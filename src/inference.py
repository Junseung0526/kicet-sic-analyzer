"""Full-Wafer Sliding-Window Inference, Coordinate Aggregation, and Visualization.

Features:
- Full-scale wafer inference via sliding-window sliced patches (SAHI-like logic).
- Global coordinate reconstruction from local patch predictions.
- Global Non-Maximum Suppression (NMS) to eliminate duplicate detections at overlap seams.
- Per-class defect tallying (BPD, TED, TSD) & structured export (JSON/CSV).
- Full-wafer defect distribution visualization: Scatter Map & Density Heatmap.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.preprocess import (
    CLASS_NAMES,
    get_slice_grid_coordinates,
    inspect_and_load_xrt,
    normalize_16bit_to_8bit,
    patch_to_full_bbox,
    preprocess_xrt_image,
)


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> List[int]:
    """Pure NumPy Non-Maximum Suppression (NMS).

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Array of shape (N,) containing confidence scores.
        iou_threshold: IoU overlap threshold.

    Returns:
        List of indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (np.maximum(0.0, x2 - x1)) * (np.maximum(0.0, y2 - y1))
    order = scores.argsort()[::-1]

    keep: List[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def compute_box_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) of two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area
    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


def is_box_overlapping_or_contained(
    boxA: List[float],
    boxB: List[float],
    iou_thresh: float = 0.10,
) -> bool:
    """Checks if two boxes overlap significantly (IoU > thresh) or if either center is inside the other."""
    # 1. IoU check
    iou = compute_box_iou(boxA, boxB)
    if iou > iou_thresh:
        return True

    # 2. Check if center of boxA is inside boxB
    cxA = (boxA[0] + boxA[2]) / 2.0
    cyA = (boxA[1] + boxA[3]) / 2.0
    if boxB[0] <= cxA <= boxB[2] and boxB[1] <= cyA <= boxB[3]:
        return True

    # 3. Check if center of boxB is inside boxA
    cxB = (boxB[0] + boxB[2]) / 2.0
    cyB = (boxB[1] + boxB[3]) / 2.0
    if boxA[0] <= cxB <= boxA[2] and boxA[1] <= cyB <= boxA[3]:
        return True

    return False


def detect_surface_particles(
    image: np.ndarray,
    existing_defects: List[Dict[str, Any]],
    min_area: int = 150,
    max_area: int = 8000,
    contrast_diff_thresh: float = 45.0,
    max_aspect_ratio: float = 3.0,
    iou_exclusion_thresh: float = 0.10,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Hybrid rule-based surface particle / dust / contaminant detector.

    Detects purely isolated, extreme-contrast surface flakes and foreign particles.
    Guarantees zero interference with deep learning crystal defects (BPD, TED, TSD):
    Any candidate blob overlapping with or containing/contained by a DL detection is strictly excluded.

    Args:
        image: Wafer image (8-bit or 16-bit, grayscale or RGB).
        existing_defects: Deep learning detected defect list (never overwritten).
        min_area: Minimum contour pixel area (default 150 to ignore micro cores of TED/TSD).
        max_area: Maximum contour pixel area to exclude macroscopic border shadows.
        contrast_diff_thresh: Intensity deviation threshold relative to local background.
        max_aspect_ratio: Max aspect ratio (w/h or h/w) to exclude linear dislocations (BPD).
        iou_exclusion_thresh: IoU overlap threshold above which particle candidate is rejected.

    Returns:
        Updated defect list including pure Particle detections, and updated per-class count dictionary.
    """
    # 1. Convert to 8-bit single-channel grayscale
    if image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[0] in (3, 4):
        gray = cv2.cvtColor(np.transpose(image[:3, :, :], (1, 2, 0)), cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if gray.dtype != np.uint8:
        gray = normalize_16bit_to_8bit(gray)

    h, w = gray.shape[:2]

    # 2. Estimate smooth local background via fast boxFilter (O(1) per pixel)
    ksize = 41
    bg = cv2.boxFilter(gray, -1, (ksize, ksize))
    diff = cv2.absdiff(gray, bg)

    # 3. Dynamic contrast anomaly thresholding (strict statistical outlier: 3.5 sigma)
    step = max(1, min(h, w) // 512)
    sub_diff = diff[::step, ::step].astype(np.float32)
    mean_diff = float(np.mean(sub_diff))
    std_diff = float(np.std(sub_diff))
    thresh_val = max(contrast_diff_thresh, mean_diff + 3.5 * std_diff)

    _, binary = cv2.threshold(diff, int(thresh_val), 255, cv2.THRESH_BINARY)

    # 4. Morphological cleaning: remove single pixel speckles and connect fragmented blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5. Extract candidate contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pre-extract existing DL defect boxes for fast spatial comparison
    crystal_boxes = [d.get("bbox_xyxy", [0, 0, 0, 0]) for d in existing_defects]

    candidate_particles: List[Dict[str, Any]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Minimum physical dimensions (must not be tiny dot like TED core)
        if bw < 12 or bh < 12:
            continue

        ar = max(bw, bh) / max(1.0, min(bw, bh))
        if ar > max_aspect_ratio:
            # Exclude long linear dislocations (BPD)
            continue

        patch_diff = diff[by : by + bh, bx : bx + bw]
        if patch_diff.size == 0:
            continue
        max_c = float(np.max(patch_diff))
        if max_c < thresh_val:
            continue

        cbox = [float(bx), float(by), float(bx + bw), float(by + bh)]

        # STRICT EXCLUSION: If candidate overlaps with ANY existing DL crystal defect, discard completely!
        is_crystal_defect_core = False
        for ebox in crystal_boxes:
            if is_box_overlapping_or_contained(cbox, ebox, iou_thresh=iou_exclusion_thresh):
                is_crystal_defect_core = True
                break

        if is_crystal_defect_core:
            continue

        candidate_particles.append(
            {
                "bbox": cbox,
                "area": area,
                "center": [float(bx + bw / 2.0), float(by + bh / 2.0)],
                "size": [float(bw), float(bh)],
                "contrast": max_c,
            }
        )

    # 6. Preserve 100% of existing DL detections (NEVER overwrite crystal defects)
    updated_defects: List[Dict[str, Any]] = [dict(d) for d in existing_defects]

    # 7. Add only verified, non-overlapping surface particles
    existing_particle_boxes: List[List[float]] = []
    for cp in candidate_particles:
        is_dup = False
        for pb in existing_particle_boxes:
            if compute_box_iou(cp["bbox"], pb) > 0.25:
                is_dup = True
                break
        if not is_dup:
            existing_particle_boxes.append(cp["bbox"])
            conf = round(float(np.clip(cp["contrast"] / 100.0 + 0.50, 0.75, 0.98)), 4)
            updated_defects.append(
                {
                    "defect_id": len(updated_defects) + 1,
                    "class_id": 3,
                    "class_name": "Particle",
                    "confidence": conf,
                    "bbox_xyxy": [round(x, 1) for x in cp["bbox"]],
                    "center_xy": [round(x, 1) for x in cp["center"]],
                    "size": [round(x, 1) for x in cp["size"]],
                }
            )

    # 8. Re-index IDs and compute updated counts
    counts = {name: 0 for name in CLASS_NAMES.values()}
    for idx, d in enumerate(updated_defects, 1):
        d["defect_id"] = idx
        cname = d.get("class_name", "Particle")
        counts[cname] = counts.get(cname, 0) + 1

    return updated_defects, counts


class WaferInferenceEngine:
    """Inference engine performing sliding-window defect detection using PyTorch YOLO."""

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """Initializes the inference engine with a trained YOLO model."""
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Please install it or use OnnxWaferInferenceEngine.")
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_names = CLASS_NAMES
        self.model = YOLO(str(self.model_path))

    def predict_patch(self, patch: np.ndarray) -> List[Dict[str, Any]]:
        """Runs model inference on a single 8-bit patch."""
        if patch.ndim == 2:
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        else:
            patch_rgb = patch

        results = self.model.predict(
            source=patch_rgb,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clses = results[0].boxes.cls.cpu().numpy().astype(int)

            for b, c, cls_id in zip(boxes, confs, clses):
                detections.append(
                    {
                        "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "conf": float(c),
                        "class_id": int(cls_id),
                    }
                )

        return detections

    def predict_full_wafer(
        self,
        image: np.ndarray,
        patch_size: int = 640,
        overlap: float = 0.2,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        detect_particles: bool = True,
    ) -> Dict[str, Any]:
        """Runs sliding-window inference across the entire wafer image."""
        h, w = image.shape[:2]
        grid_coords = get_slice_grid_coordinates(w, h, patch_size=patch_size, overlap=overlap)
        total_patches = len(grid_coords)

        raw_detections: List[Dict[str, Any]] = []
        print(f"[Inference] Scanning full wafer ({w}x{h}) using {total_patches} patches...")

        for idx, (x1, y1, x2, y2) in enumerate(grid_coords):
            patch_img = image[y1:y2, x1:x2]

            if patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
                padded = np.zeros((patch_size, patch_size), dtype=image.dtype)
                padded[: patch_img.shape[0], : patch_img.shape[1]] = patch_img
                patch_img = padded

            patch_dets = self.predict_patch(patch_img)

            for det in patch_dets:
                fx1, fy1, fx2, fy2 = patch_to_full_bbox(tuple(det["box"]), (x1, y1))
                raw_detections.append(
                    {
                        "box": [fx1, fy1, fx2, fy2],
                        "conf": det["conf"],
                        "class_id": det["class_id"],
                    }
                )

            if progress_callback is not None:
                progress_callback(idx + 1, total_patches)

        # Apply Global NMS per class
        filtered_defects: List[Dict[str, Any]] = []
        counts = {name: 0 for name in CLASS_NAMES.values()}

        for class_id, class_name in CLASS_NAMES.items():
            class_dets = [d for d in raw_detections if d["class_id"] == class_id]
            if not class_dets:
                continue

            boxes_arr = np.array([d["box"] for d in class_dets], dtype=np.float32)
            scores_arr = np.array([d["conf"] for d in class_dets], dtype=np.float32)
            keep_indices = nms_boxes(boxes_arr, scores_arr, iou_threshold=self.iou_threshold)

            for kidx in keep_indices:
                d = class_dets[kidx]
                bx1, by1, bx2, by2 = d["box"]
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                width = bx2 - bx1
                height = by2 - by1

                filtered_defects.append(
                    {
                        "defect_id": len(filtered_defects) + 1,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(d["conf"], 4),
                        "bbox_xyxy": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
                        "center_xy": [round(cx, 1), round(cy, 1)],
                        "size": [round(width, 1), round(height, 1)],
                    }
                )
                counts[class_name] += 1

        # Hybrid Surface Particle Detection
        if detect_particles:
            filtered_defects, counts = detect_surface_particles(
                image=image,
                existing_defects=filtered_defects,
            )

        return {
            "image_shape": [h, w],
            "total_defects": len(filtered_defects),
            "counts": counts,
            "defects": filtered_defects,
        }


class OnnxWaferInferenceEngine:
    """Lightweight, standalone inference engine based exclusively on ONNX Runtime.

    Zero PyTorch dependency for Windows EXE standalone distribution.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: Optional[List[str]] = None,
    ):
        """Initializes ONNX Runtime session."""
        if ort is None:
            raise ImportError("onnxruntime is not installed. Please install onnxruntime.")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = CLASS_NAMES

        if providers is None:
            available = ort.get_available_providers()
            # Prioritize CUDA -> CoreML -> CPU
            preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # e.g., [1, 3, 640, 640]
        self.imgsz = int(self.input_shape[2]) if len(self.input_shape) >= 4 else 640

    def predict_patch(self, patch: np.ndarray) -> List[Dict[str, Any]]:
        """Performs raw ONNX forward inference on a single 8-bit image patch."""
        # 1. Color formatting (H, W, 3)
        if patch.ndim == 2:
            rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        elif patch.shape[2] == 1:
            rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        else:
            rgb = patch

        # 2. Resize to imgsz if needed
        if rgb.shape[0] != self.imgsz or rgb.shape[1] != self.imgsz:
            rgb = cv2.resize(rgb, (self.imgsz, self.imgsz))

        # 3. Normalization & Blob (1, 3, H, W) float32
        blob = rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # 4. Inference
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        preds = outputs[0][0]  # shape: (channels, num_anchors) e.g. (7, 8400)

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # -> (8400, 7)

        # 5. Parse boxes and class probabilities
        # YOLOv8 format: [cx, cy, w, h, class0_score, class1_score, class2_score]
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        scores = preds[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return []

        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        x1 = cx - (w / 2.0)
        y1 = cy - (h / 2.0)
        x2 = cx + (w / 2.0)
        y2 = cy + (h / 2.0)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Local NMS per class
        detections: List[Dict[str, Any]] = []
        for cid in np.unique(class_ids):
            c_mask = class_ids == cid
            c_boxes = boxes[c_mask]
            c_scores = confidences[c_mask]

            keep = nms_boxes(c_boxes, c_scores, iou_threshold=self.iou_threshold)
            for k in keep:
                kb = c_boxes[k]
                detections.append(
                    {
                        "box": [float(kb[0]), float(kb[1]), float(kb[2]), float(kb[3])],
                        "conf": float(c_scores[k]),
                        "class_id": int(cid),
                    }
                )

        return detections

    def predict_full_wafer(
        self,
        image: np.ndarray,
        patch_size: int = 640,
        overlap: float = 0.2,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        detect_particles: bool = True,
    ) -> Dict[str, Any]:
        """Runs sliding-window ONNX inference across the entire wafer image."""
        h, w = image.shape[:2]
        grid_coords = get_slice_grid_coordinates(w, h, patch_size=patch_size, overlap=overlap)
        total_patches = len(grid_coords)

        raw_detections: List[Dict[str, Any]] = []
        print(f"[ONNX Inference] Scanning full wafer ({w}x{h}) using {total_patches} patches...")

        for idx, (x1, y1, x2, y2) in enumerate(grid_coords):
            patch_img = image[y1:y2, x1:x2]

            if patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
                padded = np.zeros((patch_size, patch_size), dtype=image.dtype)
                padded[: patch_img.shape[0], : patch_img.shape[1]] = patch_img
                patch_img = padded

            patch_dets = self.predict_patch(patch_img)

            for det in patch_dets:
                fx1, fy1, fx2, fy2 = patch_to_full_bbox(tuple(det["box"]), (x1, y1))
                raw_detections.append(
                    {
                        "box": [fx1, fy1, fx2, fy2],
                        "conf": det["conf"],
                        "class_id": det["class_id"],
                    }
                )

            if progress_callback is not None:
                progress_callback(idx + 1, total_patches)

        # Apply Global NMS per class
        filtered_defects: List[Dict[str, Any]] = []
        counts = {name: 0 for name in CLASS_NAMES.values()}

        for class_id, class_name in CLASS_NAMES.items():
            class_dets = [d for d in raw_detections if d["class_id"] == class_id]
            if not class_dets:
                continue

            boxes_arr = np.array([d["box"] for d in class_dets], dtype=np.float32)
            scores_arr = np.array([d["conf"] for d in class_dets], dtype=np.float32)
            keep_indices = nms_boxes(boxes_arr, scores_arr, iou_threshold=self.iou_threshold)

            for kidx in keep_indices:
                d = class_dets[kidx]
                bx1, by1, bx2, by2 = d["box"]
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                width = bx2 - bx1
                height = by2 - by1

                filtered_defects.append(
                    {
                        "defect_id": len(filtered_defects) + 1,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(d["conf"], 4),
                        "bbox_xyxy": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
                        "center_xy": [round(cx, 1), round(cy, 1)],
                        "size": [round(width, 1), round(height, 1)],
                    }
                )
                counts[class_name] += 1

        # Hybrid Surface Particle Detection
        if detect_particles:
            filtered_defects, counts = detect_surface_particles(
                image=image,
                existing_defects=filtered_defects,
            )

        print(f"[ONNX Inference Complete] Total defects detected: {len(filtered_defects)}")
        for cname, ctotal in counts.items():
            print(f"  - {cname}: {ctotal}")

        return {
            "image_shape": [h, w],
            "total_defects": len(filtered_defects),
            "counts": counts,
            "defects": filtered_defects,
        }


def get_inference_engine(
    model_path: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    prefer_onnx: bool = True,
    device: Optional[str] = None,
) -> Union[OnnxWaferInferenceEngine, WaferInferenceEngine]:
    """Factory creating optimal inference engine (ONNX Runtime preferred for standalone desktop apps).

    Priority:
    1. If prefer_onnx and .onnx exists -> OnnxWaferInferenceEngine
    2. If .pt exists and YOLO available -> WaferInferenceEngine
    3. Auto-fallback between available weights
    """
    # Candidate search paths if model_path is None or not existing
    default_candidates = [
        Path("best.onnx"),
        Path("runs/detect/runs/train/sic_defect_train/weights/best.onnx"),
        Path("runs/detect/runs/train/sic_defect_train/weights/best.pt"),
        Path("runs/train/sic_defect_train/weights/best.onnx"),
        Path("runs/train/sic_defect_train/weights/best.pt"),
        Path("best.pt"),
        Path("yolov8n.onnx"),
        Path("yolov8n.pt"),
    ]

    chosen_path: Optional[Path] = None
    if model_path is not None:
        p = Path(model_path)
        if p.exists():
            chosen_path = p

    if chosen_path is None:
        for cand in default_candidates:
            if cand.exists():
                chosen_path = cand
                break

    if chosen_path is None:
        raise FileNotFoundError("No trained model weights found (neither best.onnx nor best.pt).")

    # If ONNX file exists or prefer_onnx is set
    if chosen_path.suffix.lower() == ".onnx" or (prefer_onnx and chosen_path.with_suffix(".onnx").exists()):
        onnx_file = chosen_path if chosen_path.suffix.lower() == ".onnx" else chosen_path.with_suffix(".onnx")
        print(f"[Engine] Initializing Lightweight ONNX Runtime Engine: {onnx_file}")
        return OnnxWaferInferenceEngine(
            model_path=onnx_file,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    # Fallback to PyTorch YOLO
    print(f"[Engine] Initializing PyTorch YOLO Engine: {chosen_path}")
    return WaferInferenceEngine(
        model_path=chosen_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )


# ==============================================================================
# Visualization and Reporting Module
# ==============================================================================

def visualize_wafer_defects(
    results: Dict[str, Any],
    image_shape: Tuple[int, int],
    output_path: Union[str, Path] = "runs/detect/wafer_map.png",
    bg_image: Optional[np.ndarray] = None,
) -> Path:
    """Generates comprehensive wafer inspection maps:

    Left: Full-wafer defect scatter plot with circular wafer boundary and class color coding.
    Right: Defect spatial density heatmap (KDE / 2D histogram).

    Args:
        results: Dictionary containing inference defect outputs.
        image_shape: Tuple of (height, width) of full wafer.
        output_path: Destination path for the saved plot.
        bg_image: Optional 8-bit image to show as semi-transparent background.

    Returns:
        Path to the saved figure image.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    h, w = image_shape
    defects = results.get("defects", [])
    counts = results.get("counts", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=150)
    plt.subplots_adjust(wspace=0.25)

    # 1. Defect Scatter Map
    ax1.set_title(f"Full-Wafer Defect Scatter Map (Total: {results['total_defects']})", fontsize=14, weight="bold")
    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)  # Invert Y to match image coordinate system
    ax1.set_aspect("equal")

    if bg_image is not None:
        ax1.imshow(bg_image, cmap="gray", alpha=0.45)
    else:
        ax1.set_facecolor("#1a1a24")

    # Draw wafer outer circle guide
    wafer_center = (w / 2.0, h / 2.0)
    wafer_radius = min(w, h) * 0.46
    wafer_circle = plt.Circle(
        wafer_center,
        wafer_radius,
        color="#00e5ff",
        fill=False,
        linestyle="--",
        linewidth=1.8,
        label="Wafer Edge",
    )
    ax1.add_patch(wafer_circle)

    # Marker styles per defect class
    style_map = {
        "BPD": {"color": "#00b0ff", "marker": "_", "size": 100, "linewidth": 2.5, "label": f"BPD ({counts.get('BPD', 0)})"},
        "TED": {"color": "#ffea00", "marker": "o", "size": 25, "linewidth": 1.0, "label": f"TED ({counts.get('TED', 0)})"},
        "TSD": {"color": "#ff1744", "marker": "s", "size": 60, "linewidth": 1.5, "label": f"TSD ({counts.get('TSD', 0)})"},
        "Particle": {"color": "#b388ff", "marker": "x", "size": 50, "linewidth": 1.5, "label": f"Particle ({counts.get('Particle', 0)})"},
    }

    for cname, st in style_map.items():
        c_defects = [d for d in defects if d["class_name"] == cname]
        if c_defects:
            xs = [d["center_xy"][0] for d in c_defects]
            ys = [d["center_xy"][1] for d in c_defects]
            ax1.scatter(
                xs,
                ys,
                c=st["color"],
                marker=st["marker"],
                s=st["size"],
                linewidths=st.get("linewidth", 1.0),
                label=st["label"],
                alpha=0.85,
            )
        else:
            # Placeholder for legend
            ax1.scatter([], [], c=st["color"], marker=st["marker"], label=st["label"])

    ax1.legend(loc="upper right", framealpha=0.8, fontsize=10)
    ax1.set_xlabel("X Coordinate (pixels)")
    ax1.set_ylabel("Y Coordinate (pixels)")
    ax1.grid(color="#444455", linestyle=":", alpha=0.5)

    # 2. Defect Density Heatmap
    ax2.set_title("Defect Spatial Density Heatmap", fontsize=14, weight="bold")
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.set_aspect("equal")

    if defects:
        all_xs = np.array([d["center_xy"][0] for d in defects])
        all_ys = np.array([d["center_xy"][1] for d in defects])

        # 2D histogram density estimation
        bins = (64, 64)
        heatmap, xedges, yedges = np.histogram2d(all_xs, all_ys, bins=bins, range=[[0, w], [0, h]])
        heatmap_blurred = cv2.GaussianBlur(heatmap.T.astype(np.float32), (7, 7), 1.8)

        im = ax2.imshow(
            heatmap_blurred,
            extent=[0, w, h, 0],
            cmap="inferno",
            interpolation="bicubic",
            aspect="equal",
        )
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Defect Density (count / area)", fontsize=10)
    else:
        ax2.text(w / 2, h / 2, "No Defects Detected", ha="center", va="center", color="white", fontsize=14)
        ax2.set_facecolor("#1a1a24")

    # Add wafer boundary guide on heatmap too
    wafer_circle_hm = plt.Circle(
        wafer_center,
        wafer_radius,
        color="#ffffff",
        fill=False,
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
    )
    ax2.add_patch(wafer_circle_hm)
    ax2.set_xlabel("X Coordinate (pixels)")
    ax2.set_ylabel("Y Coordinate (pixels)")
    ax2.grid(color="#444455", linestyle=":", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_file), dpi=150)
    plt.close(fig)
    print(f"[Visualization] Wafer defect map successfully saved to: {out_file}")

    return out_file


# Default 4-Direction Angle Bins for TSD Classification
DEFAULT_TSD_ANGLE_BINS = [
    {"label": "A", "start": 0.0, "end": 90.0},
    {"label": "B", "start": 90.0, "end": 180.0},
    {"label": "C", "start": 180.0, "end": 270.0},
    {"label": "D", "start": 270.0, "end": 360.0},
]


def measure_bpd_length(
    image: np.ndarray,
    bbox_xyxy: List[float],
    pixel_size_um: float = 2.5,
) -> Dict[str, float]:
    """Measures linear dislocation length of BPD using Principal Component Analysis (PCA).

    Args:
        image: Full wafer or patch image (2D grayscale or RGB).
        bbox_xyxy: [x1, y1, x2, y2] bounding box coordinates.
        pixel_size_um: Micrometers per pixel ratio (default: 2.5 um/px).

    Returns:
        Dictionary with length_px and length_um.
    """
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image.shape[:2]

    # Crop with padding margin
    margin = 4
    cx1 = max(0, int(round(x1 - margin)))
    cy1 = max(0, int(round(y1 - margin)))
    cx2 = min(w, int(round(x2 + margin)))
    cy2 = min(h, int(round(y2 + margin)))

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        diag = float(np.sqrt(box_w**2 + box_h**2))
        return {"length_px": round(diag, 2), "length_um": round(diag * pixel_size_um, 2)}

    # Convert to 8-bit grayscale
    if crop.ndim == 3:
        if crop.shape[2] in (3, 4):
            gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = crop[:, :, 0]
    else:
        gray = crop

    if gray.dtype != np.uint8:
        g_min, g_max = float(gray.min()), float(gray.max())
        if g_max > g_min:
            gray = np.clip((gray - g_min) / (g_max - g_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)

    # Enhance linear structures via adaptive deviation thresholding
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median_val = float(np.median(blurred))
    diff = np.abs(blurred.astype(np.float32) - median_val)
    threshold = np.percentile(diff, 65)
    mask = (diff >= threshold).astype(np.uint8) * 255

    # Morphology closing to connect disjoint dislocation segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    pts = np.argwhere(mask > 0)  # [y, x]
    if len(pts) >= 6:
        pts_xy = pts[:, [1, 0]].astype(np.float32)  # [x, y]
        try:
            mean, eigenvectors = cv2.PCACompute(pts_xy, mean=None)
            primary_axis = eigenvectors[0]
            projected = np.dot(pts_xy - mean, primary_axis)
            pca_len = float(np.max(projected) - np.min(projected))

            # MinAreaRect check
            rect = cv2.minAreaRect(pts_xy)
            rect_len = max(rect[1][0], rect[1][1])
            length_px = float(max(pca_len, rect_len, 2.0))
        except Exception:
            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)
            length_px = float(np.sqrt(box_w**2 + box_h**2))
    else:
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        length_px = float(np.sqrt(box_w**2 + box_h**2))

    length_um = length_px * pixel_size_um
    return {
        "length_px": round(length_px, 2),
        "length_um": round(length_um, 2),
    }


def calculate_tsd_angle(
    image: np.ndarray,
    bbox_xyxy: List[float],
) -> float:
    """Calculates directional propagation angle (0 to 360 deg) of TSD core using image moments.

    Args:
        image: Full wafer or patch image.
        bbox_xyxy: [x1, y1, x2, y2] bounding box coordinates.

    Returns:
        Angle in degrees [0.0, 360.0).
    """
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image.shape[:2]

    margin = 3
    cx1 = max(0, int(round(x1 - margin)))
    cy1 = max(0, int(round(y1 - margin)))
    cx2 = min(w, int(round(x2 + margin)))
    cy2 = min(h, int(round(y2 + margin)))

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return 0.0

    if crop.ndim == 3:
        if crop.shape[2] in (3, 4):
            gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = crop[:, :, 0]
    else:
        gray = crop

    if gray.dtype != np.uint8:
        g_min, g_max = float(gray.min()), float(gray.max())
        if g_max > g_min:
            gray = np.clip((gray - g_min) / (g_max - g_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)

    ch, cw = gray.shape[:2]
    mid_x, mid_y = cw / 2.0, ch / 2.0

    # Intensity deviation weight emphasizing asymmetric core & tail
    weight = np.abs(gray.astype(np.float32) - float(np.mean(gray)))
    sum_w = float(np.sum(weight))

    if sum_w > 1e-4:
        grid_y, grid_x = np.indices((ch, cw))
        dx = float(np.sum((grid_x - mid_x) * weight)) / sum_w
        dy = float(np.sum((grid_y - mid_y) * weight)) / sum_w

        # If centroid offset is distinct, compute vector angle
        if abs(dx) > 0.08 or abs(dy) > 0.08:
            deg = float(np.degrees(np.arctan2(dy, dx)))
            angle = (deg + 360.0) % 360.0
            return round(angle, 1)

    # Secondary fallback: 2nd-order orientation moment
    m = cv2.moments(weight)
    if m["m00"] > 1e-4:
        mu20, mu02, mu11 = m["mu20"], m["mu02"], m["mu11"]
        deg = float(np.degrees(0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)))
        angle = (deg + 360.0) % 360.0
        return round(angle, 1)

    return 0.0


def classify_tsd_angle(
    angle_deg: float,
    angle_bins: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Maps continuous angle (0-360 deg) to custom discrete directional rule bin.

    Args:
        angle_deg: Angle in degrees [0, 360).
        angle_bins: List of dicts specifying label, start, and end angles.

    Returns:
        Matching bin label string (e.g. 'A', 'B', 'C', 'D' or custom name).
    """
    bins = angle_bins if angle_bins is not None else DEFAULT_TSD_ANGLE_BINS
    norm_angle = (float(angle_deg) + 360.0) % 360.0

    for b in bins:
        start = float(b.get("start", 0.0))
        end = float(b.get("end", 360.0))
        label = str(b.get("label", "?"))

        if start <= end:
            if start <= norm_angle < end or (end >= 360.0 and norm_angle >= start):
                return label
        else:
            # Wrap-around case (e.g. 315° to 45°)
            if norm_angle >= start or norm_angle < end:
                return label

    return bins[0]["label"] if bins else "A"


def analyze_domain_measurements(
    image: np.ndarray,
    defects: List[Dict[str, Any]],
    pixel_size_um: float = 2.5,
    angle_bins: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Performs domain-specific precision measurements:

    1. BPD length profiling (px & um via PCA).
    2. TSD 4-way / custom angle bin classification.

    Args:
        image: Full wafer image array.
        defects: List of defect dictionaries.
        pixel_size_um: Micrometers per pixel scaling factor.
        angle_bins: Custom angle rule bins for TSD.

    Returns:
        Structured analysis summary with enriched defect items.
    """
    bins = angle_bins if angle_bins is not None else DEFAULT_TSD_ANGLE_BINS
    bpd_lengths_um: List[float] = []
    tsd_bin_counts: Dict[str, int] = {str(b["label"]): 0 for b in bins}

    for d in defects:
        cname = d.get("class_name", "")
        bbox = d.get("bbox_xyxy", [0, 0, 0, 0])

        if cname == "BPD":
            res = measure_bpd_length(image, bbox, pixel_size_um=pixel_size_um)
            d["length_px"] = res["length_px"]
            d["length_um"] = res["length_um"]
            bpd_lengths_um.append(res["length_um"])

        elif cname == "TSD":
            ang = calculate_tsd_angle(image, bbox)
            bin_label = classify_tsd_angle(ang, bins)
            d["angle_deg"] = ang
            d["angle_label"] = bin_label
            d["sub_class"] = f"TSD-{bin_label}"
            tsd_bin_counts[bin_label] = tsd_bin_counts.get(bin_label, 0) + 1

    # BPD Statistics
    if bpd_lengths_um:
        bpd_stats = {
            "count": len(bpd_lengths_um),
            "mean_um": round(float(np.mean(bpd_lengths_um)), 2),
            "max_um": round(float(np.max(bpd_lengths_um)), 2),
            "min_um": round(float(np.min(bpd_lengths_um)), 2),
            "std_um": round(float(np.std(bpd_lengths_um)), 2),
        }
    else:
        bpd_stats = {"count": 0, "mean_um": 0.0, "max_um": 0.0, "min_um": 0.0, "std_um": 0.0}

    # TSD Statistics
    total_tsd = sum(tsd_bin_counts.values())
    tsd_stats = {
        "count": total_tsd,
        "bin_counts": tsd_bin_counts,
        "bin_ratios": {
            k: round((v / float(total_tsd)) * 100.0, 1) if total_tsd > 0 else 0.0
            for k, v in tsd_bin_counts.items()
        },
    }

    return {
        "pixel_size_um": pixel_size_um,
        "bpd_stats": bpd_stats,
        "tsd_stats": tsd_stats,
        "angle_bins": bins,
    }


def export_defect_reports(
    results: Dict[str, Any],
    output_base: Union[str, Path] = "runs/detect/wafer_defects",
) -> Tuple[Path, Path]:
    """Exports detected defect data into JSON and CSV files."""
    base_path = Path(output_base)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = base_path.with_suffix(".json")
    csv_path = base_path.with_suffix(".csv")

    # JSON export
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # CSV export
    defects = results.get("defects", [])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Defect_ID",
            "Class_ID",
            "Class_Name",
            "Sub_Class",
            "Confidence",
            "Center_X",
            "Center_Y",
            "BBox_X1",
            "BBox_Y1",
            "BBox_X2",
            "BBox_Y2",
            "Width",
            "Height",
            "Length_um",
            "TSD_Angle_deg",
            "TSD_Angle_Label",
        ])
        for d in defects:
            writer.writerow([
                d.get("defect_id", 0),
                d.get("class_id", 0),
                d.get("class_name", ""),
                d.get("sub_class", d.get("class_name", "")),
                d.get("confidence", 0.0),
                d.get("center_xy", [0, 0])[0],
                d.get("center_xy", [0, 0])[1],
                d.get("bbox_xyxy", [0, 0, 0, 0])[0],
                d.get("bbox_xyxy", [0, 0, 0, 0])[1],
                d.get("bbox_xyxy", [0, 0, 0, 0])[2],
                d.get("bbox_xyxy", [0, 0, 0, 0])[3],
                d.get("size", [0, 0])[0],
                d.get("size", [0, 0])[1],
                d.get("length_um", "N/A"),
                d.get("angle_deg", "N/A"),
                d.get("angle_label", "N/A"),
            ])

    print(f"[Report] Defect reports exported to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")

    return json_path, csv_path
