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
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.preprocess import (
    CLASS_NAMES,
    get_slice_grid_coordinates,
    inspect_and_load_xrt,
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


class WaferInferenceEngine:
    """Inference engine performing sliding-window defect detection on large wafer images."""

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """Initializes the inference engine with a trained YOLO or ONNX model."""
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = YOLO(str(self.model_path))

    def predict_patch(self, patch: np.ndarray) -> List[Dict[str, Any]]:
        """Runs model inference on a single 8-bit patch.

        Args:
            patch: (H, W) or (H, W, 3) 8-bit image array.

        Returns:
            List of detected boxes with local coordinates:
            [{"box": [x1, y1, x2, y2], "conf": 0.92, "class_id": 0}, ...]
        """
        # Convert grayscale to 3-channel if needed
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
    ) -> Dict[str, Any]:
        """Runs sliding-window inference across the entire wafer image.

        1. Slices wafer image into overlapping windows.
        2. Infers on each window.
        3. Projects coordinates to full-wafer coordinate system.
        4. Applies global Non-Maximum Suppression (NMS) to remove seam artifacts.
        5. Aggregates defect metrics and counts.

        Args:
            image: Preprocessed 8-bit full wafer image (H, W).
            patch_size: Slicing window size (default: 640).
            overlap: Slicing overlap ratio (default: 0.2).

        Returns:
            Structured results dictionary with defect list and class totals.
        """
        h, w = image.shape[:2]
        grid_coords = get_slice_grid_coordinates(w, h, patch_size=patch_size, overlap=overlap)

        raw_detections: List[Dict[str, Any]] = []

        print(f"[Inference] Scanning full wafer ({w}x{h}) using {len(grid_coords)} patches...")

        for x1, y1, x2, y2 in grid_coords:
            patch_img = image[y1:y2, x1:x2]

            # Handle edge padding if needed
            if patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
                padded = np.zeros((patch_size, patch_size), dtype=image.dtype)
                padded[: patch_img.shape[0], : patch_img.shape[1]] = patch_img
                patch_img = padded

            patch_dets = self.predict_patch(patch_img)

            # Map coordinates to full wafer
            for det in patch_dets:
                fx1, fy1, fx2, fy2 = patch_to_full_bbox(
                    tuple(det["box"]), (x1, y1)
                )
                raw_detections.append(
                    {
                        "box": [fx1, fy1, fx2, fy2],
                        "conf": det["conf"],
                        "class_id": det["class_id"],
                    }
                )

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

            for idx in keep_indices:
                d = class_dets[idx]
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

        print(f"[Inference Complete] Total defects detected: {len(filtered_defects)}")
        for cname, ctotal in counts.items():
            print(f"  - {cname}: {ctotal}")

        return {
            "image_shape": [h, w],
            "total_defects": len(filtered_defects),
            "counts": counts,
            "defects": filtered_defects,
        }


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
        writer.writerow(["Defect_ID", "Class_ID", "Class_Name", "Confidence", "Center_X", "Center_Y", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2", "Width", "Height"])
        for d in defects:
            writer.writerow([
                d["defect_id"],
                d["class_id"],
                d["class_name"],
                d["confidence"],
                d["center_xy"][0],
                d["center_xy"][1],
                d["bbox_xyxy"][0],
                d["bbox_xyxy"][1],
                d["bbox_xyxy"][2],
                d["bbox_xyxy"][3],
                d["size"][0],
                d["size"][1],
            ])

    print(f"[Report] Defect reports exported to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")

    return json_path, csv_path
