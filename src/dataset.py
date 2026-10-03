"""YOLO Dataset Preparation and YAML Configuration Module.

Processes raw XRT wafer images and their ground-truth annotations into
standard YOLO format (images/train, images/val, labels/train, labels/val).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

from src.preprocess import (
    CLASS_NAMES,
    bbox_to_yolo,
    full_to_patch_bbox,
    inspect_and_load_xrt,
    preprocess_xrt_image,
    slice_patches,
)


def load_annotation_file(ann_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads defect annotations from a JSON file.

    Expected format:
    {
        "defects": [
            {
                "class_id": 0,
                "bbox": [x1, y1, x2, y2],
                ...
            }, ...
        ]
    }
    """
    path = Path(ann_path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("defects", [])


def create_yolo_yaml(
    dataset_dir: Union[str, Path],
    yaml_filepath: Optional[Union[str, Path]] = None,
) -> Path:
    """Generates the dataset.yaml file required for Ultralytics YOLO training.

    Args:
        dataset_dir: Root directory of YOLO dataset containing images/ and labels/.
        yaml_filepath: Path to write the YAML file. Defaults to `dataset_dir / "sic_defect.yaml"`.

    Returns:
        Path to the generated YAML file.
    """
    ds_dir = Path(dataset_dir).resolve()
    if yaml_filepath is None:
        yaml_path = ds_dir / "sic_defect.yaml"
    else:
        yaml_path = Path(yaml_filepath).resolve()

    yaml_dict = {
        "path": str(ds_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): v for k, v in CLASS_NAMES.items()},
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)

    return yaml_path


def build_yolo_dataset(
    raw_dir: Union[str, Path],
    output_dir: Union[str, Path],
    patch_size: int = 640,
    overlap: float = 0.2,
    val_ratio: float = 0.2,
    bg_subsample_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[Path, Dict[str, int]]:
    """Builds a complete YOLO train/val dataset from raw XRT files and annotations.

    Steps:
    1. Scan raw_dir for .tif / .tiff files and corresponding .json annotations.
    2. Convert 16-bit to 8-bit CLAHE.
    3. Slice into overlapping patches.
    4. Project full-wafer defect boxes into patch local coordinates & YOLO format.
    5. Balance positive patches (100% kept) and negative background patches (5% subsampled).
    6. Split into train and val splits and export images and labels.
    7. Generate dataset YAML.

    Args:
        raw_dir: Directory containing raw XRT TIFFs and JSON metadata.
        output_dir: Directory to store YOLO formatted dataset (data/dataset).
        patch_size: Square patch dimensions (default: 640).
        overlap: Overlap fraction for sliding window (default: 0.2).
        val_ratio: Validation split fraction (default: 0.2).
        bg_subsample_ratio: Ratio of pure background patches to keep (default: 0.05 = 5%).
        seed: Random seed for split reproducibility.

    Returns:
        Tuple of (yaml_path, statistics_dict).
    """
    rng = random.Random(seed)
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    train_img_dir = out_path / "images" / "train"
    val_img_dir = out_path / "images" / "val"
    train_lbl_dir = out_path / "labels" / "train"
    val_lbl_dir = out_path / "labels" / "val"

    # Clean existing directories to avoid mixing old patches
    if (out_path / "images").exists():
        shutil.rmtree(out_path / "images", ignore_errors=True)
    if (out_path / "labels").exists():
        shutil.rmtree(out_path / "labels", ignore_errors=True)

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Prioritize pure lossless TIF files if present to prevent JPG compression artifact degradation
    tif_files = sorted(list(set(raw_path.glob("*.tif")) | set(raw_path.glob("*.tiff")) | set(raw_path.glob("*.TIF")) | set(raw_path.glob("*.TIFF"))))
    if tif_files:
        print(f"[Dataset] Found {len(tif_files)} high-quality lossless TIF images with ground-truth labels.")
        print(f"[Dataset] Pure TIF mode active: Excluding {len(list(raw_path.glob('*.jpg')))} lossy JPGs to maximize mAP precision.")
        raw_files = tif_files
    else:
        IMAGE_EXTS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG", "*.bmp")
        raw_files_set = set()
        for ext in IMAGE_EXTS:
            raw_files_set.update(raw_path.glob(ext))
        raw_files = sorted(list(raw_files_set))

    if not raw_files:
        raise FileNotFoundError(f"No valid image files (*.tif, *.jpg, *.png) found in {raw_path}")

    positive_samples: List[Tuple[str, np.ndarray, List[str]]] = []
    selected_negatives: List[Tuple[str, np.ndarray, List[str]]] = []
    total_raw_negative_patches = 0

    defect_counts = {name: 0 for name in CLASS_NAMES.values()}

    for file_path in raw_files:
        raw_img, meta = inspect_and_load_xrt(file_path, slice_idx=0)
        prep_img = preprocess_xrt_image(raw_img)

        # Look for annotations
        ann_file = file_path.with_suffix(".json")
        defects = load_annotation_file(ann_file)

        # Slice patches in-memory
        patches = slice_patches(
            image=prep_img,
            patch_size=patch_size,
            overlap=overlap,
            save_dir=None,  # keep in-memory for selection
        )

        stem = file_path.stem

        for p in patches:
            p_idx = p["patch_id"]
            px1, py1 = p["x1"], p["y1"]
            patch_img = p["patch"]
            base_name = f"{stem}_p{p_idx:05d}_x{px1}_y{py1}"

            yolo_lines: List[str] = []

            # Spatial pre-filtering to accelerate matching on large images
            patch_margin = 60
            px2 = px1 + patch_size
            py2 = py1 + patch_size

            nearby_defects = []
            for d in defects:
                b = d.get("bbox") or d.get("bbox_xyxy")
                if b and len(b) == 4:
                    if (px1 - patch_margin <= b[2] and b[0] <= px2 + patch_margin and
                        py1 - patch_margin <= b[3] and b[1] <= py2 + patch_margin):
                        nearby_defects.append((d, b))

            for d, fbox_list in nearby_defects:
                class_id = int(d["class_id"])
                fbox = tuple(fbox_list)  # (X1, Y1, X2, Y2)

                local_box = full_to_patch_bbox(fbox, (px1, py1), patch_size=patch_size)
                if local_box is not None:
                    # Convert to YOLO (cx, cy, w, h)
                    cx, cy, w, h = bbox_to_yolo(local_box, img_size=patch_size)
                    yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    class_name = CLASS_NAMES.get(class_id, "unknown")
                    defect_counts[class_name] = defect_counts.get(class_name, 0) + 1

            if len(yolo_lines) > 0:
                positive_samples.append((base_name, patch_img, yolo_lines))
            else:
                total_raw_negative_patches += 1
                if rng.random() < bg_subsample_ratio:
                    selected_negatives.append((base_name, patch_img, []))

    all_samples = positive_samples + selected_negatives
    rng.shuffle(all_samples)

    # Split train and val
    num_val = max(1, int(len(all_samples) * val_ratio))
    val_set = all_samples[:num_val]
    train_set = all_samples[num_val:]

    def save_split(
        sample_list: List[Tuple[str, np.ndarray, List[str]]],
        img_dir: Path,
        lbl_dir: Path,
    ) -> int:
        count = 0
        for name, img_arr, labels in sample_list:
            img_path = img_dir / f"{name}.png"
            lbl_path = lbl_dir / f"{name}.txt"

            cv2.imwrite(str(img_path), img_arr)
            with open(lbl_path, "w", encoding="utf-8") as f:
                if labels:
                    f.write("\n".join(labels) + "\n")
                else:
                    # Empty file for negative background patch
                    pass
            count += 1
        return count

    n_train = save_split(train_set, train_img_dir, train_lbl_dir)
    n_val = save_split(val_set, val_img_dir, val_lbl_dir)

    # Generate data yaml
    yaml_path = create_yolo_yaml(out_path)

    stats = {
        "total_patches": len(all_samples),
        "positive_patches": len(positive_samples),
        "total_raw_negative_patches": total_raw_negative_patches,
        "negative_patches_kept": len(selected_negatives),
        "bg_subsample_ratio": bg_subsample_ratio,
        "train_samples": n_train,
        "val_samples": n_val,
        **{f"defects_{k}": v for k, v in defect_counts.items()},
    }

    return yaml_path, stats


def generate_pseudo_labels_and_dataset(
    raw_dir: Union[str, Path] = "data/raw",
    model_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "data/dataset",
    patch_size: int = 640,
    overlap: float = 0.2,
    conf_threshold: float = 0.15,
    class_conf_thresholds: Optional[Dict[str, float]] = None,
    overwrite_existing_json: bool = False,
    val_ratio: float = 0.2,
    bg_subsample_ratio: float = 0.05,
    refine_ted_tsd: bool = True,
    device: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Runs full-wafer inference on unannotated raw images (e.g. JPGs) to generate pseudo-labels,

    and then automatically compiles them into a clean 640x640 YOLO training dataset.

    Args:
        raw_dir: Directory containing raw wafer images (.jpg, .tif, etc.).
        model_path: Base weights for pseudo-labeling (defaults to models/sic_defect_v1.2.onnx).
        output_dir: Destination YOLO dataset directory.
        patch_size: Patch resolution for YOLO training (default: 640).
        overlap: Sliding-window overlap fraction.
        conf_threshold: Global confidence cutoff fallback.
        class_conf_thresholds: Per-class confidence thresholds.
        overwrite_existing_json: Whether to re-label images that already have a .json file.
        val_ratio: Validation split fraction.
        bg_subsample_ratio: Negative background patch subsampling ratio.
        refine_ted_tsd: Apply size/contrast geometric refinement.
        device: Target compute device.

    Returns:
        Tuple of (dataset_yaml_path, statistics_dict).
    """
    from src.inference import get_inference_engine

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw image directory not found: {raw_path}")

    # Determine baseline model for inference
    candidate_models = [
        Path(model_path) if model_path else None,
        Path("models/sic_defect_v1.2.onnx"),
        Path("models/sic_defect_v1.2.pt"),
        Path("models/sic_defect_v1.0.onnx"),
        Path("models/sic_defect_v1.0.pt"),
        Path("best.onnx"),
        Path("best.pt"),
        Path("yolov8n.pt"),
    ]
    chosen_model = next((p for p in candidate_models if p is not None and p.exists()), None)
    if chosen_model is None:
        raise FileNotFoundError("No model weights found to generate pseudo-labels.")

    print(f"\n==================================================")
    print(f"[Pseudo-Labeling] Generating Automated Labels for Unannotated Images")
    print(f"  - Target Directory: {raw_path}")
    print(f"  - Generator Model:  {chosen_model}")
    print(f"  - Output Dataset:   {output_dir}")
    print(f"==================================================")

    # Initialize inference engine
    engine = get_inference_engine(
        model_path=chosen_model,
        conf_threshold=conf_threshold,
        class_conf_thresholds=class_conf_thresholds,
        prefer_onnx=True,
        device=device,
    )

    IMAGE_EXTS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG", "*.bmp")
    all_raw_images = set()
    for ext in IMAGE_EXTS:
        all_raw_images.update(raw_path.glob(ext))
    sorted_images = sorted(list(all_raw_images))

    if not sorted_images:
        raise FileNotFoundError(f"No image files found in {raw_path}")

    labeled_count = 0
    skipped_count = 0
    total_pseudo_defects = 0

    for idx, img_file in enumerate(sorted_images, 1):
        json_file = img_file.with_suffix(".json")
        if json_file.exists() and not overwrite_existing_json:
            skipped_count += 1
            continue

        print(f"[{idx}/{len(sorted_images)}] Pseudo-labeling: {img_file.name}...")
        raw_img, meta = inspect_and_load_xrt(img_file, slice_idx=0)
        prep_img = preprocess_xrt_image(raw_img)

        results = engine.predict_full_wafer(
            image=prep_img,
            patch_size=patch_size,
            overlap=overlap,
            detect_particles=True,
            refine_ted_tsd=refine_ted_tsd,
        )

        defects = results.get("defects", [])
        total_pseudo_defects += len(defects)

        # Standard JSON annotation payload
        ann_payload = {
            "image": img_file.name,
            "image_shape": list(prep_img.shape[:2]),
            "total_defects": len(defects),
            "counts": results.get("counts", {}),
            "defects": [
                {
                    "defect_id": d["defect_id"],
                    "class_id": d["class_id"],
                    "class_name": d["class_name"],
                    "confidence": d["confidence"],
                    "bbox": d["bbox_xyxy"],
                    "size": d["size"],
                    "center": d["center_xy"],
                    "reclassified_from": d.get("reclassified_from", ""),
                    "pseudo_labeled": True,
                }
                for d in defects
            ],
            "is_pseudo_label": True,
            "generator_model": chosen_model.name,
        }

        with open(json_file, "w", encoding="utf-8") as jf:
            json.dump(ann_payload, jf, indent=2)

        labeled_count += 1
        print(f"  -> Generated {len(defects)} pseudo-defects -> {json_file.name}")

    print(f"\n[Pseudo-Labeling Complete] Newly labeled: {labeled_count}, Skipped (existing): {skipped_count}")
    print(f"  - Total pseudo-defects generated: {total_pseudo_defects}")

    # Build YOLO patches dataset
    print(f"\n[Dataset] Compiling 640x640 YOLO patches from pseudo-labeled and ground-truth data...")
    yaml_path, stats = build_yolo_dataset(
        raw_dir=raw_path,
        output_dir=output_dir,
        patch_size=patch_size,
        overlap=overlap,
        val_ratio=val_ratio,
        bg_subsample_ratio=bg_subsample_ratio,
    )
    stats["pseudo_labeled_images"] = labeled_count
    stats["total_pseudo_defects"] = total_pseudo_defects
    stats["generator_model"] = chosen_model.name

    return yaml_path, stats
