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
    background_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[Path, Dict[str, int]]:
    """Builds a complete YOLO train/val dataset from raw XRT files and annotations.

    Steps:
    1. Scan raw_dir for .tif / .tiff files and corresponding .json annotations.
    2. Convert 16-bit to 8-bit CLAHE.
    3. Slice into overlapping patches.
    4. Project full-wafer defect boxes into patch local coordinates & YOLO format.
    5. Balance positive patches and negative background patches.
    6. Split into train and val splits and export images and labels.
    7. Generate dataset YAML.

    Args:
        raw_dir: Directory containing raw XRT TIFFs and JSON metadata.
        output_dir: Directory to store YOLO formatted dataset (data/dataset).
        patch_size: Square patch dimensions (default: 640).
        overlap: Overlap fraction for sliding window (default: 0.2).
        val_ratio: Validation split fraction (default: 0.2).
        background_ratio: Max ratio of background (empty) patches to positive patches.
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

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Find raw image files
    raw_files = sorted(
        list(raw_path.glob("*.tif"))
        + list(raw_path.glob("*.tiff"))
        + list(raw_path.glob("*.TIF"))
        + list(raw_path.glob("*.TIFF"))
    )

    if not raw_files:
        raise FileNotFoundError(f"No TIFF files found in {raw_path}")

    positive_samples: List[Tuple[str, np.ndarray, List[str]]] = []
    negative_samples: List[Tuple[str, np.ndarray, List[str]]] = []

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

            for d in defects:
                class_id = int(d["class_id"])
                fbox = tuple(d["bbox"])  # (X1, Y1, X2, Y2)

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
                negative_samples.append((base_name, patch_img, []))

    # Subsample negative background patches to maintain realistic yet balanced data
    max_negatives = max(1, int(len(positive_samples) * background_ratio))
    rng.shuffle(negative_samples)
    selected_negatives = negative_samples[:max_negatives]

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
        "negative_patches": len(selected_negatives),
        "train_samples": n_train,
        "val_samples": n_val,
        **{f"defects_{k}": v for k, v in defect_counts.items()},
    }

    return yaml_path, stats
