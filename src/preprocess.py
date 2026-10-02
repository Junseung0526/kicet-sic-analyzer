"""Data Inspection, Preprocessing, and High-Resolution Patch Slicing Module for SiC XRT Images.

Key Features:
- Load 2D/3D XRT TIFF files with metadata inspection (shape, dtype, min/max, slices).
- 16-bit to 8-bit dynamic range conversion using percentile min-max normalization & CLAHE.
- Fallback mock 16-bit XRT image & defect annotation generator (BPD, TED, TSD).
- Overlapping sliding-window patch slicing with exact bi-directional coordinate transformation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import roifile
import tifffile


# Defect Class Definitions
CLASS_NAMES = {
    0: "BPD",       # Basal Plane Dislocation (기저면 전위)
    1: "TED",       # Threading Edge Dislocation (관통 칼날 전위)
    2: "TSD",       # Threading Screw Dislocation (관통 나선 전위)
    3: "Particle",  # Surface Particle / Dust / Artifact (표면 이물질)
}
CLASS_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


def inspect_and_load_xrt(
    filepath: Union[str, Path],
    slice_idx: Optional[int] = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads a 2D or 3D XRT TIFF file and inspects its properties.

    Args:
        filepath: Path to the .tif / .tiff file.
        slice_idx: Index of slice to extract if 3D stack is encountered (default: 0).
                   If None, returns full 3D array if 3D.

    Returns:
        image: 2D numpy array (or 3D if slice_idx is None and data is 3D).
        meta: Dictionary of inspected metadata (shape, dtype, min, max, is_3d, num_slices).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(str(path)) as tif:
            num_pages = len(tif.pages)
            if num_pages > 1:
                # Multi-page 3D TIFF depth stack
                is_3d = True
                num_slices = num_pages
                first_page = tif.pages[0]
                p_shape = first_page.shape
                dtype_str = str(first_page.dtype)
                orig_shape = (num_pages, *p_shape)

                if slice_idx is not None:
                    idx = max(0, min(slice_idx, num_slices - 1))
                    image = tif.pages[idx].asarray()
                    selected_slice = idx
                else:
                    image = tif.asarray()
                    selected_slice = None

                val_min = float(np.min(image))
                val_max = float(np.max(image))

                meta: Dict[str, Any] = {
                    "filepath": str(path),
                    "filename": path.name,
                    "original_shape": orig_shape,
                    "dtype": dtype_str,
                    "min_val": val_min,
                    "max_val": val_max,
                    "is_3d": is_3d,
                    "num_slices": num_slices,
                }
                if selected_slice is not None:
                    meta["selected_slice"] = selected_slice
                return image, meta

    raw_data = tifffile.imread(str(path))
    original_shape = raw_data.shape
    dtype_str = str(raw_data.dtype)
    val_min = float(np.min(raw_data))
    val_max = float(np.max(raw_data))
    # Determine if truly a 3D depth stack vs 2D color image
    is_3d = False
    num_slices = 1

    if raw_data.ndim == 2:
        is_3d = False
        num_slices = 1
    elif raw_data.ndim == 3:
        # Check if the 3rd dimension is color channels (RGB/RGBA)
        if original_shape[2] in (1, 3, 4):
            is_3d = False
            num_slices = 1
        elif original_shape[0] in (1, 3, 4) and original_shape[1] > 64 and original_shape[2] > 64:
            # Channel-first 2D color image (C, H, W) -> transpose to (H, W, C)
            raw_data = np.transpose(raw_data, (1, 2, 0))
            original_shape = raw_data.shape
            is_3d = False
            num_slices = 1
        else:
            # Truly a 3D depth stack of 2D slices (e.g., shape = (D, H, W))
            is_3d = True
            num_slices = original_shape[0]
    elif raw_data.ndim >= 4:
        is_3d = True
        num_slices = original_shape[0]

    meta: Dict[str, Any] = {
        "filepath": str(path),
        "filename": path.name,
        "original_shape": original_shape,
        "dtype": dtype_str,
        "min_val": val_min,
        "max_val": val_max,
        "is_3d": is_3d,
        "num_slices": num_slices,
    }

    if is_3d and slice_idx is not None:
        idx = max(0, min(slice_idx, num_slices - 1))
        image = raw_data[idx]
        meta["selected_slice"] = idx
    else:
        image = raw_data

    return image, meta


def normalize_16bit_to_8bit(
    image: np.ndarray,
    use_percentile: bool = True,
    p_min: float = 0.5,
    p_max: float = 99.5,
) -> np.ndarray:
    """Converts 16-bit (or arbitrary dynamic range) images into safe 8-bit (uint8).

    Supports both single-channel (grayscale) and multi-channel (RGB) inputs.

    Args:
        image: Input image array (uint16, uint8, float32, etc.).
        use_percentile: Whether to clip extreme outliers using percentiles.
        p_min: Lower percentile cutoff (e.g. 0.5%).
        p_max: Upper percentile cutoff (e.g. 99.5%).

    Returns:
        8-bit single-channel grayscale image (uint8, 0~255).
    """
    # Convert multi-channel to grayscale if needed
    if image.ndim == 3 and image.shape[2] in (3, 4):
        img_gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    elif image.ndim == 3 and image.shape[0] in (3, 4):
        img_gray = cv2.cvtColor(np.transpose(image[:3, :, :], (1, 2, 0)), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image

    # If already uint8 and values span a normal range, return directly
    if img_gray.dtype == np.uint8 and not use_percentile:
        return img_gray

    img_float = img_gray.astype(np.float32)

    if use_percentile:
        low = np.percentile(img_float, p_min)
        high = np.percentile(img_float, p_max)
    else:
        low = np.min(img_float)
        high = np.max(img_float)

    if high <= low:
        # Uniform or degenerated image
        return np.zeros(img_gray.shape[:2], dtype=np.uint8)

    # Clip and scale to 0~255
    clipped = np.clip(img_float, low, high)
    normalized = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    return normalized


def apply_clahe(
    image_8bit: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances subtle dislocation contrasts in XRT images without blowing out noise.

    Args:
        image_8bit: 8-bit single-channel image (uint8).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization (e.g. (8, 8)).

    Returns:
        Contrast-enhanced 8-bit image (uint8).
    """
    if image_8bit.dtype != np.uint8:
        raise ValueError("apply_clahe requires uint8 input")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image_8bit)
    return enhanced


def preprocess_xrt_image(
    image: np.ndarray,
    clip_limit: float = 2.5,
    use_percentile: bool = True,
) -> np.ndarray:
    """End-to-end preprocessing pipeline: 16-bit Normalization + CLAHE.

    Args:
        image: Raw XRT image array (grayscale or RGB, uint8 or uint16).
        clip_limit: CLAHE clip limit.
        use_percentile: Outlier clipping toggle.

    Returns:
        Preprocessed 8-bit single-channel image ready for DL backbones (YOLO).
    """
    img_8bit = normalize_16bit_to_8bit(image, use_percentile=use_percentile)
    enhanced = apply_clahe(img_8bit, clip_limit=clip_limit)
    return enhanced


# Default bounding box half-sizes for point-annotated ImageJ ROIs
DEFECT_BOX_SIZES = {
    0: 40,  # BPD: 40x40 box (lengthy dislocation line)
    1: 16,  # TED: 16x16 box (small dot)
    2: 30,  # TSD: 30x30 box (distinct core & halo)
    3: 24,  # Particle: 24x24 box (surface dust/foreign particles)
}


def parse_kicet_roi_file(
    roi_path: Union[str, Path],
    img_shape: Tuple[int, int],
    start_id: int = 1,
) -> List[Dict[str, Any]]:
    """Parses an ImageJ .roi file from KICET annotations and converts to bounding boxes.

    Args:
        roi_path: Path to the .roi file.
        img_shape: (height, width) of the corresponding wafer image.
        start_id: Starting integer ID for detected defects.

    Returns:
        List of defect dictionaries in standard full-image coordinates.
    """
    path = Path(roi_path)
    if not path.exists():
        return []

    fname_upper = path.name.upper()
    if "BPD" in fname_upper:
        class_id = 0
    elif "TSD" in fname_upper:
        class_id = 2
    elif "TED" in fname_upper:
        class_id = 1
    elif any(k in fname_upper for k in ["PARTICLE", "DUST", "ARTIFACT", "UNKNOWN"]):
        class_id = 3
    else:
        # Ignore unclassified ROIs
        return []

    class_name = CLASS_NAMES[class_id]
    box_size = DEFECT_BOX_SIZES[class_id]
    half_box = box_size / 2.0
    h, w = img_shape[:2]

    defects: List[Dict[str, Any]] = []

    try:
        roi = roifile.roiread(str(path))
        if roi is None:
            return []

        # Multi-point or single coordinate extraction
        if hasattr(roi, "coordinates") and roi.coordinates() is not None:
            coords = np.array(roi.coordinates())
            # Shape can be (N, 2) where coords are [x, y]
            if coords.ndim == 1 and len(coords) == 2:
                coords = coords.reshape(1, 2)
            elif coords.ndim == 2:
                pass
            else:
                coords = np.empty((0, 2))
        else:
            # Fallback to bounding box center
            cx = (roi.left + roi.right) / 2.0
            cy = (roi.top + roi.bottom) / 2.0
            coords = np.array([[cx, cy]])

        for pt in coords:
            cx, cy = float(pt[0]), float(pt[1])
            bx1 = max(0.0, cx - half_box)
            by1 = max(0.0, cy - half_box)
            bx2 = min(float(w), cx + half_box)
            by2 = min(float(h), cy + half_box)

            defects.append(
                {
                    "defect_id": start_id + len(defects),
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
                    "center": [round(cx, 1), round(cy, 1)],
                }
            )
    except Exception as e:
        print(f"[Warning] Failed to parse ROI {path.name}: {e}")

    return defects


def extract_key_slice_from_3d_tiff(
    tiff_path: Union[str, Path],
    dest_path: Union[str, Path],
    key_slice_idx: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extracts a single validated 2D key reference slice from a 3D multi-page TIFF stack.

    Prevents negative background pollution by never blindly exporting empty stack slices.

    Args:
        tiff_path: Path to the 3D multi-page TIFF.
        dest_path: Destination path for the single 2D TIFF.
        key_slice_idx: Zero-based slice index matching ImageJ ROI annotations.

    Returns:
        Tuple of (slice_numpy_array, metadata_dict).
    """
    path = Path(tiff_path)
    with tifffile.TiffFile(str(path)) as tif:
        num_pages = len(tif.pages)
        valid_idx = max(0, min(key_slice_idx, num_pages - 1))
        slice_data = tif.pages[valid_idx].asarray()

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(dest), slice_data)

    meta = {
        "original_file": str(path),
        "total_slices": num_pages,
        "extracted_slice": valid_idx,
        "shape": list(slice_data.shape),
        "dtype": str(slice_data.dtype),
    }
    return slice_data, meta


def import_kicet_dataset(
    source_dir: Union[str, Path] = "/Volumes/T9 Junseung/세라믹 기술원 데이터/XRT 2D",
    target_raw_dir: Union[str, Path] = "data/raw",
    use_symlink: bool = True,
) -> List[Dict[str, Any]]:
    """Discovers KICET raw 2D XRT TIFF files and their ImageJ ROIs, and imports them to data/raw.

    Args:
        source_dir: Base directory where KICET data resides.
        target_raw_dir: Destination data/raw directory.
        use_symlink: If True, creates symlinks to large TIFFs instead of copying.

    Returns:
        List of imported dataset summaries.
    """
    src_path = Path(source_dir)
    dst_path = Path(target_raw_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"KICET source directory not found: {src_path}")

    # Find all TIFF files in source directory
    tiff_files = sorted(list(src_path.rglob("*.tif")) + list(src_path.rglob("*.tiff")))

    imported_summary: List[Dict[str, Any]] = []

    for idx, tiff_file in enumerate(tiff_files, 1):
        parent_dir = tiff_file.parent

        # Inspect image shape
        try:
            img, meta = inspect_and_load_xrt(tiff_file)
            img_shape = img.shape[:2]
        except Exception as e:
            print(f"[Skip] Could not read {tiff_file}: {e}")
            continue

        # Look for corresponding ROI files in the same directory
        raw_roi_files = sorted(list(parent_dir.glob("*.roi")))

        # Filter ROI files by filename context (e.g. avoid matching After ROIs to Before TIF)
        t_stem = tiff_file.stem.lower()
        roi_files = []
        for r in raw_roi_files:
            r_name = r.name.lower()
            if "after" in t_stem and "before" in r_name:
                continue
            if "before" in t_stem and "after" in r_name:
                continue
            roi_files.append(r)

        if not roi_files:
            print(f"[Skip] No matching ROI label files for {tiff_file.name} in {parent_dir.name}")
            continue

        all_defects: List[Dict[str, Any]] = []
        for rfile in roi_files:
            parsed = parse_kicet_roi_file(rfile, img_shape, start_id=len(all_defects) + 1)
            all_defects.extend(parsed)

        if not all_defects:
            print(f"[Skip] Zero valid defects found for {tiff_file.name}")
            continue

        # Create standardized target file name
        # E.g. "kicet_1area_after_annealing_1.tif"
        rel_parts = tiff_file.relative_to(src_path).parts
        area_prefix = ""
        for part in rel_parts:
            if "area" in part.lower():
                area_prefix = part.lower().replace(" ", "") + "_"
                break

        safe_stem = tiff_file.stem.strip().lower().replace(" ", "_")
        parent_stem = parent_dir.name.strip().lower().replace(" ", "_")
        if parent_stem in safe_stem:
            dest_name = f"kicet_{area_prefix}{safe_stem}"
        elif safe_stem in parent_stem:
            dest_name = f"kicet_{area_prefix}{parent_stem}"
        elif parent_stem.split("_")[0] == safe_stem.split("_")[0]:
            dest_name = f"kicet_{area_prefix}{safe_stem}"
        else:
            dest_name = f"kicet_{area_prefix}{parent_stem}_{safe_stem}"

        dest_tif = dst_path / f"{dest_name}.tif"
        dest_json = dst_path / f"{dest_name}.json"

        # Link or extract single 2D slice for 3D stacks
        if not dest_tif.exists():
            if meta.get("is_3d", False):
                # 3D stack: extract only the verified key slice (slice 0) to prevent empty slice pollution
                print(f"[3D Stack] Extracting key slice 0 from {meta['num_slices']} slices -> {dest_tif.name}")
                tifffile.imwrite(str(dest_tif), img)
            elif use_symlink:
                try:
                    dest_tif.symlink_to(tiff_file.resolve())
                except OSError:
                    import shutil
                    shutil.copy2(tiff_file, dest_tif)
            else:
                import shutil
                shutil.copy2(tiff_file, dest_tif)

        # Write ground truth JSON
        counts = {name: 0 for name in CLASS_NAMES.values()}
        for d in all_defects:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

        with open(dest_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_file": dest_tif.name,
                    "original_path": str(tiff_file),
                    "shape": list(img_shape),
                    "num_defects": len(all_defects),
                    "counts": counts,
                    "defects": all_defects,
                },
                f,
                indent=2,
            )

        summary_item = {
            "image": dest_tif.name,
            "shape": list(img_shape),
            "num_defects": len(all_defects),
            "counts": counts,
            "rois_found": len(roi_files),
        }
        imported_summary.append(summary_item)
        print(
            f"[Imported] {dest_tif.name}: {len(all_defects)} defects "
            f"(BPD={counts['BPD']}, TED={counts['TED']}, TSD={counts['TSD']}, Particle={counts.get('Particle', 0)})"
        )

    return imported_summary



def generate_mock_xrt_data(
    filepath: Union[str, Path],
    shape: Tuple[int, int] = (4096, 4096),
    num_defects: int = 60,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generates synthetic 16-bit XRT wafer image with artificial defects.

    Defect Models:
    - BPD (Class 0): Elongated streak / line (length 25~80px, width 1~3px)
    - TED (Class 1): Tiny dark/bright circular/elliptical dot (radius 2~5px)
    - TSD (Class 2): Prominent circular dislocation with dark core & bright halo (radius 7~16px)

    Args:
        filepath: Destination path to save the .tif image.
        shape: Full wafer image resolution (height, width).
        num_defects: Total number of artificial defects to seed.
        seed: Random seed for reproducibility.

    Returns:
        synthetic_img: 16-bit uint16 numpy array.
        defects_gt: List of defect bounding boxes in full-image coordinates:
                    [{"class_id": 0, "class_name": "BPD", "bbox": [x1, y1, x2, y2]}, ...]
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # 1. Base wafer background (smooth gradient + slight low-frequency curvature)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    # Subtle radial intensity falloff and tilt mimicking X-ray beam footprint
    cx, cy = w / 2.0, h / 2.0
    radial = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (w * 0.7)
    tilt = 0.05 * (x_coords / w) + 0.03 * (y_coords / h)

    # 16-bit baseline intensity around 30000 ~ 45000 (typical 16-bit XRT detector values)
    base_intensity = 38000.0 - (radial * 8000.0) + (tilt * 4000.0)
    # Add detector grain / Poisson-like noise
    noise = rng.normal(loc=0.0, scale=450.0, size=(h, w))
    canvas = np.clip(base_intensity + noise, 1000.0, 64000.0).astype(np.float32)

    defects_gt: List[Dict[str, Any]] = []

    # Safe margin from borders
    margin = 120
    x_min, x_max = margin, w - margin
    y_min, y_max = margin, h - margin

    for i in range(num_defects):
        # Choose class: 45% BPD, 35% TED, 20% TSD (common SiC distribution)
        rand_val = rng.random()
        if rand_val < 0.45:
            class_id = 0  # BPD
        elif rand_val < 0.80:
            class_id = 1  # TED
        else:
            class_id = 2  # TSD

        defect_x = rng.integers(x_min, x_max)
        defect_y = rng.integers(y_min, y_max)

        if class_id == 0:  # BPD (Basal Plane Dislocation - Line streak)
            angle = rng.uniform(-np.pi / 3, np.pi / 3)  # Oriented primarily along off-cut direction
            length = rng.uniform(30.0, 75.0)
            dx = (length / 2.0) * np.cos(angle)
            dy = (length / 2.0) * np.sin(angle)

            pt1 = (int(defect_x - dx), int(defect_y - dy))
            pt2 = (int(defect_x + dx), int(defect_y + dy))

            # Draw dark line streak onto temp layer
            patch_margin = int(length + 10)
            px1 = max(0, defect_x - patch_margin)
            px2 = min(w, defect_x + patch_margin)
            py1 = max(0, defect_y - patch_margin)
            py2 = min(h, defect_y + patch_margin)

            temp = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
            cv2.line(
                temp,
                (pt1[0] - px1, pt1[1] - py1),
                (pt2[0] - px1, pt2[1] - py1),
                color=255,
                thickness=rng.integers(2, 4),
            )
            # Blur line to look natural in diffraction image
            blurred = cv2.GaussianBlur(temp.astype(np.float32), (5, 5), 1.2)
            # Modulate canvas: XRT BPDs appear as contrast drop (or rise depending on g-vector)
            canvas[py1:py2, px1:px2] -= blurred * rng.uniform(25.0, 45.0)

            # Bounding box
            bx1 = min(pt1[0], pt2[0]) - 5
            bx2 = max(pt1[0], pt2[0]) + 5
            by1 = min(pt1[1], pt2[1]) - 5
            by2 = max(pt1[1], pt2[1]) + 5

        elif class_id == 1:  # TED (Threading Edge Dislocation - Small dot)
            radius = rng.integers(2, 5)
            rr = radius * 2
            px1 = max(0, defect_x - rr)
            px2 = min(w, defect_x + rr)
            py1 = max(0, defect_y - rr)
            py2 = min(h, defect_y + rr)

            temp = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
            cv2.circle(temp, (defect_x - px1, defect_y - py1), radius, 255, -1)
            blurred = cv2.GaussianBlur(temp.astype(np.float32), (3, 3), 0.8)
            canvas[py1:py2, px1:px2] -= blurred * rng.uniform(30.0, 50.0)

            bx1 = defect_x - radius - 3
            bx2 = defect_x + radius + 3
            by1 = defect_y - radius - 3
            by2 = defect_y + radius + 3

        else:  # TSD (Threading Screw Dislocation - Larger distinct core + contrast modulation)
            radius = rng.integers(7, 14)
            rr = radius * 3
            px1 = max(0, defect_x - rr)
            px2 = min(w, defect_x + rr)
            py1 = max(0, defect_y - rr)
            py2 = min(h, defect_y + rr)

            temp = np.zeros((py2 - py1, px2 - px1), dtype=np.float32)
            cv2.circle(
                temp,
                (defect_x - px1, defect_y - py1),
                radius,
                -1.0,
                -1,
            )
            # Surround with slight strain field halo
            cv2.circle(
                temp,
                (defect_x - px1, defect_y - py1),
                int(radius * 1.8),
                0.35,
                2,
            )
            blurred = cv2.GaussianBlur(temp, (9, 9), 2.5)
            canvas[py1:py2, px1:px2] += blurred * rng.uniform(40.0, 70.0)

            bx1 = defect_x - radius - 6
            bx2 = defect_x + radius + 6
            by1 = defect_y - radius - 6
            by2 = defect_y + radius + 6

        # Clip box to image boundaries
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(w, bx2)
        by2 = min(h, by2)

        defects_gt.append(
            {
                "defect_id": i,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                "center": [float(defect_x), float(defect_y)],
            }
        )

    synthetic_img = np.clip(canvas, 0, 65535).astype(np.uint16)

    # Save to disk
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), synthetic_img)

    # Save ground truth metadata alongside
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_file": out_path.name,
                "shape": list(shape),
                "num_defects": len(defects_gt),
                "defects": defects_gt,
            },
            f,
            indent=2,
        )

    return synthetic_img, defects_gt


# ==============================================================================
# Patch Slicing & Coordinate Mapping
# ==============================================================================

def get_slice_grid_coordinates(
    img_width: int,
    img_height: int,
    patch_size: int = 640,
    overlap: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """Generates upper-left and bottom-right crop bounds for overlapping patches.

    Ensures full coverage of the image without missing edges.

    Args:
        img_width: Full image width.
        img_height: Full image height.
        patch_size: Size of square patch (e.g. 640).
        overlap: Overlap fraction (0.0 <= overlap < 1.0, e.g. 0.2 for 20%).

    Returns:
        List of (x1, y1, x2, y2) patch bounds on full image coordinates.
    """
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("Overlap must be in range [0.0, 1.0)")

    stride = int(round(patch_size * (1.0 - overlap)))
    stride = max(1, stride)

    # Generate x coordinates
    x_starts: List[int] = []
    x = 0
    while x + patch_size <= img_width:
        x_starts.append(x)
        x += stride
    if not x_starts or x_starts[-1] + patch_size < img_width:
        # Align last patch to right border
        x_starts.append(max(0, img_width - patch_size))

    # Generate y coordinates
    y_starts: List[int] = []
    y = 0
    while y + patch_size <= img_height:
        y_starts.append(y)
        y += stride
    if not y_starts or y_starts[-1] + patch_size < img_height:
        # Align last patch to bottom border
        y_starts.append(max(0, img_height - patch_size))

    # Remove duplicates preserving order
    x_starts = sorted(list(set(x_starts)))
    y_starts = sorted(list(set(y_starts)))

    coords: List[Tuple[int, int, int, int]] = []
    for ys in y_starts:
        for xs in x_starts:
            coords.append((xs, ys, min(xs + patch_size, img_width), min(ys + patch_size, img_height)))

    return coords


def slice_patches(
    image: np.ndarray,
    patch_size: int = 640,
    overlap: float = 0.2,
    save_dir: Optional[Union[str, Path]] = None,
    prefix: str = "patch",
) -> List[Dict[str, Any]]:
    """Slices an image into overlapping patches and optionally saves them to disk.

    Args:
        image: 2D numpy array (e.g., preprocessed 8-bit image).
        patch_size: Width and height of sliced square patches.
        overlap: Overlap ratio (e.g. 0.2 = 20%).
        save_dir: If provided, patches will be saved as PNG images in this directory.
        prefix: Filename prefix for saved patches.

    Returns:
        List of patch metadata dictionaries:
        [
            {
                "patch_id": 0,
                "x1": 0, "y1": 0, "x2": 640, "y2": 640,
                "patch": np.ndarray (or None if saved),
                "filepath": str | None
            }, ...
        ]
    """
    h, w = image.shape[:2]
    grid_coords = get_slice_grid_coordinates(w, h, patch_size=patch_size, overlap=overlap)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None

    patches_meta: List[Dict[str, Any]] = []

    for i, (x1, y1, x2, y2) in enumerate(grid_coords):
        patch_img = image[y1:y2, x1:x2]

        # In case edge patch is smaller than patch_size, pad with reflection/zeros
        if patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
            padded = np.zeros((patch_size, patch_size), dtype=image.dtype)
            padded[: patch_img.shape[0], : patch_img.shape[1]] = patch_img
            patch_img = padded

        patch_file = None
        if save_path is not None:
            patch_file = str(save_path / f"{prefix}_{i:05d}_x{x1}_y{y1}.png")
            cv2.imwrite(patch_file, patch_img)

        patches_meta.append(
            {
                "patch_id": i,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "patch_width": patch_size,
                "patch_height": patch_size,
                "filepath": patch_file,
                "patch": patch_img if save_path is None else None,
            }
        )

    return patches_meta


# Coordinate Transformation Utilities

def full_to_patch_bbox(
    full_bbox: Tuple[float, float, float, float],
    patch_top_left: Tuple[int, int],
    patch_size: int = 640,
    min_overlap_ratio: float = 0.35,
) -> Optional[Tuple[float, float, float, float]]:
    """Transforms a bounding box from full wafer coordinates to local patch coordinates.

    Args:
        full_bbox: (X1, Y1, X2, Y2) in full wafer space.
        patch_top_left: (px, py) top-left corner of the patch in full wafer space.
        patch_size: Dimensions of the square patch.
        min_overlap_ratio: Minimum ratio of defect area that must fall inside the patch.

    Returns:
        Local (lx1, ly1, lx2, ly2) relative to the patch, or None if outside/insufficiently visible.
    """
    fx1, fy1, fx2, fy2 = full_bbox
    px, py = patch_top_left

    # Defect dimensions & area
    f_area = max(1e-6, (fx2 - fx1) * (fy2 - fy1))

    # Intersection with patch bounds [px, py, px + patch_size, py + patch_size]
    ix1 = max(fx1, float(px))
    iy1 = max(fy1, float(py))
    ix2 = min(fx2, float(px + patch_size))
    iy2 = min(fy2, float(py + patch_size))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    if (inter_area / f_area) < min_overlap_ratio:
        return None

    # Transform to local coordinates
    lx1 = ix1 - px
    ly1 = iy1 - py
    lx2 = ix2 - px
    ly2 = iy2 - py

    return (lx1, ly1, lx2, ly2)


def patch_to_full_bbox(
    patch_bbox: Tuple[float, float, float, float],
    patch_top_left: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Transforms local patch bounding box coordinates back to full wafer coordinates.

    Args:
        patch_bbox: (lx1, ly1, lx2, ly2) in local patch coordinates.
        patch_top_left: (px, py) top-left corner of the patch in full wafer space.

    Returns:
        (X1, Y1, X2, Y2) in full wafer space.
    """
    lx1, ly1, lx2, ly2 = patch_bbox
    px, py = patch_top_left
    return (lx1 + px, ly1 + py, lx2 + px, ly2 + py)


def bbox_to_yolo(
    pixel_bbox: Tuple[float, float, float, float],
    img_size: int = 640,
) -> Tuple[float, float, float, float]:
    """Converts pixel bounding box (x1, y1, x2, y2) to YOLO format (cx, cy, w, h).

    Args:
        pixel_bbox: (x1, y1, x2, y2) in local patch pixels.
        img_size: Patch width and height.

    Returns:
        (cx, cy, w, h) normalized to [0.0, 1.0].
    """
    x1, y1, x2, y2 = pixel_bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)

    # Normalize
    return (
        round(cx / img_size, 6),
        round(cy / img_size, 6),
        round(w / img_size, 6),
        round(h / img_size, 6),
    )


def yolo_to_bbox(
    yolo_bbox: Tuple[float, float, float, float],
    img_size: int = 640,
) -> Tuple[float, float, float, float]:
    """Converts YOLO format (cx, cy, w, h) to pixel bounding box (x1, y1, x2, y2).

    Args:
        yolo_bbox: Normalized (cx, cy, w, h) in [0.0, 1.0].
        img_size: Patch size in pixels.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates.
    """
    cx, cy, w, h = yolo_bbox
    pix_cx = cx * img_size
    pix_cy = cy * img_size
    pix_w = w * img_size
    pix_h = h * img_size

    x1 = pix_cx - (pix_w / 2.0)
    y1 = pix_cy - (pix_h / 2.0)
    x2 = pix_cx + (pix_w / 2.0)
    y2 = pix_cy + (pix_h / 2.0)

    return (x1, y1, x2, y2)
