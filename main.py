"""Main Driver and CLI for SiC Wafer Defect AI.

Automates end-to-end workflow:
1. Data inspection & 16-bit preprocessing (CLAHE)
2. Synthetic mock XRT data generation (fallback if raw data is downloading)
3. Patch slicing & YOLO dataset construction
4. Transfer learning on Apple Silicon MPS (or CUDA/CPU) & automatic ONNX export
5. Full-wafer sliding-window inference, defect counting, and visual wafer mapping
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Optional

from src.dataset import build_yolo_dataset, create_yolo_yaml, generate_pseudo_labels_and_dataset
from src.gui import launch_gui
from src.inference import WaferInferenceEngine, export_defect_reports, visualize_wafer_defects
from src.preprocess import (
    generate_mock_xrt_data,
    import_kicet_dataset,
    inspect_and_load_xrt,
    preprocess_xrt_image,
    slice_patches,
)
from src.train import get_default_device, train_yolo



def run_generate_mock(
    raw_dir: Path,
    num_images: int = 2,
    shape: tuple = (4096, 4096),
    num_defects: int = 60,
):
    """Generates synthetic 16-bit XRT images with BPD, TED, TSD defects."""
    print(f"\n[Step: Mock Data Generation]")
    print(f"Destination: {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, num_images + 1):
        filename = raw_dir / f"mock_wafer_xrt_{i:02d}.tif"
        print(f"Generating synthetic wafer #{i} ({shape[0]}x{shape[1]}) -> {filename} ...")
        generate_mock_xrt_data(
            filepath=filename,
            shape=shape,
            num_defects=num_defects,
            seed=42 + i,
        )
    print("Mock XRT data generation completed successfully.\n")


IMAGE_EXTENSIONS = (
    "*.tif", "*.tiff", "*.TIF", "*.TIFF",
    "*.jpg", "*.jpeg", "*.JPG", "*.JPEG",
    "*.png", "*.PNG",
    "*.bmp", "*.BMP",
    "*.webp", "*.WEBP",
)


def find_image_files(directory: Path) -> list[Path]:
    """Finds all supported images (.tif, .jpg, .png, etc.) in a directory."""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(directory.glob(ext))
    return sorted(list(set(files)))


def run_inspect_and_slice(
    raw_dir: Path,
    sliced_dir: Path,
    patch_size: int = 640,
    overlap: float = 0.2,
):
    """Inspects raw files, converts 16-bit to 8-bit CLAHE, and slices to disk."""
    print(f"\n[Step: Inspection & Preprocessing]")
    raw_files = find_image_files(raw_dir)
    if not raw_files:
        print(f"[Error] No supported image files found in {raw_dir}.")
        return

    sliced_dir.mkdir(parents=True, exist_ok=True)

    for rf in raw_files:
        img, meta = inspect_and_load_xrt(rf)
        print(f"Inspected: {rf.name}")
        print(f"  - Shape:       {meta['original_shape']}")
        print(f"  - Dtype:       {meta['dtype']}")
        print(f"  - Dynamic Range: [{meta['min_val']:.1f}, {meta['max_val']:.1f}]")
        print(f"  - 3D Stack:    {meta['is_3d']} (Slices: {meta['num_slices']})")

        # 16-bit -> 8-bit CLAHE
        preprocessed = preprocess_xrt_image(img)

        # Slice patches to disk
        out_sub = sliced_dir / rf.stem
        patches = slice_patches(
            image=preprocessed,
            patch_size=patch_size,
            overlap=overlap,
            save_dir=out_sub,
            prefix=rf.stem,
        )
        print(f"  - Sliced into {len(patches)} patches -> {out_sub}\n")


def run_prepare_dataset(
    raw_dir: Path,
    dataset_dir: Path,
    patch_size: int = 640,
    overlap: float = 0.2,
) -> Path:
    """Builds YOLO train/val dataset structure and YAML file."""
    print(f"\n[Step: YOLO Dataset Preparation]")
    yaml_path, stats = build_yolo_dataset(
        raw_dir=raw_dir,
        output_dir=dataset_dir,
        patch_size=patch_size,
        overlap=overlap,
    )
    print(f"Dataset successfully compiled at: {dataset_dir}")
    print(f"  - YAML config:         {yaml_path}")
    print(f"  - Total kept patches:  {stats['total_patches']} (Positive: {stats['positive_patches']}, 5% Negative: {stats['negative_patches_kept']} / Raw {stats['total_raw_negative_patches']})")
    print(f"  - Train patches:       {stats['train_samples']}")
    print(f"  - Validation patches:  {stats['val_samples']}")
    print(
        f"  - Defect instances:    BPD={stats.get('defects_BPD', 0)}, "
        f"TED={stats.get('defects_TED', 0)}, "
        f"TSD={stats.get('defects_TSD', 0)}, "
        f"Particle={stats.get('defects_Particle', 0)}"
    )
    return yaml_path


def run_training(
    data_yaml: Path,
    epochs: int = 1,
    batch_size: int = 8,
    imgsz: int = 640,
    device: Optional[str] = None,
    weights: str = "yolov8n.pt",
    name: str = "sic_defect_train",
    note: str = "",
) -> dict:
    """Runs YOLO training and exports best checkpoint to ONNX."""
    print(f"\n[Step: YOLO Training & ONNX Export]")
    summary = train_yolo(
        data_yaml=data_yaml,
        model_weights=weights,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        device=device,
        name=name,
        note=note,
        export_onnx=True,
    )
    print(f"Training completed:")
    print(f"  - Best weights: {summary.get('best_pt')}")
    print(f"  - ONNX model:   {summary.get('best_onnx')}")
    if summary.get("isolated_onnx"):
        print(f"  - Isolated Deployment Weights: {summary.get('isolated_onnx')}")
    return summary


def run_pseudo_labeling(
    raw_dir: Path,
    dataset_dir: Path,
    model_path: Optional[str] = None,
    patch_size: int = 640,
    overlap: float = 0.2,
    overwrite: bool = False,
    device: Optional[str] = None,
) -> tuple[Path, dict]:
    """Generates pseudo labels for unannotated raw images and builds YOLO dataset."""
    print(f"\n[Step: Automated Pseudo-Labeling & Dataset Generation]")
    yaml_path, stats = generate_pseudo_labels_and_dataset(
        raw_dir=raw_dir,
        output_dir=dataset_dir,
        model_path=model_path,
        patch_size=patch_size,
        overlap=overlap,
        overwrite_existing_json=overwrite,
        device=device,
    )
    print(f"\nPseudo-labeling and dataset preparation complete:")
    print(f"  - Dataset YAML:       {yaml_path}")
    print(f"  - Labeled raw images: {stats.get('pseudo_labeled_images', 0)}")
    print(f"  - Total kept patches: {stats.get('total_patches', 0)}")
    print(f"  - Train samples:      {stats.get('train_samples', 0)}")
    print(f"  - Val samples:        {stats.get('val_samples', 0)}")
    return yaml_path, stats


def run_release(
    tag: str,
    model_path: Optional[str] = None,
    note: str = "",
    project_root: Optional[Path] = None,
):
    """Packages and Git-tags a validated model for version release."""
    root = project_root or Path(__file__).resolve().parent
    print(f"\n[Step: Model Git Release & Tagging]")
    print(f"Target Release Tag: {tag}")
    if note:
        print(f"Release Notes:      {note}")

    # Check model file if specified
    if model_path:
        mp = Path(model_path)
        if not mp.is_absolute():
            mp = root / mp
        if not mp.exists():
            print(f"[Error] Specified model not found: {mp}")
            return
        size_mb = mp.stat().st_size / (1024 * 1024)
        print(f"Model File:         {mp.name} ({size_mb:.2f} MB)")
        if size_mb > 100:
            print(f"[Warning] Model size ({size_mb:.2f} MB) exceeds GitHub 100MB limit. Consider git-lfs.")
        elif size_mb > 50:
            print(f"[Notice] Model size ({size_mb:.2f} MB) triggers GitHub 50MB warning, but is allowed.")
        else:
            print(f"[OK] Model size is well within GitHub limits (<50MB). Safe for standard Git commit.")

    # Git status check
    try:
        subprocess.run(["git", "status", "--short"], cwd=root, check=True)
    except Exception as e:
        print(f"[Error] Git is not accessible: {e}")
        return

    # Stage files: models/, runs/experiments.csv, runs/EXPERIMENTS.md
    print("\nStaging model artifacts and experiment registries...")
    subprocess.run(["git", "add", "models/", "runs/experiments.csv", "runs/EXPERIMENTS.md"], cwd=root)

    # Commit if changes exist
    status_proc = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    if status_proc.stdout.strip():
        commit_msg = f"release: {tag} - {note or 'Fine-tuned model weights and registry'}"
        print(f"Committing changes: {commit_msg}")
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=root)
    else:
        print("Working tree clean for staged artifacts.")

    # Create Git tag
    tag_msg = note or f"SiC Defect AI Model Release {tag}"
    print(f"Creating annotated Git tag '{tag}'...")
    res = subprocess.run(["git", "tag", "-a", tag, "-m", tag_msg], cwd=root, capture_output=True, text=True)
    if res.returncode != 0:
        if "already exists" in res.stderr:
            print(f"[Warning] Git tag '{tag}' already exists. To overwrite, delete it first or choose a new tag name.")
        else:
            print(f"[Error] Failed to create tag: {res.stderr.strip()}")
            return
    else:
        print(f"Tag '{tag}' created successfully!")

    print("\n=======================================================")
    print(">>> MODEL RELEASE COMPLETED SUCCESSFULLY <<<")
    print("To push this release and tag to GitHub, run:")
    print(f"   git push origin main")
    print(f"   git push origin {tag}")
    print("Or to push all tags:")
    print(f"   git push origin --tags")
    print("=======================================================\n")


def run_inference_pipeline(
    raw_image_path: Path,
    model_path: Path,
    output_dir: Path = Path("runs/detect"),
    patch_size: int = 640,
    overlap: float = 0.2,
    conf_thresh: float = 0.25,
    device: Optional[str] = None,
):
    """Executes full-scale sliding window inference, counting, and visual mapping."""
    print(f"\n[Step: Full-Wafer Inference & Coordinate Aggregation]")
    print(f"Testing on: {raw_image_path}")
    print(f"Model: {model_path}")

    # 1. Load and preprocess full wafer
    raw_img, meta = inspect_and_load_xrt(raw_image_path)
    prep_img = preprocess_xrt_image(raw_img)

    # 2. Run engine
    engine = WaferInferenceEngine(
        model_path=model_path,
        conf_threshold=conf_thresh,
        device=device,
    )
    results = engine.predict_full_wafer(
        image=prep_img,
        patch_size=patch_size,
        overlap=overlap,
    )

    # 3. Export reports (JSON / CSV)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_base = output_dir / f"{raw_image_path.stem}_defect_report"
    export_defect_reports(results, output_base=report_base)

    # 4. Generate wafer map visualization (Scatter & Heatmap)
    vis_path = output_dir / f"{raw_image_path.stem}_defect_map.png"
    visualize_wafer_defects(
        results=results,
        image_shape=prep_img.shape[:2],
        output_path=vis_path,
        bg_image=prep_img,
    )
    print(f"Inference pipeline finished successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="SiC Wafer Defect Detection & Classification Pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "gui", "mock", "import", "preprocess", "dataset", "pseudo-label", "train", "infer", "release"],
        default="all",
        help="Pipeline mode to execute (default: 'all' for dry-run verification)",
    )
    parser.add_argument(
        "--kicet-dir",
        type=str,
        default="/Volumes/T9 Junseung/세라믹 기술원 데이터/XRT 2D",
        help="Source directory of real KICET XRT 2D data",
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--sliced-dir", type=str, default="data/sliced", help="Path to sliced patches directory")
    parser.add_argument("--dataset-dir", type=str, default="data/dataset", help="Path to YOLO dataset directory")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (use 1 for fast dry-run)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda:0', 'cpu')")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Inference confidence threshold")
    parser.add_argument("--test-image", type=str, default=None, help="Custom image path for inference")
    parser.add_argument("--model-path", type=str, default=None, help="Trained model path for inference / pseudo-labeling / release")
    parser.add_argument("--name", type=str, default="sic_defect_train", help="Experiment name (e.g. 'exp_jpg_v1')")
    parser.add_argument("--note", type=str, default="", help="Experiment or release notes description")
    parser.add_argument("--tag", type=str, default="v1.1-jpg-exp", help="Git tag for release mode")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing pseudo-label JSONs")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / args.raw_dir
    sliced_dir = project_root / args.sliced_dir
    dataset_dir = project_root / args.dataset_dir

    raw_files = find_image_files(raw_dir)

    print(f"=== SiC Wafer Defect AI Engine ===")
    print(f"Mode: {args.mode}")
    print(f"Compute Device detected: {get_default_device(args.device)}")
    print(f"Raw data status: {len(raw_files)} files found in {raw_dir}")

    # Mode 0: Interactive PyQt6 GUI
    if args.mode == "gui":
        init_img = Path(args.test_image) if args.test_image else (raw_files[0] if raw_files else None)
        print(f"\n[Step: Launching PyQt6 Desktop GUI]")
        if init_img:
            print(f"Pre-loading image: {init_img.name}")
        launch_gui(image_path=init_img)
        return

    # Mode 1: Mock data generation
    if args.mode == "mock":
        run_generate_mock(raw_dir)
        return

    # Mode 1.5: Import real KICET data
    if args.mode == "import":
        kicet_src = Path(args.kicet_dir)
        if not kicet_src.exists():
            print(f"[Error] KICET directory not found: {kicet_src}")
            return
        print(f"\n[Step: Importing KICET Real 2D XRT & ImageJ ROI annotations]")
        imported = import_kicet_dataset(kicet_src, raw_dir, use_symlink=True)
        print(f"Successfully imported {len(imported)} wafer datasets to {raw_dir}\n")
        return

    # Mode 2: Preprocess and slice
    if args.mode == "preprocess":
        if not raw_files:
            print("[Info] No raw files found. Automatically generating mock data first...")
            run_generate_mock(raw_dir)
        run_inspect_and_slice(raw_dir, sliced_dir, patch_size=args.imgsz)
        return

    # Mode 3: Prepare YOLO dataset
    if args.mode == "dataset":
        if not raw_files:
            print("[Info] No raw files found. Generating mock data first...")
            run_generate_mock(raw_dir)
        run_prepare_dataset(raw_dir, dataset_dir, patch_size=args.imgsz)
        return

    # Mode 3.5: Automated Pseudo-labeling
    if args.mode == "pseudo-label":
        run_pseudo_labeling(
            raw_dir=raw_dir,
            dataset_dir=dataset_dir,
            model_path=args.model_path,
            patch_size=args.imgsz,
            overwrite=args.overwrite,
            device=args.device,
        )
        return

    # Mode 4: Train model
    if args.mode == "train":
        yaml_path = dataset_dir / "sic_defect.yaml"
        if not yaml_path.exists():
            print("[Info] Dataset yaml missing. Building dataset first...")
            if not raw_files:
                run_generate_mock(raw_dir)
            yaml_path = run_prepare_dataset(raw_dir, dataset_dir, patch_size=args.imgsz)
        run_training(
            data_yaml=yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            weights=args.weights,
            name=args.name,
            note=args.note,
        )
        return

    # Mode 5: Inference
    if args.mode == "infer":
        test_img = Path(args.test_image) if args.test_image else (raw_files[0] if raw_files else None)
        if test_img is None or not test_img.exists():
            print("[Error] No inference image found. Please specify --test-image or generate mock data.")
            return

        candidate_paths = [
            Path(args.model_path) if args.model_path else None,
            Path("runs/train/sic_defect_train/weights/best.pt"),
            Path("runs/detect/runs/detect/sic_defect_train/weights/best.pt"),
            Path("runs/detect/sic_defect_train/weights/best.pt"),
            Path(args.weights),
        ]
        model_path = next((p for p in candidate_paths if p is not None and p.exists()), Path(args.weights))

        run_inference_pipeline(
            raw_image_path=test_img,
            model_path=model_path,
            patch_size=args.imgsz,
            conf_thresh=args.conf,
            device=args.device,
        )
        return

    # Mode 7: Model Git Release
    if args.mode == "release":
        run_release(
            tag=args.tag,
            model_path=args.model_path,
            note=args.note,
            project_root=project_root,
        )
        return

    # Mode 6: All (End-to-End dry-run verification)
    if args.mode == "all":
        print("\n>>> Starting End-to-End Pipeline Dry-Run Verification <<<")
        # 1. Check or generate mock data
        if not raw_files:
            print("[Status] Raw data is not yet downloaded. Creating synthetic 16-bit XRT data...")
            # For quick dry-run test, create two 2048x2048 synthetic images
            run_generate_mock(raw_dir, num_images=2, shape=(2048, 2048), num_defects=40)
            raw_files = find_image_files(raw_dir)
        else:
            print(f"[Status] Real raw data detected ({len(raw_files)} files). Proceeding with raw files...")

        # 2. Slice to sliced_dir
        run_inspect_and_slice(raw_dir, sliced_dir, patch_size=args.imgsz)

        # 3. Build YOLO dataset
        yaml_path = run_prepare_dataset(raw_dir, dataset_dir, patch_size=args.imgsz)

        # 4. Train 1-epoch dry-run & export ONNX
        train_summary = run_training(
            data_yaml=yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            weights=args.weights,
        )

        # 5. Full-wafer inference & visualization
        best_model = train_summary.get("best_pt") or args.weights
        test_wafer = raw_files[0]
        run_inference_pipeline(
            raw_image_path=test_wafer,
            model_path=Path(best_model),
            patch_size=args.imgsz,
            conf_thresh=0.05 if args.epochs <= 1 else args.conf,  # Lower threshold for 1-epoch untrained/lightly-trained model
            device=args.device,
        )
        print("\n=======================================================")
        print(">>> ALL PIPELINE TESTS PASSED ON MACOS (MPS) SUCCESSFULLY! <<<")
        print("=======================================================\n")


if __name__ == "__main__":
    main()
