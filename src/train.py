"""YOLO Model Training and ONNX Export Pipeline for SiC Defect AI.

Supports Apple Silicon MPS acceleration with fallback to CUDA / CPU,
and automatically exports the best checkpoint to ONNX format upon completion.
"""

from __future__ import annotations

import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Union

import torch
from ultralytics import YOLO


def get_default_device(requested_device: Optional[str] = None) -> str:
    """Detects and returns the optimal available PyTorch device.

    Priority:
    1. User explicitly requested device (if valid).
    2. Apple Silicon MPS ('mps') if available on macOS.
    3. NVIDIA CUDA ('cuda:0') if available.
    4. CPU fallback ('cpu').
    """
    if requested_device is not None and requested_device.strip() != "":
        req = requested_device.strip().lower()
        if req == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            print("[Warning] MPS requested but not available. Falling back.")
        elif req.startswith("cuda"):
            if torch.cuda.is_available():
                return req
            print(f"[Warning] {req} requested but CUDA not available. Falling back.")
        elif req == "cpu":
            return "cpu"
        else:
            return requested_device

    # Auto detection
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def log_experiment_summary(
    exp_name: str,
    base_weights: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    model_onnx: Optional[str],
    model_pt: Optional[str],
    metrics: Optional[Dict[str, Any]] = None,
    note: str = "",
    log_dir: Union[str, Path] = "runs",
) -> Path:
    """Logs experiment run details and validation metrics to CSV and Markdown tables."""
    ldir = Path(log_dir)
    ldir.mkdir(parents=True, exist_ok=True)
    csv_path = ldir / "experiments.csv"
    md_path = ldir / "EXPERIMENTS.md"

    metrics = metrics or {}
    map50 = round(float(metrics.get("map50", 0.0)), 4)
    map50_95 = round(float(metrics.get("map50_95", 0.0)), 4)
    precision = round(float(metrics.get("precision", 0.0)), 4)
    recall = round(float(metrics.get("recall", 0.0)), 4)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Append to CSV
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Experiment_Name",
                "Base_Weights",
                "Epochs",
                "Batch_Size",
                "Image_Size",
                "Device",
                "Model_ONNX",
                "Model_PT",
                "mAP50",
                "mAP50_95",
                "Precision",
                "Recall",
                "Notes",
            ])
        writer.writerow([
            timestamp,
            exp_name,
            base_weights,
            epochs,
            batch_size,
            imgsz,
            device,
            model_onnx or "N/A",
            model_pt or "N/A",
            map50,
            map50_95,
            precision,
            recall,
            note,
        ])

    # 2. Re-render Markdown summary table
    try:
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                for row in reader:
                    if row:
                        rows.append(row)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# SiC Defect AI - Model Training & Experiment Registry\n\n")
            f.write(f"*Last updated: {timestamp}*\n\n")
            f.write("| Timestamp | Experiment | Base Weights | Epochs | Device | mAP50 | mAP50-95 | ONNX Model | Notes |\n")
            f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |\n")
            for r in rows:
                ts = r[0] if len(r) > 0 else ""
                en = r[1] if len(r) > 1 else ""
                bw = Path(r[2]).name if len(r) > 2 else ""
                ep = r[3] if len(r) > 3 else ""
                dev = r[6] if len(r) > 6 else ""
                m50 = r[9] if len(r) > 9 else ""
                m95 = r[10] if len(r) > 10 else ""
                mo = Path(r[7]).name if len(r) > 7 else ""
                nt = r[13] if len(r) > 13 else ""
                f.write(f"| {ts} | **{en}** | `{bw}` | {ep} | `{dev}` | **{m50}** | {m95} | `{mo}` | {nt} |\n")
    except Exception as e:
        print(f"[Warning] Failed to generate {md_path}: {e}")

    print(f"[Registry] Experiment '{exp_name}' logged to {csv_path} & {md_path}")
    return csv_path


def train_yolo(
    data_yaml: Union[str, Path],
    model_weights: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: Optional[str] = None,
    project: str = "runs/train",
    name: str = "sic_defect_train",
    export_onnx: bool = True,
    seed: int = 42,
    note: str = "",
) -> Dict[str, Any]:
    """Trains a YOLO object detection model on the SiC defect dataset.

    Args:
        data_yaml: Path to dataset YAML configuration file.
        model_weights: Pretrained model weights (e.g. 'yolov8n.pt', 'yolov8s.pt').
        epochs: Number of training epochs.
        batch_size: Training batch size.
        imgsz: Image resolution.
        device: Target compute device ('mps', 'cuda:0', 'cpu', or None for auto).
        project: Save directory project path.
        name: Experiment run name.
        export_onnx: Whether to export best.pt to best.onnx after training.
        seed: Random seed.
        note: Descriptive note regarding this experiment run.

    Returns:
        Dictionary containing paths to best.pt, best.onnx, models/ directory outputs, and training metrics.
    """
    chosen_device = get_default_device(device)
    yaml_path = Path(data_yaml).resolve()

    if not yaml_path.exists():
        raise FileNotFoundError(f"Data configuration YAML not found: {yaml_path}")

    print(f"==================================================")
    print(f"[Train] Initializing YOLO Transfer Learning Pipeline")
    print(f"  - Experiment Name: {name}")
    print(f"  - Model Backbone:  {model_weights}")
    print(f"  - Dataset Config:  {yaml_path}")
    print(f"  - Target Device:   {chosen_device}")
    print(f"  - Epochs:          {epochs}")
    print(f"  - Batch Size:      {batch_size}")
    print(f"  - Image Size:      {imgsz}")
    if note:
        print(f"  - Note:            {note}")
    print(f"==================================================")

    # Load YOLO architecture
    model = YOLO(model_weights)

    # Train model
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=chosen_device,
        project=project,
        name=name,
        seed=seed,
        workers=4,
        cache=True,
        verbose=True,
        save=True,
        exist_ok=True,
    )

    # Locate best.pt
    save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else Path(project) / name
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = save_dir / "weights" / "last.pt"

    # Extract validation metrics
    metrics: Dict[str, Any] = {}
    if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
        metrics["map50"] = float(results.results_dict.get("metrics/mAP50(B)", 0.0))
        metrics["map50_95"] = float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))
        metrics["precision"] = float(results.results_dict.get("metrics/precision(B)", 0.0))
        metrics["recall"] = float(results.results_dict.get("metrics/recall(B)", 0.0))
    elif hasattr(results, "box"):
        metrics["map50"] = float(getattr(results.box, "map50", 0.0))
        metrics["map50_95"] = float(getattr(results.box, "map", 0.0))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Isolated model paths in models/
    isolated_pt = models_dir / f"{name}.pt"
    isolated_onnx = models_dir / f"{name}.onnx"

    if best_pt.exists():
        try:
            shutil.copy2(best_pt, isolated_pt)
            print(f"[Model Registry] Saved checkpoint to: {isolated_pt}")
        except Exception as e:
            print(f"[Warning] Failed to copy {best_pt} to {isolated_pt}: {e}")

    output_summary: Dict[str, Any] = {
        "experiment_name": name,
        "save_dir": str(save_dir),
        "best_pt": str(best_pt) if best_pt.exists() else None,
        "models_pt": str(isolated_pt) if isolated_pt.exists() else None,
        "best_onnx": None,
        "models_onnx": None,
        "device": chosen_device,
        "metrics": metrics,
    }

    # Automatic ONNX Export
    if export_onnx and best_pt.exists():
        print(f"\n[Export] Exporting best model ({best_pt}) to ONNX format for deployment...")
        trained_model = YOLO(str(best_pt))
        try:
            onnx_path = trained_model.export(
                format="onnx",
                imgsz=imgsz,
                dynamic=False,  # Fixed shape recommended for ONNX Runtime desktop apps
                opset=12,
            )
            output_summary["best_onnx"] = str(onnx_path)
            print(f"[Export] Successfully generated ONNX model: {onnx_path}")

            # 1. Copy to models/{name}.onnx for isolated experiment tracking
            try:
                shutil.copy2(onnx_path, isolated_onnx)
                output_summary["models_onnx"] = str(isolated_onnx)
                print(f"[Model Registry] Saved isolated ONNX model: {isolated_onnx}")
            except Exception as e:
                print(f"[Warning] Could not copy to {isolated_onnx}: {e}")

            # 2. Update root best.onnx as convenience pointer
            try:
                shutil.copy2(onnx_path, Path("best.onnx"))
            except Exception:
                pass

            # 3. Ensure baseline models/best_2d_base.onnx exists
            base_model = models_dir / "best_2d_base.onnx"
            if not base_model.exists():
                try:
                    shutil.copy2(onnx_path, base_model)
                    print(f"[Model Registry] Initialized baseline: {base_model}")
                except Exception:
                    pass

        except Exception as e:
            print(f"[Export Error] Failed to export ONNX: {e}")

    # Register experiment metrics in CSV and Markdown
    log_experiment_summary(
        exp_name=name,
        base_weights=model_weights,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        device=chosen_device,
        model_onnx=output_summary.get("models_onnx") or output_summary.get("best_onnx"),
        model_pt=output_summary.get("models_pt") or output_summary.get("best_pt"),
        metrics=metrics,
        note=note,
    )

    return output_summary
