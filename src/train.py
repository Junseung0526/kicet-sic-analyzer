"""YOLO Model Training and ONNX Export Pipeline for SiC Defect AI.

Supports Apple Silicon MPS acceleration with fallback to CUDA / CPU,
and automatically exports the best checkpoint to ONNX format upon completion.
"""

from __future__ import annotations

import os
from pathlib import Path
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

    Returns:
        Dictionary containing paths to best.pt, best.onnx, and training metrics.
    """
    chosen_device = get_default_device(device)
    yaml_path = Path(data_yaml).resolve()

    if not yaml_path.exists():
        raise FileNotFoundError(f"Data configuration YAML not found: {yaml_path}")

    print(f"==================================================")
    print(f"[Train] Initializing YOLO Transfer Learning Pipeline")
    print(f"  - Model Backbone: {model_weights}")
    print(f"  - Dataset Config: {yaml_path}")
    print(f"  - Target Device:  {chosen_device}")
    print(f"  - Epochs:         {epochs}")
    print(f"  - Batch Size:     {batch_size}")
    print(f"  - Image Size:     {imgsz}")
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
        verbose=True,
        save=True,
        exist_ok=True,
    )

    # Locate best.pt
    save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else Path(project) / name
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        # Fallback to last.pt if best.pt is not created (e.g. 1 epoch quick dry-run)
        best_pt = save_dir / "weights" / "last.pt"

    output_summary: Dict[str, Any] = {
        "save_dir": str(save_dir),
        "best_pt": str(best_pt) if best_pt.exists() else None,
        "best_onnx": None,
        "device": chosen_device,
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
        except Exception as e:
            print(f"[Export Error] Failed to export ONNX: {e}")

    return output_summary
