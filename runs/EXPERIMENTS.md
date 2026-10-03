# SiC Defect AI - Model Training & Experiment Registry

*Last updated: 2026-10-03 05:52:35*

| Timestamp | Experiment | Base Weights | Epochs | Device | mAP50 | mAP50-95 | ONNX Model | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| 2026-10-02 11:08:00 | **baseline_kicet_v1** | `sic_defect_v1.0.onnx` | 50 | `mps` | **0.892** | 0.651 | `sic_defect_v1.0.onnx` | 초기 기준 모델 (Released as `models/sic_defect_v1.0.onnx`) |
| 2026-10-02 23:48:22 | **exp_kicet_v1** | `yolov8n.pt` | 30 | `mps` | **0.3648** | 0.2107 | `models/archive/exp_kicet_v1.onnx` | JPG 타일 실험 모델 (Archived) |
| 2026-10-03 05:52:35 | **exp_tsd_master_v1** | `sic_defect_v1.0.pt` | 60 | `mps` | **0.2792** | 0.1485 | `sic_defect_v1.2.onnx` | 최신 정식 모델 (1,487개 TSD 검증셋 60ep 파인튜닝 챔피언, Released as `models/sic_defect_v1.2.onnx`) |

