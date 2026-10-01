# SiC Wafer XRT Defect Detection & Classification AI

> **한국세라믹기술원(KICET) 캡스톤 프로젝트**: SiC 웨이퍼 XRT(X-ray Topography) 영상을 활용한 기저면 및 관통 전위 결함 자동 검출, 미세 분류(BPD, TED, TSD), 전수 카운팅 및 웨이퍼 전면 좌표 매핑 시스템

---

## 1. 프로젝트 개요

- **개발 환경**: macOS (Apple Silicon M-시리즈, `device="mps"` 가속 지원)
- **배포 타겟**: Windows 독립 실행형 데스크톱 응용 프로그램 (`.exe`)
  - GitHub Actions / Windows PC 빌드 환경에서 ONNX Runtime 및 PyInstaller 기반 패키징
- **주요 탐지 대상 결함**:
  1. **BPD (Basal Plane Dislocation)**: 기저면 전위 (길쭉한 선/호 형태, 소자 신뢰성에 치명적)
  2. **TED (Threading Edge Dislocation)**: 관통 칼날 전위 (미세한 점/타원 형태)
  3. **TSD (Threading Screw Dislocation)**: 관통 나선 전위 (뚜렷한 코어 및 콘트라스트를 가진 원형 형태)

---

## 2. 디렉토리 구조

```text
sic-defect-ai/
├── data/
│   ├── raw/                 # 원본 2D XRT 및 3D TIFF 스택 (.tif, .tiff)
│   ├── sliced/              # 원본 영상에서 분할된 640x640 오버랩 패치
│   └── dataset/             # YOLO 형식 학습/검증 데이터셋
│       ├── images/          # train / val 이미지 (.png)
│       ├── labels/          # train / val YOLO 어노테이션 (.txt)
│       └── sic_defect.yaml  # YOLO 데이터셋 설정 파일
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # 16-bit 정규화, CLAHE, 패치 슬라이싱, 양방향 좌표 변환, Mock 데이터 생성기
│   ├── dataset.py           # YOLO 데이터셋 파이프라인 및 YAML 자동 생성
│   ├── train.py             # YOLOv8/v11 전이학습 (MPS/CUDA/CPU) 및 ONNX 자동 Export
│   └── inference.py         # 풀 웨이퍼 슬라이딩 윈도우 추론, Seam NMS, 결함 카운팅, 산점도/히트맵 시각화
├── runs/                    # 학습 가중치 (best.pt, best.onnx) 및 추론 결과 보고서/시각화 맵
├── main.py                  # 통합 실행 CLI 드라이버 스크립트
├── requirements.txt         # 의존성 패키지 목록
└── README.md                # 프로젝트 문서
```

---

## 3. 핵심 기술 및 모듈 구성

### 3.1 16-bit 고해상도 XRT 전처리 & CLAHE (`src/preprocess.py`)
- **16-bit -> 8-bit 안전 변환**: 백분위수(Percentile 0.5% ~ 99.5%) 클리핑 및 Min-Max 스케일링을 통해 극단치 노이즈를 억제하고 딥러닝 백본에 최적화된 8-bit 동적 범위 생성
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: X-ray 회절 특유의 약한 전위 콘트라스트를 국소적으로 증폭
- **Mock XRT 생성기 (`generate_mock_xrt_data`)**: 대용량 실데이터 다운로드 중에도 즉시 파이프라인을 검증할 수 있도록 16-bit 합성 웨이퍼 영상 및 BPD/TED/TSD 인공 결함 자동 생성

### 3.2 고해상도 패치 슬라이싱 & 양방향 좌표 변환 (`src/preprocess.py`)
- 수천~수만 픽셀의 대형 웨이퍼 영상과 미세 결함 크기 차이를 극복하기 위해 **640x640 오버랩(20%) 슬라이딩 윈도우** 적용
- **완벽한 양방향 좌표 매핑**:
  - Full-Wafer $(X, Y) \longleftrightarrow$ Local Patch $(x, y) \longleftrightarrow$ YOLO Normalized $(cx, cy, w, h)$

### 3.3 YOLO 파인튜닝 & ONNX 자동 Export (`src/train.py`)
- Ultralytics YOLOv8 / YOLOv11 전이 학습 지원
- macOS Apple Silicon의 **MPS(`device="mps"`)**를 기본 감지하여 하드웨어 가속 지원 (CUDA 및 CPU 클린 폴백)
- 학습 종료 즉시 배포용 포맷인 `best.onnx`로 자동 변환

### 3.4 전면 웨이퍼 추론 & 결함 통계/시각화 (`src/inference.py`)
- 슬라이딩 윈도우 추론 후 전체 웨이퍼 원본 좌표계로 복원
- 패치 경계면(seam) 중복 검출 제거를 위한 **글로벌 NMS(Non-Maximum Suppression)**
- **결과물 자동 산출**:
  - 클래스별 결함 총 카운트 (BPD, TED, TSD)
  - 결함 상세 정보 리스트 (CSV, JSON)
  - 웨이퍼 전면 시각화 맵:
    - **Defect Scatter Map**: 원형 웨이퍼 경계선 가이드 및 클래스별 결함 위치 마킹
    - **Defect Density Heatmap**: 웨이퍼 내 결함 집중 밀도 분포 시각화

---

## 4. 실행 방법

### 환경 활성화 및 의존성 설치
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 전체 파이프라인 원스톱 검증 (Dry-Run)
`data/raw/`에 데이터가 없으면 가상 16-bit XRT 생성부터 전처리, 1 epoch 학습, ONNX export, 대형 웨이퍼 슬라이딩 추론 및 시각화까지 전 과정을 한 번에 검증합니다:
```bash
python main.py --mode all --epochs 1
```

### 단계별 개별 실행
1. **모의 데이터 생성**:
   ```bash
   python main.py --mode mock
   ```
2. **전처리 및 패치 슬라이싱**:
   ```bash
   python main.py --mode preprocess
   ```
3. **YOLO 데이터셋 구성**:
   ```bash
   python main.py --mode dataset
   ```
4. **모델 학습 및 ONNX 변환**:
   ```bash
   python main.py --mode train --epochs 50 --batch-size 16 --device mps
   ```
5. **풀 웨이퍼 추론 및 리포트/시각화 생성**:
   ```bash
   python main.py --mode infer --test-image data/raw/your_wafer.tif --model-path runs/detect/sic_defect_train/weights/best.pt
   ```

---

## 5. 향후 계획 (Windows EXE 데스크톱 앱 배포)
- 2D 파이프라인 검증 완료 후 3D TIFF 스택(깊이별 전위 추적) 모듈 확장
- PyQt6 / PySide6 기반 데스크톱 UI 제작 (웨이퍼 로드, 실시간 검출, 불량률 판정, 리포트 출력)
- GitHub Actions를 통한 Windows 환경 ONNX Runtime + PyInstaller 무인 빌드 파이프라인 구축
