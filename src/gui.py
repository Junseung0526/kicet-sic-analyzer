"""Cross-Platform PyQt6 Desktop Application for SiC Wafer Defect Detection.

Engineered for macOS (Apple Silicon) & Windows Standalone EXE Deployment:
- Pure Pathlib cross-platform path handling.
- Lightweight ONNX Runtime inference core with background QThread processing.
- Interactive QGraphicsView canvas with cursor-centered smooth zooming and drag-panning.
- Real-time 16-bit dynamic range (Contrast/Brightness) adjustment and 3D TIFF stack slider.
- Class-specific overlay toggling (BPD, TED, TSD) and report exporter.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from src.inference import (
    CLASS_NAMES,
    DEFAULT_TSD_ANGLE_BINS,
    DEFAULT_TED_ANGLE_BINS,
    analyze_domain_measurements,
    calculate_tsd_angle,
    calculate_ted_angle,
    classify_tsd_angle,
    classify_ted_angle,
    get_inference_engine,
    measure_bpd_length,
)
from src.preprocess import (
    inspect_and_load_xrt,
    normalize_16bit_to_8bit,
    preprocess_xrt_image,
)

# Color Scheme for Defect Classes
DEFECT_COLORS = {
    "BPD": QColor(0, 180, 255, 230),       # Cyan-Blue
    "TED": QColor(255, 220, 0, 230),       # Bright Yellow
    "TSD": QColor(255, 40, 60, 230),        # Vibrant Red
    "Particle": QColor(179, 136, 255, 230), # Soft Purple / Violet
}

# Multilingual Translation Dictionary (KO / EN)
TRANSLATIONS = {
    "KO": {
        "window_title": "KICET SiC 웨이퍼 결함 검출 및 분류 시스템 (ONNX Runtime)",
        "lang_button": "🌐 English (EN)",
        "slice_label": "슬라이스: {cur}/{tot}",
        "brightness": "밝기:",
        "contrast": "대비:",
        "fit_view": "화면 맞춤",
        "zoom_100": "100% (1:1)",
        "tab_overview": "📊 개요 (Overview)",
        "tab_analysis": "📐 정밀 계측 (Analysis)",
        "tab_grid": "🧭 3x3 구역 (Wafer Grid)",
        "grp_file": "파일 및 모델 설정",
        "btn_open": "📂 이미지 열기 (.tif, .jpg, .png 등)",
        "no_image": "불러온 이미지 없음",
        "file_info": "파일명: {name}\n크기: {w} x {h} px\n타입: {dtype} (범위: [{min_val:.0f}, {max_val:.0f}])",
        "model_info": "모델: {name}",
        "btn_change_model": "가중치 선택",
        "grp_analysis": "자동 결함 분석",
        "btn_run": "⚡ 웨이퍼 전수 분석 실행 (ONNX)",
        "status_ready": "준비 완료",
        "status_scanning": "백그라운드에서 웨이퍼 전수 스캔 중...",
        "status_inferring": "패치 추론 중: {cur} / {tot} ({pct}%)",
        "status_complete": "완료: 결함 {total}개 검출됨 (결정 {crystal}개 / 이물질 {particle}개).",
        "status_failed": "분석 실패.",
        "status_loaded": "이미지를 성공적으로 불러왔습니다.",
        "grp_results": "결함 요약 및 클래스 오버레이",
        "table_headers": ["클래스", "개수", "색상"],
        "row_bpd": "BPD (기저면 전위)",
        "row_ted": "TED (관통 칼날 전위)",
        "row_tsd": "TSD (관통 나선 전위)",
        "row_crystal_subtotal": "결정 결함 소계",
        "row_particle": "이물질 / 먼지 (Particle)",
        "row_grand_total": "전체 검출 총합",
        "chk_bpd": "BPD",
        "chk_ted": "TED",
        "chk_tsd": "TSD",
        "chk_particle": "Particle (이물질)",
        "btn_export": "💾 보고서 내보내기 (CSV / JSON)",
        "chk_grid_overlay": "3x3 구역 격자선 표시",
        "grid_title": "웨이퍼 3x3 구역 분할 및 결함 밀도",
        "grid_intro": "구역 블록을 클릭하면 해당 1/9 영역으로 화면이 포커스됩니다.",
        "grid_hotspot": "🔥 최고 밀집 구역: {zone} (결함 {count}개, {pct}%)",
        "grid_no_data": "분석 실행 후 9개 구역별 결함 수와 밀도가 집계됩니다.",
        "grid_headers": ["구역 (Zone)", "방위", "결함 수", "점유율"],
        "slice_depth_range": "층 범위:",
        "pixel_scale_label": "웨이퍼 해상도 (Pixel Scale):",
        "analysis_bpd_title": "BPD 선 결함 길이 분석 (Dislocation Length, μm)",
        "analysis_bpd_desc": "주성분 분석(PCA) 기반 BPD 결함 장축 길이 측정 및 μm 환산",
        "analysis_bpd_summary": "BPD 결함 수: {cnt}개 | 평균: {mean} μm | 최대: {max} μm | 최소: {min} μm",
        "analysis_bpd_headers": ["ID", "중심 (X, Y)", "길이 (px)", "길이 (μm)"],
        "analysis_bpd_btn": "📈 BPD 길이 분포 히스토그램 생성",
        "analysis_conversion_title": "BPD → TED 열처리 전위 전환율 (Conversion Rate)",
        "analysis_conversion_desc": "고온 열처리(Annealing) 전/후 BPD 결함의 TED 전환율 산출",
        "lbl_before_bpd": "열처리 전 BPD 수:",
        "lbl_after_bpd": "열처리 후 잔여 BPD 수:",
        "btn_apply_current_bpd": "📥 현재 BPD 검출값 적용",
        "lbl_converted_ted": "전환된 TED 수:",
        "lbl_conversion_rate": "전환율 (Conversion Rate):",
        "conversion_formula": "공식: Conversion Rate (%) = (Before BPD - After BPD) / (Before BPD) × 100",
        "analysis_ted_title": "TED 6방향 분류 (TED Component 6-Type, <11-20>)",
        "analysis_ted_desc": "4H-SiC 버거스 벡터 6방향(60° 간격) 명암 벡터 자동 판별 및 점유율",
        "analysis_ted_headers": ["라벨", "중심 (°)", "범위 (°)", "결함 수", "점유율 (%)"],
        "btn_reset_ted_bins": "🔄 기본 6방향 (a~f)",
        "ted_gauge_title": "TED 6방향 점유율",
        "analysis_tsd_title": "TSD 4방향 판별 및 커스텀 각도 룰셋 (Propagation Angle)",
        "analysis_tsd_desc": "명암 모멘트 0~360° 중심 각도 산출 및 연구원 커스텀 각도 구간 룰셋",
        "analysis_tsd_headers": ["라벨 (Label)", "시작 각도 (°)", "종료 각도 (°)", "결함 수", "비율 (%)"],
        "btn_add_bin": "➕ 구간 추가",
        "btn_del_bin": "➖ 구간 삭제",
        "btn_reset_bins": "🔄 기본 4방향 (A~D)",
        "btn_apply_bins": "⚡ 각도 룰 적용",
        "bpd_spectrum_title": "BPD 길이 구간별 스펙트럼",
        "bpd_short": "단파 (<30μm)",
        "bpd_mid": "중파 (30~70μm)",
        "bpd_long": "장파 (>70μm)",
        "tsd_gauge_title": "TSD 각도 구간별 점유율",
        "tooltip_open": "X-선 토포그래피(XRT) 웨이퍼 영상 열기 (단축키: O)",
        "tooltip_model": "사용할 ONNX 가중치 모델 파일 선택",
        "tooltip_run": "대형 웨이퍼 결함 전수 자동 검출 실행 (단축키: R)",
        "tooltip_export": "검출 결과 및 도메인 정밀 계측 보고서 저장 (CSV / JSON)",
        "tooltip_brightness": "16-bit 영상 밝기 미세 조정 (-100 ~ +100)",
        "tooltip_contrast": "16-bit 영상 명암 대비 조절 (20% ~ 300%)",
        "tooltip_fit": "전체 웨이퍼 영상을 화면에 맞춤 (단축키: F)",
        "tooltip_100": "1:1 원본 100% 픽셀 크기로 줌 리셋",
        "tooltip_grid": "웨이퍼 3x3 9분할 격자선 표시 토글 (단축키: G)",
        "tooltip_scale": "실측 웨이퍼 해상도(μm/px) 설정 (BPD 길이 실시간 재계산)",
        "tooltip_bpd_hist": "BPD 선 결함 길이 분포 히스토그램 창 열기",
        "tooltip_before_bpd": "열처리 전(Before Annealing) BPD 총 개수 입력",
        "tooltip_after_bpd": "열처리 후(After Annealing) 잔여 BPD 개수 입력",
        "tooltip_apply_current_bpd": "현재 화면에서 검출된 BPD 개수를 계산기에 자동 입력",
        "tooltip_reset_ted_bins": "TED 6방향(a~f, 60° 간격) 프리셋으로 복원",
        "tooltip_depth_start": "3D XRT 스택 탐색 시작 레이어 (1~N)",
        "tooltip_depth_end": "3D XRT 스택 탐색 종료 레이어 (1~N)",
        "tooltip_add_bin": "새로운 사용자 정의 TSD 각도 판별 구간 추가",
        "tooltip_del_bin": "선택한 각도 판별 구간 삭제",
        "tooltip_reset_bins": "기본 4방향(A~D) 4분면 룰셋으로 복원",
        "tooltip_apply_bins": "설정한 각도 룰셋을 즉시 적용하여 결함 재분류",
        "sb_no_file": "📁 파일 없음",
        "loading_scanning": "대형 웨이퍼 결함 전수 추론 중...",
        "loading_image": "초고해상도 웨이퍼 영상 로딩 중...",
        "dialog_open_image": "SiC XRT 웨이퍼 이미지 선택",
        "dialog_select_model": "모델 가중치 선택",
        "dialog_save_report": "결함 보고서 저장",
        "export_success_title": "내보내기 완료",
        "export_success_msg": "보고서가 성공적으로 저장되었습니다:\n{path}",
        "warning_no_image_title": "이미지 없음",
        "warning_no_image_msg": "분석을 실행하기 전에 XRT 이미지를 열어주세요.",
        "warning_no_model_title": "모델 가중치 없음",
        "warning_no_model_msg": "모델 가중치를 찾을 수 없습니다:\n{path}\n학습을 진행하거나 가중치를 선택해주세요.",
        "load_error_title": "불러오기 오류",
        "load_error_msg": "이미지를 여는데 실패했습니다:\n{e}",
        "infer_error_title": "추론 오류",
        "infer_error_msg": "결함 검출 중 오류가 발생했습니다:\n{e}",
        "btn_3d_viewer": "🧊 3D 입체 뷰어",
        "tooltip_3d_viewer": "웨이퍼 3D 다중 단면 스택을 마우스로 자유롭게 XYZ 360° 회전하며 입체 탐색합니다.",
        "viewer_3d_title": "KICET SiC 웨이퍼 3D 입체 단면 및 결함 뷰어 (3D Volumetric Explorer)",
        "viewer_3d_guide": "💡 마우스 조작:\n• 좌클릭 드래그: XYZ 360° 자유 회전\n• 우클릭/휠: 줌 확대 및 축소\n• 휠 클릭 드래그: 3D 위치 이동",
        "viewer_3d_grp_camera": "3D 시점 및 제어 (Camera)",
        "viewer_3d_iso": "🔄 3D 입체 (Isometric)",
        "viewer_3d_top": "⬆️ 상단 (Top / XY)",
        "viewer_3d_front": "➡️ 정면 (Front / XZ)",
        "viewer_3d_side": "📐 측면 (Side / YZ)",
        "viewer_3d_grp_slice": "단면 층 제어 (Depth Layer / Z-Slice)",
        "viewer_3d_slice_info": "현재 층: {cur} / {tot} ({depth:.1f} μm)",
        "viewer_3d_prev": "◀ 이전 층",
        "viewer_3d_next": "다음 층 ▶",
        "viewer_3d_sync": "📥 메인 화면에 이 층 적용",
        "viewer_3d_grp_display": "표시 모드 및 렌더링 (Display)",
        "viewer_3d_mode_solid": "🧊 3D 솔리드 체적 큐브 (Solid Connected Block)",
        "viewer_3d_mode_triplanar": "📐 3D 직교 3단면 연결 탐색 (Tri-Planar: XY+XZ+YZ)",
        "viewer_3d_mode_dense": "🌫️ 연속 고밀도 체적 투시 (Dense Volume Stack)",
        "viewer_3d_mode_single": "📄 단일 직교 단면 (Single Ortho-Slice)",
        "viewer_3d_coronal_y": "횡단면 (Coronal Y):",
        "viewer_3d_sagittal_x": "종단면 (Sagittal X):",
        "viewer_3d_colormap": "컬러맵:",
        "viewer_3d_opacity": "단면 불투명도:",
        "viewer_3d_zscale": "Z축 두께 강조:",
        "viewer_3d_grp_defects": "3D 결함 분포 (3D Defects)",
        "viewer_3d_chk_defects": "3D 결함 마커 표시 (Show 3D Defects)",
        "viewer_3d_defect_stats": "검출 결함: 총 {total}개\n• BPD: {bpd}개 (하늘색 원)\n• TED: {ted}개 (노란 삼각)\n• TSD: {tsd}개 (빨간 네모)\n• Particle: {part}개 (보라 다이아)",
        "viewer_3d_close": "닫기 (Close)",
    },
    "EN": {
        "window_title": "KICET SiC Wafer Defect Detection & Classification (ONNX Runtime)",
        "lang_button": "🌐 한국어 (KO)",
        "slice_label": "Slice: {cur}/{tot}",
        "slice_depth_range": "Depth Range:",
        "brightness": "Brightness:",
        "contrast": "Contrast:",
        "fit_view": "Fit to View",
        "zoom_100": "100% (1:1)",
        "tab_overview": "📊 Overview",
        "tab_analysis": "📐 Analysis",
        "tab_grid": "🧭 Wafer Grid",
        "grp_file": "File & Model Configuration",
        "btn_open": "📂 Open Image (.tif, .jpg, .png, etc.)",
        "no_image": "No image loaded",
        "file_info": "File: {name}\nSize: {w} x {h} px\nType: {dtype} (Range: [{min_val:.0f}, {max_val:.0f}])",
        "model_info": "Model: {name}",
        "btn_change_model": "Select Weights",
        "grp_analysis": "Automated Defect Analysis",
        "btn_run": "⚡ Run Full-Wafer Analysis (ONNX)",
        "status_ready": "Ready",
        "status_scanning": "Scanning full wafer in background...",
        "status_inferring": "Inferring patches: {cur} / {tot} ({pct}%)",
        "status_complete": "Complete: {total} defects detected (Crystal {crystal} / Particle {particle}).",
        "status_failed": "Analysis failed.",
        "status_loaded": "Image loaded successfully.",
        "grp_results": "Defect Summary & Class Overlays",
        "table_headers": ["Class", "Count", "Color"],
        "row_bpd": "BPD (Basal Plane)",
        "row_ted": "TED (Threading Edge)",
        "row_tsd": "TSD (Threading Screw)",
        "row_crystal_subtotal": "Crystal Defects Subtotal",
        "row_particle": "Particle / Artifact",
        "row_grand_total": "Grand Total",
        "chk_bpd": "BPD",
        "chk_ted": "TED",
        "chk_tsd": "TSD",
        "chk_particle": "Particle",
        "btn_export": "💾 Export Report (CSV / JSON)",
        "chk_grid_overlay": "Show 3x3 Grid Overlay",
        "grid_title": "3x3 Regional Defect Distribution & Focus",
        "grid_intro": "Click any zone block to focus canvas onto that 1/9 area.",
        "grid_hotspot": "🔥 Peak Defect Density Hotspot: {zone} ({count} defects, {pct}%)",
        "grid_no_data": "Run defect analysis to compute regional density map.",
        "grid_headers": ["Zone", "Region", "Count", "Ratio"],
        "pixel_scale_label": "Pixel Scale Resolution:",
        "analysis_bpd_title": "BPD Linear Defect Length Profiling (PCA, μm)",
        "analysis_bpd_desc": "Principal Component Analysis (PCA) based BPD major-axis length & μm conversion",
        "analysis_bpd_summary": "BPD Count: {cnt} | Mean: {mean} μm | Max: {max} μm | Min: {min} μm",
        "analysis_bpd_headers": ["ID", "Center (X, Y)", "Length (px)", "Length (μm)"],
        "analysis_bpd_btn": "📈 Generate BPD Length Histogram",
        "analysis_conversion_title": "BPD → TED Annealing Conversion Rate",
        "analysis_conversion_desc": "Calculate BPD to TED dislocation conversion rate before & after annealing",
        "lbl_before_bpd": "Before Annealing BPD:",
        "lbl_after_bpd": "After Annealing BPD:",
        "btn_apply_current_bpd": "📥 Apply Current BPD",
        "lbl_converted_ted": "Converted TED Count:",
        "lbl_conversion_rate": "Conversion Rate:",
        "conversion_formula": "Formula: Conversion Rate (%) = (Before BPD - After BPD) / (Before BPD) × 100",
        "analysis_ted_title": "TED 6-Way Directional Classification (<11-20>)",
        "analysis_ted_desc": "4H-SiC Burgers vector 6-direction (60° interval) contrast vector classification",
        "analysis_ted_headers": ["Label", "Center (°)", "Range (°)", "Count", "Ratio (%)"],
        "btn_reset_ted_bins": "🔄 Reset 6-Type (a~f)",
        "ted_gauge_title": "TED 6-Way Occupancy",
        "analysis_tsd_title": "TSD 4-Way & Custom Angle Rule Sets (Propagation Angle)",
        "analysis_tsd_desc": "Image moments 0~360° centroid angle calculation & custom angle bin rules",
        "analysis_tsd_headers": ["Label", "Start Angle (°)", "End Angle (°)", "Count", "Ratio (%)"],
        "btn_add_bin": "➕ Add Bin",
        "btn_del_bin": "➖ Delete Bin",
        "btn_reset_bins": "🔄 Reset 4-Way (A~D)",
        "btn_apply_bins": "⚡ Apply Rules",
        "bpd_spectrum_title": "BPD Length Spectrum Distribution",
        "bpd_short": "Short (<30μm)",
        "bpd_mid": "Medium (30-70μm)",
        "bpd_long": "Long (>70μm)",
        "tsd_gauge_title": "TSD Angle Rule Bin Occupancy",
        "tooltip_open": "Open SiC XRT Wafer Image (Hotkey: O)",
        "tooltip_model": "Select ONNX / PT Model Weights",
        "tooltip_run": "Run Full-Wafer Defect Analysis (Hotkey: R)",
        "tooltip_export": "Export Defect Reports to CSV / JSON",
        "tooltip_brightness": "Adjust Display Brightness (-100 to +100)",
        "tooltip_contrast": "Adjust Display Contrast (20% to 300%)",
        "tooltip_fit": "Fit Entire Wafer to Viewport (Hotkey: F)",
        "tooltip_100": "Reset Zoom to 100% (1:1 Pixel Scale)",
        "tooltip_grid": "Toggle 3x3 Grid Overlay (Hotkey: G)",
        "tooltip_scale": "Wafer Pixel Scale Resolution Calibration (μm/px)",
        "tooltip_bpd_hist": "Open BPD Length Histogram Dialog",
        "tooltip_before_bpd": "Enter total BPD count before annealing",
        "tooltip_after_bpd": "Enter residual BPD count after annealing",
        "tooltip_apply_current_bpd": "Auto-fill currently detected BPD count into calculator",
        "tooltip_reset_ted_bins": "Reset to default TED 6-type preset (a~f, 60° intervals)",
        "tooltip_depth_start": "3D XRT Stack Exploration Start Layer (1~N)",
        "tooltip_depth_end": "3D XRT Stack Exploration End Layer (1~N)",
        "tooltip_add_bin": "Add New Custom TSD Angle Bin",
        "tooltip_del_bin": "Delete Selected TSD Angle Bin",
        "tooltip_reset_bins": "Reset to Standard 4-Way Angle Quadrants",
        "tooltip_apply_bins": "Apply Custom Angle Rules and Reclassify Defects",
        "sb_no_file": "📁 No File",
        "loading_scanning": "Scanning full wafer defects...",
        "loading_image": "Loading high-resolution wafer image...",
        "dialog_open_image": "Select SiC XRT Wafer Image",
        "dialog_select_model": "Select Model Weights",
        "dialog_save_report": "Save Defect Report",
        "export_success_title": "Export Success",
        "export_success_msg": "Report saved successfully:\n{path}",
        "warning_no_image_title": "No Image",
        "warning_no_image_msg": "Please open an XRT image before running analysis.",
        "warning_no_model_title": "Model Missing",
        "warning_no_model_msg": "Model weights not found at:\n{path}\nPlease train the model or select weights.",
        "load_error_title": "Load Error",
        "load_error_msg": "Failed to open image:\n{e}",
        "infer_error_title": "Inference Error",
        "infer_error_msg": "Error during defect detection:\n{e}",
        "btn_3d_viewer": "🧊 3D Volume Viewer",
        "tooltip_3d_viewer": "Inspect multi-slice wafer stack with interactive 3D XYZ mouse orbit rotation.",
        "viewer_3d_title": "KICET SiC Wafer 3D Stereoscopic Section & Defect Viewer",
        "viewer_3d_guide": "💡 Mouse Control:\n• Left-Drag: Rotate XYZ 360° freely\n• Right-Drag / Wheel: Zoom In / Out\n• Middle-Drag: Pan in 3D space",
        "viewer_3d_grp_camera": "3D View & Camera Presets",
        "viewer_3d_iso": "🔄 3D Isometric",
        "viewer_3d_top": "⬆️ Top View (XY)",
        "viewer_3d_front": "➡️ Front View (XZ)",
        "viewer_3d_side": "📐 Side View (YZ)",
        "viewer_3d_grp_slice": "Depth Layer Navigation (Z-Slice)",
        "viewer_3d_slice_info": "Current Layer: {cur} / {tot} ({depth:.1f} μm)",
        "viewer_3d_prev": "◀ Prev Layer",
        "viewer_3d_next": "Next Layer ▶",
        "viewer_3d_sync": "📥 Sync Layer to Main View",
        "viewer_3d_grp_display": "Display Mode & Rendering",
        "viewer_3d_mode_solid": "🧊 3D Solid Connected Block (Volumetric Cut)",
        "viewer_3d_mode_triplanar": "📐 3D Orthogonal Tri-Planar (XY+XZ+YZ)",
        "viewer_3d_mode_dense": "🌫️ Continuous Dense Volume Stack",
        "viewer_3d_mode_single": "📄 Single Ortho-Slice",
        "viewer_3d_coronal_y": "Coronal Cut (Y):",
        "viewer_3d_sagittal_x": "Sagittal Cut (X):",
        "viewer_3d_colormap": "Colormap:",
        "viewer_3d_opacity": "Slice Opacity:",
        "viewer_3d_zscale": "Z-Thickness Scale:",
        "viewer_3d_grp_defects": "3D Defect Distribution",
        "viewer_3d_chk_defects": "Show 3D Defect Scatter Markers",
        "viewer_3d_defect_stats": "Defects: Total {total}\n• BPD: {bpd} (Cyan circles)\n• TED: {ted} (Yellow triangles)\n• TSD: {tsd} (Red squares)\n• Particle: {part} (Purple diamonds)",
        "viewer_3d_close": "Close",
    },

}


# ==============================================================================
# Background Inference Worker (QThread)
# ==============================================================================

class InferenceWorker(QThread):
    """Background worker executing sliding-window wafer defect detection without freezing UI."""

    progress_signal = pyqtSignal(int, int)  # current, total
    finished_signal = pyqtSignal(dict)      # inference results
    error_signal = pyqtSignal(str)          # error message

    def __init__(
        self,
        image_array: np.ndarray,
        model_path: Path,
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        patch_size: int = 640,
        overlap: float = 0.2,
    ):
        super().__init__()
        self.image_array = image_array
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.patch_size = patch_size
        self.overlap = overlap

    def run(self):
        try:
            # 1. Preprocess 16-bit to 8-bit CLAHE
            preprocessed = preprocess_xrt_image(self.image_array)

            # 2. Get optimal engine (ONNX Runtime prioritized)
            engine = get_inference_engine(
                model_path=self.model_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                prefer_onnx=True,
            )

            # 3. Sliding-window inference with progress callback
            def on_progress(cur: int, tot: int):
                self.progress_signal.emit(cur, tot)

            results = engine.predict_full_wafer(
                image=preprocessed,
                patch_size=self.patch_size,
                overlap=self.overlap,
                progress_callback=on_progress,
            )
            self.finished_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))


# ==============================================================================
# Precision Analysis Histogram Dialog
# ==============================================================================

class BPDHistogramDialog(QDialog):
    """Modal dialog displaying BPD dislocation length distribution histogram."""

    def __init__(self, parent: Optional[QWidget], lengths_um: List[float], lang: str = "KO"):
        super().__init__(parent)
        self.setWindowTitle("BPD 길이 분포 히스토그램" if lang == "KO" else "BPD Length Distribution Histogram")
        self.resize(620, 460)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        fig = Figure(figsize=(6.2, 4.2), dpi=100, facecolor="#1e1e24")
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas, stretch=1)

        ax = fig.add_subplot(111)
        ax.set_facecolor("#252830")

        if lengths_um:
            data = np.array(lengths_um, dtype=np.float32)
            bins_count = min(20, max(5, int(np.sqrt(len(data)))))
            ax.hist(
                data,
                bins=bins_count,
                color="#00b4ff",
                edgecolor="#1e1e24",
                alpha=0.85,
                rwidth=0.9,
            )
            mean_val = float(np.mean(data))
            median_val = float(np.median(data))
            ax.axvline(mean_val, color="#ff5722", linestyle="--", linewidth=2.0, label=f"Mean: {mean_val:.1f} μm")
            ax.axvline(median_val, color="#00e676", linestyle=":", linewidth=2.0, label=f"Median: {median_val:.1f} μm")
            ax.legend(facecolor="#252830", edgecolor="#454c59", labelcolor="#ffffff")

        title_str = "BPD Length Distribution (μm)"
        xlabel_str = "Length (μm)"
        ylabel_str = "Defect Count"

        ax.set_title(title_str, color="#ffffff", fontsize=12, pad=10, fontweight="bold")
        ax.set_xlabel(xlabel_str, color="#dcdcdc", fontsize=10)
        ax.set_ylabel(ylabel_str, color="#dcdcdc", fontsize=10)
        ax.tick_params(colors="#a0a0a0")
        ax.grid(True, linestyle=":", alpha=0.35, color="#555566")

        for spine in ax.spines.values():
            spine.set_color("#454c59")

        fig.tight_layout()
        canvas.draw()

        btn_close = QPushButton("닫기 (Close)" if lang == "KO" else "Close")
        btn_close.setFixedHeight(32)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class Wafer3DVolumeViewerDialog(QDialog):
    """Interactive 3D volumetric inspection dialog for SiC multi-slice XRT stacks.

    Features:
    - Complete 3D volume stack pre-loaded into memory ("한번에 다 불러와서").
    - Authentic Monochrome/Grayscale XRT diffraction contrast (no artificial purple!).
    - Mode 1: 3D Solid Connected Wafer Block (윗면, 아랫면, 4면 외벽이 실제 이미지로 물리적 연결된 솔리드 체적 및 절단).
    - Mode 2: 3D Orthogonal Tri-Planar Slicer (수평 XY + 수직 XZ + 수직 YZ 3면 수직 교차 절단면 탐색).
    - Mode 3: Continuous Dense Volumetric Stack (32개 층 반투명 체적 연속 투시).
    - Mode 4: Single Ortho-Slice plane.
    - Free 3D XYZ orbit rotation via mouse left-click drag (azimuth & elevation).
    - Mouse zoom (right-click drag / wheel) and pan (middle-click drag).
    - 3D Wafer bounding box wireframe with micrometer scale.
    - 3D defect scatter positioning (BPD, TED, TSD, Particle).
    - Quick view orientation presets (Isometric, Top, Front, Side).
    - Dynamic Z-axis thickness exaggeration presets (1.0x, 2.0x, 3.5x, 5.0x).
    """

    slice_selected = pyqtSignal(int)

    def __init__(
        self,
        parent: Optional[QWidget],
        tiff_reader: Optional[Any],
        raw_data: Optional[np.ndarray],
        num_slices: int,
        initial_slice_idx: int = 0,
        defects: Optional[List[Dict[str, Any]]] = None,
        pixel_size_um: float = 3.2,
        lang: str = "KO",
    ):
        super().__init__(parent)
        self.tiff_reader = tiff_reader
        self.raw_data = raw_data
        self.num_slices = max(1, num_slices)
        self.current_slice_idx = min(self.num_slices - 1, max(0, initial_slice_idx))
        self.defects = defects or []
        self.pixel_size_um = float(pixel_size_um)
        self.lang = lang if lang in TRANSLATIONS else "KO"

        # Load complete 3D volume stack at once into memory
        self.volume_3d: Optional[np.ndarray] = None
        if self.tiff_reader is not None:
            try:
                self.volume_3d = self.tiff_reader.asarray()
            except Exception:
                self.volume_3d = None
        if self.volume_3d is None and self.raw_data is not None and self.raw_data.ndim == 3:
            self.volume_3d = self.raw_data

        # Determine dimensions
        if self.volume_3d is not None:
            self.num_slices = int(self.volume_3d.shape[0])
            self.wafer_h = int(self.volume_3d.shape[1])
            self.wafer_w = int(self.volume_3d.shape[2])
        else:
            self.wafer_w, self.wafer_h = self._probe_wafer_dimensions()

        # Tri-planar cut coordinates
        self.cur_x = self.wafer_w // 2
        self.cur_y = self.wafer_h // 2

        # Internal rendering properties - DEFAULT IS REAL GRAYSCALE (No artificial purple!)
        self._slice_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self.mesh_res: int = 48
        self.active_colormap: str = "gray"
        self.active_opacity: float = 0.88
        self.active_mode: str = "solid_block"  # "solid_block", "tri_planar", "dense_stack", "single_slice"
        self.show_defects: bool = True
        self.z_scale_factor: float = 2.0
        self.elev: float = 25.0
        self.azim: float = -55.0

        # Window setup
        self.setWindowTitle(self._tr("viewer_3d_title"))
        self.resize(1180, 840)
        self.setMinimumSize(1000, 720)

        # Debounce timer for smooth slider dragging
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._render_3d_scene)

        self._init_ui()
        self._render_3d_scene()

    def _tr(self, key: str, **kwargs) -> str:
        text = TRANSLATIONS.get(self.lang, TRANSLATIONS["KO"]).get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except Exception:
                return text
        return text

    def _probe_wafer_dimensions(self) -> Tuple[int, int]:
        if self.tiff_reader is not None and len(self.tiff_reader.pages) > 0:
            h, w = self.tiff_reader.pages[0].shape[:2]
            return int(w), int(h)
        if self.raw_data is not None:
            if self.raw_data.ndim == 3:
                h, w = self.raw_data.shape[1:3]
            else:
                h, w = self.raw_data.shape[:2]
            return int(w), int(h)
        return 2895, 2922

    def _norm8(self, arr: np.ndarray) -> np.ndarray:
        """Min-max normalizes array to uint8."""
        if arr is None or arr.size == 0:
            return np.zeros((10, 10), dtype=np.uint8)
        mn = float(arr.min())
        mx = float(arr.max())
        if mx > mn:
            return ((arr.astype(np.float32) - mn) / (mx - mn) * 255.0).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    def get_slice_8bit(self, idx: int, res: int) -> np.ndarray:
        cache_key = (idx, res)
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]

        raw = None
        if self.volume_3d is not None and idx < len(self.volume_3d):
            raw = self.volume_3d[idx]
        elif self.tiff_reader is not None and idx < len(self.tiff_reader.pages):
            raw = self.tiff_reader.pages[idx].asarray()
        elif self.raw_data is not None and idx < len(self.raw_data):
            raw = self.raw_data[idx]

        if raw is None:
            small = np.zeros((res, res), dtype=np.uint8)
        else:
            small_raw = cv2.resize(raw, (res, res), interpolation=cv2.INTER_AREA)
            small = self._norm8(small_raw)

        self._slice_cache[cache_key] = small
        return small

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ----------------------------------------------------------------------
        # LEFT: Scrollable Control Sidebar
        # ----------------------------------------------------------------------
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(370)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(4, 4, 8, 4)
        sidebar_layout.setSpacing(12)

        # 1. 3D Camera Controls Group
        grp_camera = QGroupBox(self._tr("viewer_3d_grp_camera"))
        grp_camera_layout = QVBoxLayout(grp_camera)
        grp_camera_layout.setSpacing(8)

        lbl_guide = QLabel(self._tr("viewer_3d_guide"))
        lbl_guide.setStyleSheet("color: #00b4ff; font-size: 11px; line-height: 1.4;")
        lbl_guide.setWordWrap(True)
        grp_camera_layout.addWidget(lbl_guide)

        grid_presets = QGridLayout()
        grid_presets.setSpacing(6)

        self.btn_iso = QPushButton(self._tr("viewer_3d_iso"))
        self.btn_iso.clicked.connect(self._preset_iso)
        grid_presets.addWidget(self.btn_iso, 0, 0)

        self.btn_top = QPushButton(self._tr("viewer_3d_top"))
        self.btn_top.clicked.connect(self._preset_top)
        grid_presets.addWidget(self.btn_top, 0, 1)

        self.btn_front = QPushButton(self._tr("viewer_3d_front"))
        self.btn_front.clicked.connect(self._preset_front)
        grid_presets.addWidget(self.btn_front, 1, 0)

        self.btn_side = QPushButton(self._tr("viewer_3d_side"))
        self.btn_side.clicked.connect(self._preset_side)
        grid_presets.addWidget(self.btn_side, 1, 1)

        grp_camera_layout.addLayout(grid_presets)

        self.lbl_angles = QLabel(f"시점: Elev {self.elev:.0f}°, Azim {self.azim:.0f}°")
        self.lbl_angles.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold;")
        grp_camera_layout.addWidget(self.lbl_angles)

        sidebar_layout.addWidget(grp_camera)

        # 2. Slice Depth Navigation Group
        grp_slice = QGroupBox(self._tr("viewer_3d_grp_slice"))
        grp_slice_layout = QVBoxLayout(grp_slice)
        grp_slice_layout.setSpacing(8)

        est_depth_um = (self.current_slice_idx + 1) * self.pixel_size_um
        self.lbl_slice_info = QLabel(
            self._tr("viewer_3d_slice_info", cur=self.current_slice_idx + 1, tot=self.num_slices, depth=est_depth_um)
        )
        self.lbl_slice_info.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 12px;")
        grp_slice_layout.addWidget(self.lbl_slice_info)

        # Slider & SpinBox row
        row_slice = QHBoxLayout()
        self.slider_slice = QSlider(Qt.Orientation.Horizontal)
        self.slider_slice.setRange(0, self.num_slices - 1)
        self.slider_slice.setValue(self.current_slice_idx)
        self.slider_slice.valueChanged.connect(self._on_slider_changed)
        row_slice.addWidget(self.slider_slice, stretch=1)

        self.spin_slice = QSpinBox()
        self.spin_slice.setRange(1, self.num_slices)
        self.spin_slice.setValue(self.current_slice_idx + 1)
        self.spin_slice.setFixedWidth(64)
        self.spin_slice.valueChanged.connect(self._on_spin_changed)
        row_slice.addWidget(self.spin_slice)
        grp_slice_layout.addLayout(row_slice)

        # Navigation step buttons
        row_nav = QHBoxLayout()
        self.btn_prev = QPushButton(self._tr("viewer_3d_prev"))
        self.btn_prev.clicked.connect(self._prev_slice)
        row_nav.addWidget(self.btn_prev)

        self.btn_next = QPushButton(self._tr("viewer_3d_next"))
        self.btn_next.clicked.connect(self._next_slice)
        row_nav.addWidget(self.btn_next)
        grp_slice_layout.addLayout(row_nav)

        # Sync button
        self.btn_sync = QPushButton(self._tr("viewer_3d_sync"))
        self.btn_sync.setStyleSheet("""
            QPushButton {
                background-color: #00838f;
                color: #ffffff;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #00acc1;
            }
        """)
        self.btn_sync.clicked.connect(self._on_sync_clicked)
        grp_slice_layout.addWidget(self.btn_sync)

        sidebar_layout.addWidget(grp_slice)

        # 3. Tri-Planar Cross-Cut Controls Group (Coronal Y & Sagittal X)
        self.grp_triplanar = QGroupBox(self._tr("viewer_3d_mode_triplanar"))
        grp_tri_layout = QVBoxLayout(self.grp_triplanar)
        grp_tri_layout.setSpacing(6)

        self.lbl_y_cut = QLabel(f"{self._tr('viewer_3d_coronal_y')} {self.cur_y} px")
        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.setRange(0, self.wafer_h - 1)
        self.slider_y.setValue(self.cur_y)
        self.slider_y.valueChanged.connect(self._on_y_cut_changed)
        grp_tri_layout.addWidget(self.lbl_y_cut)
        grp_tri_layout.addWidget(self.slider_y)

        self.lbl_x_cut = QLabel(f"{self._tr('viewer_3d_sagittal_x')} {self.cur_x} px")
        self.slider_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_x.setRange(0, self.wafer_w - 1)
        self.slider_x.setValue(self.cur_x)
        self.slider_x.valueChanged.connect(self._on_x_cut_changed)
        grp_tri_layout.addWidget(self.lbl_x_cut)
        grp_tri_layout.addWidget(self.slider_x)

        sidebar_layout.addWidget(self.grp_triplanar)
        self.grp_triplanar.setVisible(False)  # Only visible when Tri-Planar mode selected

        # 4. Display Mode & Appearance Group
        grp_display = QGroupBox(self._tr("viewer_3d_grp_display"))
        grp_display_layout = QVBoxLayout(grp_display)
        grp_display_layout.setSpacing(8)

        # Mode Selector
        self.combo_mode = QComboBox()
        self.combo_mode.addItem(self._tr("viewer_3d_mode_solid"), "solid_block")
        self.combo_mode.addItem(self._tr("viewer_3d_mode_triplanar"), "tri_planar")
        self.combo_mode.addItem(self._tr("viewer_3d_mode_dense"), "dense_stack")
        self.combo_mode.addItem(self._tr("viewer_3d_mode_single"), "single_slice")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        grp_display_layout.addWidget(self.combo_mode)

        # Colormap row - PURE MONOCHROME GRAYSCALE FIRST (No purple!)
        row_cmap = QHBoxLayout()
        row_cmap.addWidget(QLabel(self._tr("viewer_3d_colormap")))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItem("원본 흑백 (Grayscale)", "gray")
        self.combo_cmap.addItem("반전 흑백 (Inverted)", "gray_r")
        self.combo_cmap.addItem("X-선 고대비 (Bone)", "bone")
        self.combo_cmap.addItem("학술 Cividis", "cividis")
        self.combo_cmap.addItem("컬러 심도 (Viridis)", "viridis")
        self.combo_cmap.setCurrentIndex(0)
        self.combo_cmap.currentIndexChanged.connect(self._on_colormap_changed)
        row_cmap.addWidget(self.combo_cmap, stretch=1)
        grp_display_layout.addLayout(row_cmap)

        # Opacity row
        row_opacity = QHBoxLayout()
        row_opacity.addWidget(QLabel(self._tr("viewer_3d_opacity")))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.slider_opacity.setValue(int(self.active_opacity * 100))
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        row_opacity.addWidget(self.slider_opacity, stretch=1)
        grp_display_layout.addLayout(row_opacity)

        # Z-Scale Exaggeration row
        row_zscale = QHBoxLayout()
        row_zscale.addWidget(QLabel(self._tr("viewer_3d_zscale")))
        self.combo_zscale = QComboBox()
        self.combo_zscale.addItem("1.0x (1:1)", 1.0)
        self.combo_zscale.addItem("2.0x (권장)", 2.0)
        self.combo_zscale.addItem("3.5x", 3.5)
        self.combo_zscale.addItem("5.0x", 5.0)
        self.combo_zscale.setCurrentIndex(1)
        self.combo_zscale.currentIndexChanged.connect(self._on_zscale_changed)
        row_zscale.addWidget(self.combo_zscale, stretch=1)
        grp_display_layout.addLayout(row_zscale)

        # Detail Resolution row
        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("메시 해상도:" if self.lang == "KO" else "Mesh Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItem("고속 (48x48)", 48)
        self.combo_res.addItem("표준 (64x64)", 64)
        self.combo_res.addItem("정밀 (80x80)", 80)
        self.combo_res.setCurrentIndex(0)
        self.combo_res.currentIndexChanged.connect(self._on_res_changed)
        row_res.addWidget(self.combo_res, stretch=1)
        grp_display_layout.addLayout(row_res)

        sidebar_layout.addWidget(grp_display)

        # 5. 3D Defect Scatter Group
        grp_defects = QGroupBox(self._tr("viewer_3d_grp_defects"))
        grp_defects_layout = QVBoxLayout(grp_defects)
        grp_defects_layout.setSpacing(6)

        self.chk_defects = QCheckBox(self._tr("viewer_3d_chk_defects"))
        self.chk_defects.setChecked(self.show_defects)
        self.chk_defects.stateChanged.connect(self._on_defects_toggled)
        grp_defects_layout.addWidget(self.chk_defects)

        # Defect Counts
        cnt_bpd = sum(1 for d in self.defects if d.get("class_name") == "BPD")
        cnt_ted = sum(1 for d in self.defects if d.get("class_name") == "TED")
        cnt_tsd = sum(1 for d in self.defects if d.get("class_name") == "TSD")
        cnt_part = sum(1 for d in self.defects if d.get("class_name") == "Particle")
        total_cnt = len(self.defects)

        self.lbl_defect_stats = QLabel(
            self._tr("viewer_3d_defect_stats", total=total_cnt, bpd=cnt_bpd, ted=cnt_ted, tsd=cnt_tsd, part=cnt_part)
        )
        self.lbl_defect_stats.setStyleSheet("color: #b0bec5; font-size: 11px; line-height: 1.4;")
        grp_defects_layout.addWidget(self.lbl_defect_stats)

        sidebar_layout.addWidget(grp_defects)

        sidebar_layout.addStretch(1)

        # Close button
        btn_close = QPushButton(self._tr("viewer_3d_close"))
        btn_close.setFixedHeight(34)
        btn_close.setStyleSheet("background-color: #37474f; color: #ffffff; font-weight: bold; border-radius: 4px;")
        btn_close.clicked.connect(self.accept)
        sidebar_layout.addWidget(btn_close)

        sidebar_scroll.setWidget(sidebar_content)
        main_layout.addWidget(sidebar_scroll)

        # ----------------------------------------------------------------------
        # RIGHT: Matplotlib 3D Stereoscopic Canvas
        # ----------------------------------------------------------------------
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8.5, 7.5), facecolor="#18181b")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#18181b")
        self.ax.view_init(elev=self.elev, azim=self.azim)

        # Connect mouse release to update angle readout
        self.canvas.mpl_connect("button_release_event", self._on_canvas_released)

        canvas_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(canvas_container, stretch=1)

    def _draw_wireframe_box(self, w: int, h: int, d: int):
        top_z = max(1, d - 1)
        # Bottom rectangle (z=0)
        self.ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], [0, 0, 0, 0, 0],
                     color="#4a90e2", linestyle="--", linewidth=0.9, alpha=0.55)
        # Top rectangle (z=top_z)
        self.ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], [top_z, top_z, top_z, top_z, top_z],
                     color="#4a90e2", linestyle="--", linewidth=0.9, alpha=0.55)
        # 4 Vertical Pillars
        for x0, y0 in [(0, 0), (w, 0), (w, h), (0, h)]:
            self.ax.plot([x0, x0], [y0, y0], [0, top_z],
                         color="#4a90e2", linestyle="--", linewidth=0.9, alpha=0.55)

    def _draw_defect_scatter(self):
        bpd_x, bpd_y, bpd_z = [], [], []
        ted_x, ted_y, ted_z = [], [], []
        tsd_x, tsd_y, tsd_z = [], [], []
        part_x, part_y, part_z = [], [], []

        cur_z = float(self.current_slice_idx)
        for d in self.defects:
            bbox = d.get("bbox", [0, 0, 0, 0])
            cx = float(d.get("center_x", (bbox[0] + bbox[2]) / 2.0))
            cy = float(d.get("center_y", (bbox[1] + bbox[3]) / 2.0))
            cz = float(d.get("slice_idx", cur_z))
            cname = d.get("class_name", "")

            if cname == "BPD":
                bpd_x.append(cx); bpd_y.append(cy); bpd_z.append(cz)
            elif cname == "TED":
                ted_x.append(cx); ted_y.append(cy); ted_z.append(cz)
            elif cname == "TSD":
                tsd_x.append(cx); tsd_y.append(cy); tsd_z.append(cz)
            elif cname == "Particle":
                part_x.append(cx); part_y.append(cy); part_z.append(cz)

        if bpd_x:
            self.ax.scatter(bpd_x, bpd_y, bpd_z, c="#00b4ff", s=40, marker="o",
                            edgecolors="white", linewidths=0.5, alpha=0.9, label=f"BPD ({len(bpd_x)})")
        if ted_x:
            self.ax.scatter(ted_x, ted_y, ted_z, c="#ffd700", s=35, marker="^",
                            edgecolors="black", linewidths=0.5, alpha=0.9, label=f"TED ({len(ted_x)})")
        if tsd_x:
            self.ax.scatter(tsd_x, tsd_y, tsd_z, c="#ff283c", s=45, marker="s",
                            edgecolors="white", linewidths=0.5, alpha=0.9, label=f"TSD ({len(tsd_x)})")
        if part_x:
            self.ax.scatter(part_x, part_y, part_z, c="#b388ff", s=30, marker="d",
                            edgecolors="black", linewidths=0.5, alpha=0.8, label=f"Particle ({len(part_x)})")

        if any([bpd_x, ted_x, tsd_x, part_x]):
            self.ax.legend(loc="upper right", facecolor="#252830", edgecolor="#454c59",
                           labelcolor="#ffffff", fontsize=8)

    def _render_3d_scene(self):
        # Save camera angle
        if hasattr(self.ax, "elev") and hasattr(self.ax, "azim"):
            cur_elev = self.ax.elev if self.ax.elev is not None else self.elev
            cur_azim = self.ax.azim if self.ax.azim is not None else self.azim
        else:
            cur_elev, cur_azim = self.elev, self.azim

        self.ax.cla()
        self.ax.set_facecolor("#18181b")
        self.fig.patch.set_facecolor("#18181b")
        self.ax.xaxis.set_pane_color((0.12, 0.12, 0.16, 0.7))
        self.ax.yaxis.set_pane_color((0.12, 0.12, 0.16, 0.7))
        self.ax.zaxis.set_pane_color((0.12, 0.12, 0.16, 0.7))
        self.ax.grid(True, color="#383848", linestyle=":", alpha=0.5)
        self.ax.tick_params(colors="#c0c0c0", labelsize=8)
        self.ax.xaxis.label.set_color("#e0e0e0")
        self.ax.yaxis.label.set_color("#e0e0e0")
        self.ax.zaxis.label.set_color("#e0e0e0")

        w, h, d = self.wafer_w, self.wafer_h, self.num_slices
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_zlim(0, max(1, d - 1))
        self.ax.set_box_aspect([1.0, 1.0, 0.40 * self.z_scale_factor])

        # 1. 3D Wafer Bounding Box Wireframe
        self._draw_wireframe_box(w, h, d)

        # 2. Render Selected Mode
        res = self.mesh_res
        cmap = plt.get_cmap(self.active_colormap)
        cur_z = int(self.current_slice_idx)

        if self.active_mode == "solid_block":
            # Mode 1: Solid 3D Connected Wafer Volume & Cutaway
            # Top face at cur_z (shows real image slice at current layer)
            top_raw = self.volume_3d[cur_z] if self.volume_3d is not None else self.get_slice_8bit(cur_z, res)
            top_img = self._norm8(cv2.resize(top_raw, (res, res), interpolation=cv2.INTER_AREA))
            X_top, Y_top = np.meshgrid(np.linspace(0, w, res), np.linspace(0, h, res))
            Z_top = np.full_like(X_top, float(cur_z))
            self.ax.plot_surface(X_top, Y_top, Z_top, facecolors=cmap(top_img.astype(np.float32) / 255.0),
                                 shade=False, antialiased=False)

            # Bottom face at 0
            bot_raw = self.volume_3d[0] if self.volume_3d is not None else self.get_slice_8bit(0, res)
            bot_img = self._norm8(cv2.resize(bot_raw, (res, res), interpolation=cv2.INTER_AREA))
            Z_bot = np.full_like(X_top, 0.0)
            self.ax.plot_surface(X_top, Y_top, Z_bot, facecolors=cmap(bot_img.astype(np.float32) / 255.0),
                                 shade=False, antialiased=False)

            # Front wall (Y=0, Z from 0 to cur_z)
            if self.volume_3d is not None and cur_z >= 1:
                front_raw = self.volume_3d[:cur_z + 1, 0, :]
                front_img = self._norm8(cv2.resize(front_raw, (res, res), interpolation=cv2.INTER_AREA))
                X_front, Z_front = np.meshgrid(np.linspace(0, w, res), np.linspace(0, float(cur_z), res))
                Y_front = np.full_like(X_front, 0.0)
                self.ax.plot_surface(X_front, Y_front, Z_front, facecolors=cmap(front_img.astype(np.float32) / 255.0),
                                     shade=False, antialiased=False)

                # Left wall (X=0, Z from 0 to cur_z)
                left_raw = self.volume_3d[:cur_z + 1, :, 0]
                left_img = self._norm8(cv2.resize(left_raw, (res, res), interpolation=cv2.INTER_AREA))
                Y_left, Z_left = np.meshgrid(np.linspace(0, h, res), np.linspace(0, float(cur_z), res))
                X_left = np.full_like(Y_left, 0.0)
                self.ax.plot_surface(X_left, Y_left, Z_left, facecolors=cmap(left_img.astype(np.float32) / 255.0),
                                     shade=False, antialiased=False)

            # Highlight active cut border
            self.ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], [cur_z, cur_z, cur_z, cur_z, cur_z],
                         color="#00e5ff", linestyle="-", linewidth=1.6, alpha=0.95)

        elif self.active_mode == "tri_planar":
            # Mode 2: Orthogonal Tri-Planar Slicer (XY + XZ + YZ)
            # 1. Axial Plane (XY at cur_z)
            xy_raw = self.volume_3d[cur_z] if self.volume_3d is not None else self.get_slice_8bit(cur_z, res)
            xy_img = self._norm8(cv2.resize(xy_raw, (res, res), interpolation=cv2.INTER_AREA))
            X_xy, Y_xy = np.meshgrid(np.linspace(0, w, res), np.linspace(0, h, res))
            Z_xy = np.full_like(X_xy, float(cur_z))
            self.ax.plot_surface(X_xy, Y_xy, Z_xy, facecolors=cmap(xy_img.astype(np.float32) / 255.0),
                                 shade=False, alpha=0.90, antialiased=False)

            # 2. Coronal Plane (XZ at cur_y)
            if self.volume_3d is not None:
                xz_raw = self.volume_3d[:, self.cur_y, :]
                xz_img = self._norm8(cv2.resize(xz_raw, (res, res), interpolation=cv2.INTER_AREA))
                X_xz, Z_xz = np.meshgrid(np.linspace(0, w, res), np.linspace(0, float(d - 1), res))
                Y_xz = np.full_like(X_xz, float(self.cur_y))
                self.ax.plot_surface(X_xz, Y_xz, Z_xz, facecolors=cmap(xz_img.astype(np.float32) / 255.0),
                                     shade=False, alpha=0.90, antialiased=False)

                # 3. Sagittal Plane (YZ at cur_x)
                yz_raw = self.volume_3d[:, :, self.cur_x]
                yz_img = self._norm8(cv2.resize(yz_raw, (res, res), interpolation=cv2.INTER_AREA))
                Y_yz, Z_yz = np.meshgrid(np.linspace(0, h, res), np.linspace(0, float(d - 1), res))
                X_yz = np.full_like(Y_yz, float(self.cur_x))
                self.ax.plot_surface(X_yz, Y_yz, Z_yz, facecolors=cmap(yz_img.astype(np.float32) / 255.0),
                                     shade=False, alpha=0.90, antialiased=False)

            # Colored intersection guides
            self.ax.plot([0, w], [self.cur_y, self.cur_y], [cur_z, cur_z], color="#00e5ff", linewidth=1.5, alpha=0.9)
            self.ax.plot([self.cur_x, self.cur_x], [0, h], [cur_z, cur_z], color="#00e676", linewidth=1.5, alpha=0.9)
            self.ax.plot([self.cur_x, self.cur_x], [self.cur_y, self.cur_y], [0, d - 1], color="#ffd600", linewidth=1.5, alpha=0.9)

        elif self.active_mode == "dense_stack":
            # Mode 3: Continuous Dense Volumetric Stack (32 layers with blending)
            num_layers = min(32, d)
            levels = np.unique(np.linspace(0, d - 1, num_layers, dtype=int))
            X_st, Y_st = np.meshgrid(np.linspace(0, w, res), np.linspace(0, h, res))
            for lvl in levels:
                z_k = float(lvl)
                Z_st = np.full_like(X_st, z_k)
                s_raw = self.volume_3d[lvl] if self.volume_3d is not None else self.get_slice_8bit(lvl, res)
                s_img = self._norm8(cv2.resize(s_raw, (res, res), interpolation=cv2.INTER_AREA))
                colors = cmap(s_img.astype(np.float32) / 255.0)
                is_current = (lvl == cur_z)
                colors[..., 3] = min(0.9, self.active_opacity) if is_current else max(0.08, self.active_opacity * 0.18)
                self.ax.plot_surface(X_st, Y_st, Z_st, facecolors=colors, shade=False, antialiased=False)
                if is_current:
                    self.ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], [z_k, z_k, z_k, z_k, z_k],
                                 color="#00e5ff", linestyle="-", linewidth=1.6, alpha=0.95)

        else:
            # Mode 4: Single Ortho-Slice
            s_raw = self.volume_3d[cur_z] if self.volume_3d is not None else self.get_slice_8bit(cur_z, res)
            s_img = self._norm8(cv2.resize(s_raw, (res, res), interpolation=cv2.INTER_AREA))
            X_s, Y_s = np.meshgrid(np.linspace(0, w, res), np.linspace(0, h, res))
            Z_s = np.full_like(X_s, float(cur_z))
            colors = cmap(s_img.astype(np.float32) / 255.0)
            colors[..., 3] = self.active_opacity
            self.ax.plot_surface(X_s, Y_s, Z_s, facecolors=colors, shade=False, antialiased=False)
            self.ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], [cur_z, cur_z, cur_z, cur_z, cur_z],
                         color="#00e5ff", linestyle="-", linewidth=1.6, alpha=0.95)

        # 3. Defect Scatter Markers
        if self.show_defects and self.defects:
            self._draw_defect_scatter()

        # Labels
        self.ax.set_xlabel("X (px)", fontsize=8, labelpad=4)
        self.ax.set_ylabel("Y (px)", fontsize=8, labelpad=4)
        self.ax.set_zlabel("Z (Layer)", fontsize=8, labelpad=4)

        # Restore view angle
        self.ax.view_init(elev=cur_elev, azim=cur_azim)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._update_angle_label()

    def _update_angle_label(self):
        elev = self.ax.elev if self.ax.elev is not None else self.elev
        azim = self.ax.azim if self.ax.azim is not None else self.azim
        self.lbl_angles.setText(f"시점: Elev {elev:.0f}°, Azim {azim:.0f}°")

    def _on_canvas_released(self, event):
        self._update_angle_label()

    def _preset_iso(self):
        self.ax.view_init(elev=25, azim=-55)
        self.canvas.draw_idle()
        self._update_angle_label()

    def _preset_top(self):
        self.ax.view_init(elev=90, azim=-90)
        self.canvas.draw_idle()
        self._update_angle_label()

    def _preset_front(self):
        self.ax.view_init(elev=0, azim=-90)
        self.canvas.draw_idle()
        self._update_angle_label()

    def _preset_side(self):
        self.ax.view_init(elev=0, azim=0)
        self.canvas.draw_idle()
        self._update_angle_label()

    def _on_slider_changed(self, val: int):
        self.current_slice_idx = val
        self.spin_slice.blockSignals(True)
        self.spin_slice.setValue(val + 1)
        self.spin_slice.blockSignals(False)
        self._update_slice_info()
        self.render_timer.start(25)

    def _on_spin_changed(self, val: int):
        idx = val - 1
        self.current_slice_idx = idx
        self.slider_slice.blockSignals(True)
        self.slider_slice.setValue(idx)
        self.slider_slice.blockSignals(False)
        self._update_slice_info()
        self.render_timer.start(25)

    def _prev_slice(self):
        if self.current_slice_idx > 0:
            self.slider_slice.setValue(self.current_slice_idx - 1)

    def _next_slice(self):
        if self.current_slice_idx < self.num_slices - 1:
            self.slider_slice.setValue(self.current_slice_idx + 1)

    def _update_slice_info(self):
        est_depth_um = (self.current_slice_idx + 1) * self.pixel_size_um
        self.lbl_slice_info.setText(
            self._tr("viewer_3d_slice_info", cur=self.current_slice_idx + 1, tot=self.num_slices, depth=est_depth_um)
        )

    def _on_y_cut_changed(self, val: int):
        self.cur_y = val
        self.lbl_y_cut.setText(f"{self._tr('viewer_3d_coronal_y')} {self.cur_y} px")
        self.render_timer.start(25)

    def _on_x_cut_changed(self, val: int):
        self.cur_x = val
        self.lbl_x_cut.setText(f"{self._tr('viewer_3d_sagittal_x')} {self.cur_x} px")
        self.render_timer.start(25)

    def _on_sync_clicked(self):
        self.slice_selected.emit(self.current_slice_idx)
        self.btn_sync.setText("✅ 동기화 완료" if self.lang == "KO" else "✅ Synced")
        QTimer.singleShot(1200, lambda: self.btn_sync.setText(self._tr("viewer_3d_sync")))

    def _on_mode_changed(self):
        self.active_mode = self.combo_mode.currentData()
        self.grp_triplanar.setVisible(self.active_mode == "tri_planar")
        self._render_3d_scene()

    def _on_colormap_changed(self):
        self.active_colormap = self.combo_cmap.currentData()
        self._render_3d_scene()

    def _on_opacity_changed(self, val: int):
        self.active_opacity = val / 100.0
        self.render_timer.start(25)

    def _on_zscale_changed(self):
        self.z_scale_factor = float(self.combo_zscale.currentData())
        self._render_3d_scene()

    def _on_res_changed(self):
        self.mesh_res = int(self.combo_res.currentData())
        self._render_3d_scene()

    def _on_defects_toggled(self, state: int):
        self.show_defects = (state == Qt.CheckState.Checked.value or state == 2)
        self._render_3d_scene()

    def closeEvent(self, event):
        plt.close(self.fig)
        super().closeEvent(event)


# ==============================================================================
# Visual Feedback & Gauge Widgets
# ==============================================================================

class CanvasLoadingOverlay(QWidget):
    """Semi-transparent animated spinner overlay shown during inference or large file loading."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setVisible(False)
        self.angle: int = 0
        self.message: str = "Analyzing wafer defects..."
        self.progress_text: str = ""

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate_spinner)

    def rotate_spinner(self):
        self.angle = (self.angle + 25) % 360
        self.update()

    def show_loading(self, message: str = "Processing...", progress_text: str = ""):
        self.message = message
        self.progress_text = progress_text
        self.setVisible(True)
        self.raise_()
        self.timer.start(45)
        self.update()

    def set_progress(self, current: int, total: int, pct: int):
        self.progress_text = f"{current:,} / {total:,} patches ({pct}%)"
        self.update()

    def hide_loading(self):
        self.timer.stop()
        self.setVisible(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Semi-transparent backdrop
        painter.fillRect(self.rect(), QColor(15, 17, 23, 190))

        # 2. Centered status card
        w, h = self.width(), self.height()
        card_w, card_h = 340, 160
        card_x = (w - card_w) // 2
        card_y = (h - card_h) // 2

        # Card body & glowing neon border
        painter.setPen(QPen(QColor(0, 180, 255, 180), 1.5))
        painter.setBrush(QBrush(QColor(25, 28, 36, 245)))
        painter.drawRoundedRect(card_x, card_y, card_w, card_h, 10, 10)

        # 3. Rotating spinner arc
        cx = card_x + card_w // 2
        cy = card_y + 46
        radius = 22

        arc_pen = QPen(QColor(0, 180, 255, 240), 4)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        start_angle = int(-self.angle * 16)
        span_angle = int(240 * 16)
        painter.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, start_angle, span_angle)

        # 4. Main message text
        painter.setPen(QColor(240, 243, 246))
        font = QFont("Helvetica", 12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(card_x + 10, card_y + 88, card_w - 20, 26, Qt.AlignmentFlag.AlignCenter, self.message)

        # 5. Progress subtitle text
        if self.progress_text:
            painter.setPen(QColor(160, 170, 185))
            sub_font = QFont("Helvetica", 10)
            painter.setFont(sub_font)
            painter.drawText(card_x + 10, card_y + 116, card_w - 20, 22, Qt.AlignmentFlag.AlignCenter, self.progress_text)


class BPDSpectrumBar(QWidget):
    """Visual spectrum bar showing BPD short/medium/long length distribution."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self.cnt_short: int = 0
        self.cnt_mid: int = 0
        self.cnt_long: int = 0
        self.total: int = 0
        self.lang: str = "KO"

    def set_data(self, lengths_um: List[float], lang: str = "KO"):
        self.lang = lang
        self.cnt_short = sum(1 for x in lengths_um if x < 30.0)
        self.cnt_mid = sum(1 for x in lengths_um if 30.0 <= x <= 70.0)
        self.cnt_long = sum(1 for x in lengths_um if x > 70.0)
        self.total = len(lengths_um)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        bar_y = 6
        bar_h = 12

        # Background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(36, 40, 48)))
        painter.drawRoundedRect(0, bar_y, w, bar_h, 4, 4)

        if self.total > 0:
            w_short = int(round(w * (self.cnt_short / float(self.total))))
            w_mid = int(round(w * (self.cnt_mid / float(self.total))))
            w_long = w - w_short - w_mid

            curr_x = 0
            if w_short > 0:
                painter.setBrush(QBrush(QColor("#00e676")))  # Emerald
                painter.drawRoundedRect(curr_x, bar_y, w_short, bar_h, 3, 3)
                curr_x += w_short

            if w_mid > 0:
                painter.setBrush(QBrush(QColor("#00b4ff")))  # Cyan
                painter.drawRoundedRect(curr_x, bar_y, w_mid, bar_h, 3, 3)
                curr_x += w_mid

            if w_long > 0:
                painter.setBrush(QBrush(QColor("#ff9800")))  # Amber
                painter.drawRoundedRect(curr_x, bar_y, w_long, bar_h, 3, 3)

        # Legend text below bar
        painter.setPen(QColor(160, 168, 180))
        font = QFont("Helvetica", 9)
        painter.setFont(font)

        if self.total > 0:
            p_s = (self.cnt_short / self.total) * 100.0
            p_m = (self.cnt_mid / self.total) * 100.0
            p_l = (self.cnt_long / self.total) * 100.0
            if self.lang == "KO":
                txt = f"단파(<30μm): {self.cnt_short}개({p_s:.0f}%) | 중파(30~70μm): {self.cnt_mid}개({p_m:.0f}%) | 장파(>70μm): {self.cnt_long}개({p_l:.0f}%)"
            else:
                txt = f"Short(<30μm): {self.cnt_short} ({p_s:.0f}%) | Mid(30-70μm): {self.cnt_mid} ({p_m:.0f}%) | Long(>70μm): {self.cnt_long} ({p_l:.0f}%)"
        else:
            txt = "데이터 대기 중 (Waiting for BPD detection data...)" if self.lang == "KO" else "Waiting for BPD detection data..."

        painter.drawText(0, bar_y + bar_h + 16, w, 18, Qt.AlignmentFlag.AlignLeft, txt)


class TSDAngleGaugeBar(QWidget):
    """Visual multi-segment bar showing TSD angle rule bin distribution."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self.bin_data: List[Dict[str, Any]] = []
        self.total: int = 0
        self.lang: str = "KO"

        self.color_palette = [
            QColor("#ff283c"),  # Red/Pink (A)
            QColor("#ff9800"),  # Orange (B)
            QColor("#00b4ff"),  # Cyan (C)
            QColor("#b388ff"),  # Purple (D)
            QColor("#00e676"),  # Green (E)
            QColor("#ffd600"),  # Yellow (F)
        ]

    def set_data(self, bins: List[Dict[str, Any]], counts: Dict[str, int], lang: str = "KO"):
        self.lang = lang
        self.bin_data = []
        self.total = sum(counts.values())

        for idx, b in enumerate(bins):
            lbl = str(b.get("label", "?"))
            cnt = counts.get(lbl, 0)
            ratio = (cnt / float(self.total) * 100.0) if self.total > 0 else 0.0
            color = self.color_palette[idx % len(self.color_palette)]
            self.bin_data.append({"label": lbl, "count": cnt, "ratio": ratio, "color": color})

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        bar_y = 6
        bar_h = 12

        # Background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(36, 40, 48)))
        painter.drawRoundedRect(0, bar_y, w, bar_h, 4, 4)

        if self.total > 0:
            curr_x = 0
            for item in self.bin_data:
                seg_w = int(round(w * (item["count"] / float(self.total))))
                if seg_w > 0:
                    painter.setBrush(QBrush(item["color"]))
                    painter.drawRoundedRect(curr_x, bar_y, seg_w, bar_h, 3, 3)
                    curr_x += seg_w

        # Legend text
        painter.setPen(QColor(160, 168, 180))
        font = QFont("Helvetica", 9)
        painter.setFont(font)

        if self.total > 0:
            parts = [f"{item['label']}: {item['count']}개 ({item['ratio']:.0f}%)" for item in self.bin_data]
            txt = " | ".join(parts)
        else:
            txt = "데이터 대기 중 (Waiting for TSD detection data...)" if self.lang == "KO" else "Waiting for TSD detection data..."

        painter.drawText(0, bar_y + bar_h + 16, w, 18, Qt.AlignmentFlag.AlignLeft, txt)


class TEDAngleGaugeBar(QWidget):
    """Visual multi-segment bar showing TED 6-type directional distribution."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self.bin_data: List[Dict[str, Any]] = []
        self.total: int = 0
        self.lang: str = "KO"

        self.color_palette = [
            QColor("#00e5ff"),  # Cyan (a, 0°)
            QColor("#76ff03"),  # Lime (b, 60°)
            QColor("#ffd600"),  # Yellow (c, 120°)
            QColor("#ff9100"),  # Orange (d, 180°)
            QColor("#ff4081"),  # Pink (e, 240°)
            QColor("#b388ff"),  # Purple (f, 300°)
        ]

    def set_data(self, bins: List[Dict[str, Any]], counts: Dict[str, int], lang: str = "KO"):
        self.lang = lang
        self.bin_data = []
        self.total = sum(counts.values())

        for idx, b in enumerate(bins):
            lbl = str(b.get("label", "?"))
            cnt = counts.get(lbl, 0)
            ratio = (cnt / float(self.total) * 100.0) if self.total > 0 else 0.0
            color = self.color_palette[idx % len(self.color_palette)]
            self.bin_data.append({"label": lbl, "count": cnt, "ratio": ratio, "color": color})

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        bar_y = 6
        bar_h = 12

        # Background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(36, 40, 48)))
        painter.drawRoundedRect(0, bar_y, w, bar_h, 4, 4)

        if self.total > 0:
            curr_x = 0
            for item in self.bin_data:
                seg_w = int(round(w * (item["count"] / float(self.total))))
                if seg_w > 0:
                    painter.setBrush(QBrush(item["color"]))
                    painter.drawRoundedRect(curr_x, bar_y, seg_w, bar_h, 3, 3)
                    curr_x += seg_w

        # Legend text
        painter.setPen(QColor(160, 168, 180))
        font = QFont("Helvetica", 9)
        painter.setFont(font)

        if self.total > 0:
            parts = [f"{item['label']}: {item['count']}개 ({item['ratio']:.0f}%)" for item in self.bin_data]
            txt = " | ".join(parts)
        else:
            txt = "데이터 대기 중 (Waiting for TED detection data...)" if self.lang == "KO" else "Waiting for TED detection data..."

        painter.drawText(0, bar_y + bar_h + 16, w, 18, Qt.AlignmentFlag.AlignLeft, txt)


# ==============================================================================
# Interactive High-Resolution Wafer Graphics View
# ==============================================================================

class WaferCanvasView(QGraphicsView):
    """QGraphicsView with smooth cursor-centered wheel zoom, drag panning, and loading overlay."""

    zoom_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.overlay_items: Dict[str, List[QGraphicsItem]] = {
            "BPD": [],
            "TED": [],
            "TSD": [],
            "Particle": [],
        }

        # 3x3 Grid Overlay Elements
        self.grid_items: List[QGraphicsItem] = []
        self.show_grid: bool = False
        self.current_img_w: float = 0.0
        self.current_img_h: float = 0.0

        # Configuration for smooth navigation
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # Fast for large images
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.zoom_factor = 1.15
        self.min_zoom = 0.02
        self.max_zoom = 50.0

        # Child Animated Loading Overlay
        self.loading_overlay = CanvasLoadingOverlay(self)

    def get_zoom_level(self) -> float:
        """Returns current zoom scale factor (1.0 = 100%)."""
        return float(self.transform().m11())

    def show_loading(self, message: str = "Processing...", progress_text: str = ""):
        """Displays centered animated loading overlay."""
        self.loading_overlay.setGeometry(self.rect())
        self.loading_overlay.show_loading(message, progress_text)

    def set_loading_progress(self, current: int, total: int, pct: int):
        """Updates progress status on the loading overlay."""
        self.loading_overlay.set_progress(current, total, pct)

    def hide_loading(self):
        """Hides animated loading overlay."""
        self.loading_overlay.hide_loading()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.setGeometry(self.rect())

    def wheelEvent(self, event: QWheelEvent):
        """Cursor-centered smooth zoom."""
        if self.pixmap_item is None:
            return

        angle = event.angleDelta().y()
        if angle > 0:
            factor = self.zoom_factor
        elif angle < 0:
            factor = 1.0 / self.zoom_factor
        else:
            return

        current_scale = self.transform().m11()
        new_scale = current_scale * factor
        if self.min_zoom <= new_scale <= self.max_zoom:
            self.scale(factor, factor)
            self.zoom_changed.emit(self.get_zoom_level())

    def set_display_pixmap(
        self,
        pixmap: QPixmap,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        auto_fit: bool = False,
    ):
        """Displays or updates the main wafer image on the canvas.

        Supports proxy scale transform to preserve global full-wafer coordinate system.
        """
        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.pixmap_item.setZValue(0)
            self.pixmap_item.setTransform(QTransform.fromScale(scale_x, scale_y))
            if auto_fit:
                self.fit_in_view()
        else:
            self.pixmap_item.setPixmap(pixmap)
            self.pixmap_item.setTransform(QTransform.fromScale(scale_x, scale_y))
            if auto_fit:
                self.fit_in_view()

    def fit_in_view(self):
        """Fits image to canvas view."""
        if self.pixmap_item is not None:
            scene_rect = self.pixmap_item.sceneBoundingRect()
            self.scene.setSceneRect(scene_rect)
            self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_changed.emit(self.get_zoom_level())

    def reset_zoom_100(self):
        """Resets zoom to 100% pixel scale."""
        self.resetTransform()
        self.zoom_changed.emit(self.get_zoom_level())

    def clear_overlays(self):
        """Removes all bounding box overlays."""
        for cname, items in self.overlay_items.items():
            for item in items:
                self.scene.removeItem(item)
            items.clear()

    def add_defect_boxes(self, defects: List[Dict[str, Any]]):
        """Draws bounding boxes and labels for detected defects with measurement tooltips."""
        self.clear_overlays()

        font = QFont()
        font.setPointSize(9)
        font.setBold(True)

        for d in defects:
            cname = d.get("class_name", "UNKNOWN")
            bx1, by1, bx2, by2 = d["bbox_xyxy"]
            width = max(1.0, bx2 - bx1)
            height = max(1.0, by2 - by1)

            color = DEFECT_COLORS.get(cname, QColor(255, 255, 255))
            pen = QPen(color, 2.0)
            pen.setCosmetic(True)  # Keeps constant screen stroke width regardless of zoom

            # Bounding box
            rect_item = QGraphicsRectItem(bx1, by1, width, height)
            rect_item.setPen(pen)
            rect_item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
            rect_item.setZValue(10)

            # Tooltip with defect details & precision measurements
            conf = d.get("confidence", 0.0)
            did = d.get("defect_id", "?")
            tooltip_parts = [
                f"ID: #{did}",
                f"Class: {cname}",
                f"Conf: {conf:.3f}",
            ]

            if cname == "BPD" and "length_um" in d:
                tooltip_parts.append(f"Length: {d['length_um']:.1f} μm ({d.get('length_px', 0.0):.1f} px)")
            elif cname == "TSD":
                sub = d.get("sub_class", "TSD")
                if "angle_deg" in d:
                    tooltip_parts.append(f"Direction: {sub} ({d['angle_deg']:.1f}°)")
            elif cname == "TED":
                sub = d.get("sub_class", "TED")
                if "angle_deg" in d:
                    tooltip_parts.append(f"Direction: {sub} ({d['angle_deg']:.1f}°)")

            tooltip_parts.append(f"Coord: ({bx1:.1f}, {by1:.1f})")
            tooltip_parts.append(f"Size: {width:.1f}x{height:.1f}")
            rect_item.setToolTip("\n".join(tooltip_parts))

            self.scene.addItem(rect_item)

            if cname in self.overlay_items:
                self.overlay_items[cname].append(rect_item)

    def set_overlay_visibility(self, class_name: str, visible: bool):
        """Toggles visibility of specific defect class bounding boxes."""
        if class_name in self.overlay_items:
            for item in self.overlay_items[class_name]:
                item.setVisible(visible)

    def clear_grid_overlay(self):
        """Clears 3x3 grid lines and text labels from the scene."""
        for item in self.grid_items:
            self.scene.removeItem(item)
        self.grid_items.clear()

    def update_grid_overlay(self, width: float, height: float):
        """Builds or reconstructs the 3x3 grid lines and zone markers."""
        self.clear_grid_overlay()
        self.current_img_w = float(width)
        self.current_img_h = float(height)

        if width <= 0 or height <= 0:
            return

        zw = width / 3.0
        zh = height / 3.0

        grid_pen = QPen(QColor(0, 220, 255, 170), 2.0, Qt.PenStyle.DashLine)
        grid_pen.setCosmetic(True)

        # 1. Vertical dividing lines (2 lines)
        for i in [1, 2]:
            x = i * zw
            line = QGraphicsLineItem(x, 0.0, x, height)
            line.setPen(grid_pen)
            line.setZValue(20)
            line.setVisible(self.show_grid)
            self.scene.addItem(line)
            self.grid_items.append(line)

        # 2. Horizontal dividing lines (2 lines)
        for j in [1, 2]:
            y = j * zh
            line = QGraphicsLineItem(0.0, y, width, y)
            line.setPen(grid_pen)
            line.setZValue(20)
            line.setVisible(self.show_grid)
            self.scene.addItem(line)
            self.grid_items.append(line)

        # 3. Outer boundary rectangle
        border_pen = QPen(QColor(0, 220, 255, 120), 1.5, Qt.PenStyle.SolidLine)
        border_pen.setCosmetic(True)
        border = QGraphicsRectItem(0.0, 0.0, width, height)
        border.setPen(border_pen)
        border.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        border.setZValue(19)
        border.setVisible(self.show_grid)
        self.scene.addItem(border)
        self.grid_items.append(border)

        # 4. Zone tag labels in corners of each 1/9 block
        zone_names = [
            ["Zone 1 (NW)", "Zone 2 (N)", "Zone 3 (NE)"],
            ["Zone 4 (W)", "Zone 5 (Center)", "Zone 6 (E)"],
            ["Zone 7 (SW)", "Zone 8 (S)", "Zone 9 (SE)"],
        ]
        font = QFont("Helvetica", 14)
        font.setBold(True)

        for r in range(3):
            for c in range(3):
                tag = zone_names[r][c]
                text_item = QGraphicsSimpleTextItem(f"  {tag}  ")
                text_item.setFont(font)
                text_item.setBrush(QBrush(QColor(0, 240, 255, 230)))
                # Place near top-left of each zone with small margin
                tx = c * zw + 24.0
                ty = r * zh + 24.0
                text_item.setPos(tx, ty)
                text_item.setZValue(21)
                text_item.setVisible(self.show_grid)
                self.scene.addItem(text_item)
                self.grid_items.append(text_item)

    def set_grid_visibility(self, visible: bool):
        """Toggles visibility of the 3x3 grid overlay."""
        self.show_grid = visible
        for item in self.grid_items:
            item.setVisible(visible)

    def focus_on_zone(self, row: int, col: int):
        """Zooms and centers viewport onto specific 1/9 zone (0-2 row, 0-2 col)."""
        if self.current_img_w <= 0 or self.current_img_h <= 0:
            return
        zw = self.current_img_w / 3.0
        zh = self.current_img_h / 3.0
        zx = col * zw
        zy = row * zh
        target_rect = QRectF(zx, zy, zw, zh)
        # Small padding so grid lines are visible
        padded_rect = target_rect.adjusted(-zw * 0.02, -zh * 0.02, zw * 0.02, zh * 0.02)
        self.fitInView(padded_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(self.get_zoom_level())

    def focus_on_defect(self, bbox_xyxy: List[float]):
        """Centers and zooms viewport onto a specific defect with comfortable padding."""
        if not bbox_xyxy or len(bbox_xyxy) < 4:
            return
        bx1, by1, bx2, by2 = bbox_xyxy
        w = max(1.0, bx2 - bx1)
        h = max(1.0, by2 - by1)
        cx = bx1 + w / 2.0
        cy = by1 + h / 2.0

        # Margin around defect: 3.5x size or minimum 120 px
        span = max(max(w, h) * 3.5, 120.0)
        target_rect = QRectF(cx - span / 2.0, cy - span / 2.0, span, span)
        self.fitInView(target_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(self.get_zoom_level())


# ==============================================================================
# Main Desktop Window
# ==============================================================================

class SiCDefectMainWindow(QMainWindow):
    """Main application window for SiC Wafer Defect AI."""

    def __init__(self, initial_image_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.current_lang = "KO"  # Default Language: Korean
        self.resize(1380, 880)

        # State Variables
        self.current_file_path: Optional[Path] = None
        self.raw_data: Optional[np.ndarray] = None
        self.current_slice_2d: Optional[np.ndarray] = None
        self.current_slice_idx: int = 0
        self.num_slices: int = 1
        self.is_3d: bool = False
        self.current_meta: Optional[Dict[str, Any]] = None

        self.last_results: Optional[Dict[str, Any]] = None
        self.worker: Optional[InferenceWorker] = None

        # Precision Domain Analysis State
        self.pixel_size_um: float = 2.50
        self.current_angle_bins: List[Dict[str, Any]] = [dict(b) for b in DEFAULT_TSD_ANGLE_BINS]
        self.current_ted_angle_bins: List[Dict[str, Any]] = [dict(b) for b in DEFAULT_TED_ANGLE_BINS]
        self.bpd_defects: List[Dict[str, Any]] = []
        self.tsd_defects: List[Dict[str, Any]] = []
        self.ted_defects: List[Dict[str, Any]] = []
        self.current_defects: List[Dict[str, Any]] = []
        self.tiff_reader: Optional[Any] = None
        self.conversion_stats: Dict[str, Any] = {
            "before_bpd": 0,
            "after_bpd": 0,
            "converted_ted": 0,
            "conversion_rate_pct": 0.0,
        }

        # Performance Cache for Smooth Real-Time Sliders (Dual-Resolution Proxy + LUT)
        self.base_norm_8bit: Optional[np.ndarray] = None
        self.proxy_norm_8bit: Optional[np.ndarray] = None
        self.proxy_scale_x: float = 1.0
        self.proxy_scale_y: float = 1.0

        # Debounce timer for high-res 12k rendering after slider settles
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.render_fullres_image)

        # Default model search (Pure pathlib)
        self.model_path = self.locate_default_model()

        # Build UI
        self.init_ui()
        self.retranslate_ui()

        # Load initial file if supplied
        if initial_image_path is not None:
            self.load_image(Path(initial_image_path))

    def tr(self, key: str, **kwargs) -> Any:
        """Retrieves localized text with optional formatting parameters."""
        translations = TRANSLATIONS.get(self.current_lang, TRANSLATIONS["KO"])
        val = translations.get(key, "")
        if isinstance(val, str) and kwargs:
            try:
                return val.format(**kwargs)
            except Exception:
                return val
        return val

    def toggle_language(self):
        """Toggles between Korean (KO) and English (EN) and refreshes UI."""
        self.current_lang = "EN" if self.current_lang == "KO" else "KO"
        self.retranslate_ui()

    def locate_default_model(self) -> Path:
        """Locates best.onnx or best.pt across standard project paths."""
        candidates = [
            Path("best.onnx"),
            Path("runs") / "detect" / "runs" / "train" / "sic_defect_train" / "weights" / "best.onnx",
            Path("runs") / "detect" / "runs" / "train" / "sic_defect_train" / "weights" / "best.pt",
            Path("runs") / "train" / "sic_defect_train" / "weights" / "best.onnx",
            Path("runs") / "train" / "sic_defect_train" / "weights" / "best.pt",
            Path("best.pt"),
            Path("yolov8n.onnx"),
            Path("yolov8n.pt"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return (Path("best.onnx")).resolve()

    def create_vsep(self) -> QFrame:
        """Creates vertical separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep

    def init_ui(self):
        """Initializes UI layout, controls, and stylesheets."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ----------------------------------------------------------------------
        # LEFT: Canvas + Bottom Controls
        # ----------------------------------------------------------------------
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Graphics Canvas
        self.canvas = WaferCanvasView()
        left_layout.addWidget(self.canvas, stretch=1)

        # Bottom Controls Frame
        bottom_frame = QFrame()
        bottom_frame.setFrameShape(QFrame.Shape.StyledPanel)
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(8, 6, 8, 6)
        bottom_layout.setSpacing(12)

        # 1. 3D Slice Slider & Depth Window SpinBoxes
        self.slice_label = QLabel("Slice: 1/1")
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        bottom_layout.addWidget(self.slice_label)
        bottom_layout.addWidget(self.slice_slider, stretch=1)

        self.lbl_depth_range = QLabel(self.tr("slice_depth_range"))
        self.spin_depth_start = QSpinBox()
        self.spin_depth_start.setRange(1, 1)
        self.spin_depth_start.setEnabled(False)
        self.spin_depth_start.setFixedWidth(56)
        self.spin_depth_start.valueChanged.connect(self.on_depth_range_changed)

        self.lbl_depth_tilde = QLabel("~")

        self.spin_depth_end = QSpinBox()
        self.spin_depth_end.setRange(1, 1)
        self.spin_depth_end.setEnabled(False)
        self.spin_depth_end.setFixedWidth(56)
        self.spin_depth_end.valueChanged.connect(self.on_depth_range_changed)

        bottom_layout.addWidget(self.lbl_depth_range)
        bottom_layout.addWidget(self.spin_depth_start)
        bottom_layout.addWidget(self.lbl_depth_tilde)
        bottom_layout.addWidget(self.spin_depth_end)

        # 3D Interactive Volume Viewer Button
        self.btn_3d_viewer = QPushButton("🧊 3D 입체 뷰어")
        self.btn_3d_viewer.setEnabled(False)
        self.btn_3d_viewer.setToolTip(self.tr("tooltip_3d_viewer"))
        self.btn_3d_viewer.setStyleSheet("""
            QPushButton {
                background-color: #0d47a1;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #1976d2;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:disabled {
                background-color: #2a2d34;
                color: #6c757d;
                border: 1px solid #383a42;
            }
        """)
        self.btn_3d_viewer.clicked.connect(self.open_3d_volume_viewer)
        bottom_layout.addWidget(self.btn_3d_viewer)

        # Separator
        bottom_layout.addWidget(self.create_vsep())

        # 2. Brightness Slider
        self.lbl_brightness = QLabel("Brightness:")
        bottom_layout.addWidget(self.lbl_brightness)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setFixedWidth(120)
        self.brightness_slider.valueChanged.connect(self.on_slider_value_changed)
        self.brightness_slider.sliderReleased.connect(self.render_fullres_image)
        bottom_layout.addWidget(self.brightness_slider)

        # 3. Contrast Slider
        self.lbl_contrast = QLabel("Contrast:")
        bottom_layout.addWidget(self.lbl_contrast)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(20, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFixedWidth(120)
        self.contrast_slider.valueChanged.connect(self.on_slider_value_changed)
        self.contrast_slider.sliderReleased.connect(self.render_fullres_image)
        bottom_layout.addWidget(self.contrast_slider)

        # 4. View Reset Buttons
        self.btn_fit = QPushButton("Fit to View")
        self.btn_fit.clicked.connect(self.canvas.fit_in_view)
        bottom_layout.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("100% (1:1)")
        self.btn_100.clicked.connect(self.canvas.reset_zoom_100)
        bottom_layout.addWidget(self.btn_100)

        # 5. Quick 3x3 Grid Checkbox on bottom bar
        self.chk_bottom_grid = QCheckBox("3x3 Grid")
        self.chk_bottom_grid.setChecked(False)
        self.chk_bottom_grid.stateChanged.connect(self.on_grid_overlay_toggled)
        bottom_layout.addWidget(self.chk_bottom_grid)

        left_layout.addWidget(bottom_frame)
        splitter.addWidget(left_container)

        # ----------------------------------------------------------------------
        # RIGHT: QTabWidget 3-Tab Panel (Overview, Analysis, Wafer Grid)
        # ----------------------------------------------------------------------
        right_panel = QWidget()
        right_panel.setMinimumWidth(380)
        right_panel.setMaximumWidth(460)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Top Bar: Language Toggle Button
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addStretch(1)
        self.btn_lang = QPushButton("🌐 English (EN)")
        self.btn_lang.setFixedHeight(28)
        self.btn_lang.setStyleSheet("font-weight: bold; padding: 2px 10px;")
        self.btn_lang.clicked.connect(self.toggle_language)
        top_bar_layout.addWidget(self.btn_lang)
        right_layout.addLayout(top_bar_layout)

        # QTabWidget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3c3c46;
                background-color: #1e1e24;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #252830;
                color: #8b949e;
                padding: 7px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #007acc;
                color: #ffffff;
            }
        """)

        # Tab 1: Overview
        self.tab_overview = QWidget()
        self.init_overview_tab()
        self.tab_widget.addTab(self.tab_overview, "📊 Overview")

        # Tab 2: Precision Analysis
        self.tab_analysis = QWidget()
        self.init_analysis_tab()
        self.tab_widget.addTab(self.tab_analysis, "📐 Analysis")

        # Tab 3: Wafer 3x3 Grid
        self.tab_grid = QWidget()
        self.init_grid_tab()
        self.tab_widget.addTab(self.tab_grid, "🧭 Wafer Grid")

        right_layout.addWidget(self.tab_widget)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        # Bottom Modern Status Bar with Badges
        self.init_status_bar()

    def init_status_bar(self):
        """Initializes polished modern dark status bar with persistent information badges."""
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        sb.setStyleSheet("""
            QStatusBar {
                background-color: #14161c;
                border-top: 1px solid #282c35;
                color: #8b949e;
                font-size: 11px;
                padding: 2px 6px;
            }
            QStatusBar::item {
                border: none;
            }
        """)

        badge_style = """
            QLabel {
                background-color: #1f232b;
                color: #c9d1d9;
                border: 1px solid #363c48;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 500;
            }
        """

        self.status_file_lbl = QLabel(self.tr("sb_no_file"))
        self.status_file_lbl.setStyleSheet(badge_style)
        self.status_file_lbl.setToolTip("Currently loaded wafer image file")

        self.status_res_lbl = QLabel("📐 0 x 0 px")
        self.status_res_lbl.setStyleSheet(badge_style)
        self.status_res_lbl.setToolTip("Image pixel dimensions (Width x Height)")

        self.status_zoom_lbl = QLabel("🔍 100%")
        self.status_zoom_lbl.setStyleSheet(badge_style)
        self.status_zoom_lbl.setToolTip("Current canvas viewport zoom scale")

        self.status_engine_lbl = QLabel(f"⚡ ONNX ({self.model_path.name})")
        self.status_engine_lbl.setStyleSheet("""
            QLabel {
                background-color: #132b3d;
                color: #58a6ff;
                border: 1px solid #1f547e;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        self.status_engine_lbl.setToolTip("Inference runtime core engine (ONNX Runtime)")

        sb.addPermanentWidget(self.status_file_lbl)
        sb.addPermanentWidget(self.status_res_lbl)
        sb.addPermanentWidget(self.status_zoom_lbl)
        sb.addPermanentWidget(self.status_engine_lbl)

        self.canvas.zoom_changed.connect(self.on_canvas_zoom_changed)
        self.update_status_bar()

    def on_canvas_zoom_changed(self, zoom_val: float):
        """Callback when canvas zoom level changes."""
        pct = int(round(zoom_val * 100))
        self.status_zoom_lbl.setText(f"🔍 {pct}%")

    def update_status_bar(self):
        """Updates persistent status bar badge labels."""
        if self.current_file_path:
            name = self.current_file_path.name
            if len(name) > 30:
                name = name[:27] + "..."
            self.status_file_lbl.setText(f"📁 {name}")
        else:
            self.status_file_lbl.setText(self.tr("sb_no_file"))

        if self.current_slice_2d is not None:
            h, w = self.current_slice_2d.shape[:2]
            self.status_res_lbl.setText(f"📐 {w:,} x {h:,} px")
        else:
            self.status_res_lbl.setText("📐 0 x 0 px")

        zoom_pct = int(round(self.canvas.get_zoom_level() * 100))
        self.status_zoom_lbl.setText(f"🔍 {zoom_pct}%")

        mname = self.model_path.name if self.model_path else "best.onnx"
        self.status_engine_lbl.setText(f"⚡ ONNX ({mname})")

    def keyPressEvent(self, event: QKeyEvent):
        """Handles ergonomic global keyboard shortcuts for fast researcher workflow."""
        focused = self.focusWidget()
        if isinstance(focused, (QLineEdit, QSpinBox, QDoubleSpinBox)):
            super().keyPressEvent(event)
            return

        key = event.key()
        if key == Qt.Key.Key_O:
            self.on_open_file_dialog()
            event.accept()
        elif key == Qt.Key.Key_R:
            self.start_analysis()
            event.accept()
        elif key == Qt.Key.Key_F:
            self.canvas.fit_in_view()
            event.accept()
        elif key == Qt.Key.Key_G:
            self.chk_grid_overlay.toggle()
            event.accept()
        elif key == Qt.Key.Key_1:
            self.tab_widget.setCurrentIndex(0)
            event.accept()
        elif key == Qt.Key.Key_2:
            self.tab_widget.setCurrentIndex(1)
            event.accept()
        elif key == Qt.Key.Key_3:
            self.tab_widget.setCurrentIndex(2)
            event.accept()
        else:
            super().keyPressEvent(event)

    def init_overview_tab(self):
        """Initializes Tab 1: Overview controls, inference trigger, summary table, and overlays."""
        tab_layout = QVBoxLayout(self.tab_overview)
        tab_layout.setContentsMargins(6, 8, 6, 8)
        tab_layout.setSpacing(8)

        # Group 1: File & Model Management
        self.grp_file = QGroupBox("File & Model Configuration")
        grp_file_layout = QVBoxLayout(self.grp_file)

        self.btn_open = QPushButton("📂 Open XRT Image (.tif / .tiff)")
        self.btn_open.setFixedHeight(34)
        self.btn_open.clicked.connect(self.on_open_file_dialog)
        grp_file_layout.addWidget(self.btn_open)

        self.lbl_file_info = QLabel("No image loaded")
        self.lbl_file_info.setWordWrap(True)
        grp_file_layout.addWidget(self.lbl_file_info)

        # Model label & select
        model_hbox = QHBoxLayout()
        self.lbl_model_info = QLabel(f"Model: {self.model_path.name}")
        self.lbl_model_info.setWordWrap(True)
        self.btn_change_model = QPushButton("Select Weights")
        self.btn_change_model.clicked.connect(self.on_select_model)
        model_hbox.addWidget(self.lbl_model_info, stretch=1)
        model_hbox.addWidget(self.btn_change_model)
        grp_file_layout.addLayout(model_hbox)

        tab_layout.addWidget(self.grp_file)

        # Group 2: Analysis Execution
        self.grp_analysis = QGroupBox("Automated Defect Analysis")
        grp_analysis_layout = QVBoxLayout(self.grp_analysis)

        self.btn_run = QPushButton("⚡ Run Full-Wafer Analysis (ONNX)")
        self.btn_run.setFixedHeight(38)
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #007acc; color: white;")
        self.btn_run.clicked.connect(self.start_analysis)
        grp_analysis_layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        grp_analysis_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready")
        grp_analysis_layout.addWidget(self.lbl_status)

        tab_layout.addWidget(self.grp_analysis)

        # Group 3: Detection Summary & Overlays
        self.grp_results = QGroupBox("Defect Summary & Class Overlays")
        grp_results_layout = QVBoxLayout(self.grp_results)

        # Summary Table: 6 Rows
        self.table_summary = QTableWidget(6, 3)
        self.table_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_summary.verticalHeader().setVisible(False)
        self.init_summary_table()
        grp_results_layout.addWidget(self.table_summary)

        # Layer Checkboxes
        overlay_hbox = QHBoxLayout()
        self.chk_bpd = QCheckBox("BPD")
        self.chk_bpd.setChecked(True)
        self.chk_bpd.stateChanged.connect(lambda s: self.canvas.set_overlay_visibility("BPD", s == 2))

        self.chk_ted = QCheckBox("TED")
        self.chk_ted.setChecked(True)
        self.chk_ted.stateChanged.connect(lambda s: self.canvas.set_overlay_visibility("TED", s == 2))

        self.chk_tsd = QCheckBox("TSD")
        self.chk_tsd.setChecked(True)
        self.chk_tsd.stateChanged.connect(lambda s: self.canvas.set_overlay_visibility("TSD", s == 2))

        self.chk_particle = QCheckBox("Particle")
        self.chk_particle.setChecked(True)
        self.chk_particle.stateChanged.connect(lambda s: self.canvas.set_overlay_visibility("Particle", s == 2))

        overlay_hbox.addWidget(self.chk_bpd)
        overlay_hbox.addWidget(self.chk_ted)
        overlay_hbox.addWidget(self.chk_tsd)
        overlay_hbox.addWidget(self.chk_particle)
        grp_results_layout.addLayout(overlay_hbox)

        # Export Button
        self.btn_export = QPushButton("💾 Export Report (CSV / JSON)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_report)
        grp_results_layout.addWidget(self.btn_export)

        tab_layout.addWidget(self.grp_results)
        tab_layout.addStretch(1)

    def init_summary_table(self):
        """Initializes empty defect statistics table with 6 rows."""
        row_definitions = [
            ("row_bpd", "0", "#00b4ff", False),
            ("row_ted", "0", "#ffdc00", False),
            ("row_tsd", "0", "#ff283c", False),
            ("row_crystal_subtotal", "0", "#58a6ff", True),
            ("row_particle", "0", "#b388ff", False),
            ("row_grand_total", "0", "#ffffff", True),
        ]
        for row_idx, (key, cnt, hex_color, is_bold) in enumerate(row_definitions):
            name_text = self.tr(key)
            item_name = QTableWidgetItem(name_text)
            item_cnt = QTableWidgetItem(cnt)
            item_cnt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_col = QTableWidgetItem("■")
            item_col.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_col.setForeground(QColor(hex_color))

            if is_bold:
                bold_font = item_name.font()
                bold_font.setBold(True)
                item_name.setFont(bold_font)
                item_cnt.setFont(bold_font)
                item_name.setBackground(QColor(45, 49, 58))
                item_cnt.setBackground(QColor(45, 49, 58))
                item_col.setBackground(QColor(45, 49, 58))

            self.table_summary.setItem(row_idx, 0, item_name)
            self.table_summary.setItem(row_idx, 1, item_cnt)
            self.table_summary.setItem(row_idx, 2, item_col)

    def init_analysis_tab(self):
        """Initializes Tab 2: Precision Analysis for BPD length & TSD custom angle bins."""
        outer_layout = QVBoxLayout(self.tab_analysis)
        outer_layout.setContentsMargins(2, 2, 2, 2)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("background-color: transparent;")

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.setSpacing(10)

        # ----------------------------------------------------------------------
        # 0. Wafer Pixel Scale Resolution Calibration
        # ----------------------------------------------------------------------
        self.grp_scale = QGroupBox("Wafer Resolution Calibration")
        scale_layout = QHBoxLayout(self.grp_scale)
        scale_layout.setContentsMargins(8, 6, 8, 6)

        self.lbl_pixel_scale = QLabel(self.tr("pixel_scale_label"))
        self.lbl_pixel_scale.setStyleSheet("font-weight: bold;")
        scale_layout.addWidget(self.lbl_pixel_scale)

        self.spin_pixel_scale = QDoubleSpinBox()
        self.spin_pixel_scale.setRange(0.05, 100.0)
        self.spin_pixel_scale.setSingleStep(0.1)
        self.spin_pixel_scale.setValue(self.pixel_size_um)
        self.spin_pixel_scale.setDecimals(2)
        self.spin_pixel_scale.setSuffix(" μm/px")
        self.spin_pixel_scale.setFixedWidth(120)
        self.spin_pixel_scale.valueChanged.connect(self.on_pixel_scale_changed)
        scale_layout.addWidget(self.spin_pixel_scale)

        content_layout.addWidget(self.grp_scale)

        # ----------------------------------------------------------------------
        # 1. BPD Dislocation Length Distribution (PCA, μm)
        # ----------------------------------------------------------------------
        self.grp_bpd_analysis = QGroupBox(self.tr("analysis_bpd_title"))
        bpd_layout = QVBoxLayout(self.grp_bpd_analysis)
        bpd_layout.setContentsMargins(8, 8, 8, 8)
        bpd_layout.setSpacing(6)

        self.lbl_bpd_desc = QLabel(self.tr("analysis_bpd_desc"))
        self.lbl_bpd_desc.setWordWrap(True)
        self.lbl_bpd_desc.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        bpd_layout.addWidget(self.lbl_bpd_desc)

        # BPD summary statistics card
        self.lbl_bpd_summary = QLabel(self.tr("analysis_bpd_summary", cnt=0, mean=0.0, max=0.0, min=0.0))
        self.lbl_bpd_summary.setWordWrap(True)
        self.lbl_bpd_summary.setStyleSheet("font-weight: bold; color: #00b4ff; padding: 4px 0;")
        bpd_layout.addWidget(self.lbl_bpd_summary)

        # BPD Spectrum Bar (Short / Mid / Long)
        self.bpd_spectrum_bar = BPDSpectrumBar()
        bpd_layout.addWidget(self.bpd_spectrum_bar)

        # BPD defects table (ID, Center, Length px, Length um)
        self.table_bpd = QTableWidget(0, 4)
        self.table_bpd.setHorizontalHeaderLabels(self.tr("analysis_bpd_headers"))
        self.table_bpd.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_bpd.verticalHeader().setVisible(False)
        self.table_bpd.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_bpd.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_bpd.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_bpd.setFixedHeight(130)
        self.table_bpd.cellClicked.connect(self.on_bpd_cell_clicked)
        bpd_layout.addWidget(self.table_bpd)

        self.btn_bpd_hist = QPushButton(self.tr("analysis_bpd_btn"))
        self.btn_bpd_hist.clicked.connect(self.show_bpd_histogram)
        bpd_layout.addWidget(self.btn_bpd_hist)

        content_layout.addWidget(self.grp_bpd_analysis)

        # ----------------------------------------------------------------------
        # 2. BPD -> TED Annealing Conversion Rate Calculator
        # ----------------------------------------------------------------------
        self.grp_conversion = QGroupBox(self.tr("analysis_conversion_title"))
        conv_layout = QVBoxLayout(self.grp_conversion)
        conv_layout.setContentsMargins(8, 8, 8, 8)
        conv_layout.setSpacing(6)

        self.lbl_conv_desc = QLabel(self.tr("analysis_conversion_desc"))
        self.lbl_conv_desc.setWordWrap(True)
        self.lbl_conv_desc.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        conv_layout.addWidget(self.lbl_conv_desc)

        conv_form = QGridLayout()
        conv_form.setContentsMargins(0, 4, 0, 4)
        conv_form.setSpacing(6)

        self.lbl_before_bpd = QLabel(self.tr("lbl_before_bpd"))
        self.spin_before_bpd = QSpinBox()
        self.spin_before_bpd.setRange(0, 999999)
        self.spin_before_bpd.setValue(0)
        self.spin_before_bpd.valueChanged.connect(self.calculate_conversion_rate)

        self.lbl_after_bpd = QLabel(self.tr("lbl_after_bpd"))
        self.spin_after_bpd = QSpinBox()
        self.spin_after_bpd.setRange(0, 999999)
        self.spin_after_bpd.setValue(0)
        self.spin_after_bpd.valueChanged.connect(self.calculate_conversion_rate)

        self.btn_apply_current_bpd = QPushButton(self.tr("btn_apply_current_bpd"))
        self.btn_apply_current_bpd.clicked.connect(self.on_apply_current_bpd_to_conversion)

        conv_form.addWidget(self.lbl_before_bpd, 0, 0)
        conv_form.addWidget(self.spin_before_bpd, 0, 1)
        conv_form.addWidget(self.lbl_after_bpd, 1, 0)
        conv_form.addWidget(self.spin_after_bpd, 1, 1)
        conv_form.addWidget(self.btn_apply_current_bpd, 2, 0, 1, 2)
        conv_layout.addLayout(conv_form)

        # Output conversion rate cards
        conv_res_hbox = QHBoxLayout()

        card_ted = QFrame()
        card_ted.setStyleSheet("background-color: #252830; border: 1px solid #3c424e; border-radius: 6px; padding: 6px;")
        ted_card_layout = QVBoxLayout(card_ted)
        ted_card_layout.setContentsMargins(4, 4, 4, 4)
        self.lbl_converted_ted_title = QLabel(self.tr("lbl_converted_ted"))
        self.lbl_converted_ted_title.setStyleSheet("font-size: 10px; color: #8b949e;")
        self.lbl_converted_ted_val = QLabel("0 개")
        self.lbl_converted_ted_val.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffdc00;")
        ted_card_layout.addWidget(self.lbl_converted_ted_title)
        ted_card_layout.addWidget(self.lbl_converted_ted_val)
        conv_res_hbox.addWidget(card_ted, stretch=1)

        card_rate = QFrame()
        card_rate.setStyleSheet("background-color: #252830; border: 1px solid #3c424e; border-radius: 6px; padding: 6px;")
        rate_card_layout = QVBoxLayout(card_rate)
        rate_card_layout.setContentsMargins(4, 4, 4, 4)
        self.lbl_conversion_rate_title = QLabel(self.tr("lbl_conversion_rate"))
        self.lbl_conversion_rate_title.setStyleSheet("font-size: 10px; color: #8b949e;")
        self.lbl_conversion_rate_val = QLabel("0.0 %")
        self.lbl_conversion_rate_val.setStyleSheet("font-size: 15px; font-weight: bold; color: #8b949e;")
        rate_card_layout.addWidget(self.lbl_conversion_rate_title)
        rate_card_layout.addWidget(self.lbl_conversion_rate_val)
        conv_res_hbox.addWidget(card_rate, stretch=1)

        conv_layout.addLayout(conv_res_hbox)

        self.lbl_conv_formula = QLabel(self.tr("conversion_formula"))
        self.lbl_conv_formula.setWordWrap(True)
        self.lbl_conv_formula.setStyleSheet("color: #6b7280; font-size: 10px; font-style: italic;")
        conv_layout.addWidget(self.lbl_conv_formula)

        content_layout.addWidget(self.grp_conversion)

        # ----------------------------------------------------------------------
        # 3. TED Component 6-Type Directional Classification (<11-20>, 60 deg)
        # ----------------------------------------------------------------------
        self.grp_ted_analysis = QGroupBox(self.tr("analysis_ted_title"))
        ted_layout = QVBoxLayout(self.grp_ted_analysis)
        ted_layout.setContentsMargins(8, 8, 8, 8)
        ted_layout.setSpacing(6)

        self.lbl_ted_desc = QLabel(self.tr("analysis_ted_desc"))
        self.lbl_ted_desc.setWordWrap(True)
        self.lbl_ted_desc.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        ted_layout.addWidget(self.lbl_ted_desc)

        # TED Angle Rule Occupancy Gauge Bar (6 Colors)
        self.ted_gauge_bar = TEDAngleGaugeBar()
        ted_layout.addWidget(self.ted_gauge_bar)

        # TED 6-Type Bins Table (6 rows, 5 columns: Label, Center, Range, Count, Ratio)
        self.table_ted_bins = QTableWidget(len(self.current_ted_angle_bins), 5)
        self.table_ted_bins.setHorizontalHeaderLabels(self.tr("analysis_ted_headers"))
        self.table_ted_bins.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_ted_bins.verticalHeader().setVisible(False)
        self.table_ted_bins.setFixedHeight(145)
        self.init_ted_bins_table()
        ted_layout.addWidget(self.table_ted_bins)

        # Reset button
        self.btn_reset_ted_bins = QPushButton(self.tr("btn_reset_ted_bins"))
        self.btn_reset_ted_bins.clicked.connect(self.on_reset_ted_bins)
        ted_layout.addWidget(self.btn_reset_ted_bins)

        content_layout.addWidget(self.grp_ted_analysis)

        # ----------------------------------------------------------------------
        # 4. TSD 4-Way & Custom Angle Rule Sets (Propagation Angle)
        # ----------------------------------------------------------------------
        self.grp_tsd_analysis = QGroupBox(self.tr("analysis_tsd_title"))
        tsd_layout = QVBoxLayout(self.grp_tsd_analysis)
        tsd_layout.setContentsMargins(8, 8, 8, 8)
        tsd_layout.setSpacing(6)

        self.lbl_tsd_desc = QLabel(self.tr("analysis_tsd_desc"))
        self.lbl_tsd_desc.setWordWrap(True)
        self.lbl_tsd_desc.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        tsd_layout.addWidget(self.lbl_tsd_desc)

        # TSD Angle Rule Occupancy Gauge Bar
        self.tsd_gauge_bar = TSDAngleGaugeBar()
        tsd_layout.addWidget(self.tsd_gauge_bar)

        # TSD Custom Angle Bins Table
        self.table_tsd_bins = QTableWidget(len(self.current_angle_bins), 5)
        self.table_tsd_bins.setHorizontalHeaderLabels(self.tr("analysis_tsd_headers"))
        self.table_tsd_bins.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_tsd_bins.verticalHeader().setVisible(False)
        self.table_tsd_bins.setFixedHeight(135)
        self.init_tsd_bins_table()
        tsd_layout.addWidget(self.table_tsd_bins)

        # Bin Action Buttons (Add, Delete, Reset, Apply)
        bins_btn_grid = QGridLayout()
        bins_btn_grid.setContentsMargins(0, 0, 0, 0)
        bins_btn_grid.setSpacing(6)

        self.btn_add_bin = QPushButton(self.tr("btn_add_bin"))
        self.btn_add_bin.clicked.connect(self.on_add_angle_bin)
        bins_btn_grid.addWidget(self.btn_add_bin, 0, 0)

        self.btn_del_bin = QPushButton(self.tr("btn_del_bin"))
        self.btn_del_bin.clicked.connect(self.on_delete_angle_bin)
        bins_btn_grid.addWidget(self.btn_del_bin, 0, 1)

        self.btn_reset_bins = QPushButton(self.tr("btn_reset_bins"))
        self.btn_reset_bins.clicked.connect(self.on_reset_angle_bins)
        bins_btn_grid.addWidget(self.btn_reset_bins, 1, 0)

        self.btn_apply_bins = QPushButton(self.tr("btn_apply_bins"))
        self.btn_apply_bins.setStyleSheet("font-weight: bold; background-color: #007acc; color: white;")
        self.btn_apply_bins.clicked.connect(self.on_apply_angle_bins)
        bins_btn_grid.addWidget(self.btn_apply_bins, 1, 1)

        tsd_layout.addLayout(bins_btn_grid)
        content_layout.addWidget(self.grp_tsd_analysis)

        content_layout.addStretch(1)
        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)

    def init_tsd_bins_table(self):
        """Populates the TSD custom angle bins table with current active bins."""
        self.table_tsd_bins.setRowCount(len(self.current_angle_bins))
        for r, b in enumerate(self.current_angle_bins):
            item_lbl = QTableWidgetItem(str(b.get("label", "?")))
            item_start = QTableWidgetItem(str(b.get("start", 0.0)))
            item_end = QTableWidgetItem(str(b.get("end", 0.0)))
            item_cnt = QTableWidgetItem("0")
            item_ratio = QTableWidgetItem("0.0%")

            item_cnt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_ratio.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Count and Ratio are calculated, non-editable
            item_cnt.setFlags(item_cnt.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item_ratio.setFlags(item_ratio.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.table_tsd_bins.setItem(r, 0, item_lbl)
            self.table_tsd_bins.setItem(r, 1, item_start)
            self.table_tsd_bins.setItem(r, 2, item_end)
            self.table_tsd_bins.setItem(r, 3, item_cnt)
            self.table_tsd_bins.setItem(r, 4, item_ratio)

    def init_ted_bins_table(self):
        """Populates the TED 6-type preset table."""
        self.table_ted_bins.setRowCount(len(self.current_ted_angle_bins))
        for r, b in enumerate(self.current_ted_angle_bins):
            lbl = str(b.get("label", "?"))
            center = f"{b.get('center', 0.0):.0f}°"
            st = b.get("start", 0.0)
            en = b.get("end", 0.0)
            rng = f"{st:.0f}° ~ {en:.0f}°"
            item_lbl = QTableWidgetItem(lbl)
            item_center = QTableWidgetItem(center)
            item_rng = QTableWidgetItem(rng)
            item_cnt = QTableWidgetItem("0")
            item_ratio = QTableWidgetItem("0.0%")

            item_lbl.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_center.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_rng.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_cnt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_ratio.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            for it in (item_lbl, item_center, item_rng, item_cnt, item_ratio):
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.table_ted_bins.setItem(r, 0, item_lbl)
            self.table_ted_bins.setItem(r, 1, item_center)
            self.table_ted_bins.setItem(r, 2, item_rng)
            self.table_ted_bins.setItem(r, 3, item_cnt)
            self.table_ted_bins.setItem(r, 4, item_ratio)

    def on_reset_ted_bins(self):
        """Resets TED 6-type bins back to default preset."""
        self.current_ted_angle_bins = [dict(b) for b in DEFAULT_TED_ANGLE_BINS]
        self.init_ted_bins_table()
        if self.ted_defects:
            for d in self.ted_defects:
                ang = float(d.get("angle_deg", 0.0))
                bin_label = classify_ted_angle(ang, self.current_ted_angle_bins)
                d["angle_label"] = bin_label
                d["sub_class"] = f"TED-{bin_label}"
            self.update_analysis_ui()
            if self.last_results and "defects" in self.last_results:
                self.canvas.add_defect_boxes(self.last_results["defects"])

    def calculate_conversion_rate(self):
        """Calculates BPD to TED conversion rate in real-time."""
        before = self.spin_before_bpd.value()
        after = self.spin_after_bpd.value()
        if before > 0:
            converted = max(0, before - after)
            rate = ((before - after) / float(before)) * 100.0
            rate = max(0.0, min(100.0, rate))
            self.lbl_conversion_rate_val.setText(f"{rate:.1f} %")
            self.lbl_converted_ted_val.setText(f"{converted:,} 개")
            if rate >= 80.0:
                self.lbl_conversion_rate_val.setStyleSheet("font-size: 15px; font-weight: bold; color: #00e676;")
            elif rate >= 50.0:
                self.lbl_conversion_rate_val.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffd600;")
            else:
                self.lbl_conversion_rate_val.setStyleSheet("font-size: 15px; font-weight: bold; color: #ff5252;")
            self.conversion_stats = {
                "before_bpd": before,
                "after_bpd": after,
                "converted_ted": converted,
                "conversion_rate_pct": round(rate, 2),
            }
        else:
            self.lbl_conversion_rate_val.setText("0.0 %")
            self.lbl_conversion_rate_val.setStyleSheet("font-size: 15px; font-weight: bold; color: #8b949e;")
            self.lbl_converted_ted_val.setText("0 개")
            self.conversion_stats = {
                "before_bpd": before,
                "after_bpd": after,
                "converted_ted": 0,
                "conversion_rate_pct": 0.0,
            }

    def on_apply_current_bpd_to_conversion(self):
        """Auto-fills current detected BPD count into Before or After BPD field."""
        curr_bpd = len(self.bpd_defects)
        if self.spin_before_bpd.value() == 0:
            self.spin_before_bpd.setValue(curr_bpd)
        else:
            self.spin_after_bpd.setValue(curr_bpd)
        self.calculate_conversion_rate()

    def on_depth_range_changed(self):
        """Constrains slider range when researcher specifies depth window."""
        if not self.is_3d:
            return
        start_val = self.spin_depth_start.value()
        end_val = self.spin_depth_end.value()
        if start_val > end_val:
            self.spin_depth_end.setValue(start_val)
            end_val = start_val

        # Update slider range
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(start_val - 1, end_val - 1)
        # Clamp slider value
        cur_val = self.slice_slider.value()
        if cur_val < start_val - 1:
            cur_val = start_val - 1
            self.slice_slider.setValue(cur_val)
        elif cur_val > end_val - 1:
            cur_val = end_val - 1
            self.slice_slider.setValue(cur_val)
        self.slice_slider.blockSignals(False)

        # Trigger slice load
        self.on_slice_changed(self.slice_slider.value())

    def on_pixel_scale_changed(self, val: float):
        """Updates pixel scaling factor and recalculates BPD lengths."""
        self.pixel_size_um = float(val)
        if self.bpd_defects:
            for d in self.bpd_defects:
                px = float(d.get("length_px", 0.0))
                d["length_um"] = round(px * self.pixel_size_um, 2)
            self.update_analysis_ui()
            if self.last_results and "defects" in self.last_results:
                self.canvas.add_defect_boxes(self.last_results["defects"])

    def on_bpd_cell_clicked(self, row: int, col: int):
        """Focuses and zooms canvas viewport on clicked BPD defect."""
        if 0 <= row < len(self.bpd_defects):
            d = self.bpd_defects[row]
            self.canvas.focus_on_defect(d["bbox_xyxy"])

    def on_add_angle_bin(self):
        """Adds a new customizable angle bin row to the table."""
        row_idx = self.table_tsd_bins.rowCount()
        self.table_tsd_bins.insertRow(row_idx)

        default_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        new_label = default_labels[row_idx] if row_idx < len(default_labels) else f"Bin{row_idx+1}"

        item_lbl = QTableWidgetItem(new_label)
        item_start = QTableWidgetItem("0.0")
        item_end = QTableWidgetItem("45.0")
        item_cnt = QTableWidgetItem("0")
        item_ratio = QTableWidgetItem("0.0%")

        item_cnt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item_ratio.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item_cnt.setFlags(item_cnt.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item_ratio.setFlags(item_ratio.flags() & ~Qt.ItemFlag.ItemIsEditable)

        self.table_tsd_bins.setItem(row_idx, 0, item_lbl)
        self.table_tsd_bins.setItem(row_idx, 1, item_start)
        self.table_tsd_bins.setItem(row_idx, 2, item_end)
        self.table_tsd_bins.setItem(row_idx, 3, item_cnt)
        self.table_tsd_bins.setItem(row_idx, 4, item_ratio)

    def on_delete_angle_bin(self):
        """Deletes selected angle bin row (preserving minimum 1 row)."""
        current_row = self.table_tsd_bins.currentRow()
        if self.table_tsd_bins.rowCount() <= 1:
            QMessageBox.warning(
                self,
                "Warning" if self.current_lang == "EN" else "경고",
                "At least 1 angle bin is required." if self.current_lang == "EN" else "최소 1개 이상의 각도 구간이 필요합니다.",
            )
            return
        if current_row < 0:
            current_row = self.table_tsd_bins.rowCount() - 1
        self.table_tsd_bins.removeRow(current_row)
        self.on_apply_angle_bins()

    def on_reset_angle_bins(self):
        """Resets custom angle bins back to default 4-way A~D quadrants."""
        self.current_angle_bins = [dict(b) for b in DEFAULT_TSD_ANGLE_BINS]
        self.init_tsd_bins_table()
        self.on_apply_angle_bins()

    def on_apply_angle_bins(self):
        """Parses angle rules from table and re-classifies all detected TSD defects."""
        new_bins = []
        rows = self.table_tsd_bins.rowCount()
        for r in range(rows):
            lbl_item = self.table_tsd_bins.item(r, 0)
            start_item = self.table_tsd_bins.item(r, 1)
            end_item = self.table_tsd_bins.item(r, 2)

            label = lbl_item.text().strip() if lbl_item else f"Bin{r+1}"
            try:
                start_val = float(start_item.text().strip()) if start_item else 0.0
                end_val = float(end_item.text().strip()) if end_item else 360.0
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Warning" if self.current_lang == "EN" else "경고",
                    f"Row {r+1}: Angles must be valid floating point numbers." if self.current_lang == "EN" else f"{r+1}행: 시작 및 끝 각도는 숫자여야 합니다.",
                )
                return

            new_bins.append({
                "label": label,
                "start": start_val % 360.0,
                "end": end_val % 360.0 if end_val != 360.0 else 360.0,
            })

        self.current_angle_bins = new_bins

        # Reclassify all TSD defects
        if self.tsd_defects:
            for d in self.tsd_defects:
                ang = float(d.get("angle_deg", 0.0))
                bin_label = classify_tsd_angle(ang, self.current_angle_bins)
                d["angle_label"] = bin_label
                d["sub_class"] = f"TSD-{bin_label}"

            self.update_analysis_ui()
            if self.last_results and "defects" in self.last_results:
                self.canvas.add_defect_boxes(self.last_results["defects"])

    def show_bpd_histogram(self):
        """Opens modal popup dialog displaying BPD length distribution histogram."""
        lengths = [d["length_um"] for d in self.bpd_defects if "length_um" in d]
        if not lengths:
            QMessageBox.information(
                self,
                "BPD Analysis",
                "No BPD defects detected to plot histogram.\nPlease run defect analysis first." if self.current_lang == "EN" else "검출된 BPD 결함이 없습니다.\n먼저 결함 분석을 실행해주세요.",
            )
            return
        dlg = BPDHistogramDialog(self, lengths, lang=self.current_lang)
        dlg.exec()

    def open_3d_volume_viewer(self):
        """Launches the interactive 3D volumetric inspection dialog."""
        if not (self.is_3d and self.num_slices > 1):
            QMessageBox.information(
                self,
                "3D 뷰어" if self.current_lang == "KO" else "3D Viewer",
                "3D 다중 슬라이스 TIFF 파일이 로드되었을 때 사용할 수 있습니다."
                if self.current_lang == "KO"
                else "Available only when a 3D multi-slice TIFF file is loaded.",
            )
            return

        # Safely collect detected defects if available
        defects = []
        if getattr(self, "current_defects", None):
            defects = self.current_defects
        elif getattr(self, "last_results", None) and "defects" in self.last_results:
            defects = self.last_results["defects"]
        elif getattr(self, "bpd_defects", None) or getattr(self, "tsd_defects", None) or getattr(self, "ted_defects", None):
            defects = (getattr(self, "bpd_defects", []) or []) + (getattr(self, "tsd_defects", []) or []) + (getattr(self, "ted_defects", []) or [])

        dlg = Wafer3DVolumeViewerDialog(
            parent=self,
            tiff_reader=getattr(self, "tiff_reader", None),
            raw_data=self.raw_data,
            num_slices=self.num_slices,
            initial_slice_idx=self.current_slice_idx,
            defects=defects,
            pixel_size_um=self.pixel_size_um,
            lang=self.current_lang,
        )
        dlg.slice_selected.connect(self.slice_slider.setValue)
        dlg.exec()

    def update_analysis_ui(self):
        """Refreshes BPD and TSD analysis summary labels and table rows."""
        # 1. Update BPD table & stats
        self.table_bpd.setRowCount(len(self.bpd_defects))
        lengths_um = []
        for idx, d in enumerate(self.bpd_defects):
            did = d.get("defect_id", idx + 1)
            cx, cy = d.get("center_xy", (0, 0))
            px_len = d.get("length_px", 0.0)
            um_len = d.get("length_um", 0.0)
            lengths_um.append(um_len)

            item_id = QTableWidgetItem(f"#{did}")
            item_coord = QTableWidgetItem(f"({cx:.0f}, {cy:.0f})")
            item_px = QTableWidgetItem(f"{px_len:.1f}")
            item_um = QTableWidgetItem(f"{um_len:.1f}")

            item_id.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_coord.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_px.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_um.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table_bpd.setItem(idx, 0, item_id)
            self.table_bpd.setItem(idx, 1, item_coord)
            self.table_bpd.setItem(idx, 2, item_px)
            self.table_bpd.setItem(idx, 3, item_um)

        bpd_cnt = len(lengths_um)
        if bpd_cnt > 0:
            mean_um = round(float(np.mean(lengths_um)), 1)
            max_um = round(float(np.max(lengths_um)), 1)
            min_um = round(float(np.min(lengths_um)), 1)
        else:
            mean_um, max_um, min_um = 0.0, 0.0, 0.0

        self.lbl_bpd_summary.setText(
            self.tr("analysis_bpd_summary", cnt=bpd_cnt, mean=mean_um, max=max_um, min=min_um)
        )

        # Update BPD Spectrum Bar
        if hasattr(self, "bpd_spectrum_bar"):
            self.bpd_spectrum_bar.set_data(lengths_um, lang=self.current_lang)

        # 2. Update TSD counts and ratios in table
        tsd_counts = {str(b["label"]): 0 for b in self.current_angle_bins}
        for d in self.tsd_defects:
            lbl = d.get("angle_label", "")
            if lbl in tsd_counts:
                tsd_counts[lbl] += 1

        total_tsd = len(self.tsd_defects)
        for r, b in enumerate(self.current_angle_bins):
            if r < self.table_tsd_bins.rowCount():
                lbl = str(b["label"])
                cnt = tsd_counts.get(lbl, 0)
                ratio = (cnt / float(total_tsd) * 100.0) if total_tsd > 0 else 0.0
                item_cnt = self.table_tsd_bins.item(r, 3)
                item_ratio = self.table_tsd_bins.item(r, 4)
                if item_cnt is not None:
                    item_cnt.setText(str(cnt))
                if item_ratio is not None:
                    item_ratio.setText(f"{ratio:.1f}%")

        # Update TSD Gauge Bar
        if hasattr(self, "tsd_gauge_bar"):
            self.tsd_gauge_bar.set_data(self.current_angle_bins, tsd_counts, lang=self.current_lang)

        # 3. Update TED 6-type counts and ratios in table
        ted_counts = {str(b["label"]): 0 for b in self.current_ted_angle_bins}
        for d in self.ted_defects:
            lbl = d.get("angle_label", "")
            if lbl in ted_counts:
                ted_counts[lbl] += 1

        total_ted = len(self.ted_defects)
        for r, b in enumerate(self.current_ted_angle_bins):
            if r < self.table_ted_bins.rowCount():
                lbl = str(b["label"])
                cnt = ted_counts.get(lbl, 0)
                ratio = (cnt / float(total_ted) * 100.0) if total_ted > 0 else 0.0
                item_cnt = self.table_ted_bins.item(r, 3)
                item_ratio = self.table_ted_bins.item(r, 4)
                if item_cnt is not None:
                    item_cnt.setText(str(cnt))
                if item_ratio is not None:
                    item_ratio.setText(f"{ratio:.1f}%")

        # Update TED Gauge Bar
        if hasattr(self, "ted_gauge_bar"):
            self.ted_gauge_bar.set_data(self.current_ted_angle_bins, ted_counts, lang=self.current_lang)

        # 4. Refresh conversion rate
        self.calculate_conversion_rate()

    def init_grid_tab(self):
        """Initializes Tab 3: 3x3 Wafer Grid Density Map and Viewport Focus controls."""
        tab_layout = QVBoxLayout(self.tab_grid)
        tab_layout.setContentsMargins(6, 8, 6, 8)
        tab_layout.setSpacing(8)

        # Overlay Toggle Checkbox
        self.chk_grid_overlay = QCheckBox("Show 3x3 Grid Overlay")
        self.chk_grid_overlay.setStyleSheet("font-weight: bold; color: #00d2ff;")
        self.chk_grid_overlay.setChecked(False)
        self.chk_grid_overlay.stateChanged.connect(self.on_grid_overlay_toggled)
        tab_layout.addWidget(self.chk_grid_overlay)

        self.lbl_grid_intro = QLabel("Click any zone block to focus canvas onto that 1/9 area.")
        self.lbl_grid_intro.setWordWrap(True)
        tab_layout.addWidget(self.lbl_grid_intro)

        # 3x3 Interactive Buttons Grid
        grid_container = QGroupBox("Interactive 3x3 Wafer Zones")
        grid_layout = QGridLayout(grid_container)
        grid_layout.setContentsMargins(6, 8, 6, 8)
        grid_layout.setSpacing(6)

        zone_regions = [
            ["NW", "N", "NE"],
            ["W", "Center", "E"],
            ["SW", "S", "SE"],
        ]

        self.zone_buttons: List[List[QPushButton]] = [[None for _ in range(3)] for _ in range(3)]
        zone_id = 1
        for r in range(3):
            for c in range(3):
                reg = zone_regions[r][c]
                btn = QPushButton(f"Zone {zone_id}\n({reg})\n0 defects")
                btn.setFixedHeight(54)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #252830;
                        border: 1px solid #454c59;
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        border-color: #00d2ff;
                        background-color: #2d3440;
                    }
                """)
                # Capture current r, c in lambda
                btn.clicked.connect(lambda _, row=r, col=c: self.canvas.focus_on_zone(row, col))
                grid_layout.addWidget(btn, r, c)
                self.zone_buttons[r][c] = btn
                zone_id += 1

        tab_layout.addWidget(grid_container)

        # Hotspot Summary Label
        self.lbl_grid_hotspot = QLabel("Run defect analysis to compute regional density map.")
        self.lbl_grid_hotspot.setWordWrap(True)
        self.lbl_grid_hotspot.setStyleSheet("font-weight: bold; color: #ff9800; padding: 4px 0;")
        tab_layout.addWidget(self.lbl_grid_hotspot)

        # 3x3 Summary Table (Zone, Region, Count, Ratio)
        self.table_grid_summary = QTableWidget(9, 4)
        self.table_grid_summary.setHorizontalHeaderLabels(["Zone", "Region", "Count", "Ratio"])
        self.table_grid_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_grid_summary.verticalHeader().setVisible(False)
        self.init_grid_table()
        tab_layout.addWidget(self.table_grid_summary)

        tab_layout.addStretch(1)

    def on_grid_overlay_toggled(self, state: int):
        """Synchronizes 3x3 grid visibility across canvas and both checkboxes."""
        is_visible = (state == 2 or state == Qt.CheckState.Checked.value)
        self.canvas.set_grid_visibility(is_visible)
        if self.chk_grid_overlay.isChecked() != is_visible:
            self.chk_grid_overlay.setChecked(is_visible)
        if self.chk_bottom_grid.isChecked() != is_visible:
            self.chk_bottom_grid.setChecked(is_visible)

    def init_grid_table(self):
        """Initializes empty 9-row grid density table."""
        zone_regions = [
            ("Zone 1", "NW"), ("Zone 2", "N"), ("Zone 3", "NE"),
            ("Zone 4", "W"),  ("Zone 5", "Center"), ("Zone 6", "E"),
            ("Zone 7", "SW"), ("Zone 8", "S"), ("Zone 9", "SE"),
        ]
        for row_idx, (zname, reg) in enumerate(zone_regions):
            item_z = QTableWidgetItem(zname)
            item_r = QTableWidgetItem(reg)
            item_c = QTableWidgetItem("0")
            item_p = QTableWidgetItem("0.0%")
            item_c.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_p.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_grid_summary.setItem(row_idx, 0, item_z)
            self.table_grid_summary.setItem(row_idx, 1, item_r)
            self.table_grid_summary.setItem(row_idx, 2, item_c)
            self.table_grid_summary.setItem(row_idx, 3, item_p)

    def update_grid_stats(self, defects: Optional[List[Dict[str, Any]]] = None):
        """Calculates 3x3 regional defect density and updates buttons and table."""
        zone_regions = [
            ["NW", "N", "NE"],
            ["W", "Center", "E"],
            ["SW", "S", "SE"],
        ]
        if self.current_slice_2d is None or not defects:
            zone_id = 1
            for r in range(3):
                for c in range(3):
                    reg = zone_regions[r][c]
                    btn = self.zone_buttons[r][c]
                    if btn is not None:
                        btn.setText(f"Zone {zone_id}\n({reg})\n0 defects\n(0.0%)")
                        btn.setStyleSheet("background-color: #252830; border: 1px solid #454c59; border-radius: 4px; font-size: 11px; font-weight: bold;")
                    zone_id += 1
            self.init_grid_table()
            self.lbl_grid_hotspot.setText(self.tr("grid_no_data"))
            return

        h, w = self.current_slice_2d.shape[:2]
        zw = w / 3.0
        zh = h / 3.0
        zone_counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for d in defects:
            cx, cy = d["center_xy"]
            zr = min(2, max(0, int(cy // zh)))
            zc = min(2, max(0, int(cx // zw)))
            zone_counts[zr][zc] += 1

        total = max(1, len(defects))
        max_cnt = max(max(row) for row in zone_counts)
        hotspot_zone = ""
        hotspot_cnt = -1

        zone_id = 1
        table_row = 0
        for r in range(3):
            for c in range(3):
                cnt = zone_counts[r][c]
                pct = (cnt / float(total)) * 100.0
                reg = zone_regions[r][c]
                ztitle = f"Zone {zone_id}"

                if cnt > hotspot_cnt:
                    hotspot_cnt = cnt
                    hotspot_zone = f"{ztitle} ({reg})"

                # Heatmap background color for button
                if max_cnt > 0 and cnt > 0:
                    intensity = int((cnt / float(max_cnt)) * 120)
                    bg_color = f"rgb({intensity + 35}, 40, {max(20, 60 - intensity // 2)})"
                    border_color = "#ff5722" if cnt == max_cnt else "#00d2ff"
                else:
                    bg_color = "#252830"
                    border_color = "#454c59"

                btn = self.zone_buttons[r][c]
                if btn is not None:
                    btn.setText(f"{ztitle}\n({reg})\n{cnt} defects\n({pct:.1f}%)")
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {bg_color};
                            border: 1.5px solid {border_color};
                            border-radius: 4px;
                            font-size: 11px;
                            font-weight: bold;
                            color: #ffffff;
                        }}
                    """)

                # Table update
                if self.table_grid_summary.item(table_row, 2) is not None:
                    self.table_grid_summary.item(table_row, 2).setText(str(cnt))
                    self.table_grid_summary.item(table_row, 3).setText(f"{pct:.1f}%")

                zone_id += 1
                table_row += 1

        hotspot_pct = (hotspot_cnt / float(total)) * 100.0 if total > 0 else 0.0
        self.lbl_grid_hotspot.setText(
            self.tr("grid_hotspot", zone=hotspot_zone, count=hotspot_cnt, pct=f"{hotspot_pct:.1f}")
        )

    def retranslate_ui(self):
        """Updates all visible UI strings based on the currently selected language."""
        self.setWindowTitle(self.tr("window_title"))
        self.btn_lang.setText(self.tr("lang_button"))

        # Tabs
        self.tab_widget.setTabText(0, self.tr("tab_overview"))
        self.tab_widget.setTabText(1, self.tr("tab_analysis"))
        self.tab_widget.setTabText(2, self.tr("tab_grid"))

        # Slice label
        if self.is_3d and self.num_slices > 1:
            self.slice_label.setText(self.tr("slice_label", cur=self.current_slice_idx + 1, tot=self.num_slices))
        else:
            self.slice_label.setText(self.tr("slice_label", cur=1, tot=1))

        # Bottom controls
        self.lbl_brightness.setText(self.tr("brightness"))
        self.lbl_contrast.setText(self.tr("contrast"))
        self.btn_fit.setText(self.tr("fit_view"))
        self.btn_100.setText(self.tr("zoom_100"))

        # Tab 1: Overview
        self.grp_file.setTitle(self.tr("grp_file"))
        self.btn_open.setText(self.tr("btn_open"))
        self.btn_change_model.setText(self.tr("btn_change_model"))
        self.lbl_model_info.setText(self.tr("model_info", name=self.model_path.name))

        if self.current_file_path is None or self.current_meta is None:
            self.lbl_file_info.setText(self.tr("no_image"))
        else:
            h, w = self.current_slice_2d.shape[:2]
            self.lbl_file_info.setText(
                self.tr(
                    "file_info",
                    name=self.current_file_path.name,
                    w=w,
                    h=h,
                    dtype=self.current_meta["dtype"],
                    min_val=self.current_meta["min_val"],
                    max_val=self.current_meta["max_val"],
                )
            )

        self.grp_analysis.setTitle(self.tr("grp_analysis"))
        self.btn_run.setText(self.tr("btn_run"))

        # Status text update with modern status badges
        if self.worker is None or not self.worker.isRunning():
            if self.last_results is not None:
                counts = self.last_results.get("counts", {})
                bpd = counts.get("BPD", 0)
                ted = counts.get("TED", 0)
                tsd = counts.get("TSD", 0)
                particle = counts.get("Particle", 0)
                crystal = bpd + ted + tsd
                total = crystal + particle
                badge = '<span style="background-color: #163326; color: #00e676; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">● COMPLETED</span> '
                self.lbl_status.setText(badge + self.tr("status_complete", total=total, crystal=crystal, particle=particle))
            elif self.current_slice_2d is not None:
                badge = '<span style="background-color: #1f3747; color: #58a6ff; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">● LOADED</span> '
                self.lbl_status.setText(badge + self.tr("status_loaded"))
            else:
                badge = '<span style="background-color: #272b34; color: #8b949e; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">● READY</span> '
                self.lbl_status.setText(badge + self.tr("status_ready"))

        self.grp_results.setTitle(self.tr("grp_results"))
        self.table_summary.setHorizontalHeaderLabels(self.tr("table_headers"))

        row_keys = [
            "row_bpd",
            "row_ted",
            "row_tsd",
            "row_crystal_subtotal",
            "row_particle",
            "row_grand_total",
        ]
        for row_idx, key in enumerate(row_keys):
            item = self.table_summary.item(row_idx, 0)
            if item is not None:
                item.setText(self.tr(key))

        self.chk_bpd.setText(self.tr("chk_bpd"))
        self.chk_ted.setText(self.tr("chk_ted"))
        self.chk_tsd.setText(self.tr("chk_tsd"))
        self.chk_particle.setText(self.tr("chk_particle"))
        self.btn_export.setText(self.tr("btn_export"))

        # Tab 2: Precision Analysis
        self.lbl_pixel_scale.setText(self.tr("pixel_scale_label"))
        self.grp_bpd_analysis.setTitle(self.tr("analysis_bpd_title"))
        self.lbl_bpd_desc.setText(self.tr("analysis_bpd_desc"))
        self.table_bpd.setHorizontalHeaderLabels(self.tr("analysis_bpd_headers"))
        self.btn_bpd_hist.setText(self.tr("analysis_bpd_btn"))

        # 3D Depth Window & Viewer
        self.lbl_depth_range.setText(self.tr("slice_depth_range"))
        self.spin_depth_start.setToolTip(self.tr("tooltip_depth_start"))
        self.spin_depth_end.setToolTip(self.tr("tooltip_depth_end"))
        if hasattr(self, "btn_3d_viewer"):
            self.btn_3d_viewer.setText(self.tr("btn_3d_viewer"))
            self.btn_3d_viewer.setToolTip(self.tr("tooltip_3d_viewer"))

        # Tab 2: Precision Analysis - BPD Length
        self.lbl_pixel_scale.setText(self.tr("pixel_scale_label"))
        self.grp_bpd_analysis.setTitle(self.tr("analysis_bpd_title"))
        self.lbl_bpd_desc.setText(self.tr("analysis_bpd_desc"))
        self.table_bpd.setHorizontalHeaderLabels(self.tr("analysis_bpd_headers"))
        self.btn_bpd_hist.setText(self.tr("analysis_bpd_btn"))

        # Conversion Card
        self.grp_conversion.setTitle(self.tr("analysis_conversion_title"))
        self.lbl_conv_desc.setText(self.tr("analysis_conversion_desc"))
        self.lbl_before_bpd.setText(self.tr("lbl_before_bpd"))
        self.lbl_after_bpd.setText(self.tr("lbl_after_bpd"))
        self.btn_apply_current_bpd.setText(self.tr("btn_apply_current_bpd"))
        self.lbl_converted_ted_title.setText(self.tr("lbl_converted_ted"))
        self.lbl_conversion_rate_title.setText(self.tr("lbl_conversion_rate"))
        self.lbl_conv_formula.setText(self.tr("conversion_formula"))

        # TED 6-Type
        self.grp_ted_analysis.setTitle(self.tr("analysis_ted_title"))
        self.lbl_ted_desc.setText(self.tr("analysis_ted_desc"))
        self.table_ted_bins.setHorizontalHeaderLabels(self.tr("analysis_ted_headers"))
        self.btn_reset_ted_bins.setText(self.tr("btn_reset_ted_bins"))

        # TSD 4-Way
        self.grp_tsd_analysis.setTitle(self.tr("analysis_tsd_title"))
        self.lbl_tsd_desc.setText(self.tr("analysis_tsd_desc"))
        self.table_tsd_bins.setHorizontalHeaderLabels(self.tr("analysis_tsd_headers"))
        self.btn_add_bin.setText(self.tr("btn_add_bin"))
        self.btn_del_bin.setText(self.tr("btn_del_bin"))
        self.btn_reset_bins.setText(self.tr("btn_reset_bins"))
        self.btn_apply_bins.setText(self.tr("btn_apply_bins"))
        self.update_analysis_ui()

        # Tab 3: Grid
        self.chk_grid_overlay.setText(self.tr("chk_grid_overlay"))
        self.lbl_grid_intro.setText(self.tr("grid_intro"))
        self.table_grid_summary.setHorizontalHeaderLabels(self.tr("grid_headers"))
        if self.last_results is None:
            self.lbl_grid_hotspot.setText(self.tr("grid_no_data"))

        # Comprehensive Tooltips
        self.btn_open.setToolTip(self.tr("tooltip_open"))
        self.btn_change_model.setToolTip(self.tr("tooltip_model"))
        self.btn_run.setToolTip(self.tr("tooltip_run"))
        self.btn_export.setToolTip(self.tr("tooltip_export"))
        self.brightness_slider.setToolTip(self.tr("tooltip_brightness"))
        self.contrast_slider.setToolTip(self.tr("tooltip_contrast"))
        self.btn_fit.setToolTip(self.tr("tooltip_fit"))
        self.btn_100.setToolTip(self.tr("tooltip_100"))
        self.chk_bottom_grid.setToolTip(self.tr("tooltip_grid"))
        self.chk_grid_overlay.setToolTip(self.tr("tooltip_grid"))
        self.spin_pixel_scale.setToolTip(self.tr("tooltip_scale"))
        self.btn_bpd_hist.setToolTip(self.tr("tooltip_bpd_hist"))
        self.spin_before_bpd.setToolTip(self.tr("tooltip_before_bpd"))
        self.spin_after_bpd.setToolTip(self.tr("tooltip_after_bpd"))
        self.btn_apply_current_bpd.setToolTip(self.tr("tooltip_apply_current_bpd"))
        self.btn_reset_ted_bins.setToolTip(self.tr("tooltip_reset_ted_bins"))
        self.btn_add_bin.setToolTip(self.tr("tooltip_add_bin"))
        self.btn_del_bin.setToolTip(self.tr("tooltip_del_bin"))
        self.btn_reset_bins.setToolTip(self.tr("tooltip_reset_bins"))
        self.btn_apply_bins.setToolTip(self.tr("tooltip_apply_bins"))

        if hasattr(self, "status_file_lbl"):
            self.update_status_bar()

    # --------------------------------------------------------------------------
    # File Loading and Image Rendering
    # --------------------------------------------------------------------------

    def on_open_file_dialog(self):
        """Opens cross-platform native file dialog to choose an XRT image."""
        start_dir = Path("data") / "raw"
        if not start_dir.exists():
            start_dir = Path.cwd()

        file_str, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("dialog_open_image"),
            str(start_dir),
            "All Supported Images (*.tif *.tiff *.jpg *.jpeg *.png *.bmp *.webp);;TIFF Images (*.tif *.tiff);;JPEG Images (*.jpg *.jpeg);;PNG Images (*.png);;All Files (*.*)",
        )
        if file_str:
            self.load_image(Path(file_str))

    def on_select_model(self):
        """Allows selecting custom ONNX or PT model weights."""
        file_str, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("dialog_select_model"),
            str(self.model_path.parent if self.model_path.exists() else Path.cwd()),
            "ONNX / YOLO Models (*.onnx *.pt);;ONNX Models (*.onnx);;PyTorch Models (*.pt);;All Files (*.*)",
        )
        if file_str:
            self.model_path = Path(file_str)
            self.lbl_model_info.setText(self.tr("model_info", name=self.model_path.name))

    def load_image(self, file_path: Path):
        """Loads a 2D or 3D TIFF image and updates canvas."""
        try:
            self.current_file_path = file_path

            # Clean up existing open tiff reader
            if hasattr(self, "tiff_reader") and self.tiff_reader is not None:
                try:
                    self.tiff_reader.close()
                except Exception:
                    pass
                self.tiff_reader = None

            raw_img, meta = inspect_and_load_xrt(file_path, slice_idx=0)

            self.is_3d = meta["is_3d"]
            self.num_slices = meta["num_slices"]
            self.current_slice_idx = 0
            self.current_meta = meta

            # Configure slice slider & depth spinboxes for 3D stacks
            if self.is_3d and self.num_slices > 1:
                import tifffile
                self.tiff_reader = tifffile.TiffFile(str(file_path))
                self.current_slice_2d = raw_img

                self.slice_slider.setEnabled(True)
                self.slice_slider.blockSignals(True)
                self.slice_slider.setRange(0, self.num_slices - 1)
                self.slice_slider.setValue(0)
                self.slice_slider.blockSignals(False)

                self.spin_depth_start.setEnabled(True)
                self.spin_depth_end.setEnabled(True)
                self.spin_depth_start.blockSignals(True)
                self.spin_depth_end.blockSignals(True)
                self.spin_depth_start.setRange(1, self.num_slices)
                self.spin_depth_end.setRange(1, self.num_slices)
                self.spin_depth_start.setValue(1)
                self.spin_depth_end.setValue(self.num_slices)
                self.spin_depth_start.blockSignals(False)
                self.spin_depth_end.blockSignals(False)

                self.slice_label.setText(self.tr("slice_label", cur=1, tot=self.num_slices))
                if hasattr(self, "btn_3d_viewer"):
                    self.btn_3d_viewer.setEnabled(True)
            else:
                self.current_slice_2d = raw_img
                self.slice_slider.setEnabled(False)
                self.slice_slider.blockSignals(True)
                self.slice_slider.setRange(0, 0)
                self.slice_slider.blockSignals(False)

                self.spin_depth_start.setEnabled(False)
                self.spin_depth_end.setEnabled(False)
                self.spin_depth_start.blockSignals(True)
                self.spin_depth_end.blockSignals(True)
                self.spin_depth_start.setRange(1, 1)
                self.spin_depth_end.setRange(1, 1)
                self.spin_depth_start.setValue(1)
                self.spin_depth_end.setValue(1)
                self.spin_depth_start.blockSignals(False)
                self.spin_depth_end.blockSignals(False)
                if hasattr(self, "btn_3d_viewer"):
                    self.btn_3d_viewer.setEnabled(False)

                self.slice_label.setText(self.tr("slice_label", cur=1, tot=1))

            # File info label
            h, w = self.current_slice_2d.shape[:2]
            self.lbl_file_info.setText(
                self.tr(
                    "file_info",
                    name=file_path.name,
                    w=w,
                    h=h,
                    dtype=meta["dtype"],
                    min_val=meta["min_val"],
                    max_val=meta["max_val"],
                )
            )

            self.canvas.clear_overlays()
            self.btn_export.setEnabled(False)
            self.last_results = None
            self.bpd_defects.clear()
            self.tsd_defects.clear()
            self.ted_defects.clear()
            self.current_defects.clear()
            self.init_summary_table()
            self.update_analysis_ui()

            # Pre-compute base normalization and proxy cache
            self.prepare_slice_cache()
            self.render_fullres_image()
            self.canvas.fit_in_view()

            # Build 3x3 grid overlay and reset grid density stats
            self.canvas.update_grid_overlay(w, h)
            self.update_grid_stats(None)

            badge = '<span style="background-color: #1f3747; color: #58a6ff; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">● LOADED</span> '
            self.lbl_status.setText(badge + self.tr("status_loaded"))
            if hasattr(self, "status_file_lbl"):
                self.update_status_bar()
        except Exception as e:
            QMessageBox.critical(self, self.tr("load_error_title"), self.tr("load_error_msg", e=e))

    def on_slice_changed(self, value: int):
        """Switches current slice in 3D TIFF stack with instantaneous proxy feedback."""
        if not self.is_3d:
            return

        self.current_slice_idx = value
        self.slice_label.setText(self.tr("slice_label", cur=value + 1, tot=self.num_slices))

        # Stream requested slice from TiffFile on demand (<25ms)
        if hasattr(self, "tiff_reader") and self.tiff_reader is not None:
            self.current_slice_2d = self.tiff_reader.pages[value].asarray()
        elif self.raw_data is not None and value < len(self.raw_data):
            self.current_slice_2d = self.raw_data[value]

        self.prepare_slice_cache()

        # Immediate proxy rendering for ultra-smooth 60fps sliding feedback
        lut = self.get_brightness_contrast_lut()
        adjusted_proxy = cv2.LUT(self.proxy_norm_8bit, lut)
        ph, pw = adjusted_proxy.shape[:2]
        qimg = QImage(adjusted_proxy.data, pw, ph, pw, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self.canvas.set_display_pixmap(pixmap, scale_x=self.proxy_scale_x, scale_y=self.proxy_scale_y, auto_fit=False)

        # Schedule full-resolution rendering once sliding pauses
        self.debounce_timer.start(60)

        if hasattr(self, "status_file_lbl"):
            self.update_status_bar()

    def prepare_slice_cache(self):
        """Pre-normalizes slice and pre-generates lightweight proxy cache for 60fps sliders."""
        if self.current_slice_2d is None:
            return

        # 1. Compute full base 8-bit normalization ONCE
        self.base_norm_8bit = normalize_16bit_to_8bit(self.current_slice_2d)

        # 2. Compute lightweight proxy (max 2048x2048) for instant slider feedback
        h, w = self.base_norm_8bit.shape[:2]
        max_proxy = 2048
        if max(h, w) > max_proxy:
            scale = max_proxy / float(max(h, w))
            pw = max(1, int(round(w * scale)))
            ph = max(1, int(round(h * scale)))
            self.proxy_norm_8bit = cv2.resize(self.base_norm_8bit, (pw, ph), interpolation=cv2.INTER_AREA)
            self.proxy_scale_x = w / float(pw)
            self.proxy_scale_y = h / float(ph)
        else:
            self.proxy_norm_8bit = self.base_norm_8bit
            self.proxy_scale_x = 1.0
            self.proxy_scale_y = 1.0

    def get_brightness_contrast_lut(self) -> np.ndarray:
        """Computes 256-element Look-Up Table (LUT) for O(1) brightness/contrast scaling."""
        b_offset = float(self.brightness_slider.value())
        c_scale = float(self.contrast_slider.value()) / 100.0

        indices = np.arange(256, dtype=np.float32)
        lut = np.clip((indices - 128.0) * c_scale + 128.0 + b_offset, 0.0, 255.0).astype(np.uint8)
        return lut

    def on_slider_value_changed(self):
        """Immediately renders lightweight proxy image (<2ms) and schedules high-res render."""
        if self.proxy_norm_8bit is None:
            return

        # 1. Instant 60 FPS feedback using proxy + cv2.LUT
        lut = self.get_brightness_contrast_lut()
        adjusted_proxy = cv2.LUT(self.proxy_norm_8bit, lut)

        ph, pw = adjusted_proxy.shape[:2]
        qimg = QImage(adjusted_proxy.data, pw, ph, pw, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        # Display with scale so bounding boxes remain perfectly aligned
        self.canvas.set_display_pixmap(pixmap, scale_x=self.proxy_scale_x, scale_y=self.proxy_scale_y, auto_fit=False)

        # 2. Debounce: Schedule full 12k rendering after 80ms of inactivity
        self.debounce_timer.start(80)

    def render_fullres_image(self):
        """Renders original full-resolution (e.g. 12k x 12k) image using cv2.LUT."""
        self.debounce_timer.stop()
        if self.base_norm_8bit is None:
            return

        lut = self.get_brightness_contrast_lut()
        adjusted_full = cv2.LUT(self.base_norm_8bit, lut)

        h, w = adjusted_full.shape[:2]
        qimg = QImage(adjusted_full.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        self.canvas.set_display_pixmap(pixmap, scale_x=1.0, scale_y=1.0, auto_fit=False)

    def update_image_display(self):
        """Legacy compatibility wrapper calling render_fullres_image."""
        self.render_fullres_image()

    # --------------------------------------------------------------------------
    # Inference Pipeline
    # --------------------------------------------------------------------------

    def start_analysis(self):
        """Starts asynchronous background wafer defect inference."""
        if self.current_slice_2d is None:
            QMessageBox.warning(self, self.tr("warning_no_image_title"), self.tr("warning_no_image_msg"))
            return

        if not self.model_path.exists():
            QMessageBox.warning(
                self,
                self.tr("warning_no_model_title"),
                self.tr("warning_no_model_msg", path=self.model_path),
            )
            return

        self.btn_run.setEnabled(False)
        badge = '<span style="background-color: #3b2c14; color: #ff9800; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">● SCANNING</span> '
        self.lbl_status.setText(badge + self.tr("status_scanning"))
        self.progress_bar.setValue(0)
        self.canvas.show_loading(self.tr("loading_scanning"), "0% (0 / 1)")

        # Launch QThread worker
        self.worker = InferenceWorker(
            image_array=self.current_slice_2d,
            model_path=self.model_path,
            conf_threshold=0.20,
            patch_size=640,
            overlap=0.2,
        )
        self.worker.progress_signal.connect(self.on_inference_progress)
        self.worker.finished_signal.connect(self.on_inference_finished)
        self.worker.error_signal.connect(self.on_inference_error)
        self.worker.start()

    def on_inference_progress(self, current: int, total: int):
        """Updates progress bar and canvas overlay spinner."""
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
            badge = '<span style="background-color: #3b2c14; color: #ff9800; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">● SCANNING</span> '
            self.lbl_status.setText(badge + self.tr("status_inferring", cur=current, tot=total, pct=pct))
            self.canvas.set_loading_progress(current, total, pct)

    def on_inference_finished(self, results: Dict[str, Any]):
        """Handles completion of inference, runs precision measurements, and updates UI."""
        self.canvas.hide_loading()
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.last_results = results

        total = results.get("total_defects", 0)
        counts = results.get("counts", {})
        defects = results.get("defects", [])

        # Precision Domain Measurements (BPD Length via PCA + TSD Angle via Moments + TED 6-Type)
        if self.current_slice_2d is not None and defects:
            analyze_domain_measurements(
                image=self.current_slice_2d,
                defects=defects,
                pixel_size_um=self.pixel_size_um,
                angle_bins=self.current_angle_bins,
                ted_angle_bins=self.current_ted_angle_bins,
            )

        self.current_defects = list(defects)
        self.bpd_defects = [d for d in defects if d.get("class_name") == "BPD"]
        self.tsd_defects = [d for d in defects if d.get("class_name") == "TSD"]
        self.ted_defects = [d for d in defects if d.get("class_name") == "TED"]

        bpd_cnt = counts.get("BPD", 0)
        ted_cnt = counts.get("TED", 0)
        tsd_cnt = counts.get("TSD", 0)
        particle_cnt = counts.get("Particle", 0)
        crystal_subtotal = bpd_cnt + ted_cnt + tsd_cnt
        grand_total = crystal_subtotal + particle_cnt

        badge = '<span style="background-color: #163326; color: #00e676; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">● COMPLETED</span> '
        self.lbl_status.setText(
            badge + self.tr("status_complete", total=grand_total, crystal=crystal_subtotal, particle=particle_cnt)
        )

        # Update 6-Row Summary Table
        self.table_summary.item(0, 1).setText(str(bpd_cnt))
        self.table_summary.item(1, 1).setText(str(ted_cnt))
        self.table_summary.item(2, 1).setText(str(tsd_cnt))
        self.table_summary.item(3, 1).setText(str(crystal_subtotal))
        self.table_summary.item(4, 1).setText(str(particle_cnt))
        self.table_summary.item(5, 1).setText(str(grand_total))

        # Update Precision Analysis Tab UI (BPD table, TSD bins, TED bins, Conversion Rate)
        self.update_analysis_ui()

        # Draw Bounding Boxes on Canvas with updated tooltips
        self.canvas.add_defect_boxes(defects)

        # Apply Checkbox Filter States
        self.canvas.set_overlay_visibility("BPD", self.chk_bpd.isChecked())
        self.canvas.set_overlay_visibility("TED", self.chk_ted.isChecked())
        self.canvas.set_overlay_visibility("TSD", self.chk_tsd.isChecked())
        self.canvas.set_overlay_visibility("Particle", self.chk_particle.isChecked())

        # Update 3x3 Wafer Grid Density Map and Block Buttons
        self.update_grid_stats(defects)

        self.btn_export.setEnabled(True)

    def on_inference_error(self, err_msg: str):
        """Handles inference failure."""
        self.canvas.hide_loading()
        self.btn_run.setEnabled(True)
        badge = '<span style="background-color: #3a181c; color: #ff5252; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">● ERROR</span> '
        self.lbl_status.setText(badge + self.tr("status_failed"))
        QMessageBox.critical(self, self.tr("infer_error_title"), self.tr("infer_error_msg", e=err_msg))

    # --------------------------------------------------------------------------
    # Report Exporting
    # --------------------------------------------------------------------------

    def export_report(self):
        """Exports detected defects to user-specified CSV or JSON file with precision metrics."""
        if not self.last_results:
            return

        stem = self.current_file_path.stem if self.current_file_path else "wafer"
        default_file = Path("runs") / "inference" / f"{stem}_defects.csv"
        default_file.parent.mkdir(parents=True, exist_ok=True)

        save_str, selected_filter = QFileDialog.getSaveFileName(
            self,
            self.tr("dialog_save_report"),
            str(default_file),
            "CSV Files (*.csv);;JSON Files (*.json)",
        )
        if not save_str:
            return

        save_path = Path(save_str)
        defects = self.last_results.get("defects", [])

        if save_path.suffix.lower() == ".json":
            out_data = dict(self.last_results)
            out_data["conversion_stats"] = self.conversion_stats
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
        else:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
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
                    "Angle_deg",
                    "Angle_Label",
                    "TSD_Angle_deg",
                    "TSD_Angle_Label",
                ])
                for d in defects:
                    ang = d.get("angle_deg", "N/A")
                    ang_lbl = d.get("angle_label", "N/A")
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
                        ang,
                        ang_lbl,
                        ang,
                        ang_lbl,
                    ])

        QMessageBox.information(self, self.tr("export_success_title"), self.tr("export_success_msg", path=save_path))

    def closeEvent(self, event):
        """Safely cleans up background worker and open TIFF reader handles."""
        if hasattr(self, "tiff_reader") and self.tiff_reader is not None:
            try:
                self.tiff_reader.close()
            except Exception:
                pass
            self.tiff_reader = None
        super().closeEvent(event)


def launch_gui(image_path: Optional[Union[str, Path]] = None):
    """Launches the PyQt6 desktop application."""
    # High DPI Scaling attributes
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern cross-platform default look

    # Set modern dark polished stylesheet
    dark_stylesheet = """
        QMainWindow, QWidget {
            background-color: #17191e;
            color: #dcdcdc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            background-color: #1c1f26;
            border: 1px solid #333842;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            color: #58a6ff;
            font-size: 12px;
        }
        QPushButton {
            background-color: #272b34;
            border: 1px solid #3e4452;
            border-radius: 5px;
            padding: 6px 12px;
            color: #f0f2f5;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #343a47;
            border-color: #58a6ff;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #1c1f26;
        }
        QPushButton:disabled {
            background-color: #1e2128;
            border-color: #2b2e36;
            color: #5c6370;
        }
        QTableWidget {
            background-color: #14161b;
            border: 1px solid #313640;
            border-radius: 6px;
            gridline-color: #262a33;
            selection-background-color: #1f3d59;
            selection-color: #ffffff;
        }
        QTableWidget::item {
            padding: 4px 6px;
            border-radius: 3px;
        }
        QTableWidget::item:hover {
            background-color: #242933;
        }
        QTableWidget::item:selected {
            background-color: #1a4266;
            color: #ffffff;
        }
        QHeaderView::section {
            background-color: #1f232b;
            color: #8b949e;
            padding: 5px 8px;
            border: none;
            border-bottom: 1px solid #313640;
            border-right: 1px solid #282c35;
            font-weight: bold;
            font-size: 11px;
        }
        QProgressBar {
            border: 1px solid #313640;
            border-radius: 4px;
            text-align: center;
            background-color: #14161b;
            color: #e6edf3;
            font-size: 11px;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: #007acc;
            border-radius: 3px;
        }
        QSlider::groove:horizontal {
            height: 5px;
            background: #2a2e38;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #007acc;
            border: 1.5px solid #58a6ff;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #58a6ff;
        }
        QToolTip {
            background-color: #252830;
            color: #f0f2f5;
            border: 1px solid #58a6ff;
            border-radius: 5px;
            padding: 6px 10px;
            font-size: 12px;
        }
        QDoubleSpinBox, QSpinBox, QLineEdit {
            background-color: #14161b;
            border: 1px solid #3c424e;
            border-radius: 4px;
            padding: 4px 6px;
            color: #e6edf3;
        }
        QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
            border-color: #58a6ff;
        }
        QScrollBar:vertical {
            background: #17191e;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #333842;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #444b57;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """
    app.setStyleSheet(dark_stylesheet)

    window = SiCDefectMainWindow(initial_image_path=image_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
