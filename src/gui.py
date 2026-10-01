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
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.inference import CLASS_NAMES, get_inference_engine
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
        "btn_open": "📂 XRT 이미지 열기 (.tif / .tiff)",
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
        "analysis_bpd_title": "BPD 전위 길이 분포 분석 (Dislocation Length)",
        "analysis_bpd_desc": "BPD 결함의 선형 전파 길이(px) 필터링 및 통계 분석 (연구원 설정)",
        "analysis_bpd_min": "최소 길이 필터: {val} px",
        "analysis_bpd_max": "최대 길이 필터: {val} px",
        "analysis_bpd_stats": "평균 길이: N/A | 최대 길이: N/A | 표준편차: N/A",
        "analysis_bpd_btn": "📈 BPD 길이 분포 히스토그램 생성",
        "analysis_tsd_title": "TSD 전파 각도 및 코어 프로파일링 (Propagation Angle)",
        "analysis_tsd_desc": "웨이퍼 축 기준 전파 각도 및 코어 반경 마진 설정 (연구원 설정)",
        "analysis_tsd_angle": "기준 각도 축: {val}°",
        "analysis_tsd_margin": "코어 반경 마진: {val} px",
        "analysis_tsd_btn": "🧭 각도 벡터 맵 생성",
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
    },
    "EN": {
        "window_title": "KICET SiC Wafer Defect Detection & Classification (ONNX Runtime)",
        "lang_button": "🌐 한국어 (KO)",
        "slice_label": "Slice: {cur}/{tot}",
        "brightness": "Brightness:",
        "contrast": "Contrast:",
        "fit_view": "Fit to View",
        "zoom_100": "100% (1:1)",
        "tab_overview": "📊 Overview",
        "tab_analysis": "📐 Analysis",
        "tab_grid": "🧭 Wafer Grid",
        "grp_file": "File & Model Configuration",
        "btn_open": "📂 Open XRT Image (.tif / .tiff)",
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
        "analysis_bpd_title": "BPD Dislocation Length Distribution",
        "analysis_bpd_desc": "BPD linear propagation length (px) threshold filtering & stats",
        "analysis_bpd_min": "Min Length Filter: {val} px",
        "analysis_bpd_max": "Max Length Filter: {val} px",
        "analysis_bpd_stats": "Mean Length: N/A | Max Length: N/A | Std Dev: N/A",
        "analysis_bpd_btn": "📈 Generate BPD Length Histogram",
        "analysis_tsd_title": "TSD Propagation Angle & Core Profiling",
        "analysis_tsd_desc": "Wafer axis propagation angle vector & core radius margin",
        "analysis_tsd_angle": "Reference Angle Axis: {val}°",
        "analysis_tsd_margin": "Core Radius Margin: {val} px",
        "analysis_tsd_btn": "🧭 Generate Angle Vector Map",
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
# Interactive High-Resolution Wafer Graphics View
# ==============================================================================

class WaferCanvasView(QGraphicsView):
    """QGraphicsView with smooth cursor-centered wheel zoom and drag panning."""

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

    def reset_zoom_100(self):
        """Resets zoom to 100% pixel scale."""
        self.resetTransform()

    def clear_overlays(self):
        """Removes all bounding box overlays."""
        for cname, items in self.overlay_items.items():
            for item in items:
                self.scene.removeItem(item)
            items.clear()

    def add_defect_boxes(self, defects: List[Dict[str, Any]]):
        """Draws bounding boxes and labels for detected defects."""
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

            # Tooltip with defect details
            conf = d.get("confidence", 0.0)
            rect_item.setToolTip(f"Class: {cname}\nConf: {conf:.3f}\nCoord: ({bx1:.1f}, {by1:.1f})\nSize: {width:.1f}x{height:.1f}")

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
            Path("runs") / "train" / "sic_defect_train" / "weights" / "best.onnx",
            Path("runs") / "train" / "sic_defect_train" / "weights" / "best.pt",
            Path("best.onnx"),
            Path("yolov8n.onnx"),
            Path("yolov8n.pt"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return (Path("runs") / "train" / "sic_defect_train" / "weights" / "best.onnx").resolve()

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

        # 1. 3D Slice Slider
        self.slice_label = QLabel("Slice: 1/1")
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        bottom_layout.addWidget(self.slice_label)
        bottom_layout.addWidget(self.slice_slider, stretch=1)

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
        """Initializes Tab 2: Precision Analysis placeholder for BPD length & TSD propagation angle."""
        tab_layout = QVBoxLayout(self.tab_analysis)
        tab_layout.setContentsMargins(6, 8, 6, 8)
        tab_layout.setSpacing(10)

        # Group 1: BPD Dislocation Length Distribution
        self.grp_bpd_analysis = QGroupBox("BPD Dislocation Length Distribution")
        bpd_layout = QVBoxLayout(self.grp_bpd_analysis)

        self.lbl_bpd_desc = QLabel("BPD linear propagation length (px) threshold filtering & stats")
        self.lbl_bpd_desc.setWordWrap(True)
        bpd_layout.addWidget(self.lbl_bpd_desc)

        # Sliders for min/max length filtering
        self.lbl_bpd_min = QLabel("Min Length Filter: 10 px")
        self.slider_bpd_min = QSlider(Qt.Orientation.Horizontal)
        self.slider_bpd_min.setRange(0, 100)
        self.slider_bpd_min.setValue(10)
        self.slider_bpd_min.valueChanged.connect(lambda v: self.lbl_bpd_min.setText(self.tr("analysis_bpd_min", val=v)))
        bpd_layout.addWidget(self.lbl_bpd_min)
        bpd_layout.addWidget(self.slider_bpd_min)

        self.lbl_bpd_max = QLabel("Max Length Filter: 150 px")
        self.slider_bpd_max = QSlider(Qt.Orientation.Horizontal)
        self.slider_bpd_max.setRange(50, 400)
        self.slider_bpd_max.setValue(150)
        self.slider_bpd_max.valueChanged.connect(lambda v: self.lbl_bpd_max.setText(self.tr("analysis_bpd_max", val=v)))
        bpd_layout.addWidget(self.lbl_bpd_max)
        bpd_layout.addWidget(self.slider_bpd_max)

        self.lbl_bpd_stats = QLabel("Mean Length: N/A | Max Length: N/A | Std Dev: N/A")
        self.lbl_bpd_stats.setStyleSheet("font-weight: bold; color: #58a6ff; padding: 4px 0;")
        bpd_layout.addWidget(self.lbl_bpd_stats)

        self.btn_bpd_hist = QPushButton("📈 Generate BPD Length Histogram")
        self.btn_bpd_hist.clicked.connect(lambda: QMessageBox.information(
            self, "BPD Analysis", "BPD dislocation length profiling module initialized.\n(Feature scheduled for integration with 3D tracking)."
        ))
        bpd_layout.addWidget(self.btn_bpd_hist)

        tab_layout.addWidget(self.grp_bpd_analysis)

        # Group 2: TSD Propagation Angle & Core Profiling
        self.grp_tsd_analysis = QGroupBox("TSD Propagation Angle & Core Profiling")
        tsd_layout = QVBoxLayout(self.grp_tsd_analysis)

        self.lbl_tsd_desc = QLabel("Wafer axis propagation angle vector & core radius margin")
        self.lbl_tsd_desc.setWordWrap(True)
        tsd_layout.addWidget(self.lbl_tsd_desc)

        self.lbl_tsd_angle = QLabel("Reference Angle Axis: 45°")
        self.slider_tsd_angle = QSlider(Qt.Orientation.Horizontal)
        self.slider_tsd_angle.setRange(0, 360)
        self.slider_tsd_angle.setValue(45)
        self.slider_tsd_angle.valueChanged.connect(lambda v: self.lbl_tsd_angle.setText(self.tr("analysis_tsd_angle", val=v)))
        tsd_layout.addWidget(self.lbl_tsd_angle)
        tsd_layout.addWidget(self.slider_tsd_angle)

        self.lbl_tsd_margin = QLabel("Core Radius Margin: 12 px")
        self.slider_tsd_margin = QSlider(Qt.Orientation.Horizontal)
        self.slider_tsd_margin.setRange(2, 40)
        self.slider_tsd_margin.setValue(12)
        self.slider_tsd_margin.valueChanged.connect(lambda v: self.lbl_tsd_margin.setText(self.tr("analysis_tsd_margin", val=v)))
        tsd_layout.addWidget(self.lbl_tsd_margin)
        tsd_layout.addWidget(self.slider_tsd_margin)

        self.btn_tsd_vector = QPushButton("🧭 Generate Angle Vector Map")
        self.btn_tsd_vector.clicked.connect(lambda: QMessageBox.information(
            self, "TSD Analysis", "TSD core angular vector calculation module initialized.\n(Feature scheduled for integration with 3D tracking)."
        ))
        tsd_layout.addWidget(self.btn_tsd_vector)

        tab_layout.addWidget(self.grp_tsd_analysis)
        tab_layout.addStretch(1)

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

        # Status text update if not currently running
        if self.worker is None or not self.worker.isRunning():
            if self.last_results is not None:
                counts = self.last_results.get("counts", {})
                bpd = counts.get("BPD", 0)
                ted = counts.get("TED", 0)
                tsd = counts.get("TSD", 0)
                particle = counts.get("Particle", 0)
                crystal = bpd + ted + tsd
                total = crystal + particle
                self.lbl_status.setText(self.tr("status_complete", total=total, crystal=crystal, particle=particle))
            elif self.current_slice_2d is not None:
                self.lbl_status.setText(self.tr("status_loaded"))
            else:
                self.lbl_status.setText(self.tr("status_ready"))

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
        self.grp_bpd_analysis.setTitle(self.tr("analysis_bpd_title"))
        self.lbl_bpd_desc.setText(self.tr("analysis_bpd_desc"))
        self.lbl_bpd_min.setText(self.tr("analysis_bpd_min", val=self.slider_bpd_min.value()))
        self.lbl_bpd_max.setText(self.tr("analysis_bpd_max", val=self.slider_bpd_max.value()))
        self.lbl_bpd_stats.setText(self.tr("analysis_bpd_stats"))
        self.btn_bpd_hist.setText(self.tr("analysis_bpd_btn"))

        self.grp_tsd_analysis.setTitle(self.tr("analysis_tsd_title"))
        self.lbl_tsd_desc.setText(self.tr("analysis_tsd_desc"))
        self.lbl_tsd_angle.setText(self.tr("analysis_tsd_angle", val=self.slider_tsd_angle.value()))
        self.lbl_tsd_margin.setText(self.tr("analysis_tsd_margin", val=self.slider_tsd_margin.value()))
        self.btn_tsd_vector.setText(self.tr("analysis_tsd_btn"))

        # Tab 3: Grid
        self.chk_grid_overlay.setText(self.tr("chk_grid_overlay"))
        self.lbl_grid_intro.setText(self.tr("grid_intro"))
        self.table_grid_summary.setHorizontalHeaderLabels(self.tr("grid_headers"))
        if self.last_results is None:
            self.lbl_grid_hotspot.setText(self.tr("grid_no_data"))

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
            "TIFF Images (*.tif *.tiff *.TIF *.TIFF);;All Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*.*)",
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
            raw_img, meta = inspect_and_load_xrt(file_path, slice_idx=None)

            self.raw_data = raw_img
            self.is_3d = meta["is_3d"]
            self.num_slices = meta["num_slices"]
            self.current_slice_idx = 0
            self.current_meta = meta

            # Configure slice slider for 3D stacks
            if self.is_3d and self.num_slices > 1:
                self.slice_slider.setEnabled(True)
                self.slice_slider.setRange(0, self.num_slices - 1)
                self.slice_slider.setValue(0)
                self.slice_label.setText(self.tr("slice_label", cur=1, tot=self.num_slices))
                self.current_slice_2d = self.raw_data[0]
            else:
                self.slice_slider.setEnabled(False)
                self.slice_slider.setRange(0, 0)
                self.slice_label.setText(self.tr("slice_label", cur=1, tot=1))
                self.current_slice_2d = self.raw_data

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
            self.init_summary_table()

            # Pre-compute base normalization and proxy cache
            self.prepare_slice_cache()
            self.render_fullres_image()
            self.canvas.fit_in_view()

            # Build 3x3 grid overlay and reset grid density stats
            self.canvas.update_grid_overlay(w, h)
            self.update_grid_stats(None)

            self.lbl_status.setText(self.tr("status_loaded"))
        except Exception as e:
            QMessageBox.critical(self, self.tr("load_error_title"), self.tr("load_error_msg", e=e))

    def on_slice_changed(self, value: int):
        """Switches current slice in 3D TIFF stack."""
        if self.is_3d and self.raw_data is not None:
            self.current_slice_idx = value
            self.slice_label.setText(self.tr("slice_label", cur=value + 1, tot=self.num_slices))
            self.current_slice_2d = self.raw_data[value]
            self.prepare_slice_cache()
            self.render_fullres_image()

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
        self.lbl_status.setText(self.tr("status_scanning"))
        self.progress_bar.setValue(0)

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
        """Updates progress bar."""
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
            self.lbl_status.setText(self.tr("status_inferring", cur=current, tot=total, pct=pct))

    def on_inference_finished(self, results: Dict[str, Any]):
        """Handles completion of inference, updates overlays and summary table."""
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.last_results = results

        total = results.get("total_defects", 0)
        counts = results.get("counts", {})
        defects = results.get("defects", [])

        bpd_cnt = counts.get("BPD", 0)
        ted_cnt = counts.get("TED", 0)
        tsd_cnt = counts.get("TSD", 0)
        particle_cnt = counts.get("Particle", 0)
        crystal_subtotal = bpd_cnt + ted_cnt + tsd_cnt
        grand_total = crystal_subtotal + particle_cnt

        self.lbl_status.setText(
            self.tr("status_complete", total=grand_total, crystal=crystal_subtotal, particle=particle_cnt)
        )

        # Update 6-Row Summary Table
        self.table_summary.item(0, 1).setText(str(bpd_cnt))
        self.table_summary.item(1, 1).setText(str(ted_cnt))
        self.table_summary.item(2, 1).setText(str(tsd_cnt))
        self.table_summary.item(3, 1).setText(str(crystal_subtotal))
        self.table_summary.item(4, 1).setText(str(particle_cnt))
        self.table_summary.item(5, 1).setText(str(grand_total))

        # Draw Bounding Boxes on Canvas
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
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(self.tr("status_failed"))
        QMessageBox.critical(self, self.tr("infer_error_title"), self.tr("infer_error_msg", e=err_msg))

    # --------------------------------------------------------------------------
    # Report Exporting
    # --------------------------------------------------------------------------

    def export_report(self):
        """Exports detected defects to user-specified CSV or JSON file."""
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
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.last_results, f, indent=2)
        else:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Defect_ID", "Class_Name", "Confidence", "Center_X", "Center_Y", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2", "Width", "Height"])
                for d in defects:
                    writer.writerow([
                        d["defect_id"],
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

        QMessageBox.information(self, self.tr("export_success_title"), self.tr("export_success_msg", path=save_path))


def launch_gui(image_path: Optional[Union[str, Path]] = None):
    """Launches the PyQt6 desktop application."""
    # High DPI Scaling attributes
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern cross-platform default look

    # Set dark stylesheet
    dark_stylesheet = """
        QMainWindow, QWidget {
            background-color: #1e1e24;
            color: #dcdcdc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            border: 1px solid #3c3c46;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #58a6ff;
        }
        QPushButton {
            background-color: #2d3139;
            border: 1px solid #454c59;
            border-radius: 4px;
            padding: 5px 12px;
            color: #ffffff;
        }
        QPushButton:hover {
            background-color: #383e4a;
            border-color: #58a6ff;
        }
        QPushButton:pressed {
            background-color: #1f232b;
        }
        QTableWidget {
            background-color: #16181d;
            border: 1px solid #3c3c46;
            gridline-color: #2b2e38;
        }
        QHeaderView::section {
            background-color: #252830;
            color: #8b949e;
            padding: 4px;
            border: 1px solid #3c3c46;
        }
        QProgressBar {
            border: 1px solid #3c3c46;
            border-radius: 4px;
            text-align: center;
            background-color: #16181d;
        }
        QProgressBar::chunk {
            background-color: #007acc;
            border-radius: 3px;
        }
        QSlider::groove:horizontal {
            height: 4px;
            background: #3c3c46;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #58a6ff;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
    """
    app.setStyleSheet(dark_stylesheet)

    window = SiCDefectMainWindow(initial_image_path=image_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
