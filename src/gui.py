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
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
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
    "BPD": QColor(0, 180, 255, 230),   # Cyan-Blue
    "TED": QColor(255, 220, 0, 230),   # Bright Yellow
    "TSD": QColor(255, 40, 60, 230),    # Vibrant Red
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
        }

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

    def set_display_pixmap(self, pixmap: QPixmap):
        """Displays or updates the main wafer image on the canvas."""
        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.pixmap_item.setZValue(0)
            self.fit_in_view()
        else:
            self.pixmap_item.setPixmap(pixmap)

    def fit_in_view(self):
        """Fits image to canvas view."""
        if self.pixmap_item is not None:
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

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


# ==============================================================================
# Main Desktop Window
# ==============================================================================

class SiCDefectMainWindow(QMainWindow):
    """Main application window for SiC Wafer Defect AI."""

    def __init__(self, initial_image_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.setWindowTitle("KICET SiC Wafer Defect Detection & Classification (ONNX Runtime)")
        self.resize(1360, 860)

        # State Variables
        self.current_file_path: Optional[Path] = None
        self.raw_data: Optional[np.ndarray] = None
        self.current_slice_2d: Optional[np.ndarray] = None
        self.current_slice_idx: int = 0
        self.num_slices: int = 1
        self.is_3d: bool = False

        self.last_results: Optional[Dict[str, Any]] = None
        self.worker: Optional[InferenceWorker] = None

        # Default model search (Pure pathlib)
        self.model_path = self.locate_default_model()

        # Build UI
        self.init_ui()

        # Load initial file if supplied
        if initial_image_path is not None:
            self.load_image(Path(initial_image_path))

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

    def init_ui(self):
        """Initializes UI layout, controls, and stylesheets."""
        # Main splitter (Canvas on left, Sidebar on right)
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
        bottom_layout.addWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setFixedWidth(120)
        self.brightness_slider.valueChanged.connect(self.update_image_display)
        bottom_layout.addWidget(self.brightness_slider)

        # 3. Contrast Slider
        bottom_layout.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(20, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFixedWidth(120)
        self.contrast_slider.valueChanged.connect(self.update_image_display)
        bottom_layout.addWidget(self.contrast_slider)

        # 4. View Reset Buttons
        btn_fit = QPushButton("Fit to View")
        btn_fit.clicked.connect(self.canvas.fit_in_view)
        bottom_layout.addWidget(btn_fit)

        btn_100 = QPushButton("100% (1:1)")
        btn_100.clicked.connect(self.canvas.reset_zoom_100)
        bottom_layout.addWidget(btn_100)

        left_layout.addWidget(bottom_frame)
        splitter.addWidget(left_container)

        # ----------------------------------------------------------------------
        # RIGHT: Control Panel & Defect Statistics
        # ----------------------------------------------------------------------
        right_panel = QWidget()
        right_panel.setMinimumWidth(340)
        right_panel.setMaximumWidth(420)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Group 1: File & Model Management
        grp_file = QGroupBox("File & Model Configuration")
        grp_file_layout = QVBoxLayout(grp_file)

        btn_open = QPushButton("📂 Open XRT Image (.tif / .tiff)")
        btn_open.setFixedHeight(34)
        btn_open.clicked.connect(self.on_open_file_dialog)
        grp_file_layout.addWidget(btn_open)

        self.lbl_file_info = QLabel("No image loaded")
        self.lbl_file_info.setWordWrap(True)
        grp_file_layout.addWidget(self.lbl_file_info)

        # Model label & select
        model_hbox = QHBoxLayout()
        self.lbl_model_info = QLabel(f"Model: {self.model_path.name}")
        self.lbl_model_info.setWordWrap(True)
        btn_change_model = QPushButton("Select Weights")
        btn_change_model.clicked.connect(self.on_select_model)
        model_hbox.addWidget(self.lbl_model_info, stretch=1)
        model_hbox.addWidget(btn_change_model)
        grp_file_layout.addLayout(model_hbox)

        right_layout.addWidget(grp_file)

        # Group 2: Analysis Execution
        grp_analysis = QGroupBox("Automated Defect Analysis")
        grp_analysis_layout = QVBoxLayout(grp_analysis)

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

        right_layout.addWidget(grp_analysis)

        # Group 3: Detection Summary & Overlays
        grp_results = QGroupBox("Defect Summary & Class Overlays")
        grp_results_layout = QVBoxLayout(grp_results)

        # Summary Table
        self.table_summary = QTableWidget(4, 3)
        self.table_summary.setHorizontalHeaderLabels(["Class", "Count", "Color"])
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

        overlay_hbox.addWidget(self.chk_bpd)
        overlay_hbox.addWidget(self.chk_ted)
        overlay_hbox.addWidget(self.chk_tsd)
        grp_results_layout.addLayout(overlay_hbox)

        # Export Button
        self.btn_export = QPushButton("💾 Export Report (CSV / JSON)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_report)
        grp_results_layout.addWidget(self.btn_export)

        right_layout.addWidget(grp_results)

        right_layout.addStretch(1)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

    def create_vsep(self) -> QFrame:
        """Creates vertical separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep

    def init_summary_table(self):
        """Initializes empty defect statistics table."""
        rows = [
            ("BPD (기저면)", "0", "#00b4ff"),
            ("TED (관통칼날)", "0", "#ffdc00"),
            ("TSD (관통나선)", "0", "#ff283c"),
            ("Total Defects", "0", "#ffffff"),
        ]
        for row_idx, (name, cnt, hex_color) in enumerate(rows):
            item_name = QTableWidgetItem(name)
            item_cnt = QTableWidgetItem(cnt)
            item_cnt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_col = QTableWidgetItem("■")
            item_col.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_col.setForeground(QColor(hex_color))

            self.table_summary.setItem(row_idx, 0, item_name)
            self.table_summary.setItem(row_idx, 1, item_cnt)
            self.table_summary.setItem(row_idx, 2, item_col)

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
            "Select SiC XRT Wafer Image",
            str(start_dir),
            "TIFF Images (*.tif *.tiff *.TIF *.TIFF);;All Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*.*)",
        )
        if file_str:
            self.load_image(Path(file_str))

    def on_select_model(self):
        """Allows selecting custom ONNX or PT model weights."""
        file_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Weights",
            str(self.model_path.parent if self.model_path.exists() else Path.cwd()),
            "ONNX / YOLO Models (*.onnx *.pt);;ONNX Models (*.onnx);;PyTorch Models (*.pt);;All Files (*.*)",
        )
        if file_str:
            self.model_path = Path(file_str)
            self.lbl_model_info.setText(f"Model: {self.model_path.name}")

    def load_image(self, file_path: Path):
        """Loads a 2D or 3D TIFF image and updates canvas."""
        try:
            self.current_file_path = file_path
            raw_img, meta = inspect_and_load_xrt(file_path, slice_idx=None)

            self.raw_data = raw_img
            self.is_3d = meta["is_3d"]
            self.num_slices = meta["num_slices"]
            self.current_slice_idx = 0

            # Configure slice slider for 3D stacks
            if self.is_3d and self.num_slices > 1:
                self.slice_slider.setEnabled(True)
                self.slice_slider.setRange(0, self.num_slices - 1)
                self.slice_slider.setValue(0)
                self.slice_label.setText(f"Slice: 1/{self.num_slices}")
                self.current_slice_2d = self.raw_data[0]
            else:
                self.slice_slider.setEnabled(False)
                self.slice_slider.setRange(0, 0)
                self.slice_label.setText("Slice: 1/1")
                self.current_slice_2d = self.raw_data

            # File info label
            h, w = self.current_slice_2d.shape[:2]
            self.lbl_file_info.setText(
                f"File: {file_path.name}\n"
                f"Size: {w} x {h} px\n"
                f"Type: {meta['dtype']} (Range: [{meta['min_val']:.0f}, {meta['max_val']:.0f}])"
            )

            self.canvas.clear_overlays()
            self.btn_export.setEnabled(False)
            self.init_summary_table()
            self.update_image_display()
            self.canvas.fit_in_view()
            self.lbl_status.setText("Image loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to open image:\n{e}")

    def on_slice_changed(self, value: int):
        """Switches current slice in 3D TIFF stack."""
        if self.is_3d and self.raw_data is not None:
            self.current_slice_idx = value
            self.slice_label.setText(f"Slice: {value + 1}/{self.num_slices}")
            self.current_slice_2d = self.raw_data[value]
            self.update_image_display()

    def update_image_display(self):
        """Applies real-time brightness and contrast sliders and renders QPixmap."""
        if self.current_slice_2d is None:
            return

        # 1. Base 8-bit normalization
        norm_8bit = normalize_16bit_to_8bit(self.current_slice_2d)

        # 2. Real-time Brightness & Contrast adjustment
        # Brightness: -100 to +100 (offset)
        # Contrast: 20 to 300 (percentage scale, 100 = 1.0)
        b_offset = float(self.brightness_slider.value())
        c_scale = float(self.contrast_slider.value()) / 100.0

        adjusted = (norm_8bit.astype(np.float32) - 128.0) * c_scale + 128.0 + b_offset
        adjusted = np.clip(adjusted, 0.0, 255.0).astype(np.uint8)

        # 3. Create QPixmap
        h, w = adjusted.shape[:2]
        bytes_per_line = w
        qimg = QImage(adjusted.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        self.canvas.set_display_pixmap(pixmap)

    # --------------------------------------------------------------------------
    # Inference Pipeline
    # --------------------------------------------------------------------------

    def start_analysis(self):
        """Starts asynchronous background wafer defect inference."""
        if self.current_slice_2d is None:
            QMessageBox.warning(self, "No Image", "Please open an XRT image before running analysis.")
            return

        if not self.model_path.exists():
            QMessageBox.warning(
                self,
                "Model Missing",
                f"Model weights not found at:\n{self.model_path}\nPlease train the model or select weights.",
            )
            return

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Scanning full wafer in background...")
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
            self.lbl_status.setText(f"Inferring patches: {current} / {total} ({pct}%)")

    def on_inference_finished(self, results: Dict[str, Any]):
        """Handles completion of inference, updates overlays and summary table."""
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.last_results = results

        total = results.get("total_defects", 0)
        counts = results.get("counts", {})
        defects = results.get("defects", [])

        self.lbl_status.setText(f"Complete: {total} defects detected.")

        # Update Table
        self.table_summary.item(0, 1).setText(str(counts.get("BPD", 0)))
        self.table_summary.item(1, 1).setText(str(counts.get("TED", 0)))
        self.table_summary.item(2, 1).setText(str(counts.get("TSD", 0)))
        self.table_summary.item(3, 1).setText(str(total))

        # Draw Bounding Boxes on Canvas
        self.canvas.add_defect_boxes(defects)

        # Apply Checkbox Filter States
        self.canvas.set_overlay_visibility("BPD", self.chk_bpd.isChecked())
        self.canvas.set_overlay_visibility("TED", self.chk_ted.isChecked())
        self.canvas.set_overlay_visibility("TSD", self.chk_tsd.isChecked())

        self.btn_export.setEnabled(True)

    def on_inference_error(self, err_msg: str):
        """Handles inference failure."""
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Analysis failed.")
        QMessageBox.critical(self, "Inference Error", f"Error during defect detection:\n{err_msg}")

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
            "Save Defect Report",
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

        QMessageBox.information(self, "Export Success", f"Report saved successfully:\n{save_path}")


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
