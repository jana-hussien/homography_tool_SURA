"""
RGB <-> Thermal calibration window.

Same idea as the existing pair-calibration flow in main.py (pick 4
corresponding points on each side, compute a homography, save it), but for
finding the relationship between the single RGB camera and one chosen
thermal camera (normally the one it's physically mounted on top of, e.g.
cam5), instead of between two thermal cameras.

This is needed because RGB-derived labels (e.g. YOLO boxes) live in RGB
pixel space, and have to be projected into thermal/panorama space before
they can be assigned to a camera's quadrant. The thermal-to-thermal
homographies saved by the main tool aren't enough for that on their own.

Expected data layout for a calibration recording: the same "real capture
session" folder used by the main Stitch window (a parent folder containing
a thermal-*/ subfolder with every camera's capture JSONs, an rgb-*/
subfolder of images, and a sync.csv somewhere underneath), e.g.:

    part2/
      thermal-g.../
        data_capture_2_0_....json
        data_capture_4_0_....json
        ...
      rgb-g.../
        rgb_capture_0_0_....jpg
        ...
      sync-g.../
        sync.csv

Since a session usually has several thermal cameras, pick which one to
calibrate against RGB from the "Thermal Camera" dropdown after opening the
folder (normally the one RGB is physically mounted on top of, e.g. cam5).

Record a short hand-wave (or checkerboard) visible in both the RGB camera
and the chosen thermal camera's overlapping field of view, same as the
existing pairwise thermal calibration.
"""
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSlider, QVBoxLayout,
)

from core.homography import check_consistent_handedness, compute_homography
from core.io_utils import (
    build_camera_sequence, compute_normalization_range, discover_camera_ids,
    discover_capture_session, ensure_dir, load_sync_table, nearest_rgb_file,
    save_homography, temps_to_uint8,
)
from ui.viewer import ImageViewer

THERMAL_DISPLAY_SCALE = 12


class RGBQuadViewer(QLabel):
    """Like ui.viewer.ImageViewer's quad-click selection, but for a
    full-resolution RGB image that's scaled down to fit the window rather
    than upscaled by an integer factor. Clicks are mapped back to original
    RGB pixel coordinates by dividing out the current display scale."""

    quad_completed = pyqtSignal(list)

    def __init__(self, max_width=640, overlay_color=QColor(255, 80, 0), parent=None):
        super().__init__(parent)
        self.max_width = max_width
        self.overlay_color = overlay_color
        self.original_bgr = None
        self.display_scale = 1.0
        self.selecting = False
        self.in_progress_points = []
        self.completed_quad = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #111;")

    def set_frame_bgr(self, bgr_image):
        self.original_bgr = bgr_image
        self._refresh()

    def start_selecting(self):
        self.selecting = True
        self.in_progress_points = []
        self.completed_quad = None
        self._refresh()

    def clear_quad(self):
        self.selecting = False
        self.in_progress_points = []
        self.completed_quad = None
        self._refresh()

    def has_quad(self):
        return self.completed_quad is not None

    def _refresh(self):
        if self.original_bgr is None:
            self.setText("No RGB frame")
            self.setPixmap(QPixmap())
            return

        h, w = self.original_bgr.shape[:2]
        self.display_scale = min(1.0, self.max_width / w)
        disp_w, disp_h = int(w * self.display_scale), int(h * self.display_scale)
        resized = cv2.resize(self.original_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimg)
        pixmap = self._draw_overlay(pixmap)
        self.setFixedSize(pixmap.size())
        self.setPixmap(pixmap)

    def _draw_overlay(self, pixmap):
        if not self.in_progress_points and not self.completed_quad:
            return pixmap
        painter = QPainter(pixmap)
        painter.setPen(QPen(self.overlay_color, 2))
        painter.setFont(QFont("Arial", 10, QFont.Bold))

        points = self.completed_quad if self.completed_quad else self.in_progress_points
        scaled_points = [(x * self.display_scale, y * self.display_scale) for x, y in points]

        if self.completed_quad:
            for i in range(len(scaled_points)):
                x1, y1 = scaled_points[i]
                x2, y2 = scaled_points[(i + 1) % len(scaled_points)]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        for idx, (x, y) in enumerate(scaled_points):
            painter.setBrush(self.overlay_color)
            painter.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)
            painter.drawText(int(x) + 6, int(y) - 6, str(idx + 1))

        painter.end()
        return pixmap

    def mousePressEvent(self, event):
        if not self.selecting or self.original_bgr is None:
            super().mousePressEvent(event)
            return
        if event.button() != Qt.LeftButton:
            return

        h, w = self.original_bgr.shape[:2]
        x_orig = event.pos().x() / self.display_scale
        y_orig = event.pos().y() / self.display_scale
        x_orig = min(max(x_orig, 0), w - 1)
        y_orig = min(max(y_orig, 0), h - 1)

        self.in_progress_points.append((x_orig, y_orig))
        if len(self.in_progress_points) == 4:
            self.completed_quad = list(self.in_progress_points)
            self.in_progress_points = []
            self.selecting = False
            self.quad_completed.emit(self.completed_quad)

        self._refresh()


class RGBCalibrationWindow(QDialog):
    """Finds the homography mapping RGB pixel coordinates into one chosen
    thermal camera's pixel coordinates (cam_a="rgb", cam_b=f"cam{port}"),
    using the same compute_homography/save_homography helpers as the
    thermal-to-thermal pair calibration in main.py."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate RGB <-> Thermal")
        self.resize(1100, 700)

        self.session_folder = None
        self.capture_folder = None
        self.rgb_dir = None
        self.cam_ids = []
        self.thermal_port = None
        self.thermal_grids = []
        self.sync_rows = []
        self.frame_count = 0
        self.current_idx = 0
        self.H = None

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._advance_frame)

        layout = QVBoxLayout(self)

        source_group = QGroupBox("Source")
        open_layout = QHBoxLayout(source_group)
        self.open_button = QPushButton("Open Capture Session Folder...")
        self.open_button.clicked.connect(self.open_session_folder)
        open_layout.addWidget(self.open_button)
        self.folder_label = QLabel("Not selected")
        open_layout.addWidget(self.folder_label)
        open_layout.addWidget(QLabel("Thermal Camera:"))
        self.port_combo = QComboBox()
        self.port_combo.setEnabled(False)
        self.port_combo.currentIndexChanged.connect(self._on_port_changed)
        open_layout.addWidget(self.port_combo)
        layout.addWidget(source_group)

        images_layout = QHBoxLayout()

        thermal_box = QVBoxLayout()
        thermal_box.addWidget(QLabel("Thermal camera"))
        self.thermal_viewer = ImageViewer(scale_factor=THERMAL_DISPLAY_SCALE)
        self.thermal_viewer.quad_completed.connect(self._on_thermal_quad)
        thermal_box.addWidget(self.thermal_viewer)
        images_layout.addLayout(thermal_box)

        rgb_box = QVBoxLayout()
        rgb_box.addWidget(QLabel("RGB"))
        self.rgb_viewer = RGBQuadViewer(max_width=640)
        self.rgb_viewer.quad_completed.connect(self._on_rgb_quad)
        rgb_box.addWidget(self.rgb_viewer)
        images_layout.addLayout(rgb_box)

        layout.addLayout(images_layout)

        scrub_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)
        self.play_button.toggled.connect(self._on_play_toggled)
        scrub_layout.addWidget(self.play_button)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        self.frame_label = QLabel("Frame: 0 / 0")
        scrub_layout.addWidget(self.slider)
        scrub_layout.addWidget(self.frame_label)
        layout.addLayout(scrub_layout)

        controls_layout = QHBoxLayout()
        self.select_thermal_button = QPushButton("Select Thermal Overlap")
        self.select_thermal_button.setEnabled(False)
        self.select_thermal_button.clicked.connect(self.thermal_viewer.start_selecting)
        controls_layout.addWidget(self.select_thermal_button)

        self.select_rgb_button = QPushButton("Select RGB Overlap")
        self.select_rgb_button.setEnabled(False)
        self.select_rgb_button.clicked.connect(self.rgb_viewer.start_selecting)
        controls_layout.addWidget(self.select_rgb_button)

        self.compute_button = QPushButton("Compute Homography")
        self.compute_button.setEnabled(False)
        self.compute_button.clicked.connect(self.on_compute_clicked)
        controls_layout.addWidget(self.compute_button)

        self.save_button = QPushButton("Save Homography")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.on_save_clicked)
        controls_layout.addWidget(self.save_button)
        layout.addLayout(controls_layout)

        self.status_label = QLabel(
            "Open a capture session folder (same layout as the Stitch window's "
            "Capture Session Folder), then pick a thermal camera."
        )
        layout.addWidget(self.status_label)

    # ---- loading ----

    def open_session_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Capture Session Folder")
        if not folder:
            return
        try:
            self._load_session(Path(folder))
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _load_session(self, folder):
        capture_folder, rgb_folder, sync_csv = discover_capture_session(folder)
        if rgb_folder is None:
            raise ValueError(f"No RGB image folder found under {folder}")
        if sync_csv is None:
            raise ValueError(f"No sync.csv found under {folder}")

        cam_ids = discover_camera_ids(capture_folder)
        if not cam_ids:
            raise ValueError(f"No thermal capture JSONs found under {capture_folder}")

        self.session_folder = folder
        self.capture_folder = capture_folder
        self.rgb_dir = rgb_folder
        self.sync_rows = load_sync_table(sync_csv)
        self.cam_ids = cam_ids

        self.folder_label.setText(str(folder))

        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for cam_id in cam_ids:
            self.port_combo.addItem(f"cam{cam_id}", cam_id)
        # cam5 is normally the one RGB is physically mounted on top of.
        default_idx = cam_ids.index(5) if 5 in cam_ids else 0
        self.port_combo.setCurrentIndex(default_idx)
        self.port_combo.blockSignals(False)
        self.port_combo.setEnabled(True)

        self._load_port(cam_ids[default_idx])

    def _on_port_changed(self, _index):
        port = self.port_combo.currentData()
        if port is not None:
            self._load_port(port)

    def _load_port(self, port):
        self.thermal_port = port
        raw_seq = build_camera_sequence(self.capture_folder, port)
        self.thermal_grids = [grid for _idx, _ts, grid in raw_seq]
        self.frame_count = len(self.thermal_grids)
        self.current_idx = 0
        self.H = None

        self.thermal_viewer.clear_quad()
        self.rgb_viewer.clear_quad()
        self.select_thermal_button.setEnabled(True)
        self.select_rgb_button.setEnabled(True)
        self.compute_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.play_button.setEnabled(True)

        self.status_label.setText(f"Loaded cam{self.thermal_port} ({self.frame_count} frames).")

        self.slider.setMinimum(0)
        self.slider.setMaximum(max(self.frame_count - 1, 0))
        self.slider.setValue(0)
        self.show_frame(0)

    # ---- frame navigation ----

    def show_frame(self, idx):
        if self.frame_count == 0:
            return
        idx = max(0, min(idx, self.frame_count - 1))
        self.current_idx = idx

        # Per-frame normalization (rather than over the whole recording),
        # matching the rest of the app's default: contrast relative to
        # what's currently in view instead of washing out over a long clip.
        vmin, vmax = compute_normalization_range([self.thermal_grids[idx]])
        thermal_uint8 = temps_to_uint8(self.thermal_grids[idx], vmin, vmax)
        self.thermal_viewer.set_frame(thermal_uint8)

        rgb_name = nearest_rgb_file(self.sync_rows, idx)
        rgb_bgr = None
        if rgb_name:
            rgb_path = self.rgb_dir / rgb_name
            rgb_bgr = cv2.imread(str(rgb_path))
        self.rgb_viewer.set_frame_bgr(rgb_bgr)

        self.frame_label.setText(f"Frame: {idx + 1} / {self.frame_count}")
        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

    def _advance_frame(self):
        next_idx = self.current_idx + 1
        if next_idx >= self.frame_count:
            next_idx = 0
        self.show_frame(next_idx)

    def _on_play_toggled(self, playing):
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self.timer.start()
        else:
            self.timer.stop()

    # ---- quad selection / homography ----

    def _on_thermal_quad(self, _quad):
        self._update_compute_enabled()

    def _on_rgb_quad(self, _quad):
        self._update_compute_enabled()

    def _update_compute_enabled(self):
        ready = self.thermal_viewer.has_quad() and self.rgb_viewer.has_quad()
        self.compute_button.setEnabled(ready)

    def on_compute_clicked(self):
        # cam_a = rgb, cam_b = thermal -> H maps RGB pixel coords into this
        # thermal camera's pixel coords, same convention as the
        # thermal-to-thermal pairs (compute_homography(quad_a, quad_b)
        # solves for H with p_b = H @ p_a).
        try:
            self.H = compute_homography(self.rgb_viewer.completed_quad, self.thermal_viewer.completed_quad)
        except Exception as exc:
            QMessageBox.critical(self, "Homography Error", str(exc))
            return
        self.save_button.setEnabled(True)
        self.status_label.setText(f"Homography computed (RGB -> cam{self.thermal_port}). Ready to save.")

    def on_save_clicked(self):
        quad_rgb = self.rgb_viewer.completed_quad or []
        quad_thermal = self.thermal_viewer.completed_quad or []
        if quad_rgb and quad_thermal and not check_consistent_handedness(quad_rgb, quad_thermal):
            proceed = QMessageBox.warning(
                self, "Possible Mirrored Homography",
                "The RGB and thermal quads have opposite rotational handedness "
                "(clockwise on one image, counter-clockwise on the other). This "
                "strongly suggests the resulting homography is a mirror of reality "
                "rather than a match to it, even though each quad individually looks "
                "like a clean rectangle. Double-check both frames' orientation and "
                "that points were clicked in the same rotational direction on both.\n\n"
                "Save anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if proceed != QMessageBox.Yes:
                return
        out_dir = ensure_dir(self.session_folder)
        pair_name = f"rgb{self.thermal_port}"
        try:
            json_path, npy_path = save_homography(
                out_dir, pair_name, "rgb", f"cam{self.thermal_port}",
                quad_rgb, quad_thermal, self.H,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Saved homography to:\n{json_path}\n{npy_path}")
