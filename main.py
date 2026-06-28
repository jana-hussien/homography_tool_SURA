import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from core.homography import (
    blend_images, check_consistent_handedness, compute_homography, warp_image,
)
from core.io_utils import (
    align_camera_sequences, build_camera_sequence, compute_normalization_range,
    detect_pair_cameras, ensure_dir, save_homography, temps_to_uint8,
)
from ui.controls import ControlPanel
from ui.stitch_window import CaptureStitchWindow
from ui.rgb_calibration_window import RGBCalibrationWindow
from ui.viewer import ImageViewer

DISPLAY_SCALE = 12


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pair_folder = None
        self.pair_name = None
        self.cam_a_name = None
        self.cam_b_name = None
        self.cam_a_grids = []
        self.cam_b_grids = []
        self.frame_count = 0
        self.current_idx = 0
        self.H = None
        self.preview_enabled = False

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.advance_frame)

        self._build_ui()
        self.setFocusPolicy(Qt.StrongFocus)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)

        open_layout = QHBoxLayout()
        self.open_button = QPushButton("Open Pair Folder")
        self.open_button.clicked.connect(self.open_pair_folder)
        open_layout.addWidget(self.open_button)
        open_layout.addStretch()
        outer_layout.addLayout(open_layout)

        body_layout = QHBoxLayout()

        viewers_layout = QVBoxLayout()
        images_layout = QHBoxLayout()
        self.viewer_a = ImageViewer(scale_factor=DISPLAY_SCALE)
        self.viewer_b = ImageViewer(scale_factor=DISPLAY_SCALE)
        self.viewer_a.quad_completed.connect(self.on_quad_a_completed)
        self.viewer_b.quad_completed.connect(self.on_quad_b_completed)
        images_layout.addWidget(self.viewer_a)
        images_layout.addWidget(self.viewer_b)
        viewers_layout.addLayout(images_layout)

        self.preview_viewer = ImageViewer(scale_factor=DISPLAY_SCALE, max_width=800, max_height=400)
        self.preview_viewer.setVisible(False)
        viewers_layout.addWidget(self.preview_viewer)

        scrub_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.frame_label = QLabel("Frame: 0 / 0")
        scrub_layout.addWidget(self.slider)
        scrub_layout.addWidget(self.frame_label)
        viewers_layout.addLayout(scrub_layout)

        body_layout.addLayout(viewers_layout)

        self.control_panel = ControlPanel()
        self.control_panel.select_a_clicked.connect(self.start_select_a)
        self.control_panel.select_b_clicked.connect(self.start_select_b)
        self.control_panel.clear_a_clicked.connect(self.clear_quad_a)
        self.control_panel.clear_b_clicked.connect(self.clear_quad_b)
        self.control_panel.swap_cameras_clicked.connect(self.on_swap_cameras_clicked)
        self.control_panel.compute_clicked.connect(self.on_compute_clicked)
        self.control_panel.apply_matrix_clicked.connect(self.on_apply_matrix_clicked)
        self.control_panel.save_clicked.connect(self.on_save_clicked)
        self.control_panel.generate_examples_clicked.connect(self.on_generate_examples_clicked)
        self.control_panel.capture_stitch_mode_clicked.connect(self.open_capture_stitch_window)
        self.control_panel.rgb_calibration_clicked.connect(self.open_rgb_calibration_window)
        self.control_panel.play_toggled.connect(self.on_play_toggled)
        self.control_panel.preview_toggled.connect(self.on_preview_toggled)
        body_layout.addWidget(self.control_panel)

        outer_layout.addLayout(body_layout)

    # ---- pair loading ----

    def open_pair_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Pair Folder")
        if not folder:
            return
        try:
            self._load_pair(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _load_pair(self, folder):
        pair_folder = Path(folder)
        cam_a_id, cam_b_id = detect_pair_cameras(pair_folder)

        seq_a = build_camera_sequence(pair_folder, cam_a_id)
        seq_b = build_camera_sequence(pair_folder, cam_b_id)

        # Align by each frame's recorded frame_index rather than zipping by
        # list position, so a frame missing from only one camera (e.g. a
        # corrupt/empty capture that got removed) doesn't silently shift
        # every later frame out of sync between cam A and cam B.
        aligned = align_camera_sequences({cam_a_id: seq_a, cam_b_id: seq_b})
        self.cam_a_grids = aligned[cam_a_id]
        self.cam_b_grids = aligned[cam_b_id]

        self.pair_folder = pair_folder
        self.pair_name = f"{cam_a_id}{cam_b_id}"
        self.cam_a_name = f"cam{cam_a_id}"
        self.cam_b_name = f"cam{cam_b_id}"
        self.frame_count = min(len(self.cam_a_grids), len(self.cam_b_grids))
        self.current_idx = 0
        self.H = None

        self.viewer_a.clear_quad()
        self.viewer_b.clear_quad()
        self.preview_viewer.setVisible(False)
        self.control_panel.preview_button.setChecked(False)
        self.control_panel.set_status_a(False)
        self.control_panel.set_status_b(False)
        self.control_panel.set_compute_enabled(False)
        self.control_panel.set_post_compute_enabled(False)
        self.control_panel.set_matrix_text(None)

        self.slider.setMinimum(0)
        self.slider.setMaximum(self.frame_count - 1)
        self.slider.setValue(0)

        self.show_frame(0)

    def _update_title(self):
        self.setWindowTitle(
            f"Thermal Calibration - {self.pair_folder.name} ({self.pair_name}) - "
            f"Frame {self.current_idx + 1}/{self.frame_count}"
        )

    # ---- frame navigation ----

    def _frame_norm(self, idx):
        # Shared between cam A and cam B (rather than each auto-contrasting
        # to its own range) and recomputed per frame (rather than over the
        # whole pair sequence), matching the stitched capture preview: keeps
        # the two cameras' brightness comparable while showing contrast
        # relative to what's currently in view instead of washing out when
        # the ambient temperature drifts over a long recording.
        return compute_normalization_range([self.cam_a_grids[idx], self.cam_b_grids[idx]])

    def get_frame_a(self, idx):
        return temps_to_uint8(self.cam_a_grids[idx], *self._frame_norm(idx))

    def get_frame_b(self, idx):
        return temps_to_uint8(self.cam_b_grids[idx], *self._frame_norm(idx))

    def show_frame(self, idx):
        if self.frame_count == 0:
            return
        idx = max(0, min(idx, self.frame_count - 1))
        self.current_idx = idx
        frame_a = self.get_frame_a(idx)
        frame_b = self.get_frame_b(idx)
        self.viewer_a.set_frame(frame_a)
        self.viewer_b.set_frame(frame_b)
        self.frame_label.setText(f"Frame: {idx + 1} / {self.frame_count}")
        self._update_title()

        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

        if self.preview_enabled and self.H is not None:
            self._refresh_preview(frame_a, frame_b)

    def on_slider_changed(self, value):
        self.show_frame(value)

    def advance_frame(self):
        next_idx = self.current_idx + 1
        if next_idx >= self.frame_count:
            next_idx = 0
        self.show_frame(next_idx)

    def on_play_toggled(self, playing):
        if playing:
            self.timer.start()
        else:
            self.timer.stop()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.show_frame(self.current_idx - 1)
        elif event.key() == Qt.Key_Right:
            self.show_frame(self.current_idx + 1)
        else:
            super().keyPressEvent(event)

    # ---- camera ordering ----

    def on_swap_cameras_clicked(self):
        if self.frame_count == 0:
            return
        self.cam_a_name, self.cam_b_name = self.cam_b_name, self.cam_a_name
        self.cam_a_grids, self.cam_b_grids = self.cam_b_grids, self.cam_a_grids

        cam_a_id = int(re.search(r"\d+", self.cam_a_name).group())
        cam_b_id = int(re.search(r"\d+", self.cam_b_name).group())
        self.pair_name = f"{cam_a_id}{cam_b_id}"

        # The quads and homography were defined for the previous A/B
        # orientation and no longer apply.
        self.H = None
        self.viewer_a.clear_quad()
        self.viewer_b.clear_quad()
        self.preview_viewer.setVisible(False)
        self.control_panel.preview_button.setChecked(False)
        self.control_panel.set_status_a(False)
        self.control_panel.set_status_b(False)
        self.control_panel.set_compute_enabled(False)
        self.control_panel.set_post_compute_enabled(False)
        self.control_panel.set_matrix_text(None)

        self.show_frame(self.current_idx)

    # ---- quad selection ----

    def start_select_a(self):
        self.viewer_a.start_selecting()
        self.control_panel.set_status_a(False)
        self._update_compute_enabled()

    def start_select_b(self):
        self.viewer_b.start_selecting()
        self.control_panel.set_status_b(False)
        self._update_compute_enabled()

    def on_quad_a_completed(self, _quad):
        self.control_panel.set_status_a(True)
        self._update_compute_enabled()

    def on_quad_b_completed(self, _quad):
        self.control_panel.set_status_b(True)
        self._update_compute_enabled()

    def clear_quad_a(self):
        self.viewer_a.clear_quad()
        self.control_panel.set_status_a(False)
        self._update_compute_enabled()

    def clear_quad_b(self):
        self.viewer_b.clear_quad()
        self.control_panel.set_status_b(False)
        self._update_compute_enabled()

    def _update_compute_enabled(self):
        ready = self.viewer_a.has_quad() and self.viewer_b.has_quad()
        self.control_panel.set_compute_enabled(ready)

    # ---- homography ----

    def on_compute_clicked(self):
        try:
            self.H = compute_homography(self.viewer_a.completed_quad, self.viewer_b.completed_quad)
        except Exception as exc:
            QMessageBox.critical(self, "Homography Error", str(exc))
            return
        self.control_panel.set_matrix_text(self.H)
        self.control_panel.set_post_compute_enabled(True)
        if self.preview_enabled:
            frame_a = self.get_frame_a(self.current_idx)
            frame_b = self.get_frame_b(self.current_idx)
            self._refresh_preview(frame_a, frame_b)

    def on_apply_matrix_clicked(self):
        text = self.control_panel.matrix_display.toPlainText()
        try:
            rows = [line.split() for line in text.strip().splitlines() if line.strip()]
            H = np.array(rows, dtype=np.float64)
            if H.shape != (3, 3):
                raise ValueError(f"Expected a 3x3 matrix, got shape {H.shape}")
        except Exception as exc:
            QMessageBox.critical(self, "Matrix Error", f"Could not parse matrix: {exc}")
            return

        self.H = H
        self.control_panel.set_matrix_text(self.H)
        self.control_panel.set_post_compute_enabled(True)
        if self.preview_enabled:
            frame_a = self.get_frame_a(self.current_idx)
            frame_b = self.get_frame_b(self.current_idx)
            self._refresh_preview(frame_a, frame_b)

    def on_preview_toggled(self, enabled):
        self.preview_enabled = enabled
        self.preview_viewer.setVisible(enabled)
        if enabled and self.H is not None:
            frame_a = self.get_frame_a(self.current_idx)
            frame_b = self.get_frame_b(self.current_idx)
            self._refresh_preview(frame_a, frame_b)

    def _refresh_preview(self, frame_a, frame_b):
        panorama = self._build_panorama_frame(frame_a, frame_b)
        self.preview_viewer.set_frame(panorama)

    # ---- save / export ----

    def on_save_clicked(self):
        quad_a = self.viewer_a.completed_quad or []
        quad_b = self.viewer_b.completed_quad or []
        if quad_a and quad_b and not check_consistent_handedness(quad_a, quad_b):
            proceed = QMessageBox.warning(
                self, "Possible Mirrored Homography",
                "The two quads have opposite rotational handedness (clockwise on one "
                "image, counter-clockwise on the other). This strongly suggests the "
                "resulting homography is a mirror of reality rather than a match to "
                "it, even though each quad individually looks like a clean rectangle. "
                "Double-check both frames' orientation and that points were clicked "
                "in the same rotational direction on both.\n\nSave anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if proceed != QMessageBox.Yes:
                return
        try:
            json_path, npy_path = save_homography(
                self.pair_folder, self.pair_name, self.cam_a_name, self.cam_b_name,
                quad_a, quad_b, self.H,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Saved homography to:\n{json_path}\n{npy_path}")

    def on_generate_examples_clicked(self):
        try:
            paths = self._generate_example_images()
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            return
        QMessageBox.information(self, "Examples Generated", "\n".join(str(p) for p in paths))

    def _generate_example_images(self):
        examples_dir = ensure_dir(self.pair_folder / "examples")
        n_examples = min(5, self.frame_count)
        indices = np.linspace(0, self.frame_count - 1, n_examples, dtype=int)
        saved_paths = []

        for idx in indices:
            frame_a = self.get_frame_a(idx)
            frame_b = self.get_frame_b(idx)
            h, w = frame_b.shape[:2]
            warped_a = warp_image(frame_a, self.H, (w, h))
            blended = blend_images(warped_a, frame_b, alpha=0.5)

            side_by_side = self._hstack_padded(frame_a, frame_b)
            side_path = examples_dir / f"frame_{idx + 1:04d}_original_side_by_side.png"
            cv2.imwrite(str(side_path), side_by_side)
            saved_paths.append(side_path)

            blend_path = examples_dir / f"frame_{idx + 1:04d}_warped_blended.png"
            cv2.imwrite(str(blend_path), blended)
            saved_paths.append(blend_path)

            panorama = self._build_panorama_frame(frame_a, frame_b)
            pano_path = examples_dir / f"frame_{idx + 1:04d}_stitched_panorama.png"
            cv2.imwrite(str(pano_path), panorama)
            saved_paths.append(pano_path)

        return saved_paths

    @staticmethod
    def _hstack_padded(img_a, img_b):
        h = max(img_a.shape[0], img_b.shape[0])
        w = img_a.shape[1] + img_b.shape[1]
        canvas = np.zeros((h, w), dtype=np.uint8)
        canvas[: img_a.shape[0], : img_a.shape[1]] = img_a
        canvas[: img_b.shape[0], img_a.shape[1]:] = img_b
        return canvas

    def _build_panorama_frame(self, frame_a, frame_b):
        h_b, w_b = frame_b.shape[:2]
        h_a, w_a = frame_a.shape[:2]

        corners_a = np.array(
            [[0, 0], [w_a - 1, 0], [w_a - 1, h_a - 1], [0, h_a - 1]], dtype=np.float64
        )
        corners_h = np.hstack([corners_a, np.ones((4, 1))])
        transformed = (self.H @ corners_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]

        min_x = min(0, transformed[:, 0].min())
        min_y = min(0, transformed[:, 1].min())
        max_x = max(w_b, transformed[:, 0].max())
        max_y = max(h_b, transformed[:, 1].max())

        offset_x = -min_x
        offset_y = -min_y
        canvas_w = int(np.ceil(max_x - min_x))
        canvas_h = int(np.ceil(max_y - min_y))

        translation = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1],
        ], dtype=np.float64)

        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        warped_full_a = warp_image(frame_a, translation @ self.H, (canvas_w, canvas_h))
        mask_a = warped_full_a > 0
        canvas[mask_a] = warped_full_a[mask_a]

        warped_b = warp_image(frame_b, translation, (canvas_w, canvas_h))
        mask_b = warped_b > 0
        canvas[mask_b] = warped_b[mask_b]

        return canvas

    # ---- panorama stitching mode ----

    def open_capture_stitch_window(self):
        dialog = CaptureStitchWindow(self)
        dialog.exec_()

    def open_rgb_calibration_window(self):
        dialog = RGBCalibrationWindow(self)
        dialog.exec_()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1100, 700)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
