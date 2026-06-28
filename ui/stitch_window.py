from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)

from core.io_utils import (
    align_camera_sequences, build_camera_sequence, compute_normalization_range,
    discover_camera_ids, discover_capture_session, discover_homography_files, ensure_dir,
    heatmap_from_uint8, load_sync_table, nearest_rgb_file, temps_to_uint8,
)
from core.stitcher import build_clusters, stitch_frames
from ui.viewer import ImageViewer, RGBViewer


class OrderCamerasDialog(QDialog):
    """Lets the user pick which camera comes first within each calibrated
    pair/cluster (the first camera to appear in this order becomes that
    cluster's reference frame; the other camera(s) get warped onto it)."""

    def __init__(self, cam_ids, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Order Cameras")
        self.resize(300, 300)
        self.result = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Arrange cameras in order (top to bottom):\n"
            "For each calibrated pair, whichever camera appears first becomes "
            "the reference frame the other is warped into."
        ))

        self.list_widget = QListWidget()
        for cam_id in cam_ids:
            item = QListWidgetItem(f"cam{cam_id}")
            item.setData(Qt.UserRole, cam_id)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        up_btn = QPushButton("↑ Move Up")
        up_btn.clicked.connect(self.move_up)
        button_layout.addWidget(up_btn)
        down_btn = QPushButton("↓ Move Down")
        down_btn.clicked.connect(self.move_down)
        button_layout.addWidget(down_btn)
        layout.addLayout(button_layout)

        confirm_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_layout.addWidget(ok_btn)
        confirm_layout.addWidget(cancel_btn)
        layout.addLayout(confirm_layout)

    def move_up(self):
        row = self.list_widget.currentRow()
        if row > 0:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row - 1, item)
            self.list_widget.setCurrentRow(row - 1)

    def move_down(self):
        row = self.list_widget.currentRow()
        if row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row + 1, item)
            self.list_widget.setCurrentRow(row + 1)

    def accept(self):
        self.result = [
            self.list_widget.item(i).data(Qt.UserRole)
            for i in range(self.list_widget.count())
        ]
        super().accept()


class CaptureStitchWindow(QDialog):
    """Applies previously-saved pairwise homographies (computed from their
    own short calibration recordings) to a real, synchronized capture
    session, instead of stitching together the mismatched calibration
    footage itself. Cameras with no calibrated neighbor are shown
    unstitched. The RGB frame is shown alongside, unmodified."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stitch Real Capture + RGB")
        self.resize(1500, 800)

        self.capture_folder = None
        self.homography_folder = None
        self.rgb_folder = None
        self.sync_rows = []

        self.cam_ids = []
        # user-controlled order; first cam per cluster becomes the reference.
        # Default to the rig's usual chaining order; falls back to ascending
        # cam_ids in _build() if the loaded session's cams don't match this set.
        self.cam_order = [6, 4, 5, 2, 3]
        self.cam_grid_sequences = {}  # cam_id -> list of grids
        self.homography_entries = []
        self.clusters = []
        self.frame_count = 0
        self._last_composites = {}  # cluster index -> full-res uint8 array, for saving

        self.cluster_viewers = []
        self.cluster_labels = []

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._advance_frame)

        layout = QVBoxLayout(self)

        select_group = QGroupBox("Sources")
        select_layout = QVBoxLayout(select_group)
        self.session_label = self._add_select_row(
            select_layout, "Select Capture Session Folder...", self.select_session_folder,
        )
        self.homography_label = self._add_select_row(
            select_layout, "Select Homography Folder...", self.select_homography_folder,
        )
        self.capture_label = self._add_display_row(select_layout, "Thermal:")
        self.rgb_label = self._add_display_row(select_layout, "RGB:")
        self.sync_label = self._add_display_row(select_layout, "Sync:")
        layout.addWidget(select_group)

        self.reorder_button = QPushButton("Reorder Cameras...")
        self.reorder_button.setEnabled(False)
        self.reorder_button.clicked.connect(self.reorder_cameras)
        layout.addWidget(self.reorder_button)

        self.status_label = QLabel("Select a thermal capture folder and a homography folder to begin.")
        layout.addWidget(self.status_label)

        # The number of cluster viewers (and their size) depends on how many
        # cameras/pairs are loaded, so their combined width can exceed the
        # dialog/screen. A scroll area lets the dialog stay resizable to fit
        # the screen instead of forcing the window past it or clipping content.
        viewers_container = QWidget()
        self.viewers_layout = QHBoxLayout(viewers_container)
        viewers_scroll = QScrollArea()
        viewers_scroll.setWidgetResizable(True)
        viewers_scroll.setWidget(viewers_container)
        layout.addWidget(viewers_scroll)

        self.rgb_viewer = RGBViewer()
        rgb_box = QVBoxLayout()
        rgb_box.addWidget(QLabel("RGB (unprocessed)"))
        rgb_box.addWidget(self.rgb_viewer)
        self.viewers_layout.addLayout(rgb_box)

        slider_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)
        self.play_button.toggled.connect(self._on_play_toggled)
        slider_layout.addWidget(self.play_button)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.update_preview)
        self.frame_label = QLabel("Frame: 0 / 0")
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.frame_label)
        layout.addLayout(slider_layout)

        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save cluster:"))
        self.save_cluster_combo = QComboBox()
        self.save_cluster_combo.setEnabled(False)
        save_layout.addWidget(self.save_cluster_combo)
        self.save_button = QPushButton("Save Current Merged Frame")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_current_merged)
        save_layout.addWidget(self.save_button)
        self.save_all_button = QPushButton("Save All Merged Frames")
        self.save_all_button.setEnabled(False)
        self.save_all_button.clicked.connect(self.save_all_merged)
        save_layout.addWidget(self.save_all_button)
        layout.addLayout(save_layout)

    def _add_select_row(self, parent_layout, button_text, slot):
        row = QHBoxLayout()
        button = QPushButton(button_text)
        button.clicked.connect(slot)
        label = QLabel("Not selected")
        row.addWidget(button)
        row.addWidget(label)
        parent_layout.addLayout(row)
        return label

    def _add_display_row(self, parent_layout, prefix_text):
        row = QHBoxLayout()
        row.addWidget(QLabel(prefix_text))
        label = QLabel("Not selected")
        row.addWidget(label)
        parent_layout.addLayout(row)
        return label

    # ---- source selection ----

    def select_session_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Capture Session Folder")
        if not folder:
            return
        self.session_label.setText(folder)

        try:
            capture_folder, rgb_folder, sync_csv = discover_capture_session(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.capture_folder = capture_folder
        self.capture_label.setText(str(capture_folder))

        self.rgb_folder = rgb_folder
        self.rgb_label.setText(str(rgb_folder) if rgb_folder else "Not found")

        if sync_csv:
            try:
                self.sync_rows = load_sync_table(sync_csv)
                self.sync_label.setText(str(sync_csv))
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))
                self.sync_rows = []
                self.sync_label.setText("Not found")
        else:
            self.sync_rows = []
            self.sync_label.setText("Not found")

        self._try_build()
        self._refresh_rgb(self.slider.value())

    def select_homography_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Homography Folder")
        if not folder:
            return
        self.homography_folder = Path(folder)
        self.homography_label.setText(str(self.homography_folder))
        self._try_build()

    def reorder_cameras(self):
        dialog = OrderCamerasDialog(self.cam_order or self.cam_ids, self)
        if dialog.exec_() != QDialog.Accepted or not dialog.result:
            return
        self.cam_order = dialog.result
        if not self.homography_entries:
            return
        self.clusters = build_clusters(self.homography_entries, self.cam_order)
        self._rebuild_cluster_viewers()
        self._rebuild_save_cluster_combo()
        self.update_preview(self.slider.value())

    # ---- building clusters from capture + homography ----

    def _try_build(self):
        if self.capture_folder is None or self.homography_folder is None:
            return
        try:
            self._build()
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _build(self):
        cam_ids = discover_camera_ids(self.capture_folder)
        if not cam_ids:
            raise ValueError(f"No capture JSONs found in {self.capture_folder}")

        homography_entries = discover_homography_files(self.homography_folder)
        if not homography_entries:
            raise ValueError(f"No *_homography.json files found under {self.homography_folder}")

        raw_sequences = {cam_id: build_camera_sequence(self.capture_folder, cam_id) for cam_id in cam_ids}
        sequences = align_camera_sequences(raw_sequences)

        self.cam_ids = cam_ids
        if set(self.cam_order) != set(cam_ids):
            self.cam_order = list(cam_ids)
        self.cam_grid_sequences = sequences
        self.homography_entries = homography_entries
        self.clusters = build_clusters(homography_entries, self.cam_order)

        self.frame_count = min(len(seq) for seq in sequences.values())

        self.reorder_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.save_all_button.setEnabled(True)
        self.save_cluster_combo.setEnabled(True)
        self._rebuild_cluster_viewers()
        self._rebuild_save_cluster_combo()

        unmatched = [c["cams"][0] for c in self.clusters if len(c["cams"]) == 1]
        status = f"{len(cam_ids)} cameras, {len(self.clusters)} cluster(s)."
        if unmatched:
            status += f" No calibration for cam(s): {unmatched} (shown unstitched)."
        self.status_label.setText(status)

        self.slider.setMinimum(0)
        self.slider.setMaximum(self.frame_count - 1)
        self.slider.setValue(0)
        self.update_preview(0)

    def _rebuild_cluster_viewers(self):
        for viewer in self.cluster_viewers:
            self.viewers_layout.removeWidget(viewer)
            viewer.deleteLater()
        for label in self.cluster_labels:
            self.viewers_layout.removeWidget(label)
            label.deleteLater()
        self.cluster_viewers = []
        self.cluster_labels = []

        for i, cluster in enumerate(self.clusters):
            box = QVBoxLayout()
            cams_text = "+".join(f"cam{c}" for c in cluster["cams"])
            label = QLabel(cams_text)
            viewer = ImageViewer(scale_factor=8, max_width=900, max_height=600)
            box.addWidget(label)
            box.addWidget(viewer)
            self.viewers_layout.insertLayout(i, box)
            self.cluster_labels.append(label)
            self.cluster_viewers.append(viewer)

    def _rebuild_save_cluster_combo(self):
        previous_tag = self.save_cluster_combo.currentData()
        self.save_cluster_combo.clear()
        for cluster in self.clusters:
            tag = self._cluster_filename_tag(cluster)
            self.save_cluster_combo.addItem(tag, tag)
        if previous_tag is not None:
            idx = self.save_cluster_combo.findData(previous_tag)
            if idx >= 0:
                self.save_cluster_combo.setCurrentIndex(idx)

    # ---- preview ----

    def _compute_cluster_composites(self, idx):
        composites = []
        for cluster in self.clusters:
            # Normalize every camera in the cluster together (one shared
            # vmin/vmax), but recompute it per frame instead of over the
            # whole session, so the heatmap always shows contrast relative
            # to what's currently in view rather than washing out to one
            # end of the scale when the ambient temperature shifts (e.g.
            # moving from indoors to outdoors).
            grids = [self.cam_grid_sequences[cam_id][idx] for cam_id in cluster["cams"]]
            vmin, vmax = compute_normalization_range(grids)
            frames = [temps_to_uint8(grid, vmin, vmax) for grid in grids]
            if len(frames) > 1:
                composite = stitch_frames(frames, cluster["transforms"])
            else:
                composite = frames[0]
            composites.append(heatmap_from_uint8(composite))
        return composites

    def _compute_cluster_float_composites(self, idx):
        """Same stitching as _compute_cluster_composites, but on the raw
        float temperature grids (no uint8 normalization, no heatmap) — for
        saving actual temperature data rather than a display image."""
        composites = []
        for cluster in self.clusters:
            frames = [
                self.cam_grid_sequences[cam_id][idx].astype(np.float32)
                for cam_id in cluster["cams"]
            ]
            if len(frames) > 1:
                composite = stitch_frames(frames, cluster["transforms"])
            else:
                composite = frames[0]
            composites.append(composite)
        return composites

    def update_preview(self, idx):
        self.frame_label.setText(f"Frame: {idx + 1} / {max(self.frame_count, 1)}")
        if not self.clusters:
            return
        composites = self._compute_cluster_composites(idx)
        for i, (composite, viewer) in enumerate(zip(composites, self.cluster_viewers)):
            self._last_composites[i] = composite
            viewer.set_frame(composite)
        self._refresh_rgb(idx)

    def _on_play_toggled(self, playing):
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _advance_frame(self):
        next_idx = self.slider.value() + 1
        if next_idx > self.slider.maximum():
            next_idx = 0
        self.slider.setValue(next_idx)

    def _refresh_rgb(self, idx):
        if not self.rgb_folder or not self.sync_rows:
            self.rgb_viewer.set_frame_bgr(None)
            return
        filename = nearest_rgb_file(self.sync_rows, idx)
        if not filename:
            self.rgb_viewer.set_frame_bgr(None)
            return
        rgb_path = self.rgb_folder / filename
        image = cv2.imread(str(rgb_path))
        self.rgb_viewer.set_frame_bgr(image)

    # ---- save ----

    def _cluster_filename_tag(self, cluster):
        return "+".join(f"cam{c}" for c in cluster["cams"])

    def _selected_cluster(self):
        tag = self.save_cluster_combo.currentData()
        if tag is None:
            return None
        for cluster in self.clusters:
            if self._cluster_filename_tag(cluster) == tag:
                return cluster
        return None

    def save_current_merged(self):
        cluster = self._selected_cluster()
        if cluster is None:
            QMessageBox.warning(self, "No Cluster Selected", "Choose a cluster to save first.")
            return
        cluster_idx = self.clusters.index(cluster)
        idx = self.slider.value()
        out_dir = ensure_dir(self.capture_folder / "stitched_output")
        float_composite = self._compute_cluster_float_composites(idx)[cluster_idx]
        tag = self._cluster_filename_tag(cluster)
        out_path = out_dir / f"{tag}_frame_{idx + 1:04d}.npy"
        np.save(out_path, float_composite)
        QMessageBox.information(self, "Saved", f"Saved frame {idx + 1} (float temperatures) to {out_path}")

    def save_all_merged(self):
        cluster = self._selected_cluster()
        if cluster is None:
            QMessageBox.warning(self, "No Cluster Selected", "Choose a cluster to save first.")
            return
        cluster_idx = self.clusters.index(cluster)
        out_dir = ensure_dir(self.capture_folder / "stitched_output")
        tag = self._cluster_filename_tag(cluster)
        for idx in range(self.frame_count):
            float_composite = self._compute_cluster_float_composites(idx)[cluster_idx]
            out_path = out_dir / f"{tag}_frame_{idx + 1:04d}.npy"
            np.save(out_path, float_composite)
        QMessageBox.information(
            self, "Saved",
            f"Saved {self.frame_count} frame(s) for {tag} (float temperatures) to {out_dir}",
        )
