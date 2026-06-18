from pathlib import Path

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSlider, QVBoxLayout,
)

from core.io_utils import (
    build_camera_sequence, compute_normalization_range, detect_pair_cameras, ensure_dir,
    list_pair_folders, temps_to_uint8,
)
from core.stitcher import build_reference_transforms, load_chain_homographies, stitch_frames
from ui.viewer import ImageViewer


class OrderPairsDialog(QDialog):
    """Dialog to let the user arrange pair folders in stitch order."""
    
    def __init__(self, pair_folders, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Order Pair Folders")
        self.resize(400, 300)
        self.pair_folders = pair_folders
        self.result = None
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Arrange pairs in stitch order (top to bottom):\nUse Up/Down buttons to reorder."))
        
        self.list_widget = QListWidget()
        for pf in pair_folders:
            item = QListWidgetItem(pf.name)
            item.setData(Qt.UserRole, str(pf))
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
        self.result = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            path_str = item.data(Qt.UserRole)
            self.result.append(Path(path_str))
        super().accept()


class StitchWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stitch All Pairs")
        self.resize(900, 500)

        self.root_folder = None
        self.cam_grid_sequences = []  # per camera: list of temperature grids
        self.cam_norms = []  # per camera: (vmin, vmax)
        self.transforms = None
        self.frame_count = 0
        self._last_panorama = None

        layout = QVBoxLayout(self)

        open_layout = QHBoxLayout()
        self.open_button = QPushButton("Select Root Folder")
        self.open_button.clicked.connect(self.select_root_folder)
        self.path_label = QLabel("No folder selected")
        open_layout.addWidget(self.open_button)
        open_layout.addWidget(self.path_label)
        layout.addLayout(open_layout)

        self.preview_viewer = ImageViewer(scale_factor=4)
        layout.addWidget(self.preview_viewer, alignment=Qt.AlignCenter)

        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.frame_label = QLabel("Frame: 0 / 0")
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.frame_label)
        layout.addLayout(slider_layout)

        self.save_button = QPushButton("Save Current Panorama")
        self.save_button.clicked.connect(self.save_current)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        self.save_all_button = QPushButton("Save All Frames")
        self.save_all_button.clicked.connect(self.save_all)
        self.save_all_button.setEnabled(False)
        layout.addWidget(self.save_all_button)

    def select_root_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Root Folder Containing Pair Folders")
        if not folder:
            return
        try:
            self._load_root(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _load_root(self, folder):
        root = Path(folder)
        pair_folders = list_pair_folders(root)
        if not pair_folders:
            raise ValueError(f"No capture-session folders found in {root}")

        # Let user order the pairs
        dialog = OrderPairsDialog(pair_folders, self)
        if dialog.exec_() != QDialog.Accepted or not dialog.result:
            return
        ordered_pair_folders = dialog.result

        # Extract camera IDs from each pair in the selected order
        cam_ids = []
        pairs_to_ids = {}
        for pf in ordered_pair_folders:
            cam_a_id, cam_b_id = detect_pair_cameras(pf)
            pairs_to_ids[str(pf)] = (cam_a_id, cam_b_id)
            if not cam_ids:
                cam_ids.append(cam_a_id)
            cam_ids.append(cam_b_id)

        homographies = load_chain_homographies(ordered_pair_folders)
        self.transforms = build_reference_transforms(homographies)

        # Load sequences in the selected order
        # cam 0's frames come from the first pair folder (as cam_a); every other
        # camera's frames come from the pair folder that recorded it as cam_b.
        sequences = []
        norms = []
        for i, cam_id in enumerate(cam_ids):
            folder_for_cam = ordered_pair_folders[max(i - 1, 0)]
            raw_seq = build_camera_sequence(folder_for_cam, cam_id)
            grids = [grid for _idx, _ts, grid in raw_seq]
            sequences.append(grids)
            norms.append(compute_normalization_range(grids))

        if len(sequences) != len(self.transforms):
            raise ValueError("Number of detected cameras does not match number of homography transforms")

        self.cam_grid_sequences = sequences
        self.cam_norms = norms
        self.frame_count = min(len(s) for s in sequences)
        if self.frame_count == 0:
            raise ValueError("No frames found for one or more cameras")

        self.root_folder = root
        self.path_label.setText(str(root))
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.frame_count - 1)
        self.slider.setValue(0)
        self.save_button.setEnabled(True)
        self.save_all_button.setEnabled(True)
        self.update_preview(0)

    def _frame_for_cam(self, cam_idx, frame_idx):
        grid = self.cam_grid_sequences[cam_idx][frame_idx]
        vmin, vmax = self.cam_norms[cam_idx]
        return temps_to_uint8(grid, vmin, vmax)

    def on_slider_changed(self, value):
        self.update_preview(value)

    def update_preview(self, idx):
        if not self.cam_grid_sequences:
            return
        frames = [self._frame_for_cam(cam_idx, idx) for cam_idx in range(len(self.cam_grid_sequences))]
        panorama = stitch_frames(frames, self.transforms)
        self.preview_viewer.set_frame(panorama)
        self.frame_label.setText(f"Frame: {idx + 1} / {self.frame_count}")
        self._last_panorama = panorama

    def save_current(self):
        idx = self.slider.value()
        out_dir = ensure_dir(self.root_folder / "stitched_output")
        out_path = out_dir / f"panorama_frame_{idx + 1:04d}.png"
        cv2.imwrite(str(out_path), self._last_panorama)
        QMessageBox.information(self, "Saved", f"Saved to {out_path}")

    def save_all(self):
        out_dir = ensure_dir(self.root_folder / "stitched_output")
        for idx in range(self.frame_count):
            frames = [self._frame_for_cam(cam_idx, idx) for cam_idx in range(len(self.cam_grid_sequences))]
            panorama = stitch_frames(frames, self.transforms)
            out_path = out_dir / f"panorama_frame_{idx + 1:04d}.png"
            cv2.imwrite(str(out_path), panorama)
        QMessageBox.information(self, "Saved", f"Saved {self.frame_count} panoramas to {out_dir}")
