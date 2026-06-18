from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)


class ControlPanel(QWidget):
    select_a_clicked = pyqtSignal()
    select_b_clicked = pyqtSignal()
    clear_a_clicked = pyqtSignal()
    clear_b_clicked = pyqtSignal()
    compute_clicked = pyqtSignal()
    apply_matrix_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    generate_examples_clicked = pyqtSignal()
    capture_stitch_mode_clicked = pyqtSignal()
    swap_cameras_clicked = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    preview_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        nav_group = QGroupBox("Playback")
        nav_layout = QHBoxLayout(nav_group)
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play_toggled)
        nav_layout.addWidget(self.play_button)
        layout.addWidget(nav_group)

        quad_group = QGroupBox("Overlap Selection")
        quad_layout = QVBoxLayout(quad_group)

        self.swap_button = QPushButton("Swap Cam A / Cam B")
        self.swap_button.clicked.connect(self.swap_cameras_clicked.emit)
        quad_layout.addWidget(self.swap_button)

        row_a = QHBoxLayout()
        self.select_a_button = QPushButton("Select Cam A Overlap")
        self.clear_a_button = QPushButton("Clear A")
        self.status_a_label = QLabel("Cam A: not set")
        self.select_a_button.clicked.connect(self.select_a_clicked.emit)
        self.clear_a_button.clicked.connect(self.clear_a_clicked.emit)
        row_a.addWidget(self.select_a_button)
        row_a.addWidget(self.clear_a_button)
        quad_layout.addLayout(row_a)
        quad_layout.addWidget(self.status_a_label)

        row_b = QHBoxLayout()
        self.select_b_button = QPushButton("Select Cam B Overlap")
        self.clear_b_button = QPushButton("Clear B")
        self.status_b_label = QLabel("Cam B: not set")
        self.select_b_button.clicked.connect(self.select_b_clicked.emit)
        self.clear_b_button.clicked.connect(self.clear_b_clicked.emit)
        row_b.addWidget(self.select_b_button)
        row_b.addWidget(self.clear_b_button)
        quad_layout.addLayout(row_b)
        quad_layout.addWidget(self.status_b_label)

        layout.addWidget(quad_group)

        homography_group = QGroupBox("Homography")
        h_layout = QVBoxLayout(homography_group)
        self.compute_button = QPushButton("Compute Homography")
        self.compute_button.setEnabled(False)
        self.compute_button.clicked.connect(self.compute_clicked.emit)
        h_layout.addWidget(self.compute_button)

        self.matrix_display = QTextEdit()
        self.matrix_display.setFixedHeight(90)
        self.matrix_display.setPlaceholderText(
            "H matrix will appear here, or type/paste your own 3x3 matrix\n"
            "(space-separated values, one row per line) and click Apply"
        )
        h_layout.addWidget(self.matrix_display)

        self.apply_matrix_button = QPushButton("Apply Typed Matrix")
        self.apply_matrix_button.clicked.connect(self.apply_matrix_clicked.emit)
        h_layout.addWidget(self.apply_matrix_button)

        self.preview_button = QPushButton("Show Stitched Preview")
        self.preview_button.setCheckable(True)
        self.preview_button.setEnabled(False)
        self.preview_button.toggled.connect(self.preview_toggled.emit)
        h_layout.addWidget(self.preview_button)

        layout.addWidget(homography_group)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        self.save_button = QPushButton("Save Homography")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_clicked.emit)
        export_layout.addWidget(self.save_button)

        self.examples_button = QPushButton("Generate Example Images")
        self.examples_button.setEnabled(False)
        self.examples_button.clicked.connect(self.generate_examples_clicked.emit)
        export_layout.addWidget(self.examples_button)

        layout.addWidget(export_group)

        stitch_group = QGroupBox("Panorama")
        stitch_layout = QVBoxLayout(stitch_group)
        self.capture_stitch_button = QPushButton("Stitch Real Capture + RGB...")
        self.capture_stitch_button.clicked.connect(self.capture_stitch_mode_clicked.emit)
        stitch_layout.addWidget(self.capture_stitch_button)

        layout.addWidget(stitch_group)

        layout.addStretch()

    def _on_play_toggled(self, checked):
        self.play_button.setText("Pause" if checked else "Play")
        self.play_toggled.emit(checked)

    def set_status_a(self, defined):
        self.status_a_label.setText("Cam A: ✓ defined" if defined else "Cam A: not set")

    def set_status_b(self, defined):
        self.status_b_label.setText("Cam B: ✓ defined" if defined else "Cam B: not set")

    def set_compute_enabled(self, enabled):
        self.compute_button.setEnabled(enabled)

    def set_matrix_text(self, H):
        if H is None:
            self.matrix_display.setPlainText("")
            return
        rows = ["  ".join(f"{v: .4f}" for v in row) for row in H]
        self.matrix_display.setPlainText("\n".join(rows))

    def set_post_compute_enabled(self, enabled):
        self.save_button.setEnabled(enabled)
        self.examples_button.setEnabled(enabled)
        self.preview_button.setEnabled(enabled)
