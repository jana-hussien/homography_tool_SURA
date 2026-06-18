import cv2
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QLabel


class ImageViewer(QLabel):
    """Displays a single grayscale thermal frame, nearest-neighbor upscaled,
    with optional 4-point quad selection by mouse click."""

    quad_completed = pyqtSignal(list)

    def __init__(self, scale_factor=12, overlay_color=QColor(255, 80, 0), parent=None):
        super().__init__(parent)
        self.scale_factor = scale_factor
        self.overlay_color = overlay_color
        self.original_frame = None
        self.selecting = False
        self.in_progress_points = []
        self.completed_quad = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #111;")

    def set_frame(self, frame):
        self.original_frame = frame
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

    def set_quad(self, quad):
        self.completed_quad = list(quad) if quad else None
        self.selecting = False
        self.in_progress_points = []
        self._refresh()

    def has_quad(self):
        return self.completed_quad is not None

    def _refresh(self):
        if self.original_frame is None:
            return
        h, w = self.original_frame.shape[:2]
        upscaled = cv2.resize(
            self.original_frame,
            (w * self.scale_factor, h * self.scale_factor),
            interpolation=cv2.INTER_NEAREST,
        )
        qimg = QImage(
            upscaled.data, upscaled.shape[1], upscaled.shape[0],
            upscaled.strides[0], QImage.Format_Grayscale8,
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
        scaled_points = [(x * self.scale_factor, y * self.scale_factor) for x, y in points]

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
        if not self.selecting or self.original_frame is None:
            super().mousePressEvent(event)
            return
        if event.button() != Qt.LeftButton:
            return

        x_orig = event.pos().x() / self.scale_factor
        y_orig = event.pos().y() / self.scale_factor

        h, w = self.original_frame.shape[:2]
        x_orig = min(max(x_orig, 0), w - 1)
        y_orig = min(max(y_orig, 0), h - 1)

        self.in_progress_points.append((x_orig, y_orig))

        if len(self.in_progress_points) == 4:
            self.completed_quad = list(self.in_progress_points)
            self.in_progress_points = []
            self.selecting = False
            self.quad_completed.emit(self.completed_quad)

        self._refresh()
