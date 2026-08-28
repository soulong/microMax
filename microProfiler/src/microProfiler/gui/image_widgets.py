from __future__ import annotations

from typing import Optional

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# Display percentiles — class-level attrs on ImageViewer, set by sidebar vmin/vmax spinboxes.
# Only written on the main thread (side-effect of UI signal handlers), safe without locks.
# Accessed by _array_to_pixmap which also runs on the main thread (QPainter requirement).


def _array_to_pixmap(arr: np.ndarray, vmin: float = 0.1, vmax: float = 99.9) -> QPixmap:
    arr = np.array(arr, dtype=np.float64, copy=True)
    if arr.max() > arr.min():
        lo, hi = np.percentile(arr, (vmin, vmax))
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        arr = arr.clip(0, 1)
    arr = np.ascontiguousarray((arr * 255).round().clip(0, 255).astype(np.uint8))
    h, w = arr.shape
    img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(img)


class ImageViewer(QGraphicsView):

    display_vmin = 0.1
    display_vmax = 99.9

    zoomed = Signal()
    panned = Signal()
    view_reset = Signal()
    pixel_clicked = Signal(int, int, float)  # x, y, intensity value

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._fit_to_view = True
        self._zoom_level = 0
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self._reloading = False
        self._base_array: Optional[np.ndarray] = None
        self._overlay_mask: Optional[np.ndarray] = None
        self._overlay_visible = False
        self._overlay_alpha = 0.4
        self._last_pixel_value: Optional[int] = None
        self._composited_pixmap: Optional[QPixmap] = None
        self._base_pixmap: Optional[QPixmap] = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumSize(100, 100)
        self._show_axes = False

        # Handle pixel-click signals internally
        self.pixel_clicked.connect(self._on_pixel_clicked)

    def _on_pixel_clicked(self, x: int, y: int, value: float) -> None:
        self._last_pixel_value = int(round(value))
        self.viewport().update()

    def refresh(self) -> None:
        if self._base_array is not None:
            self.set_image(self._base_array)

    def set_show_axes(self, show: bool) -> None:
        self._show_axes = show
        self.viewport().update()

    def set_image(self, arr: np.ndarray | QImage) -> None:
        self._base_array = arr if isinstance(arr, np.ndarray) else None
        self._overlay_mask = None
        self._overlay_visible = False
        self._last_pixel_value = None
        self._composited_pixmap = None
        self._reloading = True
        self._scene.clear()
        if isinstance(arr, QImage):
            pixmap = QPixmap.fromImage(arr)
        else:
            pixmap = _array_to_pixmap(arr, self.display_vmin, self.display_vmax)
        self._base_pixmap = pixmap
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._fit_to_view = True
        self._zoom_level = 0
        self._reloading = False

    def clear_image(self) -> None:
        self._base_array = None
        self._base_pixmap = None
        self._composited_pixmap = None
        self._overlay_mask = None
        self._overlay_visible = False
        self._last_pixel_value = None
        self._fit_to_view = True
        self._zoom_level = 0
        self._reloading = True
        self._scene.clear()
        self._pixmap_item = None
        self._reloading = False

    def overlay_mask(self, mask: np.ndarray, alpha: float = 0.7) -> None:
        if self._base_array is None:
            return
        if mask.shape[:2] != self._base_array.shape[:2]:
            from skimage.transform import resize
            mask = resize(mask, self._base_array.shape[:2], order=0, preserve_range=True).astype(np.uint16)
        self._overlay_mask = mask
        self._overlay_alpha = alpha
        self._overlay_visible = True
        self._composited_pixmap = None
        self._apply_overlay()

    def set_overlay_visible(self, visible: bool) -> None:
        if visible == self._overlay_visible:
            return  # already in requested state
        self._overlay_visible = visible
        old_transform = self.transform()
        old_h = self.horizontalScrollBar().value()
        old_v = self.verticalScrollBar().value()
        old_fit = self._fit_to_view
        old_zoom = self._zoom_level

        if visible and self._overlay_mask is not None:
            pixmap = self._composited_pixmap
            if pixmap is None:
                self._apply_overlay()  # computes and caches
                pixmap = self._composited_pixmap
        elif not visible and self._base_pixmap is not None:
            pixmap = self._base_pixmap
        else:
            return

        if pixmap is not None:
            self._reloading = True
            self._scene.clear()
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._scene.addItem(self._pixmap_item)
            self._reloading = False

        # Restore view state
        if old_fit and self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        elif not old_fit:
            self.setTransform(old_transform)
            self.horizontalScrollBar().setValue(old_h)
            self.verticalScrollBar().setValue(old_v)
            self._fit_to_view = old_fit
            self._zoom_level = old_zoom

    def _apply_overlay(self) -> None:
        if self._base_array is None or self._overlay_mask is None:
            return
        old_transform = self.transform()
        old_h = self.horizontalScrollBar().value()
        old_v = self.verticalScrollBar().value()
        old_fit = self._fit_to_view
        old_zoom = self._zoom_level

        base = self._base_array.astype(np.float64)
        lo, hi = np.percentile(base[base > 0] if (base > 0).any() else base, (self.display_vmin, self.display_vmax))
        if hi > lo:
            base = (base - lo) / (hi - lo)
        base = base.clip(0, 1)

        mask = self._overlay_mask
        labels = np.unique(mask)
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.float64)
        for lbl in labels:
            if lbl == 0:
                continue
            rnd = np.random.RandomState(int(lbl) * 7 + 13)
            color = np.array([rnd.randint(60, 256) for _ in range(3)], dtype=np.float64) / 255.0
            overlay[mask == lbl, :3] = color
            overlay[mask == lbl, 3] = self._overlay_alpha

        base_rgb = np.stack([base] * 3, axis=-1)
        a = overlay[..., 3:4]
        composited = overlay[..., :3] * a + base_rgb * (1 - a)
        composited = np.ascontiguousarray((composited.clip(0, 1) * 255).astype(np.uint8))

        qimg = QImage(composited.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._composited_pixmap = pixmap  # cache for fast toggle

        self._reloading = True
        self._scene.clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._reloading = False

        # Restore view state
        if old_fit and self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        elif not old_fit:
            self.setTransform(old_transform)
            self.horizontalScrollBar().setValue(old_h)
            self.verticalScrollBar().setValue(old_v)
            self._fit_to_view = old_fit
            self._zoom_level = old_zoom

    def _reset_view(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._fit_to_view = True
            self._zoom_level = 0

    def mouseDoubleClickEvent(self, event):
        self._reset_view()
        self.view_reset.emit()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if not (event.modifiers() & Qt.ControlModifier):
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale(1.15, 1.15)
            self._zoom_level += 1
            self._fit_to_view = False
            self.zoomed.emit()
        elif delta < 0 and self._zoom_level > 0:
            self.scale(1 / 1.15, 1 / 1.15)
            self._zoom_level -= 1
            self.zoomed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._base_array is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            x = int(round(scene_pos.x()))
            y = int(round(scene_pos.y()))
            h, w = self._base_array.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                value = float(self._base_array[y, x])
                self.pixel_clicked.emit(x, y, value)
        super().mousePressEvent(event)

    @staticmethod
    def _calc_tick_step(size: int) -> int:
        target = size / 8
        magnitude = 10 ** math.floor(math.log10(target))
        residual = target / magnitude
        if residual < 1.5:
            step = int(magnitude)
        elif residual < 3.5:
            step = int(2 * magnitude)
        elif residual < 7.5:
            step = int(5 * magnitude)
        else:
            step = int(10 * magnitude)
        return max(1, step)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap_item is None:
            return
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing)

        viewport_rect = self.viewport().rect()

        # ── Axes (left y-axis + bottom x-axis) ──
        if self._show_axes and self._base_array is not None:
            h, w = self._base_array.shape[:2]
            tick_len = 6
            label_offset = 2
            axis_color = QColor(200, 200, 200)
            text_color = QColor(220, 220, 220)

            x_step = self._calc_tick_step(w)
            y_step = self._calc_tick_step(h)

            # Y-axis: draw at image left edge, clamped to viewport edge when off-screen
            y_start = max(0, self.mapToScene(0, 0).y())
            y_end = min(h - 1, self.mapToScene(viewport_rect.width(), viewport_rect.height()).y())
            y_start_tick = int(math.ceil(y_start / y_step) * y_step)
            for y in range(y_start_tick, int(y_end) + 1, y_step):
                scene_pt = self.mapFromScene(0, y)
                if scene_pt.y() < 0 or scene_pt.y() > viewport_rect.height():
                    continue
                draw_x = max(0, int(scene_pt.x()))
                painter.setPen(QPen(axis_color, 1))
                painter.drawLine(
                    draw_x, int(scene_pt.y()),
                    draw_x + tick_len, int(scene_pt.y()),
                )
                painter.setPen(text_color)
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(
                    draw_x + tick_len + label_offset,
                    int(scene_pt.y()) - 4, 40, 12,
                    Qt.AlignLeft | Qt.AlignVCenter,
                    str(y),
                )

            # X-axis: draw at image bottom edge, clamped to viewport edge when off-screen
            x_start = max(0, self.mapToScene(0, 0).x())
            x_end = min(w - 1, self.mapToScene(viewport_rect.width(), viewport_rect.height()).x())
            x_start_tick = int(math.ceil(x_start / x_step) * x_step)
            for x in range(x_start_tick, int(x_end) + 1, x_step):
                scene_pt = self.mapFromScene(x, 0)
                if scene_pt.x() < 0 or scene_pt.x() > viewport_rect.width():
                    continue
                draw_y = min(viewport_rect.height(), max(0, int(scene_pt.y())))
                painter.setPen(QPen(axis_color, 1))
                painter.drawLine(
                    int(scene_pt.x()), draw_y,
                    int(scene_pt.x()), draw_y - tick_len,
                )
                painter.setPen(text_color)
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(
                    int(scene_pt.x()) - 20,
                    draw_y - tick_len - label_offset - 12,
                    40, 12,
                    Qt.AlignCenter,
                    str(x),
                )

        # ── Pixel intensity overlay (always visible at top-left of image) ──
        if self._last_pixel_value is not None:
            pen = QPen(QColor(255, 230, 0))
            pen.setWidth(1)
            painter.setPen(pen)
            font = painter.font()
            font.setPointSize(11)
            font.setBold(True)
            painter.setFont(font)
            text = str(self._last_pixel_value)
            # Draw at a fixed viewport position (top-left of image area)
            origin_vp = self.mapFromScene(0, 0)
            draw_x = max(4, int(origin_vp.x()) + 2)
            draw_y = max(2, int(origin_vp.y()) + 2)
            painter.drawText(draw_x, draw_y, viewport_rect.width(), 20,
                             Qt.AlignLeft, text)

        painter.end()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        if not getattr(self, "_reloading", False):
            self.panned.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item and self._fit_to_view:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)


class ChannelTile(QWidget):

    def __init__(self, label: str, arr: np.ndarray | QImage | None = None, parent=None):
        super().__init__(parent)
        self._label_widget = QLabel(label)
        self._label_widget.setAlignment(Qt.AlignCenter)
        self._viewer = ImageViewer()
        self._viewer.setMinimumSize(200, 200)
        if arr is not None:
            self._viewer.set_image(arr)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._label_widget)
        layout.addWidget(self._viewer)

    def set_image(self, arr: np.ndarray | QImage) -> None:
        self._viewer.set_image(arr)


