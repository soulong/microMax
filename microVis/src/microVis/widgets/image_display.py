from __future__ import annotations

import matplotlib
import numpy as np
from natsort import natsort_key

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QDrag, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import get_cmap

_MIME_TYPE = "application/x-microvis-object"


class _PassThroughPixmapItem(QGraphicsPixmapItem):
    """QGraphicsPixmapItem that ignores all mouse events, letting them fall through."""

    def mousePressEvent(self, event):
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()


class _ThumbnailView(QGraphicsView):
    """Single image thumbnail with optional overlay polygons."""

    pixel_clicked = Signal(int, int, int, int, int)  # row_idx, x, y, pixmap_w, pixmap_h
    full_res_requested = Signal(int, int)  # row_idx, gen

    def __init__(
        self,
        rgb: np.ndarray,
        well: str,
        field: int,
        stack: int,
        timepoint: int,
        polygons: list | None = None,
        overlay_alpha: float = 0.4,
        overlay_cmap: str = "viridis",
        overlay_val: float | str | None = None,
        overlay_col: str | None = None,
        n_objects: int | None = None,
        mask: np.ndarray | None = None,
        obj_values: dict | None = None,
        thumb_size: int = 210,
        row_idx: int = -1,
        parent: QWidget | None = None,
        overlay_vmin: float = 0.0,
        overlay_vmax: float = 1.0,
        outline_only: bool = False,
    ):
        super().__init__(parent)
        self._well = well
        self._field = field
        self._stack = stack
        self._timepoint = timepoint
        self._row_idx = row_idx
        self._mask = mask
        self._thumb_mask = mask  # saved for restore on zoom-out
        self._obj_values = obj_values or {}
        self._overlay_col = overlay_col
        self._overlay_vmin = overlay_vmin
        self._overlay_vmax = overlay_vmax
        self._outline_only = outline_only
        self._is_full_res = False
        self._full_res_item = None
        self._fade_timer = None
        self._full_res_gen = 0  # generation counter to reject stale results

        # Create QPixmap from numpy RGB
        self._base_rgb = rgb.copy()
        pixmap = _array_to_qpixmap(rgb)

        # Draw overlay polygons
        if polygons:
            pixmap = _draw_polygon_overlays(pixmap, polygons, overlay_alpha, overlay_cmap,
                                            self._obj_values, overlay_vmin, overlay_vmax,
                                            outline_only=self._outline_only)

        self._thumb_pixmap = pixmap

        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        # Configure view
        self.setFixedSize(thumb_size, thumb_size)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setStyleSheet("background-color: #1e1e2e; border: 1px solid #333333;")
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self._panning = False
        self._pan_start = None
        self._last_tip = ""
        self._drag_start_pos = None  # for distinguishing click vs drag
        self.setMouseTracking(True)

        # Static tooltip fallback (when no mask for per-object hover)
        # Only show when overlay_col is set (not when Color by is None)
        if self._mask is None and overlay_col:
            parts = []
            if overlay_val is not None:
                if isinstance(overlay_val, float):
                    parts.append(f"{overlay_col}: {overlay_val:.4f}")
                else:
                    parts.append(f"{overlay_col}: {overlay_val}")
            if n_objects is not None:
                parts.append(f"objects: {n_objects}")
            if parts:
                self.setToolTip("\n".join(parts))

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            # Load full-res when zoomed past native resolution
            zoom = self.transform().m11()
            if zoom > 1.0 and not self._is_full_res:
                self._is_full_res = True
                self._full_res_gen += 1
                self.full_res_requested.emit(self._row_idx, self._full_res_gen)
            elif zoom <= 1.0 and self._is_full_res:
                self._is_full_res = False
                self.remove_full_res()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        # Check for drag initiation (left button held + moved beyond threshold)
        if self._drag_start_pos is not None and event.buttons() & Qt.LeftButton:
            dist = (event.pos() - self._drag_start_pos).manhattanLength()
            if dist > 4:
                zoom = self.transform().m11()
                # Always try object drag first if mask exists
                if self._mask is not None:
                    scene_pos = self.mapToScene(event.pos())
                    x = int(scene_pos.x())
                    y = int(scene_pos.y())
                    h, w = self._mask.shape
                    if 0 <= x < w and 0 <= y < h:
                        lbl = int(self._mask[y, x])
                        if lbl > 0:
                            self._start_object_drag(lbl)
                            return
                # Zoomed in + no object found → pan
                if zoom > 1.0:
                    self._panning = True
                    self._pan_start = event.pos()
                    self.setCursor(Qt.ClosedHandCursor)
                    event.accept()
                    return
                # Unzoomed: do nothing (no pan, no object to drag)
                return

        # Per-object tooltip on hover
        if self._mask is not None:
            scene_pos = self.mapToScene(event.pos())
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            h, w = self._mask.shape
            if 0 <= x < w and 0 <= y < h:
                lbl = int(self._mask[y, x])
                if lbl > 0 and lbl in self._obj_values:
                    val = self._obj_values[lbl]
                    if isinstance(val, float):
                        tip = f"label: {lbl}\n{self._overlay_col}: {val:.4f}"
                    else:
                        tip = f"label: {lbl}\n{self._overlay_col}: {val}"
                elif lbl > 0:
                    tip = f"label: {lbl}"
                else:
                    tip = ""
                if tip != self._last_tip:
                    self._last_tip = tip
                    if tip:
                        QToolTip.showText(QCursor.pos(), tip, self)
                    else:
                        QToolTip.hideText()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        elif event.button() == Qt.LeftButton:
            if self._panning:
                self._panning = False
                self._pan_start = None
                self.setCursor(Qt.ArrowCursor)
                event.accept()
            else:
                # Short click = emit pixel_clicked (preserves existing behavior)
                if self._drag_start_pos is not None:
                    scene_pos = self.mapToScene(event.pos())
                    x = int(scene_pos.x())
                    y = int(scene_pos.y())
                    # Scene coords are always in thumbnail pixel space (both zoomed and not).
                    # Use thumbnail pixmap for bounds check and coordinate conversion.
                    pw = self._pixmap_item.pixmap().width()
                    ph = self._pixmap_item.pixmap().height()
                    if 0 <= x < pw and 0 <= y < ph:
                        self.pixel_clicked.emit(self._row_idx, x, y, pw, ph)
                self._drag_start_pos = None
                super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_zoom()
        event.accept()

    def _start_object_drag(self, label: int) -> None:
        """Initiate a QDrag carrying the ObjectKey for a mask object."""
        from microVis.widgets.label_annotation import ObjectKey, encode_object_key

        key = ObjectKey(row_idx=self._row_idx, label=label)
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_MIME_TYPE, encode_object_key(key))
        drag.setMimeData(mime)

        # Create a small drag pixmap from the object region
        if self._mask is not None:
            ys, xs = np.where(self._mask == label)
            if len(ys) > 0:
                pix = self._pixmap_item.pixmap()
                # Scale factors between pixmap and original scene
                y_min, y_max = int(ys.min()), int(ys.max())
                x_min, x_max = int(xs.min()), int(xs.max())
                # Crop from the pixmap
                crop = pix.copy(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
                drag.setPixmap(crop.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self._drag_start_pos = None
        drag.exec(Qt.CopyAction)

    def set_pixmap(self, rgb: np.ndarray, polygons: list | None = None,
                   overlay_alpha: float = 0.4, overlay_cmap: str = "viridis",
                   obj_values: dict | None = None,
                   overlay_vmin: float | None = None,
                   overlay_vmax: float | None = None,
                   outline_only: bool | None = None) -> None:
        """Update the displayed image in-place, preserving zoom/pan state."""
        if obj_values is not None:
            self._obj_values = obj_values
        if overlay_vmin is not None:
            self._overlay_vmin = overlay_vmin
        if overlay_vmax is not None:
            self._overlay_vmax = overlay_vmax
        if outline_only is not None:
            self._outline_only = outline_only
        self._base_rgb = rgb.copy()
        pixmap = _array_to_qpixmap(rgb)
        if polygons:
            pixmap = _draw_polygon_overlays(pixmap, polygons, overlay_alpha, overlay_cmap,
                                            self._obj_values, self._overlay_vmin, self._overlay_vmax,
                                            outline_only=self._outline_only)
        self._thumb_pixmap = pixmap
        if not self._is_full_res:
            self._pixmap_item.setPixmap(pixmap)

    def render_overlay(self, overlay_alpha: float, overlay_cmap: str,
                       polygons: list | None = None,
                       outline_only: bool | None = None) -> None:
        """Re-render overlay with new styling, using cached base RGB."""
        if outline_only is not None:
            self._outline_only = outline_only
        pixmap = _array_to_qpixmap(self._base_rgb)
        if polygons:
            pixmap = _draw_polygon_overlays(pixmap, polygons, overlay_alpha, overlay_cmap,
                                            self._obj_values, self._overlay_vmin, self._overlay_vmax,
                                            outline_only=self._outline_only)
        self._thumb_pixmap = pixmap
        if not self._is_full_res:
            self._pixmap_item.setPixmap(pixmap)
        elif self._full_res_base_pixmap is not None:
            # Full-res thumbnails keep the un-overlaid base; re-render the
            # full-res item with the new styling instead of leaving it stale.
            fr_pixmap = self._full_res_base_pixmap
            fr_polygons = self._full_res_polygons
            if fr_polygons and self._overlay_col is not None:
                fr_pixmap = _draw_polygon_overlays(
                    fr_pixmap, fr_polygons, overlay_alpha, overlay_cmap,
                    self._obj_values, self._overlay_vmin, self._overlay_vmax,
                    outline_only=self._outline_only)
            self._full_res_alpha = overlay_alpha
            self._full_res_cmap = overlay_cmap
            if self._full_res_item is not None:
                self._full_res_item.setPixmap(fr_pixmap)

    def set_full_res_pixmap(self, pixmap: QPixmap, gen: int = 0,
                            mask: np.ndarray | None = None,
                            obj_values: dict | None = None,
                            overlay_col: str = "",
                            overlay_vmin: float | None = None,
                            overlay_vmax: float | None = None,
                            polygons: list | None = None,
                            overlay_alpha: float | None = None,
                            overlay_cmap: str | None = None) -> None:
        """Cross-fade from thumbnail to full-resolution pixmap.

        `pixmap` is the BASE (un-overlaid) full-res render; when overlay
        styling parameters are provided the overlay is drawn here so a later
        restyle (render_overlay) can re-render the full-res item.
        """
        from PySide6.QtCore import QTimer
        # Reject stale results (gen=0 means always accept)
        if gen != 0 and gen != self._full_res_gen:
            return
        # Update mask and overlay data for hover/drag interactivity
        if mask is not None:
            # Downscale full-res mask to thumbnail coordinate space for correct hover lookup
            thumb_w = self._thumb_pixmap.width()
            thumb_h = self._thumb_pixmap.height()
            h, w = mask.shape
            if h <= thumb_h and w <= thumb_w:
                self._mask = mask
            else:
                from skimage.transform import resize as sk_resize
                scale = min(thumb_h / h, thumb_w / w)
                self._mask = sk_resize(
                    mask, (int(h * scale), int(w * scale)),
                    order=0, preserve_range=True, anti_aliasing=False,
                ).astype(mask.dtype)
        if obj_values is not None:
            self._obj_values = obj_values
        if overlay_vmin is not None:
            self._overlay_vmin = overlay_vmin
        if overlay_vmax is not None:
            self._overlay_vmax = overlay_vmax
        if overlay_col:
            self._overlay_col = overlay_col
        # Keep the un-overlaid base so render_overlay can restyle full-res
        self._full_res_base_pixmap = pixmap
        self._full_res_polygons = polygons
        self._full_res_alpha = overlay_alpha if overlay_alpha is not None else 0.4
        self._full_res_cmap = overlay_cmap if overlay_cmap is not None else "viridis"
        if polygons and overlay_col is not None:
            pixmap = _draw_polygon_overlays(
                pixmap, polygons, self._full_res_alpha, self._full_res_cmap,
                self._obj_values, self._overlay_vmin, self._overlay_vmax,
                outline_only=self._outline_only)
        # Remove previous full-res item if any
        if self._full_res_item is not None:
            self._scene.removeItem(self._full_res_item)

        # Scale full-res to match thumbnail's scene coordinates
        thumb_w = self._thumb_pixmap.width()
        thumb_h = self._thumb_pixmap.height()
        full_w = pixmap.width()
        full_h = pixmap.height()

        self._full_res_item = _PassThroughPixmapItem(pixmap)
        self._full_res_item.setZValue(1)
        self._full_res_item.setOpacity(0.0)
        # Scale so full-res pixels align with thumbnail coordinate space
        if full_w > 0 and full_h > 0:
            self._full_res_item.setScale(min(thumb_w / full_w, thumb_h / full_h))
        self._scene.addItem(self._full_res_item)

        # Fade in with QTimer
        steps = 5
        interval = 30  # ms per step → 150ms total
        self._fade_step = 0
        self._fade_steps = steps

        def _tick():
            self._fade_step += 1
            if self._full_res_item is not None:
                self._full_res_item.setOpacity(self._fade_step / self._fade_steps)
            if self._fade_step >= self._fade_steps:
                self._fade_timer.stop()

        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(_tick)
        self._fade_timer.start(interval)

    def remove_full_res(self) -> None:
        """Remove full-res overlay, restoring thumbnail view."""
        if self._fade_timer is not None:
            self._fade_timer.stop()
            self._fade_timer = None
        if self._full_res_item is not None:
            self._scene.removeItem(self._full_res_item)
            self._full_res_item = None
        # Restore original downscaled mask for hover/drag
        self._mask = self._thumb_mask

    def reset_zoom(self) -> None:
        self._is_full_res = False
        self.remove_full_res()
        self._pixmap_item.setPixmap(self._thumb_pixmap)
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def get_view_state(self) -> dict:
        return {
            "transform": self.transform(),
            "h_scroll": self.horizontalScrollBar().value(),
            "v_scroll": self.verticalScrollBar().value(),
        }

    def restore_view_state(self, state: dict) -> None:
        self.setTransform(state["transform"])
        self.horizontalScrollBar().setValue(state["h_scroll"])
        self.verticalScrollBar().setValue(state["v_scroll"])
        # A zoomed transform on a freshly rebuilt thumbnail shows a blurry
        # scaled thumbnail — re-request the full-res render when zoomed.
        if not self._is_full_res and self.transform().m11() > 1.0:
            self.full_res_requested.emit(self._row_idx, self._full_res_gen)


class ImageDisplay(QScrollArea):
    """Scrollable area displaying image thumbnails grouped by well."""

    pixel_clicked = Signal(int, int, int, int, int)  # row_idx, x, y, pixmap_w, pixmap_h
    full_res_requested = Signal(int, int)  # row_idx, gen

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self._layout.addStretch()
        self.setWidget(container)

        # Result cache for sort/resize without reprocessing
        self._results_cache: list[dict] = []
        self._cached_thumb_size: int = 0
        # Incremental display tracking: group_key → (row_widget, h_layout)
        self._row_widgets: dict = {}
        self._colorbar_added = False
        self._cached_overlay_vmin = 0.0
        self._cached_overlay_vmax = 1.0

    def save_view_state(self) -> dict:
        return self._save_current_view_state()

    def clear(self) -> None:
        self._clear_layout()
        # Also drop the cached results/row widgets — stale worker results
        # arriving after a clear() must not attach to detached layout state.
        self._results_cache = []
        self._row_widgets = {}
        self._colorbar_added = False

    def reset_all_zoom(self) -> None:
        """Reset zoom on all visible thumbnails."""
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is None or not item.widget():
                continue
            for thumb in item.widget().findChildren(_ThumbnailView):
                thumb.reset_zoom()

    def begin_results(self, thumb_size: int) -> None:
        """Clear display and prepare for progressive thumbnail insertion."""
        self._clear_layout()
        self._results_cache = []
        self._cached_thumb_size = thumb_size
        self._row_widgets = {}
        self._colorbar_added = False

    def add_result(self, result: dict, thumb_size: int, overlay_alpha: float,
                   overlay_cmap: str, saved_state: dict | None,
                   sort_by_row: bool,
                   overlay_vmin: float = 0.0,
                   overlay_vmax: float = 1.0) -> None:
        """Incrementally insert a single result as a thumbnail."""
        self._results_cache.append(result)
        self._cached_overlay_vmin = overlay_vmin
        self._cached_overlay_vmax = overlay_vmax

        # Determine grouping key
        if sort_by_row:
            group_key = result["well"]
        else:
            group_key = (result["field"], result["stack"], result["timepoint"])

        # Remove trailing stretch
        if self._layout.count() > 0:
            last = self._layout.itemAt(self._layout.count() - 1)
            if last and last.spacerItem():
                self._layout.takeAt(self._layout.count() - 1)

        # Add colorbar on first polygon result
        if not self._colorbar_added and result.get("polygons"):
            if overlay_vmin != overlay_vmax:
                cbar = _create_colorbar_widget(overlay_cmap, overlay_vmin, overlay_vmax)
                self._layout.addWidget(cbar)
                self._colorbar_added = True

        # Find or create row widget
        if group_key not in self._row_widgets:
            row_widget = QWidget()
            row_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addStretch()
            self._row_widgets[group_key] = (row_widget, row)
            # Insert row in sorted position (left-to-right, top-to-bottom)
            self._insert_row_sorted(group_key, row_widget)
        else:
            row_widget, row = self._row_widgets[group_key]

        # Insert thumbnail in sorted position within the row
        self._add_thumbnail_sorted(row, result, thumb_size, overlay_alpha,
                                   overlay_cmap, saved_state, sort_by_row,
                                   overlay_vmin, overlay_vmax)

        # Re-add trailing stretch
        self._layout.addStretch()

    def _rebuild_display(self, thumb_size, overlay_alpha, overlay_cmap,
                         saved_state, sort_by_row):
        """Rebuild the entire display from cached results."""
        old_state = self._save_current_view_state()

        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout():
                self._clear_sub_layout(item.layout())
        self._row_widgets = {}

        if old_state:
            if saved_state:
                old_state.update(saved_state)
            saved_state = old_state

        from collections import defaultdict

        results = self._results_cache

        if sort_by_row:
            groups = defaultdict(list)
            for r in results:
                groups[r["well"]].append(r)
            for well in sorted(groups.keys(), key=natsort_key):
                row_widget = QWidget()
                row_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                row = QHBoxLayout(row_widget)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)
                self._row_widgets[well] = (row_widget, row)
                for r in sorted(groups[well],
                                key=lambda x: (natsort_key(str(x["stack"])),
                                               natsort_key(str(x["timepoint"])),
                                               natsort_key(str(x["field"])))):
                    self._add_thumbnail_column(row, r, thumb_size, overlay_alpha,
                                               overlay_cmap, saved_state,
                                               self._cached_overlay_vmin,
                                               self._cached_overlay_vmax)
                row.addStretch()
                self._layout.addWidget(row_widget)
        else:
            # Group rows by (field, stack, timepoint) combo. Use row_idx as
            # the per-thumbnail key instead of (well, field, stack, timepoint)
            # tuples — non-standard datasets collapse those tuples to a single
            # key, causing all but one result to be dropped via matching[0].
            combos_set = set()
            wells_set = set()
            for r in results:
                combos_set.add((r["field"], r["stack"], r["timepoint"]))
                wells_set.add(r["well"])
            combos = sorted(combos_set,
                            key=lambda c: (natsort_key(str(c[0])),
                                           natsort_key(str(c[1])),
                                           natsort_key(str(c[2]))))
            wells = sorted(wells_set, key=natsort_key)
            idx: dict[tuple, list[dict]] = defaultdict(list)
            for r in results:
                idx[(r["well"], r["field"], r["stack"], r["timepoint"])].append(r)
            for combo in combos:
                field, stack, timepoint = combo
                row_widget = QWidget()
                row_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                row = QHBoxLayout(row_widget)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)
                self._row_widgets[combo] = (row_widget, row)
                for well in wells:
                    # On non-standard datasets without well/field/stack/timepoint
                    # columns, every result lands under the same ("","","","")
                    # key — iterate them all instead of taking only matching[0].
                    matching = idx.get((well, field, stack, timepoint), [])
                    for r in matching:
                        self._add_thumbnail_column(row, r, thumb_size,
                                                   overlay_alpha, overlay_cmap, saved_state,
                                                   self._cached_overlay_vmin,
                                                   self._cached_overlay_vmax)
                row.addStretch()
                self._layout.addWidget(row_widget)

        has_polygons = any(r.get("polygons") for r in results)
        if has_polygons:
            cbar_widget = _create_colorbar_widget(
                overlay_cmap, self._cached_overlay_vmin, self._cached_overlay_vmax)
            self._layout.addWidget(cbar_widget)
        self._layout.addStretch()

    def _save_current_view_state(self) -> dict:
        """Save view state of currently visible thumbnails."""
        state = {}
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is None or not item.widget():
                continue
            for child in item.widget().findChildren(_ThumbnailView):
                state[child._row_idx] = child.get_view_state()
        return state

    def resort_cached(self, thumb_size, overlay_alpha, overlay_cmap,
                      saved_state, sort_by_row) -> bool:
        """Re-sort cached results without reprocessing. Returns True if cache was usable."""
        if not self._results_cache:
            return False
        self._rebuild_display(thumb_size, overlay_alpha, overlay_cmap,
                              saved_state, sort_by_row)
        return True

    def update_pixmaps_in_place(self, results: list[dict], overlay_alpha: float,
                                overlay_cmap: str,
                                remove_full_res: bool = True) -> None:
        """Update existing thumbnail pixmaps without rebuilding the layout."""
        # Build lookup: row_idx → result. row_idx uniquely identifies an image
        # group regardless of which structural metadata columns exist — using
        # (well, field, stack, timepoint) tuples collapses all thumbnails to
        # the same key on non-standard datasets, causing every thumbnail to
        # receive the last result's pixmap.
        result_map: dict[int, dict] = {}
        for r in results:
            result_map[r["row_idx"]] = r

        for row_widget, _row_layout in self._row_widgets.values():
            for thumb in row_widget.findChildren(_ThumbnailView):
                r = result_map.get(thumb._row_idx)
                if r is not None:
                    thumb.set_pixmap(r["rgb"], r.get("polygons"),
                                     overlay_alpha, overlay_cmap,
                                     outline_only=r.get("outline_only"))
                    if remove_full_res:
                        thumb.remove_full_res()

        # Merge into results cache (don't replace — keep all cached results)
        for i, cached in enumerate(self._results_cache):
            if cached["row_idx"] in result_map:
                self._results_cache[i] = result_map[cached["row_idx"]]

    def restyle_overlay(self, overlay_alpha: float, overlay_cmap: str,
                        polygon_cache: dict) -> None:
        """Re-render overlay polygons with new styling, no worker dispatch."""
        for row_widget, _row_layout in self._row_widgets.values():
            for thumb in row_widget.findChildren(_ThumbnailView):
                polygons = polygon_cache.get(thumb._row_idx)
                thumb.render_overlay(overlay_alpha, overlay_cmap, polygons)

    def _insert_row_sorted(self, group_key, row_widget):
        """Insert row widget at the correct sorted position in the layout."""
        # Build list of existing group_keys in layout order
        existing = []
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is None or not item.widget():
                continue
            for gk, (rw, _) in self._row_widgets.items():
                if rw is item.widget():
                    existing.append(gk)
                    break

        # Find insertion index
        insert_idx = len(existing)
        for i, gk in enumerate(existing):
            if natsort_key(str(group_key)) < natsort_key(str(gk)):
                insert_idx = i
                break

        # Remove trailing stretch, insert row, re-add stretch
        last = self._layout.itemAt(self._layout.count() - 1)
        has_stretch = last is not None and last.spacerItem() is not None
        if has_stretch:
            self._layout.takeAt(self._layout.count() - 1)
        self._layout.insertWidget(insert_idx, row_widget)
        if has_stretch:
            self._layout.addStretch()

    def _add_thumbnail_sorted(self, row, r, thumb_size, overlay_alpha, overlay_cmap,
                               saved_state, sort_by_row,
                               overlay_vmin=0.0, overlay_vmax=1.0):
        """Insert thumbnail at sorted position within a row."""
        if sort_by_row:
            # Sort by (stack, timepoint, field) within a well row
            new_key = (r["stack"], r["timepoint"], r["field"])
        else:
            # Sort by well name within a (field, stack, timepoint) row
            new_key = r["well"]

        # Find insertion index by comparing sort keys with existing thumbnails.
        # Uses thumb._well/_field/_stack/_timepoint directly instead of parsing
        # the label text (which is now dynamic — see _build_meta_label).
        insert_idx = row.count() - 1  # before trailing stretch
        for i in range(row.count() - 1):  # skip trailing stretch
            item = row.itemAt(i)
            if item is None or not item.layout():
                continue
            col_layout = item.layout()
            # Find the _ThumbnailView child to read sort attributes
            for j in range(col_layout.count()):
                child_item = col_layout.itemAt(j)
                if child_item and child_item.widget():
                    w = child_item.widget()
                    if isinstance(w, _ThumbnailView):
                        if sort_by_row:
                            existing_key = (w._stack, w._timepoint, w._field)
                        else:
                            existing_key = w._well
                        if natsort_key(str(new_key)) < natsort_key(str(existing_key)):
                            insert_idx = i
                        break
            if insert_idx < row.count() - 1:
                break

        col = QVBoxLayout()
        col.setSpacing(2)
        col.setContentsMargins(0, 0, 0, 0)

        meta = QLabel(r.get("meta_label", ""))
        meta.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
        meta.setAlignment(Qt.AlignCenter)
        col.addWidget(meta)

        thumb = _ThumbnailView(
            r["rgb"], r["well"], r["field"], r["stack"], r["timepoint"],
            r.get("polygons"), overlay_alpha, overlay_cmap,
            r.get("overlay_val"), r.get("overlay_col"),
            r.get("n_objects"), r.get("mask"), r.get("obj_values"),
            thumb_size, row_idx=r.get("row_idx", -1),
            overlay_vmin=overlay_vmin, overlay_vmax=overlay_vmax,
            outline_only=r.get("outline_only", False),
        )
        thumb.pixel_clicked.connect(self.pixel_clicked)
        thumb.full_res_requested.connect(self.full_res_requested)
        if saved_state:
            key = r.get("row_idx", -1)
            if key in saved_state:
                thumb.restore_view_state(saved_state[key])
        col.addWidget(thumb)
        row.insertLayout(insert_idx, col)

    def _add_thumbnail_column(self, row, r, thumb_size, overlay_alpha, overlay_cmap,
                               saved_state, overlay_vmin=0.0, overlay_vmax=1.0):
        col = QVBoxLayout()
        col.setSpacing(2)
        col.setContentsMargins(0, 0, 0, 0)

        meta = QLabel(r.get("meta_label", ""))
        meta.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
        meta.setAlignment(Qt.AlignCenter)
        col.addWidget(meta)

        thumb = _ThumbnailView(
            r["rgb"],
            r["well"],
            r["field"],
            r["stack"],
            r["timepoint"],
            r.get("polygons"),
            overlay_alpha,
            overlay_cmap,
            r.get("overlay_val"),
            r.get("overlay_col"),
            r.get("n_objects"),
            r.get("mask"),
            r.get("obj_values"),
            thumb_size,
            row_idx=r.get("row_idx", -1),
            overlay_vmin=overlay_vmin, overlay_vmax=overlay_vmax,
            outline_only=r.get("outline_only", False),
        )
        thumb.pixel_clicked.connect(self.pixel_clicked)
        thumb.full_res_requested.connect(self.full_res_requested)
        if saved_state:
            key = r.get("row_idx", -1)
            if key in saved_state:
                thumb.restore_view_state(saved_state[key])
        col.addWidget(thumb)
        row.addLayout(col)

    def _clear_layout(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout():
                self._clear_sub_layout(item.layout())


    def _clear_sub_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.deleteLater()
            sub = item.layout()
            if sub is not None:
                self._clear_sub_layout(sub)
                del sub


def _array_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert (H, W, 3) uint8 RGB array to QPixmap via copy-safe QImage."""
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    # Create pixmap immediately while rgb reference lives
    return QPixmap.fromImage(qimage.copy())


def _draw_polygon_overlays(
    pixmap: QPixmap,
    polygons: list[tuple[int, np.ndarray]],
    alpha: float = 0.4,
    cmap_name: str = "viridis",
    obj_values: dict[int, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    outline_only: bool = False,
) -> QPixmap:
    """Draw polygon overlays onto a copy of pixmap using QPainter.

    When ``outline_only`` is True, only the 1px gold outline is drawn (no
    fill) — used when a mask is selected in "Select object" but "Color by"
    is None. When False, polygons are filled with the color-mapped value
    at the given alpha and then outlined.
    """
    result = QPixmap(pixmap)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)

    cmap = get_cmap(cmap_name)
    gold_pen = QPen(QColor(0xFF, 0xD7, 0x00), 1.0)

    # Use actual metric values for coloring when available
    use_values = obj_values is not None and vmin is not None and vmax is not None and vmin != vmax

    # Build categorical mapping if values are strings
    _cat_map = None
    if obj_values:
        str_vals = {v for v in obj_values.values() if isinstance(v, str)}
        if str_vals:
            sorted_vals = sorted(str_vals)
            _cat_map = {v: i / max(len(sorted_vals) - 1, 1)
                        for i, v in enumerate(sorted_vals)}
            use_values = True

    for label_val, contour in polygons:
        if len(contour) < 3:
            continue

        path = QPainterPath()
        path.moveTo(contour[0, 1], contour[0, 0])
        for pt in contour[1:]:
            path.lineTo(pt[1], pt[0])
        path.closeSubpath()

        if not outline_only:
            if use_values and label_val in obj_values:
                val = obj_values[label_val]
                if isinstance(val, (int, float)) and vmin is not None and vmax is not None and vmin != vmax:
                    idx = (val - vmin) / (vmax - vmin)
                elif _cat_map and isinstance(val, str) and val in _cat_map:
                    idx = _cat_map[val]
                else:
                    idx = 0.5
            else:
                idx = 0.5

            rgba = cmap(np.clip(idx, 0.0, 1.0))
            fill_color = QColor.fromRgbF(rgba[0], rgba[1], rgba[2], alpha)
            painter.fillPath(path, fill_color)

        painter.setPen(gold_pen)
        painter.drawPath(path)

    painter.end()
    return result


def _create_colorbar_widget(cmap_name: str,
                            vmin: float = 0.0,
                            vmax: float = 1.0) -> QWidget:
    """Create a horizontal colorbar widget for object overlay mapping."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(4)

    fig = Figure(facecolor="#252536", figsize=(3.0, 0.45), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.45, bottom=0.12)
    ax = fig.add_subplot(111)

    cmap = get_cmap(cmap_name)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=ax, orientation="horizontal")
    ax.xaxis.tick_top()
    ax.set_facecolor("#252536")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.tick_params(colors="white", labelsize=8)

    canvas = FigureCanvasQTAgg(fig)
    canvas.setFixedWidth(300)
    canvas.setFixedHeight(45)
    canvas.setStyleSheet("background-color: #252536;")
    layout.addWidget(canvas)

    layout.addStretch()
    return widget
