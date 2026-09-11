"""Label annotation panel for classifying objects via drag-and-drop."""
from __future__ import annotations

import json
from dataclasses import dataclass

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

_MIME_TYPE = "application/x-microvis-object"

# ── Object Key ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ObjectKey:
    """Unique identifier for a mask object in an image.

    row_idx uniquely identifies one image group within the dataset (works
    for both standard plate datasets and non-standard ones where
    well/field/stack/timepoint are absent). label is an int (object ID
    within a mask).
    """

    row_idx: int
    label: int

    def to_dict(self) -> dict:
        return {
            "row_idx": self.row_idx,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ObjectKey:
        return cls(
            row_idx=int(d["row_idx"]),
            label=int(d["label"]),
        )


def encode_object_key(key: ObjectKey) -> bytes:
    return json.dumps(key.to_dict()).encode("utf-8")


def decode_object_key(data: bytes) -> ObjectKey:
    return ObjectKey.from_dict(json.loads(data.decode("utf-8")))


# ── Object Thumbnail Widget ───────────────────────────────────────────────────


class _ObjectThumb(QFrame):
    """Small cropped/masked object display inside a class box."""

    clicked = Signal(object)  # ObjectKey

    def __init__(self, key: ObjectKey, parent_box: _ClassBox, thumb_size: int = 64):
        super().__init__()
        self.key = key
        self.parent_box = parent_box
        self._thumb_size = thumb_size
        self._pixmap: QPixmap | None = None
        self._img_label: QLabel | None = None
        self._drag_start_pos = None

        self.setFixedSize(thumb_size + 4, thumb_size + 4)
        self.setStyleSheet(
            "_ObjectThumb { background-color: #1e1e2e; border: 1px solid #333333; "
            "border-radius: 4px; } "
            "_ObjectThumb:hover { border-color: #5a8a9a; }"
        )
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

        # Placeholder layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._placeholder = QLabel()
        self._placeholder.setFixedSize(thumb_size, thumb_size)
        self._placeholder.setStyleSheet(
            "background-color: #333333; border-radius: 2px;"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._placeholder)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Replace placeholder with actual cropped object image."""
        self._pixmap = pixmap
        self._placeholder.hide()
        # Remove any previous image label first — repeated set_pixmap calls
        # (e.g. re-crops after a box-to-box move) would otherwise stack
        # overlapping labels in the thumb.
        if self._img_label is not None:
            self._layout.removeWidget(self._img_label)
            self._img_label.deleteLater()
            self._img_label = None
        self._img_label = QLabel()
        self._img_label.setPixmap(
            pixmap.scaled(
                self._thumb_size,
                self._thumb_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self._img_label.setFixedSize(self._thumb_size, self._thumb_size)
        self._img_label.setStyleSheet("border: none; background: transparent;")
        self._layout.addWidget(self._img_label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if (
            self._drag_start_pos is not None
            and (event.pos() - self._drag_start_pos).manhattanLength() > 8
        ):
            self._start_drag()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drag_start_pos is not None:
            # Short click = remove
            if (event.pos() - self._drag_start_pos).manhattanLength() <= 8:
                self.clicked.emit(self.key)
            self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _start_drag(self) -> None:
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_MIME_TYPE, encode_object_key(self.key))
        drag.setMimeData(mime)

        # Use current pixmap as drag pixmap, or a placeholder
        if self._pixmap is not None:
            drag.setPixmap(
                self._pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            pix = QPixmap(48, 48)
            pix.fill(Qt.gray)
            drag.setPixmap(pix)

        drag.exec(Qt.MoveAction)
        self._drag_start_pos = None


# ── Class Box ─────────────────────────────────────────────────────────────────


class _ClassBox(QFrame):
    """Drop target for one annotation class. Displays cropped object thumbnails."""

    object_added = Signal(object, object)  # ObjectKey, class_name
    object_moved = Signal(object, object, object)  # ObjectKey, from_class, to_class

    _ACCENT_COLORS = [
        "#5a8a9a",  # teal
        "#7a5aaa",  # purple
        "#aa5a6a",  # rose
        "#5aaa7a",  # green
        "#aa8a5a",  # amber
        "#5a6aaa",  # blue
        "#8aaa5a",  # lime
        "#aa5a8a",  # pink
    ]

    _NUM_ROWS = 2  # display objects in 2 rows

    def __init__(self, class_name: str, class_index: int, panel: "LabelAnnotationPanel", parent: QWidget | None = None):
        super().__init__(parent)
        self._panel = panel
        self.class_name = class_name
        self._thumbs: dict[ObjectKey, _ObjectThumb] = {}
        self._thumb_order: list[ObjectKey] = []
        self._accent = self._ACCENT_COLORS[class_index % len(self._ACCENT_COLORS)]

        self.setAcceptDrops(True)
        self.setMinimumWidth(120)
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setObjectName("classBox")
        self.setStyleSheet(
            "QFrame#classBox { background-color: #2d2d44; "
            f"border: 2px solid {self._accent}; "
            "border-radius: 8px; padding: 4px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 4)
        layout.setSpacing(2)

        # Header: class name + count (compact)
        self._header = QLabel(f"{class_name} (0)")
        self._header.setStyleSheet(
            f"color: {self._accent}; font-weight: bold; font-size: 8pt; border: none; "
            "background: transparent;"
        )
        self._header.setAlignment(Qt.AlignCenter)
        self._header.setFixedHeight(14)
        layout.addWidget(self._header)

        # Scrollable area for object thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self._thumb_container = QWidget()
        self._thumb_grid = QGridLayout(self._thumb_container)
        self._thumb_grid.setContentsMargins(2, 2, 2, 2)
        self._thumb_grid.setSpacing(4)
        scroll.setWidget(self._thumb_container)
        layout.addWidget(scroll)

    def _relayout_grid(self) -> None:
        """Rebuild the grid layout with _NUM_ROWS rows."""
        # Remove all items from grid
        while self._thumb_grid.count():
            item = self._thumb_grid.takeAt(0)
            if item.widget():
                self._thumb_grid.removeWidget(item.widget())
        # Re-add in row-major order
        cols = max(1, (len(self._thumb_order) + self._NUM_ROWS - 1) // self._NUM_ROWS)
        for i, key in enumerate(self._thumb_order):
            thumb = self._thumbs.get(key)
            if thumb is not None:
                row = i % self._NUM_ROWS
                col = i // self._NUM_ROWS
                self._thumb_grid.addWidget(thumb, row, col)

    def _update_header(self) -> None:
        n = len(self._thumbs)
        self._header.setText(f"{self.class_name} ({n})")

    def add_object(self, key: ObjectKey, thumb_size: int = 64,
                   emit_signal: bool = True) -> _ObjectThumb:
        """Add an object (shows placeholder). Returns the thumb widget.

        emit_signal=False suppresses object_added — used by the box-to-box
        move path, which already emits object_moved (main_window dispatches
        the crop from either signal; emitting both would double-crop).
        """
        if key in self._thumbs:
            return self._thumbs[key]

        thumb = _ObjectThumb(key, self, thumb_size)
        thumb.clicked.connect(self._on_thumb_clicked)

        self._thumbs[key] = thumb
        self._thumb_order.append(key)
        self._relayout_grid()
        self._update_header()
        if emit_signal:
            self.object_added.emit(key, self.class_name)
        return thumb

    def remove_object(self, key: ObjectKey) -> None:
        """Remove an object from this class box."""
        thumb = self._thumbs.pop(key, None)
        if thumb is not None:
            self._thumb_order.remove(key)
            self._thumb_grid.removeWidget(thumb)
            thumb.deleteLater()
            self._relayout_grid()
            self._update_header()

    def has_object(self, key: ObjectKey) -> bool:
        return key in self._thumbs

    def get_object_keys(self) -> list[ObjectKey]:
        return list(self._thumbs.keys())

    def set_object_pixmap(self, key: ObjectKey, pixmap: QPixmap) -> None:
        """Update an object's thumbnail with the cropped pixmap."""
        thumb = self._thumbs.get(key)
        if thumb is not None:
            thumb.set_pixmap(pixmap)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(_MIME_TYPE):
            event.acceptProposedAction()
            self.setStyleSheet(
                "QFrame#classBox { background-color: #3a3a52; "
                f"border: 2px solid {self._accent}; "
                "border-radius: 8px; padding: 4px; }"
            )

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet(
            "QFrame#classBox { background-color: #2d2d44; "
            f"border: 2px solid {self._accent}; "
            "border-radius: 8px; padding: 4px; }"
        )

    def dropEvent(self, event) -> None:
        self.setStyleSheet(
            "QFrame#classBox { background-color: #2d2d44; "
            f"border: 2px solid {self._accent}; "
            "border-radius: 8px; padding: 4px; }"
        )
        data = event.mimeData().data(_MIME_TYPE)
        if data.isEmpty():
            return
        try:
            key = decode_object_key(bytes(data))
        except Exception:
            return
        event.acceptProposedAction()

        # If the object is already in this box, ignore
        if key in self._thumbs:
            return

        # Check if it came from another class box (move)
        source_box = event.source()
        if isinstance(source_box, _ObjectThumb) and hasattr(source_box, "parent_box"):
            src_box = source_box.parent_box
            if isinstance(src_box, _ClassBox) and src_box is not self:
                src_box.remove_object(key)
                self.object_moved.emit(key, src_box.class_name, self.class_name)
                # object_moved already triggers the crop (main_window), so
                # suppress the duplicate object_added emission — otherwise a
                # single drag dispatches two CropWorkers and stacks labels.
                self.add_object(key, emit_signal=False)
                return
        else:
            # Drag from image display — enforce cross-box uniqueness:
            # an object may only live in one class box at a time.
            for cname in self._panel._class_order:
                if cname != self.class_name:
                    other_box = self._panel._class_boxes.get(cname)
                    if other_box is not None and other_box.has_object(key):
                        other_box.remove_object(key)

        self.add_object(key)

    def _on_thumb_clicked(self, key: ObjectKey) -> None:
        self.remove_object(key)


# ── Label Annotation Panel ────────────────────────────────────────────────────


class LabelAnnotationPanel(QFrame):
    """Panel below image view showing class boxes for object annotation."""

    crop_requested = Signal(object, str)  # ObjectKey, class_name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._class_boxes: dict[str, _ClassBox] = {}
        self._class_order: list[str] = []
        self._class_counter: int = 0

        self.setStyleSheet(
            "LabelAnnotationPanel { background-color: #252536; "
            "border-top: 1px solid #333333; }"
        )
        self.setMinimumHeight(120)
        self.setMaximumHeight(250)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QLabel("Class Boxes (drag objects here)")
        header.setStyleSheet(
            "color: #888888; font-size: 8pt; border: none; background: transparent;"
        )
        layout.addWidget(header)

        # Scrollable horizontal area for class boxes
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self._boxes_container = QWidget()
        self._boxes_layout = QHBoxLayout(self._boxes_container)
        self._boxes_layout.setContentsMargins(4, 4, 4, 4)
        self._boxes_layout.setSpacing(8)
        self._boxes_layout.addStretch()
        self._scroll.setWidget(self._boxes_container)
        layout.addWidget(self._scroll)

    def add_class(self, class_name: str) -> None:
        """Add a new class box. No-op if class already exists."""
        if class_name in self._class_boxes:
            return
        box = _ClassBox(class_name, self._class_counter, self)
        self._class_counter += 1
        idx = self._boxes_layout.count() - 1
        self._boxes_layout.insertWidget(idx, box)
        self._class_boxes[class_name] = box
        self._class_order.append(class_name)
        # Connect signals to re-emit as panel-level signals
        box.object_added.connect(self._on_object_added)
        box.object_moved.connect(self._on_object_moved)

    def _on_object_added(self, key: ObjectKey, class_name: str) -> None:
        self.crop_requested.emit(key, class_name)

    def _on_object_moved(self, key: ObjectKey, from_class: str, to_class: str) -> None:
        self.crop_requested.emit(key, to_class)

    def remove_class(self, class_name: str) -> None:
        """Remove a class box and all its objects."""
        box = self._class_boxes.pop(class_name, None)
        if box is not None:
            self._boxes_layout.removeWidget(box)
            box.deleteLater()
            self._class_order.remove(class_name)

    def set_visible_classes(self, class_names: list[str]) -> None:
        """Set which class boxes are visible."""
        for name, box in self._class_boxes.items():
            box.setVisible(name in class_names)

    def get_class_box(self, class_name: str) -> _ClassBox | None:
        return self._class_boxes.get(class_name)

    def get_all_class_names(self) -> list[str]:
        return list(self._class_order)

    def get_annotations(self) -> dict[str, list[ObjectKey]]:
        """Get all annotations: class_name -> list of ObjectKeys."""
        result = {}
        for name, box in self._class_boxes.items():
            keys = box.get_object_keys()
            if keys:
                result[name] = keys
        return result

    def clear_all(self) -> None:
        """Remove all class boxes."""
        for name in list(self._class_order):
            self.remove_class(name)
        self._class_counter = 0
