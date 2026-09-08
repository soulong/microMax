"""Drag-and-drop path support for file-selection widgets.

`enable_path_drop` makes any widget accept a dragged file or folder: the
dropped path is pushed into the widget (QLineEdit.setText / editable
QComboBox.setCurrentText) and, optionally, forwarded to a callback. Applied
to every file/folder selector in the GUI so paths can be typed, browsed, OR
dragged in.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QComboBox, QLineEdit


def _first_local_path(mime_data):
    """First local filesystem path in a drag's mime data (None if absent)."""
    for url in mime_data.urls():
        if url.isLocalFile():
            return url.toLocalFile()
    return None


class _DropFilter(QObject):
    """Event filter that turns file/folder drops into a path assignment."""

    def __init__(self, on_path=None):
        super().__init__()
        self._on_path = on_path

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.DragEnter, QEvent.DragMove):
            if _first_local_path(event.mimeData()) is not None:
                event.acceptProposedAction()
                return True
            return False
        if event.type() == QEvent.Drop:
            path = _first_local_path(event.mimeData())
            if path is None:
                return False
            if isinstance(obj, QLineEdit):
                obj.setText(path)
            elif isinstance(obj, QComboBox) and obj.isEditable():
                obj.setCurrentText(path)
            if self._on_path is not None:
                self._on_path(path)
            event.acceptProposedAction()
            return True
        return False


def enable_path_drop(widget, on_path=None):
    """Make `widget` accept dragged files/directories.

    widget: a QLineEdit / editable QComboBox gets the dropped path as its
        text; any other widget only triggers `on_path`.
    on_path: optional callback invoked with the dropped path AFTER the text
        assignment (e.g. to run the same flow as the Browse button).
    """
    filt = _DropFilter(on_path)
    widget.setAcceptDrops(True)
    widget.installEventFilter(filt)
    # Keep a reference on the widget so the filter is not garbage-collected.
    widget._drop_filter = filt
    return widget
