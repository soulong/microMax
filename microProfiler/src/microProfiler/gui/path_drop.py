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


def _local_paths(mime_data):
    """All local filesystem paths in a drag's mime data (may be empty)."""
    return [u.toLocalFile() for u in mime_data.urls() if u.isLocalFile()]


class _DropFilter(QObject):
    """Event filter that turns file/folder drops into a path assignment."""

    def __init__(self, on_path=None, multi=False):
        super().__init__()
        self._on_path = on_path
        self._multi = multi

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.DragEnter, QEvent.DragMove):
            if _local_paths(event.mimeData()):
                event.acceptProposedAction()
                return True
            return False
        if event.type() == QEvent.Drop:
            paths = _local_paths(event.mimeData())
            if not paths:
                return False
            # multi=True joins several dropped files with ';' (the path-list
            # convention of multi-file selectors); otherwise the first wins.
            text = ";".join(paths) if (self._multi and len(paths) > 1) else paths[0]
            if isinstance(obj, QLineEdit):
                obj.setText(text)
            elif isinstance(obj, QComboBox) and obj.isEditable():
                obj.setCurrentText(text)
            if self._on_path is not None:
                self._on_path(text)
            event.acceptProposedAction()
            return True
        return False


def enable_path_drop(widget, on_path=None, multi=False):
    """Make `widget` accept dragged files/directories.

    widget: a QLineEdit / editable QComboBox gets the dropped path as its
        text; any other widget only triggers `on_path`.
    on_path: optional callback invoked with the dropped path(s) AFTER the
        text assignment (e.g. to run the same flow as the Browse button).
    multi: True = several dropped files are joined with ';' (for
        multi-path inputs); False = only the first file is used.
    """
    filt = _DropFilter(on_path, multi)
    widget.setAcceptDrops(True)
    widget.installEventFilter(filt)
    # Keep a reference on the widget so the filter is not garbage-collected.
    widget._drop_filter = filt
    return widget
