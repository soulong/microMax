"""Background worker that constructs an ImageDataset off the UI thread.

ImageDataset.__init__ scans files and builds metadata — synchronous and
slow for large datasets. Moving it to a background thread keeps the modal
progress dialog responsive (no "not responding" ghost window).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from microBase import ImageDataset
from microProfiler.io import clone_dataset

logger = logging.getLogger("microProfiler")


class DatasetLoadWorker(QObject):
    """Constructs an ImageDataset (and a lightweight clone) off the UI thread."""

    finished = Signal(object, object)  # dataset, clone
    error = Signal(str)

    def __init__(self, root: Path, image_pattern, mask_pattern,
                 image_subdir_pattern, parent=None):
        super().__init__(parent)
        self._root = root
        self._image_pattern = image_pattern
        self._mask_pattern = mask_pattern
        self._image_subdir_pattern = image_subdir_pattern
        self._thread: Optional[QThread] = None

    def start(self) -> None:
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.start()

    def _run(self) -> None:
        try:
            ds = ImageDataset(
                self._root,
                image_pattern=self._image_pattern,
                mask_pattern=self._mask_pattern,
                image_subdir_pattern=self._image_subdir_pattern,
            )
            clone = clone_dataset(ds)
            self.finished.emit(ds, clone)
        except Exception as e:
            logger.exception("Dataset load failed")
            self.error.emit(str(e))
        finally:
            if self._thread is not None:
                self._thread.quit()
                self._thread.finished.connect(self._thread.deleteLater)
