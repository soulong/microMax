from __future__ import annotations

import inspect
import logging
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from microBase import ImageDataset
from microProfiler.io import read_image

PreviewResult = Dict[str, object]

logger = logging.getLogger(__name__)


class PreviewWorker(QObject):

    preview_ready = Signal(object)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_event = threading.Event()
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._execute)
        self._dataset: Optional[ImageDataset] = None
        self._row_idx: int = 0

    def preview_basic(self, ds: ImageDataset, row_idx: int, channels: List[str], root_dir: Optional[Path] = None) -> bool:
        if self._thread.isRunning():
            return False
        self._cancel_event.clear()
        self._dataset = ds
        self._row_idx = row_idx
        self._op = "basic"
        self._params = {"channels": channels, "root_dir": root_dir}
        self._thread.start()
        return True

    def preview_segment(self, ds: ImageDataset, row_idx: int, seg_params: dict) -> bool:
        if self._thread.isRunning():
            return False
        self._cancel_event.clear()
        self._dataset = ds
        self._row_idx = row_idx
        self._op = "segment"
        self._params = seg_params
        self._thread.start()
        return True

    def _execute(self) -> None:
        try:
            if self._dataset is None or self._row_idx >= len(self._dataset):
                self.error.emit("Invalid dataset or row index")
                return

            row = self._dataset.metadata.iloc[self._row_idx]

            # Always collect before images. Channel columns hold absolute
            # file paths (microBase convention) — use them directly.
            before_channels: List[Tuple[str, np.ndarray]] = []
            for ch in self._dataset.intensity_colnames:
                if self._cancel_event.is_set():
                    raise InterruptedError
                before_channels.append((ch, read_image(row[ch])))

            after_channels: List[Tuple[str, np.ndarray]] = []
            extra: dict = {}

            if self._op == "basic":
                channels = self._params.get("channels", [])
                root_dir = self._params.get("root_dir")
                if root_dir is not None:
                    model_root = Path(root_dir)
                else:
                    model_root = self._dataset.root
                model_dir = model_root / ".microprofiler" / "BaSiC_model"
                any_model_found = False
                flatfield_data: Dict[str, np.ndarray] = {}
                for ch, img in before_channels:
                    if self._cancel_event.is_set():
                        raise InterruptedError
                    if ch not in channels:
                        after_channels.append((ch, img))
                        continue
                    model_path = model_dir / f"{ch}.pkl"
                    if model_path.exists():
                        any_model_found = True
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)
                        ff = model.flatfield.astype(np.float32)
                        df = model.darkfield.astype(np.float32) if hasattr(model, "darkfield") and model.darkfield is not None else 0.0
                        corrected = (img.astype(np.float32) - df) / ff
                        after_channels.append((ch, corrected))
                        flatfield_data[ch] = ff
                    else:
                        after_channels.append((ch, img))
                extra["flatfield"] = flatfield_data
                if channels and not any_model_found:
                    self.error.emit("No BaSiC model found for any channel. Fit the model first using the 'Fit Model' button.")
                    self._thread.quit()
                    return

            elif self._op == "segment":
                if self._cancel_event.is_set():
                    raise InterruptedError
                from microProfiler.segmentation.cellpose import segment_single, _get_device
                from cellpose import models as cp_models
                model_name = self._params.get("model_name", "cpdino")
                if not hasattr(self, "_cellpose_model") or self._cellpose_model_name != model_name:
                    device = _get_device()
                    self._cellpose_model = cp_models.CellposeModel(device=device, pretrained_model=model_name)
                    self._cellpose_model_name = model_name
                valid = set(inspect.signature(segment_single).parameters) - {"row", "model"}
                c1_img, c2_img, mask = segment_single(
                    row, model=self._cellpose_model, **{k: v for k, v in self._params.items() if k in valid}
                )
                extra["c1_img"] = c1_img
                extra["c2_img"] = c2_img
                extra["mask"] = mask

            result: PreviewResult = {
                "before": before_channels,
                "after": after_channels,
                "extra": extra,
                "row_idx": self._row_idx,
            }
            self.preview_ready.emit(result)

        except InterruptedError:
            pass
        except Exception as e:
            # Includes the microBase MicroMaxError subclasses (missing /
            # deleted files) — without this branch on_preview_error never
            # fires, the wait cursor stays and _preview_running is never
            # reset.
            self.error.emit(str(e))
        finally:
            self._thread.quit()

    def cancel(self) -> None:
        """Cooperate with the running preview: set the flag and wait.

        The preview checks the flag between steps and raises InterruptedError
        so the worker finishes cleanly. The thread is NEVER terminated while
        it executes Python code.
        """
        self._cancel_event.set()
        if self._thread.isRunning():
            self._thread.quit()
            if not self._thread.wait(10000):
                logger.warning("Preview worker did not stop within 10s after cancel")
