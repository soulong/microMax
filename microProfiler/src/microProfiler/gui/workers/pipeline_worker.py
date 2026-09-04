from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from microProfiler.config import PipelineConfig
from microProfiler.pipeline import run_pipeline, run_step
from microProfiler.progress_collector import SubscribableProgressCollector


logger = logging.getLogger(__name__)


class PipelineWorker(QObject):

    progress = Signal(str, int, int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_event = threading.Event()
        self._step_name: str | None = None
        self._dataset_dir: Path | None = None
        self._thread: QThread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._execute)
        # NOTE: do NOT auto-connect thread.finished -> thread.deleteLater.
        # The worker is replaced on every run (PipelineController._ensure_worker
        # calls cancel()/wait() on the previous worker), and an auto-deleted
        # QThread leaves a dangling wrapper whose isRunning()/quit()/wait()
        # raise "libshiboken: Internal C++ object already deleted". Cleanup is
        # done explicitly in _ensure_worker when the worker is replaced.
        self._result_ds = None
        self._applied_steps = None

    def run(self, cfg: PipelineConfig, dataset_dir: Path, ds=None) -> None:
        self._cancel_event.clear()
        self._cfg = cfg
        self._step_name = None
        self._dataset_dir = dataset_dir
        self._ds = ds
        self._thread.start()

    def run_step(self, cfg: PipelineConfig, step_name: str, dataset_dir: Path, ds=None) -> None:
        self._cancel_event.clear()
        self._cfg = cfg
        self._step_name = step_name
        self._dataset_dir = dataset_dir
        self._ds = ds
        self._thread.start()

    def cancel(self) -> None:
        """Cooperate with the running pipeline: set the cancel flag and wait.

        The pipeline observes the flag at every progress report (the
        progress subscription in _execute is the cancellation checkpoint),
        raising InterruptedError, so the worker finishes cleanly. The thread
        is NEVER terminated while it executes Python code.
        """
        self._cancel_event.set()
        if self._thread.isRunning():
            self._thread.quit()
            if not self._thread.wait(10000):
                logger.warning("Pipeline worker did not stop within 10s after cancel")

    def _check_cancel(self) -> None:
        if self._cancel_event.is_set():
            raise InterruptedError

    def _on_progress(self, step: str, cur: int, tot: int, msg: str) -> None:
        self._check_cancel()
        self.progress.emit(step, cur, tot, msg)

    def _execute(self) -> None:
        collector = SubscribableProgressCollector()
        collector.subscribe(self._on_progress)
        collector.cancel_check = lambda: self._cancel_event.is_set()
        try:
            if self._step_name:
                self._result_ds = run_step(
                    self._cfg, self._step_name, self._dataset_dir, progress=collector, ds=self._ds,
                )
            else:
                result_ds, applied = run_pipeline(
                    self._cfg, root_dir=self._dataset_dir, progress=collector, ds=self._ds,
                )
                self._result_ds = result_ds
                self._applied_steps = applied
            self.finished.emit()
        except InterruptedError:
            self.finished.emit()
        except SystemExit as e:
            # microBase hard-exits (print + sys.exit) on bad dataset state
            # (missing image files, invalid filter column). Without this the
            # error signal never fires and the UI stays stuck with
            # set_running(True) — the modal dialog / wait cursor never clears.
            logger.error("Pipeline failed (microBase hard-exit): %s", e)
            self.error.emit(str(e) or "Pipeline failed (microBase hard-exit)")
        except Exception as e:
            logger.error("Pipeline error: %s\n%s", e, traceback.format_exc())
            self.error.emit(str(e))
        finally:
            self._thread.quit()
