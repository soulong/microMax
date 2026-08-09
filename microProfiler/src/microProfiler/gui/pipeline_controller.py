from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QMessageBox

from microBase import SessionFile
from microProfiler.config import PipelineConfig, section_to_dataclass
from microProfiler.gui.dataset_service import DatasetService
from microProfiler.gui.interfaces import IControllerView
from microProfiler.gui.workers.pipeline_worker import PipelineWorker
from microProfiler.io import clone_dataset

logger = logging.getLogger(__name__)

_STEP_MAPPING = {
    "resize": "resize",
    "basic": "basic",
    "zproject": "zproject",
    "tile": "tile",
    "segment": "segment",
    "image_profile": "image_profile",
    "object_profile": "object_profile",
    "inference": "inference",
}


class PipelineController(QObject):

    def __init__(self, view: IControllerView):
        super().__init__(parent=view.widget())
        self._view = view
        self._ds_service = DatasetService()
        self._pending_step = None
        self._preview_block_index: int | None = None
        self._preview_pending_step = None
        self._worker_gen = 0
        self._worker: Optional[PipelineWorker] = None
        self._preview_worker: Optional[PipelineWorker] = None
        self._preview_running: bool = False
        self._random_row_idx: int | None = None
        # Per-run finished handler + run_gen, dispatched by
        # _on_worker_finished. See _ensure_worker for why the finished signal
        # is connected to a bound method (not a lambda) of this main-thread
        # QObject: a bare lambda + QueuedConnection has no QObject context, so
        # PySide6 queues the event to the worker thread, which then quits and
        # discards it — the handler never runs and the UI stays stuck.
        self._pending_finished: Optional[tuple] = None

    def _output_path(self) -> Path:
        return self._view.output_path()

    def _build_base_config(self) -> PipelineConfig:
        return PipelineConfig()

    def _build_pipeline_config(self) -> PipelineConfig:
        cfg = self._build_base_config()
        for step in self._view.get_all_step_panels():
            section = step.to_config()
            if section is not None and step.is_enabled():
                attr = _STEP_MAPPING.get(step.step_name, step.step_name)
                setattr(cfg, attr, section_to_dataclass(attr, section))
        return cfg

    def _image_profile_channel_error(self) -> Optional[str]:
        """Error message when Image Profiling is enabled with no channel
        selected (channels are unchecked by default; image_channels=None
        would silently profile ALL channels). Returns None when OK."""
        img_panel = self._view.get_step_panel("image_profile")
        if img_panel is None or not img_panel.is_enabled():
            return None
        if not hasattr(img_panel, "build_config_section"):
            return None
        section = img_panel.build_config_section() or {}
        if section.get("image_channels"):
            return None
        return (
            "Select at least one channel for Image Profiling. "
            "All channels are unchecked by default."
        )

    def _collect_profiling_table_names(self, cfg) -> set:
        """Collect table names that microProfiler will create from the config."""
        names = set()
        if cfg.image_profile:
            names.add("image")
        if cfg.object_profile and getattr(cfg.object_profile, "configs", None):
            for entry in cfg.object_profile.configs:
                name = getattr(entry, "output_table_name", None) or entry.mask_name
                if name:
                    names.add(name)
        return names

    def _drop_profiling_tables(self, db_path: Path, table_names: set) -> None:
        """Drop only the specified tables from the database, preserving others."""
        if not table_names or not db_path.exists():
            return
        import sqlite3
        try:
            conn = sqlite3.connect(str(db_path))
            for name in table_names:
                conn.execute(f"DROP TABLE IF EXISTS [{name}]")
            conn.commit()
            conn.close()
            logging.getLogger("microProfiler").info(
                "Dropped profiling tables: %s", ", ".join(sorted(table_names)))
        except Exception:
            logging.getLogger("microProfiler").warning(
                "Failed to drop profiling tables", exc_info=True)

    def _build_step_config(self, step) -> PipelineConfig:
        cfg = self._build_base_config()
        section = step.to_config()
        if section and step.is_enabled():
            attr = _STEP_MAPPING.get(step.step_name, step.step_name)
            setattr(cfg, attr, section_to_dataclass(attr, section))
        return cfg

    # ── Worker management ────────────────────────────────────────────────

    def _ensure_worker(self) -> PipelineWorker:
        self._worker_gen += 1
        if self._worker is not None:
            # Disconnect BOTH signals: a stale queued finished/error from the
            # replaced run must not fire the success handlers or pop error
            # dialogs for a run the user already replaced (the per-run gen
            # guard on finished is the backstop; error has no gen).
            for sig in (self._worker.finished, self._worker.error):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            prev_thread = self._worker._thread
            self._worker.cancel()
            if not prev_thread.wait(5000):
                logger.warning(
                    "Replaced pipeline worker still running after cancel — "
                    "it may still write result.db concurrently")
            # PipelineWorker no longer auto-deletes its QThread, so release it
            # explicitly here together with the worker object.
            prev_thread.deleteLater()
            self._worker.deleteLater()
        worker = PipelineWorker()
        self._view.progress_connect_update(worker.progress)
        worker.error.connect(self.on_pipeline_error, Qt.ConnectionType.QueuedConnection)
        # Connect finished to a bound method of self (this controller is a
        # QObject living on the main thread). This gives the QueuedConnection a
        # QObject context, so the slot is reliably queued to the main thread.
        worker.finished.connect(self._on_worker_finished, Qt.ConnectionType.QueuedConnection)
        self._worker = worker
        return worker

    def _on_worker_finished(self) -> None:
        """Main-thread dispatcher for the worker's finished signal.

        Each run registers its per-run handler + run_gen in
        ``self._pending_finished``; this slot forwards to it. The per-run
        handler still re-checks ``run_gen != self._worker_gen`` so a stale
        queued event from a cancelled/replaced run is rejected.
        """
        pending = self._pending_finished
        self._pending_finished = None
        if pending is None:
            return
        handler, run_gen = pending
        handler(run_gen)

    def _cancel_current_worker(self) -> None:
        if self._worker is not None:
            try:
                self._worker.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._worker.cancel()
        self._pending_finished = None
        if self._preview_worker is not None:
            self._preview_worker.cancel()
        self._preview_running = False
        self._view.set_running(False)
        self._view.progress_reset()

    def _save_session_yml(self, executed_steps: Optional[list] = None) -> None:
        settings: dict = {}
        for step in self._view.get_all_step_panels():
            section = step.to_config()
            if section is not None:
                settings[step.step_name] = section
        settings["image_pattern"] = self._view.get_image_pattern()
        settings["mask_pattern"] = self._view.get_mask_pattern()
        settings["image_subdir_pattern"] = self._view.get_image_subdir_pattern()
        filter_panel = self._view.get_step_panel("filter")
        if filter_panel and hasattr(filter_panel, "to_config"):
            fc = filter_panel.to_config()
            if fc and "filters" in fc:
                # Always write the key (even []): SessionFile.save deep-merges,
                # so an absent key would leave stale GUI filters in session.yml
                # that resurrect on the next Browse.
                settings["filter"] = fc["filters"]
        dataset_dir = self._output_path()
        sf = SessionFile(dataset_dir)
        sf.save(settings)
        # applied_steps must never shrink: gated in-place preprocessing steps
        # (resize/basic/zproject/tile) stay marked once actually applied
        # (run_pipeline itself unions prev_applied), so unchecking a checkbox
        # later can't cause a destructive re-run on already-processed files.
        prev_applied = set(sf.get_applied_steps())
        if executed_steps is not None:
            applied = sorted(prev_applied | set(executed_steps))
        else:
            # Inference is non-destructive and never gates a re-run — it is
            # excluded from the checked fallback so its checkbox state can
            # never mark it "applied".
            checked = [
                s.step_name for s in self._view.get_all_step_panels()
                if s.isChecked() and s.step_name != "inference"
            ]
            applied = sorted(prev_applied | set(checked))
        sf.set_applied_steps(applied)

    # ── Pipeline run handlers ────────────────────────────────────────────

    def _missing_input(self) -> bool:
        if not self._view.get_input_dir():
            QMessageBox.warning(self._view.widget(), "Missing Input", "Please select an input directory.")
            return True
        return False

    def run_preprocessing(self) -> None:
        if self._view.running or self._missing_input():
            return
        if self._view.dataset is None:
            logging.getLogger("microProfiler").info("No dataset loaded - load a dataset.")
            return
        self._view.set_running(True)

        cfg = self._build_base_config()
        executed_steps = []
        for step in self._view.get_preprocessing_steps():
            section = step.to_config()
            if section and step.is_enabled():
                attr = _STEP_MAPPING.get(step.step_name, step.step_name)
                setattr(cfg, attr, section_to_dataclass(attr, section))
                executed_steps.append(step)

        if not executed_steps:
            QMessageBox.information(
                self._view.widget(), "No Steps To Run",
                "Check the checkbox on each step panel to enable it."
            )
            self._view.set_running(False)
            return

        self._ensure_worker()
        run_gen = self._worker_gen
        self._pending_finished = (self._on_preprocessing_finished, run_gen)
        self._view.progress_reset()
        self._worker.run(cfg, dataset_dir=self._output_path(), ds=self._view.dataset)

    def _on_preprocessing_finished(self, run_gen: int) -> None:
        # The run generation is bound at connect time, so a stale queued
        # finished event from a cancelled/previous run is always rejected
        # (a new run bumps _worker_gen).
        if run_gen != self._worker_gen:
            return
        self._view.progress_finished()
        self._view.set_running(False)
        self._update_dataset_after_step("preprocessing")
        logging.getLogger("microProfiler").info("Preprocessing complete.")
        self._save_session_yml(executed_steps=self._worker._applied_steps)

    def run_segmentation(self) -> None:
        if self._view.running or self._missing_input():
            return
        if self._view.dataset is None:
            logging.getLogger("microProfiler").info("No dataset loaded - load a dataset.")
            return
        seg_panel = self._view.get_step_panel("segment")
        if seg_panel is None or not hasattr(seg_panel, "validate_object_names"):
            return
        name_error = seg_panel.validate_object_names()
        if name_error:
            QMessageBox.warning(self._view.widget(), "Invalid Names", name_error)
            return
        if hasattr(seg_panel, "validate_channels"):
            chan_error = seg_panel.validate_channels()
            if chan_error:
                QMessageBox.warning(self._view.widget(), "No Channel Selected", chan_error)
                return

        self._view.set_running(True)
        cfg = self._build_step_config(seg_panel)
        self._ensure_worker()
        run_gen = self._worker_gen
        self._pending_finished = (self._on_segmentation_finished, run_gen)
        self._view.progress_reset()
        self._worker.run_step(cfg, "segment", dataset_dir=self._output_path(), ds=self._view.dataset)

    def _on_segmentation_finished(self, run_gen: int) -> None:
        if run_gen != self._worker_gen:
            return
        self._view.progress_finished()
        self._view.set_running(False)
        logging.getLogger("microProfiler").info("Segmentation complete.")
        self._save_session_yml(executed_steps=["segment"])
        self._update_dataset_after_step("segment")

    def run_profiling(self) -> None:
        if self._view.running or self._missing_input():
            return
        if not self._view.dataset:
            logging.getLogger("microProfiler").info("No dataset loaded - load a dataset.")
            return

        obj_panel = self._view.get_step_panel("object_profile")
        if obj_panel is not None and hasattr(obj_panel, "_blocks"):
            self._sync_seg_masks_to_profiling()
            for block in obj_panel._blocks:
                obj_mask = block.get_mask_name()
                parent_mask = block.get_parent_mask_name()
                if parent_mask and parent_mask == obj_mask:
                    QMessageBox.warning(
                        self._view.widget(), "Invalid Mask Selection",
                        f"Parent mask cannot be the same as the object mask "
                        f"('{obj_mask}'). "
                        "Please choose a different parent mask or set it to None."
                    )
                    return

        # Build config first to determine which tables to drop
        cfg = self._build_base_config()
        image_panel = self._view.get_step_panel("image_profile")
        obj_panel = self._view.get_step_panel("object_profile")

        channel_err = self._image_profile_channel_error()
        if channel_err:
            QMessageBox.warning(
                self._view.widget(), "No Channels Selected", channel_err)
            return

        if image_panel and image_panel.is_enabled():
            section = image_panel.to_config()
            if section:
                cfg.image_profile = section_to_dataclass("image_profile", section)
        if obj_panel and obj_panel.is_enabled():
            section = obj_panel.to_config()
            if section:
                cfg.object_profile = section_to_dataclass("object_profile", section)

        # Drop only profiling tables, preserving other tables (e.g. microVis's label tables)
        db_path = self._output_path() / "result.db"
        table_names = self._collect_profiling_table_names(cfg)
        if db_path.exists() and table_names:
            reply = QMessageBox.question(
                self._view.widget(), "Overwrite Profiling Tables",
                f"Results database already exists at:\n{db_path}\n\n"
                f"This will drop the following tables: {', '.join(sorted(table_names))}\n"
                "Other tables in the database will be preserved.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self._drop_profiling_tables(db_path, table_names)

        self._view.set_running(True)

        self._ensure_worker()
        run_gen = self._worker_gen
        self._pending_finished = (self._on_profiling_finished, run_gen)
        self._view.progress_reset()
        self._worker.run_step(cfg, "profile", dataset_dir=self._output_path(), ds=self._view.dataset)

    def _on_profiling_finished(self, run_gen: int) -> None:
        if run_gen != self._worker_gen:
            return
        self._view.progress_finished()
        self._view.set_running(False)
        logging.getLogger("microProfiler").info("Profiling complete.")
        # Persist only the profiling steps that were actually enabled/run
        # (mirrors run_profiling's config construction); a step that was never
        # run must not enter applied_steps.
        executed = []
        image_panel = self._view.get_step_panel("image_profile")
        if image_panel is not None and image_panel.is_enabled():
            executed.append("image_profile")
        obj_panel = self._view.get_step_panel("object_profile")
        if obj_panel is not None and obj_panel.is_enabled():
            executed.append("object_profile")
        self._save_session_yml(executed_steps=executed)

    def run_inference(self) -> None:
        if self._view.running or self._missing_input():
            return
        if not self._view.dataset:
            logging.getLogger("microProfiler").info("No dataset loaded - load a dataset.")
            return
        panel = self._view.get_step_panel("inference")
        if panel is None:
            return
        if not panel.is_enabled():
            QMessageBox.information(
                self._view.widget(), "No Steps To Run",
                "Check the checkbox on the Inference panel to enable it."
            )
            return
        err = panel.validate_blocks()
        if err:
            QMessageBox.warning(
                self._view.widget(), "Invalid Inference Configuration", err)
            return

        self._view.set_running(True)
        cfg = self._build_step_config(panel)
        self._ensure_worker()
        run_gen = self._worker_gen
        self._pending_finished = (self._on_inference_finished, run_gen)
        self._view.progress_reset()
        self._worker.run_step(cfg, "infer", dataset_dir=self._output_path(), ds=self._view.dataset)

    def _on_inference_finished(self, run_gen: int) -> None:
        if run_gen != self._worker_gen:
            return
        self._view.progress_finished()
        self._view.set_running(False)
        logging.getLogger("microProfiler").info("Inference complete.")
        # Inference is non-destructive and never gates a re-run, but a
        # COMPLETED inference is recorded in applied_steps (entry 'infer')
        # like the other steps. The worker emits finished on cancel too, so
        # only record infer when the run was not cancelled.
        cancelled = bool(self._worker._cancel_event.is_set())
        self._save_session_yml(executed_steps=[] if cancelled else ["infer"])

    def run_all(self) -> None:
        if self._view.running or self._missing_input():
            return
        if self._view.dataset is None:
            QMessageBox.warning(
                self._view.widget(), "Conversion Required",
                "Please load a dataset before running the full pipeline."
            )
            return
        self._sync_seg_masks_to_profiling()

        inf_panel = self._view.get_step_panel("inference")
        if inf_panel is not None and inf_panel.is_enabled():
            err = inf_panel.validate_blocks()
            if err:
                QMessageBox.warning(
                    self._view.widget(), "Invalid Inference Configuration", err)
                return

        channel_err = self._image_profile_channel_error()
        if channel_err:
            QMessageBox.warning(
                self._view.widget(), "No Channels Selected", channel_err)
            return

        cfg = self._build_pipeline_config()
        has_steps = any(
            getattr(cfg, attr)
            for attr in ("resize", "basic", "zproject", "tile",
                         "segment", "image_profile", "object_profile",
                         "inference")
        )
        if not has_steps:
            QMessageBox.information(
                self._view.widget(), "Nothing to Run",
                "No enabled steps to run.",
            )
            return

        # Drop only profiling tables, preserving other tables (e.g. microVis's label tables)
        db_path = self._output_path() / "result.db"
        table_names = self._collect_profiling_table_names(cfg)
        if db_path.exists() and table_names:
            reply = QMessageBox.question(
                self._view.widget(), "Overwrite Profiling Tables",
                f"Results database already exists at:\n{db_path}\n\n"
                f"This will drop the following tables: {', '.join(sorted(table_names))}\n"
                "Other tables in the database will be preserved.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self._drop_profiling_tables(db_path, table_names)

        self._view.set_running(True)
        self._ensure_worker()
        run_gen = self._worker_gen
        self._pending_finished = (self._on_pipeline_finished, run_gen)
        self._view.progress_reset()
        self._worker.run(cfg, dataset_dir=self._output_path(), ds=self._view.dataset)

    def _on_pipeline_finished(self, run_gen: int):
        if run_gen != self._worker_gen:
            return
        self._view.progress_finished()
        self._view.set_running(False)
        logging.getLogger("microProfiler").info("Pipeline complete - all steps finished.")
        self._save_session_yml(executed_steps=self._worker._applied_steps)
        self._update_dataset_after_step("pipeline")

    def apply_step(self, step) -> None:
        if self._view.running:
            return
        if self._view.dataset is None:
            logging.getLogger("microProfiler").info("No dataset loaded - load a dataset.")
            return
        self._view.set_running(True)

        cfg = self._build_step_config(step)
        if step.step_name == "basic" and hasattr(cfg, "basic") and cfg.basic is not None:
            root = self._output_path()
            model_dir = root / ".microprofiler" / "BaSiC_model"
            if model_dir.exists() and any(model_dir.glob("*.pkl")):
                # Only "fit-transform" is shortened to transform-only when
                # pre-fitted models exist; an explicit "fit" request is
                # honored as-is (fit-only).
                if cfg.basic.mode == "fit-transform":
                    cfg.basic.mode = "transform"
                    logging.getLogger("microProfiler").info(
                        "Pre-fitted BaSiC models detected — running transform only"
                    )

        self._ensure_worker()
        run_gen = self._worker_gen
        self._pending_step = step
        self._pending_finished = (self._on_step_finished, run_gen)
        self._view.progress_reset()
        self._worker.run_step(cfg, step.step_name, dataset_dir=self._output_path(), ds=self._view.dataset)

    def _on_step_finished(self, run_gen: int) -> None:
        if run_gen != self._worker_gen:
            return
        step = self._pending_step
        self._view.progress_finished()
        self._view.set_running(False)
        step_name = step.step_name
        logging.getLogger("microProfiler").info(f"{step_name} complete.")
        self._save_session_yml(executed_steps=[step_name])
        self._update_dataset_after_step(step_name)

    # ── BaSiC fit ────────────────────────────────────────────────────────

    def fit_basic(self) -> None:
        if self._view.running:
            return
        if self._view.dataset is None:
            logging.getLogger("microProfiler").info("No dataset loaded - load a dataset.")
            return
        self._view.set_running(True)
        cfg = self._build_base_config()
        basic_panel = self._view.get_step_panel("basic")
        if basic_panel is not None and hasattr(basic_panel, "to_config"):
            section = basic_panel.to_config()
            if section:
                section = {**section, "mode": "fit"}
                cfg.basic = section_to_dataclass("basic", section)
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._ensure_worker()
            run_gen = self._worker_gen
            self._pending_finished = (self._on_fit_finished, run_gen)
            self._view.progress_reset()
            self._worker.run_step(cfg, "basic", dataset_dir=self._output_path(), ds=self._view.dataset)
        except Exception:
            QApplication.restoreOverrideCursor()
            self._view.set_running(False)
            raise

    def _on_fit_finished(self, run_gen: int):
        if run_gen != self._worker_gen:
            return
        self._view.progress_finished()
        self._view.set_running(False)
        logging.getLogger("microProfiler").info("BaSiC model fit complete.")
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()

    # ── Preview ──────────────────────────────────────────────────────────

    def _ensure_random_row(self) -> bool:
        if self._random_row_idx is None:
            logging.getLogger("microProfiler").info("Click 'Pick Random' first to select an image.")
            return False
        return True

    def pick_random(self, step=None) -> None:
        ds = self._view.dataset
        if ds is not None and len(ds) > 0:
            self._random_row_idx = self._ds_service.pick_random_row(ds)
            idx = self._random_row_idx
            logging.getLogger("microProfiler").info(f"Picked random row {idx}")
            for panel in self._view.get_all_step_panels():
                if hasattr(panel, "clear_preview") and callable(panel.clear_preview):
                    panel.clear_preview()
            if step and step.step_name == "basic" and hasattr(step, "_channel_tiles"):
                for ch in ds.intensity_colnames:
                    try:
                        img = self._ds_service.load_image(ds, idx, ch)
                    except FileNotFoundError:
                        logging.getLogger("microProfiler").warning(
                            f"Row {idx} missing channel file '{ch}' — skipping tile"
                        )
                        continue
                    step._channel_tiles[ch][0].set_image(img)

    def on_segment_pick(self, block_index: int) -> None:
        ds = self._view.dataset
        if ds is None or len(ds) == 0:
            return
        self._random_row_idx = self._ds_service.pick_random_row(ds)
        idx = self._random_row_idx
        logging.getLogger("microProfiler").info(
            f"Segment block {block_index}: picked random row {idx}"
        )
        seg_panel = self._view.get_step_panel("segment")
        if seg_panel is None or not hasattr(seg_panel, "_blocks"):
            return
        blocks = seg_panel._blocks
        if block_index >= len(blocks):
            return
        block = blocks[block_index]
        chan1 = block.get_chan1() or ds.intensity_colnames[:1]
        chan2 = block.get_chan2() or None
        if chan1:
            ch = chan1[0]
            try:
                c1_img = self._ds_service.load_image(ds, idx, ch)
                seg_panel.set_preview_c1(block_index, c1_img)
            except Exception:
                pass
        if chan2:
            ch = chan2[0]
            try:
                c2_img = self._ds_service.load_image(ds, idx, ch)
                seg_panel.set_preview_c2(block_index, c2_img)
            except Exception:
                pass

    def on_segment_preview(self, block_index: int) -> None:
        if self._preview_running:
            return
        ds = self._view.dataset
        if not self._ensure_random_row() or ds is None:
            return
        seg_panel = self._view.get_step_panel("segment")
        if seg_panel is None or not hasattr(seg_panel, "_blocks"):
            return
        blocks = seg_panel._blocks
        if block_index >= len(blocks):
            return
        block = blocks[block_index]
        idx = self._random_row_idx
        seg_params = block.build_config_section()
        seg_params.pop("object_name", None)
        self._preview_block_index = block_index
        self._preview_pending_step = seg_panel
        self._preview_running = True
        self._view.progress_show_status("Segmenting preview...")
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self._preview_worker is not None:
                self._preview_worker.preview_segment(ds, idx, seg_params)
        except Exception:
            QApplication.restoreOverrideCursor()
            self._preview_running = False
            raise

    def preview_step(self, step) -> None:
        if self._preview_running:
            return
        ds = self._view.dataset
        if not self._ensure_random_row() or ds is None:
            return
        idx = self._random_row_idx
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._preview_pending_step = step
            self._preview_block_index = None
            self._preview_running = True
            if step.step_name == "basic" and self._preview_worker is not None:
                chans = ds.intensity_colnames
                self._preview_worker.preview_basic(ds, idx, chans, root_dir=self._output_path())
        except Exception:
            QApplication.restoreOverrideCursor()
            self._preview_running = False
            raise

    def on_preview_ready(self, result) -> None:
        self._preview_running = False
        self._view.progress_reset()
        step = self._preview_pending_step
        if step is None:
            from PySide6.QtWidgets import QApplication
            QApplication.restoreOverrideCursor()
            return
        extra = result.get("extra", {})
        after = result.get("after", [])
        before = result.get("before", [])
        if step.step_name == "basic":
            step.update_preview_raw(dict(before))
            step.update_preview_corrected(dict(after))
            flatfield = extra.get("flatfield", {})
            if flatfield:
                step.update_preview_flatfield(flatfield)
        elif step.step_name == "segment":
            block_idx = self._preview_block_index
            if block_idx is None:
                return
            seg_panel = self._view.get_step_panel("segment")
            if seg_panel is not None:
                c1 = extra.get("c1_img")
                c2 = extra.get("c2_img")
                mask = extra.get("mask")
                if c1 is not None:
                    seg_panel.set_preview_c1(block_idx, c1)
                if c2 is not None:
                    seg_panel.set_preview_c2(block_idx, c2)
                if mask is not None:
                    seg_panel.set_preview_mask(block_idx, mask)
        self._preview_pending_step = None
        self._preview_block_index = None
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()

    def on_preview_error(self, message: str) -> None:
        self._preview_running = False
        self._view.progress_show_error(f"Preview failed: {message}")
        step_name = (
            self._preview_pending_step.step_name
            if self._preview_pending_step else "unknown"
        )
        logging.getLogger("microProfiler").error(f"Preview error ({step_name}): {message}")
        self._preview_pending_step = None
        self._preview_block_index = None
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _update_dataset_after_step(self, step_name: str) -> None:
        updated_ds = getattr(self._worker, "_result_ds", None)
        if updated_ds is None:
            logging.getLogger("microProfiler").warning(
                "No result dataset after step '%s'", step_name
            )
            return
        logger = logging.getLogger("microProfiler")
        logger.info("Updating dataset after %s: %d rows, %d channels",
                 step_name, len(updated_ds), len(updated_ds.intensity_colnames))
        self._view.dataset = updated_ds
        self._view.original_dataset = clone_dataset(updated_ds)
        filter_panel = self._view.get_step_panel("filter")
        if filter_panel is not None and hasattr(filter_panel, "_apply_filters"):
            filter_panel._apply_filters()
        self._view.update_dataset_info(updated_ds)
        seg_panel = self._view.get_step_panel("segment")
        if seg_panel is not None and hasattr(seg_panel, "populate_channels"):
            seg_panel.populate_channels(updated_ds.intensity_colnames)
        img_panel = self._view.get_step_panel("image_profile")
        if img_panel is not None and hasattr(img_panel, "populate_channels"):
            img_panel.populate_channels(updated_ds.intensity_colnames)
        obj_panel = self._view.get_step_panel("object_profile")
        if obj_panel is not None:
            if hasattr(obj_panel, "populate_channels"):
                obj_panel.populate_channels(updated_ds.intensity_colnames)
            if hasattr(obj_panel, "populate_masks"):
                self._sync_seg_masks_to_profiling()
        inf_panel = self._view.get_step_panel("inference")
        if inf_panel is not None:
            if hasattr(inf_panel, "populate_channels"):
                inf_panel.populate_channels(updated_ds.intensity_colnames)
            if hasattr(inf_panel, "populate_masks"):
                self._sync_seg_masks_to_profiling()
        if hasattr(updated_ds, '_image_pattern'):
            pat = updated_ds._image_pattern
            pat_str = pat.pattern if hasattr(pat, 'pattern') else str(pat)
            self._view.set_image_pattern(pat_str)
        if hasattr(updated_ds, '_mask_pattern'):
            pat = updated_ds._mask_pattern
            pat_str = pat.pattern if hasattr(pat, 'pattern') else str(pat)
            self._view.set_mask_pattern(pat_str)
        self._view.update_tab_status()

    def _sync_seg_masks_to_profiling(self) -> None:
        # Deterministic, deduped mask order: segment object names first
        # (so a fresh object-profile/inference block auto-selects the
        # segmentation's Object name, e.g. 'cell'), then the remaining
        # dataset mask columns.
        ordered = []
        seg_panel = self._view.get_step_panel("segment")
        seg_block_names = []
        if seg_panel is not None and hasattr(seg_panel, "get_object_names"):
            seg_block_names = seg_panel.get_object_names() or []
        for n in seg_block_names:
            prefixed = "mask_" + n
            if prefixed not in ordered:
                ordered.append(prefixed)
        if self._view.dataset is not None:
            for m in self._view.dataset.mask_colnames:
                if m not in ordered:
                    ordered.append(m)
        if not ordered:
            return
        obj_panel = self._view.get_step_panel("object_profile")
        if obj_panel is not None and hasattr(obj_panel, "populate_masks"):
            obj_panel.populate_masks(list(ordered))
        inf_panel = self._view.get_step_panel("inference")
        if inf_panel is not None and hasattr(inf_panel, "populate_masks"):
            inf_panel.populate_masks(list(ordered))

    def on_pipeline_error(self, message: str) -> None:
        self._view.progress_show_error(message)
        self._view.set_running(False)
        QMessageBox.critical(self._view.widget(), "Pipeline Error", message)
