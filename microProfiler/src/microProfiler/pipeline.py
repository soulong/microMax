from __future__ import annotations

import io
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from microBase import ImageDataset, SessionFile

from microProfiler.config import PipelineConfig
from microProfiler.logging_utils import setup_logging
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


class MetadataValidationError(ValueError):
    """Raised when an enabled step requires a metadata column that is absent."""


def _run_preprocessing_step(
    ds,
    step_name: str,
    cfg_step,
    step_fn,
    step_kwargs: dict,
    progress: ProgressCollector = NullProgressCollector(),
):
    if not cfg_step or not cfg_step.run:
        return ds

    logger.debug("%s: %s", step_name, step_kwargs)
    progress.step_start(step_name, f"Applying {step_name}...")
    ds = step_fn(ds, progress=progress, **step_kwargs)
    progress.step_end(step_name, f"{step_name} complete")
    return ds


def _run_resize(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    from microProfiler.preprocessing.resizer import resize_dataset

    return _run_preprocessing_step(
        ds, "Resize", cfg.resize, resize_dataset,
        {"scale_factor": cfg.resize.scale_factor} if cfg.resize else {},
        progress,
    )


def _run_basic(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    from microProfiler.preprocessing.basic_correction import apply_basic

    kwargs = {}
    if cfg.basic and cfg.basic.run:
        kwargs = {
            "mode": cfg.basic.mode,
            "n_image": cfg.basic.n_image,
            "working_size": cfg.basic.working_size,
            "enable_darkfield": cfg.basic.enable_darkfield,
            "root_dir": root_dir,
        }
    return _run_preprocessing_step(
        ds, "BaSiC", cfg.basic, apply_basic,
        kwargs, progress,
    )


def _run_zproject(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    if cfg.zproject and cfg.zproject.run and "stack" not in ds.metadata.columns:
        raise MetadataValidationError(
            "Z-projection is enabled but the dataset has no 'stack' column. "
            "Either disable zproject or use an image_pattern with a (?P<stack>...) group."
        )

    from microProfiler.preprocessing.z_projection import z_project_dataset

    kwargs = {}
    if cfg.zproject and cfg.zproject.run:
        kwargs = {"method": cfg.zproject.method, "delete_original": True}
    return _run_preprocessing_step(
        ds, "Z-projection", cfg.zproject, z_project_dataset,
        kwargs, progress,
    )


def _run_tile(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    if cfg.tile and cfg.tile.run and "field" not in ds.metadata.columns:
        raise MetadataValidationError(
            "Tiling is enabled but the dataset has no 'field' column. "
            "Either disable tile or use an image_pattern with a (?P<field>...) group."
        )

    from microProfiler.preprocessing.tile_splitter import tile_dataset

    kwargs = {}
    if cfg.tile and cfg.tile.run:
        kwargs = {"tile_w": cfg.tile.tile_width, "tile_h": cfg.tile.tile_height, "delete_original": True}
    return _run_preprocessing_step(
        ds, "Tile", cfg.tile, tile_dataset,
        kwargs, progress,
    )


def _run_segment(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    if not cfg.segment or not cfg.segment.run or not cfg.segment.configs:
        return ds

    from microProfiler.segmentation.cellpose import segment_dataset

    for entry in cfg.segment.configs:
        name = entry.object_name
        if not name:
            continue
        logger.debug(
            "Segment: object=%s, model=%s, chan1=%s, chan2=%s, diameter=%s, resize_factor=%s",
            name, entry.model_name, entry.chan1, entry.chan2, entry.diameter, entry.resize_factor,
        )
        progress.step_start(f"Segment ({name})", f"Starting segmentation ({name})...")
        ds = segment_dataset(
            ds,
            object_name=name,
            chan1=entry.chan1 or ds.intensity_colnames[:1],
            chan2=entry.chan2,
            merge1=entry.merge1,
            merge2=entry.merge2,
            model_name=entry.model_name,
            diameter=entry.diameter,
            resize_factor=entry.resize_factor,
            flow_threshold=entry.flow_threshold,
            cellprob_threshold=entry.cellprob_threshold,
            gpu_batch_size=entry.gpu_batch_size,
            overwrite_mask=entry.overwrite_mask,
            progress=progress,
        )
        progress.step_end(f"Segment ({name})", f"Segmentation ({name}) complete")
    return ds


def _run_profile(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    result_db: str = "result.db",
    progress: ProgressCollector = NullProgressCollector(),
):
    db_path = root_dir / result_db
    intensity_cols = ds.intensity_colnames

    image_profiling = cfg.image_profile
    object_profiling = cfg.object_profile

    # image_channels null/absent/[] => the step is skipped entirely (the
    # GUI blocks Run with no selection; the CLI treats no selection as "skip").
    if image_profiling and image_profiling.run and image_profiling.image_channels:
        from microProfiler.profiling.image_profiler import profile_images

        n_workers = image_profiling.n_workers
        channels = image_profiling.image_channels
        img_kwargs = {"db_path": db_path, "table_name": "image", "progress": progress}
        if image_profiling.image_thresholds:
            img_kwargs["thresholds"] = image_profiling.image_thresholds
        progress.step_start("Profile image", "Starting...")
        profile_images(ds, channels=channels, n_workers=n_workers, **img_kwargs)
        progress.step_end("Profile image", "Done")

    if object_profiling and object_profiling.run and object_profiling.configs:
        n_workers = object_profiling.n_workers
        from microProfiler.profiling.object_profiler import profile_objects

        for entry in object_profiling.configs:
            mask_name = entry.mask_name
            if not mask_name:
                continue
            table_name = entry.output_table_name or mask_name
            # overwrite_db: if True, drop existing table before profiling;
            # if False, BatchWriter replaces on first write anyway
            if db_path.exists() and entry.overwrite_db:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                try:
                    conn.execute(f"DROP TABLE IF EXISTS [{table_name}]")
                    conn.commit()
                    logger.info("Dropped table '%s' (overwrite_db=True)", table_name)
                finally:
                    conn.close()
            progress.step_start(f"Profile {mask_name}", "Starting...")
            profile_objects(
                ds,
                mask_name=mask_name,
                intensity_channels=entry.intensity_channels or intensity_cols,
                db_path=db_path,
                table_name=table_name,
                progress=progress,
                n_workers=n_workers,
                obj_config=entry,
            )
            progress.step_end(f"Profile {mask_name}", "Done")

    return ds


def _ds_pattern(ds, attr: str) -> Optional[str]:
    """Read a pattern off a loaded ImageDataset (compiled regex or str)."""
    pat = getattr(ds, attr, None)
    if pat is None:
        return None
    return pat.pattern if hasattr(pat, "pattern") else str(pat)


class _ProgressTee(io.TextIOBase):
    """Tee microModel's stderr to three places at once:

    - the real terminal (tqdm renders exactly like `micromodel infer`),
    - the error buffer (kept for the failure message on SystemExit),
    - the progress collector: tqdm lines (``Infer:  33%|██ 1/3 [..]``) become
      ``report(step_key, cur, tot)`` (a real status-bar bar), any other text
      line becomes ``report(step_key, 0, 0, text)`` (a status message).
    """

    _TQDM_RE = re.compile(r"^[^:\s]+:\s*\S")

    def __init__(self, err_buf, real_stderr, progress, step_key):
        super().__init__()
        self._buf = err_buf
        self._real = real_stderr
        self._progress = progress
        self._step_key = step_key
        self._pending = ""

    def _emit(self, seg: str) -> None:
        seg = re.sub(r"\x1b\[[0-9;]*m", "", seg).strip()
        if not seg:
            return
        if self._TQDM_RE.match(seg):
            m = re.search(r"(\d+)\s*/\s*(\d+)", seg)
            if m:
                self._progress.report(
                    self._step_key, int(m.group(1)), int(m.group(2)), "")
                return
        self._progress.report(self._step_key, 0, 0, seg[:200])

    def write(self, s: str) -> int:
        if self._real is not None:
            try:
                self._real.write(s)
                self._real.flush()
            except Exception:
                pass
        self._buf.write(s)
        self._pending += s
        parts = self._pending.replace("\r", "\n").split("\n")
        self._pending = parts.pop()
        for part in parts:
            self._emit(part)
        return len(s)

    def flush(self) -> None:
        if self._real is not None:
            try:
                self._real.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return bool(self._real is not None and self._real.isatty())

    def fileno(self):
        if self._real is not None and hasattr(self._real, "fileno"):
            return self._real.fileno()
        raise io.UnsupportedOperation("fileno")


class _MicroModelLogForwarder(logging.Handler):
    """Forward microModel logger records (e.g. "Fitting PCA + UMAP...",
    "Writing to <db>") to the real terminal with the CLI's format and to the
    progress collector as status messages — so the reduction stage is visible
    both in the terminal and in the GUI status bar.
    """

    def __init__(self, real_stderr, progress, step_key):
        super().__init__(level=logging.INFO)
        self._real = real_stderr
        self._progress = progress
        self._step_key = step_key
        self.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s",
            datefmt="%H:%M",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + "\n"
            if self._real is not None:
                self._real.write(line)
                self._real.flush()
            msg = record.getMessage().strip()
            if msg:
                self._progress.report(self._step_key, 0, 0, msg[:200])
        except Exception:
            self.handleError(record)


def _call_micromodel(fn, mm_cfg, err_prefix: str, progress=None,
                     step_key: str = "Inference", **kwargs):
    """Call a microModel function, converting its print + sys.exit(1) error
    paths (SystemExit) into RuntimeError with the captured stderr message, so
    the GUI worker surfaces a popup and the CLI treats the dataset as failed.

    When a progress collector is given, microModel's stderr is teed to the
    real terminal AND forwarded to the collector (live tqdm progress + status
    lines); microModel's INFO logs are forwarded the same way. When the run
    was cancelled (progress.cancel_check), a SystemExit is re-raised as
    InterruptedError — a cancel is not a failure.
    """
    import contextlib

    real_stderr = sys.stderr
    err_buf = io.StringIO()
    log_handler = None
    if progress is not None:
        tee = _ProgressTee(err_buf, real_stderr, progress, step_key)
        log_handler = _MicroModelLogForwarder(real_stderr, progress, step_key)
        mm_logger = logging.getLogger("microModel")
        mm_logger.setLevel(logging.INFO)
        mm_logger.addHandler(log_handler)
    else:
        tee = err_buf
    try:
        with contextlib.redirect_stderr(tee):
            fn(mm_cfg, **kwargs)
    except SystemExit as e:
        if (log_handler is not None
                and getattr(progress, "cancel_check", None)
                and progress.cancel_check()):
            raise InterruptedError from e
        msg = err_buf.getvalue().strip() or f"microModel failed ({e})"
        raise RuntimeError(f"{err_prefix}: {msg}") from e
    finally:
        if log_handler is not None:
            logging.getLogger("microModel").removeHandler(log_handler)


def _build_mm_inference_config(entry, cfg: PipelineConfig, ds, root_dir: Path) -> dict:
    """Build the microModel whole-image inference config dict for one block.

    output_dir is always null so the inference DB (and, with reduction, the
    fitted reducer pickles) land under the dataset dir itself.
    """
    intensity_cols = ds.intensity_colnames
    if entry.channels is None:
        channels = None
    else:
        channels = []
        for ch in entry.channels:
            try:
                channels.append(intensity_cols.index(ch) + 1)
            except ValueError:
                raise RuntimeError(
                    f"Inference channel {ch!r} not found in dataset channels "
                    f"{intensity_cols}."
                )
    mm_cfg = {
        "mode": "whole_image",
        "model": os.path.abspath(entry.model),
        "output_dir": None,
        # GUI inference uses 4 DataLoader worker processes (spawned thanks to
        # the picklable ImageDataset cache — §4.3) with the same prefetch as
        # microModel's CLI. persistent_workers=False tears the workers down
        # after each inference pass (clean teardown on completion/cancel);
        # tqdm/log output still comes from the pipeline worker thread and is
        # teed to the terminal + GUI status bar.
        "dataloader": {"num_workers": 4, "prefetch_factor": 2,
                       "persistent_workers": False},
        "data": {
            "root": [str(root_dir)],
            "channels": channels,
            "channel_layout": None,
            "image_pattern": cfg.image_pattern or _ds_pattern(ds, "_image_pattern"),
            "mask_pattern": cfg.mask_pattern or _ds_pattern(ds, "_mask_pattern"),
            "image_subdir_pattern": cfg.image_subdir_pattern or _ds_pattern(ds, "_image_subdir_pattern"),
            "mask_name": entry.mask_name,
            "max_value": float(entry.max_value),
            "label_from_dir": False,
            "label_csv": None,
            "sample_max": None,
            "sample_by": "per_dataset",
        },
        "inference": {
            "pred_class": bool(entry.pred_class),
            "feature": bool(entry.feature),
            "db_name": entry.output_db or "infer.db",
            "batch_size": 128,
        },
    }
    if entry.reduction and entry.reduction.enabled:
        mm_cfg["reduction"] = {
            "var_threshold": entry.reduction.var_threshold or 0.95,
            "color_by": entry.reduction.color_by or "pred_class",
            "sample_per_class": entry.reduction.sample_per_class or 10000,
            "reducer_pca": entry.reduction.reducer_pca,
            "reducer_umap": entry.reduction.reducer_umap,
        }
    return mm_cfg


def _run_inference(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    """Per-object inference with trained microModel bundles (whole-image).

    Each enabled block runs run_inference (+ optional plot-less PCA/UMAP
    reduction) writing <dataset>/<output_db>. microModel is imported lazily —
    a missing install raises RuntimeError (GUI popup; CLI hard-exits on the
    batch-level pre-check in cli.main before any dataset is touched).
    """
    if not cfg.inference or not cfg.inference.run or not cfg.inference.configs:
        return ds

    from microModel.infer import run_inference
    from microModel.vis import show_reduction

    for entry in cfg.inference.configs:
        if not entry.model:
            continue
        if not entry.max_value:
            raise RuntimeError(
                "Inference config error: 'max_value' is required for every "
                "inference block (e.g. 65535 for 16-bit, 255 for 8-bit images)."
            )
        model_path = os.path.abspath(entry.model)
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model bundle not found: {model_path}")

        mm_cfg = _build_mm_inference_config(entry, cfg, ds, root_dir)
        db_name = entry.output_db or "infer.db"
        label = f"{entry.mask_name or 'objects'} -> {db_name}"
        progress.step_start(
            f"Infer ({label})", f"Inferring objects with {os.path.basename(entry.model)} ({label})...")
        _call_micromodel(
            run_inference, mm_cfg,
            f"Inference failed for model {entry.model}",
            progress=progress, step_key=f"Infer ({label})")
        progress.step_end(f"Infer ({label})", f"Inference complete ({label})")
        if "reduction" in mm_cfg:
            progress.step_start(
                f"Reduction ({label})", f"Fitting PCA + UMAP ({label})...")
            _call_micromodel(
                show_reduction, mm_cfg,
                f"Reduction failed for {db_name}",
                save_plots=False, raise_on_error=True,
                progress=progress, step_key=f"Reduction ({label})")
            progress.step_end(f"Reduction ({label})", f"Reduction complete ({label})")
    return ds


_STEP_FUNCTIONS = {
    "resize": _run_resize,
    "zproject": _run_zproject,
    "basic": _run_basic,
    "tile": _run_tile,
    "segment": _run_segment,
    "profile": _run_profile,
    "infer": _run_inference,
}


def _build_dataset(cfg: PipelineConfig, root_dir: Path) -> ImageDataset:
    """Construct an ImageDataset from pipeline config patterns."""
    return ImageDataset(
        root=root_dir,
        image_pattern=cfg.image_pattern,
        mask_pattern=cfg.mask_pattern,
        image_subdir_pattern=cfg.image_subdir_pattern,
    )


def run_step(
    cfg: PipelineConfig,
    step_name: str,
    dataset_dir: Path,
    result_db: str = "result.db",
    log_file: Optional[Path] = None,
    progress: ProgressCollector = NullProgressCollector(),
    ds=None,
):
    logger = setup_logging(log_file=log_file, clear_existing=False)
    logger.info("Running step: %s", step_name)
    root_dir = dataset_dir
    logger.debug("Step '%s': root_dir=%s, db=%s", step_name, root_dir, result_db)

    fn = _STEP_FUNCTIONS.get(step_name)
    if fn is None:
        raise ValueError(
            f"Unknown step: {step_name!r}. Must be one of {list(_STEP_FUNCTIONS)}"
        )

    _PREPROC_STEPS = {"resize", "zproject", "basic", "tile"}
    if step_name in _PREPROC_STEPS:
        sf = SessionFile(dataset_dir)
        prev_applied = set(sf.get_applied_steps())
        # A fit-only BaSiC run writes shading models without touching the
        # images — it is a new intent (the GUI "Fit Model" button), so it is
        # never gated by applied_steps.
        is_fit_only = (
            step_name == "basic"
            and cfg.basic is not None
            and cfg.basic.run
            and cfg.basic.mode == "fit"
        )
        if step_name in prev_applied and not is_fit_only:
            logger.info("Skipping %s — already applied in previous run", step_name)
            if ds is None:
                ds = _build_dataset(cfg, root_dir)
            return ds

    if ds is None:
        ds = _build_dataset(cfg, root_dir)
    if step_name == "profile":
        fn(cfg, ds, root_dir, result_db, progress)
        return ds
    if step_name == "segment":
        return _run_segment(cfg, ds, root_dir, progress)
    return fn(cfg, ds, root_dir, progress)


def run_pipeline(
    cfg: PipelineConfig,
    dataset_dir: Path,
    result_db: str = "result.db",
    log_file: Optional[Path] = None,
    progress: ProgressCollector = NullProgressCollector(),
    ds=None,
) -> tuple[ImageDataset | None, list[str]]:
    logger = setup_logging(log_file=log_file, clear_existing=False)
    logger.info("Pipeline start — dataset: %s", dataset_dir)

    root_dir = dataset_dir
    logger.debug("Output dir: %s, DB: %s", root_dir, result_db)
    applied_steps = []

    sf = SessionFile(dataset_dir)
    prev_applied = set(sf.get_applied_steps())
    if prev_applied:
        logger.info("Previously applied steps detected: %s", sorted(prev_applied))

    if ds is None:
        ds = _build_dataset(cfg, root_dir)
        logger.info("Dataset loaded: %d rows, channels=%s", len(ds), ds.intensity_colnames)

    if cfg.filter:
        for f in cfg.filter:
            ds.filter_metadata(f.column, f.pattern)
        logger.info("Filters applied: %s", [(f.column, f.pattern) for f in cfg.filter])

    ds_new = ds

    if "resize" in prev_applied:
        logger.info("Skipping resize — already applied in previous run")
    else:
        ds_new = _run_resize(cfg, ds_new, root_dir, progress)
        if cfg.resize and cfg.resize.run:
            logger.info("Resize step done")
            applied_steps.append("resize")

    if "zproject" in prev_applied:
        logger.info("Skipping Z-projection — already applied in previous run")
    else:
        ds_new = _run_zproject(cfg, ds_new, root_dir, progress)
        if cfg.zproject and cfg.zproject.run:
            logger.info("Z-projection step done")
            applied_steps.append("zproject")

    if "basic" in prev_applied:
        logger.info("Skipping BaSiC — already applied in previous run")
    else:
        ds_new = _run_basic(cfg, ds_new, root_dir, progress)
        if cfg.basic and cfg.basic.run:
            logger.info("BaSiC step done")
            applied_steps.append("basic")

    if "tile" in prev_applied:
        logger.info("Skipping tiling — already applied in previous run")
    else:
        ds_new = _run_tile(cfg, ds_new, root_dir, progress)
        if cfg.tile and cfg.tile.run:
            logger.info("Tiling step done")
            applied_steps.append("tile")

    ds_new = _run_segment(cfg, ds_new, root_dir, progress)
    if cfg.segment and cfg.segment.run:
        logger.info("Segmentation step done")
        applied_steps.append("segment")

    ds_new = _run_profile(cfg, ds_new, root_dir, result_db, progress)
    if cfg.image_profile and cfg.image_profile.run:
        applied_steps.append("image_profile")
    if cfg.object_profile and cfg.object_profile.run:
        applied_steps.append("object_profile")

    # Inference is non-destructive and re-runnable — it is NOT gated by
    # applied_steps (like segment/profile), but a completed inference IS
    # recorded in applied_steps (entry 'infer') so the session reflects the
    # run, consistent with the other steps. A cancelled/aborted run never
    # reaches this point (InterruptedError propagates).
    ds_new = _run_inference(cfg, ds_new, root_dir, progress)
    if cfg.inference and cfg.inference.run and any(e.model for e in cfg.inference.configs):
        applied_steps.append("infer")

    applied_steps = sorted(set(prev_applied) | set(applied_steps))

    return ds_new, applied_steps
