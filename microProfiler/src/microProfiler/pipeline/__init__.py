from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from microBase import ImageDataset, SessionFile

from microProfiler.config import PipelineConfig
from microProfiler.log_utils import setup_logging
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector
from microProfiler.pipeline.steps import (
    PREPROC_STEPS,
    STEP_ORDER,
    _PROFILE_STEPS,
    _STEP_FUNCTIONS,
    _build_dataset,
    _is_fit_only_basic,
    _step_will_execute,
)

logger = logging.getLogger(__name__)


def apply_filters(ds, filters) -> None:
    """Apply FilterEntry list to a dataset, skipping rows with an empty
    column or pattern.

    The GUI never applies a filter row with an empty pattern (step_filter
    requires both column and pattern), but it does serialize such rows to
    session.yml — an empty regex would match everything. Skipping them here
    keeps CLI and GUI behaviour identical.
    """
    for f in filters:
        if not f.column or not f.pattern:
            logger.info("Skipping empty filter (column=%r, pattern=%r)", f.column, f.pattern)
            continue
        ds.filter_metadata(f.column, f.pattern)


class MetadataValidationError(ValueError):
    """Raised when an enabled step requires a metadata column that is absent."""


def run_step(
    cfg: PipelineConfig,
    step_name: str,
    root_dir: Path,
    result_db: str = "result.db",
    log_file: Optional[Path] = None,
    progress: ProgressCollector = NullProgressCollector(),
    ds=None,
):
    """Run a single pipeline step by its canonical section name.

    In-place preprocessing steps (resize/zproject/basic/tile) are gated by
    applied_steps — except fit-only BaSiC, which never re-triggers on
    already-processed files and is a fresh intent (GUI "Fit Model").
    """
    setup_logging(log_file=log_file, clear_existing=False)
    logger.info("Running step: %s", step_name)

    fn = _STEP_FUNCTIONS.get(step_name)
    if fn is None:
        raise ValueError(
            f"Unknown step: {step_name!r}. Must be one of {list(_STEP_FUNCTIONS)}"
        )

    if step_name in PREPROC_STEPS:
        prev_applied = set(SessionFile(root_dir).get_applied_steps())
        if step_name in prev_applied and not _is_fit_only_basic(cfg):
            logger.info("Skipping %s — already applied in previous run", step_name)
            if ds is None:
                ds = _build_dataset(cfg, root_dir)
            return ds

    if ds is None:
        ds = _build_dataset(cfg, root_dir)
    kwargs = {"result_db": result_db} if step_name in _PROFILE_STEPS else {}
    return fn(cfg, ds, root_dir, progress, **kwargs)


def run_pipeline(
    cfg: PipelineConfig,
    root_dir: Path,
    result_db: str = "result.db",
    log_file: Optional[Path] = None,
    progress: ProgressCollector = NullProgressCollector(),
    ds=None,
) -> tuple[ImageDataset | None, list[str]]:
    """Run every enabled section in STEP_ORDER with one unified gate.

    In-place preprocessing steps are skipped when already recorded in
    applied_steps (except fit-only BaSiC — see _is_fit_only_basic); the
    non-destructive steps (segment/image_profile/object_profile/inference)
    always run when enabled. Only steps that actually executed are appended
    to applied_steps.
    """
    setup_logging(log_file=log_file, clear_existing=False)
    logger.info("Pipeline start — dataset: %s", root_dir)

    prev_applied = set(SessionFile(root_dir).get_applied_steps())
    if prev_applied:
        logger.info("Previously applied steps detected: %s", sorted(prev_applied))

    if ds is None:
        ds = _build_dataset(cfg, root_dir)
        logger.info("Dataset loaded: %d rows, channels=%s", len(ds), ds.intensity_colnames)

    if cfg.filter:
        apply_filters(ds, cfg.filter)
        logger.info("Filters applied: %s", [(f.column, f.pattern) for f in cfg.filter])

    applied_steps = []
    for step_name in STEP_ORDER:
        fit_only = _is_fit_only_basic(cfg) if step_name == "basic" else False
        if step_name in PREPROC_STEPS and step_name in prev_applied and not fit_only:
            logger.info("Skipping %s — already applied in previous run", step_name)
            continue
        kwargs = {"result_db": result_db} if step_name in _PROFILE_STEPS else {}
        ds = _STEP_FUNCTIONS[step_name](cfg, ds, root_dir, progress, **kwargs)
        if _step_will_execute(cfg, step_name) and not fit_only:
            logger.info("%s step done", step_name)
            applied_steps.append(step_name)

    applied_steps = sorted(set(prev_applied) | set(applied_steps))

    return ds, applied_steps