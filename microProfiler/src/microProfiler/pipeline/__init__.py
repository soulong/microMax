from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from microBase import ImageDataset, SessionFile
from microBase.db_contracts import PROFILER_DB_NAME

from microProfiler.config import PipelineConfig
from microProfiler.log_utils import set_log_file, setup_logging
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


def run_step(
    cfg: PipelineConfig,
    step_name: str,
    root_dir: Path,
    result_db: str = PROFILER_DB_NAME,
    log_file: Optional[Path] = None,
    progress: ProgressCollector = NullProgressCollector(),
    ds=None,
) -> tuple[ImageDataset | None, list[str]]:
    """Run a single pipeline step by its canonical section name.

    In-place preprocessing steps (resize/zproject/basic/tile) are gated by
    applied_steps — except fit-only BaSiC, which never re-triggers on
    already-processed files and is a fresh intent (GUI "Fit Model").

    Returns:
        (ds, applied_steps): the dataset and the FULL applied_steps set after
        this run (previously recorded steps unioned with the step this run
        actually recorded). Callers persist exactly this list, so a fit-only
        BaSiC can never leak into applied_steps via the caller and gate a
        later fit-transform run out.
    """
    setup_logging(log_file=log_file, clear_existing=False)
    # The terminal log is mirrored next to the sources of every dataset run.
    set_log_file(Path(root_dir) / "microProfiler.log")
    logger.info("Running step: %s", step_name)

    fn = _STEP_FUNCTIONS.get(step_name)
    if fn is None:
        raise ValueError(
            f"Unknown step: {step_name!r}. Must be one of {list(_STEP_FUNCTIONS)}"
        )

    prev_applied = set(SessionFile(root_dir).get_applied_steps())
    if step_name in PREPROC_STEPS:
        if step_name in prev_applied and not (
            step_name == "basic" and _is_fit_only_basic(cfg)
        ):
            logger.info("Skipping %s — already applied in previous run", step_name)
            if ds is None:
                ds = _build_dataset(cfg, root_dir)
            return ds, sorted(prev_applied)

    if ds is None:
        ds = _build_dataset(cfg, root_dir)

    # Mirror run_pipeline: a programmatic run_step with a filter section must
    # profile the same rows as the equivalent single-section batch run.
    if cfg.filter:
        apply_filters(ds, cfg.filter)
        logger.info("Filters applied: %s", [(f.column, f.pattern) for f in cfg.filter])

    kwargs = {"result_db": result_db} if step_name in _PROFILE_STEPS else {}
    ds = fn(cfg, ds, root_dir, progress, **kwargs)
    if _step_will_execute(cfg, step_name, root_dir, ds=ds) and not (
        step_name == "basic" and _is_fit_only_basic(cfg)
    ):
        _persist_applied(root_dir, [step_name])
        prev_applied.add(step_name)
    return ds, sorted(prev_applied)


def _persist_applied(root_dir: Path, executed: list[str]) -> None:
    """Union executed steps into session.yml's applied_steps (never shrinks).

    Called immediately after each step completes inside run_pipeline/run_step,
    so a later failure or a GUI cancel still records the in-place steps that
    already ran — a destructive re-run after a partial failure is impossible.
    """
    sf = SessionFile(root_dir)
    applied = sorted(set(sf.get_applied_steps()) | set(executed))
    sf.set_applied_steps(applied)


def run_pipeline(
    cfg: PipelineConfig,
    root_dir: Path,
    result_db: str = PROFILER_DB_NAME,
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
    # The terminal log is mirrored next to the sources of every dataset run.
    set_log_file(Path(root_dir) / "microProfiler.log")
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
            # Only report the skip when THIS run requested the step: a
            # single-panel run's restricted config leaves the other sections
            # disabled, and logging their applied-gate skips is misleading.
            # Run All / Run Preprocessing enable their sections, so there the
            # message still explains why an enabled step is not re-run.
            section = getattr(cfg, step_name, None)
            if section is not None and section.run:
                logger.info("Skipping %s — already applied in previous run", step_name)
            continue
        kwargs = {"result_db": result_db} if step_name in _PROFILE_STEPS else {}
        ds = _STEP_FUNCTIONS[step_name](cfg, ds, root_dir, progress, **kwargs)
        if _step_will_execute(cfg, step_name, root_dir, ds=ds) and not fit_only:
            logger.info("%s step done", step_name)
            applied_steps.append(step_name)
            # Persist per completed step so a later failure or cancel keeps
            # the already-executed in-place steps recorded in session.yml.
            _persist_applied(root_dir, [step_name])

    applied_steps = sorted(set(prev_applied) | set(applied_steps))

    return ds, applied_steps