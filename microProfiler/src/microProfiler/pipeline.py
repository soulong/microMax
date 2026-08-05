from __future__ import annotations

import logging
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
    db_name: str = "result.db",
    progress: ProgressCollector = NullProgressCollector(),
):
    db_path = root_dir / db_name
    intensity_cols = ds.intensity_colnames

    image_profiling = cfg.image_profile
    object_profiling = cfg.object_profile

    if image_profiling and image_profiling.run and image_profiling.image_channels is not None:
        from microProfiler.profiling.image_profiler import profile_images

        n_workers = image_profiling.n_workers
        channels = image_profiling.image_channels or intensity_cols
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


_STEP_FUNCTIONS = {
    "resize": _run_resize,
    "basic": _run_basic,
    "zproject": _run_zproject,
    "tile": _run_tile,
    "segment": _run_segment,
    "profile": _run_profile,
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
    db_name: str = "result.db",
    log_file: Optional[Path] = None,
    progress: ProgressCollector = NullProgressCollector(),
    ds=None,
):
    logger = setup_logging(log_file=log_file, clear_existing=False)
    logger.info("Running step: %s", step_name)
    root_dir = dataset_dir
    logger.debug("Step '%s': root_dir=%s, db_name=%s", step_name, root_dir, db_name)

    fn = _STEP_FUNCTIONS.get(step_name)
    if fn is None:
        raise ValueError(
            f"Unknown step: {step_name!r}. Must be one of {list(_STEP_FUNCTIONS)}"
        )

    _PREPROC_STEPS = {"resize", "basic", "zproject", "tile"}
    if step_name in _PREPROC_STEPS:
        sf = SessionFile(dataset_dir)
        prev_applied = set(sf.get_applied_steps())
        if step_name in prev_applied:
            logger.info("Skipping %s — already applied in previous run", step_name)
            if ds is None:
                ds = _build_dataset(cfg, root_dir)
            return ds

    if ds is None:
        ds = _build_dataset(cfg, root_dir)
    if step_name == "profile":
        fn(cfg, ds, root_dir, db_name, progress)
        return ds
    if step_name == "segment":
        return _run_segment(cfg, ds, root_dir, progress)
    return fn(cfg, ds, root_dir, progress)


def run_pipeline(
    cfg: PipelineConfig,
    dataset_dir: Path,
    db_name: str = "result.db",
    log_file: Optional[Path] = None,
    progress: ProgressCollector = NullProgressCollector(),
    ds=None,
) -> tuple[ImageDataset | None, list[str]]:
    logger = setup_logging(log_file=log_file, clear_existing=False)
    logger.info("Pipeline start — dataset: %s", dataset_dir)

    root_dir = dataset_dir
    logger.debug("Output dir: %s, DB: %s", root_dir, db_name)
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

    if "basic" in prev_applied:
        logger.info("Skipping BaSiC — already applied in previous run")
    else:
        ds_new = _run_basic(cfg, ds_new, root_dir, progress)
        if cfg.basic and cfg.basic.run:
            logger.info("BaSiC step done")
            applied_steps.append("basic")

    if "zproject" in prev_applied:
        logger.info("Skipping Z-projection — already applied in previous run")
    else:
        ds_new = _run_zproject(cfg, ds_new, root_dir, progress)
        if cfg.zproject and cfg.zproject.run:
            logger.info("Z-projection step done")
            applied_steps.append("zproject")

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

    ds_new = _run_profile(cfg, ds_new, root_dir, db_name, progress)
    if cfg.image_profile and cfg.image_profile.run:
        applied_steps.append("image_profile")
    if cfg.object_profile and cfg.object_profile.run:
        applied_steps.append("object_profile")

    applied_steps = sorted(set(prev_applied) | set(applied_steps))

    return ds_new, applied_steps
