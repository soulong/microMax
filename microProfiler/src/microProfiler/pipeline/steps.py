"""Pipeline step runners — one thin adapter per config section.

The step keys ARE the config section names (``resize``, ``zproject``,
``basic``, ``tile``, ``segment``, ``image_profile``, ``object_profile``,
``inference``) and match the ``applied_steps`` entries, so a step that ran
via ``run_pipeline`` can be re-run via ``run_step`` and vice versa.
``STEP_ORDER`` is derived from ``config.SECTION_ATTRS``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from microBase import ImageDataset
from microBase.db_contracts import IMAGE_TABLE, PROFILER_DB_NAME

from microProfiler.config import (
    SECTION_ATTRS,
    PipelineConfig,
    resolve_inference_db,
)
from microProfiler.io import Database, rebuild_dataset
from microProfiler.pipeline.errors import MetadataValidationError
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)

# Canonical step order — derived from config.SECTION_ATTRS (the single
# source of truth) so the two lists can never drift apart.
STEP_ORDER = list(SECTION_ATTRS)

# In-place preprocessing steps: gated by applied_steps (never re-run on
# already-processed files). segment/profile/inference are non-destructive
# and always re-runnable.
PREPROC_STEPS = frozenset({"resize", "zproject", "basic", "tile"})

# Steps that need the profiling result_db path.
_PROFILE_STEPS = frozenset({"image_profile", "object_profile"})


def _run_preprocessing_step(
    ds,
    step_key: str,
    cfg_step,
    step_fn,
    step_kwargs: dict,
    progress: ProgressCollector = NullProgressCollector(),
):
    if not cfg_step or not cfg_step.run:
        return ds

    logger.debug("%s: %s", step_key, step_kwargs)
    progress.step_start(step_key, f"Applying {step_key}...")
    ds = step_fn(ds, progress=progress, **step_kwargs)
    progress.step_end(step_key, f"{step_key} complete")
    return ds


def _run_resize(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    if not (cfg.resize and cfg.resize.run):
        return ds
    # Import after the gate: a disabled step must not pull in (or import-time
    # configure) the preprocessing dependencies.
    from microProfiler.preprocessing.resizer import resize_dataset

    return _run_preprocessing_step(
        ds, "resize", cfg.resize, resize_dataset,
        {"scale_factor": cfg.resize.scale_factor}, progress,
    )


def _run_basic(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    if not (cfg.basic and cfg.basic.run):
        return ds
    # Import after the gate — the vendored basic package imports JAX.
    from microProfiler.preprocessing.basic_correction import apply_basic
    from microBase import SessionFile

    # The fit-order marker must reflect reality: models fitted before any
    # z-projection ran were fit on raw stacks and must not advertise
    # "zproject_first" (the GUI's Apply-downgrade trusts that marker).
    zproject_applied = "zproject" in set(
        SessionFile(root_dir).get_applied_steps())

    return _run_preprocessing_step(
        ds, "basic", cfg.basic, apply_basic,
        {
            "mode": cfg.basic.mode,
            "n_image": cfg.basic.n_image,
            "working_size": cfg.basic.working_size,
            "enable_darkfield": cfg.basic.enable_darkfield,
            "root_dir": root_dir,
            "zproject_applied": zproject_applied,
        },
        progress,
    )


def _run_zproject(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    if not (cfg.zproject and cfg.zproject.run):
        return ds
    if "stack" not in ds.metadata.columns:
        raise MetadataValidationError(
            "Z-projection is enabled but the dataset has no 'stack' column. "
            "Either disable zproject or use an image_pattern with a (?P<stack>...) group."
        )
    # Import after the gate.
    from microProfiler.preprocessing.z_projection import z_project_dataset

    return _run_preprocessing_step(
        ds, "zproject", cfg.zproject, z_project_dataset,
        {"method": cfg.zproject.method, "delete_original": True}, progress,
    )


def _run_tile(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    if not (cfg.tile and cfg.tile.run):
        return ds
    if "field" not in ds.metadata.columns:
        raise MetadataValidationError(
            "Tiling is enabled but the dataset has no 'field' column. "
            "Either disable tile or use an image_pattern with a (?P<field>...) group."
        )
    # Import after the gate.
    from microProfiler.preprocessing.tile_splitter import tile_dataset

    return _run_preprocessing_step(
        ds, "tile", cfg.tile, tile_dataset,
        {
            "tile_width": cfg.tile.tile_width,
            "tile_height": cfg.tile.tile_height,
            "delete_original": True,
        },
        progress,
    )


def _run_segment(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
):
    section = cfg.segment
    if not section or not section.run or not section.configs:
        return ds

    from microProfiler.segmentation.cellpose import segment_dataset

    for entry in section.configs:
        if not entry.chan1:
            # Empty chan1 means "this entry is skipped" — never the
            # first-channel fallback.
            logger.info(
                "Segmentation block '%s' skipped — no chan1 channels configured",
                entry.object_name,
            )
            progress.report(
                f"segment ({entry.object_name})", 0, 0,
                "Skipped — no chan1 channels configured",
            )
            continue
        logger.debug(
            "Segment: object=%s, model=%s, chan1=%s, chan2=%s, diameter=%s, resize_factor=%s",
            entry.object_name, entry.model_name, entry.chan1, entry.chan2,
            entry.diameter, entry.resize_factor,
        )
        progress.step_start(
            f"segment ({entry.object_name})",
            f"Starting segmentation ({entry.object_name})...",
        )
        ds = segment_dataset(
            ds,
            object_name=entry.object_name,
            chan1=entry.chan1,
            chan2=entry.chan2,
            merge1=entry.merge1,
            merge2=entry.merge2,
            model_name=entry.model_name,
            diameter=entry.diameter,
            resize_factor=entry.resize_factor,
            flow_threshold=entry.flow_threshold,
            cellprob_threshold=entry.cellprob_threshold,
            gpu_batch_size=entry.gpu_batch_size,
            edge_pixel_ratio=entry.edge_pixel_ratio,
            overwrite_mask=entry.overwrite_mask,
            progress=progress,
        )
        progress.step_end(
            f"segment ({entry.object_name})",
            f"Segmentation ({entry.object_name}) complete",
        )
    return ds


def _run_image_profile(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
    result_db: str = PROFILER_DB_NAME,
):
    section = cfg.image_profile
    if not section or not section.run:
        return ds
    if not section.image_channels:
        # Empty image_channels means "this measurement is skipped" — never
        # the all-channels fallback.
        logger.info("Image profiling skipped — no image_channels configured")
        progress.report("image_profile", 0, 0, "Skipped — no image_channels configured")
        return ds

    from microProfiler.profiling.image_profiler import profile_images

    db_path = root_dir / result_db
    img_kwargs = {"db_path": db_path, "table_name": IMAGE_TABLE, "progress": progress}
    if section.image_thresholds:
        img_kwargs["thresholds"] = section.image_thresholds
    progress.step_start("image_profile", "Starting...")
    profile_images(
        ds, channels=section.image_channels, n_workers=section.n_workers, **img_kwargs,
    )
    progress.step_end("image_profile", "Done")
    # Quarantine deletes files mid-run; rescan so this step's returned
    # dataset (and later steps) never reference rows that are gone.
    return rebuild_dataset(ds)


def _run_object_profile(
    cfg: PipelineConfig,
    ds,
    root_dir: Path,
    progress: ProgressCollector = NullProgressCollector(),
    result_db: str = PROFILER_DB_NAME,
):
    section = cfg.object_profile
    if not section or not section.run or not section.configs:
        return ds

    from microProfiler.profiling.object_profiler import profile_objects

    db_path = root_dir / result_db
    for entry in section.configs:
        if not entry.intensity_channels:
            # Empty intensity_channels means "this block is skipped" (shape
            # features included) — never the all-channels fallback.
            logger.info(
                "Object profiling block '%s' skipped — no intensity_channels configured",
                entry.mask_name,
            )
            progress.report(
                f"object_profile ({entry.mask_name})", 0, 0,
                "Skipped — no intensity_channels configured",
            )
            continue
        table_name = entry.output_table_name or entry.mask_name
        # overwrite_db: if True, drop existing table before profiling;
        # if False, BatchWriter replaces on first write anyway.
        if db_path.exists() and entry.overwrite_db:
            db = Database(db_path)
            try:
                db.drop_table(table_name)
            finally:
                db.close()
        progress.step_start(
            f"object_profile ({entry.mask_name})", "Starting...",
        )
        profile_objects(
            ds,
            mask_name=entry.mask_name,
            intensity_channels=entry.intensity_channels,
            db_path=db_path,
            table_name=table_name,
            progress=progress,
            n_workers=section.n_workers,
            obj_config=entry,
        )
        # Bookkeep the table -> mask mapping (used by the per-mask merges).
        db = Database(db_path)
        try:
            db.record_table_mask(table_name, entry.mask_name)
        finally:
            db.close()
        progress.step_end(
            f"object_profile ({entry.mask_name})", "Done",
        )
    # Quarantine deletes files mid-run; rescan so this step's returned
    # dataset (and later steps) never reference rows that are gone.
    return rebuild_dataset(ds)


def _auto_merge_infer(root_dir: Path, entry, infer_db: Path) -> None:
    """Merge one inference block's results with the profiler objects of the
    SAME mask into ``merge_<mask>.db`` next to the dataset.

    The mask comes from the infer DB's `mask_name` column (written by
    microModel), falling back to the config entry; the profiler table is
    matched through the `_table_masks` bookkeeping (falling back to the
    table-name convention). Source DBs are never modified.
    """
    from microBase import db_merge

    infer_frames = db_merge.read_infer_frames(str(infer_db))
    # The infer DB itself names its mask (mask_name column, written by
    # microModel); fall back to the config entry.
    mask = (infer_frames[0].mask if infer_frames else None) or entry.mask_name
    if not mask:
        logger.warning("Auto-merge skipped: no mask name for %s", infer_db.name)
        return

    profiler_db = root_dir / PROFILER_DB_NAME
    obj_frames: list = []
    if profiler_db.exists():
        obj_frames = [f for f in db_merge.read_profiler_frames(str(profiler_db))
                      if f.mask == mask]
    if not obj_frames:
        logger.warning(
            "Auto-merge skipped: %s has no object table for mask '%s' "
            "(profile this mask first)", profiler_db.name, mask)
        return

    fused = db_merge.fuse_frames(obj_frames + infer_frames, default_mask=mask)
    out = root_dir / f"merge_{mask}.db"
    db_merge.write_merged_db(fused, out, mask=mask)
    logger.info("Auto-merge: %d rows x %d columns -> %s",
                len(fused), len(fused.columns), out.name)


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
    section = cfg.inference
    if not section or not section.run or not section.configs:
        return ds

    from microProfiler.pipeline._micromodel_bridge import (
        _build_mm_inference_config,
        _call_micromodel,
        run_mm_inference,
        run_mm_reduction,
    )
    from microProfiler.user_defaults import update_user_defaults

    ran_any = False
    for entry in section.configs:
        if not entry.channels:
            # null/[] channels means "this block is skipped" — never the
            # all-channels fallback (checked BEFORE max_value so a block left
            # intentionally empty does not abort the run).
            logger.info(
                "Inference block '%s' skipped — no channels configured",
                entry.model,
            )
            progress.report(
                f"inference ({entry.model})", 0, 0,
                "Skipped — no channels configured",
            )
            continue
        if entry.max_value is None:
            raise RuntimeError(
                "Inference config error: 'max_value' is required for every "
                "inference block (e.g. 65535 for 16-bit, 255 for 8-bit images)."
            )
        model_path = os.path.abspath(entry.model)
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model bundle not found: {model_path}")

        mm_cfg = _build_mm_inference_config(entry, cfg, ds, root_dir)
        db_name = resolve_inference_db(entry)
        label = f"{entry.mask_name or 'objects'} -> {db_name}"
        progress.step_start(
            f"inference ({label})",
            f"Inferring objects with {os.path.basename(entry.model)} ({label})...",
        )
        _call_micromodel(
            run_mm_inference, mm_cfg,
            f"Inference failed for model {entry.model}",
            progress=progress, step_key=f"inference ({label})",
        )
        progress.step_end(f"inference ({label})", f"Inference complete ({label})")
        if "reduction" in mm_cfg:
            progress.step_start(
                f"reduction ({label})", f"Fitting DR reduction ({label})...",
            )
            _call_micromodel(
                run_mm_reduction, mm_cfg,
                f"Reduction failed for {db_name}",
                save_plots=False, raise_on_error=True,
                progress=progress, step_key=f"reduction ({label})",
            )
            progress.step_end(f"reduction ({label})", f"Reduction complete ({label})")

        # Auto-merge this block's infer results with the profiler objects
        # of the same mask -> merge_<mask>.db (a merge problem must not
        # lose the just-written inference results, so only log failures).
        try:
            _auto_merge_infer(root_dir, entry, root_dir / db_name)
        except Exception:
            logger.warning("Auto-merge failed for mask '%s'",
                           entry.mask_name, exc_info=True)
        ran_any = True

    # Remember this run's inference choices at user level (~/.micromax):
    # the Inference panel pre-fills new runs with the last run's model /
    # reducer(s) / cluster.pkl. Only saved when at least one block actually
    # ran, and only from the FIRST block (the GUI's "+ Add" copies it). A
    # run with both groups unchecked carries no `reduction` key — the
    # remembered reducer/cluster paths stay untouched then.
    try:
        first = section.configs[0]
        updates = {"model": os.path.abspath(first.model) if first.model else None}
        if first.reduction is not None:
            updates["reducer"] = first.reduction.reducer
            updates["cluster"] = first.reduction.cluster
        if ran_any:
            update_user_defaults("inference", updates)
    except Exception:
        logger.warning("Failed to remember inference defaults in ~/.micromax",
                       exc_info=True)
    return ds


_STEP_FUNCTIONS = {
    "resize": _run_resize,
    "zproject": _run_zproject,
    "basic": _run_basic,
    "tile": _run_tile,
    "segment": _run_segment,
    "image_profile": _run_image_profile,
    "object_profile": _run_object_profile,
    "inference": _run_inference,
}


def _step_will_execute(cfg: PipelineConfig, step_name: str, root_dir=None, ds=None) -> bool:
    """True when the section is enabled AND at least one unit will run.

    A block whose channel list is empty is skipped at runtime (empty means
    "skip", never "all"), so it must not be recorded in applied_steps.
    Transform-only BaSiC with no fitted models likewise transforms nothing
    (every channel is skipped with a warning) and must not be recorded —
    recording it would permanently gate future fit-transform runs.
    """
    section = getattr(cfg, step_name, None)
    if not section or not section.run:
        return False
    if step_name == "resize":
        # scale_factor == 1.0 is a no-op — recording it would gate a later
        # real resize out of applied_steps.
        return float(getattr(section, "scale_factor", 1.0)) != 1.0
    if step_name == "tile":
        # A tile size larger than the image produces zero complete tiles;
        # sources are kept (see tile_splitter) but the step must not be
        # recorded as applied.
        if ds is not None and ds.img_shape:
            h, w = ds.img_shape
            return section.tile_width <= w and section.tile_height <= h
        return True
    if step_name == "image_profile":
        return bool(section.image_channels)
    if step_name == "object_profile":
        return any(e.intensity_channels for e in section.configs)
    if step_name == "segment":
        return any(e.chan1 for e in section.configs)
    if step_name == "inference":
        return any(e.channels for e in section.configs)
    if step_name == "basic":
        if getattr(section, "mode", None) != "transform":
            # "fit" is never recorded (fit-only intent); "fit-transform"
            # fits fresh models first, so the transform has work to do.
            return True
        model_dir = Path(root_dir) / ".microprofiler" / "BaSiC_model" if root_dir else None
        return bool(model_dir and any(model_dir.glob("*.pkl")))
    return True


def _is_fit_only_basic(cfg: PipelineConfig) -> bool:
    """True when the basic section is a fit-only invocation.

    A fit-only BaSiC run writes shading models without touching the images —
    it is a new intent (the GUI "Fit Model" button), so it is never gated by
    applied_steps and never recorded as "applied".
    """
    return (
        cfg.basic is not None
        and cfg.basic.run
        and cfg.basic.mode == "fit"
    )


def _build_dataset(cfg: PipelineConfig, root_dir: Path) -> ImageDataset:
    """Construct an ImageDataset from pipeline config patterns.

    microProfiler supports ONE channel per file only (Operetta-style layouts).
    channel_layout is intentionally not exposed: the multi-channel-per-file
    path is unsupported end to end and would need a config key + GUI control
    (see microBase.ImageDataset for the low-level support).
    """
    return ImageDataset(
        root=root_dir,
        image_pattern=cfg.image_pattern,
        mask_pattern=cfg.mask_pattern,
        image_subdir_pattern=cfg.image_subdir_pattern,
    )