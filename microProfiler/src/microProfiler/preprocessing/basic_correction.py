"""BaSiC shading correction — thin wrapper around the basic/ folder.

The actual BaSiC algorithm lives in ``microProfiler.preprocessing.basic``
(copied verbatim from the original image_profiler).
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from microBase import ImageDataset
from microProfiler.io import (
    ImageReadError,
    quarantine_row,
    read_image,
    read_image_shape,
    rebuild_dataset,
    write_image,
)

# Force JAX onto CPU for BaSiC fit/transform. Must run before importing the
# vendored basic package (which imports jax) so the GPU backend never
# initialises and no VRAM is preallocated. setdefault keeps an escape hatch:
# a user who exports JAX_PLATFORMS=gpu still gets GPU.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from microProfiler.preprocessing.basic.basic import BaSiC

from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def basic_fit(
    images: List[np.ndarray],
    enable_darkfield: bool = False,
    working_size: int = 64,
) -> BaSiC:
    """Fit BaSiC model on pre-read images.

    Reading and sampling happen in ``fit_models`` so a broken file can
    quarantine its metadata row before the fit — this function only ever sees
    arrays.
    """
    shapes = {img.shape for img in images}
    if len(shapes) > 1:
        raise ValueError(
            f"BaSiC fit requires uniform image shapes, got {len(shapes)} different shapes: {shapes}"
        )
    imgs = np.stack(images)
    basic = BaSiC(
        get_darkfield=enable_darkfield,
        smoothness_flatfield=1,
        smoothness_darkfield=1,
        working_size=working_size,
        max_workers=8,
    )
    basic.fit(imgs)
    return basic


def fit_models(
    ds: ImageDataset,
    channels: List[str] | None = None,
    n_image: int = 100,
    working_size: int = 64,
    enable_darkfield: bool = False,
    root_dir: Union[str, Path, None] = None,
    progress: ProgressCollector = NullProgressCollector(),
    zproject_applied: bool = True,
) -> Path:
    """Fit BaSiC models for specified channels.

    Defaults match BasicConfig.n_image (100) and the GUI's default. Missing
    or unreadable images quarantine their metadata row (all row files are
    deleted) instead of aborting the run.

    zproject_applied: whether the z-project step has actually been applied
    to this dataset's files. Only then may the models claim the
    "zproject_first" fit order — a fit on never-projected images must not
    advertise an order it was not fit under (the GUI's Apply-downgrade
    trusts that marker).
    """
    channels = channels or ds.intensity_colnames
    metadata = ds.metadata
    root = Path(root_dir) if root_dir else ds.root

    model_dir = root / ".microprofiler" / "BaSiC_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("BaSiC fitting channels")
    fitted_any = False
    for ci, chan in enumerate(channels):
        progress.report("basic", ci, len(channels), f"Fit: channel {chan}")
        # (row_idx, path) pairs keep the row context needed to quarantine a
        # broken file; missing files quarantine their row immediately.
        entries = []
        for i in range(len(metadata)):
            value = metadata.iloc[i][chan]
            if pd.isna(value):
                continue
            path = Path(value)
            if not path.exists():
                quarantine_row(ds, i, f"missing {path.name}")
                continue
            entries.append((i, path))
        if not entries:
            logger.warning("No existing files for channel %s, skipping", chan)
            continue

        # When subsampling, a fixed seed keeps the fit reproducible across
        # runs (the fitted flatfield/darkfield — and every downstream profile
        # value — would otherwise vary run to run).
        if len(entries) > n_image:
            rng = random.Random(42)
            entries = rng.sample(entries, k=n_image)

        images = []
        for row_idx, path in entries:
            try:
                images.append(read_image(path))
            except ImageReadError as e:
                quarantine_row(ds, row_idx, str(e))
        if not images:
            logger.warning("No readable files for channel %s, skipping", chan)
            continue

        fitted_any = True
        model = basic_fit(images, enable_darkfield, working_size)

        with open(model_dir / f"{chan}.pkl", "wb") as f:
            pickle.dump(model, f)

        write_image(
            model_dir / f"{chan}_flatfield.tiff",
            model.flatfield.astype(np.float32),
        )
        if enable_darkfield:
            write_image(
                model_dir / f"{chan}_darkfield.tiff",
                model.darkfield.astype(np.float32),
            )

    progress.report("basic", len(channels), len(channels), "Fit complete")
    # Only mark the fit-order when at least one model was actually fitted —
    # an all-skipped run must not advertise fresh models.
    if fitted_any:
        # Mark that these models were fit under the zproject-first
        # preprocessing order, so the GUI "Apply" downgrade path can refuse
        # to reuse stale models fit under the old (basic-before-zproject) order.
        # When zproject never ran, the models were fit on raw stacks: write
        # the negative marker so a later Apply forces a fresh fit instead of
        # trusting models that do not match the z-projected images.
        (model_dir / ".fit_order").write_text(
            "zproject_first" if zproject_applied else "raw_fit",
            encoding="utf-8",
        )
    return model_dir


def transform_images(
    ds: ImageDataset,
    channels: List[str] | None = None,
    root_dir: Union[str, Path, None] = None,
    progress: ProgressCollector = NullProgressCollector(),
) -> ImageDataset:
    channels = channels or ds.intensity_colnames
    metadata = ds.metadata
    root = Path(root_dir) if root_dir else ds.root

    for ci, chan in enumerate(channels):
        # Channel-level progress (2/4 style), mirroring fit_models — reported
        # for every channel (skipped ones still advance the counter). The
        # per-image tqdm below keeps the fine-grained view in the terminal.
        progress.report("basic", ci, len(channels), f"Transform: channel {chan}")

        model_path = root / ".microprofiler" / "BaSiC_model" / f"{chan}.pkl"
        if not model_path.exists():
            logger.warning(
                "No BaSiC model for channel %s, skipping transform", chan)
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # (row_idx, path) pairs keep the row context needed to quarantine a
        # broken/missing file — every row file is deleted and the row skipped.
        entries = [
            (i, Path(metadata.iloc[i][chan]))
            for i in range(len(metadata))
            if pd.notna(metadata.iloc[i][chan])
        ]

        for row_idx, src in tqdm(entries, desc=f"BaSiC transform {chan}", unit="img"):
            if not src.exists():
                quarantine_row(ds, row_idx, f"missing {src.name}")
                continue
            try:
                img = read_image(src)
            except ImageReadError as e:
                quarantine_row(ds, row_idx, str(e))
                continue
            corrected = model.transform(img[None, ...])[0]
            dtype_in = img.dtype
            if dtype_in == np.uint16:
                corrected = np.clip(corrected, 0, 65535)
            elif dtype_in == np.uint8:
                corrected = np.clip(corrected, 0, 255)
            corrected = corrected.astype(dtype_in)
            write_image(src, corrected)

    progress.report("basic", len(channels), len(channels), "Transform complete")

    return rebuild_dataset(ds)


def _validate_shapes(ds: ImageDataset, n_image: int = 100) -> None:
    """Validate that all readable channel images have consistent shapes.

    A missing/unreadable file quarantines its row (all row files deleted)
    instead of aborting; only genuinely inconsistent shapes raise.
    """
    channels = ds.intensity_colnames
    metadata = ds.metadata
    sample: List[tuple] = []
    for chan in channels:
        # Channel columns hold absolute file paths (microBase convention) —
        # never join them with the (root-relative) directory column.
        entries = [
            (i, Path(metadata.iloc[i][chan]))
            for i in range(len(metadata))
            if pd.notna(metadata.iloc[i].get(chan))
        ]
        sample.extend(entries[:n_image])
    first_shape = None
    first_name = None
    for row_idx, path in sample:
        try:
            shape = read_image_shape(path)
        except ImageReadError as e:
            quarantine_row(ds, row_idx, str(e))
            continue
        if first_shape is None:
            first_shape = shape
            first_name = path.name
        elif shape != first_shape:
            raise ValueError(
                f"BaSiC requires uniform image shapes across all channels. "
                f"Got {first_shape} ({first_name}) and {shape} ({path.name}). "
                "This is checked before any processing begins."
            )


def apply_basic(
    ds: ImageDataset,
    mode: str = "fit-transform",
    n_image: int = 100,
    working_size: int = 64,
    enable_darkfield: bool = False,
    root_dir: Union[str, Path, None] = None,
    progress: ProgressCollector = NullProgressCollector(),
    zproject_applied: bool = True,
) -> ImageDataset:
    """Apply BaSiC shading correction (fit and/or transform).

    Defaults match BasicConfig.n_image (100) and the GUI's default.
    zproject_applied flows through to fit_models' fit-order marker.
    """
    if mode in ("fit", "fit-transform"):
        _validate_shapes(ds, n_image)
        fit_models(
            ds,
            n_image=n_image,
            working_size=working_size,
            enable_darkfield=enable_darkfield,
            root_dir=root_dir,
            progress=progress,
            zproject_applied=zproject_applied,
        )

    if mode in ("transform", "fit-transform"):
        return transform_images(ds, root_dir=root_dir, progress=progress)

    return ds
