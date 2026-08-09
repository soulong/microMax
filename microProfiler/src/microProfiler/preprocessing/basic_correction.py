"""BaSiC shading correction — thin wrapper around the basic/ folder.

The actual BaSiC algorithm lives in ``microProfiler.preprocessing.basic``
(copied verbatim from the original image_profiler).
"""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from microBase import ImageDataset
from microProfiler.io import read_image, write_image, rebuild_dataset
from microProfiler.preprocessing.basic.basic import BaSiC
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def basic_fit(
    image_paths: List[Path],
    n_image: int = 50,
    enable_darkfield: bool = False,
    working_size: int = 64,
    seed: int = 42,
) -> BaSiC:
    """Fit BaSiC model on a set of images.

    When subsampling (len(image_paths) > n_image), a fixed seed keeps the
    fit reproducible across runs (the fitted flatfield/darkfield — and every
    downstream profile value — would otherwise vary run to run).
    """
    if len(image_paths) > n_image:
        rng = random.Random(seed)
        image_paths = rng.sample(image_paths, k=n_image)

    imgs = [read_image(p) for p in image_paths]
    shapes = {img.shape for img in imgs}
    if len(shapes) > 1:
        raise ValueError(
            f"BaSiC fit requires uniform image shapes, got {len(shapes)} different shapes: {shapes}"
        )
    imgs = np.stack(imgs)
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
    n_image: int = 50,
    working_size: int = 64,
    enable_darkfield: bool = False,
    root_dir: Union[str, Path, None] = None,
    progress: ProgressCollector = NullProgressCollector(),
) -> Path:
    """Fit BaSiC models for specified channels."""
    channels = channels or ds.intensity_colnames
    metadata = ds.metadata
    root = Path(root_dir) if root_dir else ds.root

    model_dir = root / ".microprofiler" / "BaSiC_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("BaSiC fitting channels")
    for ci, chan in enumerate(channels):
        progress.report("BaSiC Fit", ci, len(channels), f"Fitting channel {chan}")
        paths = [
            Path(metadata.iloc[i]["directory"]) / metadata.iloc[i][chan]
            for i in range(len(metadata))
            if pd.notna(metadata.iloc[i][chan])
        ]
        paths = [p for p in paths if p.exists()]
        if not paths:
            logger.warning("No existing files for channel %s, skipping", chan)
            continue

        model = basic_fit(paths, n_image, enable_darkfield, working_size)

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

    progress.report("BaSiC Fit", len(channels), len(channels), "Fit complete")
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
        progress.report("BaSiC Transform", ci, len(channels), f"Channel {chan}")

        model_path = root / ".microprofiler" / "BaSiC_model" / f"{chan}.pkl"
        if not model_path.exists():
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        paths = [
            Path(metadata.iloc[i]["directory"]) / metadata.iloc[i][chan]
            for i in range(len(metadata))
            if pd.notna(metadata.iloc[i][chan])
        ]
        missing = [p for p in paths if not p.exists()]
        if missing:
            logger.warning("Skipping %d missing file(s) for channel %s", len(missing), chan)
        paths = [p for p in paths if p.exists()]

        for src in tqdm(paths, desc=f"BaSiC transform {chan}", unit="img"):
            img = read_image(src)
            corrected = model.transform(img[None, ...])[0]
            dtype_in = img.dtype
            if dtype_in == np.uint16:
                corrected = np.clip(corrected, 0, 65535)
            elif dtype_in == np.uint8:
                corrected = np.clip(corrected, 0, 255)
            corrected = corrected.astype(dtype_in)
            write_image(src, corrected)

    progress.report("BaSiC Transform", len(channels), len(channels), "Transform complete")

    return rebuild_dataset(ds)


def _validate_shapes(ds: ImageDataset, n_image: int = 50) -> None:
    """Validate that all channel images have consistent shapes."""
    channels = ds.intensity_colnames
    metadata = ds.metadata
    sample_paths: List[Path] = []
    for chan in channels:
        paths = [
            Path(metadata.iloc[i]["directory"]) / metadata.iloc[i][chan]
            for i in range(len(metadata))
            if pd.notna(metadata.iloc[i].get(chan))
        ]
        paths = [p for p in paths if p.exists()]
        if paths:
            sample_paths.extend(paths[:n_image])
    from microProfiler.io import read_image_shape
    if sample_paths:
        first_shape = read_image_shape(sample_paths[0])
        for p in sample_paths[1:]:
            shape = read_image_shape(p)
            if shape != first_shape:
                raise ValueError(
                    f"BaSiC requires uniform image shapes across all channels. "
                    f"Got {first_shape} and {shape} for {p.name}. "
                    "This is checked before any processing begins."
                )


def apply_basic(
    ds: ImageDataset,
    mode: str = "fit-transform",
    n_image: int = 50,
    working_size: int = 64,
    enable_darkfield: bool = False,
    root_dir: Union[str, Path, None] = None,
    progress: ProgressCollector = NullProgressCollector(),
) -> ImageDataset:
    if mode in ("fit", "fit-transform"):
        _validate_shapes(ds, n_image)
        fit_models(
            ds,
            n_image=n_image,
            working_size=working_size,
            enable_darkfield=enable_darkfield,
            root_dir=root_dir,
            progress=progress,
        )

    if mode in ("transform", "fit-transform"):
        return transform_images(ds, root_dir=root_dir, progress=progress)

    return ds
