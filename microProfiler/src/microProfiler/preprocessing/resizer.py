"""Resize images to a target scale factor."""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import zoom as ndi_zoom
from tqdm import tqdm

from microBase import ImageDataset
from microProfiler.io import read_image, write_image, rebuild_dataset
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def resize_single(img: np.ndarray, scale_factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    target = (int(round(h * scale_factor)), int(round(w * scale_factor)))
    zoom = (target[0] / h, target[1] / w)
    return ndi_zoom(img, zoom, order=1).astype(img.dtype)


def resize_dataset(
    ds: ImageDataset,
    scale_factor: float = 1.0,
    progress: ProgressCollector = NullProgressCollector(),
) -> ImageDataset:
    if scale_factor == 1.0:
        return ds

    metadata = ds.metadata
    for i in tqdm(range(len(metadata)), desc="Resizing", unit="img"):
        progress.report("Resize", i, len(metadata), "Resizing...")
        row = metadata.iloc[i]
        for ch in ds.intensity_colnames:
            if pd.notna(row[ch]):
                src = Path(row[ch])
                if not src.exists():
                    logger.warning("Missing file, skipping: %s", src)
                    continue
                img = read_image(src)
                resized = resize_single(img, scale_factor)
                write_image(src, resized)

    progress.report("Resize", len(metadata), len(metadata), "Resize complete")
    return rebuild_dataset(ds)
