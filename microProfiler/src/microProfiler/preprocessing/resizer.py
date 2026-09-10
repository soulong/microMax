"""Resize images to a target scale factor."""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom as ndi_zoom

from microBase import ImageDataset
from microProfiler.io import (
    ImageReadError,
    quarantine_row,
    read_image,
    read_mask,
    rebuild_dataset,
    write_image,
)
from microProfiler.progress import StepProgress
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def resize_single(img: np.ndarray, scale_factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    target = (int(round(h * scale_factor)), int(round(w * scale_factor)))
    zoom = (target[0] / h, target[1] / w)
    return ndi_zoom(img, zoom, order=1).astype(img.dtype)


def resize_mask_single(mask: np.ndarray, scale_factor: float) -> np.ndarray:
    """Resize a label mask with nearest-neighbour (never interpolate labels)."""
    h, w = mask.shape[:2]
    target = (int(round(h * scale_factor)), int(round(w * scale_factor)))
    zoom = (target[0] / h, target[1] / w)
    return ndi_zoom(mask, zoom, order=0).astype(mask.dtype)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a mask PNG next to the resized image (cellpose convention)."""
    out = mask if mask.dtype == np.uint8 else mask.astype(np.uint16)
    Image.fromarray(out).save(str(path))


def resize_dataset(
    ds: ImageDataset,
    scale_factor: float = 1.0,
    progress: ProgressCollector = NullProgressCollector(),
) -> ImageDataset:
    if scale_factor == 1.0:
        return ds

    metadata = ds.metadata
    total = len(metadata)
    with StepProgress("Resize", total, progress, desc="Resizing", unit="img") as sp:
        for i in range(total):
            sp.report(i, "Resizing...")
            row = metadata.iloc[i]
            # A row with a missing or unreadable channel file is unusable:
            # quarantine deletes all of its files (channels + masks) and we
            # skip the remaining channels. The final rebuild drops the row.
            broken = False
            for ch in ds.intensity_colnames:
                if pd.isna(row[ch]):
                    continue
                src = Path(row[ch])
                if not src.exists():
                    quarantine_row(ds, i, f"missing {src.name}")
                    broken = True
                    break
                try:
                    img = read_image(src)
                except ImageReadError as e:
                    quarantine_row(ds, i, str(e))
                    broken = True
                    break
                resized = resize_single(img, scale_factor)
                write_image(src, resized)
            if broken:
                continue

            # Existing masks must follow the images — an unresized mask (old
            # resolution) would misalign every downstream overlay/crop. A mask
            # that cannot be read is deleted rather than left stale.
            for mask_col in ds.mask_colnames:
                mpath = row[mask_col]
                if pd.isna(mpath):
                    continue
                mpath = Path(mpath)
                if not mpath.exists():
                    continue
                try:
                    mask = read_mask(mpath)
                except ImageReadError as e:
                    logger.warning(
                        "Resize: deleting unreadable mask %s (%s)", mpath.name, e)
                    try:
                        mpath.unlink()
                    except OSError:
                        logger.exception("Resize: failed to delete %s", mpath)
                    continue
                _write_mask(mpath, resize_mask_single(mask, scale_factor))

        sp.finish("Resize complete")
    return rebuild_dataset(ds)
