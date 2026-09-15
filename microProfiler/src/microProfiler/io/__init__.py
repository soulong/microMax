"""I/O: database and image helpers.

Image reading/writing delegates to microBase. Database wraps sqlite3.

The pipeline uses microBase's readers (``read_image(...)``, which raise ``ImageReadError``)
and ``quarantine_row`` to skip broken rows: when a file is missing or cannot
be decoded, every file of its metadata row is deleted and the row is skipped.
"""

import logging
from pathlib import Path

import pandas as pd

from microBase import ImageDataset, ImageReadError, read_image, read_mask

from microProfiler.io.database import Database

logger = logging.getLogger(__name__)


def write_image(path, data, **kwargs):
    """Write a TIFF image with zlib compression."""
    import tifffile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), data, compression="zlib", **kwargs)


def read_image_shape(path):
    """Read spatial dimensions (H, W) without loading pixel data.

    Uses PIL for format-agnostic shape probing (TIFF/PNG/JPEG). A missing or
    unreadable file raises ImageReadError (never a raw PIL error) so pipeline
    callers can quarantine the row instead of aborting.
    """
    from PIL import Image
    path = Path(path)
    try:
        with Image.open(path) as im:
            w, h = im.size  # PIL returns (W, H)
    except Exception as e:
        raise ImageReadError(path, f"failed to read image {path}: {e}") from e
    return (h, w)


def quarantine_row(ds, row_idx, reason=""):
    """Delete every file referenced by a broken metadata row.

    A row is broken when one of its image files is missing or cannot be
    decoded. All channel files, all mask files and the multi-channel
    ``__file__`` path are removed so the row disappears on the next metadata
    rebuild (no partial/mask-only rows). Deletion errors are logged, never
    raised — the pipeline must keep running.

    Returns the number of deleted files.
    """
    row = ds.metadata.iloc[row_idx]
    cols = list(ds.intensity_colnames) + list(ds.mask_colnames) + ["__file__"]
    deleted = 0
    seen = set()
    for col in cols:
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        path = Path(value)
        if path in seen:
            continue
        seen.add(path)
        try:
            if path.exists():
                path.unlink()
                deleted += 1
        except OSError:
            logger.exception("Quarantine: failed to delete %s", path)
    logger.warning(
        "Broken image row %d quarantined — deleted %d file(s) (%s); row skipped",
        row_idx, deleted, reason,
    )
    return deleted


def clone_dataset(ds: ImageDataset) -> ImageDataset:
    """Create a lightweight clone of an ImageDataset.

    Shares root/patterns but copies metadata and filters so the clone
    can be filtered independently. Cache is fresh (not shared).
    """
    return ds.clone()


def rebuild_dataset(ds: ImageDataset, keep_filters: bool = True) -> ImageDataset:
    """Create a fresh ImageDataset that re-scans the directory on disk.

    Used by preprocessing steps (resize, z-project, tile, BaSiC) after they
    modify files, so the new dataset reflects added/removed/renamed files.
    With ``keep_filters=True`` (the default) the source's filters are carried
    over and re-applied on the rescan; ``keep_filters=False`` yields the
    UNFILTERED on-disk state — used by the GUI to refresh its filter baseline
    after a step, so "clear filter" restores rows the filters previously hid.
    """
    return ImageDataset(
        root=ds.root,
        image_pattern=ds.image_pattern,
        mask_pattern=ds.mask_pattern,
        image_subdir_pattern=ds.image_subdir_pattern,
        channel_layout=ds.channel_layout,
        filters=dict(ds._filters) if keep_filters and ds._filters else None,
    )


__all__ = [
    "Database",
    "ImageReadError",
    "read_image", "write_image", "read_image_shape",
    "read_mask",
    "quarantine_row",
    "clone_dataset", "rebuild_dataset",
]
