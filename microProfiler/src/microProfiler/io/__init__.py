"""I/O: database and image helpers.

Image reading/writing delegates to microBase. Database wraps sqlite3.
"""

from microBase import read_image, read_mask
from microBase import ImageDataset

from microProfiler.io.database import Database


def write_image(path, data, **kwargs):
    """Write a TIFF image with zlib compression."""
    from pathlib import Path
    import tifffile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), data, compression="zlib", **kwargs)


def read_image_shape(path):
    """Read spatial dimensions (H, W) without loading pixel data.

    Uses PIL for format-agnostic shape probing (TIFF/PNG/JPEG).
    """
    from pathlib import Path
    from PIL import Image
    path = Path(path)
    with Image.open(path) as im:
        w, h = im.size  # PIL returns (W, H)
    return (h, w)


def clone_dataset(ds: ImageDataset) -> ImageDataset:
    """Create a lightweight clone of an ImageDataset.

    Shares root/patterns but copies metadata and filters so the clone
    can be filtered independently. Cache is fresh (not shared).
    """
    return ds.clone()


def rebuild_dataset(ds: ImageDataset) -> ImageDataset:
    """Create a fresh ImageDataset that re-scans the directory on disk.

    Used by preprocessing steps (resize, z-project, tile, BaSiC) after they
    modify files, so the new dataset reflects added/removed/renamed files.
    """
    return ImageDataset(
        root=ds.root,
        image_pattern=ds.image_pattern,
        mask_pattern=ds.mask_pattern,
        image_subdir_pattern=ds.image_subdir_pattern,
        channel_layout=ds.channel_layout,
        filters=dict(ds._filters) if ds._filters else None,
    )


__all__ = [
    "Database",
    "read_image", "write_image", "read_image_shape",
    "read_mask",
    "clone_dataset", "rebuild_dataset",
]
