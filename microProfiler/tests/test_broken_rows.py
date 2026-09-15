"""Broken-image quarantine tests.

When a pipeline step hits a missing/unreadable image file, the whole metadata
row is quarantined: every file it references (channels + masks) is deleted and
the row disappears on the next metadata rebuild — the run keeps going.
"""

import re

import numpy as np
import pytest
from PIL import Image
from tifffile import imwrite

from microBase import ImageDataset, ImageReadError
from microProfiler.io import quarantine_row, read_image
from microProfiler.preprocessing.resizer import resize_dataset
from microProfiler.preprocessing.z_projection import z_project_dataset

IMAGE_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
)
MASK_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)"
    r"_cp_masks_(?P<mask_name>\w+)\.png"
)


def _write_site(root, site, channels=(1, 2), h=64, w=64, value=1000):
    """Write one valid channel file per channel for a site (r{site} site)."""
    row = col = site
    for ch in channels:
        arr = np.full((h, w), value + ch, dtype=np.uint16)
        imwrite(str(root / f"r{row:02d}c{col:02d}f01p01-ch{ch}.tiff"), arr)


def _corrupt_file(path):
    """Overwrite a file with bytes PIL cannot decode."""
    path.write_bytes(b"not a real tiff")


def test_resize_quarantines_broken_row_and_deletes_files(tmp_path):
    """One corrupt channel quarantines the whole row: all channel files of
    that site are deleted, the other sites are resized, and the returned
    dataset no longer contains the broken row."""
    _write_site(tmp_path, 1)
    _write_site(tmp_path, 2)
    _write_site(tmp_path, 3)
    _corrupt_file(tmp_path / "r02c02f01p01-ch2.tiff")

    ds = ImageDataset(
        root=tmp_path, image_pattern=IMAGE_PATTERN, channel_layout=None,
    )
    assert len(ds) == 3

    out = resize_dataset(ds, scale_factor=0.5)

    # Both channel files (including the already-resized ch1) are gone.
    assert not (tmp_path / "r02c02f01p01-ch1.tiff").exists()
    assert not (tmp_path / "r02c02f01p01-ch2.tiff").exists()
    # The broken row is dropped; the two intact sites are resized.
    assert len(out) == 2
    assert read_image(tmp_path / "r01c01f01p01-ch1.tiff").shape == (32, 32)
    assert read_image(tmp_path / "r03c03f01p01-ch1.tiff").shape == (32, 32)


def test_quarantine_row_deletes_channels_and_masks(tmp_path):
    """quarantine_row removes every path of the row: channels + masks."""
    _write_site(tmp_path, 1, channels=(1, 2))
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[10:20, 10:20] = 1
    mask_path = tmp_path / "r01c01f01p01-ch1_cp_masks_cell.png"
    Image.fromarray(mask).save(str(mask_path))

    ds = ImageDataset(
        root=tmp_path, image_pattern=IMAGE_PATTERN,
        mask_pattern=MASK_PATTERN, channel_layout=None,
    )
    deleted = quarantine_row(ds, 0, "test")

    assert deleted == 3
    assert not (tmp_path / "r01c01f01p01-ch1.tiff").exists()
    assert not (tmp_path / "r01c01f01p01-ch2.tiff").exists()
    assert not mask_path.exists()


def test_zproject_quarantines_broken_plane(tmp_path):
    """A corrupt stack plane is deleted as a whole; the projection is built
    from the surviving planes and the original files are removed as usual."""
    row = col = 1
    for stack in (1, 2, 3):
        arr = np.full((32, 32), stack * 100, dtype=np.uint16)
        imwrite(str(tmp_path / f"r{row:02d}c{col:02d}f01p0{stack}-ch1.tiff"), arr)
    _corrupt_file(tmp_path / "r01c01f01p02-ch1.tiff")

    ds = ImageDataset(
        root=tmp_path, image_pattern=IMAGE_PATTERN, channel_layout=None,
    )
    out = z_project_dataset(ds, method="max", delete_original=True)

    # The corrupt plane is quarantined; the two read planes are consumed by
    # the projection, which survives as the single rebuilt row.
    assert not (tmp_path / "r01c01f01p01-ch1.tiff").exists()
    assert not (tmp_path / "r01c01f01p02-ch1.tiff").exists()
    assert not (tmp_path / "r01c01f01p03-ch1.tiff").exists()
    projected = tmp_path / "r01c01f01p0-ch1.tiff"
    assert projected.exists()
    assert len(out) == 1
    # Max of the two surviving planes (300), not the corrupt plane.
    assert int(read_image(projected).max()) == 300


def test_read_image_strict_raises_before_quarantine_handler(tmp_path):
    """Sanity: the strict reader is the signal the pipeline catches."""
    p = tmp_path / "broken.tiff"
    p.write_bytes(b"garbage")
    with pytest.raises(ImageReadError):
        read_image(p)


def test_zproject_quarantine_uses_global_row_position(tmp_path):
    """With MULTIPLE projection groups, a broken plane in a later group must
    quarantine its OWN metadata row. The quarantine used to receive the
    group-LOCAL enumerate position, deleting another group's files while the
    broken file survived."""
    for row, col in ((1, 1), (2, 2)):
        for stack in (1, 2):
            arr = np.full((32, 32), stack * 100, dtype=np.uint16)
            imwrite(str(tmp_path / f"r{row:02d}c{col:02d}f01p0{stack}-ch1.tiff"), arr)
    # Break the LAST row globally (index 3), which is position 1 WITHIN the
    # second group.
    _corrupt_file(tmp_path / "r02c02f01p02-ch1.tiff")

    ds = ImageDataset(
        root=tmp_path, image_pattern=IMAGE_PATTERN, channel_layout=None,
    )
    assert len(ds) == 4
    out = z_project_dataset(ds, method="max", delete_original=True)

    # The broken file itself is quarantined (previously it survived while
    # site 1's files were deleted instead).
    assert not (tmp_path / "r02c02f01p02-ch1.tiff").exists()
    # Site 1 is untouched: its two planes are consumed by the projection.
    assert not (tmp_path / "r01c01f01p01-ch1.tiff").exists()
    assert not (tmp_path / "r01c01f01p02-ch1.tiff").exists()
    projected = tmp_path / "r01c01f01p0-ch1.tiff"
    assert projected.exists()
    assert int(read_image(projected).max()) == 200
    # Site 2 keeps its single surviving plane as a plain row (a one-plane
    # group cannot be projected), so the rebuilt dataset has 2 rows.
    assert (tmp_path / "r02c02f01p01-ch1.tiff").exists()
    assert len(out) == 2
