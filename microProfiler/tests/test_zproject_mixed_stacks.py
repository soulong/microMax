"""Z-projection when channels have different z-stack depths.

Microscopes routinely acquire one channel as a z-stack and another as a
single plane. The stacked channels must be projected and the single-plane
ones carried over unchanged to the SAME stack-0 row: skipping them used to
leave every site split into two rows, each missing the other's channels —
rows every later step treats as broken (quarantine) or crashes on.
"""

import re

import numpy as np
from tifffile import imwrite

from microBase import ImageDataset
from microProfiler.io import read_image
from microProfiler.preprocessing.z_projection import z_project_dataset

# Opera-style names, e.g. r01c01f01p01-ch01t01.tiff (the optional leading
# zeros sit OUTSIDE the captures, like the real acquisition patterns).
IMAGE_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f0?(?P<field>\d+)p0?(?P<stack>\d+)"
    r"-ch0?(?P<channel>\d+)t0?(?P<timepoint>\d+)\.tiff"
)


def _make_ds(tmp_path):
    return ImageDataset(
        root=tmp_path, image_pattern=IMAGE_PATTERN, channel_layout=None,
    )


def _write(tmp_path, name, value):
    """Write one constant 8x8 uint16 plane and return its path."""
    path = tmp_path / name
    imwrite(str(path), np.full((8, 8), value, dtype=np.uint16))
    return path


def _assert_one_complete_row_per_site(ds, n_sites):
    """Every site is ONE row whose channels are all present and readable."""
    assert len(ds) == n_sites
    assert ds.intensity_colnames == ["ch1", "ch2"]
    for i in range(n_sites):
        img, _masks = ds.get_imageset(i)  # raises on a missing channel
        assert img.shape == (8, 8, 2)


def test_zproject_carries_single_plane_channel(tmp_path):
    """ch1 has ONE plane, ch2 a 2-plane stack: ch2 is projected, ch1 is
    carried over unchanged, and both land in the site's single row."""
    for site in (1, 2):
        _write(tmp_path, f"r0{site}c0{site}f01p01-ch01t01.tiff", 5 + site)
        _write(tmp_path, f"r0{site}c0{site}f01p01-ch02t01.tiff", 10)
        _write(tmp_path, f"r0{site}c0{site}f01p02-ch02t01.tiff", 20 + site)
    assert len(_make_ds(tmp_path)) == 4  # 2 sites x 2 planes

    ds = z_project_dataset(_make_ds(tmp_path), method="max", delete_original=True)

    _assert_one_complete_row_per_site(ds, 2)
    for site in (1, 2):
        # Every plane-level source is gone; only the stack-0 files remain.
        assert not (tmp_path / f"r0{site}c0{site}f01p01-ch01t01.tiff").exists()
        assert not (tmp_path / f"r0{site}c0{site}f01p01-ch02t01.tiff").exists()
        assert not (tmp_path / f"r0{site}c0{site}f01p02-ch02t01.tiff").exists()
        ch1 = read_image(tmp_path / f"r0{site}c0{site}f01p00-ch01t01.tiff")
        ch2 = read_image(tmp_path / f"r0{site}c0{site}f01p00-ch02t01.tiff")
        # ch1 is the untouched single plane, ch2 the max of its two planes.
        assert ch1.dtype == np.uint16 and int(ch1.min()) == int(ch1.max()) == 5 + site
        assert int(ch2.max()) == 20 + site
    assert len(list(tmp_path.glob("*.tiff"))) == 4


def test_zproject_single_plane_is_not_altered_by_the_method(tmp_path):
    """A carried-over plane is never run through the projection maths: its
    pixels (and dtype) survive bit-exact whatever the method."""
    plane = np.arange(64, dtype=np.float32).reshape(8, 8) + 0.25
    imwrite(str(tmp_path / "r01c01f01p01-ch01t01.tiff"), plane)
    _write(tmp_path, "r01c01f01p01-ch02t01.tiff", 10)
    _write(tmp_path, "r01c01f01p02-ch02t01.tiff", 21)

    z_project_dataset(_make_ds(tmp_path), method="mean", delete_original=True)

    carried = read_image(tmp_path / "r01c01f01p00-ch01t01.tiff")
    assert carried.dtype == np.float32
    assert np.array_equal(carried, plane)  # 'mean' would have rounded the .25
    # ch2 really is the (rounded) mean of 10 and 21.
    assert int(read_image(tmp_path / "r01c01f01p00-ch02t01.tiff").max()) == 16


def test_zproject_repairs_split_dataset_and_is_idempotent(tmp_path):
    """A dataset left split by the old behaviour (projected ch2 at stack 0,
    unprojected ch1 at stack 1) is merged by a re-run: ch1 moves to the
    stack-0 row, the already-projected ch2 file is not rewritten, and a
    further run changes nothing."""
    _write(tmp_path, "r01c01f01p01-ch01t01.tiff", 7)
    projected = _write(tmp_path, "r01c01f01p00-ch02t01.tiff", 30)
    before = projected.read_bytes()
    assert len(_make_ds(tmp_path)) == 2  # the split state: two half rows

    ds = z_project_dataset(_make_ds(tmp_path), method="max", delete_original=True)

    _assert_one_complete_row_per_site(ds, 1)
    assert not (tmp_path / "r01c01f01p01-ch01t01.tiff").exists()
    assert int(read_image(tmp_path / "r01c01f01p00-ch01t01.tiff").max()) == 7
    # Byte-identical: the file already under the projected name was left
    # alone (a rewrite would have re-encoded it with zlib compression).
    assert projected.read_bytes() == before

    snapshot = {p.name: p.read_bytes() for p in tmp_path.glob("*.tiff")}
    again = z_project_dataset(ds, method="max", delete_original=True)
    assert len(again) == 1
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.tiff")} == snapshot


def test_zproject_leaves_one_plane_site_untouched(tmp_path):
    """A site whose channels ALL have a single plane is already one complete
    row: it is not part of any projection and keeps its file names, while a
    neighbouring stacked site is projected as usual."""
    _write(tmp_path, "r01c01f01p01-ch01t01.tiff", 3)
    _write(tmp_path, "r01c01f01p01-ch02t01.tiff", 4)
    for stack in (1, 2):
        _write(tmp_path, f"r02c02f01p0{stack}-ch01t01.tiff", stack * 10)
        _write(tmp_path, f"r02c02f01p0{stack}-ch02t01.tiff", stack * 100)

    ds = z_project_dataset(_make_ds(tmp_path), method="max", delete_original=True)

    _assert_one_complete_row_per_site(ds, 2)
    assert (tmp_path / "r01c01f01p01-ch01t01.tiff").exists()
    assert (tmp_path / "r01c01f01p01-ch02t01.tiff").exists()
    assert int(read_image(tmp_path / "r02c02f01p00-ch01t01.tiff").max()) == 20
    assert int(read_image(tmp_path / "r02c02f01p00-ch02t01.tiff").max()) == 200
