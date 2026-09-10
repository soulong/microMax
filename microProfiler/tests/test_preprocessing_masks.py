"""Preprocessing vs existing masks: resize follows, zproject/tile clean up."""

import re

import numpy as np
from PIL import Image
from tifffile import imwrite

from microBase import ImageDataset
from microProfiler.preprocessing.resizer import resize_dataset
from microProfiler.preprocessing.tile_splitter import tile_dataset
from microProfiler.preprocessing.z_projection import z_project_dataset

IMAGE_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p0?(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
)
MASK_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p0?(?P<stack>\d+)-ch(?P<channel>\d+)"
    r"_cp_masks_(?P<mask_name>.+)\.png"
)


def _make_ds(tmp_path):
    return ImageDataset(
        root=tmp_path, image_pattern=IMAGE_PATTERN, mask_pattern=MASK_PATTERN,
        channel_layout=None,
    )


def _write_cell_mask(path, h, w):
    mask = np.zeros((h, w), dtype=np.uint16)
    mask[2:12, 2:12] = 1
    mask[h - 12:h - 2, w - 12:w - 2] = 2
    Image.fromarray(mask).save(str(path))


def test_resize_scales_masks(tmp_path):
    imwrite(str(tmp_path / "r01c01f01p01-ch1.tiff"),
            np.full((32, 32), 100, dtype=np.uint16))
    _write_cell_mask(tmp_path / "r01c01f01p01-ch1_cp_masks_cell.png", 32, 32)

    ds = resize_dataset(_make_ds(tmp_path), scale_factor=0.5)

    img = ds.get_imageset(0)[0]
    assert img.shape == (16, 16, 1)
    mask = np.asarray(Image.open(
        tmp_path / "r01c01f01p01-ch1_cp_masks_cell.png"))
    assert mask.shape == (16, 16)
    assert set(np.unique(mask).tolist()) - {0} == {1, 2}


def test_zproject_deletes_plane_masks(tmp_path):
    for stack in (1, 2):
        imwrite(str(tmp_path / f"r01c01f01p{stack:02d}-ch1.tiff"),
                np.full((16, 16), stack * 10, dtype=np.uint16))
        _write_cell_mask(
            tmp_path / f"r01c01f01p{stack:02d}-ch1_cp_masks_cell.png", 16, 16)

    ds = z_project_dataset(_make_ds(tmp_path), method="max", delete_original=True)

    # Projection row exists, and no mask file survived the projection.
    assert len(ds) == 1
    assert ds.mask_colnames == []
    assert not list(tmp_path.glob("*_cp_masks_*.png"))
    assert (tmp_path / "r01c01f01p00-ch1.tiff").exists()


def test_tile_keeps_sources_when_no_tiles_produced(tmp_path):
    imwrite(str(tmp_path / "r01c01f01p01-ch1.tiff"),
            np.full((8, 8), 7, dtype=np.uint16))

    ds = tile_dataset(_make_ds(tmp_path), tile_width=16, tile_height=16,
                      delete_original=True)

    # Sub-tile image produced no tiles: the source must NOT be deleted.
    assert (tmp_path / "r01c01f01p01-ch1.tiff").exists()
    assert len(ds) == 1


def test_tile_deletes_source_and_mask_after_tiling(tmp_path):
    imwrite(str(tmp_path / "r01c01f01p01-ch1.tiff"),
            np.full((32, 32), 7, dtype=np.uint16))
    _write_cell_mask(tmp_path / "r01c01f01p01-ch1_cp_masks_cell.png", 32, 32)

    tile_dataset(_make_ds(tmp_path), tile_width=16, tile_height=16,
                 delete_original=True)

    assert not (tmp_path / "r01c01f01p01-ch1.tiff").exists()
    assert not (tmp_path / "r01c01f01p01-ch1_cp_masks_cell.png").exists()
    # 4 tiles written with the synthetic field prefix.
    assert len(list(tmp_path.glob("*.tiff"))) == 4
