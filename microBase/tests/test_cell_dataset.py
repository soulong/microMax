"""Tests for microBase.CellDataset: pre-cropped single-cell loader."""

import re
import numpy as np
import pytest
from pathlib import Path
from tifffile import imwrite

from microBase import CellDataset, ConfigError, DataError, DatasetError


def _make_cell_dataset(tmp_path, n_cells=5, n_channels=2, h=32, w=32, layout="CHW"):
    """Build a folder of single-cell TIFFs.

    Filenames: cell_{i:03d}.tiff
    """
    for i in range(n_cells):
        if layout == "CHW":
            arr = np.zeros((n_channels, h, w), dtype=np.uint16)
            for ch in range(n_channels):
                arr[ch] = (ch + 1) * 1000 + i
        elif layout == "HWC":
            arr = np.zeros((h, w, n_channels), dtype=np.uint16)
            for ch in range(n_channels):
                arr[:, :, ch] = (ch + 1) * 1000 + i
        else:  # layout=None => 2D single-channel
            arr = np.zeros((h, w), dtype=np.uint16)
            arr[:] = 1000 + i
        imwrite(str(tmp_path / f"cell_{i:03d}.tiff"), arr)
    return tmp_path


def _make_cell_dataset_with_meta(tmp_path, n_cells=5, n_channels=2, h=32, w=32):
    """Build cells with parseable metadata in filenames.

    Filenames: C{channel}f{field}z{stack}t{timepoint}_{label}.tiff
    """
    for i in range(n_cells):
        arr = np.zeros((n_channels, h, w), dtype=np.uint16)
        for ch in range(n_channels):
            arr[ch] = (ch + 1) * 1000
        field = (i % 3) + 1
        label = i + 1
        fname = f"C1_f{field:02d}_z01_t01_{label:03d}.tiff"
        imwrite(str(tmp_path / fname), arr)
    return tmp_path


def test_cell_dataset_basic(tmp_path):
    _make_cell_dataset(tmp_path, n_cells=5, n_channels=2)
    ds = CellDataset(root=tmp_path, channel_layout="CHW")
    assert len(ds) == 5
    assert ds.intensity_colnames == ["ch1", "ch2"]
    assert ds.img_shape == (32, 32)


def test_cell_dataset_hwc_layout(tmp_path):
    _make_cell_dataset(tmp_path, n_cells=3, n_channels=3, layout="HWC")
    ds = CellDataset(root=tmp_path, channel_layout="HWC")
    assert len(ds) == 3
    assert ds.intensity_colnames == ["ch1", "ch2", "ch3"]


def test_cell_dataset_get_cell(tmp_path):
    _make_cell_dataset(tmp_path, n_cells=3, n_channels=2)
    ds = CellDataset(root=tmp_path, channel_layout="CHW")
    img = ds.get_cell(0)
    assert img.shape == (32, 32, 2)
    # ch1 -> 1000 + 0 (cell index 0), ch2 -> 2000 + 0
    assert img[0, 0, 0] == 1000
    assert img[0, 0, 1] == 2000


def test_cell_dataset_get_cell_index_out_of_range(tmp_path):
    _make_cell_dataset(tmp_path, n_cells=2, n_channels=1)
    ds = CellDataset(root=tmp_path, channel_layout="CHW")
    with pytest.raises(DataError):
        ds.get_cell(99)


def test_cell_dataset_with_image_pattern(tmp_path):
    """image_pattern parses metadata from filenames."""
    _make_cell_dataset_with_meta(tmp_path, n_cells=5, n_channels=2)
    ds = CellDataset(
        root=tmp_path,
        channel_layout="CHW",
        image_pattern=re.compile(
            r"C(?P<channel>\d+)_f(?P<field>\d+)_z(?P<stack>\d+)_t(?P<timepoint>\d+)_(?P<label>\d+)\.tiff"
        ),
    )
    assert len(ds) == 5
    df = ds.metadata
    assert "field" in df.columns
    assert "stack" in df.columns
    assert "timepoint" in df.columns
    assert "label" in df.columns
    # label kept verbatim as TEXT ('001', '002', ...) — no int coercion
    assert sorted(df["label"].tolist()) == ["001", "002", "003", "004", "005"]


def test_cell_dataset_image_pattern_filters_non_matching(tmp_path):
    """Files that don't match the pattern should be skipped."""
    _make_cell_dataset_with_meta(tmp_path, n_cells=3, n_channels=1)
    # Add a non-matching file
    arr = np.zeros((1, 32, 32), dtype=np.uint16)
    imwrite(str(tmp_path / "random_other_file.tiff"), arr)
    ds = CellDataset(
        root=tmp_path,
        channel_layout="CHW",
        image_pattern=re.compile(
            r"C(?P<channel>\d+)_f(?P<field>\d+)_z(?P<stack>\d+)_t(?P<timepoint>\d+)_(?P<label>\d+)\.tiff"
        ),
    )
    assert len(ds) == 3  # only 3 matching cells


def test_cell_dataset_no_pattern_minimal_metadata(tmp_path):
    """Without image_pattern, metadata has just stem, path, directory."""
    _make_cell_dataset(tmp_path, n_cells=2, n_channels=1)
    ds = CellDataset(root=tmp_path, channel_layout="CHW")
    df = ds.metadata
    assert "stem" in df.columns
    assert "path" in df.columns
    assert "directory" in df.columns
    assert "field" not in df.columns  # no pattern -> no metadata extraction


def test_cell_dataset_root_not_found_raises(tmp_path):
    with pytest.raises(DatasetError):
        CellDataset(root=tmp_path / "nonexistent")


def test_cell_dataset_bad_layout_raises(tmp_path):
    _make_cell_dataset(tmp_path, n_cells=1, n_channels=1)
    with pytest.raises(ConfigError):
        CellDataset(root=tmp_path, channel_layout="BAD")


def test_cell_dataset_none_layout_2d_cells(tmp_path):
    """channel_layout=None: 2D single-channel TIFFs, read as (H, W, 1)."""
    _make_cell_dataset(tmp_path, n_cells=4, h=40, w=48, layout=None)
    ds = CellDataset(root=tmp_path, channel_layout=None)
    assert len(ds) == 4
    assert ds.intensity_colnames == ["ch1"]
    assert ds.img_shape == (40, 48)
    img = ds.get_cell(0)
    assert img.shape == (40, 48, 1)
    # cell index 0 -> 1000 + 0
    assert img[0, 0, 0] == 1000


def test_cell_dataset_leading_zeros_in_meta_preserved(tmp_path):
    """label and field captured as '001' are kept verbatim as TEXT ('001').

    All metadata is stored as TEXT from extraction through DB storage — no
    leading-zero stripping or int coercion. The `channel` group is silently
    skipped (channels are inside the file for CellDataset).
    """
    for i in range(3):
        arr = np.zeros((2, 16, 16), dtype=np.uint16)
        arr[0] = 1000
        arr[1] = 2000
        fname = f"C1_f01_z01_t01_{i + 1:03d}.tiff"
        imwrite(str(tmp_path / fname), arr)
    ds = CellDataset(
        root=tmp_path,
        channel_layout="CHW",
        image_pattern=re.compile(
            r"C(?P<channel>\d+)_f(?P<field>\d+)_z(?P<stack>\d+)_t(?P<timepoint>\d+)_(?P<label>\d+)\.tiff"
        ),
    )
    df = ds.metadata
    # leading zeros preserved, values kept as strings (TEXT storage)
    assert sorted(df["label"].tolist()) == ["001", "002", "003"]
    assert df["field"].iloc[0] == "01"
    assert df["stack"].iloc[0] == "01"
    assert df["timepoint"].iloc[0] == "01"
    # channel group is silently skipped for CellDataset
    assert "channel" not in df.columns


def test_cell_dataset_empty_directory(tmp_path):
    """Empty directory -> empty dataset."""
    ds = CellDataset(root=tmp_path, channel_layout="CHW")
    assert len(ds) == 0
    assert ds.intensity_colnames == []


def test_cell_dataset_image_pattern_property(tmp_path):
    """image_pattern property returns the original string (compiled or str)."""
    pat = r"C1_f(?P<field>\d+)_z01_t01_\d+\.tiff"
    _make_cell_dataset_with_meta(tmp_path, n_cells=2, n_channels=1)
    ds1 = CellDataset(root=tmp_path, channel_layout="CHW", image_pattern=re.compile(pat))
    assert ds1.image_pattern == pat
    ds2 = CellDataset(root=tmp_path, channel_layout="CHW", image_pattern=pat)
    assert ds2.image_pattern == pat
    ds3 = CellDataset(root=tmp_path, channel_layout="CHW")
    assert ds3.image_pattern is None


def test_cell_dataset_skips_non_tiff(tmp_path):
    """Non-TIFF files in the directory are ignored."""
    _make_cell_dataset(tmp_path, n_cells=2, n_channels=1)
    # Add a stray .txt file
    (tmp_path / "readme.txt").write_text("not a tiff")
    ds = CellDataset(root=tmp_path, channel_layout="CHW")
    assert len(ds) == 2
