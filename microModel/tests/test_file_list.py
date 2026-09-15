"""Tests for the file-list data source (data.file_list / curated.csv).

The file-list workflow: deduplication writes curated.csv with CWD-relative
paths; train/pretrain load it via data.file_list (relative entries resolve
against the process CWD, per load_file_list's contract).
"""

import numpy as np
import pandas as pd
import pytest
import tifffile

from microBase import MicroMaxError
from microModel.dataset import FileListCellDataset
from microModel.utils import load_file_list


def _write_tiff(path, size=16, n_channels=1, seed=0):
    arr = np.random.default_rng(seed).integers(
        0, 1000, size=(size, size, n_channels)).astype(np.uint16)
    tifffile.imwrite(path, arr)


def test_load_file_list_resolves_against_cwd(tmp_path, monkeypatch):
    """Relative entries resolve against the PROCESS CWD (not the CSV's own
    directory) and labels come back as None for empty cells."""
    data_dir = tmp_path / "collections"
    data_dir.mkdir()
    _write_tiff(data_dir / "a.tif")
    _write_tiff(data_dir / "b.tif", seed=1)
    pd.DataFrame({
        "filepath": ["collections/a.tif", "collections/b.tif"],
        "label": ["mitotic", ""],
    }).to_csv(tmp_path / "curated.csv", index=False)

    monkeypatch.chdir(tmp_path)          # run from the generation directory
    paths, labels = load_file_list("curated.csv")
    assert paths == [str(data_dir / "a.tif"), str(data_dir / "b.tif")]
    assert labels == ["mitotic", None]   # empty label = unlabeled


def test_load_file_list_multiple_and_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_tiff(tmp_path / "a.tif")
    _write_tiff(tmp_path / "b.tif", seed=1)
    pd.DataFrame({"filepath": ["a.tif"]}).to_csv("one.csv", index=False)
    pd.DataFrame({"filepath": ["b.tif"]}).to_csv("two.csv", index=False)

    paths, labels = load_file_list(["one.csv", "two.csv"])
    assert len(paths) == 2 and labels == [None, None]

    with pytest.raises(MicroMaxError):
        load_file_list("missing.csv")                 # unknown list
    pd.DataFrame({"path": ["a.tif"]}).to_csv("bad.csv", index=False)
    with pytest.raises(MicroMaxError):
        load_file_list("bad.csv")                     # no filepath column


def test_file_list_cell_dataset(tmp_path):
    """get_cell mirrors CellDataset's contract: all channels, (H, W, C);
    channel count is auto-detected from the first file."""
    _write_tiff(tmp_path / "a.tif", n_channels=2, seed=0)
    _write_tiff(tmp_path / "b.tif", n_channels=2, seed=1)
    ds = FileListCellDataset([str(tmp_path / "a.tif"),
                              str(tmp_path / "b.tif")], channel_layout="HWC")
    assert ds.intensity_colnames == ["ch1", "ch2"]
    img = ds.get_cell(0)
    assert img.shape == (16, 16, 2)
    with pytest.raises(MicroMaxError):
        ds.get_cell(5)                                # out of range


def test_prepare_pretrain_data_both_sources_empty_raises():
    """file_list-mode configs keep data.file_dir null (entries commented out);
    the data prep must accept that (root ignored) and only fail with a
    clear error when NEITHER source is set. Regression: 'NoneType' object
    is not iterable at list(root)."""
    from microModel.pretrain import _prepare_pretrain_data

    base = {"max_value": 65535, "channels": [1], "channel_layout": None,
            "image_pattern": None, "sample_max": None, "sample_by": "per_class"}
    with pytest.raises(MicroMaxError, match="both"):
        _prepare_pretrain_data({"data": {**base, "file_list": None,
                                         "file_dir": None}})
    # file_list set + root null must NOT raise here (root ignored; the CSV
    # itself is validated later by load_file_list).
    cfg = {"data": {**base, "file_list": "no_such_list.csv", "file_dir": None}}
    with pytest.raises(MicroMaxError, match="no_such_list"):
        _prepare_pretrain_data(cfg)
