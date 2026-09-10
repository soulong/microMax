"""Tests for microBase.ImageDataset: whole-image dataset loader + cropping."""

import re
import pickle
import numpy as np
import pytest
from pathlib import Path
from tifffile import imwrite
from PIL import Image

from microBase import (
    ConfigError,
    DataError,
    DatasetError,
    ImageDataset,
    ImageReadError,
)


def _make_one_channel_per_file_dataset(tmp_path, n_sites=3, n_channels=2, h=64, w=64):
    """Build a dataset with one file per channel per site.

    Filenames: r{row}c{col}f{field}p{stack}-ch{channel}.tiff
    """
    for site in range(n_sites):
        row = site + 1
        col = site + 1
        field = 1
        stack = 1
        for ch in range(1, n_channels + 1):
            arr = np.random.randint(0, 65535, size=(h, w), dtype=np.uint16)
            arr[:] = ch * 1000  # constant value per channel for verification
            fname = f"r{row:02d}c{col:02d}f{field:02d}p{stack:02d}-ch{ch}.tiff"
            imwrite(str(tmp_path / fname), arr)
    return tmp_path


def _make_one_channel_per_file_dataset_zero_padded_channels(
    tmp_path, n_sites=3, n_channels=2, h=64, w=64
):
    """Like the above but channels are 2-digit zero-padded (ch01, ch02, ...).

    Regression fixture: ensures the channel group '01' yields column 'ch01'
    (verbatim capture — no leading-zero stripping).
    """
    for site in range(n_sites):
        row = site + 1
        col = site + 1
        field = 1
        stack = 1
        for ch in range(1, n_channels + 1):
            arr = np.random.randint(0, 65535, size=(h, w), dtype=np.uint16)
            arr[:] = ch * 1000
            fname = f"r{row:02d}c{col:02d}f{field:02d}p{stack:02d}-ch{ch:02d}.tiff"
            imwrite(str(tmp_path / fname), arr)
    return tmp_path


def _make_one_channel_per_file_dataset_with_masks(tmp_path, n_sites=2, h=64, w=64):
    """Build a dataset with one file per channel per site + a mask per site."""
    for site in range(n_sites):
        row = site + 1
        col = site + 1
        for ch in (1, 2):
            arr = np.random.randint(0, 65535, size=(h, w), dtype=np.uint16)
            arr[:] = ch * 1000
            fname = f"r{row:02d}c{col:02d}f01p01-ch{ch}.tiff"
            imwrite(str(tmp_path / fname), arr)
        # Mask with 3 cells: 10x10 blocks at different positions
        mask = np.zeros((h, w), dtype=np.int32)
        for i in range(3):
            y = (i * 15) % (h - 10)
            x = (i * 20) % (w - 10)
            mask[y:y+10, x:x+10] = i + 1
        mfname = f"r{row:02d}c{col:02d}f01p01-ch1_cp_masks_cell.png"
        Image.fromarray(mask.astype(np.uint16)).save(str(tmp_path / mfname))
    return tmp_path


def _make_multi_channel_per_file_dataset(tmp_path, n_sites=3, n_channels=2, h=64, w=64):
    """Build a dataset with one multi-page TIFF per site (CHW layout)."""
    for site in range(n_sites):
        row = site + 1
        col = site + 1
        arr = np.zeros((n_channels, h, w), dtype=np.uint16)
        for ch in range(n_channels):
            arr[ch] = (ch + 1) * 1000
        fname = f"r{row:02d}c{col:02d}f01p01.tiff"
        imwrite(str(tmp_path / fname), arr)
    return tmp_path


def _make_one_channel_per_file_dataset_in_subdir(tmp_path, n_sites=3, n_channels=2, h=64, w=64):
    """Build a dataset with images inside Images/r{row}c{col}/ subdirectories.

    Mirrors the Operetta layout: Images/<well>/<filename>.tiff
    Filenames: r{row}c{col}f{field}p{stack}-ch{channel}.tiff
    """
    for site in range(n_sites):
        row = site + 1
        col = site + 1
        field = 1
        stack = 1
        well_dir = tmp_path / "Images" / f"r{row:02d}c{col:02d}"
        well_dir.mkdir(parents=True, exist_ok=True)
        for ch in range(1, n_channels + 1):
            arr = np.random.randint(0, 65535, size=(h, w), dtype=np.uint16)
            arr[:] = ch * 1000
            fname = f"r{row:02d}c{col:02d}f{field:02d}p{stack:02d}-ch{ch}.tiff"
            imwrite(str(well_dir / fname), arr)
    return tmp_path


# ---- Construction tests ----

def test_image_dataset_one_channel_per_file_basic(tmp_path):
    _make_one_channel_per_file_dataset(tmp_path, n_sites=3, n_channels=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    assert len(ds) == 3
    assert ds.intensity_colnames == ["ch1", "ch2"]
    assert ds.img_shape == (64, 64)


def test_image_dataset_captured_fields_public_property(tmp_path):
    """captured_fields exposes the regex-captured metadata columns.

    The `channel` capture is consumed into the intensity-column mapping, so
    only the structural captures remain; intensity/mask cols are excluded.
    """
    _make_one_channel_per_file_dataset(tmp_path, n_sites=3, n_channels=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    assert ds.captured_fields == {"row", "col", "field", "stack"}
    assert isinstance(ds.captured_fields, set)
    assert ds.captured_fields.isdisjoint(ds.intensity_colnames)


def test_image_dataset_one_channel_per_file_preserves_zero_padded_channels(tmp_path):
    """Regex capture of '01'/'02' is kept verbatim -> intensity columns 'ch01', 'ch02'.

    All metadata is stored as TEXT from extraction through DB storage, so
    leading zeros are preserved (no coercion to int). The column name is
    'ch' + the raw captured string.
    """
    _make_one_channel_per_file_dataset_zero_padded_channels(
        tmp_path, n_sites=3, n_channels=2
    )
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    assert len(ds) == 3
    assert ds.intensity_colnames == ["ch01", "ch02"]


def test_image_dataset_one_channel_per_file_preserves_zero_padded_field(tmp_path):
    """A regex capture of '00001' for field is kept verbatim as the string '00001'."""
    # Filenames with zero-padded field '00001'
    for site in range(2):
        row, col = 1, 1
        for ch in (1, 2):
            arr = np.zeros((64, 64), dtype=np.uint16)
            arr[:] = ch * 1000
            fname = f"r{row:02d}c{col:02d}f{1:05d}p{1:03d}-ch{ch}.tiff"
            imwrite(str(tmp_path / fname), arr)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    df = ds.metadata
    assert df["field"].iloc[0] == "00001"
    assert df["stack"].iloc[0] == "001"


def test_image_dataset_well_derivation(tmp_path):
    """row + col -> well, drop row/col from metadata."""
    _make_one_channel_per_file_dataset(tmp_path, n_sites=3, n_channels=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    df = ds.metadata
    assert "well" in df.columns
    assert "row" not in df.columns
    assert "col" not in df.columns
    assert df["well"].tolist() == ["A1", "B2", "C3"]


def test_image_dataset_with_masks(tmp_path):
    _make_one_channel_per_file_dataset_with_masks(tmp_path, n_sites=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>.+)\.png"
        ),
        channel_layout=None,
    )
    assert len(ds) == 2
    assert "mask_cell" in ds.mask_colnames


def test_image_dataset_multi_channel_per_file_chw(tmp_path):
    _make_multi_channel_per_file_dataset(tmp_path, n_sites=3, n_channels=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)\.tiff"
        ),
        channel_layout="CHW",
    )
    assert len(ds) == 3
    assert ds.intensity_colnames == ["ch1", "ch2"]
    assert ds.img_shape == (64, 64)


def test_image_dataset_channel_layout_conflict_raises(tmp_path):
    """channel_layout + channel-grouped pattern should error."""
    _make_one_channel_per_file_dataset(tmp_path, n_sites=1, n_channels=1)
    with pytest.raises(ConfigError):
        ImageDataset(
            root=tmp_path,
            image_pattern=re.compile(r"r(?P<row>\d+)c(?P<col>\d+)-ch(?P<channel>\d+)\.tiff"),
            channel_layout="CHW",
        )


def test_image_dataset_no_channel_group_no_layout_implicit_ch1(tmp_path):
    """channel_layout=None + no channel group -> treat each file as implicit 'ch1'.

    This supports non-standard single-channel datasets (e.g. HPA) where every
    file is one image with no per-channel split. A warning is logged.
    """
    # Create single-channel TIFFs with simple names matching the pattern
    for site in range(2):
        row, col = site + 1, site + 1
        arr = np.zeros((64, 64), dtype=np.uint16)
        arr[:] = (site + 1) * 1000
        fname = f"r{row:02d}c{col:02d}.tiff"
        imwrite(str(tmp_path / fname), arr)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(r"r(?P<row>\d+)c(?P<col>\d+)\.tiff"),
        channel_layout=None,
    )
    assert len(ds) == 2
    assert ds.intensity_colnames == ["ch1"]


def test_image_dataset_root_not_found_raises(tmp_path):
    with pytest.raises(DatasetError):
        ImageDataset(
            root=tmp_path / "nonexistent",
            image_pattern=re.compile(r".*"),
            channel_layout=None,
        )


def test_image_subdir_pattern_as_directory_scans_recursively(tmp_path):
    """image_subdir_pattern='Images/' should scan tiff files within the Images/
    directory tree, not just yield the directory itself.

    Regression: glob('Images/') returns the directory, and p.is_file() is False,
    so zero image files were found -> zero channels detected.
    """
    _make_one_channel_per_file_dataset_in_subdir(tmp_path, n_sites=3, n_channels=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
        image_subdir_pattern="Images/",
    )
    assert len(ds) == 3
    assert ds.intensity_colnames == ["ch1", "ch2"]
    assert ds.img_shape == (64, 64)


def test_image_subdir_pattern_as_glob_scans_files(tmp_path):
    """image_subdir_pattern='Images/*/*.tiff' should also work (files matched directly)."""
    _make_one_channel_per_file_dataset_in_subdir(tmp_path, n_sites=2, n_channels=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
        image_subdir_pattern="Images/*/*.tiff",
    )
    assert len(ds) == 2
    assert ds.intensity_colnames == ["ch1", "ch2"]


# ---- Image access tests ----

def test_get_imageset_one_channel_per_file(tmp_path):
    _make_one_channel_per_file_dataset(tmp_path, n_sites=2, n_channels=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    img, masks = ds.get_imageset(0)
    assert img.shape == (64, 64, 2)
    # ch1 -> 1000, ch2 -> 2000
    assert img[0, 0, 0] == 1000
    assert img[0, 0, 1] == 2000
    assert masks == {}  # no masks requested


def test_get_imageset_with_masks(tmp_path):
    _make_one_channel_per_file_dataset_with_masks(tmp_path, n_sites=1)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>.+)\.png"
        ),
        channel_layout=None,
    )
    img, masks = ds.get_imageset(0, masks=["mask_cell"])
    assert img.shape == (64, 64, 2)
    assert "mask_cell" in masks
    assert masks["mask_cell"].shape == (64, 64)
    # 3 cells in the mask
    assert set(np.unique(masks["mask_cell"]).tolist()) - {0} == {1, 2, 3}


def test_get_imageset_caches(tmp_path):
    """Repeated calls should hit the LRU cache."""
    _make_one_channel_per_file_dataset(tmp_path, n_sites=1, n_channels=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    img1, _ = ds.get_imageset(0)
    img2, _ = ds.get_imageset(0)
    # Cached call returns the same array object
    assert img1 is img2


def test_image_dataset_pickle_roundtrip(tmp_path):
    """ImageDataset must survive a pickle round-trip (DataLoader worker spawn).

    Regression: _LRUCache held a threading.Lock, which is unpicklable — a
    torch DataLoader with num_workers>0 crashed with "cannot pickle
    '_thread.lock' object" on Windows (multiprocessing spawn).
    """
    _make_one_channel_per_file_dataset_with_masks(tmp_path, n_sites=1)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>.+)\.png"
        ),
        channel_layout=None,
    )
    # Warm the cache so the pickle must drop cached arrays, not ship them.
    img0, _ = ds.get_imageset(0)

    ds2 = pickle.loads(pickle.dumps(ds))

    # Unpickled copy is fully functional: images, masks, cropping.
    img, masks = ds2.get_imageset(0, masks=["mask_cell"])
    assert img.shape == (64, 64, 2)
    assert set(np.unique(masks["mask_cell"]).tolist()) - {0} == {1, 2, 3}
    crop, cell_mask, bbox = ds2.get_cropped_cell(0, label=1, mask_name="mask_cell", padding=2)
    assert crop.ndim == 3
    assert cell_mask.ndim == 2

    # Original is unaffected and its cache still works.
    img_again, _ = ds.get_imageset(0)
    assert img_again is img0


def test_get_imageset_multi_channel_per_file(tmp_path):
    _make_multi_channel_per_file_dataset(tmp_path, n_sites=1, n_channels=3)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)\.tiff"),
        channel_layout="CHW",
    )
    img, _ = ds.get_imageset(0)
    assert img.shape == (64, 64, 3)
    assert img[0, 0, 0] == 1000
    assert img[0, 0, 1] == 2000
    assert img[0, 0, 2] == 3000


# ---- Cropping tests ----

def test_get_cropped_cell(tmp_path):
    _make_one_channel_per_file_dataset_with_masks(tmp_path, n_sites=1)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>.+)\.png"
        ),
        channel_layout=None,
    )
    crop, cell_mask, bbox = ds.get_cropped_cell(0, label=1, mask_name="mask_cell", padding=2)
    assert crop.ndim == 3
    assert cell_mask.ndim == 2
    assert len(bbox) == 4
    # Cell 1 is at (0,0)-(9,9) with padding 2 -> (0,0)-(11,11) -> 12x12
    assert crop.shape[0] >= 10
    assert crop.shape[1] >= 10


def test_get_cropped_cells(tmp_path):
    _make_one_channel_per_file_dataset_with_masks(tmp_path, n_sites=1)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>.+)\.png"
        ),
        channel_layout=None,
    )
    results = ds.get_cropped_cells(0, mask_name="mask_cell", padding=2)
    assert len(results) == 3  # 3 cells in the mask
    for crop, cell_mask, bbox, cid in results:
        assert cid in (1, 2, 3)
        assert crop.ndim == 3


def test_filter_metadata(tmp_path):
    _make_one_channel_per_file_dataset(tmp_path, n_sites=5, n_channels=1)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    assert len(ds) == 5
    ds.filter_metadata("well", "A1")
    assert len(ds) == 1
    assert ds.metadata["well"].iloc[0] == "A1"


def test_filter_metadata_bad_column_raises(tmp_path):
    _make_one_channel_per_file_dataset(tmp_path, n_sites=1, n_channels=1)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    with pytest.raises(ConfigError):
        ds.filter_metadata("nonexistent_col", ".*")


def test_image_dataset_non_numeric_channel_no_ch_prefix(tmp_path):
    """Non-numeric channel captures stay as-is (no 'ch' prefix).

    HPA-style channels: 'blue', 'green', 'red', 'yellow' become column names
    directly, NOT 'chblue', 'chgreen', etc.
    """
    for ch_name in ("blue", "green", "red", "yellow"):
        arr = np.zeros((32, 32), dtype=np.uint16)
        arr[:] = 1000
        fname = f"000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_{ch_name}.png"
        Image.fromarray(arr.astype(np.uint16)).save(str(tmp_path / fname))
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(r"(?P<field>.+)_(?P<channel>.+)\.png"),
        channel_layout=None,
    )
    assert ds.intensity_colnames == ["blue", "green", "red", "yellow"]


def test_image_dataset_mask_files_excluded_from_images(tmp_path):
    """Files matching mask_pattern must NOT appear as intensity channels.

    HPA-style: both images and masks are .png. Without the mask-exclusion
    check, '_mask_cell.png' would match image_pattern and become a bogus
    intensity column.
    """
    # Image files (blue, green channels)
    for ch_name in ("blue", "green"):
        arr = np.zeros((32, 32), dtype=np.uint16)
        arr[:] = 1000
        fname = f"000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_{ch_name}.png"
        Image.fromarray(arr.astype(np.uint16)).save(str(tmp_path / fname))
    # Mask files — would also match image_pattern if not excluded
    for mname in ("cell", "nuclei"):
        mask = np.zeros((32, 32), dtype=np.uint16)
        mask[5:15, 5:15] = 1
        fname = f"000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_mask_{mname}.png"
        Image.fromarray(mask).save(str(tmp_path / fname))

    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(r"(?P<field>.+)_(?P<channel>.+)\.png"),
        mask_pattern=re.compile(r"(?P<field>.+)_mask_(?P<mask_name>.+)\.png"),
        channel_layout=None,
    )
    # Intensity columns should be blue and green — NOT mask_cell, mask_nuclei
    assert sorted(ds.intensity_colnames) == ["blue", "green"]
    # Masks should be detected correctly
    assert sorted(ds.mask_colnames) == ["mask_cell", "mask_nuclei"]
    # Only 1 image group (1 unique field)
    assert len(ds) == 1


def test_image_dataset_mask_name_missing_raises(tmp_path):
    """mask_pattern without mask_name group raises DatasetError."""
    _make_one_channel_per_file_dataset(tmp_path, n_sites=1, n_channels=1)
    # Mask pattern without mask_name group
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[5:15, 5:15] = 1
    Image.fromarray(mask).save(str(tmp_path / "r01c01f01p01-ch1_mask.png"))

    with pytest.raises(DatasetError, match="mask_name"):
        ImageDataset(
            root=tmp_path,
            image_pattern=re.compile(
                r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
            ),
            mask_pattern=re.compile(
                r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_mask\.png"
            ),
            channel_layout=None,
        )


def test_image_dataset_explicit_well_natsorted(tmp_path):
    """Explicit-`well` datasets sort natsorted (A2 before A10), not lexicographic."""
    for well in ("A10", "A2", "A1"):
        arr = np.zeros((32, 32), dtype=np.uint16)
        arr[:] = 1000
        imwrite(str(tmp_path / f"{well}-ch1.tiff"), arr)

    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(r"(?P<well>A\d+)-ch(?P<channel>\d+)\.tiff"),
        channel_layout=None,
    )
    wells = list(ds.metadata["well"])
    # Lexicographic would give A1, A10, A2 — natsorted gives A1, A2, A10.
    assert wells == ["A1", "A2", "A10"]


def test_get_imageset_mask_only_row_raises(tmp_path):
    """A row whose image files are missing (mask-only) must raise
    ImageReadError with a clear message, not crash with a raw TypeError."""
    _make_one_channel_per_file_dataset_with_masks(tmp_path, n_sites=2)
    # Delete site 2's image files, keeping its mask
    for ch in (1, 2):
        (tmp_path / f"r02c02f01p01-ch{ch}.tiff").unlink()
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>\w+)\.png"
        ),
        channel_layout=None,
    )
    # Row 0 (site 1) is intact
    img, masks = ds.get_imageset(0)
    assert img.shape == (64, 64, 2)
    assert "mask_cell" in masks
    # Row 1 (site 2) is mask-only -> ImageReadError for quarantine
    with pytest.raises(ImageReadError):
        ds.get_imageset(1)


def test_get_imageset_multi_channel_mask_only_first_row(tmp_path):
    """Multi-channel-per-file: auto-detect skips mask-only rows and the first
    real image defines shape/channels; get_imageset on the mask-only row
    raises ImageReadError."""
    # Site 1: mask only (no image)
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[5:15, 5:15] = 1
    Image.fromarray(mask).save(str(tmp_path / "r01c01f01p01_ch1_cp_masks_cell.png"))
    # Site 2: full CHW image + mask
    arr = np.zeros((2, 32, 32), dtype=np.uint16)
    arr[0] = 1000
    arr[1] = 2000
    imwrite(str(tmp_path / "r02c01f01p01.tiff"), arr)
    mask2 = np.zeros((32, 32), dtype=np.uint16)
    mask2[10:20, 10:20] = 1
    Image.fromarray(mask2).save(str(tmp_path / "r02c01f01p01_ch1_cp_masks_cell.png"))

    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)\.tiff"),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)_ch(?P<channel>\d+)_cp_masks_(?P<mask_name>\w+)\.png"
        ),
        channel_layout="CHW",
    )
    # Auto-detect found properties from row 1 (first row with an image)
    assert ds.img_shape == (32, 32)
    assert ds.intensity_colnames == ["ch1", "ch2"]
    # Mask-only row (index 0) raises for quarantine
    with pytest.raises(ImageReadError):
        ds.get_imageset(0)
    # Intact row works
    img, masks = ds.get_imageset(1)
    assert img.shape == (32, 32, 2)
    assert "mask_cell" in masks


def test_image_path_unknown_channel_raises(tmp_path):
    """image_path with a channel not in the metadata must raise DataError."""
    _make_one_channel_per_file_dataset(tmp_path, n_sites=1, n_channels=2)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    with pytest.raises(DataError):
        ds.image_path(0, "ch99")


def test_image_path_missing_file_raises(tmp_path):
    """image_path for a NaN (mask-only/deleted) row must raise ImageReadError."""
    _make_one_channel_per_file_dataset_with_masks(tmp_path, n_sites=2)
    for ch in (1, 2):
        (tmp_path / f"r02c02f01p01-ch{ch}.tiff").unlink()
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>\w+)\.png"
        ),
        channel_layout=None,
    )
    with pytest.raises(ImageReadError):
        ds.image_path(1, "ch1")


def test_filter_metadata_invalid_regex_raises(tmp_path):
    """An invalid filter regex must raise ConfigError, never a raw re.error."""
    _make_one_channel_per_file_dataset(tmp_path, n_sites=2, n_channels=1)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    with pytest.raises(ConfigError):
        ds.filter_metadata("field", "(")


# ---- quarantine path (missing/unreadable files) ----


def test_get_imageset_raises_on_mask_only_row(tmp_path):
    """Mask-only rows raise ImageReadError (the pipeline quarantine path)."""
    _make_one_channel_per_file_dataset_with_masks(tmp_path, n_sites=2)
    for ch in (1, 2):
        (tmp_path / f"r02c02f01p01-ch{ch}.tiff").unlink()
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>\w+)\.png"
        ),
        channel_layout=None,
    )
    with pytest.raises(ImageReadError):
        ds.get_imageset(1)
    # The intact row still loads.
    img, masks = ds.get_imageset(0)
    assert img.shape == (64, 64, 2)
    assert "mask_cell" in masks


def test_auto_detect_skips_broken_first_image(tmp_path):
    """A corrupt first file must not abort dataset construction: auto-detect
    skips it and takes shape/dtype from the next readable row."""
    (tmp_path / "site1.tiff").write_bytes(b"not a real tiff")
    arr = np.full((32, 48), 7, dtype=np.uint16)
    imwrite(str(tmp_path / "site2.tiff"), arr)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(r"site(?P<site>\d+)\.tiff"),
        channel_layout=None,
    )
    assert len(ds) == 2
    assert ds.img_shape == (32, 48)
    assert ds.img_dtype == arr.dtype
    with pytest.raises(ImageReadError):
        ds.get_imageset(0)


# ---- metadata rebuild cache invalidation ----


def test_build_metadata_clears_stale_cache(tmp_path):
    """build_metadata() must drop cached (image, mask) tuples.

    cellpose.py rebuilds metadata after saving masks; without the cache
    clear, get_imageset kept serving the pre-segmentation image/mask set.
    """
    _make_one_channel_per_file_dataset(tmp_path, n_sites=1, n_channels=1)
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        channel_layout=None,
    )
    img_before, _ = ds.get_imageset(0)
    assert img_before[0, 0] == 1000

    # Replace the on-disk image, then rebuild metadata (new content is the
    # only difference — paths and row order stay identical).
    arr = np.full((64, 64), 2000, dtype=np.uint16)
    imwrite(str(tmp_path / "r01c01f01p01-ch1.tiff"), arr)
    ds.build_metadata()

    img_after, _ = ds.get_imageset(0)
    assert img_after[0, 0] == 2000


def test_build_metadata_picks_up_new_mask(tmp_path):
    """A mask written after construction is visible after build_metadata()."""
    _make_one_channel_per_file_dataset(tmp_path, n_sites=1, n_channels=1)
    img_pat = re.compile(
        r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
    )
    mask_pat = re.compile(
        r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>.+)\.png"
    )
    ds = ImageDataset(
        root=tmp_path, image_pattern=img_pat, mask_pattern=mask_pat,
        channel_layout=None,
    )
    _, masks_before = ds.get_imageset(0)
    assert masks_before == {}

    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:20, 10:20] = 1
    Image.fromarray(mask.astype(np.uint16)).save(
        str(tmp_path / "r01c01f01p01-ch1_cp_masks_cell.png")
    )
    ds.build_metadata()
    _, masks_after = ds.get_imageset(0)
    assert "mask_cell" in masks_after


def test_mask_in_separate_directory_keeps_image_directory(tmp_path):
    """Mask rows must not overwrite the image record's directory.

    Masks may live in a sibling subtree (e.g. masks/ vs images/); the
    metadata `directory` must keep pointing at the images.
    """
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for ch in (1, 2):
        arr = np.full((32, 32), ch * 100, dtype=np.uint16)
        imwrite(str(images / f"r01c01f01p01-ch{ch}.tiff"), arr)
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[5:10, 5:10] = 1
    Image.fromarray(mask.astype(np.uint16)).save(
        str(masks / "r01c01f01p01-ch1_cp_masks_cell.png")
    )
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>.+)\.png"
        ),
        image_subdir_pattern="images",
        channel_layout=None,
    )
    assert len(ds) == 1
    assert ds.metadata.loc[0, "directory"] == "images"
    img, mask_dict = ds.get_imageset(0)
    assert img.shape == (32, 32, 2)
    assert "mask_cell" in mask_dict


def test_get_imageset_no_intensity_channels_raises(tmp_path):
    """A mask-only dataset (no image matched image_pattern) must raise
    ImageReadError instead of a raw numpy ValueError."""
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[5:10, 5:10] = 1
    Image.fromarray(mask.astype(np.uint16)).save(
        str(tmp_path / "r01c01f01p01-ch1_cp_masks_cell.png")
    )
    ds = ImageDataset(
        root=tmp_path,
        image_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
        ),
        mask_pattern=re.compile(
            r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)_cp_masks_(?P<mask_name>.+)\.png"
        ),
        channel_layout=None,
    )
    assert len(ds) == 1
    assert ds.intensity_colnames == []
    with pytest.raises(ImageReadError):
        ds.get_imageset(0)
