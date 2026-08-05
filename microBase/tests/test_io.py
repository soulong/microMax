"""Tests for microBase.io: TIFF and mask readers/writers."""

import numpy as np
import pytest

from microBase import io as mio


def _make_tiff(tmp_path, name, arr):
    """Write a TIFF using tifffile directly (no microBase code)."""
    from tifffile import imwrite
    imwrite(str(tmp_path / name), arr)


def test_read_tiff_single_channel_2d(tmp_path):
    arr = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
    _make_tiff(tmp_path, "img.tif", arr)
    out = mio.read_tiff(tmp_path / "img.tif")
    assert out.shape == (64, 64)
    assert out.dtype == arr.dtype
    np.testing.assert_array_equal(out, arr)


def test_read_tiff_squeezes_3d_single_channel(tmp_path):
    """A (1, H, W) TIFF should be squeezed to (H, W)."""
    arr = np.random.randint(0, 255, size=(1, 32, 32), dtype=np.uint8)
    _make_tiff(tmp_path, "img.tif", arr)
    out = mio.read_tiff(tmp_path / "img.tif")
    assert out.shape == (32, 32)


def test_read_tiff_channels_chw(tmp_path):
    """CHW layout: pages along axis 0."""
    arr = np.random.randint(0, 255, size=(3, 32, 32), dtype=np.uint8)
    _make_tiff(tmp_path, "img.tif", arr)
    out = mio.read_tiff_channels(tmp_path / "img.tif", [1, 2, 3], channel_layout="CHW")
    assert out.shape == (32, 32, 3)
    # Channel 0 in CHW input should map to channel 0 in HWC output
    np.testing.assert_array_equal(out[:, :, 0], arr[0])


def test_read_tiff_channels_hwc(tmp_path):
    """HWC layout: channels along last axis."""
    arr = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    _make_tiff(tmp_path, "img.tif", arr)
    out = mio.read_tiff_channels(tmp_path / "img.tif", [1, 2, 3], channel_layout="HWC")
    assert out.shape == (32, 32, 3)
    np.testing.assert_array_equal(out, arr)


def test_read_tiff_channels_subset(tmp_path):
    """Request only some channels."""
    arr = np.random.randint(0, 255, size=(3, 32, 32), dtype=np.uint8)
    _make_tiff(tmp_path, "img.tif", arr)
    out = mio.read_tiff_channels(tmp_path / "img.tif", [1, 3], channel_layout="CHW")
    assert out.shape == (32, 32, 2)
    np.testing.assert_array_equal(out[:, :, 0], arr[0])
    np.testing.assert_array_equal(out[:, :, 1], arr[2])


def test_read_mask_png(tmp_path):
    from PIL import Image
    arr = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint16)
    Image.fromarray(arr).save(str(tmp_path / "mask.png"))
    out = mio.read_mask(tmp_path / "mask.png")
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, arr)


def test_write_tiff_single_channel(tmp_path):
    arr = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
    mio.write_tiff(tmp_path / "out.tif", arr)
    out = mio.read_tiff(tmp_path / "out.tif")
    np.testing.assert_array_equal(out, arr)


def test_write_tiff_chw_roundtrip(tmp_path):
    """Write multi-channel as CHW, read back as CHW."""
    arr = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    mio.write_tiff(tmp_path / "out.tif", arr, channel_layout="CHW")
    out = mio.read_tiff_channels(tmp_path / "out.tif", [1, 2, 3], channel_layout="CHW")
    assert out.shape == (32, 32, 3)  # (H, W, C)
    np.testing.assert_array_equal(out, arr)


def test_write_mask(tmp_path):
    arr = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint16)
    mio.write_mask(tmp_path / "out.png", arr)
    out = mio.read_mask(tmp_path / "out.png")
    np.testing.assert_array_equal(out, arr)


def test_read_tiff_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        mio.read_tiff(tmp_path / "nonexistent.tif")


def test_read_tiff_channels_bad_layout_exits(tmp_path):
    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    _make_tiff(tmp_path, "img.tif", arr)
    with pytest.raises(SystemExit):
        mio.read_tiff_channels(tmp_path / "img.tif", [1], channel_layout="BAD")


def test_read_tiff_channels_none_2d(tmp_path):
    """channel_layout=None: 2D TIFF read as (H, W, 1)."""
    arr = np.random.randint(0, 65535, size=(48, 64), dtype=np.uint16)
    _make_tiff(tmp_path, "img.tif", arr)
    out = mio.read_tiff_channels(tmp_path / "img.tif", [1], channel_layout=None)
    assert out.shape == (48, 64, 1)
    assert out.dtype == arr.dtype
    np.testing.assert_array_equal(out[:, :, 0], arr)


def test_read_tiff_channels_none_squeezes_singleton_dim(tmp_path):
    """channel_layout=None: 3D (1, H, W) and (H, W, 1) also become (H, W, 1)."""
    arr_chw = np.random.randint(0, 255, size=(1, 32, 32), dtype=np.uint8)
    _make_tiff(tmp_path, "chw.tif", arr_chw)
    out1 = mio.read_tiff_channels(tmp_path / "chw.tif", [1], channel_layout=None)
    assert out1.shape == (32, 32, 1)

    arr_hwc = np.random.randint(0, 255, size=(32, 32, 1), dtype=np.uint8)
    _make_tiff(tmp_path, "hwc.tif", arr_hwc)
    out2 = mio.read_tiff_channels(tmp_path / "hwc.tif", [1], channel_layout=None)
    assert out2.shape == (32, 32, 1)


def test_read_tiff_channels_none_rejects_multi_channel_request(tmp_path):
    """channel_layout=None: requesting channels=[1,2] hard-exits."""
    arr = np.zeros((32, 32), dtype=np.uint8)
    _make_tiff(tmp_path, "img.tif", arr)
    with pytest.raises(SystemExit):
        mio.read_tiff_channels(tmp_path / "img.tif", [1, 2], channel_layout=None)


def test_read_tiff_channels_none_rejects_3d_multi_page(tmp_path):
    """channel_layout=None: a (3, H, W) TIFF hard-exits (not single-channel)."""
    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    _make_tiff(tmp_path, "img.tif", arr)
    with pytest.raises(SystemExit):
        mio.read_tiff_channels(tmp_path / "img.tif", [1], channel_layout=None)


# ---- compile_pattern / detect_tiff_properties ----


def test_compile_pattern_passes_compiled_and_none():
    import re
    assert mio.compile_pattern(None) is None
    pat = re.compile(r"x")
    assert mio.compile_pattern(pat) is pat
    out = mio.compile_pattern(r"r\d")
    assert isinstance(out, re.Pattern)
    assert out.match("r1") is not None


def test_compile_pattern_invalid_exits():
    with pytest.raises(SystemExit):
        mio.compile_pattern("(")


def test_detect_tiff_properties_chw(tmp_path):
    arr = np.random.randint(0, 65535, size=(3, 32, 32), dtype=np.uint16)
    _make_tiff(tmp_path, "img.tif", arr)
    shape, dtype, n_ch = mio.detect_tiff_properties(tmp_path / "img.tif", "CHW")
    assert shape == (32, 32)
    assert dtype == arr.dtype
    assert n_ch == 3


def test_detect_tiff_properties_hwc(tmp_path):
    arr = np.random.randint(0, 255, size=(32, 32, 2), dtype=np.uint8)
    _make_tiff(tmp_path, "img.tif", arr)
    shape, dtype, n_ch = mio.detect_tiff_properties(tmp_path / "img.tif", "HWC")
    assert shape == (32, 32)
    assert dtype == arr.dtype
    assert n_ch == 2
