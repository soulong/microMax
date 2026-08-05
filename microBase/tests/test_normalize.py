"""Tests for microBase.normalize: per_channel / global / null z-score."""

import numpy as np
import pytest

from microBase.normalize import normalize


def test_normalize_null_returns_clipped():
    """method='null' applies percentile clipping only, no z-score."""
    img = np.zeros((10, 10, 2), dtype=np.float32)
    img[:, :, 0] = np.linspace(0, 100, 100).reshape(10, 10)
    img[:, :, 1] = 50
    out = normalize(img, method="null", clip_low=0, clip_high=100)
    assert out.shape == (10, 10, 2)
    assert out.dtype == np.float32
    # Channel 1 was constant (50) -> degenerate, returns zeros
    assert np.all(out[:, :, 1] == 0)


def test_normalize_per_channel():
    """per_channel: independent z-score per channel."""
    img = np.random.randn(20, 20, 3).astype(np.float32) * 100 + 50
    out = normalize(img, method="per_channel", clip_low=0, clip_high=100)
    assert out.shape == (20, 20, 3)
    # Each channel should be approximately zero-mean
    for ch in range(3):
        assert abs(out[:, :, ch].mean()) < 0.5


def test_normalize_global():
    """global: single z-score across all channels."""
    img = np.random.randn(20, 20, 3).astype(np.float32) * 100 + 50
    out = normalize(img, method="global", clip_low=0, clip_high=100)
    assert out.shape == (20, 20, 3)
    # All channels share the same mean/std
    assert abs(out.mean()) < 0.1


def test_normalize_with_mask_zeros_background():
    """When mask is provided, background pixels are zeroed."""
    img = np.ones((10, 10, 1), dtype=np.float32) * 5
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    out = normalize(img, mask=mask, method="per_channel", clip_low=0, clip_high=100)
    # Background (mask False) should be zero
    assert np.all(out[0, 0, :] == 0)
    assert np.all(out[9, 9, :] == 0)
    # Foreground (mask True) — constant region -> degenerate -> zeros
    # (since std is 0 for constant channel)


def test_normalize_per_channel_constant_returns_zeros():
    """Constant channel -> std=0 -> returns zeros."""
    img = np.ones((10, 10, 1), dtype=np.float32) * 42
    out = normalize(img, method="per_channel", clip_low=0, clip_high=100)
    assert np.all(out == 0)


def test_normalize_global_constant_returns_zeros():
    img = np.ones((10, 10, 3), dtype=np.float32) * 42
    out = normalize(img, method="global", clip_low=0, clip_high=100)
    assert np.all(out == 0)


def test_normalize_bad_shape_exits():
    img = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(SystemExit):
        normalize(img, method="per_channel")


def test_normalize_bad_method_exits():
    img = np.zeros((10, 10, 1), dtype=np.float32)
    with pytest.raises(SystemExit):
        normalize(img, method="bad_method")


def test_normalize_none_method_treated_as_null():
    img = np.random.randn(10, 10, 2).astype(np.float32)
    out1 = normalize(img, method=None, clip_low=0, clip_high=100)
    out2 = normalize(img, method="null", clip_low=0, clip_high=100)
    np.testing.assert_array_equal(out1, out2)


def test_normalize_clipping_low_high():
    """Outliers below clip_low / above clip_high should be clipped."""
    img = np.zeros((100, 100, 1), dtype=np.float32)
    img[:, :, 0] = 50
    img[0, 0, 0] = 0       # below 5th percentile
    img[1, 1, 0] = 1000    # above 95th percentile
    out = normalize(img, method="null", clip_low=5, clip_high=95)
    # The clipped array should not contain the extreme values (they're clipped
    # to the percentile bounds, then normalized to [0, 1])
    assert out[0, 0, 0] >= 0
    assert out[0, 0, 0] <= 1
    assert out[1, 1, 0] >= 0
    assert out[1, 1, 0] <= 1
