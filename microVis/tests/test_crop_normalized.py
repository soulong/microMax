"""Unit tests for the Data-page popup's SELF-normalized cell rendering.

crop_cell_rgb_normalized must render each channel from the cell's OWN
percentile range (0 background ignored), so absolute intensity changes
leave the pixels untouched while gamma still shapes the display.
"""

import numpy as np

from microVis.worker import crop_cell_rgb_normalized


def _scene():
    """16x16 two-channel scene; object label 1 is a textured 8x8 block."""
    img = np.zeros((16, 16, 2), dtype=np.uint16)
    mask = np.zeros((16, 16), dtype=np.uint32)
    mask[4:12, 4:12] = 1
    # Channel 0: rising texture values inside the object; one outlier keeps
    # the 99.9 percentile below the max, so gamma has mid-tones to shape.
    img[4:12, 4:12, 0] = np.arange(1, 65).reshape(8, 8) * 10
    img[6, 6, 0] = 5000
    # Channel 1: a flat mid-level signal inside the object.
    img[4:12, 4:12, 1] = 200
    return img, mask


_CFG = {
    "a": {"enabled": True, "color": "red", "vmin": 0, "vmax": 65535},
    "b": {"enabled": True, "color": "green", "vmin": 0, "vmax": 65535},
}


def test_background_stays_zero_and_object_renders():
    img, mask = _scene()
    rgb = crop_cell_rgb_normalized(img, mask, 1, ["a", "b"], _CFG,
                                   0.1, 99.9, 1.0)
    assert rgb is not None and rgb.ndim == 3 and rgb.shape[2] == 3
    # Pixels far outside the object stay black.
    assert rgb[0, 0].tolist() == [0, 0, 0]
    # The textured object renders something visible.
    assert rgb.max() > 0


def test_absolute_intensity_independence():
    """Scaling a channel must not change the rendering (vs absolute ranges)."""
    img, mask = _scene()
    base = crop_cell_rgb_normalized(img, mask, 1, ["a", "b"], _CFG,
                                    0.1, 99.9, 1.0)
    # float64 scene (no uint16 overflow) scaled far beyond the original
    # range — an absolute display range would saturate, percentiles don't.
    big = img.astype(np.float64) * 1000
    scaled = crop_cell_rgb_normalized(big, mask, 1, ["a", "b"], _CFG,
                                      0.1, 99.9, 1.0)
    # Percentiles scale with the factor, so the display is identical up to
    # at most a 1-gray-level float rounding difference.
    diff = np.abs(base.astype(np.int16) - scaled.astype(np.int16))
    assert diff.max() <= 1


def test_gamma_darkens_midtones():
    img, mask = _scene()
    g1 = crop_cell_rgb_normalized(img, mask, 1, ["a", "b"], _CFG,
                                  0.1, 99.9, 1.0)
    g2 = crop_cell_rgb_normalized(img, mask, 1, ["a", "b"], _CFG,
                                  0.1, 99.9, 2.0)
    assert g1.shape == g2.shape
    # Gamma 2 darkens every lit pixel of channel 0 (values in [0, 1] shrink).
    lit = g1[..., 0] > 0
    assert lit.any()
    assert np.all(g2[..., 0][lit] <= g1[..., 0][lit])
    assert g2[..., 0][lit].mean() < g1[..., 0][lit].mean()


def test_disabled_channel_and_missing_label():
    img, mask = _scene()
    cfg = dict(_CFG)
    cfg["a"] = {**cfg["a"], "enabled": False}
    rgb = crop_cell_rgb_normalized(img, mask, 1, ["a", "b"], cfg,
                                   0.1, 99.9, 1.0)
    # The disabled channel contributes nothing: every lit pixel is pure green.
    lit = rgb.max(axis=2) > 0
    assert lit.any()
    assert np.all(rgb[lit][:, 0] == 0)
    assert np.all(rgb[lit][:, 2] == 0)
    # A label absent from the mask resolves to no image.
    assert crop_cell_rgb_normalized(img, mask, 99, ["a", "b"], _CFG,
                                    0.1, 99.9, 1.0) is None


def test_empty_channel_stays_black():
    img, mask = _scene()
    img[..., 1] = 0
    rgb = crop_cell_rgb_normalized(img, mask, 1, ["a", "b"], _CFG,
                                   0.1, 99.9, 1.0)
    assert np.all(rgb[:, :, 1] == 0)
    assert rgb[:, :, 0].max() > 0
