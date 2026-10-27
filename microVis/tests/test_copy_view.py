"""Right-click clipboard copy: CopyViewWorker rendering + crop math.

The copy must reuse the EXACT enhancement/composite path of the display,
so a single-channel copy renders only that channel (with its own
color/vmin/vmax) while the global contrast/gamma/invert still apply.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from microVis.worker import CopyViewWorker, crop_rgb_fraction  # noqa: E402

# Two channels, 8x8: ch1 bright on the left half, ch2 bright everywhere.
_IMG = np.zeros((8, 8, 2), dtype=np.uint16)
_IMG[:, :4, 0] = 1000
_IMG[:, :, 1] = 500

_CFG = {
    "ch1": {"enabled": True, "color": "red", "vmin": 0, "vmax": 1000},
    "ch2": {"enabled": True, "color": "green", "vmin": 0, "vmax": 500},
}


class _FakeDM:
    """Minimal DataModule stand-in returning a fixed imageset."""

    def __init__(self, img):
        self._img = img

    def get_imageset(self, row_idx):
        return self._img, {}


def _render(only_channel=None):
    out = []
    worker = CopyViewWorker(
        dm=_FakeDM(_IMG), row_idx=3, channel_names=["ch1", "ch2"],
        ch_config=_CFG, dmax=65535.0, contrast_method="none",
        contrast_gamma=1.0, invert=False, only_channel=only_channel)
    worker.signals.finished.connect(lambda *a: out.append(a))
    worker.run()  # QRunnable.run is plain Python — call it synchronously
    assert out, "worker did not emit finished"
    row_idx, gen, rgb = out[0]
    assert row_idx == 3 and gen == 0
    return rgb


def test_copy_current_composites_enabled_channels():
    rgb = _render(None)
    # ch1 (red, left half) + ch2 (green, everywhere): left is yellow,
    # right is pure green.
    assert rgb[0, 0, 0] == 255 and rgb[0, 0, 1] == 255 and rgb[0, 0, 2] == 0
    assert rgb[0, 6, 0] == 0 and rgb[0, 6, 1] == 255 and rgb[0, 6, 2] == 0


def test_copy_single_channel_keeps_only_that_channel():
    rgb = _render("ch1")
    # Only ch1: red on the left half, black on the right — ch2 must not leak.
    assert rgb[0, 0, 0] == 255 and rgb[0, 0, 1] == 0 and rgb[0, 0, 2] == 0
    assert rgb[0, 6, 0] == 0 and rgb[0, 6, 1] == 0 and rgb[0, 6, 2] == 0
    # Full-frame render at source resolution.
    assert rgb.shape == (8, 8, 3)


def test_crop_rgb_fraction_clamps_to_bounds():
    rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    # Full frame.
    assert crop_rgb_fraction(rgb, (0.0, 0.0, 1.0, 1.0)).shape == (4, 4, 3)
    # Center 2x2 (rows 1-2, cols 1-2).
    crop = crop_rgb_fraction(rgb, (0.25, 0.25, 0.5, 0.5))
    assert crop.shape == (2, 2, 3)
    assert crop[0, 0].tolist() == rgb[1, 1].tolist()
    # Over-large rect clamps to the frame; degenerate rect never returns
    # a zero-size crop.
    assert crop_rgb_fraction(rgb, (-0.5, -0.5, 2.0, 2.0)).shape == (4, 4, 3)
    assert crop_rgb_fraction(rgb, (0.5, 0.5, 0.0, 0.0)).shape == (1, 1, 3)
