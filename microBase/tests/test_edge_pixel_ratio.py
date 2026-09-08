"""edge_pixel_ratio — per-object edge/perimeter ratio."""

import numpy as np

from microBase import edge_pixel_ratio


def test_interior_object_has_zero_ratio():
    mask = np.zeros((50, 50), np.int32)
    mask[10:30, 10:30] = 7
    assert edge_pixel_ratio(mask) == {7: 0.0}


def test_border_object_has_positive_ratio():
    # Corner square: with the default 2px tolerance (3px band) most of its
    # ring sits inside the band.
    mask = np.zeros((50, 50), np.int32)
    mask[0:20, 0:20] = 4
    ratio = edge_pixel_ratio(mask)[4]
    assert 0.4 < ratio < 0.8


def test_tolerance_changes_the_ratio():
    mask = np.zeros((50, 50), np.int32)
    mask[0:20, 0:20] = 4
    assert (edge_pixel_ratio(mask, edge_tolerance=0)[4]
            != edge_pixel_ratio(mask, edge_tolerance=2)[4])


def test_empty_mask_returns_empty():
    assert edge_pixel_ratio(np.zeros((10, 10), np.int32)) == {}
