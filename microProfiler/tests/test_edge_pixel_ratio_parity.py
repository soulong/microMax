"""Parity: microBase.edge_pixel_ratio vs the original per-label scipy code."""

import numpy as np
from scipy.ndimage import binary_erosion, find_objects

from microBase import edge_pixel_ratio


def _reference(mask, edge_tolerance=2):
    """The original _shape_edge_pixel_ratio implementation (verbatim)."""
    H, W = mask.shape
    labels = np.unique(mask)
    labels = labels[labels != 0]
    result = {}
    slices = find_objects(mask)
    edge_mask = np.zeros((H, W), dtype=bool)
    n = edge_tolerance + 1
    edge_mask[:n, :] = True
    edge_mask[-n:, :] = True
    edge_mask[:, :n] = True
    edge_mask[:, -n:] = True
    for lbl in labels:
        sl = slices[lbl - 1]
        cell = mask[sl] == lbl
        perimeter = cell & ~binary_erosion(cell)
        perim = int(perimeter.sum())
        if perim == 0:
            result[int(lbl)] = 0.0
            continue
        result[int(lbl)] = int(np.count_nonzero(perimeter & edge_mask[sl])) / perim
    return result


def test_matches_original_scipy_implementation():
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = np.zeros((60, 70), dtype=np.int32)
        for i in range(1, int(rng.integers(1, 12)) + 1):
            y, x = rng.integers(0, 55), rng.integers(0, 65)
            h, w = rng.integers(3, 12), rng.integers(3, 12)
            m[y:y + h, x:x + w] = i
        old = _reference(m)
        new = edge_pixel_ratio(m)
        assert old.keys() == new.keys()
        for k in old:
            assert abs(old[k] - new[k]) < 1e-12
