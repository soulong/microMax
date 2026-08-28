"""Tests for microBase.cells: cropping pure functions."""

import numpy as np
import pytest

from microBase.cells import (
    crop_cell,
    crop_all_cells,
    get_labels,
)


def _make_volume_mask(n_cells=3, h=40, w=40, c=2):
    """Build a (H, W, C) volume and (H, W) mask with `n_cells` non-overlapping cells."""
    volume = np.zeros((h, w, c), dtype=np.uint16)
    mask = np.zeros((h, w), dtype=np.int32)
    # Each cell is a 10x10 block at a different position
    for i in range(n_cells):
        y = (i * 12) % (h - 10)
        x = (i * 14) % (w - 10)
        mask[y:y+10, x:x+10] = i + 1
        volume[y:y+10, x:x+10, :] = (i + 1) * 100
    return volume, mask


def test_get_labels():
    _, mask = _make_volume_mask(n_cells=3)
    ids = get_labels(mask)
    assert ids == [1, 2, 3]


def test_get_labels_empty():
    mask = np.zeros((10, 10), dtype=np.int32)
    assert get_labels(mask) == []


def test_get_labels_ignores_background():
    """Background (0) should not be in the list."""
    mask = np.array([[0, 0, 1], [0, 2, 1]], dtype=np.int32)
    assert get_labels(mask) == [1, 2]


def test_crop_cell_basic():
    volume, mask = _make_volume_mask(n_cells=1, h=40, w=40, c=2)
    crop, cell_mask, bbox = crop_cell(volume, mask, 1, padding=2)
    # Cell is at (0,0)-(9,9); with padding 2 -> (0,0)-(11,11) clamped to (0,0)-(12,12)
    assert crop.shape == (12, 12, 2)
    assert cell_mask.shape == (12, 12)
    assert bbox == (0, 0, 12, 12)
    # Background pixels are zeroed
    # At (0,0) — cell is present
    assert crop[0, 0, 0] == 100
    # At (11, 11) — outside cell, background zeroed
    assert crop[11, 11, 0] == 0


def test_crop_cell_padding_clamped():
    """Padding should be clamped to image bounds."""
    volume, mask = _make_volume_mask(n_cells=1, h=20, w=20, c=1)
    # Cell at (0,0)-(9,9); padding 100 -> clamped to image bounds (20,20)
    crop, cell_mask, bbox = crop_cell(volume, mask, 1, padding=100)
    assert crop.shape == (20, 20, 1)
    assert bbox == (0, 0, 20, 20)


def test_crop_cell_no_pixels():
    """Cell ID not in mask -> exits with error (corruption)."""
    volume, mask = _make_volume_mask(n_cells=1)
    with pytest.raises(SystemExit):
        crop_cell(volume, mask, 999)


def test_crop_cell_label_zero_exits():
    """Label 0 (background) must hard-exit, not crop the whole image."""
    volume, mask = _make_volume_mask(n_cells=1)
    with pytest.raises(SystemExit):
        crop_cell(volume, mask, 0)


def test_crop_cell_negative_label_exits():
    volume, mask = _make_volume_mask(n_cells=1)
    with pytest.raises(SystemExit):
        crop_cell(volume, mask, -3)


def test_crop_cell_background_zeroed_all_channels():
    """Background zeroing should apply to all channels."""
    volume, mask = _make_volume_mask(n_cells=1, h=30, w=30, c=3)
    crop, _, _ = crop_cell(volume, mask, 1, padding=2)
    # Find a pixel outside the cell
    # Cell at (0,0)-(9,9), padding 2 -> crop (0,0)-(11,11)
    # Background pixels are at (11, 11)
    assert crop[11, 11, 0] == 0
    assert crop[11, 11, 1] == 0
    assert crop[11, 11, 2] == 0


def test_crop_all_cells():
    volume, mask = _make_volume_mask(n_cells=3, h=50, w=50, c=2)
    results = crop_all_cells(volume, mask, padding=2)
    assert len(results) == 3
    for crop, cell_mask, bbox, cid in results:
        assert cid in (1, 2, 3)
        assert crop.ndim == 3
        assert cell_mask.ndim == 2
        assert len(bbox) == 4


def test_crop_all_cells_subset():
    """Pass labels to crop only some cells."""
    volume, mask = _make_volume_mask(n_cells=3, h=50, w=50)
    results = crop_all_cells(volume, mask, labels=[2])
    assert len(results) == 1
    assert results[0][3] == 2  # cid
