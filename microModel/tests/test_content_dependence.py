"""Content-dependence index: the dense-feature homogenization gauge."""

import numpy as np
import pytest

from microModel.monitor import content_dependence_index


def _features(image_noise):
    """(40, 36, 16) patch features: a SHARED per-position layout plus a small
    per-image perturbation. With tiny noise every image yields nearly the same
    anchor-similarity map (homogenized, CDI -> 1); with pure per-image noise
    the maps are content-specific (CDI ~ 0)."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((36, 16))
    noise = rng.standard_normal((40, 36, 16)) * image_noise
    return base[None] + noise


def test_cdi_high_when_position_dominated():
    cdi = content_dependence_index(_features(0.01))
    assert cdi is not None and cdi > 0.99


def test_cdi_low_when_content_specific():
    rng = np.random.default_rng(0)
    cdi = content_dependence_index(rng.standard_normal((40, 36, 16)))
    assert cdi is not None and abs(cdi) < 0.3


def test_cdi_none_for_degenerate_input():
    assert content_dependence_index(None) is None
    assert content_dependence_index(np.random.default_rng(0).standard_normal((1, 9, 4))) is None


def test_cdi_anchor_selects_map():
    """A single off-centre bright patch must dominate the map only for that
    anchor: with anchor=0 the map keys off patch 0's feature."""
    rng = np.random.default_rng(1)
    F = rng.standard_normal((6, 9, 8))
    c0 = content_dependence_index(F, anchor=0)
    c8 = content_dependence_index(F, anchor=8)
    assert c0 is not None and c8 is not None
    # Different anchors -> different gauge values in general.
    assert c0 != c8
