"""Unit tests for the label app's suggestion/render primitives.

Covers MLModelEngine (the active-learning per-label classifier over the
embedding space) and the display-parameter handling of _render_png. No
Flask server or model bundle is needed — both are pure numpy/sklearn.
"""

import numpy as np
import pytest

from microModel.label import MLModelEngine, _render_png


def _clustered_space(n_per_side=8, seed=0):
    """L2-normalized space with two well-separated cell clusters.

    Rows 0..n-1 sit around +e1, rows n..2n-1 around -e1 — a labeled
    positive/negative split the models must reproduce.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.1, size=(2 * n_per_side, 16)).astype(np.float32)
    W[:n_per_side] += np.r_[2.0, *([0.0] * 15)]
    W[n_per_side:] -= np.r_[2.0, *([0.0] * 15)]
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def test_ml_engine_gates_on_counts():
    """Below min_pos or min_neg the engine refuses to score (kNN fallback)."""
    W = _clustered_space()
    eng = MLModelEngine(W, "logistic", min_pos=5, min_neg=3)
    pos, neg = list(range(6)), list(range(8, 10))
    assert eng.probs(1, pos, neg) is None          # 2 negatives: too few
    assert eng.probs(1, pos, list(range(8, 13))) is not None
    assert eng.probs(1, list(range(4)), list(range(8, 13))) is None  # 4 pos


def test_ml_engine_separates_clusters():
    """Fitted probabilities must be high on the positive cluster and low on
    the negative one — a real decision boundary, not exemplar similarity.
    Random-forest probabilities on 16 training rows stay soft (~0.8), so
    the assertion only requires a clean margin, not calibration."""
    W = _clustered_space()
    n = 8
    for kind in ("logistic", "random_forest"):
        eng = MLModelEngine(W, kind, min_pos=5, min_neg=3, seed=42)
        probs = eng.probs(1, list(range(n)), list(range(n, 2 * n)))
        assert probs is not None
        assert (probs[:n] > 0.6).all()
        assert (probs[n:] < 0.4).all()


def test_ml_engine_cache_invalidated_by_bump_and_counts():
    """Cached arrays survive identical repeat calls; bump() or a changed
    decision count must re-fit (observed through object identity)."""
    W = _clustered_space()
    eng = MLModelEngine(W, "logistic", min_pos=5, min_neg=3, seed=1)
    pos, neg = list(range(6)), list(range(8, 14))
    first = eng.probs(1, pos, neg)
    assert eng.probs(1, pos, neg) is first          # cache hit
    eng.bump()
    assert eng.probs(1, pos, neg) is not first      # invalidated
    grown = eng.probs(1, pos + [7], neg)
    assert grown is not None and grown is not first  # count key changed


def test_render_png_window_and_gamma():
    """A narrower percentile window must raise contrast; gamma > 1 must
    brighten midtones. Defaults reproduce the legacy render exactly."""
    rng = np.random.default_rng(3)
    img = rng.uniform(0.2, 0.8, size=(16, 16, 1)).astype(np.float32)
    img[:4] = 0.0                                   # zero background

    default = _render_png(img, 64)
    assert _render_png(img, 64, 0.1, 99.9, 1.0) == default

    def _mean(png):
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(png)),
                          dtype=np.float32).mean()

    tight = _mean(_render_png(img, 64, lo_pct=5, hi_pct=90))
    assert tight > _mean(_render_png(img, 64, 0.1, 99.9, 1.0))
    bright = _mean(_render_png(img, 64, gamma=2.0))
    assert bright > _mean(_render_png(img, 64, 0.1, 99.9, 1.0))
    # Out-of-range query params are clamped, never crash.
    _render_png(img, 64, lo_pct=80, hi_pct=5, gamma=99)
