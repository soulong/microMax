"""Unit tests for the label app's suggestion/render primitives.

Covers SuggestEngine (per-label exemplar scoring: positives raise a cell's
score, explicit negatives PUSH their lookalikes down), the Refresh-model
logistic fit (fit_label_model) and the display-parameter handling of
_render_png. No Flask server or model bundle is needed — all pure
numpy/sklearn.
"""

import numpy as np

from microModel.label import SuggestEngine, _render_png


def _clustered_space(n_per_side=8, seed=0):
    """L2-normalized space with two well-separated cell clusters.

    Rows 0..n-1 sit around +e1, rows n..2n-1 around -e1.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.1, size=(2 * n_per_side, 16)).astype(np.float32)
    W[:n_per_side] += np.r_[2.0, *([0.0] * 15)]
    W[n_per_side:] -= np.r_[2.0, *([0.0] * 15)]
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def test_knn_score_ranks_the_positive_cluster_first():
    """Cells near the positives score above cells near the negatives."""
    W = _clustered_space()
    n = 8
    eng = SuggestEngine(W, knn_k=1, neg_weight=0.5)
    pos_part, neg_part = eng.parts(1, [0, 1], [n])
    score = pos_part - 0.5 * neg_part
    assert (score[:n] > score[n:]).all()
    # k=1 means nearest exemplar: the top positive row keeps its own peak.
    assert pos_part[0] >= pos_part[2:].max()


def test_explicit_negatives_push_lookalikes_down():
    """An explicit negative is not just a veto — it depresses the score of
    the cells it resembles, so the ranking itself changes."""
    W = _clustered_space()
    n = 8
    # A negative that is an almost exact copy of positive-cluster row 2.
    dup = W[2] + 0.001
    dup = (dup / np.linalg.norm(dup)).astype(np.float32)
    W2 = np.vstack([W, dup])
    neg_row = len(W2) - 1
    eng = SuggestEngine(W2, knn_k=1, neg_weight=2.0)
    pos_part, neg_part = eng.parts(1, [0], [neg_row])
    score = pos_part - 2.0 * neg_part
    others = [i for i in range(n) if i != 2]
    assert score[2] < score[others].min()   # the lookalike sank below the rest


def test_score_cache_bumps_on_write():
    """Cached arrays survive identical repeat calls; bump() (called after
    every write by the server) forces a recompute."""
    W = _clustered_space()
    eng = SuggestEngine(W, knn_k=1, neg_weight=0.5)
    first = eng.parts(1, [0], [])
    assert eng.parts(1, [0], []) is first            # cache hit
    eng.bump(1)
    assert eng.parts(1, [0], []) is not first        # invalidated
    grown = eng.parts(1, [0, 1], [])
    assert grown is not first                        # exemplar count changed


def test_fit_label_model_learns_the_boundary():
    """The Refresh-model logistic fit separates the two exemplar clusters
    (its P(positive) is what a refreshed label scores by); a label with
    too few exemplars on either side gets NO model — None — and keeps the
    kNN score."""
    from microModel.label import MIN_FIT_NEG, MIN_FIT_POS, fit_label_model
    W = _clustered_space()
    n = 8
    # Below the minimum on either side: refuse to fit.
    assert fit_label_model(W, [0, 1], list(range(n, n + MIN_FIT_NEG)),
                           seed=0) is None
    assert fit_label_model(W, list(range(MIN_FIT_POS)), [], seed=0) is None
    # Enough of both: P(positive) separates the clusters outright
    # (regularized probabilities are calibrated, not saturated — the
    # property that matters is clean separation around the 0.5 boundary).
    model = fit_label_model(W, list(range(n)), list(range(n, 2 * n)), seed=0)
    proba = model.predict_proba(W)[:, 1]
    assert proba[:n].min() > 0.5 > proba[n:].max()
    assert proba[:n].min() > proba[n:].max()


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
