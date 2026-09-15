"""Unit tests for deduplication's group quotas and Leiden grouping.

Group quotas implement power-compressed per-group keep counts (the
target_ratio knob); the tests pin the invariants that make the knob
trustworthy: exact total, per-group floors/caps, and the promised ratio
compression. The Leiden grouping test uses two well-separated blobs —
a data-driven check that auto resolution finds >= 2 groups.
"""

import numpy as np

from microModel.deduplication import _cluster_groups, _group_quotas


def test_quotas_sum_and_bounds():
    """Quotas sum exactly to min(total, pool size) within [floor, size]."""
    # 100:1 pool, budget 1100 -> target 10:1 is exactly feasible (1000 + 100).
    quotas, alpha = _group_quotas([10000, 100], 1100, 10)
    assert quotas.sum() == 1100
    assert alpha < 1.0                       # real compression happened
    assert quotas[0] == 1000 and quotas[1] == 100
    assert quotas[0] / quotas[1] == 10       # the promised compression


def test_quotas_cap_at_group_size():
    """When the budget makes the target infeasible the small group caps at
    its size — the big group absorbs the rest (ratio worse than the target,
    which is the best the data allows)."""
    quotas, _ = _group_quotas([10000, 100], 2000, 10)
    assert quotas.sum() == 2000
    assert quotas[1] == 100                  # capped at the group size
    assert quotas[0] == 1900


def test_quotas_already_flat_is_proportional():
    """A pool flatter than the target keeps alpha = 1 (plain shares)."""
    quotas, alpha = _group_quotas([50, 50], 20, 10)
    assert alpha == 1.0
    assert quotas.sum() == 20
    assert abs(int(quotas[0]) - int(quotas[1])) <= 1


def test_quotas_floors_win_over_tiny_budget():
    """A budget smaller than the floors keeps every group above the floor —
    floors take priority and the total overshoots instead of losing a group."""
    quotas, _ = _group_quotas([1000, 1000, 1000], 4, 5)
    assert (quotas >= 2).all()
    assert quotas.sum() == 4 or quotas.sum() == 6  # 6 when floors dominate


def test_quotas_cap_at_group_size():
    """A quota can never exceed the group's own size (alpha compression can
    over-allocate the smallest group relative to its population)."""
    quotas, _ = _group_quotas([10000, 100], 2000, 10)
    assert quotas[1] <= 100


def test_quotas_no_compression_when_ratio_unreachable():
    """target_ratio >= current ratio needs no compression: alpha = 1."""
    quotas, alpha = _group_quotas([100, 50], 30, 10)
    assert alpha == 1.0
    assert quotas.sum() == 30
    assert quotas[0] == 20 and quotas[1] == 10   # proportional split


def test_quotas_floors_protect_small_groups():
    """The small group gets at least the floor (bounded by its size)."""
    quotas, _ = _group_quotas([10000, 10], 2000, 10)
    assert quotas[1] == 10                     # its whole population
    assert quotas.sum() == 2000


def test_cluster_groups_finds_two_blobs():
    """Two well-separated dense blobs are grouped into >= 2 clusters and each
    blob is dominated by ONE cluster (>= 95% purity) — the auto resolution
    must not shred a homogeneous population into micro-groups."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.02, size=(80, 8))
    b = rng.normal(0.0, 0.02, size=(80, 8)) + np.r_[3.0, [0.0] * 7]
    W = np.vstack([a, b])
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    ids, res = _cluster_groups(W, seed=0)
    assert res is not None
    assert len(set(ids.tolist())) >= 2
    for blob in (ids[:80], ids[80:]):
        majority = np.bincount(blob).max()
        assert majority / len(blob) >= 0.95
    # The two blobs are dominated by DIFFERENT clusters.
    assert np.bincount(ids[:80]).argmax() != np.bincount(ids[80:]).argmax()
