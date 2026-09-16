"""Cluster-assisted bulk labeling over the label app's shared space.

The label server embeds every cell into one whitened-PCA + L2-normalized
space (W, rows = cells). This module partitions that space with the SAME
Leiden clustering the reduction command uses (kNN graph + RBConfiguration),
so the whole dataset breaks into ~N fine-grained morphological clusters that
the UI can label as units: review a cluster's medoid (its most typical
cell), then write one label to every member in one undoable op.

Everything here is DECISION-INDEPENDENT (like the feature cache): the
assignment only depends on W, so it is computed once per embedding and
cached to <save_dir>/features/clusters_<hash>.npz. The hash covers the
bundle identity + the exact cell list + the PCA width, so a new bundle or
data set silently recomputes.

The resolution search targets a cluster COUNT (the intuitive knob — "split
everything into about 100 groups") instead of a raw resolution: an
exponential ladder brackets the target, a short bisection refines it, and
the partition whose count lands nearest the target wins (Leiden's
count-vs-resolution relation is monotone in practice but not exact, so the
search always keeps the best seen).
"""

import hashlib
import json
import logging
import os

import numpy as np

from ..deduplication import CACHE_DIR, _error
from ..reduction import _knn_graph, _leiden_partition
from ..utils import atomic_npz_save, load_npz_cache

logger = logging.getLogger(__name__)

#: Bump when the cache layout changes (retires older clusters_*.npz files).
CLUSTER_CACHE_VERSION = 1

#: Resolution ladder: start, growth factor per step, and the floor/ceiling
#: between which the search stays (0.5 already over-partitions noise-level
#: structure; above 64 the graph usually shatters into singleton clusters).
RES_START = 0.5
RES_GROWTH = 2.0
RES_MIN = 0.05
RES_MAX = 64.0

#: Bisection steps between the two ladder entries bracketing the target —
#: each is one cheap Leiden run on the pre-built graph.
RES_BISECT_STEPS = 5


def cluster_cache_path(save_dir, key_parts):
    """Cache file <save_dir>/features/clusters_<hash>.npz for one assignment.

    `key_parts` must identity everything the assignment depends on besides
    the cluster settings (the server passes the bundle identity, a hash of
    the embedded cell list and the PCA width).
    """
    key = json.dumps({"v": CLUSTER_CACHE_VERSION, **key_parts},
                     sort_keys=True)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return os.path.join(save_dir, CACHE_DIR, f"clusters_{digest}.npz")


def _n_clusters(membership):
    """Number of clusters in a 0-based membership array."""
    return int(membership.max()) + 1 if len(membership) else 0


def resolve_resolution(graph, n_rows, target, seed):
    """Find the Leiden resolution whose cluster count lands nearest `target`.

    Runs on a PRE-BUILT kNN graph (one shared `_knn_graph` for the whole
    search): a downward ladder walks coarser while still finer than the
    target, an upward ladder climbs until the target is reached (or the
    resolution cap / all-singletons limit hits), and bisection refines
    between the two bracketing entries. Returns (resolution, membership)
    for the partition whose count is closest to the target.
    """
    best = None    # (|k - target|, resolution, membership)

    def _consider(res, k, memb):
        nonlocal best
        if best is None or abs(k - target) < best[0]:
            best = (abs(k - target), res, memb)

    def _at(res):
        memb = _leiden_partition(None, res, seed, graph=graph)
        k = _n_clusters(memb)
        logger.info("  Leiden resolution %.3f -> %d clusters", res, k)
        return k, memb

    # Coarse phase: step DOWN while the partition is still finer than the
    # target (the ladder start may already overshoot a small target).
    res = RES_START
    k, memb = _at(res)
    while k > target and res > RES_MIN and k < n_rows:
        _consider(res, k, memb)
        res /= RES_GROWTH
        k, memb = _at(res)
    _consider(res, k, memb)
    lo = (res, k)   # k <= target here (or a floor/cap stopped the walk)
    # Climb until the target is bracketed.
    while k < target and res < RES_MAX and k < n_rows:
        res *= RES_GROWTH
        k, memb = _at(res)
        _consider(res, k, memb)
    hi = (res, k)
    # Bisect between the bracketing resolutions.
    if lo[1] < hi[1] and lo[1] < target < hi[1]:
        for _ in range(RES_BISECT_STEPS):
            mid = (lo[0] + hi[0]) / 2
            k, memb = _at(mid)
            _consider(mid, k, memb)
            if k >= target:
                hi = (mid, k)
            else:
                lo = (mid, k)
    return best[1], best[2]


def medoid_rows(W, membership):
    """Per-cluster medoid: the member most similar to its whole cluster.

    W rows are L2-normalized, so dot products are cosines and a member's
    mean similarity to its cluster is (W[rows] @ sum_j W[rows_j]) / m — one
    m x D matmul, no m x m matrix. Returns (cluster_ids, medoid_rows), both
    ascending in cluster id.
    """
    ids, meds = [], []
    for c in np.unique(membership):
        rows = np.flatnonzero(membership == c)
        if len(rows) == 1:
            med = int(rows[0])
        else:
            centroid = W[rows].sum(axis=0)
            means = (W[rows] @ centroid) / float(len(rows))
            med = int(rows[int(np.argmax(means))])
        ids.append(int(c))
        meds.append(med)
    return ids, meds


def build_or_load(W, row_paths, save_dir, key_parts, target, res, seed):
    """Cluster assignment for the label app's W rows (npz-cached).

    W: the shared L2-normalized space (rows aligned with `row_paths`, the
    portable cell identities). target/res/seed steer the partition:
    an explicit `res` clusters once at that resolution, otherwise the
    count-targeted search runs. Returns (membership 0-based per W row,
    cluster_ids, medoid_row per cluster_id, resolution used) — cluster_ids
    and medoid_rows are aligned lists, both ascending in cluster id.

    The cache key rides in `key_parts`; on load the stored cell identities
    must match `row_paths` exactly (defence in depth — the key already
    covers them), otherwise the assignment is recomputed and re-written.
    """
    if res is None and target is None:
        _error("clustering needs cluster.target or cluster.res")
    if W.shape[0] < 2:
        # Degenerate pools: one singleton "cluster" per cell (empty W -> no
        # clusters) — the graph/Leiden path below needs at least 2 rows.
        memb = np.zeros(len(W), dtype=int)
        ids = [0] if len(W) else []
        meds = [0] if len(W) else []
        return memb, ids, meds, 0.0
    cpath = cluster_cache_path(save_dir, key_parts)
    if os.path.exists(cpath):
        cached = load_npz_cache(cpath)
        if cached and {"paths", "cluster", "cluster_ids", "medoid_row",
                       "res"} <= cached.keys():
            if [str(p) for p in cached["paths"]] == list(row_paths):
                memb = cached["cluster"].astype(int)
                ids = [int(i) for i in cached["cluster_ids"]]
                meds = [int(m) for m in cached["medoid_row"]]
                used = float(cached["res"])
                logger.info("Cluster cache hit: %s (%d clusters, "
                            "resolution %.3f)", os.path.basename(cpath),
                            len(ids), used)
                return memb, ids, meds, used
            logger.info("Cluster cache %s matches a different cell list "
                        "— recomputing", os.path.basename(cpath))

    graph = _knn_graph(W)   # one graph for every partition attempt
    if res is not None:
        memb = _leiden_partition(None, res, seed, graph=graph)
        used = float(res)
        logger.info("Clustering %d cells at resolution %.3f", len(W), used)
    else:
        used, memb = resolve_resolution(graph, W.shape[0], int(target), seed)
    ids, meds = medoid_rows(W, memb)
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    atomic_npz_save(cpath, paths=np.array(list(row_paths)),
                    cluster=memb.astype(np.int32),
                    cluster_ids=np.array(ids, dtype=np.int32),
                    medoid_row=np.array(meds, dtype=np.int32),
                    res=np.float64(used))
    logger.info("Clustered %d cells into %d clusters (resolution %.3f); "
                "cached to %s", len(W), len(ids), used,
                os.path.basename(cpath))
    return memb, ids, meds, used
