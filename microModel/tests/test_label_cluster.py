"""Unit tests for cluster-assisted bulk labeling (label/cluster.py + API).

Covers the resolution search for a target cluster count, the medoid
computation, the npz cluster cache, and the server-side cluster workflow:
/api/queue's cluster filter + medoid/prob sorts, the pred_label prediction
filter, and /api/annotate_cluster's scope-following whole-cluster write
(one undoable op). The server tests drive a LabelServer with a stubbed
feature build — no model bundle or Flask network needed.
"""

import os

import numpy as np
import pytest

from microModel.label import (AnnotationDB, LabelServer, SuggestEngine,
                              build_or_load, medoid_rows, resolve_resolution)
from microModel.label.cluster import cluster_cache_path
from microModel.reduction import _knn_graph


def _blob_space(n_per_cluster=12, n_clusters=3, seed=0):
    """Well-separated L2-normalized blobs: rows 0..k-1 = cluster 0, etc."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(n_clusters):
        center = np.zeros(16, dtype=np.float32)
        center[c * 2] = 3.0
        parts.append(center + rng.normal(scale=0.05,
                                         size=(n_per_cluster, 16))
                     .astype(np.float32))
    W = np.vstack(parts)
    return (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32)


# ---- resolution search ------------------------------------------------------

def test_resolve_resolution_lands_near_target():
    """The ladder + bisection search finds a partition whose cluster count
    is close to the requested target (never far above it)."""
    W = _blob_space(n_per_cluster=40, n_clusters=5)
    graph = _knn_graph(W)
    res, memb = resolve_resolution(graph, len(W), target=10, seed=0)
    k = int(memb.max()) + 1
    assert 5 <= k <= 12          # near the target, at least one per blob
    assert res > 0.5             # finer than the ladder start was needed


def test_explicit_resolution_skips_the_search():
    """The search honours the target's direction: a tiny target lands on a
    coarse partition, a huge target on the finest reachable one — and a
    fixed resolution reproduces its partition deterministically."""
    W = _blob_space(n_per_cluster=15, n_clusters=3, seed=1)
    graph = _knn_graph(W)
    _, coarse = resolve_resolution(graph, len(W), target=1, seed=0)
    _, fine = resolve_resolution(graph, len(W), target=99, seed=0)
    assert int(coarse.max()) + 1 <= 3          # coarse: at most the blobs
    assert int(fine.max()) + 1 > int(coarse.max()) + 1
    # A fixed resolution reproduces the same partition deterministically.
    from microModel.reduction import _leiden_partition
    fixed = _leiden_partition(None, 2.0, 0, graph=graph)
    assert int(fixed.max()) + 1 >= 3   # the separated blobs split apart


# ---- medoids ----------------------------------------------------------------

def test_medoid_is_the_most_typical_member():
    """The medoid row maximizes the mean cosine similarity to its cluster
    (brute-forced here against the definition)."""
    W = _blob_space(n_per_cluster=9, n_clusters=2, seed=3)
    memb = np.array([0] * 9 + [1] * 9)
    ids, meds = medoid_rows(W, memb)
    assert ids == [0, 1]
    for c, med in zip(ids, meds):
        rows = np.flatnonzero(memb == c)
        sims = W[rows] @ W[rows].T
        brute = rows[int(np.argmax(sims.mean(axis=1)))]
        assert med == int(brute)
        # A cell's mean sim via the centroid trick matches the brute force.
        means = (W[rows] @ W[rows].sum(axis=0)) / len(rows)
        assert np.allclose(means, sims.mean(axis=1), atol=1e-5)


# ---- npz cache --------------------------------------------------------------

def test_build_or_load_caches_and_invalidates(tmp_path):
    """The first call computes and writes the cache; the second call hits it
    (identical assignment). A different key (new bundle) recomputes into a
    separate cache file."""
    W = _blob_space()
    paths = [f"c{i}.tif" for i in range(len(W))]
    key = {"bundle": "bundle-a", "cells": "aaa", "pca": 8}

    memb, ids, meds, res = build_or_load(W, paths, str(tmp_path), key,
                                         target=3, res=None, seed=0)
    assert len(ids) == len(meds) == int(memb.max()) + 1
    assert sorted(set(memb.tolist())) == sorted(ids)
    assert os.path.exists(cluster_cache_path(str(tmp_path), key))

    memb2, ids2, meds2, res2 = build_or_load(W, paths, str(tmp_path), key,
                                             target=3, res=None, seed=0)
    assert (memb == memb2).all() and ids == ids2 and meds == meds2
    assert res == res2

    # A changed cell list must never be served the old assignment.
    key_b = {"bundle": "bundle-a", "cells": "bbb", "pca": 8}
    memb3, _, _, _ = build_or_load(W[:10], paths[:10], str(tmp_path), key_b,
                                   target=3, res=None, seed=0)
    assert len(memb3) == 10
    with pytest.raises(Exception):
        build_or_load(W, paths, str(tmp_path), key, target=None, res=None,
                      seed=0)


# ---- server workflow --------------------------------------------------------

class _ClusterServer:
    """A LabelServer with stubbed registry/space/clusters — the fast path
    into the API without any model or feature extraction."""

    def __init__(self, tmp_path, n_per_cluster=6, n_clusters=2):
        config = {"save_dir": str(tmp_path / "proj"),
                  "data": {"file_dir": [str(tmp_path / "imgs")]},
                  "recommend": {"knn_k": 1, "page_size": 100}}
        self.srv = LabelServer(config)
        srv = self.srv
        srv.label_mode = "multi"
        srv.db = AnnotationDB(str(tmp_path / "label_multiple.db"))
        srv.session_id = srv.db.new_session()
        srv._refresh_labels()

        n = n_per_cluster * n_clusters
        rows = [(f"cell_{i}.tif", f"imgs/cell_{i}.tif", "imgs", None)
                for i in range(n)]
        srv._registration_rows = rows
        self.W = _blob_space(n_per_cluster, n_clusters)
        srv.engine = SuggestEngine(self.W, knn_k=1, neg_weight=0.5)
        srv._row_fp = {r: fp for r, (fp, *_rest) in enumerate(rows)}
        srv._rebuild_cell_maps()

        memb = np.array([c for c in range(n_clusters)
                         for _ in range(n_per_cluster)])
        ids, meds = medoid_rows(self.W, memb)
        srv.cluster_memb = memb
        srv.cluster_ids = [i + 1 for i in ids]
        srv.cluster_medoid = {i + 1: m for i, m in zip(ids, meds)}
        srv.cluster_res_used = 1.0
        srv.cluster_target = 2
        srv.diverse_cids = [c["cell_id"] for c in srv.cells]

    def client(self):
        return self.srv.app.test_client()

    def label(self, name="A"):
        lb = self.srv.db.add_label(name)
        self.srv._refresh_labels()
        return lb["label_id"]


def test_queue_cluster_filter_and_medoid_sort(tmp_path):
    """cluster= narrows the queue to that cluster's members; sort=medoid
    leads with the medoid and the badge carries the medoid similarity."""
    cs = _ClusterServer(tmp_path)
    srv = cs.srv
    lid = cs.label()
    client = cs.client()
    # Cluster 1's cells become positives, cluster 2's explicit negatives.
    c1 = [c["filepath"] for c in srv.cells
          if srv.cluster_memb[srv.cid_row[c["cell_id"]]] == 0]
    c2 = [c["filepath"] for c in srv.cells
          if srv.cluster_memb[srv.cid_row[c["cell_id"]]] == 1]
    assert client.post("/api/annotate_batch", json={
        "filepaths": c1, "label_ids": [lid], "state": 1}).status_code == 200
    assert client.post("/api/annotate_batch", json={
        "filepaths": c2, "label_ids": [lid], "state": 0}).status_code == 200

    r = client.get(
        f"/api/queue?label_id={lid}&scope=with&cluster=1&sort=medoid")
    assert r.status_code == 200
    body = r.get_json()
    assert body["total"] == 6
    med_cell = srv.cid_cell[srv.row_cid[srv.cluster_medoid[1]]]
    assert body["cells"][0]["filepath"] == med_cell["filepath"]
    assert body["cells"][0]["score"] == pytest.approx(1.0, abs=1e-4)

    # The other cluster has no positives — and unknown ids are rejected.
    r = client.get(
        f"/api/queue?label_id={lid}&scope=with&cluster=2&sort=medoid")
    assert r.get_json()["total"] == 0
    assert client.get(
        f"/api/queue?label_id={lid}&scope=with&cluster=99"
        f"&sort=medoid").status_code == 400
    # medoid sort without a cluster filter is a hard error.
    assert client.get(
        f"/api/queue?label_id={lid}&scope=with&sort=medoid"
    ).status_code == 400


def test_annotate_cluster_writes_only_the_current_scope(tmp_path):
    """The whole-cluster write covers exactly cluster ∩ scope (an explicit
    negative is untouched), is ONE undoable op, and drains the undecided
    queue of that cluster."""
    cs = _ClusterServer(tmp_path)
    srv = cs.srv
    lid = cs.label()
    client = cs.client()
    members = [c["filepath"] for c in srv.cells
               if srv.cluster_memb[srv.cid_row[c["cell_id"]]] == 0]
    # One member already carries an explicit negative — it must survive.
    assert client.post("/api/annotate_batch", json={
        "filepaths": [members[0]], "label_ids": [lid],
        "state": 0}).status_code == 200

    r = client.post("/api/annotate_cluster", json={
        "cluster_id": 1, "label_ids": [lid], "state": 1,
        "label_id": lid, "scope": "undecided"})
    assert r.status_code == 200
    assert r.get_json()["n"] == len(members) - 1

    states = srv.db.cell_states()
    by_fp = {c["filepath"]: c["cell_id"] for c in srv.cells}
    assert states[(by_fp[members[0]], lid)] == 0          # negative kept
    for fp in members[1:]:
        assert states[(by_fp[fp], lid)] == 1              # undecided filled

    q = client.get(
        f"/api/queue?label_id={lid}&scope=undecided&cluster=1")
    assert q.get_json()["total"] == 0                     # scope drained

    u = client.post("/api/undo")
    assert u.get_json()["n"] == len(members) - 1          # one op reverts all
    states = srv.db.cell_states()
    assert states[(by_fp[members[0]], lid)] == 0
    assert (by_fp[members[1]], lid) not in states
    # An unknown cluster is rejected without writing.
    assert client.post("/api/annotate_cluster", json={
        "cluster_id": 42, "label_ids": [lid], "state": 1,
        "label_id": lid, "scope": "undecided"}).status_code == 400


def test_cluster_cold_start_shows_all_members(tmp_path):
    """Fresh project (label has no positives): the unfiltered To-label queue
    falls back to the dataset-wide diverse seeds — but inside a cluster that
    sprinkle would usually miss it, so there EVERY undecided member of the
    cluster is a candidate."""
    cs = _ClusterServer(tmp_path)
    srv = cs.srv
    lid = cs.label()
    # The diverse seeds live entirely in cluster 2 — cluster 1 gets none.
    srv.diverse_cids = [c["cell_id"] for c in srv.cells
                        if srv.cluster_memb[srv.cid_row[c["cell_id"]]] == 1]
    client = cs.client()

    r = client.get(f"/api/queue?label_id={lid}&scope=undecided&cluster=1")
    assert r.get_json()["total"] == 6      # not the empty seed intersection

    # Unfiltered queue keeps the diverse fallback (only the seeds).
    r = client.get(f"/api/queue?label_id={lid}&scope=undecided")
    assert r.get_json()["total"] == 6


def test_clusters_api_counts_follow_the_view(tmp_path):
    """in_view = cluster ∩ source ∩ scope (what clicking shows); size stays
    the full cluster; undecided is source-independent."""
    cs = _ClusterServer(tmp_path)
    srv = cs.srv
    lid = cs.label()
    client = cs.client()
    c1 = [c["filepath"] for c in srv.cells
          if srv.cluster_memb[srv.cid_row[c["cell_id"]]] == 0]
    assert client.post("/api/annotate_batch", json={
        "filepaths": c1, "label_ids": [lid], "state": 1}).status_code == 200

    r = client.get(f"/api/clusters?label_id={lid}&scope=with")
    cards = {c["id"]: c for c in r.get_json()["clusters"]}
    assert cards[1]["size"] == 6 and cards[2]["size"] == 6
    assert cards[1]["in_view"] == 6          # positives in cluster 1
    assert cards[2]["in_view"] == 0          # cluster 2 has none
    assert cards[1]["undecided"] == 0 and cards[2]["undecided"] == 6

    r = client.get(f"/api/clusters?label_id={lid}&scope=undecided")
    cards = {c["id"]: c for c in r.get_json()["clusters"]}
    assert cards[1]["in_view"] == 0 and cards[2]["in_view"] == 6

    # A source nothing matches zeroes every card without touching size.
    r = client.get(
        f"/api/clusters?label_id={lid}&scope=with&source=other")
    cards = {c["id"]: c for c in r.get_json()["clusters"]}
    assert all(c["in_view"] == 0 and c["size"] == 6
               for c in cards.values())
    # Without a label the cards only report source-matched counts.
    r = client.get("/api/clusters?source=imgs")
    cards = {c["id"]: c for c in r.get_json()["clusters"]}
    assert all(c["in_view"] == 6 for c in cards.values())


def test_queue_pred_filter_and_prob_sort(tmp_path):
    """pred_label keeps only cells whose classify argmax matches; sort=prob
    orders by P(class) — the picked class, or argmax confidence when no
    class is picked."""
    cs = _ClusterServer(tmp_path)
    srv = cs.srv
    lid = cs.label("A")
    n = len(srv.cells)
    rng = np.random.default_rng(1)
    # Two classes; rows 0..2 predict class "a", the rest "b" with varying
    # confidence. srv.cells order == row order here.
    probs = rng.uniform(0.05, 0.4, size=(n, 2)).astype(np.float32)
    probs[:3, 0] = 0.9
    probs[3:, 1] = 0.95
    srv.prob_matrix = probs
    srv.class_names = ["a", "b"]
    srv.class_index = {"a": 0, "b": 1}
    srv._pred_argmax = np.argmax(probs, axis=1)
    srv._pred_maxprob = probs[np.arange(n), srv._pred_argmax]

    r = cs.client().get(
        f"/api/queue?label_id={lid}&scope=undecided&pred_label=a&sort=prob")
    body = r.get_json()
    assert body["total"] == 3
    assert [c["pred"]["class"] for c in body["cells"]] == ["a"] * 3
    conf = [c["score"] for c in body["cells"]]
    assert conf == sorted(conf, reverse=True)

    # No class picked: the whole registry, ranked by argmax confidence.
    r = cs.client().get(
        f"/api/queue?label_id={lid}&scope=undecided&sort=prob")
    body = r.get_json()
    assert body["total"] == n
    conf = [c["score"] for c in body["cells"]]
    assert conf == sorted(conf, reverse=True)
    # An unknown class name is rejected.
    assert cs.client().get(
        f"/api/queue?label_id={lid}&scope=undecided&pred_label=zzz"
        f"&sort=prob").status_code == 400
