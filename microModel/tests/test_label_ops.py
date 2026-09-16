"""Tests for the label app's write/undo mechanics and queue ranking.

Covers the AnnotationDB write path (batch applies, undo, exports, legacy
DB migration), the Collect ranking (exemplar scores, explicit negatives
pushing lookalikes down, multi-label batches writing every target label),
the leave-one-out review that flags inconsistent decisions, and the manual
Refresh-model logistic fits (model scoring, staleness, skips, mode drops).
"""

import os
import sqlite3

import numpy as np
import pytest
import tifffile

from microModel.label import (AnnotationDB, LabelServer, SuggestEngine,
                              migrate_project_db, review_items)


# ----------------------------------------------------------------------------
# DB-level helpers
# ----------------------------------------------------------------------------

def test_migrate_project_db_routes_names(tmp_path):
    """Legacy projects route to their mode-specific names on startup.

    - label_multiple.db (v0.20 multi data) is the multi target and stays;
    - a stray v0.21 unified label.db routes by the mode recorded in ITS
      meta (multi -> label_multiple.db, single -> label_single.db);
    - a label.db whose target name is taken is set aside as .old —
      nothing is ever overwritten."""
    legacy = AnnotationDB(str(tmp_path / "label_multiple.db"))
    lb = legacy.add_label("mitotic")
    legacy.set_label(1, lb["label_id"], state=1, session_id=1)
    legacy = None  # close (garbage-collected connection released below)

    multi, single = migrate_project_db(str(tmp_path))
    assert multi == str(tmp_path / "label_multiple.db")
    assert single == str(tmp_path / "label_single.db")
    assert os.path.exists(multi)          # the multi target, untouched
    db = AnnotationDB(multi)
    assert db.list_labels()[0]["name"] == "mitotic"
    assert db.cell_states()[(1, lb["label_id"])] == 1
    assert migrate_project_db(str(tmp_path)) == (multi, single)  # idempotent


def test_migrate_unified_label_db_routes_by_meta(tmp_path):
    """A v0.21 unified label.db goes back to the project name its meta
    records; when that name is already taken it is set aside as .old."""
    import sqlite3

    def make_unified(path, mode):
        conn = sqlite3.connect(path)
        conn.executescript(
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);"
            "INSERT INTO meta VALUES ('label_mode', '%s');" % mode)
        conn.commit()
        conn.close()

    # A multi-mode label.db with no label_multiple.db around: it IS multi.
    make_unified(str(tmp_path / "label.db"), "multi")
    multi, single = migrate_project_db(str(tmp_path))
    assert os.path.exists(multi) and not os.path.exists(str(tmp_path / "label.db"))

    # A single-mode label.db routes to label_single.db.
    make_unified(str(tmp_path / "label.db"), "single")
    multi, single = migrate_project_db(str(tmp_path))
    assert os.path.exists(single) and not os.path.exists(str(tmp_path / "label.db"))

    # Both targets present: the stray label.db is set aside, never deleted.
    make_unified(str(tmp_path / "label.db"), "multi")
    migrate_project_db(str(tmp_path))
    assert os.path.exists(str(tmp_path / "label.db.old"))


def test_export_only_positives_and_clear(tmp_path):
    """Explicit negatives (state 0) are decisions but never export; a
    positive exports and a clear removes it again."""
    db = AnnotationDB(str(tmp_path / "label.db"))
    # Real flow: cells are registered before any decision exists.
    db.register_cells([("c1", "C1.tif", "src", None),
                       ("c2", "C2.tif", "src", None),
                       ("c3", "C3.tif", "src", None)])
    lb = db.add_label("bright")
    assert db.export_frame() is None
    db.set_label(1, lb["label_id"], state=0, session_id=1)
    assert db.export_frame() is None
    db.set_label(2, lb["label_id"], state=1, session_id=1)
    df = db.export_frame()
    assert list(df.columns) == ["filepath", "label"]
    assert df.iloc[0]["label"] == "bright"
    db.set_label(2, lb["label_id"], None, session_id=1)
    assert db.export_frame() is None
    st = db.stats()
    assert st["total"] == 3 and st["labeled"] == 1  # cell 1 keeps its negative


def test_undo_restores_previous_state_exactly(tmp_path):
    """Undo walks the log backwards: a cell annotated twice then re-decided
    returns to its true previous state, not just 'undecided'. Repeated
    undos keep walking back through earlier actions (undo ops are skipped
    as targets), ending cleanly at 'nothing to undo'."""
    db = AnnotationDB(str(tmp_path / "label.db"))
    a = db.add_label("a")["label_id"]
    b = db.add_label("b")["label_id"]
    db.set_label(1, a, 1, session_id=1, op_id=1)       # a+ on cell 1
    db.set_label(1, b, 0, session_id=1, op_id=2)       # b- on cell 1
    db.set_label(1, a, 0, session_id=1, op_id=3)       # a overwritten to -
    r = db.undo_last_op(1)
    assert r["n"] == 1
    assert db.cell_states()[(1, 1)] == 1               # a back to +
    assert db.cell_states()[(1, 2)] == 0               # b untouched
    r = db.undo_last_op(1)                             # undo op 2
    assert (1, 2) not in db.cell_states()
    r = db.undo_last_op(1)                             # undo op 1
    assert (1, 1) not in db.cell_states()
    assert db.cell_states() == {}
    assert db.undo_last_op(1) is None                  # nothing left


def test_old_project_db_migrates(tmp_path):
    """A pre-op_id project DB opens on the new schema and keeps working."""
    p = str(tmp_path / "label_multiple.db")
    conn = sqlite3.connect(p)
    conn.executescript("""
        CREATE TABLE cell_labels(cell_id INTEGER, label_id INTEGER,
            state INTEGER, updated_at TEXT, session_id INTEGER,
            PRIMARY KEY (cell_id, label_id));
        INSERT INTO cell_labels VALUES (5, 6, 1, 't', 1);
    """)
    conn.commit()
    conn.close()
    db = AnnotationDB(p)                            # migration runs here
    assert db.cell_states()[(5, 6)] == 1            # pre-existing decision
    db.set_label(5, 6, state=0, session_id=2)       # write on migrated table
    assert db.cell_states()[(5, 6)] == 0


def test_review_items_flags_inconsistent_decisions():
    """A positive sitting next to a negative and far from its own positives
    is flagged first, with the contradicting cell as evidence."""
    # Row 0 = "positive" but closest to the negative cluster -> suspicious.
    W = np.zeros((6, 4), dtype=np.float32)
    W[1] = W[2] = [1, 0, 0, 0]      # two consistent positives
    W[3] = W[4] = [0, 1, 0, 0]      # two consistent negatives
    W[0] = [0.1, 0.95, 0, 0]        # the mislabeled positive
    W[5] = [0.95, 0.1, 0, 0]        # a suspicious negative (mirror case)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    row_cid = {0: 100, 1: 101, 2: 102, 3: 103, 4: 104, 5: 105}
    items = review_items(W, row_cid, pos_rows=[0, 1, 2], neg_rows=[3, 4, 5])
    ids = [it["cell_id"] for it in items]
    assert 100 in ids                                # the bad positive
    assert 105 in ids                                # the bad negative
    bad = next(it for it in items if it["cell_id"] == 100)
    assert bad["state"] == 1 and bad["susp"] > 0
    assert bad["ev_cell_id"] in (103, 104, 105)      # a negative is evidence


# ----------------------------------------------------------------------------
# Server-level: one page of TIFFs + a hand-built embedding space
# ----------------------------------------------------------------------------

def _make_server(tmp_path, n_cells):
    root = tmp_path / "cells"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_cells):
        arr = rng.uniform(0, 65535, size=(16, 16)).astype(np.uint16)
        tifffile.imwrite(root / f"cell_{i:02d}.tif", arr)
    config = {
        "save_dir": str(tmp_path / "proj"),
        "model": None,
        "data": {"file_dir": [str(root)], "channels": [1],
                 "channel_layout": None, "max_value": 65535},
    }
    srv = LabelServer(config, open_browser=False)
    srv.app.run = lambda **kw: None
    srv.start()
    return srv


def _identity_space(srv, n):
    W = np.eye(n, dtype=np.float32)
    srv.engine = SuggestEngine(W, knn_k=1, neg_weight=0.5)
    srv.cid_row = {cell["cell_id"]: i for i, cell in enumerate(srv.cells)}
    srv.row_cid = {i: cell["cell_id"] for i, cell in enumerate(srv.cells)}
    return W


def test_collect_ranking_prefers_lookalikes(tmp_path):
    """Collect ranks undecided cells by the exemplar score: cells close to
    the positives lead, unrelated cells trail."""
    srv = _make_server(tmp_path, n_cells=10)
    c = srv.app.test_client()
    W = _identity_space(srv, 10)
    # Cells 2-4 become near-clones of positive cell 0.
    W[2:5] = W[0] + 0.01
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    lid = c.post("/api/labels", json={"name": "mitotic"}).get_json()["label_id"]
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[0]["filepath"]], "label_id": lid, "state": 1})
    q = c.get(f"/api/queue?scope=undecided&label_id={lid}&limit=10").get_json()
    names = [cell["filename"] for cell in q["cells"]]
    assert set(names[:3]) == {"cell_02.tif", "cell_03.tif", "cell_04.tif"}
    # Every cell carries the score the queue sorted by.
    assert all(cell["score"] is not None for cell in q["cells"])
    assert q["cells"][0]["score"] > q["cells"][-1]["score"]


def test_explicit_negative_pushes_lookalike_down(tmp_path):
    """An explicit negative is a ranking term, not just a veto: the cell it
    resembles loses the top spot to a less similar undecided cell."""
    srv = _make_server(tmp_path, n_cells=7)
    c = srv.app.test_client()
    u = np.zeros(4, dtype=np.float32); u[0] = 1.0
    v = np.zeros(4, dtype=np.float32); v[1] = 1.0
    w1 = np.zeros(4, dtype=np.float32); w1[2] = 1.0
    w2 = np.zeros(4, dtype=np.float32); w2[3] = 1.0
    def unit(x):
        return (x / np.linalg.norm(x)).astype(np.float32)
    W = np.vstack([
        unit(u + 0.05 * v),                     # 0 positive exemplar
        unit(u + 0.05 * v + 0.01 * w1),         # 1 positive exemplar
        unit(u + 0.30 * v),                     # 2 top lookalike
        unit(u - 0.30 * v),                     # 3 the other lookalike
        unit(u + 0.30 * v + 0.001 * w1),        # 4 negative ~ clone of 2
        unit(w1),                               # 5 unrelated
        unit(w2),                               # 6 unrelated
    ])
    srv.engine = SuggestEngine(W, knn_k=1, neg_weight=0.5)
    srv.cid_row = {cell["cell_id"]: i for i, cell in enumerate(srv.cells)}
    srv.row_cid = {i: cell["cell_id"] for i, cell in enumerate(srv.cells)}
    lid = c.post("/api/labels", json={"name": "mitotic"}).get_json()["label_id"]
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[0]["filepath"], srv.cells[1]["filepath"]],
        "label_id": lid, "state": 1})
    q1 = c.get(f"/api/queue?scope=undecided&label_id={lid}&limit=10").get_json()
    assert q1["cells"][0]["filename"] == "cell_02.tif"
    # Mark cell_05 (a clone of the leader) as an explicit negative.
    c.post("/api/annotate", json={
        "filepath": srv.cells[4]["filepath"], "label_id": lid, "state": 0})
    q2 = c.get(f"/api/queue?scope=undecided&label_id={lid}&limit=10").get_json()
    assert q2["cells"][0]["filename"] == "cell_03.tif"
    scores1 = {cell["filename"]: cell["score"] for cell in q1["cells"]}
    scores2 = {cell["filename"]: cell["score"] for cell in q2["cells"]}
    assert scores2["cell_02.tif"] < scores1["cell_02.tif"]


def test_undo_batch_and_multilabel_op(tmp_path):
    """A batch apply (one or several labels) is ONE undoable op; repeated
    undos walk back through successive actions, each exactly once."""
    srv = _make_server(tmp_path, n_cells=8)
    c = srv.app.test_client()
    _identity_space(srv, 8)
    l1 = c.post("/api/labels", json={"name": "a"}).get_json()["label_id"]
    l2 = c.post("/api/labels", json={"name": "b"}).get_json()["label_id"]
    fps = [cell["filepath"] for cell in srv.cells[:3]]
    resp = c.post("/api/annotate_batch", json={
        "filepaths": fps, "label_ids": [l1, l2], "state": 1}).get_json()
    assert resp["ok"] is True and resp["n"] == 6      # 3 cells x 2 labels
    assert srv.db.stats()["labeled"] == 3
    # Undo reverts BOTH labels of all three cells in one step.
    j = c.post("/api/undo").get_json()
    assert j["undone"] is True and j["n"] == 6
    assert srv.db.cell_states() == {}
    # Nothing further to undo: a clean answer, not an error.
    j = c.post("/api/undo").get_json()
    assert j["undone"] is False
    # A second action is undone independently of the first.
    c.post("/api/annotate_batch", json={
        "filepaths": [fps[0]], "label_id": l1, "state": 1})
    c.post("/api/annotate_batch", json={
        "filepaths": [fps[1]], "label_id": l2, "state": 0})
    j = c.post("/api/undo").get_json()
    assert j["undone"] is True and j["n"] == 1        # the last write
    states = srv.db.cell_states()
    assert list(states) == [(srv.cells[0]["cell_id"], l1)]
    j = c.post("/api/undo").get_json()
    assert j["undone"] is True and j["n"] == 1        # the first write
    assert srv.db.cell_states() == {}


def test_explicit_negative_survives_and_stays_out_of_export(tmp_path):
    """Apply − stores explicit negatives that stay as ranking terms and
    never leak into the positive-only training export."""
    srv = _make_server(tmp_path, n_cells=8)
    c = srv.app.test_client()
    _identity_space(srv, 8)
    lid = c.post("/api/labels", json={"name": "mitotic"}).get_json()["label_id"]
    # Cell 0 positive, cell 1 an explicit negative of the same label.
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[0]["filepath"]], "label_id": lid,
        "state": 1})
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[1]["filepath"]], "label_id": lid,
        "state": 0})
    states = srv.db.cell_states()
    assert states[(srv.cells[0]["cell_id"], lid)] == 1
    assert states[(srv.cells[1]["cell_id"], lid)] == 0
    # A later positive write leaves the negative untouched.
    resp = c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[6]["filepath"]], "label_id": lid,
        "state": 1}).get_json()
    assert resp["ok"] is True
    assert srv.db.cell_states()[(srv.cells[1]["cell_id"], lid)] == 0
    # Export: only the positives.
    df = srv.db.export_frame()
    assert list(df["filepath"]).count(srv.cells[1]["raw_path"]) == 0
    assert list(df["filepath"]).count(srv.cells[0]["raw_path"]) == 1


def test_mode_isolation_and_switch(tmp_path):
    """The two modes are fully independent projects in one save_dir:
    labels and decisions never cross, switching back restores everything,
    and the pointer file remembers the last mode."""
    srv = _make_server(tmp_path, n_cells=6)
    c = srv.app.test_client()
    _identity_space(srv, 6)
    lid = c.post("/api/labels", json={"name": "multi_only"}).get_json()["label_id"]
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[0]["filepath"]], "label_id": lid,
        "state": 1})
    # Switch to single: a fresh, empty store — nothing crossed over.
    j = c.post("/api/label_mode", json={"mode": "single"}).get_json()
    assert j["label_mode"] == "single"
    assert srv.labels_cache == [] and srv.db.cell_states() == {}
    assert srv.db.stats()["total"] == 6      # cells re-registered
    # The same NAME creates an independent label in the other store (the
    # numeric id may even coincide — they live in different databases).
    lid2 = c.post("/api/labels", json={"name": "multi_only"}).get_json()["label_id"]
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[1]["filepath"]], "label_id": lid2,
        "state": 1})
    assert srv.db.cell_states() == {(srv.cells[1]["cell_id"], lid2): 1}
    # Switch back: the multi project is exactly as it was.
    c.post("/api/label_mode", json={"mode": "multi"})
    assert [lb_["name"] for lb_ in srv.labels_cache] == ["multi_only"]
    assert srv.db.cell_states() == {(srv.cells[0]["cell_id"], lid): 1}
    assert srv._read_mode_file() == "multi"


def test_single_exclusivity_and_undo(tmp_path):
    """Single mode: writing a positive clears the cell's other positives
    (keep-set semantics) and ONE undo reverts the writes AND the clears."""
    srv = _make_server(tmp_path, n_cells=6)
    c = srv.app.test_client()
    _identity_space(srv, 6)
    c.post("/api/label_mode", json={"mode": "single"})
    la = c.post("/api/labels", json={"name": "class_a"}).get_json()["label_id"]
    lb = c.post("/api/labels", json={"name": "class_b"}).get_json()["label_id"]
    fps = [srv.cells[0]["filepath"], srv.cells[1]["filepath"]]
    c.post("/api/annotate_batch", json={
        "filepaths": fps, "label_id": la, "state": 1})
    # Mark the same cells class_b: the class_a positives must go.
    c.post("/api/annotate_batch", json={
        "filepaths": fps, "label_id": lb, "state": 1})
    states = srv.db.cell_states()
    assert states[(srv.cells[0]["cell_id"], lb)] == 1
    assert (srv.cells[0]["cell_id"], la) not in states
    # One undo reverts the whole action: the two writes AND the two clears.
    j = c.post("/api/undo").get_json()
    assert j["undone"] is True and j["n"] == 4
    states = srv.db.cell_states()
    assert states[(srv.cells[0]["cell_id"], la)] == 1
    assert (srv.cells[0]["cell_id"], lb) not in states


def test_export_files_per_mode(tmp_path):
    """Each mode writes its own export CSV; the stores stay separate."""
    srv = _make_server(tmp_path, n_cells=4)
    c = srv.app.test_client()
    lid = c.post("/api/labels", json={"name": "morph"}).get_json()["label_id"]
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[0]["filepath"]], "label_id": lid,
        "state": 1})
    multi_csv = os.path.join(srv.save_dir, "label_export.csv")
    assert os.path.exists(multi_csv)
    c.post("/api/label_mode", json={"mode": "single"})
    lid2 = c.post("/api/labels", json={"name": "class_x"}).get_json()["label_id"]
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[1]["filepath"]], "label_id": lid2,
        "state": 1})
    single_csv = os.path.join(srv.save_dir, "label_export_single.csv")
    assert os.path.exists(single_csv)
    j = c.post("/api/export").get_json()
    assert j["path"] == single_csv and j["rows"] == 1
    with open(single_csv, encoding="utf-8") as f:
        assert "class_x" in f.read()
    with open(multi_csv, encoding="utf-8") as f:
        assert "morph" in f.read()


def test_label_colors_stay_distinct(tmp_path):
    """New labels take the first UNUSED palette color (not the count-based
    one, so deletions can never cause repeats), and a DB whose labels
    somehow share a color is repaired on open: the first keeper keeps it,
    later duplicates get unused colors."""
    from microModel.label import PALETTE
    p = str(tmp_path / "l.db")
    db = AnnotationDB(p)
    a = db.add_label("a")
    b = db.add_label("b")
    # Simulate the historical corruption: two labels, one shared color.
    conn = sqlite3.connect(p)
    conn.executemany("UPDATE labels SET color = ? WHERE label_id = ?",
                     [(PALETTE[0], a["label_id"]), (PALETTE[0], b["label_id"])])
    conn.commit()
    conn.close()
    # A new label skips the taken color instead of counting.
    c = db.add_label("c")
    assert c["color"] == PALETTE[1]
    # Reopening repairs the duplicates: three distinct colors remain.
    rows = AnnotationDB(p).list_labels()
    colors = {lb["name"]: lb["color"] for lb in rows}
    assert len(set(colors.values())) == 3
    assert colors["a"] == PALETTE[0]            # first keeper keeps it
    assert colors["b"] not in (colors["a"], colors["c"])


def test_manage_negatives_scope_reviews_and_clears(tmp_path):
    """The Manage 'neg' scope lists exactly the label's explicit negatives,
    ranked most positive-like first (the likely wrongly-marked ones carry
    a certainty badge), and Remove clears a selection of them."""
    srv = _make_server(tmp_path, n_cells=6)
    c = srv.app.test_client()
    u = np.zeros(4, dtype=np.float32); u[0] = 1.0
    v = np.zeros(4, dtype=np.float32); v[1] = 1.0
    def unit(x):
        return (x / np.linalg.norm(x)).astype(np.float32)
    W = np.vstack([u, u, u,                     # 0-2 positives
                   unit(u + 0.3 * v),           # 3 negative, very pos-like
                   v, v])                       # 4-5 negatives
    srv.engine = SuggestEngine(W, knn_k=1, neg_weight=0.5)
    srv.cid_row = {cell["cell_id"]: i for i, cell in enumerate(srv.cells)}
    srv.row_cid = {i: cell["cell_id"] for i, cell in enumerate(srv.cells)}
    lid = c.post("/api/labels", json={"name": "mitotic"}).get_json()["label_id"]
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[i]["filepath"] for i in (0, 1, 2)],
        "label_id": lid, "state": 1})
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[i]["filepath"] for i in (3, 4, 5)],
        "label_id": lid, "state": 0})

    # The neg scope lists exactly the three negatives; most positive-like
    # first under Certainty descending, each with its certainty badge.
    q = c.get(f"/api/queue?scope=neg&label_id={lid}"
              f"&sort=desc&limit=10").get_json()
    names = [cell["filename"] for cell in q["cells"]]
    assert names == ["cell_03.tif", "cell_04.tif", "cell_05.tif"]
    certs = [cell["cert"] for cell in q["cells"]]
    assert certs[0] > certs[1] and all(cv is not None for cv in certs)

    # Remove the suspicious negative: it vanishes from the scope and its
    # decision is really gone from the DB.
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[3]["filepath"]], "label_id": lid,
        "state": "clear"})
    q2 = c.get(f"/api/queue?scope=neg&label_id={lid}"
               f"&limit=10").get_json()
    assert [cell["filename"] for cell in q2["cells"]] == \
        ["cell_04.tif", "cell_05.tif"]
    assert (srv.cells[3]["cell_id"], lid) not in srv.db.cell_states()
    # The flipped workflow also works: Apply + makes it a positive again.
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[3]["filepath"]], "label_id": lid, "state": 1})
    assert srv.db.cell_states()[(srv.cells[3]["cell_id"], lid)] == 1
    # A neg scope without a known label is an error, like with / without.
    r = c.get("/api/queue?scope=neg&limit=10")
    assert r.status_code == 400


def test_refresh_model_scores_stales_and_skips(tmp_path):
    """Refresh model fits a per-label logistic scorer: a refreshed label's
    Collect score becomes the model probability, new writes mark it stale
    until the next refresh, labels below the exemplar minimum keep the kNN
    score, and a mode switch drops the fits entirely."""
    srv = _make_server(tmp_path, n_cells=12)
    c = srv.app.test_client()
    u = np.zeros(4, dtype=np.float32); u[0] = 1.0
    v = np.zeros(4, dtype=np.float32); v[1] = 1.0
    def unit(x):
        return (x / np.linalg.norm(x)).astype(np.float32)
    W = np.vstack(
        [unit(u + 0.02 * v) for _ in range(5)]     # 0-4 positive exemplars
        + [unit(v) for _ in range(4)]              # 5-8 explicit negatives
        + [unit(u + 0.05 * v),                     # 9 undecided, pos-like
           unit(v + 0.2 * u),                      # 10 undecided, neg-like
           unit(u + v)])                           # 11 undecided, boundary
    srv.engine = SuggestEngine(W, knn_k=1, neg_weight=0.5)
    srv.cid_row = {cell["cell_id"]: i for i, cell in enumerate(srv.cells)}
    srv.row_cid = {i: cell["cell_id"] for i, cell in enumerate(srv.cells)}
    lid = c.post("/api/labels", json={"name": "mitotic"}).get_json()["label_id"]
    lid2 = c.post("/api/labels", json={"name": "tiny"}).get_json()["label_id"]
    c.post("/api/annotate_batch", json={
        "filepaths": [cc["filepath"] for cc in srv.cells[:5]],
        "label_id": lid, "state": 1})
    c.post("/api/annotate_batch", json={
        "filepaths": [cc["filepath"] for cc in srv.cells[5:9]],
        "label_id": lid, "state": 0})
    # A second label with 2 positives and NO negative — below the fit bar.
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[0]["filepath"], srv.cells[1]["filepath"]],
        "label_id": lid2, "state": 1})

    j = c.post("/api/refresh_model").get_json()
    assert j["fitted"] == ["mitotic"] and j["skipped"] == ["tiny"]
    st = c.get("/api/state").get_json()
    assert st["models_fitted"] == 1 and st["models_stale"] is False
    # The Collect queue now scores undecided cells with the model
    # probability: the pos-like cell leads (> 0.5), the neg-like trails
    # (< 0.5), the boundary cell sits between them.
    q = c.get(f"/api/queue?scope=undecided&label_id={lid}&limit=10").get_json()
    by_name = {cell["filename"]: cell["score"] for cell in q["cells"]}
    assert set(by_name) == {"cell_09.tif", "cell_10.tif", "cell_11.tif"}
    s9, s10, s11 = (by_name[f"cell_{i:02d}.tif"] for i in (9, 10, 11))
    assert s9 > s11 > s10 and s9 > 0.5 > s10 and 0 < s11 < 1

    # A new write marks the fit stale but the model keeps scoring — the
    # undecided cells' scores are unchanged until the next refresh.
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[11]["filepath"]], "label_id": lid,
        "state": 0})
    st = c.get("/api/state").get_json()
    assert st["models_stale"] is True
    q2 = c.get(f"/api/queue?scope=undecided&label_id={lid}&limit=10").get_json()
    assert {cell["filename"]: cell["score"]
            for cell in q2["cells"]}["cell_09.tif"] == s9
    # Refreshing again catches up (and clears the flag).
    j = c.post("/api/refresh_model").get_json()
    assert j["fitted"] == ["mitotic"]
    st = c.get("/api/state").get_json()
    assert st["models_stale"] is False

    # A mode switch drops the fits — the other DB's decisions must not
    # leak into them.
    c.post("/api/label_mode", json={"mode": "single"})
    assert srv.label_models == {} and srv._model_scores == {}
