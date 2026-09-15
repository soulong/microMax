"""Tests for the label app's write/undo/auto mechanics.

Covers the rigorous-write path introduced with the label/ package: the
auto-annotate pass must fire on BATCH threshold jumps (the old equality
check silently skipped 19 -> 22 positive jumps), "Remove auto" must reset
the per-label fired marker, every user action is one undoable op (batch
applies across several labels included), legacy project DBs migrate to the
unified label.db, explicit negatives store/export correctly, and the
leave-one-out review flags inconsistent decisions.
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

def _make_server(tmp_path, n_cells, min_positives=5):
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
        "auto_label": {"min_positives": min_positives, "threshold": 0.9},
    }
    srv = LabelServer(config, open_browser=False)
    srv.app.run = lambda **kw: None
    srv.start()
    return srv


def _identity_space(srv, n):
    W = np.eye(n, dtype=np.float32)
    srv.engine = SuggestEngine(W, knn_k=1, neg_weight=0.5)
    srv.ml_engine = None
    srv.cid_row = {cell["cell_id"]: i for i, cell in enumerate(srv.cells)}
    srv.row_cid = {i: cell["cell_id"] for i, cell in enumerate(srv.cells)}
    return W


def test_auto_fires_on_batch_jump(tmp_path):
    """One batch crossing min_positives from below MUST run the auto pass —
    the old `n_pos == threshold` equality skipped jumps like 4 -> 7."""
    srv = _make_server(tmp_path, n_cells=10)
    c = srv.app.test_client()
    W = _identity_space(srv, 10)
    lb = c.post("/api/labels", json={"name": "mitotic"}).get_json()
    lid = lb["label_id"]
    # Cells 7-9 are near-clones of positive cell 0: confident candidates.
    W[7:10] = W[0] + 0.01
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    fps = [cell["filepath"] for cell in srv.cells[:7]]
    resp = c.post("/api/annotate_batch", json={
        "filepaths": fps, "label_id": lid, "state": 1}).get_json()
    assert resp["ok"] is True
    assert resp["auto_applied"] == 3                  # cells 7-9 annotated
    q = c.get(f"/api/queue?mode=label_all&scope=with&label_id={lid}").get_json()
    autos = [cell for cell in q["cells"] if cell.get("auto")]
    assert len(autos) == 3
    # The fired label does NOT re-run on the next write: more positives in
    # the clone cluster change nothing until Remove auto resets the marker.
    resp = c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[10]["filepath"]], "label_id": lid,
        "state": 1}).get_json() if len(srv.cells) > 10 else None


def test_remove_auto_resets_and_reapplies(tmp_path):
    """Remove auto clears the auto decisions AND the fired marker, so the
    next positive write re-runs the pass (fresh, with current scores)."""
    srv = _make_server(tmp_path, n_cells=10)
    c = srv.app.test_client()
    W = _identity_space(srv, 10)
    lb = c.post("/api/labels", json={"name": "mitotic"}).get_json()
    lid = lb["label_id"]
    W[7:10] = W[0] + 0.01
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    c.post("/api/annotate_batch", json={
        "filepaths": [cell["filepath"] for cell in srv.cells[:5]],
        "label_id": lid, "state": 1})
    j = c.post("/api/auto_clear", json={"label_id": lid}).get_json()
    assert j["removed"] == 3
    q = c.get(f"/api/queue?mode=label_all&scope=with&label_id={lid}").get_json()
    assert q["total"] == 5                            # manual ones survive
    # One more positive write -> the pass re-runs and re-annotates 7-9.
    resp = c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[5]["filepath"]], "label_id": lid,
        "state": 1}).get_json()
    assert resp["auto_applied"] == 3


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


def test_explicit_negative_powers_and_survives(tmp_path):
    """Apply − stores explicit negatives that veto the kNN suggestion and
    survive Remove-auto / clears (only auto rows are ever bulk-removed)."""
    srv = _make_server(tmp_path, n_cells=8)
    c = srv.app.test_client()
    W = _identity_space(srv, 8)
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
    # A positive write runs the auto pass; the explicit negative is never
    # touched by it (only undecided cells become auto positives).
    W[2:5] = W[0] + 0.01
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    srv.db.mark_auto_fired(lid)  # below min_positives anyway; keep control
    resp = c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[6]["filepath"]], "label_id": lid,
        "state": 1}).get_json()
    assert resp["ok"] is True
    assert states[(srv.cells[1]["cell_id"], lid)] == 0
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


def test_single_auto_skips_positive_holders(tmp_path):
    """Single mode: the auto pass must never hand a second positive to a
    cell that already holds another label's positive (exclusivity)."""
    srv = _make_server(tmp_path, n_cells=8, min_positives=2)
    c = srv.app.test_client()
    c.post("/api/label_mode", json={"mode": "single"})
    W = _identity_space(srv, 8)   # AFTER the switch: cell ids are single-DB
    la = c.post("/api/labels", json={"name": "class_a"}).get_json()["label_id"]
    lb = c.post("/api/labels", json={"name": "class_b"}).get_json()["label_id"]
    # Cells 2/3 are near-clones of A-positive cell 0: the A pass picks them.
    W[2:4] = W[0] + 0.01
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[0]["filepath"], srv.cells[1]["filepath"]],
        "label_id": la, "state": 1})
    states = srv.db.cell_states()
    assert states[(srv.cells[2]["cell_id"], la)] == 1
    # Now cells 6/7 AND the A-holding 2/3 all look like B-positive clones.
    W[2:4] = W[4] + 0.01
    W[6:8] = W[4] + 0.01
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    srv._bump_engines()
    c.post("/api/annotate_batch", json={
        "filepaths": [srv.cells[4]["filepath"], srv.cells[5]["filepath"]],
        "label_id": lb, "state": 1})
    states = srv.db.cell_states()
    # Cells 6/7 (no prior positive) got B; the A-holding 2/3 did NOT.
    assert states[(srv.cells[6]["cell_id"], lb)] == 1
    assert states[(srv.cells[7]["cell_id"], lb)] == 1
    assert (srv.cells[2]["cell_id"], lb) not in states
    assert states[(srv.cells[2]["cell_id"], la)] == 1


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
