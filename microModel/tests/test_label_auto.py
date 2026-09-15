"""Tests for the label DB decision-source and the auto-annotate flow.

Covers the AnnotationDB side (manual/auto source, idempotent auto_apply,
selective clear_auto, schema migration for pre-source projects) and a
server-level smoke: crossing the positive threshold auto-annotates
confident undecided cells (fake embedding injected).
"""

import numpy as np
import pytest
import tifffile

from microBase import MicroMaxError
from microModel.label import AnnotationDB, LabelServer, SuggestEngine


def test_source_defaults_to_manual(tmp_path):
    db = AnnotationDB(str(tmp_path / "label_multiple.db"))
    db.set_label(1, 1, state=1, session_id=7)
    assert db.cell_states()[(1, 1)] == 1


def test_auto_apply_skips_decided_and_clear_keeps_manual(tmp_path):
    db = AnnotationDB(str(tmp_path / "label_multiple.db"))
    db.set_label(1, 2, state=1, session_id=7)      # manual positive
    db.set_label(2, 2, state=0, session_id=7)      # manual negative
    # Cells 2/3/4 requested, but cell 2 already has a decision -> skipped.
    assert db.auto_apply(2, [2, 3, 4], session_id=8) == 2
    assert db.auto_cells(2) == {3, 4}
    assert db.cell_states()[(3, 2)] == 1
    # A manual edit demotes an auto row back to manual.
    db.set_label(3, 2, state=0, session_id=9)
    assert 3 not in db.auto_cells(2)
    # Clearing auto removes only auto rows; manual decisions survive.
    assert db.clear_auto(2, session_id=9) == 1     # cell 4 only
    assert db.cell_states() == {(1, 2): 1, (2, 2): 0, (3, 2): 0}
    assert db.auto_cells(2) == set()


def test_old_project_db_migrates(tmp_path):
    """A pre-source project DB gains the column on open and keeps working."""
    import sqlite3
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
    assert db.auto_cells(6) == set()
    db.set_label(5, 6, state=0, session_id=2)       # write on migrated table
    assert db.cell_states()[(5, 6)] == 0


def _make_server(tmp_path, n_cells):
    from microModel.label import LabelServer
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
        "auto_label": {"min_positives": 5, "threshold": 0.9},
    }
    srv = LabelServer(config, open_browser=False)
    srv.app.run = lambda **kw: None
    srv.start()
    return srv


def test_auto_apply_triggers_on_threshold(tmp_path):
    """Crossing min_positives auto-annotates confident undecided cells;
    the Manage queue lists them with the auto flag and Remove auto clears
    exactly the auto decisions (manual ones survive)."""
    from microModel.label import SuggestEngine
    srv = _make_server(tmp_path, n_cells=10)
    c = srv.app.test_client()
    # Hand-built space: cells 0-4 mutually orthogonal, cells 5-9 elsewhere.
    W = np.eye(10, dtype=np.float32)
    W[5:] = np.full((5, 10), 0.2, dtype=np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    srv.engine = SuggestEngine(W, knn_k=1, neg_weight=0.5)
    srv.ml_engine = None
    srv.cid_row = {cell["cell_id"]: i for i, cell in enumerate(srv.cells)}
    srv.row_cid = {i: cell["cell_id"] for i, cell in enumerate(srv.cells)}

    lb = c.post("/api/labels", json={"name": "mitotic"}).get_json()
    lid = lb["label_id"]
    fps = [cell["filepath"] for cell in srv.cells[:5]]
    resp = c.post("/api/annotate_batch", json={
        "filepaths": fps, "label_id": lid, "state": 1}).get_json()
    assert resp["ok"] is True
    # At the crossing the far cells 5-9 score ~0.32 < 0.9: nothing applied.
    assert resp["auto_applied"] == 0
    # Make cells 5-9 near-clones of positive cell 0, then rerun auto: the
    # confident clones are auto-annotated, and the queue flags them.
    W[5:] = W[0] + 0.01
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    srv._bump_engines()
    j = c.post("/api/auto_apply", json={"label_id": lid}).get_json()
    assert j["applied"].get("mitotic") == 5, j
    q = c.get(f"/api/queue?mode=label_all&label_id={lid}").get_json()
    autos = [cell for cell in q["cells"] if cell.get("auto")]
    assert len(autos) == 5
    # Remove auto: the manual five survive, the auto five are gone.
    j = c.post("/api/auto_clear", json={"label_id": lid}).get_json()
    assert j["removed"] == 5
    q = c.get(f"/api/queue?mode=label_all&label_id={lid}").get_json()
    assert q["total"] == 5
