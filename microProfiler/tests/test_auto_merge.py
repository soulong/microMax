"""The inference step's auto-merge: merge_<mask>.db per inference block."""

import sqlite3

import pandas as pd

from microProfiler.io.database import Database
from microProfiler.pipeline.steps import _auto_merge_infer


def _make_profiler_db(path):
    """One object table 'cell' (bookkept) + one custom-named 'obj_nuc'."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE cell (well TEXT, label INTEGER, directory TEXT, "
                 "area REAL)")
    conn.execute("INSERT INTO cell VALUES ('A1', 1, 'C:/ds/Images', 10.0)")
    conn.execute("INSERT INTO cell VALUES ('A1', 2, 'C:/ds/Images', 20.0)")
    conn.execute("CREATE TABLE obj_nuc (well TEXT, label INTEGER, directory TEXT,"
                 " area_n REAL)")
    conn.execute("INSERT INTO obj_nuc VALUES ('A1', 1, 'C:/ds/Images', 99.0)")
    conn.commit()
    conn.close()
    db = Database(str(path))
    try:
        db.record_table_mask("cell", "cell")
        db.record_table_mask("obj_nuc", "nuclei")
    finally:
        db.close()


def _make_infer_db(path, mask_name):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE inference (uid INTEGER PRIMARY KEY, directory TEXT, "
        "mask_name TEXT, label INTEGER, pred_class TEXT, pred_prob REAL)")
    conn.execute("INSERT INTO inference VALUES (1, 'C:/ds/Images', ?, 1, "
                 "'drug', 0.9)", (mask_name,))
    conn.execute("INSERT INTO inference VALUES (2, 'C:/ds/Images', ?, 2, "
                 "'ctrl', 0.8)", (mask_name,))
    conn.execute("CREATE TABLE reduction_umap (uid INTEGER PRIMARY KEY, "
                 "umap_1 REAL, umap_2 REAL)")
    conn.execute("INSERT INTO reduction_umap VALUES (1, 1.0, 2.0)")
    conn.execute("INSERT INTO reduction_umap VALUES (2, 3.0, 4.0)")
    conn.commit()
    conn.close()


class _Entry:
    """Just the attribute _auto_merge_infer reads from an InferenceEntry."""

    def __init__(self, mask_name):
        self.mask_name = mask_name


def test_auto_merge_writes_per_mask_db(tmp_path):
    root = tmp_path
    _make_profiler_db(root / "profiler.db")
    _make_infer_db(root / "infer.db", "cell")

    _auto_merge_infer(root, _Entry("cell"), root / "infer.db")

    out = root / "merge_cell.db"
    assert out.exists()
    conn = sqlite3.connect(str(out))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    df = pd.read_sql("SELECT * FROM merged", conn)
    mapping = dict(conn.execute(
        "SELECT table_name, mask_name FROM _table_masks").fetchall())
    conn.close()

    # Fused per object: profiler measurements + infer prediction + umap.
    assert len(df) == 2
    assert {"area", "pred_class", "umap_1", "well", "label"} <= set(df.columns)
    assert (df["pred_class"] == ["drug", "ctrl"]).all()
    # Bookkeeping lets the file re-load as mask 'cell'.
    assert "merged" in tables and mapping["merged"] == "cell"

    # Sources untouched.
    src = sqlite3.connect(str(root / "profiler.db"))
    assert src.execute("SELECT COUNT(*) FROM cell").fetchone()[0] == 2
    src.close()
    src = sqlite3.connect(str(root / "infer.db"))
    assert src.execute("SELECT COUNT(*) FROM inference").fetchone()[0] == 2
    src.close()


def test_auto_merge_skips_without_matching_objects(tmp_path, caplog):
    root = tmp_path
    _make_profiler_db(root / "profiler.db")
    _make_infer_db(root / "infer.db", "spindle")   # no profiler table for it

    _auto_merge_infer(root, _Entry("spindle"), root / "infer.db")

    assert not (root / "merge_spindle.db").exists()
    assert any("no object table for mask 'spindle'" in r.message
               for r in caplog.records)


def test_auto_merge_mask_falls_back_to_entry(tmp_path):
    """A legacy infer DB without mask_name uses the entry's mask + hint."""
    root = tmp_path
    _make_profiler_db(root / "profiler.db")
    conn = sqlite3.connect(str(root / "infer.db"))
    conn.execute(
        "CREATE TABLE inference (uid INTEGER PRIMARY KEY, directory TEXT, "
        "label INTEGER, pred_class TEXT)")
    conn.execute("INSERT INTO inference VALUES (1, 'C:/ds/Images', 1, 'drug')")
    conn.commit()
    conn.close()

    _auto_merge_infer(root, _Entry("cell"), root / "infer.db")
    assert (root / "merge_cell.db").exists()
