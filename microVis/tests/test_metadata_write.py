"""Tests for metadata merge + additive metadata writes to profiler/infer DBs."""

import sqlite3

import pandas as pd
import pytest

from microVis.io.data_module import aggregate_by_well, merge_metadata
from microVis.io.infer_db import InferDB
from microVis.io.metadata_write import write_metadata_columns
from microVis.io.profiler_db import ProfilerDB


def test_aggregate_by_well_methods():
    df = pd.DataFrame({
        "well": ["A1", "A1", "A2"],
        "value": [1.0, 3.0, 10.0],
        "treat": ["x", "y", "z"],
    })
    assert aggregate_by_well(df, "value", "mean") == {"A1": 2.0, "A2": 10.0}
    assert aggregate_by_well(df, "value", "sum") == {"A1": 4.0, "A2": 10.0}
    std = aggregate_by_well(df, "value", "std")
    assert std["A1"] == pytest.approx(2 ** 0.5)
    assert "A2" not in std  # single value -> NaN, dropped
    assert aggregate_by_well(df, "treat", "mean") == {"A1": "x", "A2": "z"}
    assert aggregate_by_well(df, "missing", "mean") == {}
    assert aggregate_by_well(pd.DataFrame({"value": [1.0]}), "value") == {}


# ── merge_metadata ────────────────────────────────────────────────────────


def test_merge_metadata_adds_missing_columns_only():
    df = pd.DataFrame({"well": ["A1", "A2"], "shape_area": [1.0, 2.0]})
    meta = pd.DataFrame({"well": ["A1", "A2"], "treat": ["x", "y"], "dose": [1, 2]})
    out = merge_metadata(df, meta)
    assert list(out.columns) == ["well", "shape_area", "treat", "dose"]
    assert out["treat"].tolist() == ["x", "y"]
    # Re-merging after a DB already has the columns must not duplicate them.
    out2 = merge_metadata(out, meta)
    assert list(out2.columns) == list(out.columns)
    assert out2["shape_area"].tolist() == [1.0, 2.0]


def test_merge_metadata_noop_without_key():
    df = pd.DataFrame({"value": [1.0]})
    meta = pd.DataFrame({"well": ["A1"], "treat": ["x"]})
    assert merge_metadata(df, meta).equals(df)
    assert merge_metadata(df, None).equals(df)
    assert merge_metadata(df, meta.drop(columns="well")).equals(df)


# ── write_metadata_columns ────────────────────────────────────────────────


def test_write_metadata_columns_skips_table_without_key(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "x.db"))
    conn.execute('CREATE TABLE "t" (value REAL)')
    conn.execute('INSERT INTO "t" VALUES (1.0)')
    conn.commit()
    meta = pd.DataFrame({"well": ["A1"], "treat": ["x"]})
    assert write_metadata_columns(conn, "t", meta) == 0
    conn.close()


# ── profiler DB ───────────────────────────────────────────────────────────


def _make_profiler_db(tmp_path):
    path = tmp_path / "profiler.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        'CREATE TABLE "cell" (directory TEXT, well TEXT, label INTEGER, shape_area REAL)'
    )
    for i, well in enumerate(["A1", "A2", "A3"]):
        conn.execute('INSERT INTO "cell" VALUES (?,?,?,?)',
                     ("D:/ds", well, i + 1, float(i)))
    conn.execute('CREATE TABLE "no_well" (value REAL)')
    conn.execute('INSERT INTO "no_well" VALUES (1.0)')
    conn.commit()
    conn.close()
    return path


def test_profiler_write_metadata_additive(tmp_path):
    path = _make_profiler_db(tmp_path)
    meta = pd.DataFrame({"well": ["A1", "A2"], "treat": ["x", "y"], "dose": [1, 2]})
    db = ProfilerDB(path)
    written = db.write_metadata(meta)
    assert written == 2  # two metadata columns

    df = db.load_table("cell")
    by_well = dict(zip(df["well"], df["treat"]))
    assert by_well["A1"] == "x" and by_well["A2"] == "y" and pd.isna(by_well["A3"])
    # Original columns preserved.
    assert df["label"].tolist() == [1, 2, 3]
    assert df["shape_area"].tolist() == [0.0, 1.0, 2.0]

    # Re-write updates in place without duplicating columns.
    meta2 = pd.DataFrame({"well": ["A1", "A2"], "treat": ["u", "v"], "dose": [9, 8]})
    db.write_metadata(meta2)
    df2 = db.load_table("cell")
    assert list(df2.columns) == list(df.columns)
    assert dict(zip(df2["well"], df2["treat"]))["A1"] == "u"
    # A table without `well` is untouched.
    assert list(db.load_table("no_well").columns) == ["value"]
    db.close()


# ── infer DB ──────────────────────────────────────────────────────────────


def _make_infer_db(tmp_path):
    path = tmp_path / "infer.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        'CREATE TABLE "inference" (uid INTEGER PRIMARY KEY, directory TEXT, '
        "well TEXT, pred_class TEXT, features BLOB)"
    )
    for i, well in enumerate(["B1", "B2"]):
        conn.execute('INSERT INTO "inference" VALUES (?,?,?,?,?)',
                     (i + 1, "D:/ds", well, "x", b"\x00\x01"))
    conn.commit()
    conn.close()
    return path


def test_infer_write_metadata_preserves_uid_and_features(tmp_path):
    path = _make_infer_db(tmp_path)
    db = InferDB(path)
    meta = pd.DataFrame({"well": ["B1", "B2"], "treat": ["a", "b"]})
    assert db.write_metadata(meta) == 1
    db.close()

    conn = sqlite3.connect(str(path))
    rows = conn.execute(
        'SELECT uid, features, "treat" FROM "inference" ORDER BY uid').fetchall()
    conn.close()
    assert [r[0] for r in rows] == [1, 2]
    assert [r[1] for r in rows] == [b"\x00\x01", b"\x00\x01"]
    assert [r[2] for r in rows] == ["a", "b"]


def test_infer_write_metadata_without_well_is_noop(tmp_path):
    path = _make_infer_db(tmp_path)
    db = InferDB(path)
    assert db.write_metadata(pd.DataFrame({"other": ["a"]})) == 0
    db.close()
