"""Tests for microVis.io.profiler_db: read-only profiler.db access."""

import sqlite3

import pytest

from microVis.io.profiler_db import ProfilerDB


def _make_result_db(path):
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE "cell" (
            directory TEXT, well TEXT, label INTEGER,
            shape_area REAL, intensity_mean_ch1 REAL, features BLOB
        );
        CREATE TABLE "image" (
            directory TEXT, well TEXT, intensity_mean_ch1 REAL
        );
        CREATE TABLE "blob_only" (directory TEXT, features BLOB);
        """
    )
    for i in range(5):
        conn.execute('INSERT INTO "cell" VALUES (?,?,?,?,?,NULL)',
                     ("D:/ds", f"A{i + 1}", i + 1, float(i) * 2, float(i) / 2))
        conn.execute('INSERT INTO "image" VALUES (?,?,?)',
                     ("D:/ds", f"A{i + 1}", float(i)))
        conn.execute('INSERT INTO "blob_only" VALUES (?,NULL)', ("D:/ds",))
    conn.commit()
    conn.close()
    return path


def test_list_tables_and_columns(tmp_path):
    db = ProfilerDB(_make_result_db(tmp_path / "profiler.db"))
    # blob_only has no selectable column -> excluded; image is included.
    assert set(db.list_tables()) == {"cell", "image"}
    cols = dict((c, (t, n)) for c, t, n in db.table_columns("cell"))
    assert "directory" not in cols and "features" not in cols
    assert cols["well"] == ("TEXT", False)
    assert cols["label"] == ("INTEGER", True)
    assert cols["shape_area"] == ("REAL", True)
    db.close()


def test_load_table_drops_blob_and_directory(tmp_path):
    db = ProfilerDB(_make_result_db(tmp_path / "profiler.db"))
    df = db.load_table("cell")
    assert len(df) == 5
    assert "features" not in df.columns and "directory" not in df.columns
    assert list(df["well"]) == ["A1", "A2", "A3", "A4", "A5"]
    db.close()


def test_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ProfilerDB(tmp_path / "nope.db")
