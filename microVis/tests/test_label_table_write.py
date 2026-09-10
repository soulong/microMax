"""write_label_table must never silently replace a non-label profiler table."""

import sqlite3

import pandas as pd
import pytest

from microVis.io.data_module import DataModule


def _dm_for(db_path):
    """DataModule with only the DB attributes write_label_table touches."""
    dm = DataModule.__new__(DataModule)
    dm._db_path = db_path
    dm._df_cache = {}
    dm._db_tables = {}
    return dm


def _label_df():
    return pd.DataFrame({
        "well": ["A1"], "field": ["1"], "stack": ["1"], "timepoint": ["1"],
        "label": [1], "class": ["x"],
    })


def test_write_label_table_refuses_to_replace_profiler_table(tmp_path):
    db = tmp_path / "profiler.db"
    conn = sqlite3.connect(str(db))
    conn.execute('CREATE TABLE "cell" (well TEXT, label INTEGER, shape_area REAL)')
    conn.execute('INSERT INTO "cell" VALUES ("A1", 1, 3.0)')
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="not a label table"):
        _dm_for(db).write_label_table("cell", _label_df())

    conn = sqlite3.connect(str(db))
    assert conn.execute('SELECT COUNT(*) FROM "cell"').fetchone()[0] == 1
    conn.close()


def test_write_label_table_replaces_existing_label_table(tmp_path):
    db = tmp_path / "profiler.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        'CREATE TABLE "cell_label" (well TEXT, field TEXT, stack TEXT, '
        "timepoint TEXT, label INTEGER, class TEXT)"
    )
    conn.execute(
        "INSERT INTO \"cell_label\" VALUES ('A1', '1', '1', '1', 9, 'old')"
    )
    conn.commit()
    conn.close()

    dm = _dm_for(db)
    dm.write_label_table("cell_label", _label_df())

    conn = sqlite3.connect(str(db))
    rows = conn.execute('SELECT well, label, class FROM "cell_label"').fetchall()
    conn.close()
    assert rows == [("A1", 1, "x")]
