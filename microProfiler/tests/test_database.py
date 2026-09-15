"""profiler.db writer hardening: lazy creation, clean table layout."""

import sqlite3

import pandas as pd

from microProfiler.io import Database
from microProfiler.profiling.batch_writer import BatchWriter


def test_batch_writer_lazy_does_not_create_empty_db(tmp_path):
    db_path = tmp_path / "profiler.db"
    writer = BatchWriter(db_path, "image")
    writer.close()
    # No batch ever flushed -> no empty profiler.db left behind.
    assert not db_path.exists()


def test_batch_writer_writes_clean_table_layout(tmp_path):
    """The written DB holds only the data tables: no internal bookkeeping
    tables and no secondary indexes."""
    db_path = tmp_path / "profiler.db"
    writer = BatchWriter(db_path, "image")
    writer.add(pd.DataFrame({"well": ["A1", "A2"], "value": [1.0, 2.0]}))
    writer.close()

    conn = sqlite3.connect(str(db_path))
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"image"}
    assert [row[1] for row in conn.execute("PRAGMA index_list('image')")] == []
    assert conn.execute("SELECT COUNT(*) FROM image").fetchone()[0] == 2
    conn.close()


def test_database_quotes_arbitrary_identifiers(tmp_path):
    db = Database(tmp_path / "x.db")
    conn = db._get_conn()
    conn.execute('CREATE TABLE "we]ird" (x INTEGER)')
    conn.commit()
    assert db.row_count("we]ird") == 0
    assert "we]ird" in db.list_tables()
    db.drop_table("we]ird")
    assert "we]ird" not in db.list_tables()
    db.close()
