"""Thread-safe SQLite database operations."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Literal, Union

import pandas as pd

from microBase.db_contracts import (
    TABLE_MASKS_TABLE,
    WELL_COLUMN,
    sql_ident,
)

logger = logging.getLogger(__name__)

# profiler.db layout version. Bump when a table/column contract changes;
# readers treat a different version as "not this format" (no migration).
SCHEMA_VERSION = 1
SCHEMA_VERSION_TABLE = "_schema_version"


class Database:
    """Thread-safe SQLite database using WAL mode."""

    def __init__(self, db_path: Union[str, Path]):
        # Initialize _local FIRST so __del__/close() are safe even if the
        # mkdir below raises (e.g. drive disconnected → FileNotFoundError).
        self._local = threading.local()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Database: %s", self.db_path)

    def __del__(self) -> None:
        # __del__ must never raise — guard against partially-initialized
        # objects (e.g. __init__ failed before _local was set).
        try:
            self.close()
        except Exception:
            pass

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def close(self) -> None:
        """Close the current thread's SQLite connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def save_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: Literal["fail", "replace", "append"] = "replace",
    ) -> None:
        """Write a DataFrame to an SQLite table.

        On replace, also records the profiler.db schema version in the
        internal `_schema_version` table and creates a `well` index when the
        table has one (the well grid and metadata joins filter on it).
        """
        logger.debug("save_table: %s (%d rows, %d cols)", table_name, len(df), len(df.columns))
        conn = self._get_conn()
        # Copy so the caller's DataFrame is never mutated in place.
        df = df.copy()
        # Convert any Path values to strings (a column may mix Path/str — the
        # first-row sample alone is not a reliable type probe).
        for col in df.columns:
            if df[col].dtype == "object" and len(df) > 0 and any(isinstance(v, Path) for v in df[col]):
                df[col] = df[col].map(lambda v: str(v) if isinstance(v, Path) else v)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        if if_exists == "replace":
            self._write_schema_version(conn)
            if WELL_COLUMN in df.columns:
                table = sql_ident(table_name)
                index = sql_ident(f"idx_{table_name}_well")
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {index} "
                    f"ON {table} ({sql_ident(WELL_COLUMN)})"
                )
        conn.commit()

    def _write_schema_version(self, conn: sqlite3.Connection) -> None:
        """Record the current profiler.db layout version (idempotent)."""
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {SCHEMA_VERSION_TABLE} "
            f"(version INTEGER NOT NULL)"
        )
        conn.execute(f"DELETE FROM {SCHEMA_VERSION_TABLE}")
        conn.execute(
            f"INSERT INTO {SCHEMA_VERSION_TABLE} (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )

    def record_table_mask(self, table_name: str, mask_name: str) -> None:
        """Bookkeep which mask (object type) an object table belongs to.

        The per-mask merges (microProfiler auto-merge, microVis Select DB)
        group frames by mask — without this mapping a custom
        output_table_name would be indistinguishable from its mask.
        """
        conn = self._get_conn()
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE_MASKS_TABLE} "
            "(table_name TEXT PRIMARY KEY, mask_name TEXT)"
        )
        conn.execute(
            f"INSERT OR REPLACE INTO {TABLE_MASKS_TABLE} "
            "(table_name, mask_name) VALUES (?, ?)",
            (table_name, mask_name),
        )
        conn.commit()
        logger.debug("record_table_mask: %s -> %s", table_name, mask_name)

    def list_tables(self) -> "set[str]":
        """Return the set of table names in the database."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {row[0] for row in rows}

    def drop_table(self, name: str) -> None:
        """Drop a table if it exists (identifier-quoted name for safety)."""
        conn = self._get_conn()
        conn.execute(f"DROP TABLE IF EXISTS {sql_ident(name)}")
        conn.commit()
        logger.info("Dropped table '%s'", name)

    def row_count(self, name: str) -> int:
        """Return the number of rows in a table."""
        conn = self._get_conn()
        cur = conn.execute(f"SELECT COUNT(*) FROM {sql_ident(name)}")
        return int(cur.fetchone()[0])
