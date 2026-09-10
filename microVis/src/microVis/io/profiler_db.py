"""Read-only access to a microProfiler profiler.db.

Each plot tab owns one ``ProfilerDB`` so multiple databases can be open at
once (the DataModule still tracks a single "active" DB for the well-grid
overlay). The reader only introspects tables/columns and loads DataFrames;
all writes stay in DataModule.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from microBase.db_contracts import (
    DIRECTORY_COLUMN,
    is_numeric_sql_type,
    sql_ident as _q,
)
from microVis.io.metadata_write import write_metadata_columns
from microVis.log_utils import get_logger

logger = get_logger("microVis.profiler_db")

# Columns never offered as plot variables / never selected.
_EXCLUDED_COLUMNS = {DIRECTORY_COLUMN}


class ProfilerDB:
    """Reader for one profiler.db file (no writes, no schema changes)."""

    def __init__(self, db_path):
        self._path = Path(db_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Profiler DB not found: {self._path}")
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._tables = self._introspect()

    def _introspect(self) -> dict[str, dict[str, str]]:
        tables: dict[str, dict[str, str]] = {}
        for (name,) in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"):
            # sqlite_* are internals; _* are reserved bookkeeping tables
            # (e.g. microProfiler's _schema_version) — never plot variables.
            if name.startswith("sqlite_") or name.startswith("_"):
                continue
            cur = self._conn.execute(f"PRAGMA table_info({_q(name)})")
            tables[name] = {row[1]: row[2] for row in cur.fetchall()}
        return tables

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    def _selectable(self, table: str) -> list[str]:
        """Non-BLOB columns except the excluded bookkeeping ones."""
        return [c for c, t in self._tables.get(table, {}).items()
                if c not in _EXCLUDED_COLUMNS and t.upper() != "BLOB"]

    def list_tables(self) -> list[str]:
        """Tables with at least one selectable column (order as in the DB)."""
        return [t for t in self._tables if self._selectable(t)]

    def table_columns(self, table: str) -> list[tuple[str, str, bool]]:
        """(name, declared_type, is_numeric) for the table's plot variables."""
        out = []
        for c in self._selectable(table):
            ctype = self._tables[table][c]
            out.append((c, ctype, is_numeric_sql_type(ctype)))
        return out

    # ── Data ──────────────────────────────────────────────────────────────

    def load_table(self, table: str) -> pd.DataFrame:
        """Load every selectable column of one table into a DataFrame."""
        cols = self._selectable(table)
        if not cols:
            raise ValueError(f"Table '{table}' has no selectable columns")
        col_sql = ", ".join(_q(c) for c in cols)
        df = pd.read_sql(f'SELECT {col_sql} FROM {_q(table)}', self._conn)
        logger.info("Loaded %d rows from %s (table %s)",
                    len(df), self._path.name, table)
        return df

    # ── Metadata ──────────────────────────────────────────────────────────

    def write_metadata(self, metadata: pd.DataFrame) -> int:
        """Add/update metadata columns in every table that has a `well` column.

        Additive: existing columns/rows are untouched apart from the matched
        metadata cell values. Returns the number of columns written.
        """
        written = 0
        for table in self.list_tables():
            written += write_metadata_columns(self._conn, table, metadata)
        if written:
            self._tables = self._introspect()
        return written

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
