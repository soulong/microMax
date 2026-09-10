"""Add/update metadata columns in an existing SQLite table keyed by well.

Used by the microVis "Write to DB" action: metadata columns that are missing
from a table are added with ``ALTER TABLE`` and every row's value is set with
``UPDATE ... WHERE well = ?``. This preserves primary keys, BLOB columns
(e.g. ``features``) and every existing value — no table rewrite.
"""

from __future__ import annotations

import logging

import pandas as pd

from microBase.db_contracts import sql_ident as _q

logger = logging.getLogger(__name__)


def _sql_type(series: pd.Series) -> str:
    """SQLite column affinity for a metadata column."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def _bind(value):
    """Convert a pandas/numpy scalar into a sqlite3-bindable value."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def table_columns(conn, table: str) -> list[str]:
    """Column names of a table (empty when the table does not exist)."""
    cur = conn.execute(f"PRAGMA table_info({_q(table)})")
    return [row[1] for row in cur.fetchall()]


def write_metadata_columns(conn, table: str, metadata: pd.DataFrame | None,
                           key: str = "well") -> int:
    """Add/update metadata columns in *table*, matched by *key*.

    Returns the number of metadata columns written. Skips silently when the
    table has no key column or the metadata has no other columns.
    """
    if metadata is None or metadata.empty or key not in metadata.columns:
        return 0
    columns = table_columns(conn, table)
    if key not in columns:
        return 0
    meta_cols = [c for c in metadata.columns if c != key]
    if not meta_cols:
        return 0
    keys = [_bind(v) for v in metadata[key].tolist()]
    written = 0
    for col in meta_cols:
        if col not in columns:
            conn.execute(
                f"ALTER TABLE {_q(table)} ADD COLUMN {_q(col)} {_sql_type(metadata[col])}")
            columns.append(col)
        pairs = [(val, k)
                 for k, val in zip(keys, (_bind(v) for v in metadata[col].tolist()))]
        conn.executemany(
            f"UPDATE {_q(table)} SET {_q(col)} = ? WHERE {_q(key)} = ?", pairs)
        written += 1
    conn.commit()
    logger.info("Metadata columns written to %s: %d", table, written)
    return written
