"""Read-only access to a microModel inference database (infer.db).

The DB holds ``inference`` (one row per object) plus optional
``reduction_<method>`` (2-D DR coordinates keyed by uid) and ``find_cluster``
(cluster id/probability columns keyed by uid) tables. This module only reads;
the plotting UI consumes the joined DataFrame.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from natsort import natsorted

from microBase.db_contracts import (
    DIRECTORY_COLUMN,
    FEATURES_COLUMN,
    FIND_CLUSTER_TABLE,
    INFERENCE_TABLE,
    UID_COLUMN,
    DR_METHODS,
    is_numeric_sql_type,
    reduction_coord_prefix,
    reduction_table_name,
    sql_ident as _q,
)
from microVis.io.metadata_write import write_metadata_columns
from microVis.log_utils import get_logger

logger = get_logger("microVis.infer_db")

# Canonical DR order (shared contract).
METHOD_ORDER = DR_METHODS


def is_numeric_type(sql_type: str) -> bool:
    """True when a declared SQLite type names a numeric affinity."""
    return is_numeric_sql_type(sql_type)


class InferDB:
    """Reader for one infer.db file (no writes, no schema changes)."""

    def __init__(self, db_path):
        self._path = Path(db_path)
        if not self._path.exists():
            raise FileNotFoundError(f"infer DB not found: {self._path}")
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._tables = self._introspect()
        if INFERENCE_TABLE not in self._tables:
            self._conn.close()
            self._conn = None
            raise ValueError(
                f"Not an inference DB (no '{INFERENCE_TABLE}' table): {self._path}"
            )
        # Cached full inference + find_cluster frame (load_inference).
        self._inference_cache: pd.DataFrame | None = None

    def _introspect(self) -> dict:
        tables: dict = {}
        for (name,) in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"):
            if name.startswith("sqlite_"):
                continue
            cur = self._conn.execute(f"PRAGMA table_info({_q(name)})")
            tables[name] = {row[1]: row[2] for row in cur.fetchall()}
        return tables

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    def list_reduction_methods(self) -> list[str]:
        """DR methods with a reduction_<method> table, in canonical order."""
        return [m for m in METHOD_ORDER if reduction_table_name(m) in self._tables]

    def list_directories(self) -> list[str]:
        """Distinct non-empty source directories stored in `inference`."""
        rows = self._conn.execute(
            f'SELECT DISTINCT {_q(DIRECTORY_COLUMN)} FROM {_q(INFERENCE_TABLE)}'
        ).fetchall()
        return natsorted(str(r[0]) for r in rows if r[0])

    def list_columns(self) -> list[tuple[str, str]]:
        """Selectable (name, declared_type) pairs for color/size pickers.

        Excludes the features BLOB, the uid join key, and every BLOB column;
        find_cluster columns are appended after the inference ones.
        """
        out: list[tuple[str, str]] = []
        for c, t in self._tables.get(INFERENCE_TABLE, {}).items():
            if c in (UID_COLUMN, FEATURES_COLUMN) or t.upper() == "BLOB":
                continue
            out.append((c, t))
        for c, t in self._tables.get(FIND_CLUSTER_TABLE, {}).items():
            if c == UID_COLUMN or t.upper() == "BLOB":
                continue
            out.append((c, t))
        return out

    def coord_columns(self, method: str) -> tuple[str, str]:
        """Coordinate column names for a method (pc_1/pc_2 for pca)."""
        prefix = reduction_coord_prefix(method)
        return f"{prefix}_1", f"{prefix}_2"

    # ── Data ──────────────────────────────────────────────────────────────

    def load_scatter(self, method: str) -> pd.DataFrame:
        """Join `inference` + reduction_<method> + find_cluster on uid.

        The `features` BLOB and `uid` are excluded from the returned frame's
        selectable columns are kept (uid stays as the join key). Raises
        ValueError when the DB has no table for the requested method.
        """
        if method not in self.list_reduction_methods():
            raise ValueError(
                f"No {reduction_table_name(method)} table in {self._path}. "
                f"Available: {self.list_reduction_methods() or 'none'}")
        c1, c2 = self.coord_columns(method)
        # Non-BLOB inference columns (+ uid for the join) — features stays out.
        cols = [c for c, t in self._tables[INFERENCE_TABLE].items()
                if t.upper() != "BLOB" and c != FEATURES_COLUMN]
        col_sql = ", ".join(f"i.{_q(c)}" for c in cols)
        sql = (
            f"SELECT {col_sql}, r.{_q(c1)}, r.{_q(c2)} "
            f'FROM "{INFERENCE_TABLE}" i '
            f'JOIN "{reduction_table_name(method)}" r ON i.{UID_COLUMN} = r.{UID_COLUMN}'
        )
        df = pd.read_sql(sql, self._conn)
        if FIND_CLUSTER_TABLE in self._tables:
            cluster_cols = [c for c, t in self._tables[FIND_CLUSTER_TABLE].items()
                            if c != UID_COLUMN and t.upper() != "BLOB"]
            if cluster_cols:
                csql = ", ".join(f"c.{_q(c)}" for c in cluster_cols)
                cdf = pd.read_sql(
                    f'SELECT c.{UID_COLUMN}, {csql} FROM "{FIND_CLUSTER_TABLE}" c',
                    self._conn)
                df = df.merge(cdf, on=UID_COLUMN, how="left")
        logger.info("Loaded %d rows from %s (reduction_%s)",
                    len(df), self._path.name, method)
        return df

    def _inference_frame(self) -> pd.DataFrame:
        """Full non-BLOB `inference` + left-joined `find_cluster` frame (cached)."""
        if self._inference_cache is not None:
            return self._inference_cache
        cols = [c for c, t in self._tables[INFERENCE_TABLE].items()
                if c != FEATURES_COLUMN and t.upper() != "BLOB"]
        col_sql = ", ".join(_q(c) for c in cols)
        df = pd.read_sql(
            f'SELECT {col_sql} FROM "{INFERENCE_TABLE}"', self._conn)
        if FIND_CLUSTER_TABLE in self._tables:
            cluster_cols = [c for c, t in self._tables[FIND_CLUSTER_TABLE].items()
                            if c != UID_COLUMN and t.upper() != "BLOB"]
            if cluster_cols:
                csql = ", ".join(_q(c) for c in cluster_cols)
                cdf = pd.read_sql(
                    f'SELECT {UID_COLUMN}, {csql} FROM "{FIND_CLUSTER_TABLE}"',
                    self._conn)
                df = df.merge(cdf, on=UID_COLUMN, how="left")
        self._inference_cache = df
        return df

    def load_inference(
        self, directories: str | list[str] | None = None
    ) -> pd.DataFrame:
        """Inference rows with cluster columns, optionally directory-scoped.

        Each scope keeps rows whose stored `directory` equals it or starts
        with it (path prefix). Scopes may be the dataset's metadata directory
        values (absolute forward-slash form) or the absolute dataset dir.
        When no row matches any scope (DB written elsewhere) all rows are
        returned so the overlay still has data.
        """
        df = self._inference_frame()
        if not directories or DIRECTORY_COLUMN not in df.columns or df.empty:
            return df
        if isinstance(directories, str):
            directories = [directories]
        dirs = df[DIRECTORY_COLUMN].astype(str)
        mask = None
        for scope in directories:
            root = str(scope).replace("\\", "/").rstrip("/")
            if root in ("", "."):
                current = (dirs == ".") | dirs.str.startswith("./")
            else:
                current = (dirs == root) | dirs.str.startswith(root + "/")
            mask = current if mask is None else (mask | current)
        if mask is not None and mask.any():
            return df[mask]
        return df

    # ── Metadata ──────────────────────────────────────────────────────────

    def write_metadata(self, metadata: pd.DataFrame) -> int:
        """Add/update metadata columns in the `inference` table (by well).

        Additive: uid, features and every existing value are preserved.
        Returns the number of columns written.
        """
        written = write_metadata_columns(self._conn, INFERENCE_TABLE, metadata)
        if written:
            self._tables = self._introspect()
            self._inference_cache = None
        return written

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
