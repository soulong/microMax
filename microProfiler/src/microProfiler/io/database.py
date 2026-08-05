"""Thread-safe SQLite database operations."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Literal, Union

import pandas as pd

logger = logging.getLogger(__name__)


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
        """Write a DataFrame to an SQLite table."""
        logger.debug("save_table: %s (%d rows, %d cols)", table_name, len(df), len(df.columns))
        conn = self._get_conn()
        # Convert Path objects to strings in-place (no copy — caller doesn't reuse df)
        for col in df.columns:
            if df[col].dtype == "object" and len(df) > 0:
                sample = df[col].iloc[0]
                if isinstance(sample, Path):
                    df[col] = df[col].astype(str)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
