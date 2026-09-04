"""Batch writer for profiling results — shared DB write pattern."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from microProfiler.io.database import Database

logger = logging.getLogger(__name__)


class BatchWriter:
    """Accumulates DataFrames and flushes to SQLite or memory."""

    def __init__(
        self,
        db_path: Union[str, Path, None] = None,
        table_name: str = "results",
        batch_size: int = 100,
    ) -> None:
        self._db_path = db_path
        self._table_name = table_name
        self._batch_size = batch_size
        self._db: Optional[Database] = Database(db_path) if db_path else None
        self._first_write = True
        self._results: List[pd.DataFrame] = []
        self._batch: List[pd.DataFrame] = []
        # Column schema fixed at first flush: later batches are reindexed to
        # it so a row that lost a per-channel feature group (see
        # object_profiler._run_per_channel_regionprops) can never introduce
        # new columns on append and crash the SQLite write.
        self._columns: Optional[pd.Index] = None

    def add(self, df: pd.DataFrame) -> None:
        """Add a DataFrame to the current batch.  Flushes when full."""
        self._batch.append(df)
        if len(self._batch) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        """Flush the current batch to DB or in-memory list."""
        if not self._batch:
            return
        combined = pd.concat(self._batch, ignore_index=True)
        if self._columns is None:
            self._columns = combined.columns
        else:
            combined = combined.reindex(columns=self._columns)
        if self._db is not None:
            self._db.save_table(
                combined, self._table_name,
                if_exists="replace" if self._first_write else "append",
            )
            self._first_write = False
        else:
            self._results.extend(self._batch)
        self._batch = []

    def close(self) -> Optional[pd.DataFrame]:
        """Flush remaining data, close DB, return results (if in-memory)."""
        self.flush()
        if self._db is not None:
            self._db.close()
            self._db = None
            return None
        if not self._results:
            return None
        return pd.concat(self._results, ignore_index=True)

    def __enter__(self) -> BatchWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
