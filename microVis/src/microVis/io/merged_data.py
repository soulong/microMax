"""The Data page's integrated object table (thin wrapper over microBase).

All mask-aware merge mechanics live in :mod:`microBase.db_merge` (shared
with microProfiler's inference auto-merge): profiler object tables and
infer results are fused ONLY within the same mask (different masks stack,
tagged in the `mask` column); infer frames without mask info join the
profiler's mask with a log hint. This module adapts that engine to the
GUI: an in-memory `MergedData` handle, Excel-metadata merging and the
dataset directory scoping used by the plot view.
"""

from __future__ import annotations

import logging

import pandas as pd

from microBase import db_merge
from microBase.db_merge import (  # re-exported for MainWindow
    MERGED_TABLE,
    merge_dbs,
    write_merged_db,
)

logger = logging.getLogger("microVis.merged_data")


class MergedData:
    """The integrated per-object table built from selected DB files."""

    def __init__(self, table: pd.DataFrame, paths: list[str]):
        self.table = table
        self.paths = paths

    # ── Loading ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, paths) -> "MergedData":
        """Open every DB read-only and fuse their frames (per mask)."""
        paths = [str(p) for p in paths]
        table = merge_dbs(paths)
        return cls(table, paths)

    # ── Access ────────────────────────────────────────────────────────────

    def display_columns(self) -> list[tuple[str, bool]]:
        """(name, is_numeric) for every merged column, in order."""
        return [(c, bool(pd.api.types.is_numeric_dtype(self.table[c])))
                for c in self.table.columns]

    def frame_for_dataset(self, scopes: list[str]) -> pd.DataFrame:
        """Rows of the CURRENT dataset (directory-scoped, prefix matching).

        A scope keeps rows whose stored directory equals it or starts with
        it. When nothing matches (DB written elsewhere) ALL rows are
        returned so plots stay usable.
        """
        df = self.table
        if df.empty or "directory" not in df.columns:
            return df
        dirs = df["directory"].astype(str)
        mask = None
        for scope in scopes:
            root = str(scope).replace("\\", "/").rstrip("/")
            if root in ("", "."):
                current = (dirs == ".") | dirs.str.startswith("./")
            else:
                current = (dirs == root) | dirs.str.startswith(root + "/")
            mask = current if mask is None else (mask | current)
        if mask is not None and mask.any():
            return df[mask]
        return df


def merge_metadata_into(merged: MergedData | None,
                        metadata: pd.DataFrame | None) -> pd.DataFrame | None:
    """Display frame for the plot view: the merged table + Excel metadata.

    Left-joins only the metadata columns missing from the table, so
    re-merging after a write never duplicates columns (same contract as
    data_module.merge_metadata).
    """
    if merged is None:
        return None
    df = merged.table
    if metadata is None or "well" not in df.columns:
        return df
    missing = [c for c in metadata.columns
               if c != "well" and c not in df.columns]
    if not missing:
        return df
    return df.merge(metadata[["well"] + missing], on="well", how="left")
