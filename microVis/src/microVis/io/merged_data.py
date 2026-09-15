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

from microBase import db_merge, normalize_well
from microBase.db_merge import (  # re-exported for MainWindow
    MERGED_TABLE,
    write_merged_db,
)

logger = logging.getLogger("microVis.merged_data")


class MergedData:
    """The integrated per-object table built from selected DB files."""

    def __init__(self, table: pd.DataFrame, paths: list, masks=None):
        self.table = table
        self.paths = paths
        # Distinct mask tags across the fused sources (empty when unknown —
        # e.g. infer-only rows with NULL mask_name). A single-mask selection
        # fuses WITHOUT the tag column, so Write to DB needs this to pass
        # mask= and keep the written file re-loadable under its real mask.
        self.masks = list(masks) if masks else []

    # ── Loading ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, paths) -> "MergedData":
        """Open every DB read-only and fuse their frames (per mask)."""
        paths = [str(p) for p in paths]
        frames = []
        for p in paths:
            frames.extend(db_merge.read_db_frames(p))
        table = db_merge.fuse_frames(frames)
        masks = sorted({f.mask for f in frames if f.mask})
        return cls(table, paths, masks)

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
    data_module.merge_metadata). The well join runs on normalized keys
    ('A01' -> 'A1') so captured verbatim wells still meet the Excel
    spelling; the original well columns stay verbatim.
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
    if "well" not in metadata.columns:
        return df
    left = df.assign(__well_key__=df["well"].map(normalize_well))
    right = metadata[["well"] + missing].assign(
        __well_key__=metadata["well"].map(normalize_well))
    return left.merge(right.drop(columns=["well"]), on="__well_key__",
                      how="left").drop(columns=["__well_key__"])
