"""Build the integrated object table from any mix of profiler.db / infer.db.

Merge model
-----------
- profiler.db: every OBJECT table (has `well` + `label` columns) contributes
  its rows; the image-level `image` table and bookkeeping tables are skipped.
- infer.db: the `inference` table (minus the features BLOB) left-joined with
  every `reduction_<method>` table on `uid` (the coordinate columns are
  already method-prefixed: pc_1/pc_2, umap_1/2, ...).
- Per-table frames are folded with an OUTER merge on the intersection of the
  identity columns {well, field, stack, timepoint, directory, filename,
  ground_truth, label} present in BOTH frames. Whole-image infer rows and
  profiler objects therefore fuse into ONE row per physical object, so
  profiler measurements and infer predictions can be cross-plotted.
- Frames without a usable identity intersection (e.g. single-cell infer rows,
  which carry no `label`) are STACKED instead of joined — they stay plot-able
  but cannot be fused to mask objects.
- On column-name collisions between different sources, the incoming
  non-identity column is renamed `<db-stem>/<column>` (then
  `<db-stem>/<table>/<column>`, then a numeric suffix) so values from
  different DBs stay distinguishable.

`write_merged_db` persists the integrated table into a NEW SQLite database
(default merge.db, table `merged`) — the source DBs are never touched.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from microBase.db_contracts import (
    FEATURES_COLUMN,
    INFERENCE_TABLE,
    REDUCTION_TABLE_PREFIX,
    UID_COLUMN,
)

logger = logging.getLogger("microVis.merged_data")

# Identity columns: present in BOTH frames -> join keys for the outer merge.
KEY_COLUMNS = (
    "well", "field", "stack", "timepoint",
    "directory", "filename", "ground_truth", "label",
)

# The merged output table name inside merge.db.
MERGED_TABLE = "merged"


class MergedData:
    """The integrated per-object table built from selected DB files."""

    def __init__(self, table: pd.DataFrame, paths: list[str]):
        self.table = table
        self.paths = paths

    # ── Loading ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, paths) -> "MergedData":
        """Open every DB read-only, build per-table frames and fold them."""
        paths = [str(p) for p in paths]
        merged: pd.DataFrame | None = None
        for path in paths:
            stem = Path(path).stem
            for df, tname in cls._load_db_frames(path):
                merged = cls._fold(merged, df, tname, stem)
        if merged is None:
            raise ValueError("No selectable object data in the selected DB(s)")
        logger.info("Merged table: %d rows, %d columns from %d DB(s)",
                    len(merged), len(merged.columns), len(set(paths)))
        return cls(merged, paths)

    @staticmethod
    def _read_table(conn: sqlite3.Connection, table: str,
                    schema: dict[str, str]) -> pd.DataFrame:
        cols = [c for c, t in schema.items() if t.upper() != "BLOB"]
        col_sql = ", ".join(f'"{c}"' for c in cols)
        return pd.read_sql(f'SELECT {col_sql} FROM "{table}"', conn)

    @classmethod
    def _load_db_frames(cls, path: str) -> list[tuple[pd.DataFrame, str]]:
        """Per-table frames of one DB file (connections closed on return)."""
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables: dict[str, dict[str, str]] = {}
            for (name,) in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"):
                if name.startswith("sqlite_") or name.startswith("_"):
                    continue
                cur = conn.execute(f'PRAGMA table_info("{name}")')
                tables[name] = {r[1]: r[2] for r in cur.fetchall()}

            frames: list[tuple[pd.DataFrame, str]] = []
            if INFERENCE_TABLE in tables:
                frames.extend(cls._load_infer(conn, tables))
            else:
                frames.extend(cls._load_profiler(conn, tables))
            return frames
        finally:
            conn.close()

    @classmethod
    def _load_infer(cls, conn, tables) -> list[tuple[pd.DataFrame, str]]:
        schema = tables[INFERENCE_TABLE]
        df = cls._read_table(conn, INFERENCE_TABLE, schema)
        if df.empty:
            return []
        # Attach every reduction table's coordinates (uid is infer-internal
        # and dropped afterwards — meaningless across DBs).
        for name, tschema in tables.items():
            if not name.startswith(REDUCTION_TABLE_PREFIX):
                continue
            coords = cls._read_table(conn, name, tschema)
            if UID_COLUMN in coords.columns and not coords.empty:
                df = df.merge(coords, on=UID_COLUMN, how="left")
        if UID_COLUMN in df.columns:
            df = df.drop(columns=[UID_COLUMN])
        if FEATURES_COLUMN in df.columns:
            df = df.drop(columns=[FEATURES_COLUMN])
        return [(df, INFERENCE_TABLE)]

    @classmethod
    def _load_profiler(cls, conn, tables) -> list[tuple[pd.DataFrame, str]]:
        out = []
        for name, schema in tables.items():
            # Object tables carry the identity pair well+label; everything
            # else (image-level table, label-annotation tables without
            # measurements) is not object data.
            if not {"well", "label"} <= set(schema):
                continue
            df = cls._read_table(conn, name, schema)
            if not df.empty:
                out.append((df, name))
        return out

    # ── Folding ───────────────────────────────────────────────────────────

    @staticmethod
    def _unique_name(existing: set[str], stem: str, table: str, col: str) -> str:
        for candidate in (f"{stem}/{col}", f"{stem}/{table}/{col}"):
            if candidate not in existing:
                return candidate
        n = 2
        while f"{stem}/{table}/{col} ({n})" in existing:
            n += 1
        return f"{stem}/{table}/{col} ({n})"

    @classmethod
    def _fold(cls, acc: pd.DataFrame | None, df: pd.DataFrame,
              table: str, stem: str) -> pd.DataFrame:
        """Merge one source frame into the accumulated table.

        acc is not mutated; the incoming frame is copied when renames are
        needed. On identity-column collision with different content, the
        incoming column is renamed `<stem>/<col>` (then table-qualified).
        """
        if acc is None:
            return df

        incoming = df.copy()
        existing = set(acc.columns)
        key_set = set(KEY_COLUMNS)
        renamed = False
        for col in list(incoming.columns):
            if col in key_set or col not in existing:
                continue
            if incoming[col].equals(acc[col]):
                continue
            new_name = cls._unique_name(existing, stem, table, col)
            incoming = incoming.rename(columns={col: new_name})
            existing.add(new_name)
            renamed = True

        common = [c for c in KEY_COLUMNS
                  if c in acc.columns and c in incoming.columns]
        if "label" in common:
            return acc.merge(incoming, on=common, how="outer")
        # No object identity in common (e.g. single-cell infer rows):
        # stack instead of joining — a key-less join would fan out.
        return pd.concat([acc, incoming], axis=0, ignore_index=True)

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


def write_merged_db(df: pd.DataFrame, path) -> str:
    """Write the integrated table into a NEW SQLite DB (table `merged`).

    The parent directory is created if needed; an existing file is replaced
    table-wise (other tables in the file are untouched).
    """
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(".db")
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    try:
        df.to_sql(MERGED_TABLE, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()
    logger.info("Wrote merged table (%d rows) to %s", len(df), p)
    return str(p)


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
