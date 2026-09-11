"""Mask-aware merge of profiler.db object tables and infer.db results.

Shared by microProfiler (auto-merge after the inference step ->
``merge_<mask>.db``) and microVis (Select DB -> one integrated table).

Merge model
-----------
- Every source frame belongs to ONE mask (the segmentation object type):

  * infer.db: the `inference` table carries a `mask_name` column (written by
    microModel). Missing column / NULL (single-cell inference over
    pre-cropped cells has no mask file) -> the frame's mask is unknown and
    the caller's default applies.
  * profiler.db: object tables are mapped through the `_table_masks`
    bookkeeping table (written by microProfiler). A table that carries a
    `mask` column (a previous merge output) uses that. Otherwise the table
    name itself is taken as the mask name.
- Frames are fused ONLY with frames of the same mask: an outer merge on the
  intersection of the identity columns {well, field, stack, timepoint,
  directory, filename, ground_truth, label} present in both frames, so
  profiler measurements and infer predictions meet in ONE row per physical
  object. Frames without a usable identity intersection (no shared `label`,
  e.g. whole-image infer rows) are STACKED instead — a key-less join would
  fan out.
- Frames of different masks are NEVER fused (cell and nuclei objects are
  different physical objects even when well+label collide); they are
  stacked, and the fused table carries a `mask` column telling each row's
  group apart.
- On column-name collisions between different sources inside one mask, the
  incoming non-identity column is renamed `<db-stem>/<column>` (then
  `<db-stem>/<table>/<column>`, then a numeric suffix).

`write_merged_db` persists a fused table into a NEW SQLite database (table
`merged` plus the `_table_masks` bookkeeping row, so the output re-loads
with its mask intact). Source DBs are never touched.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from microBase.db_contracts import (
    FEATURES_COLUMN,
    INFERENCE_TABLE,
    LABEL_COLUMN,
    MASK_COLUMN,
    MASK_NAME_COLUMN,
    REDUCTION_TABLE_PREFIX,
    TABLE_MASKS_TABLE,
    UID_COLUMN,
    WELL_COLUMN,
)

logger = logging.getLogger("microBase.db_merge")

# Identity columns: present in BOTH frames -> join keys for the outer merge.
KEY_COLUMNS = (
    "well", "field", "stack", "timepoint",
    "directory", "filename", "ground_truth", "label",
)

# The fused output table name inside a written merge DB.
MERGED_TABLE = "merged"


@dataclass
class SourceFrame:
    """One table read from a DB, tagged with its origin and mask."""

    df: pd.DataFrame
    table: str                      # source table name
    stem: str                       # source DB file stem (collision prefix)
    mask: str | None = None         # owning mask name (None = unknown)
    is_infer: bool = False


# ── Reading ──────────────────────────────────────────────────────────────

def _read_table(conn: sqlite3.Connection, table: str,
                blob_free: bool = True) -> pd.DataFrame:
    """Read one table; with blob_free, BLOB columns (features) are skipped."""
    schema = [(r[1], r[2]) for r in conn.execute(f'PRAGMA table_info("{table}")')]
    cols = [c for c, t in schema if not (blob_free and t.upper() == "BLOB")]
    col_sql = ", ".join(f'"{c}"' for c in cols)
    return pd.read_sql(f'SELECT {col_sql} FROM "{table}"', conn)


def _table_columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    """name -> declared type of every column of one table."""
    return {r[1]: r[2] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def read_table_masks(db_path) -> dict[str, str]:
    """The `_table_masks` bookkeeping of a DB ({} when absent)."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return {}
    try:
        rows = conn.execute(
            f"SELECT table_name, mask_name FROM {TABLE_MASKS_TABLE}"
        ).fetchall()
        return {t: m for t, m in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def read_profiler_frames(path: str) -> list[SourceFrame]:
    """Object tables of a profiler.db (or a written merge DB).

    An object table carries the identity pair well+label; bookkeeping and
    image-level tables are skipped. Mask resolution per table: `mask`
    column (previous merge output) -> `_table_masks` bookkeeping -> the
    table name itself.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        stem = Path(path).stem
        mapping = {}
        if TABLE_MASKS_TABLE in _table_names(conn):
            rows = conn.execute(
                f"SELECT table_name, mask_name FROM {TABLE_MASKS_TABLE}"
            ).fetchall()
            mapping = {t: m for t, m in rows}
        frames = []
        for name in _table_names(conn):
            schema = _table_columns(conn, name)
            if not {WELL_COLUMN, LABEL_COLUMN} <= set(schema):
                continue
            df = _read_table(conn, name)
            if df.empty:
                continue
            # A previous merge output tags rows with their mask — split it
            # into one frame per distinct value so re-loading regroups
            # exactly (a single-value column is one frame).
            if MASK_COLUMN in df.columns:
                for mval, sub in df.groupby(df[MASK_COLUMN].astype(str)):
                    frames.append(SourceFrame(
                        df=sub.drop(columns=[MASK_COLUMN]).reset_index(drop=True),
                        table=name, stem=stem, mask=mval))
                continue
            frames.append(SourceFrame(
                df=df, table=name, stem=stem,
                mask=mapping.get(name, name)))
        return frames
    finally:
        conn.close()


def read_infer_frames(path: str) -> list[SourceFrame]:
    """The inference table of an infer.db joined with its reduction tables.

    The features BLOB and the infer-internal uid are dropped. The frame's
    mask comes from the `mask_name` column; a missing column or all-NULL
    (single-cell inference) leaves it unknown -> caller's default.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if INFERENCE_TABLE not in _table_names(conn):
            return []
        stem = Path(path).stem
        df = _read_table(conn, INFERENCE_TABLE)
        if df.empty:
            return []
        for name in _table_names(conn):
            if not name.startswith(REDUCTION_TABLE_PREFIX):
                continue
            schema = _table_columns(conn, name)
            if UID_COLUMN not in schema:
                continue
            coords = _read_table(conn, name)
            if not coords.empty:
                df = df.merge(coords, on=UID_COLUMN, how="left")
        for col in (UID_COLUMN, FEATURES_COLUMN):
            if col in df.columns:
                df = df.drop(columns=[col])
        mask = None
        if MASK_NAME_COLUMN in df.columns:
            vals = df[MASK_NAME_COLUMN].dropna().astype(str)
            mask = vals.iloc[0] if len(vals) else None
            df = df.drop(columns=[MASK_NAME_COLUMN])
        return [SourceFrame(df=df, table=INFERENCE_TABLE, stem=stem,
                            mask=mask, is_infer=True)]
    finally:
        conn.close()


def read_db_frames(path: str) -> list[SourceFrame]:
    """All source frames of one DB file, typed by its content."""
    if INFERENCE_TABLE in _table_names_of(path):
        return read_infer_frames(path)
    return read_profiler_frames(path)


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows
            if not r[0].startswith("sqlite_")]


def _table_names_of(path: str) -> list[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return _table_names(conn)
    finally:
        conn.close()


# ── Fusing ───────────────────────────────────────────────────────────────

def _unique_name(existing: set[str], stem: str, table: str, col: str) -> str:
    for candidate in (f"{stem}/{col}", f"{stem}/{table}/{col}"):
        if candidate not in existing:
            return candidate
    n = 2
    while f"{stem}/{table}/{col} ({n})" in existing:
        n += 1
    return f"{stem}/{table}/{col} ({n})"


def _fold(acc: pd.DataFrame | None, frame: SourceFrame) -> pd.DataFrame:
    """Merge one same-mask source frame into the accumulated table.

    acc is not mutated; the incoming frame is copied when renames are
    needed. On identity-column collision with different content, the
    incoming column is renamed `<stem>/<col>` (then table-qualified).
    """
    df = frame.df
    if acc is None:
        return df

    incoming = df.copy()
    existing = set(acc.columns)
    key_set = set(KEY_COLUMNS)
    for col in list(incoming.columns):
        if col in key_set or col not in existing:
            continue
        if incoming[col].equals(acc[col]):
            continue
        new_name = _unique_name(existing, frame.stem, frame.table, col)
        incoming = incoming.rename(columns={col: new_name})
        existing.add(new_name)

    common = [c for c in KEY_COLUMNS
              if c in acc.columns and c in incoming.columns]
    if LABEL_COLUMN in common:
        return acc.merge(incoming, on=common, how="outer")
    # No object identity in common (e.g. whole-image infer rows without a
    # label in both frames): stack instead of joining — a key-less join
    # would fan out.
    return pd.concat([acc, incoming], axis=0, ignore_index=True)


def fuse_frames(frames: list[SourceFrame],
                default_mask: str | None = None) -> pd.DataFrame:
    """Fuse source frames into ONE table, grouped by mask.

    Frames of the same mask are fused (outer merge on identity columns);
    different masks are stacked and tagged in the `mask` column. A frame
    with an unknown mask (single-cell infer rows) joins `default_mask` —
    normally the profiler's mask — with a log hint; with no default it
    forms its own group.
    """
    if not frames:
        raise ValueError("No selectable object data in the selected DB(s)")

    # Unknown-mask frames fall into the default group (hint in the log).
    profiler_masks = [f.mask for f in frames if not f.is_infer and f.mask]
    fallback = default_mask or (profiler_masks[0] if profiler_masks else None)
    groups: dict[str, list[SourceFrame]] = {}
    for f in frames:
        mask = f.mask
        if mask is None:
            if fallback is not None:
                logger.info(
                    "%s/%s has no mask info — assuming mask '%s' (same as "
                    "the profiler objects)", f.stem, f.table, fallback)
            mask = fallback
        groups.setdefault(mask, []).append(f)

    blocks = []
    for mask, group in groups.items():
        acc: pd.DataFrame | None = None
        for f in group:
            acc = _fold(acc, f)
        blocks.append((mask, acc))

    if len(blocks) == 1:
        return blocks[0][1]
    # Different masks: stack (column-outer aligned, other rows NaN) and tag
    # each row's group so the artifacts stay self-describing.
    logger.info("Stacking %d mask groups: %s",
                len(blocks), ", ".join(str(m) for m in groups))
    tagged = []
    for mask, block in blocks:
        block = block.copy()
        block[MASK_COLUMN] = mask
        tagged.append(block)
    return pd.concat(tagged, axis=0, ignore_index=True)


def merge_dbs(paths, default_mask: str | None = None) -> pd.DataFrame:
    """Read every DB and return the fused per-object table (one-call API)."""
    frames: list[SourceFrame] = []
    for path in paths:
        frames.extend(read_db_frames(str(path)))
    fused = fuse_frames(frames, default_mask=default_mask)
    logger.info("Merged table: %d rows, %d columns from %d DB(s)",
                len(fused), len(fused.columns), len(set(str(p) for p in paths)))
    return fused


# ── Writing ──────────────────────────────────────────────────────────────

def write_merged_db(df: pd.DataFrame, path, mask: str | None = None) -> str:
    """Write a fused table into a NEW SQLite DB (table `merged`).

    The parent directory is created if needed; the `merged` table is
    replaced, other tables in the file are untouched. `_table_masks`
    records which mask the output belongs to so re-loading the file keeps
    the grouping (mask param wins over the frame's `mask` column; a
    multi-mask table relies on its `mask` column on re-load instead).
    """
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(".db")
    p.parent.mkdir(parents=True, exist_ok=True)
    if mask is None and MASK_COLUMN in df.columns:
        uniq = df[MASK_COLUMN].dropna().astype(str).unique()
        mask = uniq[0] if len(uniq) == 1 else None
    conn = sqlite3.connect(str(p))
    try:
        df.to_sql(MERGED_TABLE, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE_MASKS_TABLE} "
            "(table_name TEXT PRIMARY KEY, mask_name TEXT)")
        conn.execute(
            f"INSERT OR REPLACE INTO {TABLE_MASKS_TABLE} "
            "(table_name, mask_name) VALUES (?, ?)",
            (MERGED_TABLE, mask))
        conn.commit()
    finally:
        conn.close()
    logger.info("Wrote merged table (%d rows, mask=%s) to %s",
                len(df), mask, p)
    return str(p)
