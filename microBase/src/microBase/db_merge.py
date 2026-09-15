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
  * profiler.db: a table that carries a `mask` column (a previous merge
    output) uses that; otherwise the table name itself is taken as the mask
    name — object tables must be named after their mask.
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
`merged`, tagged with its `mask` column so the output re-loads with its
mask intact). Source DBs are never touched.
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
    UID_COLUMN,
    sql_ident,
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
    """Read one table; with blob_free, BLOB columns (features) are skipped.

    Identifiers go through sql_ident: regex captures and class names become
    column/table names and may contain embedded double quotes.
    """
    schema = [(r[1], r[2])
              for r in conn.execute(f"PRAGMA table_info({sql_ident(table)})")]
    cols = [c for c, t in schema if not (blob_free and t.upper() == "BLOB")]
    col_sql = ", ".join(sql_ident(c) for c in cols)
    return pd.read_sql(f"SELECT {col_sql} FROM {sql_ident(table)}", conn)


def _table_columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    """name -> declared type of every column of one table."""
    return {r[1]: r[2]
            for r in conn.execute(f"PRAGMA table_info({sql_ident(table)})")}


def read_profiler_frames(path: str) -> list[SourceFrame]:
    """Object tables of a profiler.db (or a written merge DB).

    An object table is identified by its `label` column (the per-object
    identity); `well` is NOT required — datasets without a well capture are
    officially supported, and such tables merge on the identity columns they
    do carry (label, directory, ...). Bookkeeping and image-level tables are
    skipped. Mask resolution per table: `mask` column (previous merge
    output) -> the table name itself.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        stem = Path(path).stem
        frames = []
        for name in _table_names(conn):
            schema = _table_columns(conn, name)
            if LABEL_COLUMN not in schema:
                continue
            df = _read_table(conn, name)
            if df.empty:
                continue
            # A previous merge output tags rows with their mask — split it
            # into one frame per distinct value so re-loading regroups
            # exactly (a single-value column is one frame).
            if MASK_COLUMN in df.columns:
                # A NULL tag is NOT the mask "nan" — map it back to None so
                # the frame falls through to the caller's default-mask logic
                # like an infer table with NULL mask_name does. (pandas'
                # string dtype hands group keys back as NaN, hence isna.)
                key = df[MASK_COLUMN].map(
                    lambda v: None if pd.isna(v) else str(v))
                for mval, sub in df.groupby(key, dropna=False, sort=False):
                    frames.append(SourceFrame(
                        df=sub.drop(columns=[MASK_COLUMN]).reset_index(drop=True),
                        table=name, stem=stem,
                        mask=None if pd.isna(mval) else str(mval)))
                continue
            frames.append(SourceFrame(
                df=df, table=name, stem=stem, mask=name))
        return frames
    finally:
        conn.close()


def read_infer_frames(path: str) -> list[SourceFrame]:
    """The inference table of an infer.db joined with its reduction tables.

    The features BLOB and the infer-internal uid are dropped. One table can
    hold several masks (a re-run over another mask appends rows), so the
    frame is split per `mask_name` value — one SourceFrame each, exactly
    like the profiler frames. A missing column or NULL (single-cell
    inference over pre-cropped cells has no mask file) yields a frame with
    an unknown mask -> the caller's default applies.
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
        frames: list[SourceFrame] = []
        if MASK_NAME_COLUMN in df.columns:
            mask_vals = df[MASK_NAME_COLUMN]
            df = df.drop(columns=[MASK_NAME_COLUMN])
            named = mask_vals.dropna().astype(str)
            for mval, sub in df.loc[named.index].groupby(named):
                frames.append(SourceFrame(
                    df=sub.reset_index(drop=True), table=INFERENCE_TABLE,
                    stem=stem, mask=mval, is_infer=True))
            null_idx = mask_vals.index[mask_vals.isna()]
            if len(null_idx):
                frames.append(SourceFrame(
                    df=df.loc[null_idx].reset_index(drop=True),
                    table=INFERENCE_TABLE, stem=stem, mask=None,
                    is_infer=True))
        else:
            frames.append(SourceFrame(
                df=df, table=INFERENCE_TABLE, stem=stem, mask=None,
                is_infer=True))
        return frames
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
            # Identical content: keep ONE copy. Leaving the incoming column
            # in place would make the outer merge below split it into
            # `<col>_x`/`<col>_y` duplicates carrying the same values.
            incoming = incoming.drop(columns=[col])
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
    replaced, other tables in the file are untouched. When a single mask
    applies (the `mask` argument, or the frame's single distinct `mask`
    value), the written table carries a `mask` column tagging every row —
    re-loading the file then resolves the frame's mask from that column.
    Multi-mask tables already carry the column and are written as-is.
    """
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(".db")
    p.parent.mkdir(parents=True, exist_ok=True)
    if mask is None and MASK_COLUMN in df.columns:
        uniq = df[MASK_COLUMN].dropna().astype(str).unique()
        mask = uniq[0] if len(uniq) == 1 else None
    out = df
    if mask is not None and MASK_COLUMN not in df.columns:
        # Single-mask output: tag the rows so re-loading keeps the grouping
        # (the in-memory frame the caller passed stays untouched).
        out = df.copy()
        out[MASK_COLUMN] = mask
    conn = sqlite3.connect(str(p))
    try:
        out.to_sql(MERGED_TABLE, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()
    logger.info("Wrote merged table (%d rows, mask=%s) to %s",
                len(out), mask, p)
    return str(p)
