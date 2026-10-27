"""Mask-aware merge of profiler-style object tables and microModel infer DBs.

Shared by microProfiler (per-block absorb -> ``infer_<model>.db``; end-of-run
rebuild -> ``merged.db``) and microVis (loading ``merged.db``).

Merge model
-----------
- Every source frame belongs to ONE mask (the segmentation object type):

  * profiler-style DBs (profiler.db, infer_<model>.db, merged.db): a table
    that carries a `mask` column (a previous merge output) uses that;
    otherwise the table name itself is taken as the mask name — object
    tables are named after their mask.
  * raw microModel infer DBs: the `inference` table carries a `mask_name`
    column. Missing column / NULL (single-cell inference over pre-cropped
    cells has no mask file) -> the frame's mask is unknown and the caller's
    default applies.
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
- Prefixing: a caller may tag source frames with a `prefix` (microProfiler
  tags every ``infer_<model>.db`` frame with the model name). When TWO OR
  MORE distinct prefixes are in one fusion, every prefixed frame's data
  columns are renamed ``<prefix>_<column>`` up front (identity columns stay
  bare — they are the join keys), so which run produced a column never
  depends on the order. One prefix (or none) keeps bare, concise names.
- On column-name collisions between remaining sources inside one mask, the
  incoming non-identity column is renamed ``<db-stem>/<column>`` (then
  ``<db-stem>/<table>/<column>``, then a numeric suffix); prefixed frames
  use ``<prefix>_<table>_<column>`` instead.

Per-mask table writes
---------------------
``absorb_infer_db`` folds one raw microModel infer DB (inference +
reduction_<method> + find_cluster tables) into per-mask wide tables inside
``infer_<model>.db`` — the per-mask table name IS the mask name, so the file
has the same structure as profiler.db. Rows are replaced per `directory`
(microModel's re-run protection), so re-running a block refreshes its rows
without touching other masks or models.

``build_merged_db`` rebuilds ``<dataset>/merged.db`` from profiler.db and
every ``infer_<model>.db``: one table per mask, holding that mask's profiler
objects fused with all models' infer results for the same mask. The file is
rebuilt from scratch, so masks that disappeared never linger.

``write_merged_db`` persists an in-memory fused table (microVis Write to
DB) into a NEW SQLite database (table `merged`, tagged with its `mask`
column so the output re-loads with its mask intact). Source DBs are never
touched by any writer.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from microBase.db_contracts import (
    FEATURES_COLUMN,
    FIND_CLUSTER_TABLE,
    INFER_DB_STEM_PREFIX,
    INFERENCE_TABLE,
    LABEL_COLUMN,
    MASK_COLUMN,
    MASK_NAME_COLUMN,
    MERGED_DB_NAME,
    PROFILER_DB_NAME,
    REDUCTION_TABLE_PREFIX,
    UID_COLUMN,
    model_name_from_infer_db,
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
    prefix: str | None = None       # model-name namespace applied on fusion


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


def read_profiler_frames(path: str, prefix: str | None = None) -> list[SourceFrame]:
    """Object tables of a profiler-style DB (profiler.db, infer_<model>.db,
    merged.db).

    An object table is identified by its `label` column (the per-object
    identity); `well` is NOT required — datasets without a well capture are
    officially supported, and such tables merge on the identity columns they
    do carry (label, directory, ...). Bookkeeping and image-level tables are
    skipped. Mask resolution per table: `mask` column (previous merge
    output) -> the table name itself. `prefix` namespaces the frames' data
    columns on fusion (see fuse_frames).
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
                        mask=None if pd.isna(mval) else str(mval),
                        prefix=prefix))
                continue
            frames.append(SourceFrame(
                df=df, table=name, stem=stem, mask=name, prefix=prefix))
        return frames
    finally:
        conn.close()


def read_infer_frames(path: str) -> list[SourceFrame]:
    """The inference table of a raw microModel infer DB, joined with its
    reduction_<method> and find_cluster tables.

    The features BLOB and the infer-internal uid are dropped; every other
    column (predictions, coordinates, cluster IDs/confidences) meets in one
    wide row per object via the uid join. One table can hold several masks
    (a re-run over another mask appends rows), so the frame is split per
    `mask_name` value — one SourceFrame each, exactly like the profiler
    frames. A missing column or NULL (single-cell inference over
    pre-cropped cells has no mask file) yields a frame with an unknown mask
    -> the caller's default applies.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if INFERENCE_TABLE not in _table_names(conn):
            return []
        stem = Path(path).stem
        df = _read_table(conn, INFERENCE_TABLE)
        if df.empty:
            return []
        join_tables = [name for name in _table_names(conn)
                       if name.startswith(REDUCTION_TABLE_PREFIX)
                       or name == FIND_CLUSTER_TABLE]
        for name in join_tables:
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


def read_db_frames(path: str, prefix: str | None = None) -> list[SourceFrame]:
    """All source frames of one DB file, typed by its content."""
    if INFERENCE_TABLE in table_names_of(path):
        return read_infer_frames(path)
    return read_profiler_frames(path, prefix=prefix)


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows
            if not r[0].startswith("sqlite_")]


def table_names_of(path: str) -> list[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return _table_names(conn)
    finally:
        conn.close()


# ── Fusing ───────────────────────────────────────────────────────────────

def _unique_name(existing: set[str], frame: SourceFrame, col: str) -> str:
    """A free column name for a colliding incoming column.

    Prefixed frames namespace with ``<prefix>_<table>_<col>``; plain frames
    keep the historical ``<stem>/<col>`` form.
    """
    if frame.prefix:
        lead = frame.prefix
        sep = "_"
    else:
        lead = frame.stem
        sep = "/"
    for candidate in (f"{lead}{sep}{col}", f"{lead}{sep}{frame.table}{sep}{col}"):
        if candidate not in existing:
            return candidate
    n = 2
    while f"{lead}{sep}{frame.table}{sep}{col} ({n})" in existing:
        n += 1
    return f"{lead}{sep}{frame.table}{sep}{col} ({n})"


def _fold(acc: pd.DataFrame | None, frame: SourceFrame) -> pd.DataFrame:
    """Merge one same-mask source frame into the accumulated table.

    acc is not mutated; the incoming frame is copied when renames are
    needed. On identity-column collision with different content, the
    incoming column is renamed (see _unique_name).
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
        new_name = _unique_name(existing, frame, col)
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


def _prefixed_frame(frame: SourceFrame) -> SourceFrame:
    """A copy of a prefixed frame with every non-identity column renamed
    ``<prefix>_<col>``.

    Identity columns stay bare — they are the outer-merge join keys. The
    frame's mask attribute is preserved so the grouping is unaffected.
    """
    df = frame.df.copy()
    key_set = set(KEY_COLUMNS)
    df = df.rename(columns={c: f"{frame.prefix}_{c}" for c in df.columns
                            if c not in key_set})
    return SourceFrame(df=df, table=frame.table, stem=frame.stem,
                       mask=frame.mask, is_infer=frame.is_infer,
                       prefix=frame.prefix)


def fuse_frames(frames: list[SourceFrame],
                default_mask: str | None = None,
                force_prefix: bool = False) -> pd.DataFrame:
    """Fuse source frames into ONE table, grouped by mask.

    Frames of the same mask are fused (outer merge on identity columns);
    different masks are stacked and tagged in the `mask` column. A frame
    with an unknown mask (single-cell infer rows) joins `default_mask` —
    normally the profiler's mask — with a log hint; with no default it
    forms its own group. Prefixed frames' data columns are namespaced
    `<prefix>_<column>` when the fusion holds TWO OR MORE distinct
    prefixes (microProfiler tags every infer_<model>.db frame with its
    model name); a single prefixed source keeps bare names. force_prefix
    applies that namespacing whenever ANY prefixed frame exists — the
    dataset-level rule (two or more infer DBs prefix EVERY model's
    columns) for fusions where each per-mask group would hold a single
    prefix.
    """
    if not frames:
        raise ValueError("No selectable object data in the selected DB(s)")

    # Multiple prefixed sources: namespace every prefixed frame's data
    # columns up front, so 'which run produced this column' never depends on
    # the order (and two runs' identical-named — but possibly
    # different-valued — prediction columns can never silently collapse
    # into one).
    prefixes = {f.prefix for f in frames if f.prefix}
    if len(prefixes) >= 2 or (force_prefix and prefixes):
        frames = [_prefixed_frame(f) if f.prefix else f for f in frames]

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


# ── Per-mask table writes (infer_<model>.db / merged.db) ────────────────

def _write_mask_table(df: pd.DataFrame, db_path: Path, table: str) -> None:
    """Write one mask's fused rows as table `table` (the mask name).

    Re-run protection mirrors microModel's writer: rows whose `directory`
    is among the incoming rows' directories are replaced, rows of other
    directories (other masks/sites written by a different run of the SAME
    model) survive. The whole table is then rewritten (sizes are modest).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        if table in _table_names(conn):
            existing = _read_table(conn, table, blob_free=False)
            if (not existing.empty and "directory" in df.columns
                    and "directory" in existing.columns):
                dirs = set(df["directory"].astype(str))
                keep = ~existing["directory"].astype(str).isin(dirs)
                if keep.any():
                    df = pd.concat([existing[keep], df], axis=0,
                                   ignore_index=True)
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()
    logger.debug("Wrote mask table '%s' (%d rows) to %s", table, len(df),
                 db_path)


def _split_mask_blocks(fused: pd.DataFrame, frames: list[SourceFrame],
                       default_mask: str | None = None
                       ) -> list[tuple[str | None, pd.DataFrame]]:
    """Split a fused frame into (mask, block) pairs.

    A multi-mask fusion is tagged with the `mask` column; a single-mask
    fusion is untagged and the mask comes from the source frames, falling
    back to `default_mask` (the caller's answer for None-mask frames that
    fuse_frames folded into its default group).
    """
    if MASK_COLUMN in fused.columns:
        blocks = []
        key = fused[MASK_COLUMN].map(lambda v: None if pd.isna(v) else str(v))
        for mval, sub in fused.groupby(key, dropna=False, sort=False):
            blocks.append((mval,
                           sub.drop(columns=[MASK_COLUMN]).reset_index(drop=True)))
        return blocks
    for f in frames:
        if f.mask is not None:
            return [(f.mask, fused)]
    return [(default_mask, fused)]


def absorb_infer_db(raw_path, out_db, default_mask: str | None = None) -> list[str]:
    """Fold one raw microModel infer DB into per-mask tables of `out_db`.

    The raw file holds microModel's CLI layout (the `inference` table plus
    reduction_<method>/find_cluster tables); read_infer_frames already
    uid-joins them into one wide frame, which is split per mask and written
    as one table NAMED AFTER THE MASK — so `out_db` ends up with the same
    structure as profiler.db. Rows are replaced per directory (re-running
    the same block refreshes its rows). The raw file is never modified.

    default_mask names the mask of rows without `mask_name` info. Returns
    the list of mask names written.
    """
    raw = str(raw_path)
    frames = read_infer_frames(raw)
    if not frames:
        raise ValueError(f"No inference rows in {raw}")
    fused = fuse_frames(frames, default_mask=default_mask)
    blocks = _split_mask_blocks(fused, frames, default_mask)
    unnamed = [b for b in blocks if b[0] is None]
    if unnamed:
        raise ValueError(
            f"{raw}: inference rows carry no mask name and no default "
            f"applies — cannot name the output table.")

    out = Path(out_db)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for mask, block in blocks:
        _write_mask_table(block, out, str(mask))
        written.append(str(mask))
    logger.info("Absorbed %s into %s (mask tables: %s)",
                Path(raw).name, out.name, ", ".join(written))
    return written


def build_merged_db(root_dir) -> str | None:
    """(Re)build ``<dataset>/merged.db`` — the dataset-wide per-mask fusion.

    profiler.db's object tables and every ``infer_<model>.db`` are fused by
    table name (= mask name): each mask keeps ONE merged table holding its
    profiler measurements next to every model's predictions/coordinates.
    When TWO OR MORE infer DBs exist, every infer frame's data columns are
    prefixed with its model name (``<model>_<column>``); a single infer DB
    keeps bare names. The file is rebuilt from scratch, so a mask that no
    longer has sources disappears instead of lingering. With no sources at
    all a stale merged.db is deleted and None returned.
    """
    root = Path(root_dir)
    out = root / MERGED_DB_NAME
    profiler_db = root / PROFILER_DB_NAME
    infer_dbs = sorted(root.glob(f"{INFER_DB_STEM_PREFIX}*.db"))
    # A stale merged.db without any source is dropped, not served.
    if not profiler_db.exists() and not infer_dbs:
        if out.exists():
            out.unlink()
            logger.info("Removed stale %s (no profiler/infer DBs left)", out.name)
        return None

    frames: list[SourceFrame] = []
    if profiler_db.exists():
        frames.extend(read_profiler_frames(str(profiler_db)))
    # Dataset-level prefix rule: with several infer DBs every model's
    # columns carry its name — even when the models cover DIFFERENT masks
    # (each per-mask group would otherwise hold a single prefix and stay
    # bare, and a mask's columns would flip bare/prefixed the day a second
    # model is run on it). A single infer DB stays bare.
    force_prefix = len(infer_dbs) >= 2
    for db in infer_dbs:
        if INFERENCE_TABLE in table_names_of(str(db)):
            # A raw (not yet absorbed) microModel infer DB — its `inference`
            # table must not masquerade as a per-mask object table here.
            logger.warning(
                "%s is a raw microModel infer DB (absorb it first) — "
                "skipped in %s", db.name, out.name)
            continue
        prefix = model_name_from_infer_db(db.name) if force_prefix else None
        frames.extend(read_profiler_frames(str(db), prefix=prefix))
    if not frames:
        if out.exists():
            out.unlink()
            logger.info("Removed stale %s (no object tables in the sources)",
                        out.name)
        return None

    profiler_masks = [f.mask for f in frames if f.mask]
    fallback = profiler_masks[0] if profiler_masks else None
    groups: dict[str, list[SourceFrame]] = {}
    for f in frames:
        mask = f.mask or fallback
        if mask is None:
            logger.warning("%s/%s has no mask name — skipped in %s",
                           f.stem, f.table, out.name)
            continue
        groups.setdefault(mask, []).append(f)

    # Fresh file: a mask whose sources are gone must not survive as a
    # stale table.
    if out.exists():
        out.unlink()
    for mask, group in groups.items():
        fused = fuse_frames(group, default_mask=mask, force_prefix=force_prefix)
        if MASK_COLUMN in fused.columns:
            fused = fused.drop(columns=[MASK_COLUMN])
        _write_mask_table(fused, out, str(mask))
    logger.info("Built %s: %d mask table(s) from profiler.db + %d infer DB(s)",
                out.name, len(groups), len(infer_dbs))
    return str(out)


# ── In-memory write (microVis Write to DB) ───────────────────────────────

def write_merged_db(df: pd.DataFrame, path, mask: str | None = None) -> str:
    """Write a fused table into a SQLite DB (table `merged`).

    The file is overwritten WHOLESALE (an existing file is deleted first —
    same fresh-file rule as build_merged_db) so a stale table from an
    earlier write or a pipeline run never survives next to the new `merged`
    table; the default output name is therefore the pipeline's own
    `merged.db`. When a single mask applies (the `mask` argument, or the
    frame's single distinct `mask` value), the written table carries a
    `mask` column tagging every row — re-loading the file then resolves the
    frame's mask from that column. Multi-mask tables already carry the
    column and are written as-is.
    """
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(".db")
    p.parent.mkdir(parents=True, exist_ok=True)
    # Fresh file: no leftover table from an earlier write can survive.
    if p.exists():
        p.unlink()
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
