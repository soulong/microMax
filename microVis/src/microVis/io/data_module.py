from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from natsort import natsorted

from microBase import normalize_well
from microBase.db_contracts import (
    DIRECTORY_COLUMN,
    WELL_COLUMN,
    is_numeric_sql_type,
    sql_ident,
)
from microVis.log_utils import get_logger

logger = get_logger("microVis.data_module")

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_name(name: str) -> None:
    """Raise ValueError if *name* is not a safe SQL table identifier."""
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(f"Unsafe table name: {name!r}")


def _safe_str(meta: pd.DataFrame, idx, col: str) -> str:
    """Safely read a metadata cell as string. Returns '' if column missing or value NA.

    pd.isna covers float NaN, pd.NA and pd.NaT (nullable dtypes would
    otherwise render as the literal '<NA>' in keys and labels).
    """
    if col not in meta.columns:
        return ""
    v = meta.at[idx, col]
    if v is None or pd.isna(v):
        return ""
    return str(v)


def _infer_plate_dims(wells: list[str]) -> tuple[int, int]:
    """Infer plate dimensions from well names (e.g. B2, D2 → 4 rows, 2 cols)."""
    def _row_number(letters: str) -> int:
        # A=1, ..., Z=26, AA=27, ... (base-26, matching well_grid_canvas)
        n = 0
        for ch in letters:
            n = n * 26 + (ord(ch) - ord("A") + 1)
        return n

    max_row = 0
    max_col = 0
    for w in wells:
        m = re.match(r"([A-Z]+)(\d+)", w)
        if m:
            max_row = max(max_row, _row_number(m.group(1)))
            max_col = max(max_col, int(m.group(2)))
    return max_row, max_col


# Default regex patterns are imported from microBase (DEFAULT_IMAGE_PATTERN,
# DEFAULT_MASK_PATTERN, DEFAULT_IMAGE_SUBDIR_PATTERN). Persisted patterns
# live in <dataset>/session.yml via microBase.SessionFile.


class DataModule:
    """Data access layer wrapping microBase ImageDataset + profiler.db."""

    def __init__(
        self,
        measurement_dir: str,
        image_pattern: str = "",
        mask_pattern: str = "",
        image_subdir_pattern: str = "",
    ):
        self._root_dir = Path(measurement_dir)
        self._dataset: Any = None
        self._image_pattern = image_pattern
        self._mask_pattern = mask_pattern
        self._image_subdir_pattern = image_subdir_pattern
        self._db_path: Path | None = None
        self._db_conn: sqlite3.Connection | None = None
        self._db_tables: dict[str, dict[str, str]] = {}
        self._metadata: pd.DataFrame | None = None
        self._img_dtype_cache: str | None = None
        self._wells_cache: list[str] | None = None
        self._df_cache: dict[str, pd.DataFrame] = {}
        self._row_index: dict[tuple, list[int]] = {}

        self._init_dataset()
        self._init_db()
        self._build_row_index()

    # ── Initialization ─────────────────────────────────────────────

    def _init_dataset(self):
        from microBase import ImageDataset

        if not self._image_pattern or not self._mask_pattern:
            raise ValueError(
                "image_pattern and mask_pattern are required. "
                "Set them in <dataset>/session.yml or via the Data tab."
            )

        kwargs: dict[str, Any] = {
            "image_pattern": self._image_pattern,
            "mask_pattern": self._mask_pattern,
        }
        if self._image_subdir_pattern:
            kwargs["image_subdir_pattern"] = self._image_subdir_pattern

        self._dataset = ImageDataset(self._root_dir, **kwargs)
        self._metadata = self._dataset.metadata
        logger.info("ImageDataset loaded: %d rows, channels=%s, masks=%s",
                    len(self._metadata), self.channels, self.mask_names)

    def _build_row_index(self) -> None:
        """Build a composite index mapping (well, field, stack, timepoint) → [row_idx].

        All key components are strings — missing columns become "". Wells use
        the canonical form (A01 -> A1), matching get_wells()/lookup_row_indices
        and the well-grid canvas. This supports non-standard datasets (e.g.
        HPA) that only capture a subset of structural groups.
        """
        meta = self._metadata
        self._row_index = {}
        for idx in meta.index:
            key = (
                str(normalize_well(_safe_str(meta, idx, "well"))),
                _safe_str(meta, idx, "field"),
                _safe_str(meta, idx, "stack"),
                _safe_str(meta, idx, "timepoint"),
            )
            self._row_index.setdefault(key, []).append(int(idx))

    def _init_db(self, db_path: Path | None = None):
        """Open DB and introspect tables. No default — caller must pass db_path."""
        if db_path is None:
            return
        if not db_path.exists():
            logger.info("DB not found: %s — profiling unavailable", db_path)
            return

        self._db_path = db_path
        self._db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_conn.row_factory = sqlite3.Row

        self._db_tables.clear()
        cursor = self._db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        for (tname,) in cursor.fetchall():
            # sqlite_* are internals; _* are reserved bookkeeping tables —
            # never plot variables.
            if tname.startswith("sqlite_") or tname.startswith("_"):
                continue
            cur = self._db_conn.execute(f'PRAGMA table_info("{tname}")')
            self._db_tables[tname] = {row[1]: row[2] for row in cur.fetchall()}
        logger.info("DB loaded: %s, tables=%s", db_path, list(self._db_tables.keys()))

    def load_db(self, db_path: str) -> None:
        """Switch to a different DB file. Closes old connection, refreshes cache."""
        p = Path(db_path)
        if not p.exists():
            raise FileNotFoundError(f"DB file not found: {p}")
        self.clear_db()
        self._init_db(p)

    def clear_db(self) -> None:
        """Detach any loaded DB entirely: close the connection and drop the
        table introspection and cached frames.

        Select DB replaces the whole selection — without this clear, an
        infer-only re-selection would leave the previous profiler DB (and
        its tables) attached as a hidden data source.
        """
        self.close_db()
        self._db_path = None
        self._db_tables.clear()
        self._df_cache.clear()

    def _select_columns(self, table: str) -> list[str]:
        """Return non-BLOB column names for SELECT, or ['*'] if table unknown."""
        cols = self._db_tables.get(table)
        if not cols:
            return ["*"]
        return [c for c, t in cols.items() if t.upper() != "BLOB"]

    # ── Properties ─────────────────────────────────────────────────

    @property
    def dataset(self) -> Any:
        """Underlying microBase.ImageDataset."""
        return self._dataset

    @property
    def channels(self) -> list[str]:
        return list(self._dataset.intensity_colnames)

    @property
    def mask_names(self) -> list[str]:
        return [m.replace("mask_", "", 1) for m in self._dataset.mask_colnames]

    @property
    def img_dtype(self) -> str:
        if self._img_dtype_cache is not None:
            return self._img_dtype_cache
        try:
            row = self._metadata.iloc[0]
            img, _ = self._dataset.get_imageset(row.name)
            self._img_dtype_cache = str(img.dtype)
        except Exception as e:
            logger.warning("Failed to detect image dtype, defaulting to uint16: %s", e)
            self._img_dtype_cache = "uint16"
        return self._img_dtype_cache

    def get_wells(self) -> list[str]:
        if self._wells_cache is None:
            if "well" not in self._metadata.columns:
                self._wells_cache = []
            else:
                # Canonical keys (A01 -> A1): the well-grid canvas speaks
                # normalized wells, so every well comparison in the GUI goes
                # through this accessor and stays consistent with it.
                self._wells_cache = natsorted(
                    str(normalize_well(w))
                    for w in self._metadata["well"].dropna().unique()
                )
        return self._wells_cache

    def get_fields(self) -> list[str]:
        if "field" not in self._metadata.columns:
            return []
        return natsorted(str(f) for f in self._metadata["field"].dropna().unique())

    def get_stacks(self) -> list[str]:
        if "stack" not in self._metadata.columns:
            return []
        return natsorted(str(s) for s in self._metadata["stack"].dropna().unique())

    def get_timepoints(self) -> list[str]:
        if "timepoint" not in self._metadata.columns:
            return []
        return natsorted(str(t) for t in self._metadata["timepoint"].dropna().unique())

    def get_filter_values(self, col: str, max_unique: int = 100) -> list[str]:
        """Return up to max_unique values for col in natsorted order.

        All unique values are natsorted (natural order: 1, 2, 10, 20 — not
        lexicographic 1, 10, 2, 20), then the first max_unique are kept.
        Returns [] if column is missing.
        """
        if col not in self._metadata.columns:
            return []
        vals = natsorted(str(v) for v in self._metadata[col].dropna().unique())
        return vals[:max_unique]

    def get_plate_dims(self) -> tuple[int, int]:
        return _infer_plate_dims(self.get_wells())

    def directory_scopes(self) -> list[str]:
        """Directory values identifying this dataset in DB `directory` columns.

        DB writers store absolute forward-slash directories, so the dataset's
        metadata `directory` values are the exact match keys; the absolute
        dataset dir and "." are included as fallback scopes (absolute for
        path-prefix matching of sub-directories, "." for legacy root-level
        rows).
        """
        scopes = [str(self._root_dir), "."]
        if self._metadata is not None and DIRECTORY_COLUMN in self._metadata.columns:
            scopes.extend(
                str(v) for v in self._metadata[DIRECTORY_COLUMN].dropna().unique()
            )
        return scopes

    # ── DB access ──────────────────────────────────────────────────

    def get_profiling_tables(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        exclude = {DIRECTORY_COLUMN}
        for tname, cols in self._db_tables.items():
            profiling = [c for c, t in cols.items()
                         if c not in exclude and t.upper() != "BLOB"]
            if profiling:
                result[tname] = profiling
        return result

    def get_profiling_columns(self, table: str) -> list[tuple[str, str, bool]]:
        cols = self._db_tables.get(table, {})
        exclude = {DIRECTORY_COLUMN}
        result = []
        for cname, ctype in cols.items():
            if cname in exclude or ctype.upper() == "BLOB":
                continue
            result.append((cname, ctype, is_numeric_sql_type(ctype)))
        return result

    def get_table_df(self, table: str) -> pd.DataFrame | None:
        _validate_table_name(table)
        if self._db_path is None:
            return None
        if table in self._df_cache:
            return self._df_cache[table]
        # Open a temporary connection for this read
        conn = self._db_conn
        temp = conn is None
        if temp:
            conn = sqlite3.connect(str(self._db_path))
        try:
            cols = self._select_columns(table)
            col_sql = ", ".join(sql_ident(c) for c in cols) if cols != ["*"] else "*"
            df = pd.read_sql(
                f'SELECT {col_sql} FROM {sql_ident(table)}', conn)
            self._df_cache[table] = df
            return df
        except Exception as e:
            logger.warning("get_table_df(%s) failed: %s", table, e)
            return None
        finally:
            if temp:
                conn.close()

    def get_table_preview(self, table: str, limit: int = 20) -> tuple[pd.DataFrame | None, int]:
        """Fetch first `limit` rows and total row count. Skips cache."""
        _validate_table_name(table)
        if self._db_path is None:
            return None, 0
        conn = self._db_conn
        temp = conn is None
        if temp:
            conn = sqlite3.connect(str(self._db_path))
        try:
            total = conn.execute(
                f'SELECT COUNT(*) FROM {sql_ident(table)}').fetchone()[0]
            cols = self._select_columns(table)
            col_sql = ", ".join(sql_ident(c) for c in cols) if cols != ["*"] else "*"
            df = pd.read_sql(
                f'SELECT {col_sql} FROM {sql_ident(table)} LIMIT {limit}', conn)
            return df, total
        except Exception as e:
            logger.warning("get_table_preview(%s) failed: %s", table, e)
            return None, 0
        finally:
            if temp:
                conn.close()

    def invalidate_table_cache(self, table: str | None = None) -> None:
        """Clear cached DataFrames. If table is given, clear only that table."""
        if table:
            self._df_cache.pop(table, None)
        else:
            self._df_cache.clear()

    def aggregate(self, table: str, column: str, method: str) -> dict:
        _validate_table_name(table)
        df = self.get_table_df(table)
        if df is None:
            logger.warning("aggregate: get_table_df(%s) returned None", table)
            return {}
        if column not in df.columns:
            logger.warning("aggregate: column %s not in table %s, cols=%s",
                           column, table, list(df.columns[:10]))
            return {}
        if WELL_COLUMN not in df.columns:
            logger.warning("aggregate: no 'well' column in table %s, cols=%s",
                           table, list(df.columns))
            return {}

        col_type = self._db_tables.get(table, {}).get(column, "")
        is_num = is_numeric_sql_type(col_type)

        logger.debug("aggregate: table=%s col=%s is_num=%s method=%s rows=%d",
                     table, column, is_num, method, len(df))

        if not is_num:
            grouped = df.groupby(WELL_COLUMN)[column].first()
        elif method == "std":
            grouped = df.groupby(WELL_COLUMN)[column].std()
        elif method == "sum":
            grouped = df.groupby(WELL_COLUMN)[column].sum()
        else:
            grouped = df.groupby(WELL_COLUMN)[column].mean()

        # Native Python scalars: the well grid's numeric check handles
        # numpy too, but native values keep downstream code simple.
        result = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in grouped.dropna().to_dict().items()
        }
        logger.debug("aggregate: result has %d wells", len(result))
        return result

    def write_label_table(self, table_name: str, df: pd.DataFrame) -> None:
        """Write label annotations to a table in profiler.db.

        Opens a temporary connection, writes, and closes immediately.

        Args:
            table_name: Name of the table to write to.
            df: DataFrame with columns [well, field, stack, timepoint, label, class].
        """
        _validate_table_name(table_name)
        if self._db_path is None:
            raise RuntimeError("No database available")

        conn = sqlite3.connect(str(self._db_path))
        try:
            # Refuse to silently REPLACE a non-label table: the table name is
            # free-form user input, and a profiler/infer table (e.g. "cell")
            # must never be dropped because the name collided. A label table
            # is identified by its well/label/class columns.
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if exists:
                existing_cols = {
                    row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")')
                }
                if not {"well", "label", "class"} <= existing_cols:
                    raise RuntimeError(
                        f"Refusing to replace table '{table_name}': it is not a "
                        f"label table (expected at least well/label/class, found "
                        f"{sorted(existing_cols)}). Choose another table name."
                    )
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            # Update the table schema cache
            cur = conn.execute(f'PRAGMA table_info("{table_name}")')
            self._db_tables[table_name] = {row[1]: row[2] for row in cur.fetchall()}
        finally:
            conn.close()

        # Invalidate cached DataFrame for this table
        self.invalidate_table_cache(table_name)
        logger.info("Wrote %d rows to table '%s'", len(df), table_name)

    # ── Image access ───────────────────────────────────────────────

    def lookup_row_indices(
        self, wells: list[str], fields: list[str],
        stacks: list[str] | None = None,
        timepoints: list[str] | None = None,
    ) -> list[tuple[int, str, str, str, str]]:
        # Empty list = "no filter" — collect all unique values from the row
        # index for that dimension. This handles absent metadata columns
        # (widget is None → get_selected_*() returns [] → no filter).
        if not wells:
            wells = sorted({k[0] for k in self._row_index})
        else:
            # Callers may pass raw captured wells ('A01'); the index is keyed
            # canonically ('A1').
            wells = [str(normalize_well(w)) for w in wells]
        if not fields:
            fields = sorted({k[1] for k in self._row_index})
        if not stacks:
            stacks = sorted({k[2] for k in self._row_index})
        if not timepoints:
            timepoints = sorted({k[3] for k in self._row_index})
        results: list[tuple[int, str, str, str, str]] = []
        idx = self._row_index
        for well in wells:
            for field in fields:
                for stack in stacks:
                    for timepoint in timepoints:
                        row_idxs = idx.get((well, field, stack, timepoint))
                        if row_idxs:
                            for row_idx in row_idxs:
                                results.append((row_idx, well, field, stack, timepoint))
        return results

    def filter_rows_by_extra(
        self, row_indices: list[int], extra_filters: dict[str, list[str]]
    ) -> list[int]:
        """Filter row indices by extra-col selections (post-hoc).

        extra_filters: {col_name: [selected_values]}. Empty dict or empty
        list for a col = no filter on that col.
        """
        if not extra_filters:
            return row_indices
        meta = self._metadata
        keep = set(row_indices)
        for col, selected in extra_filters.items():
            if not selected:
                continue
            if col not in meta.columns:
                continue
            valid = set(meta.loc[meta[col].astype(str).isin(selected)].index)
            keep &= valid
        return [r for r in row_indices if r in keep]

    def get_imageset(self, row_idx: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self._dataset.get_imageset(row_idx)

    # ── Cleanup ────────────────────────────────────────────────────

    @property
    def db_path(self) -> "Path | None":
        """Path of the profiling DB backing this dataset (None if absent)."""
        return self._db_path

    def close_db(self) -> None:
        """Close the persistent DB connection. Cached image data remains
        available; the DataModule itself is discarded with the dataset."""
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None
            logger.info("DB connection closed (cache retained)")

    # Alias kept for readability at call sites that think of it as "final
    # teardown" — both names close exactly the same connection.
    close = close_db


def parse_plate_metadata(path: str) -> pd.DataFrame:
    """Parse a plate-shaped Excel metadata file into a long DataFrame.

    Each sheet is a plate layout:
      - Row 1: column numbers [None, 1, 2, ...]
      - Col A: row IDs [A, B, C, ...]
      - Data starts at B2 (well A1)

    Returns a DataFrame with 'well' column + one column per sheet name.
    """
    xls = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []

    for sheet_name in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        if raw.empty or raw.shape[0] < 2 or raw.shape[1] < 2:
            continue

        # Column numbers from row 0 (skip first cell which is None)
        col_numbers = raw.iloc[0, 1:].tolist()
        # Row IDs from column 0 (skip header row)
        row_ids = raw.iloc[1:, 0].tolist()

        records = []
        for r_idx, row_id in enumerate(row_ids):
            if pd.isna(row_id):
                continue
            row_id = str(row_id).strip()
            for c_idx, col_num in enumerate(col_numbers):
                if pd.isna(col_num):
                    continue
                well = f"{row_id}{int(col_num)}"
                val = raw.iloc[r_idx + 1, c_idx + 1]
                records.append({"well": well, sheet_name: val})

        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        return pd.DataFrame(columns=["well"])

    # Merge all sheets on 'well'
    result = frames[0]
    for df in frames[1:]:
        result = result.merge(df, on="well", how="outer")

    # Auto-detect column types: try numeric, fall back to string
    for col in result.columns:
        if col == "well":
            continue
        converted = pd.to_numeric(result[col], errors="coerce")
        non_null = result[col].notna().sum()
        # Coerce only when every non-null value converted — "at least one"
        # would silently blank entries like "12a".
        if non_null > 0 and converted.notna().sum() == non_null:
            result[col] = converted

    return result


def merge_metadata(
    df: pd.DataFrame,
    metadata: pd.DataFrame | None,
    key: str = "well",
) -> pd.DataFrame:
    """Left-join only the metadata columns missing from *df*.

    Re-merging after a Write-to-DB never duplicates columns (the DB table
    already carries them), and tables without the join key are returned
    unchanged. On the well key both sides are normalized ('A01' -> 'A1')
    into a temp join column so captured verbatim wells still meet the
    Excel sheet's spelling; the original well columns stay verbatim.
    """
    if df is None or metadata is None:
        return df
    if key not in df.columns or key not in metadata.columns:
        return df
    missing = [c for c in metadata.columns if c != key and c not in df.columns]
    if not missing:
        return df
    if key == "well":
        left = df.assign(__well_key__=df[key].map(normalize_well))
        right = metadata[[key] + missing].assign(
            __well_key__=metadata[key].map(normalize_well))
        return left.merge(right.drop(columns=[key]), on="__well_key__",
                          how="left").drop(columns=["__well_key__"])
    return df.merge(metadata[[key] + missing], on=key, how="left")


def aggregate_by_well(
    df: pd.DataFrame,
    column: str,
    method: str = "mean",
) -> dict:
    """Reduce a per-object DataFrame to one value per well.

    Numeric columns use the requested method (mean/sum/std); text columns take
    the first value. Returns {well: value}; missing values are dropped.
    """
    if df is None or column not in df.columns or "well" not in df.columns:
        return {}
    if not pd.api.types.is_numeric_dtype(df[column]):
        grouped = df.groupby("well")[column].first()
    elif method == "std":
        grouped = df.groupby("well")[column].std()
    elif method == "sum":
        grouped = df.groupby("well")[column].sum()
    else:
        grouped = df.groupby("well")[column].mean()
    return grouped.dropna().to_dict()
