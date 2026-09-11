"""Cross-package database / artifact contracts.

The suite talks through on-disk artifacts whose table and column names are
hard contracts between packages:

- ``profiler.db``  owned by microProfiler, read by microVis
- ``infer.db``     owned by microModel, read by microVis and probed by the
                   microProfiler inference bridge

Keeping the names and the SQLite identifier helpers here means a writer and a
reader cannot silently drift apart, and the quoting rules are identical on
every side.
"""

from __future__ import annotations

import os

# ── Artifact file names ──────────────────────────────────────────────────
PROFILER_DB_NAME = "profiler.db"
INFER_DB_NAME = "infer.db"

# ── profiler.db tables ───────────────────────────────────────────────────
IMAGE_TABLE = "image"

# ── infer.db tables ──────────────────────────────────────────────────────
INFERENCE_TABLE = "inference"
REDUCTION_TABLE_PREFIX = "reduction_"
FIND_CLUSTER_TABLE = "find_cluster"

# ── infer.db columns ─────────────────────────────────────────────────────
UID_COLUMN = "uid"
DIRECTORY_COLUMN = "directory"
FILENAME_COLUMN = "filename"
MASK_FILENAME_COLUMN = "mask_filename"
MASK_NAME_COLUMN = "mask_name"
FEATURES_COLUMN = "features"
WELL_COLUMN = "well"
LABEL_COLUMN = "label"
GROUND_TRUTH_COLUMN = "ground_truth"
PRED_CLASS_COLUMN = "pred_class"
PRED_PROB_COLUMN = "pred_prob"
PROB_COLUMN_PREFIX = "prob_"
CLUSTER_RES_PREFIX = "cluster_res_"
CLUSTER_PROB_PREFIX = "cluster_prob_"

# ── mask columns (microBase ImageDataset) ────────────────────────────────
MASK_COLUMN_PREFIX = "mask_"

# Bookkeeping table mapping profiler object-table / merged-table names to
# the mask (segmentation object type) they belong to. Written by
# microProfiler (object tables) and the merge writer (merge DBs); read by
# every mask-aware merge so differently named tables still group correctly.
TABLE_MASKS_TABLE = "_table_masks"

# Column the fused/merged output carries: which mask a row belongs to.
MASK_COLUMN = "mask"

# Dimensionality-reduction methods, canonical order. The writer (microModel),
# the reader (microVis) and the bridge all use this exact set.
DR_METHODS = ("pca", "umap", "pacmap", "localmap")

# Columns the inference table owns. Regex captures / config labels must not
# collide with these or CREATE TABLE / INSERT would produce duplicate names.
_BASE_RESERVED_INFERENCE = frozenset({
    UID_COLUMN, DIRECTORY_COLUMN, FILENAME_COLUMN, MASK_FILENAME_COLUMN,
    MASK_NAME_COLUMN, LABEL_COLUMN, GROUND_TRUTH_COLUMN, PRED_CLASS_COLUMN,
    PRED_PROB_COLUMN, FEATURES_COLUMN,
})


def reduction_table_name(method: str) -> str:
    """``pca`` -> ``reduction_pca``."""
    return f"{REDUCTION_TABLE_PREFIX}{method}"


def reduction_coord_prefix(method: str) -> str:
    """Column prefix for a reduction table: PCA uses ``pc``, others the method."""
    return "pc" if method == "pca" else method


def reserved_inference_columns(class_names=()) -> frozenset:
    """All column names an inference write may create for these classes."""
    return _BASE_RESERVED_INFERENCE | {
        f"{PROB_COLUMN_PREFIX}{c}" for c in class_names
    }


def mask_column(mask_name: str) -> str:
    """Bare mask name -> microBase metadata column (``cell`` -> ``mask_cell``)."""
    return f"{MASK_COLUMN_PREFIX}{mask_name}"


def bare_mask_name(column: str) -> str:
    """microBase metadata mask column -> bare mask name (inverse of mask_column)."""
    if column.startswith(MASK_COLUMN_PREFIX):
        return column[len(MASK_COLUMN_PREFIX):]
    return column


def sql_ident(name) -> str:
    """Double-quote a SQLite identifier, escaping embedded double quotes.

    Class names and regex captures become column names and may contain
    spaces, dots or quotes; unquoted identifiers would make CREATE / SELECT /
    INSERT invalid SQL.
    """
    return '"' + str(name).replace('"', '""') + '"'


_NUMERIC_TYPE_TOKENS = ("INT", "REAL", "FLOAT", "DOUBLE", "NUM", "DEC")


def is_numeric_sql_type(decl) -> bool:
    """True when a SQLite column declaration is numeric (INTEGER/REAL/DOUBLE/DECIMAL/...).

    Shared so a plot tab and the well grid agree on what counts as numeric.
    """
    return any(token in str(decl).upper() for token in _NUMERIC_TYPE_TOKENS)


def canonical_directory(path, root=None) -> str:
    """The DB ``directory`` value: ABSOLUTE path with forward slashes.

    All DB writers use this so ``directory`` is comparable across packages
    (microModel inference, microProfiler profiling, microVis scoping) and
    stays valid no matter where the dataset is mounted afterwards. A
    relative ``path`` is anchored at ``root`` before absolutizing; an
    absolute ``path`` ignores ``root``.
    """
    p = str(path)
    if root is not None and not os.path.isabs(p):
        p = os.path.join(str(root), p)
    return os.path.abspath(p).replace("\\", "/")


def resolve_directory(directory, root) -> str:
    """Absolute native-separator path for a stored ``directory`` value.

    Inverse of :func:`canonical_directory`. Empty means the dataset root
    itself. Stored values are absolute forward-slash paths (normalized by
    :func:`os.path.normpath`); legacy root-relative values are joined
    against the root.
    """
    if not directory:
        return os.path.abspath(str(root))
    d = str(directory)
    if os.path.isabs(d):
        return os.path.normpath(d)
    return os.path.abspath(os.path.join(str(root), d))
