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

# Dimensionality-reduction methods, canonical order. The writer (microModel),
# the reader (microVis) and the bridge all use this exact set.
DR_METHODS = ("pca", "umap", "pacmap", "localmap")

# Columns the inference table owns. Regex captures / config labels must not
# collide with these or CREATE TABLE / INSERT would produce duplicate names.
_BASE_RESERVED_INFERENCE = frozenset({
    UID_COLUMN, DIRECTORY_COLUMN, FILENAME_COLUMN, MASK_FILENAME_COLUMN,
    LABEL_COLUMN, GROUND_TRUTH_COLUMN, PRED_CLASS_COLUMN, PRED_PROB_COLUMN,
    FEATURES_COLUMN,
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


def canonical_directory(path, root) -> str:
    """Store a directory relative to *root*, with forward slashes.

    All DB writers use this so ``directory`` is comparable across packages
    (microModel inference, microProfiler profiling, microVis scoping). The
    root itself becomes ``"."``; a path outside the root keeps its ``..``
    prefix (relpath semantics) so nothing is silently escaped.
    """
    path = os.path.abspath(str(path))
    root = os.path.abspath(str(root))
    return os.path.relpath(path, root).replace("\\", "/")


def resolve_directory(directory, root) -> str:
    """Absolute path for a stored (canonical, root-relative) directory.

    Inverse of :func:`canonical_directory`. An already-absolute stored value
    (legacy DB) wins over the root, so both forms resolve. Empty directory
    means the dataset root itself.
    """
    if not directory:
        return os.path.abspath(str(root))
    if os.path.isabs(str(directory)):
        return str(directory)
    return os.path.abspath(os.path.join(str(root), str(directory)))
