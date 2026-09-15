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


def canonical_directory(path) -> str:
    """The DB ``directory`` value, PORTABLE-first.

    Stored relative to the process CWD (forward slashes) whenever the path
    lives under it — moving the dataset and the working directory together
    onto another computer/drive keeps every DB valid. Paths outside the CWD
    (e.g. another drive) fall back to absolute forward-slash. Reading is the
    inverse: :func:`resolve_directory`.
    """
    p = os.path.abspath(str(path))
    try:
        rel = os.path.relpath(p, os.getcwd())
    except ValueError:                        # different drive (Windows)
        return p.replace("\\", "/")
    if rel.startswith(".."):
        return p.replace("\\", "/")
    return rel.replace("\\", "/")


def resolve_directory(directory, root) -> str:
    """Absolute native-separator path for a stored ``directory`` value.

    Inverse of :func:`canonical_directory`. Empty means the dataset root
    itself. Resolution order for relative values: the process CWD (how
    canonical_directory writes them) first; the dataset ``root`` as the
    fallback anchor for values written from a different CWD. Absolute values
    pass through normalized (legacy DBs).
    """
    if not directory:
        return os.path.abspath(str(root))
    d = str(directory)
    if d in (".", "./"):                      # legacy rows: "." = the root
        return os.path.abspath(str(root))
    if os.path.isabs(d):
        return os.path.normpath(d)
    cand = os.path.abspath(d)
    if os.path.exists(cand):
        return cand
    alt = os.path.abspath(os.path.join(str(root), d))
    return alt if os.path.exists(alt) else cand
