"""microBase: shared library for microProfiler / microVis / microModel.

Provides:
- ImageDataset    : whole-image loader with cell cropping
- CellDataset     : pre-cropped single-cell loader
- crop_cell       : pure function for cropping a single cell
- crop_all_cells  : convenience wrapper to crop every cell in a mask
- get_labels      : pure function for listing cell labels in a mask
- edge_pixel_ratio: per-object edge-pixel/perimeter-pixel ratio
- ImageReadError  : raised by the strict readers on missing/unreadable files
- load_yaml, save_yaml, SessionFile
- build_pipeline, apply (augmentation)
- normalize
- derive_well, MetadataSchema
- db_contracts    : shared profiler.db / infer.db table/column names + SQL helpers
"""

__version__ = "0.13.0"

import os as _os
_os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
_os.environ.setdefault("ALBUMENTATIONS_NO_TELEMETRY", "1")

from .schema import (
    MetadataSchema,
    derive_well,
)
from .errors import (
    MicroMaxError,
    ConfigError,
    DatasetError,
    ImageReadError,
    DataError,
    DependencyError,
)
from .io import (
    read_image,
    read_tiff_channels,
    read_mask,
)
from .cells import (
    crop_cell,
    crop_all_cells,
    edge_pixel_ratio,
    get_labels,
)
from .config import (
    load_yaml,
    save_yaml,
    SessionFile,
    normalize_null_strings,
)
from .normalize import normalize
from .image_dataset import ImageDataset
from .cell_dataset import CellDataset
from .patterns import (
    DEFAULT_IMAGE_PATTERN,
    DEFAULT_MASK_PATTERN,
    DEFAULT_IMAGE_SUBDIR_PATTERN,
)
from .db_contracts import (
    PROFILER_DB_NAME,
    INFER_DB_NAME,
    IMAGE_TABLE,
    INFERENCE_TABLE,
    REDUCTION_TABLE_PREFIX,
    FIND_CLUSTER_TABLE,
    UID_COLUMN,
    DIRECTORY_COLUMN,
    FILENAME_COLUMN,
    MASK_FILENAME_COLUMN,
    FEATURES_COLUMN,
    WELL_COLUMN,
    LABEL_COLUMN,
    GROUND_TRUTH_COLUMN,
    PRED_CLASS_COLUMN,
    PRED_PROB_COLUMN,
    PROB_COLUMN_PREFIX,
    CLUSTER_RES_PREFIX,
    CLUSTER_PROB_PREFIX,
    MASK_COLUMN_PREFIX,
    DR_METHODS,
    reduction_table_name,
    reduction_coord_prefix,
    reserved_inference_columns,
    mask_column,
    bare_mask_name,
    sql_ident,
    is_numeric_sql_type,
    canonical_directory,
    resolve_directory,
)


__all__ = [
    "ImageDataset",
    "CellDataset",
    "crop_cell",
    "crop_all_cells",
    "edge_pixel_ratio",
    "get_labels",
    "read_image",
    "read_tiff_channels",
    "read_mask",
    "ImageReadError",
    "MicroMaxError",
    "ConfigError",
    "DatasetError",
    "DataError",
    "DependencyError",
    "load_yaml",
    "save_yaml",
    "SessionFile",
    "normalize_null_strings",
    "build_pipeline",
    "apply",
    "normalize",
    "derive_well",
    "MetadataSchema",
    "DEFAULT_IMAGE_PATTERN",
    "DEFAULT_MASK_PATTERN",
    "DEFAULT_IMAGE_SUBDIR_PATTERN",
    "PROFILER_DB_NAME",
    "INFER_DB_NAME",
    "IMAGE_TABLE",
    "INFERENCE_TABLE",
    "REDUCTION_TABLE_PREFIX",
    "FIND_CLUSTER_TABLE",
    "UID_COLUMN",
    "DIRECTORY_COLUMN",
    "FILENAME_COLUMN",
    "MASK_FILENAME_COLUMN",
    "FEATURES_COLUMN",
    "WELL_COLUMN",
    "LABEL_COLUMN",
    "GROUND_TRUTH_COLUMN",
    "PRED_CLASS_COLUMN",
    "PRED_PROB_COLUMN",
    "PROB_COLUMN_PREFIX",
    "CLUSTER_RES_PREFIX",
    "CLUSTER_PROB_PREFIX",
    "MASK_COLUMN_PREFIX",
    "DR_METHODS",
    "reduction_table_name",
    "reduction_coord_prefix",
    "reserved_inference_columns",
    "mask_column",
    "bare_mask_name",
    "sql_ident",
    "is_numeric_sql_type",
    "canonical_directory",
    "resolve_directory",
]


# ── Lazy augmentation re-exports (PEP 562) ──────────────────────────────────
# augment.py imports albumentations at module import — a heavyweight optional
# dependency. Importing `microBase` (for IO/schema/datasets) must not require
# it, so build_pipeline/apply are resolved on FIRST ATTRIBUTE ACCESS instead
# of at package import.
_AUGMENT_NAMES = {"build_pipeline", "apply"}


def __getattr__(name):
    if name in _AUGMENT_NAMES:
        from . import augment as _augment

        return getattr(_augment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _AUGMENT_NAMES)
