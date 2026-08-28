"""microBase: shared library for microProfiler / microVis / microModel.

Provides:
- ImageDataset    : whole-image loader with cell cropping
- CellDataset     : pre-cropped single-cell loader
- crop_cell       : pure function for cropping a single cell
- crop_all_cells  : convenience wrapper to crop every cell in a mask
- get_labels      : pure function for listing cell labels in a mask
- read_tiff, read_tiff_channels, read_mask
- load_yaml, save_yaml, SessionFile
- build_pipeline, apply (augmentation)
- normalize
- derive_well, MetadataSchema
"""

__version__ = "0.9.0"

import os as _os
_os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
_os.environ.setdefault("ALBUMENTATIONS_NO_TELEMETRY", "1")

from .schema import (
    MetadataSchema,
    derive_well,
)
from .io import (
    read_tiff,
    read_tiff_channels,
    read_mask,
)
from .cells import (
    crop_cell,
    crop_all_cells,
    get_labels,
)
from .config import (
    load_yaml,
    save_yaml,
    SessionFile,
    normalize_null_strings,
)
from .augment import (
    build_pipeline,
    apply,
)
from .normalize import normalize
from .image_dataset import ImageDataset
from .cell_dataset import CellDataset
from .patterns import (
    DEFAULT_IMAGE_PATTERN,
    DEFAULT_MASK_PATTERN,
    DEFAULT_IMAGE_SUBDIR_PATTERN,
)


__all__ = [
    "ImageDataset",
    "CellDataset",
    "crop_cell",
    "crop_all_cells",
    "get_labels",
    "read_tiff",
    "read_tiff_channels",
    "read_mask",
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
]
