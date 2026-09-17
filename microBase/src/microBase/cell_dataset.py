"""CellDataset: pre-cropped single-cell dataset loader.

Each TIFF file in `root` is one cell, with channels multiplexed inside
(as pages for CHW, or as last axis for HWC).

No masks. No whole-image cropping. Just reads pre-cropped cells.

Optional `image_pattern` parses each filename to extract metadata columns
(well, field, stack, timepoint, label, etc.) — useful for microModel's
infer single_cell mode where per-cell metadata is written to the DB.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted

from . import io as _io
from .errors import ConfigError, DataError, DatasetError, ImageReadError
from .schema import MetadataSchema

logger = logging.getLogger(__name__)


class CellDataset:
    """Pre-cropped single-cell dataset.

    Args:
        root: directory containing one TIFF per cell.
        channel_layout:
            None  : each TIFF is a 2D single-channel image (H, W).
            "CHW" (default) : pages along C axis, stored as (C, H, W).
            "HWC" : last axis is C, stored as (H, W, C).
        image_pattern: optional regex with named groups to parse metadata
            from filenames (e.g. r"C(?P<channel>\\d+)_f(?P<field>\\d+)_...").
            Captures become metadata columns.
    """

    def __init__(
        self,
        root,
        channel_layout="CHW",
        image_pattern=None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise DatasetError(f"CellDataset root not found: {self.root}")
        if channel_layout not in (None, "CHW", "HWC"):
            raise ConfigError(
                f"channel_layout must be None, 'CHW' or 'HWC', got '{channel_layout}'"
            )
        self.channel_layout = channel_layout
        self._image_pattern = (
            _io.compile_pattern(image_pattern, "image_pattern") if image_pattern else None
        )

        # State filled by build_metadata
        self._metadata = None
        self._intensity_colnames = []
        self._img_shape = None
        self._img_dtype = None
        self._schema = None

        self.build_metadata()

    @property
    def metadata(self):
        return self._metadata

    @property
    def intensity_colnames(self):
        return list(self._intensity_colnames)

    @property
    def img_shape(self):
        return self._img_shape

    @property
    def img_dtype(self):
        return self._img_dtype

    @property
    def image_pattern(self):
        """The original image_pattern string (None if never set).

        Mirrors ImageDataset.image_pattern — the compiled regex stays private.
        """
        if self._image_pattern is None:
            return None
        return self._image_pattern.pattern

    @property
    def schema(self):
        return self._schema

    def __len__(self):
        return 0 if self._metadata is None else len(self._metadata)

    def __repr__(self):
        return (
            f"CellDataset(root={self.root}, cells={len(self)}, "
            f"channels={self._intensity_colnames})"
        )

    def build_metadata(self):
        """Walk root, collect one record per TIFF. Parse metadata from filename if pattern set."""
        records = []
        for p in natsorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".tif", ".tiff"):
                continue
            rec = {
                "stem": p.stem,
                "path": str(p),
                "directory": str(p.parent.relative_to(self.root)),
            }
            if self._image_pattern is not None:
                m = self._image_pattern.match(p.name)
                if not m:
                    # Skip files that don't match the pattern
                    continue
                gd = m.groupdict()
                for k, v in gd.items():
                    if k == "channel":
                        # Single-cell files shouldn't have a channel group
                        # (channels are inside the file). Skip if present.
                        continue
                    if k in ("stem", "path", "directory"):
                        # Reserved record keys — a capture with one of these
                        # names would clobber the real file path/stem and
                        # break get_cell with a confusing "file not found".
                        continue
                    # Captures are used verbatim — metadata stays TEXT.
                    rec[k] = v
            records.append(rec)

        if not records:
            self._metadata = pd.DataFrame()
            self._schema = MetadataSchema.infer(set())
            return self

        df = pd.DataFrame(records)

        # Schema + well merging (no numeric coercion — all TEXT)
        captured = set(df.columns) - {"stem", "path", "directory"}
        self._schema = MetadataSchema.infer(captured)
        df = self._schema.apply_well_merge(df)

        df = df.reset_index(drop=True)
        self._metadata = df

        # Detect channel count from first TIFF
        self._auto_detect_image_properties()
        return self

    def _auto_detect_image_properties(self):
        """Read the first READABLE cell TIFF to get shape, dtype, and channels.

        Unreadable candidates are skipped (warning) instead of raising — the
        same policy as ImageDataset: a broken first file must not abort
        dataset construction, the broken row is quarantined when it is
        reached. Only when every candidate fails do the properties stay
        unset.
        """
        if len(self._metadata) == 0:
            return
        for _, row in self._metadata.iterrows():
            try:
                self._img_shape, n_channels, self._img_dtype = \
                    _io.detect_tiff_properties(row["path"], self.channel_layout)
            except ImageReadError as e:
                logger.warning(
                    "Skipping unreadable cell TIFF for shape detection: %s", e)
                continue
            self._intensity_colnames = [f"ch{i}" for i in range(1, n_channels + 1)]
            return

    def get_cell(self, idx):
        """Read one cell TIFF.

        Args:
            idx: integer index into metadata.

        Returns:
            (H, W, C) array.
        """
        if idx < 0 or idx >= len(self._metadata):
            raise DataError(f"cell index {idx} out of range (0..{len(self)-1})")
        path = self._metadata.iloc[idx]["path"]
        n_channels = len(self._intensity_colnames)
        channels_list = list(range(1, n_channels + 1))
        return _io.read_tiff_channels(
            path, channels_list, channel_layout=self.channel_layout
        )
