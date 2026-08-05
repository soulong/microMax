"""CellDataset: pre-cropped single-cell dataset loader.

Each TIFF file in `root` is one cell, with channels multiplexed inside
(as pages for CHW, or as last axis for HWC).

No masks. No whole-image cropping. Just reads pre-cropped cells.

Optional `image_pattern` parses each filename to extract metadata columns
(well, field, stack, timepoint, label, etc.) — useful for microModel's
infer single_cell mode where per-cell metadata is written to the DB.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted

from . import io as _io
from .schema import MetadataSchema, normalize_capture


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
            print(f"Error: CellDataset root not found: {self.root}", file=sys.stderr)
            sys.exit(1)
        if channel_layout not in (None, "CHW", "HWC"):
            print(
                f"Error: channel_layout must be None, 'CHW' or 'HWC', got '{channel_layout}'",
                file=sys.stderr,
            )
            sys.exit(1)
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
                    rec[k] = normalize_capture(k, v)
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
        """Read the first cell TIFF to get shape, dtype, and channel count."""
        if len(self._metadata) == 0:
            return
        path = self._metadata.iloc[0]["path"]
        self._img_shape, self._img_dtype, n_channels = _io.detect_tiff_properties(
            path, self.channel_layout)
        self._intensity_colnames = [f"ch{i}" for i in range(1, n_channels + 1)]

    def get_cell(self, idx):
        """Read one cell TIFF. Returns (img_HWC, ) tuple for forward-compat.

        Args:
            idx: integer index into metadata.

        Returns:
            (H, W, C) array.
        """
        if idx < 0 or idx >= len(self._metadata):
            print(
                f"Error: cell index {idx} out of range (0..{len(self)-1})",
                file=sys.stderr,
            )
            sys.exit(1)
        path = self._metadata.iloc[idx]["path"]
        n_channels = len(self._intensity_colnames)
        channels_list = list(range(1, n_channels + 1))
        return _io.read_tiff_channels(
            path, channels_list, channel_layout=self.channel_layout
        )
