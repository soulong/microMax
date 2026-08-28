"""ImageDataset: whole-image dataset loader.

Two file layouts supported:
- channel_layout=None  : one-channel-per-file (with a `channel` group, each
  file is one channel; without one, every file is the implicit channel `ch1`)
- channel_layout="CHW" : multi-channel-per-file (regex no `channel` group)
- channel_layout="HWC" : multi-channel-per-file (regex no `channel` group)

Provides:
- metadata DataFrame (regex-pivoted, one row per site, channels/masks as cols)
- get_imageset(row_idx, masks=None) -> (img_HWC, mask_dict)
- get_cropped_cell(row_idx, label, mask_name, padding=4)
    -> (crop_HWC, cell_mask_HW, bbox_xywh)
- get_cropped_cells(row_idx, mask_name, padding=4)
    -> list of (crop_HWC, cell_mask_HW, bbox_xywh, label)

LRU-caches raw (image, mask_dict) per row_idx so repeated crop calls don't
re-read from disk.
"""

import sys
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted, natsort_keygen

from . import cells as _cells
from . import io as _io
from .schema import MetadataSchema

logger = logging.getLogger(__name__)


def _pattern_string(pattern):
    """Convert a stored pattern (compiled regex, str, or None) to a string."""
    if pattern is None:
        return None
    return pattern.pattern if hasattr(pattern, "pattern") else str(pattern)


def _apply_filter(df, col, pat):
    """Filter a metadata frame by regex on a column, hard-exiting on an
    unknown column. Shared by build_metadata and filter_metadata."""
    if col not in df.columns:
        print(
            f"Error: filter column '{col}' not in metadata columns: "
            f"{list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return df[df[col].astype(str).str.contains(pat, regex=True, na=False)]


# Row-sort priority for the metadata DataFrame: row → col → well → field →
# stack → timepoint (well covers explicit-`well` datasets; row+col-derived
# wells are sorted via row/col before derivation). Columns absent from the
# regex are simply skipped — if none of these columns exist, the row order
# is the deterministic sorted-key order of the shared merge keys.
_ROW_SORT_PRIORITY = ["row", "col", "well", "field", "stack", "timepoint"]


class _LRUCache:
    """Tiny LRU cache. OrderedDict.move_to_end on access; popitem(last=False) on overflow.

    Thread-safe: the cache is shared between the GUI thread and background
    workers (microVis dispatches ImageWorker/FullResWorker/ObjectExportWorker
    QRunnables that all call get_imageset), so mutations are guarded by a
    lock (the LRU reorder ops are multi-step and not atomic under the GIL).

    Picklable: the lock is recreated on unpickle and the cached entries are
    dropped on pickle (a cold cache — workers spawned by a DataLoader start
    with an empty cache instead of shipping the parent's cached arrays).
    This makes ImageDataset (and therefore WholeImageCellDataset) spawnable
    by torch DataLoader worker processes on Windows, where multiprocessing
    pickles the dataset into each worker.
    """

    def __init__(self, maxsize=8):
        self.maxsize = maxsize
        self._d = OrderedDict()
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_d"] = OrderedDict()
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._d:
                return None
            self._d.move_to_end(key)
            return self._d[key]

    def put(self, key, value):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
            self._d[key] = value
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self):
        with self._lock:
            self._d.clear()


class ImageDataset:
    """Whole-image dataset.

    Args:
        root: dataset root directory.
        image_pattern: regex with named groups. With channel_layout None, a
            `channel` group makes each file one channel; without one, each
            file is the implicit channel `ch1`. With "CHW"/"HWC", no
            `channel` group is expected.
        mask_pattern: optional regex matching mask files. Must have a
            `mask_name` group plus whatever structural groups image_pattern has.
        channel_layout: None (default, one-channel-per-file), "CHW", or "HWC".
        image_subdir_pattern: optional glob pattern restricting which subdirs
            to scan (e.g. "Images/").
        filters: optional dict of {column: regex_str} to filter rows after build.
    """

    def __init__(
        self,
        root,
        image_pattern,
        mask_pattern=None,
        channel_layout=None,
        image_subdir_pattern=None,
        filters=None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            print(f"Error: dataset root not found: {self.root}", file=sys.stderr)
            sys.exit(1)
        self.channel_layout = channel_layout
        self.image_subdir_pattern = image_subdir_pattern
        self._filters = list(filters.items()) if filters else []

        # Compile patterns
        self._image_pattern = _io.compile_pattern(image_pattern, "image_pattern")
        self._mask_pattern = (
            _io.compile_pattern(mask_pattern, "mask_pattern") if mask_pattern else None
        )

        if self._image_pattern is None:
            print(
                "Error: image_pattern is required for ImageDataset. "
                "Set 'data.image_pattern' in the config.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate pattern vs channel_layout
        has_channel_group = "channel" in self._image_pattern.groupindex
        if channel_layout is None and not has_channel_group:
            # Implicit single-channel mode: each file is treated as channel "ch1".
            # No hard-exit — allows simple single-channel datasets without a
            # `channel` capture group.
            logger.warning(
                "channel_layout=None without a `channel` group in image_pattern — "
                "treating each file as the single channel 'ch1'."
            )
        if channel_layout is not None and has_channel_group:
            print(
                "Error: channel_layout conflicts with channel-grouped pattern. "
                "Either remove the `channel` group from image_pattern or set "
                "channel_layout=None.",
                file=sys.stderr,
            )
            sys.exit(1)

        # State filled by build_metadata
        self._metadata = None
        self._intensity_colnames = []
        self._mask_colnames = []
        self._img_shape = None
        self._img_dtype = None
        self._schema = None
        self._captured_fields = set()

        # LRU cache for raw (image, mask_dict) per row_idx
        self._cache = _LRUCache(maxsize=8)

        self.build_metadata()

    # ---- Public properties ----

    @property
    def metadata(self):
        return self._metadata

    @property
    def intensity_colnames(self):
        return list(self._intensity_colnames)

    @property
    def mask_colnames(self):
        return list(self._mask_colnames)

    @property
    def img_shape(self):
        return self._img_shape

    @property
    def img_dtype(self):
        return self._img_dtype

    @property
    def schema(self):
        return self._schema

    @property
    def captured_fields(self):
        """Set of regex-captured metadata column names (structural + extra)."""
        return set(self._captured_fields)

    @property
    def image_pattern(self):
        """The original image_pattern string (None if never set).

        The compiled regex stays private (``_image_pattern``) — consumers that
        need the string (config round-trips, rebuilding datasets, microModel
        configs) read this public attribute instead of reaching into privates.
        """
        return _pattern_string(self._image_pattern)

    @property
    def mask_pattern(self):
        """The original mask_pattern string (None if never set)."""
        return _pattern_string(self._mask_pattern)

    def __len__(self):
        return 0 if self._metadata is None else len(self._metadata)

    def __repr__(self):
        return (
            f"ImageDataset(root={self.root}, rows={len(self)}, "
            f"channels={self._intensity_colnames}, masks={self._mask_colnames})"
        )

    # ---- Construction helpers ----

    def _iter_image_files(self):
        """Yield (relative_dirpath, filename) for files under image_subdir_pattern.

        No suffix filtering — image_pattern.match(fname) is the authoritative
        filter. If the pattern matches files directly, yield those. If it
        matches a directory (e.g. 'Images/'), recursively scan within it.
        """
        if self.image_subdir_pattern:
            for p in self.root.glob(self.image_subdir_pattern):
                if p.is_file():
                    yield p.parent.relative_to(self.root), p.name
                elif p.is_dir():
                    for sub in p.rglob("*"):
                        if sub.is_file():
                            yield sub.parent.relative_to(self.root), sub.name
        else:
            for p in self.root.rglob("*"):
                if p.is_file():
                    yield p.parent.relative_to(self.root), p.name

    def _iter_mask_files(self):
        if self._mask_pattern is None:
            return
        for p in self.root.rglob("*"):
            if p.is_file():
                yield p.parent.relative_to(self.root), p.name

    def build_metadata(self):
        """Scan root, parse filenames with regex, build pivoted metadata DataFrame."""
        # Group 1: collect parsed records
        # Each image record: {shared_key -> {channel: filepath, ...extra meta}}
        # Mask records similar with mask_name.
        image_records = {}  # shared_key -> {channel: path, ...meta}
        mask_records = {}   # shared_key -> {mask_name: path, ...meta}
        intensity_col_names: set[str] = set()

        for reldir, fname in self._iter_image_files():
            # Skip files that match mask_pattern — they're masks, not images
            if self._mask_pattern is not None and self._mask_pattern.match(fname):
                continue
            m = self._image_pattern.match(fname)
            if not m:
                continue
            gd = m.groupdict()
            shared = self._shared_key(gd)
            rec = image_records.setdefault(shared, {"directory": str(reldir)})
            if self.channel_layout is None:
                # one-channel-per-file: channel from regex group or
                # implicit "ch1" if no channel group.
                # Prefix 'ch' only when the capture starts with a digit
                # (e.g. '01' -> 'ch01', '2' -> 'ch2'); non-numeric captures
                # like 'blue' stay as-is.
                if "channel" in gd:
                    ch_val = gd["channel"]
                    col_name = f"ch{ch_val}" if ch_val and ch_val[0].isdigit() else ch_val
                    rec[col_name] = str(self.root / reldir / fname)
                    intensity_col_names.add(col_name)
                else:
                    rec["ch1"] = str(self.root / reldir / fname)
                    intensity_col_names.add("ch1")
            else:
                # multi-channel-per-file: one file per site
                rec["__file__"] = str(self.root / reldir / fname)
            # Store structural + extra meta from regex captures (verbatim —
            # all metadata stays TEXT from extraction through DB storage).
            for k, v in gd.items():
                if k == "channel":
                    continue
                if k == "mask_name":
                    continue
                rec[k] = v

        if self._mask_pattern is not None:
            for reldir, fname in self._iter_mask_files():
                m = self._mask_pattern.match(fname)
                if not m:
                    continue
                gd = m.groupdict()
                shared = self._shared_key(gd)
                if "mask_name" not in gd:
                    raise ValueError(
                        f"mask_pattern must have a `mask_name` named group. "
                        f"File: {fname}"
                    )
                mname = gd["mask_name"]
                rec = mask_records.setdefault(shared, {"directory": str(reldir)})
                rec[f"mask_{mname}"] = str(self.root / reldir / fname)
                for k, v in gd.items():
                    if k in ("channel", "mask_name"):
                        continue
                    rec.setdefault(k, v)

        # Group 2: merge image + mask records on shared_key
        all_keys = sorted(set(image_records) | set(mask_records))
        if not all_keys:
            self._metadata = pd.DataFrame()
            self._schema = MetadataSchema.infer(set())
            return self

        # Collect all column names seen
        rows = []
        for k in all_keys:
            row = {"directory": None}
            img_rec = image_records.get(k, {})
            mask_rec = mask_records.get(k, {})
            row.update(img_rec)
            # mask_ columns override (since they're prefixed)
            for mk, mv in mask_rec.items():
                row[mk] = mv
            rows.append(row)
        df = pd.DataFrame(rows)

        # Sort rows by structural columns in microscopy-natural priority
        # (row → col → field → stack → timepoint). Only columns that exist
        # are used; absent columns are skipped. Values are natsorted so
        # field=2 comes before field=10 (not lexicographic field=10 < field=2).
        sort_cols = [c for c in _ROW_SORT_PRIORITY if c in df.columns]
        if sort_cols:
            _nk = natsort_keygen()
            df = df.sort_values(
                by=sort_cols,
                key=lambda col: col.astype(str).map(_nk),
                kind="stable",
            ).reset_index(drop=True)

        # Group 3: derive channel/mask column lists
        if self.channel_layout is None:
            ch_cols = natsorted(intensity_col_names)
        else:
            # Single file column; channels are inside the file.
            # We don't know the count until we read one — defer until first read.
            ch_cols = []  # filled by _auto_detect_image_properties
        mask_cols = natsorted([c for c in df.columns if c.startswith("mask_")])

        # Group 4: schema + well merging (no numeric coercion — all TEXT)
        captured = set()
        for c in df.columns:
            if c in ("directory", "__file__"):
                continue
            if c in intensity_col_names:
                continue
            if c.startswith("mask_"):
                continue
            captured.add(c)
        self._schema = MetadataSchema.infer(captured)
        df = self._schema.apply_well_merge(df)

        # Keep __file__ column — it's the per-site file path for
        # multi-channel-per-file mode. Don't expose it via intensity_colnames.

        # Apply filters
        for col, pat in self._filters:
            df = _apply_filter(df, col, pat)

        df = df.reset_index(drop=True)

        self._metadata = df
        self._intensity_colnames = ch_cols
        self._mask_colnames = mask_cols
        self._captured_fields = captured

        # Auto-detect image shape/dtype + channel count for multi-channel-per-file
        self._auto_detect_image_properties()

        return self

    def _auto_detect_image_properties(self):
        """Read the first image to get shape, dtype, and (for multi-channel-per-file) channels."""
        if len(self._metadata) == 0:
            return
        row = self._metadata.iloc[0]
        if self.channel_layout is None:
            # one-channel-per-file: use intensity_colnames set by build_metadata
            ch_cols = self._intensity_colnames
            if not ch_cols:
                return
            first_path = row[ch_cols[0]]
            arr = _io.read_tiff(first_path)
            self._img_shape = arr.shape  # (H, W)
            self._img_dtype = arr.dtype
        else:
            # multi-channel-per-file: open first TIFF, count channels
            if "__file__" not in row:
                return
            self._img_shape, n_channels, self._img_dtype = _io.detect_tiff_properties(
                row["__file__"], self.channel_layout)
            self._intensity_colnames = [f"ch{i}" for i in range(1, n_channels + 1)]

    def _shared_key(self, groupdict):
        """Build a hashable shared key from regex captures, ignoring channel/mask_name."""
        parts = []
        for k in sorted(groupdict.keys()):
            if k in ("channel", "mask_name"):
                continue
            parts.append(f"{k}={groupdict[k]}")
        return "|".join(parts)

    # ---- Image access ----

    def _get_row_filepaths(self, row_idx):
        """Return (image_paths_dict, mask_paths_dict) for a metadata row.

        image_paths_dict:
          - channel_layout=None: {ch_name: path}
          - channel_layout="CHW"/"HWC": {"__file__": path}  (one file)
        """
        if row_idx < 0 or row_idx >= len(self._metadata):
            print(f"Error: row_idx {row_idx} out of range (0..{len(self)-1})", file=sys.stderr)
            sys.exit(1)
        row = self._metadata.iloc[row_idx]
        img_paths = {}
        mask_paths = {}
        if self.channel_layout is None:
            for ch in self._intensity_colnames:
                p = row[ch]
                if pd.isna(p):
                    # Mask-only rows (or a site missing a channel file) have
                    # NaN image paths — reading one would crash with a raw
                    # TypeError. Hard-exit with a clear message instead.
                    print(
                        f"Error: row {row_idx} is missing the image file for "
                        f"channel '{ch}' — the file referenced by the metadata "
                        f"does not exist (mask-only or deleted file).",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                img_paths[ch] = p
        else:
            # multi-channel-per-file: the file path is stored under "__file__"
            # (kept as a private column, never re-parsed).
            if "__file__" not in row:
                print(
                    "Error: multi-channel-per-file mode requires __file__ column. "
                    "Rebuild metadata.",
                    file=sys.stderr,
                )
                sys.exit(1)
            img_paths["__file__"] = row["__file__"]
        for m in self._mask_colnames:
            mpath = row[m]
            # Skip rows where the mask file is missing (e.g. Cellpose found
            # no cells → no mask saved → mask column is NaN). Without this,
            # read_mask(NaN) would crash with TypeError.
            if pd.isna(mpath):
                continue
            mask_paths[m] = mpath
        return img_paths, mask_paths

    def get_imageset(self, row_idx, masks=None):
        """Load the multi-channel image and optional masks for a metadata row.

        Args:
            row_idx: integer row index into metadata.
            masks: optional list of mask column names to load. None = load all.

        Returns:
            (img_data, mask_dict) where:
              img_data  : (H, W, C) array
              mask_dict : {mask_col_name: (H, W) int array}
        """
        cached = self._cache.get(row_idx)
        if cached is not None:
            img_data, all_masks = cached
            if masks is None:
                return img_data, dict(all_masks)
            return img_data, {m: all_masks[m] for m in masks if m in all_masks}

        img_paths, mask_paths = self._get_row_filepaths(row_idx)

        if self.channel_layout is None:
            # one-channel-per-file: read each, stack
            arrays = []
            for ch in self._intensity_colnames:
                arrays.append(_io.read_tiff(img_paths[ch]))
            img_data = np.stack(arrays, axis=-1)  # (H, W, C)
        else:
            # multi-channel-per-file: read all pages
            n_channels = len(self._intensity_colnames)
            channels_list = list(range(1, n_channels + 1))
            img_data = _io.read_tiff_channels(
                img_paths["__file__"], channels_list, channel_layout=self.channel_layout
            )

        # Always load ALL masks so the cache holds the complete set;
        # returning only the requested subset to the caller.
        mask_dict = {}
        for mname, mpath in mask_paths.items():
            mask_dict[mname] = _io.read_mask(mpath)

        self._cache.put(row_idx, (img_data, mask_dict))
        if masks is None:
            return img_data, dict(mask_dict)
        return img_data, {m: mask_dict[m] for m in masks if m in mask_dict}

    def image_path(self, row_idx, channel):
        """Return the file path for a specific channel at a row. channel_layout=None only."""
        if self.channel_layout is not None:
            print(
                "Error: image_path() only valid for channel_layout=None "
                "(one-channel-per-file mode).",
                file=sys.stderr,
            )
            sys.exit(1)
        if row_idx < 0 or row_idx >= len(self._metadata):
            print(
                f"Error: row_idx {row_idx} out of range (0..{len(self)-1})",
                file=sys.stderr,
            )
            sys.exit(1)
        row = self._metadata.iloc[row_idx]
        p = row[channel]
        if pd.isna(p):
            print(
                f"Error: row {row_idx} is missing the image file for "
                f"channel '{channel}' — the file referenced by the metadata "
                f"does not exist (mask-only or deleted file).",
                file=sys.stderr,
            )
            sys.exit(1)
        return Path(p)

    def filter_metadata(self, column, pattern):
        """Filter rows by regex on a column. Mutates metadata."""
        self._metadata = _apply_filter(self._metadata, column, pattern).reset_index(drop=True)
        self._filters.append((column, pattern))
        self._cache.clear()

    # ---- Single-cell cropping ----

    def get_cropped_cell(self, row_idx, label, mask_name, padding=4):
        """Crop a single cell from a whole image.

        Args:
            row_idx: metadata row index.
            label: integer mask label of the cell.
            mask_name: mask column name (e.g. "mask_cell").
            padding: pixels of padding around bbox.

        Returns:
            (crop, cell_mask, bbox) where crop is (h, w, C) with background zeroed,
            cell_mask is (h, w) bool, bbox is (x, y, w, h).
            Exits with error if the cell has no pixels in the mask.
        """
        img_data, mask_dict = self.get_imageset(row_idx, masks=[mask_name])
        if mask_name not in mask_dict:
            print(
                f"Error: mask '{mask_name}' not found. Available: {list(mask_dict)}",
                file=sys.stderr,
            )
            sys.exit(1)
        mask = mask_dict[mask_name]
        return _cells.crop_cell(img_data, mask, label, padding=padding)

    def get_cropped_cells(self, row_idx, mask_name, padding=4, labels=None):
        """Crop every cell (or a subset) from a whole image.

        Args:
            row_idx: metadata row index.
            mask_name: mask column name.
            padding: pixels of padding around bbox.
            labels: optional iterable of labels to crop. None = crop all.

        Returns:
            list of (crop, cell_mask, bbox, label) tuples. Empty cells skipped.
        """
        img_data, mask_dict = self.get_imageset(row_idx, masks=[mask_name])
        if mask_name not in mask_dict:
            print(
                f"Error: mask '{mask_name}' not found. Available: {list(mask_dict)}",
                file=sys.stderr,
            )
            sys.exit(1)
        mask = mask_dict[mask_name]
        return _cells.crop_all_cells(
            img_data, mask, padding=padding, labels=labels
        )
