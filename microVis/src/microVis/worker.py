"""Background image processing worker for microVis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QRunnable, Signal
from skimage.transform import resize as sk_resize

from microVis.io.data_module import _safe_str
from microVis.log_utils import get_logger

logger = get_logger("microVis.worker")


def _downscale_image(img_data: np.ndarray, thumb_size: int) -> np.ndarray:
    """Downscale (H, W, C) array to fit within thumb_size."""
    h, w = img_data.shape[:2]
    if h <= thumb_size and w <= thumb_size:
        return img_data
    scale = thumb_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return sk_resize(
        img_data, (new_h, new_w, img_data.shape[2]),
        preserve_range=True, anti_aliasing=True,
    ).astype(img_data.dtype)


def _downscale_mask(mask_dict: dict, thumb_size: int, selected_name: str = "") -> tuple:
    """Downscale the selected mask from dict.

    ``selected_name`` is the mask name WITHOUT the ``mask_`` prefix
    (e.g. "cell"). If empty or not found in ``mask_dict``, returns
    ``(None, None)`` — no overlay is drawn for that image rather than
    silently substituting another mask. Returns ``(mask_name, small_mask)``.
    """
    if not mask_dict or not selected_name:
        return None, None
    key = f"mask_{selected_name}"
    if key not in mask_dict:
        return None, None
    name = key
    mask = mask_dict[name]
    if mask is None:
        return None, None
    h, w = mask.shape
    if h <= thumb_size and w <= thumb_size:
        return name, mask
    scale = thumb_size / max(h, w)
    small = sk_resize(
        mask, (int(h * scale), int(w * scale)),
        order=0, preserve_range=True, anti_aliasing=False,
    ).astype(mask.dtype)
    return name, small


def _enhance_channels(
    img_data: np.ndarray,
    channel_names: list[str],
    ch_config: dict,
    dmax: float,
    contrast_method: str,
    contrast_gamma: float,
    invert: bool,
) -> np.ndarray:
    """Apply per-channel contrast enhancement to an (H, W, C) image.

    Returns a float64 array in [0, 1].
    """
    from microVis.processing.contrast import apply_contrast, invert_image

    enhanced = np.zeros_like(img_data, dtype=np.float64)
    n_channels = img_data.shape[2]
    for ch_idx, ch_name in enumerate(channel_names):
        if ch_idx >= n_channels:
            break
        ch_cfg = ch_config.get(ch_name, {})
        if not ch_cfg.get("enabled", True):
            enhanced[:, :, ch_idx] = 0.0
            continue
        vmin = ch_cfg.get("vmin", 0)
        vmax = ch_cfg.get("vmax", dmax)
        band = img_data[:, :, ch_idx].astype(np.float64)
        band = np.clip((band - vmin) / max(vmax - vmin, 1e-10), 0, 1)
        if contrast_method == "gamma":
            band = apply_contrast(band, "gamma", gamma=contrast_gamma)
        elif contrast_method == "histogram_equalization":
            band = apply_contrast(band, "histogram_equalization")
        if invert:
            band = invert_image(band)
        enhanced[:, :, ch_idx] = band
    return enhanced


@dataclass
class ImageWorkerConfig:
    """Configuration for a single image processing worker."""

    row_idx: int
    well: str
    field: int
    stack: int
    timepoint: int
    raw_data: tuple | None
    thumb_size: int
    channel_names: list[str]
    ch_config: dict
    dmax: float
    contrast_method: str
    contrast_gamma: float
    invert: bool
    need_polygons: bool
    dm: object | None = None
    overlay_val: float | str | None = None
    overlay_col: str | None = None
    n_objects: int | None = None
    obj_values: dict = field(default_factory=dict)
    overlay_vmin: float = 0.0
    overlay_vmax: float = 1.0
    gen: int = 0
    sort_by_row: bool = False
    mask_cache: np.ndarray | None = None
    polygons_cache: list | None = None
    need_mask: bool = False
    meta_label: str = ""
    selected_mask_name: str = ""       # "" = no mask, else mask name (e.g. "cell")
    outline_only: bool = False


class _WorkerSignals(QObject):
    finished = Signal(dict)
    error = Signal(str)


class ImageWorker(QRunnable):
    """Process a single image at thumbnail resolution in a background thread."""

    def __init__(self, config: ImageWorkerConfig):
        super().__init__()
        self.signals = _WorkerSignals()
        self._cfg = config
        self.setAutoDelete(False)

    @property
    def gen(self) -> int:
        return self._cfg.gen

    def run(self) -> None:
        try:
            from microVis.processing.compositing import composite_image
            from microVis.processing.overlay import extract_polygons

            cfg = self._cfg

            # Load from disk if not provided (runs on worker thread, not main)
            loaded_from_disk = False
            if cfg.raw_data is not None:
                img_data, mask_dict = cfg.raw_data
            else:
                img_data, mask_dict = cfg.dm.get_imageset(cfg.row_idx)
                loaded_from_disk = True

            # Downscale to thumbnail resolution
            img_small = _downscale_image(img_data, cfg.thumb_size)
            if cfg.mask_cache is not None:
                mask_small = cfg.mask_cache
                mask_name = None
            elif cfg.selected_mask_name or cfg.need_mask or cfg.need_polygons:
                mask_name, mask_small = _downscale_mask(
                    mask_dict, cfg.thumb_size, cfg.selected_mask_name,
                )
            else:
                mask_small = None
                mask_name = None

            # Per-channel contrast enhancement
            enhanced = _enhance_channels(
                img_small, cfg.channel_names, cfg.ch_config, cfg.dmax,
                cfg.contrast_method, cfg.contrast_gamma, cfg.invert,
            )

            # Composite
            comp_config = {ch: {**c, "vmin": 0, "vmax": 1}
                           for ch, c in cfg.ch_config.items()}
            rgb = composite_image(enhanced, cfg.channel_names, comp_config, None, None)

            # Polygons from downscaled mask — use cache when available
            polygons = None
            if cfg.polygons_cache is not None:
                polygons = cfg.polygons_cache
            elif cfg.need_polygons and mask_small is not None:
                polygons = extract_polygons(mask_small)

            # Mask for thumbnail drag interaction — include when need_mask or need_polygons
            result_mask = mask_small if (cfg.need_mask or cfg.need_polygons) else None

            result = {
                "rgb": np.ascontiguousarray(rgb),
                "well": cfg.well, "field": cfg.field,
                "stack": cfg.stack, "timepoint": cfg.timepoint,
                "polygons": polygons,
                "overlay_val": cfg.overlay_val,
                "overlay_col": cfg.overlay_col,
                "n_objects": cfg.n_objects,
                "mask": result_mask,
                "obj_values": cfg.obj_values,
                "overlay_vmin": cfg.overlay_vmin,
                "overlay_vmax": cfg.overlay_vmax,
                "gen": cfg.gen,
                "thumb_size": cfg.thumb_size,
                "sort_by_row": cfg.sort_by_row,
                "row_idx": cfg.row_idx,
                "meta_label": cfg.meta_label,
                "outline_only": cfg.outline_only,
            }
            if loaded_from_disk:
                result["raw_data"] = (img_data, mask_dict)
            self.signals.finished.emit(result)
        except SystemExit as e:
            # microBase hard-exits (sys.exit) when a file/mask vanished or is
            # out of range — convert to a skip (error signal; the gen-guarded
            # _on_worker_error handler just decrements pending + logs, so the
            # thumbnail simply doesn't render).
            logger.warning("Image worker skipped row %d: %s", cfg.row_idx, e)
            self.signals.error.emit(f"Row {cfg.row_idx} skipped (missing data): {e}")
        except Exception as e:
            self.signals.error.emit(str(e))


class _FullResSignals(QObject):
    # payload (dict with rgb + polygons), row_idx, gen, mask, obj_values,
    # polygons, overlay_col, overlay_vmin, overlay_vmax
    finished = Signal(object, int, int, object, object, object, str, float, float)
    error = Signal(str)


class FullResWorker(QRunnable):
    """Load and composite a single image at full resolution."""

    def __init__(self, dm, row_idx: int,
                 channel_names: list[str], ch_config: dict, dmax: float,
                 contrast_method: str, contrast_gamma: float, invert: bool,
                 gen: int = 0, overlay_alpha: float = 0.4,
                 need_polygons: bool = True, obj_values: dict | None = None,
                 overlay_col: str | None = None,
                 overlay_vmin: float = 0.0, overlay_vmax: float = 1.0,
                 selected_mask_name: str = "", outline_only: bool = False):
        super().__init__()
        self.signals = _FullResSignals()
        self.setAutoDelete(False)
        self._dm = dm
        self._row_idx = row_idx
        self._ch_names = channel_names
        self._ch_config = ch_config
        self._dmax = dmax
        self._contrast = contrast_method
        self._gamma = contrast_gamma
        self._invert = invert
        self._gen = gen
        self._overlay_alpha = overlay_alpha
        self._need_polygons = need_polygons
        self._obj_values = obj_values or {}
        self._overlay_col = overlay_col
        self._overlay_vmin = overlay_vmin
        self._overlay_vmax = overlay_vmax
        self._selected_mask_name = selected_mask_name
        self._outline_only = outline_only

    def run(self) -> None:
        try:
            from microVis.processing.compositing import composite_image

            img_data, mask_dict = self._dm.get_imageset(self._row_idx)

            enhanced = _enhance_channels(
                img_data, self._ch_names, self._ch_config, self._dmax,
                self._contrast, self._gamma, self._invert,
            )

            comp_config = {ch: {**c, "vmin": 0, "vmax": 1}
                           for ch, c in self._ch_config.items()}
            rgb = composite_image(enhanced, self._ch_names, comp_config, None, None)
            rgb = np.ascontiguousarray(rgb)

            # Extract the selected mask for interactivity (hover/drag) + overlay.
            # If no mask selected, or the selected mask is absent for this image,
            # draw nothing (do not silently substitute another mask).
            full_mask = None
            polygons = None
            if self._need_polygons and self._selected_mask_name and mask_dict:
                full_mask = mask_dict.get(f"mask_{self._selected_mask_name}")
                if full_mask is not None:
                    from microVis.processing.overlay import extract_polygons
                    polygons = extract_polygons(full_mask)

            # QPixmap must be created on the GUI thread — the worker returns
            # the RGB array + polygons; _apply_full_res_pixmap builds the
            # pixmap and draws the overlays there (see CropWorker).
            self.signals.finished.emit(
                {"rgb": rgb, "polygons": polygons},
                self._row_idx, self._gen, full_mask, self._obj_values,
                polygons,
                self._overlay_col or "",
                self._overlay_vmin, self._overlay_vmax)
        except SystemExit as e:
            logger.warning("Full-res worker skipped row %d: %s", self._row_idx, e)
            self.signals.error.emit(f"Row {self._row_idx} skipped (missing data): {e}")
        except Exception as e:
            self.signals.error.emit(str(e))


# ── Crop Worker ───────────────────────────────────────────────────────────────


class _CropSignals(QObject):
    finished = Signal(object, object)  # np.ndarray (RGB uint8), ObjectKey
    error = Signal(str)


class CropWorker(QRunnable):
    """Background worker that crops and masks a single object from an image."""

    def __init__(
        self,
        img_data: np.ndarray,
        mask: np.ndarray,
        label: int,
        key: Any,
        channel_names: list[str],
        ch_config: dict,
        dmax: float,
        contrast_method: str = "none",
        contrast_gamma: float = 1.0,
        invert: bool = False,
        target_size: int = 64,
        padding: int = 4,
    ):
        super().__init__()
        self.signals = _CropSignals()
        self.setAutoDelete(False)
        self._img = img_data
        self._mask = mask
        self._label = label
        self._key = key
        self._ch_names = channel_names
        self._ch_config = ch_config
        self._dmax = dmax
        self._contrast = contrast_method
        self._gamma = contrast_gamma
        self._invert = invert
        self._target = target_size
        self._pad = padding

    def run(self) -> None:
        try:
            from microVis.processing.compositing import composite_image

            mask = self._mask
            label = self._label

            # Find bounding box of the object
            ys, xs = np.where(mask == label)
            if len(ys) == 0:
                self.signals.finished.emit(None, self._key)
                return

            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())

            # Add padding
            h, w = mask.shape
            y_min = max(0, y_min - self._pad)
            y_max = min(h, y_max + self._pad + 1)
            x_min = max(0, x_min - self._pad)
            x_max = min(w, x_max + self._pad + 1)

            # Crop image and mask
            crop_img = self._img[y_min:y_max, x_min:x_max, :].astype(np.float64)
            crop_mask = mask[y_min:y_max, x_min:x_max]

            # Apply mask: zero out pixels not belonging to this object
            obj_mask = (crop_mask == label).astype(np.float64)
            for ch in range(crop_img.shape[2]):
                crop_img[:, :, ch] *= obj_mask

            # Per-channel contrast
            enhanced = _enhance_channels(
                crop_img, self._ch_names, self._ch_config, self._dmax,
                self._contrast, self._gamma, self._invert,
            )

            # Composite
            comp_config = {
                ch: {**c, "vmin": 0, "vmax": 1}
                for ch, c in self._ch_config.items()
            }
            rgb = composite_image(enhanced, self._ch_names, comp_config, None, None)

            # Resize to target
            ch, cw = rgb.shape[:2]
            if ch > self._target or cw > self._target:
                from skimage.transform import resize as sk_resize
                scale = self._target / max(ch, cw)
                rgb = sk_resize(
                    rgb,
                    (int(ch * scale), int(cw * scale), 3),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(np.uint8)

            # Return as numpy array (QPixmap must be created on main thread)
            rgb = np.ascontiguousarray(rgb)

            self.signals.finished.emit(rgb, self._key)
        except SystemExit as e:
            logger.warning("Crop worker skipped %s: %s", self._key, e)
            self.signals.error.emit(f"Crop skipped (missing data): {e}")
        except Exception as e:
            self.signals.error.emit(str(e))


# -- Dataset Load Worker --


class _DatasetLoadWorker(QObject):
    """Constructs a DataModule off the UI thread.

    DataModule.__init__ scans files and builds a row index — synchronous and
    slow for large datasets. Moving it to a background thread keeps the Qt
    event loop alive so the modal progress dialog stays responsive.
    """

    finished = Signal(object)  # DataModule instance
    error = Signal(str)

    def __init__(self, path: str, image_pattern: str,
                 mask_pattern: str, image_subdir_pattern: str):
        super().__init__()
        self._path = path
        self._image_pattern = image_pattern
        self._mask_pattern = mask_pattern
        self._image_subdir_pattern = image_subdir_pattern

    def run(self) -> None:
        try:
            from microVis.io.data_module import DataModule
            dm = DataModule(
                self._path,
                image_pattern=self._image_pattern,
                mask_pattern=self._mask_pattern,
                image_subdir_pattern=self._image_subdir_pattern,
            )
            self.finished.emit(dm)
        except Exception as e:
            logger.exception("Dataset load failed")
            self.error.emit(str(e))


# -- Object Export Worker --


class _ExportSignals(QObject):
    progress = Signal(int, int)  # (current, total)
    finished = Signal(dict)  # {"count": N, "save_dir": path, "csv_path": path}
    error = Signal(str)


# CSV base column order — kept as a module constant so the writer and any
# future reader stay in sync. The CSV file is named "<mask_name>.csv" and
# rows are appended (not overwritten) on repeated exports of the same mask.
# Extra metadata columns (any regex-captured group that is neither structural
# nor a path/mask column — e.g. `stain`, `batch`) are appended after the
# base columns, sourced from `dataset.schema.extra_cols`.
_CSV_BASE_COLUMNS = [
    "filename", "well", "field", "stack", "timepoint",
    "label", "mask_name", "class_name", "channels",
]


class ObjectExportWorker(QRunnable):
    """Background worker that exports cropped/masked objects to disk.

    Uses ``microBase.ImageDataset.get_cropped_cells`` for cell extraction
    (the dataset is reached via the wrapped ``DataModule``) and writes a
    per-export ``{mask_name}.csv`` manifest alongside the TIFFs. The CSV
    is appended (not overwritten) on repeated exports of the same mask.
    """

    def __init__(
        self,
        dm: Any,
        wells: list[str],
        fields: list[int],
        stacks: list[int],
        timepoints: list[int],
        mask_name: str,
        channel_names: list[str],
        save_dir: str,
        object_mode: str,
        ch_colors: dict[str, tuple] | None = None,
        annotations: dict | None = None,
        annotated_keys: set | None = None,
        gen: int = 0,
        max_objects_per_image: int = 0,
        well_subdir: bool = False,
        extra_filters: dict[str, list[str]] | None = None,
    ):
        super().__init__()
        self.signals = _ExportSignals()
        self.setAutoDelete(False)
        self._dm = dm
        self._wells = wells
        self._fields = fields
        self._stacks = stacks
        self._timepoints = timepoints
        self._mask_name = mask_name
        self._channel_names = channel_names
        self._ch_colors = ch_colors or {}
        self._save_dir = save_dir
        self._object_mode = object_mode
        self._max_obj = max_objects_per_image
        self._annotations = annotations
        self._annotated_keys = annotated_keys
        self._gen = gen
        self._well_subdir = well_subdir
        self._extra_filters = extra_filters or {}

    def run(self) -> None:
        try:
            import csv
            from pathlib import Path

            import tifffile

            save_path = Path(self._save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            csv_path = save_path / f"{self._mask_name}.csv"

            rows = self._dm.lookup_row_indices(
                self._wells, self._fields, self._stacks, self._timepoints
            )
            # Apply extra-col filters post-hoc (only non-empty lists filter;
            # empty list for a col means "no filter" on that col).
            if self._extra_filters and rows:
                row_indices = [r[0] for r in rows]
                filtered = set(self._dm.filter_rows_by_extra(row_indices, self._extra_filters))
                rows = [r for r in rows if r[0] in filtered]
            # Filter to only annotated images when applicable.
            # annotated_keys is a set of row_idx values (int) — matches the
            # refactored ObjectKey which is (row_idx, label), so we only need
            # row_idx to scope which images to iterate.
            if self._annotated_keys is not None:
                rows = [r for r in rows if r[0] in self._annotated_keys]
            if not rows:
                self.signals.finished.emit({
                    "count": 0, "save_dir": self._save_dir, "csv_path": str(csv_path),
                    "gen": self._gen,
                })
                return

            # Build annotation lookup if needed. Keyed by (row_idx, label) so
            # it works for non-standard datasets lacking well/field/stack/timepoint.
            key_to_class: dict[tuple, str] = {}
            if self._annotations and self._object_mode == "All annotated":
                for cls_name, keys in self._annotations.items():
                    for key in keys:
                        lookup = (key.row_idx, key.label)
                        key_to_class[lookup] = cls_name

            total = len(rows)
            exported_count = 0
            skipped_rows = 0
            csv_records: list[dict] = []
            mask_col = f"mask_{self._mask_name}"

            if not self._channel_names:
                self.signals.error.emit(
                    "No channels enabled for export — enable at least one "
                    "channel in the Channel Controls panel.")
                return

            dataset = self._dm.dataset
            all_channels = list(dataset.intensity_colnames)
            # Indices of the requested (enabled) channels within the full stack,
            # so we can subset the crop to what the user actually wants exported.
            try:
                ch_indices = [all_channels.index(ch) for ch in self._channel_names]
            except ValueError:
                ch_indices = list(range(len(self._channel_names)))

            img_shape = dataset.img_shape
            if img_shape is not None and len(img_shape) >= 2:
                img_h, img_w = int(img_shape[0]), int(img_shape[1])
            else:
                img_h = img_w = 0

            # Build dynamic CSV column list: base columns + extra metadata
            # columns captured by the regex (e.g. `stain`, `batch`). Sourced
            # from `dataset.schema.extra_cols`; values are read per-row from
            # `dataset.metadata` so non-standard datasets propagate their
            # custom groups into the manifest.
            schema = getattr(dataset, "schema", None)
            extra_cols: list[str] = list(getattr(schema, "extra_cols", []) or [])
            # Defensive: drop any extra col that collides with a base column
            # (shouldn't happen since schema excludes structural cols, but
            # avoids a DictWriter collision if a user names a group `label`).
            extra_cols = [c for c in extra_cols if c not in _CSV_BASE_COLUMNS]
            csv_columns = list(_CSV_BASE_COLUMNS) + extra_cols
            meta_df = dataset.metadata

            logger.info("Export starting: %d images, mask=%s, mode=%s (full frame %dx%d)",
                        total, mask_col, self._object_mode, img_w, img_h)

            for img_idx, (row_idx, well, field, stack, timepoint) in enumerate(rows):
                try:
                    # A row whose mask is missing (NaN path, e.g. Cellpose
                    # found zero cells) makes microBase hard-exit (sys.exit).
                    # Intentional skip+warn design: the export continues with
                    # the remaining rows and reports the skipped count.
                    try:
                        cells = dataset.get_cropped_cells(row_idx, mask_col, padding=4)
                    except SystemExit as e:
                        logger.warning(
                            "Row %d (%s_f%s_z%s_t%s): mask missing — skipped (%s)",
                            row_idx, well, field, stack, timepoint, e)
                        skipped_rows += 1
                        continue
                    if not cells:
                        continue
                    logger.debug("Row %d (%s_f%s_z%s_t%s): %d cells",
                                 row_idx, well, field, stack, timepoint, len(cells))

                    # Filter by annotation class for "All annotated" mode.
                    # Lookup key is (row_idx, label) — independent of which
                    # structural metadata columns exist.
                    if self._object_mode == "All annotated":
                        cells = [
                            (crop, cell_mask, bbox, label)
                            for (crop, cell_mask, bbox, label) in cells
                            if (row_idx, int(label)) in key_to_class
                        ]
                        if not cells:
                            continue

                    # Random sample if max objects is set
                    if self._max_obj > 0 and len(cells) > self._max_obj:
                        rng = np.random.default_rng()
                        picked = rng.choice(len(cells), size=self._max_obj, replace=False)
                        cells = [cells[i] for i in picked]

                    # Build export filename stem from the mask file being used
                    # for cropping — mask filenames are unique per image group,
                    # so this works for both standard plate datasets (e.g.
                    # r1c1f1p1-ch1sk1_cp_masks_cell) and non-standard ones
                    # where well/field/stack/timepoint are absent (e.g. HPA's
                    # 000a6c98-..._mask_cell).
                    mask_path_str = _safe_str(meta_df, row_idx, mask_col)
                    img_stem = Path(mask_path_str).stem if mask_path_str else f"row{row_idx}"

                    for crop, cell_mask, bbox, label in cells:
                        label_int = int(label)
                        lookup = (row_idx, label_int)

                        # Determine class name (only for "All annotated")
                        class_name = ""
                        if self._object_mode == "All annotated" and lookup in key_to_class:
                            class_name = key_to_class[lookup]

                        # Determine save directory (class subfolder for annotated, well subdir)
                        obj_save_dir = save_path
                        if self._object_mode == "All annotated" and class_name:
                            obj_save_dir = obj_save_dir / class_name
                        if self._well_subdir:
                            obj_save_dir = obj_save_dir / well
                        obj_save_dir.mkdir(parents=True, exist_ok=True)

                        # Subset to the enabled channels. crop is (H, W, C_all)
                        # from microBase.get_cropped_cells; we slice to (H, W, C_enabled)
                        # so the exported TIFF matches self._channel_names / Labels / Colors.
                        if ch_indices and ch_indices != list(range(crop.shape[2])):
                            crop_out = crop[:, :, ch_indices]
                        else:
                            crop_out = crop

                        # Save as multi-channel ImageJ TIFF (transpose HWC → CYX)
                        multi_ch = np.moveaxis(crop_out, -1, 0)
                        fname = f"{img_stem}_{label_int}.tiff"
                        # Build per-channel (3, 256) uint8 ramp LUTs so the channel
                        # colors are honoured by Fiji/ImageJ. tifffile silently drops
                        # the 'Colors' metadata key (it's not in the IJMetadata
                        # allowlist); only 'LUTs' is encoded into the IJMetadata tag.
                        ramp = np.arange(256, dtype=np.uint32)
                        ij_luts = []
                        for ch in self._channel_names:
                            r, g, b = self._ch_colors.get(ch, (1.0, 1.0, 1.0))
                            cr = int(round(float(r) * 255))
                            cg = int(round(float(g) * 255))
                            cb = int(round(float(b) * 255))
                            lut = np.stack([
                                (ramp * cr // 255).astype(np.uint8),
                                (ramp * cg // 255).astype(np.uint8),
                                (ramp * cb // 255).astype(np.uint8),
                            ], axis=0)
                            ij_luts.append(lut)
                        tifffile.imwrite(
                            str(obj_save_dir / fname),
                            multi_ch,
                            imagej=True,
                            photometric='minisblack',
                            metadata={
                                'axes': 'CYX',
                                'Labels': list(self._channel_names),
                                'mode': 'composite',
                                'LUTs': ij_luts,
                            },
                        )

                        # Record for CSV — base columns + extra metadata cols
                        record: dict = {
                            "filename": fname,
                            "well": well,
                            "field": field,
                            "stack": stack,
                            "timepoint": timepoint,
                            "label": label_int,
                            "mask_name": self._mask_name,
                            "class_name": class_name,
                            "channels": ";".join(self._channel_names),
                        }
                        for ec in extra_cols:
                            record[ec] = _safe_str(meta_df, row_idx, ec)
                        csv_records.append(record)
                        exported_count += 1

                except Exception as e:
                    logger.warning("Failed to export from row %d: %s", row_idx, e)

                self.signals.progress.emit(img_idx + 1, total)

            # Write {mask_name}.csv manifest (append mode; one file per mask type).
            # Per AGENTS.md §3.2 — no silent fallbacks: if an existing CSV at
            # csv_path was written with a different column schema, hard-error
            # so the user knows to delete the stale file (no migrations exist).
            try:
                write_header = True
                if csv_path.exists() and csv_path.stat().st_size > 0:
                    with open(csv_path, "r", newline="", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        existing = next(reader, None)
                    if existing is not None and existing != csv_columns:
                        msg = (
                            f"Existing CSV header at {csv_path} does not match the "
                            f"current schema. Expected {csv_columns}, got {existing}. "
                            f"Delete the stale file and re-export."
                        )
                        self.signals.error.emit(msg)
                        return
                    write_header = False
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_columns)
                    if write_header:
                        writer.writeheader()
                    writer.writerows(csv_records)
            except Exception as e:
                # The manifest is part of the export contract — a lost CSV
                # with objects already written must not be reported as a
                # successful export.
                logger.warning("Failed to write %s: %s", csv_path.name, e)
                self.signals.error.emit(f"Objects exported but CSV manifest failed: {e}")
                return

            summary = f"{exported_count} objects exported to {self._save_dir}"
            if skipped_rows:
                summary += f" ({skipped_rows} rows skipped — missing masks, see log)"
            logger.info("Export complete: %s (csv=%s)", summary, csv_path)
            self.signals.finished.emit({
                "count": exported_count,
                "save_dir": self._save_dir,
                "csv_path": str(csv_path),
                "gen": self._gen,
                "skipped_rows": skipped_rows,
            })

        except Exception as e:
            logger.exception("Object export failed")
            self.signals.error.emit(str(e))
