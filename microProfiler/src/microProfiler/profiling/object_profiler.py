"""Object-level profiling — shape, intensity, and texture per labeled object."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from skimage.measure import regionprops_table

from microBase import ImageDataset, edge_pixel_ratio
from microProfiler.io import ImageReadError, quarantine_row
from microProfiler.profiling import resolve_source_directory
from microProfiler.profiling.batch_writer import BatchWriter
from microProfiler.profiling.extras import (
    _named,
    make_glcm,
    make_granularity,
    make_radial_distribution,
    measure_channel_correlation,
)
from microProfiler.config import ObjectProfileEntry
from microProfiler.progress import StepProgress
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def _lookup_mask(mask_data: Dict[str, np.ndarray], mask_name: str) -> Optional[np.ndarray]:
    """Look up a mask by name, trying both 'name' and 'mask_name' forms.

    GUI strips the 'mask_' prefix when displaying mask names, but the
    ImageDataset stores mask columns with the 'mask_' prefix. This helper
    bridges that convention gap.
    """
    if not mask_name:
        return None
    if mask_name in mask_data:
        return mask_data[mask_name]
    prefixed = f"mask_{mask_name}"
    if prefixed in mask_data:
        return mask_data[prefixed]
    return None


# ── Shape properties ─────────────────────────────────────────────────────
_SHAPE_PROPS: Tuple[str, ...] = (
    "label",
    "area",
    "eccentricity",
    "equivalent_diameter_area",
    "extent",
    "feret_diameter_max",
    "major_axis_length",
    "minor_axis_length",
    "perimeter",
    "solidity",
)

_SHAPE_RENAMES: Dict[str, str] = {
    "area": "shape_area",
    "eccentricity": "shape_eccentricity",
    "equivalent_diameter_area": "shape_equivalent_diameter_area",
    "extent": "shape_extent",
    "feret_diameter_max": "shape_feret_diameter_max",
    "major_axis_length": "shape_major_axis_length",
    "minor_axis_length": "shape_minor_axis_length",
    "perimeter": "shape_perimeter",
    "solidity": "shape_solidity",
}


# ── Helpers ──────────────────────────────────────────────────────────────


def _resolve_indices(
    requested: Optional[Sequence[str]],
    channel_names: List[str],
    param_name: str,
) -> List[int]:
    if not requested:
        return []
    unknown = [c for c in requested if c not in channel_names]
    if unknown:
        raise ValueError(
            f"'{param_name}' unknown channels: {unknown}. "
            f"Available: {channel_names}"
        )
    return [channel_names.index(c) for c in requested]


def _intensity_fns(ch_name: str) -> list:
    def _mean(mask, intensity):
        p = intensity[mask.astype(bool)]
        return float(p.mean()) if p.size > 0 else 0.0

    def _median(mask, intensity):
        p = intensity[mask.astype(bool)]
        return float(np.median(p)) if p.size > 0 else 0.0

    def _std(mask, intensity):
        p = intensity[mask.astype(bool)]
        return float(p.std()) if p.size > 0 else 0.0

    def _sum(mask, intensity):
        p = intensity[mask.astype(bool)]
        return float(p.sum()) if p.size > 0 else 0.0

    return [
        _named(_mean, f"intensity_mean_{ch_name}"),
        _named(_median, f"intensity_median_{ch_name}"),
        _named(_std, f"intensity_std_{ch_name}"),
        _named(_sum, f"intensity_sum_{ch_name}"),
    ]



def _relate_masks(
    child: np.ndarray,
    parent: np.ndarray,
) -> Dict[int, int]:
    """Assign each child object to the parent containing most of its pixels."""
    if child.shape != parent.shape:
        raise ValueError("child and parent mask shapes must match")
    child_labels = np.unique(child)
    child_labels = child_labels[child_labels != 0]
    mapping: Dict[int, int] = {}
    slices = find_objects(child)

    for lbl in child_labels:
        sl = slices[lbl - 1]
        if sl is None:
            mapping[int(lbl)] = 0
            continue
        roi_child = child[sl] == lbl
        roi_parent = parent[sl]
        vals = roi_parent[roi_child]
        vals = vals[vals != 0]
        if vals.size == 0:
            mapping[int(lbl)] = 0
        else:
            unique, counts = np.unique(vals, return_counts=True)
            mapping[int(lbl)] = int(unique[np.argmax(counts)])
    return mapping


def _run_per_channel_regionprops(
    mask: np.ndarray,
    img: np.ndarray,
    channel_names: List[str],
    groups: Dict[int, list],
) -> pd.DataFrame:
    """Run regionprops_table per channel slice and merge on label."""
    dfs: List[pd.DataFrame] = []
    for ch_idx, fns in groups.items():
        if not fns:
            continue
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else str(ch_idx)
        # skimage names extra-property columns from func.__name__ — the NaN
        # fallback MUST use the same names or BatchWriter's first-flush schema
        # locks in bogus `ch_0...` columns and the real features get dropped.
        expected_cols = [fn.__name__ for fn in fns]
        try:
            props = regionprops_table(
                mask, img[..., ch_idx],
                properties=["label"],
                extra_properties=fns,
            )
            dfs.append(pd.DataFrame(props))
        except Exception:
            # The channel's feature group failed — log and continue, but emit
            # its expected columns filled with NaN so the row keeps a stable
            # column set across the batch (BatchWriter reindexes to the first
            # flush's schema; a missing column here would otherwise surface as
            # an unrelated SQLite append error).
            logger.exception(
                "Error computing extra properties for channel %s — filling NaN", ch_name,
            )
            labels = np.unique(mask)
            labels = labels[labels != 0]
            nan_df = pd.DataFrame({"label": labels})
            nan_df[expected_cols] = np.nan
            dfs.append(nan_df)
    if not dfs:
        return pd.DataFrame()
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="label", how="outer")
    return result


# ── Public API ───────────────────────────────────────────────────────────

def measure_objects(
    mask: np.ndarray,
    img: np.ndarray,
    channel_names: List[str],
    metadata_row: Optional[Dict[str, Any]] = None,
    parent_mask: Optional[np.ndarray] = None,
    parent_mask_name: str = "Parent",
    intensity_channels: Optional[Sequence[str]] = None,
    radial_channels: Optional[Sequence[str]] = None,
    radial_kwargs: Optional[Dict] = None,
    granularity_channels: Optional[Sequence[str]] = None,
    granularity_kwargs: Optional[Dict] = None,
    glcm_channels: Optional[Sequence[str]] = None,
    glcm_kwargs: Optional[Dict] = None,
    correlation_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Measure shape, intensity, and texture for every labeled object.

    Returns a DataFrame with columns for shape, boundary, parent,
    intensity, radial, granularity, GLCM, and correlation features.
    """
    if img.ndim != 3:
        raise ValueError(f"img must be (Y, X, C), got {img.shape}")
    if mask.shape != img.shape[:2]:
        raise ValueError("mask and img spatial shapes must match")
    if len(channel_names) != img.shape[2]:
        raise ValueError(f"Got {len(channel_names)} names for {img.shape[2]} channels")

    if intensity_channels is None:
        raise ValueError(
            "measure_objects requires intensity_channels "
            "(empty channels are skipped by the pipeline, never 'all')."
        )

    intensity_idx = _resolve_indices(intensity_channels, channel_names, "intensity_channels")
    radial_idx = _resolve_indices(radial_channels, channel_names, "radial_channels")
    granularity_idx = _resolve_indices(granularity_channels, channel_names, "granularity_channels")
    glcm_idx = _resolve_indices(glcm_channels, channel_names, "glcm_channels")

    corr_pairs: List[Tuple[int, int]] = []
    if correlation_pairs:
        for a, b in correlation_pairs:
            if a not in channel_names or b not in channel_names:
                raise ValueError(f"Correlation pair ({a}, {b}) not in {channel_names}")
            corr_pairs.append((channel_names.index(a), channel_names.index(b)))

    # Step 1: Shape
    shape_props = regionprops_table(mask, properties=_SHAPE_PROPS)
    df = pd.DataFrame(shape_props).rename(
        columns={k: v for k, v in _SHAPE_RENAMES.items() if k in shape_props}
    )

    # Step 2: Edge pixel ratio (edge_pixels / perimeter_pixels) — the same
    # function segmentation uses to drop mostly-clipped objects.
    ratio_map = edge_pixel_ratio(mask)
    df["shape_edge_pixel_ratio"] = df["label"].map(ratio_map)

    # Step 3: Parent relationship
    if parent_mask is not None:
        parent_map = _relate_masks(mask, parent_mask)
        df[f"parent_{parent_mask_name}"] = df["label"].map(parent_map).fillna(0).astype(int)

    # Step 4: Build per-channel extra property groups
    groups: Dict[int, list] = {}

    def _add(ch_idx: int, fns: list) -> None:
        groups.setdefault(ch_idx, []).extend(fns)

    for idx in intensity_idx:
        _add(idx, _intensity_fns(channel_names[idx]))

    rd_kw = dict(radial_kwargs or {})
    for idx in radial_idx:
        fns = make_radial_distribution(ch_name=channel_names[idx], **rd_kw)
        _add(idx, fns)

    # Granularity is computed on the full image (CP-style) — see Step 5b below.
    # It is intentionally NOT added to regionprops' extra_properties.

    gl_kw = dict(glcm_kwargs or {})
    for idx in glcm_idx:
        fns = make_glcm(ch_name=channel_names[idx], **gl_kw)
        _add(idx, fns)

    # Step 5: Per-channel regionprops
    if groups:
        extra = _run_per_channel_regionprops(mask, img, channel_names, groups)
        df = df.merge(extra, on="label", how="left")

    # Step 5b: Granularity (full-image, per-object means).
    # Matches CellProfiler's MeasureGranularity architecture: the spectrum
    # is computed once on the full masked image, then per-object means are
    # extracted from the reconstructed image at each step.
    gr_kw = dict(granularity_kwargs or {})
    if granularity_idx:
        gran_dfs: List[pd.DataFrame] = []
        for idx in granularity_idx:
            gran_computer = make_granularity(
                ch_name=channel_names[idx], **gr_kw,
            )
            spectra = gran_computer(mask, img[..., idx])
            if not spectra:
                continue
            rows = {"label": []}
            for col in gran_computer.column_names:
                rows[col] = []
            for lbl, spec in spectra.items():
                rows["label"].append(int(lbl))
                for k, col in enumerate(gran_computer.column_names):
                    rows[col].append(float(spec[k]))
            gran_dfs.append(pd.DataFrame(rows))
        for gdf in gran_dfs:
            df = df.merge(gdf, on="label", how="left")

    # Step 6: Pearson correlation
    if corr_pairs:
        corr_dict = measure_channel_correlation(mask, img, corr_pairs)
        renamed: Dict[str, Any] = {"label": corr_dict["label"]}
        for a, b in corr_pairs:
            old_key = f"correlation_pearson_ch{a}_ch{b}"
            new_key = f"correlation_pearson_{channel_names[a]}_{channel_names[b]}"
            renamed[new_key] = corr_dict[old_key]
        df = df.merge(pd.DataFrame(renamed), on="label", how="left")

    # Step 7: Prepend metadata
    if metadata_row:
        df = pd.concat([pd.DataFrame([metadata_row] * len(df)), df], axis=1)

    # Step 8: Reorder — attributes first, then label/shape_edge_pixel_ratio/parent, then measurements
    cols = df.columns.tolist()
    meta_cols = list(metadata_row.keys()) if metadata_row else []
    priority_labels = ["label", "shape_edge_pixel_ratio"] + sorted(c for c in cols if c.startswith("parent_"))
    priority = meta_cols + [c for c in priority_labels if c not in meta_cols]
    rest = [c for c in cols if c not in priority]
    df = df[priority + rest]

    return df


def _process_one_object(
    ds: ImageDataset,
    idx: int,
    mask_name: str,
    parent_mask_name: Optional[str],
    intensity_channels: Optional[List[str]],
    correlation_pairs: Optional[List[Tuple[str, str]]],
    measure_kwargs: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """Profile objects in a single image — extracted for parallel execution."""
    try:
        row = ds.metadata.iloc[idx]
        # Skip rows where the mask file is missing (e.g. Cellpose found no
        # cells → no mask saved → mask column is NaN). Without this check,
        # get_imageset would try to read_mask(NaN) and crash.
        mask_col = f"mask_{mask_name}"
        if mask_col not in row or pd.isna(row[mask_col]):
            logger.debug("No mask file for '%s' at row %d — skipping", mask_name, idx)
            return None
        meta = {
            k: v for k, v in row.to_dict().items()
            if k not in ds.intensity_colnames and k not in ds.mask_colnames
        }
        meta["directory"] = resolve_source_directory(row, ds.intensity_colnames, ds.root)
        image_data, mask_data = ds.get_imageset(idx)
        mask = _lookup_mask(mask_data, mask_name)
        if mask is None:
            logger.debug("Mask '%s' not found for row %d, skipping", mask_name, idx)
            return None
        parent_mask = None
        if parent_mask_name is not None:
            parent_mask = _lookup_mask(mask_data, parent_mask_name)
        return measure_objects(
            mask=mask,
            img=image_data,
            channel_names=ds.intensity_colnames,
            metadata_row=meta,
            parent_mask=parent_mask,
            parent_mask_name=parent_mask_name or "Parent",
            intensity_channels=intensity_channels,
            correlation_pairs=correlation_pairs,
            **measure_kwargs,
        )
    except ImageReadError as e:
        # Missing/unreadable file: quarantine the whole row (channels + masks)
        # and skip — the batch continues.
        quarantine_row(ds, idx, str(e))
        return None
    except Exception:
        # Intentional design: a row that fails to profile is logged and
        # skipped; the batch continues by design (fix the row at the source,
        # don't abort the run). This is the documented per-row skip, distinct
        # from the run-level failure handling in profile_objects.
        logger.exception("Error profiling row %d — skipping", idx)
        return None


def _profile_object_worker(args):
    """ProcessPoolExecutor worker — receives only serializable types."""
    try:
        (image_data, mask_data, channel_names, meta, mask_name,
         parent_mask_name, intensity_channels, correlation_pairs, measure_kwargs) = args

        mask = _lookup_mask(mask_data, mask_name)
        if mask is None:
            logger.debug("Mask '%s' not found, skipping", mask_name)
            return None
        parent_mask = None
        if parent_mask_name is not None:
            parent_mask = _lookup_mask(mask_data, parent_mask_name)
        return measure_objects(
            mask=mask,
            img=image_data,
            channel_names=channel_names,
            metadata_row=meta,
            parent_mask=parent_mask,
            parent_mask_name=parent_mask_name or "Parent",
            intensity_channels=intensity_channels,
            correlation_pairs=correlation_pairs,
            **measure_kwargs,
        )
    except Exception:
        # Intentional design: same per-row skip semantics as
        # _process_one_object — log and continue.
        logger.exception("Worker error profiling image — skipping")
        return None


def profile_objects(
    ds: ImageDataset,
    obj_config: ObjectProfileEntry,
    mask_name: str,
    parent_mask_name: Optional[str] = None,
    intensity_channels: Optional[List[str]] = None,
    correlation_pairs: Optional[List[Tuple[str, str]]] = None,
    db_path: Union[str, Path, None] = None,
    table_name: Optional[str] = None,
    progress: ProgressCollector = NullProgressCollector(),
    n_workers: int = 1,
) -> Optional[pd.DataFrame]:
    """Profile all objects in a dataset for a given mask."""
    resolved = obj_config.resolved()
    mask_name = resolved.mask_name or mask_name
    parent_mask_name = resolved.parent_mask_name or parent_mask_name
    intensity_channels = resolved.intensity_channels or intensity_channels
    if not intensity_channels:
        raise ValueError(
            "profile_objects requires intensity_channels "
            "(empty channels are skipped by the pipeline, never 'all')."
        )
    correlation_pairs = resolved.correlation_pairs or correlation_pairs
    if correlation_pairs is not None:
        correlation_pairs = [tuple(p) if isinstance(p, list) else p for p in correlation_pairs]

    table_name = table_name or mask_name
    logger.debug(
        "profile_objects: mask=%s, parent=%s, intensity=%s, db=%s, table=%s, count=%d, workers=%d",
        mask_name, parent_mask_name, intensity_channels, db_path, table_name, len(ds), n_workers,
    )

    # Pre-build measure_kwargs (constant across all images)
    measure_kwargs: Dict[str, Any] = {}

    if resolved.radial_channels:
        measure_kwargs["radial_channels"] = resolved.radial_channels
        measure_kwargs["radial_kwargs"] = {"nbins": resolved.radial_bins}
    if resolved.granularity_channels:
        measure_kwargs["granularity_channels"] = resolved.granularity_channels
        measure_kwargs["granularity_kwargs"] = {
            "spectrum_length": resolved.granularity_spectrum_length,
            "subsample_size": resolved.granularity_subsample_ratio,
            "image_sample_size": resolved.granularity_background_subsample_ratio,
            "background_radius": resolved.granularity_background_radius,
        }
    if resolved.glcm_channels:
        measure_kwargs["glcm_channels"] = resolved.glcm_channels
        glcm_kwargs: Dict[str, Any] = {
            "distances": resolved.glcm_distances,
            "levels": resolved.glcm_levels,
        }
        measure_kwargs["glcm_kwargs"] = glcm_kwargs

    BATCH = 50
    n_total = len(ds)
    completed = 0
    executor = None

    with BatchWriter(db_path, table_name, BATCH) as writer:
        try:
            with StepProgress(f"Profile {mask_name}", n_total, progress, desc=f"Profiling {mask_name}", unit="img") as sp:
                if n_workers == 1:
                    for idx in range(n_total):
                        sp.report(idx, "")
                        result = _process_one_object(
                            ds, idx, mask_name, parent_mask_name,
                            intensity_channels, correlation_pairs, measure_kwargs,
                        )
                        if result is not None:
                            writer.add(result)
                        # Skips (missing mask) also advance the counter so the
                        # progress bar reaches n_total exactly.
                        completed += 1
                    sp.finish(f"Profiled {mask_name}")
                else:
                    # One executor for the whole run — chunks only bound how
                    # many images are pre-loaded into RAM at once.
                    executor = ProcessPoolExecutor(max_workers=n_workers)
                    for chunk_start in range(0, n_total, BATCH):
                        chunk_end = min(chunk_start + BATCH, n_total)
                        tasks = []
                        for idx in range(chunk_start, chunk_end):
                            row = ds.metadata.iloc[idx]
                            # Skip rows where the mask file is missing (e.g.
                            # Cellpose found no cells → no mask saved → mask
                            # column is NaN). Without this check, get_imageset
                            # would try to read_mask(NaN) and crash.
                            mask_col = f"mask_{mask_name}"
                            if mask_col not in row or pd.isna(row[mask_col]):
                                sp.tick("")
                                completed += 1
                                continue
                            meta = {
                                k: v for k, v in row.to_dict().items()
                                if k not in ds.intensity_colnames and k not in ds.mask_colnames
                            }
                            meta["directory"] = resolve_source_directory(row, ds.intensity_colnames, ds.root)
                            try:
                                image_data, mask_data = ds.get_imageset(idx)
                            except ImageReadError as e:
                                quarantine_row(ds, idx, str(e))
                                sp.tick("")
                                completed += 1
                                continue
                            tasks.append((
                                # (global row index, worker args) — futures are
                                # keyed by the global index so error logs point
                                # at the real row, not the chunk-relative slot.
                                idx, (image_data, mask_data, ds.intensity_colnames, meta,
                                      mask_name, parent_mask_name, intensity_channels,
                                      correlation_pairs, measure_kwargs),
                            ))

                        futures = {
                            executor.submit(_profile_object_worker, payload): row_idx
                            for row_idx, payload in tasks
                        }
                        chunk_completed = 0
                        for future in as_completed(futures):
                            row_idx = futures[future]
                            try:
                                result = future.result()
                                if result is not None:
                                    writer.add(result)
                                completed += 1
                                chunk_completed += 1
                                sp.tick("")
                            except InterruptedError:
                                raise
                            except BrokenProcessPool:
                                lost = len(tasks) - chunk_completed
                                logger.error("Worker process crashed — %d remaining task(s) in chunk skipped", lost)
                                raise
                            except Exception:
                                logger.exception("Object profiling failed for row %d — skipping", row_idx)
                                # A failed task still advances the bar, or it
                                # never reaches n_total and ``lost`` over-counts.
                                completed += 1
                                chunk_completed += 1
                                sp.tick("")
        except InterruptedError:
            # A cancel must propagate so the pipeline treats the run as
            # interrupted (never as a completed step): the GUI relies on
            # InterruptedError to skip success handlers and applied_steps
            # bookkeeping. Queued tasks are cancelled; running ones finish.
            # The writer still flushes on with-block exit.
            if executor is not None:
                executor.shutdown(cancel_futures=True)
            raise
        except BrokenProcessPool:
            if executor is not None:
                executor.shutdown(cancel_futures=True)
            raise
        except Exception:
            # A run-level profiling failure re-raises so the dataset is NOT
            # reported complete. Batches flushed before the failure are
            # already committed; the CLI batch loop logs the dataset failure
            # and continues, and the GUI worker surfaces the error dialog.
            # Object tables have no row-count guard — delete profiler.db (or
            # the table) before re-running after a failed object run.
            if executor is not None:
                executor.shutdown(cancel_futures=True)
            logger.exception("Object profiling failed")
            raise
        else:
            if executor is not None:
                executor.shutdown()

    return None
