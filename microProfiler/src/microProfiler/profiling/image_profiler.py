"""Image-level profiling — whole-image features."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table

from microBase import ImageDataset
from microBase.db_contracts import IMAGE_TABLE
from microProfiler.io import ImageReadError, quarantine_row
from microProfiler.profiling import resolve_source_directory
from microProfiler.profiling.batch_writer import BatchWriter
from microProfiler.progress import StepProgress
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def measure_single_image(
    image_data: np.ndarray,
    channel_names: List[str],
    intensity_channels: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Profile a single (Y, X, C) image stack at the whole-image level."""
    if image_data.ndim != 3:
        raise ValueError(f"image_data must be (Y, X, C), got {image_data.shape}")
    if len(channel_names) != image_data.shape[2]:
        raise ValueError(
            f"Got {len(channel_names)} names for {image_data.shape[2]} channels"
        )

    if intensity_channels is None:
        raise ValueError(
            "measure_single_image requires intensity_channels "
            "(empty channels are skipped by the pipeline, never 'all')."
        )
    thresholds = thresholds or {}

    result: Dict[str, Any] = {}

    for ch_name in intensity_channels:
        ch_idx = channel_names.index(ch_name)
        img = image_data[:, :, ch_idx]

        if img.size == 0 or np.all(img == 0):
            result[f"intensity_mean_{ch_name}"] = 0.0
            result[f"intensity_sum_{ch_name}"] = 0.0
            for q in [0.1, 1, 25, 75, 99, 99.9]:
                result[f"intensity_q{q}_{ch_name}"] = 0.0
        else:
            result[f"intensity_mean_{ch_name}"] = float(np.mean(img))
            result[f"intensity_sum_{ch_name}"] = float(np.sum(img))
            for q in [0.1, 1, 25, 75, 99, 99.9]:
                result[f"intensity_q{q}_{ch_name}"] = float(np.percentile(img, q))

        threshold = thresholds.get(ch_name)
        if threshold is not None:
            binary = img >= threshold
            labeled = label(binary)
            if labeled.max() > 0:
                props = regionprops_table(labeled, properties=["area", "label"])
                result[f"shape_area_{ch_name}"] = int(np.sum(props["area"]))
                result[f"shape_n_object_{ch_name}"] = len(props["label"])
                result[f"shape_mean_object_area_{ch_name}"] = float(np.mean(props["area"]))
            else:
                result[f"shape_area_{ch_name}"] = 0
                result[f"shape_n_object_{ch_name}"] = 0
                result[f"shape_mean_object_area_{ch_name}"] = 0.0

    return result


def _process_one_image(
    ds: ImageDataset,
    idx: int,
    channels: List[str],
    thresholds: Optional[Dict[str, float]],
) -> pd.DataFrame:
    """Profile a single image — extracted for parallel execution.

    Strict reads: a missing/unreadable file raises ImageReadError so the
    caller can quarantine the row. Masks are deliberately not loaded: image
    profiling has no mask dependency, and a corrupt mask must not quarantine
    a row whose images are fine (masks=[] keeps get_imageset from loading
    the full mask set).
    """
    image_data, _ = ds.get_imageset(idx, masks=[])
    # _process_one_image: keep the raw metadata row but override "directory"
    # with the absolute, forward-slash parent path — consistent with the
    # object tables (object_profiler._resolve_source_directory) so consumers
    # can join/compare across tables.
    row = ds.metadata.iloc[idx]
    excluded = set(ds.intensity_colnames) | set(ds.mask_colnames)
    meta = {k: v for k, v in row.to_dict().items() if k not in excluded}
    meta["directory"] = resolve_source_directory(row, ds.intensity_colnames, ds.root)
    measures = measure_single_image(image_data, ds.intensity_colnames, channels, thresholds)
    return pd.DataFrame([{**meta, **measures}])


def _profile_image_worker(args):
    """ProcessPoolExecutor worker — receives only serializable types."""
    image_data, channel_names, channels, thresholds, meta = args
    measures = measure_single_image(image_data, channel_names, channels, thresholds)
    return pd.DataFrame([{**meta, **measures}])


def profile_images(
    ds: ImageDataset,
    channels: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    db_path: Union[str, Path, None] = None,
    table_name: str = IMAGE_TABLE,
    progress: ProgressCollector = NullProgressCollector(),
    n_workers: int = 1,
) -> Optional[pd.DataFrame]:
    """Profile all images in a dataset at the whole-image level."""
    if not channels:
        raise ValueError(
            "profile_images requires at least one channel "
            "(empty channels are skipped by the pipeline, never 'all')."
        )
    unknown = [c for c in channels if c not in ds.intensity_colnames]
    if unknown:
        raise ValueError(
            f"image_channels unknown channels: {unknown}. "
            f"Available: {list(ds.intensity_colnames)}"
        )
    logger.debug(
        "profile_images: channels=%s, thresholds=%s, db=%s, table=%s, count=%d, workers=%d",
        channels, thresholds, db_path, table_name, len(ds), n_workers,
    )
    BATCH = 50
    n_total = len(ds)
    completed = 0
    executor = None

    with BatchWriter(db_path, table_name, BATCH) as writer:
        try:
            with StepProgress("Profile image", n_total, progress, desc="Image profiling", unit="img") as sp:
                if n_workers == 1:
                    for idx in range(n_total):
                        sp.report(idx, "")
                        try:
                            result = _process_one_image(ds, idx, channels, thresholds)
                        except InterruptedError:
                            raise
                        except ImageReadError as e:
                            # Missing/unreadable file: quarantine the row and
                            # skip it (completed still advances so the bar
                            # reaches n_total exactly).
                            quarantine_row(ds, idx, str(e))
                            completed += 1
                            continue
                        except Exception:
                            # Align with object profiler's per-row skip: one
                            # bad image logs and continues (the CLI's row-count
                            # guard still flags an incomplete table).
                            logger.exception(
                                "Image profiling failed for row %d — skipping", idx)
                            completed += 1
                            continue
                        writer.add(result)
                        completed += 1
                    sp.finish("Image profiling")
                else:
                    # One executor for the whole run (chunks only bound how
                    # many images are pre-loaded into RAM).
                    executor = ProcessPoolExecutor(max_workers=n_workers)
                    for chunk_start in range(0, n_total, BATCH):
                        chunk_end = min(chunk_start + BATCH, n_total)
                        tasks = []
                        for idx in range(chunk_start, chunk_end):
                            try:
                                image_data, _ = ds.get_imageset(idx, masks=[])
                            except ImageReadError as e:
                                quarantine_row(ds, idx, str(e))
                                completed += 1
                                sp.tick("")
                                continue
                            row = ds.metadata.iloc[idx]
                            excluded = set(ds.intensity_colnames) | set(ds.mask_colnames)
                            meta = {k: v for k, v in row.to_dict().items() if k not in excluded}
                            meta["directory"] = resolve_source_directory(row, ds.intensity_colnames, ds.root)
                            tasks.append((idx, (image_data, ds.intensity_colnames, channels, thresholds, meta)))

                        futures = {
                            executor.submit(_profile_image_worker, payload): idx
                            for idx, payload in tasks
                        }
                        for future in as_completed(futures):
                            idx = futures[future]
                            try:
                                result = future.result()
                                writer.add(result)
                                completed += 1
                                sp.tick("")
                            except InterruptedError:
                                raise
                            except Exception:
                                logger.exception(
                                    "Image profiling failed for row %d — skipping", idx)
                                completed += 1
                                sp.tick("")
        except InterruptedError:
            # A cancel must propagate so the pipeline treats the run as
            # interrupted (never as a completed step); queued tasks are
            # cancelled, running ones finish. The writer still flushes below.
            if executor is not None:
                executor.shutdown(cancel_futures=True)
            raise
        except BrokenProcessPool:
            if executor is not None:
                executor.shutdown(cancel_futures=True)
            raise
        except Exception:
            # A run-level profiling failure re-raises so the dataset is NOT
            # reported complete. Note: batches flushed before the failure are
            # already committed — the partial table is not silently accepted
            # because cli._is_dataset_complete compares the image row count
            # against the dataset size (and the GUI worker surfaces the error).
            if executor is not None:
                executor.shutdown(cancel_futures=True)
            raise
        else:
            if executor is not None:
                executor.shutdown()

    return None
