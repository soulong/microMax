"""Image-level profiling — whole-image features."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
from tqdm import tqdm

from microBase import ImageDataset
from microProfiler.profiling.batch_writer import BatchWriter
from microProfiler.profiling.object_profiler import _resolve_source_directory
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
        intensity_channels = channel_names
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
    """Profile a single image — extracted for parallel execution."""
    image_data, _ = ds.get_imageset(idx)
    # _process_one_image: keep the raw metadata row but override "directory"
    # with the absolute, forward-slash parent path — consistent with the
    # object tables (object_profiler._resolve_source_directory) so consumers
    # can join/compare across tables.
    row = ds.metadata.iloc[idx]
    excluded = set(ds.intensity_colnames) | set(ds.mask_colnames)
    meta = {k: v for k, v in row.to_dict().items() if k not in excluded}
    meta["directory"] = _resolve_source_directory(row, ds.intensity_colnames)
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
    table_name: str = "image",
    progress: ProgressCollector = NullProgressCollector(),
    n_workers: int = 1,
) -> Optional[pd.DataFrame]:
    """Profile all images in a dataset at the whole-image level."""
    channels = channels or ds.intensity_colnames
    logger.debug(
        "profile_images: channels=%s, thresholds=%s, db=%s, table=%s, count=%d, workers=%d",
        channels, thresholds, db_path, table_name, len(ds), n_workers,
    )
    BATCH = 50
    n_total = len(ds)
    completed = 0
    result_df = None

    with BatchWriter(db_path, table_name, BATCH) as writer:
        try:
            if n_workers == 1:
                for idx in tqdm(range(n_total), desc="Image profiling", unit="img"):
                    progress.report("Profile image", idx, n_total, "")
                    result = _process_one_image(ds, idx, channels, thresholds)
                    writer.add(result)
                    completed += 1
            else:
                pbar = tqdm(total=n_total, desc="Image profiling", unit="img")
                # Process in chunks to avoid pre-loading all images into RAM
                for chunk_start in range(0, n_total, BATCH):
                    chunk_end = min(chunk_start + BATCH, n_total)
                    tasks = []
                    for idx in range(chunk_start, chunk_end):
                        image_data, _ = ds.get_imageset(idx)
                        row = ds.metadata.iloc[idx]
                        excluded = set(ds.intensity_colnames) | set(ds.mask_colnames)
                        meta = {k: v for k, v in row.to_dict().items() if k not in excluded}
                        meta["directory"] = _resolve_source_directory(row, ds.intensity_colnames)
                        tasks.append((image_data, ds.intensity_colnames, channels, thresholds, meta))

                    with ProcessPoolExecutor(max_workers=n_workers) as executor:
                        futures = {
                            executor.submit(_profile_image_worker, t): idx
                            for idx, t in enumerate(tasks)
                        }
                        for future in as_completed(futures):
                            task_idx = futures[future]
                            try:
                                result = future.result()
                                writer.add(result)
                                completed += 1
                                pbar.update(1)
                                progress.report("Profile image", completed, n_total, "")
                            except InterruptedError:
                                raise
                            except Exception:
                                logger.exception("Image profiling failed for row %d — aborting", chunk_start + task_idx)
                                raise
                pbar.close()
        except InterruptedError:
            logger.info("Image profiling interrupted by user")
        except Exception:
            # A run-level profiling failure re-raises so the dataset is NOT
            # reported complete. Note: batches flushed before the failure are
            # already committed (writer.close() in finally) — the partial
            # table is not silently accepted because cli._is_dataset_complete
            # compares the image row count against the dataset size (and the
            # GUI worker surfaces the error dialog).
            logger.exception("Image profiling failed")
            raise
        finally:
            progress.step_end("Profile image", f"Profiled {completed} images")
            result_df = writer.close()

    return result_df
