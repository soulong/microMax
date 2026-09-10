"""Cellpose-SAM segmentation for microscopy images.

Supports:
    1. Single channel  → C1 = image, C2 = 0
    2. Two channel groups → C1 = merge(chan1), C2 = merge(chan2)
"""

from __future__ import annotations

import gc
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from cellpose import models
from skimage.segmentation import expand_labels
from skimage.transform import rescale, resize
from tqdm import tqdm

from microBase import ImageDataset
from microProfiler.io import (
    ImageReadError,
    quarantine_row,
    read_image,
    read_image_shape,
)
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def merge_channels(
    paths: List[Path],
    method: str = "mean",
    resize_factor: float = 1.0,
) -> np.ndarray:
    """Read and merge a list of images into a single 2D array.

    Strict reads: a missing/unreadable file raises ImageReadError so the
    per-row caller can quarantine the row.
    """
    imgs = [read_image(p) for p in paths]
    stacked = np.stack(imgs, axis=0)

    if method == "mean":
        merged = np.mean(stacked, axis=0)
    elif method == "max":
        merged = np.max(stacked, axis=0)
    elif method == "min":
        merged = np.min(stacked, axis=0)
    else:
        raise ValueError(f"Unsupported merge method: {method}")

    if resize_factor != 1.0:
        merged = rescale(
            merged, resize_factor,
            anti_aliasing=True, preserve_range=True,
        ).astype(stacked.dtype)

    return merged


def build_cellpose_image(
    row: pd.Series,
    chan1: List[str],
    chan2: Optional[List[str]],
    merge1: str,
    merge2: str,
    resize_factor: float,
) -> np.ndarray:
    """Build a (C, H, W) array from metadata for Cellpose-SAM."""
    # row[ch] is an absolute source path — use it directly (pathlib would
    # discard any leading directory component for absolute paths anyway).
    ch1_paths = [
        Path(row[ch]) for ch in chan1
        if ch in row and pd.notna(row[ch])
    ]
    if not ch1_paths:
        raise ValueError(f"Missing images for channel group 1: {chan1}")
    c1 = merge_channels(ch1_paths, merge1, resize_factor)

    if chan2:
        ch2_paths = [
            Path(row[ch]) for ch in chan2
            if ch in row and pd.notna(row[ch])
        ]
        if not ch2_paths:
            raise ValueError(f"Missing images for channel group 2: {chan2}")
        c2 = merge_channels(ch2_paths, merge2, resize_factor)
        return np.stack([c1, c2], axis=0)

    return c1[np.newaxis, ...]


def _filter_edge_objects(masks: np.ndarray, max_ratio: float) -> np.ndarray:
    """Drop labels whose edge-pixel ratio exceeds `max_ratio`.

    Objects mostly outside the frame (half-clipped cells score ~0.5) would
    pollute profiling and overlay; remaining labels keep their ids (gaps are
    fine — consumers iterate the labels present in the mask).
    """
    from microBase import edge_pixel_ratio

    ratios = edge_pixel_ratio(masks)
    dropped = [lbl for lbl, r in ratios.items() if r > max_ratio]
    if not dropped:
        return masks
    masks = np.where(np.isin(masks, dropped, invert=True), masks, 0).astype(masks.dtype)
    logger.debug("Edge filter: dropped %d object(s) (ratio > %g)", len(dropped), max_ratio)
    return masks


def segment_single(
    row: pd.Series,
    chan1: List[str],
    chan2: Optional[List[str]] = None,
    merge1: str = "mean",
    merge2: str = "mean",
    model_name: str = "cpdino",
    diameter: Optional[float] = None,
    normalize: Optional[Dict] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    resize_factor: float = 1.0,
    gpu_batch_size: int = 16,
    edge_pixel_ratio: float = 0.4,
    model=None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Segment a single image row using Cellpose-SAM."""
    device = _get_device()
    if model is None:
        model = models.CellposeModel(device=device, pretrained_model=model_name)
    orig_shape = None
    if resize_factor != 1.0:
        from microProfiler.io import read_image_shape
        first_ch_path = Path(row[chan1[0]])
        if first_ch_path.exists():
            orig_shape = read_image_shape(first_ch_path)
    img = build_cellpose_image(row, chan1, chan2, merge1, merge2, resize_factor)
    diameter_val = None if diameter is None or diameter <= 0 else int(diameter * resize_factor)
    if diameter_val is not None:
        diameter_val = min(diameter_val, 10000)
    if normalize is None:
        normalize = {"percentile": [0.1, 99.9]}
    masks, flows, _ = model.eval(
        img,
        normalize=normalize,
        diameter=diameter_val,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        batch_size=gpu_batch_size,
    )
    if resize_factor != 1.0 and orig_shape is not None:
        masks = resize(masks, orig_shape, order=0, preserve_range=True).astype(np.uint16)
        c1_img = resize(img[0], orig_shape, order=1, preserve_range=True).astype(img.dtype)
        c2_img = resize(img[1], orig_shape, order=1, preserve_range=True).astype(img.dtype) if img.shape[0] >= 2 else None
    elif resize_factor != 1.0:
        # preserve_range=True: without it skimage min-max normalizes the
        # label mask to [0,1] and astype truncates labels to 0.
        masks = rescale(masks, 1.0 / resize_factor, order=0, preserve_range=True).astype(np.uint16)
        c1_img = rescale(img[0], 1.0 / resize_factor, order=1, preserve_range=True).astype(img.dtype)
        c2_img = rescale(img[1], 1.0 / resize_factor, order=1, preserve_range=True).astype(img.dtype) if img.shape[0] >= 2 else None
    else:
        c1_img = img[0]
        c2_img = img[1] if img.shape[0] >= 2 else None
    # Label-safe gap fill: grayscale closing on a label map applies min/max
    # filtering to the label VALUES, reassigning 1-2px gaps to whichever label
    # number wins and moving object boundaries. expand_labels preserves label
    # identity while filling small background gaps.
    masks = expand_labels(masks, distance=1)
    # Preview overlay shows the SAME mask that a full run would save.
    masks = _filter_edge_objects(masks, edge_pixel_ratio)
    return c1_img, c2_img, masks


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.debug("Device selected: %s", device)
    return device


def segment_dataset(
    ds: ImageDataset,
    object_name: str = "cell",
    chan1: Optional[List[str]] = None,
    chan2: Optional[List[str]] = None,
    merge1: str = "mean",
    merge2: str = "mean",
    model_name: str = "cpdino",
    diameter: Optional[float] = None,
    normalize: Optional[Dict] = None,
    resize_factor: float = 1.0,
    overwrite_mask: bool = False,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu_batch_size: int = 16,
    edge_pixel_ratio: float = 0.4,
    progress: ProgressCollector = NullProgressCollector(),
) -> ImageDataset:
    """Run Cellpose-SAM segmentation on every image in the dataset."""
    summary: Dict = {
        "success": False,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "masks_saved": 0,
        "errors": [],
    }

    if isinstance(chan1, str):
        chan1 = [chan1]
    if not chan1:
        raise ValueError(
            "segment_dataset requires at least one chan1 channel "
            "(empty channels are skipped by the pipeline, never 'all')."
        )
    if isinstance(chan2, str):
        chan2 = [chan2]

    logger.debug(
        "segment_dataset: object=%s, model=%s, diameter=%s, chan1=%s, chan2=%s, resize=%s",
        object_name, model_name, diameter, chan1, chan2, resize_factor,
    )

    missing = [ch for ch in (chan1 + (chan2 or [])) if ch not in ds.intensity_colnames]
    if missing:
        raise ValueError(
            f"Segmentation channels not found in dataset: {missing}. "
            f"Available channels: {ds.intensity_colnames}"
        )

    device = _get_device()
    normalize = normalize or {"percentile": [0.1, 99.9]}

    # Report under the step key the pipeline started ("segment (<obj>)"),
    # not a bare "Segment" — the GUI status bar groups by key.
    step_key = f"segment ({object_name})"
    progress.report(step_key, 0, 1, "Loading Cellpose model...")
    logger.info("Loading Cellpose model '%s'...", model_name)
    model = models.CellposeModel(device=device, pretrained_model=model_name)
    diameter_val = None if diameter is None or diameter <= 0 else int(diameter * resize_factor)
    if diameter_val is not None:
        diameter_val = min(diameter_val, 10000)

    metadata = ds.metadata
    for idx in tqdm(range(len(metadata)), desc="Cellpose", unit="img", disable=(sys.stdout is None and sys.stderr is None)):
        progress.report(step_key, idx, len(metadata), f"Image {idx}")
        row = metadata.iloc[idx]
        stem_ch = chan1[0]
        stem_val = row[stem_ch] if stem_ch in row.index else None
        if stem_val is None or pd.isna(stem_val):
            summary["skipped"] += 1
            summary["errors"].append(f"Row {idx}: missing path for channel '{stem_ch}'")
            continue
        src_path = Path(stem_val)
        if not src_path.exists():
            # The source file is gone — the whole row is unusable, so delete
            # its remaining files (channels + masks) and skip it.
            quarantine_row(ds, idx, f"missing {src_path.name}")
            summary["skipped"] += 1
            summary["errors"].append(f"Source not found: {src_path.name}")
            continue

        save_stem = src_path.parent / f"{src_path.stem}_cp_masks"
        mask_path = save_stem.with_name(f"{save_stem.name}_{object_name}.png")

        if mask_path.exists() and not overwrite_mask:
            summary["skipped"] += 1
            continue

        try:
            orig_shape = None
            if resize_factor != 1.0:
                orig_path = src_path
                if orig_path.exists():
                    orig_shape = read_image_shape(orig_path)

            img = build_cellpose_image(row, chan1, chan2, merge1, merge2, resize_factor)

            masks, flows, _ = model.eval(
                img,
                batch_size=gpu_batch_size,
                normalize=normalize,
                diameter=diameter_val,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )

            if resize_factor != 1.0 and orig_shape is not None:
                masks = resize(masks, orig_shape, order=0, preserve_range=True).astype(np.uint16)
            elif resize_factor != 1.0:
                # preserve_range=True is required: without it skimage
                # min-max normalizes the label mask to [0,1] and astype
                # truncates every label to 0 (empty/corrupt mask).
                masks = rescale(masks, 1.0 / resize_factor, order=0, preserve_range=True).astype(np.uint16)

            # Label-safe gap fill (was: grayscale closing, which min/max-
            # filtered the label VALUES and moved boundaries). Objects are
            # counted AFTER the fill so n_objects always matches the mask
            # that is actually saved.
            masks = expand_labels(masks, distance=1)
            # Drop mostly-clipped objects BEFORE counting/saving, so the
            # saved mask (and every downstream overlay) is the filtered one.
            masks = _filter_edge_objects(masks, edge_pixel_ratio)
            n_objects = len(np.unique(masks)) - 1
            if n_objects <= 0:
                summary["processed"] += 1
                summary["errors"].append(f"No objects in {src_path.name}")
                # A re-segmentation finding zero objects must not leave a
                # stale mask behind — profiling would measure phantom objects.
                if overwrite_mask and mask_path.exists():
                    mask_path.unlink()
                    logger.warning("Removed stale mask %s (re-segmentation found no objects)", mask_path)
                continue

            mask_to_save = masks.astype(np.uint16)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_to_save).save(str(mask_path))
            summary["processed"] += 1
            summary["masks_saved"] += 1

        except ImageReadError as e:
            # Broken/missing channel file: quarantine the whole row and skip —
            # not a segmentation failure (nothing ran).
            quarantine_row(ds, idx, str(e))
            summary["skipped"] += 1
            summary["errors"].append(f"Broken image on {src_path.name}: {e}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            summary["failed"] += 1
            summary["errors"].append(f"GPU OOM on {src_path.name}")
        except Exception as e:
            summary["failed"] += 1
            summary["errors"].append(f"Error on {src_path.name}: {e}")

        if idx % 200 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

    logger.info(
        "Segmentation complete: %d processed, %d skipped, %d failed, %d masks saved",
        summary["processed"], summary["skipped"], summary["failed"], summary["masks_saved"],
    )
    logger.info(
        "Mask smoothing switched from closing to expand_labels — labels are "
        "preserved, but objects may differ from previous runs (re-segment to update)"
    )
    if summary["errors"]:
        logger.debug("Segmentation errors: %s", summary["errors"])

    ds.build_metadata()
    return ds
