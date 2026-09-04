"""Z-stack projection — collapse the Z (stack) dimension."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from microBase import ImageDataset
from microProfiler.io import read_image, write_image, rebuild_dataset
from microProfiler.progress import StepProgress
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def z_project_single(
    image_paths: List[Path],
    method: str = "max",
) -> np.ndarray:
    images = [read_image(p) for p in image_paths]
    stacked = np.stack(images, axis=0)
    dtype = stacked.dtype
    if method == "max":
        result = np.max(stacked, axis=0)
    elif method == "mean":
        result = np.round(np.mean(stacked, axis=0)).astype(dtype)
    elif method == "min":
        result = np.min(stacked, axis=0)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return result


def z_project_dataset(
    ds: ImageDataset,
    method: str = "max",
    delete_original: bool = True,
    progress: ProgressCollector = NullProgressCollector(),
) -> ImageDataset:
    metadata = ds.metadata

    if "stack" not in metadata.columns:
        raise ValueError("Metadata must contain a 'stack' column for Z-projection")
    if ds.channel_layout is not None:
        raise ValueError(
            "Z-projection is not supported for multi-channel-per-file (CHW/HWC) "
            "datasets — each file already holds one plane per site."
        )

    exclude = set(ds.intensity_colnames) | set(ds.mask_colnames) | {"stack", "directory"}
    group_cols = [c for c in metadata.columns if c not in exclude]
    if not group_cols:
        raise ValueError("No group columns found for Z-projection (all columns excluded)")

    grouped = metadata.groupby(group_cols, sort=False)
    all_groups = list(grouped)
    total = len(all_groups)

    with StepProgress("Z-projection", total, progress, desc="Z-projection", unit="group") as sp:
        for gi, (group_key, group_df) in enumerate(all_groups):
            sp.report(gi, f"Group {group_key}")
            if len(group_df) <= 1:
                continue

            for ch in ds.intensity_colnames:
                paths = [Path(row[ch]) for _, row in group_df.iterrows() if pd.notna(row[ch])]
                paths = [p for p in paths if p.exists()]
                if not paths:
                    logger.warning("No existing files for channel %s in group, skipping", ch)
                    continue

                projected = z_project_single(paths, method)
                src_name = paths[0].name
                m = re.compile(ds.image_pattern).match(src_name)
                if m:
                    try:
                        s_start, s_end = m.span('stack')
                        stem = src_name.rsplit('.', 1)[0]
                        ext = src_name.rsplit('.', 1)[1] if '.' in src_name else 'tiff'
                        out_name = stem[:s_start] + '0' + stem[s_end:] + '.' + ext
                    except (IndexError, ValueError):
                        out_name = src_name
                else:
                    out_name = src_name

                out_path = paths[0].parent / out_name
                if delete_original:
                    for p in paths:
                        if p.exists():
                            p.unlink()
                write_image(out_path, projected)

        sp.finish("Z-projection complete")
    return rebuild_dataset(ds)
