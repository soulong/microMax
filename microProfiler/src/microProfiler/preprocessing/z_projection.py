"""Z-stack projection — collapse the Z (stack) dimension."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from microBase import ImageDataset
from microProfiler.io import (
    ImageReadError,
    quarantine_row,
    read_image,
    rebuild_dataset,
    write_image,
)
from microProfiler.progress import StepProgress
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def z_project_single(
    images: List[np.ndarray],
    method: str = "max",
) -> np.ndarray:
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

            # Masks belong to individual planes — after projection they no
            # longer correspond to any image, and leaving them behind creates
            # orphan files that rebuild into mask-only rows. Delete them with
            # the planes (quarantine already removed broken rows' files).
            if delete_original:
                for _ri, grow in group_df.iterrows():
                    for mask_col in ds.mask_colnames:
                        mpath = grow[mask_col]
                        if pd.isna(mpath):
                            continue
                        mpath = Path(mpath)
                        try:
                            if mpath.exists():
                                mpath.unlink()
                        except OSError:
                            logger.exception(
                                "Z-projection: failed to delete %s", mpath)

            # Read every plane and channel BEFORE any projection/write. A
            # plane with a broken file is quarantined as a whole, so every
            # channel's projection uses the same surviving plane set (never a
            # per-channel mix caused by channel processing order).
            plane_paths = []  # [{channel: Path}], one entry per surviving plane
            plane_imgs = []   # [{channel: np.ndarray}]
            # enumerate() keeps a POSITIONAL row index: quarantine_row indexes
            # ds.metadata.iloc[...] and must not depend on the DataFrame's
            # index labels coinciding with positions.
            for pos, (row_idx, row) in enumerate(group_df.iterrows()):
                paths = {}
                imgs = {}
                broken = False
                for ch in ds.intensity_colnames:
                    if pd.isna(row[ch]):
                        continue
                    src = Path(row[ch])
                    if not src.exists():
                        quarantine_row(ds, pos, f"missing {src.name}")
                        broken = True
                        break
                    try:
                        imgs[ch] = read_image(src)
                    except ImageReadError as e:
                        quarantine_row(ds, pos, str(e))
                        broken = True
                        break
                    paths[ch] = src
                if broken or not paths:
                    continue
                plane_paths.append(paths)
                plane_imgs.append(imgs)

            for ch in ds.intensity_colnames:
                paths = [p[ch] for p in plane_paths if ch in p]
                imgs = [p[ch] for p in plane_imgs if ch in p]
                if len(imgs) <= 1:
                    if imgs:
                        logger.warning(
                            "Z-projection: fewer than 2 planes left for channel %s "
                            "in group %s, skipping", ch, group_key,
                        )
                    continue

                projected = z_project_single(imgs, method)
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
