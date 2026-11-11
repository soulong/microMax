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
    """Collapse every site's z-stack into one image per channel.

    Rows are grouped by every metadata column except ``stack`` (one group =
    one site); each channel's planes are projected and written under the
    stack-0 file name, so the rebuilt dataset holds one row per site.

    Channels may be acquired with DIFFERENT z-stack depths (e.g. a 2-plane
    fluorescence channel next to a single-plane one). Each channel projects
    its own planes; a channel with a single plane has nothing to collapse
    and is carried over unchanged, but still moves to the stack-0 name so
    the site stays ONE row holding all of its channels.
    """
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
    # {channel: number of single-plane images carried over unprojected} —
    # reported once after the loop instead of one log line per group.
    carried = {}

    with StepProgress("Z-projection", total, progress, desc="Z-projection", unit="group") as sp:
        for gi, (group_key, group_df) in enumerate(all_groups):
            sp.report(gi, f"Group {group_key}")
            if len(group_df) <= 1:
                continue

            projected_any = False

            # Read every plane and channel BEFORE any projection/write. A
            # plane with a broken file is quarantined as a whole, so every
            # channel's projection uses the same surviving plane set (never a
            # per-channel mix caused by channel processing order).
            plane_paths = []  # [{channel: Path}], one entry per surviving plane
            plane_imgs = []   # [{channel: np.ndarray}]
            # iterrows() yields the row's ORIGINAL metadata index label, and
            # ds.metadata always carries a contiguous RangeIndex (fresh build,
            # reset_index after every sort/filter) — so the label IS the
            # global position quarantine_row's iloc[...] expects. The
            # group-local enumerate position must never be used: with more
            # than one group it points at the wrong metadata row.
            for row_idx, row in group_df.iterrows():
                paths = {}
                imgs = {}
                broken = False
                for ch in ds.intensity_colnames:
                    if pd.isna(row[ch]):
                        continue
                    src = Path(row[ch])
                    if not src.exists():
                        quarantine_row(ds, row_idx, f"missing {src.name}")
                        broken = True
                        break
                    try:
                        imgs[ch] = read_image(src)
                    except ImageReadError as e:
                        quarantine_row(ds, row_idx, str(e))
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
                if not imgs:
                    # The channel has no surviving plane in this group (never
                    # acquired for this site, or all its planes quarantined).
                    continue

                # Every channel of the group is written under the SAME
                # stack-0 name, so the rebuilt metadata keeps one row per site.
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

                if len(imgs) == 1:
                    # Channels can have different z-stack depths: this one
                    # has a single plane, so there is nothing to collapse and
                    # the plane is carried over unchanged. It must still move
                    # to the stack-0 name — left under its own stack value it
                    # would become a second row for the site, each of the two
                    # rows missing the other one's channels.
                    if out_path == paths[0]:
                        # Already stored under the projected name.
                        continue
                    projected = imgs[0]
                    carried[ch] = carried.get(ch, 0) + 1
                else:
                    projected = z_project_single(imgs, method)
                projected_any = True

                if delete_original:
                    for p in paths:
                        if p.exists():
                            p.unlink()
                write_image(out_path, projected)

            # Masks belong to individual planes — once the group's images
            # moved to the stack-0 name they no longer correspond to any
            # image, and leaving them behind creates orphan files that
            # rebuild into mask-only rows. Delete them only when the group
            # ACTUALLY wrote something: a group left untouched (every
            # channel already under the projected name) keeps its masks.
            if delete_original and projected_any:
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

        sp.finish("Z-projection complete")

    if carried:
        logger.info(
            "Z-projection: single-plane images carried over unprojected "
            "(channels with fewer z-planes than the rest) — %s",
            ", ".join(f"{ch}: {n}" for ch, n in carried.items()),
        )
    return rebuild_dataset(ds)
