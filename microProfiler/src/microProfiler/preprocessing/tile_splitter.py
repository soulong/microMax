from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple

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


def tile_single(
    img: np.ndarray,
    tile_width: int,
    tile_height: int,
) -> List[Tuple[int, np.ndarray]]:
    # Design note: only COMPLETE tiles are produced. Right/bottom remainder
    # regions (image dims not divisible by the tile size) are intentionally
    # dropped, and an image smaller than tile_width x tile_height yields zero
    # tiles. The user is responsible for choosing tile sizes that divide the
    # image dimensions; tile_dataset's delete_original=True then deletes the
    # sources after tiling — do not enable tiling with a non-dividing size
    # unless dropping the remainder is acceptable.
    h, w = img.shape[:2]
    tiles: List[Tuple[int, np.ndarray]] = []
    tile_idx = 1
    for y in range(0, h, tile_height):
        for x in range(0, w, tile_width):
            if y + tile_height <= h and x + tile_width <= w:
                tiles.append((tile_idx, img[y : y + tile_height, x : x + tile_width]))
                tile_idx += 1
    return tiles


def tile_dataset(
    ds: ImageDataset,
    tile_width: int = 1024,
    tile_height: int = 1024,
    delete_original: bool = True,
    progress: ProgressCollector = NullProgressCollector(),
) -> ImageDataset:
    if not ds.image_pattern or "field" not in re.compile(ds.image_pattern).groupindex:
        raise ValueError(
            "Tiling requires an image_pattern with a (?P<field>...) group — "
            "tile numbering would otherwise collapse on rebuild."
        )
    metadata = ds.metadata
    all_sources: list = []
    all_masks: list = []
    total = len(metadata)

    with StepProgress("Tile", total, progress, desc="Tiling", unit="img") as sp:
        for i in range(total):
            sp.report(i, "")
            row = metadata.iloc[i]
            # Read every channel BEFORE writing any tile: a row whose file is
            # broken is quarantined as a whole and must not leave orphan tiles
            # (or half-tiled channels) behind.
            channel_imgs = {}  # {channel: (src_path, img)}
            broken = False
            for ch in ds.intensity_colnames:
                if pd.isna(row[ch]):
                    continue
                src = Path(row[ch])
                if not src.exists():
                    quarantine_row(ds, i, f"missing {src.name}")
                    broken = True
                    break
                try:
                    img = read_image(src)
                except ImageReadError as e:
                    quarantine_row(ds, i, str(e))
                    broken = True
                    break
                channel_imgs[ch] = (src, img)
            if broken:
                continue

            row_sources = []
            produced = False
            for src, img in channel_imgs.values():
                m = re.compile(ds.image_pattern).match(src.name)
                if m and "field" in m.groupdict():
                    orig_field_str = m.group("field")  # raw string, no int()
                    field_start, field_end = m.span("field")
                else:
                    orig_field_str = ""
                    field_start, field_end = len(src.stem), len(src.stem)

                tiles = tile_single(img, tile_width, tile_height)
                if tiles:
                    produced = True

                for tile_idx, tile_data in tiles:
                    # String concat: tile 1 -> "10001" + orig_field, etc.
                    new_field = str(10000 + tile_idx) + orig_field_str
                    new_filename = (
                        src.name[:field_start]
                        + new_field
                        + src.name[field_end:]
                    )
                    # Write next to the source (src.parent), never relative
                    # to the process CWD.
                    write_image(src.parent / new_filename, tile_data)
                row_sources.append(src)

            if produced:
                all_sources.extend(row_sources)
                for mask_col in ds.mask_colnames:
                    mpath = row[mask_col]
                    if pd.notna(mpath):
                        all_masks.append(Path(mpath))
            else:
                # Zero tiles (image smaller than the tile grid, or remainder
                # only): deleting the sources would destroy the dataset. Keep
                # everything and warn instead.
                name = row_sources[0].name if row_sources else f"row {i}"
                logger.warning(
                    "Tiling produced no complete tiles for %s — keeping its "
                    "source file(s) (choose a tile size that divides the "
                    "image, or the image is smaller than the tile).", name)

        sp.finish("Tiling complete")

    if delete_original:
        # Source masks describe whole images, not tiles — remove them with
        # their sources so they do not linger as orphan/mask-only rows.
        for path in all_sources + all_masks:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                logger.exception("Tiling: failed to delete %s", path)

    return rebuild_dataset(ds)
