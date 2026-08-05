from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from microBase import ImageDataset
from microProfiler.io import read_image, write_image, rebuild_dataset
from microProfiler.progress_collector import NullProgressCollector, ProgressCollector

logger = logging.getLogger(__name__)


def tile_single(
    img: np.ndarray,
    tile_w: int,
    tile_h: int,
) -> List[Tuple[int, np.ndarray]]:
    # Design note: only COMPLETE tiles are produced. Right/bottom remainder
    # regions (image dims not divisible by the tile size) are intentionally
    # dropped, and an image smaller than tile_w x tile_h yields zero tiles.
    # The user is responsible for choosing tile sizes that divide the image
    # dimensions; tile_dataset's delete_original=True then deletes the
    # sources after tiling — do not enable tiling with a non-dividing size
    # unless dropping the remainder is acceptable.
    h, w = img.shape[:2]
    tiles: List[Tuple[int, np.ndarray]] = []
    tile_idx = 1
    for y in range(0, h, tile_h):
        for x in range(0, w, tile_w):
            if y + tile_h <= h and x + tile_w <= w:
                tiles.append((tile_idx, img[y : y + tile_h, x : x + tile_w]))
                tile_idx += 1
    return tiles


def tile_dataset(
    ds: ImageDataset,
    tile_w: int = 1024,
    tile_h: int = 1024,
    delete_original: bool = True,
    progress: ProgressCollector = NullProgressCollector(),
) -> ImageDataset:
    metadata = ds.metadata
    all_sources: list = []

    for i in tqdm(range(len(metadata)), desc="Tiling", unit="img"):
        progress.report("Tile", i, len(metadata), "")
        row = metadata.iloc[i]
        img_dir = row["directory"]
        for ch in ds.intensity_colnames:
            if pd.notna(row[ch]):
                src = Path(img_dir) / row[ch]
                if not src.exists():
                    logger.warning("Missing file, skipping: %s", src)
                    continue
                img = read_image(src)

                m = ds._image_pattern.match(src.name)
                if m and "field" in m.groupdict():
                    orig_field_str = m.group("field")  # raw string, no int()
                    field_start, field_end = m.span("field")
                else:
                    orig_field_str = ""
                    field_start, field_end = len(src.stem), len(src.stem)

                tiles = tile_single(img, tile_w, tile_h)

                for tile_idx, tile_data in tiles:
                    # String concat: tile 1 -> "10001" + orig_field, etc.
                    new_field = str(10000 + tile_idx) + orig_field_str
                    new_filename = (
                        src.name[:field_start]
                        + new_field
                        + src.name[field_end:]
                    )
                    write_image(Path(img_dir) / new_filename, tile_data)
                all_sources.append(src)

    if delete_original:
        # Design note: sources are deleted after tiling even when an image
        # produced zero tiles (remainder-only or sub-tile images) — see the
        # note on tile_single. Verify tile sizes before enabling this.
        for src in all_sources:
            if src.exists():
                src.unlink()

    progress.report("Tile", len(metadata), len(metadata), "Tiling complete")
    return rebuild_dataset(ds)
