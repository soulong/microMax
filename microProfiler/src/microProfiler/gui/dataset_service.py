from __future__ import annotations

import random

from microBase import ImageDataset
from microProfiler.io import read_image


class DatasetService:

    def load_image(self, ds: ImageDataset, row_idx: int, channel: str):
        p = ds.image_path(row_idx, channel)
        if p is None:
            raise FileNotFoundError(f"Image not found: row={row_idx}, channel={channel}")
        return read_image(p)

    def pick_random_row(self, ds: ImageDataset) -> int:
        if ds is None or len(ds) == 0:
            return -1
        return random.randint(0, len(ds) - 1)
