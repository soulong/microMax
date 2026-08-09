from __future__ import annotations

import random

from microBase import ImageDataset
from microProfiler.io import read_image


class DatasetService:

    def load_image(self, ds: ImageDataset, row_idx: int, channel: str):
        try:
            p = ds.image_path(row_idx, channel)
        except SystemExit as e:
            # microBase hard-exits (print + sys.exit) when a row is missing
            # the channel file. Convert to a catchable exception so the GUI
            # pick/preview paths degrade gracefully instead of dying.
            raise FileNotFoundError(
                f"Image not found: row={row_idx}, channel={channel}"
            ) from e
        if p is None:
            raise FileNotFoundError(f"Image not found: row={row_idx}, channel={channel}")
        return read_image(p)

    def pick_random_row(self, ds: ImageDataset) -> int:
        if ds is None or len(ds) == 0:
            return -1
        return random.randint(0, len(ds) - 1)
