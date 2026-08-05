from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from microBase import ImageDataset


@dataclass
class PipelineState:

    dataset: Optional[ImageDataset] = None
    _original_dataset: Optional[ImageDataset] = field(default=None, repr=False)

    @property
    def original_dataset(self) -> Optional[ImageDataset]:
        return self._original_dataset

    @original_dataset.setter
    def original_dataset(self, value: Optional[ImageDataset]) -> None:
        self._original_dataset = value
