from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from PySide6.QtWidgets import QWidget

from microBase import ImageDataset
from microProfiler.gui.panels.base_step_panel import BaseStepPanel


class IControllerView(Protocol):
    """Interface that MainWindow implements for PipelineController."""

    @property
    def running(self) -> bool:
        ...

    def set_running(self, running: bool) -> None:
        ...

    def get_input_dir(self) -> str:
        ...

    def get_output_dir(self) -> str:
        ...

    def output_path(self) -> Path:
        ...

    def progress_reset(self) -> None:
        ...

    def progress_finished(self) -> None:
        ...

    def progress_show_status(self, message: str) -> None:
        ...

    def progress_show_error(self, message: str) -> None:
        ...

    def progress_connect_update(self, signal) -> None:
        ...

    def get_all_step_panels(self) -> List[BaseStepPanel]:
        ...

    def get_preprocessing_steps(self) -> List[BaseStepPanel]:
        ...

    def get_step_panel(self, name: str) -> Optional[BaseStepPanel]:
        ...

    def update_dataset_info(self, ds: ImageDataset) -> None:
        ...

    def update_tab_status(self) -> None:
        ...

    def refresh_step_panels(self, channels=None, masks=None) -> None:
        ...

    @property
    def dataset(self) -> Optional[ImageDataset]:
        ...

    @dataset.setter
    def dataset(self, ds: Optional[ImageDataset]) -> None:
        ...

    @property
    def original_dataset(self) -> Optional[ImageDataset]:
        ...

    @original_dataset.setter
    def original_dataset(self, ds: Optional[ImageDataset]) -> None:
        ...

    def widget(self) -> QWidget:
        ...
