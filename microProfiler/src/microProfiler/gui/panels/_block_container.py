from __future__ import annotations

from typing import Any, List, Optional, Type

from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from microProfiler.gui.panels.base_step_panel import BaseStepPanel


class BlockContainerPanel(BaseStepPanel):
    """Base class for step panels that manage a dynamic list of block widgets.

    Provides shared layout management, block add/remove, and serialization
    (load_config_section / to_config with the structured list-of-dicts format).

    Subclasses must set:
    - _block_widget_class: the widget class for blocks

    Subclasses should call _build_block_container() in their __init__.
    """

    _block_widget_class: Optional[Type[QWidget]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._blocks: List[QWidget] = []
        self._channels: List[str] = []
        self._block_container: Optional[QWidget] = None
        self._blocks_layout: Optional[QVBoxLayout] = None
        self._add_btn: Optional[QPushButton] = None
        self._add_btn_layout: Optional[QHBoxLayout] = None

    def _build_block_container(self, add_btn_text: str = "+ Add New Block") -> None:
        """Set up the block container layout with add button."""
        self._block_container = QWidget()
        self._blocks_layout = QVBoxLayout(self._block_container)
        self._blocks_layout.setContentsMargins(0, 0, 0, 0)

        self._add_btn = QPushButton(add_btn_text)
        self._add_btn.clicked.connect(self._on_add_block_clicked)
        self._add_btn_layout = QHBoxLayout()
        self._add_btn_layout.addStretch()
        self._add_btn_layout.addWidget(self._add_btn)
        self._add_btn_layout.addStretch()
        self._blocks_layout.addLayout(self._add_btn_layout)

        self._controls_layout.addWidget(self._block_container)

    def _on_add_block_clicked(self) -> None:
        """Called when the add button is clicked. Override to customize."""
        self._add_block_generic()

    def _add_block_generic(self, channels: Optional[List[str]] = None, **kwargs) -> QWidget:
        """Create and add a new block widget, returning it."""
        channels = channels or self._channels
        idx = len(self._blocks)
        block = self._block_widget_class(idx, channels, parent=self._block_container, **kwargs)
        self._connect_block_signals(block)
        self._compact_block(block)
        # Move add button to bottom
        self._blocks_layout.removeItem(self._add_btn_layout)
        if self._blocks:
            self._blocks_layout.addSpacing(4)
        self._blocks.append(block)
        self._blocks_layout.addWidget(block)
        self._blocks_layout.addLayout(self._add_btn_layout)
        return block

    def _remove_block_generic(self, block: QWidget) -> None:
        """Remove a block widget, keeping at least one."""
        if len(self._blocks) <= 1:
            return
        self._blocks.remove(block)
        self._blocks_layout.removeWidget(block)
        block.deleteLater()
        self._reindex_blocks()
        self.parameter_changed.emit()

    def _reindex_blocks(self) -> None:
        """Re-index block indices after removal."""
        for i, b in enumerate(self._blocks):
            b.block_index = i

    def _connect_block_signals(self, block: QWidget) -> None:
        """Wire the remove button and parameter signals. Override to add more."""
        block._remove_btn.clicked.connect(
            lambda checked, b=block: self._remove_block_generic(b)
        )

    def _remove_all_blocks(self) -> None:
        """Remove all block widgets from layout and list."""
        for block in list(self._blocks):
            self._blocks.remove(block)
            self._blocks_layout.removeWidget(block)
            block.deleteLater()

    # ── Serialization: structured list-of-dicts format (load_config_section) ──

    def load_config_section(self, sections: Any) -> None:
        if not sections:
            return
        if isinstance(sections, dict):
            sections = [sections]
        if not isinstance(sections, (list, tuple)):
            return
        self._remove_all_blocks()
        self._blocks_layout.removeItem(self._add_btn_layout)

        for cfg in sections:
            if not isinstance(cfg, dict):
                continue
            if self._blocks:
                self._blocks_layout.addSpacing(4)
            block = self._block_widget_class(len(self._blocks), self._channels, parent=self._block_container)
            self._connect_block_signals(block)
            self._apply_block_config(block, cfg)
            self._compact_block(block)
            self._blocks.append(block)
            self._blocks_layout.addWidget(block)

        self._blocks_layout.addLayout(self._add_btn_layout)
        self.parameter_changed.emit()

    def _apply_block_config(self, block: QWidget, cfg: dict) -> None:
        """Apply a structured config dict to a block. Override for per-panel logic."""

    # ── Helpers ──────────────────────────────────────────────────────

    def to_config(self) -> Optional[Any]:
        configs = self.build_config_section()
        if configs is None:
            return None
        return {
            "run": self.isChecked(),
            "configs": configs if isinstance(configs, list) else [configs],
        }

    def from_config(self, section: Any) -> None:
        if isinstance(section, dict):
            run_val = section.get("run")
            if run_val is not None:
                self.setChecked(bool(run_val) if not isinstance(run_val, str) else run_val.lower() in ("1", "true", "yes"))
            configs = section.get("configs", section)
            if isinstance(configs, list):
                self.load_config_section(configs)
            elif isinstance(configs, dict):
                self.load_config_section([configs])
        elif isinstance(section, list):
            self.load_config_section(section)
