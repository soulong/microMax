from __future__ import annotations

from typing import Any, List, Optional, Type

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from microProfiler.gui.panels.base_step_panel import BaseStepPanel


class BlockContainerPanel(BaseStepPanel):
    """Base class for step panels that manage a dynamic list of block widgets.

    Provides shared layout management, block add/remove, deferred-restore
    serialization, and channel/mask repopulation:

    - ``load_config_section`` builds blocks from structured config dicts and
      stashes them as ``_pending_block_configs``. Widget state that depends on
      a loaded dataset (channel checkboxes, mask combos) is applied by
      re-running ``_apply_block_config`` after ``populate_channels`` /
      ``populate_masks`` — the config dict is the single source of truth, so
      panels never hand-roll comma-joined stash formats.
    - ``_apply_block_config(block, cfg)`` is the ONLY config→widget mapping a
      subclass writes. It must be idempotent and safe to call before the
      dataset-driven widgets exist (empty checkbox lists are harmless).
    - ``_extra_config_items()`` / ``_apply_extra_config_items()`` let a panel
      carry extra top-level section keys (e.g. object_profile's ``n_workers``)
      without overriding ``to_config``/``from_config``.

    Subclasses must set ``_block_widget_class`` and call
    ``_build_block_container()`` in ``__init__``.
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
        # Deferred-restore state: structured configs waiting for a dataset to
        # load, and the last-known channel/mask lists (kept across restores).
        self._pending_block_configs: List[dict] = []
        self._last_channels: List[str] = []
        self._last_masks: List[str] = []
        # While True, _pending_block_configs are re-applied on every dataset-
        # driven repopulation. It stays active until BOTH channels and masks
        # have real data (dataset loaded), then turns off so later
        # repopulations (filter edits, post-run refresh) never resurrect the
        # restored config over the user's current selections.
        self._restore_active: bool = False

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
        self._compact_block(
            block, excluded=getattr(self, "_compact_excluded_object_names", ()) or ())
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
        """Wire the remove button and parameter signals. Override to add more.

        The remove button is connected HERE (once) — block widgets must NOT
        connect it in their own constructors.
        """
        block._remove_btn.clicked.connect(
            lambda checked, b=block: self._remove_block_generic(b)
        )

    def _remove_all_blocks(self) -> None:
        """Remove all block widgets from layout and list."""
        for block in list(self._blocks):
            self._blocks.remove(block)
            self._blocks_layout.removeWidget(block)
            block.deleteLater()

    # ── Deferred restore ─────────────────────────────────────────────────

    def _reapply_pending_configs(self) -> None:
        """Re-run _apply_block_config for the stashed configs.

        Called after populate_channels/populate_masks rebuild the
        dataset-driven widgets — at that point the config can be applied to
        real checkbox/mask widgets. The stash is NOT consumed: restore stays
        active until both channels and masks have been populated with real
        data (see populate_channels/populate_masks), so a channels-first
        repopulation can never eat the config before masks arrive.
        """
        if not self._restore_active:
            return
        for i, block in enumerate(self._blocks):
            if i < len(self._pending_block_configs):
                self._apply_block_config(block, self._pending_block_configs[i])

    def _maybe_finish_restore(self) -> None:
        """Turn restore off once both channels and masks carry real data."""
        if self._restore_active and self._channels and self._last_masks:
            self._restore_active = False

    # ── Serialization: structured list-of-dicts format (load_config_section) ──

    def load_config_section(self, sections: Any) -> None:
        if not sections:
            return
        if isinstance(sections, dict):
            sections = [sections]
        if not isinstance(sections, (list, tuple)):
            return
        # Keep the structured configs so populate_channels/populate_masks can
        # re-apply channel + mask selections when the dataset loads (at
        # restore time those widgets don't exist yet).
        self._pending_block_configs = [cfg for cfg in sections if isinstance(cfg, dict)]
        self._restore_active = True
        self._remove_all_blocks()
        self._blocks_layout.removeItem(self._add_btn_layout)

        last_channels = self._last_channels or self._channels
        last_masks = self._last_masks

        for cfg in sections:
            if not isinstance(cfg, dict):
                continue
            if self._blocks:
                self._blocks_layout.addSpacing(4)
            block = self._block_widget_class(len(self._blocks), last_channels, parent=self._block_container)
            self._connect_block_signals(block)
            self._apply_block_config(block, cfg)
            self._compact_block(
            block, excluded=getattr(self, "_compact_excluded_object_names", ()) or ())
            self._blocks.append(block)
            self._blocks_layout.addWidget(block)

        self._blocks_layout.addLayout(self._add_btn_layout)
        if last_channels:
            self.populate_channels(last_channels)
        elif last_masks:
            self.populate_masks(last_masks)
        self.parameter_changed.emit()

    def _apply_block_config(self, block: QWidget, cfg: dict) -> None:
        """Apply a structured config dict to a block. Override for per-panel logic.

        Must be idempotent and tolerant of dataset-driven widgets not existing
        yet (empty channel lists / mask combos) — it is re-run after
        populate_channels/populate_masks.
        """

    # ── Dataset-driven repopulation ─────────────────────────────────────

    def populate_channels(self, channels: List[str]) -> None:
        """Rebuild channel-driven widgets, then re-apply pending configs.

        Subclass block widgets may implement ``populate_channels`` (called via
        the ``_on_channels_changed`` hook) or rebuild their own rows.
        """
        self._last_channels = list(channels)
        self._channels = list(channels)
        self._on_channels_changed(channels)
        self._reapply_pending_configs()
        self._maybe_finish_restore()
        self.parameter_changed.emit()

    def _on_channels_changed(self, channels: List[str]) -> None:
        """Hook: distribute the channel list to block widgets.

        Rebuilt checkboxes are re-wired to parameter_changed (UniqueConnection
        makes the repeated wiring harmless).
        """
        for block in self._blocks:
            fn = getattr(block, "populate_channels", None)
            if fn is not None:
                fn(channels)
                for cb in block.findChildren(QCheckBox):
                    self._wire_param_signal(cb)

    def populate_masks(self, mask_names: List[str]) -> None:
        """Rebuild mask-driven widgets, then re-apply pending configs."""
        self._last_masks = list(mask_names)
        for block in self._blocks:
            fn = getattr(block, "populate_masks", None)
            if fn is not None:
                fn(mask_names)
        self._reapply_pending_configs()
        self._maybe_finish_restore()
        self.parameter_changed.emit()

    # ── Extra top-level section keys ─────────────────────────────────────

    def _extra_config_items(self) -> dict:
        """Extra top-level keys merged into to_config()'s section dict."""
        return {}

    def _apply_extra_config_items(self, section: dict) -> None:
        """Read extra top-level keys from a restored section dict."""

    def to_config(self) -> Optional[Any]:
        configs = self.build_config_section()
        if configs is None:
            return None
        section: dict = {
            "run": self.isChecked(),
            "configs": configs if isinstance(configs, list) else [configs],
        }
        section.update(self._extra_config_items())
        return section

    def from_config(self, section: Any) -> None:
        if isinstance(section, list):
            self.load_config_section(section)
            return
        if not isinstance(section, dict):
            return
        run_val = section.get("run")
        if isinstance(run_val, bool):
            # Strict typing: a session.yml 'run' must be a real bool (the GUI
            # only ever writes real booleans).
            self.setChecked(run_val)
        self._apply_extra_config_items(section)
        # A section without a `configs` list carries no block state (e.g.
        # session.yml written by another tool, or a bare {"run": ...}) — keep
        # the existing blocks so their widget defaults survive the restore.
        if "configs" not in section:
            return
        configs = section["configs"]
        if isinstance(configs, list):
            self.load_config_section(configs)
        elif isinstance(configs, dict):
            self.load_config_section([configs])

    def build_config_section(self) -> list:  # type: ignore[override]
        return [b.build_config_section() for b in self._blocks]