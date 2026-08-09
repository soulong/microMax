from __future__ import annotations

import logging
from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from microProfiler.gui.panels.base_step_panel import BaseStepPanel
from microProfiler.gui.state import PipelineState
from microProfiler.io import clone_dataset

logger = logging.getLogger(__name__)


class FilterPanel(BaseStepPanel):

    step_name = "filter"
    filter_changed = Signal()

    def __init__(self, state: PipelineState, parent=None):
        super().__init__(state, parent)
        self.setTitle("Filter")
        self._updating = False
        self._filter_widgets: List[Tuple[QComboBox, QLineEdit, QPushButton]] = []
        self._build_controls()
        self.setChecked(False)
        super().setCheckable(False)

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_controls(self) -> None:
        self._filters_container = QWidget()
        self._filters_layout = QVBoxLayout(self._filters_container)
        self._filters_layout.setContentsMargins(0, 0, 0, 0)
        self._filters_layout.setSpacing(4)
        self._filters_layout.setAlignment(Qt.AlignTop)
        self._controls_layout.addWidget(self._filters_container, 1)
        self._filters_layout.addStretch(1)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("+ Add Filter")
        self._add_btn.setProperty("class", "secondary")
        self._add_btn.clicked.connect(self._add_filter_row)
        self._reset_btn = QPushButton("Reset All")
        self._reset_btn.setProperty("class", "secondary")
        self._reset_btn.clicked.connect(self._reset_filters)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._reset_btn)
        btn_row.addStretch()
        self._controls_layout.addLayout(btn_row)

    # ── Filter rows ─────────────────────────────────────────────────────

    def _add_filter_row(self, column: str = "", pattern: str = "") -> None:
        meta_cols = self._get_filterable_columns()
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        col_combo = QComboBox()
        col_combo.addItems(meta_cols)
        if column and column in meta_cols:
            col_combo.setCurrentText(column)
        elif column:
            # The saved column is absent from this dataset (e.g. different
            # dataset or changed pattern). Never silently apply the pattern
            # to the first column — surface the mismatch in the dropdown.
            col_combo.addItem(f"(missing: {column})", column)
            col_combo.setCurrentIndex(col_combo.count() - 1)
            logger.warning(
                "Filter column '%s' not found in dataset columns; marked as missing "
                "(pattern not applied)", column)
        col_combo.setMaximumWidth(160)
        pat_edit = QLineEdit(pattern)
        pat_edit.setPlaceholderText("regex pattern…")
        pat_edit.setClearButtonEnabled(True)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(28, 28)
        remove_btn.setProperty("class", "danger")
        row.addWidget(col_combo)
        row.addWidget(pat_edit, 1)
        row.addWidget(remove_btn)
        widgets = (col_combo, pat_edit, remove_btn)
        self._filter_widgets.append(widgets)
        idx = self._filters_layout.count() - 1
        self._filters_layout.insertWidget(idx, row_widget)
        col_combo.currentTextChanged.connect(self._on_filter_changed)
        pat_edit.textChanged.connect(self._on_filter_changed)
        remove_btn.clicked.connect(lambda: self._remove_filter_row(widgets, row_widget))

    def _remove_filter_row(
        self,
        widgets: Tuple[QComboBox, QLineEdit, QPushButton],
        row_widget: QWidget,
    ) -> None:
        if widgets not in self._filter_widgets:
            return
        self._filter_widgets.remove(widgets)
        self._filters_layout.removeWidget(row_widget)
        row_widget.deleteLater()
        if not self._updating:
            self._apply_filters()

    def _clear_filter_rows(self) -> None:
        for combo, edit, btn in self._filter_widgets:
            combo.deleteLater()
            edit.deleteLater()
            btn.deleteLater()
        self._filter_widgets.clear()
        while self._filters_layout.count():
            item = self._filters_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _get_filterable_columns(self) -> List[str]:
        ds = self._state.original_dataset or self._state.dataset
        if ds is None or ds.metadata is None:
            return []
        skip = {"directory"}
        return [
            c for c in ds.metadata.columns
            if c not in skip and not c.startswith("ch") and not c.startswith("mask_")
        ]

    # ── Apply / Reset ───────────────────────────────────────────────────

    def _on_filter_changed(self, *_args) -> None:
        if not self._updating:
            self._apply_filters()

    def _apply_filters(self) -> None:
        if self._updating:
            return
        orig = self._state.original_dataset
        if orig is None:
            return
        ds = clone_dataset(orig)
        for combo, edit, _btn in self._filter_widgets:
            col = combo.currentText()
            pat = edit.text().strip()
            if col.startswith("(missing: "):
                # Saved column absent from this dataset — the row is a marker,
                # never apply its pattern to a different column.
                continue
            if col and pat:
                try:
                    ds.filter_metadata(col, pat)
                except Exception as exc:
                    logger.warning("Filter regex error for column '%s' pattern '%s': %s", col, pat, exc)
        self._state.dataset = ds
        self.filter_changed.emit()

    def _reset_filters(self) -> None:
        if self._state.original_dataset is not None:
            self._state.dataset = clone_dataset(self._state.original_dataset)
        self._updating = True
        self._clear_filter_rows()
        self._updating = False
        self.filter_changed.emit()

    # ── Config persistence ─────────────────────────────────────────────

    def load_from_settings(self, settings) -> None:
        stored = settings.get(self.step_name, {})
        if not stored:
            return
        filters = stored.get("filters", [])
        self._updating = True
        self._clear_filter_rows()
        for f in filters:
            self._add_filter_row(
                column=f.get("column", ""),
                pattern=f.get("pattern", ""),
            )
        self._updating = False
        self._apply_filters()

    def to_config(self) -> dict:
        filters = []
        for combo, edit, _btn in self._filter_widgets:
            # Missing-column marker rows carry the real column in userData;
            # they are excluded here so the pipeline never filters on a
            # column that does not exist in the dataset.
            col = combo.currentText()
            if col.startswith("(missing: "):
                continue
            pat = edit.text().strip()
            if col:
                filters.append({"column": col, "pattern": pat})
        return {"filters": filters}

    def from_config(self, section: dict) -> None:
        if not section:
            return
        self.load_from_settings({self.step_name: section})
