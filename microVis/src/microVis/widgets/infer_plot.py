"""Infer DB scatter tab: DR coordinates from a microModel infer.db.

Reads the joined `inference` + `reduction_<method>` (+ `find_cluster`) frame
via InferDB, filters to the selected source directory, and renders a facet-
aware scatter where color (one variable) and size (one numeric variable) are
user-bound. Exports vector PDFs with editable text.

Layout, background loading, figure lifecycle, info line and export live in
``plot_tab_base.PlotTabBase``; this file only adds the infer-specific
method/directory pickers and the scatter rendering.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from PySide6.QtWidgets import QComboBox, QFormLayout, QMessageBox, QWidget

from microVis.io.data_module import merge_metadata
from microVis.io.infer_db import InferDB, is_numeric_type
from microVis.processing import plotting as P
from microVis.widgets.plot_tab_base import (
    NONE_ITEM,
    PlotTabBase,
    _make_picker_combo,
)

_ALL = "(all directories)"


class InferPlotView(PlotTabBase):
    """Scatter builder for one infer.db (one DR method at a time)."""

    empty_load_message = "No rows for this reduction method."

    def __init__(self, parent: QWidget | None = None):
        self._dataset_dir = ""
        self._columns: list[tuple[str, str]] = []
        # Base columns (without metadata) so Merge/Clear can add/remove the
        # metadata entries without re-reading the DB.
        self._base_columns: list[tuple[str, str]] = []
        self._dataset_scopes: list[str] = []
        super().__init__(parent)

    # ── PlotTabBase hooks ─────────────────────────────────────────────────

    def _build_controls(self, form: QFormLayout) -> None:
        self._method_combo = _make_picker_combo()
        form.addRow("Reduction:", self._method_combo)

        self._dir_combo = _make_picker_combo()
        self._dir_combo.setToolTip(
            "Filter rows by the source directory stored in the infer DB.\n"
            "Type to filter long directory lists.")
        form.addRow("Directory:", self._dir_combo)

        self._add_color_size_rows(form)
        self._add_facet_rows(form)
        self._add_colors_cap_filter_rows(
            form,
            cap_label="Point cap:",
            filter_placeholder='pred_class == "x" & pred_prob > 0.8',
            filter_tooltip=(
                "pandas query expression applied before plotting.\n"
                "Operators: > < >= <= == != & | ~ ( )\n"
                "Strings: pred_class == \"x\".\n"
                "Column names with spaces/dots need backticks: `my col` > 1, "
                "`cluster_res_0.5` == 2.\n"
                "Leave empty for no filter."),
        )

    def _on_plot(self) -> None:
        if self._db is None:
            return
        method = self._method_combo.currentText()
        if not method:
            return
        self.info(f"Loading reduction_{method}...")
        self._start_load(lambda: self._db.load_scatter(method))

    def _handle_loaded(self, df: pd.DataFrame) -> None:
        df = merge_metadata(df, self._metadata)
        directory = self._dir_combo.currentText()
        if directory and directory != _ALL and "directory" in df.columns:
            df = df[df["directory"].astype(str) == directory]
        if df.empty:
            self.info("No rows for the selected directory.")
            return
        total_rows = len(df)
        df, err = P.apply_filter(df, self._filter_edit.text())
        if err:
            QMessageBox.warning(self, "Invalid Filter", err)
            self.info(f"Invalid filter: {err}", error=True)
            return
        if df.empty:
            self.info("No rows left after filtering.", error=True)
            return
        method = self._method_combo.currentText()
        c1, c2 = self._db.coord_columns(method)
        color = self._selected(self._color_combo)
        size = self._selected(self._size_combo)
        cap = self._cap_spin.value()
        facets = self._facet_selected()
        ncols = self._facet_cols.value()
        sub, sampled, total = P.apply_point_cap(df, cap)
        try:
            fig = P.make_scatter(
                sub, x=c1, y=c2, color=color, size=size,
                facet_cols=facets, ncols=ncols,
                palette=self._colors_combo.currentText(),
                cmap=self._colors_combo.currentText(),
                point_size=self._base_size.value(),
                title=f"{self._db_title}: reduction_{method}",
                # The reduction coordinates are the axes — no need to repeat
                # them in every hover tooltip.
                hover_exclude=(c1, c2),
            )
        except Exception as e:
            QMessageBox.warning(self, "Plot Failed", str(e))
            self.info(f"Plot failed: {e}", error=True)
            return
        filtered = f" (filtered {len(df):,} of {total_rows:,})" if len(df) != total_rows else ""
        note = (f"Sampled {len(sub):,} of {total:,} rows (cap {cap:,}; "
                f"set 0 for all)." if sampled else f"{len(sub):,} rows.")
        self.info(f"reduction_{method}: {note}{filtered}")
        self._set_figure(fig)

    def _column_is_numeric(self, name: str) -> bool:
        for cname, ctype in self._columns:
            if cname == name:
                return is_numeric_type(ctype)
        return False

    def _default_export_name(self) -> str:
        method = self._method_combo.currentText() or "reduction"
        return f"infer_{method}.pdf"

    def _clear_tab(self) -> None:
        self._columns = []
        self._base_columns = []
        self._method_combo.clear()
        self._dir_combo.clear()
        self._color_combo.clear()
        self._size_combo.clear()

    def _on_metadata_changed(self) -> None:
        self._columns = list(self._base_columns)
        if self._metadata is not None:
            self._append_metadata_columns()
        self._populate_column_combos()

    # ── DB wiring ─────────────────────────────────────────────────────────

    def set_db(self, db: InferDB, dataset_scopes=None) -> None:
        """Attach an infer DB and auto-plot its first reduction method.

        ``dataset_scopes`` identifies the current dataset's rows in the DB's
        `directory` column (absolute forward-slash values) so the directory
        combo preselects them.
        """
        self._db = db
        self._db_title = Path(db.path).name
        if isinstance(dataset_scopes, str):
            dataset_scopes = [dataset_scopes] if dataset_scopes else []
        self._dataset_scopes = [str(s) for s in (dataset_scopes or [])]
        self._base_columns = db.list_columns()
        self._columns = list(self._base_columns)
        if self._metadata is not None:
            self._append_metadata_columns()

        directories = db.list_directories()
        self._dir_combo.blockSignals(True)
        self._dir_combo.clear()
        self._dir_combo.addItem(_ALL)
        self._dir_combo.addItems(directories)
        self._dir_combo.blockSignals(False)
        default = self._default_directory(directories)
        if default is not None:
            self._dir_combo.setCurrentText(default)

        methods = db.list_reduction_methods()
        self._method_combo.blockSignals(True)
        self._method_combo.clear()
        self._method_combo.addItems(methods)
        self._method_combo.blockSignals(False)

        self._populate_column_combos()

        if not methods:
            self._plot_btn.setEnabled(False)
            self.info("No reduction tables in this DB — run "
                      "`micromodel reduction` (or reduction in the inference "
                      "pipeline) first.")
            return
        self._plot_btn.setEnabled(True)
        self._on_plot()

    def _default_directory(self, directories: list[str]) -> str | None:
        """Pick the current dataset's directory (exact, then prefix match).

        Stored directories are absolute forward-slash paths; scopes come from
        the current dataset's metadata, so exact matches are the normal case.
        The comparison normalizes separators and case.
        """
        if not self._dataset_scopes:
            return None

        def _norm(v) -> str:
            return os.path.normcase(str(v).replace("\\", "/")).rstrip("/")

        scopes = [_norm(s) for s in self._dataset_scopes]
        for d in directories:
            if _norm(d) in scopes:
                return d
        for d in directories:
            nd = _norm(d)
            if any(s and nd.startswith(s + "/") for s in scopes):
                return d
        return _ALL

    def _append_metadata_columns(self) -> None:
        """Add metadata columns missing from the current column list."""
        existing = {c for c, _t in self._columns}
        for c in self._metadata.columns:
            if c == "well" or c in existing:
                continue
            self._columns.append((c, str(self._metadata[c].dtype)))

    def _populate_column_combos(self) -> None:
        names = [c for c, _t in self._columns]
        numeric = [c for c, _t in self._columns if is_numeric_type(_t)]

        self._fill_combo(self._color_combo, [NONE_ITEM] + names, keep=None)
        self._fill_combo(self._size_combo, [NONE_ITEM] + numeric, keep=None)
        self._populate_facet_list(names)
        self._on_color_changed(self._color_combo.currentText())

    def _selected(self, combo: QComboBox) -> str | None:
        text = combo.currentText()
        return None if not text or text == NONE_ITEM else text
