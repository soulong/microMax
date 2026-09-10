"""Profiler DB plot tab: free-form plots over a microProfiler profiler.db table.

Controls: object/table, chart type (scatter / line mean±SEM / boxplot /
barplot mean±SEM), Y, X, color-by, size-by (scatter), facets (full
combination of the selected variables), palette/colormap, and a scatter
point cap. All figures export as vector PDFs with editable text (Type-42
fonts).

Layout, background loading, figure lifecycle, info line and export live in
``plot_tab_base.PlotTabBase``; this file only adds the profiler-specific
pickers and the per-chart rendering.
"""

from __future__ import annotations

import pandas as pd

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QMessageBox,
    QWidget,
)

from microVis.io.data_module import merge_metadata
from microVis.io.profiler_db import ProfilerDB
from microVis.processing import plotting as P
from microVis.widgets.plot_tab_base import (
    NONE_ITEM,
    PlotTabBase,
    _make_picker_combo,
)

_CHART_SCATTER = "scatter"
_CHART_LINE = "line"
_CHART_BOXPLOT = "boxplot"
_CHART_BARPLOT = "barplot"

_CHART_ORDER = [_CHART_SCATTER, _CHART_LINE, _CHART_BOXPLOT, _CHART_BARPLOT]


class ProfilerPlotView(PlotTabBase):
    """Plot builder for one profiler DB (all tables, incl. whole-image)."""

    def __init__(self, parent: QWidget | None = None):
        self._columns: list[tuple[str, str, bool]] = []
        # Table + metadata columns as shown in the pickers (same tuple shape).
        self._display_columns: list[tuple[str, str, bool]] = []
        super().__init__(parent)
        self._on_chart_changed(self._chart_combo.currentText())

    # ── PlotTabBase hooks ─────────────────────────────────────────────────

    def _build_controls(self, form: QFormLayout) -> None:
        self._object_combo = _make_picker_combo()
        self._object_combo.currentTextChanged.connect(self._on_object_changed)
        form.addRow("Object:", self._object_combo)

        self._chart_combo = QComboBox()
        self._chart_combo.addItems(_CHART_ORDER)
        self._chart_combo.currentTextChanged.connect(self._on_chart_changed)
        form.addRow("Chart:", self._chart_combo)

        self._x_combo = _make_picker_combo()
        self._x_combo.setToolTip(
            "Any variable (meta or measurement). Categorical levels are placed "
            "on their level index with the level names as axis ticks.")
        form.addRow("X:", self._x_combo)

        self._y_combo = _make_picker_combo()
        self._y_combo.setToolTip(
            "Any variable (meta or measurement). Categorical levels are placed "
            "on their level index with the level names as axis ticks.")
        form.addRow("Y:", self._y_combo)

        self._add_color_size_rows(form)
        self._add_facet_rows(form)
        self._add_colors_cap_filter_rows(
            form,
            cap_label="Scatter cap:",
            filter_placeholder="shape_area > 200 & intensity_mean_ch1 < 5000",
            filter_tooltip=(
                "pandas query expression applied before plotting.\n"
                "Operators: > < >= <= == != & | ~ ( )\n"
                "Strings: pred_class == \"x\".\n"
                "Column names with spaces/dots need backticks: `my col` > 1, "
                "`intensity_q0.1_ch1` < 5.\n"
                "Leave empty for no filter."),
        )

        self._show_points = QCheckBox("Show points on boxplot")
        self._show_points.setChecked(True)
        form.addRow("", self._show_points)

    def _on_plot(self) -> None:
        if self._db is None or not self._object_combo.currentText():
            return
        table = self._object_combo.currentText()
        chart = self._chart_combo.currentText()
        y = self._selected(self._y_combo)
        x = self._selected(self._x_combo)
        if not y:
            QMessageBox.information(self, "Nothing to Plot",
                                    "Choose a Y variable first.")
            return
        if chart == _CHART_SCATTER and not x:
            QMessageBox.information(self, "Nothing to Plot",
                                    "Scatter needs both X and Y variables.")
            return
        if chart == _CHART_LINE and not x:
            QMessageBox.information(self, "Nothing to Plot",
                                    "Line needs an X variable (the connected axis).")
            return
        self.info("Loading %s..." % table)
        # merge_metadata runs off the GUI thread inside the worker.
        self._start_load(
            lambda: merge_metadata(self._db.load_table(table), self._metadata))

    def _handle_loaded(self, df: pd.DataFrame) -> None:
        spec = self._current_spec()
        if spec is None:
            return
        try:
            self._render(df, spec)
        except Exception as e:
            QMessageBox.warning(self, "Plot Failed", str(e))
            self.info(f"Plot failed: {e}")

    def _column_is_numeric(self, name: str) -> bool:
        for cname, _t, is_num in self._display_columns:
            if cname == name:
                return is_num
        return False

    def _default_export_name(self) -> str:
        table = self._object_combo.currentText() or "plot"
        return f"{table}_{self._chart_combo.currentText().split()[0]}.pdf"

    def _clear_tab(self) -> None:
        self._columns = []
        self._display_columns = []
        self._object_combo.blockSignals(True)
        self._object_combo.clear()
        self._object_combo.blockSignals(False)
        self._on_object_changed("")

    # ── DB / table wiring ─────────────────────────────────────────────────

    def set_db(self, db: ProfilerDB, title: str = "") -> None:
        """Attach one profiler DB; repopulates object + column pickers."""
        self._db = db
        self._db_title = title or getattr(db.path, "name", str(db.path))
        self._plot_btn.setEnabled(True)
        self._populate_objects()

    def refresh_tables(self) -> None:
        self._populate_objects()

    def set_metadata(self, metadata) -> None:
        """Attach/detach merged Excel metadata (joined by `well` on load)."""
        super().set_metadata(metadata)
        # Refresh the pickers so metadata columns become selectable.
        self._on_object_changed(self._object_combo.currentText())

    def _populate_objects(self) -> None:
        current = self._object_combo.currentText()
        self._object_combo.blockSignals(True)
        self._object_combo.clear()
        tables: list[str] = []
        if self._db is not None:
            tables = self._db.list_tables()
        self._object_combo.addItems(tables)
        self._object_combo.blockSignals(False)
        if current in tables:
            self._object_combo.setCurrentText(current)
        self._on_object_changed(self._object_combo.currentText())

    def _on_object_changed(self, table: str) -> None:
        self._columns = []
        if self._db is not None and table:
            self._columns = self._db.table_columns(table)
        self._populate_columns()

    def _populate_columns(self) -> None:
        # Table columns plus the merged metadata columns (missing ones only,
        # so a DB that already stores metadata never lists duplicates).
        columns = list(self._columns)
        if self._metadata is not None:
            existing = {c for c, _t, _n in columns}
            for c in self._metadata.columns:
                if c == "well" or c in existing:
                    continue
                is_num = bool(pd.api.types.is_numeric_dtype(self._metadata[c]))
                columns.append((c, str(self._metadata[c].dtype), is_num))
        self._display_columns = columns
        names = [c for c, _t, _n in columns]
        numeric = [c for c, _t, is_num in columns if is_num]

        # X and Y both offer every variable (meta + measurements); categorical
        # axes are plotted on their level index with level-name ticks.
        self._fill_combo(self._x_combo, names)
        self._fill_combo(self._y_combo, names)
        self._fill_combo(self._color_combo, [NONE_ITEM] + names, keep=None)
        self._fill_combo(self._size_combo, [NONE_ITEM] + numeric, keep=None)
        self._populate_facet_list(names)

        self._on_color_changed(self._color_combo.currentText())

    # ── Control behavior ──────────────────────────────────────────────────

    def _on_chart_changed(self, chart: str) -> None:
        # scatter: size + point cap apply. line: points are always drawn and
        # uncapped. boxplot: the show-points toggle applies. barplot: none.
        is_scatter = chart == _CHART_SCATTER
        self._size_combo.setEnabled(is_scatter)
        self._base_size.setEnabled(True)
        self._cap_spin.setEnabled(is_scatter)
        self._show_points.setEnabled(chart == _CHART_BOXPLOT)
        self._x_combo.setToolTip(
            "X is required for scatter and line (the connected axis); "
            "optional grouping for box/bar."
            if is_scatter or chart == _CHART_LINE else
            "Optional grouping variable for box/bar.")

    def _selected(self, combo: QComboBox) -> str | None:
        text = combo.currentText()
        return None if not text or text == NONE_ITEM else text

    def _current_spec(self) -> dict | None:
        """Rebuild the spec from the live controls (used at render time)."""
        chart = self._chart_combo.currentText()
        return {
            "table": self._object_combo.currentText(),
            "chart": chart,
            "y": self._selected(self._y_combo),
            "x": self._selected(self._x_combo),
            "color": self._selected(self._color_combo),
            "size": (self._selected(self._size_combo)
                     if chart == _CHART_SCATTER else None),
            "facets": self._facet_selected(),
            "palette": self._colors_combo.currentText(),
            "ncols": self._facet_cols.value(),
            "cap": self._cap_spin.value(),
            "base_size": self._base_size.value(),
            "show_points": self._show_points.isChecked(),
            "filter": self._filter_edit.text(),
        }

    def _render(self, df: pd.DataFrame, spec: dict) -> None:
        y = spec["y"]
        x = spec["x"]
        color = spec["color"]
        facets = spec["facets"]
        palette = spec["palette"]
        ncols = spec["ncols"]
        title = f"{spec['table']}: {y}" + (f" by {x}" if x else "")

        total_rows = len(df)
        df, err = P.apply_filter(df, spec.get("filter", ""))
        if err:
            QMessageBox.warning(self, "Invalid Filter", err)
            self.info(f"Invalid filter: {err}", error=True)
            return
        if df.empty:
            self.info("No rows left after filtering.", error=True)
            return
        filtered = f" (filtered {len(df):,} of {total_rows:,})" if len(df) != total_rows else ""

        if spec["chart"] == _CHART_SCATTER:
            cap = spec["cap"]
            sub, sampled, total = P.apply_point_cap(df, cap)
            fig = P.make_scatter(
                sub, x=x, y=y, color=color, size=spec["size"],
                facet_cols=facets, palette=palette, cmap=palette,
                point_size=spec["base_size"], ncols=ncols, title=title)
            if sampled:
                self.info(f"Scatter sampled {len(sub):,} of {total:,} rows "
                          f"(cap {cap:,}; set 0 for all).{filtered}")
            else:
                self.info(f"Scatter: {len(sub):,} rows.{filtered}")
        elif spec["chart"] == _CHART_LINE:
            fig = P.make_line(
                df, y=y, x=x, color=color, facet_cols=facets,
                palette=palette, ncols=ncols, title=title)
            self.info(f"Line (mean ± SEM): {len(df):,} rows.{filtered}")
        elif spec["chart"] == _CHART_BOXPLOT:
            fig = P.make_boxplot(
                df, y=y, x=x, color=color, facet_cols=facets,
                palette=palette, show_points=spec["show_points"],
                ncols=ncols, title=title)
            self.info(f"Boxplot: {len(df):,} rows.{filtered}")
        else:
            fig = P.make_barplot_mean_sem(
                df, y=y, x=x, color=color, facet_cols=facets,
                palette=palette, ncols=ncols, title=title)
            self.info(f"Barplot (mean ± SEM): {len(df):,} rows.{filtered}")
        self._set_figure(fig)
