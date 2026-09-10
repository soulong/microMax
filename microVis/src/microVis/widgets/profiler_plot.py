"""Profiler DB plot tab: free-form plots over a microProfiler profiler.db table.

Controls: object/table, chart type (boxplot / barplot mean+SD / scatter), Y,
X, color-by, size-by (scatter), facets (full combination of the selected
variables), palette/colormap, and a scatter point cap. All figures export as
vector PDFs with editable text (Type-42 fonts).
"""

from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import CMAP_OPTIONS, DEFAULT_CMAP, QUALITATIVE_PALETTES
from microVis.io.data_module import merge_metadata
from microVis.io.profiler_db import ProfilerDB
from microVis.processing import plotting as P

_CHART_BOXPLOT = "boxplot"
_CHART_BARPLOT = "barplot (mean ± SD)"
_CHART_SCATTER = "scatter"

_NONE = "(none)"


class _LoadSignals(QObject):
    loaded = Signal(int, object)
    error = Signal(int, str)


class _TableLoadWorker(QRunnable):
    """Read one profiler.db table off the GUI thread (ProfilerDB caches nothing)."""

    def __init__(self, db: ProfilerDB, table: str, gen: int, metadata=None):
        super().__init__()
        self.signals = _LoadSignals()
        self._db = db
        self._table = table
        self._gen = gen
        self._metadata = metadata

    def run(self):
        try:
            df = self._db.load_table(self._table)
            if df is None:
                self.signals.error.emit(self._gen, f"No data in table '{self._table}'")
            else:
                df = merge_metadata(df, self._metadata)
                self.signals.loaded.emit(self._gen, df)
        except Exception as e:  # surfaced as a GUI popup
            self.signals.error.emit(self._gen, str(e))


class ProfilerPlotView(QWidget):
    """Plot builder for one profiler DB (all tables, incl. whole-image)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._db: ProfilerDB | None = None
        self._db_title = ""
        self._columns: list[tuple[str, str, bool]] = []
        # Table + metadata columns as shown in the pickers (same tuple shape).
        self._display_columns: list[tuple[str, str, bool]] = []
        # Merged Excel metadata (set by MainWindow's Merge button) — joined by
        # `well` on load; None = no metadata merging.
        self._metadata = None
        self._figure = None
        self._canvas = None
        self._toolbar = None
        self._worker = None
        # In-flight workers are retained here (setAutoDelete(False)) so Qt's
        # C++ side cannot delete a running QRunnable mid-run; handlers
        # discard them. Using only self._worker dropped the reference when a
        # second Plot click replaced it.
        self._workers: set = set()
        self._gen = 0
        self._pool = QThreadPool.globalInstance()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Left column: controls (scrollable so small windows never clip) ──
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        # ── Controls ──
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)

        self._object_combo = QComboBox()
        self._object_combo.currentTextChanged.connect(self._on_object_changed)
        form.addRow("Object:", self._object_combo)

        self._chart_combo = QComboBox()
        self._chart_combo.addItems([_CHART_BOXPLOT, _CHART_BARPLOT, _CHART_SCATTER])
        self._chart_combo.currentTextChanged.connect(self._on_chart_changed)
        form.addRow("Chart:", self._chart_combo)

        self._x_combo = QComboBox()
        self._x_combo.setToolTip(
            "Any variable (meta or measurement). Categorical levels are placed "
            "on their level index with the level names as axis ticks.")
        form.addRow("X:", self._x_combo)

        self._y_combo = QComboBox()
        self._y_combo.setToolTip(
            "Any variable (meta or measurement). Categorical levels are placed "
            "on their level index with the level names as axis ticks.")
        form.addRow("Y:", self._y_combo)

        self._color_combo = QComboBox()
        self._color_combo.currentTextChanged.connect(self._on_color_changed)
        form.addRow("Color by:", self._color_combo)

        self._size_combo = QComboBox()
        form.addRow("Size by:", self._size_combo)

        size_row = QHBoxLayout()
        self._base_size = QDoubleSpinBox()
        self._base_size.setRange(1.0, 200.0)
        self._base_size.setValue(20.0)
        self._base_size.setSuffix(" pt")
        size_row.addWidget(self._base_size)
        size_row.addStretch()
        form.addRow("Base size:", size_row)

        self._facet_list = QListWidget()
        self._facet_list.setMaximumHeight(90)
        self._facet_list.setToolTip(
            "Checked variables are faceted by their FULL combination of levels "
            "(cartesian product), one subplot per combination.")
        form.addRow("Facet by:", self._facet_list)

        facet_cols_row = QHBoxLayout()
        self._facet_cols = QSpinBox()
        self._facet_cols.setRange(1, 8)
        self._facet_cols.setValue(3)
        facet_cols_row.addWidget(QLabel("grid columns:"))
        facet_cols_row.addWidget(self._facet_cols)
        facet_cols_row.addStretch()
        form.addRow("", facet_cols_row)

        colors_row = QHBoxLayout()
        self._colors_combo = QComboBox()
        colors_row.addWidget(self._colors_combo, 1)
        form.addRow("Colors:", colors_row)

        cap_row = QHBoxLayout()
        self._cap_spin = QSpinBox()
        self._cap_spin.setRange(0, 100_000_000)
        self._cap_spin.setValue(20000)
        self._cap_spin.setSingleStep(5000)
        self._cap_spin.setSuffix("  (0 = all)")
        cap_row.addWidget(self._cap_spin, 1)
        form.addRow("Scatter cap:", cap_row)

        filter_row = QHBoxLayout()
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText(
            "shape_area > 200 & intensity_mean_ch1 < 5000")
        self._filter_edit.setToolTip(
            "pandas query expression applied before plotting.\n"
            "Operators: > < >= <= == != & | ~ ( )\n"
            "Strings: pred_class == \"x\".\n"
            "Column names with spaces/dots need backticks: `my col` > 1, "
            "`intensity_q0.1_ch1` < 5.\n"
            "Leave empty for no filter.")
        self._filter_edit.returnPressed.connect(self._on_plot)
        filter_row.addWidget(self._filter_edit, 1)
        self._filter_clear_btn = QPushButton("Clear")
        self._filter_clear_btn.setProperty("class", "secondary")
        self._filter_clear_btn.setToolTip("Clear the filter expression")
        self._filter_clear_btn.clicked.connect(self._filter_edit.clear)
        filter_row.addWidget(self._filter_clear_btn)
        form.addRow("Filter:", filter_row)

        self._show_points = QCheckBox("Show points on boxplot")
        self._show_points.setChecked(True)
        form.addRow("", self._show_points)

        controls_layout.addLayout(form)

        btn_row = QHBoxLayout()
        self._plot_btn = QPushButton("Plot")
        self._plot_btn.setProperty("class", "primary")
        self._plot_btn.clicked.connect(self._on_plot)
        btn_row.addWidget(self._plot_btn)

        self._export_btn = QPushButton("Export PDF")
        self._export_btn.setProperty("class", "primary")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self._export_btn)
        btn_row.addStretch()
        controls_layout.addLayout(btn_row)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888888; font-size: 8pt;")
        controls_layout.addWidget(self._info_label)
        controls_layout.addStretch()

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setWidget(controls)
        controls_scroll.setFixedWidth(360)
        layout.addWidget(controls_scroll)

        # ── Right column: interactive plot canvas ──
        canvas_holder = QWidget()
        self._canvas_layout = QVBoxLayout(canvas_holder)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas_holder, 1)

        self._on_chart_changed(self._chart_combo.currentText())

    # ── DB / table wiring ─────────────────────────────────────────────────

    def set_db(self, db: ProfilerDB, title: str = "") -> None:
        """Attach one profiler DB; repopulates object + column pickers."""
        self._db = db
        self._db_title = title or getattr(db.path, "name", str(db.path))
        self._populate_objects()

    def refresh_tables(self) -> None:
        self._populate_objects()

    def set_metadata(self, metadata) -> None:
        """Attach/detach merged Excel metadata (joined by `well` on load)."""
        self._metadata = metadata
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

        def _fill(combo, items, keep=None):
            prev = keep if keep is not None else combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.blockSignals(False)
            if prev in items:
                combo.setCurrentText(prev)

        # X and Y both offer every variable (meta + measurements); categorical
        # axes are plotted on their level index with level-name ticks.
        _fill(self._x_combo, names)
        _fill(self._y_combo, names)
        _fill(self._color_combo, [_NONE] + names, keep=None)
        _fill(self._size_combo, [_NONE] + numeric, keep=None)

        # Facet list with checkboxes, preserving prior checks.
        checked = set(self._facet_selected())
        self._facet_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in checked else Qt.Unchecked)
            self._facet_list.addItem(item)

        self._on_color_changed(self._color_combo.currentText())

    # ── Control behavior ──────────────────────────────────────────────────

    def _on_chart_changed(self, chart: str) -> None:
        is_scatter = chart == _CHART_SCATTER
        self._size_combo.setEnabled(is_scatter)
        self._base_size.setEnabled(True)
        self._cap_spin.setEnabled(is_scatter)
        self._show_points.setEnabled(not is_scatter)
        self._x_combo.setToolTip(
            "X is required for scatter and optional grouping for box/bar."
            if is_scatter else
            "Optional grouping variable for box/bar.")

    def _on_color_changed(self, color: str) -> None:
        """Switch the colors combo between palettes and colormaps."""
        continuous = False
        if color and color != _NONE:
            for cname, _ctype, is_num in self._display_columns:
                if cname == color:
                    continuous = is_num
                    break
        prev = self._colors_combo.currentText()
        items = CMAP_OPTIONS if continuous else QUALITATIVE_PALETTES
        self._colors_combo.blockSignals(True)
        self._colors_combo.clear()
        self._colors_combo.addItems(items)
        self._colors_combo.blockSignals(False)
        default = DEFAULT_CMAP if continuous else items[0]
        if prev in items:
            self._colors_combo.setCurrentText(prev)
        else:
            self._colors_combo.setCurrentText(default)
        self._colors_combo.setToolTip(
            "Continuous color → matplotlib colormap" if continuous
            else "Categorical color → qualitative palette")

    def _facet_selected(self) -> list[str]:
        out = []
        for i in range(self._facet_list.count()):
            item = self._facet_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.text())
        return out

    # ── Plot ──────────────────────────────────────────────────────────────

    def _selected(self, combo: QComboBox) -> str | None:
        text = combo.currentText()
        return None if not text or text == _NONE else text

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
        self._gen += 1
        self.info("Loading %s..." % table)
        worker = _TableLoadWorker(self._db, table, self._gen, self._metadata)
        worker.signals.loaded.connect(self._on_table_loaded)
        worker.signals.error.connect(self._on_load_error)
        worker.setAutoDelete(False)
        self._worker = worker
        self._workers.add(worker)
        self._pool.start(worker)

    def _on_table_loaded(self, gen: int, df: pd.DataFrame) -> None:
        self._workers.discard(self._worker)
        self._worker = None
        if gen != self._gen:
            return
        if df is None or df.empty:
            self.info("Table is empty.")
            return
        spec = self._current_spec()
        if spec is None:
            return
        try:
            self._render(df, spec)
        except Exception as e:
            QMessageBox.warning(self, "Plot Failed", str(e))
            self.info(f"Plot failed: {e}")

    def _on_load_error(self, gen: int, msg: str) -> None:
        self._workers.discard(self._worker)
        self._worker = None
        if gen != self._gen:
            return
        QMessageBox.warning(self, "Plot Failed", msg)
        self.info(f"Load failed: {msg}")

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
        elif spec["chart"] == _CHART_BOXPLOT:
            fig = P.make_boxplot(
                df, y=y, x=x, color=color, facet_cols=facets,
                palette=palette, show_points=spec["show_points"],
                ncols=ncols, title=title)
            self.info(f"Boxplot: {len(df):,} rows.{filtered}")
        else:
            fig = P.make_barplot_mean_sd(
                df, y=y, x=x, color=color, facet_cols=facets,
                palette=palette, ncols=ncols, title=title)
            self.info(f"Barplot (mean ± SD): {len(df):,} rows.{filtered}")
        self._set_figure(fig)

    def _set_figure(self, fig) -> None:
        if self._figure is not None:
            plt.close(self._figure)
        if self._canvas is not None:
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None
        if self._toolbar is not None:
            self._toolbar.setParent(None)
            self._toolbar.deleteLater()
            self._toolbar = None
        self._figure = fig
        self._canvas = FigureCanvasQTAgg(fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._canvas_layout.addWidget(self._toolbar)
        self._canvas_layout.addWidget(self._canvas, 1)
        # Hover tooltips must bind to the real (Qt) canvas, created above.
        P.attach_hover(fig, self._canvas)
        self._canvas.draw_idle()
        self._export_btn.setEnabled(True)

    def info(self, text: str, error: bool = False) -> None:
        self._info_label.setText(text)
        self._info_label.setStyleSheet(
            "color: #d05050; font-size: 8pt;" if error
            else "color: #888888; font-size: 8pt;")

    # ── Export ────────────────────────────────────────────────────────────

    def _on_export(self) -> None:
        if self._figure is None:
            return
        table = self._object_combo.currentText() or "plot"
        default = f"{table}_{self._chart_combo.currentText().split()[0]}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot as PDF", default, "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            P.save_figure_pdf(self._figure, path)
            self.info(f"Exported PDF: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", str(e))

    # ── Cleanup ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._gen += 1
        self._db = None
        self._object_combo.blockSignals(True)
        self._object_combo.clear()
        self._object_combo.blockSignals(False)
        self._on_object_changed("")
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
        self._export_btn.setEnabled(False)
        self.info("")
