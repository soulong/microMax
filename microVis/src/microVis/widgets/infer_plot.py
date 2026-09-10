"""Infer DB scatter tab: DR coordinates from a microModel infer.db.

Reads the joined `inference` + `reduction_<method>` (+ `find_cluster`) frame
via InferDB, filters to the selected source directory, and renders a scatter
where color (one variable) and size (one numeric variable) are user-bound.
Exports vector PDFs with editable text.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import CMAP_OPTIONS, DEFAULT_CMAP, QUALITATIVE_PALETTES
from microVis.io.data_module import merge_metadata
from microVis.io.infer_db import InferDB, is_numeric_type
from microVis.processing import plotting as P

_NONE = "(none)"
_ALL = "(all directories)"


class _LoadSignals(QObject):
    loaded = Signal(int, object)
    error = Signal(int, str)


class _ScatterLoadWorker(QRunnable):
    """Read + join one reduction method off the GUI thread."""

    def __init__(self, db: InferDB, method: str, gen: int):
        super().__init__()
        self.signals = _LoadSignals()
        self._db = db
        self._method = method
        self._gen = gen

    def run(self):
        try:
            df = self._db.load_scatter(self._method)
            self.signals.loaded.emit(self._gen, df)
        except Exception as e:
            self.signals.error.emit(self._gen, str(e))


class InferPlotView(QWidget):
    """Scatter builder for one infer.db (one DR method at a time)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._db: InferDB | None = None
        self._db_title = ""
        self._dataset_dir = ""
        self._columns: list[tuple[str, str]] = []
        # Base columns (without metadata) so Merge/Clear can add/remove the
        # metadata entries without re-reading the DB.
        self._base_columns: list[tuple[str, str]] = []
        # Merged Excel metadata (set by MainWindow's Merge button).
        self._metadata = None
        self._figure = None
        self._canvas = None
        self._toolbar = None
        self._worker = None
        # In-flight workers are retained here (setAutoDelete(False)) so Qt's
        # C++ side cannot delete a running QRunnable mid-run.
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

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)

        self._method_combo = QComboBox()
        form.addRow("Reduction:", self._method_combo)

        self._dir_combo = QComboBox()
        self._dir_combo.setToolTip(
            "Filter rows by the source directory stored in the infer DB.")
        form.addRow("Directory:", self._dir_combo)

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
        form.addRow("Point cap:", cap_row)

        filter_row = QHBoxLayout()
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText('pred_class == "x" & pred_prob > 0.8')
        self._filter_edit.setToolTip(
            "pandas query expression applied before plotting.\n"
            "Operators: > < >= <= == != & | ~ ( )\n"
            "Strings: pred_class == \"x\".\n"
            "Column names with spaces/dots need backticks: `my col` > 1, "
            "`cluster_res_0.5` == 2.\n"
            "Leave empty for no filter.")
        self._filter_edit.returnPressed.connect(self._on_plot)
        filter_row.addWidget(self._filter_edit, 1)
        self._filter_clear_btn = QPushButton("Clear")
        self._filter_clear_btn.setProperty("class", "secondary")
        self._filter_clear_btn.setToolTip("Clear the filter expression")
        self._filter_clear_btn.clicked.connect(self._filter_edit.clear)
        filter_row.addWidget(self._filter_clear_btn)
        form.addRow("Filter:", filter_row)

        controls_layout.addLayout(form)

        btn_row = QHBoxLayout()
        self._plot_btn = QPushButton("Plot")
        self._plot_btn.setProperty("class", "primary")
        self._plot_btn.setEnabled(False)
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

    # ── DB wiring ─────────────────────────────────────────────────────────

    def set_db(self, db: InferDB, dataset_scopes=None) -> None:
        """Attach an infer DB and auto-plot its first reduction method.

        ``dataset_scopes`` identifies the current dataset's rows in the DB's
        `directory` column (canonical relative values, or an absolute legacy
        path) so the directory combo preselects them.
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

        Stored directories are canonical (relative to their dataset root);
        scopes come from the current dataset's metadata, so exact matches are
        the normal case. Absolute legacy values are matched with a normalized
        separator/case comparison.
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

    def set_metadata(self, metadata) -> None:
        """Attach/detach merged Excel metadata (joined by `well` on load)."""
        self._metadata = metadata
        self._columns = list(self._base_columns)
        if metadata is not None:
            self._append_metadata_columns()
        self._populate_column_combos()

    def _append_metadata_columns(self) -> None:
        """Add metadata columns missing from the current column list."""
        existing = {c for c, _t in self._columns}
        for c in self._metadata.columns:
            if c == "well" or c in existing:
                continue
            self._columns.append((c, str(self._metadata[c].dtype)))

    def _populate_column_combos(self) -> None:
        names = [c for c, _t in self._columns]
        numeric = [c for c, t in self._columns if is_numeric_type(t)]

        def _fill(combo, items):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.blockSignals(False)

        _fill(self._color_combo, [_NONE] + names)
        _fill(self._size_combo, [_NONE] + numeric)
        self._on_color_changed(self._color_combo.currentText())

    def _on_color_changed(self, color: str) -> None:
        continuous = False
        if color and color != _NONE:
            for cname, ctype in self._columns:
                if cname == color:
                    continuous = is_numeric_type(ctype)
                    break
        prev = self._colors_combo.currentText()
        items = CMAP_OPTIONS if continuous else QUALITATIVE_PALETTES
        self._colors_combo.blockSignals(True)
        self._colors_combo.clear()
        self._colors_combo.addItems(items)
        self._colors_combo.blockSignals(False)
        if prev in items:
            self._colors_combo.setCurrentText(prev)
        else:
            self._colors_combo.setCurrentText(DEFAULT_CMAP if continuous else items[0])

    def _selected(self, combo: QComboBox) -> str | None:
        text = combo.currentText()
        return None if not text or text == _NONE else text

    # ── Plot ──────────────────────────────────────────────────────────────

    def _on_plot(self) -> None:
        if self._db is None:
            return
        method = self._method_combo.currentText()
        if not method:
            return
        self._gen += 1
        self.info(f"Loading reduction_{method}...")
        worker = _ScatterLoadWorker(self._db, method, self._gen)
        worker.signals.loaded.connect(self._on_loaded)
        worker.signals.error.connect(self._on_load_error)
        worker.setAutoDelete(False)
        self._worker = worker
        self._workers.add(worker)
        self._pool.start(worker)

    def _on_loaded(self, gen: int, df: pd.DataFrame) -> None:
        self._workers.discard(self._worker)
        self._worker = None
        if gen != self._gen:
            return
        if df is None or df.empty:
            self.info("No rows for this reduction method.")
            return
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
        sub, sampled, total = P.apply_point_cap(df, cap)
        try:
            fig = P.make_scatter(
                sub, x=c1, y=c2, color=color, size=size,
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

    def _on_load_error(self, gen: int, msg: str) -> None:
        self._workers.discard(self._worker)
        self._worker = None
        if gen != self._gen:
            return
        QMessageBox.warning(self, "Plot Failed", msg)
        self.info(f"Load failed: {msg}", error=True)

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
        method = self._method_combo.currentText() or "reduction"
        default = f"infer_{method}.pdf"
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
        self._columns = []
        self._method_combo.clear()
        self._dir_combo.clear()
        self._color_combo.clear()
        self._size_combo.clear()
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
        self._plot_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self.info("")
