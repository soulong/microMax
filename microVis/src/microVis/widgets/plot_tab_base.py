"""Shared skeleton for the DB plot tabs (ProfilerPlotView / InferPlotView).

Both tabs are "controls column + interactive canvas" widgets that load a
pandas frame off the GUI thread, filter it with a pandas expression and
render/export a matplotlib figure. This base owns everything the two tabs
used to duplicate:

- the layout skeleton (scrollable fixed-width controls column, canvas holder)
- the generation-guarded background loader (generic worker + worker set)
- the matplotlib figure lifecycle (close old, canvas + toolbar + hover)
- the info line, the Plot/Export button row, the filter expression row
- the Colors palette switch driven by column numericity

A subclass implements the small hooks: ``_build_controls`` (its own
pickers, in tab order, using the shared row builders), ``_handle_loaded``
(render the loaded frame), ``_column_is_numeric``, ``_default_export_name``
and optionally ``_clear_tab``.
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
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import CMAP_OPTIONS, DEFAULT_CMAP, QUALITATIVE_PALETTES
from microVis.processing import plotting as P
from microVis.widgets._event_filter import NoScrollComboBox

NONE_ITEM = "(none)"         # shared first entry for optional column pickers
CONTROLS_COLUMN_WIDTH = 360  # fixed width of the left controls column


def _make_picker_combo(placeholder: str | None = None) -> NoScrollComboBox:
    """Editable combo with a type-to-filter completer (MatchContains).

    Used for every picker whose option list can be long (columns, tables,
    directories): typing jumps to matching entries instead of scrolling.
    """
    combo = NoScrollComboBox()
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    combo.completer().setFilterMode(Qt.MatchContains)
    combo.completer().setCompletionMode(QCompleter.PopupCompletion)
    if placeholder:
        combo.lineEdit().setPlaceholderText(placeholder)
    return combo


class _LoadSignals(QObject):
    loaded = Signal(int, object)
    error = Signal(int, str)


class _CallWorker(QRunnable):
    """Run one loader callable off the GUI thread; emit loaded(df) / error(msg)."""

    def __init__(self, loader, gen: int):
        super().__init__()
        self.signals = _LoadSignals()
        self._loader = loader
        self._gen = gen

    def run(self):
        try:
            self.signals.loaded.emit(self._gen, self._loader())
        except Exception as e:  # surfaced as a GUI popup
            self.signals.error.emit(self._gen, str(e))


class PlotTabBase(QWidget):
    """Base class for one DB plot tab (see module docstring)."""

    # Info-line text used when a load returns an empty frame.
    empty_load_message = "Table is empty."

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # State shared by every tab (subclass pickers add their own).
        self._db = None
        self._db_title = ""
        self._metadata = None
        self._figure = None
        self._canvas = None
        self._toolbar = None
        self._worker = None
        # In-flight workers are retained here (setAutoDelete(False)) so Qt's
        # C++ side cannot delete a running QRunnable mid-run; handlers
        # discard the EMITTING worker (captured in the connection).
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
        self._build_controls(form)  # subclass pickers, in tab order
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
        controls_scroll.setFixedWidth(CONTROLS_COLUMN_WIDTH)
        layout.addWidget(controls_scroll)

        # ── Right column: interactive plot canvas ──
        canvas_holder = QWidget()
        self._canvas_layout = QVBoxLayout(canvas_holder)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas_holder, 1)

    # ── Subclass hooks ─────────────────────────────────────────────────────

    def _build_controls(self, form: QFormLayout) -> None:
        """Add the tab's pickers to the form, top to bottom."""
        raise NotImplementedError

    def _on_plot(self) -> None:
        """Plot button / filter Return — dispatch the tab's load + render."""
        raise NotImplementedError

    def _handle_loaded(self, df: pd.DataFrame) -> None:
        """Render a freshly loaded frame (the generation is validated)."""
        raise NotImplementedError

    def _column_is_numeric(self, name: str) -> bool:
        """Whether a display column is numeric (drives the Colors palette)."""
        raise NotImplementedError

    def _default_export_name(self) -> str:
        """Default file name for the Export PDF dialog."""
        raise NotImplementedError

    def _clear_tab(self) -> None:
        """Drop subclass state/combos when the tab is cleared."""

    # ── Shared control rows (used by _build_controls) ──────────────────────

    def _add_color_size_rows(self, form: QFormLayout) -> None:
        """'Color by' / 'Size by' / 'Base size' rows (identical in both tabs)."""
        self._color_combo = _make_picker_combo()
        self._color_combo.currentTextChanged.connect(self._on_color_changed)
        form.addRow("Color by:", self._color_combo)

        self._size_combo = _make_picker_combo()
        form.addRow("Size by:", self._size_combo)

        size_row = QHBoxLayout()
        self._base_size = QDoubleSpinBox()
        self._base_size.setRange(1.0, 200.0)
        self._base_size.setValue(20.0)
        self._base_size.setSuffix(" pt")
        size_row.addWidget(self._base_size)
        size_row.addStretch()
        form.addRow("Base size:", size_row)

    def _add_colors_cap_filter_rows(self, form: QFormLayout, cap_label: str,
                                    filter_placeholder: str,
                                    filter_tooltip: str) -> None:
        """'Colors' / point-cap / filter-expression rows (identical logic)."""
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
        form.addRow(cap_label, cap_row)

        filter_row = QHBoxLayout()
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText(filter_placeholder)
        self._filter_edit.setToolTip(filter_tooltip)
        self._filter_edit.returnPressed.connect(self._on_plot)
        filter_row.addWidget(self._filter_edit, 1)
        self._filter_clear_btn = QPushButton("Clear")
        self._filter_clear_btn.setProperty("class", "secondary")
        self._filter_clear_btn.setToolTip("Clear the filter expression")
        self._filter_clear_btn.clicked.connect(self._filter_edit.clear)
        filter_row.addWidget(self._filter_clear_btn)
        form.addRow("Filter:", filter_row)

    @staticmethod
    def _fill_combo(combo: QComboBox, items, keep=None) -> None:
        """Replace a combo's items, preserving `keep` (default: current text)."""
        prev = combo.currentText() if keep is None else keep
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.blockSignals(False)
        if prev in items:
            combo.setCurrentText(prev)

    # ── Facet rows (shared by every tab that supports faceting) ────────────

    def _add_facet_rows(self, form: QFormLayout) -> None:
        """'Facet by' checklist + grid-columns spin, in shared form/order."""
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

    def _populate_facet_list(self, names) -> None:
        """Refill the facet checklist, preserving the current checks."""
        checked = set(self._facet_selected())
        self._facet_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in checked else Qt.Unchecked)
            self._facet_list.addItem(item)

    def _facet_selected(self) -> list[str]:
        out = []
        for i in range(self._facet_list.count()):
            item = self._facet_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.text())
        return out

    # ── Metadata (Merge/Clear from MainWindow) ──────────────────────────────

    def set_metadata(self, metadata) -> None:
        """Attach/detach merged Excel metadata (joined by `well` on load)."""
        self._metadata = metadata
        self._on_metadata_changed()

    def _on_metadata_changed(self) -> None:
        """Hook: refresh the column pickers after Merge/Clear."""

    # ── Control behavior ────────────────────────────────────────────────────

    def _on_color_changed(self, color: str) -> None:
        """Switch the Colors combo between palettes and colormaps."""
        continuous = bool(color) and color != NONE_ITEM and self._column_is_numeric(color)
        prev = self._colors_combo.currentText()
        items = CMAP_OPTIONS if continuous else QUALITATIVE_PALETTES
        self._colors_combo.blockSignals(True)
        self._colors_combo.clear()
        self._colors_combo.addItems(items)
        self._colors_combo.blockSignals(False)
        default = DEFAULT_CMAP if continuous else items[0]
        self._colors_combo.setCurrentText(prev if prev in items else default)
        self._colors_combo.setToolTip(
            "Continuous color → matplotlib colormap" if continuous
            else "Categorical color → qualitative palette")

    # ── Background loading ──────────────────────────────────────────────────

    def _start_load(self, loader) -> None:
        """Bump the generation and run `loader()` on the thread pool.

        `loader` runs off the GUI thread and returns the frame to render
        (or raises — surfaced as a "Plot Failed" popup).
        """
        self._gen += 1
        worker = _CallWorker(loader, self._gen)
        # Capture the emitting worker in the connection: a STALE worker's
        # callback must discard that worker, not whoever is current now
        # (discarding the wrong one leaks both QRunnables in self._workers).
        worker.signals.loaded.connect(
            lambda gen, df, w=worker: self._on_load_finished(gen, df, w))
        worker.signals.error.connect(
            lambda gen, msg, w=worker: self._on_load_error(gen, msg, w))
        worker.setAutoDelete(False)
        self._worker = worker
        self._workers.add(worker)
        self._pool.start(worker)

    def _on_load_finished(self, gen: int, df: pd.DataFrame, worker) -> None:
        self._workers.discard(worker)
        if self._worker is worker:
            self._worker = None
        if gen != self._gen:
            return
        if df is None or df.empty:
            self.info(self.empty_load_message)
            return
        self._handle_loaded(df)

    def _on_load_error(self, gen: int, msg: str, worker) -> None:
        self._workers.discard(worker)
        if self._worker is worker:
            self._worker = None
        if gen != self._gen:
            return
        QMessageBox.warning(self, "Plot Failed", msg)
        self.info(f"Load failed: {msg}", error=True)

    # ── Figure lifecycle / info / export ────────────────────────────────────

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

    def _on_export(self) -> None:
        if self._figure is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot as PDF", self._default_export_name(), "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            P.save_figure_pdf(self._figure, path)
            self.info(f"Exported PDF: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", str(e))

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Drop the DB binding, plot state and every picker (reset)."""
        self._gen += 1
        self._db = None
        self._clear_tab()
        facet_list = getattr(self, "_facet_list", None)
        if facet_list is not None:
            facet_list.clear()
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
        self._plot_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self.info("")
