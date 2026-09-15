"""Single integrated plot area for the Data page.

One widget replaces the old per-DB plot tabs: it plots the MERGED object
table (any mix of profiler.db + infer.db files, see io/merged_data) with
the full chart toolkit — scatter / line mean±SEM / boxplot / barplot
mean±SEM, X/Y/color-by/size-by/facet-by pickers (all editable
type-to-filter combos), palette, scatter point cap and a free-form pandas
filter. Figures export as vector PDFs with editable Type-42 text.

Left-clicking a scatter point shows the corresponding single cell: the
nearest tagged point within a small radius wins (overlapping points
included); clicking empty space hides the popup. The crop itself is
resolved by MainWindow (it owns the dataset) and pushed back through
:meth:`show_cell_image` / :meth:`hide_cell_image` — the view only owns the
floating near-cursor popup.
"""

from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
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
from microVis.widgets.ui_spec import COMPACT_LINE_EDIT_STYLE, centered_row

_NONE = "(none)"
_CHART_SCATTER = "scatter"
_CHART_LINE = "line"
_CHART_BOXPLOT = "boxplot"
_CHART_BARPLOT = "barplot"
_CHART_ORDER = [_CHART_SCATTER, _CHART_LINE, _CHART_BOXPLOT, _CHART_BARPLOT]

# Width of the Data page control column. Matches the Image page's control
# boxes (the splitter pane minus the 6px pane margins minus the vertical
# scrollbar) so both pages present the same left column.
_CONTROLS_WIDTH = 260


def _make_picker_combo() -> NoScrollComboBox:
    """Editable combo with a type-to-filter completer (MatchContains)."""
    combo = NoScrollComboBox()
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    combo.completer().setFilterMode(Qt.MatchContains)
    combo.completer().setCompletionMode(QCompleter.PopupCompletion)
    return combo


def _fill_combo(combo: QComboBox, items, keep=None) -> None:
    """Replace a combo's items, preserving `keep` (default: current text)."""
    prev = combo.currentText() if keep is None else keep
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(items)
    combo.blockSignals(False)
    if prev in items:
        combo.setCurrentText(prev)


def _mpl_to_qt_xy(canvas_height: int, dpr: float,
                  x: float, y: float) -> tuple[int, int]:
    """Convert matplotlib display coords to Qt widget coords.

    matplotlib Qt events carry PHYSICAL pixels with the origin at the
    BOTTOM-left (``x = pos.x * dpr``, ``y = (height/dpr - pos.y) * dpr``);
    Qt's ``mapToGlobal`` expects LOGICAL pixels with the origin at the
    TOP-left. Dividing by the device pixel ratio and flipping y is the
    exact inverse, so a popup lands on the clicked pixel.
    """
    dpr = dpr or 1.0
    return int(round(x / dpr)), int(round(canvas_height - y / dpr))


class _CellPopup(QFrame):
    """Frameless near-cursor popup that shows one cropped single cell."""

    def __init__(self):
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self._label = QLabel()
        layout.addWidget(self._label)

    def show_pixmap(self, pixmap, global_pos: QPoint) -> None:
        self._label.setPixmap(pixmap)
        self.adjustSize()
        self.move(global_pos + QPoint(16, 16))
        self.show()
        self.raise_()


class DataPlotView(QWidget):
    """Plot builder over the merged object table (one per Data page)."""

    # A scatter point was left-clicked: (row values dict, global QPoint).
    # The plot cannot resolve the cell image itself — MainWindow owns the
    # dataset and answers via show_cell_image/hide_cell_image.
    point_picked = Signal(dict, QPoint)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._figure = None
        self._canvas = None
        self._toolbar = None
        self._popup = _CellPopup()
        self._popup_pending_pos = QCursor.pos()
        # Any interaction outside the plot canvas dismisses the single-cell
        # popup: application-wide events catch clicks on buttons/pages, wheel
        # and focus changes so the popup never lingers on screen.
        QApplication.instance().installEventFilter(self)

        layout = QHBoxLayout(self)
        # No outer margin: the plotting-control box's left edge must line up
        # with the Data page's top control box (the page layout owns the
        # margin).
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── Left column: controls (scrollable so small windows never clip) ──
        controls = QWidget()
        # Titleless rounded box around the whole plotting-control column.
        controls.setProperty("class", "panel-box")
        controls_layout = QVBoxLayout(controls)
        # Slim inner rail: the control column is kept as narrow as the Image
        # page's pane, so the form must fit without dead padding.
        controls_layout.setContentsMargins(8, 8, 8, 10)
        controls_layout.setSpacing(6)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)

        self._chart_combo = QComboBox()
        self._chart_combo.addItems(_CHART_ORDER)
        self._chart_combo.currentTextChanged.connect(self._on_chart_changed)
        form.addRow("Chart:", self._chart_combo)

        # Filter sits directly below Chart (the plot-wide data selector).
        filter_row = QHBoxLayout()
        self._filter_edit = QLineEdit()
        self._filter_edit.setStyleSheet(COMPACT_LINE_EDIT_STYLE)
        self._filter_edit.setPlaceholderText("area > 200 & pred_prob > 0.8")
        self._filter_edit.setToolTip(
            "pandas query expression applied before plotting.\n"
            "Operators: > < >= <= == != & | ~ ( )\n"
            "Strings: pred_class == \"x\".\n"
            "Column names with spaces/slashes need backticks: `my col` > 1, "
            "`infer/pred_prob` > 0.5.\n"
            "Leave empty for no filter.")
        self._filter_edit.returnPressed.connect(self._on_plot)
        filter_row.addWidget(self._filter_edit, 1)
        # Gap between the expression box and its clear button.
        filter_row.addSpacing(8)
        self._filter_clear_btn = QPushButton("Clear")
        self._filter_clear_btn.setToolTip("Clear the filter expression")
        self._filter_clear_btn.clicked.connect(self._filter_edit.clear)
        filter_row.addWidget(self._filter_clear_btn)
        form.addRow("Filter:", filter_row)

        self._x_combo = _make_picker_combo()
        self._x_combo.setToolTip(
            "Any merged column. Categorical levels are placed on their level "
            "index with the level names as axis ticks.")
        form.addRow("X:", self._x_combo)

        self._y_combo = _make_picker_combo()
        self._y_combo.setToolTip(
            "Any merged column. Categorical levels are placed on their level "
            "index with the level names as axis ticks.")
        form.addRow("Y:", self._y_combo)

        self._color_combo = _make_picker_combo()
        self._color_combo.currentTextChanged.connect(self._on_color_changed)
        form.addRow("Color by:", self._color_combo)

        self._size_combo = _make_picker_combo()
        form.addRow("Size by:", self._size_combo)

        self._facet_list = QListWidget()
        self._facet_list.setMaximumHeight(90)
        # No selection highlight: a selected row's fill hides whether its
        # checkbox is ticked; only the checkbox itself changes state.
        self._facet_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._facet_list.setToolTip(
            "Checked variables are faceted by their FULL combination of levels "
            "(cartesian product), one subplot per combination.")
        # Type-to-filter box above the list: narrows the visible columns
        # without touching the checked selection.
        facet_box = QVBoxLayout()
        facet_box.setSpacing(2)
        self._facet_filter = QLineEdit()
        self._facet_filter.setPlaceholderText("Type to filter columns...")
        self._facet_filter.setStyleSheet(COMPACT_LINE_EDIT_STYLE)
        self._facet_filter.textChanged.connect(self._apply_facet_filter)
        facet_box.addWidget(self._facet_filter)
        facet_box.addWidget(self._facet_list)
        form.addRow("Facet by:", facet_box)

        facet_cols_row = QHBoxLayout()
        self._facet_cols = QSpinBox()
        self._facet_cols.setRange(1, 8)
        self._facet_cols.setValue(3)
        self._facet_cols.setButtonSymbols(QAbstractSpinBox.NoButtons)
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
        self._cap_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        cap_row.addWidget(self._cap_spin, 1)
        form.addRow("Scatter cap:", cap_row)

        # Scatter point size sits right below the cap (they shape the same
        # scatter rendering).
        scatter_size_row = QHBoxLayout()
        self._scatter_size = QDoubleSpinBox()
        self._scatter_size.setRange(1.0, 200.0)
        self._scatter_size.setValue(20.0)
        self._scatter_size.setSuffix(" pt")
        self._scatter_size.setButtonSymbols(QAbstractSpinBox.NoButtons)
        scatter_size_row.addWidget(self._scatter_size)
        scatter_size_row.addStretch()
        form.addRow("Scatter size:", scatter_size_row)

        self._show_points = QCheckBox("Show points on boxplot")
        self._show_points.setChecked(True)
        form.addRow("", self._show_points)

        controls_layout.addLayout(form)

        # Row-count / status info line.
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #888888; font-size: 8pt;")
        controls_layout.addWidget(self._info_label)

        # Plot / Export PDF sit below the info line, centered in the column;
        # Plot is twice the natural caption length (per request).
        self._plot_btn = QPushButton("Plot")
        self._plot_btn.ensurePolished()
        self._plot_btn.setFixedWidth(self._plot_btn.sizeHint().width() * 2)
        self._plot_btn.clicked.connect(self._on_plot)

        self._export_btn = QPushButton("Export PDF")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        controls_layout.addLayout(centered_row(self._plot_btn, self._export_btn))
        controls_layout.addStretch()

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setWidget(controls)
        controls_scroll.setFixedWidth(_CONTROLS_WIDTH)
        layout.addWidget(controls_scroll)

        # ── Right column: interactive plot canvas ──
        canvas_holder = QWidget()
        self._canvas_layout = QVBoxLayout(canvas_holder)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas_holder, 1)

        self._on_chart_changed(self._chart_combo.currentText())

    # ── Data ──────────────────────────────────────────────────────────────

    def set_frame(self, df: pd.DataFrame) -> None:
        """Adopt the (re-)merged table and refresh every picker."""
        self._df = df
        names = [c for c in df.columns]
        numeric = [c for c in names
                   if pd.api.types.is_numeric_dtype(df[c])]
        _fill_combo(self._x_combo, names)
        _fill_combo(self._y_combo, names)
        _fill_combo(self._color_combo, [_NONE] + names, keep=None)
        _fill_combo(self._size_combo, [_NONE] + numeric, keep=None)

        checked = set(self._facet_selected())
        self._facet_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in checked else Qt.Unchecked)
            self._facet_list.addItem(item)
        # Keep the current text filter applied to the newly built list.
        self._apply_facet_filter(self._facet_filter.text())

        self._on_color_changed(self._color_combo.currentText())

    def clear(self) -> None:
        """Drop the merged table + figure and reset EVERY control.

        The Data-page Reset button expects a clean slate: the pickers must
        not keep the previous DB's columns/options, so all controls return
        to their construction defaults here.
        """
        self._df = None
        self._hide_cell_image()
        self._set_figure(None)
        self._export_btn.setEnabled(False)
        self.info("")
        # Chart selector (block + manual call: the handler also refreshes
        # the per-chart enabled states and tooltips).
        self._chart_combo.blockSignals(True)
        self._chart_combo.setCurrentText(_CHART_SCATTER)
        self._chart_combo.blockSignals(False)
        self._on_chart_changed(_CHART_SCATTER)
        # Filter + column pickers.
        self._filter_edit.clear()
        for combo in (self._x_combo, self._y_combo,
                      self._color_combo, self._size_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
        self._colors_combo.blockSignals(True)
        self._colors_combo.clear()
        self._colors_combo.blockSignals(False)
        # Facets.
        self._facet_list.clear()
        self._facet_filter.clear()
        self._facet_cols.setValue(3)
        # Scatter options.
        self._scatter_size.setValue(20.0)
        self._cap_spin.setValue(20000)
        self._show_points.setChecked(True)

    # ── Control behavior ──────────────────────────────────────────────────

    def _selected(self, combo: QComboBox) -> str | None:
        text = combo.currentText()
        return None if not text or text == _NONE else text

    def _on_chart_changed(self, chart: str) -> None:
        is_scatter = chart == _CHART_SCATTER
        self._size_combo.setEnabled(is_scatter)
        self._scatter_size.setEnabled(True)
        self._cap_spin.setEnabled(is_scatter)
        self._show_points.setEnabled(chart == _CHART_BOXPLOT)
        self._x_combo.setToolTip(
            "X is required for scatter and line (the connected axis); "
            "optional grouping for box/bar."
            if is_scatter or chart == _CHART_LINE else
            "Optional grouping variable for box/bar.")

    def _on_color_changed(self, color: str) -> None:
        """Switch the Colors combo between palettes and colormaps."""
        continuous = False
        if (color and color != _NONE and self._df is not None
                and color in self._df.columns):
            continuous = pd.api.types.is_numeric_dtype(self._df[color])
        prev = self._colors_combo.currentText()
        items = CMAP_OPTIONS if continuous else QUALITATIVE_PALETTES
        _fill_combo(self._colors_combo, items)
        default = DEFAULT_CMAP if continuous else items[0]
        self._colors_combo.setCurrentText(prev if prev in items else default)
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

    def _apply_facet_filter(self, text: str) -> None:
        """Hide facet columns not matching the typed text (display only).

        Checked items keep their state while hidden, so filtering the list
        never changes the selected facets.
        """
        needle = (text or "").strip().lower()
        for i in range(self._facet_list.count()):
            item = self._facet_list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    # ── Plot ──────────────────────────────────────────────────────────────

    def _on_plot(self) -> None:
        if self._df is None:
            return
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
        self._render()

    def _render(self) -> None:
        df = self._df
        chart = self._chart_combo.currentText()
        y = self._selected(self._y_combo)
        x = self._selected(self._x_combo)
        color = self._selected(self._color_combo)
        facets = self._facet_selected()
        palette = self._colors_combo.currentText()
        ncols = self._facet_cols.value()

        total_rows = len(df)
        df, err = P.apply_filter(df, self._filter_edit.text())
        if err:
            QMessageBox.warning(self, "Invalid Filter", err)
            self.info(f"Invalid filter: {err}", error=True)
            return
        if df.empty:
            self.info("No rows left after filtering.", error=True)
            return
        filtered = f" (filtered {len(df):,} of {total_rows:,})" if len(df) != total_rows else ""

        if chart == _CHART_SCATTER:
            cap = self._cap_spin.value()
            sub, sampled, total = P.apply_point_cap(df, cap)
            fig = P.make_scatter(
                sub, x=x, y=y, color=color, size=self._selected(self._size_combo),
                facet_cols=facets, palette=palette, cmap=palette,
                point_size=self._scatter_size.value(), ncols=ncols)
            if sampled:
                self.info(f"Scatter sampled {len(sub):,} of {total:,} rows "
                          f"(cap {cap:,}; set 0 for all).{filtered}")
            else:
                self.info(f"Scatter: {len(sub):,} rows.{filtered}")
        elif chart == _CHART_LINE:
            fig = P.make_line(df, y=y, x=x, color=color, facet_cols=facets,
                              palette=palette, ncols=ncols)
            self.info(f"Line (mean ± SEM): {len(df):,} rows.{filtered}")
        elif chart == _CHART_BOXPLOT:
            fig = P.make_boxplot(df, y=y, x=x, color=color, facet_cols=facets,
                                 palette=palette,
                                 show_points=self._show_points.isChecked(),
                                 ncols=ncols)
            self.info(f"Boxplot: {len(df):,} rows.{filtered}")
        else:
            fig = P.make_barplot_mean_sem(df, y=y, x=x, color=color,
                                          facet_cols=facets, palette=palette,
                                          ncols=ncols)
            self.info(f"Barplot (mean ± SEM): {len(df):,} rows.{filtered}")
        self._set_figure(fig)

    # ── Figure lifecycle / info / export ──────────────────────────────────

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
        if fig is None:
            return
        self._canvas = FigureCanvasQTAgg(fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._canvas_layout.addWidget(self._toolbar)
        self._canvas_layout.addWidget(self._canvas, 1)
        # Hover tooltips bind to the real (Qt) canvas, created above.
        P.attach_hover(fig, self._canvas)
        # Left-click picking for the single-cell popup.
        P.attach_click(fig, self._canvas, self._on_point_clicked)
        self._canvas.draw_idle()
        self._export_btn.setEnabled(True)

    def info(self, text: str, error: bool = False) -> None:
        self._info_label.setText(text)
        self._info_label.setStyleSheet(
            "color: #f04770; font-size: 8pt;" if error
            else "color: #888888; font-size: 8pt;")

    def _on_export(self) -> None:
        if self._figure is None:
            return
        chart = self._chart_combo.currentText()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot as PDF", f"merged_{chart}.pdf", "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            P.save_figure_pdf(self._figure, path)
            self.info(f"Exported PDF: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", str(e))

    # ── Click → single-cell popup ─────────────────────────────────────────

    def _on_point_clicked(self, row, x_px: int, y_px: int) -> None:
        """Left-click on the canvas: nearest tagged point (or None on a miss).

        A miss hides the popup; a hit re-positions it at the click and asks
        MainWindow for the cropped cell of that row.
        """
        if row is None or self._canvas is None:
            self._hide_cell_image()
            return
        # mpl display coords are physical px from the bottom; Qt wants logical
        # px from the top (see _mpl_to_qt_xy) — otherwise clicks near the top
        # show the popup near the bottom and vice versa.
        dpr = self._canvas.devicePixelRatioF() or 1.0
        qt_x, qt_y = _mpl_to_qt_xy(self._canvas.height(), dpr, x_px, y_px)
        global_pos = self._canvas.mapToGlobal(QPoint(qt_x, qt_y))
        row_dict = {}
        for key, val in dict(row).items():
            try:
                if val is None or pd.isna(val):
                    val = None
                elif hasattr(val, "item"):
                    val = val.item()
            except (TypeError, ValueError):
                pass
            row_dict[key] = val
        self._popup_pending_pos = global_pos
        self.point_picked.emit(row_dict, global_pos)

    def show_cell_image(self, pixmap) -> None:
        """Show the cropped cell (answered by MainWindow) at the click."""
        self._popup.show_pixmap(pixmap, getattr(self, "_popup_pending_pos",
                                                QCursor.pos()))

    def _hide_cell_image(self) -> None:
        try:
            self._popup.hide()
        except RuntimeError:
            # The popup's C++ side is already gone (app shutdown) — the
            # Python wrapper can still be reached through the app-wide
            # event filter while Qt tears objects down.
            pass

    # ── Popup auto-dismiss ────────────────────────────────────────────────

    def _is_canvas_widget(self, widget) -> bool:
        """True when `widget` is the plot canvas or one of its children."""
        if self._canvas is None:
            return False
        if widget is self._canvas:
            return True
        try:
            return self._canvas.isAncestorOf(widget)
        except (TypeError, RuntimeError):
            return False

    def eventFilter(self, watched, event):
        """Hide the cell popup on any interaction outside the canvas.

        Clicks on buttons, other panels, the nav bar, wheel/focus changes
        all dismiss it; events inside the canvas are left to the matplotlib
        click handler (which re-shows or hides the popup itself). Losing the
        application/window focus (user switched to another app) dismisses it
        too — it is a top-level tooltip window that would otherwise stay on
        screen over the other application.
        """
        try:
            popup_visible = self._popup.isVisible()
        except RuntimeError:
            # Qt is shutting down and the popup's C++ object is gone.
            return super().eventFilter(watched, event)
        etype = event.type()
        if popup_visible and etype in (
                QEvent.MouseButtonPress, QEvent.Wheel,
                QEvent.KeyPress, QEvent.FocusIn):
            if not self._is_canvas_widget(watched):
                self._hide_cell_image()
        elif popup_visible and etype in (
                QEvent.ApplicationDeactivate, QEvent.WindowDeactivate):
            self._hide_cell_image()
        return super().eventFilter(watched, event)

    def hideEvent(self, event):
        """The page was switched away: the top-level popup must go with it."""
        self._hide_cell_image()
        super().hideEvent(event)

    # Re-exported so MainWindow can hide the popup from the outside.
    hide_cell_image = _hide_cell_image
