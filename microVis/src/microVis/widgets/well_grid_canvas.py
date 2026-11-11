from __future__ import annotations

import math
import numbers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QSizePolicy, QToolTip, QVBoxLayout, QWidget

from microBase import normalize_well
from microVis._settings import get_cmap
from microVis.log_utils import get_logger

logger = get_logger("microVis.well_grid_canvas")

# Click-vs-drag threshold in display pixels (same convention as the
# thumbnail view's manhattanLength() > 4 check).
_DRAG_THRESHOLD = 4

# Accent colours for the rubber-band rectangle and its live preview ring
# (same accent as the selection ring).
_BAND_COLOR = "#5a8a9a"


def _rect_cells(xmin, xmax, ymin, ymax, n_rows, n_cols):
    """Integer (row, col) grid cells (1-based, row-major) whose unit square
    (centred on the well point, spanning ±0.5) intersects the data-space
    rectangle [xmin, xmax] × [ymin, ymax]. Clipped to the grid bounds;
    empty list when the rectangle lies outside the grid."""
    c_lo = max(1, math.ceil(xmin - 0.5))
    c_hi = min(n_cols, math.floor(xmax + 0.5))
    r_lo = max(1, math.ceil(ymin - 0.5))
    r_hi = min(n_rows, math.floor(ymax + 0.5))
    if c_lo > c_hi or r_lo > r_hi:
        return []
    return [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]


def _combine_selection(current, wells, mode):
    """Apply a band of wells to a selection according to the modifier mode:
    ``replace`` (plain drag), ``add`` (Shift) or ``toggle`` (Ctrl)."""
    if mode == "add":
        return set(current) | wells
    if mode == "toggle":
        return set(current) ^ wells
    return set(wells)  # replace


class WellGridCanvas(QWidget):
    """Well plate grid visualization using matplotlib scatter plot."""

    well_clicked = Signal(str)
    # Rubber-band selection finished: (set of well names under the band,
    # mode) with mode one of "replace" / "add" / "toggle".
    wells_band_selected = Signal(object, str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._figure = Figure(facecolor="#252536")

        # Pre-allocate BOTH axes once with a fixed GridSpec so their positions
        # never change across redraws.  The colorbar axes is simply shown or
        # hidden; it never steals space from the main axes (which is the root
        # cause of the shrinking-grid bug when using figure.colorbar(ax=…)).
        gs = GridSpec(
            1, 2,
            figure=self._figure,
            width_ratios=[20, 1],
            left=0.06, right=0.97,
            top=0.90, bottom=0.03,
            wspace=0.05,
        )
        self._axes = self._figure.add_subplot(gs[0, 0])
        self._axes.set_facecolor("#252536")
        self._cbar_ax = self._figure.add_subplot(gs[0, 1])
        self._cbar_ax.set_visible(False)

        self._canvas = FigureCanvasQTAgg(self._figure)
        # Clicks and rubber-band selection share one press/move/release state
        # machine; matplotlib's pick_event is deliberately NOT used because a
        # pick fires on mouse press and would toggle a well before a possible
        # drag even starts.
        self._canvas.mpl_connect("button_press_event", self._on_button_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_button_release)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setStyleSheet("background-color: #252536;")

        self._well_indices: dict[str, int] = {}
        self._real_wells: set[str] = set()
        self._data_map: dict[str, float | str] = {}
        self._col_name: str | None = None
        self._row_labels: list[str] = []
        self._n_rows: int = 0
        self._n_cols: int = 0

        # Incremental redraw state
        self._scatter = None
        self._annotations: list = []
        self._sel_scatter = None
        self._prev_dims: tuple[int, int] = (0, 0)
        self._prev_fmt: str = ""
        self._marker_size = 0
        # Copy of the last drawn selection (preview math during a band drag)
        self._selected_wells: set[str] = set()

        # Rubber-band gesture state (armed on press, fired on release)
        self._band_start = None       # (xdata, ydata) at press
        self._band_press_px = None    # (x, y) display coords at press
        self._band_last = None        # last valid in-axes (xdata, ydata)
        self._band_moved = False
        self._band_mode = "replace"
        self._band_rect = None        # Rectangle patch shown while dragging
        self._preview_scatter = None  # live preview ring of the pending result

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def clear(self) -> None:
        """Clear the canvas to an empty state."""
        self._cancel_band()
        self._axes.clear()
        self._axes.set_facecolor("#252536")
        self._cbar_ax.clear()
        self._cbar_ax.set_visible(False)
        self._annotations.clear()
        self._sel_scatter = None
        self._scatter = None
        self._well_indices.clear()
        self._data_map.clear()
        self._real_wells = set()
        self._col_name = None
        self._prev_dims = (0, 0)
        self._prev_fmt = ""
        self._marker_size = 0
        self._selected_wells = set()
        self._canvas.draw_idle()

    def update_grid(
        self,
        dm,
        table_name: str,
        col_val: tuple | None,
        agg: str,
        cmap: str,
        palette: str,
        fmt_name: str,
        selected_wells: set[str],
        value_map: dict[str, float | str] | None = None,
        force_redraw: bool = False,
    ) -> None:
        """Redraw the well grid. Uses incremental update when dimensions are unchanged.

        ``value_map`` is a precomputed well→value map (merged Excel metadata or
        a loaded infer DB); when given, it bypasses ``dm.aggregate``.
        """
        # A redraw mid-gesture (e.g. a color-by change while dragging) drops
        # the band artists via axes.clear(); cancel the gesture cleanly.
        self._cancel_band()
        rows, cols = dm.get_plate_dims()
        fmt_rows, fmt_cols = _get_format_dims(fmt_name)
        rows = max(rows, fmt_rows)
        cols = max(cols, fmt_cols)

        row_labels = _row_labels(rows)
        all_wells = dm.get_wells()
        # Normalize incoming well keys ('A01' -> 'A1'): the grid generates
        # canonical no-leading-zero labels, while a captured `well` keeps its
        # regex text verbatim — without normalization a 'A01' dataset would
        # never match any grid cell (all-grey grid, dead clicks/labels).
        well_set = {normalize_well(w) for w in all_wells}
        self._real_wells = well_set

        # Build well indices
        self._well_indices.clear()
        for r in range(rows):
            for c in range(cols):
                self._well_indices[f"{row_labels[r]}{c + 1}"] = r * cols + c

        self._row_labels = row_labels
        self._n_rows = rows
        self._n_cols = cols

        # Compute data mapping
        data_map: dict[str, float | str] = {}
        is_numeric = False
        self._col_name = None
        if col_val and col_val[0] is not None and table_name:
            col_name, is_numeric = col_val
            self._col_name = col_name
            if value_map is not None:
                data_map = value_map
            else:
                try:
                    data_map = dm.aggregate(table_name, col_name, agg)
                except Exception as e:
                    logger.warning("Aggregation failed for %s.%s: %s", table_name, col_name, e)
                    data_map = {}
            data_map = {normalize_well(k): v for k, v in data_map.items()}
        self._data_map = data_map
        # Copy (not reference) — the preview math during a band drag reads
        # this snapshot, while MainWindow keeps mutating its own set.
        self._selected_wells = set(selected_wells)

        marker_size = _compute_marker_size(rows, cols)
        self._marker_size = marker_size
        has_data = bool(data_map)
        cmap_obj = None
        norm = None
        cat_to_color = {}

        if has_data and is_numeric:
            # numbers.Real includes numpy integer/float scalars (np.int64 is
            # NOT an int; Excel metadata and sum aggregations produce them).
            numeric_vals = [v for v in data_map.values() if isinstance(v, numbers.Real)]
            if numeric_vals:
                vmin = min(numeric_vals)
                vmax = max(numeric_vals)
                cmap_obj = get_cmap(cmap)
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
            else:
                has_data = False

        if has_data and not is_numeric:
            unique_cats = sorted(set(str(v) for v in data_map.values()))
            base = get_cmap(palette)
            n_cats = len(unique_cats)
            cat_to_color = {
                cat: base(i / max(n_cats - 1, 1))
                for i, cat in enumerate(unique_cats)
            }

        # Build color/position arrays
        x_vals, y_vals, color_list = [], [], []
        for r in range(rows):
            for c in range(cols):
                well = f"{row_labels[r]}{c + 1}"
                x_vals.append(c + 1)
                y_vals.append(r + 1)
                if has_data and well in data_map:
                    val = data_map[well]
                    if is_numeric and isinstance(val, numbers.Real):
                        color_list.append(cmap_obj(norm(val)))
                    elif not is_numeric:
                        color_list.append(cat_to_color.get(str(val), "#3a3a5a"))
                    else:
                        color_list.append("#3a3a5a")
                elif well in well_set:
                    color_list.append("#3a3a5a")
                else:
                    color_list.append("#1a1a2a")

        offsets = np.column_stack([x_vals, y_vals])
        dims_key = (rows, cols, fmt_name)
        need_full_redraw = force_redraw or dims_key != (self._prev_dims[0], self._prev_dims[1], self._prev_fmt)

        if need_full_redraw:
            # Full redraw: clear axes and rebuild everything
            self._axes.clear()
            self._axes.set_facecolor("#252536")
            self._cbar_ax.clear()
            self._cbar_ax.set_visible(False)
            self._annotations.clear()
            self._sel_scatter = None

            self._scatter = self._axes.scatter(
                x_vals, y_vals,
                s=marker_size,
                c=color_list,
                edgecolors="#555555",
                linewidths=0.5,
                zorder=2,
            )

            for well, i in self._well_indices.items():
                if well in well_set:
                    txt = self._axes.annotate(
                        well, (x_vals[i], y_vals[i]),
                        fontsize=5, color="#aaaaaa", ha="center", va="center", zorder=3,
                    )
                    self._annotations.append(txt)

            self._axes.set_xlim(0.5, cols + 0.5)
            self._axes.set_ylim(0.5, rows + 0.5)
            self._axes.set_xticks(range(1, cols + 1))
            self._axes.set_xticklabels([str(c) for c in range(1, cols + 1)])
            self._axes.set_yticks(range(1, rows + 1))
            self._axes.set_yticklabels(row_labels)
            self._axes.tick_params(colors="#888888", labelsize=7)
            self._axes.invert_yaxis()
            self._axes.xaxis.set_ticks_position('top')
            self._axes.xaxis.set_label_position('top')
            for spine in self._axes.spines.values():
                spine.set_color("#333333")
        else:
            # Incremental update: update scatter data in-place
            self._scatter.set_offsets(offsets)
            self._scatter.set_facecolors(color_list)
            self._scatter.set_sizes(np.full(len(x_vals), marker_size))
            # Remove old selection ring
            if self._sel_scatter is not None:
                self._sel_scatter.remove()
                self._sel_scatter = None

        # Draw selection ring (on both paths)
        sel_x, sel_y = [], []
        for well in selected_wells:
            if well in well_set and well in self._well_indices:
                idx = self._well_indices[well]
                sel_x.append(x_vals[idx])
                sel_y.append(y_vals[idx])
        if sel_x:
            self._sel_scatter = self._axes.scatter(
                sel_x, sel_y,
                s=marker_size * 1.3,
                facecolors="none",
                edgecolors="#5a8a9a",
                linewidths=2,
                zorder=4,
            )

        # Colorbar
        self._cbar_ax.clear()
        self._cbar_ax.set_visible(False)
        if has_data and is_numeric and len(color_list) > 0:
            sm = ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            self._figure.colorbar(sm, cax=self._cbar_ax)
            self._cbar_ax.set_visible(True)
            self._cbar_ax.tick_params(colors="#888888", labelsize=7)
            for spine in self._cbar_ax.spines.values():
                spine.set_color("#333333")

        self._prev_dims = (rows, cols)
        self._prev_fmt = fmt_name
        self._canvas.draw()

    # ── Rubber-band selection + click state machine ──────────────────────────

    def _on_button_press(self, event) -> None:
        """Arm a click / band gesture on left-button press inside the grid."""
        if event.button != 1 or event.dblclick:
            return
        if event.inaxes != self._axes or event.xdata is None or event.ydata is None:
            return
        self._band_start = (event.xdata, event.ydata)
        self._band_press_px = (event.x, event.y)
        self._band_last = self._band_start
        self._band_moved = False
        self._band_mode = self._modifier_mode(event)

    def _on_motion(self, event) -> None:
        # While a gesture is armed, drive the rubber band and suppress the
        # hover tooltip for its duration.
        if self._band_start is not None:
            QToolTip.hideText()
            if (event.button == 1 and event.inaxes == self._axes
                    and event.xdata is not None and event.ydata is not None):
                self._band_last = (event.xdata, event.ydata)
                if not self._band_moved and self._past_drag_threshold(event):
                    self._band_moved = True
                    self._begin_band_visuals()
                if self._band_moved:
                    self._update_band_visuals(*self._band_last)
            return
        if event.inaxes != self._axes:
            QToolTip.hideText()
            return
        if event.xdata is None or event.ydata is None:
            QToolTip.hideText()
            return
        col = round(event.xdata)
        row = round(event.ydata)
        if 1 <= row <= self._n_rows and 1 <= col <= self._n_cols:
            well = f"{self._row_labels[row - 1]}{col}"
            if well in self._data_map and self._col_name:
                val = self._data_map[well]
                if isinstance(val, float):
                    text = f"{well}  {self._col_name}: {val:.4f}"
                else:
                    text = f"{well}  {self._col_name}: {val}"
            elif well in self._well_indices:
                text = well
            else:
                text = ""
            if text:
                QToolTip.showText(QCursor.pos(), text, self)
            else:
                QToolTip.hideText()
        else:
            QToolTip.hideText()

    def _on_button_release(self, event) -> None:
        """Finish the armed gesture: a click toggles one well, a drag fires
        the band selection once (with the mode captured at press time)."""
        start = self._band_start
        if start is None:
            return
        moved = self._band_moved
        mode = self._band_mode
        last = self._band_last if self._band_last is not None else start
        self._band_start = None
        self._band_last = None

        if moved:
            self._remove_band_visuals()
            xmin, xmax = min(start[0], last[0]), max(start[0], last[0])
            ymin, ymax = min(start[1], last[1]), max(start[1], last[1])
            wells = self._wells_in_rect(xmin, xmax, ymin, ymax)
            # A plain band always fires (an empty band clears the selection,
            # file-manager convention); add/toggle with an empty band are
            # no-ops, so skip the redundant refresh entirely.
            if wells or mode == "replace":
                self.wells_band_selected.emit(wells, mode)
            return

        # Plain click (below the drag threshold): hit-test with the same cell
        # rounding the hover tooltip uses — the whole cell square is
        # clickable, not just the marker. Phantom wells are ignored.
        self._remove_band_visuals()
        if event.inaxes == self._axes and event.xdata is not None and event.ydata is not None:
            well = self._well_at(event.xdata, event.ydata)
            if well is not None:
                self.well_clicked.emit(well)

    def _modifier_mode(self, event) -> str:
        """Map the keyboard modifiers held at press to the band mode:
        Ctrl → toggle, Shift → add, plain → replace."""
        gui = getattr(event, "guiEvent", None)
        mods = (gui.modifiers() if gui is not None
                else Qt.KeyboardModifier.NoModifier)
        if mods & Qt.KeyboardModifier.ControlModifier:
            return "toggle"
        if mods & Qt.KeyboardModifier.ShiftModifier:
            return "add"
        return "replace"

    def _past_drag_threshold(self, event) -> bool:
        """True once the pointer moved more than _DRAG_THRESHOLD display
        pixels away from the press point (click vs. drag decision)."""
        px = event.x if event.x is not None else self._band_press_px[0]
        py = event.y if event.y is not None else self._band_press_px[1]
        return (abs(px - self._band_press_px[0]) > _DRAG_THRESHOLD
                or abs(py - self._band_press_px[1]) > _DRAG_THRESHOLD)

    def _well_at(self, x, y):
        """Well name under a data-space point (cell rounding), or None for
        out-of-grid / phantom positions."""
        col = round(x)
        row = round(y)
        if 1 <= row <= self._n_rows and 1 <= col <= self._n_cols:
            well = f"{self._row_labels[row - 1]}{col}"
            if well in self._real_wells:
                return well
        return None

    def _well_point(self, well):
        """(x, y) data position of a well, or None when unknown."""
        idx = self._well_indices.get(well)
        if idx is None:
            return None
        r, c = divmod(idx, self._n_cols)
        return (c + 1, r + 1)

    def _wells_in_rect(self, xmin, xmax, ymin, ymax):
        """Real (non-phantom) wells whose cell squares intersect the
        data-space rectangle."""
        wells = set()
        for r, c in _rect_cells(xmin, xmax, ymin, ymax, self._n_rows, self._n_cols):
            well = f"{self._row_labels[r - 1]}{c}"
            if well in self._real_wells:
                wells.add(well)
        return wells

    # ── Band visuals ─────────────────────────────────────────────────────────

    def _begin_band_visuals(self) -> None:
        """Create the rubber-band rectangle and hide the committed selection
        ring — the live preview shows the pending result instead (WYSIWYG)."""
        if self._sel_scatter is not None:
            self._sel_scatter.set_visible(False)
        self._band_rect = Rectangle(
            (0, 0), 0, 0,
            facecolor=_BAND_COLOR, alpha=0.18,
            edgecolor=_BAND_COLOR, linewidth=1, linestyle="--", zorder=5,
        )
        self._axes.add_patch(self._band_rect)

    def _update_band_visuals(self, x, y) -> None:
        """Resize the band rectangle and refresh the preview ring for the
        wells it covers."""
        x0, y0 = self._band_start
        xmin, xmax = min(x0, x), max(x0, x)
        ymin, ymax = min(y0, y), max(y0, y)
        # Snap the rectangle to whole cell squares so its edges line up with
        # the grid cells it selects.
        self._band_rect.set_bounds(xmin - 0.5, ymin - 0.5,
                                   xmax - xmin + 1, ymax - ymin + 1)
        self._show_preview(self._wells_in_rect(xmin, xmax, ymin, ymax))
        self._canvas.draw_idle()

    def _show_preview(self, wells) -> None:
        """Draw the ring the release would produce: the band combined with
        the pre-drag selection according to the current mode."""
        result = _combine_selection(self._selected_wells, wells, self._band_mode)
        if self._preview_scatter is not None:
            self._preview_scatter.remove()
            self._preview_scatter = None
        pts = [p for p in (self._well_point(w) for w in result) if p is not None]
        if pts:
            xs, ys = zip(*pts)
            self._preview_scatter = self._axes.scatter(
                xs, ys,
                s=self._marker_size * 1.3,
                facecolors="none",
                edgecolors=_BAND_COLOR,
                linewidths=2,
                zorder=4,
            )

    def _remove_band_visuals(self) -> None:
        """Tear down band rectangle + preview ring and restore the committed
        selection ring. Safe to call without visuals (plain clicks)."""
        if self._band_rect is not None:
            self._band_rect.remove()
            self._band_rect = None
        if self._preview_scatter is not None:
            self._preview_scatter.remove()
            self._preview_scatter = None
        if self._sel_scatter is not None:
            self._sel_scatter.set_visible(True)

    def _cancel_band(self) -> None:
        """Drop an in-progress gesture without emitting (the grid was redrawn
        or cleared; artists may already be destroyed by axes.clear())."""
        self._band_start = None
        self._band_press_px = None
        self._band_last = None
        self._band_moved = False
        self._band_rect = None
        self._preview_scatter = None


def _row_labels(n: int) -> list[str]:
    labels = []
    for i in range(n):
        label = ""
        num = i
        while True:
            label = chr(65 + num % 26) + label
            num = num // 26 - 1
            if num < 0:
                break
        labels.append(label)
    return labels


def _compute_marker_size(n_rows: int, n_cols: int) -> float:
    max_dim = max(n_rows, n_cols)
    if max_dim <= 6:
        return 400
    elif max_dim <= 12:
        return 200
    elif max_dim <= 24:
        return 80
    else:
        return 40


def _get_format_dims(fmt_name: str) -> tuple[int, int]:
    from microVis._settings import PLATE_FORMATS
    return PLATE_FORMATS.get(fmt_name, (16, 24))
