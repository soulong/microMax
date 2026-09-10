from __future__ import annotations

import numbers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QSizePolicy, QToolTip, QVBoxLayout, QWidget

from microVis._settings import get_cmap
from microVis.log_utils import get_logger

logger = get_logger("microVis.well_grid_canvas")


class WellGridCanvas(QWidget):
    """Well plate grid visualization using matplotlib scatter plot."""

    well_clicked = Signal(str)

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
        self._canvas.mpl_connect("pick_event", self._on_pick)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def clear(self) -> None:
        """Clear the canvas to an empty state."""
        self._axes.clear()
        self._axes.set_facecolor("#252536")
        self._cbar_ax.clear()
        self._cbar_ax.set_visible(False)
        self._annotations.clear()
        self._sel_scatter = None
        self._scatter = None
        self._well_indices.clear()
        self._data_map.clear()
        self._col_name = None
        self._prev_dims = (0, 0)
        self._prev_fmt = ""
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
        rows, cols = dm.get_plate_dims()
        fmt_rows, fmt_cols = _get_format_dims(fmt_name)
        rows = max(rows, fmt_rows)
        cols = max(cols, fmt_cols)

        row_labels = _row_labels(rows)
        all_wells = dm.get_wells()
        well_set = set(all_wells)
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
        self._data_map = data_map

        marker_size = _compute_marker_size(rows, cols)
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
                picker=True,
                pickradius=5,
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

    def _on_pick(self, event) -> None:
        ind = event.ind
        if ind is not None and len(ind) > 0:
            idx = ind[0]
            for well, i in self._well_indices.items():
                if i == idx:
                    # Ignore clicks on empty (phantom) wells — they are drawn
                    # but not real; selecting them would add invisible entries
                    # to _selected_wells with no ring to clear.
                    if well not in self._real_wells:
                        return
                    QTimer.singleShot(0, lambda w=well: self.well_clicked.emit(w))
                    return

    def _on_motion(self, event) -> None:
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
