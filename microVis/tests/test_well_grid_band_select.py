"""Well-grid rubber-band selection: rect math, mode combination and the
press / move / release gesture state machine."""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from microVis.widgets.well_grid_canvas import (  # noqa: E402
    WellGridCanvas,
    _combine_selection,
    _rect_cells,
)


class _FakeDM:
    """Two real wells (A1, A2) on a 96-well plate — every other cell is a
    phantom (drawn but not selectable)."""

    def get_plate_dims(self):
        return 1, 2

    def get_wells(self):
        return ["A1", "A2"]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(app):
    c = WellGridCanvas()
    c.update_grid(
        _FakeDM(),
        table_name="",
        col_val=None,
        agg="mean",
        cmap="viridis",
        palette="tab10",
        fmt_name="96",
        selected_wells=set(),
        force_redraw=True,
    )
    return c


def _event(canvas, xdata, ydata, button=1, dblclick=False, guiEvent=None):
    """Fake matplotlib event over a data-space position; display coords are
    transformed so the drag threshold behaves like the real widget."""
    px, py = canvas._axes.transData.transform((xdata, ydata))
    return SimpleNamespace(
        button=button, dblclick=dblclick, inaxes=canvas._axes,
        xdata=xdata, ydata=ydata, x=px, y=py, guiEvent=guiEvent,
    )


def test_rect_cells_intersects_unit_squares():
    # Band on the centre of cell (row 1, col 1) → exactly that cell.
    assert _rect_cells(1, 1, 1, 1, 8, 12) == [(1, 1)]
    # A band with partial end cells still picks up all three columns.
    assert [c for _, c in _rect_cells(1.2, 2.8, 1, 1, 8, 12)] == [1, 2, 3]
    # Completely outside the grid → empty.
    assert _rect_cells(50, 60, 50, 60, 8, 12) == []


def test_combine_selection_modes():
    assert _combine_selection({"A1"}, {"A2"}, "replace") == {"A2"}
    assert _combine_selection({"A1"}, {"A2"}, "add") == {"A1", "A2"}
    assert _combine_selection({"A1"}, {"A1", "A2"}, "toggle") == {"A2"}


def test_wells_in_rect_filters_phantoms(canvas):
    # Rows 1-3 x cols 1-3: nine cells but only A1/A2 are real.
    assert canvas._wells_in_rect(0.5, 2.5, 0.5, 2.5) == {"A1", "A2"}


def test_band_gesture_emits_replace(canvas):
    got = []
    canvas.wells_band_selected.connect(
        lambda wells, mode: got.append((set(wells), mode)))
    canvas._on_button_press(_event(canvas, 1, 1))
    canvas._on_motion(_event(canvas, 2, 1))
    # Mid-gesture: band rectangle + live preview ring are on the axes.
    assert canvas._band_rect is not None
    assert canvas._preview_scatter is not None
    canvas._on_button_release(_event(canvas, 2, 1))
    # A1..B2 under the band; B1/B2 are phantoms → only A1/A2 selected.
    assert got == [({"A1", "A2"}, "replace")]
    assert canvas._band_rect is None and canvas._preview_scatter is None


def test_click_gesture_toggles_single_well(canvas):
    clicked = []
    banded = []
    canvas.well_clicked.connect(clicked.append)
    canvas.wells_band_selected.connect(
        lambda wells, mode: banded.append((set(wells), mode)))
    canvas._on_button_press(_event(canvas, 1, 1))
    # Tiny move stays below the drag threshold → click, not band.
    canvas._on_motion(_event(canvas, 1.02, 1.02))
    canvas._on_button_release(_event(canvas, 1.02, 1.02))
    assert clicked == ["A1"]
    assert banded == []
    assert canvas._band_start is None


def test_click_phantom_well_ignored(canvas):
    got = []
    canvas.well_clicked.connect(got.append)
    canvas._on_button_press(_event(canvas, 1, 2))  # B1 — phantom
    canvas._on_button_release(_event(canvas, 1, 2))
    assert got == []


def test_empty_band_still_emits_replace(canvas):
    # A plain band over phantom-only rows clears the selection (replace).
    got = []
    canvas.wells_band_selected.connect(
        lambda wells, mode: got.append((set(wells), mode)))
    canvas._on_button_press(_event(canvas, 1, 8))
    canvas._on_motion(_event(canvas, 3, 8))
    canvas._on_button_release(_event(canvas, 3, 8))
    assert got == [(set(), "replace")]


def test_band_mode_follows_modifiers(canvas):
    got = []
    canvas.wells_band_selected.connect(
        lambda wells, mode: got.append((set(wells), mode)))
    ctrl = SimpleNamespace(modifiers=lambda: Qt.KeyboardModifier.ControlModifier)
    shift = SimpleNamespace(modifiers=lambda: Qt.KeyboardModifier.ShiftModifier)

    canvas._on_button_press(_event(canvas, 1, 1, guiEvent=ctrl))
    canvas._on_motion(_event(canvas, 2, 1))
    canvas._on_button_release(_event(canvas, 2, 1))
    assert got[-1][1] == "toggle"

    canvas._on_button_press(_event(canvas, 1, 1, guiEvent=shift))
    canvas._on_motion(_event(canvas, 2, 1))
    canvas._on_button_release(_event(canvas, 2, 1))
    assert got[-1][1] == "add"


def test_right_button_ignored(canvas):
    canvas._on_button_press(_event(canvas, 1, 1, button=3))
    canvas._on_button_release(_event(canvas, 1, 1, button=3))
    assert canvas._band_start is None


def test_redraw_cancels_in_flight_gesture(canvas):
    got = []
    canvas.wells_band_selected.connect(
        lambda wells, mode: got.append((set(wells), mode)))
    canvas._on_button_press(_event(canvas, 1, 1))
    canvas._on_motion(_event(canvas, 2, 1))
    # A grid redraw mid-drag (color-by change) cancels the gesture…
    canvas.update_grid(
        _FakeDM(), table_name="", col_val=None, agg="mean", cmap="viridis",
        palette="tab10", fmt_name="96", selected_wells=set(),
    )
    # …so the subsequent release fires nothing.
    canvas._on_button_release(_event(canvas, 2, 1))
    assert got == []
