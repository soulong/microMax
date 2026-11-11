"""DataPlotView widget tests: facet list checked-first ordering and the
fixed export W/H / point-scale defaults."""

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from microVis.widgets.data_plot import DataPlotView  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app):
    return DataPlotView()


def _facet_names(view):
    return [view._facet_list.item(i).text()
            for i in range(view._facet_list.count())]


def _facet_item(view, name):
    for i in range(view._facet_list.count()):
        it = view._facet_list.item(i)
        if it.text() == name:
            return it
    raise AssertionError(f"facet item {name} missing")


def test_checked_facets_float_to_top(app, view):
    df = pd.DataFrame({"a": range(5), "b": range(5),
                       "c": range(5), "v": range(5)})
    view.set_frame(df)
    assert _facet_names(view) == ["a", "b", "c", "v"]

    # Checking floats the columns above all candidates, in check order.
    _facet_item(view, "c").setCheckState(Qt.Checked)
    _facet_item(view, "a").setCheckState(Qt.Checked)
    assert _facet_names(view) == ["c", "a", "b", "v"]

    # Unchecking drops the column to the FRONT of the candidate block.
    _facet_item(view, "c").setCheckState(Qt.Unchecked)
    assert _facet_names(view) == ["a", "c", "b", "v"]


def test_facet_order_survives_frame_rebuild(app, view):
    df = pd.DataFrame({"a": range(5), "b": range(5),
                       "c": range(5), "v": range(5)})
    view.set_frame(df)
    _facet_item(view, "c").setCheckState(Qt.Checked)

    # A re-merge rebuilds the list in column order but must keep the
    # checked column grouped at the top.
    view.set_frame(df)
    assert _facet_names(view) == ["c", "a", "b", "v"]
    assert view._facet_selected() == ["c"]


def test_clear_facets_restores_candidate_order(app, view):
    df = pd.DataFrame({"a": range(5), "b": range(5),
                       "c": range(5), "v": range(5)})
    view.set_frame(df)
    _facet_item(view, "c").setCheckState(Qt.Checked)
    view._clear_facets()
    assert _facet_names(view) == ["a", "b", "c", "v"]
    assert view._facet_selected() == []


def test_export_and_scale_defaults_are_fixed(app, view):
    """The export W/H pair carries the FIXED 7 × 12 default (no auto-sync)
    and Point scale is a float multiplier defaulting to 1.00."""
    assert view._pdf_w.value() == 7.0
    assert view._pdf_h.value() == 12.0
    assert view._pdf_w.decimals() == 1
    assert view._scatter_size.value() == 1.0
    assert view._scatter_size.decimals() == 2
