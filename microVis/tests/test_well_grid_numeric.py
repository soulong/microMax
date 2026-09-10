"""Well-grid numeric rendering must accept numpy scalar values."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from microVis.widgets.well_grid_canvas import WellGridCanvas  # noqa: E402


class _FakeDM:
    def get_plate_dims(self):
        return 1, 2

    def get_wells(self):
        return ["A1", "A2"]


def test_update_grid_renders_numpy_integer_values():
    """np.int64 (Excel metadata / sum aggregation) must be treated as numeric;
    it is not an int subclass, so the old isinstance check rendered gray."""
    app = QApplication.instance() or QApplication([])
    canvas = WellGridCanvas()
    canvas.update_grid(
        _FakeDM(),
        table_name="cell",
        col_val=("shape_area", True),
        agg="sum",
        cmap="viridis",
        palette="tab10",
        fmt_name="96",
        selected_wells={"A1"},
        value_map={"A1": np.int64(5), "A2": np.int64(9)},
        force_redraw=True,
    )
    assert canvas._scatter is not None
    colors = canvas._scatter.get_facecolors()
    # A1 is plotted first, A2 second (row-major); with the bug both would
    # fall through to the same neutral color.
    assert not np.allclose(colors[0], colors[1])
    app.processEvents()
