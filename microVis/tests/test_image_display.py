"""ImageDisplay layout: the overlay colorbar must stay BELOW every image row."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")


def _result(row_idx: int, well: str, poly: bool = True) -> dict:
    return {
        "row_idx": row_idx,
        "well": well,
        "field": 1,
        "stack": 1,
        "timepoint": 1,
        "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "polygons": ([(1, np.array([[2, 2], [8, 2], [8, 8], [2, 8]], float))]
                     if poly else None),
        "meta_label": well,
        "overlay_val": 1.0,
        "overlay_col": "area",
        "n_objects": 1,
        "mask": None,
        "obj_values": {1: 1.0},
        "outline_only": False,
    }


def test_colorbar_stays_below_rows():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from microVis.widgets.image_display import ImageDisplay

    disp = ImageDisplay()
    disp.begin_results(32)

    # Incremental insertion path (the one that used to place the colorbar on
    # top): add several results and check the ordering after each one.
    for i, well in enumerate(["A1", "A2", "B1"]):
        disp.add_result(_result(i, well), 32, 0.4, "viridis", None, True,
                        overlay_vmin=0.0, overlay_vmax=10.0)
        cbar = disp._colorbar_widget
        assert cbar is not None
        cbar_idx = disp._layout.indexOf(cbar)
        # The trailing stretch is the very last item; the colorbar is right
        # before it, i.e. below every image row.
        assert cbar_idx == disp._layout.count() - 2
        for row_widget, _ in disp._row_widgets.values():
            assert disp._layout.indexOf(row_widget) < cbar_idx

    # Rebuild path keeps the same invariant.
    disp.resort_cached(32, 0.4, "viridis", None, True)
    cbar = disp._colorbar_widget
    assert cbar is not None
    assert disp._layout.indexOf(cbar) == disp._layout.count() - 2

    disp.clear()
    assert disp._colorbar_widget is None
