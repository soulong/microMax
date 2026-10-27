"""Full-res zoom state machine: restore_view_state must mark the zoom.

Regression: a rebuilt thumbnail restored its zoomed transform and
re-requested the full-res render WITHOUT setting ``_is_full_res`` — the
full-res item rendered ON TOP of the base pixmap, so a later channel
toggle updated only the (hidden) base and the zoomed view kept showing
the OLD channels forever.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtGui import QTransform  # noqa: E402

from microVis.widgets.image_display import _ThumbnailView  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _thumb():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    return _ThumbnailView(rgb, "A01", 1, 1, 1, thumb_size=64, row_idx=7)


def test_restore_view_state_marks_zoom_and_bumps_gen(qt_app):
    thumb = _thumb()
    seen = []
    thumb.full_res_requested.connect(lambda row, gen: seen.append(gen))
    assert not thumb._is_full_res

    thumb.restore_view_state({
        "transform": QTransform.fromScale(4.0, 4.0),
        "h_scroll": 0, "v_scroll": 0,
    })

    # The zoom state must mirror wheelEvent's zoom-in: flagged + fresh gen —
    # the emitted gen is what the FullResWorker round-trips back through
    # set_full_res_pixmap's stale-result guard.
    assert thumb._is_full_res
    assert seen == [1] and thumb._full_res_gen == 1


def test_restore_view_state_ignores_unzoomed_transform(qt_app):
    thumb = _thumb()
    seen = []
    thumb.full_res_requested.connect(lambda row, gen: seen.append(gen))

    thumb.restore_view_state({
        "transform": QTransform.fromScale(0.9, 0.9),
        "h_scroll": 0, "v_scroll": 0,
    })

    assert not thumb._is_full_res and seen == []


def test_toggled_zoom_out_removes_full_res_after_restore(qt_app):
    """A restored zoomed thumb must behave like a wheel-zoomed one: zooming
    back out drops the zoom state AND the full-res item."""
    thumb = _thumb()
    thumb.restore_view_state({
        "transform": QTransform.fromScale(4.0, 4.0),
        "h_scroll": 0, "v_scroll": 0,
    })
    assert thumb._is_full_res

    # Simulate the in-flight render landing (wheelEvent's zoom-in also
    # emits, then the worker delivers through set_full_res_pixmap).
    from PySide6.QtGui import QPixmap
    thumb.set_full_res_pixmap(QPixmap(16, 16), gen=thumb._full_res_gen)
    assert thumb._full_res_item is not None

    thumb._is_full_res = False  # what wheelEvent's zoom-out branch does
    thumb.remove_full_res()
    assert thumb._full_res_item is None
