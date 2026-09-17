"""End-to-end simulation of the Data page's merged-DB flow (offscreen Qt).

Builds a tiny dataset (images + masks), a profiler.db and an infer.db that
describe the SAME objects, then drives MainWindow through the real flow:

select dir → load dataset → Select DB (both files) → merged table asserts →
plot → left-click a scatter point (cropped-cell popup) → click empty space
(popup hides) → Excel metadata merge → Write to DB (merge.db).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sqlite3  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

pytest.importorskip("PySide6")

IMAGE_PATTERN = r"(?P<well>[A-Z]\d{2})_f(?P<field>\d+)_ch1\.tiff"
MASK_PATTERN = r"(?P<well>[A-Z]\d{2})_f(?P<field>\d+)_ch1_cp_masks_(?P<mask_name>.+)\.png"

WELLS = ["A01", "B01"]


def _canonical(p) -> str:
    import os
    return os.path.normpath(os.path.abspath(str(p))).replace("\\", "/")


# ── Fixture builders ─────────────────────────────────────────────────────────


def _write_images(root: Path):
    """Two images; each mask has objects labeled 1 and 2 (two squares)."""
    from tifffile import imwrite

    entries = []
    img = np.zeros((32, 32), dtype=np.uint16)
    img[4:10, 4:10] = 3000    # object 1
    img[18:26, 18:26] = 5000  # object 2

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:10, 4:10] = 1
    mask[18:26, 18:26] = 2

    for well in WELLS:
        name = f"{well}_f1_ch1.tiff"
        imwrite(str(root / name), img)
        mask_name = f"{well}_f1_ch1_cp_masks_cell.png"
        Image.fromarray(mask).save(str(root / mask_name))
        entries.append((well, name))
    return entries


def _write_profiler_db(path: Path, root: Path, entries):
    """Object table `cell`: measurements for the 4 objects (2 per image)."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE image (well TEXT, directory TEXT, filename TEXT)")
    conn.execute(
        "CREATE TABLE cell (well TEXT, label INTEGER, directory TEXT, "
        "area REAL, intensity_mean REAL)")
    for well, fname in entries:
        directory = _canonical(root)
        conn.execute(
            "INSERT INTO image VALUES (?, ?, ?)", (well, directory, fname))
        conn.execute(
            "INSERT INTO cell VALUES (?, 1, ?, 36.0, ?)",
            (well, directory, 3000.0))
        conn.execute(
            "INSERT INTO cell VALUES (?, 2, ?, 64.0, ?)",
            (well, directory, 5000.0))
    conn.commit()
    conn.close()


def _write_infer_db(path: Path, root: Path, entries):
    """Whole-image-style inference table + a reduction_umap coordinate table.

    Mirrors the real microModel writer: regex captures (well here) are
    stored as extra TEXT columns — they are identity columns for the merge.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE inference ("
        "uid INTEGER PRIMARY KEY AUTOINCREMENT, "
        "directory TEXT NOT NULL, filename TEXT NOT NULL, "
        "well TEXT, "
        "label INTEGER NOT NULL DEFAULT 0, "
        "pred_class TEXT, pred_prob REAL, features BLOB)")
    conn.execute(
        "CREATE TABLE reduction_umap ("
        "uid INTEGER PRIMARY KEY, umap_1 REAL NOT NULL, umap_2 REAL NOT NULL)")
    uid = 0
    for well, fname in entries:
        directory = _canonical(root)
        for label, (u1, u2) in ((1, (2.0, 3.0)), (2, (-1.0, 0.5))):
            uid += 1
            conn.execute(
                "INSERT INTO inference (uid, directory, filename, well, label, "
                "pred_class, pred_prob) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uid, directory, fname, well, label,
                 "drug" if label == 1 else "ctrl", 0.9))
            conn.execute(
                "INSERT INTO reduction_umap VALUES (?, ?, ?)", (uid, u1, u2))
    conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(dataset, qt_app):
    from microVis.main_window import MainWindow

    win = MainWindow()
    win.select_dataset_dir(str(dataset))
    win._data_view.set_patterns(
        image=IMAGE_PATTERN, mask=MASK_PATTERN, subdir="")
    win._on_load_dataset_clicked()
    # Wait for the background dataset-load worker.
    for _ in range(3000):
        qt_app.processEvents()
        if win._dm is not None and win._loaded_dataset_dir:
            break
        qt_app.thread().msleep(5)
    assert win._dm is not None, "dataset did not load"
    qt_app.processEvents()
    yield win
    win.close()


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    entries = _write_images(root)
    _write_profiler_db(root / "profiler.db", root, entries)
    _write_infer_db(root / "infer.db", root, entries)
    return root


def _pump_until(qt_app, cond, timeout_s=10.0,
                message="condition not reached"):
    """Pump the Qt event loop until cond() holds.

    The DB merge, the plot render and the Write-to-DB run on the thread
    pool and deliver through queued signals — a single processEvents()
    cannot wait for them.
    """
    import time
    deadline = time.monotonic() + timeout_s
    while not cond() and time.monotonic() < deadline:
        qt_app.processEvents()
        qt_app.thread().msleep(5)
    qt_app.processEvents()
    assert cond(), message


def _wait_merged(window, qt_app):
    """Wait until the async Select-DB merge has landed."""
    _pump_until(qt_app, lambda: window._merged is not None,
                message="DB merge did not finish")


# ── The flow simulation ──────────────────────────────────────────────────────


def test_merged_db_flow(dataset, window, qt_app):
    from microVis.io.merged_data import MERGED_TABLE

    # ── Select DB: both files of the dataset ──
    window.load_db_files([dataset / "profiler.db", dataset / "infer.db"])
    _wait_merged(window, qt_app)

    assert window._merged is not None
    table = window._merged.table
    # 4 profiler objects fuse with 4 infer rows → 4 merged rows.
    assert len(table) == 4
    # Cross-DB columns coexist in one row.
    for col in ("well", "label", "directory", "area", "intensity_mean",
                "pred_class", "pred_prob", "umap_1", "umap_2"):
        assert col in table.columns, col
        assert table[col].notna().all()
    # Object identity survived the merge (pred_class follows the mask label).
    assert set(table["pred_class"]) == {"drug", "ctrl"}
    assert (table["area"] == 36.0).sum() == 2

    # ── Color by dropdowns: integrated table REPLACES raw profiler tables ──
    gw_labels = [window._grid_controls.column.itemText(i)
                 for i in range(window._grid_controls.column.count())]
    ov_labels = [window._image_controls.overlay_col.itemText(i)
                 for i in range(window._image_controls.overlay_col.count())]
    for labels in (gw_labels, ov_labels):
        assert "merge/area" in labels and "merge/pred_class" in labels
        # No raw profiler-table duplicates once the merged table exists.
        assert not any(lbl.startswith("cell/") for lbl in labels), labels

    # The fused-source readout next to Select DB names the inputs + result.
    status = window._data_view._db_status_label.text()
    assert "profiler.db + infer.db" in status, status
    assert status.endswith("-> merge"), status

    # ── The plot view has the columns and can render ──
    pv = window._plot_view
    pv._x_combo.setCurrentText("umap_1")
    pv._y_combo.setCurrentText("area")
    pv._color_combo.setCurrentText("pred_class")
    pv._on_plot()
    _pump_until(qt_app, lambda: pv._figure is not None,
                message="plot render did not finish")

    # ── Left-click a scatter point → popup shows the cropped cell ──
    from matplotlib.backend_bases import MouseEvent
    ax = pv._figure.axes[0]
    coll = ax.collections[0]
    offsets = np.asarray(coll.get_offsets())
    x_disp, y_disp = ax.transData.transform(offsets[0])
    event = MouseEvent("button_press_event", pv._canvas, x_disp, y_disp, button=1)
    pv._canvas.callbacks.process("button_press_event", event)
    qt_app.processEvents()
    assert pv._popup.isVisible(), "cell popup should show after a point click"

    # The popup must land AT THE CLICK, not vertically mirrored: matplotlib
    # display coords have a bottom-left origin, Qt's mapToGlobal a top-left
    # one. A naive mapping put top-half clicks at the bottom (and vice versa).
    from PySide6.QtCore import QPoint
    from microVis.widgets.data_plot import _mpl_to_qt_xy
    dpr = pv._canvas.devicePixelRatioF() or 1.0
    expected = pv._canvas.mapToGlobal(
        QPoint(*_mpl_to_qt_xy(pv._canvas.height(), dpr, x_disp, y_disp)))
    expected += QPoint(16, 16)
    # Allow a few px of window-manager nudging (offscreen screens are small
    # and may clamp the popup); the regression being guarded is the y mirror.
    actual = pv._popup.pos()
    assert abs(actual.x() - expected.x()) <= 4, (actual, expected)
    assert abs(actual.y() - expected.y()) <= 4, (actual, expected)

    # ── Click on empty canvas space → popup hides ──
    far = ax.transData.transform([[offsets[:, 0].max() + 1e6,
                                   offsets[:, 1].max() + 1e6]])[0]
    event2 = MouseEvent("button_press_event", pv._canvas, far[0], far[1], button=1)
    pv._canvas.callbacks.process("button_press_event", event2)
    qt_app.processEvents()
    assert not pv._popup.isVisible(), "popup should hide on empty-space click"

    # ── Excel metadata merge reaches the merged table ──
    import pandas as pd
    window._metadata_df = pd.DataFrame({"well": WELLS, "batch": [1, 2]})
    window._on_metadata_merge()
    qt_app.processEvents()
    assert "batch" in pv._df.columns
    assert set(pv._df["batch"]) == {1, 2}
    assert "(+ metadata)" in window._data_view._db_status_label.text()

    # Clearing the metadata drops the suffix again.
    window._on_metadata_clear()
    qt_app.processEvents()
    status = window._data_view._db_status_label.text()
    assert "(+ metadata)" not in status and status.endswith("-> merge")
    # The cleared metadata must be merged again (the write below expects it).
    window._metadata_df = pd.DataFrame({"well": WELLS, "batch": [1, 2]})
    window._on_metadata_merge()
    qt_app.processEvents()

    # ── Write to DB → merge.db with the integrated table ──
    from PySide6.QtWidgets import QMessageBox
    # Auto-answer the confirm dialog AND auto-dismiss the result dialog —
    # a modal either way would block the offscreen event loop forever.
    # The stubs must stay installed while the async write completes.
    orig_question = QMessageBox.question
    orig_information = QMessageBox.information
    orig_warning = QMessageBox.warning
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    try:
        window._on_write_to_db()
        merge_db = dataset / "merge.db"
        _pump_until(qt_app, lambda: merge_db.exists(),
                    message="merged DB write did not finish")
    finally:
        QMessageBox.question = orig_question
        QMessageBox.information = orig_information
        QMessageBox.warning = orig_warning

    conn = sqlite3.connect(str(merge_db))
    rows = conn.execute(f"SELECT COUNT(*) FROM {MERGED_TABLE}").fetchone()[0]
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({MERGED_TABLE})")}
    conn.close()
    assert rows == 4
    assert {"area", "pred_class", "umap_1", "batch", "well", "label"} <= cols

    # ── merge.db itself is selectable as a DB source ──
    from microVis.io.merged_data import MergedData
    merged_again = MergedData.load([merge_db])
    assert len(merged_again.table) == 4


def test_reset_then_reload_restores_plot_area(dataset, window, qt_app):
    """Reset → reload → Select DB must bring the lower half back.

    Regression: DataView.reset() used to detach the installed plot view, so
    the controls+canvas column stayed gone forever after a Reset, no matter
    which DB was selected afterwards.
    """
    window.load_db_files([dataset / "profiler.db", dataset / "infer.db"])
    _wait_merged(window, qt_app)
    assert window._data_view._plot_view is window._plot_view

    window._on_full_reset()
    qt_app.processEvents()
    assert window._data_view._plot_view is window._plot_view, (
        "Reset must not detach the plot view from the Data page")

    # Load the dataset again and select the same DBs.
    window.select_dataset_dir(str(dataset))
    window._data_view.set_patterns(
        image=IMAGE_PATTERN, mask=MASK_PATTERN, subdir="")
    window._on_load_dataset_clicked()
    for _ in range(3000):
        qt_app.processEvents()
        if window._dm is not None and window._loaded_dataset_dir:
            break
        qt_app.thread().msleep(5)
    assert window._dm is not None, "dataset did not load after reset"
    window.load_db_files([dataset / "profiler.db", dataset / "infer.db"])
    _wait_merged(window, qt_app)
    assert window._plot_view._df is not None
    # The view is back in the page layout (offscreen windows are never
    # shown, so check ancestry, not isVisible()).
    assert window._plot_view.isVisibleTo(window._data_view)


def test_reset_restores_plot_controls(dataset, window, qt_app):
    """The Data-page Reset must clear the plot-control pickers too.

    Regression: Reset dropped the table/figure but left every picker with
    the previous DB's columns and options.
    """
    window.load_db_files([dataset / "profiler.db", dataset / "infer.db"])
    _wait_merged(window, qt_app)
    pv = window._plot_view
    pv._chart_combo.setCurrentText("boxplot")
    pv._filter_edit.setText("area > 1")
    pv._x_combo.setCurrentText("well")
    pv._y_combo.setCurrentText("area")
    pv._color_combo.setCurrentText("pred_class")
    pv._size_combo.setCurrentText("area")
    pv._facet_filter.setText("pred")
    pv._facet_cols.setValue(5)
    pv._scatter_size.setValue(50.0)
    pv._cap_spin.setValue(100)
    pv._show_points.setChecked(False)
    qt_app.processEvents()

    window._on_full_reset()
    qt_app.processEvents()

    assert pv._df is None
    assert pv._chart_combo.currentText() == "scatter"
    assert pv._filter_edit.text() == ""
    assert pv._x_combo.currentText() == ""
    assert pv._y_combo.currentText() == ""
    assert pv._color_combo.currentText() == ""
    assert pv._size_combo.currentText() == ""
    assert pv._facet_list.count() == 0
    assert pv._facet_filter.text() == ""
    assert pv._facet_cols.value() == 3
    assert pv._colors_combo.count() == 0
    assert pv._scatter_size.value() == 20.0
    assert pv._cap_spin.value() == 20000
    assert pv._show_points.isChecked()


def test_cell_popup_dismissed_on_page_switch_and_outside_click(
        dataset, window, qt_app):
    """The cropped-cell popup must vanish as soon as the plot is left."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from matplotlib.backend_bases import MouseEvent

    # The page-switch hide relies on real visibility transitions, so show
    # the window (still offscreen) before driving the flow.
    window.show()
    qt_app.processEvents()
    window.load_db_files([dataset / "profiler.db", dataset / "infer.db"])
    _wait_merged(window, qt_app)
    pv = window._plot_view
    pv._x_combo.setCurrentText("umap_1")
    pv._y_combo.setCurrentText("area")
    pv._on_plot()
    _pump_until(qt_app, lambda: pv._figure is not None,
                message="plot render did not finish")

    def click_first_point():
        ax = pv._figure.axes[0]
        offsets = np.asarray(ax.collections[0].get_offsets())
        x_disp, y_disp = ax.transData.transform(offsets[0])
        pv._canvas.callbacks.process(
            "button_press_event",
            MouseEvent("button_press_event", pv._canvas, x_disp, y_disp,
                       button=1))
        qt_app.processEvents()

    # A point click shows the popup …
    click_first_point()
    assert pv._popup.isVisible()
    # … switching to the Image page hides it with the view.
    window._switch_tab(1)
    qt_app.processEvents()
    assert not pv._popup.isVisible(), "popup survived a page switch"
    window._switch_tab(0)
    qt_app.processEvents()

    # Re-show it, then press a control outside the canvas.
    click_first_point()
    assert pv._popup.isVisible()
    QTest.mousePress(window._data_view._btn_dataset_browse, Qt.LeftButton)
    qt_app.processEvents()
    assert not pv._popup.isVisible(), "popup survived an outside click"

    # Losing the window / application focus (user switched apps) hides it.
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    click_first_point()
    assert pv._popup.isVisible()
    QApplication.sendEvent(window, QEvent(QEvent.WindowDeactivate))
    qt_app.processEvents()
    assert not pv._popup.isVisible(), "popup survived losing the window focus"

    click_first_point()
    assert pv._popup.isVisible()
    QApplication.sendEvent(qt_app, QEvent(QEvent.ApplicationDeactivate))
    qt_app.processEvents()
    assert not pv._popup.isVisible(), "popup survived losing the app focus"


def test_channel_vmin_vmax_follow_integer_dtype(dataset, window, qt_app):
    """uint16 images show vmin/vmax as whole numbers (no decimals)."""
    assert window._dm.img_dtype == "uint16"
    rows = window._image_controls._channel_widgets
    assert rows, "channel controls should be built after a dataset load"
    for row in rows.values():
        assert row._vmin.decimals() == 0
        assert row._vmax.decimals() == 0
        assert "." not in row._vmin.text()
        assert "." not in row._vmax.text()

    # Fractional datasets keep their decimals.
    from microVis.widgets.channel_controls import ChannelControls
    fl = ChannelControls("f", {}, max_value=1.0, integer=False)
    assert fl._vmax.decimals() == 4


def test_facet_filter_hides_columns_but_keeps_selection(dataset, window, qt_app):
    """The type-to-filter box hides rows without touching the selection."""
    window.load_db_files([dataset / "profiler.db", dataset / "infer.db"])
    _wait_merged(window, qt_app)
    pv = window._plot_view
    checked_before = pv._facet_selected()

    pv._facet_filter.setText("pred")
    qt_app.processEvents()
    visible = [pv._facet_list.item(i).text()
               for i in range(pv._facet_list.count())
               if not pv._facet_list.item(i).isHidden()]
    assert visible and all("pred" in t.lower() for t in visible)
    assert pv._facet_selected() == checked_before

    pv._facet_filter.clear()
    qt_app.processEvents()
    assert all(not pv._facet_list.item(i).isHidden()
               for i in range(pv._facet_list.count()))


def test_merged_data_join_and_prefix_rules(tmp_path):
    """Unit-level checks of the fold rules (identity join, prefixing)."""
    from microVis.io.merged_data import MergedData

    p1 = tmp_path / "profiler.db"
    conn = sqlite3.connect(str(p1))
    conn.execute("CREATE TABLE cell (well TEXT, label INTEGER, directory TEXT, area REAL)")
    conn.execute("INSERT INTO cell VALUES ('A01', 1, 'C:/x', 10.0)")
    conn.commit()
    conn.close()

    p2 = tmp_path / "second.db"
    conn = sqlite3.connect(str(p2))
    conn.execute(
        "CREATE TABLE cell (well TEXT, label INTEGER, directory TEXT, "
        "area REAL, prob REAL)")
    conn.execute("INSERT INTO cell VALUES ('A01', 1, 'C:/x', 99.0, 0.5)")
    conn.commit()
    conn.close()

    merged = MergedData.load([p1, p2])
    t = merged.table
    assert len(t) == 1                       # fused, not duplicated
    assert t["area"].iloc[0] == 10.0         # first source keeps the bare name
    assert "second/area" in t.columns        # colliding incoming column prefixed
    assert t["second/area"].iloc[0] == 99.0
    assert t["prob"].iloc[0] == 0.5          # non-colliding stays bare


def test_different_masks_stack_and_log_file_in_dataset(tmp_path, qt_app):
    """Infer results of ANOTHER mask stack instead of fusing, and the
    dataset keeps its own microVis.log once loaded."""
    from tifffile import imwrite
    from PIL import Image

    root = tmp_path / "mask_ds"
    root.mkdir()
    img = np.zeros((32, 32), dtype=np.uint16)
    img[4:10, 4:10] = 3000
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:10, 4:10] = 1
    mask[18:26, 18:26] = 2
    for well in WELLS:
        imwrite(str(root / f"{well}_f1_ch1.tiff"), img)
        Image.fromarray(mask).save(
            str(root / f"{well}_f1_ch1_cp_masks_cell.png"))

    conn = sqlite3.connect(str(root / "profiler.db"))
    conn.execute("CREATE TABLE cell (well TEXT, field TEXT, directory TEXT, "
                 "label INTEGER, area REAL)")
    conn2 = sqlite3.connect(str(root / "infer.db"))
    conn2.execute(
        "CREATE TABLE inference (uid INTEGER PRIMARY KEY AUTOINCREMENT, "
        "directory TEXT, mask_name TEXT, well TEXT, field TEXT, label "
        "INTEGER, pred_class TEXT)")
    for well in WELLS:
        for label in (1, 2):
            conn.execute("INSERT INTO cell VALUES (?, '1', ?, ?, 36.0)",
                         (well, "Images", label))
            conn2.execute(
                "INSERT INTO inference (directory, mask_name, well, field, "
                "label, pred_class) VALUES (?, 'nuclei', ?, '1', ?, 'x')",
                ("Images", well, label))
    conn.commit()
    conn.close()
    conn2.commit()
    conn2.close()

    from microVis.main_window import MainWindow
    win = MainWindow()
    try:
        win.select_dataset_dir(str(root))
        win._data_view.set_patterns(
            image=IMAGE_PATTERN, mask=MASK_PATTERN, subdir="")
        win._on_load_dataset_clicked()
        for _ in range(3000):
            qt_app.processEvents()
            if win._dm is not None and win._loaded_dataset_dir:
                break
            qt_app.thread().msleep(5)
        qt_app.processEvents()

        # The dataset directory owns its log file.
        assert (root / "microVis.log").exists()

        win.load_db_files([root / "profiler.db", root / "infer.db"])
        _wait_merged(win, qt_app)
        table = win._merged.table
        # nuclei infer rows must NOT fuse with cell profiler rows.
        assert len(table) == 8
        assert set(table["mask"]) == {"cell", "nuclei"}
        cell = table[table["mask"] == "cell"]
        assert cell["area"].notna().all() and cell["pred_class"].isna().all()
        nuc = table[table["mask"] == "nuclei"]
        assert nuc["area"].isna().all() and nuc["pred_class"].notna().all()
    finally:
        win.close()


def test_legacy_relative_dir_click_crops_clicked_site(tmp_path, qt_app,
                                                      monkeypatch):
    """Legacy DBs (root-RELATIVE directory) must not break click-to-cell.

    Older microModel/microProfiler writes stored `directory = 'Images'` in
    every row — identical for ALL sites, so a directory-only match points
    at the wrong image and the crop fails (or used to crash). The clicked
    object must be located through its identity columns (well/field/...)
    and cropped from ITS OWN site's mask.
    """
    from tifffile import imwrite
    from PIL import Image

    root = tmp_path / "legacy_ds"
    root.mkdir()

    # Per-well masks at DIFFERENT pixel positions → a crop from the wrong
    # site's mask is detectable by content.
    img = np.zeros((32, 32), dtype=np.uint16)
    img[4:10, 4:10] = 3000
    img[18:26, 18:26] = 5000
    masks = {}
    for well in WELLS:
        mask = np.zeros((32, 32), dtype=np.uint8)
        r0 = 4 if well == "A01" else 6
        c0 = 4 if well == "A01" else 7
        mask[r0:r0 + 6, c0:c0 + 6] = 1          # object 1
        mask[r0 + 14:r0 + 22, c0 + 14:c0 + 20] = 2  # object 2
        imwrite(str(root / f"{well}_f1_ch1.tiff"), img)
        Image.fromarray(mask).save(
            str(root / f"{well}_f1_ch1_cp_masks_cell.png"))
        masks[well] = mask

    # DBs in the LEGACY format: root-relative directory, one identical value
    # on every row — a directory-only match therefore hits EVERY site and
    # must not be allowed to decide the location (regression of the
    # wrong-site bug: clicked B01 objects were cropped from A01's mask).
    legacy_dir = "."
    conn = sqlite3.connect(str(root / "profiler.db"))
    conn.execute(
        "CREATE TABLE cell (well TEXT, field TEXT, stack TEXT, timepoint TEXT, "
        "directory TEXT, label INTEGER, area REAL)")
    conn2 = sqlite3.connect(str(root / "infer.db"))
    conn2.execute(
        "CREATE TABLE inference (uid INTEGER PRIMARY KEY AUTOINCREMENT, "
        "directory TEXT, well TEXT, field TEXT, stack TEXT, timepoint TEXT, "
        "label INTEGER, pred_class TEXT, pred_prob REAL)")
    conn2.execute(
        "CREATE TABLE reduction_umap ("
        "uid INTEGER PRIMARY KEY, umap_1 REAL NOT NULL, umap_2 REAL NOT NULL)")
    uid = 0
    for well in WELLS:
        for label, area in ((1, 36.0), (2, 64.0)):
            conn.execute(
                "INSERT INTO cell VALUES (?, '1', '1', '1', ?, ?, ?)",
                (well, legacy_dir, label, area))
            uid += 1
            conn2.execute(
                "INSERT INTO inference (directory, well, field, stack, "
                "timepoint, label, pred_class, pred_prob) "
                "VALUES (?, ?, '1', '1', '1', ?, 'drug', 0.9)",
                (legacy_dir, well, label))
            # Distinct coordinates per object so the scatter has one point
            # per object and the clicked point identifies its row exactly.
            conn2.execute("INSERT INTO reduction_umap VALUES (?, ?, ?)",
                          (uid, float(uid), float(uid) * 10))
    conn.commit()
    conn.close()
    conn2.commit()
    conn2.close()

    from microVis.main_window import MainWindow
    win = MainWindow()
    try:
        win.select_dataset_dir(str(root))
        win._data_view.set_patterns(
            image=IMAGE_PATTERN, mask=MASK_PATTERN, subdir="")
        win._on_load_dataset_clicked()
        for _ in range(3000):
            qt_app.processEvents()
            if win._dm is not None and win._loaded_dataset_dir:
                break
            qt_app.thread().msleep(5)
        qt_app.processEvents()
        assert win._dm is not None, "dataset did not load"

        win.load_db_files([root / "profiler.db", root / "infer.db"])
        _wait_merged(win, qt_app)
        assert win._merged is not None
        assert len(win._merged.table) == 4   # 2 wells x 2 objects, fused

        pv = win._plot_view
        pv._x_combo.setCurrentText("umap_1")
        pv._y_combo.setCurrentText("umap_2")
        pv._on_plot()
        _pump_until(qt_app, lambda: pv._figure is not None,
                    message="plot render did not finish")

        # Which mask did the crop actually use?
        import microVis.main_window as MW
        real_crop = MW.crop_cell_rgb_normalized
        used = {}
        def spy(img_data, mask, label, *a, **k):
            used["mask"] = np.array(mask)
            used["label"] = label
            return real_crop(img_data, mask, label, *a, **k)
        monkeypatch.setattr(MW, "crop_cell_rgb_normalized", spy)

        # Click the scatter point of well B01, label 2 (via its plotted
        # coordinates recovered from the tagged collection).
        from matplotlib.backend_bases import MouseEvent
        ax = pv._figure.axes[0]
        coll = ax.collections[0]
        rows = coll._microvis_rows
        target = rows.index[(rows["well"] == "B01")
                            & (rows["label"] == 2)][0]
        off = np.asarray(coll.get_offsets())[target]
        x_disp, y_disp = ax.transData.transform(off)
        ev = MouseEvent("button_press_event", pv._canvas,
                        x_disp, y_disp, button=1)
        pv._canvas.callbacks.process("button_press_event", ev)
        qt_app.processEvents()

        assert pv._popup.isVisible(), "popup should show for the clicked cell"
        assert used.get("label") == 2
        # The crop MUST come from B01's own mask, not another site's.
        assert np.array_equal(used["mask"], masks["B01"]), \
            "clicked cell was cropped from the WRONG site's mask"
    finally:
        win.close()


def test_mpl_to_qt_xy_flips_and_scales():
    """mpl physical/bottom-origin → Qt logical/top-origin conversion."""
    from microVis.widgets.data_plot import _mpl_to_qt_xy

    # y=500 physical on a 600px canvas is 100px from the top.
    assert _mpl_to_qt_xy(600, 1.0, 100, 500) == (100, 100)
    # HiDPI: physical coords are dpr-scaled, Qt coords are logical.
    assert _mpl_to_qt_xy(600, 2.0, 200, 1000) == (100, 100)
    # Canvas corners map to corners.
    assert _mpl_to_qt_xy(600, 1.0, 0, 600) == (0, 0)
    assert _mpl_to_qt_xy(600, 1.0, 0, 0) == (0, 600)


def test_cell_render_widget_guard_accepts_non_widgets(qt_app):
    """The app-wide event filter also delivers non-QWidget watchers
    (e.g. a QWindow) — the cell-render exemption must not raise on them."""
    from microVis.widgets.data_plot import DataPlotView

    pv = DataPlotView()
    try:
        assert pv._is_cell_render_widget(None) is False
        assert pv._is_cell_render_widget(123) is False
        from PySide6.QtGui import QWindow
        assert pv._is_cell_render_widget(QWindow()) is False
        assert pv._is_cell_render_widget(pv._cell_low) is True
        # A child of a control (e.g. the spin box's internal line edit).
        assert pv._is_cell_render_widget(pv._cell_low.lineEdit()) is True
        # And the eventFilter itself survives a synthetic non-widget event.
        from PySide6.QtCore import QEvent
        win = QWindow()
        assert pv.eventFilter(win, QEvent(QEvent.Type.MouseButtonPress)) is False
    finally:
        pv.deleteLater()


# ── DB-only session: everything works without a loaded dataset ────────────────


def _db_only_paths(tmp_path):
    """profiler.db + infer.db describing 4 objects (no images needed)."""
    entries = [(well, f"{well}_f1_ch1.tiff") for well in WELLS]
    profiler = tmp_path / "profiler.db"
    infer = tmp_path / "infer.db"
    _write_profiler_db(profiler, tmp_path, entries)
    _write_infer_db(infer, tmp_path, entries)
    return [profiler, infer]


def test_db_merge_and_metadata_merge_without_dataset(qt_app, tmp_path):
    """Select DB(s) + Excel metadata merge must work with NO dataset loaded."""
    from microVis.main_window import MainWindow

    win = MainWindow()
    try:
        assert win._dm is None  # no dataset loaded
        win.load_db_files(_db_only_paths(tmp_path))
        _wait_merged(win, qt_app)

        assert win._merged is not None
        assert len(win._merged.table) == 4
        # The plot view adopted the integrated table and can draw it.
        assert win._plot_view._df is not None
        win._plot_view._x_combo.setCurrentText("umap_1")
        win._plot_view._y_combo.setCurrentText("umap_2")
        win._plot_view._on_plot()
        _pump_until(qt_app, lambda: win._plot_view._figure is not None,
                    message="plot render did not finish")
        assert "profiler.db + infer.db" in win._data_view._db_status_label.text()
        # The DB selection enables the metadata browse + clear actions.
        assert win._data_view.metadata_browse_button.isEnabled()

        # Excel metadata merges into the integrated table without a dataset.
        import pandas as pd
        win._metadata_df = pd.DataFrame({"well": WELLS, "batch": [1, 2]})
        win._on_metadata_merge()
        assert "batch" in win._plot_view._df.columns
        assert "(+ metadata)" in win._data_view._db_status_label.text()
    finally:
        win.close()


def test_write_to_db_without_dataset_asks_location(qt_app, tmp_path, monkeypatch):
    """Write to DB without a dataset falls back to a save dialog."""
    from PySide6.QtWidgets import QFileDialog

    from microVis.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_db_files(_db_only_paths(tmp_path))
        _wait_merged(win, qt_app)
        import pandas as pd
        win._metadata_df = pd.DataFrame({"well": WELLS, "batch": [1, 2]})
        win._on_metadata_merge()

        out = tmp_path / "chosen" / "out.db"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(out), "SQLite DB (*.db)")))
        # The success confirmation is a MODAL box — stub it like the main
        # e2e test does, or the offscreen run blocks forever.
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(
            QMessageBox, "information",
            staticmethod(lambda *a, **k: QMessageBox.Ok))
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: QMessageBox.Ok))
        win._on_write_to_db()
        _pump_until(qt_app, lambda: out.exists(),
                    message="merged DB write did not finish")

        # The written file re-loads as a fused table (with the mask tag).
        from microVis.io.merged_data import MergedData
        reloaded = MergedData.load([out])
        assert len(reloaded.table) == 4
        assert "batch" in reloaded.table.columns
    finally:
        win.close()


def test_facet_clear_unchecks_all(qt_app):
    """The facet Clear button unchecks every facet variable in one click."""
    import pandas as pd
    from PySide6.QtCore import Qt

    from microVis.widgets.data_plot import DataPlotView

    pv = DataPlotView()
    try:
        pv.set_frame(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
        assert pv._facet_list.count() == 2
        for i in range(pv._facet_list.count()):
            pv._facet_list.item(i).setCheckState(Qt.Checked)
        assert len(pv._facet_selected()) == 2
        pv._facet_clear_btn.click()
        assert pv._facet_selected() == []
    finally:
        pv.deleteLater()


def test_plot_pickers_default_empty(qt_app):
    """X/Y/Color/Size start EMPTY (no '(none)' item): empty means unset,
    and the editable combo invites typing to filter-find a column."""
    import pandas as pd

    from microVis.widgets.data_plot import DataPlotView

    pv = DataPlotView()
    try:
        pv.set_frame(pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0],
                                   "w": ["x", "y"]}))
        for combo in (pv._x_combo, pv._y_combo, pv._color_combo,
                      pv._size_combo):
            assert combo.currentText() == ""
        # All columns are listed (numeric-only for Size) for the filter.
        assert pv._x_combo.count() == 3
        assert pv._size_combo.count() == 2
        assert "(none)" not in [pv._x_combo.itemText(i)
                                for i in range(pv._x_combo.count())]
        # An empty picker is 'unset'; picking a column works as before.
        assert pv._selected(pv._x_combo) is None
        pv._x_combo.setCurrentText("a")
        assert pv._selected(pv._x_combo) == "a"
    finally:
        pv.deleteLater()


def test_wheel_blocker_passes_through_open_popup(qt_app, monkeypatch):
    """Wheel inside an OPEN combo popup must reach the popup (scroll the
    option list) instead of being redirected to the enclosing scroll area."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QComboBox, QScrollArea, QWidget

    from microVis.app import _WheelBlocker

    host = QScrollArea()
    inner = QComboBox()
    host.setWidget(inner)
    popup = QComboBox()          # stands in for the open popup container
    child = QWidget(popup)       # the wheel target inside the popup

    def _wheel():
        return QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0),
                           QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                           Qt.ScrollUpdate, False)

    monkeypatch.setattr(QApplication, "widgetAt",
                        staticmethod(lambda *a, **k: child))
    monkeypatch.setattr(QApplication, "activePopupWidget",
                        staticmethod(lambda: popup))
    blocker = _WheelBlocker()

    # Popup "open": the event passes through (False = not consumed).
    assert blocker.eventFilter(inner, _wheel()) is False

    # Popup closed: the wheel is consumed / redirected (True) so the page
    # scrolls without changing the combo value.
    monkeypatch.setattr(QApplication, "activePopupWidget",
                        staticmethod(lambda: None))
    assert blocker.eventFilter(inner, _wheel()) is True


def test_pane_input_font_matches_data_page(qt_app):
    """Image-page control panes use the SAME 9pt input text as the Data
    page (the global stylesheet's 11pt must not leak into the panes)."""
    from PySide6.QtWidgets import QLabel

    from microVis.widgets.well_grid_controls import WellGridControls

    # The offscreen font database reports identical metrics for 9pt/11pt,
    # so pin the contract on the style source: every pane input selector
    # must carry the 9pt content scale.
    from microVis.widgets.ui_spec import controls_pane_style

    style = controls_pane_style()
    for selector in ("QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QCheckBox",
                     "QLabel"):
        block = style.split(selector, 1)[1].split("}", 1)[0]
        assert "font-size: 9pt" in block, f"{selector} missing 9pt"

    bar = WellGridControls()
    try:
        bar.show()
        qt_app.processEvents()
        small = QLabel("Color by")
        small.setStyleSheet("font-size: 9pt;")
        # The pane combo resolves to the same metrics as an explicit 9pt.
        assert bar.column.fontMetrics().height() == small.fontMetrics().height()
    finally:
        bar.deleteLater()


def test_control_panes_share_fixed_width(qt_app):
    """Every page's left control rail (Image sidebar, well-grid bar, Data
    plot column) shares ONE fixed width."""
    from PySide6.QtWidgets import QApplication, QScrollArea

    from microVis.widgets.data_plot import DataPlotView
    from microVis.widgets.image_controls import ImageControls
    from microVis.widgets.ui_spec import CONTROLS_WIDTH
    from microVis.widgets.well_grid_controls import WellGridControls

    panes = [ImageControls(), WellGridControls()]
    try:
        for pane in panes:
            pane.show()
        plot = DataPlotView()
        plot.show()
        qt_app.processEvents()
        # Image sidebar + well-grid bar are the rail themselves.
        for pane in panes:
            assert pane.width() == CONTROLS_WIDTH, type(pane).__name__
        # The Data plot column's controls scroll area carries the width.
        areas = [a for a in plot.findChildren(QScrollArea)
                 if a.parentWidget() is plot]
        assert areas[0].width() == CONTROLS_WIDTH
        QApplication.processEvents()
    finally:
        for pane in panes:
            pane.deleteLater()
        plot.deleteLater()
