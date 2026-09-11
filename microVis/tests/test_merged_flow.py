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


# ── The flow simulation ──────────────────────────────────────────────────────


def test_merged_db_flow(dataset, window, qt_app):
    from microVis.io.merged_data import MERGED_TABLE

    # ── Select DB: both files of the dataset ──
    window.load_db_files([dataset / "profiler.db", dataset / "infer.db"])
    qt_app.processEvents()

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

    # ── The plot view has the columns and can render ──
    pv = window._plot_view
    pv._x_combo.setCurrentText("umap_1")
    pv._y_combo.setCurrentText("area")
    pv._color_combo.setCurrentText("pred_class")
    pv._on_plot()
    qt_app.processEvents()
    assert pv._figure is not None

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

    # ── Write to DB → merge.db with the integrated table ──
    from PySide6.QtWidgets import QMessageBox
    # Auto-answer the confirm dialog AND auto-dismiss the result dialog —
    # a modal either way would block the offscreen event loop forever.
    orig_question = QMessageBox.question
    orig_information = QMessageBox.information
    orig_warning = QMessageBox.warning
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    try:
        window._on_write_to_db()
    finally:
        QMessageBox.question = orig_question
        QMessageBox.information = orig_information
        QMessageBox.warning = orig_warning
    qt_app.processEvents()

    merge_db = dataset / "merge.db"
    assert merge_db.exists()
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
        qt_app.processEvents()
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
        qt_app.processEvents()
        assert win._merged is not None
        assert len(win._merged.table) == 4   # 2 wells x 2 objects, fused

        pv = win._plot_view
        pv._x_combo.setCurrentText("umap_1")
        pv._y_combo.setCurrentText("umap_2")
        pv._on_plot()
        qt_app.processEvents()
        assert pv._figure is not None

        # Which mask did the crop actually use?
        import microVis.main_window as MW
        real_crop = MW.crop_object_rgb
        used = {}
        def spy(img_data, mask, label, *a, **k):
            used["mask"] = np.array(mask)
            used["label"] = label
            return real_crop(img_data, mask, label, *a, **k)
        monkeypatch.setattr(MW, "crop_object_rgb", spy)

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
