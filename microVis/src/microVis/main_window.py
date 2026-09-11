from __future__ import annotations

from collections import OrderedDict
from natsort import natsort_key
from pathlib import Path
from typing import Any
import threading

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from microBase import (
    SessionFile,
    DEFAULT_IMAGE_PATTERN,
    DEFAULT_MASK_PATTERN,
    DEFAULT_IMAGE_SUBDIR_PATTERN,
)

from microVis._settings import (
    AGG_METHODS,
    CMAP_OPTIONS,
    DEFAULT_CHANNEL_COLORS,
    DEFAULT_CMAP,
    DEFAULT_PLATE,
    DTYPE_MAX,
    PLATE_FORMATS,
    QUALITATIVE_PALETTES,
)
from microVis.io.data_module import DataModule, _safe_str, aggregate_by_well
from microBase.db_contracts import DIRECTORY_COLUMN
from microVis.io import merged_data
from microVis.io.merged_data import (
    MergedData,
    merge_metadata_into,
    write_merged_db,
)
from microVis.log_utils import get_logger, set_log_file
from microVis.widgets._event_filter import RotatedLabel
from microVis.widgets.path_drop import enable_path_drop
from microVis.widgets.data_view import DataView
from microVis.widgets.image_controls import ImageControls
from microVis.widgets.image_display import ImageDisplay
from microVis.widgets.label_annotation import LabelAnnotationPanel, ObjectKey
from microVis.widgets.pixel_info import PixelInfo
from microVis.widgets.data_plot import DataPlotView
from microVis.widgets.ui_spec import (
    FORM_LABEL_WIDTH_WIDE,
    H_SPLITTER_SIZES,
    LABEL_CLASS_RATIOS,
    NO_LABEL_RATIOS,
    V_SPLITTER_SIZES,
)
from microVis.widgets.well_grid_canvas import WellGridCanvas
from microVis.widgets.well_grid_controls import WellGridControls
from microVis.worker import CropWorker, ImageWorker, ImageWorkerConfig, crop_object_rgb

logger = get_logger("microVis.main_window")

# Cap for the viewer's full-resolution row cache: "All wells" on a large
# dataset must not pin every payload in RAM forever.
RAW_CACHE_MAX_BYTES = 12 * 1024**3  # 12 GiB of payload bytes


class _RowCache:
    """row_idx → payload LRU cache capped by total CONTENT bytes.

    The viewer memoizes each loaded row's full-resolution (image, masks) so
    channel toggles, re-sorts and crops are instant. Unbounded retention let
    a large "All wells" session exhaust RAM, so once the summed payload size
    exceeds ``max_bytes`` the least-recently-used rows are evicted. A miss
    is harmless — the refresh paths simply re-dispatch an image worker.

    Methods mirror the dict subset the window uses (get / __setitem__ /
    clear / __bool__), and a lock guards the bookkeeping because
    _on_auto_all loads rows from a ThreadPoolExecutor.
    """

    def __init__(self, max_bytes: int = RAW_CACHE_MAX_BYTES):
        self._max_bytes = max_bytes
        self._entries: "OrderedDict[int, tuple]" = OrderedDict()  # row_idx → (payload, nbytes)
        self._total = 0
        self._lock = threading.Lock()

    @staticmethod
    def _payload_bytes(value) -> int:
        """Approximate payload size: sum ndarray nbytes across tuples/dicts."""
        total = 0
        stack = [value]
        while stack:
            obj = stack.pop()
            if isinstance(obj, np.ndarray):
                total += int(obj.nbytes)
            elif isinstance(obj, (tuple, list)):
                stack.extend(obj)
            elif isinstance(obj, dict):
                stack.extend(obj.values())
        return total

    def get(self, row_idx, default=None):
        with self._lock:
            entry = self._entries.get(row_idx)
            if entry is None:
                return default
            self._entries.move_to_end(row_idx)  # touch → most recently used
            return entry[0]

    def __setitem__(self, row_idx, value) -> None:
        nbytes = self._payload_bytes(value)
        with self._lock:
            old = self._entries.pop(row_idx, None)
            if old is not None:
                self._total -= old[1]
            self._entries[row_idx] = (value, nbytes)
            self._total += nbytes
            while self._total > self._max_bytes and len(self._entries) > 1:
                # Evict from the least-recently-used end; keep at least the
                # entry just inserted even if it alone exceeds the cap.
                _, (_, evicted_bytes) = self._entries.popitem(last=False)
                self._total -= evicted_bytes

    def __contains__(self, row_idx) -> bool:
        with self._lock:
            return row_idx in self._entries

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total = 0


def _build_meta_label(meta: "pd.DataFrame", row_idx: int,
                      structural_cols: list[str],
                      extra_cols: list[str]) -> str:
    """Build a display label from available metadata columns.

    Shows structural columns (well/field/stack/timepoint) that are present,
    using the conventional format: 'A1 f1 z1 t1'. If no structural columns
    exist, falls back to extra cols as 'col=val' pairs. If no extra cols
    either, falls back to 'row {row_idx}'.
    """
    parts: list[str] = []
    for col in structural_cols:
        if col in meta.columns:
            v = _safe_str(meta, row_idx, col)
            if col == "well":
                parts.append(v)
            elif col == "field":
                parts.append(f"f{v}")
            elif col == "stack":
                parts.append(f"z{v}")
            elif col == "timepoint":
                parts.append(f"t{v}")
            else:
                parts.append(f"{col}={v}")
    if parts:
        return " ".join(p for p in parts if p)
    # No structural cols — use extra cols
    for col in extra_cols:
        if col in meta.columns:
            v = _safe_str(meta, row_idx, col)
            parts.append(f"{col}={v}")
    if parts:
        return " ".join(parts)
    return f"row {row_idx}"




def _abs_norm_dir(p) -> str:
    """Normalized absolute forward-slash form of a path (join/compare key)."""
    import os
    return os.path.normcase(
        os.path.normpath(os.path.abspath(str(p))).replace("\\", "/"))


def _rel_norm_dir(p) -> str:
    """Normalized forward-slash form of a path WITHOUT anchoring it at the
    CWD — the comparison key for DB rows written by older microModel /
    microProfiler versions, whose `directory` is root-relative (e.g.
    'Images'). Absolute inputs keep their absolute form, so comparing a
    relative DB dir against an absolute one simply never matches."""
    import os
    return os.path.normcase(os.path.normpath(str(p)).replace("\\", "/"))


class MainWindow(QMainWindow):
    """Top-level application window for microVis."""

    def __init__(self, dataset_dir: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(1200, 800)
        self.resize(1500, 1000)
        self.setWindowTitle("microVis")

        # Data
        self._dm: DataModule | None = None
        self._dataset_dir: str | None = None
        self._session: SessionFile | None = None
        # session.yml is read only on the FIRST load of a given directory.
        # Subsequent loads/reloads of the same directory use current GUI state
        # as the primary source. Reset to None on full reset.
        self._loaded_dataset_dir: str | None = None

        # Selection state
        self._selected_wells: set[str] = set()

        # Channel config: {ch_name: {enabled, color: (r,g,b), vmin, vmax}}
        self._ch_config: dict = {}

        # Image display params
        self._contrast_method: str = "gamma"
        self._contrast_gamma: float = 1.0
        self._invert: bool = False
        self._overlay_table: str | None = None
        self._overlay_col: str | None = None
        self._overlay_cmap: str = "viridis"
        self._overlay_alpha: float = 0.0
        self._object_mask_selected: str = ""  # "" = None; else mask name without "mask_" prefix
        self._image_blocked: bool = False

        # Image debounce timer
        self._debounce = QTimer(singleShot=True, interval=300, timeout=self._refresh_images)

        # Performance: caches and state tracking
        self._raw_cache = _RowCache()  # row_idx → (img_data, mask_dict), capped by RAW_CACHE_MAX_BYTES
        self._mask_cache: dict[int, np.ndarray] = {}  # row_idx → downscaled mask
        self._polygon_cache: dict[int, list] = {}  # row_idx → extracted polygons
        self._gen: int = 0  # generation counter for cancelling stale workers
        self._last_state: dict = {}  # for change detection
        self._overlay_cache: tuple | None = None
        self._overlay_cache_key: str | None = None

        # Full-res zoom cache
        self._thread_pool = QThreadPool.globalInstance()
        self._pending_workers: int = 0
        # True while a channel-toggle batch is in flight — the batch-finish
        # (in-place pixmap update) must also run when every worker ERRORS,
        # otherwise _pending_workers reaches 0 with no update ever dispatched.
        self._channel_toggle_batch: bool = False
        self._channel_toggle_results: list = []
        self._shutting_down: bool = False
        # Retention registry for QRunnables: the workers run with
        # setAutoDelete(False), so a reference must be kept until the
        # finished/error signal fires (otherwise the C++ wrapper could be
        # garbage-collected mid-run) and deleteLater() must run afterwards
        # (otherwise every refresh leaks one C++ object per worker).
        self._active_workers: set = set()

        # Metadata
        self._metadata_df: pd.DataFrame | None = None
        self._metadata_merged: pd.DataFrame | None = None

        # Integrated Data-page table: any mix of profiler.db + infer.db
        # files of the current dataset, merged per object (io/merged_data).
        self._merged: MergedData | None = None

        self._build_ui()
        self._connect_signals()

        # Pre-fill directory + patterns from session.yml if provided via CLI.
        # Data is NOT auto-loaded — the user must click "Load Dataset" to
        # start the scan. This matches microProfiler's Browse → Load flow.
        if dataset_dir:
            p = Path(dataset_dir)
            self._dataset_dir = str(p)
            self._session = SessionFile(p)
            self._data_view.set_dataset_path(str(p))
            self._data_view.set_db_buttons_enabled(False)
            self._data_view.set_meta_browse_enabled(False)
            image_pat, mask_pat, subdir_pat = self._session.get_patterns()
            image_pat = image_pat or DEFAULT_IMAGE_PATTERN
            mask_pat = mask_pat or DEFAULT_MASK_PATTERN
            subdir_pat = subdir_pat or DEFAULT_IMAGE_SUBDIR_PATTERN
            self._data_view.set_patterns(
                image=image_pat,
                mask=mask_pat,
                subdir=subdir_pat,
            )

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_main_page(), stretch=1)

        # Status bar (hidden)
        self.statusBar().setVisible(False)
        self.statusBar().setMaximumHeight(0)

    def _build_main_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # ── Left nav sidebar ──
        nav = QWidget()
        nav.setObjectName("sidebar")
        nav.setFixedWidth(32)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        self._nav_data = RotatedLabel("Data")
        self._nav_data.setProperty("class", "nav-tab")
        self._nav_data.setProperty("active", "true")
        _data_font = self._nav_data.font()
        _data_font.setBold(True)
        self._nav_data.setFont(_data_font)

        self._nav_plate = RotatedLabel("Image")
        self._nav_plate.setProperty("class", "nav-tab")
        self._nav_plate.setProperty("active", "false")
        _plate_font = self._nav_plate.font()
        _plate_font.setBold(True)
        self._nav_plate.setFont(_plate_font)

        nav_layout.addWidget(self._nav_data)
        nav_layout.addWidget(self._nav_plate)
        nav_layout.addStretch()

        # ── Stacked content ──
        self._stack_content = QStackedWidget()

        # Page 0: Data View (owns THE integrated plot view)
        self._plot_view = DataPlotView()
        self._data_view = DataView()
        self._data_view.set_plot_view(self._plot_view)
        self._stack_content.addWidget(self._data_view)

        # Page 1: Plate & Images
        self._stack_content.addWidget(self._build_plate_images_tab())

        self._nav_data.clicked.connect(lambda: self._switch_tab(0))
        self._nav_plate.clicked.connect(lambda: self._switch_tab(1))

        body.addWidget(nav)
        body.addWidget(self._stack_content, stretch=1)

        outer.addLayout(body, stretch=1)

        # Pixel info bar at bottom
        self._pixel_info = PixelInfo()
        outer.addWidget(self._pixel_info)

        return page

    def _switch_tab(self, index: int) -> None:
        self._stack_content.setCurrentIndex(index)
        self._nav_data.setProperty("active", index == 0)
        self._nav_plate.setProperty("active", index == 1)
        for w in (self._nav_data, self._nav_plate):
            w.style().unpolish(w)
            w.style().polish(w)

    def _build_plate_images_tab(self) -> QWidget:
        """Image page: [well-grid | canvas] above [controls | display],
        with the label-annotation panel as a third (initially hidden)
        vertical section. Splitter geometry tokens live in ui_spec."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # ── Top splitter: Well Grid ──
        top_splitter = QSplitter(Qt.Horizontal)

        self._grid_controls = WellGridControls()
        self._grid_canvas = WellGridCanvas()

        top_splitter.addWidget(self._grid_controls)
        top_splitter.addWidget(self._grid_canvas)
        self._configure_splitter(top_splitter, H_SPLITTER_SIZES)
        self._well_grid_container = top_splitter

        # ── Middle splitter: Image View ──
        middle_splitter = QSplitter(Qt.Horizontal)

        self._image_controls = ImageControls()
        self._image_display = ImageDisplay()

        middle_splitter.addWidget(self._image_controls)
        middle_splitter.addWidget(self._image_display)
        self._configure_splitter(middle_splitter, H_SPLITTER_SIZES)

        # ── Label Annotation Panel ──
        self._label_panel = LabelAnnotationPanel()
        self._label_panel.setVisible(False)

        # ── Vertical splitter: grid + image + annotation ──
        self._v_splitter = QSplitter(Qt.Vertical)
        self._v_splitter.addWidget(top_splitter)
        self._v_splitter.addWidget(middle_splitter)
        self._v_splitter.addWidget(self._label_panel)
        self._v_splitter.setSizes(list(V_SPLITTER_SIZES))
        # The two image sections keep a visible minimum; only the label
        # panel can be dragged shut (it also hides itself when empty).
        self._v_splitter.setCollapsible(0, False)
        self._v_splitter.setCollapsible(1, False)

        layout.addWidget(self._v_splitter)
        return tab

    @staticmethod
    def _configure_splitter(splitter: QSplitter,
                            sizes: tuple[int, int]) -> None:
        """Initial sizes for a `controls | canvas` splitter; the controls
        pane cannot be collapsed away by dragging."""
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(list(sizes))
        splitter.setCollapsible(0, False)

    # ── Signal Connections ───────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # Well grid controls
        gw = self._grid_controls
        gw.plate_format.currentTextChanged.connect(self._on_grid_params_changed)
        gw.column.currentTextChanged.connect(self._on_grid_params_changed)
        gw.column.completer().activated.connect(self._on_grid_params_changed)
        gw.aggregation.currentTextChanged.connect(self._on_grid_params_changed)
        gw.colormap.currentTextChanged.connect(self._on_grid_params_changed)
        gw.palette.currentTextChanged.connect(self._on_grid_params_changed)
        gw.select_all_clicked.connect(self._on_select_all)
        gw.clear_clicked.connect(self._on_clear_selection)
        gw.image_block_toggled.connect(self._on_image_block_toggled)

        # Well grid canvas
        self._grid_canvas.well_clicked.connect(self._on_well_clicked)

        # Image controls (filter widget signals connected in _populate_image_controls)
        ic = self._image_controls
        ic.auto_all_clicked.connect(self._on_auto_all)
        ic.auto_range_changed.connect(self._on_auto_all)
        ic.reset_requested.connect(self._on_reset)
        ic.image_size_changed.connect(self._schedule_image_refresh)
        ic.channel_config_changed.connect(self._schedule_image_refresh)
        ic.contrast.currentTextChanged.connect(self._on_contrast_changed)
        ic.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        ic.overlay_col.currentTextChanged.connect(self._on_overlay_changed)
        ic.overlay_col.completer().activated.connect(self._on_overlay_changed)
        ic.overlay_cmap.currentTextChanged.connect(self._on_overlay_changed)
        ic.overlay_alpha.valueChanged.connect(self._on_overlay_changed)
        ic.object_mask_changed.connect(self._on_object_mask_changed)
        ic.sort_mode_changed.connect(self._schedule_image_refresh)

        # Image display (pixel click)
        self._image_display.pixel_clicked.connect(self._on_pixel_clicked)
        self._image_display.full_res_requested.connect(self._on_full_res_requested)

        # Label annotation controls
        ic.label_class_added.connect(self._on_label_class_added)
        ic.label_class_removed.connect(self._on_label_class_removed)
        ic.label_class_selection_changed.connect(self._on_label_class_selection_changed)
        ic.label_write_clicked.connect(self._on_label_write_to_db)

        # Object export
        ic.export_clicked.connect(self._on_export_clicked)

        # Label annotation panel crop requests
        self._label_panel.crop_requested.connect(self._on_crop_requested)

        # Data view
        self._data_view.dataset_browse_clicked.connect(self._on_dataset_browse)
        # Dropping a folder/file onto the path box or a selector button takes
        # the exact same code path as the Browse dialogs.
        enable_path_drop(self._data_view.dataset_path_edit,
                         on_path=self.select_dataset_dir)
        enable_path_drop(self._data_view.dataset_browse_button,
                         on_path=self.select_dataset_dir)
        enable_path_drop(self._data_view.select_db_button,
                         on_path=self.load_db_files, multi=True)
        enable_path_drop(self._data_view.metadata_browse_button,
                         on_path=self.load_metadata_file)
        self._data_view.dataset_path_edit.editingFinished.connect(
            self._on_dataset_path_edited)
        self._data_view.select_db_clicked.connect(self._on_select_db_browse)
        self._data_view.load_dataset_clicked.connect(self._on_load_dataset_clicked)
        self._plot_view.point_picked.connect(self._on_plot_point_picked)
        self._data_view.metadata_browse_clicked.connect(self._on_metadata_browse)
        self._data_view.metadata_merge_clicked.connect(self._on_metadata_merge)
        self._data_view.metadata_clear_clicked.connect(self._on_metadata_clear)
        self._data_view.write_to_db_clicked.connect(self._on_write_to_db)
        self._data_view.reset_clicked.connect(self._on_full_reset)

    # ── Dataset Loading ──────────────────────────────────────────────────────

    def _clear_display_state(self) -> None:
        """Clear image display, caches, grid, and label panel.

        Called when Browse selects a different directory, before the new
        directory's patterns are loaded. The UI shows an empty state until
        "Load Dataset" is clicked.
        """
        self._cancel_workers()
        self._image_display.clear()
        self._raw_cache.clear()
        self._mask_cache.clear()
        self._polygon_cache.clear()
        self._last_state.clear()
        self._overlay_cache = None
        self._overlay_cache_key = None
        # Invalidate in-flight exports too: their results must not re-enable
        # the export UI or report for a dataset no longer on screen.
        self._export_gen = getattr(self, "_export_gen", 0) + 1
        # A gen-bumped export can never reach its re-enable path (stale
        # terminal signals return early) — restore the export UI here.
        self._image_controls.set_export_enabled(True)
        # Dataset-scoped selections must not leak into the next dataset.
        self._object_mask_selected = ""
        self._label_panel.clear_all()
        self._image_controls.clear_classes()
        self._grid_canvas.clear()
        self._selected_wells = set()
        self._ch_config = {}
        self._image_controls.set_channels({})
        # Drop the previous dataset's merged metadata — otherwise the new
        # dataset's well-grid "Color by" and overlay dropdowns would list the
        # OLD dataset's metadata columns.
        self._metadata_df = None
        self._metadata_merged = None
        # Close old DataModule — it's no longer needed after browsing away.
        if self._dm is not None:
            self._dm.close_db()
            self._dm = None
        # The merged DB table is dataset-scoped — drop it with the dataset.
        self._reset_merged_data()
        self._loaded_dataset_dir = None
        self._update_window_title()

    def _update_window_title(self) -> None:
        if self._loaded_dataset_dir:
            self.setWindowTitle(f"microVis — {self._loaded_dataset_dir}")
        else:
            self.setWindowTitle("microVis")

    def _on_dataset_browse(self) -> None:
        """Select a dataset directory and read session.yml.

        Browse does NOT load data — it only selects the directory, reads
        session.yml to pre-fill GUI patterns, and clears the old display
        state when a different directory is selected. The user must click
        "Load Dataset" to actually scan data.
        """
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if not path:
            return
        self.select_dataset_dir(path)

    def select_dataset_dir(self, path: str) -> None:
        """Select (not load) a dataset directory — shared by the Browse
        button, typed paths, and directory drops onto the path box."""
        p = Path(path)
        # Re-selecting the same directory preserves the user's GUI edits —
        # session.yml is not re-read, display state is not cleared.
        if str(p) == self._dataset_dir:
            return

        # Clear old display state (images, grid, caches, etc.) when a
        # different directory is selected.
        self._clear_display_state()

        # Set the new directory + session file.
        self._dataset_dir = str(p)
        self._session = SessionFile(p)
        self._data_view.set_dataset_path(str(p))
        self._data_view.set_db_buttons_enabled(False)
        self._data_view.set_meta_browse_enabled(False)

        # Read session.yml to pre-fill GUI patterns (first browse only).
        image_pat, mask_pat, subdir_pat = self._session.get_patterns()
        image_pat = image_pat or DEFAULT_IMAGE_PATTERN
        mask_pat = mask_pat or DEFAULT_MASK_PATTERN
        subdir_pat = subdir_pat or DEFAULT_IMAGE_SUBDIR_PATTERN
        self._data_view.set_patterns(
            image=image_pat,
            mask=mask_pat,
            subdir=subdir_pat,
        )

    def _on_dataset_path_edited(self) -> None:
        """Typed dataset path: run the same selection flow as Browse/drop."""
        path = self._data_view.get_dataset_path()
        if path and str(Path(path)) != self._dataset_dir:
            self.select_dataset_dir(path)

    # ── DB selection → ONE integrated table ──────────────────────────────────

    @staticmethod
    def _split_paths(paths) -> list[str]:
        """Normalize a Browse list or a ';'-joined drop into a path list."""
        if isinstance(paths, str):
            paths = paths.split(";")
        out: list[str] = []
        for p in paths:
            p = str(p).strip()
            if p and p not in out:
                out.append(p)
        return out

    def _on_select_db_browse(self) -> None:
        """Open a file dialog for profiler.db and/or infer.db files."""
        from PySide6.QtWidgets import QFileDialog
        if self._dm is None:
            return
        start_dir = self._dataset_dir or ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select DB(s)", start_dir, "SQLite DB (*.db)")
        if paths:
            self.load_db_files(paths)

    def load_db_files(self, paths) -> None:
        """Merge any mix of profiler/infer DBs into ONE integrated table.

        Every call REPLACES the previous selection (re-select with more
        files to extend it). The last profiler-type DB is additionally
        loaded into the DataModule so the well-grid color-by keeps its
        existing table-level sources.
        """
        if self._dm is None:
            return
        paths = self._split_paths(paths)
        if not paths:
            return
        logger.info("Select DB: %d file(s) — %s",
                    len(paths), ", ".join(Path(x).name for x in paths))
        try:
            self._merged = MergedData.load(paths)
        except Exception as e:
            logger.warning("Failed to merge DBs %s: %s", paths, e)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid DB", str(e))
            return
        logger.info("Integrated table: %d rows x %d columns",
                    len(self._merged.table), len(self._merged.table.columns))

        self._refresh_merged_plot()
        self._data_view.set_meta_browse_enabled(True)

        # Well-grid / overlay machinery reads through the DataModule: keep
        # pointing it at the newest profiler-type DB (infer DBs carry no
        # well-keyed measurement tables).
        for path in reversed(paths):
            if not self._is_infer_db(path):
                self._load_profiler_into_dm(path)
                break
        self._data_view.set_write_to_db_enabled(True)

    @staticmethod
    def _is_infer_db(path: str) -> bool:
        """True when the SQLite file contains an `inference` table."""
        import sqlite3
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='inference'").fetchone()
                return row is not None
            finally:
                conn.close()
        except Exception:
            return False

    def _load_profiler_into_dm(self, path: str) -> None:
        """Point the DataModule at a profiler DB for grid/overlay columns."""
        if self._dm is None:
            return
        try:
            self._dm.load_db(path)
        except Exception as e:
            logger.warning("Failed to load profiler DB %s: %s", path, e)
            return
        # The overlay cache is keyed per DB — invalidate it so overlay data
        # is recomputed for the new database.
        self._overlay_cache = None
        self._overlay_cache_key = None
        # NOTE: filters and channels are dataset-dependent — never rebuild
        # them here (that would reset the user's field/stack selections).
        self._update_grid_columns()
        self._populate_overlay_columns()
        # Close persistent DB connection — cached data remains available.
        self._dm.close_db()
        logger.info("Active profiler DB: %s", path)

    def _reset_merged_data(self) -> None:
        """Drop the merged table and the plot view's data."""
        self._merged = None
        self._data_view.set_write_to_db_enabled(False)
        self._plot_view.clear()

    def _refresh_merged_plot(self) -> None:
        """Re-apply the Excel metadata merge and feed the plot view."""
        self._plot_view.set_frame(
            merge_metadata_into(self._merged, self._metadata_merged))

    def _merged_frame_for_dataset(self) -> pd.DataFrame:
        """Directory-scoped view of the merged table (empty df if none)."""
        if self._merged is None:
            return pd.DataFrame()
        scopes = (self._dm.directory_scopes() if self._dm
                  else [self._dataset_dir or ""])
        return self._merged.frame_for_dataset(scopes)

    def _load_dataset(
        self,
        path: str,
        image_pattern: str = "",
        mask_pattern: str = "",
        image_subdir_pattern: str = "",
    ) -> None:
        p = Path(path)
        if not p.is_dir():
            return

        # Phase 1: show modal progress dialog and construct DataModule off
        # the UI thread. DataModule.__init__ scans files + builds a row
        # index — synchronous and slow for large datasets. Moving it off the
        # UI thread keeps the dialog responsive (no "not responding" ghost).
        from PySide6.QtWidgets import QProgressDialog
        from microVis.worker import _DatasetLoadWorker

        logger.info("Loading dataset: %s", p)

        # Hold refs on self so they aren't GC'd mid-load
        self._loader_dialog = QProgressDialog("Loading dataset...", None, 0, 0, self)
        self._loader_dialog.setWindowModality(Qt.WindowModal)
        self._loader_dialog.setMinimumDuration(0)
        self._loader_dialog.setCancelButton(None)
        self._loader_dialog.setRange(0, 0)  # indeterminate
        self._loader_dialog.show()

        self._loader_worker = _DatasetLoadWorker(
            str(p), image_pattern, mask_pattern, image_subdir_pattern,
        )
        self._loader_thread = QThread()
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_worker.finished.connect(self._on_dataset_loaded)
        self._loader_worker.error.connect(self._on_dataset_load_error)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        # Stash pending patterns so phase 2 can persist them after load
        self._loader_pending = (str(p), image_pattern, mask_pattern, image_subdir_pattern)
        self._loader_thread.start()

    def _on_dataset_loaded(self, dm: Any) -> None:
        """Phase 2: populate UI after background DataModule construction."""
        # Capture and clear loader state first so a failed phase 2 doesn't
        # leave dangling refs.
        p_str, image_pattern, mask_pattern, image_subdir_pattern = self._loader_pending
        dialog = getattr(self, "_loader_dialog", None)
        thread = getattr(self, "_loader_thread", None)
        worker = getattr(self, "_loader_worker", None)
        self._loader_pending = None
        self._loader_dialog = None
        self._loader_worker = None
        self._loader_thread = None

        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        if thread is not None:
            thread.quit()
            if worker is not None:
                # deleteLater must be delivered after the loader thread's
                # event loop stops — post it from thread.finished.
                thread.finished.connect(worker.deleteLater)

        p = Path(p_str)
        try:
            self._dm = dm
            self._dataset_dir = str(p)
            self._session = SessionFile(p)
            self._data_view.set_dataset_path(str(p))
            self._data_view.set_db_buttons_enabled(True)
            self._data_view.set_meta_browse_enabled(False)

            # Reset UI state for new dataset
            self._image_display.clear()
            self._raw_cache.clear()
            self._mask_cache.clear()
            self._polygon_cache.clear()
            self._last_state.clear()
            self._overlay_cache = None
            self._overlay_cache_key = None
            self._label_panel.clear_all()
            self._grid_canvas.update_grid(
                self._dm, table_name="", col_val=(None, False), agg="mean",
                cmap="viridis", palette="Set1", fmt_name=DEFAULT_PLATE,
                selected_wells=set(), force_redraw=True,
            )

            # Init selection (no wells selected by default)
            self._selected_wells = set()

            # Init channel config.
            # On first load of this directory, read channel_colors from
            # session.yml. On subsequent loads (reload), preserve the user's
            # current GUI colors — don't re-read session.yml.
            if str(p) == self._loaded_dataset_dir:
                self._init_channel_config(use_saved=False, existing_config=dict(self._ch_config))
            else:
                self._init_channel_config(use_saved=True)

            # Populate controls
            self._populate_grid_controls()
            # Hide well grid when dataset has no well metadata
            has_wells = len(self._dm.get_wells()) > 0
            self._well_grid_container.setVisible(has_wells)
            self._populate_image_controls()
            self._populate_label_controls()

            # Close DB connection — cached data remains available
            self._dm.close_db()

            # Patterns + channel colors were already persisted to session.yml
            # by _on_load_dataset_clicked before starting the scan. This
            # phase 2 callback only populates the UI.
            self._loaded_dataset_dir = str(p)
            self._update_window_title()
            # This dataset's own log file (full DEBUG detail).
            set_log_file(Path(p_str) / "microVis.log")
            logger.info(
                "Dataset ready: %d sites, channels %s, masks %s",
                len(self._dm.dataset.metadata), list(self._dm.channels),
                list(self._dm.mask_names))

            # Initial render
            self._update_grid()
            self._schedule_image_refresh()
        except Exception:
            logger.exception("Failed to initialize UI for dataset %s", p)
            import traceback
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Dataset Load Incomplete",
                f"The dataset was scanned but the UI could not be fully "
                f"initialized:\n{traceback.format_exc(limit=5)}",
            )

    def _on_dataset_load_error(self, msg: str) -> None:
        """Phase 2 error: clean up loader state and notify the user."""
        p_str = self._loader_pending[0] if self._loader_pending else ""
        dialog = getattr(self, "_loader_dialog", None)
        thread = getattr(self, "_loader_thread", None)
        worker = getattr(self, "_loader_worker", None)
        self._loader_pending = None
        self._loader_dialog = None
        self._loader_worker = None
        self._loader_thread = None

        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        if thread is not None:
            thread.quit()
            if worker is not None:
                thread.finished.connect(worker.deleteLater)

        logger.warning("Failed to load dataset from %s — %s", p_str, msg)
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Dataset Load Failed", f"Could not load dataset:\n{msg}")
        self._dataset_dir = p_str
        self._session = SessionFile(p_str)
        self._data_view.set_dataset_path(p_str)
        self._data_view.set_db_buttons_enabled(False)
        self._data_view.set_meta_browse_enabled(False)
        self._switch_tab(0)

    def _on_load_dataset_clicked(self) -> None:
        """Load the currently selected dataset directory.

        Load Dataset is an action button: it writes the user's current GUI
        patterns to session.yml (so the next session restores them), then
        starts the data scan. Channel colors are also persisted.
        """
        # Pick up a path typed since the last selection (editingFinished may
        # not have fired yet when the button is clicked directly).
        self._on_dataset_path_edited()
        if not self._dataset_dir:
            return
        # Session persistence and the dataset load fail independently: a
        # session.yml write error (DataError — a regular Exception) must be
        # REPORTED, but must neither abort the load nor kill the app.
        try:
            image_pat, mask_pat, subdir_pat = self._data_view.get_patterns()
            # Write patterns to the SELECTED directory's session.yml, not
            # the previously loaded one. self._session was set in Browse
            # to SessionFile(self._dataset_dir).
            if self._session is not None:
                self._session.set_patterns(
                    image_pattern=image_pat,
                    mask_pattern=mask_pat,
                    image_subdir_pattern=subdir_pat,
                )
                self._persist_channel_colors()
        except Exception:
            logger.exception("Failed to persist session.yml")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Session Save Failed",
                "Could not write session.yml (disk full or permission denied). "
                "Continuing with the dataset load.",
            )
        try:
            self._load_dataset(
                self._dataset_dir,
                image_pattern=image_pat,
                mask_pattern=mask_pat,
                image_subdir_pattern=subdir_pat,
            )
        except Exception:
            logger.exception("Failed to load dataset")

    # ── Channel Config ───────────────────────────────────────────────────────

    def _init_channel_config(
        self,
        use_saved: bool = True,
        existing_config: dict | None = None,
    ) -> None:
        """Build per-channel config.

        On first load of a directory, ``use_saved=True`` reads channel_colors
        from session.yml. On subsequent loads (reload), pass
        ``existing_config=dict(self._ch_config)`` to preserve the user's
        current GUI colors for channels that still exist; defaults are used
        for new channels and removed channels are dropped.
        """
        self._ch_config.clear()
        channels = self._dm.channels
        if existing_config is not None:
            saved_colors = existing_config
        elif use_saved and self._session is not None:
            saved_colors = self._session.get_channel_colors()
        else:
            saved_colors = {}
        for i, ch in enumerate(channels):
            default_color = DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)]
            saved = saved_colors.get(ch, {}) if isinstance(saved_colors, dict) else {}
            self._ch_config[ch] = {
                "enabled": True,
                "color": saved.get("color", default_color),
                "vmin": saved.get("vmin", 0),
                "vmax": saved.get("vmax", DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)),
            }

    def _channel_max_value(self) -> float:
        """Dataset dtype max for the channel vmin/vmax spin-box range."""
        if self._dm is not None and self._dm.img_dtype is not None:
            return DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)
        return 65535.0

    def _persist_channel_colors(self) -> None:
        """Write current channel color/vmin/vmax to <dataset>/session.yml."""
        if self._session is None:
            return
        colors = {
            ch: {
                "color": cfg.get("color"),
                "vmin": cfg.get("vmin", 0),
                "vmax": cfg.get("vmax", 65535),
            }
            for ch, cfg in self._ch_config.items()
        }
        try:
            self._session.set_channel_colors(colors)
        except Exception:
            logger.warning("Failed to persist channel colors", exc_info=True)

    # ── Populate Controls ────────────────────────────────────────────────────

    def _populate_grid_controls(self) -> None:
        gw = self._grid_controls

        # Plate formats
        gw.plate_format.blockSignals(True)
        gw.plate_format.clear()
        gw.plate_format.addItems(list(PLATE_FORMATS.keys()))
        idx = gw.plate_format.findText(DEFAULT_PLATE)
        if idx >= 0:
            gw.plate_format.setCurrentIndex(idx)
        gw.plate_format.blockSignals(False)

        self._update_grid_columns()

        gw.aggregation.blockSignals(True)
        gw.aggregation.clear()
        gw.aggregation.addItems(AGG_METHODS)
        gw.aggregation.blockSignals(False)

        gw.colormap.blockSignals(True)
        gw.colormap.clear()
        gw.colormap.addItems(CMAP_OPTIONS)
        gw.colormap.setCurrentText(DEFAULT_CMAP)
        gw.colormap.blockSignals(False)

        gw.palette.blockSignals(True)
        gw.palette.clear()
        gw.palette.addItems(QUALITATIVE_PALETTES)
        gw.palette.setCurrentText("Set1")
        gw.palette.blockSignals(False)

    def _merged_color_entries(self) -> list[tuple[str, str, str, bool]]:
        """Color-by entries for every column of the merged DB table.

        Returns ``(label, data_key, column, is_numeric)`` with data_key
        ``merge`` — values resolve through :meth:`_merged_frame_for_dataset`
        (aggregated per well / per object like the infer sources were).
        """
        if self._merged is None:
            return []
        return [(f"merge/{name}", "merge", name, is_num)
                for name, is_num in self._merged.display_columns()]

    def _update_grid_columns(self) -> None:
        if self._dm is None:
            return
        gw = self._grid_controls
        gw.column.blockSignals(True)
        gw.column.clear()
        gw.column.addItem("None")
        tables = self._dm.get_profiling_tables()
        for tname in tables:
            cols = self._dm.get_profiling_columns(tname)
            for name, _ctype, is_num in cols:
                gw.column.addItem(f"{tname}/{name}", (tname, name, is_num))
        # Add merged metadata columns (numeric dtypes render with a colormap,
        # categoricals with the qualitative palette)
        if self._metadata_merged is not None:
            for col in self._metadata_merged.columns:
                if col != "well":
                    is_num = pd.api.types.is_numeric_dtype(self._metadata_merged[col])
                    gw.column.addItem(f"metadata/{col}", ("metadata", col, is_num))
        # Add merged DB columns (profiler + infer, cross-joined per object)
        for label, data_key, name, is_num in self._merged_color_entries():
            gw.column.addItem(label, (data_key, name, is_num))
        gw.column.blockSignals(False)

    def _populate_image_controls(self) -> None:
        dm = self._dm
        ic = self._image_controls

        # Top-100 natsorted for structural filters
        fields = dm.get_filter_values("field")
        stacks = dm.get_filter_values("stack")
        timepoints = dm.get_filter_values("timepoint")

        # Extra-col filters: top-100 natsorted for each extra col
        extra_cols: dict[str, list[str]] = {}
        if dm.dataset.schema.extra_cols:
            for col in dm.dataset.schema.extra_cols:
                extra_cols[col] = dm.get_filter_values(col)

        ic.set_filter_options(fields, stacks, timepoints, extra_cols)

        # Connect filter signals only for widgets that exist (missing
        # structural columns yield None widgets). UniqueConnection: the
        # widgets survive reloads of the same dataset (set_filter_options
        # no-ops on identical options), so plain connect would stack
        # duplicate handlers with every "Load Dataset".
        if ic.fields_widget:
            ic.fields_widget.selection_changed.connect(
                self._on_image_filter_changed, Qt.UniqueConnection)
        if ic.stacks_widget:
            ic.stacks_widget.selection_changed.connect(
                self._on_image_filter_changed, Qt.UniqueConnection)
        if ic.timepoints_widget:
            ic.timepoints_widget.selection_changed.connect(
                self._on_image_filter_changed, Qt.UniqueConnection)
        for widget in ic.get_extra_widgets().values():
            widget.selection_changed.connect(
                self._on_image_filter_changed, Qt.UniqueConnection)

        # Channel controls
        ic.set_channels(self._ch_config, max_value=self._channel_max_value())

        # Overlay column (DB-dependent) + cmap
        self._populate_overlay_columns()

    def _populate_overlay_columns(self) -> None:
        """Rebuild the overlay Color-by dropdown + colormap from the DB.

        DB-dependent only — never touches dataset filters or channels, so a
        DB switch cannot reset the user's field/stack/timepoint selections.
        Also re-adds metadata/* items (they depend on the loaded metadata).
        Resets the overlay state via _on_overlay_changed: the old column may
        not exist in the (new) DB.
        """
        ic = self._image_controls
        ic.overlay_col.blockSignals(True)
        ic.overlay_col.clear()
        ic.overlay_col.addItem("None")
        tables = self._dm.get_profiling_tables()
        for tname in tables:
            cols = self._dm.get_profiling_columns(tname)
            for name, _ctype, _is_num in cols:
                ic.overlay_col.addItem(f"{tname}/{name}", (tname, name))
        if self._metadata_merged is not None:
            for col in self._metadata_merged.columns:
                if col != "well":
                    ic.overlay_col.addItem(f"metadata/{col}", ("metadata", col))
        # Add merged DB columns (per-object overlay values)
        for label, data_key, name, _is_num in self._merged_color_entries():
            ic.overlay_col.addItem(label, (data_key, name))
        ic.overlay_col.blockSignals(False)

        ic.overlay_cmap.blockSignals(True)
        ic.overlay_cmap.clear()
        ic.overlay_cmap.addItems(CMAP_OPTIONS)
        ic.overlay_cmap.setCurrentText(DEFAULT_CMAP)
        ic.overlay_cmap.blockSignals(False)

        self._on_overlay_changed()

    def _populate_label_controls(self) -> None:
        """Populate mask dropdowns in label annotation + overlay controls."""
        if self._dm is None:
            return
        # Object Export panel — its own independent Object dropdown.
        self._image_controls.set_label_masks(self._dm.mask_names)
        # Object Overlay panel — Select object dropdown (display/hover/drag/crop).
        self._image_controls.set_object_masks(self._dm.mask_names)
        self._label_panel.clear_all()
        # Initialize export "Annotated" option as disabled
        self._image_controls.update_export_annotated_option(False)

    # ── Metadata ─────────────────────────────────────────────────────────────

    def _on_metadata_browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Metadata File", "", "Excel Files (*.xlsx *.xls)"
        )
        if not path:
            return
        self.load_metadata_file(path)

    def load_metadata_file(self, path: str) -> None:
        """Load a plate-metadata Excel file — shared by the Browse button
        and file drops onto the Select Metadata button."""
        try:
            from microVis.io.data_module import parse_plate_metadata
            self._metadata_df = parse_plate_metadata(path)
            self._data_view.set_metadata_label(Path(path).name)

        except Exception:
            logger.exception("Failed to load metadata from %s", path)
            self._metadata_df = None
            self._data_view.set_metadata_label(None)

    def _on_metadata_merge(self) -> None:
        if self._metadata_df is None:
            return
        self._metadata_merged = self._metadata_df.copy()
        self._overlay_cache = None
        logger.info("Merging Excel metadata (%d rows) into the integrated "
                    "table", len(self._metadata_merged))

        # Merge into the integrated table (left join by well) and refresh.
        self._refresh_merged_plot()
        self._update_overlay_with_metadata()

    def _on_metadata_clear(self) -> None:
        self._metadata_df = None
        self._metadata_merged = None
        self._overlay_cache = None
        self._data_view.set_metadata_label(None)
        logger.info("Cleared merged Excel metadata")

        self._refresh_merged_plot()
        self._update_overlay_with_metadata()

    def _update_overlay_with_metadata(self) -> None:
        ic = self._image_controls
        selected_data = ic.overlay_col.currentData()
        ic.overlay_col.blockSignals(True)
        # Remove existing metadata items (tagged with "metadata/" prefix)
        for i in range(ic.overlay_col.count() - 1, -1, -1):
            data = ic.overlay_col.itemData(i)
            if data and isinstance(data, tuple) and data[0] == "metadata":
                ic.overlay_col.removeItem(i)
        # The selected column may have been one of the removed metadata items
        # (Clear) — reset the overlay state so stale values stop rendering.
        if (selected_data is not None and isinstance(selected_data, tuple)
                and selected_data[0] == "metadata"
                and self._metadata_merged is None):
            self._overlay_table = None
            self._overlay_col = None
        # Add merged metadata columns
        if self._metadata_merged is not None:
            for col in self._metadata_merged.columns:
                if col != "well":
                    ic.overlay_col.addItem(f"metadata/{col}", ("metadata", col))
        ic.overlay_col.blockSignals(False)
        # Signals were blocked while rebuilding — redraw with the new state.
        self._on_overlay_changed()
        # Also update the well grid Color by dropdown
        self._update_grid_columns()

    def _on_write_to_db(self) -> None:
        """Write the integrated table into a NEW database.

        The table fuses profiler measurements, infer predictions/coordinates
        and (if merged) the Excel metadata columns, one row per object.
        The output file lives next to the dataset; its name comes from the
        small edit next to the button (default merge.db). Source DBs are
        never modified.
        """
        if self._merged is None:
            logger.info("Write to DB: no DBs are merged")
            return
        df = merge_metadata_into(self._merged, self._metadata_merged)
        out = str(Path(self._dataset_dir) / self._data_view.get_merge_db_name())
        logger.info("Write to DB: %d rows x %d columns -> %s",
                    len(df), len(df.columns), out)
        try:
            written_path = write_merged_db(df, out)
        except Exception:
            logger.exception("Failed to write merged DB")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Write Failed",
                f"Could not write {out}. See log for details.")
            return
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "Merged DB Written",
            f"Wrote {len(df):,} rows ({len(df.columns)} columns) to "
            f"{written_path} (table '{merged_data.MERGED_TABLE}').")
        logger.info("Merged DB written: %s (%d rows)", written_path, len(df))
        # Write-to-DB is an action button — persist patterns + channel
        # colors to session.yml (write-on-action contract).
        if self._session is not None:
            image_pat, mask_pat, subdir_pat = self._data_view.get_patterns()
            self._session.set_patterns(
                image_pattern=image_pat,
                mask_pattern=mask_pat,
                image_subdir_pattern=subdir_pat,
            )
            self._persist_channel_colors()

    # ── Grid Handlers ────────────────────────────────────────────────────────

    def _on_grid_params_changed(self) -> None:
        self._update_grid()

    def _on_well_clicked(self, well: str) -> None:
        if self._dm is None:
            return
        is_real = well in set(self._dm.get_wells())
        if well in self._selected_wells:
            self._selected_wells.discard(well)
        else:
            self._selected_wells.add(well)
        self._update_grid()
        if is_real:
            self._schedule_image_refresh()

    def _on_select_all(self) -> None:
        if self._dm is None:
            return
        self._selected_wells = set(self._dm.get_wells())
        self._update_grid()
        self._schedule_image_refresh()

    def _on_clear_selection(self) -> None:
        self._selected_wells.clear()
        self._update_grid()
        self._schedule_image_refresh()

    def _on_image_block_toggled(self, blocked: bool) -> None:
        self._image_blocked = blocked
        if not blocked:
            self._schedule_image_refresh()

    def _update_grid(self) -> None:
        if self._dm is None:
            return
        gw = self._grid_controls
        col_data = gw.column.currentData()
        if col_data and len(col_data) == 3:
            table_name, col_name, is_num = col_data
            col_val = (col_name, is_num)
        else:
            table_name = ""
            col_name = None
            col_val = (None, False)

        # Pre-computed well→value map for sources the canvas cannot aggregate
        # itself: merged Excel metadata and loaded infer DBs.
        value_map: dict[str, float | str] | None = None
        if table_name == "metadata" and col_name is not None \
                and self._metadata_merged is not None \
                and col_name in self._metadata_merged.columns:
            value_map = dict(
                zip(self._metadata_merged["well"], self._metadata_merged[col_name], strict=False)
            )
        elif table_name == "merge" and col_name is not None:
            try:
                value_map = aggregate_by_well(
                    self._merged_frame_for_dataset(), col_name,
                    gw.aggregation.currentText())
            except Exception:
                logger.warning("Failed to aggregate merged grid values", exc_info=True)
                value_map = {}

        self._grid_canvas.update_grid(
            self._dm,
            table_name=table_name,
            col_val=col_val,
            agg=gw.aggregation.currentText(),
            cmap=gw.colormap.currentText(),
            palette=gw.palette.currentText(),
            fmt_name=gw.plate_format.currentText(),
            selected_wells=self._selected_wells,
            value_map=value_map,
        )

    # ── Image Handlers ───────────────────────────────────────────────────────

    def _on_image_filter_changed(self) -> None:
        self._schedule_image_refresh()

    def _on_auto_all(self) -> None:
        if self._dm is None:
            return
        from concurrent.futures import ThreadPoolExecutor

        from microVis.worker import _downscale_image

        ic = self._image_controls
        low_pct = ic.auto_low.value()
        high_pct = ic.auto_high.value()

        # Re-sync channel config from the widgets first — a channel checkbox
        # toggled within the 300 ms refresh debounce would otherwise leave
        # _ch_config stale and Auto would compute ranges for the wrong set.
        self._ch_config = ic.get_channel_config()

        dm = self._dm
        channels = dm.channels
        enabled_chs = [ch for ch, cfg in self._ch_config.items() if cfg.get("enabled", True)]
        if not enabled_chs:
            return

        wells = sorted(self._selected_wells) if self._selected_wells else dm.get_wells()
        fields = list(ic.get_selected_fields()) or dm.get_fields()
        stacks = list(ic.get_selected_stacks()) or dm.get_stacks()
        timepoints = list(ic.get_selected_timepoints()) or dm.get_timepoints()
        rows = dm.lookup_row_indices(wells, fields, stacks, timepoints)

        # Post-hoc extra-col filter: empty list for a col means "no filter"
        # on that col (filter_rows_by_extra skips empty lists).
        extra_filters = {
            col: widget.get_selected()
            for col, widget in ic.get_extra_widgets().items()
            if widget.get_selected()
        }
        if extra_filters and rows:
            row_indices = [r[0] for r in rows]
            filtered = set(dm.filter_rows_by_extra(row_indices, extra_filters))
            rows = [r for r in rows if r[0] in filtered]

        if not rows:
            return

        max_sample = 16
        if len(rows) > max_sample:
            step = len(rows) // max_sample
            rows = rows[::step][:max_sample]

        thumb_size = 128
        ch_indices = {ch: channels.index(ch) for ch in enabled_chs if ch in channels}
        ch_samples: dict[str, list] = {ch: [] for ch in ch_indices}

        try:
            # Parallel load + downscale for uncached images
            def _load_and_downscale(row_idx):
                raw_data = self._raw_cache.get(row_idx)
                if raw_data is None:
                    raw_data = dm.get_imageset(row_idx)
                    self._raw_cache[row_idx] = raw_data
                img_data, _ = raw_data
                return _downscale_image(img_data, thumb_size)

            with ThreadPoolExecutor(max_workers=4) as pool:
                thumbnails = list(pool.map(_load_and_downscale, [r[0] for r in rows]))

            for img_small in thumbnails:
                for ch, idx in ch_indices.items():
                    if idx < img_small.shape[2]:
                        ch_samples[ch].append(img_small[:, :, idx].ravel())

            for ch, samples in ch_samples.items():
                if samples:
                    all_pixels = np.concatenate(samples)
                    self._ch_config[ch]["vmin"] = float(np.percentile(all_pixels, low_pct))
                    self._ch_config[ch]["vmax"] = float(np.percentile(all_pixels, high_pct))
        except Exception:
            logger.warning("Auto-range percentile computation failed", exc_info=True)

        self._image_controls.update_channel_values(self._ch_config)
        self._persist_channel_colors()
        self._schedule_image_refresh()

    def _on_reset(self) -> None:
        if self._dm is None:
            return
        ic = self._image_controls
        # Reset Low/High to defaults
        ic.auto_low.blockSignals(True)
        ic.auto_high.blockSignals(True)
        ic.auto_low.setValue(0.1)
        ic.auto_high.setValue(99.9)
        ic.auto_low.blockSignals(False)
        ic.auto_high.blockSignals(False)
        # Re-init channel config to defaults
        self._init_channel_config(use_saved=False)
        ic.set_channels(self._ch_config, max_value=self._channel_max_value())
        self._persist_channel_colors()
        # Reset image zoom
        self._image_display.reset_all_zoom()
        self._schedule_image_refresh()

    def _on_contrast_changed(self, method: str) -> None:
        self._contrast_method = method
        self._invert = (method == "invert")
        self._image_controls.set_gamma_visible(method == "gamma")
        self._schedule_image_refresh()

    def _on_gamma_changed(self, value: int) -> None:
        self._contrast_gamma = value / 100.0  # slider 10–300 → 0.1–3.0
        self._schedule_image_refresh()

    def _on_overlay_changed(self) -> None:
        ic = self._image_controls
        col_data = ic.overlay_col.currentData()
        if col_data and len(col_data) == 2:
            self._overlay_table, self._overlay_col = col_data
        else:
            self._overlay_table, self._overlay_col = None, None
        self._overlay_cmap = ic.overlay_cmap.currentText()
        self._overlay_alpha = ic.overlay_alpha.value() / 100.0
        self._schedule_image_refresh()

    def _on_object_mask_changed(self, mask_name: str) -> None:
        """Handle 'Select object' dropdown change in Object Overlay panel.

        ``mask_name`` is '' when 'None' is selected (no overlay / no
        interaction). Otherwise it is a mask name without the ``mask_``
        prefix. The per-row mask and polygon caches are invalidated because
        they are keyed by row_idx and become stale when the selected mask
        changes.
        """
        self._object_mask_selected = mask_name
        self._mask_cache.clear()
        self._polygon_cache.clear()
        # In-flight image workers were dispatched under the OLD mask; their
        # results would re-populate the just-cleared caches (same worker
        # generation) and the overlay fast path would render stale polygons.
        self._cancel_workers()
        self._schedule_image_refresh()

    def _schedule_image_refresh(self) -> None:
        self._debounce.start()

    def _start_worker(self, worker) -> None:
        """Track and start a QRunnable; release it when it finishes or errors."""
        self._active_workers.add(worker)

        def _release(*_args, w=worker):
            self._active_workers.discard(w)
            # QRunnables are not QObjects and have no deleteLater — dropping
            # the registry reference lets Python GC free the wrapper (the
            # pool only touches the runnable during run(), and it is never
            # auto-deleted: setAutoDelete(False)).

        worker.signals.finished.connect(_release)
        worker.signals.error.connect(_release)
        self._thread_pool.start(worker)

    def _cancel_workers(self) -> None:
        """Invalidate all pending background workers."""
        self._gen += 1
        self._pending_workers = 0
        self._channel_toggle_batch = False

    def _build_state(self) -> dict:
        """Snapshot the current filter/contrast/overlay state for change detection."""
        ic = self._image_controls
        # Normalize ch_config so colors are always tuples (avoids list != tuple)
        normalized_ch = {
            ch: {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}
            for ch, cfg in self._ch_config.items()
        }
        return {
            "wells": frozenset(self._selected_wells),
            "fields": tuple(ic.get_selected_fields()),
            "stacks": tuple(ic.get_selected_stacks()),
            "timepoints": tuple(ic.get_selected_timepoints()),
            "extra_filters": tuple(sorted(
                (col, tuple(widget.get_selected()))
                for col, widget in ic.get_extra_widgets().items()
            )),
            "ch_config": normalized_ch,
            "contrast": self._contrast_method,
            "gamma": self._contrast_gamma,
            "invert": self._invert,
            "overlay_col": self._overlay_col,
            "overlay_table": self._overlay_table,
            "overlay_cmap": self._overlay_cmap,
            "overlay_alpha": self._overlay_alpha,
            "object_mask": self._object_mask_selected,
            "sort_by_row": ic.sort_by_row.isChecked(),
            "thumb_size": int(ic.image_size.value()),
        }

    def _detect_change(self) -> str:
        """Compare current state to last state. Returns change category."""
        new_state = self._build_state()
        old = self._last_state
        self._last_state = new_state

        if not old:
            return "filters"

        if (old.get("wells") != new_state["wells"]
                or old.get("fields") != new_state["fields"]
                or old.get("stacks") != new_state["stacks"]
                or old.get("timepoints") != new_state["timepoints"]
                or old.get("extra_filters") != new_state["extra_filters"]):
            return "filters"

        contrast_changed = (
            old.get("contrast") != new_state["contrast"]
            or old.get("gamma") != new_state["gamma"]
            or old.get("invert") != new_state["invert"]
        )
        ch_config_changed = old.get("ch_config") != new_state["ch_config"]

        if ch_config_changed or contrast_changed:
            # Distinguish channel-toggle (only enabled flags changed) from contrast change
            if not contrast_changed and ch_config_changed:
                old_cfg = old.get("ch_config", {})
                new_cfg = new_state["ch_config"]
                only_enabled_changed = all(
                    old_cfg.get(ch, {}).get("vmin") == new_cfg.get(ch, {}).get("vmin")
                    and old_cfg.get(ch, {}).get("vmax") == new_cfg.get(ch, {}).get("vmax")
                    for ch in new_cfg
                    if ch in old_cfg
                )
                if only_enabled_changed:
                    return "channel_toggle"
            return "contrast"

        if (old.get("overlay_col") != new_state["overlay_col"]
                or old.get("overlay_table") != new_state["overlay_table"]
                or old.get("object_mask") != new_state["object_mask"]):
            return "overlay"

        if (old.get("overlay_cmap") != new_state["overlay_cmap"]
                or old.get("overlay_alpha") != new_state["overlay_alpha"]):
            return "overlay_styling"

        if old.get("sort_by_row") != new_state["sort_by_row"]:
            return "sort"

        if old.get("thumb_size") != new_state["thumb_size"]:
            return "image_size"

        return "none"

    def _refresh_images(self) -> None:
        if self._dm is None:
            return
        if self._image_blocked:
            return

        # Update channel config BEFORE change detection so toggles are detected
        self._ch_config = self._image_controls.get_channel_config()

        change = self._detect_change()
        if change == "none":
            return

        ic = self._image_controls
        fields = ic.get_selected_fields()
        stacks = ic.get_selected_stacks()
        timepoints = ic.get_selected_timepoints()

        # "User unchecked everything" → show nothing (only if widget exists).
        # Absent widget (column missing) → [] is passed to lookup as "no filter".
        # Every early-exit must invalidate in-flight workers (_cancel_workers
        # bumps _gen), or stale results from the previous dispatch would
        # repopulate the cleared display. The state snapshot must ALSO be
        # updated here — otherwise re-selecting the exact previous selection
        # looks like "no change" (_detect_change returns "none") and the
        # display stays empty forever.
        if ic.fields_widget is not None and not fields:
            self._cancel_workers()
            self._image_display.clear()
            self._raw_cache.clear()
            self._mask_cache.clear()
            self._polygon_cache.clear()
            self._last_state = self._build_state()
            return
        if ic.stacks_widget is not None and not stacks:
            self._cancel_workers()
            self._image_display.clear()
            self._raw_cache.clear()
            self._mask_cache.clear()
            self._polygon_cache.clear()
            self._last_state = self._build_state()
            return
        if ic.timepoints_widget is not None and not timepoints:
            self._cancel_workers()
            self._image_display.clear()
            self._raw_cache.clear()
            self._mask_cache.clear()
            self._polygon_cache.clear()
            self._last_state = self._build_state()
            return
        # Extra-col filters: widget exists but nothing selected → show nothing
        # (consistent with fields/stacks/timepoints behaviour).
        for widget in ic.get_extra_widgets().values():
            if not widget.get_selected():
                self._cancel_workers()
                self._image_display.clear()
                self._raw_cache.clear()
                self._mask_cache.clear()
                self._polygon_cache.clear()
                self._last_state = self._build_state()
                return

        # Wells exist but none selected via grid → show nothing.
        # No well column → get_wells() returns [] → no filter (all selected).
        if not self._selected_wells and self._dm.get_wells():
            self._cancel_workers()
            self._image_display.clear()
            self._raw_cache.clear()
            self._mask_cache.clear()
            self._polygon_cache.clear()
            self._last_state = self._build_state()
            return

        thumb_size = int(ic.image_size.value())
        sort_by_row = ic.sort_by_row.isChecked()
        saved_state = self._image_display.save_view_state()

        if change == "sort":
            # Re-sort the already-arrived results. In-flight workers are NOT
            # cancelled: their late results insert at the correct sorted
            # position with the current overlay (each result is added once, so
            # no duplicates) — cancelling would drop thumbnails that never
            # arrive. When nothing is cached yet, fall through to a normal
            # dispatch.
            if not self._image_display.resort_cached(
                thumb_size, self._overlay_alpha, self._overlay_cmap,
                saved_state, sort_by_row,
            ):
                self._dispatch_image_workers(thumb_size, saved_state=saved_state)
            return

        if change == "overlay_styling" and self._polygon_cache:
            # Re-render cached polygons with new styling — no workers needed.
            # In-flight workers are NOT cancelled: late results render with
            # the current overlay values at add time.
            self._image_display.restyle_overlay(
                self._overlay_alpha, self._overlay_cmap, self._polygon_cache)
            return

        if change == "channel_toggle" and self._raw_cache:
            # Re-enhance from cached raw at thumbnail res — near-instant
            self._dispatch_image_workers(thumb_size, saved_state=saved_state,
                                         channel_toggle=True)
            return

        # For overlay/contrast/resize changes with cache available, skip disk I/O
        use_cache = (
            change in ("overlay", "contrast", "image_size")
            and bool(self._raw_cache)
        )
        if change == "image_size":
            self._mask_cache.clear()
            self._polygon_cache.clear()
        if not use_cache:
            self._raw_cache.clear()
            self._mask_cache.clear()
            self._polygon_cache.clear()

        self._dispatch_image_workers(thumb_size, saved_state=saved_state)

    def _dispatch_image_workers(self, thumb_size: int, saved_state: dict | None = None,
                                channel_toggle: bool = False) -> None:
        """Load images and dispatch background workers for processing."""
        self._cancel_workers()
        # Discard any partially accumulated toggle results — a non-toggle
        # dispatch supersedes them (they hold full RGB arrays per row).
        self._channel_toggle_results = []
        self._ch_config = self._image_controls.get_channel_config()
        self._saved_state = saved_state

        ic = self._image_controls
        wells = sorted(self._selected_wells)

        # Empty get_selected_*() → [] → lookup_row_indices treats as "no
        # filter" (covers absent widgets AND user clicking "Clear"). User
        # checks all N items → filter to exactly those N (literal).
        fields = ic.get_selected_fields()
        stacks = ic.get_selected_stacks()
        timepoints = ic.get_selected_timepoints()

        rows_info = self._dm.lookup_row_indices(wells, fields, stacks, timepoints)

        # Post-hoc extra-col filter: empty list for a col means "no filter"
        # on that col (filter_rows_by_extra skips empty lists).
        extra_filters = {
            col: widget.get_selected()
            for col, widget in ic.get_extra_widgets().items()
            if widget.get_selected()
        }
        if extra_filters and rows_info:
            row_indices = [r[0] for r in rows_info]
            filtered = set(self._dm.filter_rows_by_extra(row_indices, extra_filters))
            rows_info = [r for r in rows_info if r[0] in filtered]

        rows_info.sort(key=lambda r: (natsort_key(str(r[1])), natsort_key(str(r[2])),
                                      natsort_key(str(r[3])), natsort_key(str(r[4]))))

        if not rows_info:
            self._image_display.clear()
            return

        # Use cached overlay data when overlay settings AND the loaded DB
        # haven't changed (the DB path is part of the key, otherwise a
        # different profiler.db with the same table/column names would serve
        # stale overlay data). Keying by id(_metadata_merged) is safe only
        # because _on_metadata_merge/_on_metadata_clear null the cache
        # whenever the DataFrame is replaced — an id alone could be recycled
        # by CPython for a new object.
        meta_id = id(self._metadata_merged)
        db_key = self._dm.db_path or ""
        overlay_key = f"{self._overlay_table}:{self._overlay_col}:{meta_id}:{db_key}"
        if self._overlay_cache is not None and self._overlay_cache_key == overlay_key:
            overlay_values, object_counts, per_object_values, overlay_vmin, overlay_vmax = self._overlay_cache
        else:
            overlay_values, object_counts, per_object_values, overlay_vmin, overlay_vmax = self._compute_overlay_data()
            self._overlay_cache = (overlay_values, object_counts, per_object_values, overlay_vmin, overlay_vmax)
            self._overlay_cache_key = overlay_key

        if channel_toggle:
            # Keep existing thumbnails visible — update in-place when workers finish
            self._pending_workers = len(rows_info)
            self._channel_toggle_results = []
            self._channel_toggle_batch = True
        else:
            self._image_display.begin_results(thumb_size)
            self._pending_workers = len(rows_info)
            self._channel_toggle_batch = False

        gen = self._gen
        channel_names = list(self._ch_config.keys())
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)
        # 'Select object' gates the entire overlay: no mask → no polygons, no
        # hover, no drag. Color by only fills when a mask is also selected.
        sel_mask = self._object_mask_selected
        need_polygons = bool(sel_mask)
        need_mask = bool(sel_mask)
        # Outline only when a mask is selected but Color by is not set; when
        # Color by is also set, fill polygons by value + draw outline.
        outline_only = bool(sel_mask) and (self._overlay_col is None)
        sort_by_row = self._image_controls.sort_by_row.isChecked()

        # Determine which structural and extra columns exist for label building.
        meta_df = self._dm.dataset.metadata
        schema = getattr(self._dm.dataset, "schema", None)
        extra_cols = list(getattr(schema, "extra_cols", []) or [])
        structural_cols = ["well", "field", "stack", "timepoint"]

        for row_idx, well, field, stack, timepoint in rows_info:
            raw_data = self._raw_cache.get(row_idx)

            # Use cached mask and polygons when available (skip re-extraction)
            mask_cache = self._mask_cache.get(row_idx)
            polygons_cache = self._polygon_cache.get(row_idx) if need_polygons else None

            # Build display label from available metadata columns
            meta_label = _build_meta_label(meta_df, row_idx, structural_cols, extra_cols)

            # Image-scoped obj_values to avoid label ID collisions across images
            img_key = (str(field), str(stack), str(timepoint))
            well_obj = per_object_values.get(well, {})
            obj_values = well_obj.get(img_key) or well_obj.get((-1, -1, -1), {})

            config = ImageWorkerConfig(
                row_idx=row_idx, well=well, field=field, stack=stack, timepoint=timepoint,
                raw_data=raw_data, thumb_size=thumb_size,
                channel_names=channel_names, ch_config=self._ch_config, dmax=dmax,
                contrast_method=self._contrast_method,
                contrast_gamma=self._contrast_gamma, invert=self._invert,
                need_polygons=need_polygons,
                need_mask=need_mask,
                dm=self._dm,
                overlay_val=overlay_values.get(well),
                overlay_col=self._overlay_col,
                n_objects=object_counts.get(well),
                obj_values=obj_values,
                overlay_vmin=overlay_vmin, overlay_vmax=overlay_vmax,
                gen=gen, sort_by_row=sort_by_row,
                mask_cache=mask_cache,
                polygons_cache=polygons_cache,
                meta_label=meta_label,
                selected_mask_name=sel_mask,
                outline_only=outline_only,
            )
            worker = ImageWorker(config)
            worker.signals.finished.connect(
                self._on_worker_channel_toggle_finished if channel_toggle
                else self._on_worker_finished,
                Qt.QueuedConnection,
            )
            worker.signals.error.connect(
                lambda msg, g=gen: self._on_worker_error(msg, g),
                Qt.QueuedConnection,
            )
            self._start_worker(worker)

    def _compute_overlay_data(self) -> tuple:
        """Pre-compute overlay values, object counts, and per-object values.

        The overlay source is either a profiler table (`table/col`), the merged
        Excel metadata (`metadata/col`), or a loaded infer DB (`infer:<path>`,
        directory-scoped with fallback). Infer DBs also provide object counts
        from their `label` column when no profiler table does.
        """
        overlay_values: dict[str, float | str] = {}
        # Object-level source frame (reused by the counts/per-object code
        # below); set for infer-style sources with a `label` column.
        infer_frame: pd.DataFrame | None = None

        if self._overlay_col and self._overlay_table:
            if self._overlay_table == "metadata" and self._metadata_merged is not None:
                if self._overlay_col in self._metadata_merged.columns:
                    meta = self._metadata_merged
                    overlay_values = dict(
                        zip(meta["well"], meta[self._overlay_col], strict=False)
                    )
            elif self._overlay_table == "merge":
                try:
                    scoped = self._merged_frame_for_dataset()
                    overlay_values = aggregate_by_well(
                        scoped, self._overlay_col, "mean")
                    if "label" in scoped.columns:
                        infer_frame = scoped
                except Exception:
                    logger.warning("Failed to load merged overlay values", exc_info=True)
            else:
                try:
                    overlay_values = self._dm.aggregate(
                        self._overlay_table, self._overlay_col, "mean"
                    )
                except Exception:
                    logger.warning("Failed to aggregate overlay values", exc_info=True)

        object_counts: dict[str, int] = {}
        object_table: str | None = None
        if infer_frame is not None:
            # Infer rows carry `label` in whole-image mode — counts and
            # per-object values come straight from the (directory-scoped) frame.
            if "label" in infer_frame.columns and "well" in infer_frame.columns:
                counts_df = infer_frame[infer_frame["label"].notna()]
                object_counts = counts_df.groupby("well")["label"].nunique().to_dict()
        else:
            # Prefer user-selected table if it has per-object data
            if self._overlay_table and self._overlay_table != "metadata":
                cols = [c for c, _, _ in self._dm.get_profiling_columns(self._overlay_table)]
                if "label" in cols and "well" in cols:
                    object_table = self._overlay_table
                    try:
                        df = self._dm.get_table_df(object_table)
                        if df is not None:
                            object_counts = df.groupby("well")["label"].nunique().to_dict()
                    except Exception:
                        logger.warning("Failed to compute object counts", exc_info=True)
            # Fallback: auto-discover
            if object_table is None:
                for tname in self._dm.get_profiling_tables():
                    cols = [c for c, _, _ in self._dm.get_profiling_columns(tname)]
                    if "label" in cols and "well" in cols:
                        object_table = tname
                        try:
                            df = self._dm.get_table_df(tname)
                            if df is not None:
                                object_counts = df.groupby("well")["label"].nunique().to_dict()
                        except Exception:
                            logger.warning("Failed to compute object counts", exc_info=True)
                        break

        per_object_values: dict[str, dict[tuple, dict[int, float | str]]] = {}
        overlay_vmin = 0.0
        overlay_vmax = 1.0
        source_df: pd.DataFrame | None = infer_frame
        if source_df is None and object_table and self._overlay_col:
            try:
                source_df = self._dm.get_table_df(object_table)
            except Exception:
                logger.warning("Failed to compute per-object overlay values", exc_info=True)
                source_df = None
        if source_df is not None and self._overlay_col in source_df.columns:
            try:
                odf = source_df[source_df["label"].notna()].copy()
                odf["label"] = pd.to_numeric(odf["label"], errors="coerce")
                odf = odf[odf["label"].notna()]
                odf["label"] = odf["label"].astype(int)
                # Scope per-image to avoid label ID collisions across fields/stacks/timepoints
                group_cols = ["well", "field", "stack", "timepoint"]
                has_image_cols = all(c in odf.columns for c in group_cols)
                if has_image_cols:
                    for (well, field, stack, timepoint), gdf in odf.groupby(group_cols):
                        per_object_values.setdefault(well, {})[(str(field), str(stack), str(timepoint))] = dict(
                            zip(gdf["label"], gdf[self._overlay_col], strict=True)
                        )
                else:
                    for well, wdf in odf.groupby("well"):
                        per_object_values.setdefault(well, {})[(-1, -1, -1)] = dict(
                            zip(wdf["label"], wdf[self._overlay_col], strict=True)
                        )
                # Compute global min/max for colorbar normalization
                col_vals = odf[self._overlay_col]
                if col_vals.dtype.kind in ("i", "f"):
                    overlay_vmin = float(col_vals.min())
                    overlay_vmax = float(col_vals.max())
                    if overlay_vmin == overlay_vmax:
                        overlay_vmin = 0.0
                        overlay_vmax = 1.0
            except Exception:
                logger.warning("Failed to compute per-object overlay values", exc_info=True)

        return overlay_values, object_counts, per_object_values, overlay_vmin, overlay_vmax

    def _on_worker_finished(self, result: dict) -> None:
        """Called on main thread when one image is processed."""
        if self._shutting_down:
            return
        # Discard stale results from previous generations
        if result.get("gen") != self._gen:
            return
        # Cache raw data if worker loaded from disk
        if "raw_data" in result:
            self._raw_cache[result["row_idx"]] = result["raw_data"]
        # Cache mask and polygons for overlay fast paths
        row_idx = result["row_idx"]
        if result.get("mask") is not None:
            self._mask_cache[row_idx] = result["mask"]
        if result.get("polygons") is not None:
            self._polygon_cache[row_idx] = result["polygons"]
        self._pending_workers = max(0, self._pending_workers - 1)
        if self._image_blocked:
            return
        saved_state = getattr(self, "_saved_state", None)
        self._image_display.add_result(
            result, result["thumb_size"], self._overlay_alpha,
            self._overlay_cmap, saved_state, result["sort_by_row"],
            overlay_vmin=result.get("overlay_vmin", 0.0),
            overlay_vmax=result.get("overlay_vmax", 1.0),
        )

    def _finish_channel_toggle_batch(self) -> None:
        """Apply the accumulated toggle results in place (no flash).

        Runs when the last worker of a channel-toggle batch completes —
        whether it finished or errored — so the display always converges on
        the toggled channel state.
        """
        self._channel_toggle_batch = False
        if self._image_blocked:
            return
        # Collect thumbnails currently showing full-res
        full_res_keys = set()
        from microVis.widgets.image_display import _ThumbnailView
        for row_widget, _ in self._image_display._row_widgets.values():
            for thumb in row_widget.findChildren(_ThumbnailView):
                if thumb._is_full_res and thumb._full_res_item is not None:
                    full_res_keys.add(thumb._row_idx)

        self._image_display.update_pixmaps_in_place(
            self._channel_toggle_results,
            self._overlay_alpha, self._overlay_cmap,
            remove_full_res=False,  # keep full-res visible until re-composited
        )

        # Re-dispatch full-res workers for thumbnails that were zoomed in.
        # Dispatch with the thumbnail's ACTUAL generation (not gen=0,
        # which bypasses the stale-result guard in set_full_res_pixmap).
        if full_res_keys:
            from microVis.widgets.image_display import _ThumbnailView
            redispatch: dict[int, int] = {}
            for row_widget, _ in self._image_display._row_widgets.values():
                for thumb in row_widget.findChildren(_ThumbnailView):
                    if thumb._row_idx in full_res_keys:
                        thumb._full_res_gen += 1
                        redispatch[thumb._row_idx] = thumb._full_res_gen
            for row_idx, fr_gen in redispatch.items():
                self._on_full_res_requested(row_idx, gen=fr_gen)

        self._channel_toggle_results = []

    def _on_worker_channel_toggle_finished(self, result: dict) -> None:
        """Called on main thread when a channel-toggle worker completes."""
        if self._shutting_down:
            return
        if result.get("gen") != self._gen:
            return
        # Cache raw data if worker loaded from disk
        if "raw_data" in result:
            self._raw_cache[result["row_idx"]] = result["raw_data"]
        # Cache mask and polygons for overlay fast paths
        row_idx = result["row_idx"]
        if result.get("mask") is not None:
            self._mask_cache[row_idx] = result["mask"]
        if result.get("polygons") is not None:
            self._polygon_cache[row_idx] = result["polygons"]
        self._channel_toggle_results.append(result)
        self._pending_workers = max(0, self._pending_workers - 1)
        # When all workers done, update pixmaps in-place (no flash)
        if self._pending_workers == 0:
            self._finish_channel_toggle_batch()

    def _on_worker_error(self, msg: str, gen: int) -> None:
        if self._shutting_down:
            return
        # Ignore errors from stale workers (cancelled batches) — like the
        # finished path, they must not drive _pending_workers early.
        if gen != self._gen:
            return
        self._pending_workers = max(0, self._pending_workers - 1)
        # An all-error toggle batch must still converge: run the in-place
        # update with whatever results arrived before the failures.
        if self._pending_workers == 0 and self._channel_toggle_batch:
            self._finish_channel_toggle_batch()
        logger.warning("Image worker error: %s", msg)

    def _on_pixel_clicked(self, row_idx: int, x: int, y: int,
                           pixmap_w: int, pixmap_h: int) -> None:
        if self._dm is None:
            return
        try:
            img_data, _ = self._dm.get_imageset(row_idx)
            channels = self._dm.channels

            # Convert scene coordinates to raw image coordinates.
            # x, y are in pixmap pixel space (scene coords).
            # Scale by raw/pixmap ratio to get raw image coordinates.
            h_raw, w_raw = img_data.shape[:2]
            rx = min(int(x * w_raw / pixmap_w), w_raw - 1)
            ry = min(int(y * h_raw / pixmap_h), h_raw - 1)

            # Build display label from available metadata columns.
            meta = self._dm.dataset.metadata
            schema = getattr(self._dm.dataset, "schema", None)
            extra_cols = list(getattr(schema, "extra_cols", []) or [])
            label = _build_meta_label(meta, row_idx,
                                      ["well", "field", "stack", "timepoint"],
                                      extra_cols)
            parts = [f"{label} @ ({rx},{ry})"]
            for i, ch in enumerate(channels):
                if i < img_data.shape[2]:
                    val = img_data[ry, rx, i]
                    parts.append(f"| {ch}: {val:.1f}")
            self._pixel_info.set_text("  ".join(parts))
        except Exception:
            logger.warning("Failed to read pixel info", exc_info=True)

    # ── Full-Res Zoom ───────────────────────────────────────────────────────

    def _on_full_res_requested(self, row_idx: int, gen: int = 0) -> None:
        """Load and display full-resolution image when user zooms past threshold."""
        if self._dm is None:
            return
        if self._image_blocked:
            return

        from microVis.worker import FullResWorker
        channel_names = list(self._ch_config.keys())
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)

        # 'Select object' gates the overlay in full-res just like thumbnails.
        sel_mask = self._object_mask_selected
        need_polygons = bool(sel_mask)
        outline_only = bool(sel_mask) and (self._overlay_col is None)
        obj_values = {}
        overlay_vmin = 0.0
        overlay_vmax = 1.0
        if need_polygons and self._overlay_col is not None and self._overlay_cache is not None:
            _, _, per_object_values, overlay_vmin, overlay_vmax = self._overlay_cache
            # Look up well/field/stack/timepoint from metadata for overlay scoping.
            meta = self._dm.dataset.metadata
            if row_idx in meta.index:
                well = _safe_str(meta, row_idx, "well")
                field = _safe_str(meta, row_idx, "field")
                stack = _safe_str(meta, row_idx, "stack")
                timepoint = _safe_str(meta, row_idx, "timepoint")
                well_obj = per_object_values.get(well, {})
                img_key = (field, stack, timepoint)
                obj_values = well_obj.get(img_key) or well_obj.get((-1, -1, -1), {})

        worker = FullResWorker(
            dm=self._dm, row_idx=row_idx,
            channel_names=channel_names, ch_config=self._ch_config,
            dmax=dmax,
            contrast_method=self._contrast_method,
            contrast_gamma=self._contrast_gamma,
            invert=self._invert,
            gen=gen,
            overlay_alpha=self._overlay_alpha,
            need_polygons=need_polygons,
            obj_values=obj_values,
            overlay_col=self._overlay_col,
            overlay_vmin=overlay_vmin, overlay_vmax=overlay_vmax,
            selected_mask_name=sel_mask,
            outline_only=outline_only,
        )
        worker.signals.finished.connect(self._on_full_res_finished)
        worker.signals.error.connect(
            lambda msg: logger.warning("Full-res worker error: %s", msg))
        self._start_worker(worker)

    def _on_full_res_finished(self, payload, row_idx: int, gen: int,
                               mask=None, obj_values=None, polygons=None,
                               overlay_col="",
                               overlay_vmin=0.0, overlay_vmax=1.0) -> None:
        """Apply full-res result to the matching thumbnail."""
        self._apply_full_res_pixmap(
            row_idx, payload, gen, mask,
            obj_values or {}, polygons, overlay_col,
            overlay_vmin=overlay_vmin, overlay_vmax=overlay_vmax)

    def _apply_full_res_pixmap(self, row_idx: int, payload, gen: int = 0,
                                mask=None, obj_values=None, polygons=None,
                                overlay_col="", overlay_vmin=0.0,
                                overlay_vmax=1.0) -> None:
        """Find the thumbnail widget and set its full-res pixmap.

        The QPixmap is built HERE on the GUI thread (the worker only returns
        the RGB array + polygons — QPixmap is not thread-safe).
        """
        from PySide6.QtGui import QImage, QPixmap
        rgb = payload["rgb"]
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        from microVis.widgets.image_display import _ThumbnailView
        for row_widget, _row_layout in self._image_display._row_widgets.values():
            for thumb in row_widget.findChildren(_ThumbnailView):
                if thumb._row_idx == row_idx:
                    thumb.set_full_res_pixmap(
                        pixmap, gen, mask=mask,
                        obj_values=obj_values, overlay_col=overlay_col,
                        overlay_vmin=overlay_vmin, overlay_vmax=overlay_vmax,
                        polygons=polygons,
                        overlay_alpha=self._overlay_alpha,
                        overlay_cmap=self._overlay_cmap)
                    return

    # ── Label Annotation Handlers ────────────────────────────────────────────

    def _on_label_class_added(self, class_name: str) -> None:
        """Handle new class creation from sidebar."""
        self._label_panel.add_class(class_name)
        # Show class boxes panel on first class creation
        if not self._label_panel.isVisible():
            self._label_panel.setVisible(True)
            self._set_v_splitter_ratios(LABEL_CLASS_RATIOS)
        # Update export "Annotated" option availability
        self._image_controls.update_export_annotated_option(True)

    def _on_label_class_removed(self, class_name: str) -> None:
        """Handle class deletion from sidebar."""
        self._label_panel.remove_class(class_name)
        # Hide panel if no classes remain
        if not self._label_panel.get_all_class_names():
            self._label_panel.setVisible(False)
            self._set_v_splitter_ratios(NO_LABEL_RATIOS)
            # Update export "Annotated" option availability
            self._image_controls.update_export_annotated_option(False)

    def _set_v_splitter_ratios(self, ratios: tuple[float, float, float]) -> None:
        """Redistribute the vertical splitter's current total by fractions."""
        total = sum(self._v_splitter.sizes())
        self._v_splitter.setSizes([int(total * r) for r in ratios])

    def _on_label_class_selection_changed(self) -> None:
        """Handle change in which classes are selected for display."""
        selected = self._image_controls.get_selected_class_names()
        self._label_panel.set_visible_classes(selected)

    def _on_label_write_to_db(self) -> None:
        """Write all label annotations to the database."""
        if self._dm is None:
            return

        # Write-to-DB is an action button — persist current GUI state
        # (patterns + channel colors) to session.yml.
        try:
            if self._session is not None:
                image_pat, mask_pat, subdir_pat = self._data_view.get_patterns()
                self._session.set_patterns(
                    image_pattern=image_pat,
                    mask_pattern=mask_pat,
                    image_subdir_pattern=subdir_pat,
                )
                self._persist_channel_colors()
        except SystemExit:
            # SessionFile.save raises DataError on YAML write failure — never let
            # it kill the app.
            logger.exception("Failed to persist session.yml")

        mask_name = self._image_controls.get_selected_object_mask()
        if not mask_name:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "No Mask Selected",
                "Select an object mask in the Image Controls panel "
                "('Select object') before writing labels to the database.",
            )
            return

        annotations = self._label_panel.get_annotations()
        if not annotations:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "No Annotations",
                "No labeled objects to write. Drag objects into class boxes first.",
            )
            return

        # Build DataFrame
        meta = self._dm.dataset.metadata
        rows = []
        for class_name, keys in annotations.items():
            for key in keys:
                rows.append({
                    "well": _safe_str(meta, key.row_idx, "well"),
                    "field": _safe_str(meta, key.row_idx, "field"),
                    "stack": _safe_str(meta, key.row_idx, "stack"),
                    "timepoint": _safe_str(meta, key.row_idx, "timepoint"),
                    "label": key.label,
                    "class": class_name,
                })
        df = pd.DataFrame(rows)

        # Determine table name
        table_name = self._image_controls.get_label_table_name() or f"{mask_name}_label"

        # Confirmation dialog
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Write Label to DB",
            f"Write {len(df)} label annotations to table '{table_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self._dm.write_label_table(table_name, df)
            logger.info("Wrote %d label annotations to '%s'", len(df), table_name)
        except Exception:
            logger.exception("Failed to write label annotations")
            QMessageBox.warning(
                self, "Write Failed",
                "Failed to write label annotations to database. See log for details.",
            )

    def _on_crop_requested(self, key: ObjectKey, class_name: str) -> None:
        """Dispatch a CropWorker for a dropped object."""
        if self._dm is None:
            return

        mask_name = self._image_controls.get_selected_object_mask()
        if not mask_name:
            return

        row_idx = key.row_idx

        # Load raw image data
        raw_data = self._raw_cache.get(row_idx)
        if raw_data is None:
            try:
                raw_data = self._dm.get_imageset(row_idx)
                self._raw_cache[row_idx] = raw_data
            except Exception:
                logger.warning("Failed to load image for crop: row %d", row_idx, exc_info=True)
                return

        img_data, mask_dict = raw_data

        # Get the selected mask. The dragged object came from the selected
        # mask — never substitute another mask's segmentation (§6.6: no
        # silent first-mask fallback).
        mask_full_name = f"mask_{mask_name}"
        mask = mask_dict.get(mask_full_name)
        if mask is None:
            logger.warning(
                "Selected mask '%s' not found for row %d — crop aborted "
                "(no first-mask fallback)", mask_name, row_idx)
            return

        # Get current channel/contrast settings
        ch_config = self._image_controls.get_channel_config()
        channel_names = list(ch_config.keys())
        dmax = DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0)

        # Dispatch crop worker
        worker = CropWorker(
            img_data=img_data,
            mask=mask,
            label=key.label,
            key=key,
            channel_names=channel_names,
            ch_config=ch_config,
            dmax=dmax,
            contrast_method=self._contrast_method,
            contrast_gamma=self._contrast_gamma,
            invert=self._invert,
            target_size=64,
            padding=4,
        )
        worker.signals.finished.connect(self._on_crop_finished, Qt.QueuedConnection)
        worker.signals.error.connect(
            lambda msg: logger.warning("Crop worker error: %s", msg),
            Qt.QueuedConnection,
        )
        self._start_worker(worker)

    def _on_crop_finished(self, rgb: np.ndarray, key: ObjectKey) -> None:
        """Update the class box thumbnail with the cropped image."""
        if rgb is None:
            return
        # Convert numpy RGB to QPixmap on main thread
        from PySide6.QtGui import QImage, QPixmap
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        # Find which class box contains this key and update it
        for class_name in self._label_panel.get_all_class_names():
            box = self._label_panel.get_class_box(class_name)
            if box is not None and box.has_object(key):
                box.set_object_pixmap(key, pixmap)
                break

    # ── Plot click → single-cell popup ──────────────────────────────────────

    def _on_plot_point_picked(self, row: dict, global_pos) -> None:
        """Answer the Data-plot's clicked point with the cropped single cell.

        A miss (no dataset / no object label / row or mask not resolvable)
        hides the popup instead of showing stale imagery. Any unexpected
        resolution error is logged and treated as a miss — never propagated
        back into the Qt slot (that would only print a traceback and leave
        the GUI looking dead).
        """
        try:
            pixmap = self._resolve_point_cell(row)
        except Exception:
            logger.warning("Plot click: cell resolution failed",
                           exc_info=True)
            pixmap = None
        if pixmap is None:
            self._plot_view.hide_cell_image()
            return
        self._plot_view.show_cell_image(pixmap)

    def _resolve_point_cell(self, row: dict):
        """Crop the single cell for a clicked merged-table row.

        The physical object is located through the dataset metadata
        (directory match first, then well + field/stack/timepoint) and
        cropped from the selected object mask (fallback: first mask — this
        is a read-only viewer, not annotation, so a substitute mask only
        changes the outline shown, never written anywhere).
        """
        if self._dm is None:
            return None
        label = row.get("label")
        if label is None:
            logger.info("Plot click: row has no object label — nothing to crop")
            return None
        meta = self._dm.dataset.metadata
        if meta is None:
            return None

        # Locate the metadata site of the clicked object. The identity
        # columns (well/field/stack/timepoint) pin down the SITE; the DB
        # directory only narrows datasets whose images live in several
        # subdirectories. Legacy DBs written before the absolute-directory
        # contract hold root-relative dirs ('Images'), which match EVERY
        # site — so directory alone must never decide the site.
        cand = pd.Series(True, index=meta.index)
        identity_used = False
        for col in ("well", "field", "stack", "timepoint"):
            v = row.get(col)
            if v is not None and col in meta.columns:
                cand &= meta[col].astype(str) == str(v)
                identity_used = True

        directory = row.get("directory")
        dir_match_idx = None
        if directory and DIRECTORY_COLUMN in meta.columns:
            # Current DBs hold absolute dirs → compare joined-to-root;
            # legacy DBs hold root-relative ones → compare verbatim.
            accepted = {_abs_norm_dir(directory), _rel_norm_dir(directory)}
            root = Path(self._dm.dataset.root)

            def _dir_match(v):
                keys = (_abs_norm_dir(root / str(v)), _rel_norm_dir(v))
                return keys[0] in accepted or keys[1] in accepted

            dir_mask = meta[DIRECTORY_COLUMN].map(_dir_match)
            if dir_mask.any():
                cand &= dir_mask
                dir_match_idx = meta.index[dir_mask]

        match = meta.index[cand]
        if len(match):
            row_idx = int(match[0])
        elif identity_used:
            # Identity columns existed but pinned nothing — the row does
            # not belong to this dataset; guessing a site would crop the
            # WRONG cell, so show nothing instead.
            logger.info("Plot click: row does not match any dataset image "
                        "(identity columns had no site in common)")
            return None
        else:
            # No identity columns available (e.g. whole-image infer rows):
            # fall back to the first site of the matching directory.
            row_idx = int((dir_match_idx if dir_match_idx is not None
                           else meta.index)[0])

        raw_data = self._raw_cache.get(row_idx)
        if raw_data is None:
            try:
                raw_data = self._dm.get_imageset(row_idx)
                self._raw_cache[row_idx] = raw_data
            except Exception:
                logger.warning("Plot click: image load failed for row %d",
                               row_idx, exc_info=True)
                return None
        img_data, mask_dict = raw_data

        mask_name = self._image_controls.get_selected_object_mask()
        if not mask_name and self._dm.mask_names:
            mask_name = self._dm.mask_names[0]
        mask = mask_dict.get(f"mask_{mask_name}") if mask_name else None
        if mask is None:
            logger.info("Plot click: no object mask available for the crop")
            return None
        site = self._dm.dataset.metadata.iloc[row_idx]
        site_desc = ", ".join(
            f"{c}={site[c]}" for c in ("well", "field", "stack", "timepoint")
            if c in site.index)
        logger.info("Plot click: cropping label %s from %s (%s)",
                    label, mask_name, site_desc)

        ch_config = self._image_controls.get_channel_config()
        rgb = crop_object_rgb(
            img_data, mask, int(label), list(ch_config.keys()), ch_config,
            DTYPE_MAX.get(str(self._dm.img_dtype), 65535.0),
            self._contrast_method, self._contrast_gamma, self._invert,
            target_size=96, padding=4,
        )
        if rgb is None:
            logger.info("Plot click: label %s not croppable from mask '%s' "
                        "(missing or empty object)", label, mask_name)
            return None

        from PySide6.QtGui import QImage, QPixmap
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    # ── Object Export Handler ────────────────────────────────────────────────

    def _on_export_clicked(self) -> None:
        """Handle object export button click."""
        if self._dm is None:
            return
        from PySide6.QtWidgets import QMessageBox

        # Export is an action button — persist current GUI state (patterns +
        # channel colors) to session.yml before starting the export work.
        try:
            if self._session is not None:
                image_pat, mask_pat, subdir_pat = self._data_view.get_patterns()
                self._session.set_patterns(
                    image_pattern=image_pat,
                    mask_pattern=mask_pat,
                    image_subdir_pattern=subdir_pat,
                )
                self._persist_channel_colors()
        except SystemExit:
            # SessionFile.save raises DataError on YAML write failure — never let
            # it kill the app.
            logger.exception("Failed to persist session.yml")

        ic = self._image_controls
        object_mode = ic.get_export_object_mode()
        save_dir = ic.get_export_dir()

        # Determine mask name from export mask dropdown
        mask_name = ic.get_export_mask()
        if not mask_name:
            QMessageBox.warning(self, "Export Error", "No mask selected for object extraction.")
            return

        # Determine save directory
        if not save_dir:
            save_dir = str(Path(self._dataset_dir) / "objects_exported")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Get annotations if needed
        annotations = None
        if object_mode == "Annotated":
            annotations = self._label_panel.get_annotations()
            if not annotations:
                QMessageBox.information(
                    self, "No Annotations",
                    "No annotated objects found. Create classes and annotate objects first.",
                )
                return

        # Get wells/fields based on mode
        annotated_keys = None  # set of row_idx for "Annotated"
        extra_filters: dict[str, list[str]] = {}
        if object_mode == "Selected wells":
            # Wells exist but none selected → warn, never export the whole
            # plate (matches the viewer's "wells exist but none selected →
            # show nothing" rule; "All" mode is the explicit whole-plate path).
            if not self._selected_wells and self._dm.get_wells():
                QMessageBox.warning(
                    self, "No Wells Selected",
                    "No wells are selected in the plate grid. Select at least "
                    "one well to export, or switch the mode to 'All'.",
                )
                return
            wells = sorted(self._selected_wells)
            fields = list(ic.get_selected_fields())
            stacks = list(ic.get_selected_stacks())
            timepoints = list(ic.get_selected_timepoints())
            # Extra-col filters only apply to "Selected wells" — "Selected
            # wells (all objects)" exports all images in the wells by user
            # intent, "All" exports everything, "Annotated" is scoped to
            # annotated images.
            for col, widget in ic.get_extra_widgets().items():
                selected = widget.get_selected()
                if selected:
                    extra_filters[col] = selected
        elif object_mode == "Selected wells (all objects)":
            # Selected wells only — the Image Filters (fields/stacks/
            # timepoints/extra cols) are intentionally ignored. Empty
            # wells/fields/stacks/timepoints means "no filter" in the worker.
            if not self._selected_wells and self._dm.get_wells():
                QMessageBox.warning(
                    self, "No Wells Selected",
                    "No wells are selected in the plate grid. Select at least "
                    "one well to export, or switch the mode to 'All'.",
                )
                return
            wells = sorted(self._selected_wells)
            fields = []
            stacks = []
            timepoints = []
        elif object_mode == "Annotated" and annotations:
            # Only scan images that contain annotated objects. annotated_keys
            # is a set of row_idx values — the worker uses it to filter rows
            # after lookup_row_indices. Empty wells/fields/stacks/timepoints
            # means "no filter" (lookup_row_indices returns all rows), then
            # annotated_keys scopes to annotated images.
            annotated_keys = set()
            for keys in annotations.values():
                for key in keys:
                    annotated_keys.add(key.row_idx)
            wells = []
            fields = []
            stacks = []
            timepoints = []
        else:  # "All"
            wells = self._dm.get_wells()
            fields = self._dm.get_fields()
            stacks = self._dm.get_stacks()
            timepoints = self._dm.get_timepoints()

        # Disable UI during export
        ic.set_export_enabled(False)
        self._pixel_info.set_text("Exporting objects...")

        # Run export in background thread
        self._export_gen = getattr(self, "_export_gen", 0) + 1
        gen = self._export_gen

        from microVis.worker import ObjectExportWorker

        max_objects = ic.get_export_max_objects()
        ch_config = ic.get_channel_config()
        channel_names = [ch for ch in self._dm.channels if ch_config.get(ch, {}).get("enabled", False)]
        ch_colors = {
            ch: cfg.get("color", (0.0, 1.0, 0.0))
            for ch, cfg in ch_config.items()
            if ch in channel_names
        }
        well_subdir = ic.get_export_well_subdir()

        worker = ObjectExportWorker(
            dm=self._dm,
            wells=wells,
            fields=fields,
            stacks=stacks,
            timepoints=timepoints,
            mask_name=mask_name,
            channel_names=channel_names,
            ch_colors=ch_colors,
            annotated_keys=annotated_keys,
            save_dir=str(save_path),
            object_mode=object_mode,
            annotations=annotations,
            gen=gen,
            max_objects_per_image=max_objects,
            well_subdir=well_subdir,
            extra_filters=extra_filters,
        )
        worker.signals.progress.connect(
            lambda c, t, g=gen: self._on_export_progress(c, t, g),
            Qt.QueuedConnection)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(
            lambda msg, g=gen: self._on_export_error(msg, g),
            Qt.QueuedConnection,
        )
        self._start_worker(worker)

    def _on_export_progress(self, current: int, total: int, gen: int = 0) -> None:
        """Update export progress (gen-guarded: a stale export from a
        previous dataset must not overwrite the status bar)."""
        if self._shutting_down:
            return
        if gen != getattr(self, "_export_gen", 0):
            return
        self._pixel_info.set_text(f"Exporting: {current}/{total} images...")

    def _on_export_finished(self, result: dict) -> None:
        """Handle export completion."""
        if result.get("gen") != getattr(self, "_export_gen", 0):
            return
        if self._shutting_down:
            return
        self._image_controls.set_export_enabled(True)
        count = result.get("count", 0)
        save_dir = result.get("save_dir", "")
        skipped = result.get("skipped_rows", 0)
        msg = f"Exported {count} objects to {save_dir}"
        if skipped:
            msg += f" ({skipped} rows skipped — missing masks)"
        self._pixel_info.set_text(msg)
        logger.info("%s", msg)

    def _on_export_error(self, msg: str, gen: int) -> None:
        """Handle export error."""
        if self._shutting_down:
            return
        # Ignore stale export failures (e.g. after browsing away mid-export
        # bumped _export_gen) — they must not re-enable the UI or report for
        # a dataset no longer on screen.
        if gen != getattr(self, "_export_gen", 0):
            return
        self._image_controls.set_export_enabled(True)
        self._pixel_info.set_text(f"Export error: {msg}")
        logger.warning("Export error: %s", msg)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def _on_full_reset(self) -> None:
        """Reset everything to initial startup state without restarting the app."""
        self._shutting_down = True

        # Invalidate all pending workers
        self._cancel_workers()
        self._export_gen = getattr(self, "_export_gen", 0) + 1
        self._thread_pool.waitForDone(2000)
        QApplication.processEvents()

        # Close DataModule
        if self._dm is not None:
            self._dm.close_db()
            self._dm = None

        # Clear data references
        self._dataset_dir = None
        self._session = None
        self._loaded_dataset_dir = None
        self._update_window_title()
        self._metadata_df = None
        self._metadata_merged = None

        # Drop the integrated table and the plot view's data
        self._reset_merged_data()

        # Clear caches
        self._raw_cache.clear()
        self._mask_cache.clear()
        self._polygon_cache.clear()
        self._last_state.clear()
        self._overlay_cache = None
        self._overlay_cache_key = None

        # Reset state fields
        self._selected_wells = set()
        self._ch_config = {}
        self._contrast_method = "gamma"
        self._contrast_gamma = 1.0
        self._invert = False
        self._overlay_table = None
        self._overlay_col = None
        self._overlay_cmap = "viridis"
        self._overlay_alpha = 0.0
        self._object_mask_selected = ""
        self._image_blocked = False
        self._pending_workers = 0
        self._channel_toggle_batch = False
        self._channel_toggle_results = []

        # Reset widgets
        self._image_display.clear()
        self._image_display.reset_all_zoom()
        self._label_panel.clear_all()
        self._label_panel.setVisible(False)
        self._v_splitter.setSizes(list(V_SPLITTER_SIZES))
        self._grid_canvas.clear()
        # Restore well grid visibility to default; next dataset load re-evaluates
        self._well_grid_container.setVisible(True)

        # Reset image controls
        ic = self._image_controls
        ic.set_channels({})
        ic.set_filter_options([], [], [])
        ic.overlay_col.blockSignals(True)
        ic.overlay_col.clear()
        ic.overlay_col.addItem("None")
        ic.overlay_col.blockSignals(False)
        ic.overlay_cmap.blockSignals(True)
        ic.overlay_cmap.clear()
        ic.overlay_cmap.addItems(CMAP_OPTIONS)
        ic.overlay_cmap.setCurrentText(DEFAULT_CMAP)
        ic.overlay_cmap.blockSignals(False)
        ic.contrast.blockSignals(True)
        ic.contrast.setCurrentText("gamma")
        ic.contrast.blockSignals(False)
        ic.gamma_slider.blockSignals(True)
        ic.gamma_slider.setValue(100)
        ic.gamma_slider.blockSignals(False)
        ic.set_gamma_visible(True)
        ic.auto_low.blockSignals(True)
        ic.auto_high.blockSignals(True)
        ic.auto_low.setValue(0.1)
        ic.auto_high.setValue(99.9)
        ic.auto_low.blockSignals(False)
        ic.auto_high.blockSignals(False)
        ic.set_label_masks([])
        ic.set_object_masks([])
        ic.clear_classes()
        ic.set_export_enabled(True)
        ic.update_export_annotated_option(False)
        # Reset widget state that _on_full_reset's field resets must stay in
        # sync with (otherwise e.g. the alpha slider shows 40% while
        # _overlay_alpha is 0.0 and the next touch snaps the rendering).
        ic.overlay_alpha.blockSignals(True)
        ic.overlay_alpha.setValue(0)
        ic.overlay_alpha.blockSignals(False)
        ic.sort_by_row.blockSignals(True)
        ic.sort_by_row.setChecked(False)
        ic.sort_by_row.blockSignals(False)
        for w in (getattr(ic, "_export_dir_input", None),
                  getattr(ic, "_export_well_subdir", None),
                  getattr(ic, "_label_table_name", None),
                  getattr(ic, "_class_input", None)):
            if w is not None:
                w.blockSignals(True)
                w.clear()
                w.blockSignals(False)
        max_obj = getattr(ic, "_export_max_obj", None)
        if max_obj is not None:
            max_obj.blockSignals(True)
            max_obj.setValue(0)
            max_obj.blockSignals(False)

        # Reset grid controls
        gw = self._grid_controls
        gw.plate_format.blockSignals(True)
        gw.plate_format.clear()
        gw.plate_format.addItems(list(PLATE_FORMATS.keys()))
        idx = gw.plate_format.findText(DEFAULT_PLATE)
        if idx >= 0:
            gw.plate_format.setCurrentIndex(idx)
        gw.plate_format.blockSignals(False)
        gw.column.blockSignals(True)
        gw.column.clear()
        gw.column.addItem("None")
        gw.column.blockSignals(False)
        gw.aggregation.blockSignals(True)
        gw.aggregation.clear()
        gw.aggregation.addItems(AGG_METHODS)
        gw.aggregation.blockSignals(False)
        gw.colormap.blockSignals(True)
        gw.colormap.clear()
        gw.colormap.addItems(CMAP_OPTIONS)
        gw.colormap.setCurrentText(DEFAULT_CMAP)
        gw.colormap.blockSignals(False)
        gw.palette.blockSignals(True)
        gw.palette.clear()
        gw.palette.addItems(QUALITATIVE_PALETTES)
        gw.palette.setCurrentText("Set1")
        gw.palette.blockSignals(False)
        # The image-block toggle is a plain button (no signals to block) —
        # reset its label/style so it agrees with _image_blocked = False.
        gw.reset_image_block()

        # Reset data view
        self._data_view.reset()

        # Reset pixel info
        self._pixel_info.set_text("")

        # Switch to Data tab
        self._switch_tab(0)

        self._shutting_down = False
        logger.info("Session reset to initial state")

    def closeEvent(self, event) -> None:
        self._shutting_down = True
        self._cancel_workers()
        self._thread_pool.waitForDone(2000)
        # Flush Windows message queue to clear "Not Responding" state
        QApplication.processEvents()
        if self._dm is not None:
            self._dm.close_db()
        super().closeEvent(event)
