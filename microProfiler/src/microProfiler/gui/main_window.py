from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import warnings

from PySide6.QtCore import Qt, QEvent, QObject
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
from microProfiler.gui.path_drop import enable_path_drop
from microProfiler.gui.pipeline_controller import PipelineController
from microProfiler.gui.state import PipelineState
from microProfiler.gui.ui_spec import (
    PAGE_MARGIN,
    PANEL_CONTENT_STYLE,
    PATTERN_EDIT_STYLE,
    PATTERN_LABEL_STYLE,
    SCROLL_CONTENT_SPACING,
    STATUS_BAR_HEIGHT,
    WINDOW_SIZE,
    run_button,
)
from microProfiler.gui.workers.preview_worker import PreviewWorker
from microProfiler.gui.workers.dataset_load_worker import DatasetLoadWorker
from microProfiler.gui.sidebar import Sidebar
from microProfiler.user_defaults import (
    ensure_user_defaults,
    get_user_defaults,
    update_user_defaults,
)
from microProfiler.gui.panels import (
    BaSiCStepPanel, FilterPanel,
    ImageProfilingStepPanel, InferenceStepPanel, ObjectProfilingStepPanel,
    ResizeStepPanel, SegmentStepPanel, TileStepPanel, ZProjectStepPanel,
)
from microProfiler.gui.panels.base_step_panel import BaseStepPanel
from microProfiler.gui.image_widgets import ImageViewer
from microProfiler.gui.dpi import dp
from microProfiler.log_utils import set_log_file, setup_logging


class WindowWheelFilter(QObject):

    def __init__(self, window: QMainWindow):
        super().__init__()
        self._window = window

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.Wheel:
            return False
        # A combo popup (or any other top-level popup) is its own window —
        # its wheel events must NOT be swallowed by the main-window scroller.
        if isinstance(watched, QWidget) and watched.window() is not self._window:
            return False
        if event.modifiers() & Qt.ControlModifier:
            return False
        mw = self._window
        if not hasattr(mw, "_stack"):
            return False
        current_idx = mw._stack.currentIndex()
        scroll_area = self._page_scroll_area(mw, current_idx)
        if scroll_area is not None:
            vbar = scroll_area.verticalScrollBar()
            delta = event.angleDelta().y()
            vbar.setValue(vbar.value() - delta)
        return True

    @staticmethod
    def _page_scroll_area(window, page_idx: int):
        page = window._stack.widget(page_idx)
        if page is None:
            return None
        scroll = page.findChild(QScrollArea)
        return scroll


def _pattern_label(text: str) -> QLabel:
    """Pattern-row label on the shared 9pt panel scale (Input page)."""
    lbl = QLabel(text)
    lbl.setStyleSheet(PATTERN_LABEL_STYLE)
    return lbl


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._state = PipelineState()
        # The input dir is the dataset dir: results (profiler.db, masks,
        # session.yml, infer.db) are always written next to the sources.
        self._loaded_dataset_dir = None
        self._pending_filters = None

        self.setWindowTitle("microProfiler")
        self.setMinimumSize(*WINDOW_SIZE)
        self._restore_window_size()
        self._running = False

        setup_logging(clear_existing=False)
        warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

        self._build_ui()
        self._ctrl = PipelineController(self)
        self._connect_signals()
        self._apply_compact_widths()
        self.setFocus()

    # ── IControllerView implementation ──────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    def set_running(self, running: bool) -> None:
        self._set_running(running)

    def get_input_dir(self) -> str:
        return self._input_dir.text()

    def get_image_pattern(self) -> Optional[str]:
        return self._custom_image_pattern.text().strip() or None

    def set_image_pattern(self, pattern: str) -> None:
        self._custom_image_pattern.setText(pattern)

    def get_mask_pattern(self) -> Optional[str]:
        return self._custom_mask_pattern.text().strip() or None

    def set_mask_pattern(self, pattern: str) -> None:
        self._custom_mask_pattern.setText(pattern)

    def get_image_subdir_pattern(self) -> Optional[str]:
        return self._custom_image_subdir_pattern.text().strip() or None

    def _clear_progress_error(self) -> None:
        """Drop the error style class set by progress_show_error."""
        if self._progress_label.property("class") == "error":
            self._progress_label.setProperty("class", "status-bar")
            self._progress_label.style().polish(self._progress_label)

    def progress_reset(self) -> None:
        self._clear_progress_error()
        self._progress_label.setVisible(False)

    def progress_finished(self) -> None:
        self._clear_progress_error()
        self._progress_label.setVisible(False)

    def progress_show_status(self, message: str) -> None:
        self._clear_progress_error()
        self._progress_label.setText(message)
        self._progress_label.setVisible(True)

    def progress_show_error(self, message: str) -> None:
        self._progress_label.setText(f"Error: {message}")
        self._progress_label.setProperty("class", "error")
        self._progress_label.style().polish(self._progress_label)
        self._progress_label.setVisible(True)

    def progress_connect_update(self, signal) -> None:
        signal.connect(self._on_log_progress, Qt.ConnectionType.QueuedConnection)

    def _tqdm_bar(self, step: str, current: int, total: int, width: int = 40) -> str:
        pct = current / total if total > 0 else 0
        filled = int(pct * width)
        bar = "\u2588" * filled + "\u2591" * (width - filled)
        return f"[{step}] {bar} {current}/{total} ({pct*100:.0f}%)"

    def _on_log_progress(self, step: str, current: int, total: int, message: str) -> None:
        if message and (total <= 0 or current == 0):
            # Status/start lines (e.g. "Starting...", microModel's INFO logs) —
            # show the text instead of a meaningless 0/N bar. step_start emits
            # (0, 1, msg), so current==0 with a message is a text line too.
            self._progress_label.setText(f"[{step}] {message}")
        else:
            self._progress_label.setText(self._tqdm_bar(step, current, total))
        self._progress_label.setVisible(True)

    def get_all_step_panels(self):
        return self._all_step_panels

    def get_preprocessing_steps(self):
        return self._preprocessing_steps

    def get_step_panel(self, name: str):
        mapping = {
            "resize": self._resize_panel,
            "basic": self._basic_panel,
            "zproject": self._zproject_panel,
            "tile": self._tile_panel,
            "segment": self._segment_panel,
            "image_profile": self._image_profile_panel,
            "object_profile": self._object_profile_panel,
            "inference": self._inference_panel,
            "filter": self._filter_panel,
        }
        return mapping.get(name)

    def update_dataset_info(self, ds) -> None:
        self._update_dataset_info(ds)

    def update_tab_status(self) -> None:
        self._update_tab_status()

    def refresh_step_panels(self, channels=None, masks=None) -> None:
        """Repopulate channel/mask-driven panels after dataset state changes.

        Channels go to segment / image_profile / object_profile / inference
        panels. Masks: pass an explicit list (e.g. [] on reset) to use it
        verbatim, or leave None to use the controller's segment-first mask
        ordering (segment Object names first, then dataset masks).
        """
        ch = list(channels) if channels is not None else []
        self._segment_panel.populate_channels(ch)
        self._image_profile_panel.populate_channels(ch)
        self._object_profile_panel.populate_channels(ch)
        self._inference_panel.populate_channels(ch)
        if masks is None:
            self._ctrl._sync_seg_masks_to_profiling()
        else:
            self._object_profile_panel.populate_masks(list(masks))
            self._inference_panel.populate_masks(list(masks))

    @property
    def dataset(self):
        return self._state.dataset

    @dataset.setter
    def dataset(self, ds):
        self._state.dataset = ds

    @property
    def original_dataset(self):
        return self._state.original_dataset

    @original_dataset.setter
    def original_dataset(self, ds):
        self._state.original_dataset = ds

    def widget(self):
        return self

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self._sidebar = Sidebar()
        main_layout.addWidget(self._sidebar)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(0)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # ── Create step panels ─────────────────────────────────────────
        self._resize_panel = ResizeStepPanel(self._state)
        self._basic_panel = BaSiCStepPanel(self._state)
        self._zproject_panel = ZProjectStepPanel(self._state)
        self._tile_panel = TileStepPanel(self._state)
        self._segment_panel = SegmentStepPanel(self._state)
        self._image_profile_panel = ImageProfilingStepPanel(self._state)
        self._object_profile_panel = ObjectProfilingStepPanel(self._state)
        self._inference_panel = InferenceStepPanel(self._state)
        self._filter_panel = FilterPanel(self._state)

        self._all_step_panels = [
            self._resize_panel, self._zproject_panel,
            self._basic_panel, self._tile_panel,
            self._segment_panel, self._image_profile_panel,
            self._object_profile_panel, self._inference_panel,
        ]

        self._preprocessing_steps = [
            self._resize_panel, self._zproject_panel,
            self._basic_panel, self._tile_panel,
        ]

        # ── QStackedWidget ─────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── Page 0: Input + Filter ────────────────────────────────────
        input_page = QWidget()
        # The Input page is a control panel: same 9pt content scale as the
        # step cards.
        input_page.setStyleSheet(PANEL_CONTENT_STYLE)
        input_layout = QVBoxLayout(input_page)
        input_layout.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN,
                                        PAGE_MARGIN, PAGE_MARGIN)

        input_group = QGroupBox("Input")
        input_form = QFormLayout(input_group)
        input_form.setLabelAlignment(Qt.AlignLeft)
        input_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._input_dir = QLineEdit()
        enable_path_drop(self._input_dir)
        self._input_browse = QPushButton("Browse...")
        self._load_dataset_btn = run_button("Load Dataset")
        # Requested: twice the natural caption length.
        self._load_dataset_btn.ensurePolished()
        self._load_dataset_btn.setFixedWidth(
            self._load_dataset_btn.sizeHint().width() * 2)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Reset the pipeline and clear the dataset")
        input_row = QHBoxLayout()
        # The path box takes 2/3 of the free width; Browse / Load Dataset
        # follow it immediately and the remaining third stays empty before
        # the right-aligned Reset.
        input_row.addWidget(self._input_dir, 2)
        input_row.addWidget(self._input_browse)
        input_row.addWidget(self._load_dataset_btn)
        input_row.addStretch(1)
        input_row.addSpacing(12)
        input_row.addWidget(self._reset_btn)
        input_form.addRow("Dataset:", input_row)

        img_row = QHBoxLayout()
        self._custom_image_pattern = QLineEdit()
        self._custom_image_pattern.setStyleSheet(PATTERN_EDIT_STYLE)
        self._custom_image_pattern.setPlaceholderText(r"(?P<well>[A-Z]\d+)_f(?P<field>\d+)_ch(?P<channel>\d+)\.tif")
        img_row.addWidget(self._custom_image_pattern, 1)
        input_form.addRow(_pattern_label("Image pattern:"), img_row)

        mask_row = QHBoxLayout()
        self._custom_mask_pattern = QLineEdit()
        self._custom_mask_pattern.setStyleSheet(PATTERN_EDIT_STYLE)
        self._custom_mask_pattern.setPlaceholderText(r".*_masks_(?P<mask_name>.+)\.png")
        mask_row.addWidget(self._custom_mask_pattern, 1)
        input_form.addRow(_pattern_label("Mask pattern:"), mask_row)

        search_row = QHBoxLayout()
        self._custom_image_subdir_pattern = QLineEdit()
        self._custom_image_subdir_pattern.setStyleSheet(PATTERN_EDIT_STYLE)
        self._custom_image_subdir_pattern.setPlaceholderText("Leave empty to search directly")
        search_row.addWidget(self._custom_image_subdir_pattern, 1)
        input_form.addRow(_pattern_label("Image subdir:"), search_row)

        self._dataset_info_label = QLabel("")
        self._dataset_info_label.setWordWrap(True)
        self._dataset_info_label.setProperty("class", "placeholder")
        self._dataset_info_label.setVisible(False)
        input_form.addRow(self._dataset_info_label)

        input_layout.addWidget(input_group)

        # Filter section inside the input tab: it gets only a small share of
        # the leftover height (requested: half its previous allocation), so
        # the trailing spacer absorbs the rest while extra filter rows can
        # still grow the box.
        input_layout.addWidget(self._filter_panel, 1)

        input_layout.addStretch(3)
        self._stack.addWidget(input_page)

        # ── Pages 1-4: one scrollable page per pipeline stage ──────────
        # Each page is the step card(s) inside a scroll area with the stage's
        # Run button pinned BELOW the scroll area, so it stays reachable
        # however long the card content grows.
        self._run_pre_btn = run_button("Run Preprocessing")
        self._stack.addWidget(self._make_step_page(
            [self._resize_panel, self._zproject_panel,
             self._basic_panel, self._tile_panel],
            self._run_pre_btn))

        self._run_seg_btn = run_button("Run Segmentation")
        self._stack.addWidget(self._make_step_page(
            [self._segment_panel], self._run_seg_btn))

        self._run_prof_btn = run_button("Run Profiling")
        self._stack.addWidget(self._make_step_page(
            [self._image_profile_panel, self._object_profile_panel],
            self._run_prof_btn))

        self._run_infer_btn = run_button("Run Inference")
        self._stack.addWidget(self._make_step_page(
            [self._inference_panel], self._run_infer_btn))

        # ── Progress bar at bottom ─────────────────────────────────────
        self._progress_label = QLabel("")
        self._progress_label.setVisible(False)
        self._progress_label.setFixedHeight(dp(STATUS_BAR_HEIGHT))
        self._progress_label.setProperty("class", "status-bar")

        right_layout.addWidget(self._stack, 1)
        right_layout.addWidget(self._progress_label)

        main_layout.addWidget(right_widget, 1)

        self._wheel_filter = WindowWheelFilter(self)
        QApplication.instance().installEventFilter(self._wheel_filter)

        # Seed pattern fields with microBase defaults at startup.
        self._custom_image_pattern.setText(DEFAULT_IMAGE_PATTERN)
        self._custom_mask_pattern.setText(DEFAULT_MASK_PATTERN)
        self._custom_image_subdir_pattern.setText(DEFAULT_IMAGE_SUBDIR_PATTERN)

    def _make_step_page(self, panels, run_btn: QPushButton) -> QWidget:
        """Build one stacked page: step card(s) in a scroll area + Run button.

        The button sits outside the scroll area (pinned to the page bottom)
        so it never scrolls out of reach on long pages.
        """
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN,
                                       PAGE_MARGIN, PAGE_MARGIN)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(SCROLL_CONTENT_SPACING)
        for p in panels:
            inner_layout.addWidget(p)
        inner_layout.addStretch()
        scroll.setWidget(inner)

        page_layout.addWidget(scroll, 1)
        page_layout.addWidget(run_btn)
        return page

    # ── Signal connections ────────────────────────────────────────────────

    def _connect_signals(self):
        self._sidebar.navigation_changed.connect(self._on_navigation_changed)
        self._sidebar.thread_count_changed.connect(self._on_thread_count_changed)
        self._sidebar.display_range_changed.connect(self._on_vmin_vmax_changed)
        self._input_browse.clicked.connect(self._browse_input)
        self._input_dir.textChanged.connect(self._on_input_dir_edited)
        self._load_dataset_btn.clicked.connect(self._load_dataset)
        self._reset_btn.clicked.connect(self._reset_all)
        self._sidebar.run_all_clicked.connect(self._ctrl.run_all)
        self._run_pre_btn.clicked.connect(self._ctrl.run_preprocessing)
        self._run_seg_btn.clicked.connect(self._ctrl.run_segmentation)
        self._run_prof_btn.clicked.connect(self._ctrl.run_profiling)
        self._run_infer_btn.clicked.connect(self._ctrl.run_inference)
        self._sidebar.cancel_clicked.connect(self._ctrl._cancel_current_worker)

        self._ctrl._preview_worker = PreviewWorker()
        self._ctrl._preview_worker.preview_ready.connect(self._ctrl.on_preview_ready)
        self._ctrl._preview_worker.error.connect(self._ctrl.on_preview_error)

        for step in self._all_step_panels:
            if hasattr(step, "_pick_btn"):
                step._pick_btn.clicked.connect(lambda checked, s=step: self._ctrl.pick_random(s))
            if hasattr(step, "_preview_btn"):
                step._preview_btn.clicked.connect(lambda checked, s=step: self._ctrl.preview_step(s))
            if hasattr(step, "_fit_btn"):
                step._fit_btn.clicked.connect(self._ctrl.fit_basic)
            if hasattr(step, "_apply_btn"):
                step._apply_btn.clicked.connect(lambda checked, s=step: self._ctrl.apply_step(s))

        self._segment_panel.pick_requested.connect(self._ctrl.on_segment_pick)
        self._segment_panel.preview_requested.connect(self._ctrl.on_segment_preview)

        self._filter_panel.filter_changed.connect(self._on_filter_changed)
        self._segment_panel.parameter_changed.connect(self._ctrl._sync_seg_masks_to_profiling)

    def _on_navigation_changed(self, page_id: str) -> None:
        index_map = {
            "input": 0, "preprocess": 1, "segment": 2,
            "profile": 3, "inference": 4,
        }
        if page_id in index_map:
            self._stack.setCurrentIndex(index_map[page_id])
            if page_id in ("profile", "inference"):
                self._ctrl._sync_seg_masks_to_profiling()

    def _on_thread_count_changed(self, value: int) -> None:
        self._image_profile_panel.set_n_workers(value)
        self._object_profile_panel.set_n_workers(value)

    def _on_vmin_vmax_changed(self, vmin: float, vmax: float) -> None:
        ImageViewer.display_vmin = vmin
        ImageViewer.display_vmax = vmax
        for w in QApplication.instance().allWidgets():
            if isinstance(w, ImageViewer):
                w.refresh()

    # ── Input / browse ──────────────────────────────────────────────────

    def _browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if not path:
            return
        self._input_dir.setText(path)
        # session.yml is read only on the FIRST browse of a given directory
        # (pre-fills GUI fields); re-browsing the SAME directory preserves the
        # user's GUI edits and the loaded dataset — session.yml is never
        # re-read into the GUI (§3.16).
        if path == self._loaded_dataset_dir:
            return
        self._on_input_changed()

        sf = SessionFile(path)
        session_data = sf.load()
        if not session_data:
            return
        params = {k: v for k, v in session_data.items() if not k.startswith("_")}
        applied = session_data.get("applied_steps", [])

        saved_img = params.get("image_pattern", "")
        saved_mask = params.get("mask_pattern", "")
        saved_search = params.get("image_subdir_pattern", "")
        if saved_img:
            self.set_image_pattern(saved_img)
        if saved_mask:
            self.set_mask_pattern(saved_mask)
        if saved_search:
            self._custom_image_subdir_pattern.setText(saved_search)

        for step in self._all_step_panels:
            step.from_config(params.get(step.step_name, {}))
        for step in self._all_step_panels:
            if step.step_name in applied:
                step.setChecked(True)
        # Filters are applied after the dataset is loaded (the panel needs
        # the dataset's metadata columns to populate the dropdowns).
        if "filter" in session_data:
            self._pending_filters = session_data["filter"]

    def _on_input_dir_edited(self, text: str) -> None:
        # Manual edit (not Browse): if the input dir no longer matches the
        # loaded dataset's input, invalidate the dataset (Run would otherwise
        # process the old dataset's files while writing to the new directory).
        if self._loaded_dataset_dir is not None and text != self._loaded_dataset_dir:
            self._on_input_changed()

    def _load_dataset(self) -> None:
        if self._running:
            return
        path = Path(self._input_dir.text())
        if not path.exists():
            QMessageBox.warning(self, "Invalid Directory", "Input directory does not exist.")
            return

        # Use current GUI values. session.yml is read once at Browse time;
        # the user's manual edits take precedence over any saved values.
        image_pattern = self.get_image_pattern()
        mask_pattern = self.get_mask_pattern()
        raw_pattern = self.get_image_subdir_pattern()

        try:
            img_pat = re.compile(image_pattern) if image_pattern else None
            msk_pat = re.compile(mask_pattern) if mask_pattern else None
        except re.error as e:
            QMessageBox.warning(self, "Invalid Pattern", f"Regex compile error:\n{e}")
            return

        # Phase 1: show modal progress dialog and construct ImageDataset off
        # the UI thread. ImageDataset.__init__ scans files + builds metadata
        # — synchronous and slow for large datasets. Moving it off the UI
        # thread keeps the dialog responsive (no "not responding" ghost).
        # The input dir IS the dataset dir: all artifacts land there.
        root = path

        self._loader_dialog = QProgressDialog("Loading dataset...", None, 0, 0, self)
        self._loader_dialog.setWindowModality(Qt.WindowModal)
        self._loader_dialog.setMinimumDuration(0)
        self._loader_dialog.setCancelButton(None)
        self._loader_dialog.setRange(0, 0)  # indeterminate
        self._loader_dialog.show()

        # Stash pending params for phase 2 (UI population)
        self._loader_pending = (image_pattern, mask_pattern, raw_pattern, str(path))

        self._loader_worker = DatasetLoadWorker(
            root, img_pat, msk_pat, raw_pattern,
        )
        self._loader_worker.finished.connect(self._on_dataset_loaded)
        self._loader_worker.error.connect(self._on_dataset_load_error)
        self._loader_worker.start()

    def _on_dataset_loaded(self, ds, clone) -> None:
        """Phase 2: populate UI after background ImageDataset construction."""
        image_pattern, mask_pattern, raw_pattern, input_path = self._loader_pending
        dialog = getattr(self, "_loader_dialog", None)
        worker = getattr(self, "_loader_worker", None)
        self._loader_pending = None
        self._loader_dialog = None
        self._loader_worker = None

        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        if worker is not None:
            worker.deleteLater()

        try:
            self._state.dataset = ds
            self._state.original_dataset = clone

            # Re-apply filters restored from session.yml (Browse defers them
            # because the filter panel needs the dataset's metadata columns);
            # otherwise start with a clean filter state.
            if self._pending_filters:
                self._filter_panel.load_from_settings(
                    {"filter": {"filters": self._pending_filters}})
                self._pending_filters = None
                if hasattr(self._filter_panel, "_apply_filters"):
                    self._filter_panel._apply_filters()
            else:
                self._filter_panel._reset_filters()

            self.refresh_step_panels(ds.intensity_colnames)
            self._basic_panel.set_preview_channels(ds.intensity_colnames)

            # Initialize the filter-edit caches: otherwise the first filter
            # edit sees unset (None) values, triggers a full panel refresh and
            # wipes the user's channel selections on every panel.
            self._last_filter_channels = tuple(ds.intensity_colnames)
            self._last_filter_masks = tuple(ds.mask_colnames)

            # max_value is read from config and always trusted; a mismatch
            # against the dataset dtype is a non-blocking warning (§Q12) so the
            # user can check the yml — the configured value is still used.
            try:
                mismatches = self._inference_panel.max_value_mismatches(ds.img_dtype)
            except Exception:
                mismatches = []
            if mismatches:
                QMessageBox.warning(
                    self, "max_value Mismatch",
                    "Configured max_value does not match the dataset dtype:\n\n"
                    + "\n".join(mismatches)
                    + "\n\nThe configured value is used as-is — check the "
                      "config/session.yml if unexpected.")
            self._inference_panel.set_dataset_dtype(ds.img_dtype)

            self._update_dataset_info(ds)

            # Mirror the terminal log into the dataset directory (one log per
            # dataset; a later load retargets the file handler).
            set_log_file(Path(input_path) / "microProfiler.log")
            logging.getLogger("microProfiler").info(
                f"Dataset loaded: {len(ds)} rows, channels={ds.intensity_colnames}"
            )

            # Load Dataset writes the user's current patterns to session.yml
            # so the next session restores them. Only patterns are written
            # here — step params / applied_steps / filter are written by the
            # Run/Apply action buttons via PipelineController._save_session_yml.
            try:
                sf = SessionFile(input_path)
                sf.set_patterns(
                    image_pattern=image_pattern or "",
                    mask_pattern=mask_pattern or "",
                    image_subdir_pattern=raw_pattern or "",
                )
            except Exception:
                logging.getLogger("microProfiler").warning(
                    "Failed to persist patterns to session.yml", exc_info=True)

            self._loaded_dataset_dir = input_path
            self._update_window_title()
            self._update_tab_status()
        except Exception as e:
            self._state.dataset = None
            self._state.original_dataset = None
            self._loaded_dataset_dir = None
            self._update_window_title()
            self._update_tab_status()
            QMessageBox.warning(self, "Load Failed", f"Could not load dataset:\n{e}")

    def _on_dataset_load_error(self, msg: str) -> None:
        """Phase 2 error: clean up loader state and notify the user."""
        dialog = getattr(self, "_loader_dialog", None)
        worker = getattr(self, "_loader_worker", None)
        self._loader_pending = None
        self._loader_dialog = None
        self._loader_worker = None

        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        if worker is not None:
            worker.deleteLater()

        self._state.dataset = None
        self._state.original_dataset = None
        self._loaded_dataset_dir = None
        self._update_window_title()
        self._update_tab_status()
        # A failed load must not leave a Browse-restored config pending —
        # populating a later dataset with stale channel selections is wrong.
        img_panel = getattr(self, "_image_profile_panel", None)
        if img_panel is not None and hasattr(img_panel, "_pending_settings"):
            del img_panel._pending_settings
        QMessageBox.warning(self, "Load Failed", f"Could not load dataset:\n{msg}")

    def _on_input_changed(self):
        # Reset dataset state when the input directory changes.
        # session.yml is read at Browse time, not here.
        self._state.dataset = None
        self._state.original_dataset = None
        self._loaded_dataset_dir = None
        self._pending_filters = None
        self._update_window_title()
        # Structured configs restored from a previous directory must not
        # leak into the next dataset's object-profile blocks.
        obj_panel = getattr(self, "_object_profile_panel", None)
        if obj_panel is not None:
            obj_panel._pending_block_configs = []
            obj_panel._restore_active = False
        inf_panel = getattr(self, "_inference_panel", None)
        if inf_panel is not None:
            inf_panel._pending_block_configs = []
            inf_panel._restore_active = False
        img_panel = getattr(self, "_image_profile_panel", None)
        if img_panel is not None and hasattr(img_panel, "_pending_settings"):
            del img_panel._pending_settings
        self._clear_dataset_info()
        if hasattr(self, '_filter_panel') and self._filter_panel is not None:
            self._filter_panel._reset_filters()
        self._update_tab_status()

    # ── Compact widths ──────────────────────────────────────────────────

    def _apply_compact_widths(self) -> None:
        for panel in self._all_step_panels:
            BaseStepPanel._compact_block(panel)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _update_window_title(self) -> None:
        if self._loaded_dataset_dir:
            self.setWindowTitle(f"microProfiler — {self._loaded_dataset_dir}")
        else:
            self.setWindowTitle("microProfiler")

    # ── Window state (~/.micromax) ───────────────────────────────────────

    def _restore_window_size(self) -> None:
        """Start at the last closed window size (fallback: WINDOW_SIZE)."""
        window = get_user_defaults().get("window") or {}
        try:
            width = int(window.get("width", 0))
            height = int(window.get("height", 0))
        except (TypeError, ValueError):
            width = height = 0
        if width >= WINDOW_SIZE[0] and height >= WINDOW_SIZE[1]:
            self.resize(width, height)
        else:
            self.resize(*WINDOW_SIZE)

    def _save_window_size(self) -> None:
        """Remember the current window size for the next start.

        A maximized window stores its normal (restored) size so the next
        start comes up with a sensible non-maximized geometry.
        """
        size = self.normalGeometry().size() if self.isMaximized() else self.size()
        update_user_defaults(
            "window", {"width": int(size.width()), "height": int(size.height())})

    def _update_tab_status(self) -> None:
        ds = self._state.dataset
        has_ds = ds is not None and len(ds) > 0
        self._run_pre_btn.setEnabled(has_ds)
        self._run_seg_btn.setEnabled(has_ds)
        self._run_prof_btn.setEnabled(has_ds)
        self._run_infer_btn.setEnabled(has_ds)

    def _on_filter_changed(self) -> None:
        ds = self._state.dataset
        if ds is not None:
            self._update_dataset_info(ds)
            ch = tuple(ds.intensity_colnames)
            masks = tuple(ds.mask_colnames)
            if ch != getattr(self, "_last_filter_channels", None):
                self.refresh_step_panels(ds.intensity_colnames)
                self._last_filter_channels = ch
            if masks != getattr(self, "_last_filter_masks", None):
                self._ctrl._sync_seg_masks_to_profiling()
                self._last_filter_masks = masks

    # ── Dataset info display ──────────────────────────────────────────────

    def _update_dataset_info(self, ds) -> None:
        self._dataset_info_label.setVisible(True)
        self._dataset_info_label.setProperty("class", "")
        self._dataset_info_label.style().polish(self._dataset_info_label)
        n = len(ds)
        ch = ", ".join(ds.intensity_colnames) if ds.intensity_colnames else "—"
        shape = ds.img_shape
        dtype = ds.img_dtype
        masks = ", ".join(ds.mask_colnames) if ds.mask_colnames else "—"

        lines = [f"Image groups: {n}"]
        meta = ds.metadata
        if meta is not None:
            structural = ["well", "row", "col", "field", "stack", "timepoint", "channel", "tile"]
            for col in structural:
                if col in meta.columns:
                    n_unique = meta[col].nunique()
                    vals = meta[col].unique()[:10]
                    val_str = ", ".join(str(v) for v in vals)
                    if meta[col].nunique() > 10:
                        val_str += "…"
                    lines.append(f"{col}: {n_unique} ({val_str})")
            for extra_col in ds.schema.extra_cols:
                if extra_col in meta.columns:
                    n_unique = meta[extra_col].nunique()
                    vals = meta[extra_col].unique()[:10]
                    val_str = ", ".join(str(v) for v in vals)
                    if n_unique > 10:
                        val_str += "…"
                    lines.append(f"{extra_col}: {n_unique} ({val_str})")

        lines.append(f"Channels: {ch}")
        if shape:
            lines.append(f"Dimensions: {shape[0]}×{shape[1]}")
        if dtype is not None:
            lines.append(f"Data type: {dtype}")
        if ds.mask_colnames:
            lines.append("")
            lines.append(f"Masks: {masks}")
        self._dataset_info_label.setText("\n".join(lines))

    def _clear_dataset_info(self) -> None:
        self._dataset_info_label.setVisible(False)
        self._dataset_info_label.setText("Load a dataset to see dataset information.")
        self._dataset_info_label.setProperty("class", "placeholder")
        self._dataset_info_label.style().polish(self._dataset_info_label)

    # ── Reset ───────────────────────────────────────────────────────────

    def _reset_all(self) -> None:
        if self._running:
            return
        for step in self._all_step_panels:
            step.setChecked(False)
        self._state.dataset = None
        self._state.original_dataset = None
        self._ctrl._random_row_idx = None
        self._last_filter_channels = None
        self._last_filter_masks = None
        self._loaded_dataset_dir = None
        self._pending_filters = None
        obj_panel = getattr(self, "_object_profile_panel", None)
        if obj_panel is not None:
            obj_panel._pending_block_configs = []
        inf_panel = getattr(self, "_inference_panel", None)
        if inf_panel is not None:
            inf_panel._pending_block_configs = []
        self._filter_panel._reset_filters()
        self._basic_panel._clear_preview()
        self._basic_panel.set_preview_channels([])
        self._segment_panel.clear_preview()
        self.refresh_step_panels([], [])
        self._input_dir.clear()
        self._clear_dataset_info()
        self._update_window_title()
        self._update_tab_status()

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._input_browse.setEnabled(not running)
        self._load_dataset_btn.setEnabled(not running)
        self._reset_btn.setEnabled(not running)
        for step in self._all_step_panels:
            step.setEnabled(not running)
        self._filter_panel.setEnabled(not running)
        self._sidebar.set_action_buttons_enabled(not running)
        self._sidebar.set_cancel_visible(running)

    def closeEvent(self, event):
        # Remember the window size and complete the shared GUI config before
        # any shutdown work so a slow/cancelled worker never prevents the
        # geometry/preferences from being persisted.
        self._save_window_size()
        ensure_user_defaults(
            "inference", {"model": None, "reducer": None, "cluster": None})
        # Cooperative shutdown: signal every worker's cancel flag and wait.
        # Threads are never terminated while they execute Python code.
        workers = [self._ctrl._worker, self._ctrl._preview_worker]
        loader = getattr(self, "_loader_worker", None)
        if loader is not None:
            workers.append(loader)
        for w in workers:
            if w is None or not hasattr(w, "_thread"):
                continue
            # A finished worker's QThread is scheduled for deletion via
            # deleteLater; its C++ wrapper may already be gone, in which case
            # isRunning() raises RuntimeError. Guard so closing the window
            # after a completed run doesn't crash.
            try:
                thread = w._thread
                if thread is None or not thread.isRunning():
                    continue
            except RuntimeError:
                continue
            if hasattr(w, "_cancel_event"):
                w._cancel_event.set()
            try:
                thread.quit()
                thread.wait(5000)
            except RuntimeError:
                pass
        super().closeEvent(event)
