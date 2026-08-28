from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml
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
    normalize_null_strings,
    DEFAULT_IMAGE_PATTERN,
    DEFAULT_MASK_PATTERN,
    DEFAULT_IMAGE_SUBDIR_PATTERN,
)
from microProfiler.gui.pipeline_controller import PipelineController
from microProfiler.gui.state import PipelineState
from microProfiler.gui.workers.preview_worker import PreviewWorker
from microProfiler.gui.workers.dataset_load_worker import DatasetLoadWorker
from microProfiler.gui.sidebar import Sidebar
from microProfiler.gui.panels import (
    BaSiCStepPanel, FilterPanel,
    ImageProfilingStepPanel, InferenceStepPanel, ObjectProfilingStepPanel,
    ResizeStepPanel, SegmentStepPanel, TileStepPanel, ZProjectStepPanel,
)
from microProfiler.gui.panels.base_step_panel import BaseStepPanel
from microProfiler.gui.image_widgets import ImageViewer
from microProfiler.gui.dpi import dp
from microProfiler.log_utils import setup_logging


class WindowWheelFilter(QObject):

    def __init__(self, window: QMainWindow):
        super().__init__()
        self._window = window

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.Wheel:
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


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._state = PipelineState()
        self._output_manually_set = False
        self._loaded_dataset_dir = None
        self._loaded_input_dir = None
        self._pending_filters = None

        self.setWindowTitle("microProfiler")
        self.resize(1200, 800)
        self.setMinimumSize(1200, 800)
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

    def get_output_dir(self) -> str:
        return self._output_dir.text()

    def output_path(self) -> Path:
        return self._output_path()

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

    def progress_reset(self) -> None:
        self._progress_label.setVisible(False)

    def progress_finished(self) -> None:
        self._progress_label.setVisible(False)

    def progress_show_status(self, message: str) -> None:
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
        input_layout = QVBoxLayout(input_page)
        input_layout.setContentsMargins(4, 4, 4, 4)

        input_group = QGroupBox("Input")
        input_form = QFormLayout(input_group)
        input_form.setLabelAlignment(Qt.AlignLeft)
        input_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._input_dir = QLineEdit()
        self._input_browse = QPushButton("Browse...")
        self._input_browse.setProperty("class", "secondary")
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_dir, 1)
        input_row.addWidget(self._input_browse)
        input_form.addRow("Input dir:", input_row)

        self._output_dir = QLineEdit()
        self._output_browse = QPushButton("Browse...")
        self._output_browse.setProperty("class", "secondary")
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir, 1)
        output_row.addWidget(self._output_browse)
        input_form.addRow("Output dir:", output_row)

        self._load_dataset_btn = QPushButton("Load Dataset")
        self._load_dataset_btn.setProperty("class", "primary")
        self._load_dataset_btn.setFixedHeight(dp(32))
        format_row = QHBoxLayout()
        format_row.addStretch()
        format_row.addWidget(self._load_dataset_btn)
        format_row.addStretch()
        input_form.addRow(format_row)

        img_row = QHBoxLayout()
        self._custom_image_pattern = QLineEdit()
        self._custom_image_pattern.setPlaceholderText(r"(?P<well>[A-Z]\d+)_f(?P<field>\d+)_ch(?P<channel>\d+)\.tif")
        img_row.addWidget(self._custom_image_pattern, 1)
        input_form.addRow("Image pattern:", img_row)

        mask_row = QHBoxLayout()
        self._custom_mask_pattern = QLineEdit()
        self._custom_mask_pattern.setPlaceholderText(r".*_masks_(?P<mask_name>.+)\.png")
        mask_row.addWidget(self._custom_mask_pattern, 1)
        input_form.addRow("Mask pattern:", mask_row)

        search_row = QHBoxLayout()
        self._custom_image_subdir_pattern = QLineEdit()
        self._custom_image_subdir_pattern.setPlaceholderText("Leave empty to search directly")
        search_row.addWidget(self._custom_image_subdir_pattern, 1)
        input_form.addRow("Image subdir pattern:", search_row)

        self._dataset_info_label = QLabel("")
        self._dataset_info_label.setWordWrap(True)
        self._dataset_info_label.setProperty("class", "placeholder")
        self._dataset_info_label.setVisible(False)
        input_form.addRow(self._dataset_info_label)

        input_layout.addWidget(input_group)

        # Filter section inside input tab (no outer QGroupBox wrapper)
        input_layout.addWidget(self._filter_panel)

        input_layout.addStretch()
        self._stack.addWidget(input_page)

        # ── Page 1: Pre-process ────────────────────────────────────────
        pre_page = QWidget()
        pre_layout = QVBoxLayout(pre_page)
        pre_layout.setContentsMargins(4, 4, 4, 4)

        self._pre_scroll = QScrollArea()
        self._pre_scroll.setWidgetResizable(True)
        pre_scroll_inner = QWidget()
        pre_scroll_layout = QVBoxLayout(pre_scroll_inner)
        pre_scroll_layout.setContentsMargins(0, 0, 0, 0)
        pre_scroll_layout.setSpacing(12)
        for p in self._preprocessing_steps:
            pre_scroll_layout.addWidget(p)
        self._run_pre_btn = QPushButton("Run Preprocessing")
        self._run_pre_btn.setProperty("class", "primary")
        self._run_pre_btn.setFixedHeight(dp(32))
        pre_scroll_layout.addWidget(self._run_pre_btn)
        pre_scroll_layout.addStretch()
        self._pre_scroll.setWidget(pre_scroll_inner)
        pre_layout.addWidget(self._pre_scroll, 1)
        self._stack.addWidget(pre_page)

        # ── Page 2: Segmentation ───────────────────────────────────────
        seg_page = QWidget()
        seg_layout = QVBoxLayout(seg_page)
        seg_layout.setContentsMargins(4, 4, 4, 4)

        self._seg_scroll = QScrollArea()
        self._seg_scroll.setWidgetResizable(True)
        seg_scroll_inner = QWidget()
        seg_scroll_layout = QVBoxLayout(seg_scroll_inner)
        seg_scroll_layout.setContentsMargins(0, 0, 0, 0)
        seg_scroll_layout.setSpacing(12)
        seg_scroll_layout.addWidget(self._segment_panel)
        self._run_seg_btn = QPushButton("Run Segmentation")
        self._run_seg_btn.setProperty("class", "primary")
        self._run_seg_btn.setFixedHeight(dp(32))
        seg_scroll_layout.addWidget(self._run_seg_btn)
        seg_scroll_layout.addStretch()
        self._seg_scroll.setWidget(seg_scroll_inner)
        seg_layout.addWidget(self._seg_scroll, 1)
        self._stack.addWidget(seg_page)

        # ── Page 3: Profiling ──────────────────────────────────────────
        prof_page = QWidget()
        prof_layout = QVBoxLayout(prof_page)
        prof_layout.setContentsMargins(4, 4, 4, 4)

        self._prof_scroll = QScrollArea()
        self._prof_scroll.setWidgetResizable(True)
        prof_scroll_inner = QWidget()
        prof_scroll_layout = QVBoxLayout(prof_scroll_inner)
        prof_scroll_layout.setContentsMargins(0, 0, 0, 0)
        prof_scroll_layout.setSpacing(12)
        prof_scroll_layout.addWidget(self._image_profile_panel)
        prof_scroll_layout.addWidget(self._object_profile_panel)
        self._run_prof_btn = QPushButton("Run Profiling")
        self._run_prof_btn.setProperty("class", "primary")
        self._run_prof_btn.setFixedHeight(dp(32))
        prof_scroll_layout.addWidget(self._run_prof_btn)
        prof_scroll_layout.addStretch()
        self._prof_scroll.setWidget(prof_scroll_inner)
        prof_layout.addWidget(self._prof_scroll, 1)
        self._stack.addWidget(prof_page)

        # ── Page 4: Inference ──────────────────────────────────────────
        infer_page = QWidget()
        infer_layout = QVBoxLayout(infer_page)
        infer_layout.setContentsMargins(4, 4, 4, 4)

        self._infer_scroll = QScrollArea()
        self._infer_scroll.setWidgetResizable(True)
        infer_scroll_inner = QWidget()
        infer_scroll_layout = QVBoxLayout(infer_scroll_inner)
        infer_scroll_layout.setContentsMargins(0, 0, 0, 0)
        infer_scroll_layout.setSpacing(12)
        infer_scroll_layout.addWidget(self._inference_panel)
        self._run_infer_btn = QPushButton("Run Inference")
        self._run_infer_btn.setProperty("class", "primary")
        self._run_infer_btn.setFixedHeight(dp(32))
        infer_scroll_layout.addWidget(self._run_infer_btn)
        infer_scroll_layout.addStretch()
        self._infer_scroll.setWidget(infer_scroll_inner)
        infer_layout.addWidget(self._infer_scroll, 1)
        self._stack.addWidget(infer_page)

        # ── Progress bar at bottom ─────────────────────────────────────
        self._progress_label = QLabel("")
        self._progress_label.setVisible(False)
        self._progress_label.setFixedHeight(dp(24))
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

    # ── Signal connections ────────────────────────────────────────────────

    def _connect_signals(self):
        self._sidebar.navigation_changed.connect(self._on_navigation_changed)
        self._sidebar.load_config_clicked.connect(self._load_config)
        self._sidebar.reset_all_clicked.connect(self._reset_all)
        self._sidebar.thread_count_changed.connect(self._on_thread_count_changed)
        self._sidebar.display_range_changed.connect(self._on_vmin_vmax_changed)
        self._input_browse.clicked.connect(self._browse_input)
        self._output_browse.clicked.connect(self._browse_output)
        self._input_dir.textChanged.connect(self._on_input_dir_edited)
        self._output_dir.textChanged.connect(self._on_output_changed)
        self._load_dataset_btn.clicked.connect(self._load_dataset)
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
        if not self._output_manually_set:
            self._output_dir.blockSignals(True)
            self._output_dir.setText(path)
            self._output_dir.blockSignals(False)
        output_path = self._output_path()
        # session.yml is read only on the FIRST browse of a given directory
        # (pre-fills GUI fields); re-browsing the SAME directory preserves the
        # user's GUI edits and the loaded dataset — session.yml is never
        # re-read into the GUI (§3.16).
        if output_path == self._loaded_dataset_dir:
            return
        self._on_input_changed()

        sf = SessionFile(output_path)
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

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir.setText(path)
            self._output_manually_set = True

    def _on_output_changed(self) -> None:
        self._output_manually_set = True
        # Manual edit (not Browse): if the output dir no longer matches the
        # loaded dataset's dir, invalidate the dataset so a Run can't process
        # stale data into the new directory.
        if self._loaded_dataset_dir is not None and self._output_path() != self._loaded_dataset_dir:
            self._on_input_changed()

    def _on_input_dir_edited(self, text: str) -> None:
        # Manual edit (not Browse): if the input dir no longer matches the
        # loaded dataset's input, invalidate the dataset (Run would otherwise
        # process the old dataset's files while writing to the new layout).
        if self._loaded_input_dir is not None and text != self._loaded_input_dir:
            self._on_input_changed()

    def _is_converted(self, output_path: Path) -> bool:
        """True when the output dir holds a converted layout (subdir with images).

        Uses the configured image_subdir_pattern (default 'images' — the old
        hardcoded 'image' never matched the default layout on case-sensitive
        filesystems); falls back to a case-insensitive subdir match.
        """
        subdir = (self.get_image_subdir_pattern() or "images").strip("/\\")
        candidates = [output_path / subdir]
        if output_path.is_dir():
            candidates += [
                p for p in output_path.iterdir()
                if p.is_dir() and p.name.lower() == subdir.lower()
            ]
        for image_subdir in candidates:
            if image_subdir.is_dir() and bool(
                list(image_subdir.glob("*.tiff")) or list(image_subdir.glob("*.tif"))
                or list(image_subdir.glob("*.png")) or list(image_subdir.glob("*.jpg"))
            ):
                return True
        return False

    def _load_dataset(self) -> None:
        if self._running:
            return
        path = Path(self._input_dir.text())
        if not path.exists():
            QMessageBox.warning(self, "Invalid Directory", "Input directory does not exist.")
            return

        output_path = self._output_path()
        is_converted = self._is_converted(output_path)

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
        root = output_path if is_converted else path

        self._loader_dialog = QProgressDialog("Loading dataset...", None, 0, 0, self)
        self._loader_dialog.setWindowModality(Qt.WindowModal)
        self._loader_dialog.setMinimumDuration(0)
        self._loader_dialog.setCancelButton(None)
        self._loader_dialog.setRange(0, 0)  # indeterminate
        self._loader_dialog.show()

        # Stash pending params for phase 2 (UI population)
        self._loader_pending = (output_path, is_converted, image_pattern,
                                mask_pattern, raw_pattern, str(path))

        self._loader_worker = DatasetLoadWorker(
            root, img_pat, msk_pat, raw_pattern,
        )
        self._loader_worker.finished.connect(self._on_dataset_loaded)
        self._loader_worker.error.connect(self._on_dataset_load_error)
        self._loader_worker.start()

    def _on_dataset_loaded(self, ds, clone) -> None:
        """Phase 2: populate UI after background ImageDataset construction."""
        output_path, is_converted, image_pattern, mask_pattern, raw_pattern, input_path = self._loader_pending
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
            self._loaded_input_dir = input_path

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

            logging.getLogger("microProfiler").info(
                f"{'Converted' if is_converted else 'Raw'} dataset loaded: {len(ds)} rows, channels={ds.intensity_colnames}"
            )

            # Load Dataset writes the user's current patterns to session.yml
            # so the next session restores them. Only patterns are written
            # here — step params / applied_steps / filter are written by the
            # Run/Apply action buttons via PipelineController._save_session_yml.
            try:
                sf = SessionFile(output_path)
                sf.set_patterns(
                    image_pattern=image_pattern or "",
                    mask_pattern=mask_pattern or "",
                    image_subdir_pattern=raw_pattern or "",
                )
            except Exception:
                logging.getLogger("microProfiler").warning(
                    "Failed to persist patterns to session.yml", exc_info=True)

            self._loaded_dataset_dir = output_path
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
        output_path = self._loader_pending[0] if self._loader_pending else None
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
        QMessageBox.warning(self, "Load Failed", f"Could not load dataset:\n{msg}")

    def _on_input_changed(self):
        # Reset dataset state when the input/output directory changes.
        # session.yml is read at Browse time, not here.
        self._state.dataset = None
        self._state.original_dataset = None
        self._loaded_dataset_dir = None
        self._loaded_input_dir = None
        self._pending_filters = None
        self._update_window_title()
        # Structured configs restored from a previous directory must not
        # leak into the next dataset's object-profile blocks.
        obj_panel = getattr(self, "_object_profile_panel", None)
        if obj_panel is not None:
            obj_panel._pending_block_configs = []
        inf_panel = getattr(self, "_inference_panel", None)
        if inf_panel is not None:
            inf_panel._pending_block_configs = []
        self._clear_dataset_info()
        if hasattr(self, '_filter_panel') and self._filter_panel is not None:
            self._filter_panel._reset_filters()
        self._update_tab_status()

    # ── Compact widths ──────────────────────────────────────────────────

    def _apply_compact_widths(self) -> None:
        for panel in self._all_step_panels:
            BaseStepPanel._compact_block(panel)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _output_path(self) -> Path:
        txt = self._output_dir.text() or self._input_dir.text()
        return Path(txt)

    def _update_window_title(self) -> None:
        if self._loaded_dataset_dir:
            self.setWindowTitle(f"microProfiler — {self._loaded_dataset_dir}")
        else:
            self.setWindowTitle("microProfiler")

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

    # ── Config load / reset ─────────────────────────────────────────────

    def _load_config(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File", "",
            "YAML files (*.yml *.yaml);;All files (*)",
        )
        if not file_path:
            return
        try:
            with open(file_path, encoding="utf-8") as f:
                data = normalize_null_strings(yaml.safe_load(f) or {})
        except Exception as e:
            logging.getLogger("microProfiler").error("Failed to load config: %s", e)
            return
        params = {k: v for k, v in data.items() if not k.startswith("_")}
        # Top-level patterns + filter are part of the PipelineConfig schema;
        # apply them to the GUI too (not only the step sections).
        if data.get("image_pattern"):
            self.set_image_pattern(data["image_pattern"])
        if data.get("mask_pattern"):
            self.set_mask_pattern(data["mask_pattern"])
        if data.get("image_subdir_pattern"):
            self._custom_image_subdir_pattern.setText(data["image_subdir_pattern"])
        if data.get("filter"):
            self._filter_panel.load_from_settings({"filter": {"filters": data["filter"]}})
        for step in self._all_step_panels:
            step.from_config(params.get(step.step_name, {}))
        if self._state.dataset is not None:
            self.refresh_step_panels(
                self._state.dataset.intensity_colnames,
                self._state.dataset.mask_colnames,
            )
        applied = data.get("applied_steps", [])
        for step in self._all_step_panels:
            if step.step_name in applied:
                step.setChecked(True)
        logging.getLogger("microProfiler").info("Config loaded from %s", file_path)

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
        self._loaded_input_dir = None
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
        self._output_dir.clear()
        self._output_manually_set = False
        self._clear_dataset_info()
        self._update_window_title()
        self._update_tab_status()
        logging.getLogger("microProfiler").info("Pipeline reset complete.")

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._input_browse.setEnabled(not running)
        self._output_browse.setEnabled(not running)
        self._load_dataset_btn.setEnabled(not running)
        for step in self._all_step_panels:
            step.setEnabled(not running)
        self._filter_panel.setEnabled(not running)
        self._sidebar.set_action_buttons_enabled(not running)
        self._sidebar.set_cancel_visible(running)

    def closeEvent(self, event):
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
