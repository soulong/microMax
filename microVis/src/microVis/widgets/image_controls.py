from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import CONTRAST_METHODS
from microVis.widgets._event_filter import NoScrollComboBox, NoScrollDoubleSpinBox, NoScrollSlider
from microVis.widgets.channel_controls import ChannelControls


class _MultiSelectCombo(QWidget):
    """Compact multi-select with checkboxes in a horizontal layout + Select All / Clear."""

    selection_changed = Signal()

    def __init__(self, label: str, items: list[str], checked_first: bool = False,
                 parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Label + buttons row
        header = QHBoxLayout()
        header.setSpacing(4)
        header.addWidget(QLabel(label))

        sel_all = QPushButton("All")
        sel_all.setProperty("class", "secondary")
        sel_all.setFixedSize(42, 18)
        sel_all.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
        sel_all.clicked.connect(lambda: self.set_all_checked(True))
        header.addWidget(sel_all)

        clear_btn = QPushButton("Clear")
        clear_btn.setProperty("class", "secondary")
        clear_btn.setFixedSize(48, 18)
        clear_btn.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
        clear_btn.clicked.connect(lambda: self.set_all_checked(False))
        header.addWidget(clear_btn)
        header.addStretch()
        layout.addLayout(header)

        # Checkboxes in a scrollable horizontal row
        checks_scroll = QScrollArea()
        checks_scroll.setWidgetResizable(True)
        checks_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        checks_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        checks_scroll.setMaximumHeight(24)
        checks_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        checks_inner = QWidget()
        self._checks_layout = QHBoxLayout(checks_inner)
        self._checks_layout.setSpacing(4)
        self._checks_layout.setContentsMargins(0, 0, 0, 0)
        self._checkboxes: dict[str, QCheckBox] = {}
        for i, item_text in enumerate(items):
            cb = QCheckBox(item_text)
            cb.setStyleSheet("QCheckBox { font-size: 7pt; spacing: 2px; } QCheckBox::indicator { width: 12px; height: 12px; }")
            cb.setChecked(i == 0 if checked_first else True)
            cb.toggled.connect(lambda: self.selection_changed.emit())
            self._checks_layout.addWidget(cb)
            self._checkboxes[item_text] = cb
        self._checks_layout.addStretch()

        checks_scroll.setWidget(checks_inner)
        layout.addWidget(checks_scroll)

    def get_options(self) -> list[str]:
        return list(self._checkboxes.keys())

    def get_selected(self) -> list[str]:
        return [t for t, cb in self._checkboxes.items() if cb.isChecked()]

    def set_all_checked(self, checked: bool) -> None:
        for cb in self._checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.selection_changed.emit()


class ImageControls(QScrollArea):
    """Left sidebar controls for the image viewer."""

    auto_all_clicked = Signal()
    auto_range_changed = Signal()
    image_size_changed = Signal()
    channel_config_changed = Signal()
    reset_requested = Signal()
    sort_mode_changed = Signal()

    # Class labeling signals
    object_mask_changed = Signal(str)       # selected object mask name ("" for None)
    label_class_added = Signal(str)         # class_name
    label_class_removed = Signal(str)       # class_name removed
    label_class_selection_changed = Signal()  # selected classes changed
    label_write_clicked = Signal()          # write to db requested

    # Object export signals
    export_clicked = Signal()               # export button pressed

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(260)
        self.setMaximumWidth(320)

        container = QWidget()
        container.setStyleSheet("""
            QComboBox, QDoubleSpinBox, QSpinBox, QSlider {
                min-height: 18px;
                max-height: 22px;
                font-size: 8pt;
                padding: 2px 3px;
                min-width: 0;
            }
            QLabel {
                font-size: 8pt;
            }
            QPushButton {
                font-size: 9pt;
                padding: 2px 6px;
            }
            QGroupBox {
                font-size: 8pt;
                font-weight: bold;
                color: #5a8a9a;
                border: 1px solid #3a3a4a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)

        # ── Image Filters ──
        grp_filters = QGroupBox("Image Filters")
        filter_layout = QVBoxLayout(grp_filters)
        filter_layout.setSpacing(2)
        self._filter_container = filter_layout
        self._fields_widget: _MultiSelectCombo | None = None
        self._stacks_widget: _MultiSelectCombo | None = None
        self._timepoints_widget: _MultiSelectCombo | None = None
        self._extra_widgets: dict[str, _MultiSelectCombo] = {}
        self._layout.addWidget(grp_filters)

        # ── Channel Setting ──
        grp_channels = QGroupBox("Channel Setting")
        ch_layout = QVBoxLayout(grp_channels)
        ch_layout.setSpacing(2)

        # Channel controls container (dynamic, cleared on set_channels)
        self._ch_container = QVBoxLayout()
        self._ch_container.setSpacing(2)
        ch_layout.addLayout(self._ch_container)
        self._channel_widgets: dict[str, ChannelControls] = {}

        def _row(label_text, widget, target_layout=ch_layout):
            r = QHBoxLayout()
            r.setSpacing(4)
            r.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(60)
            r.addWidget(lbl)
            r.addWidget(widget, stretch=1)
            target_layout.addLayout(r)

        # Low / High
        lowhigh_row = QHBoxLayout()
        lowhigh_row.setSpacing(4)
        lbl_lo = QLabel("Low")
        lbl_lo.setFixedWidth(28)
        lowhigh_row.addWidget(lbl_lo)
        self._auto_low = NoScrollDoubleSpinBox()
        self._auto_low.setRange(0.0, 100.0)
        self._auto_low.setValue(0.1)
        self._auto_low.setDecimals(2)
        self._auto_low.setButtonSymbols(QDoubleSpinBox.NoButtons)
        lowhigh_row.addWidget(self._auto_low, stretch=1)
        lbl_hi = QLabel("High")
        lbl_hi.setFixedWidth(28)
        lowhigh_row.addWidget(lbl_hi)
        self._auto_high = NoScrollDoubleSpinBox()
        self._auto_high.setRange(0.0, 100.0)
        self._auto_high.setValue(99.9)
        self._auto_high.setDecimals(2)
        self._auto_high.setButtonSymbols(QDoubleSpinBox.NoButtons)
        lowhigh_row.addWidget(self._auto_high, stretch=1)
        ch_layout.addLayout(lowhigh_row)

        self._auto_low.valueChanged.connect(lambda: self.auto_range_changed.emit())
        self._auto_high.valueChanged.connect(lambda: self.auto_range_changed.emit())

        # Auto / Reset (centered)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        self._auto_all_btn = QPushButton("Auto")
        self._auto_all_btn.setProperty("class", "secondary")
        self._auto_all_btn.setFixedSize(64, 24)
        self._auto_all_btn.clicked.connect(self.auto_all_clicked)
        btn_row.addWidget(self._auto_all_btn)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setProperty("class", "secondary")
        self._reset_btn.setFixedSize(64, 24)
        self._reset_btn.clicked.connect(self.reset_requested)
        btn_row.addWidget(self._reset_btn)
        btn_row.addStretch()
        ch_layout.addLayout(btn_row)

        # Image size
        self._image_size = NoScrollDoubleSpinBox()
        self._image_size.setRange(50, 1000)
        self._image_size.setValue(256)
        self._image_size.setDecimals(0)
        self._image_size.setSingleStep(10)
        self._image_size.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self._image_size.valueChanged.connect(lambda: self.image_size_changed.emit())
        _row("Img size", self._image_size)

        # Contrast
        self._contrast = NoScrollComboBox()
        self._contrast.addItems(CONTRAST_METHODS)
        self._contrast.setCurrentIndex(2)
        _row("Transform", self._contrast)

        # Gamma slider (below Transform)
        self._gamma_slider = NoScrollSlider(Qt.Horizontal)
        self._gamma_slider.setRange(10, 300)
        self._gamma_slider.setSingleStep(10)
        self._gamma_slider.setPageStep(10)
        self._gamma_slider.setValue(100)
        self._gamma_slider.setVisible(True)
        self._gamma_slider_label = QLabel("Gamma: 1.00")
        self._gamma_slider_label.setProperty("class", "muted")
        self._gamma_slider_label.setVisible(True)
        self._gamma_slider.valueChanged.connect(
            lambda v: self._gamma_slider_label.setText(f"Gamma: {v / 100:.2f}")
        )
        ch_layout.addWidget(self._gamma_slider_label)
        ch_layout.addWidget(self._gamma_slider)

        # Sort mode
        sort_row = QHBoxLayout()
        sort_row.setSpacing(4)
        sort_row.setContentsMargins(0, 0, 0, 0)
        sort_lbl = QLabel("Group by")
        sort_lbl.setFixedWidth(60)
        sort_row.addWidget(sort_lbl)
        self._sort_by_col = QRadioButton("Col")
        self._sort_by_col.setChecked(True)
        self._sort_by_col.setStyleSheet("font-size: 8pt; spacing: 4px;")
        self._sort_by_col.toggled.connect(lambda: self.sort_mode_changed.emit())
        sort_row.addWidget(self._sort_by_col)
        self._sort_by_row = QRadioButton("Row")
        self._sort_by_row.setStyleSheet("font-size: 8pt; spacing: 4px;")
        self._sort_by_row.toggled.connect(lambda: self.sort_mode_changed.emit())
        sort_row.addWidget(self._sort_by_row)
        ch_layout.addLayout(sort_row)

        self._layout.addWidget(grp_channels)

        # ── Object Overlay ──
        grp_overlay = QGroupBox("Object Overlay")
        overlay_layout = QVBoxLayout(grp_overlay)
        overlay_layout.setSpacing(3)

        self._object_mask = NoScrollComboBox()
        self._object_mask.setEditable(True)
        self._object_mask.setInsertPolicy(QComboBox.NoInsert)
        self._object_mask.completer().setFilterMode(Qt.MatchContains)
        self._object_mask.completer().setCompletionMode(QCompleter.PopupCompletion)
        self._object_mask.addItem("None")
        self._object_mask.setCurrentIndex(0)
        self._object_mask.lineEdit().setPlaceholderText("None = no overlay")
        self._object_mask.currentTextChanged.connect(self._on_object_mask_changed)
        _row("Select object", self._object_mask, overlay_layout)

        self._overlay_col = NoScrollComboBox()
        self._overlay_col.setEditable(True)
        self._overlay_col.setInsertPolicy(QComboBox.NoInsert)
        self._overlay_col.completer().setFilterMode(Qt.MatchContains)
        self._overlay_col.completer().setCompletionMode(QCompleter.PopupCompletion)
        self._overlay_col.lineEdit().setPlaceholderText("Type to filter...")
        _row("Color by", self._overlay_col, overlay_layout)

        self._overlay_cmap = NoScrollComboBox()
        _row("Colors", self._overlay_cmap, overlay_layout)

        self._overlay_alpha = NoScrollSlider(Qt.Horizontal)
        self._overlay_alpha.setRange(0, 100)
        self._overlay_alpha.setValue(0)
        _row("Alpha", self._overlay_alpha, overlay_layout)

        clear_btn_row = QHBoxLayout()
        clear_btn_row.setSpacing(8)
        clear_btn_row.addStretch()
        self._overlay_clear_btn = QPushButton("Clear")
        self._overlay_clear_btn.setProperty("class", "secondary")
        self._overlay_clear_btn.setFixedSize(64, 24)
        self._overlay_clear_btn.clicked.connect(lambda: self._overlay_col.setCurrentIndex(0))
        clear_btn_row.addWidget(self._overlay_clear_btn)
        clear_btn_row.addStretch()
        overlay_layout.addLayout(clear_btn_row)

        self._layout.addWidget(grp_overlay)

        # ── Object Label ──
        grp_label = QGroupBox("Object Label")
        label_layout = QVBoxLayout(grp_label)
        label_layout.setSpacing(3)

        # Class name input + Add button
        class_input_row = QHBoxLayout()
        class_input_row.setSpacing(4)
        class_input_row.setContentsMargins(0, 0, 0, 0)
        lbl_class = QLabel("Class")
        lbl_class.setFixedWidth(60)
        class_input_row.addWidget(lbl_class)
        self._class_input = QLineEdit()
        self._class_input.setPlaceholderText("New class name...")
        self._class_input.setStyleSheet(
            "min-height: 18px; max-height: 22px; font-size: 8pt; padding: 2px 3px;"
        )
        self._class_input.returnPressed.connect(self._on_add_class)
        class_input_row.addWidget(self._class_input, stretch=1)
        self._add_class_btn = QPushButton("Add")
        self._add_class_btn.setProperty("class", "secondary")
        self._add_class_btn.setFixedSize(36, 20)
        self._add_class_btn.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
        self._add_class_btn.clicked.connect(self._on_add_class)
        class_input_row.addWidget(self._add_class_btn)
        self._remove_class_btn = QPushButton("Del")
        self._remove_class_btn.setProperty("class", "secondary")
        self._remove_class_btn.setFixedSize(36, 20)
        self._remove_class_btn.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
        self._remove_class_btn.clicked.connect(self._on_remove_class)
        class_input_row.addWidget(self._remove_class_btn)
        label_layout.addLayout(class_input_row)

        # Selected classes (multi-select dropdown)
        self._class_select_label = QLabel("Selected classes")
        self._class_select_label.setStyleSheet(
            "font-size: 8pt; color: #aaaaaa; padding-top: 2px;"
        )
        self._class_select_label.setVisible(False)
        label_layout.addWidget(self._class_select_label)

        self._class_select_scroll = QScrollArea()
        self._class_select_scroll.setWidgetResizable(True)
        self._class_select_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._class_select_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._class_select_scroll.setMaximumHeight(24)
        self._class_select_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self._class_select_scroll.setVisible(False)

        self._class_select_container = QWidget()
        self._class_select_layout = QHBoxLayout(self._class_select_container)
        self._class_select_layout.setSpacing(4)
        self._class_select_layout.setContentsMargins(0, 0, 0, 0)
        self._class_checkboxes: dict[str, QCheckBox] = {}
        self._class_select_layout.addStretch()
        self._class_select_scroll.setWidget(self._class_select_container)
        label_layout.addWidget(self._class_select_scroll)

        # Table name input
        table_row = QHBoxLayout()
        table_row.setSpacing(4)
        table_row.setContentsMargins(0, 0, 0, 0)
        lbl_table = QLabel("Table name")
        lbl_table.setFixedWidth(60)
        table_row.addWidget(lbl_table)
        self._label_table_name = QLineEdit()
        self._label_table_name.setStyleSheet(
            "min-height: 18px; max-height: 22px; font-size: 8pt; padding: 2px 3px;"
        )
        table_row.addWidget(self._label_table_name, stretch=1)
        label_layout.addLayout(table_row)

        # Save Label button (centered)
        write_btn_row = QHBoxLayout()
        write_btn_row.setSpacing(8)
        write_btn_row.addStretch()
        self._write_labels_btn = QPushButton("Write Label to DB")
        self._write_labels_btn.setProperty("class", "secondary")
        self._write_labels_btn.setFixedSize(150, 24)
        self._write_labels_btn.clicked.connect(self.label_write_clicked)
        write_btn_row.addWidget(self._write_labels_btn)
        write_btn_row.addStretch()
        label_layout.addLayout(write_btn_row)
        self._layout.addWidget(grp_label)

        # ── Object Export ──
        grp_export = QGroupBox("Object Export")
        export_layout = QVBoxLayout(grp_export)
        export_layout.setSpacing(3)

        # Object mask selection dropdown
        self._export_mask_combo = NoScrollComboBox()
        _row("Object", self._export_mask_combo, export_layout)

        # Object range selection dropdown
        self._export_object_combo = NoScrollComboBox()
        self._export_object_combo.addItems([
            "Selected images",
            "Selected wells",
            "Annotated",
            "All",
        ])
        self._export_object_combo.setCurrentIndex(0)
        self._export_object_combo.setToolTip(
            "Selected images: objects from currently selected wells, passing the "
            "Image Filters (fields/stacks/timepoints/extra cols)\n"
            "Selected wells: all objects from the currently selected wells, "
            "ignoring the Image Filters\n"
            "Annotated: only manually class-labeled objects\n"
            "All: all objects from the entire dataset"
        )
        _row("Obj range", self._export_object_combo, export_layout)

        # Max objects per image
        self._export_max_obj = NoScrollDoubleSpinBox()
        self._export_max_obj.setRange(0, 10000)
        self._export_max_obj.setValue(0)
        self._export_max_obj.setDecimals(0)
        self._export_max_obj.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self._export_max_obj.setToolTip(
            "Max objects to randomly sample per image (0 = no limit)"
        )
        _row("Max obj", self._export_max_obj, export_layout)

        # Save directory selection
        dir_row = QHBoxLayout()
        dir_row.setSpacing(4)
        dir_row.setContentsMargins(0, 0, 0, 0)
        lbl_dir = QLabel("Save dir")
        lbl_dir.setFixedWidth(60)
        dir_row.addWidget(lbl_dir)
        self._export_dir_input = QLineEdit()
        self._export_dir_input.setPlaceholderText("objects_exported")
        self._export_dir_input.setStyleSheet(
            "min-height: 18px; max-height: 22px; font-size: 8pt; padding: 2px 3px;"
        )
        dir_row.addWidget(self._export_dir_input, stretch=1)
        self._export_dir_btn = QPushButton("...")
        self._export_dir_btn.setProperty("class", "secondary")
        self._export_dir_btn.setFixedSize(24, 20)
        self._export_dir_btn.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
        self._export_dir_btn.clicked.connect(self._on_browse_export_dir)
        dir_row.addWidget(self._export_dir_btn)
        export_layout.addLayout(dir_row)

        # Well subdirectory toggle
        self._export_well_subdir = NoScrollComboBox()
        self._export_well_subdir.addItems(["No", "Yes"])
        self._export_well_subdir.setCurrentIndex(1)
        self._export_well_subdir.setToolTip(
            "Yes: write images under per-well subdirectories (e.g. {save_dir}/{well}/)\n"
            "No: write images directly in the save directory"
        )
        _row("Well dir", self._export_well_subdir, export_layout)

        # Export button (centered)
        export_btn_row = QHBoxLayout()
        export_btn_row.setSpacing(8)
        export_btn_row.addStretch()
        self._export_btn = QPushButton("Save object to Disk")
        self._export_btn.setProperty("class", "secondary")
        self._export_btn.setFixedSize(150, 24)
        self._export_btn.clicked.connect(self.export_clicked)
        export_btn_row.addWidget(self._export_btn)
        export_btn_row.addStretch()
        export_layout.addLayout(export_btn_row)
        self._layout.addWidget(grp_export)

        self._layout.addStretch()
        self.setWidget(container)

    def set_filter_options(
        self,
        fields: list[str],
        stacks: list[str],
        timepoints: list[str],
        extra_cols: dict[str, list[str]] | None = None,
    ) -> None:
        # No-op when nothing changed: rebuilding would reset every filter's
        # checked state to checked_first (losing the user's multi-selection,
        # e.g. when a DB switch repopulates controls). Only rebuild when the
        # option sets actually differ (e.g. a different dataset was loaded).
        extra_cols = {k: v for k, v in (extra_cols or {}).items() if v}
        current = (
            self._fields_widget.get_options() if self._fields_widget else None,
            self._stacks_widget.get_options() if self._stacks_widget else None,
            self._timepoints_widget.get_options() if self._timepoints_widget else None,
            {c: w.get_options() for c, w in self._extra_widgets.items()},
        )
        new = (
            fields or None,
            stacks or None,
            timepoints or None,
            extra_cols,
        )
        if current == new:
            return

        for w in [self._fields_widget, self._stacks_widget, self._timepoints_widget]:
            if w is not None:
                w.deleteLater()
        for w in self._extra_widgets.values():
            w.deleteLater()
        self._extra_widgets.clear()
        while self._filter_container.count():
            item = self._filter_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Only create widgets for non-empty option lists — missing structural
        # columns (e.g. HPA dataset with no stack/timepoint) yield empty lists,
        # and the corresponding filter selector is simply omitted.
        self._fields_widget = (
            _MultiSelectCombo("Fields", fields, checked_first=True) if fields else None
        )
        self._stacks_widget = (
            _MultiSelectCombo("Stacks", stacks, checked_first=True) if stacks else None
        )
        self._timepoints_widget = (
            _MultiSelectCombo("Timepoints", timepoints, checked_first=True) if timepoints else None
        )

        if self._fields_widget:
            self._filter_container.addWidget(self._fields_widget)
        if self._stacks_widget:
            self._filter_container.addWidget(self._stacks_widget)
        if self._timepoints_widget:
            self._filter_container.addWidget(self._timepoints_widget)

        # Extra-col filters: only first item checked by default (matches
        # Fields/Stacks/Timepoints behavior). Wells are the sole exception —
        # they default to unselected (handled via the well grid canvas).
        if extra_cols:
            for col, values in extra_cols.items():
                if not values:
                    continue
                w = _MultiSelectCombo(col, values, checked_first=True)
                self._extra_widgets[col] = w
                self._filter_container.addWidget(w)

    def get_selected_fields(self) -> list[str]:
        return self._fields_widget.get_selected() if self._fields_widget else []

    def get_selected_stacks(self) -> list[str]:
        return self._stacks_widget.get_selected() if self._stacks_widget else []

    def get_selected_timepoints(self) -> list[str]:
        return self._timepoints_widget.get_selected() if self._timepoints_widget else []

    def get_extra_widgets(self) -> dict[str, _MultiSelectCombo]:
        return self._extra_widgets

    def set_channels(self, ch_config: dict, max_value: float | None = None) -> None:
        for w in self._channel_widgets.values():
            w.deleteLater()
        self._channel_widgets.clear()
        while self._ch_container.count():
            item = self._ch_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for ch_name, cfg in ch_config.items():
            row = ChannelControls(ch_name, cfg, max_value=max_value)
            row.config_changed.connect(lambda ch=ch_name: self.channel_config_changed.emit())
            self._ch_container.addWidget(row)
            self._channel_widgets[ch_name] = row

    def get_channel_config(self) -> dict:
        result = {}
        for ch_name, w in self._channel_widgets.items():
            result[ch_name] = w.get_config()
        return result

    def update_channel_values(self, ch_config: dict) -> None:
        for ch_name, cfg in ch_config.items():
            if ch_name in self._channel_widgets:
                self._channel_widgets[ch_name].set_values(
                    cfg.get("vmin", 0), cfg.get("vmax", 65535)
                )

    def set_gamma_visible(self, visible: bool) -> None:
        self._gamma_slider.setVisible(visible)
        self._gamma_slider_label.setVisible(visible)

    # ── Public API ──────────────────────────────────────────────────

    @property
    def contrast(self) -> NoScrollComboBox:
        return self._contrast

    @property
    def gamma_slider(self) -> NoScrollSlider:
        return self._gamma_slider

    @property
    def auto_low(self) -> NoScrollDoubleSpinBox:
        return self._auto_low

    @property
    def auto_high(self) -> NoScrollDoubleSpinBox:
        return self._auto_high

    @property
    def image_size(self) -> NoScrollDoubleSpinBox:
        return self._image_size

    @property
    def overlay_col(self) -> NoScrollComboBox:
        return self._overlay_col

    @property
    def object_mask(self) -> NoScrollComboBox:
        return self._object_mask

    @property
    def overlay_cmap(self) -> NoScrollComboBox:
        return self._overlay_cmap

    @property
    def overlay_alpha(self) -> NoScrollSlider:
        return self._overlay_alpha

    @property
    def sort_by_row(self) -> QRadioButton:
        return self._sort_by_row

    @property
    def fields_widget(self) -> _MultiSelectCombo | None:
        return self._fields_widget

    @property
    def stacks_widget(self) -> _MultiSelectCombo | None:
        return self._stacks_widget

    @property
    def timepoints_widget(self) -> _MultiSelectCombo | None:
        return self._timepoints_widget

    # ── Class Labeling API ─────────────────────────────────────────

    def set_object_masks(self, mask_names: list[str]) -> None:
        """Populate the 'Select object' dropdown in Object Overlay.

        'None' is always the first item (default = no overlay/hover/drag).
        Also sets the default label table name to '{first_mask}_label'.
        """
        cur = self._object_mask.currentText().strip()
        self._object_mask.blockSignals(True)
        self._object_mask.clear()
        self._object_mask.addItem("None")
        for name in mask_names:
            self._object_mask.addItem(name)
        # Restore previous selection if still present, else default to None
        if cur and cur != "None":
            idx = self._object_mask.findText(cur)
            self._object_mask.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self._object_mask.setCurrentIndex(0)
        self._object_mask.blockSignals(False)
        # Default label table name follows the first available mask
        if mask_names and not self._label_table_name.text().strip():
            self._label_table_name.setText(f"{mask_names[0]}_label")

    def _on_object_mask_changed(self, mask_name: str) -> None:
        """Update default label table name and emit signal on selection change."""
        name = mask_name.strip()
        if name and name != "None":
            # Only auto-fill the table name when the user hasn't customized
            # it — a custom name must not be overwritten by mask changes.
            current = self._label_table_name.text().strip()
            if not current or current.endswith("_label"):
                self._label_table_name.setText(f"{name}_label")
        self.object_mask_changed.emit("" if name == "None" else name)

    def get_selected_object_mask(self) -> str:
        """Return the selected object mask name ('' for None)."""
        name = self._object_mask.currentText().strip()
        return "" if name == "None" else name

    def set_label_masks(self, mask_names: list[str]) -> None:
        """Populate the export mask dropdown (kept for Object Export group)."""
        self._export_mask_combo.blockSignals(True)
        self._export_mask_combo.clear()
        self._export_mask_combo.addItems(mask_names)
        self._export_mask_combo.blockSignals(False)

    def get_label_table_name(self) -> str:
        """Return the table name (always user-editable, defaults to {mask}_label)."""
        return self._label_table_name.text().strip()

    def get_selected_class_names(self) -> list[str]:
        """Return currently checked class names."""
        return [n for n, cb in self._class_checkboxes.items() if cb.isChecked()]

    def _on_add_class(self) -> None:
        name = self._class_input.text().strip()
        if not name or name in self._class_checkboxes:
            return
        self._class_input.clear()

        # Show the multi-select section on first class
        self._class_select_label.setVisible(True)
        self._class_select_scroll.setVisible(True)

        cb = QCheckBox(name)
        cb.setStyleSheet(
            "QCheckBox { font-size: 7pt; spacing: 2px; } "
            "QCheckBox::indicator { width: 12px; height: 12px; }"
        )
        cb.setChecked(True)
        cb.toggled.connect(lambda: self.label_class_selection_changed.emit())
        # Insert before the trailing stretch
        idx = self._class_select_layout.count() - 1
        self._class_select_layout.insertWidget(idx, cb)
        self._class_checkboxes[name] = cb

        self.label_class_added.emit(name)

    def _on_remove_class(self) -> None:
        """Remove the last added class."""
        if not self._class_checkboxes:
            return
        # Get last class name (dict preserves insertion order in Python 3.7+)
        name = list(self._class_checkboxes.keys())[-1]
        cb = self._class_checkboxes.pop(name)
        self._class_select_layout.removeWidget(cb)
        cb.deleteLater()
        # Hide multi-select section if no classes left
        if not self._class_checkboxes:
            self._class_select_label.setVisible(False)
            self._class_select_scroll.setVisible(False)
        self.label_class_removed.emit(name)

    # ── Object Export API ──────────────────────────────────────────────

    def _on_browse_export_dir(self) -> None:
        """Open directory dialog for export save location."""
        from PySide6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if dir_path:
            self._export_dir_input.setText(dir_path)

    def get_export_mask(self) -> str:
        """Return the selected export mask name."""
        return self._export_mask_combo.currentText()

    def get_export_object_mode(self) -> str:
        """Return the selected object export mode."""
        return self._export_object_combo.currentText()

    def get_export_dir(self) -> str:
        """Return the export directory path (empty = default)."""
        return self._export_dir_input.text().strip()

    def get_export_max_objects(self) -> int:
        """Return max objects per image (0 = no limit)."""
        return int(self._export_max_obj.value())

    def get_export_well_subdir(self) -> bool:
        """Return whether well subdirectories are enabled."""
        return self._export_well_subdir.currentText() == "Yes"

    def set_export_enabled(self, enabled: bool) -> None:
        """Enable/disable export controls."""
        self._export_btn.setEnabled(enabled)
        self._export_mask_combo.setEnabled(enabled)
        self._export_object_combo.setEnabled(enabled)
        self._export_max_obj.setEnabled(enabled)
        self._export_dir_input.setEnabled(enabled)
        self._export_dir_btn.setEnabled(enabled)
        self._export_well_subdir.setEnabled(enabled)

    def update_export_annotated_option(self, has_annotations: bool) -> None:
        """Enable/disable the 'Annotated' option based on annotation state."""
        model = self._export_object_combo.model()
        item = model.item(self._export_object_combo.findText("Annotated"))
        if item:
            item.setEnabled(has_annotations)
        # If current selection is "Annotated" but no annotations, switch to
        # "Selected images" — NOT "All" or "Selected wells", which would
        # silently export every object in the dataset.
        if not has_annotations and self._export_object_combo.currentText() == "Annotated":
            self._export_object_combo.setCurrentIndex(
                self._export_object_combo.findText("Selected images")
            )
