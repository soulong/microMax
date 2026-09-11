from __future__ import annotations

from typing import List, Optional

import numpy as np
from cellpose import models as cp_models
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from microProfiler.gui.path_drop import enable_path_drop
from microProfiler.gui.dpi import dp
from microProfiler.gui.panels.base_step_panel import BaseStepPanel, make_hsep
from microProfiler.gui.panels._block_container import BlockContainerPanel
from microProfiler.gui.image_widgets import ImageViewer



class SegmentBlockWidget(QWidget):

    def __init__(
        self,
        block_index: int,
        channels: List[str],
        parent=None,
    ):
        super().__init__(parent)
        self.block_index = block_index
        self._channels = channels
        self._syncing = False
        self.setProperty("class", "block-card")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        # One spacing for every block (same as Object/Inference) so rows are
        # equally spaced in every step, separator or not.
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        # Row 1: Object name + Model + Browse + Remove
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Object name:"))
        self._object_name = QLineEdit("cell")
        self._object_name.setToolTip("Name for this segmentation (used as mask prefix)")
        row1.addWidget(self._object_name)
        row1.addWidget(QLabel("Model:"))
        self._model_name = QComboBox()
        self._model_name.setEditable(True)
        # Dragging a model file onto the combo fills its path (the inner
        # QLineEdit owns the drop events of an editable combo).
        enable_path_drop(self._model_name.lineEdit())
        self._model_name.addItems(cp_models.MODEL_NAMES)
        self._model_name.setCurrentText("cpdino")
        self._model_name.setToolTip("Built-in cellpose model or path to custom model file")
        row1.addWidget(self._model_name)
        self._model_browse = QPushButton("Browse...")
        self._model_browse.clicked.connect(self._browse_model)
        row1.addWidget(self._model_browse)
        row1.addStretch()
        # Give the destructive action a gap from the fields before it.
        row1.addSpacing(12)
        self._remove_btn = QPushButton("✕ Remove")
        self._remove_btn.setToolTip("Remove this segmentation block")
        row1.addWidget(self._remove_btn)
        layout.addLayout(row1)

        # Row 2: Resize factor + Diameter + thresholds
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Resize factor:"))
        self._resize_factor = QDoubleSpinBox()
        self._resize_factor.setRange(0.01, 10.0)
        self._resize_factor.setValue(0.5)
        self._resize_factor.setSingleStep(0.1)
        self._resize_factor.setDecimals(2)
        self._resize_factor.setToolTip("Downscale factor before segmentation (speeds up inference)")
        row2.addWidget(self._resize_factor)
        row2.addWidget(QLabel("Diameter:"))
        self._diameter = QSpinBox()
        self._diameter.setRange(0, 10000)
        self._diameter.setValue(0)
        self._diameter.setSpecialValueText("Auto")
        self._diameter.setToolTip("Expected object diameter in pixels (0 = auto-detect)")
        row2.addWidget(self._diameter)
        row2.addWidget(QLabel("Edge ratio:"))
        self._edge_ratio = QDoubleSpinBox()
        self._edge_ratio.setRange(0.0, 1.0)
        self._edge_ratio.setValue(0.4)
        self._edge_ratio.setSingleStep(0.05)
        self._edge_ratio.setDecimals(2)
        self._edge_ratio.setToolTip(
            "Drop masks whose edge-pixel/perimeter ratio exceeds this "
            "(1 = keep everything): mostly-clipped objects score ~0.5, "
            "interior objects 0.0. Applies to the saved mask AND the "
            "preview overlay.")
        row2.addWidget(self._edge_ratio)
        row2.addWidget(QLabel("Flow thresh:"))
        self._flow_threshold = QDoubleSpinBox()
        self._flow_threshold.setRange(0.0, 10.0)
        self._flow_threshold.setValue(0.4)
        self._flow_threshold.setSingleStep(0.1)
        self._flow_threshold.setToolTip("Cellpose flow error threshold (higher = more permissive)")
        row2.addWidget(self._flow_threshold)
        row2.addWidget(QLabel("Cell prob:"))
        self._cellprob_threshold = QDoubleSpinBox()
        self._cellprob_threshold.setRange(0.0, 10.0)
        self._cellprob_threshold.setValue(0.0)
        self._cellprob_threshold.setSingleStep(0.1)
        self._cellprob_threshold.setToolTip("Cell probability threshold (lower = more permissive)")
        row2.addWidget(self._cellprob_threshold)
        row2.addWidget(QLabel("GPU batch:"))
        self._gpu_batch_size = QSpinBox()
        self._gpu_batch_size.setRange(1, 512)
        self._gpu_batch_size.setValue(32)
        self._gpu_batch_size.setToolTip("Batch size for GPU inference (reduce if OOM)")
        row2.addWidget(self._gpu_batch_size)
        self._overwrite_mask = QCheckBox("Overwrite mask")
        self._overwrite_mask.setChecked(False)
        self._overwrite_mask.setToolTip("Re-run segmentation even if mask files already exist")
        row2.addWidget(self._overwrite_mask)
        row2.addStretch()
        layout.addLayout(row2)

        # Section break: input channels.
        layout.addWidget(make_hsep())

        # Row 3: Chan1 + Merge1
        row3 = QHBoxLayout()
        self._chan1_row = row3
        row3.addWidget(QLabel("Chan1:"))
        self._chan1_checkboxes: List[QCheckBox] = []
        self._chan1_placeholder: Optional[QLabel] = None
        if self._channels:
            for ch in self._channels:
                cb = QCheckBox(ch)
                cb.toggled.connect(self._on_chan1_toggled)
                self._chan1_checkboxes.append(cb)
                row3.addWidget(cb)
        else:
            self._chan1_placeholder = QLabel("Load a dataset to configure")
            self._chan1_placeholder.setProperty("class", "placeholder")
            row3.addWidget(self._chan1_placeholder)
        row3.addSpacing(10)
        row3.addWidget(QLabel("Merge1:"))
        self._merge1 = QComboBox()
        self._merge1.addItems(["mean", "max", "min"])
        self._merge1.setToolTip("How to combine selected Chan1 channels (mean/max/min)")
        row3.addWidget(self._merge1)
        row3.addStretch()
        layout.addLayout(row3)

        # Row 4: Chan2 + Merge2
        row4 = QHBoxLayout()
        self._chan2_row = row4
        row4.addWidget(QLabel("Chan2:"))
        self._chan2_checkboxes: List[QCheckBox] = []
        self._chan2_placeholder: Optional[QLabel] = None
        if self._channels:
            for ch in self._channels:
                cb = QCheckBox(ch)
                cb.toggled.connect(self._on_chan2_toggled)
                self._chan2_checkboxes.append(cb)
                row4.addWidget(cb)
        else:
            self._chan2_placeholder = QLabel("Load a dataset to configure")
            self._chan2_placeholder.setProperty("class", "placeholder")
            row4.addWidget(self._chan2_placeholder)
        row4.addSpacing(10)
        row4.addWidget(QLabel("Merge2:"))
        self._merge2 = QComboBox()
        self._merge2.addItems(["mean", "max", "min"])
        self._merge2.setToolTip("How to combine selected Chan2 channels (mean/max/min)")
        row4.addWidget(self._merge2)
        row4.addStretch()
        layout.addLayout(row4)

        # Section break: preview controls.
        layout.addWidget(make_hsep())

        # Row 5: left-aligned Pick Random, Preview Segment, Hide Mask — one
        # identical fixed width so the three controls line up and never
        # stretch with the card.
        row5 = QHBoxLayout()
        self._pick_btn = QPushButton("Pick Random")
        self._preview_btn = QPushButton("Preview Segment")
        self._mask_toggle_btn = QPushButton("Hide Mask")
        self._mask_toggle_btn.setCheckable(True)
        self._mask_toggle_btn.setChecked(True)
        self._mask_toggle_btn.setEnabled(False)
        self._mask_toggle_btn.clicked.connect(self._on_mask_toggle)
        button_w = max(
            self._pick_btn.sizeHint().width(),
            self._preview_btn.sizeHint().width(),
            self._mask_toggle_btn.sizeHint().width(),
        )
        for _btn in (self._pick_btn, self._preview_btn, self._mask_toggle_btn):
            _btn.setFixedWidth(button_w)
        row5.addWidget(self._pick_btn)
        row5.addWidget(self._preview_btn)
        row5.addWidget(self._mask_toggle_btn)
        # Keep the fixed-width buttons packed at the left; the stretch eats
        # the leftover width so they never spread out.
        row5.addStretch()
        layout.addLayout(row5)

        # Preview row — centered C1/C2 images, tightly packed (2/3 size).
        self._preview_container = QWidget()
        self._preview_container.setMinimumHeight(dp(146))
        preview_container_layout = QVBoxLayout(self._preview_container)
        preview_container_layout.setContentsMargins(0, 0, 0, 0)
        preview_container_layout.setSpacing(0)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(0)
        preview_row.setContentsMargins(0, 0, 0, 0)

        preview_row.addStretch()

        c1_col = QVBoxLayout()
        c1_col.setSpacing(0)
        c1_col.setAlignment(Qt.AlignTop)
        c1_label = QLabel("C1 (merged):")
        c1_label.setAlignment(Qt.AlignCenter)
        c1_label.setContentsMargins(0, 0, 0, 0)
        c1_col.addWidget(c1_label)
        self._c1_view = ImageViewer()
        self._c1_view.setMinimumSize(dp(133), dp(133))
        self._c1_view.set_show_axes(True)
        c1_col.addWidget(self._c1_view)
        preview_row.addLayout(c1_col, 1)

        c2_col = QVBoxLayout()
        c2_col.setSpacing(0)
        c2_col.setAlignment(Qt.AlignTop)
        c2_label = QLabel("C2 (merged):")
        c2_label.setAlignment(Qt.AlignCenter)
        c2_label.setContentsMargins(0, 0, 0, 0)
        c2_col.addWidget(c2_label)
        self._c2_view = ImageViewer()
        self._c2_view.setMinimumSize(dp(133), dp(133))
        self._c2_view.set_show_axes(True)
        c2_col.addWidget(self._c2_view)
        preview_row.addLayout(c2_col, 1)

        preview_row.addStretch()

        preview_container_layout.addLayout(preview_row)
        self._preview_container.setVisible(False)
        layout.addWidget(self._preview_container)

        # Sync zoom/pan/reset across viewers
        self._c1_view.zoomed.connect(lambda: self._sync_views(self._c1_view))
        self._c1_view.panned.connect(lambda: self._sync_views(self._c1_view))
        self._c2_view.zoomed.connect(lambda: self._sync_views(self._c2_view))
        self._c2_view.panned.connect(lambda: self._sync_views(self._c2_view))
        self._c1_view.view_reset.connect(self._reset_all_views)
        self._c2_view.view_reset.connect(self._reset_all_views)

    def show_preview(self):
        self._preview_container.setVisible(True)

    def get_chan1(self) -> List[str]:
        return [cb.text() for cb in self._chan1_checkboxes if cb.isChecked()]

    def get_chan2(self) -> List[str]:
        return [cb.text() for cb in self._chan2_checkboxes if cb.isChecked()]

    def _on_chan1_toggled(self, *_) -> None:
        if self._chan1_checkboxes and not any(
            cb.isChecked() for cb in self._chan1_checkboxes
        ):
            self._c1_view.clear_image()

    def _on_chan2_toggled(self, *_) -> None:
        if self._chan2_checkboxes and not any(
            cb.isChecked() for cb in self._chan2_checkboxes
        ):
            self._c2_view.clear_image()

    def populate_channels(self, channels: List[str]) -> None:
        """Rebuild the Chan1/Chan2 checkbox rows.

        Currently-checked selections are preserved across the rebuild so a
        filtered-dataset or post-run refresh never wipes the user's picks;
        restored configs re-check their channels via _apply_block_config
        (base-class deferred restore).
        """
        self._channels = channels

        # Snapshot current selections before the widgets are destroyed.
        saved_chan1 = {cb.text() for cb in self._chan1_checkboxes if cb.isChecked()}
        saved_chan2 = {cb.text() for cb in self._chan2_checkboxes if cb.isChecked()}
        self._chan1_checkboxes.clear()
        self._chan2_checkboxes.clear()

        # Helper to rebuild one channel row
        def _rebuild_row(row, placeholder_attr, checkbox_list, checked_set, toggled_handler):
            placeholder = getattr(self, placeholder_attr, None)
            if placeholder is not None:
                for i in range(row.count() - 1, -1, -1):
                    item = row.itemAt(i)
                    if item and item.widget() is placeholder:
                        row.removeItem(item)
                        placeholder.deleteLater()
                        setattr(self, placeholder_attr, None)
                        break

            # Remove all QCheckBox widgets from the row (iterate backwards)
            for i in range(row.count() - 1, -1, -1):
                item = row.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QCheckBox):
                    w = item.widget()
                    row.removeItem(item)
                    w.deleteLater()

            # Find insertion point: right after the label
            insert_at = -1
            for i in range(row.count()):
                w = row.itemAt(i).widget()
                if isinstance(w, QLabel) and w.text().startswith("Chan"):
                    insert_at = i + 1
                    break

            # Insert checkboxes
            for ch in channels:
                cb = QCheckBox(ch)
                cb.setChecked(ch in checked_set)
                cb.toggled.connect(toggled_handler)
                row.insertWidget(insert_at, cb)
                checkbox_list.append(cb)
                insert_at += 1

        # Both Chan1 and Chan2 keep their current selections across the
        # rebuild; stored configs (session.yml) re-check via _apply_block_config.
        _rebuild_row(self._chan1_row, "_chan1_placeholder", self._chan1_checkboxes, saved_chan1, self._on_chan1_toggled)
        _rebuild_row(self._chan2_row, "_chan2_placeholder", self._chan2_checkboxes, saved_chan2, self._on_chan2_toggled)

    def build_config_section(self) -> dict:
        chan1 = self.get_chan1()
        chan2 = self.get_chan2() or None
        return {
            "object_name": self._object_name.text().strip(),
            "model_name": self._model_name.currentText(),
            "chan1": chan1,
            "chan2": chan2,
            "merge1": self._merge1.currentText(),
            "merge2": self._merge2.currentText(),
            "resize_factor": self._resize_factor.value(),
            "diameter": self._diameter.value() if self._diameter.value() > 0 else None,
            "flow_threshold": self._flow_threshold.value(),
            "cellprob_threshold": self._cellprob_threshold.value(),
            "edge_pixel_ratio": self._edge_ratio.value(),
            "gpu_batch_size": self._gpu_batch_size.value(),
            "overwrite_mask": self._overwrite_mask.isChecked(),
        }

    def _browse_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Cellpose Model", "", "All files (*)",
        )
        if path:
            self._model_name.setCurrentText(path)

    def _on_mask_toggle(self) -> None:
        visible = self._mask_toggle_btn.isChecked()
        self._mask_toggle_btn.setText("Hide Mask" if visible else "Show Mask")
        self._c1_view.set_overlay_visible(visible)
        self._c2_view.set_overlay_visible(visible)

    def _sync_views(self, source: ImageViewer) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            t = source.transform()
            h = source.horizontalScrollBar().value()
            v_bar = source.verticalScrollBar().value()
            for view in (self._c1_view, self._c2_view):
                if view is not source:
                    view.setTransform(t)
                    view.horizontalScrollBar().setValue(h)
                    view.verticalScrollBar().setValue(v_bar)
                    view._fit_to_view = source._fit_to_view
                    view._zoom_level = source._zoom_level
        finally:
            self._syncing = False

    def _reset_all_views(self) -> None:
        for v in (self._c1_view, self._c2_view):
            v._reset_view()


class SegmentStepPanel(BlockContainerPanel):

    step_name = "segment"
    _block_widget_class = SegmentBlockWidget

    pick_requested = Signal(int)
    preview_requested = Signal(int)

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("Segmentation")
        self._build_block_container("+ Add New Segmentation")
        self._channels: List[str] = []
        self._add_block_generic([])

    def _connect_block_signals(self, block: SegmentBlockWidget) -> None:
        super()._connect_block_signals(block)
        self._wire_param_signal(block._object_name)
        self._wire_param_signal(block._resize_factor)
        self._wire_param_signal(block._diameter)
        self._wire_param_signal(block._flow_threshold)
        self._wire_param_signal(block._edge_ratio)
        self._wire_param_signal(block._cellprob_threshold)
        self._wire_param_signal(block._gpu_batch_size)
        self._wire_param_signal(block._model_name)
        self._wire_param_signal(block._merge1)
        self._wire_param_signal(block._merge2)
        self._wire_param_signal(block._overwrite_mask)
        for cb in block._chan1_checkboxes + block._chan2_checkboxes:
            self._wire_param_signal(cb)
        block._pick_btn.clicked.connect(
            lambda: (self.pick_requested.emit(block.block_index), block.show_preview())
        )
        block._preview_btn.clicked.connect(
            lambda: (self.preview_requested.emit(block.block_index), block.show_preview())
        )

    def _apply_block_config(self, block: SegmentBlockWidget, cfg: dict) -> None:
        # Only set the object name when the config carries one — never clear
        # the widget's "cell" default (empty configs wipe it otherwise).
        obj_name = cfg.get("object_name")
        if obj_name:
            block._object_name.setText(str(obj_name))
        model_name = cfg.get("model_name", "")
        if model_name:
            idx = block._model_name.findText(str(model_name))
            if idx >= 0:
                block._model_name.setCurrentIndex(idx)
            else:
                block._model_name.setCurrentText(str(model_name))
        BaseStepPanel._set_widget(block._resize_factor, cfg.get("resize_factor", 0.5), "resize_factor")
        BaseStepPanel._set_widget(block._diameter, cfg.get("diameter", 0), "diameter")
        BaseStepPanel._set_widget(block._flow_threshold, cfg.get("flow_threshold", 0.4), "flow_threshold")
        BaseStepPanel._set_widget(block._edge_ratio, cfg.get("edge_pixel_ratio", 0.4), "edge_pixel_ratio")
        BaseStepPanel._set_widget(block._cellprob_threshold, cfg.get("cellprob_threshold", 0.0), "cellprob_threshold")
        BaseStepPanel._set_widget(block._gpu_batch_size, cfg.get("gpu_batch_size", 32), "gpu_batch_size")
        merge1 = cfg.get("merge1", "")
        if merge1:
            idx = block._merge1.findText(str(merge1))
            if idx >= 0:
                block._merge1.setCurrentIndex(idx)
        merge2 = cfg.get("merge2", "")
        if merge2:
            idx = block._merge2.findText(str(merge2))
            if idx >= 0:
                block._merge2.setCurrentIndex(idx)
        # Restore overwrite_mask checkbox
        if hasattr(block, "_overwrite_mask"):
            block._overwrite_mask.setChecked(bool(cfg.get("overwrite_mask", False)))
        # Channel checks are applied by the base-class deferred restore: this
        # method is re-run after populate_channels built the checkboxes.
        BaseStepPanel._set_checked_states(block._chan1_checkboxes, cfg.get("chan1"))
        BaseStepPanel._set_checked_states(block._chan2_checkboxes, cfg.get("chan2"))

    def validate_object_names(self) -> Optional[str]:
        names = []
        for block in self._blocks:
            name = block._object_name.text().strip()
            if not name:
                return "Object name cannot be empty in one of the blocks."
            if name in names:
                return f"Duplicate object name: '{name}'. Each block must have a unique name."
            names.append(name)
        return None

    def get_object_names(self) -> List[str]:
        return [b._object_name.text().strip() for b in self._blocks if b._object_name.text().strip()]

    def build_config_section(self) -> List[dict]:
        return [block.build_config_section() for block in self._blocks]

    def set_preview_c1(self, block_index: int, img: Optional[np.ndarray]) -> None:
        if 0 <= block_index < len(self._blocks):
            block = self._blocks[block_index]
            if img is None:
                block._c1_view.clear_image()
            else:
                block._c1_view.set_image(img)
            block.show_preview()

    def set_preview_c2(self, block_index: int, img: Optional[np.ndarray]) -> None:
        if 0 <= block_index < len(self._blocks):
            block = self._blocks[block_index]
            if img is None:
                block._c2_view.clear_image()
            else:
                block._c2_view.set_image(img)
            block.show_preview()

    def clear_preview(self) -> None:
        for block in self._blocks:
            block._c1_view.clear_image()
            block._c2_view.clear_image()
            block._mask_toggle_btn.setChecked(False)
            block._mask_toggle_btn.setText("Show Mask")
            block._mask_toggle_btn.setEnabled(False)
            block._preview_container.setVisible(False)

    def set_preview_mask(self, block_index: int, mask: np.ndarray) -> None:
        if 0 <= block_index < len(self._blocks):
            self._blocks[block_index]._c1_view.overlay_mask(mask)
            self._blocks[block_index]._c2_view.overlay_mask(mask)
            self._blocks[block_index]._mask_toggle_btn.setEnabled(True)
