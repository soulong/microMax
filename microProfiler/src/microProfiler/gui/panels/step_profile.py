from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from microProfiler.config import default_n_workers
from microProfiler.gui.panels.base_step_panel import BaseStepPanel, dp, make_hsep
from microProfiler.gui.panels._block_container import BlockContainerPanel


class ObjectProfileBlockWidget(QWidget):

    def __init__(
        self,
        block_index: int,
        channels: List[str],
        parent=None,
    ):
        super().__init__(parent)
        self.block_index = block_index
        self._channels = channels
        self.setProperty("class", "block-card")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # Row: Mask name + Parent mask + Remove button (all in one row)
        row_top = QHBoxLayout()
        row_top.addWidget(QLabel("Mask name:"))
        self._object_mask = QComboBox()
        self._object_mask.setEditable(True)
        self._object_mask.setToolTip("Mask to use as object boundary")
        row_top.addWidget(self._object_mask)
        row_top.addWidget(QLabel("Parent mask:"))
        self._parent_mask = QComboBox()
        self._parent_mask.setEditable(True)
        self._parent_mask.addItems(["None"])
        self._parent_mask.setToolTip("Optional parent mask for hierarchical segmentation")
        row_top.addWidget(self._parent_mask)
        row_top.addWidget(QLabel("Output table:"))
        self._output_table = QLineEdit()
        self._output_table.setMaximumWidth(150)
        self._output_table.setToolTip("SQLite table name for results (defaults to mask name)")
        self._table_synced = True
        self._object_mask.currentTextChanged.connect(self._sync_output_table)
        self._output_table.textEdited.connect(self._on_output_table_edited)
        row_top.addWidget(self._output_table)
        self._overwrite_db = QCheckBox("Overwrite DB")
        self._overwrite_db.setChecked(False)
        self._overwrite_db.setToolTip("Drop existing table before profiling (otherwise BatchWriter replaces on first write)")
        row_top.addWidget(self._overwrite_db)
        row_top.addStretch()
        self._remove_btn = QPushButton("✕ Remove")
        self._remove_btn.setProperty("class", "danger")
        self._remove_btn.setToolTip("Remove this object profiling block")
        row_top.addWidget(self._remove_btn)
        layout.addLayout(row_top)

        # -- Intensity --
        layout.addWidget(make_hsep())
        self._intensity_ch_layout = QHBoxLayout()
        self._intensity_ch_layout.setContentsMargins(0, 0, 0, 0)
        _lbl = QLabel("Intensity:")
        _lbl.setFixedWidth(dp(85))
        self._intensity_ch_layout.addWidget(_lbl)
        self._intensity_cbs: List[QCheckBox] = []
        self._intensity_placeholder: Optional[QLabel] = None
        if self._channels:
            for ch in self._channels:
                cb = QCheckBox(ch)
                self._intensity_ch_layout.addWidget(cb)
                self._intensity_cbs.append(cb)
        else:
            self._intensity_placeholder = QLabel("Load a dataset to configure")
            self._intensity_placeholder.setProperty("class", "placeholder")
            self._intensity_ch_layout.addWidget(self._intensity_placeholder)
        layout.addLayout(self._intensity_ch_layout)

        # -- Radial + Bins (single row) --
        layout.addWidget(make_hsep())
        self._radial_layout = QHBoxLayout()
        self._radial_layout.setContentsMargins(0, 0, 0, 0)
        _lbl = QLabel("Radial:")
        _lbl.setFixedWidth(dp(85))
        self._radial_layout.addWidget(_lbl)
        self._radial_cbs: List[QCheckBox] = []
        self._radial_placeholder: Optional[QLabel] = None
        if self._channels:
            for ch in self._channels:
                cb = QCheckBox(ch)
                self._radial_layout.addWidget(cb)
                self._radial_cbs.append(cb)
        else:
            self._radial_placeholder = QLabel("Load a dataset to configure")
            self._radial_placeholder.setProperty("class", "placeholder")
            self._radial_layout.addWidget(self._radial_placeholder)
        self._radial_layout.addSpacing(10)
        self._radial_layout.addWidget(QLabel("Bins:"))
        self._radial_bins = QSpinBox()
        self._radial_bins.setRange(1, 50)
        self._radial_bins.setValue(4)
        self._radial_bins.setFixedWidth(50)
        self._radial_bins.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._radial_bins.setToolTip(
            "Number of concentric rings from object edge to center. "
            "Use fewer bins for coarse resolution, more bins for fine detail. "
            "Useful for detecting spatial gradients in protein localization."
        )
        self._radial_layout.addWidget(self._radial_bins)
        layout.addLayout(self._radial_layout)

        # -- Granularity + spectrum length + subsample + image_sample + background radius --
        layout.addWidget(make_hsep())
        self._gran_layout = QHBoxLayout()
        self._gran_layout.setContentsMargins(0, 0, 0, 0)
        _lbl = QLabel("Granularity:")
        _lbl.setFixedWidth(dp(85))
        self._gran_layout.addWidget(_lbl)
        self._gran_cbs: List[QCheckBox] = []
        self._gran_placeholder: Optional[QLabel] = None
        if self._channels:
            for ch in self._channels:
                cb = QCheckBox(ch)
                self._gran_layout.addWidget(cb)
                self._gran_cbs.append(cb)
        else:
            self._gran_placeholder = QLabel("Load a dataset to configure")
            self._gran_placeholder.setProperty("class", "placeholder")
            self._gran_layout.addWidget(self._gran_placeholder)
        self._gran_layout.addSpacing(10)
        self._gran_layout.addWidget(QLabel("Spectrum len:"))
        self._gran_spectrum_length = QSpinBox()
        self._gran_spectrum_length.setRange(1, 64)
        self._gran_spectrum_length.setValue(8)
        self._gran_spectrum_length.setFixedWidth(50)
        self._gran_spectrum_length.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._gran_spectrum_length.setToolTip(
            "Number of granularity spectrum bins. Each bin is one additional "
            "erosion+reconstruction step with disk(1) in the subsampled grid. "
            "Larger values extend the spectrum to coarser scales; 16 is the "
            "CellProfiler default and covers a useful size range."
        )
        self._gran_layout.addWidget(self._gran_spectrum_length)
        self._gran_layout.addWidget(QLabel("Subsample:"))
        self._gran_subsample_ratio = QDoubleSpinBox()
        self._gran_subsample_ratio.setRange(0.05, 1.0)
        self._gran_subsample_ratio.setValue(0.5)
        self._gran_subsample_ratio.setSingleStep(0.05)
        self._gran_subsample_ratio.setFixedWidth(60)
        self._gran_subsample_ratio.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._gran_subsample_ratio.setToolTip(
            "Subsampling factor for the granularity calculation. "
            "Smaller values are faster and let the spectrum cover coarser "
            "feature sizes; the CellProfiler default is 0.25."
        )
        self._gran_layout.addWidget(self._gran_subsample_ratio)
        self._gran_layout.addWidget(QLabel("BG sample:"))
        self._gran_background_subsample_ratio = QDoubleSpinBox()
        self._gran_background_subsample_ratio.setRange(0.05, 1.0)
        self._gran_background_subsample_ratio.setValue(0.25)
        self._gran_background_subsample_ratio.setSingleStep(0.05)
        self._gran_background_subsample_ratio.setFixedWidth(60)
        self._gran_background_subsample_ratio.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._gran_background_subsample_ratio.setToolTip(
            "Subsampling factor for the background-reduction step. "
            "CellProfiler default is 0.25; small factors are preferred when "
            "the structures of interest are large."
        )
        self._gran_layout.addWidget(self._gran_background_subsample_ratio)
        self._gran_layout.addWidget(QLabel("BG radius:"))
        self._gran_background_radius = QSpinBox()
        self._gran_background_radius.setRange(1, 256)
        self._gran_background_radius.setValue(10)
        self._gran_background_radius.setFixedWidth(50)
        self._gran_background_radius.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._gran_background_radius.setToolTip(
            "Radius of the structuring element used to estimate the image "
            "background (erosion then dilation on the BG-subsampled grid). "
            "CellProfiler default is 10. Should be larger than the largest "
            "feature of interest so the background is fully removed."
        )
        self._gran_layout.addWidget(self._gran_background_radius)
        layout.addLayout(self._gran_layout)

        # -- GLCM + Distances + Levels (single row, no Angles widget) --
        layout.addWidget(make_hsep())
        self._glcm_layout = QHBoxLayout()
        self._glcm_layout.setContentsMargins(0, 0, 0, 0)
        _lbl = QLabel("GLCM:")
        _lbl.setFixedWidth(dp(85))
        self._glcm_layout.addWidget(_lbl)
        self._glcm_cbs: List[QCheckBox] = []
        self._glcm_placeholder: Optional[QLabel] = None
        if self._channels:
            for ch in self._channels:
                cb = QCheckBox(ch)
                self._glcm_layout.addWidget(cb)
                self._glcm_cbs.append(cb)
        else:
            self._glcm_placeholder = QLabel("Load a dataset to configure")
            self._glcm_placeholder.setProperty("class", "placeholder")
            self._glcm_layout.addWidget(self._glcm_placeholder)
        self._glcm_layout.addSpacing(10)
        self._glcm_layout.addWidget(QLabel("Distances:"))
        self._glcm_distances = QLineEdit("2")
        self._glcm_distances.setFixedWidth(50)
        # Only digits/commas/spaces are accepted — build_config_section
        # int-parses this text, so invalid characters would raise a raw
        # ValueError inside the Qt slot.
        self._glcm_distances.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"[0-9,\s]*"), self)
        )
        self._glcm_distances.setToolTip(
            "Pixel offsets for Gray-Level Co-occurrence Matrix. "
            "Larger distances capture longer-range texture patterns. "
            "Common values: 1-3 for fine texture, 5-10 for coarse texture."
        )
        self._glcm_layout.addWidget(self._glcm_distances)
        self._glcm_layout.addWidget(QLabel("Levels:"))
        self._glcm_levels = QSpinBox()
        self._glcm_levels.setRange(2, 256)
        self._glcm_levels.setValue(256)
        self._glcm_levels.setFixedWidth(50)
        self._glcm_levels.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._glcm_levels.setToolTip(
            "Grayscale quantization bins. More levels capture finer intensity variations "
            "but increase computation. 8-32 typical range."
        )
        self._glcm_layout.addWidget(self._glcm_levels)
        layout.addLayout(self._glcm_layout)

        # -- Correlation --
        layout.addWidget(make_hsep())
        self._corr_layout = QHBoxLayout()
        self._corr_layout.setContentsMargins(0, 0, 0, 0)
        _lbl = QLabel("Correlation:")
        _lbl.setFixedWidth(dp(85))
        self._corr_layout.addWidget(_lbl)
        self._corr_cbs: List[QCheckBox] = []
        self._corr_placeholder: Optional[QLabel] = None
        if self._channels:
            for ch in self._channels:
                for other in self._channels:
                    if other > ch:
                        pair_cb = QCheckBox(f"{ch}-{other}")
                        self._corr_layout.addWidget(pair_cb)
                        self._corr_cbs.append(pair_cb)
        else:
            self._corr_placeholder = QLabel("Load a dataset to configure")
            self._corr_placeholder.setProperty("class", "placeholder")
            self._corr_layout.addWidget(self._corr_placeholder)
        layout.addLayout(self._corr_layout)
        # Add tooltip to correlation checkbox container
        for cb in self._corr_cbs:
            cb.setToolTip(
                "Select channel pairs to measure Pearson correlation. "
                "Values range from -1 (perfect anti-correlation) to +1 (perfect correlation). "
                "0 indicates no linear relationship."
            )

        # Ensure content left-aligns (stretch at end of each row)
        self._left_align_content()

    def _left_align_content(self):
        for lyt in [
            self._intensity_ch_layout, self._radial_layout, self._gran_layout,
            self._glcm_layout, self._corr_layout,
        ]:
            lyt.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            # Only remove stretch spacers (addStretch with zero size), preserve fixed spacing
            for i in range(lyt.count() - 1, -1, -1):
                item = lyt.itemAt(i)
                sp = item.spacerItem() if item else None
                if sp and sp.sizeHint().width() == 0 and sp.sizeHint().height() == 0:
                    lyt.removeItem(item)
            lyt.addStretch()

    # ── channel helpers ────────────────────────────────────────────────

    def get_mask_name(self) -> str:
        return self._object_mask.currentText()

    def get_parent_mask_name(self) -> Optional[str]:
        txt = self._parent_mask.currentText()
        return None if txt == "None" else txt

    def get_output_table_name(self) -> str:
        return self._output_table.text().strip() or self.get_mask_name()

    def _sync_output_table(self, text: str) -> None:
        if self._table_synced:
            self._output_table.setText(text)

    def _on_output_table_edited(self) -> None:
        self._table_synced = False
        # Re-sync if user clears the field
        if not self._output_table.text().strip():
            self._table_synced = True
            self._output_table.setText(self._object_mask.currentText())

    def populate_channels(self, channels: List[str]) -> None:
        """Rebuild the five channel rows (all unchecked by default).

        Currently-checked selections are preserved across the rebuild (a
        filtered-dataset or post-run refresh must not wipe the user's picks);
        restored configs re-check their channels via _apply_block_config after
        this runs (base-class deferred restore).
        """
        self._channels = channels

        # Snapshot current selections before the widgets are destroyed.
        saved = {
            "intensity": {cb.text() for cb in self._intensity_cbs if cb.isChecked()},
            "radial": {cb.text() for cb in self._radial_cbs if cb.isChecked()},
            "granularity": {cb.text() for cb in self._gran_cbs if cb.isChecked()},
            "glcm": {cb.text() for cb in self._glcm_cbs if cb.isChecked()},
            "correlation": {cb.text() for cb in self._corr_cbs if cb.isChecked()},
        }

        # Remove placeholders and existing checkboxes
        BaseStepPanel._remove_placeholder(self._intensity_ch_layout, "_intensity_placeholder", self)
        BaseStepPanel._remove_placeholder(self._radial_layout, "_radial_placeholder", self)
        BaseStepPanel._remove_placeholder(self._gran_layout, "_gran_placeholder", self)
        BaseStepPanel._remove_placeholder(self._glcm_layout, "_glcm_placeholder", self)
        BaseStepPanel._remove_placeholder(self._corr_layout, "_corr_placeholder", self)
        BaseStepPanel._clear_checkboxes(self._intensity_ch_layout, self._intensity_cbs)
        BaseStepPanel._clear_checkboxes(self._radial_layout, self._radial_cbs)
        BaseStepPanel._clear_checkboxes(self._gran_layout, self._gran_cbs)
        BaseStepPanel._clear_checkboxes(self._glcm_layout, self._glcm_cbs)
        BaseStepPanel._clear_checkboxes(self._corr_layout, self._corr_cbs)

        if not channels:
            for layout, attr in [
                (self._intensity_ch_layout, "_intensity_placeholder"),
                (self._radial_layout, "_radial_placeholder"),
                (self._gran_layout, "_gran_placeholder"),
                (self._glcm_layout, "_glcm_placeholder"),
                (self._corr_layout, "_corr_placeholder"),
            ]:
                placeholder = QLabel("Load a dataset to configure")
                placeholder.setProperty("class", "placeholder")
                setattr(self, attr, placeholder)
                layout.insertWidget(1, placeholder)
            self._left_align_content()
            return

        for i, ch in enumerate(channels):
            cb = QCheckBox(ch)
            cb.setChecked(ch in saved["intensity"])
            self._intensity_ch_layout.insertWidget(1 + i, cb)
            self._intensity_cbs.append(cb)

            cb = QCheckBox(ch)
            cb.setChecked(ch in saved["radial"])
            self._radial_layout.insertWidget(1 + i, cb)
            self._radial_cbs.append(cb)

            cb = QCheckBox(ch)
            cb.setChecked(ch in saved["granularity"])
            self._gran_layout.insertWidget(1 + i, cb)
            self._gran_cbs.append(cb)

            cb = QCheckBox(ch)
            cb.setChecked(ch in saved["glcm"])
            self._glcm_layout.insertWidget(1 + i, cb)
            self._glcm_cbs.append(cb)

            for other in channels:
                if other > ch:
                    pair = f"{ch}-{other}"
                    pair_cb = QCheckBox(pair)
                    pair_cb.setChecked(pair in saved["correlation"])
                    self._corr_layout.insertWidget(1 + len(self._corr_cbs), pair_cb)
                    self._corr_cbs.append(pair_cb)

        self._left_align_content()

    def populate_masks(self, mask_names: List[str]) -> None:
        stripped = [n.removeprefix("mask_") for n in mask_names]
        current = self._object_mask.currentText()
        self._object_mask.clear()
        for name in stripped:
            self._object_mask.addItem(name)
        if stripped:
            if current in stripped:
                self._object_mask.setCurrentText(current)
            else:
                self._object_mask.setCurrentIndex(0)
        current_parent = self._parent_mask.currentText()
        self._parent_mask.clear()
        self._parent_mask.addItems(["None"] + stripped)
        if current_parent in (["None"] + stripped):
            self._parent_mask.setCurrentText(current_parent)
        elif stripped:
            self._parent_mask.setCurrentIndex(0)

    def get_correlation_pairs(self) -> Optional[List[List[str]]]:
        pairs = []
        for cb in self._corr_cbs:
            if cb.isChecked():
                a, b = cb.text().split("-")
                pairs.append([a, b])
        return pairs or None

    def build_config_section(self) -> dict:
        glcm_d_str = self._glcm_distances.text().strip()
        glcm_d = None
        if glcm_d_str:
            glcm_d = [int(x.strip()) for x in glcm_d_str.split(",") if x.strip()]

        return {
            "mask_name": self.get_mask_name() or None,
            "parent_mask_name": self.get_parent_mask_name(),
            "output_table_name": self.get_output_table_name() or None,
            "overwrite_db": self._overwrite_db.isChecked(),
            "intensity_channels": BaseStepPanel._checked_checkboxes(self._intensity_cbs) or None,
            "radial_channels": BaseStepPanel._checked_checkboxes(self._radial_cbs) or None,
            "radial_bins": self._radial_bins.value(),
            "gran_channels": BaseStepPanel._checked_checkboxes(self._gran_cbs) or None,
            "gran_spectrum_length": self._gran_spectrum_length.value(),
            "gran_subsample_ratio": self._gran_subsample_ratio.value(),
            "gran_background_subsample_ratio": self._gran_background_subsample_ratio.value(),
            "gran_background_radius": self._gran_background_radius.value(),
            "glcm_channels": BaseStepPanel._checked_checkboxes(self._glcm_cbs) or None,
            "glcm_distances": glcm_d,
            "glcm_levels": self._glcm_levels.value(),
            "correlation_pairs": self.get_correlation_pairs(),
        }


class ImageProfilingStepPanel(BaseStepPanel):

    step_name = "image_profile"

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("Image Profiling")
        self._n_workers_value = default_n_workers()
        self._image_ch_cbs: List[QCheckBox] = []
        self._threshold_spins: Dict[str, QDoubleSpinBox] = {}
        self._build_controls()

    def _build_controls(self):
        self._image_grid = QWidget()
        self._image_grid_layout = QGridLayout(self._image_grid)
        self._image_grid_layout.setContentsMargins(0, 0, 0, 0)
        self._image_grid_layout.setHorizontalSpacing(4)
        self._image_grid_layout.setVerticalSpacing(2)
        self._image_ch_label = QLabel("Channels:")
        self._image_grid_layout.addWidget(self._image_ch_label, 0, 0)
        self._threshold_label = QLabel("Thresholds:")
        self._image_grid_layout.addWidget(self._threshold_label, 1, 0)
        self._image_ch_placeholder = QLabel("Load a dataset to configure")
        self._image_ch_placeholder.setProperty("class", "placeholder")
        self._image_grid_layout.addWidget(self._image_ch_placeholder, 0, 1, 1, -1)
        self._threshold_placeholder = QLabel("Load a dataset to configure")
        self._threshold_placeholder.setProperty("class", "placeholder")
        self._image_grid_layout.addWidget(self._threshold_placeholder, 1, 1, 1, -1)
        self._controls_layout.addWidget(self._image_grid)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clear_grid_section(layout, checkbox_list, row):
        i = 0
        while i < layout.count():
            item = layout.itemAt(i)
            w = item.widget()
            if isinstance(w, QCheckBox):
                r, _, _, _ = layout.getItemPosition(i)
                if r == row:
                    layout.removeItem(item)
                    w.deleteLater()
                    continue
            i += 1
        checkbox_list.clear()

    def _clear_thresholds(self):
        for w in self._threshold_spins.values():
            self._image_grid_layout.removeWidget(w)
            w.deleteLater()
        self._threshold_spins.clear()

    def _re_add_placeholder(self, placeholder_attr, layout, row=0, col=0, rowspan=1, colspan=1):
        placeholder = QLabel("Load a dataset to configure")
        placeholder.setProperty("class", "placeholder")
        if isinstance(layout, QGridLayout):
            layout.addWidget(placeholder, row, col, rowspan, colspan)
        else:
            layout.addWidget(placeholder)
        setattr(self, placeholder_attr, placeholder)

    def get_thresholds(self) -> Optional[Dict[str, float]]:
        result = {}
        for ch, w in self._threshold_spins.items():
            val = w.value()
            if val > 0:
                result[ch] = val
        return result or None

    def get_n_workers(self) -> int:
        return self._n_workers_value

    def set_n_workers(self, value: int) -> None:
        self._n_workers_value = value
        self.parameter_changed.emit()

    # ── Channel population ────────────────────────────────────────────

    def populate_channels(self, channels: List[str]) -> None:
        saved_thresholds = {}
        for ch, w in self._threshold_spins.items():
            saved_thresholds[ch] = w.value()

        # Save currently-checked channels before rebuild destroys widgets
        saved_image = set()
        for cb in self._image_ch_cbs:
            if cb.isChecked():
                saved_image.add(cb.text())

        # A config restored before the dataset loaded (Browse/from_config) —
        # re-apply it after rebuilding instead of stashing comma-joined text.
        pending = getattr(self, "_pending_settings", None)

        BaseStepPanel._remove_placeholder(self._image_grid_layout, "_image_ch_placeholder", self)
        BaseStepPanel._remove_placeholder(self._image_grid_layout, "_threshold_placeholder", self)
        self._clear_grid_section(self._image_grid_layout, self._image_ch_cbs, 0)
        self._clear_thresholds()

        if not channels:
            self._re_add_placeholder("_image_ch_placeholder", self._image_grid_layout, 0, 1, 1, -1)
            self._re_add_placeholder("_threshold_placeholder", self._image_grid_layout, 1, 1, 1, -1)
        else:
            if pending is not None:
                saved_channels = {c for c in pending.get("image_channels") or []}
                pending_thresholds = pending.get("image_thresholds") or {}
            else:
                saved_channels = saved_image
                pending_thresholds = {}
            for col_idx, ch in enumerate(channels):
                col = col_idx + 1
                cb = QCheckBox(ch)
                if pending is not None:
                    cb.setChecked(ch in saved_channels)
                elif saved_image:
                    cb.setChecked(ch in saved_image)
                else:
                    # Fresh default: NOTHING checked — the user picks the
                    # channels explicitly (an empty selection skips the step).
                    cb.setChecked(False)
                self._image_grid_layout.addWidget(cb, 0, col)
                self._image_ch_cbs.append(cb)
                self._wire_param_signal(cb)

                th_widget = QDoubleSpinBox()
                th_widget.setRange(-99999.0, 99999.0)
                th_widget.setDecimals(3)
                th_widget.setValue(0.0)
                th_widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
                th_widget.setMinimumWidth(70)
                th_widget.setMaximumWidth(90)
                th_key = f"threshold_{ch}"
                if ch in pending_thresholds:
                    th_widget.setValue(float(pending_thresholds[ch]))
                elif ch in saved_thresholds:
                    th_widget.setValue(saved_thresholds[ch])
                self._image_grid_layout.addWidget(th_widget, 1, col)
                self._threshold_spins[ch] = th_widget
                self._wire_param_signal(th_widget)

            self._image_grid_layout.setColumnStretch(len(channels) + 1, 1)

        if hasattr(self, "_pending_settings"):
            del self._pending_settings

    # ── Config building ───────────────────────────────────────────────

    def build_config_section(self) -> dict:
        image_ch = BaseStepPanel._checked_checkboxes(self._image_ch_cbs) or None
        return {
            "n_workers": self._n_workers_value,
            "image_channels": image_ch,
            "image_thresholds": self.get_thresholds(),
        }

    # ── Settings persistence ──────────────────────────────────────────

    def load_config_section(self, section) -> None:
        if not section:
            return
        if "n_workers" in section:
            self._n_workers_value = int(section["n_workers"])
        BaseStepPanel._set_checked_states(self._image_ch_cbs, section.get("image_channels"))
        thresholds = section.get("image_thresholds") or {}
        for ch, w in self._threshold_spins.items():
            if ch in thresholds:
                w.setValue(float(thresholds[ch]))
        # Keep the raw section for populate_channels: at restore time (Browse)
        # the channel grid doesn't exist yet — when the dataset loads,
        # populate_channels rebuilds it from this config instead of defaults.
        self._pending_settings = dict(section)


class ObjectProfilingStepPanel(BlockContainerPanel):

    step_name = "object_profile"
    _block_widget_class = ObjectProfileBlockWidget

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("Object Profiling")
        self._n_workers_value = default_n_workers()
        self._build_block_container("+ Add New Object Profiling")
        self._add_block_generic([])

    def _connect_block_signals(self, block: ObjectProfileBlockWidget) -> None:
        super()._connect_block_signals(block)
        self._wire_param_signal(block._object_mask)
        self._wire_param_signal(block._parent_mask)
        self._wire_param_signal(block._output_table)
        self._wire_param_signal(block._overwrite_db)
        self._wire_param_signal(block._radial_bins)
        self._wire_param_signal(block._gran_spectrum_length)
        self._wire_param_signal(block._gran_subsample_ratio)
        self._wire_param_signal(block._gran_background_subsample_ratio)
        self._wire_param_signal(block._gran_background_radius)
        self._wire_param_signal(block._glcm_distances)
        self._wire_param_signal(block._glcm_levels)
        for cb_list in (block._intensity_cbs, block._radial_cbs, block._gran_cbs,
                        block._glcm_cbs, block._corr_cbs):
            for cb in cb_list:
                self._wire_param_signal(cb)

    def _on_add_block_clicked(self) -> None:
        channels = list(self._blocks[0]._channels) if self._blocks else []
        block = self._add_block_generic(channels)
        # Copy settings from first block
        if len(self._blocks) > 1 and self._blocks[0] is not block:
            src = self._blocks[0]
            block._object_mask.clear()
            for i in range(src._object_mask.count()):
                block._object_mask.addItem(src._object_mask.itemText(i))
            block._object_mask.setCurrentIndex(src._object_mask.currentIndex())
            block._parent_mask.clear()
            for i in range(src._parent_mask.count()):
                block._parent_mask.addItem(src._parent_mask.itemText(i))
            block._parent_mask.setCurrentIndex(src._parent_mask.currentIndex())
            block._output_table.setText(src._output_table.text())
            block._table_synced = src._table_synced
            for src_cbs, dst_cbs in [
                (src._intensity_cbs, block._intensity_cbs),
                (src._radial_cbs, block._radial_cbs),
                (src._gran_cbs, block._gran_cbs),
                (src._glcm_cbs, block._glcm_cbs),
                (src._corr_cbs, block._corr_cbs),
            ]:
                checked = {cb.text() for cb in src_cbs if cb.isChecked()}
                for cb in dst_cbs:
                    cb.setChecked(cb.text() in checked)
            block._radial_bins.setValue(src._radial_bins.value())
            block._gran_spectrum_length.setValue(src._gran_spectrum_length.value())
            block._gran_subsample_ratio.setValue(src._gran_subsample_ratio.value())
            block._gran_background_subsample_ratio.setValue(src._gran_background_subsample_ratio.value())
            block._gran_background_radius.setValue(src._gran_background_radius.value())
            block._glcm_distances.setText(src._glcm_distances.text())
            block._glcm_levels.setValue(src._glcm_levels.value())

    def get_n_workers(self) -> int:
        return self._n_workers_value

    def set_n_workers(self, value: int) -> None:
        self._n_workers_value = value
        self.parameter_changed.emit()

    # n_workers is an extra top-level section key carried by this panel.
    def _extra_config_items(self) -> dict:
        return {"n_workers": self._n_workers_value}

    def _apply_extra_config_items(self, section: dict) -> None:
        nw = section.get("n_workers")
        if nw is not None:
            self._n_workers_value = int(nw)

    def get_mask_name(self) -> str:
        if self._blocks:
            return self._blocks[0].get_mask_name()
        return ""

    def get_parent_mask_name(self) -> Optional[str]:
        if self._blocks:
            return self._blocks[0].get_parent_mask_name()
        return None

    def _apply_block_config(self, block: ObjectProfileBlockWidget, cfg: dict) -> None:
        mask_name = cfg.get("mask_name", "")
        if mask_name:
            idx = block._object_mask.findText(str(mask_name))
            if idx >= 0:
                block._object_mask.setCurrentIndex(idx)
        parent_name = cfg.get("parent_mask_name", "")
        if parent_name:
            idx = block._parent_mask.findText(str(parent_name))
            if idx >= 0:
                block._parent_mask.setCurrentIndex(idx)
        table_name = cfg.get("output_table_name", "")
        if table_name:
            block._output_table.setText(str(table_name))
            block._table_synced = False
        if hasattr(block, "_overwrite_db"):
            block._overwrite_db.setChecked(bool(cfg.get("overwrite_db", False)))
        BaseStepPanel._set_widget(block._radial_bins, cfg.get("radial_bins", 4), "radial_bins")
        BaseStepPanel._set_widget(block._gran_spectrum_length, cfg.get("gran_spectrum_length", 8), "gran_spectrum_length")
        BaseStepPanel._set_widget(block._gran_subsample_ratio, cfg.get("gran_subsample_ratio", 0.5), "gran_subsample_ratio")
        BaseStepPanel._set_widget(block._gran_background_subsample_ratio, cfg.get("gran_background_subsample_ratio", 0.25), "gran_background_subsample_ratio")
        BaseStepPanel._set_widget(block._gran_background_radius, cfg.get("gran_background_radius", 10), "gran_background_radius")
        glcm_d = cfg.get("glcm_distances", "")
        if glcm_d:
            if isinstance(glcm_d, list):
                block._glcm_distances.setText(",".join(str(d) for d in glcm_d))
            else:
                block._glcm_distances.setText(str(glcm_d))
        BaseStepPanel._set_widget(block._glcm_levels, cfg.get("glcm_levels", 256), "glcm_levels")
        BaseStepPanel._set_checked_states(block._intensity_cbs, cfg.get("intensity_channels"))
        BaseStepPanel._set_checked_states(block._radial_cbs, cfg.get("radial_channels"))
        BaseStepPanel._set_checked_states(block._gran_cbs, cfg.get("gran_channels"))
        BaseStepPanel._set_checked_states(block._glcm_cbs, cfg.get("glcm_channels"))
        corr_pairs = cfg.get("correlation_pairs") or []
        for cb in block._corr_cbs:
            pair = cb.text().split("-")
            cb.setChecked(pair in corr_pairs)
