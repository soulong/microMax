from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from microProfiler.gui.panels.base_step_panel import BaseStepPanel
from microProfiler.gui.panels._block_container import BlockContainerPanel

logger = logging.getLogger(__name__)

DEFAULT_MAX_VALUE = 65535.0

_COLOR_BY_OPTIONS = ["pred_class", "directory", "pred_prob", "ground_truth"]


def _hsep():
    s = QFrame()
    s.setFrameShape(QFrame.HLine)
    s.setFrameShadow(QFrame.Sunken)
    s.setProperty("class", "separator")
    return s


def read_bundle_meta(model_path: str) -> dict:
    """Load a microModel bundle's meta dict (lazy torch import).

    Raises ImportError when microModel/torch are missing and RuntimeError for
    anything that is not a microModel bundle — callers surface these as popups.
    """
    import importlib.util
    if importlib.util.find_spec("microModel") is None:
        raise ImportError(
            "Inference requires the 'microModel' package, which is not "
            "installed. Install microModel and restart microProfiler."
        )
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Inference requires 'torch' (installed with microModel), which is "
            "not installed."
        ) from e
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise RuntimeError("Not a microModel bundle (missing 'state_dict').")
    meta = bundle.get("meta") or {}
    if not isinstance(meta, dict):
        raise RuntimeError("Bundle 'meta' is not a dict.")
    return meta


class InferenceBlockWidget(QWidget):

    def __init__(
        self,
        block_index: int,
        channels: List[str],
        on_remove: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.block_index = block_index
        self._on_remove = on_remove
        self._channels = list(channels)
        self.setProperty("class", "block-card")
        self._classify_capable: Optional[bool] = None
        self._capability_checked_path: Optional[str] = None
        self._color_by_auto = True
        self._max_value = DEFAULT_MAX_VALUE
        self._var_threshold = 0.95
        self._stored_channels: Optional[List[str]] = None
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # Row 1: model path + browse + remove
        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model:"))
        self._model_path = QLineEdit()
        self._model_path.setObjectName("checkpoint_path")
        self._model_path.setPlaceholderText("Path to a microModel bundle (*.pt)")
        self._model_path.setToolTip(
            "Trained microModel bundle (SSL pretrain model.pt or train model.pt)")
        row_model.addWidget(self._model_path, 1)
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setProperty("class", "secondary")
        self._browse_btn.setToolTip("Pick a bundle; capabilities are read from its meta")
        row_model.addWidget(self._browse_btn)
        self._remove_btn = QPushButton("✕ Remove")
        self._remove_btn.setProperty("class", "danger")
        self._remove_btn.setToolTip("Remove this inference block")
        row_model.addWidget(self._remove_btn)
        if self._on_remove:
            self._remove_btn.clicked.connect(self._on_remove)
        layout.addLayout(row_model)

        # Row 1b: model info (from the bundle meta) — always visible between
        # the model selection and the mask row; "—" until a bundle is read.
        row_info = QHBoxLayout()
        self._model_type_label = QLabel("Model type: —")
        self._model_type_label.setToolTip(
            "SL - Classify = train bundle (predictions available); "
            "SSL - Features = pretrain bundle (features only)")
        row_info.addWidget(self._model_type_label)
        row_info.addSpacing(10)
        self._model_ch_info_label = QLabel("Input Channel Number: —")
        self._model_ch_info_label.setToolTip(
            "Number of input channels the model was trained with (bundle meta)")
        row_info.addWidget(self._model_ch_info_label)
        row_info.addStretch()
        layout.addLayout(row_info)

        # Row 2: mask + channels (checkable row, ◀/▶ to reorder)
        row_mask = QHBoxLayout()
        self._mask_row = row_mask
        row_mask.addWidget(QLabel("Mask:"))
        self._mask_combo = QComboBox()
        self._mask_combo.setEditable(True)
        self._mask_combo.setToolTip("Mask type to infer objects from (dataset mask columns)")
        row_mask.addWidget(self._mask_combo)
        row_mask.addSpacing(10)
        row_mask.addWidget(QLabel("Channels:"))
        self._channels_row = QHBoxLayout()
        self._channels_row.setSpacing(4)
        self._channels_row.setContentsMargins(0, 0, 0, 0)
        row_mask.addLayout(self._channels_row)
        self._ch_cbs: List[QCheckBox] = []
        self._chan_placeholder: Optional[QLabel] = None
        self._move_left_btn = QPushButton("◀")
        self._move_left_btn.setProperty("class", "secondary")
        self._move_left_btn.setFixedWidth(24)
        self._move_left_btn.setToolTip(
            "Move the selected channel left (earlier in the model's input "
            "channel order). Click a channel first.")
        self._move_left_btn.setEnabled(False)
        self._move_left_btn.clicked.connect(lambda: self._move_channel(-1))
        self._move_right_btn = QPushButton("▶")
        self._move_right_btn.setProperty("class", "secondary")
        self._move_right_btn.setFixedWidth(24)
        self._move_right_btn.setToolTip(
            "Move the selected channel right (later in the model's input "
            "channel order). Click a channel first.")
        self._move_right_btn.setEnabled(False)
        self._move_right_btn.clicked.connect(lambda: self._move_channel(1))
        row_mask.addWidget(self._move_left_btn)
        row_mask.addWidget(self._move_right_btn)
        row_mask.addStretch()
        if self._channels:
            self.set_channel_state(list(self._channels), set())
        else:
            self._chan_placeholder = QLabel("Load a dataset to configure")
            self._chan_placeholder.setProperty("class", "placeholder")
            self._channels_row.addWidget(self._chan_placeholder)
        layout.addLayout(row_mask)

        # Row 3: outputs + DB + max_value
        layout.addWidget(_hsep())
        row_out = QHBoxLayout()
        self._feature_cb = QCheckBox("feature")
        self._feature_cb.setChecked(True)
        self._feature_cb.setToolTip("Write the backbone feature vectors (features BLOB)")
        row_out.addWidget(self._feature_cb)
        self._pred_class_cb = QCheckBox("pred_class")
        self._pred_class_cb.setChecked(True)
        self._pred_class_cb.setToolTip("Write the predicted class (requires a train bundle)")
        row_out.addWidget(self._pred_class_cb)
        self._pred_prob_cb = QCheckBox("pred_prob")
        self._pred_prob_cb.setChecked(True)
        self._pred_prob_cb.setToolTip("Write the argmax softmax probability (requires pred_class)")
        row_out.addWidget(self._pred_prob_cb)
        row_out.addSpacing(10)
        row_out.addWidget(QLabel("DB:"))
        self._output_db = QLineEdit("infer.db")
        self._output_db.setMaximumWidth(150)
        self._output_db.setToolTip("SQLite DB file name written under the dataset dir (table names are fixed)")
        row_out.addWidget(self._output_db)
        row_out.addSpacing(10)
        row_out.addWidget(QLabel("max_value:"))
        self._max_value_label = QLabel(f"{DEFAULT_MAX_VALUE:g}")
        self._max_value_label.setToolTip(
            "Maximum possible intensity of the input dtype, read from config "
            "(65535 for 16-bit, 255 for 8-bit). Trusted as configured.")
        row_out.addWidget(self._max_value_label)
        row_out.addStretch()
        layout.addLayout(row_out)

        # Row 4: reduction group
        layout.addWidget(_hsep())
        self._reduction_group = QGroupBox("Dimension reduction (PCA + UMAP)")
        self._reduction_group.setCheckable(True)
        self._reduction_group.setChecked(False)
        self._reduction_group.setToolTip(
            "Fit PCA (0.95 cumulative variance) + UMAP on the features and write "
            "reduction_pca / reduction_umap / reduction_pca_variance tables. "
            "No plots are produced. Provided reducers transform directly; "
            "otherwise reducers are fit (and saved) per dataset.")
        red_layout = QVBoxLayout(self._reduction_group)
        row_red = QHBoxLayout()
        row_red.addWidget(QLabel("PCA reducer:"))
        self._reducer_pca_path = QLineEdit()
        self._reducer_pca_path.setObjectName("reducer_path")
        self._reducer_pca_path.setPlaceholderText("reducer_pca.pkl (optional — fit if empty)")
        row_red.addWidget(self._reducer_pca_path, 1)
        self._pca_browse_btn = QPushButton("Browse...")
        self._pca_browse_btn.setProperty("class", "secondary")
        row_red.addWidget(self._pca_browse_btn)
        row_red.addSpacing(10)
        row_red.addWidget(QLabel("UMAP reducer:"))
        self._reducer_umap_path = QLineEdit()
        self._reducer_umap_path.setObjectName("reducer_path")
        self._reducer_umap_path.setPlaceholderText("reducer_umap.pkl (optional — fit if empty)")
        row_red.addWidget(self._reducer_umap_path, 1)
        self._umap_browse_btn = QPushButton("Browse...")
        self._umap_browse_btn.setProperty("class", "secondary")
        row_red.addWidget(self._umap_browse_btn)
        red_layout.addLayout(row_red)
        row_red2 = QHBoxLayout()
        row_red2.addWidget(QLabel("Color by:"))
        self._color_by = QComboBox()
        self._color_by.addItems(_COLOR_BY_OPTIONS)
        self._color_by.setToolTip("Stratification column for reducer fitting (pred_class for SL, directory for SSL)")
        row_red2.addWidget(self._color_by)
        row_red2.addSpacing(10)
        row_red2.addWidget(QLabel("Samples per class:"))
        self._sample_per_class = QSpinBox()
        self._sample_per_class.setRange(1, 1000000)
        self._sample_per_class.setValue(10000)
        self._sample_per_class.setFixedWidth(80)
        self._sample_per_class.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._sample_per_class.setToolTip("Stratified sample size used to FIT the reducers (0 = no sampling)")
        row_red2.addWidget(self._sample_per_class)
        row_red2.addStretch()
        red_layout.addLayout(row_red2)
        layout.addWidget(self._reduction_group)

        # ── behavior wiring ──
        self._browse_btn.clicked.connect(self._on_browse)
        self._pca_browse_btn.clicked.connect(self._on_browse_pca)
        self._umap_browse_btn.clicked.connect(self._on_browse_umap)
        self._pred_class_cb.toggled.connect(self._on_pred_class_toggled)
        self._on_pred_class_toggled(self._pred_class_cb.isChecked())
        self._color_by.currentIndexChanged.connect(self._on_color_by_changed)

    def _on_color_by_changed(self, *_):
        self._color_by_auto = False

    def _on_pred_class_toggled(self, checked: bool) -> None:
        # microModel writes pred_prob only when pred_class is on — gate the
        # checkbox instead of silently dropping the output.
        self._pred_prob_cb.setEnabled(checked)
        if not checked:
            self._pred_prob_cb.setChecked(False)

    def _update_capability_ui(self) -> None:
        if self._classify_capable is None:
            self._pred_class_cb.setEnabled(True)
            return
        if self._classify_capable:
            self._pred_class_cb.setEnabled(True)
            if self._color_by_auto:
                self._color_by.setCurrentText("pred_class")
                self._color_by_auto = False
        else:
            self._pred_class_cb.setChecked(False)
            self._pred_class_cb.setEnabled(False)
            if self._color_by_auto:
                self._color_by.setCurrentText("directory")
                self._color_by_auto = False

    # ── Capability (bundle meta) ────────────────────────────────────────

    def _update_model_info(self, meta: Optional[dict] = None) -> None:
        """Refresh the always-visible model info row from the bundle meta."""
        if not meta:
            self._model_type_label.setText("Model type: —")
            self._model_ch_info_label.setText("Input Channel Number: —")
            return
        is_classify = "num_classes" in meta
        self._model_type_label.setText(
            "Model type: SL - Classify" if is_classify else "Model type: SSL - Features")
        in_chans = meta.get("in_chans")
        ch_text = str(in_chans) if in_chans is not None else "—"
        self._model_ch_info_label.setText(f"Input Channel Number: {ch_text}")

    def ensure_capability(self) -> Optional[str]:
        """Read the bundle meta (once per path) and gate the output checkboxes.

        Returns an error string on failure (caller shows a popup) or None.
        """
        path = self.get_model_path()
        if not path:
            self._update_model_info(None)
            return None
        if self._capability_checked_path == path:
            return None
        try:
            meta = read_bundle_meta(path)
        except ImportError as e:
            self._update_model_info(None)
            return str(e)
        except Exception as e:
            self._update_model_info(None)
            return f"Could not read model bundle {path}:\n{e}"
        self._classify_capable = "num_classes" in meta
        self._capability_checked_path = path
        self._update_model_info(meta)
        self._update_capability_ui()
        return None

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Bundle", "",
            "PyTorch bundles (*.pt);;All files (*)",
        )
        if not path:
            return
        self._model_path.setText(path)
        err = self.ensure_capability()
        if err:
            QMessageBox.warning(self, "Model Bundle", err)

    def _on_browse_pca(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-fitted PCA Reducer", "",
            "Pickle files (*.pkl);;All files (*)",
        )
        if path:
            self._reducer_pca_path.setText(path)

    def _on_browse_umap(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-fitted UMAP Reducer", "",
            "Pickle files (*.pkl);;All files (*)",
        )
        if path:
            self._reducer_umap_path.setText(path)

    # ── Accessors ───────────────────────────────────────────────────────

    def get_model_path(self) -> str:
        return self._model_path.text().strip()

    def get_output_db(self) -> str:
        return self._output_db.text().strip() or "infer.db"

    def get_mask_name(self) -> str:
        return self._mask_combo.currentText().strip() or ""

    def get_max_value(self) -> float:
        return self._max_value

    def set_max_value(self, value: float) -> None:
        self._max_value = float(value)
        self._max_value_label.setText(f"{self._max_value:g}")

    def get_checked_channels(self) -> List[str]:
        """Checked channels in row order (left-to-right).

        The order is meaningful: it maps 1:1 to the model's input channel
        order — click a channel, then ◀/▶ to match the trained model's
        channels.
        """
        return [cb.text() for cb in self._ch_cbs if cb.isChecked()]

    def set_channel_state(self, order: List[str], checked) -> None:
        """Rebuild the channel checkboxes with an explicit order + checked set."""
        for cb in self._ch_cbs:
            self._channels_row.removeWidget(cb)
            cb.deleteLater()
        self._ch_cbs = []
        for name in order:
            cb = QCheckBox(name)
            cb.setChecked(name in checked)
            cb.pressed.connect(lambda ch=cb: self._on_channel_pressed(ch))
            cb.setToolTip(
                "Check to include this channel in the model input; click it "
                "to select it, then use ◀/▶ to change its position.")
            self._channels_row.addWidget(cb)
            self._ch_cbs.append(cb)
        self._order_current: Optional[QCheckBox] = None
        self._update_move_buttons()

    # ── Channel reorder (◀/▶) ──────────────────────────────────────────

    def _on_channel_pressed(self, cb: QCheckBox) -> None:
        self._order_current = cb
        self._update_move_buttons()

    def _update_move_buttons(self) -> None:
        cur = self._order_current
        enabled_left = cur is not None and cur in self._ch_cbs and self._ch_cbs.index(cur) > 0
        enabled_right = (
            cur is not None and cur in self._ch_cbs
            and self._ch_cbs.index(cur) < len(self._ch_cbs) - 1
        )
        self._move_left_btn.setEnabled(enabled_left)
        self._move_right_btn.setEnabled(enabled_right)

    def _move_channel(self, delta: int) -> None:
        cur = self._order_current
        if cur is None or cur not in self._ch_cbs:
            return
        i = self._ch_cbs.index(cur)
        j = i + delta
        if j < 0 or j >= len(self._ch_cbs):
            return
        self._ch_cbs[i], self._ch_cbs[j] = self._ch_cbs[j], self._ch_cbs[i]
        for cb in self._ch_cbs:
            self._channels_row.removeWidget(cb)
        for k, cb in enumerate(self._ch_cbs):
            self._channels_row.insertWidget(k, cb)
        self._update_move_buttons()

    def is_reduction_enabled(self) -> bool:
        return self._reduction_group.isChecked()

    # ── Population ──────────────────────────────────────────────────────

    def populate_channels(self, channels: List[str]) -> None:
        self._channels = list(channels)
        if channels:
            BaseStepPanel._remove_placeholder(self._channels_row, "_chan_placeholder", self)
            stored = self._stored_channels
            self._stored_channels = None
            if stored is not None:
                # Config order wins: stored (ordered, checked) channels first,
                # then dataset channels not covered by the stored list.
                order = [c for c in stored if c in channels] + [c for c in channels if c not in stored]
                checked = set(stored)
            else:
                # Repopulation: preserve the user's current order + checks.
                current = [cb.text() for cb in self._ch_cbs]
                checked = {cb.text() for cb in self._ch_cbs if cb.isChecked()}
                order = [c for c in current if c in channels] + [c for c in channels if c not in current]
            self.set_channel_state(order, checked)
        else:
            for cb in self._ch_cbs:
                self._channels_row.removeWidget(cb)
                cb.deleteLater()
            self._ch_cbs = []
            self._update_move_buttons()
            if self._chan_placeholder is None:
                self._chan_placeholder = QLabel("Load a dataset to configure")
                self._chan_placeholder.setProperty("class", "placeholder")
                self._channels_row.addWidget(self._chan_placeholder)

    def populate_masks(self, mask_names: List[str]) -> None:
        stripped = [n.removeprefix("mask_") for n in mask_names]
        current = self._mask_combo.currentText()
        self._mask_combo.blockSignals(True)
        self._mask_combo.clear()
        self._mask_combo.addItems(stripped)
        self._mask_combo.blockSignals(False)
        if current and current in stripped:
            self._mask_combo.setCurrentText(current)
        elif stripped:
            # Fresh/stale selection: default to the first mask — the segment
            # Object name (e.g. 'cell') thanks to the sync ordering.
            self._mask_combo.setCurrentIndex(0)

    # ── Config serialization ────────────────────────────────────────────

    def build_config_section(self) -> dict:
        section = {
            "model": self.get_model_path(),
            "mask_name": self.get_mask_name() or None,
            "channels": self.get_checked_channels() or None,
            "feature": self._feature_cb.isChecked(),
            "pred_class": self._pred_class_cb.isChecked(),
            "pred_prob": self._pred_prob_cb.isChecked(),
            "output_db": self.get_output_db(),
            "max_value": self._max_value,
        }
        if self._reduction_group.isChecked():
            section["reduction"] = {
                "enabled": True,
                "var_threshold": self._var_threshold,
                "color_by": self._color_by.currentText(),
                "sample_per_class": self._sample_per_class.value(),
                "reducer_pca": self._reducer_pca_path.text().strip() or None,
                "reducer_umap": self._reducer_umap_path.text().strip() or None,
            }
        return section


class InferenceStepPanel(BlockContainerPanel):

    step_name = "inference"
    _block_widget_class = InferenceBlockWidget

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("Inference (microModel)")
        self._last_channels: List[str] = []
        self._last_masks: List[str] = []
        self._pending_block_configs: List[dict] = []
        self._default_max_value = DEFAULT_MAX_VALUE
        self._build_block_container("+ Add New Inference Block")
        self._add_block_generic([])

    def _connect_block_signals(self, block: InferenceBlockWidget) -> None:
        super()._connect_block_signals(block)
        for w in (block._model_path, block._output_db, block._mask_combo,
                  block._feature_cb, block._pred_class_cb, block._pred_prob_cb,
                  block._color_by, block._sample_per_class,
                  block._reducer_pca_path, block._reducer_umap_path):
            self._wire_param_signal(w)
        block._reduction_group.toggled.connect(
            self.parameter_changed, Qt.UniqueConnection)
        # Channel row: check/uncheck AND ◀/▶ reorder both persist to config.
        for cb in block._ch_cbs:
            cb.toggled.connect(self.parameter_changed, Qt.UniqueConnection)
        block._move_left_btn.clicked.connect(
            self.parameter_changed, Qt.UniqueConnection)
        block._move_right_btn.clicked.connect(
            self.parameter_changed, Qt.UniqueConnection)

    def _on_add_block_clicked(self) -> None:
        channels = list(self._blocks[0]._channels) if self._blocks else []
        block = self._add_block_generic(channels)
        if len(self._blocks) > 1 and self._blocks[0] is not block:
            src = self._blocks[0]
            block._model_path.setText(src.get_model_path())
            # Refresh the copied block's model info row from the bundle meta
            # (silent: a stale/unreadable copied path must not block adding).
            try:
                block.ensure_capability()
            except Exception:
                pass
            block._output_db.setText(src.get_output_db())
            block.set_max_value(src.get_max_value())
            block._feature_cb.setChecked(src._feature_cb.isChecked())
            block._pred_class_cb.setChecked(src._pred_class_cb.isChecked())
            block._pred_prob_cb.setChecked(src._pred_prob_cb.isChecked())
            block._mask_combo.clear()
            for i in range(src._mask_combo.count()):
                block._mask_combo.addItem(src._mask_combo.itemText(i))
            block._mask_combo.setCurrentText(src.get_mask_name())
            block._reduction_group.setChecked(src.is_reduction_enabled())
            block._reducer_pca_path.setText(src._reducer_pca_path.text())
            block._reducer_umap_path.setText(src._reducer_umap_path.text())
            block._color_by.setCurrentText(src._color_by.currentText())
            block._sample_per_class.setValue(src._sample_per_class.value())
            if src._ch_cbs:
                src_order = [cb.text() for cb in src._ch_cbs]
                block.set_channel_state(src_order, set(src.get_checked_channels()))
            else:
                # Dataset not loaded yet — carry the source's selection so it
                # applies when populate_channels runs after the load.
                block._stored_channels = src.get_checked_channels()
        else:
            block.set_max_value(self._default_max_value)

    def populate_channels(self, channels: List[str]) -> None:
        self._last_channels = list(channels)
        self._channels = list(channels)
        for block in self._blocks:
            # The block preserves its current order + checked state across
            # repopulation (and applies stored config order when restoring).
            block.populate_channels(channels)
            self._connect_block_signals(block)
        self._pending_block_configs = []

    def populate_masks(self, mask_names: List[str]) -> None:
        self._last_masks = list(mask_names)
        pending = getattr(self, "_pending_block_configs", None)
        for i, block in enumerate(self._blocks):
            block.populate_masks(mask_names)
            if pending and i < len(pending):
                saved = pending[i].get("mask_name")
                if saved:
                    block._mask_combo.setCurrentText(str(saved).removeprefix("mask_"))

    def load_config_section(self, sections: Any) -> None:
        if not sections:
            return
        if isinstance(sections, dict):
            sections = [sections]
        if not isinstance(sections, (list, tuple)):
            return
        self._pending_block_configs = [cfg for cfg in sections if isinstance(cfg, dict)]
        self._remove_all_blocks()
        self._blocks_layout.removeItem(self._add_btn_layout)

        last_channels = self._last_channels or self._channels
        last_masks = getattr(self, "_last_masks", [])

        for cfg in sections:
            if not isinstance(cfg, dict):
                continue
            if self._blocks:
                self._blocks_layout.addSpacing(4)
            block = self._block_widget_class(len(self._blocks), last_channels, parent=self._block_container)
            self._connect_block_signals(block)
            if last_masks:
                block.populate_masks(last_masks)
            self._apply_block_config(block, cfg)
            self._compact_block(block)
            self._blocks.append(block)
            self._blocks_layout.addWidget(block)

        self._blocks_layout.addLayout(self._add_btn_layout)
        if last_channels:
            self.populate_channels(last_channels)
        self.parameter_changed.emit()

    def _apply_block_config(self, block: InferenceBlockWidget, cfg: dict) -> None:
        model = cfg.get("model", "")
        if model:
            block._model_path.setText(str(model))
        db_name = cfg.get("output_db", "")
        if db_name:
            block._output_db.setText(str(db_name))
        max_value = cfg.get("max_value")
        if max_value:
            block.set_max_value(float(max_value))
        BaseStepPanel._set_widget(block._feature_cb, cfg.get("feature", True), "feature")
        BaseStepPanel._set_widget(block._pred_class_cb, cfg.get("pred_class", True), "pred_class")
        BaseStepPanel._set_widget(block._pred_prob_cb, cfg.get("pred_prob", True), "pred_prob")
        mask_name = cfg.get("mask_name", "")
        if mask_name:
            mask_name = str(mask_name).removeprefix("mask_")
            idx = block._mask_combo.findText(mask_name)
            if idx >= 0:
                block._mask_combo.setCurrentIndex(idx)
            else:
                block._mask_combo.setCurrentText(mask_name)
        stored = cfg.get("channels")
        if stored:
            block._stored_channels = list(stored)
        red = cfg.get("reduction") or {}
        block._reduction_group.setChecked(bool(red.get("enabled", False)))
        # var_threshold has no GUI widget — keep the config value verbatim so
        # a YAML with var_threshold: 0.99 survives a GUI round-trip.
        block._var_threshold = float(red.get("var_threshold", 0.95))
        if red.get("reducer_pca"):
            block._reducer_pca_path.setText(str(red["reducer_pca"]))
        if red.get("reducer_umap"):
            block._reducer_umap_path.setText(str(red["reducer_umap"]))
        BaseStepPanel._set_widget(block._color_by, red.get("color_by", "pred_class"), "color_by")
        BaseStepPanel._set_widget(block._sample_per_class, red.get("sample_per_class", 10000), "sample_per_class")
        block._color_by_auto = False

    def build_config_section(self) -> list:  # type: ignore[override]
        return [b.build_config_section() for b in self._blocks]

    def set_dataset_dtype(self, dtype) -> None:
        """Refresh per-block max_value defaults from the dataset dtype.

        Config values are always trusted (never overwritten); blocks still at
        the untouched default adopt the dtype-derived value.
        """
        derived = _max_value_for_dtype(dtype)
        self._default_max_value = derived
        for block in self._blocks:
            if block.get_max_value() == DEFAULT_MAX_VALUE:
                block.set_max_value(derived)

    def max_value_mismatches(self, dtype) -> List[str]:
        """Config max_value vs dataset dtype — non-blocking warnings (Q12)."""
        derived = _max_value_for_dtype(dtype)
        out = []
        for block in self._blocks:
            if not block._model_path.text().strip():
                continue
            if block.get_max_value() != derived:
                out.append(
                    f"- model '{block._model_path.text().strip()}': configured "
                    f"max_value={block.get_max_value():g} but dataset dtype "
                    f"{dtype} implies {derived:g}. The configured value is "
                    f"used as-is — check the config/session.yml if "
                    f"unexpected."
                )
        return out

    def validate_blocks(self) -> Optional[str]:
        """Pre-Run validation. Returns an error string or None."""
        seen: dict = {}
        for block in self._blocks:
            model = block.get_model_path()
            if not model:
                return "Every inference block needs a model bundle path."
            db = block.get_output_db()
            if db in seen:
                return (
                    f"Duplicate output_db name '{db}' in inference blocks. "
                    f"Rename one of them."
                )
            seen[db] = True
            if block.is_reduction_enabled() and not block._feature_cb.isChecked():
                return (
                    "Dimension reduction requires the 'feature' output. "
                    "Check 'feature' in the block before running."
                )
            if block._ch_cbs and not block.get_checked_channels():
                return (
                    "Select at least one channel in every inference block. "
                    "The channel order (click a channel, then ◀/▶) must match "
                    "the trained model's input channel order."
                )
            err = block.ensure_capability()
            if err:
                return err
        return None


def _max_value_for_dtype(dtype) -> float:
    """Map a dataset img_dtype to the max possible intensity.

    Accepts both np.dtype instances and numpy scalar type classes.
    """
    if dtype is None:
        return DEFAULT_MAX_VALUE
    try:
        name = getattr(np.dtype(dtype), "name", None) or str(dtype)
    except Exception:
        name = str(dtype)
    if name in ("uint8", "int8"):
        return 255.0
    return DEFAULT_MAX_VALUE
