from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from microBase.db_contracts import INFER_DB_NAME

from microProfiler.gui.dpi import dp
from microProfiler.gui.panels.base_step_panel import BaseStepPanel, make_hsep
from microProfiler.gui.panels._block_container import BlockContainerPanel
from microProfiler.gui.path_drop import enable_path_drop
from microProfiler.pipeline._micromodel_bridge import read_bundle_meta
from microProfiler.user_defaults import get_user_defaults

logger = logging.getLogger(__name__)

DEFAULT_MAX_VALUE = 65535.0

# DR methods offered in the Dimension-reduction group (canonical order).
_DR_METHOD_OPTIONS = ["pca", "umap", "pacmap", "localmap"]

# Image dtype -> max possible intensity (matches microVis's DTYPE_MAX).
_DTYPE_MAX = {
    "uint8": 255.0,
    "int8": 127.0,
    "uint16": 65535.0,
    "uint32": 4294967295.0,
    "float32": 1.0,
    "float64": 1.0,
}


class InferenceBlockWidget(QWidget):

    def __init__(
        self,
        block_index: int,
        channels: List[str],
        parent=None,
    ):
        super().__init__(parent)
        self.block_index = block_index
        self._channels = list(channels)
        self.setProperty("class", "block-card")
        self._classify_capable: Optional[bool] = None
        self._capability_checked_path: Optional[str] = None
        self._max_value = DEFAULT_MAX_VALUE
        # True once a max_value came from an explicit source (config restore
        # or block copy) — the dataset-dtype default must never overwrite it.
        self._max_value_explicit = False
        # No-widget reduction keys, round-tripped verbatim from YAML.
        # color_by / sample_per_class only shape the reducer fit and the
        # PDF plots (which this pipeline never writes) — the tables always
        # contain every object, so neither has a GUI widget.
        self._color_by = "pred_class"
        self._cluster_res = None
        self._sample_per_class = 10000
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)
        # Rows stretch to the full card width (no AlignLeft) so the path
        # boxes — not the buttons — absorb the extra space.
        layout.setAlignment(Qt.AlignTop)

        # Common width for row-leading labels: every input starts at the
        # same x, right after its label column.
        label_w = dp(80)

        def _row_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setFixedWidth(label_w)
            return lbl

        def _browse_button() -> QPushButton:
            btn = QPushButton("Browse...")
            btn.setToolTip("Browse... (you can also drag a file onto the input box)")
            return btn

        # Row 1: model path + browse (the path box absorbs all extra width,
        # so Browse sits right after it and Remove ends up right-aligned)
        row_model = QHBoxLayout()
        row_model.addWidget(_row_label("Model:"))
        self._model_path = QLineEdit()
        self._model_path.setObjectName("checkpoint_path")
        self._model_path.setPlaceholderText("Path to a microModel bundle (*.pt)")
        self._model_path.setToolTip(
            "Trained microModel bundle (SSL pretrain model.pt or train model.pt)")
        enable_path_drop(self._model_path, on_path=lambda _: self.ensure_capability())
        # Path box = 2/3 of the free width; Browse follows it immediately.
        row_model.addWidget(self._model_path, 2)
        self._browse_btn = _browse_button()
        self._browse_btn.setToolTip(
            "Pick a bundle; capabilities are read from its meta")
        row_model.addWidget(self._browse_btn)
        row_model.addStretch(1)
        # Gap before the destructive action at the right edge.
        row_model.addSpacing(12)
        self._remove_btn = QPushButton("✕ Remove")
        self._remove_btn.setToolTip("Remove this inference block")
        row_model.addWidget(self._remove_btn)
        layout.addLayout(row_model)

        # Row 2: mask + channels (checkable row, ◀/▶ to reorder)
        row_mask = QHBoxLayout()
        self._mask_row = row_mask
        row_mask.addWidget(_row_label("Mask:"))
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
        self._move_left_btn.setFixedWidth(24)
        self._move_left_btn.setToolTip(
            "Move the selected channel left (earlier in the model's input "
            "channel order). Click a channel first.")
        self._move_left_btn.setEnabled(False)
        self._move_left_btn.clicked.connect(lambda: self._move_channel(-1))
        self._move_right_btn = QPushButton("▶")
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
        row_out = QHBoxLayout()
        self._feature_cb = QCheckBox("feature")
        self._feature_cb.setChecked(True)
        self._feature_cb.setToolTip("Write the backbone feature vectors (features BLOB)")
        row_out.addWidget(self._feature_cb)
        self._pred_class_cb = QCheckBox("pred_class")
        self._pred_class_cb.setChecked(True)
        self._pred_class_cb.setToolTip("Write the predicted class (requires a train bundle)")
        row_out.addWidget(self._pred_class_cb)
        row_out.addSpacing(10)
        row_out.addWidget(QLabel("DB:"))
        self._output_db = QLineEdit(INFER_DB_NAME)
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

        # Row 4: dimension reduction — NO checkbox: a reducer file OR at
        # least one checked DR method means the stage runs.
        layout.addWidget(make_hsep())
        row_red = QHBoxLayout()
        row_red.addWidget(_row_label("Reducer:"))
        self._reducer_path = QLineEdit()
        self._reducer_path.setObjectName("reducer_path")
        self._reducer_path.setPlaceholderText(
            "reducer pickles (*.pkl), ';'-separated — empty + no method = skip")
        self._reducer_path.setToolTip(
            "One or more pre-fitted reducer pickles (pca/umap/pacmap/"
            "localmap), separated with ';'. When any reducer is given, the "
            "DR-method checkboxes are ignored (execution follows the "
            "canonical pca -> umap -> pacmap -> localmap order). Empty + no "
            "method checked = dimension reduction is skipped.")
        enable_path_drop(self._reducer_path, multi=True)
        row_red.addWidget(self._reducer_path, 2)
        self._reducer_browse_btn = QPushButton("Browse...")
        self._reducer_browse_btn.setToolTip(
            "Browse... (multi-select; you can also drag files onto the box)")
        row_red.addWidget(self._reducer_browse_btn)
        row_red.addStretch(1)
        layout.addLayout(row_red)
        # DR-method selection — only used when NO reducer file is given.
        row_meth = QHBoxLayout()
        meth_lbl = QLabel("DR methods:")
        meth_lbl.setFixedWidth(label_w)
        row_meth.addWidget(meth_lbl)
        self._method_cbs = {}
        for m in _DR_METHOD_OPTIONS:
            cb = QCheckBox(m)
            cb.setChecked(False)
            cb.setToolTip(
                "Methods fitted fresh when no reducer file is given. Any "
                "checked method (or a reducer file) makes the DR stage run; "
                "none checked + no reducer = skipped. Ignored entirely when "
                "reducer file(s) are provided.")
            row_meth.addWidget(cb)
            self._method_cbs[m] = cb
        row_meth.addStretch()
        layout.addLayout(row_meth)
        # Dim the method checkboxes while reducers are in charge.
        self._reducer_path.textChanged.connect(self._update_method_enabled)
        self._update_method_enabled()

        # Row 5: cluster prediction — runs when a cluster.pkl is provided.
        row_cl = QHBoxLayout()
        row_cl.addWidget(_row_label("Cluster file:"))
        self._cluster_path = QLineEdit()
        self._cluster_path.setObjectName("reducer_path")
        self._cluster_path.setPlaceholderText("cluster.pkl — empty = no prediction")
        self._cluster_path.setToolTip(
            "Predict clusters for every object from a baseline cluster.pkl "
            "(kNN vote over its stored points — cluster IDs stay "
            "baseline-aligned) and write the find_cluster table. Requires "
            "a cluster.pkl produced by `micromodel reduction`; empty = no "
            "prediction.")
        enable_path_drop(self._cluster_path)
        row_cl.addWidget(self._cluster_path, 2)
        self._cluster_browse_btn = _browse_button()
        row_cl.addWidget(self._cluster_browse_btn)
        row_cl.addStretch(1)
        layout.addLayout(row_cl)

        # ── behavior wiring ──
        self._browse_btn.clicked.connect(self._on_browse)
        self._reducer_browse_btn.clicked.connect(self._on_browse_reducer)
        self._cluster_browse_btn.clicked.connect(self._on_browse_cluster)

    def _update_capability_ui(self) -> None:
        if self._classify_capable is None:
            self._pred_class_cb.setEnabled(True)
            return
        if self._classify_capable:
            self._pred_class_cb.setEnabled(True)
        else:
            # SSL bundles extract features only — no predictions to gate on.
            self._pred_class_cb.setChecked(False)
            self._pred_class_cb.setEnabled(False)

    # ── Capability (bundle meta) ────────────────────────────────────────

    def ensure_capability(self) -> Optional[str]:
        """Read the bundle meta (once per path) and gate the output checkboxes.

        Returns an error string on failure (caller shows a popup) or None.
        The model type / input-channel count are written to the LOG, not to
        the GUI (the model-info row was removed).
        """
        path = self.get_model_path()
        if not path:
            return None
        if self._capability_checked_path == path:
            return None
        try:
            meta = read_bundle_meta(path)
        except ImportError as e:
            return str(e)
        except Exception as e:
            return f"Could not read model bundle {path}:\n{e}"
        self._classify_capable = "num_classes" in meta
        self._capability_checked_path = path
        in_chans = meta.get("in_chans")
        logger.info(
            "Loaded model bundle %s: %s, input channels: %s", path,
            "SL - Classify" if self._classify_capable else "SSL - Features",
            in_chans if in_chans is not None else "unknown")
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

    def _update_method_enabled(self, *_):
        """DR-method checkboxes matter only without reducer files."""
        has_reducer = bool(self._reducer_path.text().strip())
        for cb in self._method_cbs.values():
            cb.setEnabled(not has_reducer)

    def _selected_methods(self) -> List[str]:
        return [m for m, cb in self._method_cbs.items() if cb.isChecked()]

    def _on_browse_reducer(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Pre-fitted Reducer(s)", "",
            "Pickle files (*.pkl);;All files (*)",
        )
        if paths:
            existing = [p for p in self._reducer_path.text().split(";") if p.strip()]
            self._reducer_path.setText(";".join(existing + paths))

    def _on_browse_cluster(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Baseline cluster.pkl", "",
            "Pickle files (*.pkl);;All files (*)",
        )
        if path:
            self._cluster_path.setText(path)

    # ── Accessors ───────────────────────────────────────────────────────

    def get_model_path(self) -> str:
        return self._model_path.text().strip()

    def get_output_db(self) -> str:
        return self._output_db.text().strip() or INFER_DB_NAME

    def get_mask_name(self) -> str:
        return self._mask_combo.currentText().strip() or ""

    def get_max_value(self) -> float:
        return self._max_value

    def set_max_value(self, value: float, explicit: bool = False) -> None:
        self._max_value = float(value)
        self._max_value_label.setText(f"{self._max_value:g}")
        if explicit:
            self._max_value_explicit = True

    def get_checked_channels(self) -> List[str]:
        """Checked channels in row order (left-to-right).

        The order is meaningful: it maps 1:1 to the model's input channel
        order — click a channel, then ◀/▶ to match the trained model's
        channels.
        """
        return [cb.text() for cb in self._ch_cbs if cb.isChecked()]

    def set_channel_state(self, order: List[str], checked) -> None:
        """Rebuild the channel checkboxes with an explicit order + checked set."""
        if self._chan_placeholder is not None:
            # Channels arrived — the "Load a dataset to configure" hint goes.
            self._channels_row.removeWidget(self._chan_placeholder)
            self._chan_placeholder.deleteLater()
            self._chan_placeholder = None
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

    def is_reduction_or_cluster_enabled(self) -> bool:
        """True when any reduction/cluster stage has something to run.

        There are no group checkboxes anymore: a reducer file OR at least
        one checked DR method makes the reduction stage run, and a
        cluster.pkl path makes the cluster prediction run.
        """
        dr_runs = bool(self._reducer_path.text().strip()
                       or self._selected_methods())
        return dr_runs or bool(self._cluster_path.text().strip())

    # ── Population ──────────────────────────────────────────────────────

    def populate_channels(self, channels: List[str]) -> None:
        """Rebuild the channel checkboxes, preserving the current order/checks.

        Restored configs re-apply their ordered channel selection via
        _apply_block_config after this runs (base-class deferred restore).
        """
        self._channels = list(channels)
        if channels:
            BaseStepPanel._remove_placeholder(self._channels_row, "_chan_placeholder", self)
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
            "output_db": self.get_output_db(),
            "max_value": self._max_value,
        }
        reducers = [p.strip() for p in self._reducer_path.text().split(";")
                    if p.strip()]
        methods = self._selected_methods()
        cluster_file = self._cluster_path.text().strip()
        if reducers or methods or cluster_file:
            section["reduction"] = {
                # Derived flags (no group checkboxes in the GUI): the
                # reduction runs with a reducer file or any checked method;
                # the cluster runs with a cluster file.
                "enabled": bool(reducers or methods),
                "reducer": reducers or None,
                "cluster_enabled": bool(cluster_file),
                "cluster": cluster_file or None,
                # Method checkboxes apply only without reducers (the bridge
                # ignores them otherwise — serialize None so the YAML stays
                # honest). [] = nothing fitted; a cluster file still uses PCA
                # for its reference embedding.
                "method": (None if reducers else methods),
                # No-widget YAML keys, round-tripped verbatim.
                "color_by": self._color_by,
                "cluster_res": self._cluster_res,
                "sample_per_class": self._sample_per_class,
            }
        return section


class InferenceStepPanel(BlockContainerPanel):

    step_name = "inference"
    _block_widget_class = InferenceBlockWidget
    # These QLineEdits keep their natural width (model path / reducer paths).
    _compact_excluded_object_names = frozenset({"checkpoint_path", "reducer_path"})

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("Inference (microModel)")
        self._default_max_value = DEFAULT_MAX_VALUE
        self._build_block_container("+ Add New Inference Block")
        self._add_block_generic([])
        self._apply_user_defaults()

    def _apply_user_defaults(self) -> None:
        """Pre-fill the first block from ~/.micromax (the LAST run's model /
        reducer(s) / cluster.pkl — saved by the pipeline after each run).

        A remembered reducer/cluster path IS the "run it" flag (there are no
        group checkboxes anymore): the stage runs because the path is
        present, and the user skips it by clearing the path. A config load
        or a session.yml restore overwrites these afterwards. The model's
        capability is intentionally NOT read here (a full torch.load at
        startup would stall the GUI) — validate_blocks re-checks it before
        every run.
        """
        defaults = get_user_defaults().get("inference") or {}
        if not defaults:
            return
        block = self._blocks[0]
        model = defaults.get("model")
        if model:
            block._model_path.setText(str(model))
        reducer = defaults.get("reducer") or []
        if isinstance(reducer, str):
            reducer = [reducer]
        if reducer:
            block._reducer_path.setText(";".join(str(p) for p in reducer))
        cluster = defaults.get("cluster")
        if cluster:
            block._cluster_path.setText(str(cluster))

    def _connect_block_signals(self, block: InferenceBlockWidget) -> None:
        super()._connect_block_signals(block)
        for w in (block._model_path, block._output_db, block._mask_combo,
                  block._feature_cb, block._pred_class_cb,
                  block._reducer_path, block._cluster_path,
                  *block._method_cbs.values()):
            self._wire_param_signal(w)
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
            block.set_max_value(src.get_max_value(), explicit=src._max_value_explicit)
            block._feature_cb.setChecked(src._feature_cb.isChecked())
            block._pred_class_cb.setChecked(src._pred_class_cb.isChecked())
            block._mask_combo.clear()
            for i in range(src._mask_combo.count()):
                block._mask_combo.addItem(src._mask_combo.itemText(i))
            block._mask_combo.setCurrentText(src.get_mask_name())
            block._reducer_path.setText(src._reducer_path.text())
            block._cluster_path.setText(src._cluster_path.text())
            for m, cb in src._method_cbs.items():
                block._method_cbs[m].setChecked(cb.isChecked())
            if src._ch_cbs:
                src_order = [cb.text() for cb in src._ch_cbs]
                block.set_channel_state(src_order, set(src.get_checked_channels()))
            # NOTE: when the dataset is not loaded yet, the copied block has
            # no checkboxes — the copy's selection is not carried over (the
            # base-class deferred restore covers config-driven restores).
        else:
            block.set_max_value(self._default_max_value)

    def _apply_block_config(self, block: InferenceBlockWidget, cfg: dict) -> None:
        model = cfg.get("model", "")
        if model:
            block._model_path.setText(str(model))
        db_name = cfg.get("output_db", "")
        if db_name:
            block._output_db.setText(str(db_name))
        max_value = cfg.get("max_value")
        if max_value is not None:
            block.set_max_value(float(max_value), explicit=True)
        BaseStepPanel._set_widget(block._feature_cb, cfg.get("feature", True), "feature")
        BaseStepPanel._set_widget(block._pred_class_cb, cfg.get("pred_class", True), "pred_class")
        mask_name = cfg.get("mask_name", "")
        if mask_name:
            mask_name = str(mask_name).removeprefix("mask_")
            idx = block._mask_combo.findText(mask_name)
            if idx >= 0:
                block._mask_combo.setCurrentIndex(idx)
            else:
                block._mask_combo.setCurrentText(mask_name)
        # Channel order is meaningful: config order wins, then the remaining
        # dataset channels. Applied by the base-class deferred restore after
        # populate_channels built the checkboxes.
        channels_cfg = cfg.get("channels")
        if channels_cfg and block._channels:
            order = (
                [c for c in channels_cfg if c in block._channels]
                + [c for c in block._channels if c not in channels_cfg]
            )
            block.set_channel_state(order, set(channels_cfg))
        red = cfg.get("reduction") or {}
        # The GUI derives "run" from the paths/methods, so the stored
        # enabled / cluster_enabled flags are ignored on restore.
        # color_by / cluster_res / sample_per_class have no GUI widget —
        # keep the config values verbatim so a YAML with custom values
        # survives a round-trip.
        block._color_by = red.get("color_by", "pred_class")
        block._cluster_res = red.get("cluster_res")
        block._sample_per_class = red.get("sample_per_class", 10000)
        reducer = red.get("reducer") or []
        if isinstance(reducer, str):
            reducer = [reducer]
        if reducer:
            block._reducer_path.setText(";".join(str(p) for p in reducer))
        # method: null (reducers in charge / no explicit choice) leaves every
        # checkbox unchecked — the GUI default is "nothing fitted".
        method = red.get("method")
        if method is None:
            method = []
        for m, cb in block._method_cbs.items():
            cb.setChecked(m in method)
        if red.get("cluster"):
            block._cluster_path.setText(str(red["cluster"]))

    def set_dataset_dtype(self, dtype) -> None:
        """Refresh per-block max_value defaults from the dataset dtype.

        Config/explicit values are always trusted (never overwritten). Every
        non-explicit block adopts the new dtype's value on EVERY dataset
        change — the previous check only fired while the value still equalled
        DEFAULT_MAX_VALUE, so switching from a uint8 dataset to a uint16 one
        kept max_value=255 and silently ran inference on the wrong intensity
        domain.
        """
        derived = _max_value_for_dtype(dtype)
        self._default_max_value = derived
        for block in self._blocks:
            if not block._max_value_explicit:
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
            if block.is_reduction_or_cluster_enabled() and not block._feature_cb.isChecked():
                return (
                    "Dimension reduction / Cluster require the 'feature' "
                    "output. Check 'feature' in the block before running."
                )
            # NOTE: a block with no checked channels is NOT an error — it is
            # skipped at runtime (empty channels = skip, never "all").
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
    return _DTYPE_MAX.get(name, DEFAULT_MAX_VALUE)
