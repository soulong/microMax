from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from microProfiler.gui.dpi import dp
from microProfiler.gui.panels.base_step_panel import BaseStepPanel
from microProfiler.gui.ui_spec import step_button
from microProfiler.gui.image_widgets import ChannelTile


class BaSiCStepPanel(BaseStepPanel):
    step_name = "basic"
    _FIELD_MAP = {
        "_mode": "mode",
        "_n_image": "n_image",
        "_working_size": "working_size",
        "_darkfield": "enable_darkfield",
    }

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("BaSiC Shading Correction")
        self._build_controls()
        self._build_preview()

    def _build_controls(self):
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Mode:"))
        self._mode = QComboBox()
        self._mode.addItems(["fit-transform", "fit", "transform"])
        self._mode.setToolTip("fit: learn shading model, transform: apply existing, fit-transform: both")
        row1.addWidget(self._mode)
        row1.addWidget(QLabel("Fit images:"))
        self._n_image = QSpinBox()
        self._n_image.setRange(1, 1000)
        self._n_image.setValue(100)
        self._n_image.setToolTip("Number of images used to fit the BaSiC model")
        row1.addWidget(self._n_image)
        row1.addWidget(QLabel("Working size:"))
        self._working_size = QSpinBox()
        self._working_size.setRange(16, 512)
        self._working_size.setValue(64)
        self._working_size.setToolTip("Downscaled size (px) for faster model fitting")
        row1.addWidget(self._working_size)
        self._darkfield = QCheckBox("Darkfield")
        self._darkfield.setToolTip("Enable darkfield correction in addition to flatfield")
        row1.addWidget(self._darkfield)
        row1.addStretch()
        self._controls_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addStretch()
        self._fit_btn = QPushButton("Fit Model")
        self._fit_btn.setProperty("class", "secondary")
        row2.addWidget(self._fit_btn)
        self._pick_btn = QPushButton("Pick Random")
        self._pick_btn.setProperty("class", "secondary")
        row2.addWidget(self._pick_btn)
        self._preview_btn = QPushButton("Preview Transform")
        self._preview_btn.setProperty("class", "secondary")
        row2.addWidget(self._preview_btn)
        self._apply_btn = step_button("▶ Apply")
        row2.addWidget(self._apply_btn)
        self._controls_layout.addLayout(row2)

        for w in (self._mode, self._n_image, self._working_size, self._darkfield):
            self._wire_param_signal(w)

    def _build_preview(self):
        # Per-channel preview container: each row is chN: [raw] | [corrected] | [flatfield]
        self._preview_container = QVBoxLayout()
        self._preview_container.setSpacing(2)
        self._preview_layout.addLayout(self._preview_container)

    def set_preview_channels(self, channel_names):
        self._clear_preview()
        if not channel_names:
            return
        for ch in channel_names:
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(QLabel(f"{ch}:"))
            raw_tile = ChannelTile("raw", np.zeros((64, 64)))
            raw_tile._viewer.setMinimumSize(dp(200), dp(200))
            corr_tile = ChannelTile("corrected", np.zeros((64, 64)))
            corr_tile._viewer.setMinimumSize(dp(200), dp(200))
            ff_tile = ChannelTile("flatfield", np.zeros((64, 64)))
            ff_tile._viewer.setMinimumSize(dp(200), dp(200))
            row.addWidget(raw_tile)
            row.addWidget(corr_tile)
            row.addWidget(ff_tile)
            row.addStretch()
            # Store refs for later update
            if not hasattr(self, "_channel_tiles"):
                self._channel_tiles = {}
            self._channel_tiles[ch] = (raw_tile, corr_tile, ff_tile)
            self._preview_container.addLayout(row)

    def clear_preview(self):
        if hasattr(self, "_channel_tiles"):
            for ch in self._channel_tiles:
                blank = np.zeros((64, 64))
                self._channel_tiles[ch][0].set_image(blank)
                self._channel_tiles[ch][1].set_image(blank)
                self._channel_tiles[ch][2].set_image(blank)

    def update_preview_raw(self, channel_data: dict):
        for ch, arr in channel_data.items():
            if hasattr(self, "_channel_tiles") and ch in self._channel_tiles:
                self._channel_tiles[ch][0].set_image(arr)

    def update_preview_corrected(self, channel_data: dict):
        for ch, arr in channel_data.items():
            if hasattr(self, "_channel_tiles") and ch in self._channel_tiles:
                self._channel_tiles[ch][1].set_image(arr)

    def update_preview_flatfield(self, channel_data: dict):
        for ch, arr in channel_data.items():
            if hasattr(self, "_channel_tiles") and ch in self._channel_tiles:
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = (arr - mn) / (mx - mn)
                self._channel_tiles[ch][2].set_image(arr)

    def _clear_preview(self):
        while self._preview_container.count():
            item = self._preview_container.takeAt(0)
            if item.layout():
                self._clear_layout(item.layout())
            elif item.widget():
                item.widget().deleteLater()
        if hasattr(self, "_channel_tiles"):
            del self._channel_tiles

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.layout():
                BaSiCStepPanel._clear_layout(item.layout())
            elif item.widget():
                item.widget().deleteLater()
