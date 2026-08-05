from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)

from microProfiler.gui.panels.base_step_panel import BaseStepPanel, dp


class TileStepPanel(BaseStepPanel):
    step_name = "tile"
    _FIELD_MAP = {
        "_tile_w": ("tile_width", int, 1024),
        "_tile_h": ("tile_height", int, 1024),
    }

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("Tile Splitting")
        self._build_controls()

    def _build_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("W:"))
        self._tile_w = QSpinBox()
        self._tile_w.setRange(64, 65536)
        self._tile_w.setValue(1024)
        self._tile_w.setSingleStep(256)
        self._tile_w.setToolTip("Tile width in pixels")
        row.addWidget(self._tile_w)
        row.addWidget(QLabel("H:"))
        self._tile_h = QSpinBox()
        self._tile_h.setRange(64, 65536)
        self._tile_h.setValue(1024)
        self._tile_h.setSingleStep(256)
        self._tile_h.setToolTip("Tile height in pixels")
        row.addWidget(self._tile_h)
        row.addStretch()
        self._apply_btn = QPushButton("▶ Apply")
        self._apply_btn.setProperty("class", "primary")
        self._apply_btn.setFixedHeight(dp(28))
        row.addWidget(self._apply_btn)
        self._controls_layout.addLayout(row)

        for w in (self._tile_w, self._tile_h):
            self._wire_param_signal(w)
