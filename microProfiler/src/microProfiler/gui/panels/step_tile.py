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
        "_tile_width": "tile_width",
        "_tile_height": "tile_height",
    }

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("Tile Splitting")
        self._build_controls()

    def _build_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("W:"))
        self._tile_width = QSpinBox()
        self._tile_width.setRange(64, 65536)
        self._tile_width.setValue(1024)
        self._tile_width.setSingleStep(256)
        self._tile_width.setToolTip("Tile width in pixels")
        row.addWidget(self._tile_width)
        row.addWidget(QLabel("H:"))
        self._tile_height = QSpinBox()
        self._tile_height.setRange(64, 65536)
        self._tile_height.setValue(1024)
        self._tile_height.setSingleStep(256)
        self._tile_height.setToolTip("Tile height in pixels")
        row.addWidget(self._tile_height)
        row.addStretch()
        self._apply_btn = QPushButton("▶ Apply")
        self._apply_btn.setProperty("class", "primary")
        self._apply_btn.setFixedHeight(dp(28))
        row.addWidget(self._apply_btn)
        self._controls_layout.addLayout(row)

        for w in (self._tile_width, self._tile_height):
            self._wire_param_signal(w)
