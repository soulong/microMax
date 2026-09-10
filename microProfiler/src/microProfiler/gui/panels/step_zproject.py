from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton

from microProfiler.gui.panels.base_step_panel import BaseStepPanel
from microProfiler.gui.ui_spec import step_button


class ZProjectStepPanel(BaseStepPanel):
    step_name = "zproject"
    _FIELD_MAP = {"_method": "method"}

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("Z-Projection")
        self._build_controls()

    def _build_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Method:"))
        self._method = QComboBox()
        self._method.addItems(["max", "mean", "min"])
        self._method.setToolTip("max: brightest pixel, mean: average intensity, min: darkest pixel")
        row.addWidget(self._method)
        row.addStretch()
        self._apply_btn = step_button("▶ Apply")
        row.addWidget(self._apply_btn)
        self._controls_layout.addLayout(row)

        for w in (self._method,):
            self._wire_param_signal(w)
