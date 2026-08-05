from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton

from microProfiler.gui.panels.base_step_panel import BaseStepPanel, dp


class ResizeStepPanel(BaseStepPanel):
    step_name = "resize"
    _FIELD_MAP = {"_scale_factor": ("scale_factor", float, 0.5)}

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.setTitle("Resize")
        self._build_controls()

    def _build_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Scale:"))
        self._scale_factor = QDoubleSpinBox()
        self._scale_factor.setRange(0.1, 4.0)
        self._scale_factor.setValue(0.5)
        self._scale_factor.setSingleStep(0.1)
        self._scale_factor.setToolTip("Resize factor (0.5 = half size, 2.0 = double)")
        row.addWidget(self._scale_factor)
        row.addStretch()
        self._apply_btn = QPushButton("▶ Apply")
        self._apply_btn.setProperty("class", "primary")
        self._apply_btn.setFixedHeight(dp(28))
        row.addWidget(self._apply_btn)
        self._controls_layout.addLayout(row)

        for w in (self._scale_factor,):
            self._wire_param_signal(w)
