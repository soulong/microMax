from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from microVis._settings import CHANNEL_COLORS
from microVis.widgets._event_filter import NoScrollComboBox, NoScrollDoubleSpinBox


class ChannelControls(QWidget):
    """Per-channel control block: checkbox + color on row 1, vmin/vmax on row 2."""

    config_changed = Signal(str)

    def __init__(self, ch_name: str, cfg: dict, max_value: float | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._ch_name = ch_name
        # Spin-box range follows the dataset dtype (e.g. 0..4294967295 for
        # uint32, 0..1 for float) — a hardcoded 16-bit range would clamp
        # vmax and silently corrupt the display of wider data.
        self._max_value = float(max_value) if max_value is not None else 65535.0

        self.setStyleSheet("""
            QDoubleSpinBox, QComboBox {
                padding: 2px 3px;
                min-width: 0;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(1)

        # Row 1: checkbox + color
        top = QHBoxLayout()
        top.setSpacing(4)

        self._toggle = QCheckBox(ch_name)
        self._toggle.setChecked(cfg.get("enabled", True))
        self._toggle.toggled.connect(lambda: self.config_changed.emit(self._ch_name))
        top.addWidget(self._toggle)

        self._color = NoScrollComboBox()
        color_names = list(CHANNEL_COLORS.keys())
        self._color.addItems(color_names)
        current_color = cfg.get("color", "green")
        for i, cname in enumerate(color_names):
            if isinstance(current_color, str):
                if cname == current_color:
                    self._color.setCurrentIndex(i)
                    break
            elif CHANNEL_COLORS[cname] == tuple(current_color):
                self._color.setCurrentIndex(i)
                break
        self._color.currentTextChanged.connect(lambda: self.config_changed.emit(self._ch_name))
        top.addWidget(self._color, stretch=1)
        top.addStretch()
        root.addLayout(top)

        # Row 2: vmin / vmax
        bottom = QHBoxLayout()
        bottom.setSpacing(4)
        bottom.setContentsMargins(20, 0, 0, 0)

        bottom.addWidget(QLabel("vmin"))
        self._vmin = NoScrollDoubleSpinBox()
        self._vmin.setRange(0, self._max_value)
        # Decimals >= 1 so auto-range percentiles (e.g. 1234.7) round-trip
        # without truncation drift across reloads. Float datasets (max 1.0)
        # get finer decimals + step so [0, 1] contrast is adjustable.
        self._vmin.setDecimals(4 if self._max_value <= 1.0 else 2)
        self._vmin.setSingleStep(0.0001 if self._max_value <= 1.0 else 1.0)
        self._vmin.setValue(cfg.get("vmin", 0))
        self._vmin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._vmin.valueChanged.connect(lambda: self.config_changed.emit(self._ch_name))
        self._vmin.setContextMenuPolicy(Qt.NoContextMenu)
        bottom.addWidget(self._vmin, stretch=1)

        bottom.addWidget(QLabel("vmax"))
        self._vmax = NoScrollDoubleSpinBox()
        self._vmax.setRange(0, self._max_value)
        self._vmax.setDecimals(4 if self._max_value <= 1.0 else 2)
        self._vmax.setSingleStep(0.0001 if self._max_value <= 1.0 else 1.0)
        self._vmax.setValue(cfg.get("vmax", self._max_value))
        self._vmax.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._vmax.valueChanged.connect(lambda: self.config_changed.emit(self._ch_name))
        self._vmax.setContextMenuPolicy(Qt.NoContextMenu)
        bottom.addWidget(self._vmax, stretch=1)

        root.addLayout(bottom)

    def get_config(self) -> dict:
        color_name = self._color.currentText()
        return {
            "enabled": self._toggle.isChecked(),
            "color": CHANNEL_COLORS.get(color_name, (0, 1, 0)),
            "vmin": self._vmin.value(),
            "vmax": self._vmax.value(),
        }

    def set_values(self, vmin: float, vmax: float) -> None:
        self._vmin.blockSignals(True)
        self._vmax.blockSignals(True)
        self._vmin.setValue(vmin)
        self._vmax.setValue(vmax)
        self._vmin.blockSignals(False)
        self._vmax.blockSignals(False)
