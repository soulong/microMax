from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from microProfiler.gui.dpi import dp
from microProfiler.gui.state import PipelineState

logger = logging.getLogger(__name__)


def make_hsep() -> QFrame:
    """Horizontal separator line shared by step panels."""
    s = QFrame()
    s.setFrameShape(QFrame.HLine)
    s.setFrameShadow(QFrame.Sunken)
    s.setProperty("class", "separator")
    return s


class BaseStepPanel(QGroupBox):

    step_name: str = "step"
    parameter_changed = Signal()
    # widget attribute -> config key
    _FIELD_MAP: Dict[str, str] = {}

    def __init__(self, state: PipelineState, parent=None):
        super().__init__(parent)
        self._state = state
        self.setCheckable(True)
        self.setChecked(False)
        self.setProperty("class", "card")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self._controls_widget = QWidget()
        self._controls_widget.setStyleSheet("background-color: #252536;")
        self._controls_layout = QVBoxLayout(self._controls_widget)
        self._controls_layout.setContentsMargins(0, 0, 0, 0)
        self._controls_layout.setSpacing(2)

        self._preview_widget = QWidget()
        self._preview_widget.setStyleSheet("background-color: #252536;")
        self._preview_widget.setProperty("class", "card-inner")
        self._preview_layout = QVBoxLayout(self._preview_widget)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)

        body = QVBoxLayout(self)
        body.setContentsMargins(6, 6, 6, 6)
        body.setSpacing(0)
        body.addWidget(self._controls_widget)
        body.addWidget(self._preview_widget)

    def is_enabled(self) -> bool:
        return self.isChecked()

    @staticmethod
    def _remove_placeholder(layout, placeholder_attr, obj):
        old = getattr(obj, placeholder_attr, None)
        if old is not None:
            layout.removeWidget(old)
            old.deleteLater()
            setattr(obj, placeholder_attr, None)

    @staticmethod
    def _clear_checkboxes(layout, checkbox_list):
        i = 0
        while i < layout.count():
            item = layout.itemAt(i)
            w = item.widget()
            if isinstance(w, QCheckBox):
                layout.removeItem(item)
                w.deleteLater()
            else:
                i += 1
        checkbox_list.clear()

    @staticmethod
    def _checked_checkboxes(cbs):
        return [cb.text() for cb in cbs if cb.isChecked()]

    @staticmethod
    def _set_checked_states(cbs, channels):
        if channels is None:
            return
        channel_set = set(channels)
        for cb in cbs:
            cb.setChecked(cb.text() in channel_set)

    def _wire_param_signal(self, widget: QObject) -> None:
        if isinstance(widget, (QLineEdit, QSpinBox, QDoubleSpinBox)):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.parameter_changed, Qt.UniqueConnection)
            if hasattr(widget, "textChanged"):
                widget.textChanged.connect(self.parameter_changed, Qt.UniqueConnection)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(self.parameter_changed, Qt.UniqueConnection)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self.parameter_changed, Qt.UniqueConnection)

    @staticmethod
    def _compact_block(block: QWidget, max_width: int = 200) -> None:
        from PySide6.QtWidgets import QAbstractSpinBox
        excluded = getattr(block, "_compact_excluded_object_names", ()) or ()
        for child in block.findChildren(QSpinBox):
            child.setMaximumWidth(max_width)
            child.setButtonSymbols(QAbstractSpinBox.NoButtons)
        for child in block.findChildren(QDoubleSpinBox):
            child.setMaximumWidth(max_width)
            child.setButtonSymbols(QAbstractSpinBox.NoButtons)
        for child in block.findChildren(QComboBox):
            child.setMaximumWidth(max_width)
        for child in block.findChildren(QLineEdit):
            name = child.objectName()
            if name in excluded:
                continue
            if child.maximumWidth() > max_width or child.maximumWidth() == 16777215:
                child.setMaximumWidth(160)

    def build_config_section(self) -> Optional[dict]:
        if not self._FIELD_MAP:
            return None
        config = {}
        for widget_attr, key in self._FIELD_MAP.items():
            widget = getattr(self, widget_attr, None)
            if widget is None:
                continue
            config[key] = self._read_widget(widget)
        return config

    def load_config_section(self, section: dict) -> None:
        if not section or not self._FIELD_MAP:
            return
        for widget_attr, key in self._FIELD_MAP.items():
            if key in section:
                widget = getattr(self, widget_attr, None)
                if widget is not None:
                    self._set_widget(widget, section[key], key)

    def to_config(self) -> Optional[dict]:
        section = self.build_config_section()
        if section is not None:
            section["run"] = self.isChecked()
        return section

    def from_config(self, section: dict) -> None:
        if not section:
            return
        run_val = section.pop("run", None)
        if run_val is not None:
            self.setChecked(bool(run_val) if not isinstance(run_val, str) else run_val.lower() in ("1", "true", "yes"))
        self.load_config_section(section)
        if run_val is not None:
            section["run"] = run_val

    @staticmethod
    def _read_widget(widget: QObject) -> Any:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return widget.text() if hasattr(widget, "text") else str(widget)

    @staticmethod
    def _set_widget(widget: QObject, value: Any, field_name: str = "") -> None:
        """Set a widget value with type coercion, logging failures instead of silent pass."""
        if value is None:
            return
        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "yes"))
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                else:
                    widget.setCurrentText(str(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            else:
                if hasattr(widget, "setText"):
                    widget.setText(str(value))
        except (ValueError, TypeError) as e:
            logger.warning("Failed to set %s=%r on %s: %s", field_name, value, type(widget).__name__, e)
