from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLabel, QSlider


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores scroll-wheel.

    Overrides event() — wheelEvent does not work because QAbstractSpinBox
    processes wheel events through an internal event filter on its child
    QLineEdit, completely outside the normal wheelEvent dispatch chain.
    """

    def event(self, event):
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            self.clearFocus()
            return True
        return super().event(event)


class NoScrollComboBox(QComboBox):
    """QComboBox that ignores scroll-wheel."""

    def event(self, event):
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            self.clearFocus()
            return True
        return super().event(event)


class NoScrollSlider(QSlider):
    """QSlider that ignores scroll-wheel."""

    def event(self, event):
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            self.clearFocus()
            return True
        return super().event(event)


class RotatedLabel(QLabel):
    """QLabel that paints text rotated -90 degrees (top-to-bottom)."""

    clicked = Signal()

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(-90)
        fm = self.fontMetrics()
        rect = fm.boundingRect(self.text())
        painter.drawText(-rect.width() / 2, rect.height() / 4, self.text())
        painter.end()

    def mousePressEvent(self, event):
        self.clicked.emit()

    def sizeHint(self):
        fm = self.fontMetrics()
        rect = fm.boundingRect(self.text())
        from PySide6.QtCore import QSize
        return QSize(rect.height() + 8, (rect.width() + 8) * 2)

    def minimumSizeHint(self):
        return self.sizeHint()
