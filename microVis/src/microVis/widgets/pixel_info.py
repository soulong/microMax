from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget


class PixelInfo(QFrame):
    """Bottom bar showing pixel intensity information on image click."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("pixel-info")
        self.setStyleSheet(
            "QFrame#pixel-info {"
            "  background-color: #1e1e2e;"
            "  border-top: 1px solid #333333;"
            "  padding: 4px 12px;"
            "  max-height: 28px;"
            "}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 2, 12, 2)

        self._label = QLabel("Click an image to see pixel intensities")
        self._label.setProperty("class", "muted")
        layout.addWidget(self._label)

    def set_text(self, text: str) -> None:
        self._label.setText(text)
        self._label.setProperty("class", "")
        self._label.style().polish(self._label)
