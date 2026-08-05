from __future__ import annotations

from PySide6.QtWidgets import QApplication


def dp(px: int) -> int:
    """Scale pixel value by device pixel ratio for DPI-aware sizing."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return px
    return int(px * screen.devicePixelRatio())
