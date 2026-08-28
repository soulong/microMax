from __future__ import annotations

import sys
from importlib.resources import files

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

from microProfiler.log_utils import _ensure_std_streams, setup_logging


def _resource(name: str) -> str | None:
    """Return path to bundled resource file."""
    try:
        return str(files("microProfiler.resources") / name)
    except Exception:
        return None


def main() -> None:
    _ensure_std_streams()
    setup_logging()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("microProfiler")
    app.setOrganizationName("microProfiler")

    # Dark palette for title bar and system dialogs
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 46))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(30, 30, 46))
    palette.setColor(QPalette.AlternateBase, QColor(37, 37, 54))
    palette.setColor(QPalette.ToolTipBase, QColor(45, 45, 68))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(45, 45, 68))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.BrightText, QColor(240, 71, 112))
    palette.setColor(QPalette.Highlight, QColor(76, 201, 240))
    palette.setColor(QPalette.HighlightedText, QColor(30, 30, 46))
    app.setPalette(palette)

    icon_path = _resource("icon.ico")
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    qss_path = _resource("style.qss")
    if qss_path:
        with open(qss_path, encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("microProfiler")
    except Exception:
        pass  # non-Windows or ctypes unavailable — safe to ignore

    from microProfiler.gui.main_window import MainWindow

    icon = QIcon(icon_path) if icon_path else None

    window = MainWindow()
    if icon:
        window.setWindowIcon(icon)
    window.show()

    # Reinforce icon after window is fully rendered (fixes taskbar icon lost after AppUserModelID)
    if icon:
        QTimer.singleShot(200, lambda: window.setWindowIcon(icon))

    sys.exit(app.exec())
