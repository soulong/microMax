"""QApplication bootstrap for the microVis desktop GUI."""
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QEvent, QObject, QTimer
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QScrollArea

from microVis.log_utils import setup_logging


class _WheelBlocker(QObject):
    """Global event filter that redirects wheel events to QScrollAreas.

    Normal wheel (no modifiers) → redirected to the nearest QScrollArea so
    the panel scrolls regardless of which widget the cursor is over.
    Ctrl+wheel → passed through naturally so widgets like _ThumbnailView can
    zoom (or other Ctrl+wheel actions).

    Also prevents accidental value changes on unfocused QAbstractSpinBox /
    QComboBox widgets by consuming their wheel events when no QScrollArea
    ancestor is available to redirect to.
    """

    def eventFilter(self, watched, event):
        try:
            if event.type() != QEvent.Type.Wheel:
                return False

            pos = event.globalPosition().toPoint()
            target = QApplication.instance().widgetAt(pos)
            if target is None:
                return False

            # ── Walk parent chain ────────────────────────────────────────────
            has_spin_or_combo = False
            found_scroll: QScrollArea | None = None

            w = target
            for _ in range(12):
                if w is None:
                    break
                if isinstance(w, (QAbstractSpinBox, QComboBox)):
                    has_spin_or_combo = True
                    if w.hasFocus():
                        return False        # Focused → let it self-handle
                elif isinstance(w, QScrollArea) and found_scroll is None:
                    found_scroll = w        # nearest QScrollArea ancestor
                w = w.parentWidget()

            # ── Ctrl+wheel: natural dispatch (zoom, etc.) ───────────────────
            if event.modifiers() & Qt.ControlModifier:
                return False

            # ── Normal wheel: redirect to the nearest QScrollArea ────────────
            if found_scroll is not None:
                vbar = found_scroll.verticalScrollBar()
                hbar = found_scroll.horizontalScrollBar()
                dx = event.angleDelta().x()
                dy = event.angleDelta().y()
                # Horizontal swipe (touchpad) → horizontal scrollbar
                if dx != 0 and hbar.minimum() < hbar.maximum():
                    hbar.setValue(hbar.value() - dx)
                # Vertical swipe (mouse wheel or touchpad) → vertical priority
                if dy != 0:
                    if vbar.minimum() < vbar.maximum():
                        vbar.setValue(vbar.value() - dy)
                    elif hbar.minimum() < hbar.maximum():
                        hbar.setValue(hbar.value() - dy)
                return True

            # ── No scroll area: block unfocused spinbox/combo value change ──
            if has_spin_or_combo:
                return True
            return False
        except KeyboardInterrupt:
            sys.exit(0)


def run_app(dataset_dir: str | None = None) -> None:
    """Launch the microVis desktop application.

    Args:
        dataset_dir: Optional path to a measurement directory to load on startup.
    """
    os.environ.setdefault("QT_LOGGING_RULES", "qt.gui.icc=false")

    setup_logging()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("microVis")
    except Exception:
        pass

    # Block scroll-wheel on all input widgets app-wide
    app.installEventFilter(_WheelBlocker(app))
    app.setApplicationName("microVis")
    app.setOrganizationName("microVis")
    from microVis import __version__
    app.setApplicationVersion(__version__)

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

    resources = Path(__file__).parent / "resources"

    icon_path = resources / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    qss_path = resources / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    from microVis.main_window import MainWindow

    icon = QIcon(str(icon_path)) if icon_path.exists() else None

    window = MainWindow(dataset_dir=dataset_dir)
    if icon:
        window.setWindowIcon(icon)
    window.show()

    # Reinforce icon after window is fully rendered (fixes taskbar icon lost after AppUserModelID)
    if icon:
        QTimer.singleShot(200, lambda: window.setWindowIcon(icon))

    sys.exit(app.exec())
