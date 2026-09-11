from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from microProfiler.gui.dpi import dp
from microProfiler.gui.ui_spec import SIDEBAR_WIDTH


def _make_divider() -> QFrame:
    
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setProperty("class", "sidebar-divider")
    line.setFixedHeight(1)
    return line


class SidebarItem(QWidget):

    clicked = Signal(str)

    def __init__(self, page_id: str, label: str, parent=None):
        super().__init__(parent)
        self._page_id = page_id
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("active", False)
        self.setFixedHeight(dp(28))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(0)

        self._text_label = QLabel(label)
        self._text_label.setProperty("class", "sidebar-label")
        self._text_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self._text_label)

    def set_active(self, active: bool) -> None:
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
        for child in self.findChildren(QLabel):
            self.style().unpolish(child)
            self.style().polish(child)
            child.update()

    def mousePressEvent(self, event):
        self.clicked.emit(self._page_id)
        super().mousePressEvent(event)


class Sidebar(QWidget):

    navigation_changed = Signal(str)
    run_all_clicked = Signal()
    cancel_clicked = Signal()
    thread_count_changed = Signal(int)
    display_range_changed = Signal(float, float)

    PAGES = [
        ("input", "Input"),
        ("preprocess", "Preprocess"),
        ("segment", "Segmentation"),
        ("profile", "Profiling"),
        ("inference", "Inference"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(dp(SIDEBAR_WIDTH))
        self.setObjectName("sidebar")
        self._items: dict[str, SidebarItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(0)

        # ── Input ──
        self._add_item("input")
        layout.addWidget(_make_divider())

        # ── Processing ──
        self._add_item("preprocess")
        layout.addWidget(_make_divider())

        # ── Analysis ──
        self._add_item("segment")
        self._add_item("profile")
        self._add_item("inference")

        layout.addStretch()

        # ── Footer buttons ──
        layout.addWidget(_make_divider())

        def _make_spin_row(label_text: str, spin, tooltip: str) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setContentsMargins(8, 2, 8, 2)
            lbl = QLabel(label_text)
            lbl.setProperty("class", "sidebar-label")
            row.addWidget(lbl)
            spin.setProperty("class", "sidebar-spin")
            spin.setFixedHeight(dp(18))
            spin.setFixedWidth(dp(36))
            spin.setToolTip(tooltip)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            row.addWidget(spin)
            return row

        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setRange(0.0, 100.0)
        self._vmin_spin.setDecimals(2)
        self._vmin_spin.setValue(0.1)
        self._vmin_spin.setSingleStep(1.0)
        self._vmin_spin.valueChanged.connect(lambda _: self._emit_display_range())
        layout.addLayout(_make_spin_row("vmin", self._vmin_spin, "Lower percentile bound for display contrast"))

        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(0.0, 100.0)
        self._vmax_spin.setDecimals(2)
        self._vmax_spin.setValue(99.9)
        self._vmax_spin.setSingleStep(1.0)
        self._vmax_spin.valueChanged.connect(lambda _: self._emit_display_range())
        layout.addLayout(_make_spin_row("vmax", self._vmax_spin, "Upper percentile bound for display contrast"))

        # Thread count spinbox
        n_cpu = os.cpu_count() or 1
        self._thread_spin = QSpinBox()
        self._thread_spin.setRange(1, 64)
        self._thread_spin.setValue(max(1, n_cpu // 2))
        self._thread_spin.setToolTip("Number of worker processes for parallel profiling")
        self._thread_spin.valueChanged.connect(self.thread_count_changed.emit)
        layout.addLayout(_make_spin_row("Thread", self._thread_spin, "Number of worker processes for parallel profiling"))

        # Run All / Cancel are the only footer action buttons; both use the
        # shared button style (thin outline, accent text) with a small side
        # inset so they do not touch the sidebar edges.
        self._run_all_btn = QPushButton("Run All")
        self._run_all_btn.setProperty("class", "sidebar-footer-btn")
        self._run_all_btn.clicked.connect(self.run_all_clicked.emit)
        layout.addWidget(self._run_all_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setProperty("class", "sidebar-footer-btn")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self.cancel_clicked.emit)
        layout.addWidget(self._cancel_btn)

        self._items["input"].set_active(True)

    def _add_item(self, page_id: str) -> None:
        for pid, label in self.PAGES:
            if pid == page_id:
                item = SidebarItem(page_id, label)
                item.clicked.connect(self._on_item_clicked)
                self.layout().addWidget(item)
                self._items[page_id] = item
                return

    def set_cancel_visible(self, visible: bool) -> None:
        self._cancel_btn.setVisible(visible)

    def set_action_buttons_enabled(self, enabled: bool) -> None:
        """Enable/disable Run All (disabled while a run is active)."""
        self._run_all_btn.setEnabled(enabled)

    def _emit_display_range(self) -> None:
        self.display_range_changed.emit(self._vmin_spin.value(), self._vmax_spin.value())

    def _on_item_clicked(self, page_id: str) -> None:
        for key, item in self._items.items():
            item.set_active(key == page_id)
        self.navigation_changed.emit(page_id)
