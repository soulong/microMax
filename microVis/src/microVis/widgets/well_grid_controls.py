from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from microVis.widgets._event_filter import NoScrollComboBox
from microVis.widgets.ui_spec import (
    CONTROLS_MARGIN,
    CONTROLS_MAX_WIDTH,
    CONTROLS_MIN_WIDTH,
    CONTROLS_SPACING,
    controls_pane_style,
    centered_row,
    form_row,
    small_button,
)


class WellGridControls(QScrollArea):
    """Left sidebar controls for the well plate grid."""

    select_all_clicked = Signal()
    clear_clicked = Signal()
    image_block_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(CONTROLS_MIN_WIDTH)
        self.setMaximumWidth(CONTROLS_MAX_WIDTH)

        container = QWidget()
        container.setStyleSheet(controls_pane_style())
        layout = QVBoxLayout(container)
        layout.setContentsMargins(CONTROLS_MARGIN, CONTROLS_MARGIN,
                                  CONTROLS_MARGIN, CONTROLS_MARGIN)
        layout.setSpacing(CONTROLS_SPACING)

        # Format
        self._plate_fmt = NoScrollComboBox()
        layout.addLayout(form_row("Format", self._plate_fmt))

        # Color by
        self._column = NoScrollComboBox()
        self._column.setEditable(True)
        self._column.setInsertPolicy(QComboBox.NoInsert)
        self._column.completer().setFilterMode(Qt.MatchContains)
        self._column.completer().setCompletionMode(QCompleter.PopupCompletion)
        self._column.lineEdit().setPlaceholderText("Type to filter...")
        layout.addLayout(form_row("Color by", self._column))

        # Aggregation
        self._agg = NoScrollComboBox()
        layout.addLayout(form_row("Agg", self._agg))

        # Select All / Clear (centered)
        self._select_all_btn = small_button("Select All", width=80)
        self._select_all_btn.clicked.connect(self.select_all_clicked)
        self._clear_btn = small_button("Clear", width=64)
        self._clear_btn.clicked.connect(self.clear_clicked)
        layout.addLayout(centered_row(self._select_all_btn, self._clear_btn))

        # Colors
        self._cmap = NoScrollComboBox()
        layout.addLayout(form_row("Colors", self._cmap))

        # Palette
        self._palette = NoScrollComboBox()
        layout.addLayout(form_row("Palette", self._palette))

        # Image block toggle (centered)
        self._image_blocked = False
        self._image_block_btn = small_button("Block Image", width=100)
        self._image_block_btn.setToolTip(
            "Block image loading and display.\n"
            "Already shown images stay visible."
        )
        self._image_block_btn.clicked.connect(self._on_toggle_image_block)
        layout.addLayout(centered_row(self._image_block_btn))

        layout.addStretch()

        self.setWidget(container)

    # ── Public API ──────────────────────────────────────────────────

    @property
    def plate_format(self) -> NoScrollComboBox:
        return self._plate_fmt

    @property
    def column(self) -> NoScrollComboBox:
        return self._column

    @property
    def aggregation(self) -> NoScrollComboBox:
        return self._agg

    @property
    def colormap(self) -> NoScrollComboBox:
        return self._cmap

    @property
    def palette(self) -> NoScrollComboBox:
        return self._palette

    def _on_toggle_image_block(self) -> None:
        self._image_blocked = not self._image_blocked
        if self._image_blocked:
            self._image_block_btn.setText("Show Image")
            self._image_block_btn.setProperty("class", "danger")
            self._image_block_btn.setStyleSheet(
                "QPushButton { background-color: #c04040; color: white; "
                "font-size: 9pt; padding: 2px 6px; }"
            )
        else:
            self._image_block_btn.setText("Block Image")
            self._image_block_btn.setProperty("class", "secondary")
            self._image_block_btn.setStyleSheet("")
        self._image_block_btn.style().unpolish(self._image_block_btn)
        self._image_block_btn.style().polish(self._image_block_btn)
        self.image_block_toggled.emit(self._image_blocked)

    def reset_image_block(self) -> None:
        """Reset the image-block toggle to the unblocked state (full reset)."""
        self._image_blocked = False
        self._image_block_btn.setText("Block Image")
        self._image_block_btn.setProperty("class", "secondary")
        self._image_block_btn.setStyleSheet("")
        self._image_block_btn.style().unpolish(self._image_block_btn)
        self._image_block_btn.style().polish(self._image_block_btn)
