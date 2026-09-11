from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QHBoxLayout,
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
        outer = QVBoxLayout(container)
        outer.setContentsMargins(CONTROLS_MARGIN, CONTROLS_MARGIN,
                                 CONTROLS_MARGIN, CONTROLS_MARGIN)
        outer.setSpacing(0)

        # Titleless rounded box around the whole well-grid control bar. It
        # sits at the same pane margin as the Image-controls group boxes so
        # both columns line up vertically.
        box = QWidget()
        box.setProperty("class", "panel-box")
        box.setStyleSheet(controls_pane_style())
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 12, 12, 12)
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

        # Select All / Clear stay left; Block Image sits after Clear at the
        # right edge of the box. Widths follow the captions (no stretching).
        self._select_all_btn = small_button("Select All")
        self._select_all_btn.clicked.connect(self.select_all_clicked)
        self._clear_btn = small_button("Clear")
        self._clear_btn.clicked.connect(self.clear_clicked)
        self._image_blocked = False
        self._image_block_btn = small_button("Block Image")
        self._image_block_btn.setToolTip(
            "Block image loading and display.\n"
            "Already shown images stay visible."
        )
        self._image_block_btn.clicked.connect(self._on_toggle_image_block)
        action_row = QHBoxLayout()
        action_row.setSpacing(4)
        action_row.addWidget(self._select_all_btn)
        action_row.addWidget(self._clear_btn)
        action_row.addStretch()
        action_row.addWidget(self._image_block_btn)
        layout.addLayout(action_row)

        # Colors
        self._cmap = NoScrollComboBox()
        layout.addLayout(form_row("Colors", self._cmap))

        # Palette
        self._palette = NoScrollComboBox()
        layout.addLayout(form_row("Palette", self._palette))

        layout.addStretch()

        outer.addWidget(box)
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
        self._image_block_btn.setText(
            "Show Image" if self._image_blocked else "Block Image")
        self.image_block_toggled.emit(self._image_blocked)

    def reset_image_block(self) -> None:
        """Reset the image-block toggle to the unblocked state (full reset)."""
        self._image_blocked = False
        self._image_block_btn.setText("Block Image")
