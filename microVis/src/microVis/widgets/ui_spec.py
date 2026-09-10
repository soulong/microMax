"""Central UI sizing/style tokens for the microVis panel widgets.

Every hard-coded control size, spacing and shared inline style used by
the left control panes and the small buttons/checkboxes lives HERE, so
the layout can be re-tuned in one place instead of hunting magic numbers
across widget files. Colors and look-and-feel rules stay in
``resources/style.qss``; this module only owns *geometry* and the few
per-widget font/padding tweaks Qt cannot express per-class in the app
stylesheet.

Naming convention: tokens are plain ``UPPER_CASE`` ints/tuples; style
strings end with ``_STYLE``; reusable widget builders are small functions
at the bottom.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QWidget

# ── Left control panes (WellGridControls / ImageControls) ────────────────────
CONTROLS_MIN_WIDTH = 260      # splitter never shrinks a pane below this
CONTROLS_MAX_WIDTH = 320      # splitter never grows a pane beyond this
CONTROLS_MARGIN = 6           # inner padding of a control pane
CONTROLS_SPACING = 4          # vertical gap between the group boxes

# ── Form rows ("label: input") inside the control panes ──────────────────────
FORM_LABEL_WIDTH = 60         # fixed label column (Image/Well-grid rows)
FORM_LABEL_WIDTH_WIDE = 120   # wider label column (Data-page pattern rows)
ROW_SPACING = 4               # gap between label and field

# ── Compact inputs (combos / spinboxes / line edits in control panes) ────────
CONTROL_MIN_H = 18
CONTROL_MAX_H = 22
# Per-widget variant for single line edits (same look as COMPACT_INPUT_STYLE,
# for the few inputs that need the tweak individually).
COMPACT_LINE_EDIT_STYLE = (
    f"min-height: {CONTROL_MIN_H}px; max-height: {CONTROL_MAX_H}px; "
    "font-size: 8pt; padding: 2px 3px;"
)
# One shared font/padding tweak — applied per-pane because the app-wide
# stylesheet must keep normal-sized inputs for the Data page.
COMPACT_INPUT_STYLE = f"""
QComboBox, QDoubleSpinBox, QSpinBox, QSlider {{
    min-height: {CONTROL_MIN_H}px;
    max-height: {CONTROL_MAX_H}px;
    font-size: 8pt;
    padding: 2px 3px;
    min-width: 0;
}}
QLabel {{
    font-size: 8pt;
}}
QPushButton {{
    font-size: 9pt;
    padding: 2px 6px;
}}
"""
# QGroupBox header look for the ImageControls panes (WellGridControls has
# no group boxes, so it omits this block via `controls_pane_style`).
GROUP_BOX_STYLE = """
QGroupBox {
    font-size: 8pt;
    font-weight: bold;
    color: #5a8a9a;
    border: 1px solid #3a3a4a;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
"""

# ── Small action buttons ─────────────────────────────────────────────────────
BTN_STD_WIDTH = 64            # Auto / Reset / Clear
BTN_WIDE_WIDTH = 150          # full-action buttons (Write Label to DB / Export)
BTN_HEIGHT = 24
BTN_MINI_WIDTH = 36           # Add / Del / "..."
BTN_MINI_HEIGHT = 20
# Font/padding shrink for the tiny buttons (Add/Del/...) and the mini
# All/Clear pair inside multi-select filters.
BTN_MINI_STYLE = "font-size: 8pt; padding: 1px 4px;"

# ── Horizontal checkbox strips (filter multi-select, class checkboxes) ───────
CHECK_STRIP_MAX_H = 24
SMALL_CHECKBOX_STYLE = (
    "QCheckBox { font-size: 7pt; spacing: 2px; } "
    "QCheckBox::indicator { width: 12px; height: 12px; }"
)

# ── Splitters of the Image page ──────────────────────────────────────────────
H_SPLITTER_SIZES = (280, 600)         # controls | canvas (both horizontal splitters)
V_SPLITTER_SIZES = (250, 750, 0)      # well grid | image view | label panel (hidden)
LABEL_CLASS_RATIOS = (0.25, 0.50, 0.25)   # when the label panel first appears
NO_LABEL_RATIOS = (0.30, 0.70, 0.0)       # after the last label class is removed


def controls_pane_style(include_group_boxes: bool = False) -> str:
    """Stylesheet for a left control pane (compact inputs + small buttons).

    include_group_boxes: add the group-box header block — only ImageControls
    uses group boxes; WellGridControls lays its rows out flat.
    """
    extra = GROUP_BOX_STYLE if include_group_boxes else ""
    return COMPACT_INPUT_STYLE + extra


def form_row(label_text: str, widget: QWidget,
             label_width: int = FORM_LABEL_WIDTH) -> QHBoxLayout:
    """Build a `label: input` row; the field stretches, the label does not."""
    row = QHBoxLayout()
    row.setSpacing(ROW_SPACING)
    row.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(label_width)
    row.addWidget(lbl)
    row.addWidget(widget, stretch=1)
    return row


def small_button(text: str, width: int = BTN_STD_WIDTH,
                 height: int = BTN_HEIGHT,
                 mini_style: bool = False) -> QPushButton:
    """Secondary-class action button with the shared fixed size."""
    btn = QPushButton(text)
    btn.setProperty("class", "secondary")
    btn.setFixedSize(width, height)
    if mini_style:
        btn.setStyleSheet(BTN_MINI_STYLE)
    return btn


def centered_row(*widgets: QWidget, spacing: int = 8) -> QHBoxLayout:
    """A row that centers its widgets between two stretches."""
    row = QHBoxLayout()
    row.setSpacing(spacing)
    row.addStretch()
    for w in widgets:
        row.addWidget(w)
    row.addStretch()
    return row


def check_strip() -> QScrollArea:
    """Horizontal, borderless strip that hosts a row of small checkboxes."""
    strip = QScrollArea()
    strip.setWidgetResizable(True)
    strip.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    strip.setMaximumHeight(CHECK_STRIP_MAX_H)
    strip.setStyleSheet("QScrollArea { border: none; background: transparent; }")
    return strip
