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

# ── Main window ──────────────────────────────────────────────────────────────
WINDOW_DEFAULT_SIZE = (1500, 1000)   # first start, before a saved size exists
WINDOW_MIN_SIZE = (1200, 800)        # panes never squeeze below this

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
# Panel content scale (see the GUI style section of AGENTS.md): 9pt labels /
# inputs / buttons, while window and group titles stay 11pt. Input padding and
# the button look come from the app stylesheet; this block only caps heights.
CONTROL_MIN_H = 20
CONTROL_MAX_H = 24
# Per-widget variant for single line edits (same look as COMPACT_INPUT_STYLE,
# for the few inputs that need the tweak individually).
COMPACT_LINE_EDIT_STYLE = (
    f"min-height: {CONTROL_MIN_H}px; max-height: {CONTROL_MAX_H}px; "
    "font-size: 9pt;"
)
COMPACT_INPUT_STYLE = f"""
QComboBox, QDoubleSpinBox, QSpinBox, QSlider {{
    min-height: {CONTROL_MIN_H}px;
    max-height: {CONTROL_MAX_H}px;
    min-width: 0;
}}
QLabel {{
    font-size: 9pt;
}}
"""

# ── Pattern regex inputs (Data page; shared wording/style with
#    microProfiler's Input page) ─────────────────────────────────────────────
PATTERN_LABEL_STYLE = "font-size: 9pt;"
PATTERN_EDIT_STYLE = "font-family: Consolas, monospace; font-size: 9pt;"

# ── Small action buttons ─────────────────────────────────────────────────────
BTN_STD_WIDTH = 64            # Auto / Reset / Clear
BTN_WIDE_WIDTH = 150          # full-action buttons (Write Label to DB / Export)
BTN_HEIGHT = 26               # matches the stylesheet's natural button height
BTN_MINI_WIDTH = 36           # Add / Del / "..."
BTN_MINI_HEIGHT = 20
# Font/padding shrink for the tiny buttons (Add/Del/...) and the mini
# All/Clear pair inside multi-select filters.
BTN_MINI_STYLE = "font-size: 8pt; padding: 1px 4px;"

# ── Horizontal checkbox strips (filter multi-select, class checkboxes) ───────
CHECK_STRIP_MAX_H = 24
SMALL_CHECKBOX_STYLE = (
    "QCheckBox { font-size: 8pt; spacing: 2px; } "
    "QCheckBox::indicator { width: 12px; height: 12px; }"
)

# ── Splitters of the Image page ──────────────────────────────────────────────
H_SPLITTER_SIZES = (280, 600)         # controls | canvas (both horizontal splitters)
V_SPLITTER_SIZES = (220, 780, 0)      # well grid | image view | label panel (hidden)
LABEL_CLASS_RATIOS = (0.25, 0.50, 0.25)   # when the label panel first appears
NO_LABEL_RATIOS = (0.30, 0.70, 0.0)       # after the last label class is removed


def controls_pane_style() -> str:
    """Compact 9pt content style for a left control pane.

    Group-box and button look-and-feel come from the app stylesheet; this
    only caps the compact input heights and label size.
    """
    return COMPACT_INPUT_STYLE


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


def small_button(text: str, width: int | None = None,
                 height: int = BTN_HEIGHT,
                 mini_style: bool = False) -> QPushButton:
    """Action button; height is fixed, width=None sizes to the text."""
    btn = QPushButton(text)
    if width is None:
        btn.setFixedHeight(height)
    else:
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
