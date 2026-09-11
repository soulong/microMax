"""Central UI sizing tokens for the microProfiler main window.

Geometry (page margins, compact-width caps) lives HERE so re-tuning the
layout means editing one file instead of hunting magic numbers across the
GUI package. Colors and look-and-feel stay in ``resources/style.qss``;
sizes that must scale with the display go through
:func:`microProfiler.gui.dpi.dp` at the call site.
"""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton

# ── Main window ──────────────────────────────────────────────────────────────
WINDOW_SIZE = (1200, 800)     # initial AND minimum window size
SIDEBAR_WIDTH = 100           # fixed width of the left navigation sidebar

# ── Stacked step pages ───────────────────────────────────────────────────────
PAGE_MARGIN = 4               # outer margin of every stacked page
PAGE_SPACING = 12             # vertical gap between step cards inside a page
SCROLL_CONTENT_SPACING = 12   # gap between cards inside a page's scroll area

# ── Buttons ──────────────────────────────────────────────────────────────────
# Button size/look comes from the app stylesheet (compact neutral blocks, see
# AGENTS.md "GUI style conventions"); no fixed heights here.
STATUS_BAR_HEIGHT = 24        # bottom progress/status line

# ── Step-card compaction (BaseStepPanel._compact_block) ──────────────────────
COMPACT_MAX_WIDTH = 200       # cap for spins/combos inside step cards
COMPACT_LINEEDIT_WIDTH = 160  # cap for non-path line edits inside step cards

# ── Panel content scale ──────────────────────────────────────────────────────
# Step cards and the Input page are control panels: their labels/inputs/
# checkboxes/buttons use the 9pt panel scale, while QGroupBox titles stay on
# the 11pt title scale (QSS). Auxiliary text (hints, status, mini buttons)
# stays 8pt.
PANEL_CONTENT_STYLE = """
QLabel, QCheckBox, QRadioButton {
    font-size: 9pt;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    font-size: 9pt;
}
QPushButton {
    font-size: 9pt;
}
QLabel[class="hint"] {
    font-size: 8pt;
}
"""

# Pattern regex line edits (Input page) share microVis's Data-page style.
PATTERN_LABEL_STYLE = "font-size: 9pt;"
PATTERN_EDIT_STYLE = "font-family: Consolas, monospace; font-size: 9pt;"


def run_button(text: str) -> QPushButton:
    """Action button ("Run ..." / Load Dataset) — the shared button style."""
    return QPushButton(text)


def step_button(text: str) -> QPushButton:
    """Action button for the per-step Apply actions — the shared style."""
    return QPushButton(text)
