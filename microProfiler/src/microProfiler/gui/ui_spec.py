"""Central UI sizing tokens for the microProfiler main window.

Geometry (page margins, button heights, compact-width caps) lives HERE so
re-tuning the layout means editing one file instead of hunting magic
numbers across the GUI package. Colors and look-and-feel stay in
``resources/style.qss``; sizes that must scale with the display go
through :func:`microProfiler.gui.dpi.dp` at the call site.
"""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from microProfiler.gui.dpi import dp

# ── Main window ──────────────────────────────────────────────────────────────
WINDOW_SIZE = (1200, 800)     # initial AND minimum window size
SIDEBAR_WIDTH = 100           # fixed width of the left navigation sidebar

# ── Stacked step pages ───────────────────────────────────────────────────────
PAGE_MARGIN = 4               # outer margin of every stacked page
PAGE_SPACING = 12             # vertical gap between step cards inside a page
SCROLL_CONTENT_SPACING = 12   # gap between cards inside a page's scroll area

# ── Buttons ──────────────────────────────────────────────────────────────────
RUN_BTN_HEIGHT = 32           # "Run Preprocessing/.../Inference" + Load Dataset
STEP_BTN_HEIGHT = 28          # per-panel Apply/Pick/Preview buttons
SIDEBAR_BTN_HEIGHT = 28       # sidebar footer buttons (Load/Reset/Run All/Cancel)
STATUS_BAR_HEIGHT = 24        # bottom progress/status line

# ── Step-card compaction (BaseStepPanel._compact_block) ──────────────────────
COMPACT_MAX_WIDTH = 200       # cap for spins/combos inside step cards
COMPACT_LINEEDIT_WIDTH = 160  # cap for non-path line edits inside step cards


def run_button(text: str) -> QPushButton:
    """Primary action button with the shared run-button height."""
    btn = QPushButton(text)
    btn.setProperty("class", "primary")
    btn.setFixedHeight(dp(RUN_BTN_HEIGHT))
    return btn


def step_button(text: str) -> QPushButton:
    """Primary action button for the per-step Apply/Pick actions."""
    btn = QPushButton(text)
    btn.setProperty("class", "primary")
    btn.setFixedHeight(dp(STEP_BTN_HEIGHT))
    return btn
