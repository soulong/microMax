from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from microVis.widgets.ui_spec import COMPACT_LINE_EDIT_STYLE, FORM_LABEL_WIDTH_WIDE

DEFAULT_MERGE_DB = "merge.db"


class DataView(QWidget):
    """Data tab: dataset selection + DB selection + ONE integrated plot area.

    The dataset directory is a type/browse/drop line edit (same style as
    microProfiler's Input dir). "Select DB" accepts any number of profiler
    and/or infer DB files of the current dataset; their objects are merged
    into one table rendered by the single plot view set via
    :meth:`set_plot_view`. "Write to DB" persists the integrated table
    (profiler + infer + merged Excel metadata) into a NEW database whose
    file name the small edit after the button controls (default merge.db).
    """

    dataset_browse_clicked = Signal()
    load_dataset_clicked = Signal()
    select_db_clicked = Signal()
    metadata_browse_clicked = Signal()
    metadata_merge_clicked = Signal()
    metadata_clear_clicked = Signal()
    write_to_db_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # Scoped style hook: the Data page's buttons are compact (see
        # QWidget#data-view rules in resources/style.qss).
        self.setObjectName("data-view")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # All fixed controls live in one top container whose height is capped
        # at its size hint: the plot area below absorbs the extra space, so
        # the controls never drift to the bottom when the page is empty.
        top = QWidget()
        top.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        # ── Row 1: dataset path line edit + browse + Reset ──
        row1 = QHBoxLayout()
        row1.setAlignment(Qt.AlignBottom)
        self._dataset_edit = QLineEdit()
        self._dataset_edit.setPlaceholderText(
            "Dataset directory — type, browse, or drop a folder here")
        row1.addWidget(self._dataset_edit, 1)

        self._btn_dataset_browse = QPushButton("Browse...")
        self._btn_dataset_browse.setProperty("class", "primary")
        self._btn_dataset_browse.setToolTip("Browse for a dataset directory")
        self._btn_dataset_browse.clicked.connect(self.dataset_browse_clicked)
        row1.addWidget(self._btn_dataset_browse)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setProperty("class", "primary")
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self.reset_clicked)
        row1.addWidget(self._btn_reset)
        top_layout.addLayout(row1)

        # ── Pattern inputs (always visible so the button row never moves) ──

        # Helper to build a label+input row
        def _pattern_row(label_text: str, placeholder: str) -> tuple[QWidget, QLineEdit]:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(FORM_LABEL_WIDTH_WIDE)
            lbl.setStyleSheet("font-weight: bold; color: #7a9aaa;")
            row.addWidget(lbl)
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
            row.addWidget(edit, stretch=1)
            container = QWidget()
            container.setLayout(row)
            return container, edit

        pat1, self._pattern_image_edit = _pattern_row(
            "Image Pattern",
            r"e.g. (?P<field>\d+)...ch(?P<channel>\d+)\.tiff",
        )
        top_layout.addWidget(pat1)

        pat2, self._pattern_mask_edit = _pattern_row(
            "Mask Pattern",
            r"e.g. ...cp_masks_(?P<mask_name>.+)\.png",
        )
        top_layout.addWidget(pat2)

        pat3, self._pattern_subdir_edit = _pattern_row(
            "Image Subdir",
            "e.g. Images/  (leave empty to scan root)",
        )
        top_layout.addWidget(pat3)

        # ── Button row: Load Dataset + Select DB + metadata actions ──
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)

        self._btn_load_dataset = QPushButton("Load Dataset")
        self._btn_load_dataset.setProperty("class", "primary")
        self._btn_load_dataset.setEnabled(False)
        self._btn_load_dataset.clicked.connect(self.load_dataset_clicked)
        btn_row.addWidget(self._btn_load_dataset)

        self._btn_select_db = QPushButton("Select DB")
        self._btn_select_db.setProperty("class", "primary")
        self._btn_select_db.setEnabled(False)
        self._btn_select_db.setToolTip(
            "Pick one or more profiler.db and/or infer.db files of THIS "
            "dataset (multi-select). Their object tables are merged into a "
            "single integrated table so measurements and predictions can be "
            "cross-plotted. You can also drop files on this button.")
        self._btn_select_db.clicked.connect(self.select_db_clicked)
        btn_row.addWidget(self._btn_select_db)

        btn_row.addStretch()

        self._btn_meta_browse = QPushButton("Select Metadata")
        self._btn_meta_browse.setProperty("class", "primary")
        self._btn_meta_browse.setEnabled(False)
        self._btn_meta_browse.clicked.connect(self.metadata_browse_clicked)
        btn_row.addWidget(self._btn_meta_browse)

        self._btn_merge = QPushButton("Merge")
        self._btn_merge.setProperty("class", "primary")
        self._btn_merge.setEnabled(False)
        self._btn_merge.clicked.connect(self.metadata_merge_clicked)
        btn_row.addWidget(self._btn_merge)

        self._btn_meta_clear = QPushButton("Clear")
        self._btn_meta_clear.setProperty("class", "primary")
        self._btn_meta_clear.setEnabled(False)
        self._btn_meta_clear.clicked.connect(self.metadata_clear_clicked)
        btn_row.addWidget(self._btn_meta_clear)

        self._btn_write_db = QPushButton("Write to DB")
        self._btn_write_db.setProperty("class", "primary")
        self._btn_write_db.setEnabled(False)
        self._btn_write_db.setToolTip(
            "Write the integrated table (profiler + infer + merged metadata "
            "columns) into a NEW database next to the dataset.")
        self._btn_write_db.clicked.connect(self._on_write_to_db)
        btn_row.addWidget(self._btn_write_db)

        # Output DB name for Write to DB (relative to the dataset dir).
        self._merge_db_edit = QLineEdit(DEFAULT_MERGE_DB)
        self._merge_db_edit.setToolTip(
            "File name of the database written by 'Write to DB' "
            "(inside the dataset directory).")
        self._merge_db_edit.setFixedWidth(110)
        self._merge_db_edit.setStyleSheet(COMPACT_LINE_EDIT_STYLE)
        btn_row.addWidget(self._merge_db_edit)

        top_layout.addLayout(btn_row)

        layout.addWidget(top)
        # Pin the controls to the top even while the plot area is hidden
        # (an empty plot slot must not shift them down/center them).
        layout.setAlignment(top, Qt.AlignTop)

        # ── Integrated plot area (single view, set by MainWindow) ──
        self._plot_view: QWidget | None = None
        self._plot_slot = QVBoxLayout()
        self._plot_slot.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._plot_slot, 1)

    # ── Public methods ─────────────────────────────────────────────────────

    def set_patterns(self, image: str, mask: str, subdir: str) -> None:
        self._pattern_image_edit.setText(image)
        self._pattern_mask_edit.setText(mask)
        self._pattern_subdir_edit.setText(subdir)
        self._btn_load_dataset.setEnabled(True)
        self._btn_reset.setEnabled(True)

    def get_patterns(self) -> tuple[str, str, str]:
        return (
            self._pattern_image_edit.text().strip(),
            self._pattern_mask_edit.text().strip(),
            self._pattern_subdir_edit.text().strip(),
        )

    # The dataset directory is a type/browse/drop line edit; the browse
    # button is exposed so MainWindow can wire drag-and-drop onto it too.
    @property
    def dataset_path_edit(self) -> QLineEdit:
        return self._dataset_edit

    @property
    def dataset_browse_button(self) -> QPushButton:
        return self._btn_dataset_browse

    @property
    def select_db_button(self) -> QPushButton:
        return self._btn_select_db

    @property
    def metadata_browse_button(self) -> QPushButton:
        return self._btn_meta_browse

    def get_dataset_path(self) -> str:
        return self._dataset_edit.text().strip()

    def set_dataset_path(self, text: str) -> None:
        self._dataset_edit.setText(text)

    def set_meta_browse_enabled(self, enabled: bool) -> None:
        self._btn_meta_browse.setEnabled(enabled)

    def set_metadata_label(self, text: str | None) -> None:
        has_meta = text is not None
        self._btn_merge.setEnabled(has_meta)
        self._btn_meta_clear.setEnabled(has_meta)

    def set_write_to_db_enabled(self, enabled: bool) -> None:
        """Write to DB needs merged DB data (not the Excel metadata)."""
        self._btn_write_db.setEnabled(enabled)

    def set_db_buttons_enabled(self, enabled: bool) -> None:
        self._btn_select_db.setEnabled(enabled)

    def get_merge_db_name(self) -> str:
        """Output DB file name for Write to DB (defaults to merge.db)."""
        name = self._merge_db_edit.text().strip()
        return name if name else DEFAULT_MERGE_DB

    # ── Integrated plot area ───────────────────────────────────────────────

    def set_plot_view(self, widget: QWidget) -> None:
        """Install THE plot view (idempotent: re-installing is a no-op)."""
        if self._plot_view is widget:
            return
        self.clear_plot_view()
        self._plot_view = widget
        self._plot_slot.addWidget(widget)
        widget.setVisible(True)

    def clear_plot_view(self) -> None:
        """Remove the plot view widget (ownership stays with MainWindow)."""
        if self._plot_view is None:
            return
        self._plot_slot.removeWidget(self._plot_view)
        self._plot_view.setVisible(False)
        self._plot_view = None

    def reset(self) -> None:
        """Reset to initial startup state."""
        self._dataset_edit.clear()
        self._pattern_image_edit.clear()
        self._pattern_mask_edit.clear()
        self._pattern_subdir_edit.clear()
        self._btn_load_dataset.setEnabled(False)
        self._btn_reset.setEnabled(False)
        self._btn_select_db.setEnabled(False)
        self._btn_meta_browse.setEnabled(False)
        self._btn_merge.setEnabled(False)
        self._btn_meta_clear.setEnabled(False)
        self._btn_write_db.setEnabled(False)
        self.clear_plot_view()

    def _on_write_to_db(self) -> None:
        reply = QMessageBox.question(
            self,
            "Write Merged Database",
            "This writes the integrated table (profiler + infer objects plus "
            "the merged metadata columns) into a NEW database file inside "
            "the dataset directory. The source DBs are not modified.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.write_to_db_clicked.emit()
