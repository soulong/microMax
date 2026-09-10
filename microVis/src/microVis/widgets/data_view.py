from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class DataView(QWidget):
    """Data tab: dataset selection + DB plot tabs.

    The dataset directory is a type/browse/drop line edit (same style as
    microProfiler's Input dir). Two DB selectors open plot tabs in the area
    below: the profiler DB (profiler.db object tables) and the infer DB
    (microModel inference/reduction scatter).
    """

    dataset_browse_clicked = Signal()
    load_dataset_clicked = Signal()
    profiler_db_browse_clicked = Signal()
    infer_db_browse_clicked = Signal()
    metadata_browse_clicked = Signal()
    metadata_merge_clicked = Signal()
    metadata_clear_clicked = Signal()
    write_to_db_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
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
            lbl.setFixedWidth(120)
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

        # ── Button row: Load Dataset + two DB selectors + metadata actions ──
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)

        self._btn_load_dataset = QPushButton("Load Dataset")
        self._btn_load_dataset.setProperty("class", "primary")
        self._btn_load_dataset.setEnabled(False)
        self._btn_load_dataset.clicked.connect(self.load_dataset_clicked)
        btn_row.addWidget(self._btn_load_dataset)

        self._btn_profiler_db = QPushButton("Select Profiler DB")
        self._btn_profiler_db.setProperty("class", "primary")
        self._btn_profiler_db.setEnabled(False)
        self._btn_profiler_db.setToolTip(
            "Pick one or more microProfiler profiler.db files (object profiling). "
            "Each DB opens its own plot tab; you can also drop files on this button.")
        self._btn_profiler_db.clicked.connect(self.profiler_db_browse_clicked)
        btn_row.addWidget(self._btn_profiler_db)

        self._btn_infer_db = QPushButton("Select Infer DB")
        self._btn_infer_db.setProperty("class", "primary")
        self._btn_infer_db.setEnabled(False)
        self._btn_infer_db.setToolTip(
            "Pick one or more microModel infer.db files (inference + reduction "
            "tables). Each DB opens its own scatter tab; you can also drop "
            "files on this button.")
        self._btn_infer_db.clicked.connect(self.infer_db_browse_clicked)
        btn_row.addWidget(self._btn_infer_db)

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
        self._btn_write_db.clicked.connect(self._on_write_to_db)
        btn_row.addWidget(self._btn_write_db)

        top_layout.addLayout(btn_row)

        layout.addWidget(top)
        # Pin the controls to the top even while the plot tabs are hidden
        # (an empty hidden tab item must not shift them down/center them).
        layout.setAlignment(top, Qt.AlignTop)

        # ── Plot tabs (filled by MainWindow when a DB is selected) ──
        self._plot_tabs = QTabWidget()
        self._plot_tabs.setVisible(False)
        layout.addWidget(self._plot_tabs, 1)

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
    def profiler_db_browse_button(self) -> QPushButton:
        return self._btn_profiler_db

    @property
    def infer_db_browse_button(self) -> QPushButton:
        return self._btn_infer_db

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
        self._btn_write_db.setEnabled(has_meta)

    def set_db_buttons_enabled(self, enabled: bool) -> None:
        self._btn_profiler_db.setEnabled(enabled)
        self._btn_infer_db.setEnabled(enabled)

    # ── Plot tabs ──────────────────────────────────────────────────────────

    def show_plot_tab(self, title: str, widget: QWidget) -> None:
        """Add (or activate) a plot tab. Re-showing the same widget activates
        its existing tab instead of adding a duplicate."""
        index = self._plot_tabs.indexOf(widget)
        if index < 0:
            index = self._plot_tabs.addTab(widget, title)
        else:
            self._plot_tabs.setTabText(index, title)
        self._plot_tabs.setCurrentIndex(index)
        self._plot_tabs.setVisible(True)

    def clear_plot_tabs(self) -> None:
        """Remove and delete every plot tab (dataset change / reset).

        Each tab is a per-DB widget owned by MainWindow, which drops its
        references before calling this.
        """
        while self._plot_tabs.count():
            w = self._plot_tabs.widget(0)
            self._plot_tabs.removeTab(0)
            if w is not None:
                w.deleteLater()
        self._plot_tabs.setVisible(False)

    def reset(self) -> None:
        """Reset to initial startup state."""
        self._dataset_edit.clear()
        self._pattern_image_edit.clear()
        self._pattern_mask_edit.clear()
        self._pattern_subdir_edit.clear()
        self._btn_load_dataset.setEnabled(False)
        self._btn_reset.setEnabled(False)
        self._btn_profiler_db.setEnabled(False)
        self._btn_infer_db.setEnabled(False)
        self._btn_meta_browse.setEnabled(False)
        self._btn_merge.setEnabled(False)
        self._btn_meta_clear.setEnabled(False)
        self._btn_write_db.setEnabled(False)
        self.clear_plot_tabs()

    def _on_write_to_db(self) -> None:
        reply = QMessageBox.question(
            self,
            "Write Metadata to Databases",
            "This will add/update the merged metadata columns (matched by well) "
            "in every loaded profiler DB and infer DB.\n\n"
            "Existing columns and rows are preserved.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.write_to_db_clicked.emit()
