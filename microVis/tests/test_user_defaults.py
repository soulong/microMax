"""microVis ~/.micromax user defaults: round-trip, merge, window geometry."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import microVis.user_defaults as ud  # noqa: E402


def _point_defaults(tmp_path, monkeypatch):
    f = tmp_path / ".micromax"
    monkeypatch.setattr(ud, "DEFAULTS_FILE", f)
    return f


def test_round_trip_and_merge(tmp_path, monkeypatch):
    _point_defaults(tmp_path, monkeypatch)
    assert ud.get_user_defaults() == {}
    ud.update_user_defaults("window", {"width": 1600, "height": 900})
    ud.update_user_defaults("other", {"key": 1})
    ud.update_user_defaults("window", {"height": 950})
    data = ud.get_user_defaults()
    assert data["window"] == {"width": 1600, "height": 950}
    assert data["other"] == {"key": 1}
    assert ud.DEFAULTS_FILE.exists()


def test_shared_file_keeps_other_app_section(tmp_path, monkeypatch):
    """One ~/.micromax file holds both apps' sections side by side."""
    from microBase import load_yaml, save_yaml

    f = _point_defaults(tmp_path, monkeypatch)
    save_yaml(f, {
        "microprofiler": {"window": {"width": 1200, "height": 800}},
        "microvis": {"window": {"width": 1500, "height": 1000}},
    })

    # microVis reads/updates only its own section; microProfiler's survives.
    assert ud.get_user_defaults() == {"window": {"width": 1500, "height": 1000}}
    ud.update_user_defaults("window", {"height": 1100})
    data = load_yaml(f)
    assert data["microprofiler"]["window"] == {"width": 1200, "height": 800}
    assert data["microvis"]["window"] == {"width": 1500, "height": 1100}


def test_main_window_restores_and_saves_size(tmp_path, monkeypatch):
    _point_defaults(tmp_path, monkeypatch)
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    ud.update_user_defaults("window", {"width": 1300, "height": 850})

    from microVis.main_window import MainWindow

    win = MainWindow()
    try:
        assert (win.width(), win.height()) == (1300, 850)
        win.resize(1340, 870)
        win._save_window_size()
    finally:
        win.close()
    app.processEvents()
    assert ud.get_user_defaults()["window"] == {"width": 1340, "height": 870}


# ── Logging: INFO only ───────────────────────────────────────────────────────


def test_set_log_file_is_info_only(tmp_path):
    import logging

    import microVis.log_utils as lu

    logging.getLogger().setLevel(logging.INFO)
    path = tmp_path / "microVis.log"
    lu.set_log_file(path)
    try:
        assert lu._FILE_HANDLER is not None
        assert lu._FILE_HANDLER.level == logging.INFO
        test_logger = logging.getLogger("microVis.test_level")
        test_logger.debug("debug-should-not-appear")
        test_logger.info("info-should-appear")
        lu._FILE_HANDLER.flush()
        text = path.read_text(encoding="utf-8")
        assert "info-should-appear" in text
        assert "debug-should-not-appear" not in text
    finally:
        root = logging.getLogger()
        root.removeHandler(lu._FILE_HANDLER)
        lu._FILE_HANDLER.close()
        lu._FILE_HANDLER = None
