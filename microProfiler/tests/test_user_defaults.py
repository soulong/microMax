"""~/.micromax user defaults: persistence + inference panel pre-fill."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import microProfiler.user_defaults as ud  # noqa: E402
from microProfiler.user_defaults import (  # noqa: E402
    ensure_user_defaults,
    get_user_defaults,
    update_user_defaults,
)


@pytest.fixture
def defaults_dir(tmp_path, monkeypatch):
    """Point the shared ~/.micromax config file at a temp path."""
    import microProfiler.user_defaults as ud
    f = tmp_path / ".micromax"
    monkeypatch.setattr(ud, "DEFAULTS_FILE", f)
    return f


def test_update_and_get_round_trip(defaults_dir):
    update_user_defaults("inference", {"model": "D:/m/model.pt"})
    update_user_defaults("inference", {"reducer": ["D:/r/umap.pkl"],
                                       "cluster": "D:/r/cluster.pkl"})
    data = get_user_defaults()
    assert data["inference"]["model"] == "D:/m/model.pt"
    assert data["inference"]["reducer"] == ["D:/r/umap.pkl"]
    assert data["inference"]["cluster"] == "D:/r/cluster.pkl"
    assert ud.DEFAULTS_FILE.exists()


def test_merge_keeps_other_sections(defaults_dir):
    update_user_defaults("inference", {"model": "a.pt"})
    update_user_defaults("other", {"key": 1})
    update_user_defaults("inference", {"cluster": "c.pkl"})
    data = get_user_defaults()
    # Sections and earlier keys survive an unrelated update.
    assert data["inference"] == {"model": "a.pt", "cluster": "c.pkl"}
    assert data["other"] == {"key": 1}


def test_get_defaults_missing_file(defaults_dir):
    assert get_user_defaults() == {}


def test_ensure_user_defaults_fills_missing_keys(defaults_dir):
    """The shared file is completed after the first run: existing values are
    kept and missing keys get their defaults."""
    update_user_defaults("window", {"width": 1200, "height": 800})
    update_user_defaults("inference", {"model": "D:/m/model.pt"})

    ensure_user_defaults(
        "inference", {"model": None, "reducer": None, "cluster": None})

    inf = get_user_defaults()["inference"]
    assert inf["model"] == "D:/m/model.pt"      # existing value untouched
    assert inf["reducer"] is None and inf["cluster"] is None
    assert get_user_defaults()["window"] == {"width": 1200, "height": 800}


def test_ensure_user_defaults_creates_missing_file(defaults_dir):
    ensure_user_defaults("inference", {"model": None})
    assert defaults_dir.exists()
    assert get_user_defaults()["inference"]["model"] is None


# ── The pipeline save hook ───────────────────────────────────────────────────


def _inference_cfg(model_path, reducer=None, cluster=None, channels=None):
    from microProfiler.config import PipelineConfig
    from microProfiler.config import (
        InferenceConfig, InferenceEntry, InferenceReductionConfig)

    red = None
    if reducer is not None or cluster is not None:
        red = InferenceReductionConfig(
            enabled=True, reducer=reducer, cluster_enabled=bool(cluster),
            cluster=cluster)
    entry = InferenceEntry(
        model=model_path, mask_name="cell",
        # channels=null/[] means the block is SKIPPED — never "all".
        channels=channels if channels is not None else ["ch1"],
        max_value=65535.0, reduction=red)
    return PipelineConfig(inference=InferenceConfig(run=True, configs=[entry]))


def _run_inference_isolated(tmp_path, monkeypatch, cfg):
    """Run _run_inference with microModel and the auto-merge stubbed out."""
    import microProfiler.pipeline._micromodel_bridge as bridge
    import microProfiler.pipeline.steps as steps
    from microProfiler.progress_collector import NullProgressCollector

    monkeypatch.setattr(bridge, "_build_mm_inference_config",
                        lambda *a, **k: {})
    monkeypatch.setattr(bridge, "_call_micromodel", lambda *a, **k: None)
    monkeypatch.setattr(steps, "_auto_merge_infer", lambda *a, **k: None)
    steps._run_inference(cfg, ds=None, root_dir=tmp_path,
                         progress=NullProgressCollector())


def test_run_inference_remembers_model_reducers_cluster(
        tmp_path, monkeypatch, defaults_dir):
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")   # only existence is checked (microModel stubbed)
    cfg = _inference_cfg(
        str(model), reducer=["D:/r/pca.pkl", "D:/r/umap.pkl"],
        cluster="D:/r/cluster.pkl")
    _run_inference_isolated(tmp_path, monkeypatch, cfg)

    inf = get_user_defaults()["inference"]
    assert inf["model"] == str(model)
    assert inf["reducer"] == ["D:/r/pca.pkl", "D:/r/umap.pkl"]
    assert inf["cluster"] == "D:/r/cluster.pkl"


def test_run_without_reduction_keeps_remembered_paths(
        tmp_path, monkeypatch, defaults_dir):
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    update_user_defaults("inference", {
        "model": "D:/old/model.pt",
        "reducer": ["D:/old/umap.pkl"], "cluster": "D:/old/cluster.pkl"})

    # No reduction key on the entry (both GUI groups unchecked).
    cfg = _inference_cfg(str(model), reducer=None, cluster=None)
    # _inference_cfg only builds a reduction block when a path is given.
    cfg.inference.configs[0].reduction = None
    _run_inference_isolated(tmp_path, monkeypatch, cfg)

    inf = get_user_defaults()["inference"]
    assert inf["model"] == str(model)          # model follows the run
    assert inf["reducer"] == ["D:/old/umap.pkl"]   # remembered paths survive
    assert inf["cluster"] == "D:/old/cluster.pkl"


def test_skipped_blocks_save_nothing(tmp_path, monkeypatch, defaults_dir):
    # channels null/[] = block skipped at runtime -> nothing ran -> no save.
    cfg = _inference_cfg("D:/nonexistent/model.pt", reducer=["D:/r/u.pkl"],
                         channels=[])
    _run_inference_isolated(tmp_path, monkeypatch, cfg)
    assert get_user_defaults() == {}


# ── The GUI pre-fill ─────────────────────────────────────────────────────────


def test_panel_prefills_from_user_defaults(defaults_dir, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    update_user_defaults("inference", {
        "model": "D:/m/model.pt",
        "reducer": ["D:/r/pca.pkl", "D:/r/umap.pkl"],
        "cluster": "D:/r/cluster.pkl"})

    from microProfiler.gui.panels.step_inference import (
        InferenceBlockWidget, InferenceStepPanel)

    panel = InferenceStepPanel.__new__(InferenceStepPanel)
    panel._blocks = []
    block = InferenceBlockWidget(0, [])
    panel._blocks.append(block)
    panel._apply_user_defaults()

    assert block.get_model_path() == "D:/m/model.pt"
    assert block._reducer_path.text() == "D:/r/pca.pkl;D:/r/umap.pkl"
    assert block._cluster_path.text() == "D:/r/cluster.pkl"
    # No group checkboxes anymore — the remembered paths are the run flags.
    assert block.is_reduction_or_cluster_enabled()

    # No defaults (fresh machine) -> untouched blank panel.
    defaults_dir.unlink()
    panel2 = InferenceStepPanel.__new__(InferenceStepPanel)
    panel2._blocks = []
    block2 = InferenceBlockWidget(0, [])
    panel2._blocks.append(block2)
    panel2._apply_user_defaults()
    assert block2.get_model_path() == ""
    assert not block2._reducer_path.text()
    assert not block2._cluster_path.text()
    assert not block2.is_reduction_or_cluster_enabled()


# ── Window size persistence ──────────────────────────────────────────────────


def test_inference_dr_methods_default_none(defaults_dir):
    """Fresh inference blocks start with no DR method selected."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from microProfiler.gui.panels.step_inference import InferenceBlockWidget

    block = InferenceBlockWidget(0, ["ch1"])
    assert block._method_cbs
    assert all(not cb.isChecked() for cb in block._method_cbs.values())
    # Nothing selected and no reducer -> the DR stage is not scheduled.
    assert block.is_reduction_or_cluster_enabled() is False


def test_inference_reduction_flags_derive_from_paths_and_methods(defaults_dir):
    """No group checkboxes: paths / checked methods are the run flags."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from microProfiler.gui.panels.step_inference import InferenceBlockWidget

    block = InferenceBlockWidget(0, ["ch1"])
    assert "reduction" not in block.build_config_section()

    # A checked DR method alone schedules the reduction.
    block._method_cbs["pca"].setChecked(True)
    red = block.build_config_section()["reduction"]
    assert red["enabled"] is True
    assert red["cluster_enabled"] is False
    assert red["method"] == ["pca"]
    assert red["cluster"] is None

    # A reducer file schedules it too and takes over from the methods.
    block._reducer_path.setText("D:/r/pca.pkl")
    red = block.build_config_section()["reduction"]
    assert red["enabled"] is True
    assert red["reducer"] == ["D:/r/pca.pkl"]
    assert red["method"] is None

    # Clearing everything skips the stage again.
    block._method_cbs["pca"].setChecked(False)
    block._reducer_path.clear()
    assert "reduction" not in block.build_config_section()

    # A cluster file alone schedules cluster prediction only.
    block._cluster_path.setText("D:/r/cluster.pkl")
    red = block.build_config_section()["reduction"]
    assert red["enabled"] is False
    assert red["cluster_enabled"] is True
    assert red["cluster"] == "D:/r/cluster.pkl"


def test_main_window_restores_and_saves_size(defaults_dir):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    update_user_defaults("window", {"width": 1360, "height": 860})

    from microProfiler.gui.main_window import MainWindow

    win = MainWindow()
    try:
        assert (win.width(), win.height()) == (1360, 860)
        win.resize(1420, 900)
        win._save_window_size()
    finally:
        win.close()
    app.processEvents()
    assert get_user_defaults()["window"] == {"width": 1420, "height": 900}


def test_main_window_falls_back_without_saved_size(defaults_dir):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from microProfiler.gui.main_window import MainWindow
    from microProfiler.gui.ui_spec import WINDOW_SIZE

    win = MainWindow()
    try:
        assert (win.width(), win.height()) == WINDOW_SIZE
    finally:
        win.close()
