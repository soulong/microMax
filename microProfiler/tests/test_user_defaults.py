"""~/.micromax user defaults: persistence + inference panel pre-fill."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import microProfiler.user_defaults as ud  # noqa: E402
from microProfiler.user_defaults import (  # noqa: E402
    get_user_defaults,
    update_user_defaults,
)


@pytest.fixture
def defaults_dir(tmp_path, monkeypatch):
    """Point ~/.micromax at a temp dir for the whole test."""
    import microProfiler.user_defaults as ud
    d = tmp_path / "micromax"
    monkeypatch.setattr(ud, "DEFAULTS_DIR", d)
    monkeypatch.setattr(ud, "DEFAULTS_FILE", d / "microprofiler.yml")
    return d


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
    assert block._reduction_group.isChecked()
    assert block._reducer_path.text() == "D:/r/pca.pkl;D:/r/umap.pkl"
    assert block._cluster_group.isChecked()
    assert block._cluster_path.text() == "D:/r/cluster.pkl"

    # No defaults (fresh machine) -> untouched blank panel.
    defaults_dir.joinpath("microprofiler.yml").unlink()
    panel2 = InferenceStepPanel.__new__(InferenceStepPanel)
    panel2._blocks = []
    block2 = InferenceBlockWidget(0, [])
    panel2._blocks.append(block2)
    panel2._apply_user_defaults()
    assert block2.get_model_path() == ""
    assert not block2._reduction_group.isChecked()
    assert not block2._cluster_group.isChecked()
