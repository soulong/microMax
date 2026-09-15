"""CLI completeness check: inference-only configs and reduction-table
expectations must match what the bridge actually asks microModel to write."""

import sqlite3

import pytest

from microProfiler.cli import _is_dataset_complete
from microProfiler.config import (
    InferenceConfig,
    InferenceEntry,
    InferenceReductionConfig,
    PipelineConfig,
)
from microProfiler.pipeline._micromodel_bridge import expected_reduction_tables


def _make_infer_db(path, tables):
    conn = sqlite3.connect(str(path))
    for name in tables:
        conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        conn.execute(f'CREATE TABLE "{name}" (uid INTEGER)')
    conn.commit()
    conn.close()


def _inference_cfg(reduction=None):
    cfg = PipelineConfig()
    cfg.inference = InferenceConfig(
        run=True,
        configs=[InferenceEntry(
            model="model.pt", mask_name="cell", channels=["ch1"],
            output_db="infer.db", reduction=reduction,
        )],
    )
    return cfg


def test_inference_only_dataset_can_be_skipped(tmp_path):
    """A config that enables ONLY inference must still be skippable once its
    infer DB exists (previously an empty profiler-table set returned False
    before the inference check)."""
    cfg = _inference_cfg()
    assert _is_dataset_complete(cfg, tmp_path) is False

    _make_infer_db(tmp_path / "infer.db", ["inference"])
    assert _is_dataset_complete(cfg, tmp_path) is True


def test_expected_reduction_tables_mirrors_bridge_method_choice():
    # Cluster + explicit non-PCA method: the bridge writes reduction_umap,
    # so the completeness check must demand exactly that (not hardcoded pca).
    red = InferenceReductionConfig(
        enabled=True, cluster_enabled=True, cluster="cluster.pkl",
        method=["umap"],
    )
    entry = InferenceEntry(model="m.pt", channels=["ch1"], reduction=red)
    assert expected_reduction_tables(entry) == frozenset(
        {"reduction_umap", "find_cluster"})

    # Cluster-only: cheap default [pca] (matches the bridge).
    red = InferenceReductionConfig(
        enabled=False, cluster_enabled=True, cluster="cluster.pkl")
    entry = InferenceEntry(model="m.pt", channels=["ch1"], reduction=red)
    assert expected_reduction_tables(entry) == frozenset(
        {"reduction_pca", "find_cluster"})

    # Reduction group, no explicit method -> default [pca, umap].
    red = InferenceReductionConfig(enabled=True)
    entry = InferenceEntry(model="m.pt", channels=["ch1"], reduction=red)
    assert expected_reduction_tables(entry) == frozenset(
        {"reduction_pca", "reduction_umap"})


def test_empty_method_fits_nothing():
    """Explicit [] = no DR method at all: the bridge schedules no reduction
    stage and the completeness check expects no reduction table."""
    from pathlib import Path
    from types import SimpleNamespace

    from microProfiler.pipeline._micromodel_bridge import (
        _build_mm_inference_config,
    )

    red = InferenceReductionConfig(enabled=True, method=[])
    entry = InferenceEntry(
        model="m.pt", mask_name="cell", channels=["ch1"],
        max_value=65535.0, reduction=red)
    cfg = PipelineConfig()
    ds = SimpleNamespace(
        intensity_colnames=["ch1"], image_pattern="p",
        mask_pattern="q", image_subdir_pattern="",
    )
    mm_cfg = _build_mm_inference_config(entry, cfg, ds, Path("."))
    assert "reduction" not in mm_cfg
    assert expected_reduction_tables(entry) == frozenset()


def test_empty_method_with_cluster_still_uses_pca_reference():
    """A checked Cluster group needs the PCA reference embedding even when
    no DR method is selected."""
    from pathlib import Path
    from types import SimpleNamespace

    from microProfiler.pipeline._micromodel_bridge import (
        _build_mm_inference_config,
    )

    red = InferenceReductionConfig(
        enabled=True, cluster_enabled=True, cluster="cluster.pkl", method=[])
    entry = InferenceEntry(
        model="m.pt", mask_name="cell", channels=["ch1"],
        max_value=65535.0, reduction=red)
    cfg = PipelineConfig()
    ds = SimpleNamespace(
        intensity_colnames=["ch1"], image_pattern="p",
        mask_pattern="q", image_subdir_pattern="",
    )
    mm_cfg = _build_mm_inference_config(entry, cfg, ds, Path("."))
    assert mm_cfg["reduction"]["method"] == ["pca"]
    assert expected_reduction_tables(entry) == frozenset(
        {"reduction_pca", "find_cluster"})


def test_missing_reduction_table_marks_incomplete(tmp_path):
    red = InferenceReductionConfig(enabled=True, method=["pacmap"])
    cfg = _inference_cfg(red)

    _make_infer_db(tmp_path / "infer.db", ["inference"])
    assert _is_dataset_complete(cfg, tmp_path) is False

    _make_infer_db(tmp_path / "infer.db", ["inference", "reduction_pacmap"])
    assert _is_dataset_complete(cfg, tmp_path) is True


def test_config_yaml_preserves_empty_method_list():
    """`method: []` in YAML must survive as [] (fit nothing). The section
    coercion used to normalize it to None, silently re-enabling the pca+umap
    default."""
    from microProfiler.config import section_to_dataclass

    section = {
        "run": True,
        "configs": [{
            "model": "model.pt", "mask_name": "cell", "channels": ["ch1"],
            "max_value": 65535.0,
            "reduction": {"enabled": True, "method": []},
        }],
    }
    cfg = section_to_dataclass("inference", section)
    red = cfg.configs[0].reduction
    assert red.method == []
    assert expected_reduction_tables(cfg.configs[0]) == frozenset()


def test_pure_preprocessing_config_is_never_complete(tmp_path):
    """A config with no profiling/inference outputs has no notion of
    'complete' — the dataset must always run (the applied_steps gate alone
    protects the destructive steps)."""
    from microProfiler.config import ResizeConfig, SegmentConfig, SegmentEntry

    resize_only = PipelineConfig(resize=ResizeConfig(run=True, scale_factor=0.5))
    assert _is_dataset_complete(resize_only, tmp_path) is False

    segment_only = PipelineConfig(segment=SegmentConfig(
        run=True, configs=[SegmentEntry(chan1=["ch1"], object_name="cell")]))
    assert _is_dataset_complete(segment_only, tmp_path) is False


def test_unknown_top_level_key_raises(tmp_path):
    """A typo'd top-level section (`zproejct:`) must raise instead of being
    silently dropped."""
    import yaml

    from microProfiler.config import load_config

    cfg_file = tmp_path / "cfg.yml"
    cfg_file.write_text(
        "image_pattern: 'img_(?P<channel>\\d+)\\.tiff'\n"
        "zproejct:\n"
        "  run: true\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown top-level config keys"):
        load_config(cfg_file)
