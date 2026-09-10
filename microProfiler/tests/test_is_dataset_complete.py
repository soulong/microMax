"""CLI completeness check: inference-only configs and reduction-table
expectations must match what the bridge actually asks microModel to write."""

import sqlite3

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


def test_missing_reduction_table_marks_incomplete(tmp_path):
    red = InferenceReductionConfig(enabled=True, method=["pacmap"])
    cfg = _inference_cfg(red)

    _make_infer_db(tmp_path / "infer.db", ["inference"])
    assert _is_dataset_complete(cfg, tmp_path) is False

    _make_infer_db(tmp_path / "infer.db", ["inference", "reduction_pacmap"])
    assert _is_dataset_complete(cfg, tmp_path) is True
