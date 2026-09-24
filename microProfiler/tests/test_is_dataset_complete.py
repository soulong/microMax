"""CLI completeness check: the NEW output style (per-model infer DBs with
per-mask tables + reduction columns inside them + merged.db) is what the
skip decision inspects, mirroring what the bridge actually writes."""

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

MODEL_NAME = "puncta"


def _make_db(path, tables, columns=("id INTEGER",)):
    conn = sqlite3.connect(str(path))
    for name, cols in tables.items():
        conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        conn.execute(f'CREATE TABLE "{name}" ({", ".join(cols)})')
    conn.commit()
    conn.close()


def _mask_cols(extra=()):
    return ["well TEXT", "label INTEGER", "directory TEXT", *extra]


def _inference_cfg(reduction=None):
    cfg = PipelineConfig()
    cfg.inference = InferenceConfig(
        run=True,
        configs=[InferenceEntry(
            model="model.pt", model_name=MODEL_NAME, mask_name="cell",
            channels=["ch1"], max_value=65535.0, reduction=reduction,
        )],
    )
    return cfg


def test_inference_only_dataset_can_be_skipped(tmp_path):
    """A config that enables ONLY inference must still be skippable once its
    per-model DB holds the mask table + merged.db exists."""
    cfg = _inference_cfg()
    assert _is_dataset_complete(cfg, tmp_path) is False

    _make_db(tmp_path / f"infer_{MODEL_NAME}.db",
             {"cell": _mask_cols()})
    # Still incomplete: merged.db (the run's final output) is missing.
    assert _is_dataset_complete(cfg, tmp_path) is False

    _make_db(tmp_path / "merged.db", {"cell": _mask_cols()})
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
    stage and the completeness check expects no reduction columns."""
    from pathlib import Path
    from types import SimpleNamespace

    from microProfiler.pipeline._micromodel_bridge import (
        SCRATCH_INFER_DB_NAME,
        _build_mm_inference_config,
    )

    red = InferenceReductionConfig(enabled=True, method=[])
    entry = InferenceEntry(
        model="m.pt", model_name=MODEL_NAME, mask_name="cell",
        channels=["ch1"], max_value=65535.0, reduction=red)
    cfg = PipelineConfig()
    ds = SimpleNamespace(
        intensity_colnames=["ch1"], image_pattern="p",
        mask_pattern="q", image_subdir_pattern="",
    )
    mm_cfg = _build_mm_inference_config(entry, cfg, ds, Path("."))
    assert "reduction" not in mm_cfg
    # microModel writes its RAW layout into the transient scratch file.
    assert mm_cfg["inference"]["db_name"] == SCRATCH_INFER_DB_NAME
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
        model="m.pt", model_name=MODEL_NAME, mask_name="cell",
        channels=["ch1"], max_value=65535.0, reduction=red)
    cfg = PipelineConfig()
    ds = SimpleNamespace(
        intensity_colnames=["ch1"], image_pattern="p",
        mask_pattern="q", image_subdir_pattern="",
    )
    mm_cfg = _build_mm_inference_config(entry, cfg, ds, Path("."))
    assert mm_cfg["reduction"]["method"] == ["pca"]
    assert expected_reduction_tables(entry) == frozenset(
        {"reduction_pca", "find_cluster"})


def test_missing_reduction_columns_marks_incomplete(tmp_path):
    """Reduction results live as COLUMNS of the mask table — a pacmap run
    whose table lacks `localmap_1` is reprocessed."""
    red = InferenceReductionConfig(enabled=True, method=["pacmap"])
    cfg = _inference_cfg(red)

    _make_db(tmp_path / f"infer_{MODEL_NAME}.db", {"cell": _mask_cols()})
    assert _is_dataset_complete(cfg, tmp_path) is False

    _make_db(tmp_path / f"infer_{MODEL_NAME}.db", {"cell": _mask_cols(
        ("pacmap_1 REAL", "pacmap_2 REAL"))})
    _make_db(tmp_path / "merged.db", {"cell": _mask_cols()})
    assert _is_dataset_complete(cfg, tmp_path) is True


def test_cluster_columns_checked_in_mask_table(tmp_path):
    """cluster_enabled -> the mask table must carry a cluster_res column."""
    red = InferenceReductionConfig(
        enabled=False, cluster_enabled=True, cluster="cluster.pkl")
    cfg = _inference_cfg(red)

    _make_db(tmp_path / f"infer_{MODEL_NAME}.db", {"cell": _mask_cols(
        ("pc_1 REAL", "pc_2 REAL"))})
    assert _is_dataset_complete(cfg, tmp_path) is False

    _make_db(tmp_path / f"infer_{MODEL_NAME}.db", {"cell": _mask_cols(
        ("pc_1 REAL", "pc_2 REAL", "cluster_res_1 INTEGER"))})
    _make_db(tmp_path / "merged.db", {"cell": _mask_cols()})
    assert _is_dataset_complete(cfg, tmp_path) is True


def test_config_yaml_preserves_empty_method_list():
    """`method: []` in YAML must survive as [] (fit nothing). The section
    coercion used to normalize it to None, silently re-enabling the pca+umap
    default."""
    from microProfiler.config import section_to_dataclass

    section = {
        "run": True,
        "configs": [{
            "model": "model.pt", "model_name": "puncta",
            "mask_name": "cell", "channels": ["ch1"],
            "max_value": 65535.0,
            "reduction": {"enabled": True, "method": []},
        }],
    }
    cfg = section_to_dataclass("inference", section)
    red = cfg.configs[0].reduction
    assert red.method == []
    assert expected_reduction_tables(cfg.configs[0]) == frozenset()


def test_cluster_res_with_group_off_needs_no_find_cluster(tmp_path):
    """cluster_enabled gates the WHOLE cluster group: a leftover template
    cluster_res must not make the run do fresh Leiden, nor the completeness
    check demand cluster columns (a fully processed dataset was
    reprocessed forever because of it)."""
    from pathlib import Path
    from types import SimpleNamespace

    from microProfiler.pipeline._micromodel_bridge import (
        _build_mm_inference_config,
    )

    red = InferenceReductionConfig(
        enabled=True, cluster_enabled=False, cluster=None,
        cluster_res=[0.5, 1], method=["pca", "umap", "localmap"],
    )
    entry = InferenceEntry(
        model="m.pt", model_name=MODEL_NAME, mask_name="cell",
        channels=["ch1"], max_value=65535.0, reduction=red)
    cfg = _inference_cfg(red)

    # Expected tables: the three DR tables, NO find_cluster.
    assert expected_reduction_tables(entry) == frozenset(
        {"reduction_pca", "reduction_umap", "reduction_localmap"})

    # The bridge passes cluster_res as null — microModel never clusters.
    ds = SimpleNamespace(
        intensity_colnames=["ch1"], image_pattern="p",
        mask_pattern="q", image_subdir_pattern="",
    )
    mm_cfg = _build_mm_inference_config(entry, cfg, ds, Path(tmp_path))
    assert mm_cfg["reduction"]["cluster_res"] is None

    # The user's scenario: the mask table HAS every reduction column the
    # run writes (cluster_res absent) — the dataset must count as complete.
    _make_db(tmp_path / f"infer_{MODEL_NAME}.db", {"cell": _mask_cols(
        ("pc_1 REAL", "umap_1 REAL", "localmap_1 REAL"))})
    _make_db(tmp_path / "merged.db", {"cell": _mask_cols()})
    assert _is_dataset_complete(cfg, tmp_path) is True


def test_cluster_res_with_group_on_writes_find_cluster():
    """Fresh Leiden is a sub-option of the cluster group: cluster_enabled +
    cluster_res (no baseline) runs it — and the check mirrors that."""
    red = InferenceReductionConfig(
        enabled=True, cluster_enabled=True, cluster=None,
        cluster_res=[0.5, 1], method=["pca"],
    )
    entry = InferenceEntry(model="m.pt", channels=["ch1"], reduction=red)

    assert expected_reduction_tables(entry) == frozenset(
        {"reduction_pca", "find_cluster"})


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


def test_object_profile_requires_merged_db(tmp_path):
    """A dataset with profiler tables but no merged.db (old-style results)
    is reprocessed so the end-of-run rebuild recreates the fused output."""
    from microProfiler.config import ObjectProfileConfig, ObjectProfileEntry

    cfg = PipelineConfig(object_profile=ObjectProfileConfig(
        run=True,
        configs=[ObjectProfileEntry(
            mask_name="cell", intensity_channels=["ch1"])]))

    _make_db(tmp_path / "profiler.db", {"cell": _mask_cols()})
    assert _is_dataset_complete(cfg, tmp_path) is False

    _make_db(tmp_path / "merged.db", {"cell": _mask_cols()})
    assert _is_dataset_complete(cfg, tmp_path) is True


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
