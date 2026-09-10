"""run_step contract tests: it returns (ds, applied_steps) and owns the
applied-steps gate, so GUI callers persist exactly what actually ran."""

import re

import numpy as np
from tifffile import imwrite

from microBase import ImageDataset, SessionFile
from microProfiler.config import (
    BasicConfig,
    FilterEntry,
    ImageProfileConfig,
    PipelineConfig,
    ResizeConfig,
    TileConfig,
)
from microProfiler.pipeline import run_step
from microProfiler.pipeline.steps import _step_will_execute
import microProfiler.pipeline as pipeline_pkg

IMAGE_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
)


def _make_dataset(tmp_path):
    imwrite(str(tmp_path / "r01c01f01p01-ch1.tiff"),
            np.zeros((8, 8), dtype=np.uint16))
    return ImageDataset(
        root=tmp_path, image_pattern=IMAGE_PATTERN, channel_layout=None,
    )


def test_run_step_fit_only_basic_not_recorded(tmp_path, monkeypatch):
    """A fit-only BaSiC run must never appear in applied_steps.

    The GUI's generic Apply handler used to save [step_name] unconditionally,
    permanently gating a later fit-transform run out.
    """
    ds = _make_dataset(tmp_path)
    cfg = PipelineConfig()
    cfg.basic = BasicConfig(run=True, mode="fit")

    # Stub the actual BaSiC fit: this test covers the persistence contract,
    # not the algorithm (which needs fitted models).
    monkeypatch.setitem(
        pipeline_pkg._STEP_FUNCTIONS, "basic",
        lambda cfg, ds, root_dir, progress: ds,
    )

    out_ds, applied = run_step(cfg, "basic", tmp_path, ds=ds)
    assert out_ds is ds
    assert applied == []
    assert SessionFile(tmp_path).get_applied_steps() == []


def test_run_step_noop_section_not_recorded(tmp_path):
    """An enabled section whose block list is empty executes nothing and
    must not be recorded in applied_steps."""
    ds = _make_dataset(tmp_path)
    cfg = PipelineConfig()
    cfg.image_profile = ImageProfileConfig(run=True, image_channels=[])

    _, applied = run_step(cfg, "image_profile", tmp_path, ds=ds)
    assert applied == []
    assert "image_profile" not in SessionFile(tmp_path).get_applied_steps()


def test_run_step_records_and_returns_union(tmp_path, monkeypatch):
    """A step that executes is recorded and returned as the full union."""
    ds = _make_dataset(tmp_path)
    SessionFile(tmp_path).set_applied_steps(["zproject"])
    cfg = PipelineConfig()
    cfg.resize = ResizeConfig(run=True, scale_factor=0.5)
    monkeypatch.setitem(
        pipeline_pkg._STEP_FUNCTIONS, "resize",
        lambda cfg, ds, root_dir, progress: ds,
    )

    _, applied = run_step(cfg, "resize", tmp_path, ds=ds)
    assert applied == ["resize", "zproject"]
    assert SessionFile(tmp_path).get_applied_steps() == ["resize", "zproject"]


def test_run_step_applies_filters(tmp_path, monkeypatch):
    """run_step must apply cfg.filter like run_pipeline does (it only
    received the filter panel's live-filtered dataset in the GUI)."""
    for field in (1, 2):
        imwrite(str(tmp_path / f"r01c01f{field:02d}p01-ch1.tiff"),
                np.zeros((8, 8), dtype=np.uint16))

    captured = {}

    def _fake_step(cfg, ds, root_dir, progress):
        captured["n_rows"] = len(ds)
        return ds

    monkeypatch.setitem(pipeline_pkg._STEP_FUNCTIONS, "resize", _fake_step)

    cfg = PipelineConfig()
    cfg.image_pattern = IMAGE_PATTERN.pattern
    cfg.resize = ResizeConfig(run=True, scale_factor=0.5)
    cfg.filter = [FilterEntry(column="field", pattern="1")]
    run_step(cfg, "resize", tmp_path, ds=None)
    assert captured["n_rows"] == 1


def test_resize_scale_1_is_not_recorded(tmp_path):
    """scale_factor == 1.0 is a no-op: recording it would gate a later real
    resize out of applied_steps forever."""
    ds = _make_dataset(tmp_path)
    cfg = PipelineConfig()
    cfg.resize = ResizeConfig(run=True, scale_factor=1.0)

    _, applied = run_step(cfg, "resize", tmp_path, ds=ds)
    assert applied == []
    assert SessionFile(tmp_path).get_applied_steps() == []


class _FakeShape:
    img_shape = (8, 8)


def test_tile_larger_than_image_is_not_recorded():
    cfg = PipelineConfig()
    cfg.tile = TileConfig(run=True, tile_width=16, tile_height=16)
    assert _step_will_execute(cfg, "tile", None, ds=_FakeShape()) is False

    cfg.tile = TileConfig(run=True, tile_width=8, tile_height=8)
    assert _step_will_execute(cfg, "tile", None, ds=_FakeShape()) is True
