"""Regression tests for the BaSiC .fit_order marker.

The marker tells the GUI's Apply-downgrade which preprocessing order the
models were FIT under. It used to be written unconditionally as
"zproject_first", so models fitted before any z-projection advertised an
order they were not fit under — and the Apply path trusted the marker and
transformed z-projected images with models fitted on raw stacks.
"""

import re

import numpy as np
import pytest
from tifffile import imwrite

import microProfiler.preprocessing.basic_correction as bc
import microProfiler.pipeline.steps as steps
from microBase import ImageDataset, SessionFile
from microProfiler.config import BasicConfig, PipelineConfig

IMAGE_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
)


class _FakeModel:
    # fit_models writes flatfield (and darkfield when enabled) as TIFFs.
    flatfield = np.ones((8, 8), dtype=np.float32)


@pytest.fixture
def dataset(tmp_path):
    imwrite(str(tmp_path / "r01c01f01p01-ch1.tiff"),
            np.zeros((8, 8), dtype=np.uint16))
    return ImageDataset(root=tmp_path, image_pattern=IMAGE_PATTERN,
                        channel_layout=None)


def _marker_path(tmp_path):
    return tmp_path / ".microprofiler" / "BaSiC_model" / ".fit_order"


def test_fit_models_marks_zproject_first(tmp_path, dataset, monkeypatch):
    """A fit AFTER z-projection advertises the zproject-first order."""
    monkeypatch.setattr(bc, "basic_fit", lambda *a, **k: _FakeModel())
    bc.fit_models(dataset, root_dir=tmp_path, zproject_applied=True)
    assert _marker_path(tmp_path).read_text(encoding="utf-8") == "zproject_first"


def test_fit_models_marks_raw_fit_before_zproject(tmp_path, dataset, monkeypatch):
    """A fit on never-projected images must NOT claim zproject_first."""
    monkeypatch.setattr(bc, "basic_fit", lambda *a, **k: _FakeModel())
    bc.fit_models(dataset, root_dir=tmp_path, zproject_applied=False)
    assert _marker_path(tmp_path).read_text(encoding="utf-8") == "raw_fit"


def test_run_basic_passes_session_zproject_state(tmp_path, dataset, monkeypatch):
    """_run_basic derives zproject_applied from session.yml applied_steps,
    so the marker reflects reality whichever run path is used."""
    captured = {}

    def _fake_apply_basic(ds, **kwargs):
        captured.update(kwargs)
        return ds

    monkeypatch.setattr(bc, "apply_basic", _fake_apply_basic)
    cfg = PipelineConfig()
    cfg.basic = BasicConfig(run=True, mode="fit")

    SessionFile(tmp_path).set_applied_steps(["zproject"])
    steps._run_basic(cfg, dataset, tmp_path)
    assert captured["zproject_applied"] is True

    SessionFile(tmp_path).set_applied_steps([])
    steps._run_basic(cfg, dataset, tmp_path)
    assert captured["zproject_applied"] is False
