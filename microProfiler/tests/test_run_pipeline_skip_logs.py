"""run_pipeline skip-log context tests.

The applied-steps gate must only announce "Skipping X — already applied" for
steps this run actually requested: Run All / Run Preprocessing enable their
sections, while a single-panel run's restricted config leaves the other
sections disabled and must stay quiet.
"""

import logging
import re

import numpy as np
from tifffile import imwrite

from microBase import ImageDataset, SessionFile
from microProfiler.config import PipelineConfig, ResizeConfig
from microProfiler.pipeline import run_pipeline

IMAGE_PATTERN = re.compile(
    r"r(?P<row>\d+)c(?P<col>\d+)f(?P<field>\d+)p(?P<stack>\d+)-ch(?P<channel>\d+)\.tiff"
)


def _make_dataset(tmp_path):
    imwrite(str(tmp_path / "r01c01f01p01-ch1.tiff"),
            np.zeros((8, 8), dtype=np.uint16))
    return ImageDataset(
        root=tmp_path, image_pattern=IMAGE_PATTERN, channel_layout=None,
    )


def test_single_panel_run_does_not_log_preproc_skips(tmp_path, caplog):
    """A restricted config (all sections None) must not log applied-gate skips
    for preprocessing steps it never requested."""
    ds = _make_dataset(tmp_path)
    SessionFile(tmp_path).set_applied_steps(["resize", "basic"])

    with caplog.at_level(logging.INFO):
        run_pipeline(PipelineConfig(), root_dir=tmp_path, ds=ds)

    assert "Skipping resize" not in caplog.text
    assert "Skipping basic" not in caplog.text


def test_run_all_enabled_step_still_logs_skip(tmp_path, caplog):
    """When the run enables resize and it is already applied, the skip is
    still reported (Run All / Run Preprocessing behavior)."""
    ds = _make_dataset(tmp_path)
    SessionFile(tmp_path).set_applied_steps(["resize"])

    cfg = PipelineConfig()
    cfg.resize = ResizeConfig(run=True, scale_factor=0.5)

    with caplog.at_level(logging.INFO):
        run_pipeline(cfg, root_dir=tmp_path, ds=ds)

    assert "Skipping resize — already applied in previous run" in caplog.text


def test_run_pipeline_mirrors_terminal_log_into_dataset(tmp_path):
    """The terminal log is also written to <dataset>/microProfiler.log."""
    ds = _make_dataset(tmp_path)
    run_pipeline(PipelineConfig(), root_dir=tmp_path, ds=ds)

    log_path = tmp_path / "microProfiler.log"
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "Pipeline start" in text
