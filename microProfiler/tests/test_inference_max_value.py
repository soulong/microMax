"""Inference panel max_value must follow the dataset dtype across switches."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from microProfiler.gui.panels.step_inference import (  # noqa: E402
    InferenceBlockWidget,
    InferenceStepPanel,
)


def _app():
    return QApplication.instance() or QApplication([])


def test_set_dataset_dtype_refreshes_non_explicit_blocks():
    """uint8 -> uint16 must update an adopted 255 back to 65535.

    Regression: the old check only fired while the value still equalled
    DEFAULT_MAX_VALUE, so a uint8 dataset left max_value=255 for every later
    dataset and inference silently divided by the wrong maximum.
    """
    _app()
    panel = InferenceStepPanel.__new__(InferenceStepPanel)
    panel._blocks = []
    panel._default_max_value = 65535.0

    adopted = InferenceBlockWidget(0, ["ch1"])
    panel._blocks.append(adopted)

    panel.set_dataset_dtype("uint8")
    assert adopted.get_max_value() == 255.0
    panel.set_dataset_dtype("uint16")
    assert adopted.get_max_value() == 65535.0

    explicit = InferenceBlockWidget(1, ["ch1"])
    explicit.set_max_value(3000.0, explicit=True)
    panel._blocks.append(explicit)
    panel.set_dataset_dtype("uint8")
    assert explicit.get_max_value() == 3000.0
