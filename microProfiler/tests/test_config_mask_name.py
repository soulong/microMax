"""Mask names in configs are bare; the 'mask_' prefix is internal."""

import pytest

from microProfiler.config import section_to_dataclass


def test_prefixed_mask_names_are_rejected():
    with pytest.raises(ValueError, match="bare"):
        section_to_dataclass("object_profile", {
            "run": True,
            "configs": [{"mask_name": "mask_cell", "intensity_channels": ["ch1"]}],
        })
    with pytest.raises(ValueError, match="bare"):
        section_to_dataclass("object_profile", {
            "run": True,
            "configs": [{"mask_name": "nucleus", "parent_mask_name": "mask_cell",
                         "intensity_channels": ["ch1"]}],
        })
    with pytest.raises(ValueError, match="bare"):
        section_to_dataclass("segment", {
            "run": True,
            "configs": [{"object_name": "mask_cell", "chan1": ["ch1"]}],
        })


def test_prefixed_inference_mask_name_is_rejected():
    with pytest.raises(ValueError, match="bare"):
        section_to_dataclass("inference", {
            "run": True,
            "configs": [{"model": "model.pt", "mask_name": "mask_cell",
                         "channels": ["ch1"]}],
        })


def test_bare_mask_names_are_accepted():
    cfg = section_to_dataclass("object_profile", {
        "run": True,
        "configs": [{"mask_name": "cell", "intensity_channels": ["ch1"]}],
    })
    assert cfg.configs[0].mask_name == "cell"
