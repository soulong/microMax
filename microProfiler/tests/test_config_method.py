"""Reduction `method` parsing: scalar wrap, []-fits-nothing, zproject typo.

The reduction method list is the ONE field whose explicit empty list is
meaningful (fit nothing); every other empty list falls back to the
dataclass default.
"""

from microProfiler.config import section_to_dataclass


def _inference_section(reduction):
    return {
        "run": True,
        "configs": [{
            "model": "model.pt",
            "channels": ["ch1"],
            "max_value": 65535,
            "feature": True,
            "reduction": reduction,
        }],
    }


def test_scalar_method_string_is_wrapped():
    """A hand-written `method: pca` must parse as a one-entry list, not
    explode into per-character entries at validation time."""
    cfg = section_to_dataclass("inference", _inference_section(
        {"enabled": True, "reducer": ["pca"], "method": "pacmap"}))
    assert cfg.configs[0].reduction.method == ["pacmap"]


def test_empty_method_list_survives():
    """Explicit [] = fit nothing — it must not be conflated with null."""
    cfg = section_to_dataclass("inference", _inference_section(
        {"enabled": True, "method": []}))
    assert cfg.configs[0].reduction.method == []


def test_zproject_empty_method_falls_back_to_default():
    """zproject.method is a scalar enum: a [] there is a typo and falls back
    to the declared default instead of leaking into runtime."""
    cfg = section_to_dataclass("zproject", {"run": True, "method": []})
    assert cfg.method.value == "max"


def test_unknown_reduction_method_still_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown"):
        section_to_dataclass("inference", _inference_section(
            {"enabled": True, "method": ["nope"]}))
