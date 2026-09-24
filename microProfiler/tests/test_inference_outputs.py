"""The new inference output style: per-model DBs (one table per mask) and
the config validation that keeps them consistent."""

import sqlite3

import pytest

from microProfiler.config import infer_db_name, section_to_dataclass


def _block(channels, model_name="puncta", mask_name="cell"):
    return {
        "model": "model.pt", "model_name": model_name, "mask_name": mask_name,
        "max_value": 65535, "channels": channels,
    }


def test_infer_db_name_from_model_name():
    from microProfiler.config import InferenceEntry

    entry = InferenceEntry(model="m.pt", model_name="puncta")
    assert infer_db_name(entry) == "infer_puncta.db"


def test_config_model_name_replaces_output_db():
    """`model_name` names the per-model DB; `output_db` no longer exists."""
    cfg = section_to_dataclass(
        "inference", {"run": True, "configs": [_block(["ch1"])]})
    entry = cfg.configs[0]
    assert entry.model_name == "puncta"
    assert infer_db_name(entry) == "infer_puncta.db"
    assert not hasattr(entry, "output_db")


def test_config_strips_trailing_db_suffix():
    """A user-typed `puncta.db` becomes the plain model name `puncta`."""
    cfg = section_to_dataclass(
        "inference",
        {"run": True, "configs": [_block(["ch1"], model_name="puncta.db")]})
    assert cfg.configs[0].model_name == "puncta"


def test_config_rejects_bad_model_name():
    with pytest.raises(ValueError, match="must not be empty"):
        section_to_dataclass(
            "inference", {"run": True, "configs": [_block(["ch1"],
                                                          model_name="")]})
    with pytest.raises(ValueError, match="path separators"):
        section_to_dataclass(
            "inference",
            {"run": True, "configs": [_block(["ch1"], model_name="a/b")]})


def test_config_requires_mask_name():
    """The mask name IS the output table name — an active block without it
    has no defined output."""
    block = _block(["ch1"])
    block["mask_name"] = None
    with pytest.raises(ValueError, match="mask_name' must not be empty"):
        section_to_dataclass("inference", {"run": True, "configs": [block]})


def test_config_rejects_duplicate_model_mask_pair():
    """Two blocks over the SAME (model_name, mask_name) would overwrite each
    other's rows in the same per-mask table — config validation rejects them
    up front (GUI and CLI share this one validation source)."""
    section = {
        "run": True,
        "configs": [_block(["ch1"]), _block(["ch2"])],
    }
    with pytest.raises(ValueError, match="Duplicate inference"):
        section_to_dataclass("inference", section)

    # Same model over a DIFFERENT mask is fine (one table per mask).
    section["configs"][1] = _block(["ch2"], mask_name="nuclei")
    parsed = section_to_dataclass("inference", section)
    assert [e.mask_name for e in parsed.configs] == ["cell", "nuclei"]
    assert [infer_db_name(e) for e in parsed.configs] == \
        ["infer_puncta.db", "infer_puncta.db"]

    # Same mask under a DIFFERENT model is fine too (one DB per model).
    section["configs"][1] = _block(["ch2"], model_name="mito")
    parsed = section_to_dataclass("inference", section)
    assert [infer_db_name(e) for e in parsed.configs] == \
        ["infer_puncta.db", "infer_mito.db"]


def test_names_required_even_for_skipped_blocks():
    """Like `model`, model_name/mask_name are validated for EVERY entry —
    a skipped (no-channels) block still must be a complete, runnable
    config once channels are filled in."""
    block = _block([], model_name="", mask_name=None)
    with pytest.raises(ValueError, match="model_name' must not be empty"):
        section_to_dataclass("inference", {"run": True, "configs": [block]})
