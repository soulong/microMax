"""Tests for microBase.config: YAML load/save and SessionFile."""

import pytest

from microBase.config import (
    load_yaml,
    save_yaml,
    SessionFile,
    _deep_merge,
    normalize_null_strings,
)


def test_save_and_load_yaml(tmp_path):
    data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    path = tmp_path / "config.yml"
    save_yaml(path, data)
    out = load_yaml(path)
    assert out == data


def test_load_yaml_missing_exits(tmp_path):
    with pytest.raises(SystemExit):
        load_yaml(tmp_path / "nonexistent.yml")


def test_load_yaml_invalid_exits(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text("not: valid: yaml: [")
    with pytest.raises(SystemExit):
        load_yaml(path)


def test_load_yaml_empty_returns_empty_dict(tmp_path):
    path = tmp_path / "empty.yml"
    path.write_text("")
    assert load_yaml(path) == {}


def test_save_yaml_creates_parent_dirs(tmp_path):
    path = tmp_path / "subdir" / "nested" / "config.yml"
    save_yaml(path, {"k": "v"})
    assert path.exists()


def test_deep_merge_basic():
    base = {"a": 1, "b": 2}
    overlay = {"b": 3, "c": 4}
    assert _deep_merge(base, overlay) == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"x": {"y": 1, "z": 2}}
    overlay = {"x": {"y": 10, "w": 3}}
    assert _deep_merge(base, overlay) == {"x": {"y": 10, "z": 2, "w": 3}}


def test_deep_merge_overlay_wins_on_conflict():
    base = {"k": "old"}
    overlay = {"k": "new"}
    assert _deep_merge(base, overlay) == {"k": "new"}


def test_session_file_save_and_load(tmp_path):
    sf = SessionFile(tmp_path)
    sf.save({"image_pattern": "abc", "applied_steps": ["resize"]})
    out = sf.load()
    assert out["image_pattern"] == "abc"
    assert out["applied_steps"] == ["resize"]


def test_session_file_save_merges_not_overwrites(tmp_path):
    sf = SessionFile(tmp_path)
    sf.save({"image_pattern": "abc", "channel_colors": {"ch1": "green"}})
    sf.save({"mask_pattern": "def"})  # different key
    out = sf.load()
    # Both keys should be present (merge, not overwrite)
    assert out["image_pattern"] == "abc"
    assert out["mask_pattern"] == "def"
    assert out["channel_colors"] == {"ch1": "green"}


def test_session_file_load_missing_returns_empty(tmp_path):
    sf = SessionFile(tmp_path)
    assert sf.load() == {}


def test_session_file_get_set_patterns(tmp_path):
    sf = SessionFile(tmp_path)
    sf.set_patterns(
        image_pattern=r"r\d+c\d+",
        mask_pattern=r"_cp_masks\.png",
        image_subdir_pattern="Images/",
    )
    img, mask, sub = sf.get_patterns()
    assert img == r"r\d+c\d+"
    assert mask == r"_cp_masks\.png"
    assert sub == "Images/"


def test_session_file_get_set_applied_steps(tmp_path):
    sf = SessionFile(tmp_path)
    sf.set_applied_steps(["resize", "segment"])
    assert sf.get_applied_steps() == ["resize", "segment"]


def test_session_file_get_set_channel_colors(tmp_path):
    sf = SessionFile(tmp_path)
    colors = {"ch1": {"color": "green", "vmin": 0, "vmax": 65535}}
    sf.set_channel_colors(colors)
    out = sf.get_channel_colors()
    assert out == colors


def test_session_file_path_is_dataset_root_session_yml(tmp_path):
    sf = SessionFile(tmp_path)
    assert sf.path == tmp_path / "session.yml"


def test_session_file_exists(tmp_path):
    sf = SessionFile(tmp_path)
    assert sf.exists() is False
    sf.save({"k": "v"})
    assert sf.exists() is True


# ---- normalize_null_strings ----

def test_normalize_null_strings_passes_none_through():
    assert normalize_null_strings(None) is None


def test_normalize_null_strings_converts_case_variants():
    assert normalize_null_strings("null") is None
    assert normalize_null_strings("Null") is None
    assert normalize_null_strings("NULL") is None
    assert normalize_null_strings("none") is None
    assert normalize_null_strings("None") is None
    assert normalize_null_strings("NONE") is None
    assert normalize_null_strings("nOne") is None
    assert normalize_null_strings("NUll") is None


def test_normalize_null_strings_round_trip_via_load_yaml(tmp_path):
    """All common null spellings loaded through load_yaml become Python None."""
    path = tmp_path / "nulls.yml"
    path.write_text(
        "v1: null\n"
        "v2: Null\n"
        "v3: NULL\n"
        "v4: None\n"
        "v5: none\n"
        "v6: NONE\n"
        "v7: nOne\n"
        "v8: ~\n"
        "v9:\n"
    )
    data = load_yaml(path)
    for key in ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"):
        assert data[key] is None, f"{key} should be None, got {data[key]!r}"


def test_normalize_null_strings_recurses_into_dicts_and_lists():
    data = {
        "a": ["None", "none", "x", 1, {"b": "Null"}],
        "c": {"nested": "NULL", "keep": "value"},
    }
    out = normalize_null_strings(data)
    assert out == {
        "a": [None, None, "x", 1, {"b": None}],
        "c": {"nested": None, "keep": "value"},
    }


def test_normalize_null_strings_leaves_dict_keys_untouched():
    """A user may legitimately have a key spelled like 'None'; we don't touch keys."""
    out = normalize_null_strings({"None": "value", "null": "keep"})
    assert out == {"None": "value", "null": "keep"}


def test_normalize_null_strings_preserves_numbers_and_empty_string():
    assert normalize_null_strings(0) == 0
    assert normalize_null_strings(1.5) == 1.5
    assert normalize_null_strings(True) is True
    assert normalize_null_strings("") == ""
    assert normalize_null_strings("none!") == "none!"


def test_normalize_null_strings_quoted_null_in_yaml_becomes_none(tmp_path):
    """Even quoted 'null'/'None' strings normalize to None after load."""
    path = tmp_path / "quoted.yml"
    path.write_text('q1: "null"\nq2: "None"\nq3: "none"\n')
    data = load_yaml(path)
    assert data == {"q1": None, "q2": None, "q3": None}
