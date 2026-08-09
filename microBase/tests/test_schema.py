"""Tests for microBase.schema: well derivation and MetadataSchema."""

import pytest

from microBase.schema import (
    derive_well,
    _row_to_letter,
    MetadataSchema,
    normalize_capture,
)


def test_row_to_letter_basic():
    assert _row_to_letter(1) == "A"
    assert _row_to_letter(2) == "B"
    assert _row_to_letter(26) == "Z"
    assert _row_to_letter(27) == "AA"
    assert _row_to_letter(28) == "AB"


def test_row_to_letter_passes_through_alpha():
    assert _row_to_letter("A") == "A"
    assert _row_to_letter("a") == "A"
    assert _row_to_letter("P") == "P"


def test_row_to_letter_string_numeric():
    assert _row_to_letter("1") == "A"
    assert _row_to_letter("26") == "Z"


def test_derive_well_basic():
    assert derive_well(1, 1) == "A1"
    assert derive_well(2, 3) == "B3"
    assert derive_well(26, 12) == "Z12"
    assert derive_well(27, 1) == "AA1"


def test_derive_well_non_numeric_col_raises():
    """Non-numeric column values must raise a clear ValueError (metadata is TEXT)."""
    with pytest.raises(ValueError, match="not numeric"):
        derive_well("A", "1.5")
    with pytest.raises(ValueError, match="not numeric"):
        derive_well("A", "A")


# ---- normalize_capture ----
# normalize_capture is a verbatim pass-through: all metadata is stored as TEXT
# from extraction through DB storage, so '01' stays '01', 'blue' stays 'blue'.
# derive_well still computes well labels (e.g. 'A1') from row+col by doing its
# own int() internally.

def test_normalize_capture_channel_passes_through_verbatim():
    """channel captures are kept verbatim — '01' stays '01' (TEXT storage)."""
    assert normalize_capture("channel", "01") == "01"
    assert normalize_capture("channel", "00002") == "00002"
    assert normalize_capture("channel", "1") == "1"
    assert normalize_capture("channel", "12") == "12"
    # String channel names (e.g. HPA '_blue', '_green') are preserved as-is
    assert normalize_capture("channel", "blue") == "blue"
    assert normalize_capture("channel", "green") == "green"


def test_normalize_capture_structural_cols_pass_through_verbatim():
    """field/stack/timepoint/row/col/label are kept verbatim (TEXT storage)."""
    for col in ("field", "stack", "timepoint", "row", "col", "label"):
        assert normalize_capture(col, "01") == "01"
        assert normalize_capture(col, "00001") == "00001"
        assert normalize_capture(col, "1") == "1"
        assert normalize_capture(col, "12") == "12"


def test_normalize_capture_well_passes_through_verbatim():
    """A captured well 'A01' stays 'A01' (TEXT storage). derive_well handles
    the row+col -> 'A1' normalization separately via its own int()."""
    assert normalize_capture("well", "A01") == "A01"
    assert normalize_capture("well", "A1") == "A1"
    assert normalize_capture("well", "B12") == "B12"


def test_normalize_capture_extra_cols_pass_through_verbatim():
    """Extra cols are kept verbatim — no leading-zero stripping, no coercion."""
    assert normalize_capture("plate_id", "001") == "001"
    assert normalize_capture("cycle", "01") == "01"
    assert normalize_capture("plate_id", "PlateA") == "PlateA"
    assert normalize_capture("custom_group", "A01") == "A01"
    assert normalize_capture("custom_group", "Field02_Cell05") == "Field02_Cell05"


def test_normalize_capture_passes_none_and_non_string_through():
    assert normalize_capture("field", None) is None
    assert normalize_capture("field", 1) == 1
    assert normalize_capture("channel", 5) == 5


def test_normalize_capture_passes_non_numeric_through_verbatim():
    """Non-numeric input for any column is returned verbatim."""
    assert normalize_capture("field", "abc") == "abc"
    assert normalize_capture("field", "00abc") == "00abc"
    assert normalize_capture("channel", "0abc") == "0abc"


def test_metadata_schema_no_well():
    """No row/col -> no well derivation."""
    schema = MetadataSchema.infer({"field", "timepoint"})
    assert schema.derived_well is False
    assert "well" not in schema.structural_cols
    assert "field" in schema.structural_cols
    assert "timepoint" in schema.structural_cols
    assert schema.extra_cols == []


def test_metadata_schema_with_row_col():
    """Both row and col -> derive well, mark for removal."""
    schema = MetadataSchema.infer({"row", "col", "field", "timepoint"})
    assert schema.derived_well is True
    assert schema.structural_cols["well"] == "row_col"
    assert schema.structural_cols["row"] == "row"
    assert schema.structural_cols["col"] == "col"
    assert "row" not in schema.captured_fields  # discarded
    assert "col" not in schema.captured_fields
    assert "field" in schema.structural_cols


def test_metadata_schema_with_explicit_well():
    """If well is already in captures, don't derive from row/col."""
    schema = MetadataSchema.infer({"well", "row", "col"})
    assert schema.derived_well is False
    assert schema.structural_cols["well"] == "well"


def test_metadata_schema_extra_cols():
    """Non-structural captures go into extra_cols."""
    schema = MetadataSchema.infer({"field", "plate_id", "cycle"})
    assert schema.extra_cols == ["cycle", "plate_id"]  # sorted


def test_apply_well_merge_with_row_col():
    import pandas as pd
    schema = MetadataSchema.infer({"row", "col", "field"})
    df = pd.DataFrame({
        "row": [1, 2, 3],
        "col": [1, 2, 3],
        "field": [1, 1, 1],
    })
    merged = schema.apply_well_merge(df)
    assert "well" in merged.columns
    assert "row" not in merged.columns
    assert "col" not in merged.columns
    assert merged["well"].tolist() == ["A1", "B2", "C3"]


def test_apply_well_merge_no_op_when_not_derived():
    import pandas as pd
    schema = MetadataSchema.infer({"field", "timepoint"})
    df = pd.DataFrame({"field": [1, 2], "timepoint": [0, 0]})
    merged = schema.apply_well_merge(df)
    # Unchanged
    assert "well" not in merged.columns
    assert merged["field"].tolist() == [1, 2]
