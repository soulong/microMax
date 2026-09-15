"""Tests for microBase.schema: well derivation and MetadataSchema."""

import pytest

from microBase.schema import (
    derive_well,
    normalize_well,
    _row_to_letter,
    MetadataSchema,
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


def test_derive_well_non_numeric_row_raises():
    """Mixed row values ('1.5', 'A1') must raise instead of producing garbage wells."""
    with pytest.raises(ValueError, match="row value"):
        derive_well("1.5", 1)
    with pytest.raises(ValueError, match="row value"):
        derive_well("A1", 1)
    with pytest.raises(ValueError, match="row value"):
        derive_well(1.5, 1)


def test_derive_well_alpha_row_ok():
    """Alphabetic rows pass through _row_to_letter ('A' -> 'A1')."""
    assert derive_well("A", 1) == "A1"
    assert derive_well("a", 1) == "A1"
    assert derive_well("P", 12) == "P12"


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


def test_normalize_well_strips_leading_zeros():
    """Captured 'A01' style wells normalize to the canonical 'A1' grid key."""
    assert normalize_well("A01") == "A1"
    assert normalize_well("A1") == "A1"
    assert normalize_well("P012") == "P12"
    assert normalize_well("AA03") == "AA3"


def test_normalize_well_uppercases_row_letters():
    """Lowercase captured wells meet their uppercase grid counterparts."""
    assert normalize_well("a01") == "A1"
    assert normalize_well("p12") == "P12"


def test_normalize_well_passthrough():
    """Anything that is not a <letters><digits> well passes through as-is."""
    assert normalize_well("A") == "A"
    assert normalize_well("01") == "01"
    assert normalize_well("weird") == "weird"
    assert normalize_well("A1b") == "A1b"
    assert normalize_well(None) is None
