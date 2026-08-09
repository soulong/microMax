"""Schema helpers: well derivation and metadata column classification.

The row/col auto-merge rule: if both `row` and `col` are captured,
derive `well` and drop `row`/`col`.
"""

from dataclasses import dataclass

from natsort import natsorted


STRUCTURAL_COLS = {"well", "field", "stack", "timepoint", "channel", "row", "col"}
REGEX_META_COLS = {"ext", "tile", "mask_name"}


def _row_to_letter(row_val):
    """1 -> A, ..., 26 -> Z, 27 -> AA. Passes through alphabetic input."""
    if isinstance(row_val, str):
        if row_val.isalpha():
            return row_val.upper()
        try:
            row_val = int(row_val)
        except ValueError:
            return str(row_val)
    if row_val <= 0:
        return str(row_val)
    letters = ""
    n = int(row_val)
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def derive_well(row_val, col_val):
    """Build a well label like 'A1' from row+col values.

    Raises ValueError with a clear message if the column value is not
    numeric (metadata is TEXT, so e.g. '1.5' or 'A' reach this point).
    """
    try:
        col_int = int(col_val)
    except (TypeError, ValueError):
        raise ValueError(
            f"derive_well: column value {col_val!r} is not numeric — "
            f"row/col captures must be integers to derive a well label."
        ) from None
    return f"{_row_to_letter(row_val)}{col_int}"


def normalize_capture(_col_name, value):
    """Return a regex-captured value verbatim.

    Kept as a pass-through for API compatibility. All metadata is stored
    as TEXT from extraction through DB storage, so no value normalization
    is applied: '01' stays '01', 'blue' stays 'blue', '000a6c98-...' stays
    '000a6c98-...'. `derive_well` still computes well labels (e.g. 'A1')
    from row/col by doing its own int() internally.
    None is passed through.
    """
    return value


@dataclass(frozen=True)
class MetadataSchema:
    """Describes how regex-captured columns map to structural metadata."""

    structural_cols: dict
    extra_cols: list
    captured_fields: set
    derived_well: bool

    @classmethod
    def infer(cls, parsed_columns):
        """Partition captured regex groups into structural vs extra columns.

        If both `row` and `col` are present, derives `well` and marks them
        for removal from the saved metadata.
        """
        captured = set(parsed_columns)

        structural = {}
        extra = []
        derived_well = False

        # Auto-derive well from row+col if both present and well not already present
        if "row" in captured and "col" in captured and "well" not in captured:
            structural["well"] = "row_col"
            structural["row"] = "row"
            structural["col"] = "col"
            derived_well = True
            captured.discard("row")
            captured.discard("col")
        elif "well" in captured:
            structural["well"] = "well"

        for col in ("field", "stack", "timepoint", "channel"):
            if col in captured:
                structural[col] = col

        for col in natsorted(captured):
            if col not in STRUCTURAL_COLS and col not in REGEX_META_COLS:
                extra.append(col)

        return cls(
            structural_cols=structural,
            extra_cols=extra,
            captured_fields=captured,
            derived_well=derived_well,
        )

    def apply_well_merge(self, df):
        """If derived_well, build the well column from row+col and drop them."""
        if not self.derived_well:
            return df
        if "row" in df.columns and "col" in df.columns:
            df = df.copy()
            df["well"] = df.apply(
                lambda r: derive_well(r["row"], r["col"]), axis=1
            )
            df = df.drop(columns=["row", "col"])
        return df
