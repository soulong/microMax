"""Schema helpers: well derivation and metadata column classification.

The row/col auto-merge rule: if both `row` and `col` are captured,
derive `well` and drop `row`/`col`.

All metadata is stored as TEXT from extraction through DB storage: regex
captures are used verbatim (no leading-zero stripping, no int coercion —
'01' stays '01', 'blue' stays 'blue'). `derive_well` is the only exception:
it computes well labels (e.g. 'A1') from row/col by doing its own int()
internally.
"""

from dataclasses import dataclass

from natsort import natsorted

from microBase.errors import DatasetError


STRUCTURAL_COLS = {"well", "field", "stack", "timepoint", "channel", "row", "col"}
# Captured columns that are internal bookkeeping, not user-facing extra
# metadata: `ext`/`tile` come from optional pattern groups with no analytical
# meaning, and `mask_name`/`channel` are consumed into mask/intensity columns.
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

    row accepts integers, numeric strings, or pure alphabetic strings
    (e.g. 'A' — _row_to_letter handles both); mixed values like '1.5' or
    'A1' raise ValueError. col must be an integer (metadata is TEXT, so
    e.g. '1.5' or 'A' reach this point).
    """
    if isinstance(row_val, str):
        if not (row_val.isalpha() or row_val.isdigit()):
            raise ValueError(
                f"derive_well: row value {row_val!r} is not numeric or alphabetic — "
                f"row/col captures must be integers (or an alphabetic row) to "
                f"derive a well label."
            )
    elif not isinstance(row_val, int) or isinstance(row_val, bool):
        try:
            if int(row_val) != row_val:
                raise TypeError
        except (TypeError, ValueError):
            raise ValueError(
                f"derive_well: row value {row_val!r} is not numeric or alphabetic — "
                f"row/col captures must be integers (or an alphabetic row) to "
                f"derive a well label."
            ) from None
    try:
        col_int = int(col_val)
    except (TypeError, ValueError):
        raise ValueError(
            f"derive_well: column value {col_val!r} is not numeric — "
            f"row/col captures must be integers to derive a well label."
        ) from None
    return f"{_row_to_letter(row_val)}{col_int}"


def normalize_well(well_val):
    """Canonical well key: UPPERCASE row letters + strip leading zeros.

    A directly-captured `well` keeps its regex text verbatim ('A01' or even
    'a01' stays as captured), while derived wells and GUI/grid code generate
    uppercase 'A1' — comparing or joining the two forms therefore needs this
    normalization. Anything that is not a `<letters><digits>` well shape
    passes through unchanged.
    """
    text = str(well_val)
    i = 0
    while i < len(text) and text[i].isalpha():
        i += 1
    if i == 0 or i == len(text) or not text[i:].isdigit():
        return well_val
    return f"{text[:i].upper()}{int(text[i:])}"


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
        """If derived_well, build the well column from row+col and drop them.

        A non-numeric row/col capture (e.g. an optional group that matched
        nothing) is a dataset-layout mistake, so the ValueError is wrapped
        in DatasetError — build_metadata must only ever raise MicroMaxError
        subclasses.
        """
        if not self.derived_well:
            return df
        if "row" in df.columns and "col" in df.columns:
            df = df.copy()
            try:
                df["well"] = df.apply(
                    lambda r: derive_well(r["row"], r["col"]), axis=1
                )
            except (ValueError, TypeError) as e:
                raise DatasetError(
                    f"could not derive 'well' from row/col captures: {e}"
                ) from e
            df = df.drop(columns=["row", "col"])
        return df
