"""Tests for microBase.patterns default regex patterns."""
import re

from microBase import (
    DEFAULT_IMAGE_PATTERN,
    DEFAULT_MASK_PATTERN,
    DEFAULT_IMAGE_SUBDIR_PATTERN,
)


def test_constants_are_nonempty_strings():
    assert isinstance(DEFAULT_IMAGE_PATTERN, str) and DEFAULT_IMAGE_PATTERN
    assert isinstance(DEFAULT_MASK_PATTERN, str) and DEFAULT_MASK_PATTERN
    assert isinstance(DEFAULT_IMAGE_SUBDIR_PATTERN, str) and DEFAULT_IMAGE_SUBDIR_PATTERN


def test_image_pattern_compiles_with_expected_groups():
    pat = re.compile(DEFAULT_IMAGE_PATTERN)
    for grp in ("row", "col", "field", "stack", "channel", "timepoint"):
        assert grp in pat.groupindex, f"missing group: {grp}"


def test_mask_pattern_compiles_with_expected_groups():
    pat = re.compile(DEFAULT_MASK_PATTERN)
    for grp in ("row", "col", "field", "stack", "channel", "timepoint", "mask_name"):
        assert grp in pat.groupindex, f"missing group: {grp}"


def test_image_pattern_matches_without_leading_zeros():
    pat = re.compile(DEFAULT_IMAGE_PATTERN)
    m = pat.match("r1c1f1p1-ch1sk1-ae66af39-4df5-4339-8491-c46e95b7a949.tiff")
    assert m is not None
    assert m.group("row") == "1"
    assert m.group("col") == "1"
    assert m.group("field") == "1"
    assert m.group("stack") == "1"
    assert m.group("channel") == "1"
    assert m.group("timepoint") == "1"


def test_image_pattern_matches_with_leading_zeros():
    pat = re.compile(DEFAULT_IMAGE_PATTERN)
    m = pat.match("r01c01f01p01-ch01sk01.tiff")
    assert m is not None
    # row/col have no 0? prefix → capture all digits verbatim
    assert m.group("row") == "01"
    assert m.group("col") == "01"
    # field/stack/channel/timepoint have 0? prefix → leading zero is consumed
    # by the optional 0?, so the capture group gets the remaining digits
    assert m.group("field") == "1"
    assert m.group("stack") == "1"
    assert m.group("channel") == "1"
    assert m.group("timepoint") == "1"


def test_image_pattern_matches_with_t_prefix():
    pat = re.compile(DEFAULT_IMAGE_PATTERN)
    m = pat.match("r1c1f1p1-ch1t1.tiff")
    assert m is not None
    assert m.group("timepoint") == "1"


def test_mask_pattern_matches():
    pat = re.compile(DEFAULT_MASK_PATTERN)
    m = pat.match("r1c1f1p1-ch1sk1-ae66af39-4df5-4339-8491-c46e95b7a949_cp_masks_cell.png")
    assert m is not None
    assert m.group("mask_name") == "cell"


def test_mask_pattern_matches_with_leading_zeros():
    pat = re.compile(DEFAULT_MASK_PATTERN)
    m = pat.match("r01c01f01p01-ch01sk01_cp_masks_nuclei.png")
    assert m is not None
    assert m.group("mask_name") == "nuclei"
