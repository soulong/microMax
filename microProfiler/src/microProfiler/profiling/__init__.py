"""Profiling: image-level and object-level feature extraction."""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence


def resolve_source_directory(
    row: Dict[str, Any], intensity_colnames: Sequence[str]
) -> str:
    """Derive the output 'directory' from the source file path.

    Returns the parent directory of the first available source file path
    (absolute, forward slashes), matching how microModel's infer.py computes
    the 'directory' column. Shared by the image and object profilers so their
    tables can be joined on 'directory'.
    """
    source_path = None
    if "__file__" in row and row["__file__"]:
        source_path = row["__file__"]
    else:
        for col in intensity_colnames:
            if col in row and row[col]:
                source_path = row[col]
                break
    if not source_path:
        return ""
    return os.path.dirname(str(source_path)).replace("\\", "/")