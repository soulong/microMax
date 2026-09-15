"""Profiling: image-level and object-level feature extraction."""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from microBase.db_contracts import canonical_directory


def resolve_source_directory(
    row: Dict[str, Any], intensity_colnames: Sequence[str], root
) -> str:
    """Derive the output 'directory' from the source file path.

    Returns the parent directory of the first available source file path,
    stored PORTABLE-FIRST (CWD-relative via microBase canonical_directory; the shared
    microBase canonical form, matching microModel infer.db writers).
    Shared by the image and object profilers so their tables can be joined
    on 'directory'.
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
    p = str(source_path)
    if not os.path.isabs(p):
        p = os.path.join(str(root), p)   # rows may carry root-relative paths
    return canonical_directory(os.path.dirname(p))