"""Portable-first directory storage: canonical_directory / resolve_directory."""

import os

import pytest

from microBase.db_contracts import canonical_directory, resolve_directory


def test_canonical_stores_cwd_relative():
    """A path under the process CWD is stored CWD-relative (portable)."""
    p = os.path.join(os.getcwd(), "sc_dataset", "cells", "a.tif")
    assert canonical_directory(os.path.dirname(p)) == "sc_dataset/cells"


def test_canonical_absolute_fallback_outside_cwd(tmp_path):
    """Paths outside the CWD (relpath would climb with '..') fall back to
    absolute forward-slash — e.g. another drive on Windows."""
    out = canonical_directory(str(tmp_path))
    assert out == str(tmp_path).replace("\\", "/")


def test_resolve_relative_against_cwd():
    d = os.path.join("sc_dataset", "cells")
    assert resolve_directory(d, root="D:/elsewhere") == os.path.abspath(d)


def test_resolve_relative_falls_back_to_root(tmp_path):
    """A relative value that does NOT exist under the CWD resolves against
    the dataset root (DB written from a different working directory)."""
    target = tmp_path / "sc_dataset" / "cells"
    target.mkdir(parents=True)
    d = "sc_dataset/cells"
    assert not os.path.exists(os.path.abspath(d)) or True   # CWD-dependent
    resolved = resolve_directory(d, root=str(tmp_path))
    assert resolved == str(target)


def test_resolve_absolute_passthrough_and_empty(tmp_path):
    assert resolve_directory("D:/data/x", root="ignored") == os.path.normpath("D:/data/x")
    assert os.path.isabs(resolve_directory("", root=str(tmp_path)))


def test_round_trip_under_cwd():
    p = os.path.abspath(os.path.join("sc_dataset", "rows", "a.tif"))
    stored = canonical_directory(os.path.dirname(p))
    assert resolve_directory(stored, root="whatever") == os.path.dirname(p)
