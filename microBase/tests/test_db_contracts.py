"""Tests for the shared profiler.db / infer.db contract module."""

import os

from microBase import (
    DR_METHODS,
    bare_mask_name,
    canonical_directory,
    is_numeric_sql_type,
    mask_column,
    reduction_coord_prefix,
    reduction_table_name,
    reserved_inference_columns,
    resolve_directory,
    sql_ident,
)


def test_table_and_coord_names():
    assert reduction_table_name("pca") == "reduction_pca"
    assert reduction_table_name("localmap") == "reduction_localmap"
    # PCA is the historical exception: pc_1/pc_2, others method_1/method_2.
    assert reduction_coord_prefix("pca") == "pc"
    assert reduction_coord_prefix("umap") == "umap"
    assert DR_METHODS == ("pca", "umap", "pacmap", "localmap")


def test_reduction_coord_prefix_covers_every_method():
    assert [reduction_coord_prefix(m) for m in DR_METHODS] == [
        "pc", "umap", "pacmap", "localmap"]


def test_sql_ident_escapes_quotes():
    assert sql_ident("plain") == '"plain"'
    assert sql_ident('a"b') == '"a""b"'
    assert sql_ident("prob_nuclear punctae") == '"prob_nuclear punctae"'


def test_reserved_inference_columns_includes_generated_probs():
    reserved = reserved_inference_columns(["a", "b"])
    assert "uid" in reserved
    assert "features" in reserved
    assert "pred_class" in reserved
    assert "prob_a" in reserved
    assert "prob_b" in reserved
    # A regex capture named like an existing column must be detectable.
    assert "directory" in reserved


def test_mask_column_helpers_roundtrip():
    assert mask_column("cell") == "mask_cell"
    assert bare_mask_name("mask_cell") == "cell"
    # Already-bare input passes through.
    assert bare_mask_name("cell") == "cell"


def test_is_numeric_sql_type():
    assert is_numeric_sql_type("INTEGER")
    assert is_numeric_sql_type("REAL")
    assert is_numeric_sql_type("DECIMAL(10,2)")
    assert not is_numeric_sql_type("TEXT")
    assert not is_numeric_sql_type("BLOB")


def test_canonical_directory(tmp_path):
    root = tmp_path / "dataset"
    sub = root / "images" / "wellA"
    sub.mkdir(parents=True)
    # The DB `directory` contract: PORTABLE-FIRST — CWD-relative when the
    # path lives under the CWD, absolute forward-slash otherwise (tmp_path
    # is outside the CWD, so an absolute value is the correct fallback).
    expected = str(sub.resolve()).replace("\\", "/")
    assert canonical_directory(sub) == expected
    # A relative path under the CWD stores CWD-relative (portable).
    assert canonical_directory("images/wellA") == "images/wellA"

    # Forward slashes on every platform.
    assert "\\" not in canonical_directory(sub)


def test_resolve_directory_roundtrip(tmp_path):
    root = tmp_path / "dataset"
    sub = root / "images" / "wellA"
    sub.mkdir(parents=True)

    stored = canonical_directory(sub)
    assert resolve_directory(stored, root) == str(sub.resolve())
    # Root-relative "." resolves to the root itself (legacy rows).
    assert resolve_directory(".", root) == str(root.resolve())
    # Empty resolves to the root.
    assert resolve_directory("", root) == str(root.resolve())
    # Windows separators in stored values still resolve.
    assert resolve_directory("images\\wellA", root) == str(sub.resolve())
