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
    assert canonical_directory(sub, root) == "images/wellA"
    assert canonical_directory(root, root) == "."
    # Paths outside the root keep their leading .. (never silently escaped).
    assert canonical_directory(tmp_path, root).startswith("..")
    # Forward slashes on every platform.
    assert "\\" not in canonical_directory(sub, root)


def test_resolve_directory_roundtrip(tmp_path):
    root = tmp_path / "dataset"
    sub = root / "images" / "wellA"
    sub.mkdir(parents=True)

    rel = canonical_directory(sub, root)
    assert resolve_directory(rel, root) == str(sub.resolve())
    # Root-relative "." resolves to the root itself.
    assert resolve_directory(".", root) == str(root.resolve())
    # Empty resolves to the root.
    assert resolve_directory("", root) == str(root.resolve())
    # An already-absolute stored value (legacy) wins over the root.
    assert resolve_directory(str(sub), tmp_path) == str(sub)
    # Windows separators in stored values still resolve.
    assert resolve_directory("images\\wellA", root) == str(sub.resolve())
