"""Tests for microVis.io.infer_db: read-only infer.db access."""

import sqlite3

import numpy as np
import pytest

from microVis.io.infer_db import InferDB


def _make_infer_db(path):
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE "inference" (
            uid INTEGER PRIMARY KEY AUTOINCREMENT,
            directory TEXT,
            filename TEXT,
            well TEXT,
            field TEXT,
            pred_class TEXT,
            pred_prob REAL,
            features BLOB
        );
        CREATE TABLE "reduction_umap" (
            uid INTEGER PRIMARY KEY,
            umap_1 REAL,
            umap_2 REAL
        );
        CREATE TABLE "find_cluster" (
            uid INTEGER PRIMARY KEY,
            "cluster_res_0.5" INTEGER,
            "cluster_prob_0.5" REAL
        );
        """
    )
    for uid, directory in ((1, r"D:/data/ds1"), (2, r"D:/data/ds1"),
                           (3, r"D:/data/ds2")):
        conn.execute(
            'INSERT INTO "inference" (uid, directory, filename, well, field, '
            "pred_class, pred_prob, features) VALUES (?,?,?,?,?,?,?,?)",
            (uid, directory, f"cell{uid}.tiff", f"A{uid}", "1",
             "classA" if uid < 3 else "classB", 0.9,
             np.array([1.0, 2.0], dtype=np.float32).tobytes()),
        )
        conn.execute(
            'INSERT INTO "reduction_umap" (uid, umap_1, umap_2) VALUES (?,?,?)',
            (uid, float(uid), float(uid) * 2),
        )
        conn.execute(
            'INSERT INTO "find_cluster" (uid, "cluster_res_0.5", "cluster_prob_0.5") '
            "VALUES (?,?,?)", (uid, uid, 0.8),
        )
    conn.commit()
    conn.close()
    return path


def test_methods_directories_and_columns(tmp_path):
    db = InferDB(_make_infer_db(tmp_path / "infer.db"))
    assert db.list_reduction_methods() == ["umap"]
    assert db.list_directories() == ["D:/data/ds1", "D:/data/ds2"]
    cols = dict(db.list_columns())
    assert "features" not in cols and "uid" not in cols
    assert "pred_class" in cols
    assert "cluster_res_0.5" in cols
    db.close()


def test_load_scatter_joins_and_excludes_features(tmp_path):
    db = InferDB(_make_infer_db(tmp_path / "infer.db"))
    df = db.load_scatter("umap")
    assert len(df) == 3
    assert list(df["uid"]) == [1, 2, 3]
    assert "features" not in df.columns
    assert "umap_1" in df.columns and "umap_2" in df.columns
    assert "cluster_res_0.5" in df.columns
    assert df.loc[df["uid"] == 1, "umap_2"].iloc[0] == 2.0
    assert db.coord_columns("pca") == ("pc_1", "pc_2")
    assert db.coord_columns("localmap") == ("localmap_1", "localmap_2")
    db.close()


def test_missing_method_raises(tmp_path):
    db = InferDB(_make_infer_db(tmp_path / "infer.db"))
    with pytest.raises(ValueError, match="reduction_pca"):
        db.load_scatter("pca")
    db.close()


def test_load_inference_columns_and_directory_filter(tmp_path):
    db = InferDB(_make_infer_db(tmp_path / "infer.db"))
    df = db.load_inference()
    assert len(df) == 3
    assert "features" not in df.columns
    assert "pred_class" in df.columns
    assert "cluster_res_0.5" in df.columns

    scoped = db.load_inference("D:/data/ds1")
    assert sorted(scoped["uid"]) == [1, 2]
    # No directory matches -> fall back to all rows.
    fallback = db.load_inference("D:/elsewhere")
    assert len(fallback) == 3
    db.close()


def test_load_inference_scopes_list_and_relative_directories(tmp_path):
    """The canonical DB form is relative to the dataset root; scopes come from
    the dataset's own metadata directory values."""
    path = tmp_path / "infer.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE "inference" (
            uid INTEGER PRIMARY KEY AUTOINCREMENT,
            directory TEXT, filename TEXT, well TEXT
        );
        """
    )
    for uid, directory in ((1, "."), (2, "images/wellA"), (3, "images/wellB")):
        conn.execute(
            'INSERT INTO "inference" (uid, directory, filename, well) VALUES (?,?,?,?)',
            (uid, directory, f"cell{uid}.tiff", f"A{uid}"),
        )
    conn.commit()
    conn.close()

    db = InferDB(path)
    scoped = db.load_inference(["images/wellA"])
    assert sorted(scoped["uid"]) == [2]
    # Multiple scopes are a union.
    both = db.load_inference(["images/wellA", "images/wellB"])
    assert sorted(both["uid"]) == [2, 3]
    # "." matches root-level rows only.
    root_rows = db.load_inference(["."])
    assert sorted(root_rows["uid"]) == [1]
    # Legacy absolute values keep working.
    assert len(db.load_inference(str(tmp_path))) == 3
    db.close()


def test_load_inference_cache_invalidated_by_write_metadata(tmp_path):
    path = _make_infer_db(tmp_path / "infer.db")
    db = InferDB(path)
    before = db.load_inference()
    assert "treat" not in before.columns
    import pandas as pd
    db.write_metadata(pd.DataFrame({"well": ["A1"], "treat": ["x"]}))
    after = db.load_inference()
    assert "treat" in after.columns
    db.close()


def test_non_inference_db_rejected(tmp_path):
    path = tmp_path / "other.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="inference"):
        InferDB(path)
