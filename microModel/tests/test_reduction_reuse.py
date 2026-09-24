"""Regression tests for microModel.reduction.

The critical case: a LOADED (pre-fit) nonlinear reducer must transform the
CURRENT dataset's features. Reusing its ``.embedding_`` (the baseline's fit
embedding) silently wrote baseline coordinates against this dataset's uids.
"""

import pickle
import sqlite3

import numpy as np
import pacmap
import pytest

from microModel.reduction import run_reduction


def _write_features_db(path, feats):
    """Write a minimal inference table with float32 feature BLOBs."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE inference ("
        "uid INTEGER PRIMARY KEY AUTOINCREMENT, "
        "directory TEXT NOT NULL, filename TEXT NOT NULL, features BLOB)")
    conn.executemany(
        "INSERT INTO inference (directory, filename, features) VALUES (?, ?, ?)",
        [(".", f"cell_{i}.tiff", np.asarray(f, dtype=np.float32).tobytes())
         for i, f in enumerate(feats)])
    conn.commit()
    conn.close()


def test_loaded_pacmap_transforms_current_features(tmp_path):
    """A reducer fitted on dataset A, reused on dataset B, must produce
    B's transformed coordinates (not A's embedding rows)."""
    rng = np.random.default_rng(0)
    feats_a = rng.normal(size=(60, 8)).astype(np.float32)
    feats_b = rng.normal(size=(40, 8)).astype(np.float32)

    reducer = pacmap.PaCMAP(
        n_components=2, random_state=0, save_tree=True).fit(feats_a)
    pickle_path = tmp_path / "reduction_pacmap.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(reducer, f)

    root_b = tmp_path / "dataset_b"
    root_b.mkdir()
    _write_features_db(root_b / "infer.db", feats_b)

    config = {
        "data": {"file_dir": [str(root_b)]},
        "inference": {"db_name": "infer.db"},
        "reduction": {
            "method": ["pacmap"],
            "reduction_pacmap": str(pickle_path),
            "cluster_res": [],
            "show_cluster_image": False,
            "sample_per_class": 0,
        },
    }
    run_reduction(config, save_plots=False, raise_on_error=True)

    conn = sqlite3.connect(str(root_b / "infer.db"))
    rows = conn.execute(
        "SELECT uid, pacmap_1, pacmap_2 FROM reduction_pacmap ORDER BY uid"
    ).fetchall()
    conn.close()
    assert len(rows) == len(feats_b)

    expected = reducer.transform(feats_b)
    got = np.array([[r[1], r[2]] for r in rows])
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_empty_method_list_fits_nothing(tmp_path):
    """Explicit `method: []` means NO DR fitting — it must not fall back to
    the [pca, umap] legacy default (only null means default; the same
    contract is written down in microProfiler's config validation)."""
    config = {
        "data": {"file_dir": [str(tmp_path)]},
        "inference": {"db_name": "infer.db"},
        "reduction": {"method": [], "cluster_res": []},
    }
    with pytest.raises(ValueError, match="No valid reduction"):
        run_reduction(config, save_plots=False, raise_on_error=True)


def test_loaded_reducer_reuse_with_plots_and_clusters(tmp_path):
    """The documented baseline-reuse flow with save_plots=True (CLI default)
    AND cluster finding — plots and tables both consume the embedded rows.

    Regression: the plot embedding took the LOADED reducer's baseline
    .embedding_ (60 baseline rows) while labels/cluster ids belong to the
    current 40-row dataset — the cluster-ID ordering crashed on the boolean
    index, and the uncolored plot silently drew baseline coordinates. The
    loaded reducer must transform the current features for the tables and
    the plots alike.
    """
    rng = np.random.default_rng(1)
    feats_a = rng.normal(size=(60, 8)).astype(np.float32)
    feats_b = rng.normal(size=(40, 8)).astype(np.float32)

    reducer = pacmap.PaCMAP(
        n_components=2, random_state=0, save_tree=True).fit(feats_a)
    pickle_path = tmp_path / "reduction_pacmap.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(reducer, f)

    root_b = tmp_path / "dataset_b"
    root_b.mkdir()
    _write_features_db(root_b / "infer.db", feats_b)

    config = {
        "data": {"file_dir": [str(root_b)]},
        "inference": {"db_name": "infer.db"},
        "reduction": {
            "method": ["pacmap"],
            "reduction_pacmap": str(pickle_path),
            "cluster_res": [0.5],
            "show_cluster_image": False,
            "sample_per_class": 0,
        },
    }
    # Plots land next to the DB (save_dirs falls back to the DB's directory).
    run_reduction(config, save_plots=True, raise_on_error=True)

    assert (root_b / "reduction_pacmap.pdf").exists()

    conn = sqlite3.connect(str(root_b / "infer.db"))
    rows = conn.execute(
        "SELECT uid, pacmap_1, pacmap_2 FROM reduction_pacmap ORDER BY uid"
    ).fetchall()
    # Resolution tags contain a dot, so the column name needs quoting.
    cluster_rows = conn.execute(
        'SELECT uid, "cluster_res_0.5" FROM find_cluster ORDER BY uid'
    ).fetchall()
    conn.close()

    # The tables hold THIS dataset's rows, transformed by the loaded reducer.
    assert len(rows) == len(feats_b)
    expected = reducer.transform(feats_b)
    got = np.array([[r[1], r[2]] for r in rows])
    np.testing.assert_allclose(got, expected, atol=1e-5)
    # Every current-dataset uid got a cluster id (1-based).
    assert len(cluster_rows) == len(feats_b)
    assert all(isinstance(r[1], int) and r[1] >= 1 for r in cluster_rows)


def test_sample_per_class_never_subsamples_the_fit(tmp_path):
    """sample_per_class thins ONLY the plot points: the reducers (and the
    cluster whitening) must be fit on ALL rows even when it is tiny."""
    rng = np.random.default_rng(2)
    # Distinct rows so any small subsample's mean differs from the global one.
    feats = (np.arange(50 * 8, dtype=np.float64).reshape(50, 8)
             + rng.normal(scale=1e-3, size=(50, 8))).astype(np.float32)

    root = tmp_path / "dataset"
    root.mkdir()
    _write_features_db(root / "infer.db", feats)

    config = {
        "data": {"file_dir": [str(root)]},
        "inference": {"db_name": "infer.db"},
        "reduction": {
            "method": ["pca"],
            "cluster_res": [],
            "show_cluster_image": False,
            "sample_per_class": 2,
        },
    }
    run_reduction(config, save_plots=False, raise_on_error=True)

    # The fitted PCA saw every row: its mean_ equals the full-data mean. (A
    # fit on a 2-point subsample — the old behavior — would be far off.)
    with open(root / "reduction_pca.pkl", "rb") as f:
        pca = pickle.load(f)
    np.testing.assert_allclose(
        pca.mean_, feats.mean(axis=0, dtype=np.float64), atol=1e-3)

    # And the table still carries every object.
    conn = sqlite3.connect(str(root / "infer.db"))
    rows = conn.execute("SELECT uid FROM reduction_pca ORDER BY uid").fetchall()
    conn.close()
    assert len(rows) == len(feats)
