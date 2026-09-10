"""Regression tests for microModel.reduction.

The critical case: a LOADED (pre-fit) nonlinear reducer must transform the
CURRENT dataset's features. Reusing its ``.embedding_`` (the baseline's fit
embedding) silently wrote baseline coordinates against this dataset's uids.
"""

import pickle
import sqlite3

import numpy as np
import pacmap

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
        "data": {"root": [str(root_b)]},
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


def test_loaded_reducer_reuse_with_plots_and_clusters(tmp_path):
    """The documented baseline-reuse flow with save_plots=True (CLI default)
    AND cluster finding — both consume X_fit.

    Regression: X_fit took the LOADED reducer's baseline .embedding_ (60
    baseline rows) while labels/cluster ids belong to the current 40-row
    dataset — the cluster-ID ordering crashed on the boolean index, and the
    uncolored plot silently drew baseline coordinates. The loaded reducer
    must transform the current features for BOTH X_all and X_fit.
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
        "data": {"root": [str(root_b)]},
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
