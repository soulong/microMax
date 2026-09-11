"""infer.db schema evolution and reserved-column protection."""

import sqlite3

import pytest
import torch

from microModel.infer import _init_db, _reject_reserved_extra_cols, _write_db


def _columns(conn, table="inference"):
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_init_db_rebuilds_on_schema_change(tmp_path):
    db = tmp_path / "infer.db"
    conn = sqlite3.connect(str(db))
    _init_db(conn, "single_cell", extra_cols=["a"], prob_cols=[])
    assert _columns(conn) == ["uid", "directory", "filename", "mask_name",
                              "ground_truth", "a", "pred_class", "pred_prob",
                              "features"]

    # Same schema again: no rebuild, no error.
    _init_db(conn, "single_cell", extra_cols=["a"], prob_cols=[])
    assert _columns(conn)[2] == "filename"

    # Changed schema (new metadata + class probs): dropped and recreated.
    _init_db(conn, "whole_image", extra_cols=["a", "b"], prob_cols=["prob_x"])
    assert _columns(conn) == [
        "uid", "directory", "filename", "mask_filename", "mask_name", "label",
        "ground_truth", "a", "b", "pred_class", "pred_prob", "prob_x", "features",
    ]
    conn.close()


def test_reject_reserved_extra_cols():
    with pytest.raises(ValueError, match="collide"):
        _reject_reserved_extra_cols(["label"], [])
    with pytest.raises(ValueError, match="collide"):
        _reject_reserved_extra_cols(["prob_nuc"], ["nuc"])
    with pytest.raises(ValueError, match="collide"):
        _reject_reserved_extra_cols(["features"], [])
    # Non-reserved captures pass.
    _reject_reserved_extra_cols(["treatment", "dose"], [])


def test_write_db_rerun_with_changed_schema(tmp_path):
    """A features-only re-run on a DB that had classify columns must rebuild
    the table instead of failing the INSERT on the old schema."""
    db = tmp_path / "infer.db"
    meta = [{"directory": ".", "filename": "a.tiff", "ground_truth": None}]
    logits = [torch.tensor([[2.0, 1.0]])]
    feats = [torch.tensor([[1.0, 2.0]])]

    _write_db(str(db), meta, logits, feats, ["x", "y"], True,
              write_pred_class=True)
    conn = sqlite3.connect(str(db))
    cols = _columns(conn)
    assert "prob_x" in cols and "prob_y" in cols
    assert conn.execute("SELECT COUNT(*) FROM inference").fetchone()[0] == 1
    conn.close()

    _write_db(str(db), meta, [], feats, ["x", "y"], True,
              write_pred_class=False)
    conn = sqlite3.connect(str(db))
    cols = _columns(conn)
    assert "prob_x" not in cols
    rows = conn.execute("SELECT filename, pred_class, features FROM inference").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "a.tiff"
    assert rows[0][1] is None
    assert rows[0][2] is not None
