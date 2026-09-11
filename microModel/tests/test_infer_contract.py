"""infer.db writer output matches the shared microBase.db_contracts schema."""

import sqlite3

import torch

from microBase.db_contracts import (
    DIRECTORY_COLUMN,
    FEATURES_COLUMN,
    FILENAME_COLUMN,
    GROUND_TRUTH_COLUMN,
    LABEL_COLUMN,
    MASK_FILENAME_COLUMN,
    MASK_NAME_COLUMN,
    PRED_CLASS_COLUMN,
    PRED_PROB_COLUMN,
    PROB_COLUMN_PREFIX,
    UID_COLUMN,
)
from microModel.infer import _write_db


def _columns(path):
    conn = sqlite3.connect(str(path))
    cols = [row[1] for row in conn.execute("PRAGMA table_info(inference)")]
    conn.close()
    return cols


def test_single_cell_columns_match_contract(tmp_path):
    db = tmp_path / "infer.db"
    meta = [{"directory": ".", "filename": "a.tiff", "ground_truth": "gt",
             "dose": "5"}]
    _write_db(
        str(db), meta,
        all_logits=[torch.tensor([[2.0, 1.0]])],
        all_features=[torch.tensor([[1.0, 2.0]])],
        class_names=["a", "b"], write_features=True,
        extra_cols=["dose"], mode="single_cell", write_pred_class=True,
    )
    assert _columns(db) == [
        UID_COLUMN, DIRECTORY_COLUMN, FILENAME_COLUMN, MASK_NAME_COLUMN,
        GROUND_TRUTH_COLUMN, "dose", PRED_CLASS_COLUMN, PRED_PROB_COLUMN,
        f"{PROB_COLUMN_PREFIX}a", f"{PROB_COLUMN_PREFIX}b", FEATURES_COLUMN,
    ]
    # Single-cell inference has no mask file — the column stays NULL.
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT mask_name FROM inference").fetchone()
    conn.close()
    assert row[0] is None


def test_whole_image_columns_match_contract(tmp_path):
    db = tmp_path / "infer.db"
    meta = [{"directory": ".", "filename": '["a.tiff"]', "label": 1,
             "mask_filename": "/m/mask.png", "ground_truth": None}]
    _write_db(
        str(db), meta, all_logits=[], all_features=[torch.tensor([[1.0]])],
        class_names=[], write_features=True, mode="whole_image",
        write_pred_class=False, mask_name="cell",
    )
    assert _columns(db) == [
        UID_COLUMN, DIRECTORY_COLUMN, FILENAME_COLUMN, MASK_FILENAME_COLUMN,
        MASK_NAME_COLUMN, LABEL_COLUMN, GROUND_TRUTH_COLUMN,
        PRED_CLASS_COLUMN, PRED_PROB_COLUMN, FEATURES_COLUMN,
    ]
    # The bare mask name lands on every row (per-mask merge grouping key).
    conn = sqlite3.connect(str(db))
    rows = conn.execute(f"SELECT {MASK_NAME_COLUMN}, label FROM inference"
                        ).fetchall()
    conn.close()
    assert rows == [("cell", 1)]
