"""Mask-aware merge engine: mask resolution, per-mask fusion, round-trip."""

import sqlite3

import pandas as pd
import pytest

from microBase import db_merge
from microBase.db_contracts import TABLE_MASKS_TABLE


def _write_profiler(path, masks):
    """masks: {table_name: (mask_name or None-for-convention, rows)}."""
    conn = sqlite3.connect(str(path))
    mapping = []
    for table, (mask, rows) in masks.items():
        conn.execute(
            f"CREATE TABLE {table} (well TEXT, label INTEGER, directory TEXT, "
            f"area_{table} REAL)")
        for well, label, area in rows:
            conn.execute(
                f"INSERT INTO {table} VALUES (?, ?, 'C:/ds/Images', ?)",
                (well, label, area))
        if mask is not None:
            mapping.append((table, mask))
    if mapping:
        conn.execute(
            f"CREATE TABLE {TABLE_MASKS_TABLE} "
            "(table_name TEXT PRIMARY KEY, mask_name TEXT)")
        conn.executemany(
            f"INSERT INTO {TABLE_MASKS_TABLE} VALUES (?, ?)", mapping)
    conn.commit()
    conn.close()


def _write_infer(path, mask_name, rows):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE inference (uid INTEGER PRIMARY KEY, directory TEXT, "
        "mask_name TEXT, label INTEGER, pred_class TEXT, features BLOB)")
    for uid, label, cls in rows:
        conn.execute(
            "INSERT INTO inference VALUES (?, 'C:/ds/Images', ?, ?, ?, NULL)",
            (uid, mask_name, label, cls))
    conn.commit()
    conn.close()


def test_profiler_mask_resolution_rules(tmp_path):
    """mask column (merge output) > _table_masks bookkeeping > table name."""
    p = tmp_path / "profiler.db"
    _write_profiler(p, {"cell": (None, [("A1", 1, 10.0)]),
                        "nuc": ("nuclei", [("A1", 1, 99.0)])})
    frames = db_merge.read_profiler_frames(str(p))
    by_table = {f.table: f for f in frames}
    # No bookkeeping row for 'cell' -> table-name convention.
    assert by_table["cell"].mask == "cell"
    # Bookkeeping wins for custom table names.
    assert by_table["nuc"].mask == "nuclei"


def test_infer_mask_from_column_or_none(tmp_path):
    p = tmp_path / "infer.db"
    _write_infer(p, "cell", [(1, 1, "drug"), (2, 2, "ctrl")])
    frames = db_merge.read_infer_frames(str(p))
    assert frames[0].mask == "cell"
    assert "mask_name" not in frames[0].df.columns
    assert "features" not in frames[0].df.columns

    # Legacy DBs without the column -> unknown mask.
    p2 = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(p2))
    conn.execute("CREATE TABLE inference (uid INTEGER PRIMARY KEY, "
                 "directory TEXT, label INTEGER)")
    conn.execute("INSERT INTO inference VALUES (1, 'C:/x', 1)")
    conn.commit()
    conn.close()
    assert db_merge.read_infer_frames(str(p2))[0].mask is None


def test_same_mask_fuses_different_masks_stack(tmp_path):
    """cell objects fuse with cell infer rows; nuclei stays its own block."""
    p_prof = tmp_path / "profiler.db"
    _write_profiler(p_prof, {"cell": (None, [("A1", 1, 10.0)]),
                             "nuc": ("nuclei", [("A1", 1, 99.0)])})
    p_infer = tmp_path / "infer.db"
    _write_infer(p_infer, "cell", [(1, 1, "drug")])

    frames = db_merge.read_db_frames(str(p_prof)) + \
        db_merge.read_db_frames(str(p_infer))
    fused = db_merge.fuse_frames(frames)

    # 1 fused cell row + 1 stacked nuclei row; groups tagged.
    assert len(fused) == 2
    assert set(fused["mask"]) == {"cell", "nuclei"}
    cell = fused[fused["mask"] == "cell"].iloc[0]
    assert cell["area_cell"] == 10.0
    assert cell["pred_class"] == "drug"     # cross-source fusion happened
    nuc = fused[fused["mask"] == "nuclei"].iloc[0]
    assert nuc["area_nuc"] == 99.0
    assert pd.isna(nuc["pred_class"])


def test_unknown_infer_mask_joins_profiler_default(tmp_path, caplog):
    p_prof = tmp_path / "profiler.db"
    _write_profiler(p_prof, {"cell": (None, [("A1", 1, 10.0)])})
    p_infer = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(p_infer))
    conn.execute("CREATE TABLE inference (uid INTEGER PRIMARY KEY, "
                 "directory TEXT, label INTEGER, pred_class TEXT)")
    conn.execute("INSERT INTO inference VALUES (1, 'C:/ds/Images', 1, 'drug')")
    conn.commit()
    conn.close()

    with caplog.at_level("INFO", logger="microBase.db_merge"):
        fused = db_merge.merge_dbs([str(p_prof), str(p_infer)])
    # Fused into ONE row with the profiler's mask assumption + log hint.
    assert len(fused) == 1
    assert "mask" not in fused.columns
    assert any("assuming mask 'cell'" in r.message for r in caplog.records)


def test_write_and_reload_round_trip(tmp_path):
    p_prof = tmp_path / "profiler.db"
    _write_profiler(p_prof, {"cell": (None, [("A1", 1, 10.0)])})
    frames = db_merge.read_db_frames(str(p_prof))
    fused = db_merge.fuse_frames(frames)

    out = tmp_path / "merge_cell.db"
    written = db_merge.write_merged_db(fused, out, mask="cell")
    assert written.endswith("merge_cell.db")
    # No mask column for a single-group table; the bookkeeping carries it.
    assert "mask" not in fused.columns
    again = db_merge.read_db_frames(out)
    assert [(f.table, f.mask) for f in again] == [("merged", "cell")]

    # A multi-mask fused table round-trips through its mask column.
    _write_profiler(tmp_path / "p2.db",
                    {"nuc": ("nuclei", [("A1", 1, 99.0)])})
    both = db_merge.fuse_frames(
        db_merge.read_profiler_frames(str(p_prof))
        + db_merge.read_profiler_frames(str(tmp_path / "p2.db")))
    out2 = tmp_path / "merge_both.db"
    db_merge.write_merged_db(both, out2)
    masks = sorted(f.mask for f in db_merge.read_db_frames(out2))
    assert masks == ["cell", "nuclei"]


def test_fuse_without_frames_raises():
    with pytest.raises(ValueError):
        db_merge.fuse_frames([])
