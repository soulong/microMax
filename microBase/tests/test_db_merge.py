"""Mask-aware merge engine: mask resolution, per-mask fusion, round-trip."""

import sqlite3

import pandas as pd
import pytest

from microBase import db_merge


def _write_profiler(path, masks):
    """masks: {table_name: rows}. Tables are named after their mask."""
    conn = sqlite3.connect(str(path))
    for table, rows in masks.items():
        conn.execute(
            f"CREATE TABLE {table} (well TEXT, label INTEGER, directory TEXT, "
            f"area_{table} REAL)")
        for well, label, area in rows:
            conn.execute(
                f"INSERT INTO {table} VALUES (?, ?, 'C:/ds/Images', ?)",
                (well, label, area))
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
    """A `mask` column (previous merge output) wins over the table name."""
    p = tmp_path / "profiler.db"
    _write_profiler(p, {"cell": [("A1", 1, 10.0)],
                        "nuclei": [("A1", 1, 99.0)]})
    frames = db_merge.read_profiler_frames(str(p))
    by_table = {f.table: f for f in frames}
    # Table-name convention.
    assert by_table["cell"].mask == "cell"
    assert by_table["nuclei"].mask == "nuclei"

    # A previous merge output tags its rows: the tag wins over the name.
    conn = sqlite3.connect(str(p))
    conn.execute(
        "CREATE TABLE merged (well TEXT, label INTEGER, area REAL, mask TEXT)")
    conn.execute("INSERT INTO merged VALUES ('A1', 1, 5.0, 'cell')")
    conn.commit()
    conn.close()
    merged_frames = [f for f in db_merge.read_profiler_frames(str(p))
                     if f.table == "merged"]
    assert [f.mask for f in merged_frames] == ["cell"]


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


def test_quoted_identifiers_read(tmp_path):
    """Table/column names with embedded double quotes (legal class-name
    captures) must read back — identifiers go through sql_ident."""
    p = tmp_path / "profiler.db"
    conn = sqlite3.connect(str(p))
    # Column name: prob_we"ird (embedded quote doubled inside the identifier).
    conn.execute('CREATE TABLE cell (well TEXT, label INTEGER, '
                 '"prob_we""ird" REAL)')
    conn.execute('INSERT INTO cell VALUES (?, 1, 0.5)', ("A1",))
    conn.commit()
    conn.close()

    frames = db_merge.read_profiler_frames(str(p))
    assert len(frames) == 1
    assert frames[0].mask == "cell"
    assert 'prob_we"ird' in frames[0].df.columns
    assert frames[0].df["prob_we\"ird"].iloc[0] == 0.5


def test_null_mask_tag_is_not_the_string_nan(tmp_path):
    """A merge DB whose mask column holds NULLs must NOT come back with a
    mask literally called 'nan' — NULL means unknown (default-mask logic)."""
    p = tmp_path / "merged.db"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE merged (well TEXT, label INTEGER, "
                 "directory TEXT, mask TEXT)")
    conn.execute("INSERT INTO merged VALUES ('A1', 1, 'C:/ds/Images', NULL)")
    conn.execute("INSERT INTO merged VALUES ('A1', 2, 'C:/ds/Images', 'cell')")
    conn.commit()
    conn.close()

    frames = db_merge.read_profiler_frames(str(p))
    masks = sorted(f.mask if f.mask is not None else "?" for f in frames)
    assert masks == ["?", "cell"]       # NULL group survived, no "nan"
    for f in frames:
        assert "mask" not in f.df.columns

    # The unknown group falls through to the caller's default-mask logic —
    # it joins the default mask (logged hint) instead of forming a bogus
    # "nan" group, and the fused table stays a single untagged block.
    fused = db_merge.fuse_frames(
        db_merge.read_db_frames(str(p)), default_mask="cell")
    assert len(fused) == 2
    assert "mask" not in fused.columns
    assert sorted(fused["label"].tolist()) == [1, 2]


def test_same_mask_fuses_different_masks_stack(tmp_path):
    """cell objects fuse with cell infer rows; nuclei stays its own block."""
    p_prof = tmp_path / "profiler.db"
    _write_profiler(p_prof, {"cell": [("A1", 1, 10.0)],
                             "nuclei": [("A1", 1, 99.0)]})
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
    assert nuc["area_nuclei"] == 99.0
    assert pd.isna(nuc["pred_class"])


def test_unknown_infer_mask_joins_profiler_default(tmp_path, caplog):
    p_prof = tmp_path / "profiler.db"
    _write_profiler(p_prof, {"cell": [("A1", 1, 10.0)]})
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
    _write_profiler(p_prof, {"cell": [("A1", 1, 10.0)]})
    frames = db_merge.read_db_frames(str(p_prof))
    fused = db_merge.fuse_frames(frames)

    out = tmp_path / "merge_cell.db"
    written = db_merge.write_merged_db(fused, out, mask="cell")
    assert written.endswith("merge_cell.db")
    # The in-memory frame stays untouched; the WRITTEN table carries the
    # mask column so re-loading resolves the frame's mask from it.
    assert "mask" not in fused.columns
    again = db_merge.read_db_frames(out)
    assert [(f.table, f.mask) for f in again] == [("merged", "cell")]

    # A multi-mask fused table round-trips through its mask column.
    _write_profiler(tmp_path / "p2.db",
                    {"nuclei": [("A1", 1, 99.0)]})
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


def test_no_well_object_table_still_merges(tmp_path):
    """Datasets without a well capture are officially supported: an object
    table without `well` merges on the identity columns both frames carry
    (label + directory) instead of being silently skipped."""
    p_prof = tmp_path / "profiler.db"
    conn = sqlite3.connect(str(p_prof))
    conn.execute("CREATE TABLE cell (label INTEGER, directory TEXT, area REAL)")
    conn.execute("INSERT INTO cell VALUES (1, 'C:/ds/Images', 10.0)")
    conn.commit()
    conn.close()
    p_infer = tmp_path / "infer.db"
    conn = sqlite3.connect(str(p_infer))
    conn.execute(
        "CREATE TABLE inference (uid INTEGER PRIMARY KEY, directory TEXT, "
        "mask_name TEXT, label INTEGER, pred_class TEXT)")
    conn.execute("INSERT INTO inference VALUES (1, 'C:/ds/Images', 'cell', 1, 'drug')")
    conn.commit()
    conn.close()

    fused = db_merge.merge_dbs([str(p_prof), str(p_infer)])
    assert len(fused) == 1
    row = fused.iloc[0]
    assert row["area"] == 10.0
    assert row["pred_class"] == "drug"   # cross-source fusion happened


def test_multi_mask_infer_table_splits_per_mask(tmp_path):
    """One infer table holding several mask_name values yields one frame per
    mask — rows are never folded into the first mask."""
    p_prof = tmp_path / "profiler.db"
    _write_profiler(p_prof, {"cell": [("A1", 1, 10.0)],
                             "nuclei": [("A1", 1, 99.0)]})
    p_infer = tmp_path / "infer.db"
    conn = sqlite3.connect(str(p_infer))
    conn.execute(
        "CREATE TABLE inference (uid INTEGER PRIMARY KEY, directory TEXT, "
        "mask_name TEXT, label INTEGER, pred_class TEXT)")
    conn.executemany(
        "INSERT INTO inference VALUES (?, 'C:/ds/Images', ?, ?, ?)",
        [(1, "cell", 1, "drug"), (2, "nuclei", 1, "ctrl"),
         (3, "cell", 2, "drug")])
    conn.commit()
    conn.close()

    frames = db_merge.read_infer_frames(str(p_infer))
    assert sorted(f.mask for f in frames) == ["cell", "nuclei"]
    cell_frames = [f for f in frames if f.mask == "cell"]
    assert len(cell_frames[0].df) == 2

    fused = db_merge.merge_dbs([str(p_prof), str(p_infer)])
    cell = fused[(fused["mask"] == "cell") & (fused["label"] == 1)].iloc[0]
    assert cell["pred_class"] == "drug"
    nuc = fused[fused["mask"] == "nuclei"].iloc[0]
    assert nuc["pred_class"] == "ctrl"
    assert nuc["area_nuclei"] == 99.0


def test_equal_columns_do_not_duplicate(tmp_path):
    """Two same-mask frames carrying an IDENTICAL extra column keep one copy
    — the fused table never grows `<col>_x`/`<col>_y` duplicates."""
    p_prof = tmp_path / "profiler.db"
    _write_profiler(p_prof, {"cell": [("A1", 1, 10.0)]})
    p_infer = tmp_path / "infer.db"
    conn = sqlite3.connect(str(p_infer))
    conn.execute(
        "CREATE TABLE inference (uid INTEGER PRIMARY KEY, directory TEXT, "
        "mask_name TEXT, label INTEGER, area_cell REAL, pred_class TEXT)")
    conn.execute(
        "INSERT INTO inference VALUES (1, 'C:/ds/Images', 'cell', 1, 10.0, 'drug')")
    conn.commit()
    conn.close()

    fused = db_merge.merge_dbs([str(p_prof), str(p_infer)])
    assert len(fused) == 1
    assert "area_cell_x" not in fused.columns
    assert "area_cell_y" not in fused.columns
    assert fused["area_cell"].iloc[0] == 10.0
    assert fused["pred_class"].iloc[0] == "drug"
