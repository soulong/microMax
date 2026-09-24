"""Mask-aware merge engine: mask resolution, per-mask fusion, round-trip."""

import sqlite3

import pandas as pd
import pytest

from microBase import db_merge
from microBase.db_contracts import infer_db_file_name


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


def _write_mask_db(path, mask_name, rows, cls_by_uid=None):
    """A NEW-STYLE per-model infer DB: one wide table named after the mask.

    rows: (uid, directory, label). Same directories/labels as the profiler
    fixtures so same-mask frames fuse into one row per object.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {mask_name} ("
        "directory TEXT, label INTEGER, well TEXT, pred_class TEXT, "
        "pred_prob REAL, umap_1 REAL, umap_2 REAL)")
    for uid, directory, label in rows:
        cls = (cls_by_uid or {}).get(uid, "drug")
        conn.execute(
            f"INSERT INTO {mask_name} VALUES (?, ?, 'A1', ?, 0.9, ?, ?)",
            (directory, label, cls, float(uid), -float(uid)))
    conn.commit()
    conn.close()


def test_multiple_prefixed_sources_get_model_prefix(tmp_path):
    """With infer frames from TWO OR MORE prefixed DBs, EVERY prefixed
    source's data columns are namespaced `<prefix>_<col>` (identity columns
    stay bare) — which model produced a column never depends on the order,
    and identical-named prediction columns can never collapse into one."""
    prof = tmp_path / "profiler.db"
    _write_profiler(prof, {"cell": [("A1", 1, 10.0)]})
    a = tmp_path / "infer_puncta.db"
    _write_mask_db(a, "cell", [(1, "C:/ds/Images", 1)])
    b = tmp_path / "infer_mito.db"
    _write_mask_db(b, "cell", [(1, "C:/ds/Images", 1)], cls_by_uid={1: "ctrl"})

    frames = (db_merge.read_profiler_frames(str(prof))
              + db_merge.read_profiler_frames(str(a), prefix="puncta")
              + db_merge.read_profiler_frames(str(b), prefix="mito"))
    fused = db_merge.fuse_frames(frames)

    # Fused per object (not stacked): profiler + BOTH models in one row.
    assert len(fused) == 1
    assert fused["area_cell"].iloc[0] == 10.0
    assert fused["puncta_pred_class"].iloc[0] == "drug"
    assert fused["mito_pred_class"].iloc[0] == "ctrl"
    assert "pred_class" not in fused.columns
    # Identity columns stay bare — they are the join keys.
    assert fused["well"].iloc[0] == "A1"
    assert fused["label"].iloc[0] == 1
    assert fused["directory"].iloc[0] == "C:/ds/Images"


def test_single_prefixed_source_keeps_bare_names(tmp_path):
    """One prefixed infer DB in the fusion keeps the concise bare names."""
    prof = tmp_path / "profiler.db"
    _write_profiler(prof, {"cell": [("A1", 1, 10.0)]})
    a = tmp_path / "infer_puncta.db"
    _write_mask_db(a, "cell", [(1, "C:/ds/Images", 1)])

    frames = (db_merge.read_profiler_frames(str(prof))
              + db_merge.read_profiler_frames(str(a), prefix="puncta"))
    fused = db_merge.fuse_frames(frames)

    assert fused["pred_class"].iloc[0] == "drug"
    assert not any(c.startswith("puncta_") for c in fused.columns)


def _write_raw_infer(path, mask_name, rows, cls_by_uid=None):
    """A raw microModel infer DB: inference + reduction_umap + find_cluster.

    rows: (uid, directory, label). Repeated calls append rows (the CREATEs
    are IF NOT EXISTS), so one file can hold several masks. cls_by_uid
    defaults to a fixed class.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS inference (uid INTEGER PRIMARY KEY, "
        "directory TEXT, filename TEXT, mask_name TEXT, label INTEGER, "
        "pred_class TEXT, pred_prob REAL, features BLOB)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reduction_umap (uid INTEGER PRIMARY KEY, "
        "umap_1 REAL, umap_2 REAL)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS find_cluster (uid INTEGER PRIMARY KEY, "
        "cluster_res_1 INTEGER, cluster_prob_1 REAL)")
    for uid, directory, label in rows:
        cls = (cls_by_uid or {}).get(uid, "drug")
        conn.execute(
            "INSERT INTO inference VALUES (?, ?, ?, ?, ?, ?, 0.9, NULL)",
            (uid, directory, f"{uid}.tiff", mask_name, label, cls))
        conn.execute("INSERT INTO reduction_umap VALUES (?, ?, ?)",
                     (uid, float(uid), -float(uid)))
        conn.execute("INSERT INTO find_cluster VALUES (?, ?, ?)",
                     (uid, label, 0.8))
    conn.commit()
    conn.close()


def test_absorb_writes_one_wide_table_per_mask(tmp_path):
    """absorb_infer_db folds inference + reduction + find_cluster into ONE
    table named after the mask — the file ends up profiler.db-shaped."""
    raw = tmp_path / "infer.db"
    _write_raw_infer(raw, "cell", [(1, "C:/ds", 1), (2, "C:/ds", 2)])
    out = tmp_path / "infer_puncta.db"

    written = db_merge.absorb_infer_db(raw, out, default_mask="cell")

    assert written == ["cell"]
    conn = sqlite3.connect(str(out))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    df = pd.read_sql("SELECT * FROM cell", conn)
    conn.close()
    assert tables == {"cell"}
    assert len(df) == 2
    # The uid join brought the coordinates and cluster columns along; the
    # infer-internal uid / features BLOB / mask_name are gone.
    for col in ("pred_class", "pred_prob", "umap_1", "umap_2",
                "cluster_res_1", "cluster_prob_1", "label", "directory"):
        assert col in df.columns, col
    for col in ("uid", "features", "mask_name"):
        assert col not in df.columns, col
    assert (df["umap_1"] == [1.0, 2.0]).all()
    assert (df["cluster_res_1"] == [1, 2]).all()

    # The raw file is untouched.
    src = sqlite3.connect(str(raw))
    assert src.execute("SELECT COUNT(*) FROM inference").fetchone()[0] == 2
    src.close()


def test_absorb_splits_masks_into_tables(tmp_path):
    raw = tmp_path / "infer.db"
    _write_raw_infer(raw, "cell", [(1, "C:/ds", 1)])
    _write_raw_infer(raw, "nuclei", [(2, "C:/ds", 1)])
    out = tmp_path / "infer_puncta.db"

    written = db_merge.absorb_infer_db(raw, out)
    assert sorted(written) == ["cell", "nuclei"]
    conn = sqlite3.connect(str(out))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert tables == {"cell", "nuclei"}


def test_absorb_replaces_rows_per_directory(tmp_path):
    """Re-absorbing the same directories replaces their rows (re-run
    protection, mirroring microModel's writer) instead of duplicating."""
    raw = tmp_path / "infer.db"
    _write_raw_infer(raw, "cell", [(1, "C:/ds", 1)])
    out = tmp_path / "infer_puncta.db"
    db_merge.absorb_infer_db(raw, out, default_mask="cell")

    # Same directory, new prediction: the row is replaced.
    raw2 = tmp_path / "infer2.db"
    _write_raw_infer(raw2, "cell", [(9, "C:/ds", 1)], cls_by_uid={9: "ctrl"})
    db_merge.absorb_infer_db(raw2, out, default_mask="cell")

    conn = sqlite3.connect(str(out))
    df = pd.read_sql("SELECT * FROM cell", conn)
    conn.close()
    assert len(df) == 1
    assert (df["pred_class"] == "ctrl").all()

    # A DIFFERENT directory appends (the other run's rows survive).
    raw3 = tmp_path / "infer3.db"
    _write_raw_infer(raw3, "cell", [(5, "C:/other", 3)])
    db_merge.absorb_infer_db(raw3, out, default_mask="cell")
    conn = sqlite3.connect(str(out))
    df = pd.read_sql("SELECT * FROM cell", conn)
    conn.close()
    assert len(df) == 2


def test_absorb_without_any_mask_raises(tmp_path):
    raw = tmp_path / "infer.db"
    conn = sqlite3.connect(str(raw))
    conn.execute(
        "CREATE TABLE inference (uid INTEGER PRIMARY KEY, directory TEXT, "
        "label INTEGER, pred_class TEXT)")
    conn.execute("INSERT INTO inference VALUES (1, 'C:/ds', 1, 'drug')")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="mask name"):
        db_merge.absorb_infer_db(raw, tmp_path / "infer_x.db")


def _profiler_and_infer(root, infer_names=("puncta",)):
    """profiler.db (cell + nuclei) + one absorbed infer DB per name."""
    _write_profiler(root / "profiler.db",
                    {"cell": [("A1", 1, 10.0)],
                     "nuclei": [("A1", 1, 99.0)]})
    for name in infer_names:
        raw = root / f"raw_{name}.db"
        # Same directory+label as the profiler objects -> the frames fuse.
        _write_raw_infer(raw, "cell", [(1, "C:/ds/Images", 1)])
        db_merge.absorb_infer_db(raw, root / infer_db_file_name(name),
                                 default_mask="cell")
        raw.unlink()


def test_build_merged_db_single_infer_db_bare(tmp_path):
    """merged.db: profiler objects fused with the infer rows per mask; ONE
    infer DB keeps bare column names; each mask stays its own table."""
    _profiler_and_infer(tmp_path)

    out = db_merge.build_merged_db(tmp_path)
    assert out and out.endswith("merged.db")
    conn = sqlite3.connect(out)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    cell = pd.read_sql("SELECT * FROM cell", conn)
    nuclei = pd.read_sql("SELECT * FROM nuclei", conn)
    conn.close()

    assert tables == {"cell", "nuclei"}
    assert len(cell) == 1      # fused: profiler object + infer row
    assert cell["area_cell"].iloc[0] == 10.0
    assert cell["pred_class"].iloc[0] == "drug"
    assert "umap_1" in cell.columns
    # Bare names with a single infer DB.
    assert not any(c.startswith("puncta_") for c in cell.columns)
    # The nuclei table is the profiler table alone (its mask has no infer).
    assert len(nuclei) == 1
    assert nuclei["area_nuclei"].iloc[0] == 99.0
    assert "pred_class" not in nuclei.columns


def test_build_merged_db_multi_infer_db_prefixes(tmp_path):
    _profiler_and_infer(tmp_path, infer_names=("puncta", "mito"))

    out = db_merge.build_merged_db(tmp_path)
    conn = sqlite3.connect(out)
    cell = pd.read_sql("SELECT * FROM cell", conn)
    conn.close()

    assert len(cell) == 1
    assert "puncta_pred_class" in cell.columns
    assert "mito_pred_class" in cell.columns
    assert "pred_class" not in cell.columns
    assert cell["puncta_pred_class"].iloc[0] == "drug"


def test_build_merged_db_prefixes_disjoint_masks(tmp_path):
    """The dataset-level prefix rule does not depend on mask overlap: two
    infer DBs over DIFFERENT masks still namespace every model's columns —
    a mask's columns must never flip bare/prefixed when a second model is
    run on it later."""
    _write_profiler(tmp_path / "profiler.db",
                    {"cell": [("A1", 1, 10.0)],
                     "nuclei": [("A1", 1, 99.0)]})
    raw_a = tmp_path / "raw_a.db"
    _write_raw_infer(raw_a, "cell", [(1, "C:/ds/Images", 1)])
    db_merge.absorb_infer_db(raw_a, tmp_path / infer_db_file_name("puncta"),
                             default_mask="cell")
    raw_b = tmp_path / "raw_b.db"
    _write_raw_infer(raw_b, "nuclei", [(1, "C:/ds/Images", 1)])
    db_merge.absorb_infer_db(raw_b, tmp_path / infer_db_file_name("mito"),
                             default_mask="nuclei")

    out = db_merge.build_merged_db(tmp_path)
    conn = sqlite3.connect(out)
    cell = pd.read_sql("SELECT * FROM cell", conn)
    nuclei = pd.read_sql("SELECT * FROM nuclei", conn)
    conn.close()

    assert len(cell) == 1
    assert cell["puncta_pred_class"].iloc[0] == "drug"
    assert "pred_class" not in cell.columns
    assert len(nuclei) == 1
    assert nuclei["mito_pred_class"].iloc[0] == "drug"
    assert "pred_class" not in nuclei.columns


def test_build_merged_db_rebuilt_from_scratch(tmp_path):
    """A stale mask table in merged.db does not survive a rebuild."""
    _profiler_and_infer(tmp_path)
    stale = tmp_path / "merged.db"
    conn = sqlite3.connect(str(stale))
    conn.execute("CREATE TABLE ghost (label INTEGER)")
    conn.commit()
    conn.close()

    db_merge.build_merged_db(tmp_path)
    conn = sqlite3.connect(str(stale))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "ghost" not in tables


def test_build_merged_db_skips_raw_infer_dbs(tmp_path):
    """A raw (not absorbed) infer DB must not masquerade as an object
    table — the `inference` table is not a per-mask profiler table."""
    _write_profiler(tmp_path / "profiler.db", {"cell": [("A1", 1, 10.0)]})
    _write_raw_infer(tmp_path / "infer_raw.db", "cell", [(1, "C:/ds", 1)])

    out = db_merge.build_merged_db(tmp_path)
    conn = sqlite3.connect(out)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    cell = pd.read_sql("SELECT * FROM cell", conn)
    conn.close()
    assert tables == {"cell"}
    assert "pred_class" not in cell.columns    # raw rows did not sneak in


def test_build_merged_db_without_sources_removes_stale_output(tmp_path):
    stale = tmp_path / "merged.db"
    stale.write_bytes(b"x")
    assert db_merge.build_merged_db(tmp_path) is None
    assert not stale.exists()


def test_build_merged_db_profiler_only(tmp_path):
    """No infer DBs: merged.db mirrors profiler.db's object tables."""
    _write_profiler(tmp_path / "profiler.db", {"cell": [("A1", 1, 10.0)]})
    out = db_merge.build_merged_db(tmp_path)
    assert out
    conn = sqlite3.connect(out)
    df = pd.read_sql("SELECT * FROM cell", conn)
    conn.close()
    assert len(df) == 1
    assert df["area_cell"].iloc[0] == 10.0
