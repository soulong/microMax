"""Annotation storage for the label app — append-friendly SQLite projects.

The dual-mode entry point keeps TWO fully independent projects in one
save_dir: the multi-label project in ``label_multiple.db`` and the
single-label project in ``label_single.db``. Each holds the label
registry, the per-cell current
decisions, the append-only decision history and the per-root display
settings. Every write lands in the history; an undo restores the previous
state from that history. Annotation passes therefore accumulate safely —
re-running the command on the same save_dir resumes the project, and
adding roots or labels never destroys existing annotations.

Tables
------
meta            project-level key/values (model bundle the decisions were
                made with, for consistency checks)
labels          label registry with a user-controlled display order (drag
                in the UI) and the auto-annotate bookkeeping
                (``auto_fired_at``); deletion is explicit and UI-confirmed
                and removes the label's decisions — the append-only log
                keeps the history
sessions        one row per server start (the session_id stamped on writes)
sources         per-source-root display settings (channels/layout/
                max_value) as they were when the root was last configured —
                legacy cells stay displayable even when their root leaves
                the config
cells           one row per known cell (normcase absolute filepath is the
                identity); registered on every startup, never removed, so
                cells from roots that dropped out of the config stay
                annotated and exportable
cell_labels     CURRENT decision per (cell, label): state 1 = positive,
                0 = explicit negative; re-annotating upserts this row
annotation_log  append-only history of every decision (including the
                cleared state NULL). ``op_id`` groups the rows of ONE user
                action (a batch apply, an auto run, an undo) so undo can
                revert exactly the last action.

Projects created before the dual-mode naming stored their decisions in
``label_multiple.db`` / ``annotations.db`` (multi) or
``annotations_single.db`` (single); the v0.21 unification briefly used a
single ``label.db``. ``migrate_project_db`` routes all of those to the
mode-specific names on the first startup so existing annotations survive.
"""

import datetime
import json
import logging
import os
import sqlite3
from contextlib import closing

import pandas as pd

logger = logging.getLogger(__name__)

DB_NAME = "label_multiple.db"           # the MULTI-label project database
DB_NAME_SINGLE = "label_single.db"      # the SINGLE-label project database
# Two fully independent projects live side by side in one save_dir (the
# dual-mode entry point): switching modes swaps the whole store, so a
# cell can carry different label sets in each. Shared between the modes:
# the feature cache, the embedding space and the suggest engines — all
# decision-independent (see server.py).
LEGACY_MULTI_DBS = ["annotations.db"]        # pre-0.14.1 multi project
LEGACY_DB_SINGLE = "annotations_single.db"   # pre-0.14.1 single project
UNIFIED_DB = "label.db"                      # v0.21 unified name (routed)

STATE_POS = 1
STATE_NEG = 0

# Auto-assigned label colors (cycled in registration order).
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#008080", "#f032e6", "#9a6324", "#469990", "#800000",
           "#000075", "#808000", "#e6c229", "#a9a9a9"]


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _db_label_mode(path):
    """The mode recorded in a project DB's meta ('multi' when unreadable)."""
    try:
        conn = sqlite3.connect(path)
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'label_mode'").fetchone()
        finally:
            conn.close()
        if row and row[0] == "single":
            return "single"
    except sqlite3.Error:
        pass
    return "multi"


def migrate_project_db(save_dir):
    """Route the project DBs to their mode-specific names.

    multi = ``label_multiple.db``, single = ``label_single.db``. Legacy
    files are renamed in when their target is missing: the pre-0.14.1
    ``annotations.db`` / ``annotations_single.db``, and the v0.21 unified
    ``label.db`` (the mode recorded in ITS meta decides which project it
    holds). A ``label.db`` whose target name is already taken is set
    aside as ``label.db.old`` — nothing is ever overwritten or destroyed.
    Returns (multi_db_path, single_db_path).
    """
    multi = os.path.join(save_dir, DB_NAME)
    single = os.path.join(save_dir, DB_NAME_SINGLE)
    if not os.path.exists(multi):
        for legacy in LEGACY_MULTI_DBS:
            old = os.path.join(save_dir, legacy)
            if os.path.exists(old):
                os.rename(old, multi)
                logger.info("Migrated %s -> %s (annotations preserved)",
                            legacy, DB_NAME)
                break
    if not os.path.exists(single) and \
            os.path.exists(os.path.join(save_dir, LEGACY_DB_SINGLE)):
        os.rename(os.path.join(save_dir, LEGACY_DB_SINGLE), single)
        logger.info("Migrated %s -> %s (annotations preserved)",
                    LEGACY_DB_SINGLE, DB_NAME_SINGLE)
    unified = os.path.join(save_dir, UNIFIED_DB)
    if os.path.exists(unified):
        target = single if _db_label_mode(unified) == "single" else multi
        if not os.path.exists(target):
            os.rename(unified, target)
            logger.info("Migrated %s -> %s (annotations preserved)",
                        UNIFIED_DB, os.path.basename(target))
        else:
            stray = unified + ".old"
            logger.warning("%s duplicates %s — set aside as %s",
                           UNIFIED_DB, os.path.basename(target), stray)
            os.rename(unified, stray)
    return multi, single


class AnnotationDB:
    """Thin SQLite wrapper for the annotation project (see module docstring).

    A fresh connection per call keeps the Flask threads safe without
    locking; SQLite serializes the (rare) writes itself.
    """

    def __init__(self, path):
        self.path = path
        with closing(self._connect()) as conn, conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta(
                key   TEXT PRIMARY KEY,
                value TEXT);
            CREATE TABLE IF NOT EXISTS labels(
                label_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL UNIQUE,
                color         TEXT NOT NULL,
                sort_order    INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT NOT NULL,
                auto_fired_at TEXT);
            CREATE TABLE IF NOT EXISTS sessions(
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sources(
                source         TEXT PRIMARY KEY,
                channels       TEXT,
                channel_layout TEXT,
                max_value      REAL,
                n_channels     INTEGER);
            CREATE TABLE IF NOT EXISTS cells(
                cell_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT NOT NULL UNIQUE,
                raw_path TEXT NOT NULL,
                source   TEXT NOT NULL,
                preset   TEXT,
                added_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cell_labels(
                cell_id    INTEGER NOT NULL REFERENCES cells(cell_id),
                label_id   INTEGER NOT NULL REFERENCES labels(label_id),
                state      INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                session_id INTEGER,
                source     TEXT NOT NULL DEFAULT 'manual',
                PRIMARY KEY (cell_id, label_id));
            CREATE TABLE IF NOT EXISTS annotation_log(
                log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT NOT NULL,
                session_id INTEGER,
                op_id      INTEGER NOT NULL DEFAULT 0,
                cell_id    INTEGER NOT NULL,
                label_id   INTEGER NOT NULL,
                state      INTEGER,
                source     TEXT NOT NULL DEFAULT 'manual');
            CREATE TABLE IF NOT EXISTS undone_ops(
                op_id     INTEGER PRIMARY KEY,
                undone_at TEXT NOT NULL);
            """)
            # Migrations for projects created before a column existed — an
            # existing annotation project must keep working untouched.
            for table, column, ddl in (
                    ("cell_labels", "source",
                     "ALTER TABLE cell_labels ADD COLUMN source "
                     "TEXT NOT NULL DEFAULT 'manual'"),
                    ("annotation_log", "source",
                     "ALTER TABLE annotation_log ADD COLUMN source "
                     "TEXT NOT NULL DEFAULT 'manual'"),
                    ("annotation_log", "op_id",
                     "ALTER TABLE annotation_log ADD COLUMN op_id "
                     "INTEGER NOT NULL DEFAULT 0"),
                    ("labels", "auto_fired_at",
                     "ALTER TABLE labels ADD COLUMN auto_fired_at TEXT")):
                cols = {r[1] for r in conn.execute(
                    f"PRAGMA table_info({table})").fetchall()}
                if column not in cols:
                    conn.execute(ddl)

    def _connect(self):
        return sqlite3.connect(self.path)

    def new_session(self):
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO sessions(started_at) VALUES (?)", (_now(),))
            return cur.lastrowid

    # -- labels -------------------------------------------------------------

    def add_label(self, name, color=None):
        """Create a label (idempotent: an existing name returns its row)."""
        name = str(name).strip()
        if not name:
            raise ValueError("label name must not be empty")
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT label_id, name, color FROM labels WHERE name = ?",
                (name,)).fetchone()
            if row is not None:
                return {"label_id": row[0], "name": row[1],
                        "color": row[2], "created": False}
            n = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
            color = color or PALETTE[n % len(PALETTE)]
            sort_order = conn.execute(
                "SELECT COALESCE(MAX(sort_order), 0) + 1 FROM labels"
            ).fetchone()[0]
            cur = conn.execute(
                "INSERT INTO labels(name, color, sort_order, created_at) "
                "VALUES (?,?,?,?)", (name, color, sort_order, _now()))
            return {"label_id": cur.lastrowid, "name": name,
                    "color": color, "created": True}

    def delete_label(self, label_id):
        """Delete a label and its decisions (UI double-confirms first).

        Returns the number of removed decision rows; the append-only
        annotation_log keeps every historical record.
        """
        with closing(self._connect()) as conn, conn:
            n = conn.execute(
                "DELETE FROM cell_labels WHERE label_id = ?",
                (label_id,)).rowcount
            conn.execute("DELETE FROM labels WHERE label_id = ?",
                         (label_id,))
        return n

    def reorder_labels(self, label_ids):
        """Persist a new display order (the UI's drag-and-drop result)."""
        with closing(self._connect()) as conn, conn:
            for i, lid in enumerate(label_ids):
                conn.execute(
                    "UPDATE labels SET sort_order = ? WHERE label_id = ?",
                    (i + 1, int(lid)))

    def mark_auto_fired(self, label_id):
        """Record that the auto-annotate pass ran for this label.

        The pass runs once per label until Remove-auto resets the marker —
        that is what makes "Remove auto" stick instead of being re-applied
        by the next manual write.
        """
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE labels SET auto_fired_at = ? WHERE label_id = ?",
                (_now(), int(label_id)))

    def get_meta(self, key):
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?",
                               (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key, value):
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                         (key, str(value)))

    def list_labels(self):
        """All labels in display order, with pos/neg counts + auto state."""
        with closing(self._connect()) as conn, conn:
            rows = conn.execute("""
                SELECT l.label_id, l.name, l.color,
                       COALESCE(SUM(cl.state = 1), 0),
                       COALESCE(SUM(cl.state = 0), 0),
                       l.auto_fired_at IS NOT NULL
                FROM labels l
                LEFT JOIN cell_labels cl ON cl.label_id = l.label_id
                GROUP BY l.label_id
                ORDER BY l.sort_order, l.label_id""").fetchall()
        return [{"label_id": r[0], "name": r[1], "color": r[2],
                 "n_pos": int(r[3]), "n_neg": int(r[4]),
                 "auto_fired": bool(r[5])} for r in rows]

    # -- cells / sources ----------------------------------------------------

    def upsert_source(self, source, channels, channel_layout, max_value,
                      n_channels):
        """Record the display settings of a configured root."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?)",
                (os.path.normcase(source), json.dumps(channels),
                 channel_layout, float(max_value), int(n_channels)))

    def source_settings(self, source):
        """Display settings recorded for a source root (or None)."""
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT channels, channel_layout, max_value FROM sources "
                "WHERE source = ?", (os.path.normcase(source),)).fetchone()
        if row is None:
            return None
        return {"channels": json.loads(row[0]), "channel_layout": row[1],
                "max_value": row[2]}

    def register_cells(self, rows):
        """Insert new cells (INSERT OR IGNORE) and refresh preset labels.

        rows: iterable of (normcase filepath, raw path, source, preset).
        Returns the full cell table ordered by cell_id — including cells
        from earlier sessions whose roots are no longer in the config.
        """
        ts = _now()
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT OR IGNORE INTO cells(filepath, raw_path, source, "
                "preset, added_at) VALUES (?,?,?,?,?)",
                [(fp, raw, src, preset, ts) for fp, raw, src, preset in rows])
            # The preset is informational (label_csv / folder name); refresh
            # it so an updated label_csv is reflected. Annotations are not
            # touched here.
            conn.executemany(
                "UPDATE cells SET preset = ? WHERE filepath = ?",
                [(preset, fp) for fp, _, _, preset in rows])
            out = conn.execute(
                "SELECT cell_id, filepath, raw_path, source, preset "
                "FROM cells ORDER BY cell_id").fetchall()
        return [{"cell_id": r[0], "filepath": r[1], "raw_path": r[2],
                 "source": r[3], "preset": r[4]} for r in out]

    # -- decisions ------------------------------------------------------------

    def cell_states(self):
        """All current decisions as {(cell_id, label_id): state}."""
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT cell_id, label_id, state FROM cell_labels").fetchall()
        return {(r[0], r[1]): int(r[2]) for r in rows}

    def auto_cells(self, label_id):
        """Cell ids currently auto-annotated positive for one label."""
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT cell_id FROM cell_labels WHERE label_id = ? AND "
                "state = 1 AND source = 'auto'",
                (label_id,)).fetchall()
        return {r[0] for r in rows}

    def next_op_id(self):
        """A fresh monotonically increasing op id (one per user action)."""
        with closing(self._connect()) as conn, conn:
            return int(conn.execute(
                "SELECT COALESCE(MAX(op_id), 0) + 1 FROM annotation_log"
            ).fetchone()[0])

    def _write_one(self, conn, cell_id, label_id, state, ts, session_id,
                   source):
        """Upsert/clear ONE current decision (inside an open transaction)."""
        if state is None:
            conn.execute(
                "DELETE FROM cell_labels WHERE cell_id = ? AND label_id = ?",
                (cell_id, label_id))
        else:
            conn.execute(
                "INSERT INTO cell_labels(cell_id, label_id, state, "
                "updated_at, session_id, source) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(cell_id, label_id) DO UPDATE SET "
                "state = excluded.state, "
                "updated_at = excluded.updated_at, "
                "session_id = excluded.session_id, "
                "source = excluded.source",
                (cell_id, label_id, int(state), ts, session_id, source))

    def set_label(self, cell_id, label_id, state, session_id, op_id=0,
                  source="manual"):
        """Upsert one decision; state None clears it. Always logged."""
        ts = _now()
        with closing(self._connect()) as conn, conn:
            self._write_one(conn, cell_id, label_id, state, ts, session_id,
                            source)
            conn.execute(
                "INSERT INTO annotation_log(ts, session_id, op_id, cell_id, "
                "label_id, state, source) VALUES (?,?,?,?,?,?,?)",
                (ts, session_id, op_id, cell_id, label_id, state, source))

    def apply_batch(self, cell_ids, label_ids, state, session_id, op_id,
                    source="manual"):
        """Write ONE decision to MANY cells × MANY labels in ONE transaction.

        All-or-nothing: a partially applied batch can never exist, so the
        UI and the DB can never disagree. Every (cell, label) pair is
        logged under the same op_id — the whole user action (e.g. "Apply +
        3 labels on 12 cells") is ONE undoable step. Returns the number of
        (cell, label) pairs written.
        """
        if isinstance(label_ids, int):
            label_ids = [label_ids]
        label_ids = [int(l) for l in label_ids]
        cell_ids = [int(c) for c in cell_ids]
        ts = _now()
        with closing(self._connect()) as conn, conn:
            for lid in label_ids:
                for cid in cell_ids:
                    self._write_one(conn, cid, lid, state, ts, session_id,
                                    source)
                conn.executemany(
                    "INSERT INTO annotation_log(ts, session_id, op_id, "
                    "cell_id, label_id, state, source) VALUES (?,?,?,?,?,?,?)",
                    [(ts, session_id, op_id, cid, lid, state, source)
                     for cid in cell_ids])
        return len(cell_ids) * len(label_ids)

    def auto_apply(self, label_id, cell_ids, session_id, op_id=0):
        """Bulk-insert AUTO positive decisions (undecided cells only).

        The WHERE-style ON CONFLICT DO NOTHING guard makes it idempotent
        even if the caller's snapshot is stale. Every write lands in the
        log with source='auto'. Returns the number of decisions written.
        """
        ts = _now()
        rows = [(int(cid), int(label_id), STATE_POS, ts, session_id, "auto")
                for cid in cell_ids]
        with closing(self._connect()) as conn, conn:
            cur = conn.executemany(
                "INSERT INTO cell_labels(cell_id, label_id, state, "
                "updated_at, session_id, source) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(cell_id, label_id) DO NOTHING",
                rows)
            applied = cur.rowcount
            conn.executemany(
                "INSERT INTO annotation_log(ts, session_id, op_id, cell_id, "
                "label_id, state, source) VALUES (?,?,?,?,?,?, 'auto')",
                [(ts, session_id, op_id, int(cid), int(label_id), STATE_POS)
                 for cid in cell_ids])
        return applied

    def clear_auto(self, label_id, session_id, op_id=0):
        """Remove every AUTO decision of one label (undo an auto-apply run).

        Each removal is logged with state NULL, and the label's fired
        marker resets so a later write can run the auto pass again.
        Returns rows removed.
        """
        ts = _now()
        with closing(self._connect()) as conn, conn:
            ids = [r[0] for r in conn.execute(
                "SELECT cell_id FROM cell_labels WHERE label_id = ? AND "
                "source = 'auto'", (label_id,)).fetchall()]
            conn.execute(
                "DELETE FROM cell_labels WHERE label_id = ? AND "
                "source = 'auto'", (label_id,))
            conn.executemany(
                "INSERT INTO annotation_log(ts, session_id, op_id, cell_id, "
                "label_id, state, source) VALUES (?,?,?,?,?,NULL,'auto-clear')",
                [(ts, session_id, op_id, cid, label_id) for cid in ids])
            conn.execute(
                "UPDATE labels SET auto_fired_at = NULL WHERE label_id = ?",
                (int(label_id),))
        return len(ids)

    def clear_other_positives(self, cell_ids, keep_label_ids, session_id,
                              op_id):
        """Single-label exclusivity: one cell holds at most ONE positive.

        Removes every positive decision of the given cells whose label is
        NOT in the keep-set (the labels written by the same user action) —
        keep-set semantics make the result independent of write order even
        for a multi-label batch. Explicit negatives stay. Each removal is
        logged with state NULL under the SAME op_id as the write, so one
        undo reverts the whole action (writes + clears together).

        Returns the list of removed (cell_id, label_id) pairs.
        """
        keeps = {int(l) for l in keep_label_ids}
        ts = _now()
        removed = []
        with closing(self._connect()) as conn, conn:
            for cid in cell_ids:
                rows = conn.execute(
                    "SELECT label_id FROM cell_labels "
                    "WHERE cell_id = ? AND state = 1",
                    (int(cid),)).fetchall()
                for (lid,) in rows:
                    if lid in keeps:
                        continue
                    conn.execute(
                        "DELETE FROM cell_labels WHERE cell_id = ? AND "
                        "label_id = ?", (int(cid), lid))
                    conn.execute(
                        "INSERT INTO annotation_log(ts, session_id, op_id, "
                        "cell_id, label_id, state, source) "
                        "VALUES (?,?,?,?,?,NULL,'exclusive')",
                        (ts, session_id, op_id, int(cid), lid))
                    removed.append((int(cid), lid))
        return removed

    def undo_last_op(self, session_id):
        """Revert the newest not-yet-undone user action of this session.

        The previous state of every (cell, label) touched by the op is the
        state of its latest log row BEFORE the op (None = undecided), so
        the restore is exact no matter how many passes touched the cell in
        between. Reverted ops are recorded in ``undone_ops`` and pure-undo
        ops are never targets, so repeated undos walk BACKWARD through the
        real actions (op3, op2, op1, then nothing) — each exactly once.
        Undoing an auto run also resets that label's fired marker, so auto
        can run again later.

        Returns {"op_id", "n", "labels"} or None when nothing is undoable.
        """
        ts = _now()
        with closing(self._connect()) as conn, conn:
            ops = conn.execute(
                "SELECT l.op_id, MIN(l.source), MAX(l.source) "
                "FROM annotation_log l "
                "WHERE l.session_id = ? AND l.op_id > 0 AND l.op_id NOT IN "
                "(SELECT op_id FROM undone_ops) "
                "GROUP BY l.op_id ORDER BY l.op_id DESC",
                (session_id,)).fetchall()
            # Skip ops that consist purely of 'undo' rows: they are the
            # result of previous undos, not user actions.
            op = next((r[0] for r in ops
                       if not (r[1] == "undo" and r[2] == "undo")), None)
            if op is None:
                return None
            op_min = conn.execute(
                "SELECT MIN(log_id) FROM annotation_log WHERE op_id = ?",
                (op,)).fetchone()[0]
            rows = conn.execute(
                "SELECT cell_id, label_id, source FROM annotation_log "
                "WHERE op_id = ? GROUP BY cell_id, label_id",
                (op,)).fetchall()
            new_op = int(conn.execute(
                "SELECT COALESCE(MAX(op_id), 0) + 1 FROM annotation_log"
            ).fetchone()[0])
            touched, auto_touched = set(), set()
            for cell_id, label_id, source in rows:
                prev = conn.execute(
                    "SELECT state FROM annotation_log "
                    "WHERE cell_id = ? AND label_id = ? AND log_id < ? "
                    "ORDER BY log_id DESC LIMIT 1",
                    (cell_id, label_id, op_min)).fetchone()
                prev_state = prev[0] if prev else None
                self._write_one(conn, cell_id, label_id, prev_state, ts,
                                session_id, "undo")
                conn.execute(
                    "INSERT INTO annotation_log(ts, session_id, op_id, "
                    "cell_id, label_id, state, source) "
                    "VALUES (?,?,?,?,?,?,'undo')",
                    (ts, session_id, new_op, cell_id, label_id, prev_state))
                touched.add(label_id)
                if source == "auto":
                    auto_touched.add(label_id)
            for lid in auto_touched:
                conn.execute(
                    "UPDATE labels SET auto_fired_at = NULL "
                    "WHERE label_id = ?", (lid,))
            conn.execute("INSERT INTO undone_ops VALUES (?,?)", (op, ts))
            return {"op_id": op, "n": len(rows),
                    "labels": sorted(touched)}

    def stats(self):
        """Global counters for the header bar (undecided = nothing yet)."""
        with closing(self._connect()) as conn, conn:
            total = conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(DISTINCT cell_id) FROM cell_labels"
            ).fetchone()[0]
        return {"total": int(total), "labeled": int(labeled),
                "undecided": int(total) - int(labeled)}

    def export_frame(self):
        """Positive decisions as a label_csv frame (raw-case paths).

        Multi-labels are ';' joined (micromodel train's multi-label mode);
        cells without any positive label export nothing — a closed-world
        multi-label classifier treats unlisted cells as negatives anyway.
        """
        with closing(self._connect()) as conn, conn:
            rows = conn.execute("""
                SELECT c.raw_path, l.name
                FROM cell_labels cl
                JOIN cells c ON c.cell_id = cl.cell_id
                JOIN labels l ON l.label_id = cl.label_id
                WHERE cl.state = 1
                ORDER BY c.cell_id, l.label_id""").fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["filepath", "label"])
        return df.groupby("filepath", sort=False)["label"].agg(
            ";".join).reset_index()
