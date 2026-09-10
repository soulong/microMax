"""Interactive multi-label annotation web server with model-assisted suggestions.

`micromodel label` opens a local web page (Flask, same pattern as
reduction_vis) where every cropped single-cell image can receive any number
of labels. The label registry, the per-cell decisions and the full decision
history live in one append-friendly SQLite DB (`label.db` single-label mode,
`label_multiple.db` multi-label mode) inside the
config's save_dir, so re-running the command on the same save_dir — with the
same or new data roots, and adding new labels — never destroys existing
annotations.

Model assistance (optional but recommended):
  The config's `model` may be an SSL pretrain bundle OR a train bundle. All
  cells are embedded with the same extraction path as infer/deduplication
  (bundle-meta normalization, teacher backbone for SSL bundles), projected
  into a whitened-PCA + L2-normalized space and cached per root (the
  deduplication cache layout — adding files or changing the bundle invalidates
  it automatically). On top of that space a suggest engine provides:

    - kNN scoring per label: score(x, L) = mean of the top-k cosine
      similarities from x to L's positive cells (k = knn_k, default 1 =
      nearest-exemplar retrieval), minus neg_weight * the same quantity
      against L's explicit-negative cells. "Never labeled" is NOT a
      negative (positive-unlabeled); only explicit user rejections count.
    - a classify bundle's per-class probabilities as a second suggestion
      source for labels whose name equals a class name.

  Queue modes driven by the scores (see /api/queue):
    diverse    greedy farthest-point subset — the cold-start batch that
               covers the latent space before any label exists
    unlabeled  cells without any annotation yet (stable order)
    confirm    cells with at least one suggestion >= threshold, most
               confident first — review queue for auto-annotation
    label_top  for one label: undecided cells ranked by score — the fastest
               way to collect positives for that label
    uncertain  for one label: undecided cells nearest the threshold band —
               active-learning picks, most informative for a future
               classifier
    review     for one label: already-decided cells ranked by a
               leave-one-out kNN consistency margin — the re-labeling half
               of active learning (suspected mislabels surface first)
    all        everything (with source filter)

Outputs, all under save_dir:
  label.db / label_multiple.db   labels / cells / cell_labels / annotation_log / sources
  label_export.csv label_csv format (filepath,label with ';' joined
                   multi-labels) — directly consumable by micromodel train
  features/        per-root feature cache (deduplication layout)
"""

import os
import io
import sys
import json
import sqlite3
import datetime
import threading
import webbrowser
from contextlib import closing

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from flask import Flask, jsonify, request, Response

from microBase import (CellDataset, MicroMaxError, read_tiff_channels,
                       build_pipeline, apply)

from .utils import (logger, set_seed, select_device, load_label_csv,
                    resolve_max_value, add_file_logging, copy_config_file,
                    atomic_npz_save, load_npz_cache)
from .dataset import SingleCellDataset
from .backbone import load_model_from_bundle, load_ssl_backbone_from_bundle
from .infer import _forward_pass, _validate_channel_count, _resolve_gt
from .deduplication import (_error, _parse_roots, _root_names,
                        _extract_root_features, _cache_path, _build_space,
                        _resolve_max_value_entry)

# Artifact names, all under save_dir.
DB_NAME = "label.db"                        # single-label project DB
DB_NAME_MULTI = "label_multiple.db"         # multi-label project DB
MODE_FILE = "label_mode.txt"                # last used mode (startup restore)
EXPORT_NAME = "label_export.csv"
EXPORT_NAME_SINGLE = "label_export_single.csv"

# Pre-rename project files, migrated to the new names on startup so
# existing annotation projects survive the rename untouched.
_LEGACY_DBS = [("annotations.db", DB_NAME_MULTI),
               ("annotations_single.db", DB_NAME)]

# Auto-assigned label colors (cycled in registration order).
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#008080", "#f032e6", "#9a6324", "#469990", "#800000",
           "#000075", "#808000", "#e6c229", "#a9a9a9"]

# Composite colors for multi-channel display (channel 1..8, additive RGB).
CHANNEL_COLORS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.6, 1.0),
                  (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0),
                  (1.0, 0.5, 0.0), (1.0, 1.0, 1.0)]

# Decision state codes (cell_labels.state and the API).
STATE_POS = 1
STATE_NEG = 0


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------------
# Annotation storage — append-friendly SQLite
# ----------------------------------------------------------------------------

class AnnotationDB:
    """Thin SQLite wrapper for the annotation project.

    Tables:
      labels          label registry with a user-controlled display order
                      (drag in the UI); deletion is explicit and UI-
                      confirmed and removes the label's decisions — the
                      append-only log keeps the history
      meta            project-level key/values (e.g. the model bundle the
                      annotations were made with, for consistency checks)
      cells           one row per known cell (normcase absolute filepath is
                      the identity); registered on every startup, never
                      removed, so cells from roots that dropped out of the
                      config stay annotated and exportable
      cell_labels     CURRENT decision per (cell, label): state 1 = positive,
                      0 = explicit negative; re-annotating upserts this row
      annotation_log  append-only history of every decision (including the
                      cleared state NULL) — multiple annotation passes
                      accumulate here and are never overwritten
      sources         per-source-root display settings (channels/layout/
                      max_value) as they were when the root was last
                      configured — legacy cells stay displayable even when
                      their root leaves the config
    """

    def __init__(self, path):
        self.path = path
        with closing(self._connect()) as conn, conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta(
                key   TEXT PRIMARY KEY,
                value TEXT);
            CREATE TABLE IF NOT EXISTS labels(
                label_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL UNIQUE,
                color      TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL);
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
                PRIMARY KEY (cell_id, label_id));
            CREATE TABLE IF NOT EXISTS annotation_log(
                log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT NOT NULL,
                session_id INTEGER,
                cell_id    INTEGER NOT NULL,
                label_id   INTEGER NOT NULL,
                state      INTEGER);
            """)

    def _connect(self):
        # A fresh connection per call keeps the Flask threads safe without
        # locking; SQLite serializes the (rare) writes itself.
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
            _error("label name must not be empty")
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
        """All labels in display order, with positive/negative counts."""
        with closing(self._connect()) as conn, conn:
            rows = conn.execute("""
                SELECT l.label_id, l.name, l.color,
                       COALESCE(SUM(cl.state = 1), 0),
                       COALESCE(SUM(cl.state = 0), 0)
                FROM labels l
                LEFT JOIN cell_labels cl ON cl.label_id = l.label_id
                GROUP BY l.label_id
                ORDER BY l.sort_order, l.label_id""").fetchall()
        return [{"label_id": r[0], "name": r[1], "color": r[2],
                 "n_pos": int(r[3]), "n_neg": int(r[4])} for r in rows]

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

    def set_label(self, cell_id, label_id, state, session_id):
        """Upsert one decision; state None clears it. Always logged."""
        ts = _now()
        with closing(self._connect()) as conn, conn:
            if state is None:
                conn.execute(
                    "DELETE FROM cell_labels WHERE cell_id = ? AND "
                    "label_id = ?", (cell_id, label_id))
            else:
                conn.execute(
                    "INSERT INTO cell_labels(cell_id, label_id, state, "
                    "updated_at, session_id) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(cell_id, label_id) DO UPDATE SET "
                    "state = excluded.state, "
                    "updated_at = excluded.updated_at, "
                    "session_id = excluded.session_id",
                    (cell_id, label_id, int(state), ts, session_id))
            conn.execute(
                "INSERT INTO annotation_log(ts, session_id, cell_id, "
                "label_id, state) VALUES (?,?,?,?,?)",
                (ts, session_id, cell_id, label_id, state))

    def clear_other_positives(self, cell_id, keep_label_id, session_id):
        """Single-label mode: a cell holds at most ONE positive label.

        Removes every other positive decision of the cell (each removal is
        logged, so the history stays complete); explicit negatives are kept.
        Returns the number of cleared labels.
        """
        ts = _now()
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT label_id FROM cell_labels WHERE cell_id = ? AND "
                "state = 1 AND label_id != ?",
                (cell_id, keep_label_id)).fetchall()
            for (lid,) in rows:
                conn.execute(
                    "DELETE FROM cell_labels WHERE cell_id = ? AND "
                    "label_id = ?", (cell_id, lid))
                conn.execute(
                    "INSERT INTO annotation_log(ts, session_id, cell_id, "
                    "label_id, state) VALUES (?,?,?,?,NULL)",
                    (ts, session_id, cell_id, lid))
        return len(rows)

    def stats(self):
        """Global counters for the header bar."""
        with closing(self._connect()) as conn, conn:
            total = conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(DISTINCT cell_id) FROM cell_labels"
            ).fetchone()[0]
        return {"total": int(total), "labeled": int(labeled)}

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


# ----------------------------------------------------------------------------
# Feature extraction — train-bundle variant (the SSL variant is deduplication's)
# ----------------------------------------------------------------------------

def _extract_root_features_cls(entry, meta, model, device, output_dir,
                               bundle_id, seed, dl_cfg):
    """Cache-aware feature + probability extraction for a train bundle.

    Mirrors deduplication._extract_root_features (same cache layout, keyed by
    bundle identity — a bundle file is either SSL or classify, so the two
    never collide) but runs the classification model: row features come from
    model(x)'s second output, per-class probabilities from the first
    (sigmoid for multi-label bundles, softmax otherwise).
    """
    root = entry["path"]
    cell_ds = CellDataset(root, channel_layout=entry["channel_layout"],
                          image_pattern=entry["image_pattern"])
    if len(cell_ds) == 0:
        _error(f"no TIFF files found in {root}")
    _validate_channel_count(len(cell_ds.intensity_colnames),
                            entry["channels"], meta, root)
    md = cell_ds.metadata

    rel_paths = sorted(
        os.path.relpath(p, root).replace("\\", "/") for p in md["path"])
    cpath = _cache_path(output_dir, entry, bundle_id, rel_paths, None, seed)

    if os.path.exists(cpath):
        cached = load_npz_cache(cpath)
        if cached:
            paths = [str(p) for p in cached["paths"]]
            raw_paths = [str(p) for p in cached["paths_raw"]]
            feats = cached["feats"]
            probs = cached["probs"]
            logger.info("Feature cache hit: %s (%d cells, dim %d)",
                        os.path.basename(cpath), len(paths), feats.shape[1])
            return paths, raw_paths, feats, probs

    required_meta = ("augmentation_infer", "normalize_method",
                     "normalize_with_masking", "clip_low", "clip_high",
                     "normalize_fixed_reference")
    missing = [k for k in required_meta if k not in meta]
    if missing:
        _error(f"bundle meta missing required keys: {missing}")
    dataset = SingleCellDataset(
        [(cell_ds, int(i)) for i in np.arange(len(md))],
        {"_": 0}, ["_"] * len(md),
        channels=entry["channels"],
        augmentation_spec=meta["augmentation_infer"],
        normalize_method=meta["normalize_method"],
        clip_low=meta["clip_low"], clip_high=meta["clip_high"],
        with_masking=meta["normalize_with_masking"],
        fixed_reference=bool(meta["normalize_fixed_reference"]),
        max_value=_resolve_max_value_entry(entry))

    num_workers = dl_cfg.get("num_workers", 4)
    loader_kwargs = dict(batch_size=dl_cfg.get("batch_size", 128),
                         shuffle=False,
                         num_workers=num_workers,
                         persistent_workers=dl_cfg.get(
                             "persistent_workers", True) and num_workers > 0)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = dl_cfg.get("prefetch_factor", 2)
    loader = DataLoader(dataset, **loader_kwargs)

    logger.info("Extracting train-bundle features for %s (%d cells) ...",
                root, len(md))
    all_logits, all_feats = _forward_pass(loader, model, device, True, True,
                                          write_pred_class=True, pool_fn=None)
    feats = torch.cat(all_feats, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0)
    # Multi-label bundles train with BCE -> independent sigmoids; everything
    # else is a softmax distribution (same convention as infer).
    if meta.get("loss") == "bce":
        probs = torch.sigmoid(logits).numpy()
    else:
        probs = torch.softmax(logits, dim=1).numpy()

    raw_paths = [os.path.abspath(md.iloc[int(i)]["path"]) for i in
                 range(len(md))]
    paths = [os.path.normcase(p) for p in raw_paths]

    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    atomic_npz_save(cpath, paths=np.array(paths),
                    paths_raw=np.array(raw_paths), feats=feats, probs=probs)
    logger.info("Cached features to %s", cpath)
    return paths, raw_paths, feats, probs


# ----------------------------------------------------------------------------
# Suggestion engine — kNN scoring over the whitened-PCA space
# ----------------------------------------------------------------------------

def _diverse_order(W, m, seed):
    """Greedy farthest-point ordering: m row indices spread over the space.

    Starting from a seeded random row, each step appends the row with the
    LOWEST maximum cosine similarity to the already chosen set — the classic
    coverage-maximizing batch for the cold start, where every labeled cell
    teaches the kNN something new.
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    m = int(min(m, n))
    chosen = [int(rng.integers(n))]
    max_sim = W @ W[chosen[0]]
    for _ in range(m - 1):
        nxt = int(np.argmin(max_sim))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, W @ W[nxt])
    return chosen


class SuggestEngine:
    """Per-label nearest-exemplar scores in the L2-normalized space.

    For label L with positive rows P and explicit-negative rows G:
      pos_part(x) = mean of the top-k cosine similarities from x to P
      neg_part(x) = mean of the top-k cosine similarities from x to G
      score       = pos_part - neg_weight * neg_part
    k = min(knn_k, |P|); the default knn_k=1 makes this a pure
    nearest-exemplar score — with few, mutually-diverse positives a top-k
    MEAN would dilute the one truly similar exemplar (measured: top-4 mean
    ~0.01 while the nearest same-type neighbor sits at ~0.8), so the max is
    the right primary signal and larger k only smooths once many exemplars
    accumulate. A suggestion fires when pos_part >= threshold AND
    pos_part > neg_part (negatives act as a veto and a ranking term).

    Score arrays are cached per label and invalidated wholesale by bump()
    after every annotation write (recomputation is one small matmul).
    """

    def __init__(self, W, knn_k, neg_weight):
        self.W = W
        self.n = W.shape[0]
        self.knn_k = int(knn_k)
        self.neg_weight = float(neg_weight)
        self._cache = {}       # label_id -> (version, (npos, nneg), parts)
        self.version = 0
        self._lock = threading.Lock()

    def bump(self):
        """Invalidate all cached scores (call after any annotation write)."""
        with self._lock:
            self.version += 1
            self._cache.clear()

    @staticmethod
    def _topk_mean(sims, k):
        """Mean of the k largest similarities per row (vectorized)."""
        if sims.shape[1] > k:
            part = np.partition(sims, -k, axis=1)[:, -k:]
        else:
            part = sims
        return part.mean(axis=1)

    def parts(self, label_id, pos_rows, neg_rows):
        """(pos_part, neg_part) score arrays over ALL cells for one label."""
        with self._lock:
            hit = self._cache.get(label_id)
            if hit is not None and hit[0] == self.version and \
                    hit[1] == (len(pos_rows), len(neg_rows)):
                return hit[2]

        pos_part = np.full(self.n, -1.0, dtype=np.float32)
        neg_part = np.zeros(self.n, dtype=np.float32)
        if pos_rows:
            k = min(self.knn_k, len(pos_rows))
            sims = self.W @ self.W[pos_rows].T
            pos_part = self._topk_mean(sims, k)
        if neg_rows:
            k = min(self.knn_k, len(neg_rows))
            sims = self.W @ self.W[neg_rows].T
            neg_part = self._topk_mean(sims, k)
        with self._lock:
            self._cache[label_id] = (self.version,
                                     (len(pos_rows), len(neg_rows)),
                                     (pos_part, neg_part))
        return pos_part, neg_part

    def neighbors(self, row, k):
        """Top-k most similar cells of one row (excluding itself)."""
        sims = self.W @ self.W[row]
        sims[row] = -2.0  # never return the query cell itself
        k = int(min(k, self.n - 1))
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]


# ----------------------------------------------------------------------------
# Image rendering — TIFF crop -> percentile-normalized PNG
# ----------------------------------------------------------------------------

def _render_png(img_hwc, max_px):
    """Percentile-normalize a float [0,1] HWC crop and encode a PNG.

    Per-channel robust range (0.1% / 99.9%) computed over the NONZERO
    pixels — cropped cells have exact-zero background, and including it
    would wash the cell out. One channel renders grayscale; several
    channels are additively composited with CHANNEL_COLORS.
    """
    from PIL import Image

    h, w, c = img_hwc.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(c):
        ch = img_hwc[:, :, i]
        vals = ch[ch > 0]
        if vals.size:
            lo, hi = np.percentile(vals, [0.1, 99.9])
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        if c == 1:
            canvas[..., 0] = canvas[..., 1] = canvas[..., 2] = norm
        else:
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            canvas += norm[..., None] * np.asarray(color, dtype=np.float32)
    u8 = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)

    img = Image.fromarray(u8)
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# The server
# ----------------------------------------------------------------------------

class LabelServer:
    """Flask app wiring the DB, the suggest engine and the embedded UI."""

    def __init__(self, config, config_path=None, port=5000,
                 open_browser=True):
        self.config = config
        self.config_path = config_path
        self.port = port
        self.open_browser = open_browser

        # ---- config -------------------------------------------------------
        save_dir = config.get("save_dir")
        if not save_dir:
            _error("config 'save_dir' is required (the annotation project "
                    "directory holding the label DB)")
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        data_cfg = config.get("data", {})
        if "root" not in data_cfg:
            _error("config 'data.root' is required")
        self.roots = _parse_roots(data_cfg)
        for entry, name in zip(self.roots,
                               _root_names([e["path"] for e in self.roots])):
            entry["name"] = name
        # Global max_value default (per-root may override); validated the
        # same way as deduplication.
        if data_cfg.get("max_value") is not None:
            resolve_max_value(data_cfg)
        self.preset_from_dir = bool(data_cfg.get("label_from_dir", False))

        rec = config.get("recommend", {})
        self.knn_k = int(rec.get("knn_k", 1))
        self.neg_weight = float(rec.get("neg_weight", 0.5))
        self.auto_threshold = float(rec.get("auto_threshold", 0.75))
        self.uncertain_band = float(rec.get("uncertain_band", 0.15))
        self.diverse_size = int(rec.get("diverse_size", 48))
        self.neighbors_k = int(rec.get("neighbors", 24))
        self.seed = int(config.get("seed", 42))
        self.pca_components = int(
            config.get("space", {}).get("pca_components", 50))
        self.dl_cfg = config.get("dataloader", {})

        # ---- runtime state (filled by start()) ----------------------------
        self.db = None
        self.session_id = None
        self.cells = []            # full DB cell table, ordered by cell_id
        self.cell_by_fp = {}       # normcase filepath -> cell dict
        self.cid_row = {}          # cell_id -> W row (cells with features)
        self.row_cid = {}          # W row -> cell_id (reverse)
        self.cid_cell = {}         # cell_id -> cell dict
        self.engine = None         # SuggestEngine, or None (no model)
        self.aug_infer = None      # bundle inference pipeline (square display)
        self.diverse_cids = []     # cold-start cell_ids (model only)
        self.prob_matrix = None    # (n_feature_rows, n_classes), classify only
        self.class_names = []
        self.class_index = {}      # class name -> prob column
        self.root_entries = {}     # normcase source -> config entry
        self.labels_cache = []     # refreshed on label changes
        self.global_entry = {"channels": data_cfg.get("channels"),
                             "channel_layout": data_cfg.get("channel_layout"),
                             "max_value": data_cfg.get("max_value")}

        self.app = Flask(__name__)
        # Library errors become clean HTTP 400 responses instead of killing
        # the server process (requests handle data problems all the time).
        self.app.register_error_handler(
            MicroMaxError, lambda e: (jsonify({"error": str(e)}), 400))
        self.app.route("/")(self._serve_html)
        self.app.route("/api/state")(self._api_state)
        self.app.route("/api/labels", methods=["POST"])(self._api_add_label)
        self.app.route("/api/label_mode",
                       methods=["POST"])(self._api_label_mode)
        self.app.route("/api/labels_delete",
                       methods=["POST"])(self._api_delete_label)
        self.app.route("/api/labels_reorder",
                       methods=["POST"])(self._api_reorder_labels)
        self.app.route("/api/labels_from_model",
                       methods=["POST"])(self._api_labels_from_model)
        self.app.route("/api/queue")(self._api_queue)
        self.app.route("/api/cell")(self._api_cell)
        self.app.route("/api/annotate",
                       methods=["POST"])(self._api_annotate)
        self.app.route("/api/annotate_batch",
                       methods=["POST"])(self._api_annotate_batch)
        self.app.route("/api/neighbors")(self._api_neighbors)
        self.app.route("/api/image")(self._api_image)
        self.app.route("/api/export", methods=["POST"])(self._api_export)

    # ------------------------------------------------------------------
    # Startup: DB registration, feature extraction, engine build
    # ------------------------------------------------------------------

    def start(self):
        add_file_logging(self.save_dir)
        if self.config_path is not None:
            copy_config_file(self.config_path, self.save_dir)
        # Labeling mode: 'multi' (default) = a cell may hold any number of
        # positive labels; 'single' = mutually exclusive (a new positive
        # clears the cell's other positives — classic classification). The
        # two modes keep FULLY INDEPENDENT databases (label.db vs
        # annotations_single.db) — switching modes switches stores; the
        # pointer file restores whichever mode the last session used.
        self.label_mode = "multi"
        mode_file = os.path.join(self.save_dir, MODE_FILE)
        if os.path.exists(mode_file):
            with open(mode_file, encoding="utf-8") as f:
                self.label_mode = f.read().strip() or "multi"
        if self.label_mode not in ("multi", "single"):
            self.label_mode = "multi"
        self._migrate_legacy_dbs()
        self.db = AnnotationDB(self._db_path(self.label_mode))
        self.db.set_meta("label_mode", self.label_mode)
        self.session_id = self.db.new_session()
        self._refresh_labels()

        # Resume: every write already lives in the DB, so annotations are
        # simply loaded from disk (they are human decisions — valid in any
        # embedding space). Warn when the model bundle changed since the
        # previous session: suggestions would then be computed in a new
        # space while old labels persist, which is fine but worth knowing.
        prev_labels = self.db.list_labels()
        prev_stats = self.db.stats()
        if prev_labels or prev_stats["labeled"]:
            logger.info("Resuming %s-label project: %d labels, %d cells "
                        "already annotated — loaded from %s",
                        self.label_mode, len(prev_labels),
                        prev_stats["labeled"], self._db_path_name())
        model_path = self.config.get("model")
        if model_path:
            norm_model = os.path.normcase(os.path.abspath(model_path))
            prev_model = self.db.get_meta("model")
            if prev_model and prev_model != norm_model:
                logger.warning("Model bundle changed since the previous "
                               "session (%s -> %s); existing annotations "
                               "are kept, but suggestions are now computed "
                               "in the NEW embedding space", prev_model,
                               norm_model)
            self.db.set_meta("model", norm_model)

        # ---- index roots, register cells ---------------------------------
        new_rows = []
        for entry in self.roots:
            root = entry["path"]
            cell_ds = CellDataset(root,
                                  channel_layout=entry["channel_layout"],
                                  image_pattern=entry["image_pattern"])
            if len(cell_ds) == 0:
                _error(f"no TIFF files found in {root}")
            md = cell_ds.metadata
            n_avail = len(cell_ds.intensity_colnames)
            disp_channels = entry["channels"] or list(range(1, n_avail + 1))
            self.db.upsert_source(root, disp_channels,
                                  entry["channel_layout"],
                                  _resolve_max_value_entry(entry), n_avail)
            self.root_entries[os.path.normcase(root)] = entry

            label_map = {}
            if entry["label_csv"]:
                if not os.path.exists(entry["label_csv"]):
                    _error(f"label_csv not found: {entry['label_csv']}")
                label_map = load_label_csv(entry["label_csv"])
            for p in md["path"]:
                raw = os.path.abspath(p)
                fp = os.path.normcase(raw)
                preset = _resolve_gt(
                    label_map, self.preset_from_dir, fp,
                    os.path.dirname(raw).replace("\\", "/"))
                new_rows.append((fp, raw, root, preset))
            logger.info("Root %s: %d cells", root, len(md))

        # Overlapping roots would give one file two identities.
        fps = [r[0] for r in new_rows]
        if len(set(fps)) != len(fps):
            dupes = sorted({fp for fp in fps if fps.count(fp) > 1})
            _error(f"overlapping roots: {len(dupes)} cells appear more than "
                   f"once, e.g. {dupes[:5]}")

        # Kept for mode switches: the OTHER mode's DB re-registers the same
        # cells from here (INSERT OR IGNORE — idempotent).
        self._registration_rows = new_rows
        self.model_path = os.path.abspath(model_path) if model_path else None

        self.cells = self.db.register_cells(new_rows)
        self.cell_by_fp = {c["filepath"]: c for c in self.cells}
        self.cid_cell = {c["cell_id"]: c for c in self.cells}
        # Short display names for the meta line and the source filter: the
        # path relative to the prefix shared by ALL known sources (same idea
        # as deduplication's curated/ folder names). Legacy sources are
        # included so old cells keep a readable name; on a short-name
        # collision the full sanitized path takes over.
        self.source_names = self._build_source_names()
        logger.info("Cell registry: %d cells (%d from this config)",
                    len(self.cells), len(new_rows))

        # ---- model + features (optional) ----------------------------------
        if self.model_path:
            self._build_features(self.model_path)
        else:
            logger.warning("No model configured — manual mode (no "
                           "suggestions, no similarity queues)")

        st = self.db.stats()
        logger.info("Label ready: %d cells, %d labeled, "
                    "http://127.0.0.1:%d", st["total"], st["labeled"],
                    self.port)
        url = f"http://127.0.0.1:{self.port}"
        if self.open_browser:
            webbrowser.open(url)
        self.app.run(host="127.0.0.1", port=self.port, debug=False,
                     threaded=True)

    def _build_source_names(self):
        """Short, collision-free display names for every known source root.

        Each name is the path relative to the prefix shared by ALL known
        sources (e.g. D:\\Model\\sc_dataset -> opencell_single_cell,
        p53_2026-07-10_63x). Display only — the full path remains the
        identity used by the API.
        """
        paths = sorted({c["source"] for c in self.cells})
        try:
            common = os.path.commonpath(paths)
        except ValueError:  # different drives — nothing shared
            common = None
        names, used = {}, set()
        for p in paths:
            if common:
                rel = os.path.relpath(p, common)
                if rel == ".":
                    rel = os.path.basename(p)
            else:
                rel = p
            name = rel.replace("\\", "_").replace("/", "_") \
                      .replace(":", "").replace(" ", "_")
            if name in used:  # collision — fall back to the full path
                name = p.replace(":", "").replace("\\", "_") \
                        .replace("/", "_").replace(" ", "_")
            used.add(name)
            names[p] = name
        return names

    def _build_features(self, model_path):
        """Extract (or load cached) features and build the suggest engine."""
        if not os.path.exists(model_path):
            _error(f"model bundle not found: {model_path}")
        device = select_device()
        set_seed(self.seed)
        logger.info("Loading bundle from %s", model_path)
        bundle = torch.load(model_path, map_location=device,
                            weights_only=False)
        meta = bundle["meta"]
        bundle_id = json.dumps([
            os.path.normcase(model_path),
            os.path.getmtime(model_path),
            os.path.getsize(model_path),
        ])

        feats_list, path_lists = [], []
        probs_by_fp = {}
        is_cls = False
        if "state_dict" in bundle and "num_classes" in meta:
            # Train bundle: model features + per-class probabilities.
            is_cls = True
            model = load_model_from_bundle(bundle, device)
            self.class_names = [str(c) for c in meta.get("class_names", [])]
            self.class_index = {n: i for i, n in enumerate(self.class_names)}
            for entry in self.roots:
                paths, _, feats, probs = _extract_root_features_cls(
                    entry, meta, model, device, self.save_dir, bundle_id,
                    self.seed, self.dl_cfg)
                feats_list.append(feats)
                path_lists.append(paths)
                for p, pv in zip(paths, probs):
                    probs_by_fp[p] = pv
            logger.info("Train bundle: %d classes", len(self.class_names))
        elif "state_dict" in bundle:
            # SSL bundle: teacher-branch features via the deduplication path
            # (identical extraction + cache layout).
            model, feat_dim, pool_fn, meta = \
                load_ssl_backbone_from_bundle(bundle, device)
            logger.info("SSL backbone loaded: feat_dim=%d", feat_dim)
            for entry in self.roots:
                paths, _, feats = _extract_root_features(
                    entry, meta, model, device, pool_fn, self.save_dir,
                    bundle_id, None, self.seed, self.dl_cfg)
                feats_list.append(feats)
                path_lists.append(paths)
        else:
            _error("bundle has no 'state_dict' key (unsupported bundle)")

        # Display preprocessing: the same geometric pipeline the model sees
        # at inference (resize/pad to the square model input — small cells
        # upscale, large ones downscale), applied before the percentile
        # render so every cell shows as a uniform square.
        aug_spec = meta.get("augmentation_infer")
        self.aug_infer = build_pipeline(aug_spec) if aug_spec else None

        feats_all = np.ascontiguousarray(
            np.vstack([f.astype(np.float32, copy=False) for f in feats_list]))
        flat_paths = [p for lst in path_lists for p in lst]
        if len(set(flat_paths)) != len(flat_paths):
            _error("feature paths contain duplicates — overlapping roots?")

        # Align feature rows with the cell registry; cells whose root is not
        # in this run's config simply have no row (no suggestions for them).
        w_rows, prob_rows = [], []
        for flat_row, fp in enumerate(flat_paths):
            cell = self.cell_by_fp.get(fp)
            if cell is None:
                continue
            self.cid_row[cell["cell_id"]] = len(w_rows)
            self.row_cid[len(w_rows)] = cell["cell_id"]
            w_rows.append(flat_row)
            if is_cls:
                prob_rows.append(probs_by_fp[fp])
        if len(w_rows) < len(feats_all):
            logger.info("%d feature rows match the registry (%d extracted)",
                        len(w_rows), len(feats_all))
        if not w_rows:
            _error("no registry cell matches the extracted features")
        W_sub = np.ascontiguousarray(feats_all[w_rows])

        _, W = _build_space(W_sub, self.pca_components, seed=self.seed)
        self.engine = SuggestEngine(W, self.knn_k, self.neg_weight)
        self.diverse_cids = [self.row_cid[r]
                             for r in _diverse_order(W, self.diverse_size,
                                                     self.seed)
                             if r in self.row_cid]
        if is_cls:
            self.prob_matrix = np.ascontiguousarray(
                np.vstack([np.asarray(p, np.float32) for p in prob_rows]))

        # The extractor is no longer needed — display and scoring are
        # numpy/DB only.
        del model, bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Suggest engine ready: %d cells in space, k=%d, "
                    "neg_weight=%.2f", W.shape[0], self.knn_k,
                    self.neg_weight)

    # ------------------------------------------------------------------
    # Label cache (refreshed on every label change)
    # ------------------------------------------------------------------

    def _refresh_labels(self):
        self.labels_cache = self.db.list_labels()

    def _label_ids(self):
        return [lb["label_id"] for lb in self.labels_cache]

    def _label_name(self, label_id):
        for lb in self.labels_cache:
            if lb["label_id"] == label_id:
                return lb["name"]
        return None

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------

    def _label_rows(self, states):
        """Per-label positive/negative W rows from the current states."""
        by_label = {}
        for (cid, lid), st in states.items():
            by_label.setdefault(lid, {1: [], 0: []})[st].append(cid)
        out = {}
        for lid, d in by_label.items():
            out[lid] = (
                [self.cid_row[c] for c in d[1] if c in self.cid_row],
                [self.cid_row[c] for c in d[0] if c in self.cid_row])
        return out

    def _review_items(self, pos_rows, neg_rows):
        """Leave-one-out kNN consistency check over DECIDED cells of a label.

        The active-learning complement to the `uncertain` queue: instead of
        picking new informative cells, flag already-labeled cells whose
        embedding contradicts their own decision (likely mislabels — slips
        of the hand, or suggestions accepted when the label had too few
        exemplars to be trustworthy). A positive is suspicious when its
        nearest OTHER positive (self excluded — the self-similarity of 1.0
        would otherwise mask everything) is FARTHER than its nearest
        explicit negative; symmetrically for negatives. Evaluable requires
        >= 2 own-side and >= 1 opposite-side exemplars — with fewer there
        is nothing to be inconsistent with.

        Returns dicts sorted by suspicion margin descending, positives
        before negatives on ties dropped (only susp > 0 kept):
          {cell_id, state, susp, own_sim, ev_sim, ev_row}
        where ev_* points at the contradicting neighbor for the UI.
        """
        W = self.engine.W
        items = []
        if len(pos_rows) >= 2 and neg_rows:
            PP = W[pos_rows] @ W[pos_rows].T
            np.fill_diagonal(PP, -2.0)
            CM = W[pos_rows] @ W[neg_rows].T
            own, contra = PP.max(axis=1), CM.max(axis=1)
            for i, row in enumerate(pos_rows):
                susp = float(contra[i] - own[i])
                if susp > 0:
                    items.append({"cell_id": self.row_cid[int(row)],
                                  "state": STATE_POS,
                                  "susp": round(susp, 4),
                                  "own_sim": round(float(own[i]), 4),
                                  "ev_sim": round(float(contra[i]), 4),
                                  "ev_row": int(neg_rows[int(np.argmax(CM[i]))])})
        if len(neg_rows) >= 2 and pos_rows:
            GG = W[neg_rows] @ W[neg_rows].T
            np.fill_diagonal(GG, -2.0)
            CM = W[neg_rows] @ W[pos_rows].T
            own, contra = GG.max(axis=1), CM.max(axis=1)
            for i, row in enumerate(neg_rows):
                susp = float(contra[i] - own[i])
                if susp > 0:
                    items.append({"cell_id": self.row_cid[int(row)],
                                  "state": STATE_NEG,
                                  "susp": round(susp, 4),
                                  "own_sim": round(float(own[i]), 4),
                                  "ev_sim": round(float(contra[i]), 4),
                                  "ev_row": int(pos_rows[int(np.argmax(CM[i]))])})
        items.sort(key=lambda it: -it["susp"])
        return items

    def _cell_payload(self, cell, states, label_rows, thr):
        """One cell for the UI: identity + current labels + suggestions."""
        cid = cell["cell_id"]
        labels = {}
        for lid in self._label_ids():
            st = states.get((cid, lid))
            if st is not None:
                labels[str(lid)] = st
        suggest = []
        if self.engine is not None:
            row = self.cid_row.get(cid)
            if row is not None:
                for lb in self.labels_cache:
                    lid = lb["label_id"]
                    if (cid, lid) in states:
                        continue  # already decided — no suggestion needed
                    pos_rows, neg_rows = label_rows.get(lid, ([], []))
                    if not pos_rows:
                        continue
                    pos_part, neg_part = self.engine.parts(
                        lid, pos_rows, neg_rows)
                    p = float(pos_part[row])
                    if p >= thr and p > float(neg_part[row]):
                        suggest.append({"label_id": lid, "score": round(p, 4),
                                        "src": "knn"})
            if self.prob_matrix is not None and row is not None:
                # Second source: the classify head's own probability for a
                # label sharing a class name.
                pvec = self.prob_matrix[row]
                for ci, name in enumerate(self.class_names):
                    p = float(pvec[ci])
                    if p < thr:
                        continue
                    lid = next((lb["label_id"] for lb in self.labels_cache
                                if lb["name"] == name), None)
                    if lid is None or str(lid) in labels:
                        continue
                    existing = next((s for s in suggest
                                     if s["label_id"] == lid), None)
                    if existing is not None:
                        if p > existing["score"]:
                            existing.update({"score": round(p, 4),
                                             "src": "model"})
                    else:
                        suggest.append({"label_id": lid,
                                        "score": round(p, 4), "src": "model"})
        suggest.sort(key=lambda s: -s["score"])
        if self.label_mode == "single":
            # Mutually exclusive classes: suggest only the single best
            # candidate — a cell can end up with at most one positive.
            suggest = suggest[:1]
        return {"filepath": cell["filepath"], "raw_path": cell["raw_path"],
                "filename": os.path.basename(cell["raw_path"]),
                "source": cell["source"],
                "source_name": self.source_names.get(cell["source"],
                                                     cell["source"]),
                "preset": cell["preset"],
                "labels": labels, "suggest": suggest}

    # ------------------------------------------------------------------
    # API endpoints
    # ------------------------------------------------------------------

    def _serve_html(self):
        return self._render_html()

    def _api_state(self):
        st = self.db.stats()
        return jsonify({
            "labels": self.labels_cache,
            "sources": [{"path": p, "name": self.source_names.get(p, p)}
                        for p in sorted({c["source"] for c in self.cells})],
            "total": st["total"],
            "labeled": st["labeled"],
            "has_model": self.engine is not None,
            "class_names": self.class_names,
            "threshold": self.auto_threshold,
            "uncertain_band": self.uncertain_band,
            "neighbors_k": self.neighbors_k,
            "label_mode": self.label_mode,
        })

    def _db_path(self, mode):
        """Each labeling mode keeps a fully independent project DB."""
        return os.path.join(
            self.save_dir, DB_NAME if mode == "single" else DB_NAME_MULTI)

    def _migrate_legacy_dbs(self):
        """Rename pre-0.14.1 project DBs to the new file names."""
        for old_name, new_name in _LEGACY_DBS:
            old = os.path.join(self.save_dir, old_name)
            new = os.path.join(self.save_dir, new_name)
            if os.path.exists(old) and not os.path.exists(new):
                os.rename(old, new)
                logger.info("Migrated %s -> %s", old_name, new_name)

    def _db_path_name(self):
        return os.path.basename(self.db.path)

    def _switch_mode(self, mode):
        """Point the project at the other mode's independent DB.

        Labels and decisions never cross between the two stores. Cells are
        re-registered from the config roots (INSERT OR IGNORE, idempotent)
        and features come back from the on-disk cache, so the switch costs
        seconds rather than a re-extraction.
        """
        self.label_mode = mode
        with open(os.path.join(self.save_dir, MODE_FILE), "w",
                  encoding="utf-8") as f:
            f.write(mode)
        self.db = AnnotationDB(self._db_path(mode))
        self.db.set_meta("label_mode", mode)
        self.session_id = self.db.new_session()
        self.cells = self.db.register_cells(self._registration_rows)
        self.cell_by_fp = {c["filepath"]: c for c in self.cells}
        self.cid_cell = {c["cell_id"]: c for c in self.cells}
        self.source_names = self._build_source_names()
        self._refresh_labels()
        if self.model_path:
            self._build_features(self.model_path)
        st = self.db.stats()
        logger.info("Switched to %s-label DB (%s): %d labels, %d cells, "
                    "%d labeled", mode, self._db_path_name(),
                    len(self.labels_cache), st["total"], st["labeled"])

    def _api_label_mode(self):
        """Switch the project labeling mode (each mode has its own DB)."""
        data = request.get_json(force=True)
        mode = data.get("mode")
        if mode not in ("multi", "single"):
            return jsonify({"error": "mode must be 'multi' or 'single'"}), 400
        if mode != self.label_mode:
            self._switch_mode(mode)
        return jsonify({"ok": True, "label_mode": mode})

    def _api_add_label(self):
        data = request.get_json(force=True)
        lb = self.db.add_label(data.get("name"), data.get("color"))
        self._refresh_labels()
        return jsonify(lb)

    def _api_delete_label(self):
        """Delete a label (UI double-confirms); its decisions go with it."""
        data = request.get_json(force=True)
        try:
            lid = int(data.get("label_id"))
        except (TypeError, ValueError):
            return jsonify({"error": f"bad label_id: {data.get('label_id')!r}"}), 400
        removed = self.db.delete_label(lid)
        self._refresh_labels()
        if self.engine is not None:
            self.engine.bump()
        logger.info("Label %s deleted (%d decision rows removed)",
                    lid, removed)
        self._auto_export()
        return jsonify({"ok": True, "removed_annotations": removed})

    def _api_reorder_labels(self):
        """Persist the UI's drag-and-drop label order (full permutation)."""
        data = request.get_json(force=True)
        ids = [int(i) for i in data.get("label_ids", [])]
        known = sorted(lb["label_id"] for lb in self.labels_cache)
        # A partial reorder would leave the remaining labels with stale
        # sort_order values and an ambiguous merged order — require all.
        if sorted(ids) != known:
            return jsonify({"error": "label_ids must be a permutation of "
                                     "the existing labels"}), 400
        self.db.reorder_labels(ids)
        self._refresh_labels()
        return jsonify({"ok": True})

    def _api_labels_from_model(self):
        """Create labels from the classify bundle's class names (idempotent)."""
        if not self.class_names:
            return jsonify({"error": "no class names — the bundle is not a "
                                     "classify bundle"}), 400
        created = []
        for name in self.class_names:
            lb = self.db.add_label(name)
            if lb["created"]:
                created.append(lb)
        self._refresh_labels()
        return jsonify({"created": created})

    def _api_queue(self):
        a = request.args
        mode = a.get("mode", "unlabeled")
        label_id = a.get("label_id", type=int)
        source = a.get("source") or None
        offset = max(0, a.get("offset", 0, type=int) or 0)
        limit = min(500, max(1, a.get("limit", 200, type=int) or 200))
        thr = float(a.get("threshold", self.auto_threshold))

        states = self.db.cell_states()
        touched = {cid for (cid, _lid) in states}  # cells with any decision
        label_rows = self._label_rows(states) \
            if self.engine is not None else {}
        labels_list = self.labels_cache

        cells = [c for c in self.cells
                 if source is None or c["source"] == source]
        review_by_cid = {}  # review mode: cell_id -> suspicion payload

        # ---- candidate selection per mode --------------------------------
        if mode == "all":
            picked = cells

        elif mode == "unlabeled":
            picked = [c for c in cells if c["cell_id"] not in touched]

        elif mode == "diverse":
            if self.engine is None:
                return jsonify({"error": "diverse queue needs a model"}), 400
            by_id = {c["cell_id"]: c for c in cells}
            picked = [by_id[cid] for cid in self.diverse_cids
                      if cid in by_id and cid not in touched]

        elif mode in ("label_top", "uncertain", "review"):
            if self.engine is None:
                return jsonify({"error": f"{mode} queue needs a model"}), 400
            if not label_id or label_id not in self._label_ids():
                return jsonify({"error": f"{mode} queue needs a known "
                                         f"label_id"}), 400
            pos_rows, neg_rows = label_rows.get(label_id, ([], []))
            if mode == "review":
                # Decided cells whose embedding contradicts their label,
                # most suspicious first — the re-labeling half of active
                # learning.
                items = self._review_items(pos_rows, neg_rows)
                by_id = {c["cell_id"]: c for c in cells}
                picked = [by_id[it["cell_id"]] for it in items
                          if it["cell_id"] in by_id]
                review_by_cid = {it["cell_id"]: it for it in items}
            else:
                pos_part, neg_part = self.engine.parts(label_id, pos_rows,
                                                       neg_rows)
                scored = []
                for c in cells:
                    if (c["cell_id"], label_id) in states:
                        continue
                    row = self.cid_row.get(c["cell_id"])
                    if row is None:
                        continue
                    p = float(pos_part[row])
                    if p <= float(neg_part[row]):
                        # Explicit negatives veto this cell (same rule as the
                        # suggestion path) — never queue a vetoed cell.
                        continue
                    scored.append((c, p))
                if mode == "label_top":
                    # Most-confident positives first — the fastest harvest.
                    scored.sort(key=lambda t: -t[1])
                else:
                    # uncertain: only cells inside the configured band around
                    # the decision threshold, nearest boundary first (active
                    # learning). A missing/zero band falls back to all.
                    band = self.uncertain_band
                    if band and band > 0:
                        scored = [t for t in scored
                                  if abs(t[1] - thr) <= band]
                    scored.sort(key=lambda t: abs(t[1] - thr))
                picked = [c for c, _ in scored]

        elif mode == "confirm":
            if self.engine is None:
                return jsonify({"error": "confirm queue needs a model"}), 400
            cands = {}  # cell_id -> best confidence across sources/labels
            for lb in labels_list:
                lid = lb["label_id"]
                pos_rows, neg_rows = label_rows.get(lid, ([], []))
                if pos_rows:
                    pos_part, neg_part = self.engine.parts(lid, pos_rows,
                                                           neg_rows)
                    ok = (pos_part >= thr) & (pos_part > neg_part)
                    for row in np.nonzero(ok)[0]:
                        cid = self.row_cid.get(int(row))
                        if cid is not None and (cid, lid) not in states:
                            cands[cid] = max(cands.get(cid, -1.0),
                                             float(pos_part[row]))
            if self.prob_matrix is not None:
                rows_arr = np.array(
                    [self.cid_row.get(c["cell_id"], -1) for c in cells],
                    dtype=np.int64)
                valid = rows_arr >= 0
                for lb in labels_list:
                    ci = self.class_index.get(lb["name"], -1)
                    if ci < 0:
                        continue
                    col = np.full(len(cells), -1.0, dtype=np.float32)
                    if valid.any():
                        col[valid] = self.prob_matrix[rows_arr[valid], ci]
                    for i in np.nonzero(col >= thr)[0]:
                        cid = cells[int(i)]["cell_id"]
                        if (cid, lb["label_id"]) not in states:
                            cands[cid] = max(cands.get(cid, -1.0),
                                             float(col[i]))
            by_id = {c["cell_id"]: c for c in cells}
            picked = sorted(
                (by_id[cid] for cid in cands if cid in by_id),
                key=lambda c: -cands[c["cell_id"]])
        else:
            return jsonify({"error": f"unknown queue mode: {mode}"}), 400

        total = len(picked)
        page = picked[offset:offset + limit]
        payload = []
        for c in page:
            pl = self._cell_payload(c, states, label_rows, thr)
            it = review_by_cid.get(c["cell_id"])
            if it is not None:
                ev = self.cid_cell.get(self.row_cid.get(it["ev_row"]))
                pl["review"] = {
                    "label_id": label_id, "state": it["state"],
                    "susp": it["susp"], "own_sim": it["own_sim"],
                    "ev_sim": it["ev_sim"],
                    "ev_filename": (os.path.basename(ev["raw_path"])
                                    if ev else None)}
            payload.append(pl)
        return jsonify({"cells": payload, "total": total,
                        "offset": offset, "mode": mode})

    def _api_cell(self):
        """Single-cell payload (for jumps from the neighbor strip)."""
        filepath = os.path.normcase(request.args.get("filepath", ""))
        thr = float(request.args.get("threshold", self.auto_threshold))
        cell = self.cell_by_fp.get(filepath)
        if cell is None:
            return jsonify({"error": f"unknown cell: {filepath}"}), 404
        states = self.db.cell_states()
        label_rows = self._label_rows(states) \
            if self.engine is not None else {}
        return jsonify(self._cell_payload(cell, states, label_rows, thr))

    def _api_annotate(self):
        data = request.get_json(force=True)
        return self._handle_annotate(
            data.get("filepath"), data.get("label_id"), data.get("state"))

    def _api_annotate_batch(self):
        data = request.get_json(force=True)
        ok, failed = 0, 0
        for fp in data.get("filepaths", []):
            resp, code = self._handle_annotate(
                fp, data.get("label_id"), data.get("state"))
            ok += code == 200
            failed += code != 200
        if failed:
            return jsonify({"error": f"{failed} writes failed"}), 500
        return jsonify({"ok": True, "n": ok})

    def _handle_annotate(self, filepath, label_id, state):
        """Shared write path: validate, upsert/clear, log, bump the engine.

        state: 1 = positive, 0 = explicit negative, "clear"/None = remove
        the decision (the append-only log keeps the history either way).
        """
        cell = self.cell_by_fp.get(os.path.normcase(str(filepath)))
        if cell is None:
            return jsonify({"error": f"unknown cell: {filepath}"}), 400
        try:
            lid = int(label_id)
        except (TypeError, ValueError):
            return jsonify({"error": f"bad label_id: {label_id!r}"}), 400
        if self._label_name(lid) is None:
            return jsonify({"error": f"unknown label_id: {label_id}"}), 400
        if state in (None, "clear", "none"):
            db_state = None
        else:
            try:
                s = int(state)
            except (TypeError, ValueError):
                return jsonify({"error": f"bad state: {state!r}"}), 400
            if s == STATE_POS:
                db_state = STATE_POS
            elif s == STATE_NEG:
                db_state = STATE_NEG
            else:
                return jsonify({"error": f"bad state: {state!r}"}), 400
        self.db.set_label(cell["cell_id"], lid, db_state, self.session_id)
        if self.label_mode == "single" and db_state == STATE_POS:
            # Single-label mode is mutually exclusive: the new positive
            # clears the cell's other positives (each logged; negatives
            # stay). The client mirrors this locally.
            self.db.clear_other_positives(cell["cell_id"], lid,
                                          self.session_id)
        if self.engine is not None:
            self.engine.bump()
        # Refresh the label cache so per-label pos/neg counts served by
        # /api/state reflect the write immediately.
        self._refresh_labels()
        self._auto_export()
        st = self.db.stats()
        return jsonify({"ok": True, "labeled": st["labeled"]}), 200

    def _api_neighbors(self):
        a = request.args
        filepath = a.get("filepath", "")
        k = min(64, max(1, a.get("k", self.neighbors_k, type=int) or 24))
        cell = self.cell_by_fp.get(os.path.normcase(filepath))
        if cell is None:
            return jsonify({"error": f"unknown cell: {filepath}"}), 400
        row = self.cid_row.get(cell["cell_id"])
        if row is None:
            return jsonify({"error": "cell has no features (its root is not "
                                     "in this run's config)"}), 400
        states = self.db.cell_states()
        out = []
        for r, sim in self.engine.neighbors(row, k):
            nb = self.cid_cell.get(self.row_cid[r])
            labels = {}
            for (cid, lid), st in states.items():
                if cid == nb["cell_id"]:
                    labels[str(lid)] = st
            out.append({"filepath": nb["filepath"],
                        "raw_path": nb["raw_path"],
                        "filename": os.path.basename(nb["raw_path"]),
                        "sim": round(sim, 4), "labels": labels})
        return jsonify({"neighbors": out})

    def _api_image(self):
        a = request.args
        filepath = os.path.normcase(a.get("filepath", ""))
        max_px = min(1024, max(32, a.get("max_px", 384, type=int) or 384))
        cell = self.cell_by_fp.get(filepath)
        if cell is None:
            return jsonify({"error": f"unknown cell: {filepath}"}), 404
        settings = self.db.source_settings(cell["source"])
        if settings is None:
            # Legacy cell whose root predates the sources table — fall back
            # to the global config defaults.
            settings = {"channels": self.global_entry["channels"] or [1],
                        "channel_layout":
                            self.global_entry["channel_layout"],
                        "max_value": self.global_entry["max_value"] or 65535}
        try:
            img = read_tiff_channels(cell["raw_path"], settings["channels"],
                                     channel_layout=settings["channel_layout"])
            if img.ndim == 2:
                img = img[:, :, None]
            img = img.astype(np.float32) / float(settings["max_value"])
            if self.aug_infer is not None:
                # Model-input preprocessing: every cell renders at the
                # bundle's square input size (small upscaled, large
                # downscaled) — same geometry the model was fed.
                img, _ = apply(self.aug_infer, img, None)
            else:
                # No model: still a uniform square by zero-padding.
                h, w = img.shape[:2]
                s = max(h, w)
                canvas = np.zeros((s, s, img.shape[2]), dtype=np.float32)
                canvas[:h, :w] = img
                img = canvas
            png = _render_png(img, max_px)
        except SystemExit:
            # microBase raises on missing files; keep the server alive.
            logger.exception("image read failed: %s", cell["raw_path"])
            return jsonify({"error": "image read failed"}), 500
        except Exception as e:
            logger.exception("image render failed: %s", cell["raw_path"])
            return jsonify({"error": str(e)}), 500
        # Rendered pixels can change between code versions (preprocessing
        # updates), so always revalidate instead of caching hard — the cost
        # on loopback is negligible.
        resp = Response(png, mimetype="image/png")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    def _auto_export(self):
        """Re-write this mode's label CSV after every decision write — the
        training-ready file is always current, no manual export step."""
        df = self.db.export_frame()
        export_name = (EXPORT_NAME_SINGLE if self.label_mode == "single"
                       else EXPORT_NAME)
        path = os.path.join(self.save_dir, export_name)
        if df is None:
            if os.path.exists(path):
                os.remove(path)
                logger.info("No positive annotations left; removed %s",
                            export_name)
            return
        df.to_csv(path, index=False)
        logger.debug("Auto-exported %d rows -> %s", len(df), export_name)

    def _api_export(self):
        df = self.db.export_frame()
        if df is None:
            return jsonify({"error": "no positive annotations to export"}), 400
        # Each mode exports to its own CSV (single-label rows never carry
        # ';', multi-label rows join with it).
        export_name = (EXPORT_NAME_SINGLE if self.label_mode == "single"
                       else EXPORT_NAME)
        path = os.path.join(self.save_dir, export_name)
        df.to_csv(path, index=False)
        logger.info("Exported %d labeled cells to %s", len(df), path)
        return jsonify({"ok": True, "path": path, "rows": int(len(df))})

    # ------------------------------------------------------------------
    # Embedded UI — vanilla JS single page (same approach as reduction_vis)
    # ------------------------------------------------------------------

    def _render_html(self):
        return _HTML


_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>microModel — Cell Labeling</title>
<style>
:root { --bg:#f7f7f8; --line:#ddd; --acc:#4363d8; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); height:100vh; display:flex; flex-direction:column; font-size:13px; color:#222; }
#topbar { background:#fff; border-bottom:1px solid var(--line); padding:6px 12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
#topbar .brand { font-weight:700; margin-right:4px; }
#topbar select, #topbar input { font-size:13px; padding:2px 4px; }
#stats { color:#555; }
button { font-size:13px; padding:3px 10px; border:1px solid #bbb; background:#fff; border-radius:4px; cursor:pointer; }
button:hover { background:#eee; }
button.primary { background:var(--acc); color:#fff; border-color:var(--acc); }
button:disabled { opacity:.45; cursor:default; }
#main { flex:1; display:flex; min-height:0; }
#sidebar { width:235px; background:#fff; border-right:1px solid var(--line); padding:10px; overflow-y:auto; flex-shrink:0; }
#sidebar h3 { font-size:12px; color:#666; margin:8px 0 4px; text-transform:uppercase; letter-spacing:.4px; }
.lbl-row { display:flex; align-items:center; gap:6px; padding:4px 6px; border-radius:4px; cursor:pointer; margin-bottom:2px; }
.lbl-row:hover { background:#f0f0f2; }
.lbl-row.active { outline:2px solid var(--acc); }
.lbl-row.dragging { opacity:.4; }
.lbl-row.dragover { box-shadow:0 -2px 0 var(--acc); }
.lbl-row .dot { width:12px; height:12px; border-radius:3px; flex-shrink:0; }
.lbl-row .cnt { margin-left:auto; color:#888; font-size:12px; }
.lbl-row .del { display:none; color:#c33; cursor:pointer; font-weight:700; padding:0 4px; border-radius:3px; }
.lbl-row:hover .del { display:inline-block; }
.lbl-row .del.armed { display:inline-block; background:#c33; color:#fff; font-size:11px; font-weight:600; padding:1px 6px; }
#content { flex:1; display:flex; flex-direction:column; min-width:0; }
#banner { padding:6px 12px; background:#fffbe8; border-bottom:1px solid #eee; display:none; align-items:center; gap:8px; }
#view { flex:1; overflow:auto; padding:14px; }
#detail { display:none; max-width:1150px; margin:0 auto; }
#detail .top { display:flex; gap:16px; flex-wrap:wrap; }
#detail .imgbox { background:#fff; border:1px solid var(--line); border-radius:6px; padding:10px; align-self:flex-start; }
#detail img.cell { image-rendering:pixelated; max-width:min(46vw,520px); max-height:56vh; display:block; }
#meta { margin:2px 0; font-size:12px; color:#555; word-break:break-all; }
#review-warn { display:none; margin:8px 0 2px; padding:6px 10px; background:#fff3f3; border:1px solid #f3c1c1; border-radius:4px; color:#8a3b3b; font-size:12px; line-height:1.6; }
#chips { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
.chip { display:inline-flex; align-items:center; gap:6px; border:1.5px solid #ccc; border-radius:16px; padding:3px 12px; cursor:pointer; background:#fff; user-select:none; }
.chip .key { display:inline-block; min-width:16px; height:16px; line-height:16px; text-align:center; border-radius:8px; background:#eee; font-size:11px; color:#555; padding:0 2px; }
.chip .sug { font-size:11px; color:#b8860b; font-weight:600; }
.chip.pos { border-color:var(--c); background:var(--c); color:#fff; }
.chip.pos .key { background:rgba(255,255,255,.3); color:#fff; }
.chip.neg { border-color:#999; background:#e9e9e9; color:#888; text-decoration:line-through; }
#navbar { display:flex; gap:8px; align-items:center; margin-top:10px; flex-wrap:wrap; }
#neighbors { margin-top:10px; display:none; }
#neighbors .strip { display:flex; flex-wrap:wrap; gap:8px; padding:4px 0; }
#neighbors .nb { border:2px solid transparent; border-radius:4px; cursor:pointer; position:relative; flex-shrink:0; }
#neighbors .nb img { width:116px; height:116px; object-fit:contain; background:#fff; display:block; }
#neighbors .nb.sel { border-color:var(--acc); }
#neighbors .nb .sim { position:absolute; bottom:3px; left:3px; font-size:10px; background:rgba(0,0,0,.55); color:#fff; padding:0 4px; border-radius:3px; }
#grid { display:none; grid-template-columns:repeat(auto-fill,minmax(112px,1fr)); gap:8px; }
#grid .cellbox { background:#fff; border:2px solid var(--line); border-radius:4px; padding:4px; cursor:pointer; position:relative; }
#grid .cellbox.sel { border-color:var(--acc); }
#grid .cellbox img { width:100%; height:96px; object-fit:contain; display:block; }
#grid .cellbox .dots { position:absolute; top:6px; right:6px; display:flex; gap:2px; }
#grid .cellbox .dot { width:9px; height:9px; border-radius:50%; }
#grid .cellbox .dot.neg { outline:1.5px solid #999; opacity:.65; }
#help { font-size:11px; color:#777; line-height:1.8; margin-top:12px; border-top:1px solid var(--line); padding-top:8px; }
#toast { position:fixed; bottom:18px; left:50%; transform:translateX(-50%); background:#333; color:#fff; padding:8px 16px; border-radius:6px; display:none; z-index:10; }
#busy { position:fixed; inset:0; background:rgba(255,255,255,.65); display:none; align-items:center; justify-content:center; z-index:99; font-size:15px; color:#333; cursor:wait; }
</style>
</head>
<body>
<div id="topbar">
  <span class="brand">🔬 micromodel label</span>
  <select id="mode">
    <option value="unlabeled" title="Walk through every cell that has no annotation yet.">Unlabeled</option>
    <option value="diverse" title="Farthest-point sampling over the embedding - one cell covers a whole region. Label this batch first in a new project and suggestions start working immediately.">Diverse (cold start)</option>
    <option value="confirm" title="Cells with automatic label suggestions, most confident first. Space = accept all suggestions and move on - the fastest review loop.">Confirm suggestions (auto)</option>
    <option value="label_top" title="Undecided cells ranked by similarity to the label selected on the left - the fastest way to collect positives.">By label (ranked)</option>
    <option value="uncertain" title="Undecided cells whose score sits near the threshold - active learning: labeling these helps a future classifier the most.">Uncertain band (active learning)</option>
    <option value="review" title="Already-labeled cells that look more like the opposite class - likely mislabels, ranked by suspicion. Click a chip to flip or clear.">Review suspicious labels</option>
    <option value="all" title="Every cell (filterable by source).">All</option>
  </select>
  <select id="source"><option value="">All sources</option></select>
  <span id="stats"></span>
  <span style="flex:1"></span>
  <label title="Suggestion threshold (0-1): a label is suggested when the cell's nearest positive exemplar is at least this similar and beats the nearest explicit negative. Lower = more suggestions, higher = fewer but safer. Applies to queues loaded afterwards.">Threshold
    <input id="thr" type="number" step="0.05" min="0" max="1" style="width:58px">
  </label>
  <label title="Checked: after applying labels to selected neighbors, jump to the next cell automatically. Unchecked: stay on this cell - keep the selection and maybe apply another label."><input id="cb-adv" type="checkbox" checked> Auto-advance</label>
  <label title="Multi-label: a cell may hold any number of positive labels. Single-label: mutually exclusive classes - setting a new positive clears the cell's other positives, and only the single best suggestion is shown. Switch anytime; both modes keep fully independent databases, nothing is lost.">Mode
    <select id="label-mode">
      <option value="multi">Multi-label</option>
      <option value="single">Single-label (exclusive)</option>
    </select>
  </label>
</div>
<div id="busy"><div>Switching label mode…</div></div>
<div id="main">
  <div id="sidebar">
    <h3>Labels (click = set the current label)</h3>
    <div id="labels"></div>
    <div style="display:flex;gap:4px;margin-top:6px">
      <input id="new-label" placeholder="New label name" style="flex:1;font-size:13px;padding:3px 6px">
      <button id="btn-add" class="primary">Add</button>
    </div>
    <button id="btn-model-labels" style="display:none;margin-top:6px;width:100%">Create labels from model classes</button>
    <div id="help">
      <b>Shortcuts</b><br>
      1-9 toggle positive · Shift+1-9 negative<br>
      Space next (confirm queue = accept)<br>
      Left/Right prev-next · Z undo · G grid<br>
      Click chip = positive · Shift+click = negative<br>
      Grid view: click to open · Ctrl+click multi-select<br>
      Click the same label again = clear the decision<br>
      Single-label mode: a new positive clears old ones<br>
      Drag label rows to reorder · hover X to delete (confirm twice)<br>
      Apply to neighbors: sidebar label, or current positives<br>
      label_csv is re-written after every click - train anytime
    </div>
  </div>
  <div id="content">
    <div id="banner"></div>
    <div id="view">
      <div id="detail">
        <div class="top">
          <div class="imgbox"><img id="cell-img" class="cell"></div>
          <div style="flex:1;min-width:300px">
            <div id="meta"></div>
            <div id="review-warn"></div>
            <div id="chips"></div>
            <div id="navbar">
              <button id="btn-prev">← Prev</button>
              <button id="btn-next" class="primary">Next ␣</button>
              <button id="btn-accept" style="display:none">✓ Accept all</button>
              <button id="btn-undo">Undo Z</button>
              <span id="pos" style="color:#888"></span>
            </div>
          </div>
        </div>
        <div id="neighbors">
          <div style="display:flex;align-items:center;gap:8px;margin:16px 0 4px">
            <h3 style="font-size:12px;color:#666;flex:1;margin:0;font-weight:600">Similar neighbors (click = select/deselect · double-click = jump · select some, then Apply on the right)</h3>
            <button id="btn-nb-apply" title="The label selected on the left wins; otherwise the current cell's positive labels are applied. With 'Auto-advance' checked, jumps to the next cell afterwards.">Apply label to selected</button>
            <button id="btn-nb-all">Select all</button>
            <button id="btn-nb-none">Clear selection</button>
          </div>
          <div class="strip" id="nb-strip"></div>
        </div>
      </div>
      <div id="grid"></div>
    </div>
  </div>
</div>
<div id="toast"></div>
<script>
"use strict";
const $ = s => document.querySelector(s);
const S = { labels:[], labelById:{}, mode:'unlabeled', labelFilter:null, source:'',
            thr:0.65, nbK:24, queue:[], offset:0, pos:-1, view:'detail',
            sel:new Set(), nbSel:new Set(), nbSelFor:null, dragId:null, queueTotal:0,
            autoAdvance: localStorage.getItem('label_autoadvance') !== '0',
            labelMode:'multi',
            undo:[], hasModel:false, classNames:[], total:0, labeled:0, done:false };
const cur = () => S.queue[S.pos] || null;
const esc = s => (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function toast(m) { const t=$('#toast'); t.textContent=m; t.style.display='block'; clearTimeout(t._h); t._h=setTimeout(()=>t.style.display='none',2400); }
async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(()=>({}));
  if (!r.ok) { toast(j.error || ('HTTP '+r.status)); throw new Error(j.error || r.status); }
  return j;
}
// v = display pipeline version: bump when rendering changes so browsers
// drop their cached renders of the old pipeline.
const IMG_V = 2;
const imgURL = (fp, px) => '/api/image?filepath=' + encodeURIComponent(fp) + '&max_px=' + px + '&v=' + IMG_V;

// ---- state / stats ---------------------------------------------------------
async function refreshStats() {
  const j = await api('/api/state');
  S.labels = j.labels; S.hasModel = j.has_model; S.classNames = j.class_names;
  S.thr = j.threshold; S.nbK = j.neighbors_k; S.total = j.total; S.labeled = j.labeled;
  S.labelById = {}; S.labels.forEach((l,i)=>{ l.idx = i+1; S.labelById[l.label_id] = l; });
  $('#thr').value = S.thr;
  $('#stats').textContent = `${S.labeled} / ${S.total} labeled`;
  $('#btn-model-labels').style.display = S.classNames.length ? 'block' : 'none';
  const src = $('#source'), keep = src.value;
  src.innerHTML = '<option value="">All sources</option>' +
    j.sources.map(s=>`<option value="${esc(s.path)}">${esc(s.name)}</option>`).join('');
  src.value = keep;
  S.labelMode = j.label_mode || 'multi';
  $('#label-mode').value = S.labelMode;
  if (!S.hasModel) ['diverse','confirm','label_top','uncertain','review'].forEach(v=>{
    const o=$('#mode').querySelector(`option[value=${v}]`); if(o){o.disabled=true;}
  });
  renderLabels();
}
function renderLabels() {
  $('#labels').innerHTML = S.labels.map(l =>
    `<div class="lbl-row ${S.labelFilter===l.label_id?'active':''}" draggable="true" data-lid="${l.label_id}">` +
    `<span class="dot" style="background:${l.color}"></span>` +
    `<span style="color:#aaa;font-size:11px">${l.idx}</span><span>${esc(l.name)}</span>` +
    `<span class="cnt">${l.n_pos}/${l.n_neg}</span>` +
    `<span class="del" title="Delete label (confirm twice)">✕</span></div>`).join('')
    || '<div style="color:#999">No labels yet - add one below.</div>';
  document.querySelectorAll('.lbl-row').forEach(el => {
    const lid = +el.dataset.lid;
    el.onclick = () => {
      S.labelFilter = (S.labelFilter === lid) ? null : lid;
      renderLabels();
      // Only label-dependent queues reload; other modes keep the current
      // view AND the neighbor selection (order of selecting cells/labels
      // is free).
      if (['label_top','uncertain','review'].includes(S.mode)) loadQueue();
    };
    // ---- drag to reorder ------------------------------------------------
    el.ondragstart = (ev) => { S.dragId = lid; el.classList.add('dragging');
      ev.dataTransfer.effectAllowed = 'move'; };
    el.ondragend = () => { S.dragId = null; el.classList.remove('dragging'); };
    el.ondragover = (ev) => { ev.preventDefault(); if (S.dragId && S.dragId !== lid) el.classList.add('dragover'); };
    el.ondragleave = () => el.classList.remove('dragover');
    el.ondrop = async (ev) => {
      ev.preventDefault(); el.classList.remove('dragover');
      if (!S.dragId || S.dragId === lid) return;
      const from = S.labels.findIndex(l => l.label_id === S.dragId);
      const to = S.labels.findIndex(l => l.label_id === lid);
      if (from < 0 || to < 0) return;
      const [moved] = S.labels.splice(from, 1);
      S.labels.splice(to, 0, moved);
      S.labels.forEach((l, i) => l.idx = i + 1);   // keyboard numbers follow
      renderLabels(); render();
      await api('/api/labels_reorder', { method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ label_ids: S.labels.map(l => l.label_id) }) });
    };
    // ---- delete with double confirm -------------------------------------
    const del = el.querySelector('.del');
    del.onclick = (ev) => {
      ev.stopPropagation();
      if (del.dataset.armed) { deleteLabel(lid); }
      else {
        del.dataset.armed = '1'; del.textContent = 'Confirm delete'; del.classList.add('armed');
        setTimeout(() => { if (del.isConnected) { del.dataset.armed = '';
          del.textContent = 'X'; del.classList.remove('armed'); } }, 3000);
      }
    };
  });
}
async function deleteLabel(lid) {
  const name = (S.labelById[lid] || {}).name || lid;
  await api('/api/labels_delete', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ label_id: lid }) });
  if (S.labelFilter === lid) S.labelFilter = null;
  S.queue.forEach(c => delete c.labels[lid]);
  toast(`Label "${name}" deleted (with all of its decisions)`);
  await refreshStats();
  if (['label_top','uncertain','review'].includes(S.mode)) loadQueue();
  else render();
}

// ---- queue -----------------------------------------------------------------
function queueParams(extra) {
  const p = new URLSearchParams({ mode: S.mode, threshold: S.thr, ...extra });
  if (S.source) p.set('source', S.source);
  if (['label_top','uncertain','review'].includes(S.mode)) {
    if (!S.labelFilter) return null;
    p.set('label_id', S.labelFilter);
  }
  return p;
}
async function loadQueue() {
  const p = queueParams({ limit: 300, offset: 0 });
  if (!p) { toast('Select a label on the left first'); return; }
  const j = await api('/api/queue?' + p);
  S.queue = j.cells; S.queueTotal = j.total;
  S.offset = 0; S.pos = j.cells.length ? 0 : -1; S.done = false;
  S.sel.clear(); render();
  if (!S.queue.length) showBanner(queueEmptyText());
}
function queueEmptyText() {
  if (S.mode==='diverse') return 'The diverse cold-start batch is fully labeled - switch to "Confirm suggestions" or "By label" to continue.';
  if (S.mode==='confirm') return 'No suggestions yet (each label needs at least a few positives). Label a batch with Diverse / Unlabeled / By label first.';
  if (S.mode==='label_top') return 'The ranked queue for this label is finished.';
  if (S.mode==='uncertain') return 'The uncertain band is empty - no undecided cells near the decision boundary.';
  if (S.mode==='review') return 'No suspicious labels found (need at least 2 positives and 1 negative to evaluate).';
  if (S.mode==='unlabeled') return 'No unlabeled cells left.';
  return 'Queue is empty';
}
function showBanner(html, withButtons) {
  const b = $('#banner');
  if (!html) { b.style.display='none'; return; }
  b.style.display = 'flex';
  b.innerHTML = html;
}
async function extendQueue() {
  const p = queueParams({ limit: 300, offset: S.offset + S.queue.length });
  if (!p) return false;
  const j = await api('/api/queue?' + p);
  if (!j.cells.length) { S.done = true; return false; }
  S.queue = S.queue.concat(j.cells); S.queueTotal = j.total;
  return true;
}

// ---- rendering -------------------------------------------------------------
function render() {
  if (S.view === 'grid') { renderGrid(); }
  else { showBanner(null); renderDetail(); }
}
function renderDetail() {
  const c = cur(); if (!c) return;
  $('#detail').style.display = 'block'; $('#grid').style.display = 'none';
  $('#cell-img').src = imgURL(c.filepath, 440);
  $('#meta').innerHTML = `<b>${esc(c.filename)}</b> · ${esc(c.source_name || c.source)}` +
    (c.preset ? ` · preset: <b>${esc(c.preset)}</b>` : '');
  const rw = $('#review-warn');
  if (c.review) {
    rw.style.display = 'block';
    rw.innerHTML = `⚠ Review: this cell is currently a <b>${c.review.state===1?'positive':'negative'}</b>` +
      ` but looks more like the opposite side - nearest opposite <b>${esc(c.review.ev_filename||'?')}</b> at ` +
      `<b>${c.review.ev_sim.toFixed(2)}</b> vs nearest same-side ${c.review.own_sim.toFixed(2)}` +
      ` (suspicion ${c.review.susp.toFixed(2)}). If it is a mislabel: click the chip to flip, click again to clear.`;
  } else rw.style.display = 'none';
  $('#chips').innerHTML = S.labels.map(l => {
    const st = c.labels[l.label_id];
    const sug = (c.suggest||[]).find(s => s.label_id === l.label_id);
    let cls = 'chip'; if (st === 1) cls += ' pos'; else if (st === 0) cls += ' neg';
    return `<span class="${cls}" style="--c:${l.color}" data-lid="${l.label_id}"` +
      ` title="Click = positive · Shift+click = negative · click again = clear${S.labelMode==='single'?' (single-label: a new positive clears the others)':''}">` +
      `<span class="key">${l.idx<=9?l.idx:'·'}</span>${esc(l.name)}` +
      (sug ? `<span class="sug">★${sug.score.toFixed(2)}${sug.src==='model'?'M':'K'}</span>` : '') +
      '</span>';
  }).join('');
  document.querySelectorAll('#chips .chip').forEach(el => el.onclick = ev =>
    toggleLabel(+el.dataset.lid, ev.shiftKey ? 0 : 1));
  $('#btn-accept').style.display =
    (S.mode==='confirm' && (c.suggest||[]).length) ? 'inline-block' : 'none';
  $('#pos').textContent = `${S.offset + S.pos + 1} / ${S.queueTotal}` + (S.done ? ' (end)' : '');
  loadNeighbors(c);
}
function renderGrid() {
  $('#detail').style.display = 'none'; $('#grid').style.display = 'grid';
  $('#grid').innerHTML = S.queue.map((c,i) =>
    `<div class="cellbox ${S.sel.has(c.filepath)?'sel':''}" data-i="${i}">` +
    `<img loading="lazy" src="${imgURL(c.filepath,112)}">` +
    `<div class="dots">${S.labels.filter(l=>c.labels[l.label_id]!==undefined).map(l=>
      `<span class="dot ${c.labels[l.label_id]===0?'neg':''}" style="background:${l.color}"></span>`).join('')}</div></div>`
  ).join('');
  document.querySelectorAll('#grid .cellbox').forEach(el => el.onclick = ev => {
    const i = +el.dataset.i, c = S.queue[i];
    if (ev.ctrlKey || ev.metaKey) {
      S.sel.has(c.filepath) ? S.sel.delete(c.filepath) : S.sel.add(c.filepath);
      el.classList.toggle('sel'); showSelBar();
    } else { S.pos = i; S.view = 'detail'; render(); }
  });
  showSelBar();
}
function showSelBar() {
  if (S.view !== 'grid' || !S.sel.size) { showBanner(null); return; }
  const opts = S.labels.map(l=>`<option value="${l.label_id}">${esc(l.name)}</option>`).join('');
  showBanner(`${S.sel.size} selected <select id="sel-label">${opts}</select>` +
    ` <button id="sel-pos" class="primary">Set positive</button>` +
    ` <button id="sel-neg">Set negative</button> <button id="sel-clear">Clear selection</button>`);
  $('#sel-pos').onclick = ()=>batchApply(1);
  $('#sel-neg').onclick = ()=>batchApply(0);
  $('#sel-clear').onclick = ()=>{ S.sel.clear(); renderGrid(); };
}
async function batchApply(state) {
  const lid = +($('#sel-label')||{}).value; if (!lid) { toast('Pick a label first'); return; }
  await api('/api/annotate_batch', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ filepaths:[...S.sel], label_id:lid, state }) });
  S.queue.forEach(c => { if (S.sel.has(c.filepath)) c.labels[lid] = state; });
  toast(`Applied to ${S.sel.size} cells`); S.sel.clear(); renderGrid(); refreshStats();
}

// ---- actions ---------------------------------------------------------------
async function toggleLabel(lid, state) {
  const c = cur(); if (!c) return;
  const prev = (lid in c.labels) ? c.labels[lid] : null;
  const next = (prev === state) ? 'clear' : state;   // same state again = clear
  S.undo.push({ fp:c.filepath, lid, prev });
  await api('/api/annotate', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ filepath:c.filepath, label_id:lid, state:next }) });
  if (next === 'clear') delete c.labels[lid]; else c.labels[lid] = next;
  if (S.labelMode === 'single' && next === 1) {
    // Mirror the server's mutual exclusion locally: other positives go.
    Object.keys(c.labels).forEach(k => { if (+k !== lid && c.labels[k] === 1) delete c.labels[k]; });
  }
  c.suggest = (c.suggest||[]).filter(s => s.label_id !== lid);
  renderDetail(); refreshStats();
}
async function undo() {
  const a = S.undo.pop(); if (!a) { toast('Nothing to undo'); return; }
  await api('/api/annotate', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ filepath:a.fp, label_id:a.lid,
                           state: a.prev === null ? 'clear' : a.prev }) });
  const c = S.queue.find(x => x.filepath === a.fp);
  if (c) { if (a.prev === null) delete c.labels[a.lid]; else c.labels[a.lid] = a.prev; }
  render(); refreshStats();
}
async function acceptAll() {
  const c = cur(); if (!c || !(c.suggest||[]).length) return;
  const byLabel = {};
  c.suggest.forEach(s => (byLabel[s.label_id] = byLabel[s.label_id] || []).push(c.filepath));
  for (const [lid, fps] of Object.entries(byLabel)) {
    await api('/api/annotate_batch', { method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ filepaths:fps, label_id:+lid, state:1 }) });
    c.labels[lid] = 1;
  }
  c.suggest = []; renderDetail(); refreshStats(); next();
}
async function next() {
  if (S.view === 'grid') return;
  if (S.pos + 1 < S.queue.length) { S.pos++; renderDetail(); return; }
  toast('Fetching more cells…');
  if (await extendQueue()) { S.pos++; renderDetail(); }
  else showBanner('End of this queue &nbsp; <button onclick="loadQueue()">Reload</button>');
}
function prev() { if (S.pos > 0) { S.pos--; render(); } else toast('Already at the first cell'); }

// ---- neighbors -------------------------------------------------------------
async function loadNeighbors(c) {
  const box = $('#neighbors');
  if (!S.hasModel) { box.style.display = 'none'; return; }
  box.style.display = 'block';
  // Selection survives label switches and re-renders; only moving to a
  // DIFFERENT cell starts a fresh selection.
  if (S.nbSelFor !== c.filepath) { S.nbSel.clear(); S.nbSelFor = c.filepath; }
  const j = await api('/api/neighbors?filepath=' + encodeURIComponent(c.filepath) + '&k=' + S.nbK);
  $('#nb-strip').innerHTML = j.neighbors.map(n =>
    `<div class="nb ${S.nbSel.has(n.filepath)?'sel':''}" data-fp="${esc(n.filepath)}" title="${esc(n.filename)}">` +
    `<img loading="lazy" src="${imgURL(n.filepath,116)}">` +
    `<div class="sim">${n.sim.toFixed(2)}</div></div>`).join('');
  document.querySelectorAll('#nb-strip .nb').forEach(el => {
    el.onclick = () => { const fp = el.dataset.fp;
      S.nbSel.has(fp) ? S.nbSel.delete(fp) : S.nbSel.add(fp);
      el.classList.toggle('sel'); };
    el.ondblclick = () => jumpTo(el.dataset.fp);
  });
}
async function jumpTo(fp) {
  const j = await api('/api/cell?filepath=' + encodeURIComponent(fp) + '&threshold=' + S.thr);
  // Replace the rest of the queue from the next position with the jumped
  // cell so ← returns to where the user came from.
  S.queue = S.queue.slice(0, S.pos + 1).concat([j]);
  S.pos++; renderDetail();
}
$('#btn-nb-all').onclick = () => {
  document.querySelectorAll('#nb-strip .nb').forEach(el => { S.nbSel.add(el.dataset.fp); el.classList.add('sel'); });
};
$('#btn-nb-none').onclick = () => {
  S.nbSel.clear();
  document.querySelectorAll('#nb-strip .nb').forEach(el => el.classList.remove('sel'));
};
$('#btn-nb-apply').onclick = async () => {
  const c = cur(); if (!c) return;
  if (!S.nbSel.size) { toast('Select some neighbors first'); return; }
  // Label choice is order-free: a label selected in the sidebar wins;
  // otherwise the current cell's positive labels are applied.
  const lids = S.labelFilter
    ? [S.labelFilter]
    : Object.entries(c.labels).filter(([,st]) => st === 1).map(([lid]) => +lid);
  if (!lids.length) { toast('Select a label on the left, or give the current cell a positive first'); return; }
  const n = S.nbSel.size;
  for (const lid of lids) {
    await api('/api/annotate_batch', { method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ filepaths:[...S.nbSel], label_id:lid, state:1 }) });
  }
  await refreshStats();
  if (S.autoAdvance) {
    // Applied neighbors are done — move to the next cell (the new cell's
    // neighbor strip starts a fresh selection).
    toast(`Applied ${lids.length} label(s) to ${n} neighbors - next cell`);
    await next();
  } else {
    // Stay on this cell: keep the selection (apply another label to the
    // same cells, or undo) and just refresh the strip's label dots.
    toast(`Applied ${lids.length} label(s) to ${n} neighbors`);
    renderDetail();
  }
};
$('#cb-adv').onchange = () => {
  S.autoAdvance = $('#cb-adv').checked;
  localStorage.setItem('label_autoadvance', S.autoAdvance ? '1' : '0');
};

// ---- top-level actions -----------------------------------------------------
$('#btn-add').onclick = async () => {
  const name = $('#new-label').value.trim(); if (!name) return;
  await api('/api/labels', { method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
  $('#new-label').value = ''; await refreshStats(); toast(`Label "${name}" added`);
};
$('#new-label').addEventListener('keydown', e => { if (e.key === 'Enter') $('#btn-add').onclick(); });
$('#btn-model-labels').onclick = async () => {
  const j = await api('/api/labels_from_model', { method:'POST' });
  await refreshStats(); toast(`Created ${j.created.length} labels`);
};
function setBusy(on) {
  const b = document.querySelector('#busy');
  b.style.display = on ? 'flex' : 'none';
}
function updateModeTip() {
  const o = $('#mode').selectedOptions[0];
  $('#mode').title = o ? o.title : '';
}
$('#mode').onchange = () => { S.mode = $('#mode').value; updateModeTip(); refreshStats().then(loadQueue); };
$('#label-mode').onchange = async () => {
  const mode = $('#label-mode').value;
  setBusy(true);   // the other DB re-registers cells + reloads features
  try {
    await api('/api/label_mode', { method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ mode }) });
  // The other mode runs on its own database: label ids do not cross, so
  // drop every piece of state tied to the previous store and reload.
  S.labelMode = mode;
  S.labelFilter = null;
  S.nbSel.clear(); S.sel.clear(); S.undo = [];
    toast(mode === 'single'
          ? 'Switched to the single-label project (independent label.db)'
          : 'Switched to the multi-label project (independent label_multiple.db)');
    await refreshStats();
    await loadQueue();
  } finally {
    setBusy(false);
  }
};
$('#source').onchange = () => { S.source = $('#source').value; loadQueue(); };
$('#thr').onchange = () => { S.thr = parseFloat($('#thr').value) || S.thr; toast('Threshold applies to queues loaded afterwards'); };
$('#btn-prev').onclick = prev;
$('#btn-next').onclick = next;
$('#btn-accept').onclick = acceptAll;
$('#btn-undo').onclick = undo;

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (S.view === 'grid') { if (e.key==='g'||e.key==='G') { S.view='detail'; render(); } return; }
  const k = e.key;
  if (k >= '1' && k <= '9') { const l = S.labels[+k-1]; if (l) toggleLabel(l.label_id, e.shiftKey ? 0 : 1); }
  else if (k === ' ') { e.preventDefault(); const c = cur();
    if (S.mode === 'confirm' && c && (c.suggest||[]).length) acceptAll(); else next(); }
  else if (k === 'ArrowRight') next();
  else if (k === 'ArrowLeft') prev();
  else if (k === 'z' || k === 'Z') undo();
  else if (k === 'g' || k === 'G') { S.view = 'grid'; render(); }
});

updateModeTip();
$('#cb-adv').checked = S.autoAdvance;
refreshStats().then(loadQueue);
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run_label(config, config_path=None, port=5000, no_browser=False):
    """micromodel label — launch the annotation web server (blocking)."""
    server = LabelServer(config, config_path=config_path, port=port,
                            open_browser=not no_browser)
    server.start()
