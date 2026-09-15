"""The label app's Flask server — API wiring over DB + engines + imaging.

`micromodel label` opens a local web page (Flask, same pattern as
reduction_vis): a GRID BATCH WORKTABLE. Every queue mode renders as one
page of thumbnails (K per page); the user selects images, checks target
labels, and Apply + / Apply − / Remove writes selected label(s) to
selected image(s) in ONE server action (single transaction, one undoable
op). Double-click a thumbnail for a zoomed view where labels can be
toggled on the zoomed cell directly (chips or keys 1-9) and ←/→ walk the
queue. The label registry, the per-cell decisions and the full decision
history live in one SQLite DB PER MODE (`label_multiple.db` multi /
`label_single.db` single — the pointer file `label_mode.txt` restores the
last session's mode; the two stores are fully independent projects while
the decision-independent feature cache / engines are shared) so re-running the
command on the same save_dir — with the same or new data roots, and
adding new labels — never destroys existing annotations.

Model assistance (optional but recommended):
  The config's `model` may be an SSL pretrain bundle OR a train bundle. All
  cells are embedded with the same extraction path as infer/deduplication
  (bundle-meta normalization, teacher backbone for SSL bundles), projected
  into a whitened-PCA + L2-normalized space and cached per root. On top of
  that space the engines provide kNN scores, a classify head's per-class
  probabilities and a per-label trained classifier (see engines.py).

Queue modes driven by the scores (see /api/queue) — four, matching the
Collect -> Auto -> Manage workflow:
  label_top  "Collect": for the selected label, undecided cells ranked by
             score (farthest-point spread while the label has no exemplars
             at all); a seeded Shuffle reshuffles the queue
  label_all  "Manage": ALL labeled cells — union / with / without the
             selected label — most uncertain first, plus a "suspicious"
             ranking that surfaces likely mislabels (review_items)
  unlabeled  cells without any annotation yet (stable order)
  all        everything (with source filter)

Auto-annotate (auto_label config): once a label holds >= min_positives
positives, every positive write runs the auto pass — every undecided cell
scoring >= threshold becomes an AUTO positive, jointly over all eligible
labels so nothing is missed. Each label runs ONCE per crossing (tracked
by labels.auto_fired_at), so "Remove auto" sticks; it resets the marker
and a later write can re-run the pass.

Every write is grouped into one op in the annotation log; /api/undo
reverts the newest op exactly (see db.py).
"""

import json
import logging
import os
import webbrowser

import numpy as np
import torch
from flask import Flask, Response, jsonify, request

from microBase import (CellDataset, MicroMaxError, apply, build_pipeline,
                       read_tiff_channels)

from ..deduplication import (_build_space, _error, _extract_root_features,
                             _parse_roots, _resolve_max_value_entry,
                             _root_names)
from ..utils import (add_file_logging, copy_config_file, load_file_list,
                     load_label_csv, resolve_max_value, select_device,
                     set_seed)
from ..backbone import load_model_from_bundle, load_ssl_backbone_from_bundle
from ..infer import _resolve_gt
from .db import (STATE_NEG, STATE_POS, AnnotationDB, DB_NAME,
                 DB_NAME_SINGLE, migrate_project_db)
from .engines import MLModelEngine, SuggestEngine, _diverse_order, review_items
from .features import _extract_root_features_cls
from .imaging import RENDER_VERSION, RenderCache, _render_png
from .ui import HTML_PAGE

EXPORT_NAME = "label_export.csv"             # multi-label project export
EXPORT_NAME_SINGLE = "label_export_single.csv"   # single-label project export
MODE_FILE = "label_mode.txt"                 # last used mode (startup restore)


class LabelServer:
    """Flask app wiring the DB, the suggest engines and the embedded UI."""

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
        self.file_list = data_cfg.get("file_list") or None
        self.list_paths = self.list_labels = None
        if self.file_list:
            # File-list mode: the CSVs (filepath[, label] columns — e.g.
            # deduplication's curated.csv) ARE the registry; data.file_dir
            # is ignored. Implicit roots are the listed files' distinct parent
            # directories: per-root dict overrides don't exist here, the
            # global data.* defaults hold for every file, and the CSV's own
            # label column (when present) is the preset. The per-root
            # "only" set keeps registration + feature extraction on the
            # listed subset instead of the whole folder.
            if data_cfg.get("file_dir"):
                logging.getLogger(__name__).warning(
                    "data.file_list takes precedence: data.file_dir is ignored")
            self.list_paths, self.list_labels = load_file_list(self.file_list)
            if isinstance(self.file_list, str):
                fl_names = [self.file_list]
            else:
                fl_names = [str(p) for p in self.file_list]
            pseudo = {k: v for k, v in data_cfg.items()
                      if k not in ("file_dir", "label_csv", "file_list")}
            pseudo["file_dir"] = [{"path": d} for d in sorted(
                {os.path.dirname(p) for p in self.list_paths})]
            self.roots = _parse_roots(pseudo)
            for entry in self.roots:
                entry["only"] = {
                    os.path.normcase(p) for p in self.list_paths
                    if os.path.dirname(p) == entry["path"]}
            for entry, name in zip(self.roots,
                                   _root_names([e["path"]
                                                for e in self.roots])):
                entry["name"] = name
            logging.getLogger(__name__).info(
                "File list %s: %d cells in %d folder(s)", fl_names,
                len(self.list_paths), len(self.roots))
        else:
            if "file_dir" not in data_cfg:
                _error("config 'data.file_dir' (or 'data.file_list') is required")
            self.roots = _parse_roots(data_cfg)
            for entry, name in zip(self.roots,
                                   _root_names([e["path"]
                                                for e in self.roots])):
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
        self.diverse_size = int(rec.get("diverse_size", 48))
        self.page_size = int(rec.get("page_size", 100))
        # Model-based (active learning) suggester over the same space; a
        # label engages only once it has enough explicit pos AND neg.
        self.ml_model = str(rec.get("ml_model", "logistic")).lower()
        if self.ml_model not in ("none", "logistic", "random_forest"):
            _error("recommend.ml_model must be one of "
                   "none | logistic | random_forest, "
                   f"got {self.ml_model!r}")
        self.ml_min_pos = int(rec.get("ml_min_pos", 5))
        self.ml_min_neg = int(rec.get("ml_min_neg", 3))
        # Auto-annotate: once a label holds >= min_positives positives, every
        # positive write runs the auto pass — undecided cells scoring >=
        # threshold become AUTO positives (the closed-world workflow labels
        # broadly, then prunes by uncertainty in the Manage view).
        auto_cfg = config.get("auto_label", {})
        self.auto_min_positives = int(auto_cfg.get("min_positives", 20))
        self.auto_apply_threshold = float(auto_cfg.get("threshold", 0.9))
        self.seed = int(config.get("seed", 42))
        self.pca_components = int(
            config.get("space", {}).get("pca_components", 50))
        self.dl_cfg = config.get("dataloader", {})

        # ---- runtime state (filled by start()) ----------------------------
        # Labeling mode: 'multi' (default) = a cell may hold any number of
        # positive labels; 'single' = mutually exclusive (a new positive
        # clears the cell's other positives — classic classification). The
        # two modes keep FULLY INDEPENDENT databases (label.db vs
        # label_single.db) — switching modes switches stores, so the two
        # annotation projects never mix; the pointer file restores
        # whichever mode the last session used. Everything decision-
        # independent (feature cache, embedding space, suggest engines,
        # display pipeline) is built ONCE and shared by both modes.
        self.label_mode = "multi"
        self.db = None
        self.session_id = None
        self.cells = []            # full DB cell table, ordered by cell_id
        self.cell_by_fp = {}       # normcase filepath -> cell dict
        self.cid_cell = {}         # cell_id -> cell dict
        self.cid_row = {}          # cell_id -> W row (cells with features)
        self.row_cid = {}          # W row -> cell_id (reverse)
        self._row_fp = {}          # W row -> normcase filepath (mode-independent)
        self._registration_rows = []   # config rows, replayed on mode switch
        self.engine = None         # SuggestEngine, or None (no model)
        self.ml_engine = None      # MLModelEngine, or None (ml_model: none)
        self.aug_infer = None      # bundle inference pipeline (square display)
        self.diverse_cids = []     # cold-start cell_ids (model only)
        self.prob_matrix = None    # (n_feature_rows, n_classes), classify only
        self.class_names = []
        self.class_index = {}      # class name -> prob column
        self.root_entries = {}     # normcase source -> config entry
        self.labels_cache = []     # refreshed on label changes
        self.render_cache = RenderCache()
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
        self.app.route("/api/annotate",
                       methods=["POST"])(self._api_annotate)
        self.app.route("/api/annotate_batch",
                       methods=["POST"])(self._api_annotate_batch)
        self.app.route("/api/auto_apply",
                       methods=["POST"])(self._api_auto_apply)
        self.app.route("/api/auto_clear",
                       methods=["POST"])(self._api_auto_clear)
        self.app.route("/api/undo", methods=["POST"])(self._api_undo)
        self.app.route("/api/image")(self._api_image)
        self.app.route("/api/export", methods=["POST"])(self._api_export)

    # ------------------------------------------------------------------
    # Startup: DB registration, feature extraction, engine build
    # ------------------------------------------------------------------

    def start(self):
        # One thumbnail = one HTTP request, which floods the terminal with
        # werkzeug access-log lines. Keep only warnings and errors — the
        # server URL is announced by this app's own logger below.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        log = logging.getLogger(__name__)
        add_file_logging(self.save_dir)
        if self.config_path is not None:
            copy_config_file(self.config_path, self.save_dir)

        # Two independent project DBs (multi / single); legacy files are
        # renamed into the new names on first startup. The pointer file
        # restores the last session's mode.
        migrate_project_db(self.save_dir)
        self.label_mode = self._read_mode_file()
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
            log.info("Resuming %s-label project: %d labels, %d cells "
                     "already annotated — loaded from %s",
                     self.label_mode, len(prev_labels),
                     prev_stats["labeled"],
                     os.path.basename(self._db_path(self.label_mode)))
        model_path = self.config.get("model")
        if model_path:
            norm_model = os.path.normcase(os.path.abspath(model_path))
            prev_model = self.db.get_meta("model")
            if prev_model and prev_model != norm_model:
                log.warning("Model bundle changed since the previous "
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

            if entry.get("only") is not None:
                # File-list mode: register exactly the listed subset and
                # take presets from the CSVs' own label column.
                md = md[[os.path.normcase(os.path.abspath(p)) in entry["only"]
                         for p in md["path"]]].reset_index(drop=True)
                if len(md) == 0:
                    _error(f"none of the file-list files exist in {root}")
                label_map = {os.path.normcase(p): l for p, l in
                             zip(self.list_paths, self.list_labels) if l}
            else:
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
            log.info("Root %s: %d cells", root, len(md))

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

        self._rebuild_cell_maps()
        log.info("Cell registry: %d cells (%d from this config)",
                 len(self.cells), len(new_rows))

        # ---- model + features (optional) ----------------------------------
        if self.model_path:
            self._build_features(self.model_path)
        else:
            log.warning("No model configured — manual mode (no "
                        "suggestions, no similarity queues)")

        st = self.db.stats()
        log.info("Label ready: %d cells, %d labeled, %d undecided, "
                 "http://127.0.0.1:%d", st["total"], st["labeled"],
                 st["undecided"], self.port)
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

    # ------------------------------------------------------------------
    # Dual mode: two independent projects (multi / single) in one save_dir
    # ------------------------------------------------------------------

    def _db_path(self, mode):
        """Each labeling mode keeps a fully independent project DB."""
        return os.path.join(
            self.save_dir, DB_NAME if mode == "multi" else DB_NAME_SINGLE)

    def _read_mode_file(self):
        """The last session's mode from the pointer file (default multi)."""
        mode_file = os.path.join(self.save_dir, MODE_FILE)
        if os.path.exists(mode_file):
            with open(mode_file, encoding="utf-8") as f:
                mode = f.read().strip()
            if mode in ("multi", "single"):
                return mode
        return "multi"

    def _rebuild_cell_maps(self):
        """(Re)register cells and rebuild the id maps for the CURRENT mode.

        Each mode's DB has its own cell_id sequence, so a mode switch
        re-registers the config rows into the new store (INSERT OR IGNORE —
        idempotent) and re-derives cell_id <-> feature-row from the
        mode-independent row->filepath table.
        """
        self.cells = self.db.register_cells(self._registration_rows)
        self.cell_by_fp = {c["filepath"]: c for c in self.cells}
        self.cid_cell = {c["cell_id"]: c for c in self.cells}
        self.cid_row, self.row_cid = {}, {}
        for row, fp in self._row_fp.items():
            cell = self.cell_by_fp.get(fp)
            if cell is not None:
                self.cid_row[cell["cell_id"]] = row
                self.row_cid[row] = cell["cell_id"]
        # Short display names for the meta line and the source filter (see
        # _build_source_names): rebuilt here so legacy sources of the other
        # mode's DB keep readable names too.
        self.source_names = self._build_source_names()

    def _switch_mode(self, mode):
        """Point the project at the other mode's independent DB.

        Everything decision-independent is SHARED and stays built: the
        feature cache, the embedding space, the suggest engines and the
        display pipeline. The switch swaps the store, re-registers the
        cells and invalidates the score caches ONCE — they were computed
        from the other mode's decisions, and the cache key (version +
        decision counts) carries no mode, so a stale hit would leak the
        other project's scores. No model reload: the switch is sub-second.
        """
        log = logging.getLogger(__name__)
        self.label_mode = mode
        with open(os.path.join(self.save_dir, MODE_FILE), "w",
                  encoding="utf-8") as f:
            f.write(mode)
        self.db = AnnotationDB(self._db_path(mode))
        self.db.set_meta("label_mode", mode)
        self.session_id = self.db.new_session()
        self._bump_engines()
        self._rebuild_cell_maps()
        self._refresh_labels()
        st = self.db.stats()
        log.info("Switched to %s-label DB (%s): %d labels, %d cells, "
                 "%d labeled", mode,
                 os.path.basename(self._db_path(mode)),
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

    def _build_features(self, model_path):
        """Extract (or load cached) features and build the suggest engines."""
        log = logging.getLogger(__name__)
        if not os.path.exists(model_path):
            _error(f"model bundle not found: {model_path}")
        device = select_device()
        set_seed(self.seed)
        log.info("Loading bundle from %s", model_path)
        bundle = torch.load(model_path, map_location=device,
                            weights_only=False)
        meta = bundle["meta"]
        # Same key convention as deduplication's cache: normcase+abspath so a
        # relative `model:` config path and its absolute form hit the SAME
        # shared features/ cache entry instead of silently re-extracting.
        bundle_id = json.dumps([
            os.path.normcase(os.path.abspath(model_path)),
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
                    self.seed, self.dl_cfg, only=entry.get("only"))
                feats_list.append(feats)
                path_lists.append(paths)
                for p, pv in zip(paths, probs):
                    probs_by_fp[p] = pv
            log.info("Train bundle: %d classes", len(self.class_names))
        elif "state_dict" in bundle:
            # SSL bundle: teacher-branch features via the deduplication path
            # (identical extraction + cache layout).
            model, feat_dim, meta = \
                load_ssl_backbone_from_bundle(bundle, device)
            log.info("SSL backbone loaded: feat_dim=%d", feat_dim)
            # The readout config rides in the cache key (pooling changes the
            # feature space — stale caches must never be served).
            bundle_id = json.dumps(json.loads(bundle_id)
                                   + [model.source, model.mask_weighted])
            for entry in self.roots:
                paths, _, feats = _extract_root_features(
                    entry, meta, model, device, self.save_dir,
                    bundle_id, None, self.seed, self.dl_cfg,
                    only=entry.get("only"))
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

        # Row -> filepath table: decision-independent and SHARED by both
        # modes (each mode's DB re-derives its own cell_id maps from it in
        # _rebuild_cell_maps). Cells whose root is not in this run's config
        # simply have no row (no suggestions for them).
        self._row_fp = {}
        w_rows, prob_rows = [], []
        for flat_row, fp in enumerate(flat_paths):
            if self.cell_by_fp.get(fp) is None:
                continue
            self._row_fp[len(w_rows)] = fp
            w_rows.append(flat_row)
            if is_cls:
                prob_rows.append(probs_by_fp[fp])
        if len(self._row_fp) < len(feats_all):
            log.info("%d feature rows match the registry (%d extracted)",
                     len(self._row_fp), len(feats_all))
        if not self._row_fp:
            _error("no registry cell matches the extracted features")
        W_sub = np.ascontiguousarray(feats_all[w_rows])

        _, W = _build_space(W_sub, self.pca_components, seed=self.seed)
        self.engine = SuggestEngine(W, self.knn_k, self.neg_weight)
        self.ml_engine = None if self.ml_model == "none" else MLModelEngine(
            W, self.ml_model, self.ml_min_pos, self.ml_min_neg, self.seed)
        self.diverse_cids = [self.row_cid[r]
                             for r in _diverse_order(W, self.diverse_size,
                                                     self.seed)
                             if r in self.row_cid]
        if is_cls:
            self.prob_matrix = np.ascontiguousarray(
                np.vstack([np.asarray(p, np.float32) for p in prob_rows]))

        # The extractor is no longer needed — display and scoring are
        # numpy/DB only. The cell_id maps for the current mode derive from
        # the freshly built row table.
        del model, bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
        self._rebuild_cell_maps()
        log.info("Suggest engine ready: %d cells in space, k=%d, "
                 "neg_weight=%.2f", W.shape[0], self.knn_k, self.neg_weight)

    # ------------------------------------------------------------------
    # Label cache (refreshed on every label change)
    # ------------------------------------------------------------------

    def _refresh_labels(self):
        self.labels_cache = self.db.list_labels()

    def _bump_label(self, label_id):
        """Invalidate the affected label's cached scores after a write."""
        if self.engine is not None:
            self.engine.bump(label_id)
        if self.ml_engine is not None:
            self.ml_engine.bump(label_id)

    def _bump_engines(self):
        """Invalidate every suggestion cache (label deletion, undo, ...)."""
        if self.engine is not None:
            self.engine.bump()
        if self.ml_engine is not None:
            self.ml_engine.bump()

    def _label_scores(self, label_id, pos_rows, neg_rows):
        """Confidence array over ALL cells for one label (None: no data).

        The trained model's probability when the label engaged, else the
        kNN positive-similarity score. With explicit negatives the kNN
        variant keeps the veto — a cell closer to a negative than to any
        positive scores -1 and never auto-applies.
        """
        if not pos_rows:
            return None
        if self.ml_engine is not None:
            probs = self.ml_engine.probs(label_id, pos_rows, neg_rows)
            if probs is not None:
                return probs
        pos_part, neg_part = self.engine.parts(label_id, pos_rows, neg_rows)
        if neg_rows:
            return np.where(pos_part > neg_part, pos_part, -1.0)
        return pos_part

    def _maybe_auto_apply_all(self, force_label=None):
        """Run the auto-annotate pass for every eligible label.

        Eligible = n_pos >= auto_label.min_positives. A forced run (the UI
        button, ``force_label`` set) reruns one label regardless of its
        fired marker; the regular post-write pass runs a label only ONCE
        per crossing (labels.auto_fired_at) so Remove-auto sticks. Every
        still-undecided cell scoring >= auto_label.threshold becomes an
        AUTO positive — jointly over all eligible labels, so a cell gets
        every label it deserves and nothing is missed. Returns the
        {label_name: applied_count} map.
        """
        log = logging.getLogger(__name__)
        if self.engine is None:
            return {}
        states = self.db.cell_states()
        label_rows = self._label_rows(states)
        # Single mode is mutually exclusive: auto-annotating a cell that
        # already holds ANOTHER label's positive would break exclusivity
        # (the old design did exactly that) — restrict candidates to cells
        # with no positive at all. cell_states keys are (cell_id, label_id).
        pos_holders = {cid for (cid, _lid), s in states.items()
                       if s == STATE_POS} \
            if self.label_mode == "single" else set()
        applied = {}
        # ONE op for the whole pass: "the auto run" is a single undoable
        # step no matter how many labels it annotated.
        op_id = self.db.next_op_id() if self.labels_cache else 0
        for lb in self.labels_cache:
            lid = lb["label_id"]
            if lb["n_pos"] < self.auto_min_positives:
                continue
            if force_label is not None:
                if lid != force_label:
                    continue
            elif lb.get("auto_fired"):
                continue
            scores = self._label_scores(lid, *label_rows.get(lid, ([], [])))
            if scores is None:
                continue
            candidates = [
                cid for cid in self.row_cid.values()
                if (cid, lid) not in states
                and cid not in pos_holders
                and float(scores[self.cid_row[cid]]) >=
                self.auto_apply_threshold]
            n = self.db.auto_apply(lid, candidates, self.session_id, op_id)
            self.db.mark_auto_fired(lid)
            if n:
                applied[lb["name"]] = n
                self._bump_label(lid)
                log.info("Auto-apply '%s': %d cells (threshold %.2f, %d "
                         "positives)", lb["name"], n,
                         self.auto_apply_threshold, lb["n_pos"])
        if applied:
            self._refresh_labels()
            self._auto_export()
        return applied

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
            if self.ml_engine is not None and row is not None:
                # Third source: the per-label trained model's probability
                # (only labels past the pos/neg count gates return scores).
                for lb in self.labels_cache:
                    lid = lb["label_id"]
                    if (cid, lid) in states:
                        continue  # already decided — no suggestion needed
                    pos_rows, neg_rows = label_rows.get(lid, ([], []))
                    probs = self.ml_engine.probs(lid, pos_rows, neg_rows)
                    if probs is None:
                        continue
                    p = float(probs[row])
                    if p < 0.5:
                        continue
                    existing = next((s for s in suggest
                                     if s["label_id"] == lid), None)
                    if existing is not None:
                        if p > existing["score"]:
                            existing.update({"score": round(p, 4),
                                             "src": "aml"})
                    else:
                        suggest.append({"label_id": lid,
                                        "score": round(p, 4), "src": "aml"})
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
        return HTML_PAGE

    def _api_state(self):
        st = self.db.stats()
        return jsonify({
            "labels": self.labels_cache,
            "sources": [{"path": p, "name": self.source_names.get(p, p)}
                        for p in sorted({c["source"] for c in self.cells})],
            "total": st["total"],
            "labeled": st["labeled"],
            "undecided": st["undecided"],
            "has_model": self.engine is not None,
            "ml_model": self.ml_model,
            "class_names": self.class_names,
            "threshold": self.auto_threshold,
            "auto_min_positives": self.auto_min_positives,
            "auto_threshold": self.auto_apply_threshold,
            "page_size": self.page_size,
            "label_mode": self.label_mode,
        })

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
        self._bump_label(lid)
        logging.getLogger(__name__).info(
            "Label %s deleted (%d decision rows removed)", lid, removed)
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
        # Manage scopes: with (the label's positives), without (labeled
        # cells missing the label), union (every cell with >= 1 positive).
        scope = a.get("scope", "with")
        # Seeded reshuffle (client Shuffle button): > 0 permutes the queue
        # deterministically so a fresh random sample reaches the page.
        shuffle = a.get("shuffle", 0, type=int) or 0

        states = self.db.cell_states()
        touched = {cid for (cid, _lid) in states}  # cells with any decision
        label_rows = self._label_rows(states) \
            if self.engine is not None else {}
        labels_list = self.labels_cache

        cells = [c for c in self.cells
                 if source is None or c["source"] == source]
        cert_by_cid = {}    # label_all mode: cell_id -> label certainty
        susp_by_cid = {}    # label_all mode: cell_id -> review evidence
        auto_set = None     # label_all mode: cell_ids auto-annotated

        # ---- candidate selection per mode --------------------------------
        if mode == "all":
            picked = cells

        elif mode == "unlabeled":
            picked = [c for c in cells if c["cell_id"] not in touched]

        elif mode == "label_all":
            # Manage view over DECIDED cells (works without a model — pure
            # DB read). scope picks the membership:
            #   with    — cells positive for the selected label (verify it);
            #   without — labeled cells that do NOT carry the selected
            #             label (candidates missing it);
            #   union   — every cell carrying ANY label (the whole pool a
            #             user should re-check, nothing drops out).
            # A selected label rides along in every scope so each cell can
            # carry its certainty for it: the best score any suggestion
            # source assigns (trained model / classify head / kNN
            # nearest-exemplar — max-merge, same rule as suggestions).
            # Lower = the label is less certain on that cell, i.e. an
            # auto-annotation worth re-checking. sort=review replaces the
            # ranking with the leave-one-out consistency check (the most
            # suspicious decisions first, evidence attached).
            if scope not in ("with", "without", "union"):
                scope = "with"
            known = label_id in self._label_ids()
            if scope != "union" and (label_id is None or not known):
                return jsonify({"error": f"{mode} scope {scope!r} needs a "
                                         f"known label_id"}), 400
            lids = self._label_ids()

            def _has_any_pos(c):
                cid = c["cell_id"]
                return any(states.get((cid, l)) == STATE_POS for l in lids)

            if scope == "with":
                picked = [c for c in cells
                          if states.get((c["cell_id"], label_id)) == STATE_POS]
            elif scope == "without":
                picked = [c for c in cells if _has_any_pos(c)
                          and states.get((c["cell_id"], label_id)) != STATE_POS]
            else:
                picked = [c for c in cells if _has_any_pos(c)]
            auto_set = self.db.auto_cells(label_id) \
                if (known and scope == "with") else None
            if auto_set and a.get("auto_only") == "1":
                picked = [c for c in picked if c["cell_id"] in auto_set]

            review_mode = (a.get("sort") == "review" and known
                           and scope == "with" and self.engine is not None)
            if review_mode:
                # Likely mislabels first: the leave-one-out check over this
                # label's decided positives vs its explicit negatives.
                pos_rows, neg_rows = label_rows.get(label_id, ([], []))
                items = review_items(self.engine.W, self.row_cid,
                                     pos_rows, neg_rows)
                order = {}
                for rank, it in enumerate(items):
                    order[it["cell_id"]] = (rank, it)
                picked = [c for c in picked if c["cell_id"] in order]
                picked.sort(key=lambda c: order[c["cell_id"]][0])
                for c in picked:
                    susp_by_cid[c["cell_id"]] = order[c["cell_id"]][1]
            elif self.engine is not None and label_id is not None:
                self._compute_certainties(picked, label_id, label_rows,
                                          cert_by_cid)
                reverse = a.get("sort", "desc") != "asc"
                # Stable two-pass sort: cell_id ascending as the base order,
                # then certainty — ties keep their registration order in
                # BOTH directions (a reverse=True tuple sort would flip
                # ties too).
                picked.sort(key=lambda c: c["cell_id"])
                picked.sort(key=lambda c: cert_by_cid.get(c["cell_id"], -1.0),
                            reverse=reverse)

        elif mode == "label_top":
            # Collect view: undecided cells ranked for the selected label —
            # the fastest way to reach the auto-annotate threshold. Without
            # any exemplar yet (cold start) the queue falls back to the
            # farthest-point spread, so the very first batch still covers
            # the whole space.
            if self.engine is None:
                return jsonify({"error": f"{mode} queue needs a model"}), 400
            if not label_id or label_id not in self._label_ids():
                return jsonify({"error": f"{mode} queue needs a known "
                                         f"label_id"}), 400
            pos_rows, neg_rows = label_rows.get(label_id, ([], []))
            if not pos_rows and not neg_rows:
                by_id = {c["cell_id"]: c for c in cells}
                picked = [by_id[cid] for cid in self.diverse_cids
                          if cid in by_id and (cid, label_id) not in states]
            else:
                pos_part, neg_part = self.engine.parts(label_id, pos_rows,
                                                       neg_rows)
                # When the trained model engages for this label it replaces
                # the kNN ranking outright.
                ml_probs = None
                if self.ml_engine is not None:
                    ml_probs = self.ml_engine.probs(label_id, pos_rows,
                                                    neg_rows)
                scored = []
                for c in cells:
                    if (c["cell_id"], label_id) in states:
                        continue
                    row = self.cid_row.get(c["cell_id"])
                    if row is None:
                        continue
                    if ml_probs is not None:
                        scored.append((c, float(ml_probs[row])))
                        continue
                    p = float(pos_part[row])
                    if p <= float(neg_part[row]):
                        # Explicit negatives veto this cell (same rule as
                        # the suggestion path) — never queue a vetoed cell.
                        continue
                    scored.append((c, p))
                # Two rankings: score descending (default — the fastest
                # harvest) or active-learning uncertainty, i.e. the cells
                # nearest the decision center (0.5 for the trained model,
                # else the kNN threshold) first — the most informative picks
                # when the confident head of the queue is exhausted or all
                # one morphology.
                center = 0.5 if ml_probs is not None else thr
                if a.get("sort", "desc") == "unc":
                    scored.sort(key=lambda t: abs(t[1] - center))
                else:
                    scored.sort(key=lambda t: -t[1])
                picked = [c for c, _ in scored]

        else:
            return jsonify({"error": f"unknown queue mode: {mode}"}), 400

        if shuffle and len(picked) > 1:
            # Seeded reshuffle of the queue (client Shuffle button): a new
            # random sample reaches the page while the same seed keeps the
            # pages stable while flipping.
            rng = np.random.default_rng(shuffle + self.seed)
            picked = [picked[i] for i in rng.permutation(len(picked))]

        total = len(picked)
        page = picked[offset:offset + limit]
        payload = []
        for c in page:
            pl = self._cell_payload(c, states, label_rows, thr)
            if c["cell_id"] in cert_by_cid:
                pl["cert"] = round(cert_by_cid[c["cell_id"]], 4)
            if c["cell_id"] in susp_by_cid:
                it = susp_by_cid[c["cell_id"]]
                ev = self.cid_cell.get(it["ev_cell_id"])
                pl["susp"] = {"susp": it["susp"], "own_sim": it["own_sim"],
                              "ev_sim": it["ev_sim"],
                              "ev_file": ev["raw_path"] if ev else None,
                              "ev_state": it["state"]}
            if auto_set is not None:
                pl["auto"] = c["cell_id"] in auto_set
            payload.append(pl)
        return jsonify({"cells": payload, "total": total,
                        "offset": offset, "mode": mode})

    def _compute_certainties(self, picked, label_id, label_rows, out):
        """Best certainty per picked cell for one label (vectorized).

        certainty = max(trained-model P(positive), classify-head P(class),
                        kNN mean similarity to the label's OTHER positives —
                        self excluded, since in the `with` scope every
                        listed cell is a positive and its self-similarity
                        of 1.0 would mask every real neighbor). One matmul
                        for the whole page's worth of cells instead of one
                        per cell.
        """
        rows = [self.cid_row[c["cell_id"]] for c in picked
                if c["cell_id"] in self.cid_row]
        if not rows:
            return
        W = self.engine.W
        pos_rows, neg_rows = label_rows.get(label_id, ([], []))
        knn_cert = None
        others = pos_rows
        if len(others) > 1:
            SIM = W[rows] @ W[others].T
            # Exclude each cell's similarity to ITSELF (row == pos_row).
            # equal.outer already pairs every row with every positive —
            # (len(rows), len(others)), same shape as SIM.
            self_mask = np.equal.outer(
                np.asarray(rows, dtype=np.int64),
                np.asarray(others, dtype=np.int64))
            if self_mask.any():
                SIM = np.where(self_mask, -2.0, SIM)
            k = max(1, self.engine.knn_k)
            kk = min(k, SIM.shape[1])
            part = np.partition(SIM, -kk, axis=1)[:, -kk:]
            knn_cert = part.mean(axis=1).astype(np.float32)
        elif len(others) == 1:
            sims = W[rows] @ W[others[0]]
            # Single other positive: the same top-1 mean.
            knn_cert = sims.astype(np.float32)

        cert = knn_cert
        if self.ml_engine is not None:
            ml_probs = self.ml_engine.probs(
                label_id, pos_rows, neg_rows)
            if ml_probs is not None:
                ml = ml_probs[rows]
                cert = ml if cert is None else np.maximum(cert, ml)
        if self.prob_matrix is not None:
            ci = self.class_index.get(self._label_name(label_id), -1)
            if ci >= 0:
                head = self.prob_matrix[rows, ci]
                cert = head if cert is None else np.maximum(cert, head)
        if cert is None:
            return
        for c, v in zip((c for c in picked if c["cell_id"] in self.cid_row),
                        cert):
            out[c["cell_id"]] = float(v)

    def _api_annotate(self):
        data = request.get_json(force=True)
        return self._annotate_core([data.get("filepath")],
                                   [data.get("label_id")],
                                   data.get("state"))

    def _api_annotate_batch(self):
        data = request.get_json(force=True)
        # One user action may carry several target labels: "Apply + 3
        # labels on 12 cells" is ONE transaction and ONE undoable op.
        lids = data.get("label_ids")
        if not lids and data.get("label_id") is not None:
            lids = [data.get("label_id")]
        return self._annotate_core(data.get("filepaths", []),
                                   lids or [], data.get("state"))

    def _annotate_core(self, filepaths, label_ids, state):
        """Shared write path for cells × labels (ONE undoable op).

        state: 1 = positive, 0 = explicit negative, "clear"/None = remove
        the decision. The whole batch is one transaction; the engines'
        score caches are invalidated ONLY for the written labels; the
        auto-annotate pass runs afterwards for every eligible label so a
        threshold crossing can never be skipped (the old equality check
        missed batch jumps like 19 -> 22 positives). The export CSV is
        rewritten exactly once per request.
        """
        log = logging.getLogger(__name__)
        if not isinstance(filepaths, list) or not filepaths:
            return jsonify({"error": "filepaths must be a non-empty list"}), 400
        if not isinstance(label_ids, list) or not label_ids:
            return jsonify({"error": "label_ids must be a non-empty list"}), 400
        lids = []
        for lid in label_ids:
            try:
                lid = int(lid)
            except (TypeError, ValueError):
                return jsonify({"error": f"bad label_id: {lid!r}"}), 400
            if self._label_name(lid) is None:
                return jsonify({"error": f"unknown label_id: {lid}"}), 400
            lids.append(lid)
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

        cids = []
        unknown = []
        for fp in filepaths:
            cell = self.cell_by_fp.get(os.path.normcase(str(fp)))
            if cell is None:
                unknown.append(str(fp))
            else:
                cids.append(cell["cell_id"])
        if unknown:
            return jsonify({"error": f"unknown cell(s): {unknown[:3]}"
                                      f"{'...' if len(unknown) > 3 else ''}"}), 400

        op_id = self.db.next_op_id()
        n = self.db.apply_batch(cids, lids, db_state, self.session_id, op_id)
        if self.label_mode == "single" and db_state == STATE_POS:
            # Single-label mode is mutually exclusive: a cell holds at most
            # ONE positive. Keep-set semantics — every positive written by
            # THIS action stays, all other positives of those cells go
            # (each logged under the same op, so one undo reverts the
            # whole action). Negatives are untouched.
            cleared = self.db.clear_other_positives(cids, lids,
                                                    self.session_id, op_id)
            for lid in {lid for _cid, lid in cleared}:
                self._bump_label(lid)
        for lid in lids:
            self._bump_label(lid)
        # Refresh the label cache so per-label pos/neg counts served by
        # /api/state reflect the write immediately.
        self._refresh_labels()
        # Auto-annotate runs after every positive write: any label that has
        # crossed its positive threshold and not yet fired annotates its
        # confident undecided cells (the written labels, or another one that
        # became eligible through this write's state change).
        auto_applied = {}
        if db_state == STATE_POS:
            auto_applied = self._maybe_auto_apply_all()
        self._auto_export()
        st = self.db.stats()
        names = ", ".join(self._label_name(l) for l in lids)
        log.info("Annotate [%s] state=%s on %d cells (op %d)%s",
                 names, db_state, len(cids), op_id,
                 f" auto={auto_applied}" if auto_applied else "")
        return jsonify({"ok": True, "n": n, "labeled": st["labeled"],
                        "auto_applied": sum(auto_applied.values()),
                        "auto_labels": auto_applied}), 200

    def _api_auto_apply(self):
        """Auto-annotate one label (label_id given, forced rerun) or every
        eligible label that has not fired yet. Idempotent: decided cells
        skip."""
        data = request.get_json(force=True, silent=True) or {}
        force = None
        if data.get("label_id") is not None:
            try:
                force = int(data["label_id"])
            except (TypeError, ValueError):
                return jsonify({"error": "bad label_id"}), 400
            if force not in self._label_ids():
                return jsonify({"error": f"unknown label_id: {force}"}), 400
        applied = self._maybe_auto_apply_all(force_label=force)
        st = self.db.stats()
        return jsonify({"ok": True, "applied": applied,
                        "labeled": st["labeled"]})

    def _api_auto_clear(self):
        """Remove every AUTO decision of one label (two-step confirm in UI)."""
        data = request.get_json(force=True, silent=True) or {}
        try:
            lid = int(data.get("label_id"))
        except (TypeError, ValueError):
            return jsonify({"error": "bad label_id"}), 400
        if lid not in self._label_ids():
            return jsonify({"error": f"unknown label_id: {lid}"}), 400
        removed = self.db.clear_auto(lid, self.session_id,
                                     self.db.next_op_id())
        self._bump_label(lid)
        self._refresh_labels()
        self._auto_export()
        logging.getLogger(__name__).info(
            "Auto-clear '%s': %d auto decisions removed",
            self._label_name(lid), removed)
        st = self.db.stats()
        return jsonify({"ok": True, "removed": removed,
                        "labeled": st["labeled"]})

    def _api_undo(self):
        """Revert the newest user action of this session (one op from the
        log). Works for batch applies, single writes and auto runs;
        repeated undos keep walking back through earlier actions. Returns
        what was reverted.
        """
        result = self.db.undo_last_op(self.session_id)
        if result is None:
            return jsonify({"ok": True, "undone": False, "n": 0,
                            "message": "nothing to undo"})
        self._bump_engines()
        self._refresh_labels()
        self._auto_export()
        st = self.db.stats()
        logging.getLogger(__name__).info(
            "Undo op %d: %d decision(s) restored (labels %s)",
            result["op_id"], result["n"], result["labels"])
        return jsonify({"ok": True, "undone": True, "n": result["n"],
                        "labels": result["labels"],
                        "labeled": st["labeled"]})

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

        def _fparam(name, default):
            v = a.get(name, type=float)
            return default if v is None else v

        lo, hi, gamma = _fparam("lo", 0.1), _fparam("hi", 99.9), \
            _fparam("gamma", 1.0)
        key = (cell["raw_path"], os.path.getmtime(cell["raw_path"]),
               tuple(settings["channels"]), settings["channel_layout"],
               float(settings["max_value"]), max_px, float(lo), float(hi),
               float(gamma), self.aug_infer is not None, RENDER_VERSION)
        png = self.render_cache.get(key)
        if png is None:
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
                png = _render_png(img, max_px, lo_pct=lo, hi_pct=hi,
                                  gamma=gamma)
            except Exception as e:
                # Includes the microBase MicroMaxError subclasses (missing
                # files) — keep the server alive.
                logging.getLogger(__name__).exception(
                    "image render failed: %s", cell["raw_path"])
                return jsonify({"error": str(e)}), 500
            self.render_cache.put(key, png)
        return Response(png, mimetype="image/png")

    def _export_path(self):
        """Each mode writes its own training-ready CSV (single-label rows
        never carry ';', multi-label rows join with it)."""
        name = (EXPORT_NAME_SINGLE if self.label_mode == "single"
                else EXPORT_NAME)
        return os.path.join(self.save_dir, name)

    def _auto_export(self):
        """Re-write this mode's label CSV after every decision write — the
        training-ready file is always current, no manual export step."""
        df = self.db.export_frame()
        path = self._export_path()
        if df is None:
            if os.path.exists(path):
                os.remove(path)
                logging.getLogger(__name__).info(
                    "No positive annotations left; removed %s",
                    os.path.basename(path))
            return
        df.to_csv(path, index=False)

    def _api_export(self):
        df = self.db.export_frame()
        if df is None:
            return jsonify({"error": "no positive annotations to export"}), 400
        path = self._export_path()
        df.to_csv(path, index=False)
        logging.getLogger(__name__).info("Exported %d labeled cells to %s",
                                         len(df), path)
        return jsonify({"ok": True, "path": path, "rows": int(len(df))})


# Entry point (imported by cli.main) — keep the historical import path.
def run_label(config, config_path=None, port=5000, no_browser=False):
    """micromodel label — launch the annotation web server (blocking)."""
    server = LabelServer(config, config_path=config_path, port=port,
                         open_browser=not no_browser)
    server.start()
