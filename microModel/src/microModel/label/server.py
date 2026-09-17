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
  that space SuggestEngine scores every cell for one label as

      score = top-k mean similarity to the label's POSITIVE exemplars
              − neg_weight × top-k mean similarity to its EXPLICIT NEGATIVES

  (undecided is never a negative). That single transparent score drives the
  Collect queue order AND the grid badge; writing positives or negatives
  bumps the cached scores, so the ranking always reflects the current
  exemplars. A classify bundle additionally offers its own class
  probabilities as a second suggestion source. When a label holds enough
  explicit exemplars on BOTH sides, the UI's Refresh-model button
  (POST /api/refresh_model) fits a per-label logistic scorer over them —
  negatives then define a real decision boundary instead of only nudging a
  similarity — and that P(positive) becomes the label's primary score
  everywhere. The fit is MANUAL by design: new writes mark the models
  stale (button status) but keep scoring until the user refreshes again.

ONE unified queue (see /api/queue), fully determined by
image source × label × scope radio × sort radio — optionally narrowed to
one Leiden cluster (cluster-assisted bulk labeling, see label/cluster.py)
and/or the classify bundle's argmax prediction:
  undecided  cells without a decision for the label, ranked by the
             label's primary score; seeded Shuffle reshuffles and
             sort=unc ranks the smallest pos/neg margin first (active
             learning); farthest-point spread while the label has no
             positives at all (needs a model)
  with       the label's positives — certainty ranking, or sort=review's
             leave-one-out "suspicious" ranking (review_items)
  without    labeled cells missing the label (candidates)
  neg        the label's explicit negatives (certainty = positive-
             likeness, so descending surfaces likely mislabels first)
  union      every cell carrying any label
Decided scopes are pure DB reads and work without a model.

Every write is grouped into one op in the annotation log; /api/undo
reverts the newest op exactly (see db.py).
"""

import hashlib
import json
import logging
import os
import threading
import webbrowser

import numpy as np
import torch
from flask import Flask, Response, jsonify, request

from microBase import (CellDataset, MicroMaxError, apply, build_pipeline,
                       canonical_directory, read_tiff_channels)

from ..deduplication import (_build_space, _error, _extract_root_features,
                             _parse_roots, _resolve_max_value_entry,
                             _root_names)
from ..utils import (add_file_logging, copy_config_file, load_file_list,
                     load_label_csv, resolve_max_value, select_device,
                     set_seed)
from ..backbone import load_model_from_bundle, load_ssl_backbone_from_bundle
from ..infer import _resolve_gt
from .cluster import build_or_load
from .db import (STATE_NEG, STATE_POS, AnnotationDB, DB_NAME,
                 DB_NAME_SINGLE, migrate_project_db)
from .engines import (MIN_FIT_NEG, MIN_FIT_POS, SuggestEngine,
                      _diverse_order, fit_label_model, review_items)
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
        self.diverse_size = int(rec.get("diverse_size", 48))
        self.page_size = int(rec.get("page_size", 100))
        self.seed = int(config.get("seed", 42))
        self.pca_components = int(
            config.get("space", {}).get("pca_components", 50))
        self.dl_cfg = config.get("dataloader", {})

        # Cluster-assisted bulk labeling: `cluster.target` asks the resolution
        # search for ~N Leiden clusters over the shared space; `cluster.res`
        # clusters once at an explicit resolution instead. Both null (or the
        # section missing) keeps the feature off — behavior identical to a
        # config without the section.
        clu = config.get("cluster") or {}
        self.cluster_target = clu.get("target")
        self.cluster_res = clu.get("res")
        if self.cluster_target is not None:
            self.cluster_target = int(self.cluster_target)
        if self.cluster_res is not None:
            self.cluster_res = float(self.cluster_res)
        if self.cluster_res is not None and self.cluster_target is not None:
            logging.getLogger(__name__).info(
                "cluster.res is set: the explicit resolution overrides "
                "cluster.target (%s)", self.cluster_target)

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
        # In-memory mirror of the cell_labels table (see _get_states) —
        # keeps the per-request hot path free of full-table scans.
        self._states_cache = None
        self._states_lock = threading.Lock()
        # Serializes DB write + mirror update in _annotate_core so two
        # concurrent annotates apply to the DB and the mirror in the SAME
        # order (the mirror can never diverge from the DB).
        self._write_lock = threading.Lock()
        self.cells = []            # full DB cell table, ordered by cell_id
        self.cell_by_fp = {}       # portable identity -> cell dict
        self.cid_cell = {}         # cell_id -> cell dict
        self.cid_row = {}          # cell_id -> W row (cells with features)
        self.row_cid = {}          # W row -> cell_id (reverse)
        self._row_fp = {}          # W row -> portable identity (mode-independent)
        self._registration_rows = []   # config rows, replayed on mode switch
        self.engine = None         # SuggestEngine, or None (no model)
        # Per-label logistic refits (the manual Refresh-model button):
        # label_id -> fitted sklearn model over the shared space. New writes
        # mark them stale (button status) but they keep scoring until the
        # next refresh; a mode switch drops them entirely (the other mode's
        # decisions live in a different DB — its labels need their own fit).
        self.label_models = {}
        self._model_scores = {}    # label_id -> full-W P(positive) array
        self._model_stale = False
        self.aug_infer = None      # bundle inference pipeline (square display)
        self.diverse_cids = []     # cold-start cell_ids (model only)
        self.prob_matrix = None    # (n_feature_rows, n_classes), classify only
        self.class_names = []
        self.class_index = {}      # class name -> prob column
        # Classify-bundle per-row argmax: W row -> predicted class index and
        # its probability (the /api/queue pred_label filter and the prob
        # sort read these instead of scanning prob_matrix per request).
        self._pred_argmax = None       # int array over W rows
        self._pred_maxprob = None      # float32 array over W rows
        # Cluster-assisted bulk labeling (built once with the features,
        # decision-independent, shared by both modes). cluster_memb is the
        # 0-based partition over W rows; ids/medoids are 1-based for the API.
        self.bundle_id = None
        self.cluster_memb = None
        self.cluster_ids = []          # 1-based, ascending
        self.cluster_medoid = {}       # 1-based cluster id -> W row
        self.cluster_res_used = None
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
        self.app.route("/api/clusters")(self._api_clusters)
        self.app.route("/api/annotate",
                       methods=["POST"])(self._api_annotate)
        self.app.route("/api/annotate_batch",
                       methods=["POST"])(self._api_annotate_batch)
        self.app.route("/api/annotate_cluster",
                       methods=["POST"])(self._api_annotate_cluster)
        self.app.route("/api/undo", methods=["POST"])(self._api_undo)
        self.app.route("/api/refresh_model",
                       methods=["POST"])(self._api_refresh_model)
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
            # Stored PORTABLE (same convention as the cell paths) so the
            # project stays valid when the working tree moves.
            model_portable = canonical_directory(os.path.abspath(model_path))
            prev_model = self.db.get_meta("model")
            if prev_model and prev_model != model_portable:
                log.warning("Model bundle changed since the previous "
                            "session (%s -> %s); existing annotations "
                            "are kept, but suggestions are now computed "
                            "in the NEW embedding space", prev_model,
                            model_portable)
            self.db.set_meta("model", model_portable)

        # ---- index roots, register cells ---------------------------------
        # Every stored path is PORTABLE (canonical_directory: CWD-relative
        # forward-slash when under the CWD, absolute fallback) — the label
        # project survives moving the working tree and the dataset together,
        # and label_export.csv is a CWD-relative file list like curated.csv.
        new_rows = []
        for entry in self.roots:
            root = entry["path"]
            root_portable = canonical_directory(root)
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
                raw_portable = canonical_directory(raw)
                fp = os.path.normcase(raw_portable)
                # label_map / _resolve_gt work on absolute normcase keys.
                preset = _resolve_gt(
                    label_map, self.preset_from_dir, os.path.normcase(raw),
                    os.path.dirname(raw).replace("\\", "/"))
                new_rows.append((fp, raw_portable, root_portable, preset))
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
        # The new project store starts empty — drop the previous mode's
        # states mirror so nothing leaks across the switch.
        self._states_cache = None
        # The refit models were trained on the OTHER mode's decisions —
        # drop them instead of leaking scores across projects.
        self.label_models.clear()
        self._model_scores.clear()
        self._model_stale = False
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

        # Final cache identity (SSL bundles extend it with the readout) —
        # also the space identity the cluster cache keys on.
        self.bundle_id = bundle_id
        self.prob_matrix = None
        self._pred_argmax = None
        self._pred_maxprob = None

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

        # Row -> portable-identity table: decision-independent and SHARED by
        # both modes (each mode's DB re-derives its own cell_id maps from it
        # in _rebuild_cell_maps). Cells whose root is not in this run's
        # config simply have no row (no suggestions for them).
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
        self.diverse_cids = [self.row_cid[r]
                             for r in _diverse_order(W, self.diverse_size,
                                                     self.seed)
                             if r in self.row_cid]
        if is_cls:
            self.prob_matrix = np.ascontiguousarray(
                np.vstack([np.asarray(p, np.float32) for p in prob_rows]))
            # Per-row argmax prediction: the pred_label queue filter and the
            # confidence (sort=prob) ranking read these flat arrays.
            self._pred_argmax = np.argmax(self.prob_matrix, axis=1)
            self._pred_maxprob = self.prob_matrix[
                np.arange(len(self._pred_argmax)), self._pred_argmax]

        # The extractor is no longer needed — display and scoring are
        # numpy/DB only. The cell_id maps for the current mode derive from
        # the freshly built row table.
        del model, bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
        self._rebuild_cell_maps()
        log.info("Suggest engine ready: %d cells in space, k=%d, "
                 "neg_weight=%.2f", W.shape[0], self.knn_k, self.neg_weight)
        self._build_clusters()

    def _build_clusters(self):
        """Leiden clusters over the shared space (once, decision-independent).

        Configured via cluster.target / cluster.res; the assignment only
        depends on the embedding (bundle identity + cell list + PCA width),
        so it is cached to <save_dir>/features/ and survives mode switches.
        Without a model there is no space to cluster — the feature logs a
        hint and stays off.
        """
        log = logging.getLogger(__name__)
        if self.cluster_target is None and self.cluster_res is None:
            return
        if self.engine is None:
            log.warning("cluster configured but no model — clustering off")
            return
        row_paths = [self._row_fp[r] for r in range(len(self._row_fp))]
        key = {"bundle": self.bundle_id,
               "cells": hashlib.md5(
                   json.dumps(row_paths).encode("utf-8")).hexdigest(),
               "pca": self.pca_components}
        memb, ids, meds, used = build_or_load(
            self.engine.W, row_paths, self.save_dir, key,
            self.cluster_target, self.cluster_res, self.seed)
        # 1-based display ids in the API/UI (0 would read as "no cluster").
        self.cluster_memb = memb
        self.cluster_ids = [int(i) + 1 for i in ids]
        self.cluster_medoid = {int(i) + 1: int(m) for i, m in zip(ids, meds)}
        self.cluster_res_used = used
        log.info("Clusters ready: %d clusters over %d cells (resolution "
                 "%.3f%s)", len(self.cluster_ids), len(memb), used,
                 f", target {self.cluster_target}"
                 if self.cluster_target is not None else "")

    # ------------------------------------------------------------------
    # Label cache (refreshed on every label change)
    # ------------------------------------------------------------------

    def _refresh_labels(self):
        self.labels_cache = self.db.list_labels()

    def _bump_label(self, label_id):
        """Invalidate the affected label's cached scores after a write."""
        if self.engine is not None:
            self.engine.bump(label_id)

    def _bump_engines(self):
        """Invalidate every suggestion cache (label deletion, undo, ...)."""
        if self.engine is not None:
            self.engine.bump()

    def _mark_models_stale(self):
        """A write landed: refit models no longer match the decisions.

        They keep scoring (they are still the best available ranking) until
        the user clicks Refresh model again — the fit is deliberately
        manual, the staleness is only a status hint.
        """
        self._model_stale = True

    def _model_proba(self, label_id):
        """P(positive) over ALL W rows from the label's fitted logistic.

        Full-array predictions are cached per label: the model only changes
        on an explicit refresh, so the Collect queue and every badge read
        the same array between refreshes. None when the label has no refit
        (it then runs on the kNN exemplar score).
        """
        model = self.label_models.get(label_id)
        if model is None or self.engine is None:
            return None
        if label_id not in self._model_scores:
            self._model_scores[label_id] = model.predict_proba(
                self.engine.W)[:, 1].astype(np.float32)
        return self._model_scores[label_id]

    def _api_refresh_model(self):
        """Fit a per-label logistic scorer for every eligible label.

        Eligible = at least MIN_FIT_POS positives AND MIN_FIT_NEG explicit
        negatives (a logistic needs both classes to mean anything); labels
        below the bar keep the kNN score and are reported as skipped. A
        fresh fit replaces the previous model and clears the stale flag.
        Pure CPU and sub-second per label.
        """
        if self.engine is None:
            return jsonify({"error": "refresh model needs a model bundle"}), 400
        log = logging.getLogger(__name__)
        label_rows = self._label_rows(self._get_states())
        fitted, skipped = [], []
        for lb in self.labels_cache:
            lid = lb["label_id"]
            pos_rows, neg_rows = label_rows.get(lid, ([], []))
            model = fit_label_model(self.engine.W, pos_rows, neg_rows,
                                    self.seed)
            self._model_scores.pop(lid, None)   # recompute on the next read
            if model is None:
                self.label_models.pop(lid, None)
                skipped.append(lb["name"])
            else:
                self.label_models[lid] = model
                fitted.append(lb["name"])
        self._model_stale = False
        if fitted or skipped:
            log.info("Refresh model: fitted %s; skipped %s (< %d positives "
                     "or < %d negatives)", fitted or "none", skipped,
                     MIN_FIT_POS, MIN_FIT_NEG)
        return jsonify({"ok": True, "fitted": fitted, "skipped": skipped,
                        "min_pos": MIN_FIT_POS, "min_neg": MIN_FIT_NEG})

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

    def _get_states(self):
        """A consistent snapshot of the current decisions as
        {(cell_id, label_id): state}, from an in-memory mirror of the
        cell_labels table.

        Queue paging, cluster cards and the header stats read decisions on
        EVERY request — the mirror keeps those reads off the SQLite table.
        It loads lazily and is kept in step by the same write paths that
        touch the DB (_annotate_core mirrors its exact rows); actions that
        revert rows wholesale (undo, label deletion) drop it for a lazy
        reload. Callers get a private COPY so they can iterate freely while
        a concurrent annotate mutates the mirror under the lock.
        """
        with self._states_lock:
            if self._states_cache is None:
                self._states_cache = self.db.cell_states()
            return dict(self._states_cache)

    def _drop_states_cache(self):
        """Invalidate the states mirror (next read reloads from the DB)."""
        with self._states_lock:
            self._states_cache = None

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

    def _score_parts(self, lid, label_rows):
        """(pos_part, neg_part) arrays for one label, or None without positives."""
        pos_rows, neg_rows = label_rows.get(lid, ([], []))
        if not pos_rows:
            return None
        return self.engine.parts(lid, pos_rows, neg_rows)

    def _score_at(self, lid, row, label_rows):
        """The label's primary score for one W row, or None.

        A refreshed (logistic) label scores by its model's P(positive);
        everything else by the transparent exemplar score
        pos_part − neg_weight × neg_part. Either way this is the exact
        number the Collect queue sorts by, so badges and order always agree.
        """
        proba = self._model_proba(lid)
        if proba is not None:
            return float(proba[row])
        parts = self._score_parts(lid, label_rows)
        if parts is None:
            return None
        pos_part, neg_part = parts
        return float(pos_part[row]) - self.neg_weight * float(neg_part[row])

    def _cell_payload(self, cell, states, label_rows):
        """One cell for the UI: identity + current labels + suggestion scores.

        Suggestions carry the same transparent score the Collect queue ranks
        by, for every label that has at least one positive exemplar and is
        not decided on this cell yet. A classify bundle's own class
        probabilities merge in as a second source. Exact negatives lower a
        label's score, so they change the badges and the order too.
        """
        cid = cell["cell_id"]
        labels = {}
        for lid in self._label_ids():
            st = states.get((cid, lid))
            if st is not None:
                labels[str(lid)] = st
        suggest = []
        row = self.cid_row.get(cid) if self.engine is not None else None
        if row is not None:
            for lb in self.labels_cache:
                lid = lb["label_id"]
                if (cid, lid) in states:
                    continue  # already decided — no suggestion needed
                score = self._score_at(lid, row, label_rows)
                if score is None or score <= 0:
                    continue
                suggest.append({"label_id": lid, "score": round(score, 4)})
            if self.prob_matrix is not None:
                # Second source: the classify head's own probability for a
                # label sharing a class name.
                pvec = self.prob_matrix[row]
                for ci, name in enumerate(self.class_names):
                    p = float(pvec[ci])
                    if p < 0.5:
                        continue
                    lid = next((lb["label_id"] for lb in self.labels_cache
                                if lb["name"] == name), None)
                    if lid is None or str(lid) in labels:
                        continue
                    existing = next((s for s in suggest
                                     if s["label_id"] == lid), None)
                    if existing is not None:
                        if p > existing["score"]:
                            existing.update({"score": round(p, 4)})
                    else:
                        suggest.append({"label_id": lid,
                                        "score": round(p, 4)})
        suggest.sort(key=lambda s: -s["score"])
        if self.label_mode == "single":
            # Mutually exclusive classes: suggest only the single best
            # candidate — a cell can end up with at most one positive.
            suggest = suggest[:1]
        # The classify bundle's own argmax prediction (queue filter value +
        # tooltip info); None with an SSL bundle or without features.
        pred = None
        if row is not None and self._pred_argmax is not None:
            ci = int(self._pred_argmax[row])
            pred = {"class": (self.class_names[ci]
                              if ci < len(self.class_names) else str(ci)),
                    "prob": round(float(self.prob_matrix[row, ci]), 4)}
        return {"filepath": cell["filepath"], "raw_path": cell["raw_path"],
                "filename": os.path.basename(cell["raw_path"]),
                "source": cell["source"],
                "source_name": self.source_names.get(cell["source"],
                                                     cell["source"]),
                "preset": cell["preset"], "pred": pred,
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
            "models_fitted": len(self.label_models),
            "models_stale": self._model_stale,
            "fit_min_pos": MIN_FIT_POS,
            "fit_min_neg": MIN_FIT_NEG,
            "class_names": self.class_names,
            "knn_k": self.knn_k,
            "neg_weight": self.neg_weight,
            "page_size": self.page_size,
            "label_mode": self.label_mode,
            "cluster": {"enabled": self.cluster_memb is not None,
                        "n": len(self.cluster_ids),
                        "res": self.cluster_res_used,
                        "target": self.cluster_target},
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
        # The label's decision rows went with it — drop the mirror.
        self._drop_states_cache()
        self._refresh_labels()
        self._bump_label(lid)
        # A deleted label cannot keep a model fitted on its decisions.
        self.label_models.pop(lid, None)
        self._model_scores.pop(lid, None)
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

    def _api_clusters(self):
        """Cluster cards for the sidebar's Clusters tab, largest first.

        Per card:
          size      — the FULL cluster size (decision/source independent).
          undecided — with a label_id: members without a decision for that
                      label (whole cluster) — the per-cluster progress badge.
          in_view   — how many members the CURRENT queue context (source ∩
                      scope ∩ label, the exact member logic /api/queue
                      uses) would show. When the source/scope filters hide
                      parts of a cluster, the card says so instead of
                      promising cells clicking it will not show.
        The medoid identifies the card (its thumbnail).
        """
        if self.cluster_memb is None:
            return jsonify({"error": "clustering is off (set cluster.target "
                                     "or cluster.res in the config)"}), 400
        label_id = request.args.get("label_id", type=int)
        source = request.args.get("source") or None
        scope = request.args.get("scope", "undecided")
        if scope not in ("undecided", "with", "without", "neg", "union"):
            scope = "undecided"
        lids = self._label_ids()
        have_label = label_id is not None and label_id in lids
        states = self._get_states()
        out = []
        for cid in self.cluster_ids:
            rows = np.flatnonzero(self.cluster_memb == cid - 1)
            med_cell = self.cid_cell.get(self.row_cid.get(
                self.cluster_medoid[cid]))
            item = {"id": cid, "size": int(len(rows)),
                    "medoid_filepath":
                        med_cell["filepath"] if med_cell else None,
                    "medoid_raw_path":
                        med_cell["raw_path"] if med_cell else None}
            n_view = n_und = 0
            for r in rows:
                cid_ = self.row_cid.get(int(r))
                if cid_ is None:
                    continue
                if have_label:
                    if (cid_, label_id) not in states:
                        n_und += 1
                    if not self._in_scope(states, cid_, scope, label_id,
                                          lids):
                        continue
                if source and self.cid_cell[cid_]["source"] != source:
                    continue
                n_view += 1
            if have_label:
                item["undecided"] = n_und
            item["in_view"] = n_view
            out.append(item)
        out.sort(key=lambda it: -it["size"])
        return jsonify({"clusters": out, "res": self.cluster_res_used})

    def _in_scope(self, states, cid, scope, label_id, lids):
        """Is cell_id a member of the given queue scope?

        The ONE membership definition shared by the queue and the
        whole-cluster write — an annotate_cluster request can never touch a
        cell the cluster queue is not currently showing.
        """
        st = states.get((cid, label_id))
        if scope == "undecided":
            return st is None
        if scope == "with":
            return st == STATE_POS
        if scope == "without":
            return (st != STATE_POS
                    and any(states.get((cid, l)) == STATE_POS for l in lids))
        if scope == "neg":
            return st == STATE_NEG
        return any(states.get((cid, l)) == STATE_POS for l in lids)   # union

    def _api_queue(self):
        """ONE unified queue per (source, label, scope, cluster, prediction).

        The client picks an image source (or all), a label in the sidebar
        and a scope radio; the queue is that intersection with the chosen
        ranking:
          undecided — cells without a decision for this label, ranked by
                      the label's primary score (model P after a
                      Refresh-model click, else exemplar similarity minus
                      the explicit-negative penalty); farthest-point
                      spread while the label has no positives yet;
                      sort=unc ranks the smallest pos/neg margin first
                      (active learning), shuffle reshuffles. Needs a model
                      (the scope is defined by the scores).
          with      — the label's positives; ranked by certainty (best
                      positive-likeness) or, with sort=review, by the
                      leave-one-out mislabel check.
          without   — labeled cells missing this label (candidates).
          neg       — the label's EXPLICIT negatives; certainty = how
                      positive-like, so descending surfaces the likely
                      wrongly-marked ones first (review / un-do them).
          union     — every cell carrying ANY label.
        Two orthogonal filters narrow every scope:
          cluster=?     — only members of this Leiden cluster (1-based id
                          from /api/clusters; the cluster-assisted bulk
                          pass — see label/cluster.py).
          pred_label=?  — classify bundle only: only cells whose argmax
                          prediction is this class name.
        And two cross-cutting rankings override the per-scope order:
          sort=medoid   — similarity to the cluster's medoid, most typical
                          member first (needs the cluster filter).
          sort=prob     — classify bundle only: P(pred_label class) first,
                          or each cell's argmax confidence when no class is
                          picked; the sorted-by value rides in `score`.
        Certainty is the best available positive-likeness (model P /
        classify head / kNN similarity — max merge). The decided scopes
        are pure DB reads and work without a model.
        """
        a = request.args
        label_id = a.get("label_id", type=int)
        source = a.get("source") or None
        offset = max(0, a.get("offset", 0, type=int) or 0)
        limit = min(500, max(1, a.get("limit", 200, type=int) or 200))
        scope = a.get("scope", "undecided")
        if scope not in ("undecided", "with", "without", "neg", "union"):
            scope = "undecided"
        sort = a.get("sort", "desc")
        if sort not in ("desc", "unc", "review", "medoid", "prob"):
            sort = "desc"
        # Seeded reshuffle (client Shuffle button): > 0 permutes the queue
        # deterministically so a fresh random sample reaches the page.
        shuffle = a.get("shuffle", 0, type=int) or 0
        # Cluster filter (1-based display id as served by /api/clusters) and
        # the classify bundle's prediction filter (class name or empty).
        cluster_id = a.get("cluster", type=int)
        pred_label = a.get("pred_label") or None

        if label_id is None or label_id not in self._label_ids():
            return jsonify({"error": "queue needs a known label_id"}), 400
        if sort == "medoid" and cluster_id is None:
            return jsonify({"error": "sort=medoid needs a cluster filter"}), 400
        if sort == "prob" and self.prob_matrix is None:
            return jsonify({"error": "sort=prob needs a classify "
                                     "bundle"}), 400

        states = self._get_states()
        label_rows = self._label_rows(states) \
            if self.engine is not None else {}
        cells = [c for c in self.cells
                 if source is None or c["source"] == source]
        if cluster_id is not None:
            if (self.cluster_memb is None
                    or cluster_id not in self.cluster_medoid):
                return jsonify({"error":
                                f"unknown cluster: {cluster_id}"}), 400
            # Cluster members are feature rows; a cell without features
            # belongs to no cluster and drops out here.
            wanted = {int(r)
                      for r in np.flatnonzero(
                          self.cluster_memb == cluster_id - 1)}
            cells = [c for c in cells
                     if self.cid_row.get(c["cell_id"]) in wanted]
        if pred_label is not None and self.prob_matrix is not None:
            ci = self.class_index.get(pred_label, -1)
            if ci < 0:
                return jsonify({"error":
                                f"unknown pred_label: {pred_label}"}), 400
            keep = []
            for c in cells:
                row = self.cid_row.get(c["cell_id"])
                if row is not None and int(self._pred_argmax[row]) == ci:
                    keep.append(c)
            cells = keep
        cert_by_cid = {}    # decided scopes: cell_id -> label certainty
        susp_by_cid = {}    # with scope, sort=review: review evidence

        # ---- member selection + ranking per scope ------------------------
        if scope == "undecided":
            if self.engine is None:
                return jsonify({"error": "the undecided queue needs a "
                                         "model"}), 400
            pos_rows, neg_rows = label_rows.get(label_id, ([], []))
            if not pos_rows and cluster_id is None:
                # Cold start (label has no positives yet): the farthest-
                # point spread over the whole space seeds the first picks.
                # Only for the UNFILTERED queue — inside a cluster this
                # dataset-wide sprinkle would usually miss the cluster
                # entirely; there every undecided member is a candidate and
                # the medoid order provides the spread instead.
                by_id = {c["cell_id"]: c for c in cells}
                picked = [by_id[cid] for cid in self.diverse_cids
                          if cid in by_id and (cid, label_id) not in states]
            elif not pos_rows:
                picked = sorted(
                    (c for c in cells
                     if (c["cell_id"], label_id) not in states),
                    key=lambda c: c["cell_id"])
            else:
                proba = self._model_proba(label_id)
                pos_part, neg_part = (None, None) if proba is not None \
                    else self.engine.parts(label_id, pos_rows, neg_rows)
                scored = []
                for c in cells:
                    if (c["cell_id"], label_id) in states:
                        continue
                    row = self.cid_row.get(c["cell_id"])
                    if row is None:
                        continue
                    if proba is not None:
                        # Refit model: uncertainty = closeness to the 0.5
                        # decision boundary (the margin sort=unc ranks by).
                        s = float(proba[row])
                        margin = abs(s - 0.5)
                    else:
                        # One transparent score everywhere: exemplar
                        # similarity minus the explicit-negative penalty.
                        # Negatives PUSH cells down instead of merely
                        # vetoing them.
                        p = float(pos_part[row])
                        n = float(neg_part[row])
                        s = p - self.neg_weight * n
                        margin = abs(p - n)
                    scored.append((c, s, margin))
                if sort == "unc":
                    # Active learning: smallest margin first — the most
                    # informative cells once the confident head of the
                    # queue is exhausted.
                    scored.sort(key=lambda t: t[2])
                else:
                    scored.sort(key=lambda t: -t[1])
                picked = [c for c, _s, _m in scored]
        else:
            lids = self._label_ids()
            picked = [c for c in cells
                      if self._in_scope(states, c["cell_id"], scope,
                                        label_id, lids)]

            review_mode = (sort == "review" and scope == "with"
                           and self.engine is not None)
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
            elif self.engine is not None:
                self._compute_certainties(picked, label_id, label_rows,
                                          cert_by_cid)
                # Stable two-pass sort: cell_id ascending as the base order,
                # then certainty — descending by default, sort=unc flips to
                # the least certain first; ties keep their registration
                # order in BOTH directions (a reverse=True tuple sort would
                # flip ties too).
                picked.sort(key=lambda c: c["cell_id"])
                picked.sort(key=lambda c: cert_by_cid.get(c["cell_id"], -1.0),
                            reverse=(sort != "unc"))

        # ---- cross-cutting rankings (cluster medoid / class probability) -
        score_override = {}   # cell_id -> the exact value this queue sorted by
        if sort in ("medoid", "prob"):
            rows, keep = [], []
            for c in picked:
                row = self.cid_row.get(c["cell_id"])
                if row is not None:
                    rows.append(row)
                    keep.append(c)
            if sort == "medoid":
                vals = np.asarray(
                    self.engine.W[rows] @ self.engine.W[
                        self.cluster_medoid[cluster_id]])
            else:
                # P(class): the picked pred_label class, else each cell's
                # own argmax confidence.
                ci = self.class_index.get(pred_label, -1) \
                    if pred_label else -1
                vals = (self.prob_matrix[np.asarray(rows), ci] if ci >= 0
                        else self._pred_maxprob[np.asarray(rows)])
            order = np.argsort(-vals, kind="stable")
            picked = [keep[i] for i in order]
            for c, v in zip(picked, vals[order]):
                score_override[c["cell_id"]] = float(v)
        elif shuffle and len(picked) > 1:
            # Seeded reshuffle of the queue (client Shuffle button): a new
            # random sample reaches the page while the same seed keeps the
            # pages stable while flipping.
            rng = np.random.default_rng(shuffle + self.seed)
            picked = [picked[i] for i in rng.permutation(len(picked))]

        total = len(picked)
        page = picked[offset:offset + limit]
        payload = []
        for c in page:
            pl = self._cell_payload(c, states, label_rows)
            if c["cell_id"] in score_override:
                # The exact value this queue sorted by (medoid similarity /
                # class probability) — the grid badge.
                pl["score"] = round(score_override[c["cell_id"]], 4)
            elif scope == "undecided":
                # The exact score this queue sorted by — the grid badge.
                row = self.cid_row.get(c["cell_id"])
                score = self._score_at(label_id, row, label_rows) \
                    if row is not None else None
                pl["score"] = None if score is None else round(score, 4)
            if c["cell_id"] in cert_by_cid:
                pl["cert"] = round(cert_by_cid[c["cell_id"]], 4)
            if c["cell_id"] in susp_by_cid:
                it = susp_by_cid[c["cell_id"]]
                ev = self.cid_cell.get(it["ev_cell_id"])
                pl["susp"] = {"susp": it["susp"], "own_sim": it["own_sim"],
                              "ev_sim": it["ev_sim"],
                              "ev_file": ev["raw_path"] if ev else None,
                              "ev_state": it["state"]}
            payload.append(pl)
        return jsonify({"cells": payload, "total": total,
                        "offset": offset, "scope": scope})

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
            # Same self-exclusion as the multi-positive branch: in the
            # `with` scope the listed cell may BE that one positive, and
            # its 1.0 self-similarity is not evidence for itself.
            self_mask = np.asarray(rows, dtype=np.int64) == others[0]
            if self_mask.any():
                sims = np.where(self_mask, -2.0, sims)
            knn_cert = sims.astype(np.float32)

        cert = knn_cert
        if self.prob_matrix is not None:
            ci = self.class_index.get(self._label_name(label_id), -1)
            if ci >= 0:
                head = self.prob_matrix[rows, ci]
                cert = head if cert is None else np.maximum(cert, head)
        # A refreshed logistic scorer is the sharpest certainty available —
        # it is the only source that has actually seen the explicit
        # negatives.
        proba = self._model_proba(label_id)
        if proba is not None:
            p_arr = proba[np.asarray(rows, dtype=np.int64)]
            cert = p_arr if cert is None else np.maximum(cert, p_arr)
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

    def _api_annotate_cluster(self):
        """Whole-cluster write: cluster ∩ current scope members × targets.

        The UI sends the cluster queue's exact context (label_id + scope +
        source), and the members are expanded server-side through the SAME
        `_in_scope` membership the queue uses — so the N shown above the
        grid (queue total) is exactly the number of cells this request
        writes. The write goes through _annotate_core: one transaction,
        one undoable op (Ctrl+Z reverts the whole cluster), engine bumps
        and the auto export included.
        """
        data = request.get_json(force=True)
        try:
            cluster_id = int(data.get("cluster_id"))
        except (TypeError, ValueError):
            return jsonify({"error":
                            f"bad cluster_id: {data.get('cluster_id')!r}"}), 400
        if (self.cluster_memb is None
                or cluster_id not in self.cluster_medoid):
            return jsonify({"error": f"unknown cluster: {cluster_id}"}), 400
        scope = data.get("scope", "undecided")
        if scope not in ("undecided", "with", "without", "neg", "union"):
            scope = "undecided"
        lids_all = self._label_ids()
        label_id = data.get("label_id")
        try:
            label_id = int(label_id) if label_id is not None else None
        except (TypeError, ValueError):
            return jsonify({"error": f"bad label_id: {label_id!r}"}), 400
        if label_id is None or label_id not in lids_all:
            return jsonify({"error": "annotate_cluster needs a known "
                                     "label_id (the queue's label)"}), 400
        source = data.get("source") or None

        states = self._get_states()
        cids = []
        for row in np.flatnonzero(self.cluster_memb == cluster_id - 1):
            cid = self.row_cid.get(int(row))
            if cid is None:
                continue
            if source and self.cid_cell[cid]["source"] != source:
                continue
            if self._in_scope(states, cid, scope, label_id, lids_all):
                cids.append(cid)
        if not cids:
            return jsonify({"error": "the cluster has no members in this "
                                     "scope"}), 400
        filepaths = [self.cid_cell[cid]["filepath"] for cid in cids]
        lids = data.get("label_ids") or []
        logging.getLogger(__name__).info(
            "Cluster write: cluster %d, %d member(s) in scope %s",
            cluster_id, len(cids), scope)
        return self._annotate_core(filepaths, lids, data.get("state"))

    def _annotate_core(self, filepaths, label_ids, state):
        """Shared write path for cells × labels (ONE undoable op).

        state: 1 = positive, 0 = explicit negative, "clear"/None = remove
        the decision. The whole batch is one transaction; the engine's
        score cache is invalidated for the written labels so the next queue
        request ranks with the new exemplars; the export CSV is rewritten
        exactly once per request.
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

        # DB write and mirror update happen under one lock so concurrent
        # annotates land in the same order on both sides.
        with self._write_lock:
            op_id = self.db.next_op_id()
            n = self.db.apply_batch(cids, lids, db_state, self.session_id,
                                    op_id)
            # Mirror the exact rows written into the states cache (state
            # None means row deletion, mirroring _write_one).
            with self._states_lock:
                if self._states_cache is not None:
                    for cid in cids:
                        for lid in lids:
                            if db_state is None:
                                self._states_cache.pop((cid, lid), None)
                            else:
                                self._states_cache[(cid, lid)] = int(db_state)
            if self.label_mode == "single" and db_state == STATE_POS:
                # Single-label mode is mutually exclusive: a cell holds at
                # most ONE positive. Keep-set semantics — every positive
                # written by THIS action stays, all other positives of those
                # cells go (each logged under the same op, so one undo
                # reverts the whole action). Negatives are untouched.
                cleared = self.db.clear_other_positives(cids, lids,
                                                        self.session_id, op_id)
                with self._states_lock:
                    if self._states_cache is not None:
                        for pair in cleared:
                            self._states_cache.pop(pair, None)
                for lid in {lid for _cid, lid in cleared}:
                    self._bump_label(lid)
        for lid in lids:
            self._bump_label(lid)
        self._mark_models_stale()
        # Refresh the label cache so per-label pos/neg counts served by
        # /api/state reflect the write immediately.
        self._refresh_labels()
        self._auto_export()
        st = self.db.stats()
        names = ", ".join(self._label_name(l) for l in lids)
        log.info("Annotate [%s] state=%s on %d cells (op %d)",
                 names, db_state, len(cids), op_id)
        return jsonify({"ok": True, "n": n, "labeled": st["labeled"]}), 200

    def _api_undo(self):
        """Revert the newest user action of this session (one op from the
        log). Works for batch applies and single writes; repeated undos
        keep walking back through earlier actions. Returns what was reverted.
        """
        result = self.db.undo_last_op(self.session_id)
        if result is None:
            return jsonify({"ok": True, "undone": False, "n": 0,
                            "message": "nothing to undo"})
        # The undo rewrote an arbitrary set of rows — drop the mirror and
        # let the next read reload it from the DB.
        self._drop_states_cache()
        self._bump_engines()
        self._mark_models_stale()
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

        # The DB stores portable paths; reading/getmtime need the absolute
        # form (relative entries resolve against the CWD they were stored in).
        raw_path = os.path.abspath(cell["raw_path"])
        lo, hi, gamma = _fparam("lo", 0.1), _fparam("hi", 99.9), \
            _fparam("gamma", 1.0)
        key = (raw_path, os.path.getmtime(raw_path),
               tuple(settings["channels"]), settings["channel_layout"],
               float(settings["max_value"]), max_px, float(lo), float(hi),
               float(gamma), self.aug_infer is not None, RENDER_VERSION)
        png = self.render_cache.get(key)
        if png is None:
            try:
                img = read_tiff_channels(raw_path, settings["channels"],
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
                    "image render failed: %s", raw_path)
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
