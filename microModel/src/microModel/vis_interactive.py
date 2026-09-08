"""Interactive reduction viewer with live image inspection (Flask server).

Reads the reduction_<method> tables (pca/umap/pacmap/localmap), the optional
find_cluster table and the inference table from infer.db. One plot tab is
offered per method that actually has a table; color_by accepts every
inference-table and find_cluster column except uid/features (cluster_res*
columns color categorically, numeric columns continuously). The bundle meta
uses the same field names for augmentation_infer, normalize_method,
normalize_with_masking, clip_low, clip_high, channels, channel_layout.

Image loading:
  - single_cell: load cell TIFF directly from {directory}/{filename}
    (filename is a single TIFF containing all channels); channels selected
    via config.
  - whole_image: filename is a JSON array of channel TIFF filenames;
    load each channel from {directory}/{channel_filename}, load mask from
    {directory}/{mask_filename} (full path stored in DB), then crop using
    label.
"""

import os
import re
import sys
import json
import sqlite3
import webbrowser

import numpy as np
import torch
from flask import Flask, jsonify, request

from microBase import (
    build_pipeline,
    apply,
    read_image,
    read_tiff_channels,
    read_mask,
    crop_cell,
    get_labels,
)

from .utils import (logger, resolve_output_paths, resolve_max_value,
                    stratified_sample_indices, parse_pred_prob)
from .dataset import _to_float_max


def main(config, port=5000):
    server = VisInteractiveServer(config, port)
    server.start()


class VisInteractiveServer:
    def __init__(self, config, port):
        self.config = config
        self.port = port
        self.mode = config.get("mode", "single_cell")
        self.data_cfg = config["data"]
        self.inf_cfg = config["inference"]
        self.red_cfg = config.get("reduction", {})

        self.db_name = self.inf_cfg.get("db_name", "infer.db")
        self.base_output_dir = self.config.get("output_dir")
        self.data_roots = self.data_cfg["root"]

        # Load model bundle to get augmentation_infer + normalize defaults.
        # SSL pretrain bundles and train bundles carry the same meta fields
        # needed here.
        model_path = config.get("model")
        if not isinstance(model_path, str) or not model_path:
            print("Error: config 'model' must be a string path to a model bundle "
                  "(train model.pt or SSL model.pt)", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(model_path):
            print(f"Error: model bundle not found: {model_path}", file=sys.stderr)
            sys.exit(1)
        logger.info("Loading model bundle from %s", model_path)
        bundle = torch.load(model_path, map_location="cpu", weights_only=False)
        self.meta = bundle["meta"]

        # Build augmentation_infer pipeline from bundle meta
        if "augmentation_infer" not in self.meta:
            print("Error: bundle meta missing 'augmentation_infer' (unsupported "
                  "pre-0.2.1 bundle format)", file=sys.stderr)
            sys.exit(1)
        aug_infer_spec = self.meta["augmentation_infer"]
        self.aug_infer_pipeline = build_pipeline(aug_infer_spec) if aug_infer_spec else None

        # Images are converted to float [0, 1] by data.max_value on load
        self.max_value = resolve_max_value(self.data_cfg)

        # Display defaults: percentile-based norm (per_channel or per_image).
        # Mask default comes from the bundle's training-time setting.
        self.norm_method_default = "per_channel"
        self.norm_with_masking_default = self.meta.get("normalize_with_masking", False)

        self.scatter_data = []
        self._available_channels = None
        self._color_by_cols = []
        # DR methods with a reduction_<method> table (canonical order).
        self._dr_methods = []

        self.app = Flask(__name__)
        self.app.route("/")(self._serve_html)
        self.app.route("/api/scatter")(self._api_scatter)
        self.app.route("/api/image")(self._api_image)
        self.app.route("/api/config")(self._api_config)

    def start(self):
        self._discover()
        self._available_channels = self._detect_available_channels()
        url = f"http://localhost:{self.port}"
        logger.info("Starting interactive viewer at %s", url)
        webbrowser.open(url)
        self.app.run(host="127.0.0.1", port=self.port, debug=False)

    # ------------------------------------------------------------------
    # Discovery: load reduction coords from each DB
    # ------------------------------------------------------------------

    def _discover(self):
        out_pairs = resolve_output_paths(self.data_roots, self.base_output_dir)
        seen = set()
        for data_dir, out_dir in out_pairs:
            db_path = os.path.join(out_dir, self.db_name)
            key = db_path
            if key in seen:
                continue
            seen.add(key)
            if not os.path.exists(db_path):
                logger.warning("DB not found: %s", db_path)
                continue
            try:
                self._load_coords(db_path)
            except Exception as e:
                logger.warning("Failed to load %s: %s", db_path, e)

        logger.info("Loaded %d scatter points from %d DB(s)",
                    len(self.scatter_data), len(seen))

    def _load_coords(self, db_path):
        conn = sqlite3.connect(db_path)
        # Discover every reduction_<method> table written by vis-reduction.
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'reduction_%'")
        tables = sorted(row[0] for row in cur.fetchall())
        if not tables:
            logger.warning(
                "No reduction_* tables found in %s. "
                "Run 'micromodel vis-reduction' first.", db_path)
            conn.close()
            return

        # Per-method 2D coords by uid. pca stores pc_1/pc_2, every other
        # method stores <method>_1/<method>_2. Stale tables missing their
        # coordinate columns are skipped with a warning.
        coords_by_uid = {}
        for table in tables:
            method = table[len("reduction_"):]
            if method not in ("pca", "umap", "pacmap", "localmap"):
                continue
            x_col = "pc_1" if method == "pca" else f"{method}_1"
            y_col = "pc_2" if method == "pca" else f"{method}_2"
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            if x_col not in cols or y_col not in cols:
                logger.warning("Table %s lacks %s/%s columns; skipping",
                               table, x_col, y_col)
                continue
            for uid, x, y in conn.execute(f"SELECT uid, {x_col}, {y_col} FROM {table}"):
                d = coords_by_uid.setdefault(int(uid), {})
                d[f"{method}_x"] = float(x)
                d[f"{method}_y"] = float(y)
            if method not in self._dr_methods:
                self._dr_methods.append(method)
        # Canonical method order (alphabetical sqlite_master order otherwise).
        canonical = ("pca", "umap", "pacmap", "localmap")
        self._dr_methods = ([m for m in canonical if m in self._dr_methods]
                            + [m for m in self._dr_methods if m not in canonical])

        # find_cluster: one 1-based cluster_res<resolution> column per Leiden
        # resolution; exposed to the frontend as categorical color_by options.
        # Resolution tags may contain a dot ("cluster_res0.5"), so every
        # identifier is quoted in the SELECT.
        cluster_by_uid = {}
        cluster_cols = []
        has_fc = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='find_cluster'"
        ).fetchone()
        if has_fc:
            all_cols = [row[1] for row in conn.execute("PRAGMA table_info(find_cluster)")]
            cluster_cols = [c for c in all_cols if c.startswith("cluster_")]
            if cluster_cols:
                sel = ", ".join(["uid"] + [f'"{c}"' for c in cluster_cols])
                for row in conn.execute(f"SELECT {sel} FROM find_cluster"):
                    d = cluster_by_uid.setdefault(int(row[0]), {})
                    for col, val in zip(cluster_cols, row[1:]):
                        # cluster_res<tag> holds integer IDs; the
                        # cluster_prob<tag> confidence columns are floats.
                        if col.startswith("cluster_prob"):
                            d[col] = float(val) if val is not None else None
                        else:
                            d[col] = int(val) if val is not None else None
            for col in cluster_cols:
                if col not in self._color_by_cols:
                    self._color_by_cols.append(col)

        # Get inference table columns dynamically. Everything except uid and
        # the features BLOB is a color_by candidate (and merged into the
        # scatter payload) — prediction/probability/path columns included.
        cur = conn.execute("PRAGMA table_info(inference)")
        infer_cols = [row[1] for row in cur.fetchall()]
        _exclude_color = {"uid", "features"}
        for col in infer_cols:
            if col not in self._color_by_cols and col not in _exclude_color:
                self._color_by_cols.append(col)

        # Load all from inference table (single table, no JOINs needed).
        # Column names may contain spaces (prob_<class> columns derive from
        # arbitrary class names) — quote every identifier.
        col_select = ", ".join(f'"{c}"' for c in infer_cols)
        cur = conn.execute(f"SELECT {col_select} FROM inference")
        col_names = [desc[0] for desc in cur.description]

        db_points = []
        for row in cur.fetchall():
            d = dict(zip(col_names, row))
            uid = int(d.get("uid"))
            coords = coords_by_uid.get(uid)
            if not coords:
                continue

            directory = str(d.get("directory", ""))
            filename = str(d.get("filename", ""))
            label = int(d.get("label", 0)) if d.get("label") is not None else 0
            mask_filename = str(d.get("mask_filename", "")) if d.get("mask_filename") else ""
            # source_image: for frontend hover display (first channel or filename)
            if self.mode == "whole_image":
                try:
                    ch_names = json.loads(filename)
                    display_name = ch_names[0] if ch_names else filename
                except (json.JSONDecodeError, TypeError):
                    display_name = filename
            else:
                display_name = filename
            src = (directory + "/" + display_name) if directory else display_name

            # Multi-label display data: [name, prob] pairs sorted by
            # probability (desc) — the frontend shows the top N (page
            # control). Source: the per-class prob_<name> columns written by
            # every classify bundle (features-only SSL rows carry none and
            # show no ranking).
            prob_items = []
            for col in infer_cols:
                if col.startswith("prob_") and d.get(col) is not None:
                    try:
                        prob_items.append((col[len("prob_"):], float(d[col])))
                    except (TypeError, ValueError):
                        pass
            ml_pairs = sorted(prob_items, key=lambda t: -t[1])

            # cid: OpenCell filenames embed CID#####; whole_image rows have
            # no CID in the name and fall back to the mask cell label.
            m_cid = re.search(r"CID\d+", filename)
            cid = m_cid.group(0) if m_cid else (
                str(label) if self.mode == "whole_image" else "")

            point = {
                "pred_class": str(d.get("pred_class")) if d.get("pred_class") else "unknown",
                "pred_prob": parse_pred_prob(d.get("pred_prob")),
                "source_image": src,
                "directory": directory,
                "filename": filename,
                "label": label,
                "mask_filename": mask_filename,
                "cid": cid,
                "ml": [[n, round(p, 4)] for n, p in ml_pairs],
            }
            # One <method>_x/<method>_y pair per available DR method.
            point.update(coords)
            # Cluster assignments + every remaining inference column land in
            # the point payload for color_by support (uid/features excluded).
            point.update(cluster_by_uid.get(uid, {}))
            _exclude_point = {"uid", "features"}
            for col in infer_cols:
                if col not in _exclude_point:
                    val = d.get(col)
                    if hasattr(val, "item"):
                        val = val.item()
                    point[col] = val
            db_points.append(point)
        conn.close()

        # Sub-sample to avoid browser overload (stratified by first color_by)
        db_points = self._subsample_points(db_points)
        self.scatter_data.extend(db_points)

    def _subsample_points(self, points):
        """Sub-sample points using reduction.sample_per_class (null/0 = keep all)."""
        sample_per_class = self.red_cfg.get("sample_per_class", 1000) or 0
        if sample_per_class <= 0 or len(points) <= sample_per_class:
            return points

        color_by_vals = self.red_cfg.get("color_by")
        if isinstance(color_by_vals, str):
            color_by_vals = [color_by_vals]
        color_by_vals = color_by_vals or []
        first_cb = color_by_vals[0] if color_by_vals else "pred_class"
        seed = 42
        labels = [pt.get(first_cb, "unknown") for pt in points]
        selected = stratified_sample_indices(
            len(points), labels, sample_per_class, seed,
            uniform=(first_cb == "pred_prob"))
        result = [points[i] for i in selected]
        logger.info("Sub-sampled %d -> %d points for display (sample_per_class=%d, color_by=%s)",
                    len(points), len(result), sample_per_class, first_cb)
        return result

    # ------------------------------------------------------------------
    # Channel detection: determine available channels from actual data
    # ------------------------------------------------------------------

    def _detect_available_channels(self):
        """Determine available channels from the first scatter point."""
        if not self.scatter_data:
            return self.meta.get("channels", [1])
        pt = self.scatter_data[0]
        try:
            if self.mode == "whole_image":
                # filename is a JSON array of channel filenames
                ch_names = json.loads(pt["filename"])
                n_ch = len(ch_names)
            else:
                # single-cell: read raw TIFF array to count channels
                full_path = os.path.join(pt["directory"], pt["filename"])
                from tifffile import TiffFile
                with TiffFile(full_path) as tif:
                    arr = tif.asarray()
                # channel_layout from bundle meta (same as inference)
                channel_layout = self.meta.get("channel_layout", "CHW")
                if arr.ndim == 2:
                    n_ch = 1
                elif arr.ndim == 3:
                    if arr.shape[0] == 1:
                        n_ch = 1
                    elif arr.shape[-1] == 1:
                        n_ch = 1
                    else:
                        n_ch = arr.shape[0] if channel_layout == "CHW" else arr.shape[-1]
                else:
                    n_ch = arr.shape[0] if channel_layout == "CHW" else arr.shape[-1]
            if n_ch > 0:
                return list(range(1, n_ch + 1))
        except Exception as e:
            logger.warning("Failed to detect channels: %s", e)
        return self.meta.get("channels", [1])

    # ------------------------------------------------------------------
    # Image resolution: load a cell crop for a scatter point
    # ------------------------------------------------------------------

    def _resolve_single_cell(self, directory, filename, channels, label):
        """Load a single-cell TIFF directly from {directory}/{filename}.

        The cell TIFF contains all channels. Channel selection uses the
        `channels` list (1-based indices). channel_layout comes from the
        bundle meta (same source as inference uses).
        Returns img_HWC.
        """
        full_path = os.path.join(directory, filename) if directory else filename
        if not os.path.exists(full_path):
            raise ValueError(f"Cell TIFF not found: {full_path}")
        channel_layout = self.meta.get("channel_layout", "CHW")
        ch_indices = list(channels) if channels else [1]
        img_hwc = read_tiff_channels(full_path, ch_indices, channel_layout=channel_layout)
        return img_hwc

    def _resolve_whole_image(self, directory, filename_json, mask_filename,
                             channels, label):
        """Load all channel TIFFs + mask, then crop the cell by label.

        filename_json: JSON array of channel TIFF filenames (in channel order).
        mask_filename: mask full filepath (stored in DB).
        Returns cropped img_HWC.
        """
        ch_filenames = json.loads(filename_json)
        if not ch_filenames:
            raise ValueError("Empty channel filename list")

        # Load each channel TIFF and stack into HWC
        arrays = []
        for ch_fname in ch_filenames:
            ch_path = os.path.join(directory, ch_fname) if directory else ch_fname
            arrays.append(read_image(ch_path))
        img_hwc = np.stack(arrays, axis=-1)  # (H, W, C)

        # Load mask — mask_filename is a full filepath stored in DB
        if not mask_filename:
            raise ValueError("Missing mask_filename in DB for whole-image mode")
        mask = read_mask(mask_filename)

        # Crop the cell using label (mask object ID). crop_cell hard-exits
        # (sys.exit) when the label has zero pixels in the mask; validate
        # first so a stale DB row (e.g. mask regenerated after inference)
        # yields a JSON error instead of killing the viewer server.
        if int(label) not in get_labels(mask):
            raise ValueError(f"Cell {label} not found in mask {mask_filename}")
        crop_hwc, _, _ = crop_cell(img_hwc, mask, int(label), padding=4)

        # Channel selection
        if channels is not None and crop_hwc.shape[2] != len(channels):
            ch_idx = [c - 1 for c in channels]
            ch_idx = [i for i in ch_idx if i < crop_hwc.shape[2]]
            crop_hwc = crop_hwc[:, :, ch_idx]
        return crop_hwc

    # ------------------------------------------------------------------
    # Flask endpoints
    # ------------------------------------------------------------------

    def _serve_html(self):
        return self._render_html()

    def _api_config(self):
        class_names = []
        if self.scatter_data:
            names = set(d["pred_class"] for d in self.scatter_data)
            class_names = sorted(names)

        # color_by options: every inference-table + find_cluster column
        # except uid/features, with the two prediction columns first (the
        # common choices).
        all_cols = list(self._color_by_cols)
        color_by_cols = ([c for c in ("pred_class", "pred_prob") if c in all_cols]
                         + [c for c in all_cols
                            if c not in ("pred_class", "pred_prob")])

        # DR methods with a table, already in canonical order
        methods = list(self._dr_methods)

        # Channels: all available from data (not just training channels)
        channels = self._available_channels or self.meta.get("channels", [1])

        return jsonify({
            "channels": channels,
            "class_names": class_names,
            "mode": self.mode,
            "norm_default": self.norm_method_default,
            "mask_default": self.norm_with_masking_default,
            "color_by_cols": color_by_cols,
            "methods": methods,
        })

    def _api_scatter(self):
        return jsonify(self.scatter_data)

    def _api_image(self):
        src = request.args.get("src", "")
        label_str = request.args.get("cid", "0")
        norm_mode = request.args.get("norm", self.norm_method_default)
        mask_str = request.args.get("mask", "1" if self.norm_with_masking_default else "0")
        ch_str = request.args.get("channels", "")
        disp_size = request.args.get("size", "original")

        label = int(label_str)
        do_mask = mask_str == "1"

        if ch_str:
            requested_channels = [int(c) for c in ch_str.split(",")]
        else:
            requested_channels = self._available_channels or self.meta.get("channels", [1])

        try:
            # Find the scatter point matching src + label to get directory/filename/mask_filename
            point = None
            for d in self.scatter_data:
                if d["source_image"] == src and d["label"] == label:
                    point = d
                    break
            if point is None:
                return jsonify({"error": f"Point not found: src={src}, cid={label}"}), 400

            directory = point["directory"]
            filename = point["filename"]
            mask_filename = point.get("mask_filename", "")

            if self.mode == "single_cell":
                img_hwc = self._resolve_single_cell(
                    directory, filename, requested_channels, label)
            elif self.mode == "whole_image":
                img_hwc = self._resolve_whole_image(
                    directory, filename, mask_filename,
                    requested_channels, label)
            else:
                return jsonify({"error": f"Unknown mode: {self.mode}"}), 400

            if img_hwc.ndim == 2:
                img_hwc = img_hwc[:, :, None]

            # Adjust requested_channels if image has fewer channels. Filtering (not
            # truncating) keeps the labels honest: truncating [1,3,5] to two
            # entries would render channel-index 1's data under the label
            # "ch3". Only in-range requests are kept (labels stay the true
            # 1-based indices).
            actual_n = img_hwc.shape[2]
            requested_channels = [c for c in requested_channels if c <= actual_n]

            mask = (img_hwc != 0).any(axis=2).astype(np.uint8) if do_mask else None
            # size="model": convert to float [0, 1] by data.max_value, then
            # apply bundle's augmentation_infer (no ToFloat in new specs)
            # size="original": only the float conversion (no resize, no pad)
            if disp_size == "model" and self.aug_infer_pipeline is not None:
                img_hwc = _to_float_max(img_hwc, self.max_value)
                img_hwc, mask = apply(self.aug_infer_pipeline, img_hwc, mask)
            else:
                # size="original": only the float conversion (no resize, no
                # pad) — divide by the CONFIGURED data.max_value (e.g. 4095
                # for 12-bit), not the dtype's hardcoded max, so both display
                # modes agree on the [0, 1] domain.
                img_hwc = _to_float_max(img_hwc, self.max_value)

            # If mask ON, zero background pixels (non-cell within bbox).
            mask_bool = mask.astype(bool) if mask is not None else None
            if mask_bool is not None:
                img_hwc[~mask_bool] = 0

            # Convert to uint16 for display
            disp = np.floor(img_hwc * 65535).clip(0, 65535).astype(np.uint16)

            # Percentile display range (0.1% / 99.9%):
            # per_channel: each channel independent
            # per_image: all channels share same values
            display_range = {}
            if norm_mode == "per_image":
                if mask_bool is not None:
                    region = disp[mask_bool]
                else:
                    region = disp.reshape(-1, disp.shape[2])
                if region.size == 0:
                    region = disp.reshape(-1, disp.shape[2])
                p01, p99 = np.percentile(region, [0.1, 99.9])
                for i, c in enumerate(requested_channels):
                    if i < disp.shape[2]:
                        display_range[f"ch{c}"] = [float(p01), float(p99)]
            else:  # per_channel
                for i, c in enumerate(requested_channels):
                    if i >= disp.shape[2]:
                        continue
                    ch_2d = disp[:, :, i]
                    if mask_bool is not None:
                        region = ch_2d[mask_bool]
                    else:
                        region = ch_2d
                    if region.size == 0:
                        region = ch_2d
                    p01, p99 = np.percentile(region, [0.1, 99.9])
                    display_range[f"ch{c}"] = [float(p01), float(p99)]

            result = {}
            for i, c in enumerate(requested_channels):
                if i < disp.shape[2]:
                    ch_data = disp[:, :, i].flatten().tolist()
                    result[f"ch{c}"] = ch_data
            result["shape"] = [disp.shape[0], disp.shape[1]]
            result["display_range"] = display_range
            result["dtype_max"] = 65535
            return jsonify(result)

        except SystemExit as e:
            # microBase hard-exits (sys.exit) on missing files / out-of-range
            # channels; convert to a JSON error so the server never dies.
            logger.exception("Image fetch error: %s", e)
            return jsonify({"error": "Data error (missing or mismatched "
                                      "image/mask file)"}), 500
        except Exception as e:
            logger.exception("Image fetch error: %s", e)
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # HTML / JS frontend
    # ------------------------------------------------------------------

    def _render_html(self):
        return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>microModel — Interactive Reduction View</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; height: 100vh; display: flex; flex-direction: column; }
#controls { background: #fff; padding: 8px 16px; border-bottom: 1px solid #ddd; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 13px; }
#controls label { display: inline-flex; align-items: center; gap: 4px; font-size: 13px; }
#controls input[type=number] { width: 60px; font-size: 13px; }
#controls select { font-size: 13px; }
#controls .norm-group { display: inline-flex; align-items: center; gap: 2px; }
#controls .norm-group label { gap: 2px; }
.ch-control { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
#dataset-row { background: #e8e8e8; padding: 3px 16px; font-size: 12px; color: #555; border-bottom: 1px solid #ddd; min-height: 22px; font-weight: 600; }
#image-row { background: #fff; border-bottom: 1px solid #ddd; padding: 6px 16px; display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; overflow-x: auto; }
#image-row .img-card { border: 1px solid #ddd; border-radius: 4px; padding: 4px; text-align: center; flex-shrink: 0; }
#image-row .img-card canvas { display: block; image-rendering: pixelated; }
#image-row .img-card .label { font-size: 11px; font-weight: 600; }
#info-panel { font-size: 13px; color: #333; padding: 4px 8px; white-space: nowrap; font-weight: 600; flex-shrink: 0; }
#plot-section { flex: 1; display: flex; flex-direction: column; min-height: 300px; padding: 6px 12px; }
#plot-tabs { display: flex; gap: 2px; margin-bottom: 4px; }
#plot-tabs .tab { padding: 4px 16px; border: 1px solid #ccc; background: #eee; cursor: pointer; font-size: 13px; border-radius: 4px 4px 0 0; }
#plot-tabs .tab.active { background: #fff; border-bottom-color: #fff; font-weight: 600; }
#plot-container { flex: 1; display: flex; align-items: flex-start; justify-content: center; min-height: 200px; }
</style>
</head>
<body>
<div id="controls">
  <span>Normalize:</span>
  <div class="norm-group">
    <label><input type="radio" name="norm-mode" value="per_channel" checked> per_ch</label>
    <label><input type="radio" name="norm-mode" value="per_image"> per_img</label>
  </div>
  <label><input type="checkbox" id="cb-mask" checked> Mask</label>
  <span>Size:</span>
  <div class="norm-group">
    <label><input type="radio" name="display-size" value="original"> original</label>
    <label><input type="radio" name="display-size" value="model" checked> model</label>
  </div>
  <label>Color_by:
    <select id="color-by-select"></select>
  </label>
  <label>Top_N:
    <input type="number" id="top-n" value="5" min="1" max="30" step="1">
  </label>
  <label><input type="checkbox" id="show-src"> src</label>
  <label><input type="checkbox" id="show-cid"> cid</label>
  <span>Ch:</span>
  <div id="color-controls" style="display:inline-flex;flex-wrap:wrap;gap:4px;align-items:center;"></div>
</div>
<div id="dataset-row"></div>
<div id="image-row">
</div>
<div id="plot-section">
  <div id="plot-tabs"></div>
  <div id="plot-container">
    <div id="plot-div" style="max-width:700px;"></div>
  </div>
</div>
<script>
var scatterData = [];
var selectedPoint = null;
var pinnedPoints = new Set();
var configData = {};
var activeTab = 'pca';
var drMethods = ['pca', 'umap'];
var _lastData = null, _lastChannels = null, _lastPointData = null, _lastNormMode = 'per_channel';
var _savedChannelVals = {};
var _colorBy = 'pred_class';
// Same 60-color palette as the static reduction PDFs (tab20/20b/20c).
var PALETTE60 = ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5','#393b79','#5254a3','#6b6ecf','#9c9ede','#637939','#8ca252','#b5cf6b','#cedb9c','#8c6d31','#bd9e39','#e7ba52','#e7cb94','#843c39','#ad494a','#d6616b','#e7969c','#7b4173','#a55194','#ce6dbd','#de9ed6','#3182bd','#6baed6','#9ecae1','#c6dbef','#e6550d','#fd8d3c','#fdae6b','#fdd0a2','#31a354','#74c476','#a1d99b','#c7e9c0','#756bb1','#9e9ac8','#bcbddc','#dadaeb','#636363','#969696','#bdbdbd','#d9d9d9'];
// Uniform scatter color when no color_by is selected (matches the PDFs).
var NONE_COLOR = '#87CEEB';

function showError(msg) {
  var row = document.getElementById('image-row');
  while (row.firstChild) row.removeChild(row.firstChild);
  var span = document.createElement('span');
  span.style.cssText = 'color:red;padding:8px;';
  span.textContent = msg;
  row.appendChild(span);
}

async function loadConfig() {
  var r = await fetch('/api/config');
  configData = await r.json();
  var normDefault = configData.norm_default || 'per_channel';
  var rb = document.querySelector('input[name="norm-mode"][value="' + normDefault + '"]');
  if (rb) rb.checked = true;
  var maskDefault = configData.mask_default !== undefined ? configData.mask_default : true;
  document.getElementById('cb-mask').checked = maskDefault;
  buildChannelWidgets(configData.channels);
  buildColorBySelect(configData.color_by_cols || ['pred_class']);
  buildTabs(configData.methods || ['pca', 'umap']);
  loadScatter();
}

// One tab per DR method that has a reduction_<method> table in the DBs.
function buildTabs(methods) {
  drMethods = methods;
  if (drMethods.indexOf(activeTab) < 0) activeTab = drMethods[0] || 'pca';
  var bar = document.getElementById('plot-tabs');
  bar.innerHTML = '';
  drMethods.forEach(function(m) {
    var b = document.createElement('button');
    b.className = 'tab' + (m === activeTab ? ' active' : '');
    b.id = 'tab-' + m;
    b.textContent = m.toUpperCase();
    b.addEventListener('click', function() { switchTab(m); });
    bar.appendChild(b);
  });
}

function buildChannelWidgets(channels) {
  var defaultColors = ['red','green','blue','cyan','magenta','yellow'];
  var container = document.getElementById('color-controls');
  container.innerHTML = '';
  channels.forEach(function(ch, i) {
    var wrap = document.createElement('div');
    wrap.className = 'ch-control';

    var cb = document.createElement('input');
    cb.type = 'checkbox'; cb.id = 'chk-' + ch; cb.checked = true;
    cb.addEventListener('change', function() { if (selectedPoint) fetchImage(); });

    var label = document.createElement('span');
    label.textContent = 'ch' + ch + ':';

    var colSel = document.createElement('select');
    colSel.id = 'color-' + ch;
    defaultColors.forEach(function(c) {
      var opt = document.createElement('option');
      opt.value = c; opt.textContent = c;
      if (c === defaultColors[i % defaultColors.length]) opt.selected = true;
      colSel.appendChild(opt);
    });
    colSel.addEventListener('change', function() { if (selectedPoint) fetchImage(); });

    wrap.appendChild(cb);
    wrap.appendChild(label);
    wrap.appendChild(colSel);
    container.appendChild(wrap);
  });
}

function buildColorBySelect(cols) {
  var sel = document.getElementById('color-by-select');
  sel.innerHTML = '';
  // "none" = uniform light-blue points (matches color_by: null in the PDFs).
  var noneOpt = document.createElement('option');
  noneOpt.value = '__none__'; noneOpt.textContent = 'none';
  sel.appendChild(noneOpt);
  cols.forEach(function(c) {
    var opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    if (c === _colorBy) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', function() {
    _colorBy = sel.value;
    renderPlot();
  });
}

function isNumericColumn(points, key) {
  for (var i = 0; i < Math.min(points.length, 100); i++) {
    var v = points[i][key];
    if (v === null || v === undefined) continue;
    if (typeof v !== 'number' && (typeof v !== 'string' || isNaN(parseFloat(v))))
      return false;
  }
  return true;
}

function computeColors(points, colorBy) {
  if (!colorBy || colorBy === '__none__') {
    return { colors: points.map(function() { return NONE_COLOR; }),
             type: 'none', categories: [], colorMap: {} };
  }

  // pred_class is always categorical; cluster_res* columns hold integer IDs
  // from find_cluster but must also color categorically (cluster ID order
  // encodes plot-distance, not magnitude — cluster_prob* confidences stay
  // numeric). Everything else keeps the numeric-check heuristic (numeric ->
  // viridis, else categorical).
  if (colorBy !== 'pred_class' && colorBy.indexOf('cluster_res') !== 0) {
    var values = points.map(function(d) { return d[colorBy]; });
    var numeric = isNumericColumn(points, colorBy);

    if (numeric) {
      var nums = values.map(function(v) { return parseFloat(v); });
      var min = Math.min.apply(null, nums);
      var max = Math.max.apply(null, nums);
      function viridis(t) {
        var r, g, b;
        if (t < 0.25) { r = 68 + t*4*(59-68); g = 1 + t*4*(82-1); b = 84 + t*4*(139-84); }
        else if (t < 0.5) { r = 59 + (t-0.25)*4*(81-59); g = 82 + (t-0.25)*4*(171-82); b = 139 + (t-0.25)*4*(41-139); }
        else if (t < 0.75) { r = 81 + (t-0.5)*4*(177-81); g = 171 + (t-0.5)*4*(204-171); b = 41 + (t-0.5)*4*(95-41); }
        else { r = 177 + (t-0.75)*4*(253-177); g = 204 + (t-0.75)*4*(231-204); b = 95 + (t-0.75)*4*(37-95); }
        return 'rgb(' + Math.round(r) + ',' + Math.round(g) + ',' + Math.round(b) + ')';
      }
      // Continuous columns keep the RAW numeric array: the trace colors via
      // the shared Plotly coloraxis (added to the layout in renderPlot),
      // which is what actually renders the colorbar.
      return { values: nums, colors: nums.map(function(n) {
        var t = max > min ? (n - min) / (max - min) : 0.5;
        return viridis(Math.max(0, Math.min(1, t)));
      }), type: 'continuous', min: min, max: max };
    }
  }

  // Categorical: unique values, numeric-aware sort (cluster IDs ascend with
  // the legend), 60-color palette shared with the static PDFs.
  var strVals = points.map(function(d) {
    var v = d[colorBy];
    return (v === null || v === undefined) ? '' : String(v);
  });
  var cats = [];
  strVals.forEach(function(v) { if (cats.indexOf(v) < 0) cats.push(v); });
  cats.sort(function(a, b) {
    var fa = parseFloat(a), fb = parseFloat(b);
    if (!isNaN(fa) && !isNaN(fb)) return fa - fb;
    return a < b ? -1 : (a > b ? 1 : 0);
  });
  var colorMap = {};
  cats.forEach(function(c, i) { colorMap[c] = PALETTE60[i % PALETTE60.length]; });
  return { colors: strVals.map(function(v) { return colorMap[v] || '#333'; }),
           type: 'categorical', categories: cats, colorMap: colorMap };
}

function getPlotTrace(points, mode) {
  var xKey = mode + '_x';
  var yKey = mode + '_y';
  var colorInfo = computeColors(points, _colorBy);
  var marker = { size: 4, color: colorInfo.colors, opacity: 0.7 };
  if (colorInfo.type === 'continuous') {
    // Numeric column -> raw values + the shared coloraxis (its colorbar is
    // attached to the layout in renderPlot). Precomputed rgb strings would
    // leave the coloraxis unreferenced and the colorbar invisible.
    marker.color = colorInfo.values;
    marker.coloraxis = 'coloraxis';
  }
  var trace = {
    x: points.map(function(d) { return d[xKey]; }),
    y: points.map(function(d) { return d[yKey]; }),
    mode: 'markers', type: 'scattergl',
    marker: marker,
    text: points.map(function(d) {
      // Hover shows ONLY the top-N classes by probability (N from the
      // Top_N control), highest first. No separate pred_class/pred_prob line.
      var topN = parseInt(document.getElementById('top-n').value, 10) || 5;
      var ml = d.ml || [];
      var lines = [];
      var n = Math.min(topN, ml.length);
      for (var i = 0; i < n; i++) {
        lines.push((i + 1) + '. ' + ml[i][0] + ': ' + ml[i][1].toFixed(3));
      }
      if (_colorBy && _colorBy !== 'pred_class' && _colorBy !== '__none__') {
        lines.push(_colorBy + ': ' + d[_colorBy]);
      }
      if (document.getElementById('show-src').checked) {
        lines.push('src: ' + d.source_image.split(/[/\\\\]/).pop());
      }
      if (document.getElementById('show-cid').checked) {
        lines.push('cid: ' + (d.cid || d.label));
      }
      return lines.join('<br>');
    }),
    hoverinfo: 'text',
    customdata: points.map(function(d,i) { return [d.source_image, d.label, i]; }),
    selectedpoints: []
  };
  trace._colorInfo = colorInfo;
  return trace;
}

function updatePlotSize() {
  var container = document.getElementById('plot-container');
  if (!container) return;
  var maxW = 700;
  var w = Math.min(maxW, Math.max(200, container.clientWidth - 10));
  var h = w * 2 / 3;
  Plotly.relayout('plot-div', { width: w, height: h });
}

async function loadScatter() {
  var r = await fetch('/api/scatter');
  scatterData = await r.json();
  renderPlot();
  window.addEventListener('resize', updatePlotSize);
}

function renderPlot() {
  var trace = getPlotTrace(scatterData, activeTab);

  var container = document.getElementById('plot-container');
  var maxW = 700;
  var w = Math.min(maxW, Math.max(200, container.clientWidth - 10));
  var h = w * 2 / 3;
  var layout = {
    width: w, height: h,
    margin: { l: 50, r: 20, t: 10, b: 40 },
    dragmode: 'zoom', hovermode: 'closest',
    title: activeTab.toUpperCase()
  };

  var traces = [trace];
  var colorInfo = trace._colorInfo;
  if (colorInfo && colorInfo.type === 'categorical' && colorInfo.categories.length <= 60) {
    colorInfo.categories.forEach(function(cat) {
      var dummyTrace = {
        x: [null], y: [null],
        mode: 'markers', type: 'scattergl',
        marker: { size: 8, color: colorInfo.colorMap[cat] },
        name: cat,
        showlegend: true
      };
      traces.push(dummyTrace);
    });
    trace.showlegend = false;
  } else if (colorInfo && colorInfo.type === 'continuous') {
    layout.coloraxis = { colorscale: 'Viridis', cmin: colorInfo.min, cmax: colorInfo.max,
                         colorbar: { title: _colorBy, thickness: 10 } };
  }

  Plotly.newPlot('plot-div', traces, layout, { responsive: true });
  document.getElementById('plot-div').on('plotly_click', function(eventData) { handleClick(eventData); });
}

function switchTab(tab) {
  if (tab === activeTab || drMethods.indexOf(tab) < 0) return;
  activeTab = tab;
  var tabs = document.querySelectorAll('#plot-tabs .tab');
  tabs.forEach(function(el) {
    el.classList.toggle('active', el.id === 'tab-' + tab);
  });
  renderPlot();
}

function handleClick(eventData) {
  if (!eventData.points || eventData.points.length === 0) return;
  var pt = eventData.points[0];
  var idx = pt.customdata ? pt.customdata[2] : pt.pointNumber;
  if (typeof idx !== 'number' || idx < 0 || idx >= scatterData.length) return;
  var pointData = scatterData[idx];
  if (pinnedPoints.has(idx)) {
    pinnedPoints.delete(idx);
    if (selectedPoint && selectedPoint.idx === idx) {
      selectedPoint = null;
      document.getElementById('image-row').innerHTML = '';
      document.getElementById('dataset-row').innerHTML = '';
    }
  } else {
    pinnedPoints.add(idx);
    selectedPoint = { idx: idx, data: pointData };
    fetchImage();
  }
  updateSelection();
}

function updateSelection() {
  var selArr = Array.from(pinnedPoints);
  Plotly.restyle('plot-div', { selectedpoints: [selArr] });
}

async function fetchImage() {
  if (!selectedPoint) return;
  var d = selectedPoint.data;
  var normMode = document.querySelector('input[name="norm-mode"]:checked').value;
  var mask = document.getElementById('cb-mask').checked ? 1 : 0;
  var dispSize = document.querySelector('input[name="display-size"]:checked').value;
  var channels = configData.channels.filter(function(ch) { return document.getElementById('chk-' + ch).checked; });
  var url = '/api/image?src=' + encodeURIComponent(d.source_image) + '&cid=' + d.label + '&norm=' + normMode + '&mask=' + mask + '&size=' + dispSize + '&channels=' + channels.join(',');
  try {
    var r = await fetch(url);
    var data = await r.json();
    if (data.error) {
      showError('Error: ' + data.error);
      return;
    }
    _savedChannelVals = {};
    _lastData = data; _lastChannels = channels;
    _lastPointData = d; _lastNormMode = normMode;
    renderChannels(data, channels, d, normMode);
  } catch(e) {
    showError('Error: ' + e);
  }
}

function renderChannels(data, channels, pointData, normMode) {
  var h = data.shape[0], w = data.shape[1];
  var row = document.getElementById('image-row');
  row.innerHTML = '';

  var dsRow = document.getElementById('dataset-row');
  dsRow.innerHTML = '';
  if (pointData && pointData.directory) {
    dsRow.textContent = pointData.directory;
  }

  var compositeBuf = new Float32Array(w * h * 3);

  channels.forEach(function(ch) {
    var chData = data['ch' + ch];
    if (!chData) return;
    var colEl = document.getElementById('color-' + ch);
    var colorName = colEl ? colEl.value : 'red';
    var colorMap = { red:[255,0,0], green:[0,255,0], blue:[0,0,255], cyan:[0,255,255], magenta:[255,0,255], yellow:[255,255,0] };
    var rgb = colorMap[colorName] || [255,255,255];

    var loVal, hiVal, loMin, loMax, loStep;
    var dr = (data.display_range && data.display_range['ch' + ch]) ? data.display_range['ch' + ch] : null;
    loMin = 0; loMax = 65535; loStep = 1;
    loVal = _savedChannelVals[ch + '-lo'] !== undefined ? _savedChannelVals[ch + '-lo'] : (dr ? dr[0] : 0);
    hiVal = _savedChannelVals[ch + '-hi'] !== undefined ? _savedChannelVals[ch + '-hi'] : (dr ? dr[1] : 65535);
    var minV = loVal, maxV = hiVal, range = maxV - minV || 1;

    var canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    var ctx = canvas.getContext('2d');
    var imgData = ctx.createImageData(w, h);

    for (var i = 0; i < chData.length && i < w*h; i++) {
      var v = Math.max(0, Math.min(255, ((chData[i] - minV) / range) * 255));
      var idx4 = i * 4;
      imgData.data[idx4] = (v * rgb[0] / 255);
      imgData.data[idx4+1] = (v * rgb[1] / 255);
      imgData.data[idx4+2] = (v * rgb[2] / 255);
      imgData.data[idx4+3] = 255;

      var normV = Math.max(0, Math.min(1, (chData[i] - minV) / range));
      var idx3 = i * 3;
      compositeBuf[idx3] += normV * rgb[0] / 255;
      compositeBuf[idx3+1] += normV * rgb[1] / 255;
      compositeBuf[idx3+2] += normV * rgb[2] / 255;
    }
    ctx.putImageData(imgData, 0, 0);

    var card = document.createElement('div');
    card.className = 'img-card';

    var ctrlRow = document.createElement('div');
    ctrlRow.style.cssText = 'display:flex;gap:4px;align-items:center;justify-content:center;min-height:28px;font-size:12px;';
    var loLbl = document.createElement('span');
    loLbl.textContent = 'Lo:';
    var loInput = document.createElement('input');
    loInput.type = 'number'; loInput.id = 'lo-' + ch;
    loInput.style.cssText = 'width:65px;font-size:12px;padding:1px 2px;text-align:center;';
    loInput.min = loMin; loInput.max = loMax; loInput.step = loStep; loInput.value = loVal;
    loInput.addEventListener('change', function() {
      var loV = parseFloat(loInput.value);
      var hi = document.getElementById('hi-' + ch);
      _savedChannelVals[ch + '-lo'] = loV;
      if (hi && loV >= parseFloat(hi.value)) {
        hi.value = (loV + loStep).toFixed(loStep >= 1 ? 0 : 1);
        _savedChannelVals[ch + '-hi'] = parseFloat(hi.value);
      }
      if (selectedPoint) renderChannels(_lastData, _lastChannels, _lastPointData, _lastNormMode);
    });
    var hiLbl = document.createElement('span');
    hiLbl.textContent = 'Hi:';
    var hiInput = document.createElement('input');
    hiInput.type = 'number'; hiInput.id = 'hi-' + ch;
    hiInput.style.cssText = 'width:65px;font-size:12px;padding:1px 2px;text-align:center;';
    hiInput.min = loMin; hiInput.max = loMax; hiInput.step = loStep; hiInput.value = hiVal;
    hiInput.addEventListener('change', function() {
      var hiV = parseFloat(hiInput.value);
      var lo = document.getElementById('lo-' + ch);
      _savedChannelVals[ch + '-hi'] = hiV;
      if (lo && hiV <= parseFloat(lo.value)) {
        lo.value = (hiV - loStep).toFixed(loStep >= 1 ? 0 : 1);
        _savedChannelVals[ch + '-lo'] = parseFloat(lo.value);
      }
      if (selectedPoint) renderChannels(_lastData, _lastChannels, _lastPointData, _lastNormMode);
    });
    ctrlRow.appendChild(loLbl);
    ctrlRow.appendChild(loInput);
    ctrlRow.appendChild(hiLbl);
    ctrlRow.appendChild(hiInput);

    card.appendChild(ctrlRow);
    var labelDiv = document.createElement('div');
    labelDiv.className = 'label';
    labelDiv.textContent = 'ch' + ch + ' (' + colorName + ')';
    card.appendChild(labelDiv);
    card.appendChild(canvas);
    row.appendChild(card);
  });

  var compCanvas = document.createElement('canvas');
  compCanvas.width = w; compCanvas.height = h;
  var compCtx = compCanvas.getContext('2d');
  var compImgData = compCtx.createImageData(w, h);
  for (var i = 0; i < w * h; i++) {
    var idx3 = i * 3;
    var idx4 = i * 4;
    compImgData.data[idx4] = Math.min(255, Math.max(0, compositeBuf[idx3] * 255));
    compImgData.data[idx4+1] = Math.min(255, Math.max(0, compositeBuf[idx3+1] * 255));
    compImgData.data[idx4+2] = Math.min(255, Math.max(0, compositeBuf[idx3+2] * 255));
    compImgData.data[idx4+3] = 255;
  }
  compCtx.putImageData(compImgData, 0, 0);
  var compCard = document.createElement('div');
  compCard.className = 'img-card';
  var compCtrlRow = document.createElement('div');
  compCtrlRow.style.cssText = 'display:flex;align-items:center;justify-content:center;min-height:28px;font-size:12px;';
  if (pointData) {
    var parts = [pointData.pred_class + ' (' + (pointData.pred_prob*100).toFixed(1) + '%)'];
    if (pointData.mask_filename) {
      parts.push(pointData.mask_filename.split(/[/\\\\]/).pop());
    }
    parts.push('cid:' + pointData.label);
    compCtrlRow.textContent = parts.join(' | ');
  }
  compCard.appendChild(compCtrlRow);
  var compLabelDiv = document.createElement('div');
  compLabelDiv.className = 'label';
  compLabelDiv.textContent = 'Composite';
  compCard.appendChild(compLabelDiv);
  compCard.appendChild(compCanvas);
  row.appendChild(compCard);
}

// Tab buttons are built dynamically from /api/config (one per DR method) —
// no hardcoded per-tab listeners here.
// Hover content controls: changing Top_N / src / cid re-renders the traces
// (tooltips are baked into the trace text arrays).
document.getElementById('top-n').addEventListener('input', renderPlot);
document.getElementById('show-src').addEventListener('change', renderPlot);
document.getElementById('show-cid').addEventListener('change', renderPlot);
var normRadios = document.querySelectorAll('input[name="norm-mode"]');
normRadios.forEach(function(el) {
  el.addEventListener('change', function() { if (selectedPoint) fetchImage(); });
});
document.getElementById('cb-mask').addEventListener('change', function() { if (selectedPoint) fetchImage(); });
var sizeRadios = document.querySelectorAll('input[name="display-size"]');
sizeRadios.forEach(function(el) {
  el.addEventListener('change', function() { if (selectedPoint) fetchImage(); });
});

loadConfig();
</script>
</body>
</html>"""
