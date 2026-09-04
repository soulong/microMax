"""Inference + feature extraction -> infer.db.

The bundle determines the model mode, and inference.pred_class /
inference.feature control what is written:
  - features: load SSL bundle (model.pt; no num_classes in meta), extract
      backbone features only.
      pred_class/pred_prob = NULL (requesting pred_class logs a warning).
  - classify: load train bundle (model.pt), predict + extract features.

Model mode is detected from the bundle: train bundles carry "state_dict" +
"num_classes" in meta; SSL bundles carry a full "state_dict" without
"num_classes".

DB schema — single `inference` table + 3 lazily-created reduction tables.
No _meta, no migrations.
"""

import os
import sys
import json
import sqlite3

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from microBase import CellDataset, ImageDataset

from .utils import (logger, set_seed, select_device, load_label_csv,
                    resolve_output_paths, copy_config_file,
                    add_file_logging, resolve_max_value)
from .dataset import (WholeImageCellDataset, SingleCellDataset, subsample)
from .backbone import load_model_from_bundle, load_ssl_backbone_from_bundle


# ----------------------------------------------------------------------------
# DB schema — single `inference` table + 3 lazily-created reduction tables.
# ----------------------------------------------------------------------------

def _resolve_gt(label_map, label_from_dir, abs_path, file_dir):
    """Resolve ground-truth label for a source file.

    label_map (path -> label) wins; else label_from_dir uses the parent
    directory basename; else None.
    """
    if abs_path and abs_path in label_map:
        return label_map[abs_path]
    if label_from_dir and file_dir:
        return os.path.basename(file_dir)
    return None

def _to_native(val):
    """Convert numpy/pandas types to native Python types for SQLite."""
    if val is None:
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


def _init_db(conn, mode, extra_cols=None):
    """Create the single `inference` table."""
    extra_cols = extra_cols or []
    cols = [
        "uid INTEGER PRIMARY KEY AUTOINCREMENT",
        "directory TEXT NOT NULL",
        "filename TEXT NOT NULL",
    ]
    if mode == "whole_image":
        cols.append("mask_filename TEXT")
        cols.append("label INTEGER NOT NULL DEFAULT 0")
    cols.append("ground_truth TEXT")
    for c in extra_cols:
        cols.append(f"{c} TEXT")
    cols.append("pred_class TEXT")
    cols.append("pred_prob REAL")
    cols.append("features BLOB")
    col_defs = ", ".join(cols)
    conn.execute(f"CREATE TABLE IF NOT EXISTS inference ({col_defs})")
    conn.commit()


def _write_db(db_path, meta_rows, all_logits, all_features,
              class_names, write_features, extra_cols=None, mode="single_cell",
              write_pred_class=False):
    """Write meta + predictions + features into the `inference` table.

    write_pred_class: write pred_class/pred_prob (classify-capable bundle only).
    write_features: write the features BLOB.
    """
    extra_cols = extra_cols or []
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    logger.info("Writing to %s", db_path)

    if write_pred_class and all_logits:
        logits_all = torch.cat(all_logits, dim=0)
        probs_all = torch.softmax(logits_all, dim=1)
        preds_all = probs_all.argmax(1).tolist()
    else:
        probs_all = None
        preds_all = None

    if write_features and all_features:
        feats_all = torch.cat(all_features, dim=0).numpy()
    else:
        feats_all = None

    conn = sqlite3.connect(db_path)
    _init_db(conn, mode, extra_cols)

    n = len(meta_rows)
    if write_pred_class and preds_all is not None and n != len(preds_all):
        print(
            f"Error: meta_rows length {n} != predictions length {len(preds_all)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if write_features and feats_all is not None and n != len(feats_all):
        print(
            f"Error: meta_rows length {n} != features length {len(feats_all)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Re-run protection: replace THIS dataset's rows instead of appending.
    # Scoped to the written directories so multiple datasets sharing one
    # output DB accumulate instead of overwriting each other. Rows with an
    # empty/missing directory (source path unavailable) must also be deleted,
    # or re-runs accumulate duplicates for them.
    dirs: set[str] = set()
    has_null_dir = False
    for m in meta_rows:
        d = m.get("directory")
        if d is None or d == "":
            has_null_dir = True
        else:
            dirs.add(str(d))
    if has_null_dir:
        conn.execute(
            "DELETE FROM inference WHERE directory IS NULL OR directory = ''")
    if dirs:
        ph = ", ".join("?" * len(dirs))
        existing = conn.execute(
            f"SELECT COUNT(*) FROM inference WHERE directory IN ({ph})",
            tuple(dirs)).fetchone()[0]
        if existing:
            logger.warning("inference table already has %d rows for this "
                           "dataset's directories; deleting them "
                           "(rows replaced on re-run)", existing)
            conn.execute(f"DELETE FROM inference WHERE directory IN ({ph})",
                         tuple(dirs))

    base_cols = ["directory", "filename"]
    if mode == "whole_image":
        base_cols.append("mask_filename")
        base_cols.append("label")
    base_cols.append("ground_truth")
    all_cols = base_cols + extra_cols + ["pred_class", "pred_prob", "features"]
    col_names = ", ".join(all_cols)
    placeholders = ", ".join("?" * len(all_cols))

    rows = []
    for i, meta in enumerate(meta_rows):
        if write_pred_class:
            pred_idx = int(preds_all[i])
            pred_class = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)
            pred_prob = float(probs_all[i, pred_idx])
        else:
            pred_class = None
            pred_prob = None
        feat_blob = feats_all[i].tobytes() if feats_all is not None else None
        row = [_to_native(meta.get(c)) for c in base_cols + extra_cols]
        row.extend([pred_class, pred_prob, feat_blob])
        rows.append(tuple(row))

    conn.executemany(
        f"INSERT INTO inference ({col_names}) VALUES ({placeholders})",
        rows,
    )
    wrote = "meta" + ("+pred" if write_pred_class else "") + ("+features" if write_features else "")
    logger.info("Wrote %d rows (%s) to %s", len(rows), wrote, db_path)

    conn.commit()
    conn.close()
    return db_path


# ----------------------------------------------------------------------------
# Forward pass
# ----------------------------------------------------------------------------

def _forward_pass(loader, model, device, write_features, classify_mode,
                  write_pred_class=False, pool_fn=None):
    """Run forward pass on the dataloader. Returns (all_logits, all_features).

    classify_mode=True: model is a classification Model; forward returns (logits, pooled).
    classify_mode=False: model is a backbone-only feature extractor; forward returns
        raw features; pool_fn is applied to get pooled features.
    write_pred_class: collect logits (only meaningful when classify_mode=True).
    write_features: collect features.
    """
    all_logits = []
    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer"):
            x = batch[0].to(device)
            if classify_mode:
                logits, features = model(x)
                if write_pred_class:
                    all_logits.append(logits.cpu())
            else:
                raw = model(x)
                features = pool_fn(raw) if pool_fn is not None else raw
                if features.ndim > 2:
                    features = features.mean(dim=(2, 3)) if features.ndim == 4 else features.mean(dim=1)
            if write_features:
                all_features.append(features.cpu())
    return all_logits, all_features


# ----------------------------------------------------------------------------
# Single-cell inference
# ----------------------------------------------------------------------------

def _validate_channel_count(n_avail, channels, meta, data_dir):
    """Hard-exit if the resolved data channel count mismatches the model's in_chans.

    Mirrors pretrain's in_chans-vs-data.channels check: the model architecture
    is fixed by the channel count it was built with, so a mismatch would
    otherwise crash mid-forward with a cryptic conv-shape error.
    """
    resolved = len(channels) if channels is not None else n_avail
    in_chans = meta.get("in_chans")
    if in_chans is not None and resolved != in_chans:
        print(
            f"Error: {data_dir}: data channels resolve to {resolved} "
            f"(channels={channels}, {n_avail} available) but the model bundle "
            f"was built with in_chans={in_chans}",
            file=sys.stderr,
        )
        sys.exit(1)


def _run_single_cell(data_dir, meta, model, device,
                     batch_size, db_path, write_features, write_pred_class,
                     label_map, label_from_dir,
                     channels, channel_layout, image_pattern, max_value,
                     classify_mode, pool_fn,
                     dl_num_workers=4, dl_prefetch_factor=2,
                     dl_persistent_workers=True,
                     sample_max=None, sample_by='per_class', seed=42):
    logger.info("Running single-cell inference on %s", data_dir)

    cell_ds = CellDataset(data_dir, channel_layout=channel_layout,
                          image_pattern=image_pattern)
    if len(cell_ds) == 0:
        print(f"Error: no TIFF files found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    _validate_channel_count(len(cell_ds.intensity_colnames), channels, meta, data_dir)

    _required_meta = ("augmentation_infer", "normalize_method", "normalize_with_masking",
                      "clip_low", "clip_high", "normalize_fixed_reference")
    _missing = [k for k in _required_meta if k not in meta]
    if _missing:
        print(f"Error: bundle meta missing required keys: {_missing}", file=sys.stderr)
        sys.exit(1)
    augmentation_spec = meta["augmentation_infer"]
    normalize_method = meta["normalize_method"]
    with_masking = meta["normalize_with_masking"]
    clip_low = meta["clip_low"]
    clip_high = meta["clip_high"]
    fixed_reference = bool(meta["normalize_fixed_reference"])

    md = cell_ds.metadata
    _exclude = {"stem", "path", "directory", "channel", "ext", "mask_name"}
    _exclude.update(cell_ds.intensity_colnames)
    extra_cols = [c for c in md.columns if c not in _exclude]

    entries = []
    for i in range(len(md)):
        row = md.iloc[i]
        path = row["path"]
        abs_path = os.path.abspath(path)
        file_dir = os.path.dirname(path).replace("\\", "/")
        gt = _resolve_gt(label_map, label_from_dir, abs_path, file_dir)
        entry = {
            "idx": i,
            "directory": file_dir,
            "filename": os.path.basename(path),
            "ground_truth": gt,
        }
        for col in extra_cols:
            entry[col] = row.get(col)
        entries.append(entry)

    logger.info("Loaded %d single-cell images (extra metadata cols: %s)",
                len(entries), extra_cols or "none")

    if sample_max is not None:
        entries = subsample(entries, sample_max, sample_by, seed,
                            label_key="ground_truth")
        logger.info("Sub-sampled to %d entries (sample_max=%s, sample_by=%s)",
                    len(entries), sample_max, sample_by)

    dataset = SingleCellDataset(
        [(cell_ds, e["idx"]) for e in entries],
        {"_": 0}, ["_"] * len(entries),
        channels=channels,
        augmentation_spec=augmentation_spec,
        normalize_method=normalize_method,
        clip_low=clip_low, clip_high=clip_high,
        with_masking=with_masking,
        fixed_reference=fixed_reference, max_value=max_value)
    loader_kwargs = dict(batch_size=batch_size, shuffle=False,
                         num_workers=dl_num_workers,
                         persistent_workers=dl_persistent_workers)
    if dl_num_workers > 0:
        loader_kwargs["prefetch_factor"] = dl_prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    all_logits, all_features = _forward_pass(
        loader, model, device, write_features, classify_mode,
        write_pred_class=write_pred_class, pool_fn=pool_fn)

    base_keys = ("directory", "filename", "ground_truth")
    meta_rows = [
        {k: e.get(k) for k in base_keys + tuple(extra_cols)}
        for e in entries
    ]
    class_names = meta.get("class_names", []) if classify_mode else []
    return _write_db(db_path, meta_rows, all_logits, all_features,
                     class_names, write_features, extra_cols=extra_cols,
                     mode="single_cell", write_pred_class=write_pred_class)


# ----------------------------------------------------------------------------
# Whole-image inference
# ----------------------------------------------------------------------------

def _run_whole_image(data_dir, image_pattern, mask_pattern, meta,
                     model, device, batch_size, db_path,
                     write_features, write_pred_class, image_subdir_pattern,
                     channel_layout, channels, label_from_dir, label_map,
                     mask_name_cfg, max_value,
                     classify_mode, pool_fn,
                     dl_num_workers=4, dl_prefetch_factor=2,
                     dl_persistent_workers=True,
                     sample_max=None, sample_by='per_class', seed=42):
    logger.info("Running whole-image inference on %s", data_dir)

    _required_meta = ("augmentation_infer", "normalize_method", "normalize_with_masking",
                      "clip_low", "clip_high", "normalize_fixed_reference")
    _missing = [k for k in _required_meta if k not in meta]
    if _missing:
        print(f"Error: bundle meta missing required keys: {_missing}", file=sys.stderr)
        sys.exit(1)
    augmentation_spec = meta["augmentation_infer"]
    normalize_method = meta["normalize_method"]
    with_masking = meta["normalize_with_masking"]
    clip_low = meta["clip_low"]
    clip_high = meta["clip_high"]
    fixed_reference = bool(meta["normalize_fixed_reference"])

    image_ds = ImageDataset(
        data_dir, image_pattern, mask_pattern,
        channel_layout=channel_layout,
        image_subdir_pattern=image_subdir_pattern or None,
    )
    if len(image_ds) == 0:
        print(f"Error: no whole-image + mask pairs found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    _validate_channel_count(len(image_ds.intensity_colnames), channels, meta, data_dir)

    mask_cols = image_ds.mask_colnames
    if not mask_cols:
        print(f"Error: no mask columns found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    if mask_name_cfg is not None:
        mask_name = f"mask_{mask_name_cfg}"
        if mask_name not in mask_cols:
            print(
                f"Error: mask_name '{mask_name_cfg}' -> column '{mask_name}' "
                f"not found. Available mask columns: {mask_cols}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        mask_name = mask_cols[0]
    logger.info("Using mask column: %s", mask_name)

    ds = WholeImageCellDataset(
        image_ds, mask_name, channels=channels,
        augmentation_spec=augmentation_spec,
        normalize_method=normalize_method, clip_low=clip_low, clip_high=clip_high,
        with_masking=with_masking, fixed_reference=fixed_reference, padding=4,
        max_value=max_value)

    logger.info("Whole-image: %d cells across %d fields", len(ds), len(image_ds))

    if sample_max is not None:
        entries = []
        for idx in range(len(ds)):
            row_idx, label = ds._flat_index[idx]
            source_path = ds.row_source_path(row_idx)
            abs_path = os.path.abspath(source_path) if source_path else None
            file_dir = os.path.dirname(source_path).replace("\\", "/") if source_path else ""
            gt = _resolve_gt(label_map, label_from_dir, abs_path, file_dir)
            entries.append({"idx": idx, "ground_truth": gt})
        entries = subsample(entries, sample_max, sample_by, seed,
                            label_key="ground_truth")
        ds.subsample(e["idx"] for e in entries)
        logger.info("Sub-sampled to %d cells", len(ds))

    loader_kwargs = dict(batch_size=batch_size, shuffle=False,
                         num_workers=dl_num_workers,
                         persistent_workers=dl_persistent_workers)
    if dl_num_workers > 0:
        loader_kwargs["prefetch_factor"] = dl_prefetch_factor
    loader = DataLoader(ds, **loader_kwargs)

    all_logits, all_features = _forward_pass(
        loader, model, device, write_features, classify_mode,
        write_pred_class=write_pred_class, pool_fn=pool_fn)

    md = image_ds.metadata
    _exclude = {"path", "stem", "__file__", "directory", "channel", "ext", "mask_name"}
    _exclude.update(image_ds.intensity_colnames)
    _exclude.update(image_ds.mask_colnames)
    extra_cols = [c for c in md.columns if c not in _exclude]

    meta_rows = []
    for idx in range(len(ds)):
        row_idx, label = ds._flat_index[idx]
        row = md.iloc[row_idx]
        ch_filenames = ds.row_channel_filenames(row_idx)
        filename_json = json.dumps(ch_filenames)
        mask_fname = ds.row_mask_filename(row_idx)
        source_path = ds.row_source_path(row_idx)
        abs_path = os.path.abspath(source_path) if source_path else None
        file_dir = os.path.dirname(source_path).replace("\\", "/") if source_path else ""
        gt = _resolve_gt(label_map, label_from_dir, abs_path, file_dir)
        meta_entry = {
            "directory": file_dir,
            "filename": filename_json,
            "label": int(label),
            "mask_filename": mask_fname,
            "ground_truth": gt,
        }
        for col in extra_cols:
            meta_entry[col] = row.get(col)
        meta_rows.append(meta_entry)

    class_names = meta.get("class_names", []) if classify_mode else []
    return _write_db(db_path, meta_rows, all_logits, all_features,
                     class_names, write_features, extra_cols=extra_cols,
                     mode="whole_image", write_pred_class=write_pred_class)


# ----------------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------------

def run_inference(config, config_path=None):
    """Inference (+ optional feature extraction) -> infer.db.

    Output is controlled by inference.pred_class and inference.feature:
      - pred_class (default = bundle is classify-capable): write pred_class/pred_prob.
        Explicitly requesting it with an SSL bundle logs a warning and skips.
      - feature (default true): write the features BLOB.
    """
    mode = config.get("mode", "single_cell")
    data_cfg = config["data"]
    inf_cfg = config["inference"]

    model_path = config["model"]
    if not isinstance(model_path, str) or not model_path:
        print("Error: config 'model' must be a string path to a model bundle "
              "(train model.pt or SSL model.pt)", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    logger.info("Loading model from %s", model_path)
    device = select_device()
    set_seed(42)
    bundle = torch.load(model_path, map_location=device, weights_only=False)
    meta = bundle["meta"]

    # Determine model mode: classify (train bundle) vs features (SSL bundle).
    # Train bundle: "state_dict" + "num_classes" in meta.
    # SSL bundle: full "state_dict" without "num_classes".
    if "state_dict" in bundle and "num_classes" in meta:
        classify_mode = True
        model = load_model_from_bundle(bundle, device)
        pool_fn = None  # Model.forward handles pooling
        logger.info("Classify mode: %d classes", meta["num_classes"])
    elif "state_dict" in bundle:
        classify_mode = False
        model, feat_dim, pool_fn, meta = load_ssl_backbone_from_bundle(bundle, device)
        logger.info("Features-only mode: feat_dim=%d", feat_dim)
    else:
        print("Error: bundle has no 'state_dict' key (unsupported pre-0.2.1 "
              "bundle format)", file=sys.stderr)
        sys.exit(1)

    # What to write (defaults: pred_class = bundle capability, feature = true).
    write_pred_class = bool(inf_cfg.get("pred_class", classify_mode))
    write_features = bool(inf_cfg.get("feature", True))
    if not classify_mode and write_pred_class:
        logger.warning("pred_class requested but the model bundle is an SSL "
                       "pretrain bundle (no classifier); writing features only")
        write_pred_class = False
    if not write_pred_class and not write_features:
        logger.warning("Both inference.pred_class and inference.feature are false; "
                       "writing metadata only")

    # All data.* settings must be explicit
    required_data_keys = [
        "root", "channels", "channel_layout", "image_pattern", "max_value",
        "label_from_dir", "label_csv", "sample_max", "sample_by",
    ]
    if mode == "whole_image":
        required_data_keys += ["mask_pattern", "image_subdir_pattern", "mask_name"]

    missing = [k for k in required_data_keys if k not in data_cfg]
    if missing:
        print(
            f"Error: missing data.* keys in inference config: {missing}. "
            f"All data.* settings must be explicit (null is allowed; "
            f"missing is not). Required for mode='{mode}': {required_data_keys}",
            file=sys.stderr,
        )
        sys.exit(1)

    data_roots = data_cfg["root"]
    channels_cfg = data_cfg["channels"]
    channel_layout_cfg = data_cfg["channel_layout"]
    image_pattern_cfg = data_cfg["image_pattern"]
    max_value = resolve_max_value(data_cfg)
    label_csv = data_cfg["label_csv"]
    label_from_dir = data_cfg["label_from_dir"]
    sample_max = data_cfg["sample_max"]
    sample_by = data_cfg["sample_by"]

    output_dir = config.get("output_dir")
    db_name = inf_cfg.get("db_name", "infer.db")
    batch_size = inf_cfg.get("batch_size", 128)
    seed = 42

    dl_cfg = config.get("dataloader", {})
    dl_num_workers = dl_cfg.get("num_workers", 4)
    dl_prefetch_factor = dl_cfg.get("prefetch_factor", 2)
    dl_persistent_workers = dl_cfg.get("persistent_workers", True) and dl_num_workers > 0

    label_map = {}
    if label_csv and os.path.exists(label_csv):
        label_map = load_label_csv(label_csv)

    out_pairs = resolve_output_paths(data_roots, output_dir)
    if output_dir:
        add_file_logging(output_dir)

    if config_path is not None:
        copy_targets = [output_dir] if output_dir else [r for r, _ in out_pairs]
        for target in copy_targets:
            copy_config_file(config_path, target)

    results = []
    for i, (data_dir, out_dir) in enumerate(out_pairs):
        logger.info("")
        if data_dir == out_dir:
            logger.info("Inference %d/%d on %s", i + 1, len(out_pairs), data_dir)
        else:
            logger.info("Inference %d/%d on %s -> %s", i + 1, len(out_pairs), data_dir, out_dir)
        os.makedirs(out_dir, exist_ok=True)

        db_path = os.path.join(out_dir, db_name)

        try:
            if mode == "single_cell":
                path = _run_single_cell(
                    data_dir, meta, model, device,
                    batch_size, db_path, write_features, write_pred_class,
                    label_map, label_from_dir,
                    channels_cfg, channel_layout_cfg, image_pattern_cfg, max_value,
                    classify_mode, pool_fn,
                    dl_num_workers=dl_num_workers, dl_prefetch_factor=dl_prefetch_factor,
                    dl_persistent_workers=dl_persistent_workers,
                    sample_max=sample_max, sample_by=sample_by, seed=seed)
            elif mode == "whole_image":
                path = _run_whole_image(
                    data_dir, image_pattern_cfg, data_cfg["mask_pattern"],
                    meta, model, device,
                    batch_size, db_path, write_features, write_pred_class,
                    data_cfg["image_subdir_pattern"],
                    channel_layout_cfg, channels_cfg,
                    label_from_dir, label_map,
                    data_cfg["mask_name"], max_value,
                    classify_mode, pool_fn,
                    dl_num_workers=dl_num_workers, dl_prefetch_factor=dl_prefetch_factor,
                    dl_persistent_workers=dl_persistent_workers,
                    sample_max=sample_max, sample_by=sample_by, seed=seed)
            else:
                print(f"Error: unknown inference mode: {mode}", file=sys.stderr)
                sys.exit(1)
            results.append(path)
        except Exception as e:
            print(f"Error: inference failed for {data_dir}: {e}", file=sys.stderr)
            sys.exit(1)

    return results
