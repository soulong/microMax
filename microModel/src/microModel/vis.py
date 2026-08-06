"""Visualization: multi-view augmentation preview, reduction plot, training plots.

show_augmentation adapts to both pretrain configs (augmentation_views) and
train configs (augmentation_train + augmentation_infer).
show_reduction fits PCA + UMAP on infer.db features, writes reduction
tables, saves plots.
"""

import os
import sys
import time
import sqlite3
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import umap

from microBase import (
    CellDataset,
    build_pipeline,
    apply,
    normalize,
)

from .dataset import _compute_ref_stats, _normalize_fixed, _to_float_max

from .utils import (
    logger,
    resolve_output_paths,
    shorten_labels,
    save_reducer,
    load_reducer,
    validate_pca,
    validate_umap_pipeline,
    add_file_logging,
    resolve_max_value,
    stratified_sample_indices,
)


# ----------------------------------------------------------------------------
# Augmentation preview — supports both pretrain (multi-view) and train configs
# ----------------------------------------------------------------------------

def show_augmentation(config):
    """Render augmentation preview.

    For pretrain configs (has 'augmentation_views'): renders ALL N views per
    sample by default; vis_augment.num_views is an optional explicit cap.
    For train configs (has 'augmentation_train' + 'augmentation_infer'):
    renders raw + eval + num_views augmented views per sample.
    """
    data_cfg = config["data"]
    norm_cfg = config.get("normalize", {})
    vis_cfg = config.get("vis_augment", {})

    num_samples = vis_cfg.get("num_samples", 1)
    num_views = vis_cfg.get("num_views", 8)

    root = data_cfg["root"]
    if isinstance(root, list):
        root = root[0]

    normalize_method = norm_cfg.get("method", "per_channel")
    with_masking = norm_cfg.get("with_masking", False)
    clip_low = norm_cfg.get("clip_low", 0.05)
    clip_high = norm_cfg.get("clip_high", 99.95)
    fixed_reference = norm_cfg.get("fixed_reference", False)
    max_value = resolve_max_value(data_cfg)
    channel_layout = data_cfg.get("channel_layout", "CHW")
    image_pattern = data_cfg.get("image_pattern")
    channels = data_cfg.get("channels")

    cell_ds = CellDataset(root, channel_layout=channel_layout, image_pattern=image_pattern)
    if len(cell_ds) == 0:
        print(f"Error: no TIFF files found in {root}", file=sys.stderr)
        sys.exit(1)

    if channels is None:
        n_avail = len(cell_ds.intensity_colnames)
        channels = list(range(1, n_avail + 1))
    else:
        channels = list(channels)

    # Determine config type: pretrain (augmentation_views) vs train
    aug_views_cfg = config.get("augmentation_views")
    is_pretrain = aug_views_cfg is not None

    if is_pretrain:
        # Pretrain: build all view pipelines. Show ALL configured views by
        # default (auto-adapts to augmentation_views); an explicit
        # vis_augment.num_views caps the grid (e.g. for many local views).
        view_pipelines = [build_pipeline(spec) for spec in aug_views_cfg]
        # "Raw" display must be deterministic: use augmentation_infer if
        # present, else the resize-only steps of the first view.
        aug_infer_cfg = config.get("augmentation_infer")
        if aug_infer_cfg:
            raw_pipeline = build_pipeline(aug_infer_cfg)
        else:
            _RESIZE_STEPS = {"LongestMaxSize", "PadIfNeeded", "Resize"}
            resize_spec = [s for s in aug_views_cfg[0] if next(iter(s)) in _RESIZE_STEPS]
            raw_pipeline = build_pipeline(resize_spec) if resize_spec else None
        n_views_to_show = len(view_pipelines)
        explicit = vis_cfg.get("num_views")
        if explicit is not None:
            n_views_to_show = max(1, min(int(explicit), n_views_to_show))
        view_names = [f"V#{i+1}" for i in range(n_views_to_show)]
        view_pipelines_to_show = view_pipelines[:n_views_to_show]
    else:
        # Train: build train + infer pipelines
        aug_train_cfg = config.get("augmentation_train", [])
        aug_infer_cfg = config.get("augmentation_infer", [])
        raw_pipeline = build_pipeline(aug_infer_cfg) if aug_infer_cfg else None
        train_pipeline = build_pipeline(aug_train_cfg) if aug_train_cfg else None
        n_views_to_show = num_views
        view_names = [f"V#{i+1}" for i in range(n_views_to_show)]
        view_pipelines_to_show = [train_pipeline] * n_views_to_show

    n_ch = len(channels)
    # Layout: Raw + N views per sample per channel
    n_cols = 1 + n_views_to_show
    n_rows = num_samples * n_ch
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Shared frame: one global x/y window across all samples/views so that
    # different view sizes (e.g. 224 global vs 112 local) are visually
    # apparent instead of each subplot auto-fitting its own image.
    max_h = max_w = 0

    sample_indices = random.sample(range(len(cell_ds)), min(num_samples, len(cell_ds)))
    logger.info("Showing %d cells x %d views (pretrain=%s, norm=%s, mask=%s, fixed=%s)",
                len(sample_indices), n_views_to_show, is_pretrain, normalize_method,
                with_masking, fixed_reference)

    for s, sample_idx in enumerate(sample_indices):
        cell_orig = _to_float_max(cell_ds.get_cell(sample_idx), max_value)
        if channels is not None:
            ch_idx = [c - 1 for c in channels]
            cell_orig = cell_orig[:, :, ch_idx]
        mask = (cell_orig != 0).any(axis=2).astype(np.uint8) if with_masking else None
        # Fixed-reference mode mirrors the training datasets: stats computed on
        # the raw cell, applied to every view (photometric aug survives).
        ref_stats = None
        if fixed_reference:
            ref_stats = _compute_ref_stats(
                cell_orig, None, with_masking, clip_low, clip_high, normalize_method)

        # Raw view: raw_pipeline + normalize
        raw_hwc, raw_mask = apply(raw_pipeline, cell_orig, mask)
        if ref_stats is not None:
            raw_norm = _normalize_fixed(raw_hwc, raw_mask, ref_stats)
        else:
            raw_norm = normalize(raw_hwc, mask=raw_mask, method=normalize_method,
                                 clip_low=clip_low, clip_high=clip_high)

        # Augmented views
        aug_views = []
        for pipe in view_pipelines_to_show:
            aug_hwc, aug_mask = apply(pipe, cell_orig, mask)
            if ref_stats is not None:
                aug_hwc = _normalize_fixed(aug_hwc, aug_mask, ref_stats)
            else:
                aug_hwc = normalize(aug_hwc, mask=aug_mask, method=normalize_method,
                                    clip_low=clip_low, clip_high=clip_high)
            aug_views.append((aug_hwc, aug_mask))

        all_views = [(raw_norm, raw_mask)] + aug_views
        all_names = ["Raw"] + view_names

        for img, _ in all_views:
            max_h = max(max_h, img.shape[0])
            max_w = max(max_w, img.shape[1])

        for c in range(n_ch):
            row = s * n_ch + c

            # Shared display range (all z-scored)
            fgs = []
            for img, vmask in all_views:
                ch_data = img[:, :, c]
                if vmask is not None:
                    fg = ch_data[vmask.astype(bool)]
                else:
                    fg = ch_data.reshape(-1)
                if fg.size > 0:
                    fgs.append(fg)
            if fgs:
                pooled = np.concatenate(fgs)
                shared_min, shared_max = np.percentile(pooled, [0.1, 99.9])
            else:
                shared_min, shared_max = -3.0, 3.0

            for v_idx, ((img, view_mask), name) in enumerate(zip(all_views, all_names)):
                ch_data = img[:, :, c]
                _imshow(axes[row][v_idx], ch_data, f"{name} ch{channels[c]}",
                        vmin=shared_min, vmax=shared_max, mask=view_mask)

    # Shared frame across all samples/views — views keep aspect="equal", so a
    # 112px local view renders visibly smaller than a 224px global view.
    for ax in axes.flat:
        ax.set_xlim(0, max_w)
        ax.set_ylim(0, max_h)

    plt.suptitle(f"Augmentation Preview: {len(sample_indices)} cells "
                 f"({'pretrain' if is_pretrain else 'train'}) — shared frame {max_w}x{max_h}",
                 fontsize=11)
    fig.subplots_adjust(top=0.93)
    plt.tight_layout()

    output_dir = config.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    if config.get("output_dir"):
        add_file_logging(output_dir)
    aug_path = os.path.join(output_dir, "augment_preview.pdf")
    fig.savefig(aug_path, dpi=150, bbox_inches="tight")
    logger.info("Augmentation preview saved to %s", aug_path)
    plt.close(fig)


def _imshow(ax, data, title, cmap="gray", vmin=None, vmax=None, mask=None):
    if vmin is None or vmax is None:
        fg = data[data != 0]
        if fg.size > 0:
            vmin, vmax = np.percentile(fg, [1, 99])
        else:
            vmin, vmax = 0.0, 1.0
    d = data.astype(np.float64) if not np.issubdtype(data.dtype, np.floating) else data
    if vmax > vmin:
        disp = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    else:
        disp = d.clip(0, 1)
    if mask is not None:
        disp = disp.copy()
        disp[~mask.astype(bool)] = 0
    ax.imshow(disp, cmap=cmap, aspect="equal", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7)


# ----------------------------------------------------------------------------
# Reduction view (PCA + UMAP)
# ----------------------------------------------------------------------------

def _plot_reduction_scatter(X, labels, label_names, title, xlabel, ylabel,
                            save_path, pred_probs=None, continuous=False):
    fig, ax = plt.subplots(figsize=(9, 6))

    if continuous:
        prob_arr = np.array(pred_probs, dtype=np.float64) if pred_probs is not None else np.array(labels, dtype=np.float64)
        sort_idx = np.argsort(prob_arr)
        X_plot = X[sort_idx]
        sc = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=prob_arr[sort_idx],
                        cmap="viridis", alpha=0.8, s=10, edgecolors="none", vmin=0, vmax=1)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Prediction Probability")
    else:
        unique_classes = sorted(set(labels))
        centroids = np.array([X[np.array(labels) == cls].mean(axis=0) for cls in unique_classes])
        n_cls = len(unique_classes)
        if n_cls > 1:
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(centroids)
            neighbor_mask = np.zeros_like(dists, dtype=bool)
            k = min(5, n_cls - 2)
            for i in range(n_cls):
                neighbor_mask[i, np.argpartition(dists[i], k + 1)[1:k + 1]] = True
            neighbor_mask |= neighbor_mask.T

        palette = []
        for cmap_name in ['tab20', 'tab20b']:
            cmap_obj = plt.colormaps[cmap_name]
            palette.extend([cmap_obj(i) for i in range(min(cmap_obj.N, 20))])

        assigned_color = {}
        sizes = [np.sum(np.array(labels) == cls) for cls in unique_classes]
        order = sorted(range(n_cls), key=lambda i: sizes[i], reverse=True)
        for idx in order:
            cls = unique_classes[idx]
            if n_cls > 1:
                neighbor_colors = [assigned_color[unique_classes[nb]]
                                   for nb in range(n_cls)
                                   if neighbor_mask[idx, nb] and unique_classes[nb] in assigned_color]
            else:
                neighbor_colors = []
            if not neighbor_colors:
                assigned_color[cls] = palette[len(assigned_color) % len(palette)]
            else:
                best_c = None
                best_dist = -1.0
                for pc in palette:
                    pc_arr = np.array(pc[:3])
                    min_d = min(np.sqrt(np.sum((pc_arr - np.array(nc[:3])) ** 2)) for nc in neighbor_colors)
                    if min_d > best_dist:
                        best_dist = min_d
                        best_c = pc
                assigned_color[cls] = best_c

        label_name_map = {}
        for i in range(len(labels)):
            lab = labels[i]
            if lab not in label_name_map:
                label_name_map[lab] = label_names[i]

        if pred_probs is not None:
            prob_arr = np.array(pred_probs, dtype=np.float64)
            sort_idx = np.argsort(prob_arr)
            X_plot = X[sort_idx]
            colors_plot = [assigned_color[labels[i]] for i in sort_idx]
        else:
            X_plot = X
            colors_plot = [assigned_color[l] for l in labels]

        ax.scatter(X_plot[:, 0], X_plot[:, 1],
                   c=colors_plot, alpha=0.8, s=10, edgecolors='none')

        from matplotlib.patches import Patch
        legend_handles = [Patch(color=assigned_color[cls], label=label_name_map[cls]) for cls in unique_classes]
        ax.legend(handles=legend_handles, fontsize=7, markerscale=3, loc="best")

        for cls_idx in unique_classes:
            mask = np.array(labels) == cls_idx
            center = X[mask].mean(axis=0)
            ax.annotate(label_name_map[cls_idx], center, fontsize=9, weight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Reduction plot saved to %s", save_path)
    plt.close(fig)


def show_reduction(config):
    inf_cfg = config["inference"]
    red_cfg = config.get("reduction", {})

    data_roots = config["data"]["root"]
    base_output_dir = config.get("output_dir")
    db_name = inf_cfg.get("db_name", "infer.db")
    seed = 42

    color_by_vals = red_cfg.get("color_by", "pred_class")
    if isinstance(color_by_vals, str):
        color_by_vals = [color_by_vals]
    valid_color_by = {"pred_class", "directory", "pred_prob", "ground_truth"}
    color_by_vals = [cb for cb in color_by_vals if cb is not None]
    filtered = [cb for cb in color_by_vals if cb in valid_color_by]
    dropped = [cb for cb in color_by_vals if cb not in valid_color_by]
    if dropped:
        logger.warning("Unknown color_by %s; dropping (valid: %s)", dropped, sorted(valid_color_by))
    color_by_vals = filtered
    if not color_by_vals:
        logger.error("No valid color_by values; aborting reduction.")
        return
    first_cb = color_by_vals[0]

    save_dir = base_output_dir if base_output_dir else os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    if base_output_dir:
        add_file_logging(save_dir)

    # Phase 1: load all DBs and merge features
    out_pairs = resolve_output_paths(data_roots, base_output_dir)
    seen = set()
    db_entries = []
    for data_dir, output_dir in out_pairs:
        db_path = os.path.join(output_dir, db_name)
        if db_path in seen:
            continue
        seen.add(db_path)
        if not os.path.exists(db_path):
            logger.warning("Database not found: %s. Run inference first.", db_path)
            continue
        feats, dicts = _load_inference_features(db_path)
        if feats is None:
            continue
        logger.info("Loaded %d feature vectors (dim=%d) from %s",
                    feats.shape[0], feats.shape[1], db_path)
        db_entries.append((db_path, feats, dicts))

    if not db_entries:
        logger.error("No data loaded from any DB; aborting reduction.")
        return

    feats_merged = np.concatenate([e[1] for e in db_entries], axis=0)
    dicts_merged = []
    for _, _, dicts in db_entries:
        dicts_merged.extend(dicts)
    logger.info("Merged %d feature vectors from %d DB(s)",
                feats_merged.shape[0], len(db_entries))

    # Phase 2: sample subset for fitting
    sample_per_class = red_cfg.get("sample_per_class", 1000)
    reducer_pca_path = red_cfg.get("reducer_pca")
    reducer_umap_path = red_cfg.get("reducer_umap")
    has_prefitted = bool(reducer_pca_path or reducer_umap_path)

    dirs_merged = [d["directory"] for d in dicts_merged]
    probs_merged = [d.get("pred_prob") or 0.0 for d in dicts_merged]
    first_labels_merged, _ = _extract_color_data(first_cb, dicts_merged, dirs_merged, probs_merged)

    if sample_per_class > 0 and not has_prefitted:
        fit_indices = _sample_fit_indices(
            len(dicts_merged), first_labels_merged, first_cb, sample_per_class, seed)
        feats_fit = feats_merged[fit_indices]
        dicts_fit = [dicts_merged[i] for i in fit_indices]
        dirs_fit = [dirs_merged[i] for i in fit_indices]
        probs_fit = [probs_merged[i] for i in fit_indices]
        if len(fit_indices) < len(dicts_merged):
            logger.info("Sampled %d of %d rows for reducer fitting (sample_per_class=%d)",
                        len(fit_indices), len(dicts_merged), sample_per_class)
    else:
        feats_fit, dicts_fit = feats_merged, dicts_merged
        dirs_fit, probs_fit = dirs_merged, probs_merged

    if len(dicts_fit) < 2:
        logger.error("Too few samples for reduction view (< 2), got %d", len(dicts_fit))
        return

    # Phase 3: fit (or load) PCA + UMAP
    var_threshold = red_cfg.get("var_threshold", 0.95)

    if reducer_pca_path:
        if not os.path.exists(reducer_pca_path):
            print(f"Error: reducer_pca not found: {reducer_pca_path}", file=sys.stderr)
            sys.exit(1)
        pca_full = load_reducer(reducer_pca_path)
        validate_pca(pca_full, feats_merged.shape[1])
        logger.info("PCA: loaded reducer from %s", reducer_pca_path)
    else:
        pca_full = PCA(n_components=var_threshold)
        pca_full.fit(feats_fit)
        if pca_full.n_components_ < 2:
            pca_full = PCA(n_components=2).fit(feats_fit)
        save_reducer(pca_full, os.path.join(save_dir, "reducer_pca.pkl"))
        logger.info("PCA %dD: explained variance ratio = %.4f",
                    pca_full.n_components_, pca_full.explained_variance_ratio_.sum())

    raw_var = pca_full.explained_variance_
    var_ratio = pca_full.explained_variance_ratio_
    cum_ratio = np.cumsum(var_ratio)
    n_pca = pca_full.n_components_
    pc_axis_labels = [f"PC {i+1} ({var_ratio[i]*100:.1f}%)" for i in range(n_pca)]

    if reducer_umap_path:
        if not os.path.exists(reducer_umap_path):
            print(f"Error: reducer_umap not found: {reducer_umap_path}", file=sys.stderr)
            sys.exit(1)
        umap_pipeline = load_reducer(reducer_umap_path)
        validate_umap_pipeline(umap_pipeline, feats_merged.shape[1])
        pca_pre = umap_pipeline.get("pca_pre")
        reducer = umap_pipeline["umap"]
        logger.info("UMAP: loaded reducer from %s", reducer_umap_path)
    else:
        n_pre = pca_full.n_components_
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)
            if n_pre > 2:
                pca_pre = PCA(n_components=n_pre)
                pca_pre.fit(feats_fit)
                logger.info("UMAP preprocessing PCA: %d -> %d components, explained variance = %.4f",
                            feats_fit.shape[1], n_pre, pca_pre.explained_variance_ratio_.sum())
            else:
                pca_pre = None
            reducer = umap.UMAP(random_state=seed)
            reducer.fit(pca_pre.transform(feats_fit) if pca_pre is not None else feats_fit)
            save_reducer({"pca_pre": pca_pre, "umap": reducer},
                         os.path.join(save_dir, "reducer_umap.pkl"))
        logger.info("UMAP: fitted on %d points", feats_fit.shape[0])

    # Phase 4: transform fit subset + plot
    X_pca_fit = pca_full.transform(feats_fit)
    X_pca_pre_fit = pca_pre.transform(feats_fit) if pca_pre is not None else feats_fit
    X_umap_fit = reducer.transform(X_pca_pre_fit)

    for cb in color_by_vals:
        labels, label_names = _extract_color_data(cb, dicts_fit, dirs_fit, probs_fit)
        continuous = (cb == "pred_prob")
        _plot_reduction_scatter(
            X_pca_fit[:, :2], labels, label_names,
            f"PCA of feature vectors (colored by {cb})",
            pc_axis_labels[0], pc_axis_labels[1],
            os.path.join(save_dir, f"feature_pca_{cb}.pdf"),
            pred_probs=probs_fit, continuous=continuous)
        _plot_reduction_scatter(
            X_umap_fit, labels, label_names,
            f"UMAP of feature vectors (colored by {cb})", "UMAP 1", "UMAP 2",
            os.path.join(save_dir, f"feature_umap_{cb}.pdf"),
            pred_probs=probs_fit, continuous=continuous)

    # Phase 5: batch-transform all DBs + write tables
    pc_cols = ", ".join(f"pc_{i+1} REAL NOT NULL" for i in range(n_pca))
    pc_col_names = ", ".join(f"pc_{i+1}" for i in range(n_pca))
    pc_ph = ", ".join("?" * n_pca)

    feats_merged_all = np.concatenate([e[1] for e in db_entries], axis=0)
    offsets = []
    _cursor = 0
    for _, feats_all, _ in db_entries:
        offsets.append((_cursor, _cursor + len(feats_all)))
        _cursor += len(feats_all)

    t_tf0 = time.perf_counter()
    X_pca_merged = pca_full.transform(feats_merged_all)
    t_pca_full = time.perf_counter()

    if pca_pre is not None and _pca_equivalent(pca_full, pca_pre):
        X_pca_pre_merged = X_pca_merged
    else:
        X_pca_pre_merged = (pca_pre.transform(feats_merged_all)
                            if pca_pre is not None else feats_merged_all)
    t_pca_pre = time.perf_counter()

    X_umap_merged = reducer.transform(X_pca_pre_merged)
    t_umap = time.perf_counter()
    logger.info("Batch transform on %d points: pca=%.1fs pca_pre=%.1fs umap=%.1fs",
                len(feats_merged_all),
                t_pca_full - t_tf0, t_pca_pre - t_pca_full, t_umap - t_pca_pre)

    for (start, end), (db_path, feats_all, db_dicts_all) in zip(offsets, db_entries):
        t_db0 = time.perf_counter()
        X_pca_all = X_pca_merged[start:end]
        X_umap_all = X_umap_merged[start:end]
        uids = [d["uid"] for d in db_dicts_all]

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")

        # The pc_1..pc_k schema depends on the auto-selected component count,
        # which can change between runs — drop before recreating.
        conn.execute("DROP TABLE IF EXISTS reduction_pca")
        conn.execute("DROP TABLE IF EXISTS reduction_pca_variance")

        conn.execute(
            f"CREATE TABLE IF NOT EXISTS reduction_pca ("
            f"uid INTEGER PRIMARY KEY, {pc_cols})")
        pca_cols = X_pca_all.T.tolist()
        conn.executemany(
            f"INSERT OR REPLACE INTO reduction_pca (uid, {pc_col_names}) "
            f"VALUES (?, {pc_ph})",
            list(zip(uids, *pca_cols)))

        conn.execute("""CREATE TABLE IF NOT EXISTS reduction_umap (
            uid INTEGER PRIMARY KEY,
            umap_1 REAL NOT NULL,
            umap_2 REAL NOT NULL)""")
        umap_cols = X_umap_all.T.tolist()
        conn.executemany(
            "INSERT OR REPLACE INTO reduction_umap (uid, umap_1, umap_2) VALUES (?, ?, ?)",
            list(zip(uids, *umap_cols)))

        conn.execute("""CREATE TABLE IF NOT EXISTS reduction_pca_variance (
            pc_id INTEGER PRIMARY KEY,
            variance REAL NOT NULL,
            variance_ratio REAL NOT NULL,
            cumulative_variance_ratio REAL NOT NULL)""")
        conn.executemany(
            "INSERT OR REPLACE INTO reduction_pca_variance "
            "(pc_id, variance, variance_ratio, cumulative_variance_ratio) VALUES (?, ?, ?, ?)",
            [(i + 1, float(raw_var[i]), float(var_ratio[i]), float(cum_ratio[i]))
             for i in range(n_pca)])

        conn.commit()
        conn.close()
        t_db1 = time.perf_counter()
        logger.info("Wrote %d PCA (%dD) + %d UMAP (2D) + %d variance rows to %s "
                    "(db_write=%.1fs)",
                    len(feats_all), n_pca, len(feats_all), n_pca, db_path,
                    t_db1 - t_db0)

    logger.info("Reduction complete: fitted on %d samples, transformed %d DB(s), "
                "reducers+plots saved to %s",
                feats_fit.shape[0], len(db_entries), save_dir)


def _load_inference_features(db_path):
    """Load all rows from the inference table. Returns (feats, dicts) or (None, None)."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='inference'")
        if not cur.fetchone():
            logger.error("No 'inference' table in %s. Run inference first.", db_path)
            return None, None
        cur = conn.execute("PRAGMA table_info(inference)")
        all_cols = [row[1] for row in cur.fetchall()]
        col_select = ", ".join(all_cols)
        cur = conn.execute(f"SELECT {col_select} FROM inference ORDER BY uid")
        col_names = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        if not rows:
            logger.error("No rows in inference table in %s.", db_path)
            return None, None
        dicts = [dict(zip(col_names, r)) for r in rows]
        feats_arr = []
        for d in dicts:
            f = d["features"]
            if f is None:
                print(
                    f"Error: row uid={d.get('uid')} has NULL features. "
                    "Run inference with inference.feature: true",
                    file=sys.stderr,
                )
                sys.exit(1)
            feats_arr.append(np.frombuffer(f, dtype=np.float32))
        feats = np.stack(feats_arr, axis=0)
        return feats, dicts
    finally:
        conn.close()


def _extract_color_data(cb, dicts, dirs, probs):
    """Extract (raw_values, display_names) for a color_by variable."""
    if cb == "pred_class":
        return ([d.get("pred_class") or "unknown" for d in dicts],
                [d.get("pred_class") or "unknown" for d in dicts])
    if cb == "directory":
        labs = shorten_labels(dirs)
        return labs, labs
    if cb == "pred_prob":
        return probs, [f"{p:.2f}" for p in probs]
    if cb == "ground_truth":
        labs = [d.get("ground_truth") or "__unlabeled__" for d in dicts]
        return labs, labs
    return ([d.get("pred_class") or "unknown" for d in dicts],
            [d.get("pred_class") or "unknown" for d in dicts])


def _sample_fit_indices(n_orig, labels, color_by, sample_per_class, seed):
    """Select indices for reducer fitting, stratified by `labels`."""
    return stratified_sample_indices(
        n_orig, labels, sample_per_class, seed,
        uniform=(color_by == "pred_prob"))


def _pca_equivalent(a, b):
    """True if two fitted PCA objects produce identical transforms."""
    if a is b:
        return True
    if (getattr(a, "n_components", None) != getattr(b, "n_components", None)
            or getattr(a, "n_features_in_", None) != getattr(b, "n_features_in_", None)):
        return False
    ca = getattr(a, "components_", None)
    cb = getattr(b, "components_", None)
    if ca is None or cb is None:
        return False
    return np.array_equal(ca, cb)


# ----------------------------------------------------------------------------
# Training plots (for train)
# ----------------------------------------------------------------------------

def plot_training_results(model, device, val_loader, num_classes, label_to_idx,
                          train_loss_history, val_acc_history, save_path=None):
    if not train_loss_history:
        logger.warning("No training history to plot.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(train_loss_history, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)

    axes[1].plot(val_acc_history, marker="s", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True)

    model.eval()
    yt_all, yp_all = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits, _ = model(x)
            yt_all.extend(y.tolist())
            yp_all.extend(logits.argmax(1).cpu().tolist())
    cm = confusion_matrix(yt_all, yp_all, labels=list(range(num_classes)))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    im = axes[2].imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    axes[2].set_title("Confusion Matrix")
    cls_names = list(label_to_idx.keys())
    axes[2].set_xticks(range(num_classes), cls_names, rotation=45)
    axes[2].set_yticks(range(num_classes), cls_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            axes[2].text(j, i, f"{val:.2f}",
                         ha="center", va="center",
                         color="white" if val > 0.5 else "black")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training plots saved to %s", save_path)

    plt.close(fig)

    from sklearn.metrics import accuracy_score, f1_score
    final_acc = float(accuracy_score(yt_all, yp_all))
    final_f1 = float(f1_score(yt_all, yp_all, average="macro", zero_division=0))
    return {"cm": cm, "final_loss": train_loss_history[-1],
            "final_acc": val_acc_history[-1], "final_f1": final_f1,
            "final_val_acc": final_acc, "final_val_f1": final_f1}


# ----------------------------------------------------------------------------
# Pretrain loss curve (for SSL pretrain)
# ----------------------------------------------------------------------------

def plot_pretrain_loss(loss_history, dino_history=None, ibot_history=None,
                       koleo_history=None, save_path=None):
    """Plot pretrain loss curve. For DINOv2, also plot dino/ibot/koleo components.

    loss_history: list of per-epoch total loss.
    dino_history/ibot_history/koleo_history: optional per-epoch component losses
        (DINOv2 only); must be the same length as loss_history to be plotted.
    save_path: output PDF path. If None, the figure is not saved.
    """
    if not loss_history:
        logger.warning("No pretrain loss history to plot.")
        return None

    epochs = range(1, len(loss_history) + 1)
    has_components = (dino_history is not None and len(dino_history) == len(loss_history)
                      and ibot_history is not None and len(ibot_history) == len(loss_history)
                      and koleo_history is not None and len(koleo_history) == len(loss_history))

    if has_components:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, loss_history, label="total", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Total Loss")
        axes[0].grid(True)
        axes[0].legend()
        axes[1].plot(epochs, dino_history, label="dino", linewidth=1.2)
        axes[1].plot(epochs, ibot_history, label="ibot", linewidth=1.2)
        axes[1].plot(epochs, koleo_history, label="koleo", linewidth=1.2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("DINOv2 Loss Components")
        axes[1].grid(True)
        axes[1].legend()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, loss_history, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Pretrain Loss")
        ax.grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Pretrain loss curve saved to %s", save_path)
    plt.close(fig)
    return save_path
