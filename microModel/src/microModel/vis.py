"""Visualization: multi-view augmentation preview, reduction plot, training plots.

show_augmentation adapts to both pretrain configs (augmentation_views) and
train configs (augmentation_train + augmentation_infer).
show_reduction fits the configured DR methods (pca/umap/pacmap/localmap) on
infer.db features, writes one reduction_<method> table per method plus the
optional find_cluster table, and saves one multi-page PDF per DR method.
"""

import os
import sys
import json
import time
import sqlite3
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
import igraph as ig
import leidenalg
import umap
import pacmap

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
    sql_ident,
    validate_pca,
    validate_umap_pipeline,
    add_file_logging,
    resolve_max_value,
    stratified_sample_indices,
    parse_pred_prob,
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
# Reduction view (pca / umap / pacmap / localmap + optional clustering)
# ----------------------------------------------------------------------------

#: Canonical DR method order — `reduction.method` is filtered against this.
DR_METHODS = ("pca", "umap", "pacmap", "localmap")

#: Pretty axis prefix per DR method (plot titles / axis labels).
DR_AXIS_NAMES = {"pca": "PC", "umap": "UMAP", "pacmap": "PaCMAP",
                 "localmap": "LocalMAP"}

#: PCA component count feeding UMAP (visualization only — a fixed 50 keeps
#: the embedding input stable; the clustering space has its own
#: variance-based rule below).
UMAP_PRE_COMPONENTS = 50

#: Whitened clustering space dimensionality: the smallest component count k
#: covering CLUSTER_VARIANCE_TARGET of the feature variance (adaptive across
#: backbones/feature dims, measured on the fit subset), capped at
#: CLUSTER_MAX_COMPONENTS. The cap matters because whiten=True rescales EVERY
#: retained component to unit variance — an uncapped tail (e.g. 95% of a
#: 384-d DINOv3 embedding needs ~133 comps) would noise-amplify directions
#: carrying ~0.1% variance each to the same footing as the signal.
CLUSTER_VARIANCE_TARGET = 0.95
CLUSTER_MAX_COMPONENTS = 100

#: Neighbors per point in the kNN graph feeding Leiden clustering; the same
#: k is the vote pool when a baseline cluster.pkl predicts new data.
LEIDEN_N_NEIGHBORS = 15

#: Default scatter color when color_by is null (light blue).
SINGLE_COLOR = "#87CEEB"

#: Images-per-cluster / clusters-per-row in the cluster contact sheet.
CONTACT_SHEET_PER_CLUSTER = 8
CONTACT_SHEET_GROUPS_PER_ROW = 5

#: Sheet representatives are drawn RANDOMLY from the densest
#: CONTACT_SHEET_DENSITY_KEEP fraction of each cluster (local density =
#: mean distance to the nearest reference members). The sparse tail is
#: outlier morphology, not the cluster's typical look; the random draw
#: inside the dense core keeps natural variety instead of near-duplicates
#: of the centroid.
CONTACT_SHEET_DENSITY_KEEP = 0.75

#: Reference-subsample cap for the density estimate (exact within-cluster
#: kNN density via cKDTree costs seconds on 18k-member clusters; distances
#: to a fixed random subsample rank members identically at a fraction of
#: the cost).
CONTACT_SHEET_DENSITY_REF = 2000

#: Fixed RNG seed for the sheet sampling (per-cluster streams derive from
#: it), so regenerated sheets are reproducible.
CONTACT_SHEET_SEED = 42

#: Minimum foreground (nonzero-pixel) fraction for a sheet representative.
#: Near-empty segmentation slivers carry almost no information, so their
#: generic DINOv3 features land right at a cluster centroid and would
#: otherwise dominate the nearest-centroid representative slots. 10% keeps
#: those (and just-above-threshold sparse fragments) out; normal cells sit
#: far above it (dataset median ~42%).
CONTACT_SHEET_MIN_FOREGROUND = 0.10

#: Cap on candidates scanned per cluster while skipping degenerate crops.
CONTACT_SHEET_MAX_SCAN = 60

#: Max cluster count that still gets a representative-cell contact sheet —
#: Leiden at high resolution can return hundreds of micro-clusters, which
#: would blow up the sheet layout (and PDF page size).
CONTACT_SHEET_MAX_CLUSTERS = 60

#: A color_by column with more distinct numeric values than this renders as
#: a continuous viridis page instead of one category per value (which would
#: explode the legend/annotations). Integer ID columns that must stay
#: categorical regardless (cluster_res<tag>) are forced so in
#: _color_column_continuous.
CONTINUOUS_COLOR_MAX_CLASSES = 20


def _foreground_fraction(img_u8):
    """Fraction of pixels with any nonzero channel in a (H, W, 3) uint8 image.

    Zeros stay exactly zero through normalization, so this equals the raw
    crop's foreground fraction.
    """
    if img_u8 is None:
        return 0.0
    return float(np.any(img_u8 > 0, axis=2).mean())


def _build_palette():
    """60-entry categorical palette (tab20 + tab20b + tab20c)."""
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap_obj = plt.colormaps[cmap_name]
        palette.extend([cmap_obj(i) for i in range(cmap_obj.N)])
    return palette


def _res_tag(res):
    """Compact resolution tag for table columns / file names (1.0 -> '1')."""
    return f"{float(res):g}"


def _leiden_partition(W, resolution, seed):
    """Leiden communities over a kNN graph of W (one 0-based label per row).

    Each point is connected to its LEIDEN_N_NEIGHBORS nearest neighbors; the
    union of the directed kNN edges forms an undirected unweighted graph.
    Leiden's RBConfiguration model puts `resolution` in the modularity term:
    higher resolutions yield more, tighter clusters, and the cluster count
    emerges from the data instead of being configured.
    """
    n_neighbors = min(LEIDEN_N_NEIGHBORS, W.shape[0] - 1)
    A = kneighbors_graph(W, n_neighbors, mode="connectivity", include_self=False)
    A = A.maximum(A.T).tocoo()  # undirected graph: keep an edge from either side
    g = ig.Graph(n=W.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())))
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=float(resolution), seed=seed)
    return np.asarray(part.membership, dtype=int)


def _plot_reduction_page(X, title, xlabel, ylabel, pdf, labels=None,
                         label_names=None, pred_probs=None, continuous=False,
                         cont_values=None, cont_range=None, cont_label=""):
    """Write one DR scatter page into an open PdfPages.

    Data points are rasterized (non-editable bitmap, rendered at the dpi
    given in pdf.savefig); axes, ticks, legend text and annotations stay
    editable vector objects. The legend shows colored TEXT only (no marker
    dots). labels=None draws every point in the default light blue with no
    legend/annotations (the color_by: null case).

    continuous=True renders a viridis-colored page from cont_values (any
    numeric color_by column) with the cont_range color limits and a colorbar
    titled cont_label; pred_probs still controls draw order (low first, so
    high-confidence points stay on top).
    """
    # Plot region 1.2x wider than tall. set_box_aspect takes HEIGHT/WIDTH,
    # so the wider-than-square box asks for 1/1.2; the figsize leaves the
    # width for it.
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    ax.set_box_aspect(1 / 1.2)

    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], c=SINGLE_COLOR, alpha=0.8, s=10,
                   edgecolors="none", rasterized=True)
    elif continuous:
        if cont_values is None:
            raise ValueError("continuous scatter requires cont_values")
        vals = np.asarray(cont_values, dtype=np.float64)
        # Draw ascending (NaN/missing first) so the largest values stay on top.
        order = np.argsort(np.where(np.isnan(vals), -np.inf, vals))
        if cont_range is not None:
            cmin, cmax = cont_range
        else:
            cmin, cmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        sc = ax.scatter(X[order, 0], X[order, 1], c=vals[order],
                        cmap="viridis", alpha=0.8, s=10, edgecolors="none",
                        vmin=cmin, vmax=cmax, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cont_label or "value")
    else:
        labels_arr = np.asarray(labels)
        unique_classes = sorted(set(labels))
        n_cls = len(unique_classes)
        # Beyond 60 categories (e.g. coloring by a high-cardinality column
        # like filename) the legend and centroid annotations would drown in
        # noise and the pairwise centroid distances below would not fit
        # memory — fall back to plain palette cycling with no annotations.
        huge = n_cls > 60
        if 1 < n_cls <= 60:
            centroids = np.array([X[labels_arr == cls].mean(axis=0)
                                  for cls in unique_classes])
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(centroids)
            neighbor_mask = np.zeros_like(dists, dtype=bool)
            k = min(5, n_cls - 2)
            for i in range(n_cls):
                neighbor_mask[i, np.argpartition(dists[i], k + 1)[1:k + 1]] = True
            neighbor_mask |= neighbor_mask.T

        palette = _build_palette()
        # Assign colors largest-class-first, always picking the palette entry
        # most distant from the colors already used by spatially neighboring
        # classes (centroids in the DR plot) so close classes stay distinct.
        assigned_color = {}
        sizes = [np.sum(labels_arr == cls) for cls in unique_classes]
        order = sorted(range(n_cls), key=lambda i: sizes[i], reverse=True)
        for idx in order:
            cls = unique_classes[idx]
            if huge:
                assigned_color[cls] = palette[len(assigned_color) % len(palette)]
                continue
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
                   c=colors_plot, alpha=0.8, s=10, edgecolors='none',
                   rasterized=True)

        # Legend = colored text only (invisible handles carry no marker).
        # With many classes the tall legend dwarfs the plot and per-class
        # annotations overlap, so BOTH scale their font down with the class
        # count instead of disappearing; beyond 60 classes the legend is
        # dropped entirely (the centroid annotations still label everything).
        if n_cls <= 30:
            leg_fs, ann_fs, ann_pad = 7, 9, 0.3
        elif n_cls <= 60:
            leg_fs, ann_fs, ann_pad = 5.5, 6.5, 0.2
        else:
            leg_fs, ann_fs, ann_pad = None, 5, 0.15
        if leg_fs is not None:
            handles = [Line2D([], [], linestyle="none") for _ in unique_classes]
            leg = ax.legend(handles, [label_name_map[cls] for cls in unique_classes],
                            fontsize=leg_fs, loc="best", framealpha=0.85)
            for text, cls in zip(leg.get_texts(), unique_classes):
                text.set_color(assigned_color[cls])

        if not huge:
            for cls in unique_classes:
                center = X[labels_arr == cls].mean(axis=0)
                ax.annotate(label_name_map[cls], center, fontsize=ann_fs,
                            weight="bold", ha="center", va="center",
                            color=assigned_color[cls],
                            bbox=dict(boxstyle=f"round,pad={ann_pad}", fc="white",
                                      ec="gray", alpha=0.8))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _to_rgb_display(img_hwc):
    """(H, W, C) raw crop -> (H, W, 3) uint8 contact-sheet image.

    Each channel is percentile-normalized independently; up to 3 channels map
    to R/G/B, more channels average to grayscale.
    """
    chans = []
    for c in range(img_hwc.shape[2]):
        ch = img_hwc[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        chans.append(np.clip((ch - lo) / (hi - lo), 0, 1))
    if len(chans) == 1:
        rgb = np.stack([chans[0]] * 3, axis=-1)
    elif len(chans) == 2:
        rgb = np.stack([chans[0], chans[1], np.zeros_like(chans[0])], axis=-1)
    elif len(chans) == 3:
        rgb = np.stack(chans, axis=-1)
    else:
        gray = np.mean(chans, axis=0)
        rgb = np.stack([gray] * 3, axis=-1)
    return (rgb * 255).astype(np.uint8)


def _build_cell_view(config, mode, channels, channel_layout):
    """Inference-input context for the contact sheet, from the model bundle.

    Loads model.pt for augmentation_infer + normalize meta so contact-sheet
    cells are rendered exactly as the model saw them at inference (same
    resize/crop pipeline -> uniform image size, same normalization).
    """
    model_path = config.get("model")
    if not isinstance(model_path, str) or not model_path or not os.path.exists(model_path):
        print(f"Error: the cluster contact sheet needs the model bundle "
              f"(inference-time preprocessing), but model is missing or not "
              f"found: {model_path}", file=sys.stderr)
        sys.exit(1)
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    meta = bundle.get("meta") or {}
    if "augmentation_infer" not in meta:
        print("Error: bundle meta missing 'augmentation_infer' — cannot "
              "rebuild the inference input for the contact sheet", file=sys.stderr)
        sys.exit(1)
    aug_spec = meta["augmentation_infer"]
    return {
        "aug": build_pipeline(aug_spec) if aug_spec else None,
        "channels": channels,
        "channel_layout": channel_layout,
        "with_masking": bool(meta.get("normalize_with_masking", False)),
        "clip_low": meta.get("clip_low", 0.05),
        "clip_high": meta.get("clip_high", 99.95),
        "normalize_method": meta.get("normalize_method", "per_channel"),
        "fixed_reference": bool(meta.get("normalize_fixed_reference", False)),
        "max_value": resolve_max_value(config["data"]),
    }


def _load_cell_image(d, mode, view):
    """Load one DB row's cell exactly as inference saw it.

    Order mirrors dataset._cell_to_tensor: load -> float by max_value ->
    channel subset -> foreground mask -> bundle augmentation_infer pipeline
    (this fixes the image size) -> normalize (fixed-reference aware).
    Returns a (H, W, 3) uint8 display image, or None on failure.
    """
    from microBase import read_tiff_channels, read_image, read_mask, crop_cell, get_labels
    directory = d.get("directory") or ""
    try:
        if mode == "whole_image":
            ch_files = json.loads(d["filename"])
            arrays = [read_image(os.path.join(directory, f) if directory else f)
                      for f in ch_files]
            img = np.stack(arrays, axis=-1)
            mask_m = read_mask(d.get("mask_filename") or "")
            label = int(d.get("label") or 0)
            if label not in get_labels(mask_m):
                return None
            img, _, _ = crop_cell(img, mask_m, label, padding=4)
        else:
            path = os.path.join(directory, d["filename"]) if directory else d["filename"]
            img = read_tiff_channels(path, view["channels"],
                                     channel_layout=view["channel_layout"])
        img = _to_float_max(img, view["max_value"])
        if mode == "whole_image" and view["channels"] is not None:
            ch_idx = [c - 1 for c in view["channels"] if c <= img.shape[2]]
            img = img[:, :, ch_idx]
        mask = ((img != 0).any(axis=2).astype(np.uint8)
                if view["with_masking"] else None)
        ref_stats = None
        if view["fixed_reference"]:
            ref_stats = _compute_ref_stats(
                img, None, view["with_masking"], view["clip_low"],
                view["clip_high"], view["normalize_method"])
        if view["aug"] is not None:
            img, mask = apply(view["aug"], img, mask)
        if ref_stats is not None:
            img = _normalize_fixed(img, mask, ref_stats)
        else:
            img = normalize(img, mask=mask, method=view["normalize_method"],
                            clip_low=view["clip_low"], clip_high=view["clip_high"])
        return _to_rgb_display(img)
    except (Exception, SystemExit) as e:
        # microBase readers hard-exit (SystemExit) on bad files — degrade to
        # a warning + empty slot instead of aborting the whole run.
        logger.warning("Contact sheet: failed to load %s/%s: %s",
                       directory, d.get("filename"), e)
        return None


def _write_cluster_sheet(ids_all, W, dicts, path, mode, view):
    """Contact-sheet PDF for one Leiden partition: representative cells per cluster.

    Representatives are CONTACT_SHEET_PER_CLUSTER members drawn RANDOMLY from
    the cluster's dense core: local density is each member's mean distance to
    the CONTACT_SHEET_DENSITY_REF-nearest reference members (a fixed random
    subsample — exact cKDTree kNN costs seconds on 18k-member clusters), and
    only the densest CONTACT_SHEET_DENSITY_KEEP fraction is eligible, so the
    sheet shows what a TYPICAL member looks like (the sparse tail is outlier
    morphology) while the random draw keeps natural within-cluster variety.
    Candidates whose crop foreground fraction is below
    CONTACT_SHEET_MIN_FOREGROUND are skipped during the walk. Every cell is
    rendered as its inference-mode input (bundle augmentation_infer pipeline
    -> uniform image size). Layout: CONTACT_SHEET_PER_CLUSTER images per
    cluster, CONTACT_SHEET_GROUPS_PER_ROW clusters per row; cluster blocks
    are ordered by 1-based cluster ID.
    """
    n_ids = int(ids_all.max())  # IDs are 1-based
    n_per = CONTACT_SHEET_PER_CLUSTER
    groups = CONTACT_SHEET_GROUPS_PER_ROW
    rows = -(-n_ids // groups)  # ceil
    # A narrow empty spacer column after each cluster group keeps adjacent
    # clusters visually separated.
    width_ratios = []
    for _ in range(groups):
        width_ratios += [1.0] * n_per + [0.45]
    fig, axes = plt.subplots(
        rows, groups * (n_per + 1),
        figsize=(1.15 * sum(width_ratios), 1.55 * rows),
        squeeze=False, gridspec_kw={"width_ratios": width_ratios})
    for cid in range(1, n_ids + 1):
        member = np.where(ids_all == cid)[0]
        r, g = divmod(cid - 1, groups)
        base = g * (n_per + 1)
        if member.size == 0:
            for j in range(n_per + 1):
                axes[r][base + j].axis("off")
            continue
        # Per-cluster RNG stream: deterministic across regeneration.
        rng = np.random.default_rng([CONTACT_SHEET_SEED, cid])
        member_W = W[member]
        # Local density: mean distance to the k nearest reference members
        # (squared distances via the ||a-b||^2 expansion, chunked to cap the
        # m x ref matrix; smaller = denser). Ranking against a fixed random
        # reference subsample matches exact-kNN ranking at much lower cost.
        ref = member_W[rng.choice(member.size, size=min(
            CONTACT_SHEET_DENSITY_REF, member.size), replace=False)]
        k_near = min(LEIDEN_N_NEIGHBORS, ref.shape[0])
        ref_sq = (ref ** 2).sum(axis=1)
        density = np.empty(member.size)
        for chunk in np.array_split(np.arange(member.size),
                                    max(1, -(-member.size // 4096))):
            d2 = ((member_W[chunk] ** 2).sum(axis=1)[:, None] + ref_sq
                  - 2.0 * member_W[chunk] @ ref.T)
            nearest_sq = np.partition(d2, k_near - 1, axis=1)[:, :k_near]
            density[chunk] = np.sqrt(np.maximum(nearest_sq, 0.0)).mean(axis=1)
        # Eligible pool = the densest fraction; walk it in random order.
        cutoff = np.quantile(density, CONTACT_SHEET_DENSITY_KEEP)
        core = np.where(density <= cutoff)[0]
        order = member[rng.permutation(core)[:CONTACT_SHEET_MAX_SCAN]]
        # Walk candidates in random core order, skipping near-empty crops
        # (see CONTACT_SHEET_MIN_FOREGROUND) until n_per valid representatives.
        imgs = []
        scanned = 0
        for idx in order:
            if len(imgs) >= n_per or scanned >= CONTACT_SHEET_MAX_SCAN:
                break
            scanned += 1
            img = _load_cell_image(dicts[idx], mode, view)
            if img is None or _foreground_fraction(img) < CONTACT_SHEET_MIN_FOREGROUND:
                continue
            imgs.append(img)
        if len(imgs) < n_per:
            logger.warning("Cluster %d: only %d/%d representatives pass the "
                           "%.0f%% foreground filter (scanned %d candidates)",
                           cid, len(imgs), n_per,
                           CONTACT_SHEET_MIN_FOREGROUND * 100, scanned)
        for j in range(n_per):
            ax = axes[r][base + j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(imgs):
                ax.imshow(imgs[j])
            if j == 0:
                ax.set_title(f"cluster {cid} (n={member.size})", fontsize=8)
        axes[r][base + n_per].axis("off")  # spacer after the group
    # Any trailing cluster slot beyond n_ids stays empty.
    for cid in range(n_ids, rows * groups):
        r, g = divmod(cid, groups)
        base = g * (n_per + 1)
        for j in range(n_per + 1):
            axes[r][base + j].axis("off")
    fig.subplots_adjust(wspace=0.06, hspace=0.3)
    with PdfPages(path) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cluster image sheet (%d clusters) saved to %s", n_ids, path)


def _dr_transform(method, reducer, feats):
    """Transform raw feature rows to a 2D embedding with one method's reducer.

    PCA reducers are sliced to 2 columns so pre-fitted >2-component pickles
    (older runs) keep working. pacmap/localmap are fit with save_tree=True so
    out-of-sample transform needs no basis data.
    """
    if method == "pca":
        return reducer.transform(feats)[:, :2]
    if method == "umap":
        pca_pre = reducer.get("pca_pre")
        model = reducer["umap"]
        return model.transform(pca_pre.transform(feats) if pca_pre is not None else feats)
    return reducer.transform(feats)  # pacmap / localmap (n_components=2)


def _fit_embedding(method, reducer, feats_fit):
    """Embedding of the FIT subset itself.

    The nonlinear methods store their fit embedding — transforming the fit
    rows would only re-approximate it.
    """
    if method == "pca":
        return reducer.transform(feats_fit)
    if method == "umap":
        return reducer["umap"].embedding_
    return reducer.embedding_  # pacmap / localmap


def show_reduction(config, save_plots=True, raise_on_error=False):
    """Fit/load the configured DR methods on infer.db features and write tables.

    reduction.method selects from pca / umap / pacmap / localmap. PCA is fit
    with exactly 2 components; UMAP is preceded by a plain PCA to
    UMAP_PRE_COMPONENTS dimensions; PaCMAP/LocalMAP reduce their input
    internally. Every method writes a reduction_<method> table (pca columns
    pc_1/pc_2, others <method>_1/<method>_2) into each DB; the legacy
    reduction_pca_variance table is no longer written (and dropped if a stale
    copy exists).

    reduction.cluster_res lists Leiden resolutions, each producing one
    partition of a kNN graph over a whitened PCA space of the features (the
    space keeps the smallest component count covering
    CLUSTER_VARIANCE_TARGET of the variance, capped at
    CLUSTER_MAX_COMPONENTS; higher resolution = more clusters, and the
    cluster count emerges from the data). Cluster IDs are
    1-based and ordered by each cluster's centroid distance to the origin of
    the reference DR plot (UMAP when present, else the first configured
    method), so nearby-in-the-plot clusters get nearby IDs. The fitted
    whitening PCA + per-resolution kNN classifiers persist to cluster.pkl.
    reduction.cluster pointing at a saved cluster.pkl switches to baseline
    mode: every stored resolution is PREDICTED for the new data by a kNN vote
    over the baseline points (cluster_res ignored), so cluster IDs stay
    aligned with the baseline across datasets. Both go to the find_cluster
    table — one cluster_res<resolution> ID column per resolution plus one
    cluster_prob<resolution> column holding the kNN vote confidence (max
    class probability in [0, 1], the assignment's reliability score). IDs
    become extra color pages in every method's PDF, and each resolution gets
    a representative-cell sheet (cluster_res<resolution>.pdf) unless
    reduction.show_cluster_image is false: CONTACT_SHEET_PER_CLUSTER members
    per cluster, drawn randomly from its densest
    CONTACT_SHEET_DENSITY_KEEP fraction (typical look + natural variety),
    rendered as their inference-mode input via the bundle's
    augmentation_infer pipeline so all images share one size.

    One multi-page PDF per method (reduction_<method>.pdf): one page per
    color_by entry, then one per cluster resolution. color_by accepts ANY
    inference-table or find_cluster column except uid/features; columns whose
    values are numeric with more than CONTINUOUS_COLOR_MAX_CLASSES distinct
    values render as continuous viridis pages, everything else as labeled
    categorical pages. color_by null = a single page with all points in the
    default light blue.

    save_plots=False skips the PDF outputs (tables are still written) —
    used by microProfiler, which is a data-only analysis suite.
    raise_on_error=True converts logged-error abort paths into raised
    RuntimeError/ValueError (the CLI keeps its print + sys.exit semantics).
    Save location: output_dir when set, otherwise every processed dataset's
    own directory (reducer pickles + plots are written once per dataset dir so
    each dataset folder is self-contained).
    """
    inf_cfg = config["inference"]
    red_cfg = config.get("reduction", {})

    data_roots = config["data"]["root"]
    base_output_dir = config.get("output_dir")
    db_name = inf_cfg.get("db_name", "infer.db")
    seed = 42

    # DR methods: canonical order, unknown names dropped with a warning.
    methods_raw = red_cfg.get("method") or ["pca", "umap"]
    if isinstance(methods_raw, str):
        methods_raw = [methods_raw]
    unknown = [m for m in methods_raw if m not in DR_METHODS]
    if unknown:
        logger.warning("Unknown reduction method(s) %s; dropping (valid: %s)",
                       unknown, list(DR_METHODS))
    methods = [m for m in DR_METHODS if m in methods_raw]
    if not methods:
        if raise_on_error:
            raise ValueError("No valid reduction.method; aborting reduction.")
        logger.error("No valid reduction.method; aborting reduction.")
        return

    # color_by is validated AFTER the DBs are loaded: any inference-table or
    # find_cluster column (uid/features excepted) is a valid variable.

    # cluster_res: list of Leiden resolutions; null/empty/nonpositive entries
    # are dropped. Ignored entirely when `cluster` (baseline predict) is set.
    cluster_res_list = []
    for res in (red_cfg.get("cluster_res") or []):
        res_f = float(res)
        if res_f > 0 and res_f not in cluster_res_list:
            cluster_res_list.append(res_f)

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
        feats, dicts = _load_inference_features(db_path, raise_on_error=raise_on_error)
        if feats is None:
            continue
        logger.info("Loaded %d feature vectors (dim=%d) from %s",
                    feats.shape[0], feats.shape[1], db_path)
        db_entries.append((db_path, feats, dicts))

    if not db_entries:
        if raise_on_error:
            raise RuntimeError("No data loaded from any DB; aborting reduction.")
        logger.error("No data loaded from any DB; aborting reduction.")
        return

    # Save location: output_dir when set, else every processed dataset's own
    # directory (reducer pickles + plots are written once per dataset dir).
    if base_output_dir:
        save_dirs = [base_output_dir]
    else:
        save_dirs = sorted({os.path.dirname(db_path) for db_path, _, _ in db_entries})
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)
    if base_output_dir:
        add_file_logging(save_dirs[0])

    feats_all = np.concatenate([e[1] for e in db_entries], axis=0)
    dicts_all = []
    for _, _, dicts in db_entries:
        dicts_all.extend(dicts)
    n_all = feats_all.shape[0]
    logger.info("Merged %d feature vectors from %d DB(s)", n_all, len(db_entries))

    # color_by: null = a single light-blue page. Any inference-table or
    # find_cluster column (uid/features excepted) is valid; unknown names are
    # dropped with a warning. Numeric columns with more distinct values than
    # CONTINUOUS_COLOR_MAX_CLASSES render as continuous pages.
    color_by_vals = red_cfg.get("color_by")
    if color_by_vals is None:
        color_by_vals = []
    elif isinstance(color_by_vals, str):
        color_by_vals = [color_by_vals]
    valid_color_by = set()
    for _, _, dicts in db_entries:
        for d in dicts:
            valid_color_by.update(d.keys())
    valid_color_by -= {"uid", "features"}
    dropped = [cb for cb in color_by_vals if cb not in valid_color_by]
    if dropped:
        logger.warning("Unknown color_by %s; dropping (valid: %s)",
                       dropped, sorted(valid_color_by))
    color_by_vals = [cb for cb in color_by_vals if cb in valid_color_by]
    first_cb = color_by_vals[0] if color_by_vals else None
    cb_continuous = {cb: _color_column_continuous(cb, dicts_all)
                     for cb in color_by_vals}

    # Phase 2: sample subset for fitting (null/0 = use all points)
    sample_per_class = red_cfg.get("sample_per_class") or 0
    # Sampling gates whichever reducer still needs FITTING.
    needs_fit = any(not red_cfg.get(f"reduction_{m}") for m in methods)

    dirs_all = [d["directory"] for d in dicts_all]
    probs_all = [parse_pred_prob(d.get("pred_prob")) for d in dicts_all]
    if sample_per_class > 0 and needs_fit:
        if first_cb is not None:
            first_labels_all, _ = _extract_color_data(first_cb, dicts_all, dirs_all, probs_all)
        else:
            first_labels_all = None
        fit_indices = _sample_fit_indices(
            n_all, first_labels_all, sample_per_class, seed,
            uniform=(first_cb is None or cb_continuous.get(first_cb, False)))
        feats_fit = feats_all[fit_indices]
        dicts_fit = [dicts_all[i] for i in fit_indices]
        dirs_fit = [dirs_all[i] for i in fit_indices]
        probs_fit = [probs_all[i] for i in fit_indices]
        logger.info("Sampled %d of %d rows for reducer fitting (sample_per_class=%d)",
                    len(fit_indices), n_all, sample_per_class)
    else:
        fit_indices = np.arange(n_all)
        feats_fit, dicts_fit = feats_all, dicts_all
        dirs_fit, probs_fit = dirs_all, probs_all
    sampled = len(fit_indices) < n_all

    if len(dicts_fit) < 2:
        if raise_on_error:
            raise RuntimeError(
                f"Too few samples for reduction view (< 2), got {len(dicts_fit)}")
        logger.error("Too few samples for reduction view (< 2), got %d", len(dicts_fit))
        return

    # Phase 3: fit (or load) one reducer per method
    reducers = {}
    for m in methods:
        pre_path = red_cfg.get(f"reduction_{m}")
        if pre_path:
            if not os.path.exists(pre_path):
                if raise_on_error:
                    raise RuntimeError(f"reduction_{m} not found: {pre_path}")
                print(f"Error: reduction_{m} not found: {pre_path}", file=sys.stderr)
                sys.exit(1)
            reducers[m] = load_reducer(pre_path)
            if m == "pca":
                validate_pca(reducers[m], feats_all.shape[1])
            elif m == "umap":
                validate_umap_pipeline(reducers[m], feats_all.shape[1])
            logger.info("%s: loaded reducer from %s", m.upper(), pre_path)
            continue
        if m == "pca":
            reducers[m] = PCA(n_components=2).fit(feats_fit)
        elif m == "umap":
            dim = feats_fit.shape[1]
            n_pre = min(UMAP_PRE_COMPONENTS, feats_fit.shape[0], dim)
            pca_pre = (PCA(n_components=n_pre).fit(feats_fit)
                       if n_pre < dim else None)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)
                model = umap.UMAP(random_state=seed)
                model.fit(pca_pre.transform(feats_fit) if pca_pre is not None else feats_fit)
            reducers[m] = {"pca_pre": pca_pre, "umap": model}
        elif m == "pacmap":
            # PaCMAP reduces its input internally (apply_pca=True default);
            # save_tree keeps the neighbor index inside the pickled reducer.
            reducers[m] = pacmap.PaCMAP(
                n_components=2, random_state=seed, save_tree=True).fit(feats_fit)
        else:  # localmap
            reducers[m] = pacmap.LocalMAP(
                n_components=2, random_state=seed, save_tree=True).fit(feats_fit)
        for d in save_dirs:
            save_reducer(reducers[m], os.path.join(d, f"reduction_{m}.pkl"))
        logger.info("%s: fitted on %d points", m.upper(), feats_fit.shape[0])

    # Phase 4: transform. The fit subset feeds the plots + cluster ordering;
    # the full merge feeds the DB tables. When nothing was sampled the fit
    # embeddings ARE the full embeddings (transform would only re-approximate
    # them for the nonlinear methods).
    X_fit = {m: _fit_embedding(m, reducers[m], feats_fit) for m in methods}
    X_all = X_fit if not sampled else {
        m: _dr_transform(m, reducers[m], feats_all) for m in methods}

    # Phase 5: cluster finding — Leiden over a kNN graph in a whitened PCA
    # space, one partition per resolution. IDs are 1-based, ordered by
    # cluster-centroid distance to the origin of the reference DR plot (UMAP
    # preferred) so plot-nearby clusters get nearby IDs. Fit fresh (cluster:
    # null — per-resolution kNN classifiers persisted to cluster.pkl) or
    # load a baseline cluster.pkl and kNN-predict, so new datasets inherit
    # the baseline's cluster IDs (cluster_res is ignored then; every stored
    # resolution is predicted).
    cluster_ids = {}
    cluster_probs = {}
    cluster_path = red_cfg.get("cluster")
    if cluster_path:
        if not os.path.exists(cluster_path):
            if raise_on_error:
                raise RuntimeError(f"cluster not found: {cluster_path}")
            print(f"Error: cluster not found: {cluster_path}", file=sys.stderr)
            sys.exit(1)
        cluster_obj = load_reducer(cluster_path)
        validate_pca(cluster_obj["pca_whiten"], feats_all.shape[1],
                     name="cluster.pkl pca_whiten")
        W = cluster_obj["pca_whiten"].transform(feats_all)
        W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-8)
        cluster_res_list = sorted(cluster_obj["models"])
        logger.info("Clustering: loaded baseline %s — kNN-predicting all stored "
                    "resolutions %s (cluster_res ignored)",
                    cluster_path, cluster_res_list)
        for res in cluster_res_list:
            knn = cluster_obj["models"][res]["knn"]
            cluster_ids[res] = knn.predict(W).astype(int)
            # Assignment confidence: the fraction of the k nearest baseline
            # points voting for the winning cluster (max class probability).
            cluster_probs[res] = knn.predict_proba(W).max(axis=1)
    elif cluster_res_list:
        dim = feats_fit.shape[1]
        # Clustering-space dimensionality: smallest k covering
        # CLUSTER_VARIANCE_TARGET of the feature variance (measured on the
        # fit subset), capped at CLUSTER_MAX_COMPONENTS (see the constants'
        # whiten noise-amplification note).
        n_cap = min(CLUSTER_MAX_COMPONENTS, feats_fit.shape[0], dim)
        probe = PCA(n_components=n_cap).fit(feats_fit)
        n_white = int(np.searchsorted(
            np.cumsum(probe.explained_variance_ratio_),
            CLUSTER_VARIANCE_TARGET) + 1)
        n_white = max(1, min(n_white, n_cap))
        pca_w = PCA(n_components=n_white, whiten=True).fit(feats_fit)
        W = pca_w.transform(feats_all)
        W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-8)
        ref = "umap" if "umap" in methods else methods[0]
        logger.info("Clustering: whitened PCA %dd (%.0f%% variance, target "
                    "%.0f%%, cap %d) -> Leiden on kNN graph, resolutions=%s "
                    "(ID order ref: %s)",
                    n_white, 100 * float(pca_w.explained_variance_ratio_.sum()),
                    100 * CLUSTER_VARIANCE_TARGET, CLUSTER_MAX_COMPONENTS,
                    cluster_res_list, ref)
        cluster_obj = {"pca_whiten": pca_w, "models": {}}
        for res in cluster_res_list:
            raw_ids = _leiden_partition(W, res, seed)
            n_cl = int(raw_ids.max()) + 1
            # Order clusters by their centroid distance to the reference
            # plot's origin. A cluster absent from the fit subset (reducer
            # sampling) sorts last via an infinite distance.
            ref_fit_ids = raw_ids[fit_indices]
            cent = np.array([
                X_fit[ref][ref_fit_ids == c].mean(axis=0)
                if np.any(ref_fit_ids == c) else np.full(2, np.inf)
                for c in range(n_cl)])
            order = np.argsort(np.linalg.norm(cent, axis=1))
            id_map = np.empty(n_cl, dtype=int)
            id_map[order] = np.arange(1, n_cl + 1)
            ids = id_map[raw_ids]
            # Baseline handle: a kNN vote over these very points lets future
            # datasets inherit this partition's IDs (predict branch above);
            # predict_proba doubles as each point's own assignment confidence.
            knn = KNeighborsClassifier(
                n_neighbors=min(LEIDEN_N_NEIGHBORS, n_all - 1)).fit(W, ids)
            cluster_ids[res] = ids
            cluster_probs[res] = knn.predict_proba(W).max(axis=1)
            cluster_obj["models"][res] = {"knn": knn, "n_clusters": n_cl}
            logger.info("Clustering: resolution %g -> %d clusters", res, n_cl)
        for d in save_dirs:
            save_reducer(cluster_obj, os.path.join(d, "cluster.pkl"))

    # Phase 6: plots — one multi-page PDF per method
    if save_plots:
        for d in save_dirs:
            for m in methods:
                axis_name = DR_AXIS_NAMES[m]
                pdf_path = os.path.join(d, f"reduction_{m}.pdf")
                with PdfPages(pdf_path) as pdf:
                    if not color_by_vals and not cluster_res_list:
                        _plot_reduction_page(
                            X_fit[m], f"{axis_name} of feature vectors",
                            f"{axis_name} 1", f"{axis_name} 2", pdf)
                    for cb in color_by_vals:
                        labels, label_names = _extract_color_data(
                            cb, dicts_fit, dirs_fit, probs_fit)
                        if cb_continuous[cb]:
                            # Continuous page: pred_prob keeps its fixed [0,1]
                            # scale and title; other numeric columns scale to
                            # their own range.
                            vals = np.array(
                                [np.nan if v is None else float(v) for v in labels],
                                dtype=np.float64)
                            if cb == "pred_prob":
                                cmin, cmax, clabel = 0.0, 1.0, "Prediction Probability"
                            else:
                                cmin = float(np.nanmin(vals))
                                cmax = float(np.nanmax(vals))
                                if not cmax > cmin:
                                    cmax = cmin + 1.0
                                clabel = cb
                            _plot_reduction_page(
                                X_fit[m], f"{axis_name} of feature vectors (colored by {cb})",
                                f"{axis_name} 1", f"{axis_name} 2", pdf,
                                labels=labels, label_names=label_names,
                                pred_probs=probs_fit, continuous=True,
                                cont_values=vals, cont_range=(cmin, cmax),
                                cont_label=clabel)
                        else:
                            _plot_reduction_page(
                                X_fit[m], f"{axis_name} of feature vectors (colored by {cb})",
                                f"{axis_name} 1", f"{axis_name} 2", pdf,
                                labels=labels, label_names=label_names,
                                pred_probs=probs_fit)
                    for res in cluster_res_list:
                        ids_fit = cluster_ids[res][fit_indices]
                        n_cl = len(np.unique(ids_fit))
                        _plot_reduction_page(
                            X_fit[m],
                            f"{axis_name} of feature vectors (colored by Leiden "
                            f"cluster, res={_res_tag(res)}, n={n_cl})",
                            f"{axis_name} 1", f"{axis_name} 2", pdf,
                            labels=list(ids_fit),
                            label_names=[str(i) for i in ids_fit])
                logger.info("Reduction plot saved to %s", pdf_path)
        # Representative-cell sheet per resolution (CONTACT_SHEET_PER_CLUSTER
        # random dense-core members per cluster, uniform cell size). Gated by
        # reduction.show_cluster_image — false skips the sheets entirely,
        # including the model-bundle load they need.
        if not red_cfg.get("show_cluster_image", True):
            logger.info("reduction.show_cluster_image is false — skipping "
                        "cluster_res*.pdf sheets")
        elif cluster_res_list:
            mode = config.get("mode", "single_cell")
            channels_cfg = config["data"].get("channels")
            channels = list(channels_cfg) if channels_cfg else [1]
            channel_layout = config["data"].get("channel_layout")
            cell_view = _build_cell_view(config, mode, channels, channel_layout)
            for d in save_dirs:
                for res in cluster_res_list:
                    n_cl = int(cluster_ids[res].max())
                    if n_cl > CONTACT_SHEET_MAX_CLUSTERS:
                        logger.info("Skipping cluster sheet for resolution %g: "
                                    "%d clusters > %d", res, n_cl,
                                    CONTACT_SHEET_MAX_CLUSTERS)
                        continue
                    _write_cluster_sheet(
                        cluster_ids[res], W, dicts_all,
                        os.path.join(d, f"cluster_res{_res_tag(res)}.pdf"),
                        mode, cell_view)

    # Phase 7: write per-DB tables
    t0 = time.perf_counter()
    offsets = []
    _cursor = 0
    for _, feats_db, _ in db_entries:
        offsets.append((_cursor, _cursor + len(feats_db)))
        _cursor += len(feats_db)

    for (start, end), (db_path, feats_db, db_dicts) in zip(offsets, db_entries):
        t_db0 = time.perf_counter()
        uids = [d["uid"] for d in db_dicts]
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")

        # Legacy table from the variance-tracking schema — no longer written.
        conn.execute("DROP TABLE IF EXISTS reduction_pca_variance")

        for m in methods:
            col = "pc" if m == "pca" else m
            # Drop first: the old PCA schema could carry pc_1..pc_k columns.
            conn.execute(f"DROP TABLE IF EXISTS reduction_{m}")
            conn.execute(
                f"CREATE TABLE reduction_{m} ("
                f"uid INTEGER PRIMARY KEY, {col}_1 REAL NOT NULL, {col}_2 REAL NOT NULL)")
            arr = X_all[m][start:end]
            conn.executemany(
                f"INSERT OR REPLACE INTO reduction_{m} (uid, {col}_1, {col}_2) "
                f"VALUES (?, ?, ?)",
                list(zip(uids, arr[:, 0].tolist(), arr[:, 1].tolist())))

        if cluster_res_list:
            # One cluster_res<tag> ID column plus one cluster_prob<tag>
            # confidence column (kNN vote fraction, [0, 1]) per resolution.
            # Resolution tags may contain a dot ("0.5"), so every identifier
            # is quoted.
            id_cols = [f"cluster_res{_res_tag(r)}" for r in cluster_res_list]
            prob_cols = [f"cluster_prob{_res_tag(r)}" for r in cluster_res_list]
            cols = ", ".join(
                [f"{sql_ident(c)} INTEGER NOT NULL" for c in id_cols]
                + [f"{sql_ident(c)} REAL NOT NULL" for c in prob_cols])
            col_names = ", ".join(sql_ident(c) for c in id_cols + prob_cols)
            # The uid placeholder is written literally below; ph covers the
            # id+prob columns only.
            ph = ", ".join("?" * (len(id_cols) + len(prob_cols)))
            conn.execute("DROP TABLE IF EXISTS find_cluster")
            conn.execute(
                f"CREATE TABLE find_cluster (uid INTEGER PRIMARY KEY, {cols})")
            conn.executemany(
                f"INSERT OR REPLACE INTO find_cluster (uid, {col_names}) "
                f"VALUES (?, {ph})",
                list(zip(uids,
                         *[cluster_ids[r][start:end].tolist()
                           for r in cluster_res_list],
                         *[cluster_probs[r][start:end].tolist()
                           for r in cluster_res_list])))

        conn.commit()
        conn.close()
        logger.info("Wrote %s tables + cluster info (%d rows) to %s (db_write=%.1fs)",
                    "/".join(methods), end - start, db_path,
                    time.perf_counter() - t_db0)

    logger.info("Reduction complete: methods=%s, cluster_res=%s, fitted on %d "
                "samples, transformed %d DB(s), outputs in %s",
                methods, cluster_res_list or None, feats_fit.shape[0], len(db_entries),
                save_dirs)


def _load_inference_features(db_path, raise_on_error=False):
    """Load all rows from the inference table. Returns (feats, dicts) or (None, None).

    raise_on_error=True converts the NULL-features abort into a raised
    RuntimeError (microProfiler's bridge) instead of print + sys.exit (CLI).
    """
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
        # Merge the find_cluster columns (cluster_res<resolution> IDs +
        # cluster_prob<resolution> confidences) into the row dicts so every
        # one of them can serve as a color_by variable. Resolution tags may
        # contain a dot, so identifiers are quoted. Rows missing from a stale
        # find_cluster table simply keep no keys (d.get -> None downstream).
        has_fc = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='find_cluster'"
        ).fetchone()
        if has_fc:
            fc_cols = [row[1] for row in conn.execute("PRAGMA table_info(find_cluster)")
                       if row[1] != "uid"]
            if fc_cols:
                sel = ", ".join(["uid"] + [sql_ident(c) for c in fc_cols])
                fc_by_uid = {}
                for row in conn.execute(f"SELECT {sel} FROM find_cluster"):
                    fc_by_uid[int(row[0])] = dict(zip(fc_cols, row[1:]))
                for d in dicts:
                    extra = fc_by_uid.get(d["uid"])
                    if extra:
                        d.update(extra)
        feats_arr = []
        for d in dicts:
            f = d["features"]
            if f is None:
                msg = (
                    f"row uid={d.get('uid')} has NULL features. "
                    "Run inference with inference.feature: true"
                )
                if raise_on_error:
                    raise RuntimeError(f"{db_path}: {msg}")
                print(f"Error: {msg}", file=sys.stderr)
                sys.exit(1)
            feats_arr.append(np.frombuffer(f, dtype=np.float32))
        feats = np.stack(feats_arr, axis=0)
        return feats, dicts
    finally:
        conn.close()


def _color_column_continuous(cb, dicts):
    """Classify a color_by column: True = continuous page, False = categorical.

    Probability columns (pred_prob, cluster_prob<tag>) are always continuous
    — kNN vote confidences only take ~15 distinct fractions, which the
    distinct-count heuristic below would misread as categories. Other
    columns qualify as continuous when every non-null value parses as a
    number AND there are more distinct values than
    CONTINUOUS_COLOR_MAX_CLASSES. cluster_res<tag> columns hold Leiden IDs
    whose numeric order encodes plot distance, not magnitude — they stay
    categorical no matter how many clusters a resolution produced.
    """
    if cb.startswith("cluster_res"):
        return False
    if cb == "pred_prob" or cb.startswith("cluster_prob"):
        return True
    nums = set()
    for d in dicts:
        v = d.get(cb)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            try:
                nums.add(float(str(v)))
            except ValueError:
                return False  # any non-numeric value -> categorical
        else:
            nums.add(float(v))
        if len(nums) > CONTINUOUS_COLOR_MAX_CLASSES:
            return True
    return False


def _extract_color_data(cb, dicts, dirs, probs):
    """Extract (raw_values, display_names) for a color_by variable.

    The curated columns keep their tailored handling; any other
    inference-table / find_cluster column falls through to its raw values
    (None -> "__missing__" so the categorical sort never sees mixed types).
    """
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
    vals = [d.get(cb) for d in dicts]
    return vals, ["__missing__" if v is None else str(v) for v in vals]


def _sample_fit_indices(n_orig, labels, sample_per_class, seed, uniform):
    """Select indices for reducer fitting, stratified by `labels`.

    uniform=True samples without stratification (color_by null, or a
    continuous column — stratifying by raw floats would create one stratum
    per distinct value).
    """
    return stratified_sample_indices(n_orig, labels, sample_per_class, seed,
                                     uniform=uniform)


# ----------------------------------------------------------------------------
# Training plots (for train)
# ----------------------------------------------------------------------------

def plot_training_results(model, device, val_loader, num_classes, label_to_idx,
                          train_loss_history, val_acc_history, save_path=None,
                          multi_label=False):
    """Loss/accuracy history + a third panel that depends on the label mode:
    single-label draws the normalized confusion matrix; multi-label draws a
    per-class F1 bar chart at the 0.5 threshold. Returns a dict with the
    final metrics plus "cm" (single-label) or "per_class_f1" (multi-label).
    """
    if not train_loss_history:
        logger.warning("No training history to plot.")
        return None

    # Imported here (not at module top) to keep vis import light for GUI users.
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    # x-axis is the 1-based epoch number (package-wide convention).
    ep = range(1, len(train_loss_history) + 1)
    axes[0].plot(ep, train_loss_history, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)

    axes[1].plot(range(1, len(val_acc_history) + 1), val_acc_history,
                 marker="s", color="orange")
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
            if multi_label:
                yt_all.extend(y.long().tolist())
                yp_all.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
            else:
                yt_all.extend(y.tolist())
                yp_all.extend(logits.argmax(1).cpu().tolist())

    # Class order for the confusion-matrix axes follows the label_to_idx
    # mapping (sorted by index), not dict insertion order.
    cls_names = sorted(label_to_idx, key=label_to_idx.get)
    if multi_label:
        # No confusion matrix exists for multi-label — bar-chart the F1 of
        # every category instead.
        per_class = f1_score(yt_all, yp_all, average=None,
                             zero_division=0) if yt_all else np.zeros(num_classes)
        per_class_f1 = {cls_names[i]: float(per_class[i])
                        for i in range(min(num_classes, len(per_class)))}
        axes[2].bar(range(num_classes), per_class[:num_classes], color="steelblue")
        axes[2].set_xticks(range(num_classes), cls_names, rotation=45, ha="right")
        axes[2].set_ylabel("F1")
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Per-class F1 (threshold 0.5)")
        axes[2].grid(True, axis="y")
        cm = None
    else:
        cm = confusion_matrix(yt_all, yp_all, labels=list(range(num_classes)))
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        im = axes[2].imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("True")
        axes[2].set_title("Confusion Matrix")
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

    final_acc = float(accuracy_score(yt_all, yp_all)) if yt_all else 0.0
    final_f1 = float(f1_score(yt_all, yp_all, average="macro", zero_division=0)) if yt_all else 0.0
    result = {"final_loss": train_loss_history[-1],
              "final_acc": val_acc_history[-1], "final_f1": final_f1,
              "final_val_acc": final_acc, "final_val_f1": final_f1}
    if cm is not None:
        result["cm"] = cm
    else:
        result["per_class_f1"] = per_class_f1
    return result


# ----------------------------------------------------------------------------
# Pretrain loss curve (for SSL pretrain)
# ----------------------------------------------------------------------------

def plot_pretrain_loss(loss_history, component_histories=None, save_path=None):
    """Plot the pretrain total loss plus one curve per loss component.

    loss_history: list of per-epoch total loss.
    component_histories: {component_name: [per-epoch values]} — whatever the
        method's train_step returned (dino/ibot/gram/recon/dist/adv/...).
        Components whose history length does not match loss_history are
        skipped. save_path: output PDF path. If None, the figure is not saved.
    """
    if not loss_history:
        logger.warning("No pretrain loss history to plot.")
        return None

    epochs = range(1, len(loss_history) + 1)
    components = {
        name: hist for name, hist in (component_histories or {}).items()
        if isinstance(hist, list) and len(hist) == len(loss_history)
    }

    if components:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, loss_history, label="total", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Total Loss")
        axes[0].grid(True)
        axes[0].legend()
        for name, hist in components.items():
            axes[1].plot(epochs, hist, label=name, linewidth=1.2)
        axes[1].set_title("Loss Components")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
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


# ----------------------------------------------------------------------------
# DINOv3 training-quality diagnostics
# ----------------------------------------------------------------------------

#: Head-collapse threshold in logit space (DINOv3 collapse-guard value):
#: healthy heads sit at std ~0.02-0.06, collapsed heads at ~0.002.
HEAD_LOGITS_STD_COLLAPSE_THRESHOLD = 5e-3


def plot_head_track(head_std_history=None, head_entropy_history=None,
                    sts_history=None, gram_masked_history=None,
                    gram_unmasked_history=None, save_path=None):
    """DINOv3 monitoring plots: head collapse signals + Gram split stats.

    head_std_history:   per-epoch student head-logit std (collapse if it
        drops below ~5e-3).
    head_entropy_history: normalized softmax entropy in [0, 1] (-> 1 collapse).
    sts_history:        mean teacher-student CLS cosine similarity.
    gram_masked/unmasked_history: per-epoch Gram losses on masked/unmasked
        patch tokens (gram anchoring enabled).
    """
    series = [
        (head_std_history, "head logits std", "Head logits std (collapse < 5e-3)"),
        (head_entropy_history, "head entropy", "Head entropy (norm., -> 1 = collapse)"),
        (sts_history, "teacher-student sim", "Teacher-student CLS similarity"),
    ]
    present = [(h, lbl, ttl) for h, lbl, ttl in series if h]
    has_gram = (gram_masked_history or gram_unmasked_history)

    n_plots = len(present) + (1 if has_gram else 0)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, (hist, label, title) in zip(axes, present):
        epochs = range(1, len(hist) + 1)
        ax.plot(epochs, hist, marker="o", label=label, linewidth=1.4)
        if title.startswith("Head logits std"):
            ax.axhline(HEAD_LOGITS_STD_COLLAPSE_THRESHOLD, color="red",
                       linestyle="--", linewidth=1.0,
                       label="collapse threshold")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend(fontsize=8)

    if has_gram and len(axes) > len(present):
        ax = axes[-1]
        if gram_masked_history:
            ax.plot(range(1, len(gram_masked_history) + 1), gram_masked_history,
                    marker="o", label="gram (masked)", linewidth=1.2)
        if gram_unmasked_history:
            ax.plot(range(1, len(gram_unmasked_history) + 1), gram_unmasked_history,
                    marker="s", label="gram (unmasked)", linewidth=1.2)
        ax.set_title("Gram loss split (masked / unmasked)")
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.suptitle("DINOv3 training-quality diagnostics", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Head-track diagnostics saved to %s", save_path)
    plt.close(fig)


def _to_display_rgb(x_cpu):
    """Normalized input tensor -> (H,W,3) uint8 display image.

    Takes the first channel of a (…,C,H,W)/(C,H,W) tensor and min-max
    rescales it, so the user sees the exact pixels fed to the model
    (brightness rescaled for display only).
    """
    a = np.asarray(x_cpu)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = a[0]
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        hi = lo + 1.0
    g = ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.stack([g] * 3, axis=-1)


def plot_patch_similarity(maps_list, save_path, inputs=None, anchor_idx=None,
                          title=None):
    """Patch cosine-similarity maps (DINOv3 paper Fig. 3 style).

    maps_list: list of samples; each sample = n_anchors (grid, grid) maps.
    Healthy training shows object-like high-similarity blocks; uniform/blurry
    maps indicate dense-feature collapse or degradation.

    inputs (optional): one (H,W,3) display per sample -> prepended first
    column so the user sees the raw-ish input beside each anchor map.
    anchor_idx (optional): sorted anchor patch indices -> per-anchor subplot
    labels (row/col + flat index).
    """
    if not maps_list:
        return
    n_samples = len(maps_list)
    n_cols = len(maps_list[0])
    layout_cols = n_cols + (1 if inputs is not None else 0)
    fig, axes = plt.subplots(n_samples, layout_cols,
                             figsize=(3.2 * layout_cols, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_samples):
        col = 0
        if inputs is not None:
            im = axes[i][0].imshow(inputs[i])
            axes[i][0].set_xticks([])
            axes[i][0].set_yticks([])
            if i == 0:
                axes[i][0].set_title("input (normalized)", fontsize=8)
            col = 1
        for j in range(n_cols):
            ax = axes[i][col + j]
            sim = maps_list[i][j]
            im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                label = f"anchor {j + 1}"
                if anchor_idx:
                    g = sim.shape[0]
                    r, c = divmod(anchor_idx[j], g)
                    label += f"\n(idx {anchor_idx[j]}, r{r} c{c})"
                ax.set_title(label, fontsize=8)
    fig.suptitle(title or f"Patch similarity maps ({n_samples} samples)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    try:
        fig.tight_layout()
    except ValueError:
        # colorbar axes are not tight_layout-compatible; fall back to subplots_adjust
        fig.subplots_adjust(wspace=0.05, hspace=0.1)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Patch similarity maps saved to %s", save_path)
    plt.close(fig)


def plot_attention_maps(maps_list, save_path, inputs=None, title=None):
    """CLS->patch attention heat maps, per single head + mean head.

    maps_list: list of samples; each sample is a dict
    {'per_head': [h (grid, grid) arrays, one per attention head],
     'mean': (grid, grid)}. Object-like highlight = attention is segmenting
    the cell (healthy DINO training); scattered/uniform = degenerate.
    Comparing the heads side by side shows head-specific foci (nucleus,
    cytosol, background...).

    inputs (optional): one (H,W,3) display per sample -> prepended first
    column so the user sees the exact input image each row was computed from.
    """
    if not maps_list:
        return
    n_samples = len(maps_list)
    h = len(maps_list[0]["per_head"])
    n_cols = h + 1 + (1 if inputs is not None else 0)
    fig, axes = plt.subplots(n_samples, n_cols,
                             figsize=(3 * n_cols, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    for i, smp in enumerate(maps_list):
        col = 0
        if inputs is not None:
            axes[i][0].imshow(inputs[i])
            axes[i][0].set_xticks([])
            axes[i][0].set_yticks([])
            if i == 0:
                axes[i][0].set_title("input (normalized)", fontsize=8)
            col = 1
        for hi, mp in enumerate(smp["per_head"]):
            ax = axes[i][col + hi]
            im = ax.imshow(mp, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"head {hi + 1}", fontsize=8)
        axm = axes[i][col + h]
        im = axm.imshow(smp["mean"], cmap="viridis", vmin=0.0, vmax=1.0)
        axm.set_xticks([])
        axm.set_yticks([])
        if i == 0:
            axm.set_title("mean head", fontsize=8)
    fig.suptitle(title or f"CLS attention maps ({n_samples} samples)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    try:
        fig.tight_layout()
    except ValueError:
        # colorbar axes are not tight_layout-compatible; fall back to subplots_adjust
        fig.subplots_adjust(wspace=0.05, hspace=0.1)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("CLS attention maps saved to %s", save_path)
    plt.close(fig)


def plot_attention_combined(samples, save_path, inputs=None, anchor_idx=None,
                            title=None):
    """Combined diagnostic PDF: one row per image containing the input, the
    patch-similarity anchors (coolwarm) and the per-head + mean CLS attention
    maps (viridis) side by side.

    samples: list; each = {'anchors': [4 (g, g) cosine-sim maps],
    'attn': {'per_head': [6 (g, g) maps, one per attention head],
    'mean': (g, g)}}.
    inputs (optional): one (H,W,3) display per row -> prepended column.
    anchor_idx (optional): sorted anchor patch indices for the anchor labels.
    """
    if not samples:
        return
    n_rows = len(samples)
    n_anchors = len(samples[0]["anchors"])
    n_heads = len(samples[0]["attn"]["per_head"])
    base_cols = n_anchors + n_heads + 1  # anchors + heads + mean
    n_cols = base_cols + (1 if inputs is not None else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, smp in enumerate(samples):
        col = 0
        if inputs is not None:
            axes[i][0].imshow(inputs[i])
            axes[i][0].set_xticks([])
            axes[i][0].set_yticks([])
            if i == 0:
                axes[i][0].set_title("input (normalized)", fontsize=8)
            col = 1
        im = None
        for j, mp in enumerate(smp["anchors"]):
            ax = axes[i][col + j]
            im = ax.imshow(mp, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                label = f"anchor {j + 1}"
                if anchor_idx:
                    g = mp.shape[0]
                    r, c = divmod(anchor_idx[j], g)
                    label += f"\n(idx {anchor_idx[j]}, r{r} c{c})"
                ax.set_title(label, fontsize=8)
        for hi, mp in enumerate(smp["attn"]["per_head"]):
            ax = axes[i][col + n_anchors + hi]
            im = ax.imshow(mp, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"head {hi + 1}", fontsize=8)
        mean = smp["attn"]["mean"]
        axm = axes[i][col + n_anchors + n_heads]
        im = axm.imshow(mean, cmap="viridis", vmin=0.0, vmax=1.0)
        axm.set_xticks([])
        axm.set_yticks([])
        if i == 0:
            axm.set_title("mean head", fontsize=8)
    fig.suptitle(title or f"Attention + patch similarity ({n_rows} samples)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    try:
        fig.tight_layout()
    except ValueError:
        # colorbar axes are not tight_layout-compatible; fall back to subplots_adjust
        fig.subplots_adjust(wspace=0.05, hspace=0.1)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Combined attention maps saved to %s", save_path)
    plt.close(fig)
