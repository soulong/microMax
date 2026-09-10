"""Feature-space reduction and cluster finding (CLI: micromodel reduction).

run_reduction fits the configured DR methods (pca/umap/pacmap/localmap) on
infer.db features, writes one reduction_<method> table per method plus the
optional find_cluster table, and saves one multi-page PDF per DR method.
"""

import os
import sys
import json
import time
import sqlite3
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
import igraph as ig
import leidenalg
import umap
import pacmap

from microBase import (
    MicroMaxError,
    build_pipeline,
    apply,
    normalize,
)
from microBase.db_contracts import (
    CLUSTER_PROB_PREFIX,
    CLUSTER_RES_PREFIX,
    DR_METHODS,
    FEATURES_COLUMN,
    FIND_CLUSTER_TABLE,
    INFERENCE_TABLE,
    INFER_DB_NAME,
    PRED_PROB_COLUMN,
    UID_COLUMN,
    reduction_coord_prefix,
    reduction_table_name,
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
# Reduction view (pca / umap / pacmap / localmap + optional clustering)
# ----------------------------------------------------------------------------

#: Pretty axis prefix per DR method (plot titles / axis labels).
#: DR_METHODS (the canonical order) comes from microBase.db_contracts.
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
CLUSTER_VARIANCE_TARGET = 0.90
CLUSTER_MAX_COMPONENTS = 100

#: Neighbors per point in the kNN graph feeding Leiden clustering; the same
#: k is the vote pool when a baseline cluster.pkl predicts new data.
LEIDEN_N_NEIGHBORS = 15

#: Default scatter color when color_by is null (light blue).
SINGLE_COLOR = "#87CEEB"

#: Images-per-cluster in the cluster contact sheet.
CONTACT_SHEET_PER_CLUSTER = 10

#: Max cluster rows in the cluster contact sheet; cluster blocks fill
#: COLUMN-major (top -> bottom, then the next column) with as many columns
#: as the cluster count needs.
CONTACT_SHEET_MAX_ROWS = 10

#: Cluster-block columns per sheet PAGE. Each block column is one cluster's
#: row of images, so capping the columns per page caps the page width — and
#: with it the size every cell is rendered at. Higher resolutions produce
#: more clusters; without the cap they would all be squeezed onto one
#: ever-wider page (60 clusters -> a ~6x-wider sheet -> cells shrunk 6x the
#: moment the viewer fits the page to the screen). Extra clusters continue
#: onto the next page of the same PDF at the SAME cell size.
CONTACT_SHEET_GROUPS_PER_PAGE = 2

#: Each page hugs its content: rows fill first (up to CONTACT_SHEET_MAX_ROWS),
#: then each extra block-column widens the page (up to
#: CONTACT_SHEET_GROUPS_PER_PAGE) — saved without bbox tightening, so there
#: is no blank page margin around the grid and the per-cell size is identical
#: on every page.

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
        # Beyond 20 classes the legend outgrows the plot area entirely, so it
        # is dropped and classes are labeled by the centroid annotations
        # alone; those scale their font down with the class count and, past
        # 60, disappear as well (pairwise centroid colors then go away too).
        if n_cls <= 20:
            leg_fs, ann_fs, ann_pad = 7, 9, 0.3
        elif n_cls <= 60:
            leg_fs, ann_fs, ann_pad = None, 6.5, 0.2
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
        raise MicroMaxError(f"Error: the cluster contact sheet needs the model bundle "
              f"(inference-time preprocessing), but model is missing or not "
              f"found: {model_path}")
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    meta = bundle.get("meta") or {}
    if "augmentation_infer" not in meta:
        raise MicroMaxError("Error: bundle meta missing 'augmentation_infer' — cannot "
              "rebuild the inference input for the contact sheet")
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
    from microBase.db_contracts import resolve_directory
    # directory is stored relative to the row's dataset root (_root injected
    # at load time); resolving here also accepts legacy absolute values.
    directory = resolve_directory(d.get("directory") or "", d.get("_root", ""))
    try:
        if mode == "whole_image":
            ch_files = json.loads(d["filename"])
            arrays = [read_image(os.path.join(directory, f)) for f in ch_files]
            img = np.stack(arrays, axis=-1)
            mask_m = read_mask(d.get("mask_filename") or "")
            label = int(d.get("label") or 0)
            if label not in get_labels(mask_m):
                return None
            img, _, _ = crop_cell(img, mask_m, label, padding=4)
        else:
            path = os.path.join(directory, d["filename"])
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
        # microBase readers raise on bad files — degrade to
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
    CONTACT_SHEET_MIN_FOREGROUND are skipped during the walk; clusters where
    EVERY candidate fails it (glass scratches / dust — Leiden groups debris
    coherently) fall back to their unfiltered crops, still titled "no
    cell-like crops". Every cell is rendered as its inference-mode input
    (bundle augmentation_infer pipeline -> uniform image size).

    Layout: cluster blocks fill COLUMN-major (top -> bottom, then the next
    column), at most CONTACT_SHEET_MAX_ROWS rows, and at most
    CONTACT_SHEET_GROUPS_PER_PAGE block columns per PAGE — so the page width,
    and with it the rendered size of every cell, stays constant no matter how
    many clusters the resolution produced; the remaining clusters continue
    onto the following pages of the same PDF. Blocks are ordered by 1-based
    cluster ID and each page hugs its grid (no blank margins, no bbox
    tightening).
    """
    n_ids = int(ids_all.max())  # IDs are 1-based
    n_per = CONTACT_SHEET_PER_CLUSTER
    rows = min(n_ids, CONTACT_SHEET_MAX_ROWS)
    groups = -(-n_ids // rows)  # ceil -> number of cluster-block columns
    per_page = CONTACT_SHEET_GROUPS_PER_PAGE

    # ---- Phase A: representatives per cluster (the slow image loading) ----
    # reps[cid] = (imgs, debris, n_member); imgs=None marks an empty cluster.
    reps = {}
    for cid in range(1, n_ids + 1):
        member = np.where(ids_all == cid)[0]
        if member.size == 0:
            reps[cid] = (None, False, 0)
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
        debris = not imgs
        if debris:
            # Debris clusters (glass scratches, dust — Leiden groups them
            # coherently) fail the foreground filter wholesale. Show their
            # real crops anyway — an empty row would read like a bug.
            for idx in order:
                if len(imgs) >= n_per:
                    break
                img = _load_cell_image(dicts[idx], mode, view)
                if img is not None:
                    imgs.append(img)
        if len(imgs) < n_per:
            logger.warning("Cluster %d: only %d/%d representatives pass the "
                           "%.0f%% foreground filter (scanned %d candidates)",
                           cid, len(imgs), n_per,
                           CONTACT_SHEET_MIN_FOREGROUND * 100, scanned)
        reps[cid] = (imgs, debris, member.size)

    # ---- Phase B: paginate the cluster blocks, one figure per page ----
    with PdfPages(path) as pdf:
        for page_start in range(0, groups, per_page):
            page_groups = min(per_page, groups - page_start)
            # A narrow empty spacer column after each cluster group keeps
            # adjacent clusters visually separated.
            width_ratios = []
            for _ in range(page_groups):
                width_ratios += [1.0] * n_per + [0.45]
            fig, axes = plt.subplots(
                rows, page_groups * (n_per + 1),
                figsize=(1.15 * page_groups * (n_per + 0.45), 1.35 * rows),
                squeeze=False, gridspec_kw={"width_ratios": width_ratios})
            first_cid = page_start * rows + 1
            last_cid = min(first_cid + page_groups * rows, n_ids + 1)
            for cid in range(first_cid, last_cid):
                imgs, debris, n_member = reps[cid]
                g, r = divmod(cid - first_cid, rows)  # column-major in page
                base = g * (n_per + 1)
                if imgs is None:  # empty cluster: blank block, no title
                    for j in range(n_per + 1):
                        axes[r][base + j].axis("off")
                    continue
                for j in range(n_per):
                    ax = axes[r][base + j]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if j < len(imgs):
                        ax.imshow(imgs[j])
                    if j == 0:
                        title = "cluster %d (n=%d)" % (cid, n_member)
                        if debris:
                            title += "\nno cell-like crops"
                        ax.set_title(title, fontsize=8)
                axes[r][base + n_per].axis("off")  # spacer after the group
            # Trailing cluster slots beyond n_ids on this page stay empty.
            for cid in range(last_cid, first_cid + page_groups * rows):
                g, r = divmod(cid - first_cid, rows)
                base = g * (n_per + 1)
                for j in range(n_per + 1):
                    axes[r][base + j].axis("off")
            # Near-zero page margins: the grid owns the whole page.
            fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.01,
                                wspace=0.06, hspace=0.15)
            # No bbox tightening: the page size derives directly from the
            # grid, so every page renders cells at the same physical size.
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    logger.info("Cluster image sheet (%d clusters, %d page(s)) saved to %s",
                n_ids, -(-groups // per_page), path)


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


def run_reduction(config, save_plots=True, raise_on_error=False):
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
    table — one cluster_res_<resolution> ID column per resolution plus one
    cluster_prob_<resolution> column holding the kNN vote confidence (max
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
    db_name = inf_cfg.get("db_name", INFER_DB_NAME)
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
        # Each row's directory is stored relative to ITS dataset root —
        # remember which root so contact-sheet crops resolve correctly.
        for d in dicts:
            d["_root"] = str(data_dir)
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
    loaded_methods = set()
    for m in methods:
        pre_path = red_cfg.get(f"reduction_{m}")
        if pre_path:
            if not os.path.exists(pre_path):
                if raise_on_error:
                    raise RuntimeError(f"reduction_{m} not found: {pre_path}")
                raise MicroMaxError(f"Error: reduction_{m} not found: {pre_path}")
            reducers[m] = load_reducer(pre_path)
            loaded_methods.add(m)
            if m == "pca":
                validate_pca(reducers[m], feats_all.shape[1])
            elif m == "umap":
                validate_umap_pipeline(reducers[m], feats_all.shape[1])
            logger.info("%s: loaded reducer from %s", m.upper(), pre_path)
            continue
        if m == "pca":
            reducers[m] = PCA(n_components=2, random_state=seed).fit(feats_fit)
        elif m == "umap":
            dim = feats_fit.shape[1]
            n_pre = min(UMAP_PRE_COMPONENTS, feats_fit.shape[0], dim)
            pca_pre = (PCA(n_components=n_pre, random_state=seed).fit(feats_fit)
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
    # the full merge feeds the DB tables. When nothing was sampled, a
    # FRESHLY fitted nonlinear reducer's fit embedding IS the full embedding
    # (transform would only re-approximate it). A LOADED reducer's
    # .embedding_ belongs to its baseline fit set — never this dataset's
    # rows — so loaded methods must always transform the current features.
    X_fit = {m: _fit_embedding(m, reducers[m], feats_fit) for m in methods}
    X_all = {
        m: (
            _dr_transform(m, reducers[m], feats_all)
            if sampled or m in loaded_methods
            else X_fit[m]
        )
        for m in methods
    }

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
            raise MicroMaxError(f"Error: cluster not found: {cluster_path}")
        cluster_obj = load_reducer(cluster_path)
        if (not isinstance(cluster_obj, dict)
                or "pca_whiten" not in cluster_obj
                or "models" not in cluster_obj):
            msg = (f"{cluster_path} is not a cluster.pkl "
                   "(missing pca_whiten/models — point reduction.cluster at "
                   "the cluster.pkl written by a previous reduction run)")
            if raise_on_error:
                raise RuntimeError(msg)
            raise MicroMaxError(f"Error: {msg}")
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
        probe = PCA(n_components=n_cap, random_state=seed).fit(feats_fit)
        n_white = int(np.searchsorted(
            np.cumsum(probe.explained_variance_ratio_),
            CLUSTER_VARIANCE_TARGET) + 1)
        n_white = max(1, min(n_white, n_cap))
        pca_w = PCA(n_components=n_white, whiten=True,
                    random_state=seed).fit(feats_fit)
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
        uids = [d[UID_COLUMN] for d in db_dicts]
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")

        # Legacy table from the variance-tracking schema — no longer written.
        conn.execute("DROP TABLE IF EXISTS reduction_pca_variance")

        for m in methods:
            col = reduction_coord_prefix(m)
            table = reduction_table_name(m)
            arr = X_all[m]
            # Defensive guard against a reducer whose embedding does not
            # cover this dataset (e.g. a stale pre-fit pickle): never write
            # coordinates against mismatched uids / silently truncate.
            if len(arr) != end - start:
                msg = (
                    f"{m}: embedding has {len(arr)} rows but the DB block "
                    f"holds {end - start} rows — the reducer does not match "
                    f"this dataset (re-fit it or provide a matching pickle)."
                )
                conn.close()
                if raise_on_error:
                    raise RuntimeError(msg)
                raise MicroMaxError(f"Error: {msg}")
            # Drop first: the old PCA schema could carry pc_1..pc_k columns.
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(
                f"CREATE TABLE {table} ("
                f"{UID_COLUMN} INTEGER PRIMARY KEY, "
                f"{col}_1 REAL NOT NULL, {col}_2 REAL NOT NULL)")
            arr = arr[start:end]
            conn.executemany(
                f"INSERT OR REPLACE INTO {table} ({UID_COLUMN}, {col}_1, {col}_2) "
                f"VALUES (?, ?, ?)",
                list(zip(uids, arr[:, 0].tolist(), arr[:, 1].tolist())))

        if cluster_res_list:
            # One cluster_res_<tag> ID column plus one cluster_prob_<tag>
            # confidence column (kNN vote fraction, [0, 1]) per resolution.
            # Resolution tags may contain a dot ("0.5"), so every identifier
            # is quoted.
            id_cols = [f"{CLUSTER_RES_PREFIX}{_res_tag(r)}" for r in cluster_res_list]
            prob_cols = [f"{CLUSTER_PROB_PREFIX}{_res_tag(r)}" for r in cluster_res_list]
            cols = ", ".join(
                [f"{sql_ident(c)} INTEGER NOT NULL" for c in id_cols]
                + [f"{sql_ident(c)} REAL NOT NULL" for c in prob_cols])
            col_names = ", ".join(sql_ident(c) for c in id_cols + prob_cols)
            # The uid placeholder is written literally below; ph covers the
            # id+prob columns only.
            ph = ", ".join("?" * (len(id_cols) + len(prob_cols)))
            conn.execute(f"DROP TABLE IF EXISTS {FIND_CLUSTER_TABLE}")
            conn.execute(
                f"CREATE TABLE {FIND_CLUSTER_TABLE} "
                f"({UID_COLUMN} INTEGER PRIMARY KEY, {cols})")
            conn.executemany(
                f"INSERT OR REPLACE INTO {FIND_CLUSTER_TABLE} "
                f"({UID_COLUMN}, {col_names}) VALUES (?, {ph})",
                list(zip(uids,
                         *[cluster_ids[r][start:end].tolist()
                           for r in cluster_res_list],
                         *[cluster_probs[r][start:end].tolist()
                           for r in cluster_res_list])))
        else:
            # A re-run without clustering must not leave a stale find_cluster
            # table behind (microVis merges it into the scatter).
            conn.execute(f"DROP TABLE IF EXISTS {FIND_CLUSTER_TABLE}")

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
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (INFERENCE_TABLE,))
        if not cur.fetchone():
            logger.error("No '%s' table in %s. Run inference first.",
                         INFERENCE_TABLE, db_path)
            return None, None
        cur = conn.execute(f"PRAGMA table_info({INFERENCE_TABLE})")
        all_cols = [row[1] for row in cur.fetchall()]
        # Column names come from class names / regex captures and may contain
        # spaces, dots or quotes — quote every identifier (inference creates
        # them with sql_ident too).
        col_select = ", ".join(sql_ident(c) for c in all_cols)
        cur = conn.execute(
            f"SELECT {col_select} FROM {INFERENCE_TABLE} ORDER BY {UID_COLUMN}")
        col_names = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        if not rows:
            logger.error("No rows in %s table in %s.", INFERENCE_TABLE, db_path)
            return None, None
        dicts = [dict(zip(col_names, r)) for r in rows]
        # Merge the find_cluster columns (cluster_res_<resolution> IDs +
        # cluster_prob_<resolution> confidences) into the row dicts so every
        # one of them can serve as a color_by variable. Resolution tags may
        # contain a dot, so identifiers are quoted. Rows missing from a stale
        # find_cluster table simply keep no keys (d.get -> None downstream).
        has_fc = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (FIND_CLUSTER_TABLE,)
        ).fetchone()
        if has_fc:
            fc_cols = [
                row[1]
                for row in conn.execute(f"PRAGMA table_info({FIND_CLUSTER_TABLE})")
                if row[1] != UID_COLUMN
            ]
            if fc_cols:
                sel = ", ".join([UID_COLUMN] + [sql_ident(c) for c in fc_cols])
                fc_by_uid = {}
                for row in conn.execute(f"SELECT {sel} FROM {FIND_CLUSTER_TABLE}"):
                    fc_by_uid[int(row[0])] = dict(zip(fc_cols, row[1:]))
                for d in dicts:
                    extra = fc_by_uid.get(d[UID_COLUMN])
                    if extra:
                        d.update(extra)
        feats_arr = []
        for d in dicts:
            f = d[FEATURES_COLUMN]
            if f is None:
                msg = (
                    f"row uid={d.get(UID_COLUMN)} has NULL features. "
                    "Run inference with inference.feature: true"
                )
                if raise_on_error:
                    raise RuntimeError(f"{db_path}: {msg}")
                raise MicroMaxError(f"Error: {msg}")
            feats_arr.append(np.frombuffer(f, dtype=np.float32))
        feats = np.stack(feats_arr, axis=0)
        return feats, dicts
    finally:
        conn.close()


def _color_column_continuous(cb, dicts):
    """Classify a color_by column: True = continuous page, False = categorical.

    Probability columns (pred_prob, cluster_prob_<tag>) are always continuous
    — kNN vote confidences only take ~15 distinct fractions, which the
    distinct-count heuristic below would misread as categories. Other
    columns qualify as continuous when every non-null value parses as a
    number AND there are more distinct values than
    CONTINUOUS_COLOR_MAX_CLASSES. cluster_res_<tag> columns hold Leiden IDs
    whose numeric order encodes plot distance, not magnitude — they stay
    categorical no matter how many clusters a resolution produced.
    """
    if cb.startswith(CLUSTER_RES_PREFIX):
        return False
    if cb == PRED_PROB_COLUMN or cb.startswith(CLUSTER_PROB_PREFIX):
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

