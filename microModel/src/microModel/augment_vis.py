"""Augmentation preview (CLI: micromodel augment-vis).

show_augmentation adapts to both pretrain configs (augmentation_views) and
train configs (augmentation_train + augmentation_infer): it renders one
multi-page grid of raw + augmented views per sample and channel.
"""

import os
import sys
import random

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from microBase import (
    CellDataset,
    build_pipeline,
    apply,
    normalize,
)

from .dataset import _compute_ref_stats, _normalize_fixed, _to_float_max
from .utils import logger, add_file_logging, resolve_max_value


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
