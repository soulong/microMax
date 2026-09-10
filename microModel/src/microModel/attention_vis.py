"""Offline DINOv3 attention diagnostics (CLI: micromodel attention-vis).

run_attention_vis loads a trained SSL bundle (config.resume.ssl_model), recomputes
per-head/mean CLS attention maps and patch-similarity anchor maps on a fixed
subset of the training data, and writes one combined vis_attention.pdf.
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from microBase import MicroMaxError

from . import __version__
from .utils import logger, select_device, set_seed, add_file_logging
from .dataset import SSLMultiViewDataset
from .models import build_ssl_model
from .monitor import compute_patch_similarity_maps, compute_cls_attention_maps
from .pretrain import (_load_checkpoint_state, _merge_method_config,
                       _METHOD_ARCH_KEYS, _prepare_pretrain_data)
from .plots import plot_attention_combined, _to_display_rgb


def run_attention_vis(config, config_path=None):
    """Offline DINOv3 attention + patch-similarity visualization.

    Loads a trained SSL bundle (config.resume.ssl_model), re-computes the
    per-head/mean CLS attention maps and the patch-similarity anchor maps on
    a fixed subset of the same data, and writes a SINGLE combined PDF
    ({output_dir}/vis_attention.pdf) — no training happens.
    """
    if config.get("method") != "dinov3":
        raise MicroMaxError("Error: attention-vis currently supports method 'dinov3' only")
    resume_path = (config.get("resume") or {}).get("ssl_model")
    if not resume_path or not os.path.exists(resume_path):
        raise MicroMaxError(f"Error: resume.ssl_model not found ({resume_path!r}) — point it "
              "at the trained bundle (e.g. <run>/model.pt)")

    device = select_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    seed = 42
    output_dir = config.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)
    add_file_logging(output_dir)
    logger.info("microModel %s attention-vis (config=%s, device=%s)",
                __version__, config_path, device)
    set_seed(seed)

    (all_pairs, resolved_channels, _, _, max_value, normalize_method,
     with_masking, clip_low, clip_high, fixed_reference,
     augmentation_infer) = _prepare_pretrain_data(config)

    # ---- Fixed deterministic subset (same recipe as the UMAP check) ----
    n_samples = min(max(1, int((config.get("monitoring") or {})
                               .get("patch_similarity_samples", 6))),
                    len(all_pairs))
    n_umap = (config.get("training") or {}).get("n_image_umap")
    if n_umap:
        n_samples = min(n_samples, int(n_umap), len(all_pairs))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(all_pairs), size=n_samples,
                               replace=False).tolist())
    by_cell = {}
    for i in chosen:
        cell_ds, idx = all_pairs[i]
        by_cell.setdefault(id(cell_ds), (cell_ds, []))[1].append(idx)
    from torch.utils.data import ConcatDataset, DataLoader
    check_datasets = [
        SSLMultiViewDataset(
            cell_ds, indices, resolved_channels,
            augmentation_specs=[augmentation_infer],
            normalize_method=normalize_method, clip_low=clip_low,
            clip_high=clip_high, with_masking=with_masking,
            fixed_reference=fixed_reference, max_value=max_value)
        for cell_ds, indices in by_cell.values()
    ]
    loader = DataLoader(
        check_datasets[0] if len(check_datasets) == 1
        else ConcatDataset(check_datasets),
        batch_size=32, shuffle=False, num_workers=0)
    logger.info("attention-vis: %d images, bundle=%s", n_samples, resume_path)

    # ---- Load bundle into a dinov3 model (gram keys handled) ----
    # Architecture comes from the bundle meta (locked at training time); the
    # config carries no backbone block.
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta") or {}
    if not meta.get("backbone"):
        raise MicroMaxError(f"Error: bundle {resume_path} meta lacks 'backbone' — not an "
              f"SSL bundle")
    backbone_cfg = {"name": meta["backbone"], "pretrained": False,
                    "in_chans": meta.get("in_chans", len(resolved_channels))}
    if backbone_cfg["in_chans"] != len(resolved_channels):
        raise MicroMaxError(f"Error: bundle in_chans={backbone_cfg['in_chans']} does not match "
            f"len(data.channels)={len(resolved_channels)}; the model "
            f"architecture is fixed by the channel count the bundle was "
            f"trained with")
    # Method block: the bundle's saved config wins for the architecture keys
    # (the saved heads were built with them — building from the current
    # config's defaults would crash load_state_dict); current-config keys
    # win everywhere else, exactly like pretrain's continue-resume merge.
    # Monitoring is forced off: the per-step diag tensors are never consumed
    # by this offline pass.
    method_cfg = config.get("dinov3", {})
    saved_method_cfg = (ckpt.get("config") or {}).get("dinov3")
    if isinstance(saved_method_cfg, dict):
        method_cfg = _merge_method_config(saved_method_cfg, method_cfg,
                                          _METHOD_ARCH_KEYS)
    model = build_ssl_model(
        "dinov3", backbone_cfg, {**method_cfg,
                                 "monitoring": {"enabled": False}}, device)
    _load_checkpoint_state(model, ckpt, method="dinov3")
    model.eval()
    logger.info("Loaded bundle from epoch %d", ckpt.get("epoch", -1))

    # ---- Recompute maps on the fixed subset ----
    from .plots import plot_attention_combined, _to_display_rgb
    samples = []      # list of {'anchors': [...], 'attn': {...}}
    all_inputs = []
    anchor_idx = None
    grid = None
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            for i in range(len(x)):
                x1 = x[i:i + 1]
                amaps, grid, anchors = compute_patch_similarity_maps(
                    model, x1, n_anchors=4)
                att_maps, _ = compute_cls_attention_maps(
                    model.teacher_backbone, x1)
                if amaps and att_maps:
                    samples.append({"anchors": amaps, "attn": att_maps[0]})
                    all_inputs.append(_to_display_rgb(x[i].detach().cpu()))
                    if anchor_idx is None:
                        anchor_idx = anchors
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break
    if not samples:
        logger.warning("attention-vis produced no samples")
        return
    save_path = os.path.join(output_dir, "vis_attention.pdf")
    plot_attention_combined(samples, save_path, inputs=all_inputs,
                            anchor_idx=anchor_idx,
                            title=f"Attention + patch similarity "
                                  f"(offline, {len(samples)} samples)")
    logger.info("attention-vis PDF saved to %s (%d samples)",
                save_path, len(samples))
