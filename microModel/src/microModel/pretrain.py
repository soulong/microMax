"""SSL pretraining loop: build dataset, model, optimizer; run epochs; save SSL bundle.

Method-agnostic: dispatches train_step + criterion to the registered SSL method.
Each model file (models/<method>.py) owns its train_step function.
"""

import os
import sys
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import __version__
from .utils import (logger, set_seed, select_device, copy_config_file,
                    add_file_logging, atomic_torch_save, merge_locked_normalize,
                    resolve_channels, resolve_max_value, build_cell_datasets)
from .dataset import SSLMultiViewDataset, subsample
from .models import build_ssl_model, get_train_step, get_criterion
from .monitor import (MetricsTracker, head_collapse_metrics, gram_split_metrics,
                      compute_patch_similarity_maps, compute_cls_attention_maps,
                      maps_to_grid, resize_map)


def _save_bundle(output_dir, epoch, model, opt, loss_history, config, meta, method,
                 final=False, dino_loss_history=None, ibot_loss_history=None,
                 koleo_loss_history=None, gram_loss_history=None,
                 head_std_history=None, head_entropy_history=None,
                 teacher_student_sim_history=None, gram_masked_history=None,
                 gram_unmasked_history=None):
    """Save a complete SSL bundle: full model state (backbone + heads +
    momentum nets), meta, config, optimizer state, epoch, loss history.

    Every saved .pt is a complete bundle — usable for exact resume, train
    transfer, and feature extraction. final=True writes model.pt (final
    epoch); otherwise model_{epoch}.pt (save_interval epochs). The epoch
    is 1-based, matching the filename (no zero padding).
    """
    bundle = {
        "state_dict": model.state_dict(),
        "meta": meta,
        "config": config,
        "optimizer_state_dict": opt.state_dict(),
        "epoch": epoch,
        "loss_history": loss_history,
        "method": method,
    }
    # DINOv2 component histories (dino/ibot/koleo) are persisted so a
    # 'continue' resume restores them (absent in old bundles -> [] fallback).
    # DINOv3 additionally persists the gram anchoring loss history and the
    # monitoring diagnostics (head collapse / gram split) so their PDF curves
    # continue seamlessly across resumed runs.
    if dino_loss_history is not None:
        bundle["dino_loss_history"] = dino_loss_history
    if ibot_loss_history is not None:
        bundle["ibot_loss_history"] = ibot_loss_history
    if koleo_loss_history is not None:
        bundle["koleo_loss_history"] = koleo_loss_history
    if gram_loss_history is not None:
        bundle["gram_loss_history"] = gram_loss_history
    if head_std_history is not None:
        bundle["head_std_history"] = head_std_history
    if head_entropy_history is not None:
        bundle["head_entropy_history"] = head_entropy_history
    if teacher_student_sim_history is not None:
        bundle["teacher_student_sim_history"] = teacher_student_sim_history
    if gram_masked_history is not None:
        bundle["gram_masked_history"] = gram_masked_history
    if gram_unmasked_history is not None:
        bundle["gram_unmasked_history"] = gram_unmasked_history
    fname = "model.pt" if final else f"model_{epoch}.pt"
    path = os.path.join(output_dir, fname)
    atomic_torch_save(bundle, path)
    logger.info("SSL bundle saved to %s (epoch %d)", path, epoch)


def _run_umap_check(model, method, loader, device, epoch, seed, save_path):
    """Extract backbone features of the fixed UMAP-check subset (deterministic
    augmentation_infer pipeline) and save a fresh UMAP scatter PDF.

    Auxiliary monitoring aid: failures are caught by the caller (logged and
    skipped) so a plotting hiccup never aborts training.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    model.eval()
    feats = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if method in ("dinov2", "dinov3"):
                f = model.student_backbone.encode(x)[:, 0]
            else:
                f = model.backbone(x)
                if f.ndim == 4:
                    f = f.mean(dim=(2, 3))
                elif f.ndim == 3:
                    f = f.mean(dim=1)
            feats.append(f.cpu().numpy())
    X = np.concatenate(feats, axis=0)

    reducer = umap.UMAP(random_state=seed)
    emb = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], s=12, alpha=0.8, edgecolors="none")
    ax.set_title(f"UMAP check — epoch {epoch} ({len(X)} images)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP check saved to %s (epoch %d, %d images)", save_path, epoch, len(X))


def _run_attention_summary_check(model, loader, device, save_path, n_samples,
                                 tracker, step):
    """Combined diagnostic PDF at save_interval / final epochs: one row per
    image with the input, the patch-similarity anchors and the per-head +
    mean CLS attention maps side by side. Also writes two TensorBoard images
    (patch anchors / mean attention). Skips when no umap loader is available;
    failures are caught by the caller.
    """
    if loader is None:
        logger.warning("attention summary check skipped (no UMAP-check "
                       "loader; set training.n_image_umap to enable)")
        return
    from .vis import plot_attention_combined, _to_display_rgb
    model.eval()
    n_samples = max(1, int(n_samples))
    samples = []      # list of {'anchors': [...], 'attn': {...}}
    all_inputs = []
    anchor_idx = None
    grid = None
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            need = n_samples - len(samples)
            for i in range(min(need, len(x))):
                x1 = x[i:i + 1]
                amaps, grid, anchors = compute_patch_similarity_maps(
                    model, x1, n_anchors=4)
                att_maps, _ = compute_cls_attention_maps(
                    model.student_backbone, x1)
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
        logger.warning("attention summary check skipped (no valid samples)")
        return
    plot_attention_combined(samples, save_path, inputs=all_inputs,
                            anchor_idx=anchor_idx,
                            title=f"Attention + patch similarity (epoch {step})")
    # TensorBoard images: input + patch-anchor grid / input + mean attention
    target = all_inputs[0].shape[:2]
    tracker.add_image("patch_similarity/sample_0",
                      np.hstack([all_inputs[0],
                                 resize_map(maps_to_grid(samples[0]["anchors"],
                                                         grid), target)
                                 .clip(0, 255).astype(np.uint8)]), step)
    mean8 = (np.clip(samples[0]["attn"]["mean"], 0.0, 1.0) * 255.0).astype(np.uint8)
    tracker.add_image("attention/sample_0",
                      np.hstack([all_inputs[0],
                                 resize_map(np.stack([mean8] * 3, axis=-1),
                                            target)
                                 .clip(0, 255).astype(np.uint8)]), step)
    logger.info("Attention + patch similarity maps saved to %s (epoch %d, "
                "%d samples)", save_path, step, len(samples))


def _load_checkpoint_state(model, ckpt, method=None):
    """Restore model state from a bundle (full model state_dict).

    For DINOv3 phase-2 gram runs, the fresh model carries a frozen gram
    teacher (gram_backbone, plus optionally gram_teacher/gram_ema buffers)
    that a phase-1 bundle (trained without gram) does not have — such bundles
    are loaded non-strictly, keeping every gram-* key at its
    gram.ckpt-initialized value instead of failing on a mismatch.
    """
    if "state_dict" not in ckpt:
        print("Error: bundle has no 'state_dict' key (unsupported pre-0.2.1 "
              "bundle format)", file=sys.stderr)
        sys.exit(1)
    meta = ckpt.get("meta") or {}
    if "method" not in meta:
        print(
            "Error: bundle is not an SSL bundle (meta has no 'method') — "
            "resume.ssl_model expects a pretrain checkpoint, not a train "
            "bundle.",
            file=sys.stderr,
        )
        sys.exit(1)
    state = ckpt["state_dict"]
    if method == "dinov3":
        model_keys = set(model.state_dict())
        bundle_keys = set(state)
        missing = model_keys - bundle_keys
        gram_prefixes = ("gram_backbone.", "gram_teacher.")
        if missing and all(k.startswith(gram_prefixes) for k in missing):
            # Bundle predates the gram teacher: every schema-diff key belongs
            # to the gram modules, so they stay at the values loaded from
            # gram.ckpt. Anything else missing is real schema drift and must
            # fail loudly.
            logger.info(
                "Bundle has no gram-* state (%d gram keys kept from "
                "dinov3.gram.ckpt); loading non-strictly", len(missing))
            model.load_state_dict(state, strict=False)
            return
    model.load_state_dict(state)


def _try_resume(config, device, method):
    """Load SSL checkpoint if resume.ssl_model is set. Returns (checkpoint, config)."""
    resume_path = config.get("resume", {}).get("ssl_model")
    if not resume_path:
        return None, config

    if not os.path.exists(resume_path):
        print(f"Error: resume SSL model not found: {resume_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    logger.info("Loaded SSL checkpoint from epoch %d", ckpt.get("epoch", -1))

    saved_cfg = ckpt.get("config")
    if saved_cfg is None:
        print(f"Error: SSL checkpoint at {resume_path} has no 'config' key", file=sys.stderr)
        sys.exit(1)

    # Locked keys (bundle wins on mismatch): method, backbone.name, normalize.*
    # augmentation_views is deliberately NOT locked: the current run's view
    # flow always wins (handled right below), so any later run — including the
    # DINOv3 gram phase-2 config appending its gram-teacher crops — can freely
    # redefine augmentation.
    locked_keys = [
        ("method",),
        ("backbone", "name"),
    ]
    for key_path in locked_keys:
        sv = saved_cfg
        cv = config
        ok = True
        for k in key_path:
            if not isinstance(sv, dict) or k not in sv:
                ok = False
                break
            if not isinstance(cv, dict) or k not in cv:
                ok = False
                break
            sv = sv[k]
            cv = cv[k]
        if ok and str(cv) != str(sv):
            logger.warning("Locked %s differs: bundle=%s, config=%s; using bundle value",
                           ".".join(key_path), sv, cv)
            # Apply bundle value to config
            target = config
            for k in key_path[:-1]:
                target = target.setdefault(k, {})
            target[key_path[-1]] = sv

    # ---- augmentation_views: always prefer the NEW config (all methods) ----
    # Policy: if the current run provides augmentation_views, it wins —
    # regardless of what the bundle saved. This lets e.g. the DINOv3 gram
    # phase-2 config append its gram-teacher crops to the phase-1 view flow,
    # or any later run adjust views freely. The bundle value is only echoed
    # as an informational note.
    sv_views = saved_cfg.get("augmentation_views")
    cv_views = config.get("augmentation_views")
    if cv_views is not None and sv_views is not None \
            and str(cv_views) != str(sv_views):
        logger.info("Using new augmentation_views from config (%d views; "
                    "bundle had %d) — bundle value ignored by policy",
                    len(cv_views), len(sv_views))

    merge_locked_normalize(saved_cfg, config)

    # Method-specific block (e.g. byol/dinov3 head dims) — bundle wins for the
    # keys it saved, so the saved model state (heads included) loads without
    # shape mismatch. A leaf-wise merge (not full replace) lets a NEW run
    # extend the block, while every leaf the bundle saved still wins. The one
    # deliberate exception is the DINOv3 gram phase-2 upgrade: a run that
    # explicitly sets gram.use_loss=true while the bundle was trained with it
    # off is a Gram-anchoring stage — the new gram settings are kept.
    final_method = (ckpt.get("meta") or {}).get("method", method)
    sm = saved_cfg.get(final_method, {})
    cm = config.get(final_method, {})
    if str(cm) != str(sm):
        merged = dict(cm)
        for k, v in sm.items():
            cur = merged.get(k)
            if isinstance(v, dict) and isinstance(cur, dict):
                if (cur.get("use_loss") is True and v.get("use_loss") is False):
                    continue  # gram phase-2 upgrade: keep the new gram block
                sub = dict(cur)
                for sk, sv in v.items():
                    sub[sk] = sv
                merged[k] = sub
            else:
                merged[k] = v
        config[final_method] = merged

    return ckpt, config


def _build_records(cell_ds):
    """Walk a CellDataset's metadata DataFrame and return list of (cell_ds, idx) pairs."""
    return [(cell_ds, i) for i in range(len(cell_ds.metadata))]


def _resolve_monitoring_cfg(user_cfg):
    """Merge the user's `monitoring:` block over defaults (None entries drop
    back to defaults). enabled=false disables everything (current behaviour)."""
    defaults = {
        "enabled": True,
        "tensorboard": True,
        "csv": True,
        "log_every_steps": 10,
        "patch_similarity_samples": 6,
        "attention_maps": True,
        "gram_split_stats": True,
        "head_logits_track": True,
    }
    for k, v in (user_cfg or {}).items():
        if v is not None:
            defaults[k] = v
    return defaults


def _build_step_info(method, method_cfg, train_cfg, global_step, total_steps, warmup_steps):
    """Build the step_info dict for train_step (method-specific keys).

    Shared by AMP/non-AMP paths. Method-specific keys (DINOv2/DINOv3 teacher
    temperature, weight decay schedule, koleo weight) are added only for
    dinov2/dinov3.
    """
    step_info = {
        "global_step": global_step,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "lr_peak": train_cfg.get("lr", 0.0005 if method in ("dinov2", "dinov3") else 0.05),
        "lr_final": method_cfg.get("lr_final", 1e-6),
        "momentum_start": method_cfg.get(
            "momentum_start", 0.992 if method in ("dinov2", "dinov3") else 0.996),
        "momentum_end": method_cfg.get("momentum_end", 1.0),
    }
    if method in ("dinov2", "dinov3"):
        step_info.update({
            "koleo_weight": method_cfg.get("koleo_weight", 0.1),
            "teacher_temp_start": method_cfg.get("teacher_temp_start", 0.04),
            "teacher_temp_end": method_cfg.get("teacher_temp_end", 0.07),
            "weight_decay_start": method_cfg.get("weight_decay_start", 0.04),
            "weight_decay_end": method_cfg.get("weight_decay_end", 0.4),
        })
    return step_info


def _prepare_pretrain_data(config):
    """Resolve data roots -> (cell, idx) pairs plus the channels/normalize/
    input settings shared by pretrain_ssl and the offline vis_attention
    command.

    -> (all_pairs, resolved_channels, channel_layout, image_pattern,
        max_value, normalize_method, with_masking, clip_low, clip_high,
        fixed_reference, augmentation_infer)
    """
    seed = 42
    data_cfg = config["data"]
    norm_cfg = config.get("normalize", {})
    normalize_method = norm_cfg.get("method", "per_channel")
    with_masking = norm_cfg.get("with_masking", False)
    clip_low = norm_cfg.get("clip_low", 0.05)
    clip_high = norm_cfg.get("clip_high", 99.95)
    fixed_reference = norm_cfg.get("fixed_reference", False)
    max_value = resolve_max_value(data_cfg)

    root = data_cfg["root"]
    roots = [root] if isinstance(root, str) else list(root)
    channels = data_cfg.get("channels")
    channel_layout = data_cfg.get("channel_layout", "CHW")
    image_pattern = data_cfg.get("image_pattern")

    datasets = build_cell_datasets(roots, channel_layout, image_pattern)
    all_pairs = []
    root_of_pair = []
    resolved_per_root = {}
    for r, cell_ds in datasets:
        n_avail = len(cell_ds.intensity_colnames)
        resolved_per_root[r] = resolve_channels(channels, n_avail, r)
        recs = _build_records(cell_ds)
        all_pairs.extend(recs)
        root_of_pair.extend([r] * len(recs))

    if not all_pairs:
        print(f"Error: no records found in {roots}", file=sys.stderr)
        sys.exit(1)

    # All roots must resolve to the same channel set — one bundle carries a
    # single `channels` meta, so a heterogeneous resolution would silently
    # train on the wrong channels for some roots (last-root-wins bug).
    unique_resolved = {tuple(v) for v in resolved_per_root.values()}
    if len(unique_resolved) > 1:
        print(
            f"Error: data roots resolve to different channel sets: "
            f"{ {r: v for r, v in resolved_per_root.items()} }. "
            f"Give every root the same channel count or set data.channels "
            f"explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    resolved_channels = list(next(iter(unique_resolved))) if unique_resolved else channels

    # Subsample
    sample_max = data_cfg.get("sample_max")
    sample_by = data_cfg.get("sample_by", "per_dataset")
    if sample_max is not None:
        # Wrap pairs as dicts for subsample; the real root is carried along so
        # sample_by='per_dataset' caps per data root (not across all roots).
        items = [{"pair": p, "root": r} for p, r in zip(all_pairs, root_of_pair)]
        items = subsample(items, sample_max, sample_by, seed,
                          label_key="__unlabeled__", root_key="root")
        all_pairs = [it["pair"] for it in items]
        logger.info("Sub-sampled to %d records (sample_max=%s, sample_by=%s)",
                    len(all_pairs), sample_max, sample_by)

    augmentation_infer = config.get("augmentation_infer")
    if augmentation_infer is None:
        print(
            "Error: augmentation_infer is required in pretrain config (must be a "
            "deterministic pipeline for inference/UMAP-check). Set it explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    return (all_pairs, resolved_channels, channel_layout, image_pattern,
            max_value, normalize_method, with_masking, clip_low, clip_high,
            fixed_reference, augmentation_infer)


def pretrain_ssl(config, config_path=None):
    """Generic SSL pretraining loop. Dispatches on config['method']."""
    method = config.get("method")
    if method not in ("byol", "dinov2", "dinov3"):
        print(f"Error: unknown SSL method '{method}'. Available: byol, dinov2, dinov3",
              file=sys.stderr)
        sys.exit(1)

    mode = config.get("mode", "single_cell")
    if mode not in ("single_cell",):
        print(f"Error: pretrain mode '{mode}' is not supported. Available: "
              f"single_cell (whole_image is planned but not yet implemented)",
              file=sys.stderr)
        sys.exit(1)

    device = select_device()
    if device.type == "cuda":
        # fp16 matmul reduced-precision reduction produced rare NaN on some
        # GPUs (cuBLAS accumulation overflow) even with small, finite
        # activations. Force fp32 accumulation for fp16 matmuls.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    checkpoint, config = _try_resume(config, device, method)
    # _try_resume may have replaced config["method"] with the bundle's method
    # (locked key, bundle wins) — re-read it so every downstream dispatch
    # (method_cfg, build_ssl_model, meta, optimizer/criterion) uses the method
    # the model was ACTUALLY built with.
    method = config["method"]
    if method not in ("byol", "dinov2", "dinov3"):
        print(f"Error: unknown SSL method '{method}'. Available: byol, dinov2, dinov3",
              file=sys.stderr)
        sys.exit(1)

    # resume.type: 'continue' = exact same-data extension (schedules/optimizer/
    # epoch counter continue where the interrupted run left off); 'transfer' =
    # domain transfer on new data — fresh run (epoch 0, schedules restart,
    # optimizer state reset) initialized from the bundle weights.
    resume_type = config.get("resume", {}).get("type", "continue")
    if resume_type not in ("continue", "transfer"):
        print(f"Error: unknown resume.type '{resume_type}'. "
              f"Available: continue, transfer", file=sys.stderr)
        sys.exit(1)

    data_cfg = config["data"]
    train_cfg = config["training"]
    backbone_cfg = config["backbone"]
    method_cfg = config.get(method, {})
    norm_cfg = config.get("normalize", {})
    aug_views_cfg = config.get("augmentation_views", [])
    monitoring_cfg = _resolve_monitoring_cfg(config.get("monitoring"))
    if monitoring_cfg["enabled"]:
        logger.info("Monitoring enabled: tensorboard=%s csv=%s "
                    "log_every_steps=%d patch_similarity_samples=%d",
                    monitoring_cfg["tensorboard"], monitoring_cfg["csv"],
                    monitoring_cfg["log_every_steps"],
                    monitoring_cfg["patch_similarity_samples"])

    # Validate views count
    if method == "byol" and len(aug_views_cfg) < 2:
        print(f"Error: BYOL requires at least 2 views, got {len(aug_views_cfg)}", file=sys.stderr)
        sys.exit(1)
    if method == "dinov2" and len(aug_views_cfg) < 3:
        print(f"Error: DINOv2 requires at least 3 views (2 global + 1 local), "
              f"got {len(aug_views_cfg)}", file=sys.stderr)
        sys.exit(1)
    if method == "dinov3":
        gram_cfg = method_cfg.get("gram", {}) or {}
        n_gram_views = 2 if (gram_cfg.get("use_loss", False)
                             and not gram_cfg.get("ema_teacher", False)) else 0
        min_views = 3 + n_gram_views
        if len(aug_views_cfg) < min_views:
            print(f"Error: DINOv3 requires at least {min_views} views "
                  f"(2 global + 1 local{' + 2 gram-teacher crops' if n_gram_views else ''}), "
                  f"got {len(aug_views_cfg)}", file=sys.stderr)
            sys.exit(1)

    seed = 42
    output_dir = config.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)
    add_file_logging(output_dir)
    tracker = (MetricsTracker(output_dir,
                              tensorboard=monitoring_cfg["tensorboard"],
                              csv_enabled=monitoring_cfg["csv"],
                              log_every_steps=monitoring_cfg["log_every_steps"])
               if monitoring_cfg["enabled"] else None)
    if config_path is not None:
        dst = copy_config_file(config_path, output_dir)
        logger.info("Pretrain config copied to %s", dst)
    logger.info("microModel %s pretrain starting (method=%s, config=%s, device=%s)",
                __version__, method, config_path, device)

    set_seed(seed)

    # ---- Resolve data (shared with the offline vis_attention command) ----
    (all_pairs, resolved_channels, channel_layout, image_pattern,
     max_value, normalize_method, with_masking, clip_low, clip_high,
     fixed_reference, augmentation_infer) = _prepare_pretrain_data(config)

    logger.info("Found %d records, channels=%s", len(all_pairs), resolved_channels)
    logger.info("SSL method: %s, views: %d", method, len(aug_views_cfg))

    set_seed(seed)

    # ---- Build dataset + dataloader ----
    # SSLMultiViewDataset takes (cell_dataset, indices, channels, ...)
    # We group pairs by cell_dataset to pass indices per dataset. But since
    # SSLMultiViewDataset takes a single cell_dataset, we need to handle
    # multi-root by building one dataset per root and concatenating.
    # Simpler: build one SSLMultiViewDataset per cell_dataset, then ConcatDataset.
    from torch.utils.data import ConcatDataset
    by_cell_ds = {}
    for cell_ds, idx in all_pairs:
        by_cell_ds.setdefault(id(cell_ds), (cell_ds, []))[1].append(idx)

    datasets = []
    for cell_ds_id, (cell_ds, indices) in by_cell_ds.items():
        ds = SSLMultiViewDataset(
            cell_ds, indices, resolved_channels,
            augmentation_specs=aug_views_cfg,
            normalize_method=normalize_method, clip_low=clip_low, clip_high=clip_high,
            with_masking=with_masking, fixed_reference=fixed_reference,
            max_value=max_value)
        datasets.append(ds)
    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = ConcatDataset(datasets)

    dl_cfg = config.get("dataloader", {})
    num_workers = dl_cfg.get("num_workers", 4)
    prefetch_factor = dl_cfg.get("prefetch_factor", 2)
    persistent_workers = dl_cfg.get("persistent_workers", True) and num_workers > 0
    batch_size = train_cfg.get("batch_size", 128)
    if method == "dinov2" and batch_size % 2 != 0:
        # dinov2.train_step chunks the teacher's global-view outputs in pairs
        # (views[0:2]) — an odd batch crashes lightly's DINOLoss stack.
        print(f"Error: DINOv2 requires an EVEN training.batch_size "
              f"(global views are paired); got {batch_size}", file=sys.stderr)
        sys.exit(1)
    loader_kwargs = dict(
        batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=(device.type == "cuda"),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(full_dataset, **loader_kwargs)

    # ---- Build model + optimizer + criterion ----
    # backbone.in_chans is NOT a config key — it is derived from the resolved
    # data channels. On resume the bundle's in_chans wins (the model was built
    # with it), so a changed data.channels hard-exits instead of crashing.
    if checkpoint is not None and "meta" in checkpoint:
        backbone_cfg["in_chans"] = checkpoint["meta"].get(
            "in_chans", len(resolved_channels))
    else:
        backbone_cfg["in_chans"] = len(resolved_channels)
    if backbone_cfg["in_chans"] != len(resolved_channels):
        print(
            f"Error: bundle in_chans={backbone_cfg['in_chans']} does not match "
            f"len(data.channels)={len(resolved_channels)}; the model architecture "
            f"is fixed by the channel count the bundle was trained with",
            file=sys.stderr,
        )
        sys.exit(1)
    model = build_ssl_model(method, backbone_cfg, method_cfg, device)

    # DINOv3 gram anchoring phase 2: initialize the frozen gram teacher from
    # gram.ckpt (an earlier SSL bundle, typically the phase-1 model.pt). When
    # no ckpt is given, the gram teacher is initialized later from the EMA
    # teacher at gram.it_load_ema_teacher.
    if method == "dinov3":
        gram_cfg = method_cfg.get("gram", {}) or {}
        if gram_cfg.get("use_loss", False) and not gram_cfg.get("ema_teacher", False):
            gram_ckpt = gram_cfg.get("ckpt")
            gram_ckpt = None if gram_ckpt == "ignore" else gram_ckpt
            if gram_ckpt:
                model.load_gram_from_bundle(gram_ckpt)

    # Bundle meta — needed by every saved bundle (infer/train consumers).
    if method in ("dinov2", "dinov3"):
        feat_dim = model.student_backbone.vit.num_features
    else:
        feat_dim = getattr(model.backbone, "num_features", None)
    meta = {
        "method": method,
        "backbone": backbone_cfg["name"],
        "in_chans": backbone_cfg["in_chans"],
        "channels": resolved_channels,
        "channel_layout": channel_layout,
        "max_value": max_value,
        "feat_dim": feat_dim,
        "augmentation_infer": augmentation_infer,
        "normalize_method": normalize_method,
        "normalize_with_masking": with_masking,
        "normalize_fixed_reference": fixed_reference,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "image_pattern": image_pattern,
    }

    # Separate parameters into decay (ndim >= 2: weight matrices, embeddings)
    # and no-decay (ndim < 2: biases, LayerNorm gamma/beta) groups. Standard
    # ViT practice — applying wd=0.4 to biases/LN is too aggressive.
    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    optimizer_name = train_cfg.get("optimizer", "sgd" if method == "byol" else "adamw").lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {"params": decay_params, "weight_decay": train_cfg.get("weight_decay", 0.0)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=train_cfg.get("lr", 0.06),
            momentum=0.9,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": train_cfg.get("weight_decay", 0.04)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=train_cfg.get("lr", 0.001),
            betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        )
    else:
        print(f"Error: unknown optimizer '{optimizer_name}'", file=sys.stderr)
        sys.exit(1)

    criterion = get_criterion(method, method_cfg, device)
    train_step_fn = get_train_step(method)

    amp_enabled = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    grad_clip = train_cfg.get("grad_clip", 10.0)

    epochs = train_cfg.get("epochs", 10)
    save_interval = train_cfg.get("save_interval")
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    if not isinstance(grad_accum_steps, int) or grad_accum_steps < 1:
        print(f"Error: training.grad_accum_steps must be an integer >= 1, "
              f"got {grad_accum_steps}", file=sys.stderr)
        sys.exit(1)

    start_epoch = 0
    loss_history = []
    dino_loss_history = []
    ibot_loss_history = []
    koleo_loss_history = []
    gram_loss_history = []
    head_std_history = []
    head_entropy_history = []
    sts_history = []
    gram_masked_history = []
    gram_unmasked_history = []
    if checkpoint is not None:
        _load_checkpoint_state(model, checkpoint, method=method)
        if resume_type == "continue":
            # Bundle 'epoch' is 1-based (last completed epoch), which equals
            # the next 0-based loop index to continue from. Exact extension:
            # optimizer state + loss history are restored so the run
            # continues where it left off.
            start_epoch = checkpoint.get("epoch", 0)
            logger.info("Resuming from epoch %d", start_epoch)
            # Phase-handoff extension: when the checkpoint has already passed
            # the configured epoch target (e.g. phase-2 reuses a smaller
            # epoch count than phase-1 ended at), treat `epochs` as EXTRA
            # epochs to run on top of the checkpoint instead of empty-running.
            if start_epoch >= epochs:
                logger.info(
                    "Checkpoint at epoch %d >= configured epochs=%d — "
                    "running %d additional epochs (new target %d)",
                    start_epoch, epochs, epochs, start_epoch + epochs)
                epochs = start_epoch + epochs
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            loss_history = checkpoint.get("loss_history", [])
            # DINOv2/DINOv3 component histories (absent in old bundles -> []).
            dino_loss_history = checkpoint.get("dino_loss_history", [])
            ibot_loss_history = checkpoint.get("ibot_loss_history", [])
            koleo_loss_history = checkpoint.get("koleo_loss_history", [])
            gram_loss_history = checkpoint.get("gram_loss_history", [])
            # DINOv3 monitoring diagnostics (absent in old bundles -> []).
            head_std_history = checkpoint.get("head_std_history", [])
            head_entropy_history = checkpoint.get("head_entropy_history", [])
            sts_history = checkpoint.get("teacher_student_sim_history", [])
            gram_masked_history = checkpoint.get("gram_masked_history", [])
            gram_unmasked_history = checkpoint.get("gram_unmasked_history", [])
        else:
            # Transfer: domain-transfer / pretrained-weight init. The run
            # restarts at epoch 0 with fresh schedules and a fresh optimizer
            # (AdamW/SGD moments are stale for the new data); only the model
            # weights are carried over.
            logger.info("Transfer: fresh run (epoch 0) initialized from bundle "
                        "weights; optimizer state and loss history reset")

    # ---- SSL step tracking ----
    # Schedules (LR / momentum / weight decay / teacher temp) span the full
    # run (epochs * len(loader)). In 'continue' mode global_step is the
    # absolute step position (start_epoch * len(loader)), so schedules extend
    # seamlessly where the interrupted run left off; in 'transfer'/fresh mode
    # it starts at 0 and schedules restart from their start values.
    total_steps = max(1, epochs * len(loader))
    global_step = start_epoch * len(loader)
    if method == "dinov3" and resume_type == "continue":
        # Restore the number of gram-teacher refreshes already done so the
        # rep_update schedule continues seamlessly (official behavior).
        model.set_gram_resume_updates(global_step)
    if resume_type == "transfer":
        # Transfer: short warmup (default 1 epoch) — pretrained weights only
        # need a brief warmup for the fresh optimizer to estimate gradient
        # statistics. The full warmup_epochs is for fresh pretrain from scratch.
        transfer_warmup = method_cfg.get("transfer_warmup_epochs", 1)
        warmup_steps = int(transfer_warmup * len(loader))
    else:
        warmup_steps = int(method_cfg.get("warmup_epochs", 30) * len(loader))

    # ---- UMAP check setup (training.n_image_umap) ----
    # A fixed random image subset (picked once with the training seed) is
    # re-embedded with a FRESH UMAP at each save_interval epoch and at the
    # final epoch, so per-epoch PDFs are directly comparable. Features use
    # the deterministic augmentation_infer pipeline. Disabled when 0/null.
    n_image_umap = train_cfg.get("n_image_umap")
    if n_image_umap is not None:
        try:
            n_image_umap = int(n_image_umap)
        except (TypeError, ValueError):
            print(f"Error: training.n_image_umap must be an integer, got {n_image_umap!r}",
                  file=sys.stderr)
            sys.exit(1)
        if n_image_umap < 0:
            print("Error: training.n_image_umap must be >= 0 (0 = disabled)",
                  file=sys.stderr)
            sys.exit(1)
    umap_check_loader = None
    last_umap_epoch = None
    if n_image_umap:
        n_pick = min(n_image_umap, len(all_pairs))
        if n_pick < 5:
            logger.warning("n_image_umap=%s but only %d images available; "
                           "UMAP check disabled (< 5)", n_image_umap, len(all_pairs))
        else:
            rng = np.random.default_rng(seed)
            chosen = sorted(rng.choice(len(all_pairs), size=n_pick, replace=False).tolist())
            by_cell = {}
            for i in chosen:
                cell_ds, idx = all_pairs[i]
                by_cell.setdefault(id(cell_ds), (cell_ds, []))[1].append(idx)
            check_datasets = [
                SSLMultiViewDataset(
                    cell_ds, indices, resolved_channels,
                    augmentation_specs=[augmentation_infer],
                    normalize_method=normalize_method, clip_low=clip_low,
                    clip_high=clip_high, with_masking=with_masking,
                    fixed_reference=fixed_reference, max_value=max_value)
                for cell_ds, indices in by_cell.values()
            ]
            umap_check_loader = DataLoader(
                check_datasets[0] if len(check_datasets) == 1 else ConcatDataset(check_datasets),
                batch_size=32, shuffle=False, num_workers=0)
            logger.info("UMAP check enabled: %d fixed images, fresh UMAP at "
                        "epoch 0 baseline + each save_interval + final epoch",
                        n_pick)

    # ---- Pre-training baseline UMAP check (epoch 0) ----
    # Features BEFORE any training (fresh pretrained init, or transfer
    # weights). Skipped when actually resuming with resume.type=continue —
    # the run is an exact extension of a previous one, so the baseline
    # belongs to that run.
    if umap_check_loader is not None and (checkpoint is None or resume_type != "continue"):
        try:
            _run_umap_check(model, method, umap_check_loader, device, 0, seed,
                            os.path.join(output_dir, "umap_check_epoch_0.pdf"))
        except Exception as e:
            logger.error("UMAP check failed at epoch 0: %s", e)

    # ---- Training loop ----
    gram_enabled = bool((method_cfg.get("gram") or {}).get("use_loss", False))
    diag_enabled = (monitoring_cfg["enabled"] and method == "dinov3")
    for epoch in range(start_epoch, epochs):
        model.train()
        tot_loss = 0.0
        tot_dino = 0.0
        tot_ibot = 0.0
        tot_koleo = 0.0
        tot_gram = 0.0
        tot_head_std = 0.0
        tot_head_entropy = 0.0
        tot_sts = 0.0
        tot_gram_masked = 0.0
        tot_gram_unmasked = 0.0
        n_diag = 0
        n_gram_diag = 0
        n_batches = 0
        batch_in_epoch = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            # batch is a list of N view tensors (each (B, C, H, W))
            # DataLoader auto-collates the list returned by SSLMultiViewDataset
            do_step = ((batch_in_epoch + 1) % grad_accum_steps == 0) or \
                (batch_in_epoch + 1 == len(loader))
            step_info = _build_step_info(
                method, method_cfg, train_cfg, global_step, total_steps, warmup_steps)
            if amp_enabled:
                # AMP: forward under autocast; train_step does its own scaled
                # backward + optimizer step via the GradScaler.
                with torch.amp.autocast("cuda"):
                    if method == "dinov3":
                        loss, dino_l, ibot_l, koleo_l, gram_l = train_step_fn(
                            model, batch, optimizer, epoch, epochs,
                            device, criterion, step_info,
                            scaler, grad_clip, do_step)
                    else:
                        loss, dino_l, ibot_l, koleo_l = train_step_fn(
                            model, batch, optimizer, epoch, epochs,
                            device, criterion, step_info,
                            scaler, grad_clip, do_step)
                        gram_l = None
            else:
                if method == "dinov3":
                    loss, dino_l, ibot_l, koleo_l, gram_l = train_step_fn(
                        model, batch, optimizer, epoch, epochs,
                        device, criterion, step_info,
                        scaler, grad_clip, do_step)
                else:
                    loss, dino_l, ibot_l, koleo_l = train_step_fn(
                        model, batch, optimizer, epoch, epochs,
                        device, criterion, step_info,
                        scaler, grad_clip, do_step)
                    gram_l = None
            if not math.isfinite(loss):
                print(f"Error: non-finite loss ({loss}) at epoch {epoch + 1}; "
                      f"aborting to avoid writing a corrupted SSL bundle",
                      file=sys.stderr)
                sys.exit(1)
            tot_loss += loss
            if dino_l is not None:
                tot_dino += dino_l
                tot_ibot += ibot_l
                tot_koleo += koleo_l
            if gram_l is not None:
                tot_gram += gram_l

            # Monitoring: DINOv3 diagnostics + step-level metrics (CSV/TB).
            if diag_enabled and tracker is not None and getattr(model, "_last_diag", None):
                diag = model._last_diag
                step_metrics = {"loss": loss, "dino": dino_l, "ibot": ibot_l,
                                "koleo": koleo_l, "gram": gram_l}
                if monitoring_cfg.get("head_logits_track", True):
                    hm = head_collapse_metrics(diag, model._student_temp)
                    step_metrics.update(hm)
                    tot_head_std += hm["head_logits_std"]
                    tot_head_entropy += hm["head_entropy"]
                    tot_sts += hm["teacher_student_sim"]
                    n_diag += 1
                gram_diag = diag.get("gram")
                if (gram_diag is not None and gram_l is not None
                        and monitoring_cfg.get("gram_split_stats", True)):
                    gm = gram_split_metrics(criterion[3], *gram_diag)
                    step_metrics.update(gm)
                    tot_gram_masked += gm.get("gram_masked", 0.0)
                    tot_gram_unmasked += gm.get("gram_unmasked", 0.0)
                    n_gram_diag += 1
                for k in ("lr", "momentum", "weight_decay", "teacher_temp"):
                    v = diag.get(k)
                    if v is not None:
                        step_metrics[k] = v
                if global_step % tracker.log_every_steps == 0:
                    tracker.add_step(global_step, epoch + 1, step_metrics)
            n_batches += 1
            batch_in_epoch += 1
            global_step += 1

        avg_loss = tot_loss / n_batches if n_batches else 0.0
        loss_history.append(avg_loss)
        if method in ("dinov2", "dinov3") and n_batches:
            dino_loss_history.append(tot_dino / n_batches)
            ibot_loss_history.append(tot_ibot / n_batches)
            koleo_loss_history.append(tot_koleo / n_batches)
            if method == "dinov3" and gram_enabled:
                gram_loss_history.append(tot_gram / n_batches if n_batches else 0.0)
                logger.info("  epoch=%d  loss=%.6f  (dino=%.4f  ibot=%.4f  "
                            "koleo=%.4f  gram=%.4f)",
                            epoch + 1, avg_loss,
                            tot_dino / n_batches, tot_ibot / n_batches,
                            tot_koleo / n_batches, tot_gram / n_batches)
            elif method == "dinov3":
                logger.info("  epoch=%d  loss=%.6f  (dino=%.4f  ibot=%.4f  "
                            "koleo=%.4f)",
                            epoch + 1, avg_loss,
                            tot_dino / n_batches, tot_ibot / n_batches,
                            tot_koleo / n_batches)
            else:
                logger.info("  epoch=%d  loss=%.6f  (dino=%.4f  ibot=%.4f  koleo=%.4f)",
                            epoch + 1, avg_loss, tot_dino / n_batches, tot_ibot / n_batches,
                            tot_koleo / n_batches)
        else:
            logger.info("  epoch=%d  loss=%.6f", epoch + 1, avg_loss)

        # Monitoring: per-epoch scalars (CSV/TB) + diagnostic histories.
        if tracker is not None:
            epoch_metrics = {"loss": avg_loss}
            if method in ("dinov2", "dinov3") and n_batches:
                epoch_metrics.update({
                    "dino": tot_dino / n_batches,
                    "ibot": tot_ibot / n_batches,
                    "koleo": tot_koleo / n_batches,
                })
                if method == "dinov3" and gram_enabled:
                    epoch_metrics["gram"] = tot_gram / n_batches
                if method == "dinov3" and n_diag:
                    epoch_metrics.update({
                        "head_logits_std": tot_head_std / n_diag,
                        "head_entropy": tot_head_entropy / n_diag,
                        "teacher_student_sim": tot_sts / n_diag,
                    })
                    if n_gram_diag:
                        epoch_metrics.update({
                            "gram_masked": tot_gram_masked / n_gram_diag,
                            "gram_unmasked": tot_gram_unmasked / n_gram_diag,
                        })
            tracker.add_epoch(epoch + 1, epoch_metrics)
        if method == "dinov3" and n_diag:
            head_std_history.append(tot_head_std / n_diag)
            head_entropy_history.append(tot_head_entropy / n_diag)
            sts_history.append(tot_sts / n_diag)
            if n_gram_diag:
                gram_masked_history.append(tot_gram_masked / n_gram_diag)
                gram_unmasked_history.append(tot_gram_unmasked / n_gram_diag)

        if save_interval and (epoch + 1) % save_interval == 0:
            _save_bundle(output_dir, epoch + 1, model, optimizer, loss_history,
                         config, meta, method,
                         dino_loss_history=dino_loss_history if method in ("dinov2", "dinov3") else None,
                         ibot_loss_history=ibot_loss_history if method in ("dinov2", "dinov3") else None,
                         koleo_loss_history=koleo_loss_history if method in ("dinov2", "dinov3") else None,
                         gram_loss_history=gram_loss_history if method == "dinov3" else None,
                         head_std_history=head_std_history if method == "dinov3" else None,
                         head_entropy_history=head_entropy_history if method == "dinov3" else None,
                         teacher_student_sim_history=sts_history if method == "dinov3" else None,
                         gram_masked_history=gram_masked_history if method == "dinov3" else None,
                         gram_unmasked_history=gram_unmasked_history if method == "dinov3" else None)
            if umap_check_loader is not None:
                try:
                    _run_umap_check(model, method, umap_check_loader, device, epoch + 1, seed,
                                    os.path.join(output_dir, f"umap_check_epoch_{epoch + 1}.pdf"))
                    last_umap_epoch = epoch + 1
                except Exception as e:
                    logger.error("UMAP check failed at epoch %d: %s", epoch + 1, e)
            _run_diag_plots(model, method, umap_check_loader, device, output_dir,
                            epoch + 1, head_std_history, head_entropy_history,
                            sts_history, gram_masked_history, gram_unmasked_history,
                            monitoring_cfg, tracker)

    logger.info("SSL pretraining done. Final loss: %.6f",
                loss_history[-1] if loss_history else 0.0)

    # ---- Save final SSL bundle ----
    # 1-based epoch of the last completed epoch. If the loop never ran
    # (resume past epochs), save the loaded checkpoint state as-is.
    last_epoch = epochs if start_epoch < epochs else start_epoch
    _save_bundle(output_dir, last_epoch, model, optimizer, loss_history,
                 config, meta, method, final=True,
                 dino_loss_history=dino_loss_history if method in ("dinov2", "dinov3") else None,
                 ibot_loss_history=ibot_loss_history if method in ("dinov2", "dinov3") else None,
                 koleo_loss_history=koleo_loss_history if method in ("dinov2", "dinov3") else None,
                 gram_loss_history=gram_loss_history if method == "dinov3" else None,
                 head_std_history=head_std_history if method == "dinov3" else None,
                 head_entropy_history=head_entropy_history if method == "dinov3" else None,
                 teacher_student_sim_history=sts_history if method == "dinov3" else None,
                 gram_masked_history=gram_masked_history if method == "dinov3" else None,
                 gram_unmasked_history=gram_unmasked_history if method == "dinov3" else None)
    if umap_check_loader is not None and last_umap_epoch != last_epoch:
        try:
            _run_umap_check(model, method, umap_check_loader, device, last_epoch, seed,
                            os.path.join(output_dir, f"umap_check_epoch_{last_epoch}.pdf"))
        except Exception as e:
            logger.error("UMAP check failed at final epoch %d: %s", last_epoch, e)
    if last_umap_epoch != last_epoch:
        _run_diag_plots(model, method, umap_check_loader, device, output_dir,
                        last_epoch, head_std_history, head_entropy_history,
                        sts_history, gram_masked_history, gram_unmasked_history,
                        monitoring_cfg, tracker)
    bundle_path = os.path.join(output_dir, "model.pt")

    if loss_history:
        from .vis import plot_pretrain_loss
        try:
            plot_pretrain_loss(
                loss_history,
                dino_history=dino_loss_history if method in ("dinov2", "dinov3") else None,
                ibot_history=ibot_loss_history if method in ("dinov2", "dinov3") else None,
                koleo_history=koleo_loss_history if method in ("dinov2", "dinov3") else None,
                gram_history=gram_loss_history if method == "dinov3" else None,
                save_path=os.path.join(output_dir, "loss_curve.pdf"))
        except Exception as e:
            logger.error("Loss curve plot failed: %s", e)

    if tracker is not None:
        tracker.close()
        tracker = None
    return {"bundle_path": bundle_path, "method": method, "model": model}


def _run_diag_plots(model, method, loader, device, output_dir, epoch,
                    head_std_history, head_entropy_history, sts_history,
                    gram_masked_history, gram_unmasked_history,
                    monitoring_cfg, tracker):
    """DINOv3 diagnostic PDFs at save_interval / final epochs (best-effort)."""
    if method != "dinov3" or tracker is None:
        return
    try:
        from .vis import plot_head_track
        plot_head_track(
            head_std_history, head_entropy_history, sts_history,
            gram_masked_history, gram_unmasked_history,
            save_path=os.path.join(output_dir, "head_track.pdf"))
    except Exception as e:
        logger.error("Head-track plot failed at epoch %d: %s", epoch, e)
    if monitoring_cfg.get("attention_maps", True) and loader is not None:
        try:
            _run_attention_summary_check(
                model, loader, device,
                os.path.join(output_dir, f"attention_epoch_{epoch}.pdf"),
                monitoring_cfg.get("patch_similarity_samples", 6), tracker, epoch)
        except Exception as e:
            logger.error("Attention summary check failed at epoch %d: %s", epoch, e)


def vis_attention(config, config_path=None):
    """Offline DINOv3 attention + patch-similarity visualization.

    Loads a trained SSL bundle (config.resume.ssl_model), re-computes the
    per-head/mean CLS attention maps and the patch-similarity anchor maps on
    a fixed subset of the same data, and writes a SINGLE combined PDF
    ({output_dir}/vis_attention.pdf) — no training happens.
    """
    if config.get("method") != "dinov3":
        print("Error: vis-attention currently supports method 'dinov3' only",
              file=sys.stderr)
        sys.exit(1)
    resume_path = (config.get("resume") or {}).get("ssl_model")
    if not resume_path or not os.path.exists(resume_path):
        print(f"Error: resume.ssl_model not found ({resume_path!r}) — point it "
              "at the trained bundle (e.g. <run>/model.pt)", file=sys.stderr)
        sys.exit(1)

    device = select_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    seed = 42
    output_dir = config.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)
    add_file_logging(output_dir)
    logger.info("microModel %s vis-attention (config=%s, device=%s)",
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
    logger.info("vis-attention: %d images, bundle=%s", n_samples, resume_path)

    # ---- Load bundle into a dinov3 model (gram keys handled) ----
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta") or {}
    backbone_cfg = dict(config["backbone"])
    backbone_cfg["in_chans"] = meta.get("in_chans", len(resolved_channels))
    if backbone_cfg["in_chans"] != len(resolved_channels):
        print(
            f"Error: bundle in_chans={backbone_cfg['in_chans']} does not match "
            f"len(data.channels)={len(resolved_channels)}; the model "
            f"architecture is fixed by the channel count the bundle was "
            f"trained with",
            file=sys.stderr,
        )
        sys.exit(1)
    model = build_ssl_model("dinov3", backbone_cfg, config.get("dinov3", {}),
                            device)
    _load_checkpoint_state(model, ckpt, method="dinov3")
    model.eval()
    logger.info("Loaded bundle from epoch %d", ckpt.get("epoch", -1))

    # ---- Recompute maps on the fixed subset ----
    from .vis import plot_attention_combined, _to_display_rgb
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
                    model.student_backbone, x1)
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
        logger.warning("vis-attention produced no samples")
        return
    save_path = os.path.join(output_dir, "vis_attention.pdf")
    plot_attention_combined(samples, save_path, inputs=all_inputs,
                            anchor_idx=anchor_idx,
                            title=f"Attention + patch similarity "
                                  f"(offline, {len(samples)} samples)")
    logger.info("vis-attention PDF saved to %s (%d samples)",
                save_path, len(samples))
