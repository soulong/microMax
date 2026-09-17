"""SSL pretraining loop: build dataset, model, optimizer; run epochs; save SSL bundle.

Method-agnostic: dispatches train_step + criterion to the registered SSL method.
Each model file (models/<method>.py) owns its train_step function.
"""

import os
import sys
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from microBase import MicroMaxError

from . import __version__
from .utils import (logger, set_seed, select_device, copy_config_file,
                    add_file_logging, atomic_torch_save, merge_locked_normalize,
                    resolve_channels, resolve_max_value, build_cell_datasets,
                    load_file_list)
from .dataset import SSLMultiViewDataset, FileListCellDataset, subsample
from .models import build_ssl_model, get_train_step, get_criterion
from .backbone import (pool_embedding, num_prefix_tokens,
                       validate_embed_source)
from .monitor import (MetricsTracker, head_collapse_metrics, gram_split_metrics,
                      compute_patch_similarity_maps, compute_cls_attention_maps,
                      maps_to_grid, resize_map, content_dependence_index)

# Images used for the content-dependence index inside each UMAP check.
CDI_IMAGES = 60


def _is_self_resume(resume_path, output_dir):
    """True when resume.ssl_model lives inside this run's output_dir — an
    interrupted continuation of the SAME run (vs a new phase over another
    run's bundle)."""
    if not resume_path:
        return False
    out = os.path.normcase(os.path.abspath(output_dir))
    return os.path.normcase(os.path.abspath(resume_path)).startswith(out)


def _align_history(hist, start_epoch):
    """Front-pad a monitor history with None so that index i == epoch i.

    Histories that only start mid-run (gram stats, CDI — enabled in some
    phases but not others) would otherwise be plotted at the far left of
    head_track.pdf, sharing an 'Epoch' axis they don't belong to.
    """
    return [None] * max(0, start_epoch - len(hist)) + list(hist)


def _append_at(hist, epoch, value):
    """Place value at 1-based index epoch, front-padding None gaps (sparse
    per-save metrics like CDI keep their true epoch positions)."""
    while len(hist) < epoch - 1:
        hist.append(None)
    if len(hist) == epoch - 1:
        hist.append(value)
    else:
        hist[epoch - 1] = value


def _resolve_gram_refresh_state(bundle_meta, output_dir, resume_path, start_step):
    """-> (origin, count) for the gram refresh schedule of this anchoring run.

    Self-resume — the resume bundle lives inside this run's output_dir, i.e.
    an interrupted continuation of the SAME anchoring phase — restores the
    persisted origin + true refresh count, so the schedule and its
    max_updates budget are not reset by the interruption. Any other resume
    (a NEW anchoring phase over a phase-1/other bundle) starts a fresh local
    schedule at the resume step: the earlier history contains no gram
    refreshes and must not pre-exhaust max_updates.
    """
    if not _is_self_resume(resume_path, output_dir):
        return int(start_step), 0
    saved = (bundle_meta or {}).get("gram_refresh")
    if saved:
        return int(saved.get("origin", start_step)), int(saved.get("count", 0))
    return int(start_step), 0


def _save_bundle(output_dir, epoch, model, opt, loss_history, config, meta, method,
                 component_histories=None, monitor_histories=None, final=False,
                 scaler=None):
    """Save a complete SSL bundle: full model state (backbone + heads +
    momentum nets + training-time heads), meta, config, optimizer state,
    GradScaler state, epoch, loss history.

    Every saved .pt is a complete bundle — usable for exact resume, train
    transfer, and feature extraction. final=True writes model.pt (final
    epoch); otherwise model_{epoch}.pt (save_interval epochs). The epoch
    is 1-based, matching the filename (no zero padding).

    component_histories: {component_name: [per-epoch values]} — whatever the
    method's train_step returned (dino/ibot/gram/recon/dist/adv/...).
    monitor_histories: {name: [per-epoch values]} — method-specific
    diagnostic curves (head collapse / gram split) so their PDFs continue
    across resumed runs.
    """
    # meta["ssl_method"] is the single method key (no top-level "method").
    meta = dict(meta)
    meta.setdefault("ssl_method", method)
    # Persist the gram refresh schedule state (local origin + true refresh
    # count) so an interrupted anchoring run resumes its schedule instead of
    # resetting it (see _resolve_gram_refresh_state).
    if (method == "dinov3" and getattr(model, "gram_use_loss", False)
            and getattr(model, "gram_backbone", None) is not None):
        meta["gram_refresh"] = {"origin": int(model._gram_step_origin),
                                "count": int(model._num_gram_updates)}
    bundle = {
        "state_dict": model.state_dict(),
        "meta": meta,
        "config": config,
        "optimizer_state_dict": opt.state_dict(),
        # AMP scale: without it a resumed run restarts at 65536 and skips a
        # few steps until the scale re-converges.
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        # Bundle key naming mirrors the train bundles (meta["ssl_method"]
        # identifies the method; no separate top-level "method").
        "train_loss_history": loss_history,
        "component_histories": component_histories or {},
        "monitor_histories": monitor_histories or {},
    }
    fname = "model.pt" if final else f"model_{epoch}.pt"
    path = os.path.join(output_dir, fname)
    atomic_torch_save(bundle, path)
    logger.info("SSL bundle saved to %s (epoch %d)", path, epoch)


def _run_umap_check(model, method, loader, device, epoch, seed, save_path,
                    embed_source="patch", mask_weighted=False):
    """Extract backbone features of the fixed UMAP-check subset (deterministic
    augmentation_infer pipeline), save a fresh UMAP scatter PDF, and return
    the content-dependence index (CDI) of the first CDI_IMAGES images' patch
    features — the dense-feature homogenization gauge (~1.0 = similarity maps
    input-independent; low = maps track content). Returns None when fewer
    than 2 images are available.

    Features come from the teacher (EMA) branch — official DINO-family
    evaluation uses the teacher backbone — pooled per the run's
    monitoring.embed_source. The umap loader yields masks only when
    mask_weighted is set (single deterministic view per image).

    Auxiliary monitoring aid: failures are caught by the caller (logged and
    skipped) so a plotting hiccup never aborts training.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    model.eval()
    feats = []
    cdi_patches = []
    n_prefix = None
    with torch.no_grad():
        for batch in loader:
            # Without return_mask the batch is [view_batch]; with it, the item
            # is (views, masks) -> batch = [[view_batch], [mask_batch]].
            views = batch[0]
            x = (views[0] if isinstance(views, list) else views).to(device)
            # teacher_backbone.encode returns the FULL token grid (CLS +
            # registers + patches); pool it per the run's embed_source.
            tokens = model.teacher_backbone.encode(x)
            if n_prefix is None:
                n_prefix = num_prefix_tokens(model.teacher_backbone)
            if len(cdi_patches) < CDI_IMAGES:
                cdi_patches.append(
                    tokens[:, n_prefix:].detach().float().cpu())
            patch_mask = None
            if mask_weighted:
                m = batch[-1]
                patch_mask = (m[0] if isinstance(m, list) else m).to(device)
            f = pool_embedding(tokens, n_prefix, embed_source, patch_mask)
            feats.append(f.cpu().numpy())
    X = np.concatenate(feats, axis=0)

    # n_jobs=1 is required by the fixed random_state (reproducible embedding);
    # passing it explicitly silences umap's override warning.
    reducer = umap.UMAP(random_state=seed, n_jobs=1)
    emb = reducer.fit_transform(X)

    # epoch 0 = the pre-training baseline snapshot (before epoch 1 runs).
    stage = "baseline (before training)" if epoch == 0 else f"epoch {epoch}"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], s=12, alpha=0.8, edgecolors="none")
    ax.set_title(f"UMAP check — {stage} ({len(X)} images)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP check saved to %s (%s, %d images)", save_path, stage, len(X))

    # Dense-feature homogenization gauge over the first CDI_IMAGES images
    # (anchor = grid centre patch). Returned to the caller for the
    # monitor_histories curve; None when the subset is too small.
    if not cdi_patches:
        return None
    patches = torch.cat(cdi_patches, dim=0)[:CDI_IMAGES].numpy()
    if len(patches) < 2:
        return None
    grid = int(round((patches.shape[1]) ** 0.5))
    cdi = content_dependence_index(
        patches, anchor=(grid // 2) * grid + grid // 2)
    logger.info("Content-dependence index: %.3f (~1.0 = similarity maps "
                "homogenized; low = content-tracking)", cdi)
    return cdi


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
    from .plots import plot_attention_combined, _to_display_rgb
    model.eval()
    n_samples = max(1, int(n_samples))
    samples = []      # list of {'anchors': [...], 'attn': {...}}
    all_inputs = []
    anchor_idx = None
    grid = None
    with torch.no_grad():
        for batch in loader:
            # The umap loader yields a view LIST per sample (single view here)
            # — collated to [view_batch]; unwrapped either way.
            views = batch[0]
            x = (views[0] if isinstance(views, list) else views).to(device)
            need = n_samples - len(samples)
            for i in range(min(need, len(x))):
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

    For DINOv3 gram runs the bundle and the freshly built model may disagree
    on the gram modules ONLY — such bundles load non-strictly, in either
    direction:
    - bundle has no gram state (phase-1 bundle into a phase-2 model): every
      missing key is a gram module, which keeps the value it was initialized
      with from the resume checkpoint (the gram anchor).
    - bundle has gram state but the current config disabled gram: those
      bundle keys are unexpected (the model never built the modules) and are
      simply dropped.
    Any other missing/unexpected key is real schema drift and must fail
    loudly via the strict load below.
    """
    if "state_dict" not in ckpt:
        raise MicroMaxError("Error: bundle has no 'state_dict' key (unsupported pre-0.2.1 "
              "bundle format)")
    meta = ckpt.get("meta") or {}
    # Bundles written before the meta-key rename carry 'method'; current
    # ones carry 'ssl_method' (pretrain saves meta["ssl_method"]). Either
    # proves this is an SSL pretrain bundle.
    if "ssl_method" not in meta and "method" not in meta:
        raise MicroMaxError("Error: bundle is not an SSL bundle (meta has no 'ssl_method') — "
            "resume.ssl_model expects a pretrain checkpoint, not a train "
            "bundle.")
    state = ckpt["state_dict"]
    if method == "dinov3":
        model_keys = set(model.state_dict())
        bundle_keys = set(state)
        missing = model_keys - bundle_keys
        unexpected = bundle_keys - model_keys
        gram_prefixes = ("gram_backbone.", "gram_teacher.")
        if (missing and all(k.startswith(gram_prefixes) for k in missing)) or \
                (unexpected and all(k.startswith(gram_prefixes) for k in unexpected)):
            logger.info(
                "Bundle/model gram schema diff (%d missing / %d unexpected "
                "keys); loading non-strictly", len(missing), len(unexpected))
            model.load_state_dict(state, strict=False)
            return
    model.load_state_dict(state)


# Method-block keys that determine saved state_dict SHAPES (the DINO/iBOT
# heads' MLP widths and prototype counts, flat or nested under dino:/ibot:).
# On resume the bundle wins for these — building the heads with a changed
# value would crash load_state_dict. Everything else in the method block
# (schedules, loss weights, temps, masking, gram) follows the current config.
_METHOD_ARCH_KEYS = (
    "head_hidden_dim", "head_bottleneck_dim", "head_nlayers", "head_n_prototypes",
    "ibot_head_hidden_dim", "ibot_head_bottleneck_dim", "ibot_head_nlayers",
    "ibot_head_n_prototypes",
    "hidden_dim", "bottleneck_dim", "nlayers", "out_dim",
)


def _merge_method_config(saved, current, arch_keys):
    """Merge the bundle's method block over the current run's block.

    The CURRENT config wins for every key it sets (re-tuned schedules /
    weights take effect; logged so overrides are never silent). Keys the
    bundle saved but the config omits are adopted silently — omission means
    "keep the bundle's value", which makes continue-resume exact by default.
    `arch_keys` are locked to the bundle with a warning when the config
    explicitly differs (the saved heads were built with them). Sub-blocks
    (gram / nested dino:/ibot:) merge by the same rule recursively — which
    subsumes the old gram phase-2 upgrade special case: a config that turns
    gram.use_loss on simply wins over the bundle's false.
    """
    merged = dict(current)
    for k, sv in saved.items():
        if k not in merged:
            merged[k] = sv
            continue
        cv = merged[k]
        if isinstance(sv, dict) and isinstance(cv, dict):
            merged[k] = _merge_method_config(sv, cv, arch_keys)
        elif k in arch_keys and str(cv) != str(sv):
            logger.warning("Locked method key %s differs: bundle=%s, "
                           "config=%s; using bundle value", k, sv, cv)
            merged[k] = sv
        elif str(cv) != str(sv):
            logger.info("Using new value for %s: %s (bundle had %s)", k, cv, sv)
    return merged


def _try_resume(config, device, method):
    """Load SSL checkpoint if resume.ssl_model is set. Returns (checkpoint, config)."""
    resume_path = config.get("resume", {}).get("ssl_model")
    if not resume_path:
        return None, config

    if not os.path.exists(resume_path):
        raise MicroMaxError(f"Error: resume SSL model not found: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    logger.info("Loaded SSL checkpoint from epoch %d", ckpt.get("epoch", -1))

    saved_cfg = ckpt.get("config")
    if saved_cfg is None:
        raise MicroMaxError(f"Error: SSL checkpoint at {resume_path} has no 'config' key")

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

    # Method-specific block (e.g. dinov3 schedules + head dims): merge the
    # bundle's block over the current one. Architecture keys are locked to
    # the bundle so the saved model state (heads included) loads without
    # shape mismatch; every other key follows the CURRENT config, so a
    # stage-2 run can re-tune schedules/loss weights freely (logged, never
    # silent). See _merge_method_config.
    final_method = (ckpt.get("meta") or {}).get("ssl_method", method)
    sm = saved_cfg.get(final_method, {})
    cm = config.get(final_method, {})
    if str(cm) != str(sm):
        config[final_method] = _merge_method_config(sm, cm, _METHOD_ARCH_KEYS)

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
        # UMAP-check embedding readout (teacher branch) + binary background
        # drop in the patch mean. Training itself is unaffected.
        "embed_source": "patch",
        "mask_weighted": True,
    }
    for k, v in (user_cfg or {}).items():
        if v is not None:
            defaults[k] = v
    return defaults


def _build_step_info(method, method_cfg, train_cfg, global_step, total_steps, warmup_steps):
    """Build the step_info dict for train_step (method-specific keys).

    Shared by AMP/non-AMP paths. Method-specific keys (DINOv3 teacher
    temperature, weight decay schedule, koleo weight) are added only for
    dinov3.
    """
    step_info = {
        "global_step": global_step,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        # Must match run_pretrain's optimizer default — train_step drives
        # every group's lr from this value each step, so the optimizer's
        # initial lr is only a placeholder.
        "lr_peak": train_cfg.get("lr", 0.0005),
        "lr_final": method_cfg.get("lr_final", 1e-6),
        "momentum_start": method_cfg.get("momentum_start", 0.992),
        "momentum_end": method_cfg.get("momentum_end", 1.0),
    }
    if method == "dinov3":
        step_info.update({
            # Note: the KoLeo loss weight itself lives on the model
            # (dinov3.koleo_loss_weight config) — schedules here only cover
            # lr/momentum/teacher-temp/weight-decay.
            "teacher_temp_start": method_cfg.get("teacher_temp_start", 0.04),
            "teacher_temp_end": method_cfg.get("teacher_temp_end", 0.07),
            "weight_decay_start": method_cfg.get("weight_decay_start", 0.04),
            "weight_decay_end": method_cfg.get("weight_decay_end", 0.4),
        })
    return step_info


def _prepare_pretrain_data(config):
    """Resolve data roots -> (cell, idx) pairs plus the channels/normalize/
    input settings shared by run_pretrain and the offline run_attention_vis
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

    root = data_cfg.get("file_dir")
    # file_dir may be null (all entries commented out) — legal in file_list
    # mode, where the CSVs are the pool and file_dir is ignored.
    roots = [root] if isinstance(root, str) else list(root or [])
    channels = data_cfg.get("channels")
    channel_layout = data_cfg.get("channel_layout", "CHW")
    image_pattern = data_cfg.get("image_pattern")

    file_list = data_cfg.get("file_list")
    if not file_list and not roots:
        raise MicroMaxError("Error: data.file_list and data.file_dir are both "
              "empty; set one of them (a data source is required)")
    resolved_per_root = {}
    if file_list:
        # File-list mode: the CSVs ARE the pool — data.file_dir is ignored.
        # Labels (if a column exists) are meaningless for SSL and skipped.
        if root:
            logger.warning("data.file_list takes precedence: data.file_dir "
                           "is ignored")
        paths, _labels = load_file_list(file_list)
        cell_ds = FileListCellDataset(paths, channel_layout)
        n_avail = len(cell_ds.intensity_colnames)
        resolved_channels = resolve_channels(channels, n_avail, "file_list")
        all_pairs = [(cell_ds, i) for i in range(len(paths))]
        root_of_pair = ["file_list"] * len(all_pairs)
        logger.info("File list: %d cells from %s", len(paths), file_list)
    else:
        datasets = build_cell_datasets(roots, channel_layout, image_pattern)
        all_pairs = []
        root_of_pair = []
        for r, cell_ds in datasets:
            n_avail = len(cell_ds.intensity_colnames)
            resolved_per_root[r] = resolve_channels(channels, n_avail, r)
            recs = _build_records(cell_ds)
            all_pairs.extend(recs)
            root_of_pair.extend([r] * len(recs))

        if not all_pairs:
            raise MicroMaxError(f"Error: no records found in {roots}")

        # All roots must resolve to the same channel set — one bundle carries a
        # single `channels` meta, so a heterogeneous resolution would silently
        # train on the wrong channels for some roots (last-root-wins bug).
        unique_resolved = {tuple(v) for v in resolved_per_root.values()}
        if len(unique_resolved) > 1:
            raise MicroMaxError(f"Error: data roots resolve to different channel sets: "
                f"{ {r: v for r, v in resolved_per_root.items()} }. "
                f"Give every root the same channel count or set data.channels "
                f"explicitly.")
        resolved_channels = list(next(iter(unique_resolved))) if unique_resolved else channels

    # Subsample
    sample_max = data_cfg.get("sample_max")
    sample_by = data_cfg.get("sample_by", "per_dataset")
    if sample_max is not None:
        # Wrap pairs as dicts for subsample; the real root is carried along so
        # sample_by='per_dataset' caps per data root (not across all roots).
        items = [{"pair": p, "root": r} for p, r in zip(all_pairs, root_of_pair)]
        items = subsample(items, sample_max, sample_by, seed,
                          label_key=None, root_key="root")
        all_pairs = [it["pair"] for it in items]
        logger.info("Sub-sampled to %d records (sample_max=%s, sample_by=%s)",
                    len(all_pairs), sample_max, sample_by)

    augmentation_infer = config.get("augmentation_infer")
    if augmentation_infer is None:
        raise MicroMaxError("Error: augmentation_infer is required in pretrain config (must be a "
            "deterministic pipeline for inference/UMAP-check). Set it explicitly.")
    return (all_pairs, resolved_channels, channel_layout, image_pattern,
            max_value, normalize_method, with_masking, clip_low, clip_high,
            fixed_reference, augmentation_infer)




def run_pretrain(config, config_path=None):
    """Generic SSL pretraining loop. Dispatches on config['method']."""
    method = config.get("method")
    if method != "dinov3":
        raise MicroMaxError(f"Error: unknown SSL method '{method}'. Available: dinov3")

    mode = config.get("mode", "single_cell")
    if mode not in ("single_cell",):
        raise MicroMaxError(f"Error: pretrain mode '{mode}' is not supported. Available: "
              f"single_cell (whole_image is planned but not yet implemented)")

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
    if method != "dinov3":
        raise MicroMaxError(f"Error: unknown SSL method '{method}'. Available: dinov3")

    # resume.type: 'continue' = exact same-data extension (schedules/optimizer/
    # epoch counter continue where the interrupted run left off); 'transfer' =
    # domain transfer on new data — fresh run (epoch 0, schedules restart,
    # optimizer state reset) initialized from the bundle weights.
    resume_type = config.get("resume", {}).get("type", "continue")
    if resume_type not in ("continue", "transfer"):
        raise MicroMaxError(f"Error: unknown resume.type '{resume_type}'. "
              f"Available: continue, transfer")

    train_cfg = config["training"]
    # backbone block: required for dinov3 (name/pretrained).
    backbone_cfg = config.get("backbone") or {}
    if method == "dinov3" and not backbone_cfg.get("name"):
        raise MicroMaxError("Error: dinov3 requires a backbone: block with name/pretrained "
              "(e.g. name: vit_small_patch16_dinov3)")
    method_cfg = config.get(method, {})
    aug_views_cfg = config.get("augmentation_views", [])
    monitoring_cfg = _resolve_monitoring_cfg(config.get("monitoring"))
    if monitoring_cfg["enabled"]:
        logger.info("Monitoring enabled: tensorboard=%s csv=%s "
                    "log_every_steps=%d patch_similarity_samples=%d",
                    monitoring_cfg["tensorboard"], monitoring_cfg["csv"],
                    monitoring_cfg["log_every_steps"],
                    monitoring_cfg["patch_similarity_samples"])
    # UMAP-check embedding readout (teacher branch): monitoring.embed_source
    # + monitoring.mask_weighted. Training itself is unaffected — the SSL
    # losses use CLS and patch tokens directly.
    embed_source = validate_embed_source(monitoring_cfg["embed_source"])
    umap_mask_weighted = bool(monitoring_cfg["mask_weighted"]) \
        and embed_source != "cls"
    if monitoring_cfg["enabled"]:
        logger.info("UMAP-check embedding: source=%s mask_weighted=%s",
                    embed_source, umap_mask_weighted)

    # Validate views count
    if method == "dinov3":
        gram_cfg = method_cfg.get("gram", {}) or {}
        n_gram_views = 2 if (gram_cfg.get("use_loss", False)
                             and not gram_cfg.get("ema_teacher", False)) else 0
        min_views = 3 + n_gram_views
        if len(aug_views_cfg) < min_views:
            raise MicroMaxError(f"Error: DINOv3 requires at least {min_views} views "
                  f"(2 global + 1 local{' + 2 gram-teacher crops' if n_gram_views else ''}), "
                  f"got {len(aug_views_cfg)}")

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

    # ---- Resolve data ----
    # The config data block (shared with the offline run_attention_vis command).
    (all_pairs, resolved_channels, channel_layout, image_pattern,
     max_value, normalize_method, with_masking, clip_low, clip_high,
     fixed_reference, augmentation_infer) = _prepare_pretrain_data(config)
    cfg_root = config["data"].get("file_dir")
    data_roots = [cfg_root] if isinstance(cfg_root, str) else list(cfg_root or [])

    logger.info("Found %d records, channels=%s", len(all_pairs), resolved_channels)
    logger.info("SSL method: %s, views: %d", method, len(aug_views_cfg))

    set_seed(seed)

    # ---- Build dataset + dataloader ----
    # One SSLMultiViewDataset per cell_dataset, then ConcatDataset.
    by_cell_ds = {}
    for cell_ds, idx in all_pairs:
        by_cell_ds.setdefault(id(cell_ds), (cell_ds, []))[1].append(idx)

    datasets = []
    for cell_ds_id, (cell_ds, indices) in by_cell_ds.items():
        ds = SSLMultiViewDataset(
            cell_ds, indices, resolved_channels,
            augmentation_specs=aug_views_cfg,
            normalize_method=normalize_method, clip_low=clip_low,
            clip_high=clip_high, with_masking=with_masking,
            fixed_reference=fixed_reference, max_value=max_value)
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
    # with it), so a changed data.channels raises a clear error instead of crashing.
    if checkpoint is not None and "meta" in checkpoint:
        backbone_cfg["in_chans"] = checkpoint["meta"].get(
            "in_chans", len(resolved_channels))
    else:
        backbone_cfg["in_chans"] = len(resolved_channels)
    if backbone_cfg["in_chans"] != len(resolved_channels):
        raise MicroMaxError(f"Error: bundle in_chans={backbone_cfg['in_chans']} does not match "
            f"len(data.channels)={len(resolved_channels)}; the model architecture "
            f"is fixed by the channel count the bundle was trained with")
    # The in-model diagnostics flag rides on the method block (build_dinov3
    # reads method_cfg["monitoring"]["enabled"]) — wire the resolved top-level
    # monitoring block through on a copy, so `monitoring.enabled: false` also
    # stops the model from building per-step diag tensors. method_cfg itself
    # stays clean (it is what gets saved into the bundle config).
    # The dinov3 gram anchor IS the phase-1 model the run resumes from — always
    # resume.ssl_model, there is no separate gram.ckpt to configure (or forget).
    # gram.it_load_ema_teacher (optional) can still overwrite the anchor with
    # the EMA teacher at a chosen global step.
    gram_anchor = None
    if method == "dinov3":
        gram_cfg = method_cfg.get("gram") or {}
        if gram_cfg.get("use_loss", False) and not gram_cfg.get("ema_teacher", False):
            gram_anchor = (config.get("resume") or {}).get("ssl_model")
            if not gram_anchor:
                raise MicroMaxError("Error: dinov3.gram.use_loss is on but resume.ssl_model is "
                      "not set; the gram anchor is the phase-1 model the run resumes from")
    model = build_ssl_model(
        method, backbone_cfg, {**method_cfg, "monitoring": monitoring_cfg},
        device)

    # DINOv3 gram anchoring: initialize the frozen gram teacher from the
    # resume checkpoint (the phase-1 model).
    if gram_anchor:
        model.load_gram_from_bundle(gram_anchor)

    # Bundle meta — needed by every saved bundle (infer/train consumers).
    # No feat_dim: consumers rebuild the backbone and read num_features live.
    meta = {
        "ssl_method": method,
        "backbone": backbone_cfg["name"],
        "in_chans": backbone_cfg["in_chans"],
        "channels": resolved_channels,
        "channel_layout": channel_layout,
        "max_value": max_value,
        "augmentation_infer": augmentation_infer,
        "normalize_method": normalize_method,
        "normalize_with_masking": with_masking,
        "normalize_fixed_reference": fixed_reference,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "image_pattern": image_pattern,
        "data_root": data_roots,
    }

    # Parameter groups. Standard ViT practice: decay (ndim >= 2: weight
    # matrices, embeddings) vs no-decay (ndim < 2: biases, LayerNorm).
    # training.encoder_lr (optional) scales the schedule LR for trainable
    # encoder params ("encoder." prefix): head/decoder params train at
    # training.lr while the encoder gets a lower LR, without touching the
    # schedule code (train_step multiplies group["lr_scale"]).
    # Must match _build_step_info's lr_peak default — train_step overwrites
    # every group's lr from the schedule each step.
    lr = float(train_cfg.get("lr", 0.0005))
    encoder_lr = train_cfg.get("encoder_lr")
    enc_scale = (float(encoder_lr) / lr) if encoder_lr else 1.0
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if enc_scale != 1.0:
        enc_named = [(n, p) for n, p in trainable if n.split(".")[0] == "encoder"]
        rest_named = [(n, p) for n, p in trainable if n.split(".")[0] != "encoder"]
        logger.info("encoder_lr=%s (scale %.3f): %d encoder params, %d "
                    "head/decoder params", encoder_lr, enc_scale,
                    len(enc_named), len(rest_named))
    else:
        enc_named, rest_named = [], trainable

    def _param_groups(named_params, lr_scale, default_wd):
        groups = []
        decay = [p for _, p in named_params if p.ndim >= 2]
        no_decay = [p for _, p in named_params if p.ndim < 2]
        if decay:
            groups.append({"params": decay, "weight_decay": default_wd,
                           "lr_scale": lr_scale})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0,
                           "lr_scale": lr_scale})
        return groups

    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            _param_groups(rest_named, 1.0, train_cfg.get("weight_decay", 0.0))
            + _param_groups(enc_named, enc_scale, train_cfg.get("weight_decay", 0.0)),
            lr=lr,
            momentum=0.9,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            _param_groups(rest_named, 1.0, train_cfg.get("weight_decay", 0.04))
            + _param_groups(enc_named, enc_scale, train_cfg.get("weight_decay", 0.04)),
            lr=lr,
            betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        )
    else:
        raise MicroMaxError(f"Error: unknown optimizer '{optimizer_name}'")

    criterion = get_criterion(method, method_cfg, device)
    train_step_fn = get_train_step(method)

    amp_enabled = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    grad_clip = train_cfg.get("grad_clip", 10.0)

    epochs = train_cfg.get("epochs", 10)
    save_interval = train_cfg.get("save_interval")
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    if not isinstance(grad_accum_steps, int) or grad_accum_steps < 1:
        raise MicroMaxError(f"Error: training.grad_accum_steps must be an integer >= 1, "
              f"got {grad_accum_steps}")

    start_epoch = 0
    loss_history = []
    # Per-component loss histories: {name: [per-epoch value]} — the keys are
    # whatever the method's train_step returns (dino/ibot/gram/recon/dist/
    # adv/adv_acc). Plus method-specific diagnostic curves for the PDFs.
    component_histories = {}
    monitor_histories = {"head_std": [], "head_entropy": [],
                         "teacher_student_sim": [], "gram_masked": [],
                         "gram_unmasked": []}
    if checkpoint is not None:
        _load_checkpoint_state(model, checkpoint, method=method)
        if resume_type == "continue":
            # Bundle 'epoch' is 1-based (last completed epoch), which equals
            # the next 0-based loop index to continue from. Exact extension:
            # optimizer state + loss history are restored so the run
            # continues where it left off.
            start_epoch = checkpoint.get("epoch", 0)
            logger.info("Resuming: %d epochs completed, continuing at epoch %d",
                        start_epoch, start_epoch + 1)
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
            if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            loss_history = checkpoint.get("train_loss_history", [])
            component_histories = dict(checkpoint.get("component_histories", {}))
            for k, v in (checkpoint.get("monitor_histories") or {}).items():
                monitor_histories[k] = _align_history(v, start_epoch)
            if not _is_self_resume((config.get("resume") or {}).get("ssl_model"),
                                   output_dir):
                # New anchoring phase over another run's bundle: its CDI
                # points belong to the previous phase's own head_track.pdf.
                monitor_histories["content_dependence"] = []
        else:
            # Transfer: domain-transfer / pretrained-weight init. The run
            # restarts from epoch 1 with fresh schedules and a fresh optimizer
            # (AdamW/SGD moments are stale for the new data); only the model
            # weights are carried over.
            logger.info("Transfer: fresh run starting at epoch 1, initialized from bundle "
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
        # Gram refresh schedule on the LOCAL time axis of this anchoring run:
        # a self-resume (bundle inside this output_dir) restores origin +
        # true refresh count; any other resume starts a fresh schedule here
        # (the earlier history has no gram refreshes to account for).
        origin, count = _resolve_gram_refresh_state(
            checkpoint.get("meta"), output_dir,
            (config.get("resume") or {}).get("ssl_model"), global_step)
        model.set_gram_origin(origin, count)
        if count:
            logger.info("Gram refresh schedule resumed: origin=%d, %d update(s) done",
                        origin, count)
        else:
            # Models without the gram head never get the attribute (treated
            # as disabled); None means the refresh budget is uncapped, i.e.
            # the schedule is active and starts fresh on this run's axis.
            gram_max = getattr(model, "_gram_max_updates", 0)
            if gram_max is None or gram_max > 0:
                logger.info("Gram refresh schedule starts fresh at local step 0 "
                            "(first update after %d batches)",
                            model._gram_update_frequency)
            else:
                logger.info("Gram refresh disabled (max_updates=0): anchor frozen "
                            "at the resume model for the whole run")
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
            raise MicroMaxError(f"Error: training.n_image_umap must be an integer, got {n_image_umap!r}")
        if n_image_umap < 0:
            raise MicroMaxError("Error: training.n_image_umap must be >= 0 (0 = disabled)")
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
                    fixed_reference=fixed_reference, max_value=max_value,
                    return_mask=umap_mask_weighted)
                for cell_ds, indices in by_cell.values()
            ]
            umap_check_loader = DataLoader(
                check_datasets[0] if len(check_datasets) == 1 else ConcatDataset(check_datasets),
                batch_size=32, shuffle=False, num_workers=0)
            logger.info("UMAP check enabled: %d fixed images, fresh UMAP at "
                        "pre-training baseline + each save_interval + final epoch",
                        n_pick)

    # ---- Pre-training baseline UMAP check (before epoch 1) ----
    # Features BEFORE any training (fresh pretrained init, or transfer
    # weights). Skipped when actually resuming with resume.type=continue —
    # the run is an exact extension of a previous one, so the baseline
    # belongs to that run.
    if umap_check_loader is not None and (checkpoint is None or resume_type != "continue"):
        try:
            _run_umap_check(model, method, umap_check_loader, device, 0, seed,
                            os.path.join(output_dir, "umap_check_baseline.pdf"),
                            embed_source=embed_source,
                            mask_weighted=umap_mask_weighted)
        except Exception as e:
            logger.error("UMAP check failed on baseline: %s", e)

    # ---- Training loop ----
    # Uniform train_step protocol: every registered method returns
    # (loss_value, {component_name: float}) — run_pretrain aggregates,
    # logs and persists components generically, so adding a loss/head never
    # touches this loop.
    diag_enabled = (monitoring_cfg["enabled"] and method == "dinov3")
    for epoch in range(start_epoch, epochs):
        model.train()
        tot_loss = 0.0
        tot_comp = {}
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
            # batch is a list of N view tensors (each (B, C, H, W)) for the
            # SSL methods. DataLoader auto-collates the datasets' return
            # values.
            do_step = ((batch_in_epoch + 1) % grad_accum_steps == 0) or \
                (batch_in_epoch + 1 == len(loader))
            step_info = _build_step_info(
                method, method_cfg, train_cfg, global_step, total_steps, warmup_steps)
            if amp_enabled:
                # AMP: forward under autocast; train_step does its own scaled
                # backward + optimizer step via the GradScaler.
                with torch.amp.autocast("cuda"):
                    loss_val, components = train_step_fn(
                        model, batch, optimizer, epoch, epochs,
                        device, criterion, step_info,
                        scaler, grad_clip, do_step)
            else:
                loss_val, components = train_step_fn(
                    model, batch, optimizer, epoch, epochs,
                    device, criterion, step_info,
                    scaler, grad_clip, do_step)
            if not math.isfinite(loss_val):
                raise MicroMaxError(f"Error: non-finite loss ({loss_val}) at epoch {epoch + 1}; "
                      f"aborting to avoid writing a corrupted SSL bundle")
            tot_loss += loss_val
            for k, v in components.items():
                tot_comp[k] = tot_comp.get(k, 0.0) + v

            # Monitoring: DINOv3 diagnostics (head collapse / gram split) +
            # step-level metrics (CSV/TB). Other methods have no _last_diag.
            if diag_enabled and tracker is not None and getattr(model, "_last_diag", None):
                diag = model._last_diag
                step_metrics = {"loss": loss_val, **components}
                if (monitoring_cfg.get("head_logits_track", True)
                        and "head_logits" in diag):
                    hm = head_collapse_metrics(diag, model._student_temp)
                    step_metrics.update(hm)
                    tot_head_std += hm["head_logits_std"]
                    tot_head_entropy += hm["head_entropy"]
                    tot_sts += hm["teacher_student_sim"]
                    n_diag += 1
                gram_diag = diag.get("gram")
                if (gram_diag is not None
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
        avg_comp = {k: v / n_batches for k, v in tot_comp.items()} if n_batches else {}
        loss_history.append(avg_loss)
        for k, v in avg_comp.items():
            component_histories.setdefault(k, []).append(v)
        comp_txt = "  ".join(f"{k}={v:.4f}" for k, v in avg_comp.items())
        logger.info("  epoch=%d  loss=%.6f%s", epoch + 1, avg_loss,
                    f"  ({comp_txt})" if comp_txt else "")

        # Monitoring: per-epoch scalars (CSV/TB) + DINOv3 diagnostic histories.
        if tracker is not None:
            epoch_metrics = {"loss": avg_loss, **avg_comp}
            if n_diag:
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
            monitor_histories["head_std"].append(tot_head_std / n_diag)
            monitor_histories["head_entropy"].append(tot_head_entropy / n_diag)
            monitor_histories["teacher_student_sim"].append(tot_sts / n_diag)
            if n_gram_diag:
                monitor_histories["gram_masked"].append(tot_gram_masked / n_gram_diag)
                monitor_histories["gram_unmasked"].append(tot_gram_unmasked / n_gram_diag)

        if save_interval and (epoch + 1) % save_interval == 0:
            _save_bundle(output_dir, epoch + 1, model, optimizer, loss_history,
                         config, meta, method,
                         component_histories=component_histories,
                         monitor_histories=monitor_histories, scaler=scaler)
            if umap_check_loader is not None:
                try:
                    cdi = _run_umap_check(
                        model, method, umap_check_loader, device, epoch + 1, seed,
                        os.path.join(output_dir, f"umap_check_epoch_{epoch + 1}.pdf"),
                        embed_source=embed_source,
                        mask_weighted=umap_mask_weighted)
                    if cdi is not None:
                        # Place at the ABSOLUTE epoch (front-padding gaps) so
                        # sparse CDI points keep their true positions on the
                        # shared head_track epoch axis.
                        _append_at(monitor_histories.setdefault(
                            "content_dependence", []), epoch + 1, cdi)
                    last_umap_epoch = epoch + 1
                except Exception as e:
                    logger.error("UMAP check failed at epoch %d: %s", epoch + 1, e)
            _run_diag_plots(model, method, umap_check_loader, device, output_dir,
                            epoch + 1, monitor_histories, monitoring_cfg, tracker)

    logger.info("SSL pretraining done. Final loss: %.6f",
                loss_history[-1] if loss_history else 0.0)

    # ---- Save final SSL bundle ----
    # 1-based epoch of the last completed epoch. If the loop never ran
    # (resume past epochs), save the loaded checkpoint state as-is.
    last_epoch = epochs if start_epoch < epochs else start_epoch
    _save_bundle(output_dir, last_epoch, model, optimizer, loss_history,
                 config, meta, method, final=True,
                 component_histories=component_histories,
                 monitor_histories=monitor_histories, scaler=scaler)
    if umap_check_loader is not None and last_umap_epoch != last_epoch:
        try:
            cdi = _run_umap_check(
                model, method, umap_check_loader, device, last_epoch, seed,
                os.path.join(output_dir, f"umap_check_epoch_{last_epoch}.pdf"),
                embed_source=embed_source,
                mask_weighted=umap_mask_weighted)
            if cdi is not None:
                _append_at(monitor_histories.setdefault(
                    "content_dependence", []), last_epoch, cdi)
        except Exception as e:
            logger.error("UMAP check failed at final epoch %d: %s", last_epoch, e)
    if last_umap_epoch != last_epoch:
        _run_diag_plots(model, method, umap_check_loader, device, output_dir,
                        last_epoch, monitor_histories, monitoring_cfg, tracker)
    bundle_path = os.path.join(output_dir, "model.pt")

    if loss_history:
        from .plots import plot_pretrain_loss
        try:
            plot_pretrain_loss(
                loss_history,
                component_histories=component_histories,
                save_path=os.path.join(output_dir, "loss_curve.pdf"))
        except Exception as e:
            logger.error("Loss curve plot failed: %s", e)

    if tracker is not None:
        tracker.close()
        tracker = None
    return {"bundle_path": bundle_path, "method": method, "model": model}


def _run_diag_plots(model, method, loader, device, output_dir, epoch,
                    monitor_histories, monitoring_cfg, tracker):
    """DINOv3 diagnostic PDFs at save_interval / final epochs (best-effort)."""
    if method != "dinov3" or tracker is None:
        return
    try:
        from .plots import plot_head_track
        plot_head_track(
            monitor_histories.get("head_std", []),
            monitor_histories.get("head_entropy", []),
            monitor_histories.get("teacher_student_sim", []),
            monitor_histories.get("gram_masked", []),
            monitor_histories.get("gram_unmasked", []),
            cdi_history=monitor_histories.get("content_dependence"),
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
