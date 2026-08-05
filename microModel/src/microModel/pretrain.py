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
                    resolve_channels, build_cell_datasets)
from .dataset import SSLMultiViewDataset, subsample
from .models import build_ssl_model, get_train_step, get_criterion


def _save_bundle(output_dir, epoch, model, opt, loss_history, config, meta, method,
                 final=False):
    """Save a complete SSL bundle: full model state (backbone + heads +
    momentum nets), meta, config, optimizer state, epoch, loss history.

    Every saved .pt is a complete bundle — usable for exact resume, train
    transfer, and feature extraction. final=True writes ssl_model.pt (final
    epoch); otherwise ssl_model_{epoch}.pt (save_interval epochs). The epoch
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
    fname = "ssl_model.pt" if final else f"ssl_model_{epoch}.pt"
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
            if method == "dinov2":
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


def _load_checkpoint_state(model, ckpt):
    """Restore model state from a bundle.

    New bundles carry the full model state_dict (exact resume, teacher/
    momentum included). Old bundles carry only backbone_state_dict — heads
    are re-initialized with a warning.
    """
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        return
    if "backbone_state_dict" in ckpt:
        _load_backbone_state_legacy(model, ckpt["backbone_state_dict"])
        logger.warning("Old bundle: only backbone state restored; "
                       "projection/prediction heads re-initialized")
        return
    print("Error: bundle has neither 'state_dict' nor 'backbone_state_dict'",
          file=sys.stderr)
    sys.exit(1)


def _load_backbone_state_legacy(model, state_dict):
    """Load a backbone state dict into an SSL model (student/online network).

    BYOL: model.backbone (online network). DINOv2: model.student_backbone.
    Momentum/teacher networks are re-initialized as a deepcopy of the student.
    """
    import copy
    if hasattr(model, "student_backbone"):
        model.student_backbone.load_state_dict(state_dict)
        model.teacher_backbone.load_state_dict(copy.deepcopy(state_dict))
    elif hasattr(model, "backbone"):
        model.backbone.load_state_dict(state_dict)
        if hasattr(model, "backbone_momentum"):
            model.backbone_momentum.load_state_dict(copy.deepcopy(state_dict))
    else:
        print("Error: cannot load backbone state into SSL model", file=sys.stderr)
        sys.exit(1)


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

    # Locked keys (bundle wins on mismatch): method, backbone.name, augmentation_views, normalize.*
    locked_keys = [
        ("method",),
        ("backbone", "name"),
        ("augmentation_views",),
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

    merge_locked_normalize(saved_cfg, config)

    # Method-specific block (e.g. byol/dinov2 head dims) — bundle wins so the
    # saved model state (heads included) loads without shape mismatch.
    sm = saved_cfg.get(method, {})
    cm = config.get(method, {})
    if str(cm) != str(sm):
        logger.warning("Locked %s block differs: bundle=%s, config=%s; using bundle value",
                       method, sm, cm)
        config[method] = sm

    return ckpt, config


def _build_records(cell_ds):
    """Walk a CellDataset's metadata DataFrame and return list of (cell_ds, idx) pairs."""
    return [(cell_ds, i) for i in range(len(cell_ds.metadata))]


def _build_step_info(method, method_cfg, train_cfg, global_step, total_steps, warmup_steps):
    """Build the step_info dict for train_step (method-specific keys).

    Shared by AMP/non-AMP paths. Method-specific keys (DINOv2 teacher
    temperature, weight decay schedule, koleo weight) are added only for dinov2.
    """
    step_info = {
        "global_step": global_step,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "lr_peak": train_cfg.get("lr", 0.0005 if method == "dinov2" else 0.05),
        "lr_final": method_cfg.get("lr_final", 1e-6),
        "momentum_start": method_cfg.get(
            "momentum_start", 0.992 if method == "dinov2" else 0.996),
        "momentum_end": method_cfg.get("momentum_end", 1.0),
    }
    if method == "dinov2":
        step_info.update({
            "koleo_weight": method_cfg.get("koleo_weight", 0.1),
            "teacher_temp_start": method_cfg.get("teacher_temp_start", 0.04),
            "teacher_temp_end": method_cfg.get("teacher_temp_end", 0.07),
            "weight_decay_start": method_cfg.get("weight_decay_start", 0.04),
            "weight_decay_end": method_cfg.get("weight_decay_end", 0.4),
        })
    return step_info


def pretrain_ssl(config, config_path=None):
    """Generic SSL pretraining loop. Dispatches on config['method']."""
    method = config.get("method")
    if method not in ("byol", "dinov2"):
        print(f"Error: unknown SSL method '{method}'. Available: byol, dinov2", file=sys.stderr)
        sys.exit(1)

    device = select_device()
    if device.type == "cuda":
        # fp16 matmul reduced-precision reduction produced rare NaN on some
        # GPUs (cuBLAS accumulation overflow) even with small, finite
        # activations. Force fp32 accumulation for fp16 matmuls.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    checkpoint, config = _try_resume(config, device, method)

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

    # Validate views count
    if method == "byol" and len(aug_views_cfg) < 2:
        print(f"Error: BYOL requires at least 2 views, got {len(aug_views_cfg)}", file=sys.stderr)
        sys.exit(1)
    if method == "dinov2" and len(aug_views_cfg) < 3:
        print(f"Error: DINOv2 requires at least 3 views (2 global + 1 local), "
              f"got {len(aug_views_cfg)}", file=sys.stderr)
        sys.exit(1)

    normalize_method = norm_cfg.get("method", "per_channel")
    with_masking = norm_cfg.get("with_masking", False)
    clip_low = norm_cfg.get("clip_low", 0.05)
    clip_high = norm_cfg.get("clip_high", 99.95)

    seed = 42
    output_dir = config.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)
    add_file_logging(output_dir)
    if config_path is not None:
        dst = copy_config_file(config_path, output_dir)
        logger.info("Pretrain config copied to %s", dst)
    logger.info("microModel %s pretrain starting (method=%s, config=%s, device=%s)",
                __version__, method, config_path, device)

    set_seed(seed)

    # ---- Build CellDataset + records from each root ----
    root = data_cfg["root"]
    roots = [root] if isinstance(root, str) else list(root)
    channels = data_cfg.get("channels")
    channel_layout = data_cfg.get("channel_layout", "CHW")
    image_pattern = data_cfg.get("image_pattern")

    datasets = build_cell_datasets(roots, channel_layout, image_pattern)
    all_pairs = []
    resolved_channels = channels
    for r, cell_ds in datasets:
        n_avail = len(cell_ds.intensity_colnames)
        resolved_channels = resolve_channels(channels, n_avail, r)
        all_pairs.extend(_build_records(cell_ds))

    if not all_pairs:
        print(f"Error: no records found in {roots}", file=sys.stderr)
        sys.exit(1)

    logger.info("Found %d records, channels=%s", len(all_pairs), resolved_channels)
    logger.info("SSL method: %s, views: %d", method, len(aug_views_cfg))

    # Subsample
    sample_max = data_cfg.get("sample_max")
    sample_by = data_cfg.get("sample_by", "per_dataset")
    if sample_max is not None:
        # Wrap pairs as dicts for subsample (uses root_key, label_key defaults)
        items = [{"pair": p, "root": "__root__"} for p in all_pairs]
        items = subsample(items, sample_max, sample_by, seed,
                          label_key="__unlabeled__", root_key="root")
        all_pairs = [it["pair"] for it in items]
        logger.info("Sub-sampled to %d records (sample_max=%s, sample_by=%s)",
                    len(all_pairs), sample_max, sample_by)

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
            with_masking=with_masking)
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

    # Bundle meta — needed by every saved bundle (infer/train consumers).
    if method == "dinov2":
        feat_dim = model.student_backbone.vit.num_features
    else:
        feat_dim = getattr(model.backbone, "num_features", None)
    augmentation_infer = config.get("augmentation_infer")
    if augmentation_infer is None:
        print(
            "Error: augmentation_infer is required in pretrain config (must be a "
            "deterministic pipeline for inference/UMAP-check). Set it explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    meta = {
        "method": method,
        "backbone": backbone_cfg["name"],
        "in_chans": backbone_cfg["in_chans"],
        "channels": resolved_channels,
        "channel_layout": channel_layout,
        "feat_dim": feat_dim,
        "augmentation_infer": augmentation_infer,
        "normalize_method": normalize_method,
        "normalize_with_masking": with_masking,
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
    if checkpoint is not None:
        _load_checkpoint_state(model, checkpoint)
        if resume_type == "continue":
            # Bundle 'epoch' is 1-based (last completed epoch), which equals
            # the next 0-based loop index to continue from. Exact extension:
            # optimizer state + loss history are restored so the run
            # continues where it left off.
            start_epoch = checkpoint.get("epoch", 0)
            logger.info("Resuming from epoch %d", start_epoch)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            loss_history = checkpoint.get("loss_history", [])
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
    umap_check_loader = None
    last_umap_epoch = None
    if n_image_umap is not None and int(n_image_umap) < 0:
        print("Error: training.n_image_umap must be >= 0 (0 = disabled)",
              file=sys.stderr)
        sys.exit(1)
    if n_image_umap:
        n_pick = min(int(n_image_umap), len(all_pairs))
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
                    clip_high=clip_high, with_masking=with_masking)
                for cell_ds, indices in by_cell.values()
            ]
            umap_check_loader = DataLoader(
                check_datasets[0] if len(check_datasets) == 1 else ConcatDataset(check_datasets),
                batch_size=32, shuffle=False, num_workers=0)
            logger.info("UMAP check enabled: %d fixed images, fresh UMAP at each "
                        "save_interval + final epoch", n_pick)

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs):
        model.train()
        tot_loss = 0.0
        tot_dino = 0.0
        tot_ibot = 0.0
        tot_koleo = 0.0
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
                    loss, dino_l, ibot_l, koleo_l = train_step_fn(
                        model, batch, optimizer, epoch, epochs,
                        device, criterion, step_info,
                        scaler, grad_clip, do_step)
            else:
                loss, dino_l, ibot_l, koleo_l = train_step_fn(
                    model, batch, optimizer, epoch, epochs,
                    device, criterion, step_info,
                    scaler, grad_clip, do_step)
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
            n_batches += 1
            batch_in_epoch += 1
            global_step += 1

        avg_loss = tot_loss / n_batches if n_batches else 0.0
        loss_history.append(avg_loss)
        if method == "dinov2" and n_batches:
            dino_loss_history.append(tot_dino / n_batches)
            ibot_loss_history.append(tot_ibot / n_batches)
            koleo_loss_history.append(tot_koleo / n_batches)
            logger.info("  epoch=%d  loss=%.6f  (dino=%.4f  ibot=%.4f  koleo=%.4f)",
                        epoch + 1, avg_loss, tot_dino / n_batches, tot_ibot / n_batches,
                        tot_koleo / n_batches)
        else:
            logger.info("  epoch=%d  loss=%.6f", epoch + 1, avg_loss)

        if save_interval and (epoch + 1) % save_interval == 0:
            _save_bundle(output_dir, epoch + 1, model, optimizer, loss_history,
                         config, meta, method)
            if umap_check_loader is not None:
                try:
                    _run_umap_check(model, method, umap_check_loader, device, epoch + 1, seed,
                                    os.path.join(output_dir, f"umap_check_epoch_{epoch + 1}.pdf"))
                    last_umap_epoch = epoch + 1
                except Exception as e:
                    logger.error("UMAP check failed at epoch %d: %s", epoch + 1, e)

    logger.info("SSL pretraining done. Final loss: %.6f",
                loss_history[-1] if loss_history else 0.0)

    # ---- Save final SSL bundle ----
    # 1-based epoch of the last completed epoch. If the loop never ran
    # (resume past epochs), save the loaded checkpoint state as-is.
    last_epoch = epochs if start_epoch < epochs else start_epoch
    _save_bundle(output_dir, last_epoch, model, optimizer, loss_history,
                 config, meta, method, final=True)
    if umap_check_loader is not None and last_umap_epoch != last_epoch:
        try:
            _run_umap_check(model, method, umap_check_loader, device, last_epoch, seed,
                            os.path.join(output_dir, f"umap_check_epoch_{last_epoch}.pdf"))
        except Exception as e:
            logger.error("UMAP check failed at final epoch %d: %s", last_epoch, e)
    bundle_path = os.path.join(output_dir, "ssl_model.pt")

    if loss_history:
        from .vis import plot_pretrain_loss
        try:
            plot_pretrain_loss(
                loss_history,
                dino_history=dino_loss_history if method == "dinov2" else None,
                ibot_history=ibot_loss_history if method == "dinov2" else None,
                koleo_history=koleo_loss_history if method == "dinov2" else None,
                save_path=os.path.join(output_dir, "loss_curve.pdf"))
        except Exception as e:
            logger.error("Loss curve plot failed: %s", e)

    return {"bundle_path": bundle_path, "method": method, "model": model}
