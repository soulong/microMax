"""Classification training from SSL backbone or scratch.

Loads an SSL bundle (model.pt) or a timm pretrained backbone, adds a
ClassificationHead, and trains with FocalLoss. Saves a train bundle
(model.pt) compatible with infer.py's classify mode.

If the SSL bundle's method is "dinov2", the ViT is rebuilt via timm directly
with special init args (pos_embed, dynamic_img_size, init_values) and cls-token
pooling is used. Otherwise (BYOL or conv), build_backbone is used.
"""

import os
import sys
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from . import __version__
from .utils import (logger, set_seed, select_device, load_label_csv, copy_config_file,
                    add_file_logging, atomic_torch_save, merge_locked_normalize,
                    resolve_channels, resolve_max_value, build_cell_datasets)
from .dataset import SingleCellDataset, stratified_split, subsample
from .backbone import (
    build_backbone, build_dinov2_vit, cls_token_pool_fn,
    ClassificationHead, FocalLoss, Model, load_backbone_weights,
)


def _save_bundle(output_dir, epoch, state_dict, meta, config, opt, best_state,
                 best_val, train_loss_history, val_acc_history, val_f1_history,
                 final=False):
    """Save a complete train bundle: state_dict, meta, config, optimizer
    state, epoch, histories, best_val, best_state.

    final=True writes model.pt (final epoch, eval/best weights); otherwise
    model_{epoch}.pt (every save_interval epoch; filename epoch is 1-based,
    no zero padding — model_9.pt = 9th epoch completed; the bundle 'epoch'
    key stays 0-based last-completed, matching the final bundle). Interval
    bundles store the raw training weights at that epoch so the optimizer
    state matches for exact resume. Every saved .pt is a complete bundle —
    usable for exact resume (resume.sl_model) and inference.
    """
    bundle = {
        "state_dict": state_dict,
        "meta": meta,
        "config": config,
        "optimizer_state_dict": opt.state_dict(),
        "epoch": epoch,
        "train_loss_history": train_loss_history,
        "val_acc_history": val_acc_history,
        "val_f1_history": val_f1_history,
        "best_val": best_val,
        "best_state": best_state,
    }
    fname = "model.pt" if final else f"model_{epoch + 1}.pt"
    path = os.path.join(output_dir, fname)
    atomic_torch_save(bundle, path)
    logger.info("Train bundle saved to %s (epoch %d)", path, epoch)


def _infer_state_dict_in_chans(state_dict):
    """Infer the backbone's input channel count from a state dict.

    The first 4-dim weight tensor (first conv / patch-embed projection) has
    shape (out_ch, in_ch, k, k). Returns None if no such tensor exists (then
    callers skip the validation and strict load_state_dict still catches it).
    """
    for key, tensor in state_dict.items():
        if tensor.ndim == 4 and key.endswith("weight"):
            return tensor.shape[1]
    return None


def _try_resume(config, device):
    """Load train checkpoint if resume.sl_model is set. Returns (checkpoint, config)."""
    resume_path = config.get("resume", {}).get("sl_model")
    if not resume_path:
        return None, config

    if not os.path.exists(resume_path):
        print(f"Error: resume train model not found: {resume_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    logger.info("Loaded train checkpoint from epoch %d", ckpt.get("epoch", -1))

    meta = ckpt.get("meta") or {}
    if "num_classes" not in meta and "class_names" not in meta:
        print(
            f"Error: {resume_path} is not a train bundle (meta has no "
            f"'num_classes'/'class_names') — resume.sl_model expects a "
            f"classification checkpoint, not an SSL bundle. Use "
            f"resume.ssl_model for SSL backbone transfer.",
            file=sys.stderr,
        )
        sys.exit(1)

    saved_cfg = ckpt.get("config")
    if saved_cfg is None:
        print(f"Error: train checkpoint at {resume_path} has no 'config' key", file=sys.stderr)
        sys.exit(1)

    # Locked model-specific settings (bundle wins)
    locked = [
        ("model", "backbone"), ("model", "pretrained"), ("model", "focal_gamma"),
    ]
    for section, key in locked:
        sv = saved_cfg.get(section, {}).get(key)
        if sv is not None:
            cv = config.get(section, {}).get(key)
            if str(cv) != str(sv):
                logger.warning("Locked %s.%s differs: bundle=%s, config=%s; using bundle value",
                               section, key, sv, cv)
            config.setdefault(section, {})[key] = sv

    # Locked augmentation + normalize
    for aug_key in ("augmentation_train", "augmentation_infer"):
        sv = saved_cfg.get(aug_key)
        if sv is not None:
            cv = config.get(aug_key)
            if cv != sv:
                logger.warning("Locked %s differs from bundle; using bundle value", aug_key)
            config[aug_key] = sv

    merge_locked_normalize(saved_cfg, config)

    # Fallback training settings
    fallback = [
        ("training", "epochs"), ("training", "lr"), ("training", "weight_decay"),
        ("training", "betas"), ("training", "patience"),
    ]
    for section, key in fallback:
        if key not in config.get(section, {}):
            sv = saved_cfg.get(section, {}).get(key)
            if sv is not None:
                config.setdefault(section, {})[key] = sv
                logger.info("  Using %s.%s=%s from bundle (not set in config)", section, key, sv)

    if "model_state_dict" not in ckpt and "state_dict" in ckpt:
        ckpt["model_state_dict"] = ckpt["state_dict"]

    return ckpt, config


def _build_records_from_cell_dataset(cell_ds, root, label_from_dir, label_csv):
    """Walk a CellDataset's metadata DataFrame and return a list of record dicts."""
    label_map = {}
    if label_csv:
        # A configured-but-missing CSV is a typo — never silently fall back to
        # directory labels (§2 no-guessing).
        if not os.path.exists(label_csv):
            print(f"Error: label_csv file not found: {label_csv}", file=sys.stderr)
            sys.exit(1)
        label_map = load_label_csv(label_csv)

    records = []
    md = cell_ds.metadata
    for i in range(len(md)):
        row = md.iloc[i]
        path = row["path"]
        abs_path = os.path.abspath(path)
        if abs_path in label_map:
            label = label_map[abs_path]
        elif label_from_dir:
            d = row.get("directory", "")
            label = os.path.basename(str(d).replace("\\", "/")) if d else None
        else:
            label = None
        rec = {
            "idx": i,
            "path": path,
            "label": label,
            "root": root,
            "directory": row.get("directory", ""),
            "stem": row.get("stem", ""),
        }
        for col in ("well", "field", "stack", "timepoint"):
            if col in row:
                rec[col] = row[col]
        records.append(rec)
    return records


def _build_model_from_ssl(ssl_bundle, model_cfg, num_classes, device):
    """Build a classification Model from an SSL pretrain bundle.

    For DINOv2: rebuild ViT via build_dinov2_vit + cls-token pooling.
    For BYOL/conv: use build_backbone + global mean pool.
    Returns (model, method, meta).
    """
    meta = ssl_bundle["meta"]
    method = meta.get("method")
    in_chans = meta["in_chans"]

    if method == "dinov2":
        backbone = build_dinov2_vit(meta["backbone"], in_chans, pretrained=False)
        feat_dim = backbone.num_features
        pool_fn = cls_token_pool_fn
    else:
        # BYOL / conv backbone
        backbone, feat_dim, pool_fn = build_backbone(
            meta["backbone"], in_chans, pretrained=False)

    # Load SSL backbone weights from the bundle's full state_dict
    load_backbone_weights(backbone, ssl_bundle, method)
    logger.info("Loaded SSL backbone weights (method=%s)", method)

    head = ClassificationHead(feat_dim, num_classes)
    model = Model(backbone, pool_fn, head).to(device)
    return model, method, meta


def train(config, config_path=None):
    """Classification training from SSL backbone or scratch."""
    method = config.get("method", "classification")
    if method not in ("classification",):
        print(f"Error: train method '{method}' is not supported. Available: "
              f"classification (segmentation is planned but not yet implemented)",
              file=sys.stderr)
        sys.exit(1)

    mode = config.get("mode", "single_cell")
    if mode not in ("single_cell",):
        print(f"Error: train mode '{mode}' is not supported. Available: "
              f"single_cell (whole_image is planned but not yet implemented)",
              file=sys.stderr)
        sys.exit(1)

    resume_cfg = config.get("resume", {})
    if resume_cfg.get("sl_model") and resume_cfg.get("ssl_model"):
        print("Error: resume.sl_model and resume.ssl_model are mutually exclusive; "
              "set only one.", file=sys.stderr)
        sys.exit(1)

    device = select_device()
    checkpoint, config = _try_resume(config, device)

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    norm_cfg = config.get("normalize", {})
    aug_train_cfg = config.get("augmentation_train", [])
    aug_infer_cfg = config.get("augmentation_infer", [])

    normalize_method = norm_cfg.get("method", "per_channel")
    with_masking = norm_cfg.get("with_masking", False)
    clip_low = norm_cfg.get("clip_low", 0.05)
    clip_high = norm_cfg.get("clip_high", 99.95)
    fixed_reference = norm_cfg.get("fixed_reference", False)
    max_value = resolve_max_value(data_cfg)
    label_from_dir = data_cfg.get("label_from_dir", True)
    label_csv = data_cfg.get("label_csv")

    seed = 42
    output_dir = config.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)
    add_file_logging(output_dir)
    if config_path is not None:
        dst = copy_config_file(config_path, output_dir)
        logger.info("Train config copied to %s", dst)
    logger.info("microModel %s train starting (config=%s, device=%s)",
                __version__, config_path, device)

    set_seed(seed)

    # ---- Build CellDataset + records from each root ----
    root = data_cfg["root"]
    roots = [root] if isinstance(root, str) else list(root)
    channels = data_cfg.get("channels")
    channel_layout = data_cfg.get("channel_layout", "CHW")
    image_pattern = data_cfg.get("image_pattern")

    datasets = build_cell_datasets(roots, channel_layout, image_pattern)
    all_records = []
    resolved_per_root = {}
    for r, cell_ds in datasets:
        n_avail = len(cell_ds.intensity_colnames)
        resolved_per_root[r] = resolve_channels(channels, n_avail, r)
        recs = _build_records_from_cell_dataset(
            cell_ds, r, label_from_dir, label_csv)
        for rec in recs:
            rec["cell_dataset"] = cell_ds
        all_records.extend(recs)

    if not all_records:
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

    # Drop unlabeled
    n_unlabeled = sum(1 for r in all_records if r["label"] is None)
    if n_unlabeled > 0:
        logger.warning("Dropping %d records with no label", n_unlabeled)
        all_records = [r for r in all_records if r["label"] is not None]
        if not all_records:
            print("Error: all records are unlabeled", file=sys.stderr)
            sys.exit(1)

    labels = [r["label"] for r in all_records]
    logger.info("Found %d records, %d classes, channels=%s",
                len(all_records), len(set(labels)), resolved_channels)
    logger.info("Class distribution: %s", dict(Counter(labels)))

    all_labels = sorted(set(labels))
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
    num_classes = len(label_to_idx)

    # Subsample
    sample_max = data_cfg.get("sample_max")
    sample_by = data_cfg.get("sample_by", "per_class")
    if sample_max is not None:
        all_records = subsample(all_records, sample_max, sample_by, seed)
        logger.info("Sub-sampled to %d records (sample_max=%s, sample_by=%s)",
                    len(all_records), sample_max, sample_by)
        labels = [r["label"] for r in all_records]
        all_labels = sorted(set(labels))
        label_to_idx = {lab: i for i, lab in enumerate(all_labels)}
        num_classes = len(label_to_idx)

    set_seed(seed)

    val_ratio = train_cfg.get("val_ratio", 0.25)
    if not (0 <= val_ratio < 1):
        print(f"Error: training.val_ratio must be in [0, 1), got {val_ratio!r} "
              f"(0 disables validation)", file=sys.stderr)
        sys.exit(1)
    train_records, val_records = stratified_split(all_records, val_ratio, seed)
    logger.info("Train: %d  Val: %d", len(train_records), len(val_records))

    if not val_records and val_ratio > 0:
        print(
            "Error: validation split is empty — every class needs at least "
            "two samples for a non-empty val set (val_ratio "
            f"{val_ratio}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Build train/val datasets ----
    train_ds = SingleCellDataset(
        [(r["cell_dataset"], r["idx"]) for r in train_records],
        label_to_idx,
        labels=[r["label"] for r in train_records],
        channels=resolved_channels,
        augmentation_spec=aug_train_cfg,
        normalize_method=normalize_method,
        clip_low=clip_low, clip_high=clip_high,
        with_masking=with_masking,
        fixed_reference=fixed_reference,
        max_value=max_value)
    val_ds = SingleCellDataset(
        [(r["cell_dataset"], r["idx"]) for r in val_records],
        label_to_idx,
        labels=[r["label"] for r in val_records],
        channels=resolved_channels,
        augmentation_spec=aug_infer_cfg,
        normalize_method=normalize_method,
        clip_low=clip_low, clip_high=clip_high,
        with_masking=with_masking,
        fixed_reference=fixed_reference,
        max_value=max_value)

    # ---- Build model ----
    ssl_bundle_path = config.get("resume", {}).get("ssl_model")
    ssl_method = None
    # The backbone actually trained (name + in_chans) — from the SSL bundle
    # meta when transferring, else from the train config. The eval-model
    # rebuild and the saved bundle meta must use THIS, not model_cfg, or a
    # config/bundle mismatch silently builds the wrong architecture.
    trained_backbone = model_cfg["backbone"]
    trained_in_chans = len(resolved_channels)
    if ssl_bundle_path:
        if not os.path.exists(ssl_bundle_path):
            print(f"Error: SSL model not found: {ssl_bundle_path}", file=sys.stderr)
            sys.exit(1)
        logger.info("Loading SSL backbone from %s", ssl_bundle_path)
        ssl_bundle = torch.load(ssl_bundle_path, map_location=device, weights_only=False)
        # resume.ssl_model must point at an SSL pretrain bundle (meta has
        # 'method') — a train bundle's meta has no 'method' and would
        # silently take the generic backbone path below (which only works
        # by accident). Same contract pretrain enforces via
        # _load_checkpoint_state.
        if "method" not in (ssl_bundle.get("meta") or {}):
            print(
                "Error: resume.ssl_model must point at an SSL pretrain bundle "
                f"(meta has 'method'); {ssl_bundle_path} is not an SSL bundle. "
                "Use resume.sl_model for train-bundle resumes.",
                file=sys.stderr,
            )
            sys.exit(1)
        model, ssl_method, ssl_meta = _build_model_from_ssl(
            ssl_bundle, model_cfg, num_classes, device)
        trained_backbone = ssl_meta["backbone"]
        trained_in_chans = ssl_meta["in_chans"]
    else:
        # From scratch (timm pretrained). On resume the checkpoint state dict
        # fully overwrites these weights, so the ImageNet load is skipped.
        if checkpoint is not None:
            ckpt_state = checkpoint.get("model_state_dict")
            ckpt_in_chans = _infer_state_dict_in_chans(ckpt_state) if ckpt_state else None
            if ckpt_in_chans is not None and ckpt_in_chans != len(resolved_channels):
                print(
                    f"Error: data.channels ({len(resolved_channels)}) does not match the "
                    f"resume checkpoint's input channels ({ckpt_in_chans}); re-run with "
                    f"matching data.channels or without resume.sl_model",
                    file=sys.stderr,
                )
                sys.exit(1)
        backbone, feat_dim, pool_fn = build_backbone(
            model_cfg["backbone"], len(resolved_channels),
            model_cfg.get("pretrained", True) if checkpoint is None else False)
        head = ClassificationHead(feat_dim, num_classes)
        model = Model(backbone, pool_fn, head).to(device)

    # Train bundle meta — known before training; shared by every saved
    # bundle (interval model_{epoch}.pt + final model.pt).
    meta = {
        "class_names": [lab for lab in sorted(label_to_idx, key=label_to_idx.get)],
        "channels": resolved_channels,
        "channel_layout": channel_layout,
        "max_value": max_value,
        "in_chans": trained_in_chans,
        "backbone": trained_backbone,
        "num_classes": num_classes,
        "augmentation_train": aug_train_cfg,
        "augmentation_infer": aug_infer_cfg,
        "normalize_method": normalize_method,
        "normalize_with_masking": with_masking,
        "normalize_fixed_reference": fixed_reference,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "image_pattern": image_pattern,
        "ssl_method": ssl_method,
    }

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))))
    criterion = FocalLoss(gamma=model_cfg.get("focal_gamma", 2.0))

    amp_enabled = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    dl_cfg = config.get("dataloader", {})
    num_workers = dl_cfg.get("num_workers", 4)
    prefetch_factor = dl_cfg.get("prefetch_factor", 2)
    persistent_workers = dl_cfg.get("persistent_workers", True) and num_workers > 0
    batch_size = train_cfg.get("batch_size", 128)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         persistent_workers=persistent_workers,
                         pin_memory=(device.type == "cuda"))
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    epochs = train_cfg.get("epochs", 10)
    if epochs is None or int(epochs) < 1:
        print(f"Error: training.epochs must be a positive integer, got {epochs!r}",
              file=sys.stderr)
        sys.exit(1)
    epochs = int(epochs)
    patience = train_cfg.get("patience", 5)
    save_interval = train_cfg.get("save_interval")

    start_epoch = 0
    train_loss_history = []
    val_acc_history = []
    val_f1_history = []
    best_acc = 0.0
    best_state = None
    patience_counter = 0

    if checkpoint is not None:
        start_epoch = checkpoint.get("epoch", -1) + 1
        logger.info("Resuming from epoch %d", start_epoch)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss_history = checkpoint.get("train_loss_history", [])
        val_acc_history = checkpoint.get("val_acc_history", [])
        val_f1_history = checkpoint.get("val_f1_history", [])
        best_acc = checkpoint.get("best_val", 0.0)
        best_state = checkpoint.get("best_state")

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs):
        model.train()
        tot_loss = 0.0
        n_batches = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    logits, _ = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
            tot_loss += loss.item()
            n_batches += 1
        avg_loss = tot_loss / n_batches if n_batches else 0.0
        train_loss_history.append(avg_loss)

        # Validation (skipped entirely when val_ratio: 0 disabled the split)
        model.eval()
        yt, yp = [], []
        if val_records:
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    logits, _ = model(x)
                    yt.extend(y.tolist())
                    yp.extend(logits.argmax(1).cpu().tolist())
        acc = float(accuracy_score(yt, yp)) if yt else 0.0
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0)) if yt else 0.0
        val_acc_history.append(acc)
        val_f1_history.append(f1)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if save_interval and (epoch + 1) % save_interval == 0:
            _save_bundle(output_dir, epoch, model.state_dict(), meta, config,
                         opt, best_state, best_acc,
                         train_loss_history, val_acc_history, val_f1_history)

        logger.info("  loss=%.4f  val_acc=%.4f  val_f1=%.4f  patience=%d/%d%s",
                    avg_loss, acc, f1, patience_counter, patience, marker)

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Training done. Best val accuracy: %.4f", best_acc)

    # ---- Rebuild eval model from best_state ----
    # Must use the backbone that was actually trained (trained_backbone/
    # trained_in_chans), not model_cfg — a config/bundle mismatch otherwise
    # builds the wrong architecture and load_state_dict fails or the eval
    # model silently has random weights.
    if ssl_method == "dinov2":
        eval_backbone = build_dinov2_vit(trained_backbone, trained_in_chans,
                                         pretrained=False)
        eval_feat_dim = eval_backbone.num_features
        eval_pool_fn = cls_token_pool_fn
    else:
        eval_backbone, eval_feat_dim, eval_pool_fn = build_backbone(
            trained_backbone, trained_in_chans, pretrained=False)
    eval_head = ClassificationHead(eval_feat_dim, num_classes)
    eval_model = Model(eval_backbone, eval_pool_fn, eval_head)

    if best_state is not None:
        eval_model.load_state_dict(best_state)
    else:
        logger.warning("best_state is None. Using last model weights.")
        eval_model.load_state_dict(model.state_dict())
    eval_model.eval()

    # ---- Plots + report ----
    from .vis import plot_training_results
    eval_model.to(device)
    result = plot_training_results(
            eval_model, device, val_loader, num_classes,
            label_to_idx, train_loss_history, val_acc_history,
            save_path=os.path.join(output_dir, "training_plot.pdf"))

    if result is not None:
        report_path = os.path.join(output_dir, "training_report.txt")
        class_names_sorted = [lab for lab in sorted(label_to_idx, key=label_to_idx.get)]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Training Report\n")
            f.write("===============\n\n")
            f.write(f"Backbone: {trained_backbone}\n")
            f.write(f"SSL method: {ssl_method or 'scratch'}\n")
            f.write(f"Input channels: {len(resolved_channels)}\n")
            f.write(f"Num classes: {num_classes}\n")
            f.write(f"Classes: {class_names_sorted}\n\n")
            f.write(f"Total epochs: {len(train_loss_history)}\n")
            f.write(f"Best validation accuracy: {best_acc:.4f}\n\n")
            f.write("Per-epoch metrics:\n")
            f.write(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Acc':>8}  {'Val F1':>8}\n")
            f.write("-" * 42 + "\n")
            for e in range(len(train_loss_history)):
                f.write(f"{e:>6}  {train_loss_history[e]:>11.6f}  {val_acc_history[e]:>8.4f}  {val_f1_history[e]:>8.4f}\n")
            f.write("\n")
            f.write("Final metrics (on full validation set):\n")
            f.write(f"  Train loss: {result['final_loss']:.6f}\n")
            f.write(f"  Validation accuracy: {result['final_acc']:.4f}\n")
            f.write(f"  Validation F1 (macro): {result['final_val_f1']:.4f}\n\n")
            f.write("Confusion matrix:\n")
            cm = result["cm"]
            header = "true\\pred\t" + "\t".join(str(i) for i in range(num_classes))
            f.write(header + "\n")
            for i in range(cm.shape[0]):
                row_str = f"{class_names_sorted[i] if i < len(class_names_sorted) else i}\t"
                row_str += "\t".join(str(int(cm[i, j])) for j in range(cm.shape[1]))
                f.write(row_str + "\n")
        logger.info("Training report saved to %s", report_path)

    # ---- Save final train bundle ----
    bundle_path = os.path.join(output_dir, "model.pt")
    _save_bundle(
        output_dir, epoch if start_epoch < epochs else start_epoch - 1,
        eval_model.state_dict(), meta, config, opt, best_state, best_acc,
        train_loss_history, val_acc_history, val_f1_history, final=True)
    logger.info("Train bundle saved to %s", bundle_path)
    logger.info("  num_classes=%d  best_val_acc=%.4f  ssl_method=%s",
                num_classes, best_acc, ssl_method)

    return {"best_acc": best_acc, "bundle_path": bundle_path,
            "num_classes": num_classes, "label_to_idx": label_to_idx,
            "model": eval_model, "channels": resolved_channels}
