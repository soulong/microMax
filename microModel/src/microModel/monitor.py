"""Training quality monitoring: per-step/epoch metrics (metrics.csv +
TensorBoard) and DINOv3-specific diagnostics.

Two layers, kept deliberately separable:

  1. MetricsTracker — method-agnostic CSV + TensorBoard writers. Every SSL
     method can log its (loss, component) scalars at step/epoch granularity.
  2. Diagnostic helpers — DINOv3-only signals computed from tensors that
     dinov3.train_step already has in scope (no extra forward passes):
       - head_collapse_metrics: student head-logit std / normalized entropy
         and teacher-student CLS cosine similarity (the L2-norm + Sinkhorn
         design should keep head logits alive; std < ~5e-3 signals collapse).
       - gram_split_metrics: the masked vs unmasked Gram losses (official
         `gram.compute_stats` semantics) — a rising unmasked Gram is a red
         flag for dense-feature degradation.
       - compute_patch_similarity_maps: per-image cosine-similarity maps from
         a few anchor patches against all patches (paper Fig. 3 style);
         segment-like highlights = healthy dense features.
"""

import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .utils import logger

#: Column order of metrics.csv (missing keys are written as blanks so
#: appending across continue-resume runs keeps a consistent header).
COLUMNS = [
    "tag", "step", "epoch", "loss", "dino", "ibot", "koleo", "recon", "dist",
    "gram", "gram_masked", "gram_unmasked", "head_logits_std", "head_entropy",
    "teacher_student_sim", "lr", "momentum", "weight_decay", "teacher_temp",
]


class MetricsTracker:
    """CSV + TensorBoard sinks for scalar metrics.

    tensorboard=False / csv=False disable each sink independently; a
    TensorBoard init failure only logs a warning (training continues).
    """

    def __init__(self, output_dir, tensorboard=True, csv_enabled=True, log_every_steps=10):
        self.output_dir = output_dir
        self.log_every_steps = max(1, int(log_every_steps))
        self.writer = None
        self._csv_file = None
        self._csv_writer = None

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(output_dir, "tensorboard")
                self.writer = SummaryWriter(log_dir=tb_dir)
                logger.info("TensorBoard metrics -> %s", tb_dir)
            except Exception as e:  # noqa: BLE001 — monitoring must never abort training
                logger.warning("TensorBoard unavailable, disabled: %s", e)

        if csv_enabled:
            self.csv_path = os.path.join(output_dir, "metrics.csv")
            self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=COLUMNS, extrasaction="ignore")
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()
            logger.info("Per-step metrics CSV -> %s", self.csv_path)

    # -- sinks ---------------------------------------------------------
    def add_step(self, step, epoch, metrics):
        """Per-step (throttled by the caller) metrics row + TB scalars."""
        self._write_row("step", step, epoch, metrics)

    def add_epoch(self, epoch, metrics):
        """Per-epoch metrics row + TB scalars (x-axis = epoch)."""
        self._write_row("epoch", epoch, epoch, metrics)

    def add_image(self, name, image_hw_c, step):
        if self.writer is None:
            return
        try:
            self.writer.add_image(name, image_hw_c, step, dataformats="HWC")
        except Exception as e:  # noqa: BLE001
            logger.warning("TensorBoard image failed: %s", e)

    def close(self):
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:  # noqa: BLE001
                pass
            self._csv_file = None
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:  # noqa: BLE001
                pass
            self.writer = None

    # -- internals -----------------------------------------------------
    def _write_row(self, tag, step, epoch, metrics):
        row = {"tag": tag, "step": step, "epoch": epoch}
        for k in COLUMNS:
            if k in ("tag", "step", "epoch"):
                continue
            v = metrics.get(k)
            row[k] = "" if v is None else f"{v:.6g}"
        if self._csv_writer is not None:
            try:
                self._csv_writer.writerow(row)
                self._csv_file.flush()  # durable even on hard exits
            except Exception:  # noqa: BLE001
                logger.warning("metrics.csv write failed (step %d)", step)
        if self.writer is not None:
            for k, v in metrics.items():
                if v is None:
                    continue
                try:
                    self.writer.add_scalar(f"{k}/{tag}", float(v), int(step))
                except Exception:  # noqa: BLE001
                    pass


# ---------------------------------------------------------------------------
# DINOv3 diagnostics (computed from tensors already available in train_step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def head_collapse_metrics(diag, student_temp=0.1):
    """-> {head_logits_std, head_entropy, teacher_student_sim} floats.

    diag: {head_logits (B,K), student_cls (B,D), teacher_cls (B,D)}.
    """
    logits = diag["head_logits"].float()
    k = max(1, logits.shape[-1])
    out = {"head_logits_std": float(logits.std(dim=-1).mean())}
    probs = F.softmax(logits / student_temp, dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()
    out["head_entropy"] = float(entropy / math.log(k))  # normalized to [0, 1]
    s = F.normalize(diag["student_cls"].float(), dim=-1)
    t = F.normalize(diag["teacher_cls"].float(), dim=-1)
    out["teacher_student_sim"] = float((s * t).sum(dim=-1).mean())
    return out


@torch.no_grad()
def gram_split_metrics(gram_criterion, student_patches, teacher_patches, mask_patch):
    """-> {gram_masked, gram_unmasked} floats (official compute_stats semantics).

    student_patches/teacher_patches: (B, P, D); mask_patch: (B, P) bool over
    the patch tokens only (no class/reg prefix). Student side detached.
    """
    if mask_patch is None or student_patches.shape != teacher_patches.shape:
        return {}
    target = teacher_patches
    m = mask_patch.any(dim=-1)
    if not m.any():
        return {}
    sp_m = student_patches.detach()[mask_patch]
    tp_m = target[mask_patch]
    sp_u = student_patches.detach()[~mask_patch]
    tp_u = target[~mask_patch]
    return {
        "gram_masked": float(gram_criterion(sp_m, tp_m, img_level=False)),
        "gram_unmasked": float(gram_criterion(sp_u, tp_u, img_level=False)),
    }


@torch.no_grad()
def compute_patch_similarity_maps(model, x, n_anchors=4):
    """-> (maps, grid, anchors) for one input image.

    model: DINOv3 (uses teacher_backbone, the EMA branch official evaluation
    uses). x: (1, C, H, W) normalized tensor on the right device. maps: list
    of n_anchors (grid, grid) cosine-sim maps; anchors: the chosen flat
    patch indices (attention_vis annotates the sheets with them).
    """
    n_prefix = model.teacher_backbone.vit.num_prefix_tokens
    features = model.teacher_backbone.encode(x)  # (1, N, D)
    patches = features[:, n_prefix:]  # drop class/reg tokens
    patches = F.normalize(patches[0].float(), dim=-1)  # (P, D)
    p = patches.shape[0]
    grid = int(round(math.sqrt(p)))
    if grid * grid != p:
        return [], grid, []
    anchors = _pick_anchors(p, grid, n_anchors)
    maps = []
    for a in anchors:
        sim = patches @ patches[a]  # (P,)
        maps.append(sim.reshape(grid, grid).cpu().numpy())
    return maps, grid, anchors


def _pick_anchors(p, grid, n_anchors):
    """Anchor patch indices: quantile positions + patch-grid centre."""
    anchors = {p // 4, p // 2, 3 * p // 4,
               (grid // 2) * grid + grid // 2}
    anchors = sorted(list(anchors))[:n_anchors]
    return [max(0, min(p - 1, a)) for a in anchors]


def resize_map(arr, target):
    """(grid, grid) or (grid, grid, C) ndarray -> nearest-neighbour resized
    to the target (h, w) pixel size (aligns heat maps with their input image
    for side-by-side TensorBoard previews)."""
    a = np.asarray(arr)
    if a.ndim == 3:  # per-channel planes handled independently
        return np.stack([resize_map(a[..., c], target)
                         for c in range(a.shape[-1])], axis=-1)
    import torch.nn.functional as F
    t = torch.from_numpy(a.astype(np.float32))[None, None]
    return F.interpolate(t, size=target, mode="nearest")[0, 0].numpy()


def maps_to_grid(maps, grid, cmap_name="coolwarm"):
    """Stack one sample's anchor maps side by side -> HxW RGB uint8 array."""
    if not maps:
        return np.zeros((grid, grid, 3), dtype=np.uint8)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    stack = np.hstack(maps)  # (grid, grid * len(maps))
    norm = (stack - (-1.0)) / 2.0  # [-1, 1] -> [0, 1]
    rgb = cmap(norm.clip(0.0, 1.0))[..., :3]
    return (rgb * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLS attention maps (DINO-style segmentation visualization)
# ---------------------------------------------------------------------------

def compute_cls_attention_maps(backbone, x, layer=-1):
    """Last-layer CLS->patch attention maps -> (maps, grid).

    backbone: DINOv3 masked wrapper (.vit holds the raw timm backbone).
    x: (B, C, H, W) normalized tensor on the right device.
    timm 1.0.28 exposes no attention weights (no need_weights / out_attn), so
    the token stream is replayed and the LAST block's q@k -> softmax weights
    are recomputed by hand (both timm DINOv3/Eva and classic timm
    VisionTransformer branches). Returns per-sample (grid, grid) mean-head
    CLS-attention maps.
    """
    vit = backbone.vit
    from timm.models.vision_transformer import VisionTransformer
    if isinstance(vit, VisionTransformer):
        _cls_attn = _classic_cls_attn_weights(vit, x, layer)
    else:
        _cls_attn = _eva_cls_attn_weights(vit, x, layer)
    # _cls_attn: (B, heads, n_patch) -> expose every single head PLUS the
    # mean head so users can inspect head-specific foci (nucleus/cytosol/...).
    attn = _cls_attn.float()
    samples = []
    grid = None
    for i in range(attn.shape[0]):
        per_head = []
        for hi in range(attn.shape[1]):
            p = attn[i, hi].numel()
            g = int(round(math.sqrt(p)))
            if g * g == p:
                per_head.append(attn[i, hi].reshape(g, g).cpu().numpy())
                grid = g
        if not per_head:
            continue
        mean_map = np.mean(np.stack(per_head, axis=0), axis=0)
        samples.append({"per_head": per_head, "mean": mean_map})
    return samples, grid


@torch.no_grad()
def _eva_cls_attn_weights(vit, x, layer):
    """Replay the Eva token stream and recompute the last block's CLS row."""
    from timm.layers import apply_rot_embed_cat
    x = vit.patch_embed(x)
    x, rot = vit._pos_embed(x)
    x = vit.norm_pre(x)
    rope_mixed = getattr(vit, "rope_mixed", False)
    stop = layer if layer < 0 else layer + 1
    blocks = list(vit.blocks)
    stop = len(blocks) + stop if stop < 0 else stop
    for i, blk in enumerate(blocks[:stop]):
        rope = rot[i] if rope_mixed else rot
        x = blk(x, rope=rope)
    blk = blocks[stop]
    a = blk.attn
    xh = blk.norm1(x)
    rope = rot[stop] if rope_mixed else rot
    # qkv -> (3, B, heads, N, dim)  (mirror Eva Attention.forward exactly)
    if a.q_bias is None:
        qkv = a.qkv(xh)
    else:
        qkv_bias = torch.cat((a.q_bias, a.k_bias, a.v_bias))
        if getattr(a, "qkv_bias_separate", False):
            qkv = a.qkv(xh) + qkv_bias
        else:
            qkv = torch.nn.functional.linear(xh, weight=a.qkv.weight, bias=qkv_bias)
    B, N, _ = xh.shape
    qkv = qkv.reshape(B, N, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = a.q_norm(q), a.k_norm(k)
    if rope is not None:
        npt = a.num_prefix_tokens
        half = getattr(a, "rotate_half", False)
        q = torch.cat([q[:, :, :npt, :],
                       apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)],
                      dim=2)
        k = torch.cat([k[:, :, :npt, :],
                       apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)],
                      dim=2)
    q = q * a.scale
    attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
    npt = a.num_prefix_tokens
    return attn[:, :, 0, npt:]  # (B, heads, n_patch)


@torch.no_grad()
def _classic_cls_attn_weights(vit, x, layer):
    """Replay a classic timm VisionTransformer and recompute the last block's
    CLS row (learned pos-embed path)."""
    B = x.shape[0]
    x = vit.patch_embed(x)
    if x.ndim == 4:
        x = x.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
    prefix = []
    if vit.cls_token is not None:
        prefix.append(vit.cls_token.expand(B, -1, -1))
    if getattr(vit, "reg_token", None) is not None:
        prefix.append(vit.reg_token.expand(B, -1, -1))
    if prefix:
        x = torch.cat(prefix + [x], dim=1)
    pos = vit.pos_embed
    if pos.shape[1] != x.shape[1] and vit.dynamic_img_size:
        # resample for size-agnostic classic ViTs
        from timm.layers.pos_embed import resample_abs_pos_embed
        grid_size = int(math.sqrt(x.shape[1] - vit.num_prefix_tokens))
        pos = resample_abs_pos_embed(pos, (grid_size, grid_size),
                                     num_prefix_tokens=vit.num_prefix_tokens,
                                     antialias=False)
    if not vit.no_embed_class:
        x = x + pos
    x = vit.pos_drop(x)
    x = vit.norm_pre(x)
    blocks = list(vit.blocks)
    stop = layer if layer < 0 else layer + 1
    stop = len(blocks) + stop if stop < 0 else stop
    for blk in blocks[:stop]:
        x = blk(x)
    blk = blocks[stop]
    a = blk.attn
    xh = blk.norm1(x)
    qkv = a.qkv(xh).reshape(B, xh.shape[1], 3, a.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if getattr(a, "q_norm", None) is not None:
        q, k = a.q_norm(q), a.k_norm(k)
    q = q * a.scale
    attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
    npt = vit.num_prefix_tokens
    return attn[:, :, 0, npt:]  # (B, heads, n_patch)