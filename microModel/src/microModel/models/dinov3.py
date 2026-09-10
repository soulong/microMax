"""DINOv3 SSL model (Meta AI recipe, arXiv:2508.10104) on a timm ViT backbone.

The backbone stays a plain timm ViT (rebuilt via build_dino_vit in
backbone.py — pos_embed="learn", dynamic_img_size=True, init_values=1e-5),
ported 1:1 from facebookresearch/dinov3 (which is built around a custom ViT
+ FSDP; microModel replaces that with any timm ViT that has a class token):

  - Heads: DINO head (dino/CLS prototypes) and iBOT head (patch prototypes)
    are the official DINOHead — MLP -> bottleneck(256) -> L2-norm -> Linear
    (no BatchNorm, no weight-norm). The L2-norm + Sinkhorn-Knopp teacher
    centering is the anti-collapse mechanism.
  - DINO loss: cross-entropy; teacher targets are Sinkhorn-Knopp-centered
    softmax (not softmax+EMA-center). Optionally ignores the diagonal
    (global crop i vs itself) — the DINOv3 default.
  - iBOT loss: patch-level CE on the masked patches only; teacher targets
    Sinkhorn-Knopp over all masked patches jointly.
  - KoLeo loss on the student's pre-head CLS tokens of the global views.
  - Gram anchoring (optional): a regularization that pushes the Gram matrix
    of the student's patch features toward that of a frozen "gram teacher" —
    either the EMA teacher (gram.ema_teacher) or a separate frozen backbone
    initialized from an earlier checkpoint (gram.ckpt). Ported GramLoss with
    normalize / remove_neg / tokens_used / img_level options; supports higher
    resolution gram-teacher crops (gram.crops_size) with spatial resampling
    of the teacher features to the student patch grid. With gram.rep_update
    the gram teacher is periodically refreshed from the EMA teacher.

Backbone masking (iBOT): DINOv3 block masks (random rectangular blocks,
same generator as the official MaskingGenerator) with the linear mask-ratio
schedule and batch-level mask probability, applied to the student's global
views only.

This module is a drop-in SSL method for microModel: models/__init__.py
registers "dinov3", pretrain.py dispatches to it, and every saved bundle has
the "teacher_backbone.vit.*" / "student_backbone.vit.*" prefixes so
extract_backbone_state_dict and the downstream train/infer consumers know
which branch to pull. Downstream extraction uses the TEACHER branch by
default (EMA/Polyak average — official DINO-family evaluation practice).

References:
  - https://arxiv.org/abs/2508.10104
  - https://github.com/facebookresearch/dinov3
"""

import copy
import math
import os
import random as _random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from lightly.models.modules import MaskedVisionTransformerTIMM
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule
from timm.models.vision_transformer import VisionTransformer

from microBase import MicroMaxError

from ..backbone import build_dino_vit
from ..utils import logger


def _update_momentum_with_buffers(model, model_ema, m):
    """EMA of parameters AND float buffers (lightly's update_momentum only
    does parameters).

    Buffers matter for BatchNorm in the projection heads: the teacher's BN
    running stats must track the student's, otherwise the teacher targets are
    computed with stale statistics. Non-float buffers (e.g. BN's
    num_batches_tracked) are left untouched.
    """
    for ema_param, param in zip(model_ema.parameters(), model.parameters()):
        ema_param.data.mul_(m).add_(param.data, alpha=1 - m)
    for ema_buffer, buffer in zip(model_ema.buffers(), model.buffers()):
        if ema_buffer.dtype.is_floating_point:
            ema_buffer.data.mul_(m).add_(buffer.data, alpha=1 - m)


# ---------------------------------------------------------------------------
# Masked wrapper for timm DINOv3 (Eva-based) backbones
# ---------------------------------------------------------------------------

class _MaskedEva(nn.Module):
    """Masked-patch wrapper for timm DINOv3 (Eva / RoPE) backbones.

    Mirrors the interface that dinov3.py relies on from lightly's
    MaskedVisionTransformerTIMM so both backbones are treated identically:
      - ``self.vit``            -> the raw timm backbone (attribute-compatible:
                                  num_prefix_tokens, embed_dim, patch_embed ...)
      - ``self.mask_token``     -> learnable mask-token parameter
      - ``encode(x, mask=None)`` -> (B, prefix+patches, D) normalized features
      - ``forward(x)``          -> encode without masking (features[:, 0] = CLS)

    Eva uses RoPE (no learned pos_embed) and register tokens; masking follows
    the official DINOv3 approach — masked patch tokens are replaced with the
    mask token before the transformer blocks (no positional reordering needed
    since positions are encoded by RoPE inside each block).
    """

    def __init__(self, vit, mask_token=None):
        super().__init__()
        self.vit = vit
        self.mask_token = (mask_token if mask_token is not None
                           else nn.Parameter(torch.zeros(1, 1, vit.embed_dim)))

    def encode(self, images, mask=None):
        x = self.vit.patch_embed(images)  # dynamic: (B, H, W, C)
        if mask is not None:
            # patch-only boolean mask (drop the region of prefix tokens)
            patch_mask = mask[:, self.vit.num_prefix_tokens:]
            flat = x.flatten(1, 2)  # (B, P, C)
            flat = flat.clone()
            flat[patch_mask] = self.mask_token.to(dtype=flat.dtype)
            x = flat.reshape(x.shape[0], x.shape[1], x.shape[2], -1) if x.ndim == 4 \
                else flat
        x, rot_pos_embed = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        rope_mixed = getattr(self.vit, "rope_mixed", False)
        for i, blk in enumerate(self.vit.blocks):
            rope = rot_pos_embed[i] if rope_mixed else rot_pos_embed
            x = blk(x, rope=rope)
        x = self.vit.norm(x)
        return x

    def forward(self, images, mask=None):
        return self.encode(images, mask=mask)


def _apply_drop_path(vit, drop_path_rate):
    """Set uniform stochastic-depth on the timm backbone (works for both the
    classic VisionTransformer blocks (drop_path1/drop_path2) and Eva blocks)."""
    if not drop_path_rate:
        return
    for mod in vit.modules():
        for attr in ("drop_path1", "drop_path2"):
            dp = getattr(mod, attr, None)
            if dp is not None and hasattr(dp, "drop_prob"):
                dp.drop_prob = drop_path_rate


def _build_masked_vit(vit, weight_init=""):
    """Wrap a timm ViT for patch masking: classic VisionTransformer -> lightly
    wrapper; timm DINOv3 (Eva) -> _MaskedEva.
    """
    if isinstance(vit, VisionTransformer):
        return MaskedVisionTransformerTIMM(
            vit=vit, antialias=False, pos_embed_initialization="skip",
            weight_initialization=weight_init)
    if hasattr(vit, "_pos_embed") and getattr(vit, "num_prefix_tokens", 0) >= 1 \
            and getattr(vit, "embed_dim", None) is not None:
        logger.info("Wrapping %s with _MaskedEva (timm DINOv3 backbone)",
                    type(vit).__name__)
        return _MaskedEva(vit)
    raise ValueError(f"Unsupported timm backbone for DINOv3: {type(vit).__name__}")


# ---------------------------------------------------------------------------
# Block masking (ported from dinov3/data/masking.py)
# ---------------------------------------------------------------------------

class _MaskingGenerator:
    """Random-rectangle block mask generator (DINOv3 MaskingGenerator)."""

    def __init__(self, input_size, num_masking_patches=None, min_num_patches=4,
                 max_num_patches=None, min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = _random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(_random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = _random.randint(0, self.height - h)
                left = _random.randint(0, self.width - w)
                num_masked = mask[top:top + h, left:left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def complete_mask_randomly(self, mask, num_masking_patches):
        shape = mask.shape
        m2 = mask.flatten()
        to_add = np.random.choice(np.where(~m2)[0],
                                  size=num_masking_patches - m2.sum(), replace=False)
        m2[to_add] = True
        return m2.reshape(shape)

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return self.complete_mask_randomly(mask, num_masking_patches)

    def get_shape(self):
        return self.height, self.width


def _make_masks(model, n_global_views, grid_h, grid_w, device):
    """DINOv3-style batch masks for the student's global views.

    Returns (mask_full, masks_weight):
      - mask_full:  (B, grid_h*grid_w + n_prefix) bool, prefix tokens unmasked
      - masks_weight: (n_masked,) per-sample normalized weights for the
        iBOT loss (mirrors dinov3/data/collate.py)
    """
    n_prefix = model.teacher_backbone.vit.num_prefix_tokens
    n_patches = grid_h * grid_w
    B = n_global_views
    n_masked_samples = int(B * model._mask_sample_probability)
    if n_masked_samples <= 0:
        mask_full = torch.zeros((B, n_patches + n_prefix), dtype=torch.bool, device=device)
        masks_weight = None
        return mask_full, masks_weight
    probs = torch.linspace(
        model._mask_ratio_min, model._mask_ratio_max, n_masked_samples + 1).tolist()
    gen = _MaskingGenerator((grid_h, grid_w),
                            max_num_patches=int(0.5 * n_patches))
    masks_list = []
    for i in range(n_masked_samples):
        num = int(n_patches * probs[i + 1])
        masks_list.append(np.asarray(gen(num), dtype=np.uint8))
    for _ in range(n_masked_samples, B):
        masks_list.append(np.zeros((grid_h, grid_w), dtype=np.uint8))
    _random.shuffle(masks_list)
    masks = np.stack(masks_list, axis=0).reshape(B, n_patches)
    mask_full = torch.zeros((B, n_patches + n_prefix), dtype=torch.bool, device=device)
    mask_full[:, n_prefix:] = torch.from_numpy(masks.astype(bool)).to(device)

    per_sample = mask_full.sum(dim=-1).clamp(min=1.0).float()
    mask_flat = mask_full.reshape(B, n_patches + n_prefix)
    masks_weight = (1.0 / per_sample).repeat_interleave(
        mask_flat.sum(dim=-1).long())  # (n_masked,) same order as mask_full[mask_full]
    return mask_full, masks_weight


# ---------------------------------------------------------------------------
# Losses (ported from dinov3/loss/, single-process)
# ---------------------------------------------------------------------------

class DINOLoss(nn.Module):
    """DINO loss with Sinkhorn-Knopp teacher centering (DINOv3).

    Prototype count is never configured here — every shape comes from the
    tensors at forward time; only the student temperature is a parameter.
    """

    def __init__(self, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        """teacher_output: (B, K) logits -> (B, K) assignment probabilities."""
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()  # (K, B)
        B = Q.shape[1]
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q
        for _ in range(n_iterations):
            # normalize rows: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            # normalize columns: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B  # columns sum to 1, i.e. Q is an assignment
        return Q.t()

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        """Cross-entropy between student logits and teacher probs.

        student_logits: (S, B, K), teacher_probs: (T, B, K) sum to 1 over K.
        """
        S, B, K = student_logits.shape
        T = teacher_probs.shape[0]
        student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        if ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(S, T)
            loss[range(min_st), range(min_st)] = 0.0
            return loss.sum() / (B * S * T - B * min_st)
        loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
        return loss / (B * S * T)


class iBOTPatchLoss(nn.Module):
    """iBOT patch-level loss with Sinkhorn-Knopp teacher centering (DINOv3)."""

    def __init__(self, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_patch_tokens, teacher_temp,
                               n_masked_patches, n_iterations=3):
        """teacher_patch_tokens: (n_masked, K) -> (n_masked, K) assignments."""
        teacher_patch_tokens = teacher_patch_tokens.float()
        Q = torch.exp(teacher_patch_tokens / teacher_temp).t()  # (K, n_masked)
        K = Q.shape[0]
        B = int(n_masked_patches)  # total number of masked patches

        sum_Q = torch.sum(Q)
        Q /= sum_Q
        for _ in range(n_iterations):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    def lossfunc(self, t, s, temp):
        return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)

    def forward_masked(self, student_patch_tokens_masked, teacher_patch_tokens_masked,
                       masks_weight):
        """Masked-patch CE (teacher probs precomputed via sinkhorn_knopp_teacher).

        student/teacher: (n_masked, K); masks_weight: (n_masked,) from _make_masks.
        """
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss = self.lossfunc(t, s, self.student_temp) * masks_weight
        return -loss.sum() / t.shape[0]


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic regularizer (DINOv3, on pre-head CLS tokens)."""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.fill_diagonal_(-1.0)
        _, indices = torch.max(dots, dim=1)
        return indices

    def forward(self, student_output, eps=1e-8):
        """student_output: (B, D) pre-head features -> scalar."""
        with torch.autocast(device_type=str(student_output.device).split(":")[0],
                            enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            indices = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[indices])
            loss = -torch.log(distances + eps).mean()
        return loss


class GramLoss(nn.Module):
    """Gram anchoring loss (DINOv3, loss/gram_loss.py)."""

    def __init__(self, apply_norm=True, remove_neg=False, remove_only_teacher_neg=False):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg
        if self.remove_neg and self.remove_only_teacher_neg:
            raise ValueError("remove_neg and remove_only_teacher_neg are exclusive.")

    def forward(self, output_feats, target_feats, img_level=True):
        """MSE between the Gram matrices of student (output) and gram teacher.

        img_level=True: features (B, N, D) — Gram per image.
        img_level=False: flattened (B*N, D) or (n_sel, D) — Gram over the batch.
        """
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)
        if not img_level and len(target_feats.shape) == 3:
            target_feats = target_feats.flatten(0, 1)
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)
        if not img_level and len(output_feats.shape) == 3:
            output_feats = output_feats.flatten(0, 1)
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0
        elif self.remove_only_teacher_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0
        return self.mse_loss(student_sim, target_sim)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class DINOv3Head(nn.Module):
    """DINO head: MLP -> bottleneck -> L2-norm -> prototype Linear (DINOv3).

    in_dim (backbone embed) -> hidden_dim (x nlayers-1) -> bottleneck_dim ->
    L2 normalize -> Linear(bottleneck_dim, out_dim, bias=False).
    No BatchNorm, no weight-norm: anti-collapse comes from the bottleneck
    L2-norm and the Sinkhorn-Knopp teacher targets.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            mlp = [nn.Linear(in_dim, bottleneck_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            mlp = layers
        self.mlp = nn.Sequential(*mlp)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.bottleneck_dim = bottleneck_dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self):
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


class DINOv3HeadPair(Module):
    """Wrapper holding the dino (cls-token) and ibot (patch-token) heads."""

    def __init__(self, dino_head, ibot_head):
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DINOv3(nn.Module):
    """DINOv3 model: teacher + student ViT backbones, each with a head pair,
    plus an optional frozen gram-teacher backbone for the Gram anchoring loss.

    Teacher is frozen (momentum-updated). Student receives patch masking on
    global views. Gram teacher is either the EMA teacher (gram.ema_teacher)
    or a separate frozen backbone initialized from an earlier bundle
    (gram.ckpt) and optionally refreshed from the EMA teacher.
    """

    def __init__(self, vit_name="vit_small_patch16_224",
                 drop_path_rate=0.1, in_chans=3, pretrained=False,
                 head_cfg=None, gram_cfg=None, mask_cfg=None, loss_cfg=None):
        super().__init__()
        head_cfg = head_cfg or {}
        gram_cfg = dict(gram_cfg or {})
        mask_cfg = mask_cfg or {}
        loss_cfg = loss_cfg or {}

        vit_teacher = build_dino_vit(vit_name, in_chans, pretrained)
        input_dim = vit_teacher.embed_dim

        # Teacher/student backbones (same wrappers + key prefixes for every
        # bundle, so extraction and downstream consumers are shared). The
        # wrapper type follows the timm backbone: classic VisionTransformer
        # -> lightly MaskedVisionTransformerTIMM; timm DINOv3 (Eva/RoPE) ->
        # _MaskedEva.
        weight_init = "skip" if pretrained else ""
        self.teacher_backbone = _build_masked_vit(vit_teacher, weight_init)
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        _apply_drop_path(self.student_backbone.vit, drop_path_rate)
        _freeze_eval_module(self.teacher_backbone)

        # Heads (DINOv3 keeps dino and ibot heads separate).
        self._head_input_dim = input_dim
        self._head_cfg = head_cfg
        dino_head_fn = partial_head(input_dim, unbox_head_cfg(head_cfg, "dino"))
        ibot_head_fn = partial_head(input_dim, unbox_head_cfg(head_cfg, "ibot"))
        student_dino_head = dino_head_fn()
        student_ibot_head = ibot_head_fn()
        self.student_head = DINOv3HeadPair(student_dino_head, student_ibot_head)
        # Teacher heads start as a copy of the student heads (teacher EMA
        # begins at the student, original DINO design).
        teacher_dino_head = dino_head_fn()
        teacher_dino_head.load_state_dict(student_dino_head.state_dict())
        teacher_ibot_head = ibot_head_fn()
        teacher_ibot_head.load_state_dict(student_ibot_head.state_dict())
        self.teacher_head = DINOv3HeadPair(teacher_dino_head, teacher_ibot_head)
        _freeze_eval_module(self.teacher_head)

        # ---- loss weighting (config) ----
        self._dino_loss_weight = float(loss_cfg.get("dino_loss_weight", 1.0))
        self._ibot_loss_weight = float(loss_cfg.get("ibot_loss_weight", 1.0))
        self._koleo_loss_weight = float(loss_cfg.get("koleo_loss_weight", 0.1))
        self._global_ignore_diagonal = bool(loss_cfg.get("global_ignore_diagonal", True))
        self._sinkhorn_n_iterations = int(loss_cfg.get("sinkhorn_n_iterations", 3))
        self._student_temp = float(loss_cfg.get("student_temp", 0.1))
        self._reweight_dino_local = bool(loss_cfg.get("reweight_dino_local_loss", False))
        self._local_loss_weight_schedule = loss_cfg.get("local_loss_weight_schedule")

        # ---- masking (config) ----
        self._mask_ratio_min = float(mask_cfg.get("mask_ratio_min", 0.1))
        self._mask_ratio_max = float(mask_cfg.get("mask_ratio_max", 0.5))
        self._mask_sample_probability = float(mask_cfg.get("mask_sample_probability", 0.5))

        # ---- freeze last layer (anti-collapse, DINO-style) ----
        self._freeze_last_layer_epochs = int(loss_cfg.get("freeze_last_layer_epochs", 1))

        # ---- Gram anchoring ----
        self.gram_use_loss = bool(gram_cfg.get("use_loss", False))
        self._gram_img_level = bool(gram_cfg.get("img_level", True))
        self._gram_tokens_used = gram_cfg.get("tokens_used", "all")
        self._gram_loss_weight = float(gram_cfg.get("loss_weight", 1.0))
        self._gram_loss_weight_schedule = gram_cfg.get("loss_weight_schedule")
        self.gram_backbone = None
        self._gram_teacher_initialized = False
        self._num_gram_updates = 0
        self._n_gram_views = 0
        if self.gram_use_loss:
            gram_ema_teacher = bool(gram_cfg.get("ema_teacher", False))
            self._gram_ema_teacher = gram_ema_teacher
            self._gram_rep_update = bool(gram_cfg.get("rep_update", True))
            self._gram_update_frequency = int(gram_cfg.get("update_frequency", 50000))
            self._gram_it_first_update = int(gram_cfg.get("it_first_update", 0))
            self._gram_max_updates = gram_cfg.get("max_updates")
            self._gram_it_load_ema_teacher = int(gram_cfg.get("it_load_ema_teacher", -1))
            self._gram_ckpt = gram_cfg.get("ckpt")
            if self._gram_ckpt == "ignore":
                self._gram_ckpt = None
            self._gram_teacher_resize_method = gram_cfg.get(
                "global_teacher_resize_method", "bicubic")
            self._gram_teacher_resize_antialias = bool(
                gram_cfg.get("global_teacher_resize_antialias", False))
            gram_crops_size = gram_cfg.get("crops_size")

            if gram_ema_teacher and self._gram_ckpt is not None:
                raise ValueError(
                    "dinov3.gram: ema_teacher and ckpt are mutually exclusive; "
                    "set one of them.")
            if self._gram_ckpt is None and self._gram_it_load_ema_teacher < 0:
                raise ValueError(
                    "dinov3.gram: no gram.ckpt provided, so gram.it_load_ema_teacher "
                    "must be set to a non-negative iteration.")
            if gram_ema_teacher and self._gram_rep_update:
                raise ValueError("dinov3.gram: rep_update requires a ckpt-based "
                                 "gram teacher (ema_teacher must be false).")
            if self._gram_tokens_used not in ("all", "masked", "unmasked"):
                raise ValueError("dinov3.gram.tokens_used must be one of "
                                 "all | masked | unmasked.")
            if self._gram_tokens_used in ("masked", "unmasked"):
                if self._gram_img_level:
                    raise ValueError("dinov3.gram: tokens_used=masked/unmasked "
                                     "requires img_level=false.")
            if not gram_ema_teacher:
                if gram_crops_size is None:
                    raise ValueError(
                        "dinov3.gram.crops_size must be set when using a ckpt-based "
                        "gram teacher (the augmentation_views list must end with "
                        "2 high-res gram-teacher crop views).")
                # Frozen gram teacher backbone (same architecture + wrapper type as
                # the teacher/student; no heads attached).
                vit_gram = build_dino_vit(vit_name, in_chans, pretrained=False)
                self.gram_backbone = _build_masked_vit(vit_gram, weight_init="skip")
                _freeze_eval_module(self.gram_backbone)
                self._n_gram_views = 2
            else:
                self._n_gram_views = 0
            logger.info("GRAM enabled: ema_teacher=%s, ckpt=%s, "
                        "it_load_ema_teacher=%d, rep_update=%s (freq=%d, "
                        "first=%d, max=%s), tokens_used=%s, img_level=%s, "
                        "weight=%s, crops_size=%s",
                        gram_ema_teacher, self._gram_ckpt,
                        self._gram_it_load_ema_teacher, self._gram_rep_update,
                        self._gram_update_frequency, self._gram_it_first_update,
                        self._gram_max_updates, self._gram_tokens_used,
                        self._gram_img_level, self._gram_loss_weight, gram_crops_size)

        # Monitoring hooks: when enabled, train_step fills self._last_diag
        # with detached diagnostic tensors (head logits, CLS features, gram
        # patches) for monitor.py — zero overhead when disabled.
        self._monitoring_enabled = False
        self._last_diag = None

    def load_gram_from_bundle(self, path):
        """Initialize the frozen gram-teacher backbone from an SSL bundle."""
        if not self.gram_use_loss or self.gram_backbone is None:
            return
        if not path or not os.path.exists(path):
            raise MicroMaxError(f"Error: dinov3.gram.ckpt not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict")
        if not sd:
            raise MicroMaxError(f"Error: {path} is not an SSL bundle (no 'state_dict')")
        prefix = "teacher_backbone.vit."
        vt = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not vt:
            raise MicroMaxError(f"Error: bundle {path} has no '{prefix}' weights "
                  f"(not a DINOv3 SSL bundle)")
        self.gram_backbone.vit.load_state_dict(vt)
        self._gram_teacher_initialized = True
        _freeze_eval_module(self.gram_backbone)
        logger.info("Gram teacher initialized from %s (backbone %d keys)",
                    path, len(vt))

    def gram_load_ema_teacher(self):
        """Copy the EMA teacher backbone into the gram teacher (official
        gram_load_ema_teacher; heads are not copied)."""
        if self.gram_backbone is None:
            return
        with torch.no_grad():
            for gt, tt in zip(self.gram_backbone.parameters(),
                              self.teacher_backbone.parameters()):
                gt.copy_(tt)
            for gb, tb in zip(self.gram_backbone.buffers(),
                              self.teacher_backbone.buffers()):
                if gb.dtype.is_floating_point:
                    gb.copy_(tb)
        self._gram_teacher_initialized = True
        logger.info("Gram teacher loaded from EMA teacher")

    def update_gram(self, m=0.0):
        """Refresh the gram teacher from the EMA teacher (official update_gram)."""
        if self.gram_backbone is None:
            return
        with torch.no_grad():
            for gp, tp in zip(self.gram_backbone.parameters(),
                              self.teacher_backbone.parameters()):
                gp.mul_(m).add_(tp, alpha=1 - m)
            for gb, tb in zip(self.gram_backbone.buffers(),
                              self.teacher_backbone.buffers()):
                if gb.dtype.is_floating_point:
                    gb.mul_(m).add_(tb, alpha=1 - m)
        logger.info("Gram teacher updated from EMA teacher (m=%.3f)", m)

    def maybe_update_gram(self, global_step):
        """Drive the gram-teacher lifecycle at the given global step."""
        if not self.gram_use_loss or self.gram_backbone is None:
            return
        if self._gram_it_load_ema_teacher >= 0 and global_step == self._gram_it_load_ema_teacher:
            self.gram_load_ema_teacher()
        if (self._gram_rep_update
                and global_step + 1 >= self._gram_it_first_update
                and (global_step + 1) % self._gram_update_frequency == 0
                and (self._gram_max_updates is None
                     or self._num_gram_updates < self._gram_max_updates)):
            self.update_gram(m=0.0)
            self._num_gram_updates += 1

    def set_gram_resume_updates(self, start_iter):
        """Restore the number of gram updates already done when resuming.

        maybe_update_gram fires when (global_step + 1) is a positive multiple
        of the update frequency and >= the first-update step. After completing
        steps 0..start_iter-1 that is exactly the multiples m of frequency
        with m >= first and m <= start_iter (floor accounting, not ceil).
        """
        if not self.gram_use_loss or self.gram_backbone is None:
            return
        if start_iter > 0 and start_iter >= self._gram_it_first_update:
            freq = self._gram_update_frequency
            # First trigger point: the smallest POSITIVE multiple of freq that
            # is >= first (step counting starts at global_step=0, so m=0 is
            # never a trigger).
            m0 = max(freq, -(-self._gram_it_first_update // freq) * freq)
            if start_iter >= m0:
                self._num_gram_updates = (start_iter - m0) // freq + 1

    def forward_teacher(self, x):
        """Teacher forward (unmasked). Returns (cls_tokens, features)."""
        features = self.teacher_backbone.encode(x)
        return features[:, 0], features

    def forward_student(self, x, mask=None):
        """Student forward. Returns (cls_tokens, features, masked_features_or_None).

        mask: (B, seq_len) bool. When given, masked_features contains the
        features at masked positions.
        """
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, features, masked_features

    def train(self, mode=True):
        super().train(mode)
        self.teacher_backbone.eval()
        self.teacher_head.eval()
        if self.gram_backbone is not None:
            self.gram_backbone.eval()
        return self


def _freeze_eval_module(module: Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def unbox_head_cfg(head_cfg, key):
    """Head hyperparameters (hidden/bottleneck/nlayers/out_dim) for one of
    the 'dino'/'ibot' heads.

    Per-key resolution, most specific wins: a nested '<key>:' block (e.g.
    dinov3.ibot.{...}), then the flat '<key>_head_*' keys (e.g.
    ibot_head_n_prototypes), then the shared 'head_*' keys (the dino head's
    values), then the built-in default.
    """
    nested = head_cfg.get(key)
    if not isinstance(nested, dict):
        nested = {}
    vals = {}
    for name, cfg_key, default in (
            ("hidden_dim", "head_hidden_dim", 2048),
            ("bottleneck_dim", "head_bottleneck_dim", 256),
            ("nlayers", "head_nlayers", 3),
            ("out_dim", "head_n_prototypes", 65536)):
        vals[name] = nested.get(
            name,
            head_cfg.get(f"{key}_{cfg_key}",
                         head_cfg.get(cfg_key, default)))
    return {"hidden_dim": int(vals["hidden_dim"]),
            "bottleneck_dim": int(vals["bottleneck_dim"]),
            "nlayers": int(vals["nlayers"]),
            "out_dim": int(vals["out_dim"])}


def partial_head(in_dim, cfg):
    return lambda: DINOv3Head(
        in_dim, cfg["out_dim"], hidden_dim=cfg["hidden_dim"],
        bottleneck_dim=cfg["bottleneck_dim"], nlayers=cfg["nlayers"])


def _schedule_value(step, total_steps, per_epoch, sched):
    """linear warmup -> cosine (or constant) schedule, DINOv3-style."""
    warmup_iters = per_epoch * int(sched.get("warmup_epochs", 0))
    start = float(sched.get("start", 0.0))
    peak = float(sched.get("peak", start))
    end = float(sched.get("end", start))
    cosine_ep = sched.get("cosine_epochs")
    if step < warmup_iters:
        return linear_warmup_schedule(
            step=step, warmup_steps=max(warmup_iters, 1), start_value=start, end_value=peak)
    if cosine_ep is not None:
        total_cos = max(1, per_epoch * int(cosine_ep))
    else:
        total_cos = max(1, total_steps - warmup_iters)
    return cosine_schedule(step=step - warmup_iters, max_steps=total_cos,
                           start_value=peak, end_value=end)


def build_dinov3(backbone_cfg, method_cfg, device):
    """Build a DINOv3 model from config.

    backbone_cfg: {name (any timm ViT with a class token), in_chans, pretrained}.
    method_cfg: {drop_path_rate, head_*, dino/ibot/koleo weights,
                 mask_*, freeze_last_layer_epochs, gram: {...}, schedules ...}
    """
    vit_name = backbone_cfg.get("name", "vit_small_patch16_224")
    in_chans = backbone_cfg["in_chans"]
    pretrained = backbone_cfg.get("pretrained", False)
    model = DINOv3(
        vit_name=vit_name,
        drop_path_rate=method_cfg.get("drop_path_rate", 0.1),
        in_chans=in_chans,
        pretrained=pretrained,
        head_cfg=method_cfg,
        gram_cfg=method_cfg.get("gram"),
        mask_cfg=method_cfg,
        loss_cfg=method_cfg,
    )
    model.to(device)
    model._monitoring_enabled = bool(
        (method_cfg.get("monitoring") or {}).get("enabled", True))
    model._last_diag = None
    logger.info("DINOv3 model built: vit=%s, in_chans=%d, embed_dim=%d, "
                "pretrained=%s, prototypes=%s, gram=%s, monitoring=%s",
                vit_name, in_chans, model._head_input_dim, pretrained,
                model.student_head.dino_head.last_layer.out_features,
                model.gram_use_loss, model._monitoring_enabled)
    return model


def train_step(model, batch, optimizer, epoch, total_epochs, device, criterion,
               step_info, scaler=None, grad_clip=None, step=True):
    """One DINOv3 training step. Returns (loss_value, components_dict) — the
    uniform protocol consumed by run_pretrain (components: dino/ibot/koleo,
    plus gram when the Gram anchoring loss is enabled).

    criterion = (dino_criterion, ibot_criterion, koleo_criterion,
                 gram_criterion_or_None)
    step_info = {global_step, total_steps, warmup_steps, lr_peak, lr_final,
                 momentum_start, momentum_end, koleo_weight,
                 teacher_temp_start, teacher_temp_end,
                 weight_decay_start, weight_decay_end}
    Views convention (see config):
      - views[0:2] = global views (2 required; teacher + masked student)
      - views[2:2+n_local] = local views (student, unmasked)
      - views[2+n_local:] = gram-teacher crops (only when a ckpt-based gram
        teacher is used; exactly 2 views)
    """
    dino_criterion, ibot_criterion, koleo_criterion, gram_criterion = criterion
    views = batch
    n_gram = getattr(model, "_n_gram_views", 0)
    n_local = len(views) - 2 - n_gram
    if n_local < 1 or 2 + n_local + n_gram != len(views):
        raise MicroMaxError(f"Error: DINOv3 requires 2 global + >=1 local views"
            f"{' + 2 gram-teacher crops' if n_gram else ''} "
            f"(total {2 + n_gram} + n_local), got {len(views)}")

    global_views = torch.cat(views[:2]).to(device)
    local_views = torch.cat(views[2:2 + n_local]).to(device)
    gram_views = (torch.cat(views[2 + n_local:]).to(device) if n_gram else None)
    B = len(global_views)  # 2 * batch_size
    bs = B // 2          # batch_size (samples per view)

    # ---- patch mask for the student's global views (DINOv3 blocks) ----
    _, _, in_h, in_w = global_views.shape
    patch = model.teacher_backbone.vit.patch_embed.patch_size
    patch_h, patch_w = (patch, patch) if isinstance(patch, int) else patch
    grid_h, grid_w = in_h // patch_h, in_w // patch_w
    mask_full, masks_weight = _make_masks(model, B, grid_h, grid_w,
                                          global_views.device)

    device_type = str(device).split(":")[0]

    # ---- teacher (unmasked global views, no grad) ----
    with torch.no_grad():
        teacher_cls_token, teacher_features = model.forward_teacher(global_views)
        with torch.amp.autocast(device_type=device_type, enabled=False):
            teacher_cls_out = model.teacher_head.dino_head.forward(teacher_cls_token)
            teacher_masked_out = model.teacher_head.ibot_head.forward(
                teacher_features[mask_full])

    # ---- student (masked global + unmasked local views) ----
    student_global_cls, student_global_features, student_global_masked = \
        model.forward_student(global_views, mask=mask_full)
    student_local_cls, _, _ = model.forward_student(local_views, mask=None)
    with torch.amp.autocast(device_type=device_type, enabled=False):
        student_global_cls_out = model.student_head.dino_head.forward(student_global_cls)
        student_global_masked_out = model.student_head.ibot_head.forward(
            student_global_masked)
        student_local_cls_out = model.student_head.dino_head.forward(student_local_cls)

    global_step = step_info["global_step"]
    teacher_temp = linear_warmup_schedule(
        step=global_step,
        warmup_steps=step_info["warmup_steps"],
        start_value=step_info.get("teacher_temp_start", 0.04),
        end_value=step_info.get("teacher_temp_end", 0.07),
    )

    # ---- losses (DINOv3 recipe, weights/scales from the official code) ----
    teacher_cls_centered = dino_criterion.sinkhorn_knopp_teacher(
        teacher_cls_out, teacher_temp,
        n_iterations=model._sinkhorn_n_iterations).unflatten(0, (2, bs))
    n_masked = mask_full.sum()
    if n_masked > 0:
        teacher_masked_centered = ibot_criterion.sinkhorn_knopp_teacher(
            teacher_masked_out, teacher_temp, n_masked_patches=n_masked,
            n_iterations=model._sinkhorn_n_iterations)
    else:
        teacher_masked_centered = None

    n_g, n_l = 2, n_local
    dg_terms = n_g * (n_g - 1) if model._global_ignore_diagonal else n_g * n_g
    dl_terms = n_g * n_l
    dg_scale = dg_terms / (dg_terms + dl_terms)
    dl_scale = dl_terms / (dg_terms + dl_terms)

    dino_global = dino_criterion(
        student_global_cls_out.unflatten(0, (2, bs)),
        teacher_cls_centered,
        ignore_diagonal=model._global_ignore_diagonal,
    )
    dino_local = dino_criterion(
        student_local_cls_out.unflatten(0, (n_l, bs)),
        teacher_cls_centered,
    )
    if model._reweight_dino_local and model._local_loss_weight_schedule:
        per_epoch = max(1, step_info["total_steps"] // max(1, total_epochs))
        local_weight = _schedule_value(global_step, step_info["total_steps"],
                                       per_epoch, model._local_loss_weight_schedule)
    else:
        local_weight = 1.0
    koleo = sum(koleo_criterion(t) for t in student_global_cls.chunk(2)) / n_g

    if n_masked > 0:
        ibot = ibot_criterion.forward_masked(
            student_global_masked_out, teacher_masked_centered,
            masks_weight=masks_weight)
    else:
        ibot = torch.zeros((), device=device)

    gram_loss = None
    gram_weight = 0.0
    if model.gram_use_loss and gram_criterion is not None:
        n_prefix = model.teacher_backbone.vit.num_prefix_tokens
        student_patches = student_global_features[:, n_prefix:]
        if model._gram_ema_teacher:
            teacher_patches = teacher_features[:, n_prefix:]
        else:
            with torch.no_grad():
                gram_features = model.gram_backbone.encode(gram_views)
            teacher_patches = gram_features[:, n_prefix:]
            # Resample gram-teacher features to the student patch grid when
            # the gram-teacher crops render at a different resolution.
            if teacher_patches.shape[1] != student_patches.shape[1]:
                _, _, g_h, g_w = gram_views.shape
                N = (g_h // patch_h) * (g_w // patch_w)
                N_student = grid_h * grid_w
                if teacher_patches.shape[1] != N or student_patches.shape[1] != N_student:
                    raise MicroMaxError(f"Error: gram-teacher patch count {teacher_patches.shape[1]} "
                        f"!= ({g_h}//{patch_h})^2 and student {student_patches.shape[1]} "
                        f"!= ({in_h}//{patch_h})^2")
                S = int(round(math.sqrt(N)))
                S_student = int(round(math.sqrt(N_student)))
                patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (S, S))
                patches_hw = F.interpolate(
                    patches_hw, size=(S_student, S_student),
                    mode=model._gram_teacher_resize_method,
                    align_corners=False,
                    antialias=model._gram_teacher_resize_antialias)
                teacher_patches = patches_hw.flatten(-2, -1).transpose(-2, -1)
        if model._monitoring_enabled and model._gram_tokens_used == "all":
            # Full (pre-selection) patch tensors for the masked/unmasked
            # Gram split statistics; mask_patch covers patch tokens only.
            _diag_gram = (student_patches.detach(), teacher_patches.detach(),
                          mask_full[:, n_prefix:])
        else:
            _diag_gram = None
        if model._gram_tokens_used == "masked":
            student_patches = student_patches[mask_full]
            teacher_patches = teacher_patches[mask_full]
        elif model._gram_tokens_used == "unmasked":
            student_patches = student_patches[~mask_full]
            teacher_patches = teacher_patches[~mask_full]
        gram_loss = gram_criterion(student_patches, teacher_patches,
                                   img_level=model._gram_img_level)
        if model._gram_loss_weight_schedule:
            per_epoch = max(1, step_info["total_steps"] // max(1, total_epochs))
            gram_weight = _schedule_value(global_step, step_info["total_steps"],
                                          per_epoch, model._gram_loss_weight_schedule)
        else:
            gram_weight = model._gram_loss_weight

    loss = (model._dino_loss_weight * dg_scale * dino_global
            + model._dino_loss_weight * dl_scale * local_weight * dino_local
            + model._koleo_loss_weight * n_g * koleo
            + model._ibot_loss_weight * ibot)
    if gram_loss is not None:
        loss = loss + gram_weight * gram_loss
    if not torch.isfinite(loss).item():
        raise MicroMaxError(f"Error: non-finite DINOv3 loss ({loss.item()}) at global step "
              f"{global_step}")

    # Monitoring hook: detached diagnostic tensors for monitor.py (no extra
    # forward passes; all tensors are already in scope).
    if model._monitoring_enabled:
        model._last_diag = {
            "head_logits": student_global_cls_out.detach(),   # (2B, K)
            "student_cls": student_global_cls.detach(),       # pre-head
            "teacher_cls": teacher_cls_token.detach(),
        }
        if model.gram_use_loss and _diag_gram is not None:
            model._last_diag["gram"] = _diag_gram
    else:
        model._last_diag = None

    # Freeze the heads' last layer during the warm-up epochs (anti-collapse).
    freeze = (model._freeze_last_layer_epochs > 0 and epoch < model._freeze_last_layer_epochs)

    if step:
        weight_decay = cosine_schedule(
            step=global_step, max_steps=step_info["total_steps"],
            start_value=step_info.get("weight_decay_start", 0.04),
            end_value=step_info.get("weight_decay_end", 0.4),
        )
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                group["weight_decay"] = weight_decay

        lr_peak = step_info.get("lr_peak", 0.0005)
        lr_final = step_info.get("lr_final", 1e-6)
        lr_warmup_steps = step_info["warmup_steps"]
        if lr_warmup_steps > 0 and global_step < lr_warmup_steps:
            lr = linear_warmup_schedule(
                step=global_step, warmup_steps=lr_warmup_steps,
                start_value=0.0, end_value=lr_peak)
        else:
            lr = cosine_schedule(
                step=global_step - lr_warmup_steps,
                max_steps=max(1, step_info["total_steps"] - lr_warmup_steps),
                start_value=lr_peak, end_value=lr_final)
        # Per-group LR scale: reserved for fine-tune stages that give the
        # encoder a different LR than the heads (pretrain builds lr_scale=1
        # groups, so dinov3 always uses the plain schedule value).
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)

        if scaler is not None:
            scaler.scale(loss).backward()
            if freeze:
                _zero_last_layer_grads(model)
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if freeze:
                _zero_last_layer_grads(model)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad()

        momentum = cosine_schedule(
            step=global_step, max_steps=step_info["total_steps"],
            start_value=step_info.get("momentum_start", 0.992),
            end_value=step_info.get("momentum_end", 1.0),
        )
        _update_momentum_with_buffers(model.student_backbone, model.teacher_backbone, m=momentum)
        _update_momentum_with_buffers(model.student_head, model.teacher_head, m=momentum)

        # Gram-teacher lifecycle (it_load_ema_teacher / rep_update).
        model.maybe_update_gram(global_step)

        if model._monitoring_enabled and model._last_diag is not None:
            model._last_diag.update({
                "lr": float(lr),
                "momentum": float(momentum),
                "weight_decay": float(weight_decay),
                "teacher_temp": float(teacher_temp),
            })
    else:
        if scaler is not None:
            scaler.scale(loss).backward()
            if freeze:
                _zero_last_layer_grads(model)
        else:
            loss.backward()
            if freeze:
                _zero_last_layer_grads(model)

    components = {"dino": dino_global.item(), "ibot": ibot.item(),
                  "koleo": koleo.item()}
    if gram_loss is not None:
        components["gram"] = gram_loss.item()
    return loss.item(), components


def _zero_last_layer_grads(model):
    """Drop gradients of the student heads' last layer (DINO freeze trick)."""
    for head in (model.student_head.dino_head, model.student_head.ibot_head):
        if head.last_layer is not None:
            for p in head.last_layer.parameters():
                p.grad = None