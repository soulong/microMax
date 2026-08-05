"""DINOv2 SSL model.

Uses lightly for MaskedVisionTransformerTIMM, DINOv2ProjectionHead, DINOLoss,
IBOTPatchLoss, KoLeoLoss, random_block_mask, update_drop_path_rate. Teacher
momentum uses the local _update_momentum_with_buffers (lightly's
update_momentum only EMAs parameters; ours also syncs BN buffers for
head_batch_norm=true). The ViT backbone is rebuilt via timm directly (not via
build_backbone) because ViT requires special init args (pos_embed="learn",
dynamic_img_size=True, init_values=1e-5); build_dinov2_vit in backbone.py
owns that and accepts any timm ViT with a class token. The input channel
count is derived from data.channels by pretrain_ssl (not a config key).
backbone.pretrained is honored: pretrained=True loads ImageNet weights into
both teacher and student (weight_initialization="skip" keeps them — lightly's
default MAE-style init would otherwise overwrite the loaded weights).

Multi-view convention:
  - views[0:2] are global views (2 required).
  - views[2:] are local views (any count: 1, 2, 4, 8, ...).
  - Minimum 3 views total (2 global + 1 local).

References:
  - https://docs.lightly.ai/self-supervised-learning/examples/dinov2.html
"""

import copy
import sys
from functools import partial

import torch
from torch import nn
from torch.nn import Module

from lightly.models.modules import DINOv2ProjectionHead, MaskedVisionTransformerTIMM
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.utils import random_block_mask, update_drop_path_rate
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule

from ..backbone import build_dinov2_vit
from ..utils import logger

# Collapse guard: the DINO/iBOT softmax cross-entropies vanish when the
# projection head outputs collapse to a constant (uniform softmax over the
# 65536-dim code space) — the loss then locks at the uniform-CE floor forever.
# The guard tracks the student head's logit spread and re-initializes the
# heads when it collapses (see _maybe_reinit_heads / DINOv2.reinit_heads).
# Threshold in logit space: healthy heads sit at std ~0.02-0.06, collapsed
# heads at ~0.002 — 5e-3 fires while there is still gradient signal left.
_LOGITS_STD_THRESHOLD = 5e-3
_REINIT_COOLDOWN = 200


def _update_momentum_with_buffers(model, model_ema, m):
    """EMA of parameters AND float buffers (lightly's update_momentum only
    does parameters).

    Buffers matter for BatchNorm in the projection heads (head_batch_norm=true):
    the teacher's BN running stats must track the student's, otherwise the
    teacher targets are computed with stale statistics. Non-float buffers
    (e.g. BN's num_batches_tracked) are left untouched.
    """
    for ema_param, param in zip(model_ema.parameters(), model.parameters()):
        ema_param.data.mul_(m).add_(param.data, alpha=1 - m)
    for ema_buffer, buffer in zip(model_ema.buffers(), model.buffers()):
        if ema_buffer.dtype.is_floating_point:
            ema_buffer.data.mul_(m).add_(buffer.data, alpha=1 - m)


def _maybe_reinit_heads(model):
    """Re-initialize the projection heads if the student logits collapsed.

    Returns True when heads were re-initialized (caller may log it).
    """
    cooldown = getattr(model, "_reinit_cooldown", 0)
    if cooldown > 0:
        model._reinit_cooldown = cooldown - 1
        return False
    ema = getattr(model, "_logits_std_ema", None)
    if ema is None or ema >= _LOGITS_STD_THRESHOLD:
        return False
    logger.warning(
        "Student head logits collapsed (EMA std=%.4g < %g); "
        "re-initializing projection heads",
        ema, _LOGITS_STD_THRESHOLD)
    model.reinit_heads()
    model._logits_std_ema = None
    model._reinit_cooldown = _REINIT_COOLDOWN
    return True


def _freeze_eval_module(module: Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2Head(Module):
    """Wrapper holding dino (cls-token) and ibot (patch-token) projection heads."""

    def __init__(self, dino_head, ibot_head):
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2(nn.Module):
    """DINOv2 model: teacher + student ViT backbones, each with a DINOv2Head.

    Teacher is frozen (momentum-updated). Student receives patch masking on
    global views.

    input_dim=None derives the projection-head input from the ViT's embed_dim;
    a non-None value must match it. in_chans/pretrained are passed through to
    the timm ViT factory (in_chans comes from data.channels via pretrain_ssl).
    """

    def __init__(self, vit_name="vit_small_patch16_224", input_dim=None,
                 drop_path_rate=0.1, ibot_separate_head=False,
                 in_chans=3, pretrained=False, head_batch_norm=True):
        super().__init__()
        vit_teacher = build_dinov2_vit(vit_name, in_chans, pretrained)
        if input_dim is not None and input_dim != vit_teacher.embed_dim:
            print(
                f"Error: dinov2.input_dim={input_dim} does not match ViT "
                f"'{vit_name}' embed_dim={vit_teacher.embed_dim}; set it to "
                f"{vit_teacher.embed_dim} or remove it (null = derived)",
                file=sys.stderr,
            )
            sys.exit(1)
        input_dim = vit_teacher.embed_dim

        # Teacher: ViT wrapped in MaskedVisionTransformerTIMM for patch masking.
        # weight_initialization="skip" preserves the timm-loaded weights
        # (pretrained=True); the default lightly MAE-style init would overwrite
        # them. pos_embed_initialization="skip" keeps the loaded pos_embed.
        weight_init = "skip" if pretrained else ""
        self.teacher_backbone = MaskedVisionTransformerTIMM(
            vit=vit_teacher, antialias=False, pos_embed_initialization="skip",
            weight_initialization=weight_init)
        # Student: deepcopy of teacher, with drop_path_rate applied
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate(self.student_backbone.vit, drop_path_rate, mode="uniform")
        _freeze_eval_module(self.teacher_backbone)

        # Head config — kept for reinit_heads() (collapse guard).
        self._head_input_dim = input_dim
        self._head_batch_norm = head_batch_norm
        self._ibot_separate_head = ibot_separate_head

        # Heads: dino (cls) + ibot (patch). If ibot_separate_head=False, share the head.
        # head_batch_norm=true adds BatchNorm inside the heads (original DINO
        # design) — it normalizes the head output scale and is a critical
        # anti-collapse mechanism: without it the 65536-dim head outputs
        # collapse to a constant (uniform softmax) and the loss locks at the
        # uniform-CE floor forever.
        dino_head_fn = partial(DINOv2ProjectionHead, input_dim=input_dim,
                               batch_norm=head_batch_norm)
        teacher_dino_head = dino_head_fn()
        student_dino_head = dino_head_fn()
        if ibot_separate_head:
            teacher_ibot_head = dino_head_fn()
            student_ibot_head = dino_head_fn()
        else:
            teacher_ibot_head = teacher_dino_head
            student_ibot_head = student_dino_head
        self.teacher_head = DINOv2Head(teacher_dino_head, teacher_ibot_head)
        self.student_head = DINOv2Head(student_dino_head, student_ibot_head)
        _freeze_eval_module(self.teacher_head)

    def reinit_heads(self):
        """Re-initialize the projection heads (student fresh, teacher copy).

        Used by the collapse guard: when the student head's outputs collapse to
        a constant (uniform softmax), the DINO/iBOT gradients vanish and the
        loss locks at the uniform-CE floor forever. Fresh heads restore the
        gradient signal; the teacher restarts as a copy of the fresh student
        (original DINO: "teacher and student start with the same weights").
        """
        dino_head_fn = partial(DINOv2ProjectionHead, input_dim=self._head_input_dim,
                               batch_norm=self._head_batch_norm)
        device = next(self.student_backbone.parameters()).device
        student_dino_head = dino_head_fn().to(device)
        student_ibot_head = dino_head_fn().to(device) if self._ibot_separate_head \
            else student_dino_head
        self.student_head = DINOv2Head(student_dino_head, student_ibot_head)
        # Teacher starts as a copy of the fresh student (original DINO:
        # "teacher and student start with the same weights"), then diverges
        # via the momentum update. A state-dict copy is used instead of
        # copy.deepcopy — DINOv2ProjectionHead uses torch.nn.utils.weight_norm,
        # which copy.deepcopy cannot handle in torch >= 2.13.
        teacher_dino_head = dino_head_fn().to(device)
        teacher_dino_head.load_state_dict(student_dino_head.state_dict())
        if self._ibot_separate_head:
            teacher_ibot_head = dino_head_fn().to(device)
            teacher_ibot_head.load_state_dict(student_ibot_head.state_dict())
        else:
            teacher_ibot_head = teacher_dino_head
        self.teacher_head = DINOv2Head(teacher_dino_head, teacher_ibot_head)
        _freeze_eval_module(self.teacher_head)

    def forward_teacher(self, x):
        """Teacher forward on global views. Returns (cls_tokens, features)."""
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(self, x, mask=None):
        """Student forward. Returns (cls_tokens, masked_features_or_None).

        mask: (B, seq_len) bool — patch mask (True = masked). When given,
        forward_student also returns the features at masked positions.
        """
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features


def build_dinov2(backbone_cfg, method_cfg, device):
    """Build a DINOv2 model from config.

    backbone_cfg: {name (any timm ViT with a class token), pretrained} —
    in_chans is injected by pretrain_ssl from the resolved data.channels (not
    a config key).
    method_cfg: {input_dim (null = derived from ViT), drop_path_rate,
                 ibot_separate_head, head_batch_norm, ...}
    """
    vit_name = backbone_cfg.get("name", "vit_small_patch16_224")
    in_chans = backbone_cfg["in_chans"]
    pretrained = backbone_cfg.get("pretrained", False)
    model = DINOv2(
        vit_name=vit_name,
        input_dim=method_cfg.get("input_dim"),
        drop_path_rate=method_cfg.get("drop_path_rate", 0.1),
        ibot_separate_head=method_cfg.get("ibot_separate_head", False),
        in_chans=in_chans,
        pretrained=pretrained,
        head_batch_norm=method_cfg.get("head_batch_norm", True),
    )
    model.to(device)
    logger.info("DINOv2 model built: vit=%s, in_chans=%d, input_dim=%d, pretrained=%s, "
                "head_batch_norm=%s",
                vit_name, in_chans, model.student_head.dino_head.layers[0].in_features,
                pretrained, model._head_batch_norm)
    return model


def train_step(model, batch, optimizer, epoch, total_epochs, device, criterion,
               step_info, scaler=None, grad_clip=None, step=True):
    """One DINOv2 training step. Returns (loss, dino_loss, ibot_loss, koleo_loss)
    floats (the components are used for per-epoch logging by pretrain_ssl).

    criterion = (dino_criterion, ibot_criterion, koleo_criterion)
    step_info = {global_step, total_steps, warmup_steps, koleo_weight}
    scaler: optional torch.amp.GradScaler for AMP (scale loss, step, update).
    grad_clip: optional global gradient norm clip (None = no clipping).
    step: False accumulates gradients (backward only) for gradient
        accumulation; optimizer.step/zero_grad/clip/update, the weight-decay
        schedule and the teacher momentum update happen only when step=True.

    Views convention:
      - batch[0:2] = global views (2 required)
      - batch[2:] = local views (any count >= 1)
    """
    dino_criterion, ibot_criterion, koleo_criterion = criterion
    koleo_weight = step_info.get("koleo_weight", 0.1)

    views = batch
    if len(views) < 3:
        print(
            f"Error: DINOv2 requires at least 3 views (2 global + 1 local), "
            f"got {len(views)}", file=sys.stderr,
        )
        sys.exit(1)

    # Collapse guard: re-initialize the heads if the previous step's EMA of
    # the student logit spread says they collapsed (uniform softmax -> the
    # DINO/iBOT gradients vanish and the loss locks at the uniform-CE floor).
    if _maybe_reinit_heads(model):
        # reinit_heads replaced the head modules; the optimizer still owns the
        # OLD (orphaned) head params, so the fresh heads would never be
        # trained. Re-point the optimizer at the model's current parameters
        # (fresh heads get fresh AdamW state; hyperparams unchanged).
        # Preserve the decay/no-decay param group structure (group 0 = decay
        # ndim>=2, group 1 = no-decay ndim<2) created by pretrain_ssl.
        decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        optimizer.param_groups[0]["params"] = decay_params
        optimizer.param_groups[1]["params"] = no_decay_params

    global_views = torch.cat(views[:2]).to(device)
    local_views = torch.cat(views[2:]).to(device)

    B = len(global_views)
    # The mask grid must match the ACTUAL input size, not the ViT's default
    # img_size (e.g. the *_dinov2 timm variants default to 518, not 224) —
    # otherwise mask/features shapes mismatch. Compute from the input and the
    # patch size, mirroring timm's dynamic patch embed (floor division). The
    # masked region starts after ALL prefix tokens (cls + reg tokens).
    _, _, in_h, in_w = global_views.shape
    patch = model.teacher_backbone.vit.patch_embed.patch_size
    patch_h, patch_w = (patch, patch) if isinstance(patch, int) else patch
    grid_h, grid_w = in_h // patch_h, in_w // patch_w
    n_prefix = model.teacher_backbone.vit.num_prefix_tokens
    mask = global_views.new_zeros((B, grid_h * grid_w + n_prefix), dtype=torch.bool)
    block_mask = random_block_mask(size=(B, grid_h, grid_w), device=mask.device)
    mask[:, n_prefix:] = block_mask.flatten(start_dim=1)

    # The projection heads run in fp32 (autocast off) — fp16 matmuls in the
    # 65536-dim heads produced rare NaN on some GPUs even with finite
    # activations (same cuBLAS issue that motivated BYOL's fp32 heads); a
    # fresh head's weight distribution after reinit_heads can hit it.
    device_type = str(device).split(":")[0]
    with torch.no_grad():
        teacher_cls_token, teacher_features = model.forward_teacher(global_views)
        with torch.amp.autocast(device_type=device_type, enabled=False):
            teacher_cls_out = model.teacher_head.dino_head.forward(teacher_cls_token)
            teacher_masked_out = model.teacher_head.ibot_head.forward(teacher_features[mask])

    student_global_cls_token, student_global_masked_features = model.forward_student(
        global_views, mask=mask)
    student_local_cls_token, _ = model.forward_student(local_views, mask=None)
    with torch.amp.autocast(device_type=device_type, enabled=False):
        student_global_cls_out = model.student_head.dino_head.forward(student_global_cls_token)
        student_global_masked_out = model.student_head.ibot_head.forward(student_global_masked_features)
        student_local_cls_out = model.student_head.dino_head.forward(student_local_cls_token)
    student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

    # Collapse-guard EMA: track the student head's logit spread (std across
    # the 65536 code dims, per sample, averaged). A decaying spread toward
    # ~0 means the softmax is becoming uniform -> vanishing gradients.
    logits_std = float(torch.std(student_global_cls_out.detach(), dim=-1).mean())
    if getattr(model, "_logits_std_ema", None) is None:
        model._logits_std_ema = logits_std
    else:
        model._logits_std_ema = 0.95 * model._logits_std_ema + 0.05 * logits_std

    global_step = step_info["global_step"]
    teacher_temp = linear_warmup_schedule(
        step=global_step,
        warmup_steps=step_info["warmup_steps"],
        start_value=step_info.get("teacher_temp_start", 0.04),
        end_value=step_info.get("teacher_temp_end", 0.07),
    )
    dino_loss = dino_criterion(
        teacher_out=teacher_cls_out.chunk(2),
        student_out=student_cls_out.chunk(len(views)),
        teacher_temp=teacher_temp,
    )
    ibot_loss = ibot_criterion(
        teacher_out=teacher_masked_out,
        student_out=student_global_masked_out,
        mask=block_mask,
        teacher_temp=teacher_temp,
    )
    koleo_loss = koleo_weight * sum(
        koleo_criterion(t) for t in student_global_cls_token.chunk(2)
    )
    loss = dino_loss + ibot_loss + koleo_loss

    if step:
        # Weight decay schedule (param groups only; safe before step)
        weight_decay = cosine_schedule(
            step=global_step, max_steps=step_info["total_steps"],
            start_value=step_info.get("weight_decay_start", 0.04),
            end_value=step_info.get("weight_decay_end", 0.4),
        )
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                group["weight_decay"] = weight_decay

        # LR schedule: linear warmup (0 -> lr_peak over warmup_steps) then
        # cosine decay (lr_peak -> lr_final over remaining steps). Reuses
        # warmup_steps (same as teacher temperature warmup) — DINOv2 default.
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
        for group in optimizer.param_groups:
            group["lr"] = lr

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad()

        # Momentum update teacher (params AND buffers — buffers keep the BN
        # running stats of the teacher head in sync with the student)
        momentum = cosine_schedule(
            step=global_step, max_steps=step_info["total_steps"],
            start_value=step_info.get("momentum_start", 0.992),
            end_value=step_info.get("momentum_end", 1.0),
        )
        _update_momentum_with_buffers(model.student_backbone, model.teacher_backbone, m=momentum)
        _update_momentum_with_buffers(model.student_head, model.teacher_head, m=momentum)
    else:
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    return (loss.item(), dino_loss.item(), ibot_loss.item(), koleo_loss.item())
