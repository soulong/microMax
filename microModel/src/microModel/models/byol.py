"""BYOL (Bootstrap Your Own Latent) SSL model.

Uses lightly for projection/prediction heads, NegativeCosineSimilarity loss,
and momentum update utilities. Backbone is a generic timm conv backbone
via build_backbone.

References:
  - https://docs.lightly.ai/self-supervised-learning/examples/byol.html
"""

import copy
import sys
import torch
from torch import nn

from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule

from ..backbone import build_backbone
from ..utils import logger


class BYOL(nn.Module):
    """BYOL model: online backbone + projection + prediction, with momentum target."""

    def __init__(self, backbone, feat_dim,
                 proj_hidden=1024, proj_out=256,
                 pred_hidden=1024, pred_out=256):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(feat_dim, proj_hidden, proj_out)
        self.prediction_head = BYOLPredictionHead(proj_out, pred_hidden, pred_out)
        # Momentum (target) network — deepcopy + freeze
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        """Online forward: backbone -> flatten -> projection -> prediction -> p.

        The MLP heads run in fp32 (autocast disabled) — fp16 matmul
        accumulation in the heads produced rare NaN on some GPUs even with
        small, finite activations (cuBLAS reduced-precision reduction).
        """
        y = self.backbone(x).flatten(start_dim=1).float()
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            z = self.projection_head(y)
            p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        """Target forward: momentum backbone -> momentum projection -> z (detached)."""
        y = self.backbone_momentum(x).flatten(start_dim=1).float()
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            z = self.projection_head_momentum(y)
        return z.detach()


def build_byol(backbone_cfg, method_cfg, device):
    """Build a BYOL model from config.

    backbone_cfg: {name, pretrained} — in_chans is injected by pretrain_ssl
    from the resolved data.channels (not a config key).
    method_cfg: {proj_hidden_dim, proj_out_dim, pred_hidden_dim, pred_out_dim,
                 warmup_epochs, transfer_warmup_epochs, momentum_start,
                 momentum_end, lr_final}
    """
    backbone, feat_dim, _ = build_backbone(
        backbone_cfg["name"],
        backbone_cfg["in_chans"],
        backbone_cfg.get("pretrained", True),
    )
    model = BYOL(
        backbone, feat_dim,
        proj_hidden=method_cfg.get("proj_hidden_dim", 1024),
        proj_out=method_cfg.get("proj_out_dim", 256),
        pred_hidden=method_cfg.get("pred_hidden_dim", 1024),
        pred_out=method_cfg.get("pred_out_dim", 256),
    )
    model.to(device)
    logger.info("BYOL model built: backbone=%s, feat_dim=%d, proj_out=%d",
                backbone_cfg["name"], feat_dim, method_cfg.get("proj_out_dim", 256))
    return model


def train_step(model, batch, optimizer, epoch, total_epochs, device, criterion,
               step_info, scaler=None, grad_clip=None, step=True):
    """One BYOL training step. Returns loss value.

    BYOL uses exactly 2 views. batch = list of N view tensors (each (B, C, H, W));
    only the first 2 are used.

    criterion = NegativeCosineSimilarity()
    step_info: dict with global_step, total_steps, warmup_steps, lr_peak,
        lr_final, momentum_start, momentum_end.
    scaler: optional torch.amp.GradScaler for AMP (scale loss, step, update).
    grad_clip: optional global gradient norm clip (None = no clipping).
    step: False accumulates gradients (backward only) for gradient
        accumulation; optimizer.step/zero_grad/clip/update happen only when
        step=True (and at the end of an accumulation window). The momentum
        (target) EMA update also runs only when step=True, right after the
        optimizer step — the canonical BYOL/DINO timing (once per optimizer
        step, teacher lags the student).
    """
    global_step = step_info["global_step"]
    momentum_val = cosine_schedule(
        step=global_step,
        max_steps=step_info["total_steps"],
        start_value=step_info.get("momentum_start", 0.996),
        end_value=step_info.get("momentum_end", 1.0),
    )

    views = batch
    if len(views) < 2:
        print("Error: BYOL requires at least 2 views, got "
              f"{len(views)}", file=sys.stderr)
        sys.exit(1)

    x0 = views[0].to(device)
    x1 = views[1].to(device)
    p0 = model(x0)
    z0 = model.forward_momentum(x0)
    p1 = model(x1)
    z1 = model.forward_momentum(x1)
    loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))

    if scaler is not None:
        scaler.scale(loss).backward()
        if step:
            # LR schedule: linear warmup (0 -> lr_peak) then cosine decay
            lr_peak = step_info.get("lr_peak", 0.05)
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
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
    else:
        loss.backward()
        if step:
            # LR schedule: linear warmup (0 -> lr_peak) then cosine decay
            lr_peak = step_info.get("lr_peak", 0.05)
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
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    if step:
        optimizer.zero_grad()
        # Momentum EMA update for the target network — once per optimizer
        # step, after the student step (canonical BYOL/DINO timing). Not run
        # on accumulation micro-batches (step=False).
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)
    # Components are None for BYOL (no dino/ibot/koleo split); the 4-tuple
    # keeps pretrain_ssl's unpacking uniform across methods.
    return (loss.item(), None, None, None)
