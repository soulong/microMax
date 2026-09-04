"""Model definitions: FocalLoss, backbone builder, classification head, Model, bundle loader."""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import VisionTransformer

from .utils import logger


def build_dinov2_vit(vit_name, in_chans, pretrained=False):
    """Build a timm ViT usable for DINO-style SSL (cls-token pooling).

    Two timm model families are accepted:
      1. Classic timm `VisionTransformer` (DINOv2 checkpoints such as
         `vit_*_patch14_dinov2.*`): created with pos_embed="learn",
         dynamic_img_size=True, init_values=1e-5.
      2. timm DINOv3 backbones (`vit_*_patch16_dinov3*`, e.g.
         `vit_small_patch16_dinov3`): these are implemented on the Eva
         architecture (RoPE, register tokens, num_prefix_tokens=5) and do NOT
         accept the VisionTransformer-specific kwargs — they are created with
         a plain create_model call (dynamic_img_size is built-in).
    Anything without a patch_embed / blocks / a class token hard-exits
    (no silent fallback).

    pretrained=True loads timm weights (ImageNet/LVD); the first conv is
    adapted to in_chans via timm's adapt_input_conv.
    """
    try:
        vit = timm.create_model(
            vit_name, pretrained=pretrained, in_chans=in_chans, num_classes=0,
            pos_embed="learn", dynamic_img_size=True, init_values=1e-5)
    except TypeError:
        # timm DINOv3 (Eva-based) backbones reject the VisionTransformer
        # kwargs — rebuild without them.
        try:
            vit = timm.create_model(
                vit_name, pretrained=pretrained, in_chans=in_chans, num_classes=0)
        except Exception as e:
            print(f"Error: failed to create DINOv3 ViT '{vit_name}': {e}",
                  file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error: failed to create DINOv2 ViT '{vit_name}': {e}",
              file=sys.stderr)
        sys.exit(1)
    ok = (hasattr(vit, "patch_embed") and hasattr(vit, "blocks")
          and getattr(vit, "num_prefix_tokens", 0) >= 1
          and getattr(vit, "embed_dim", None) is not None)
    if not ok:
        print(
            f"Error: '{vit_name}' is not a DINO-style timm ViT (needs a "
            f"patch_embed, blocks and a class token); got {type(vit).__name__}.",
            file=sys.stderr,
        )
        sys.exit(1)
    return vit


def cls_token_pool_fn(x):
    """ViT cls-token pooling: (B, N, C) -> (B, C)."""
    if x.ndim == 3:
        return x[:, 0]
    return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_backbone(name, in_chans=1, pretrained=True):
    """Build a timm backbone (conv or ViT) for feature extraction.

    Returns (model, feat_dim, pool_fn):
      - model: timm model with num_classes=0 (feature extractor)
      - feat_dim: model.num_features
      - pool_fn: callable that pools raw features to (B, feat_dim)
        - 2D (B, C) -> identity
        - 4D (B, C, H, W) -> global mean pool over spatial dims
        - 3D (B, N, C) -> mean over token dim N
    """
    timm_name = name.replace("-", "_")
    try:
        m = timm.create_model(timm_name, pretrained=pretrained,
                              in_chans=in_chans, num_classes=0)
    except Exception as e:
        raise ValueError(f"Failed to create backbone '{name}' (timm: '{timm_name}'): {e}") from e
    feat = m.num_features
    logger.info("Backbone '%s' created: in_chans=%d, feature_dim=%d, pretrained=%s",
                name, in_chans, feat, pretrained)

    def pool_fn(x):
        if x.ndim == 2:
            return x
        if x.ndim == 4:
            return x.mean(dim=(2, 3))
        return x.mean(dim=1)
    return m, feat, pool_fn


def extract_backbone_state_dict(state_dict, method):
    """Pull the student/online backbone sub-dict out of a full SSL model state_dict.

    DINOv2 stores the ViT inside MaskedVisionTransformerTIMM under
    'student_backbone.vit.*' (the wrapper's mask_token param is excluded by
    the prefix); BYOL/conv models store the online backbone under
    'backbone.*'. The prefix is stripped so the result loads directly into a
    standalone backbone.
    """
    prefix = "student_backbone.vit." if method in ("dinov2", "dinov3") else "backbone."
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


class ClassificationHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class Model(nn.Module):
    def __init__(self, backbone, pool_fn, head):
        super().__init__()
        self.backbone = backbone
        self.pool_fn = pool_fn
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.pool_fn(feats)
        return self.head(pooled), pooled


def load_model_from_bundle(bundle, device=None):
    """Reconstruct a classification Model from a train bundle.

    For SSL-derived ViT backbones (meta.ssl_method == "dinov2"), rebuilds the
    ViT via build_dinov2_vit (special init args) and uses cls-token pooling.
    Otherwise uses build_backbone.
    """
    meta = bundle["meta"]
    ssl_method = meta.get("ssl_method")
    in_chans = meta["in_chans"]

    if ssl_method in ("dinov2", "dinov3"):
        backbone = build_dinov2_vit(meta["backbone"], in_chans, pretrained=False)
        feat_dim = backbone.num_features
        pool_fn = cls_token_pool_fn
    else:
        # Conv backbone (BYOL or scratch) — use generic build_backbone
        backbone, feat_dim, pool_fn = build_backbone(
            meta["backbone"], in_chans, pretrained=False)

    head = ClassificationHead(feat_dim, meta["num_classes"])
    model = Model(backbone, pool_fn, head)
    model.load_state_dict(bundle["state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    logger.info("Model loaded from bundle: %d classes, backbone=%s, ssl_method=%s",
                meta["num_classes"], meta["backbone"], ssl_method)
    return model


def load_backbone_weights(backbone, bundle, method):
    """Load a bare backbone's weights from an SSL bundle.

    The bundle carries the full model state_dict; the backbone is extracted
    by prefix (student_backbone.vit.* for DINOv2, backbone.* for BYOL/conv).
    """
    if "state_dict" not in bundle:
        print("Error: SSL bundle has no 'state_dict' key (unsupported "
              "pre-0.2.1 bundle format)", file=sys.stderr)
        sys.exit(1)
    backbone.load_state_dict(extract_backbone_state_dict(bundle["state_dict"], method))


def load_ssl_backbone_from_bundle(bundle, device=None):
    """Reconstruct a backbone-only feature extractor from an SSL pretrain bundle.

    Returns (model, feat_dim, pool_fn, meta).
    """
    meta = bundle["meta"]
    method = meta.get("method")
    in_chans = meta["in_chans"]

    if method in ("dinov2", "dinov3"):
        backbone = build_dinov2_vit(meta["backbone"], in_chans, pretrained=False)
        feat_dim = backbone.num_features
        pool_fn = cls_token_pool_fn
    else:
        # BYOL / conv backbone
        backbone, feat_dim, pool_fn = build_backbone(
            meta["backbone"], in_chans, pretrained=False)

    # Load only the backbone weights from the SSL bundle
    load_backbone_weights(backbone, bundle, method)

    if device is not None:
        backbone = backbone.to(device)
    backbone.eval()
    logger.info("SSL backbone loaded: method=%s, backbone=%s, feat_dim=%d",
                method, meta["backbone"], feat_dim)
    return backbone, feat_dim, pool_fn, meta
