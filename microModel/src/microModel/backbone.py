"""Model definitions: FocalLoss, backbone builder, embedding extractor, classification head, Model, bundle loader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from microBase import MicroMaxError

from .utils import logger


def build_dino_vit(vit_name, in_chans, pretrained=False):
    """Build a timm ViT usable for DINO-style SSL (cls-token pooling).

    Two timm model families are accepted:
      1. Classic timm `VisionTransformer`: created with pos_embed="learn",
         dynamic_img_size=True, init_values=1e-5.
      2. timm DINOv3 backbones (`vit_*_patch16_dinov3*`, e.g.
         `vit_small_patch16_dinov3`): these are implemented on the Eva
         architecture (RoPE, register tokens, num_prefix_tokens=5) and do NOT
         accept the VisionTransformer-specific kwargs — they are created with
         a plain create_model call (dynamic_img_size is built-in).
    Anything without a patch_embed / blocks / a class token raises MicroMaxError
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
            raise MicroMaxError(f"Error: failed to create DINOv3 ViT '{vit_name}': {e}")
    except Exception as e:
        raise MicroMaxError(f"Error: failed to create DINO-style ViT '{vit_name}': {e}")
    ok = (hasattr(vit, "patch_embed") and hasattr(vit, "blocks")
          and getattr(vit, "num_prefix_tokens", 0) >= 1
          and getattr(vit, "embed_dim", None) is not None)
    if not ok:
        raise MicroMaxError(f"Error: '{vit_name}' is not a DINO-style timm ViT (needs a "
            f"patch_embed, blocks and a class token); got {type(vit).__name__}.")
    return vit


# ----------------------------------------------------------------------------
# Embedding readout: one pooling concept for every consumer (infer features,
# classify Model, train, pretrain UMAP check).
# ----------------------------------------------------------------------------

EMBED_SOURCES = ("cls", "patch", "cls_patch")


def validate_embed_source(value):
    """Validate an embed_source config value -> normalized string."""
    if value not in EMBED_SOURCES:
        raise MicroMaxError(f"Error: embed_source must be one of {list(EMBED_SOURCES)}; "
            f"got {value!r}")
    return value


def token_grid(backbone, x):
    """Full token grid (B, N, C) from a timm ViT or the pretrain masked-ViT
    wrapper (which exposes encode() instead of forward_features)."""
    if hasattr(backbone, "encode"):
        return backbone.encode(x)
    return backbone.forward_features(x)


def num_prefix_tokens(backbone):
    """Number of non-patch prefix tokens (CLS + registers) on the wrapped ViT."""
    vit = getattr(backbone, "vit", backbone)
    return int(vit.num_prefix_tokens)


def is_vit_backbone(backbone):
    """True for token-grid backbones, False for conv/feature-map backbones.

    ViT here means either a timm ViT (carries a `num_prefix_tokens`
    attribute) or the pretrain masked-ViT wrapper (exposes `encode()`;
    its `.vit` holds the inner ViT). Conv nets like timm's ConvNeXt ALSO
    define `forward_features` — but it returns a (B, C, H, W) feature map,
    not a token grid — so `forward_features` alone must never select the
    token path.
    """
    inner = getattr(backbone, "vit", None)
    if inner is not None and \
            getattr(inner, "num_prefix_tokens", None) is not None:
        return True
    return (hasattr(backbone, "encode")
            or getattr(backbone, "num_prefix_tokens", None) is not None)


def pool_embedding(tokens, num_prefix, source, patch_mask=None):
    """Pool the token grid into an image embedding per source.

    tokens: (B, N, C) with the first num_prefix tokens being CLS + registers.
      - cls:       the CLS token.
      - patch:     mean over the patch tokens (prefix excluded).
      - cls_patch: concat([cls, patch_mean]) — feature dim doubles.
    patch_mask: optional (B, H, W) 0/1 foreground mask (same input resolution
      as the tokens' image). It is averaged onto the patch grid and BINARIZED:
      a patch counts iff it contains any foreground pixel (no coefficients);
      pure-background patches are dropped from the mean. Rows without any
      foreground patch fall back to the plain full mean.
    """
    cls = tokens[:, 0]
    if source == "cls":
        return cls
    patches = tokens[:, num_prefix:]
    if patch_mask is not None:
        grid = int(round(patches.shape[1] ** 0.5))
        if grid * grid != patches.shape[1]:
            raise MicroMaxError(f"Error: patch token count {patches.shape[1]} is not a "
                f"square grid; cannot apply a patch mask")
        m = patch_mask
        if m.dim() == 4:
            m = m.squeeze(1)
        w = F.adaptive_avg_pool2d(m.float().unsqueeze(1), (grid, grid)).flatten(1)
        w = (w > 0).float()
        denom = w.sum(dim=1)
        covered = (patches * w.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1.0).unsqueeze(-1)
        empty = denom == 0
        if empty.any():
            covered[empty] = patches[empty].mean(dim=1)
        patch_mean = covered
    else:
        patch_mean = patches.mean(dim=1)
    if source == "patch":
        return patch_mean
    return torch.cat([cls, patch_mean], dim=-1)


class EmbedExtractor(nn.Module):
    """Backbone wrapper whose forward outputs the pooled image embedding (B, D).

    Routing (see is_vit_backbone): ViT backbones (timm ViTs, or the pretrain
    masked-ViT wrapper's encode) are pooled on the token grid per `source`
    (see pool_embedding); conv backbones (ConvNeXt & co.) are spatially
    mean-pooled — source/mask do not apply, `source="patch"` is the only
    meaningful value (a hard error otherwise: no CLS token exists).
    mask_weighted enables the binary foreground-patch selection on ViTs —
    the mask is then passed per forward call (forward(x, mask=...)).

    fc_norm: timm applies its final LayerNorm AFTER pooling on the Eva/avg
    path (DINOv3), so the pooled embedding is normed here to stay bit-equal
    with the backbone's own forward; backbones that already norm the token
    grid carry an Identity fc_norm, which is skipped.
    """

    def __init__(self, backbone, source="patch", mask_weighted=False):
        super().__init__()
        self.backbone = backbone
        self.source = validate_embed_source(source)
        self.mask_weighted = bool(mask_weighted)
        self.vit = is_vit_backbone(backbone)
        if not self.vit and self.source != "patch":
            raise MicroMaxError(
                f"Error: conv backbone '{type(backbone).__name__}' has no CLS "
                f"token — embed_source must be 'patch', got {self.source!r}")
        fc = getattr(backbone, "fc_norm", None)
        self.fc_norm = None if fc is None or isinstance(fc, nn.Identity) else fc
        if self.vit:
            base_dim = int(getattr(backbone, "embed_dim", None)
                           or getattr(backbone, "num_features"))
        else:
            base_dim = int(backbone.num_features)
        self.feat_dim = base_dim * (2 if (self.vit and source == "cls_patch") else 1)

    def forward(self, x, mask=None):
        if not self.vit:
            # Conv backbones: global spatial mean. timm conv models return
            # either the (B, C, H, W) feature map (forward_features) or the
            # already-pooled (B, C) rows (num_classes=0 forward) — and the
            # stub-based tests pass a (B, N, C) sequence; accept all three.
            out = self.backbone(x)
            if out.ndim == 4:
                return out.mean(dim=(2, 3))
            if out.ndim == 3:
                return out.mean(dim=1)
            return out
        patch_mask = mask if (self.mask_weighted and mask is not None) else None
        emb = pool_embedding(token_grid(self.backbone, x),
                             num_prefix_tokens(self.backbone), self.source, patch_mask)
        if self.fc_norm is not None:
            emb = self.fc_norm(emb)
        return emb


class FocalLoss(nn.Module):
    """Multi-class focal loss over exclusive int class indices.

    gamma=0 degenerates to plain cross-entropy; label_smoothing is forwarded
    to the inner CE (smoothing mass is spread uniformly over all classes).
    """

    def __init__(self, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing)
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BCELoss(nn.Module):
    """Multi-label loss: per-class BCE over logits with multi-hot targets.

    Targets are (B, num_classes) float multi-hot vectors — classes absent
    from a sample's label set count as negatives (closed-world assumption).
    Label smoothing softens targets towards 0.5 (1 -> 1-ls/2, 0 -> ls/2).

    gamma > 0 adds the multi-label analogue of focal loss: every per-class
    BCE term is down-weighted by (1 - p_t) ** gamma, so the easy majority
    (background classes, confidently-absent) stops drowning the rare
    positives in a long-tail label set. gamma=0 is plain BCE.
    """

    def __init__(self, label_smoothing=0.0, gamma=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.gamma = gamma

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            ls = self.label_smoothing
            targets = targets * (1 - ls) + ls / 2
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                 reduction="none")
        if self.gamma > 0:
            p = torch.sigmoid(logits)
            # p_t = probability of the true class per element (present -> p,
            # absent -> 1-p); easy elements -> p_t ~ 1 -> weight ~ 0.
            p_t = p * targets + (1 - p) * (1 - targets)
            bce = bce * (1 - p_t) ** self.gamma
        return bce.mean()


def build_backbone(name, in_chans=1, pretrained=True):
    """Build a timm backbone (conv or ViT) for feature extraction.

    Returns (model, feat_dim) — the raw timm model with num_classes=0; pooling
    is the caller's EmbedExtractor's job.
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
    return m, feat


def extract_backbone_state_dict(state_dict, method, branch="teacher"):
    """Pull the backbone sub-dict out of a full SSL model state_dict.

    Branch selection (DINO-style EMA teacher vs student):
      - dinov3 stores the ViT inside a masked wrapper:
          teacher -> 'teacher_backbone.vit.*', student -> 'student_backbone.vit.*'
        (the wrapper's mask_token param is excluded by the prefix)
    The prefix is stripped so the result loads directly into a standalone
    backbone. The TEACHER (EMA/Polyak-averaged student) is the default
    extraction branch, matching official DINO-family evaluation practice.
    """
    if method == "dinov3":
        prefix = "teacher_backbone.vit." if branch == "teacher" \
            else "student_backbone.vit."
    else:
        raise MicroMaxError(f"Error: cannot extract a backbone from SSL method '{method}' "
            f"(supported: dinov3)")
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


class ClassificationHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class Model(nn.Module):
    """Classification model: an EmbedExtractor backbone + a linear head."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        pooled = self.backbone(x)
        return self.head(pooled), pooled


def load_model_from_bundle(bundle, device=None):
    """Reconstruct a classification Model from a train bundle.

    The pooling is the one the head was TRAINED with — for DINOv3 bundles
    that is the mean-patch readout (source="patch", matching timm's built-in
    Eva average pooling), so embed_source does not apply here.
    """
    meta = bundle["meta"]
    ssl_method = meta.get("ssl_method")
    in_chans = meta["in_chans"]

    if ssl_method == "dinov3":
        vit = build_dino_vit(meta["backbone"], in_chans, pretrained=False)
        backbone = EmbedExtractor(vit, source="patch")
    else:
        # Conv backbone (scratch or non-ViT SSL) — global mean pooling.
        conv, _ = build_backbone(meta["backbone"], in_chans, pretrained=False)
        backbone = EmbedExtractor(conv)

    head = ClassificationHead(backbone.feat_dim, meta["num_classes"])
    model = Model(backbone, head)
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
    by prefix (teacher_backbone.vit.* / student_backbone.vit.* for dinov3).
    """
    if "state_dict" not in bundle:
        raise MicroMaxError("Error: SSL bundle has no 'state_dict' key (unsupported "
              "pre-0.2.1 bundle format)")
    backbone.load_state_dict(extract_backbone_state_dict(bundle["state_dict"], method))


def load_ssl_backbone_from_bundle(bundle, device=None,
                                  embed_source="patch", mask_weighted=False):
    """Reconstruct an EmbedExtractor feature extractor from an SSL pretrain bundle.

    embed_source selects the readout (cls | patch | cls_patch) and
    mask_weighted enables the binary foreground-patch selection (the mask is
    then passed per forward call — callers whose datasets do not produce
    masks keep this False; the extractor then plain-means the patches).
    Returns (EmbedExtractor, feat_dim, meta).
    """
    meta = bundle["meta"]
    method = meta.get("ssl_method")
    in_chans = meta["in_chans"]

    if method == "dinov3":
        backbone = build_dino_vit(meta["backbone"], in_chans, pretrained=False)
    else:
        # Conv backbone
        backbone, _ = build_backbone(meta["backbone"], in_chans, pretrained=False)

    # Load only the backbone weights from the SSL bundle
    load_backbone_weights(backbone, bundle, method)

    model = EmbedExtractor(backbone, source=embed_source, mask_weighted=mask_weighted)
    if device is not None:
        model = model.to(device)
    model.eval()
    logger.info("SSL backbone loaded: method=%s, backbone=%s, embed_source=%s, "
                "mask_weighted=%s, feat_dim=%d",
                method, meta["backbone"], model.source, model.mask_weighted,
                model.feat_dim)
    return model, model.feat_dim, meta
