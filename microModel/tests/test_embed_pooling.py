"""Embedding pooling: cls / patch / cls_patch readouts + binary mask drop.

pool_embedding acts on the raw token grid; EmbedExtractor wraps a backbone
and must reproduce the backbone's own forward for source="patch" (the timm
Eva/DINOv3 default is mean-of-patch-tokens with fc_norm applied after
pooling) — the parity test below pins that contract.
"""

import pytest
import torch
from torch import nn

from microModel.backbone import (EmbedExtractor, pool_embedding,
                                 num_prefix_tokens, validate_embed_source)
from microBase import MicroMaxError


def _tokens(batch=2, prefix=1, grid=3, dim=4, seed=0):
    """Token grid (B, prefix + grid*grid, C) with distinguishable values."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, prefix + grid * grid, dim, generator=g)


# ---------------------------------------------------------------------------
# pool_embedding on synthetic tokens
# ---------------------------------------------------------------------------

def test_pool_cls_and_patch_and_concat():
    tokens = _tokens(prefix=1, grid=3, dim=4)
    cls = tokens[:, 0]
    patches = tokens[:, 1:]

    assert torch.equal(pool_embedding(tokens, 1, "cls"), cls)
    assert torch.allclose(pool_embedding(tokens, 1, "patch"), patches.mean(dim=1))
    concat = pool_embedding(tokens, 1, "cls_patch")
    assert concat.shape == (2, 8)
    assert torch.allclose(concat[:, :4], cls)
    assert torch.allclose(concat[:, 4:], patches.mean(dim=1))


def test_pool_prefix_registers_excluded():
    """num_prefix > 1 (CLS + registers): the patch mean skips ALL prefixes."""
    tokens = _tokens(prefix=5, grid=3, dim=4)
    assert torch.allclose(pool_embedding(tokens, 5, "patch"),
                          tokens[:, 5:].mean(dim=1))


def test_pool_binary_mask_drops_background_patches():
    """A patch counts iff it contains any foreground pixel (binary, no
    coefficients); pure-background patches are excluded from the mean."""
    tokens = _tokens(batch=1, prefix=1, grid=2, dim=4)   # 4 patches, 2x2 grid
    # Mask covers ONLY the top-left pixel: patch 0 has foreground, 1..3 none.
    mask = torch.zeros(1, 4, 4)
    mask[0, 0, 0] = 1.0

    out = pool_embedding(tokens, 1, "patch", patch_mask=mask)
    assert torch.allclose(out, tokens[:, 1:2].mean(dim=1))   # patch 0 only

    # Two foreground patches (top-left and bottom-right pixels).
    mask[0, 3, 3] = 1.0
    out = pool_embedding(tokens, 1, "patch", patch_mask=mask)
    assert torch.allclose(out, tokens[:, 1:2].mean(dim=1).add(
        tokens[:, 4:5].mean(dim=1)).div(2))


def test_pool_mask_all_background_falls_back_to_full_mean():
    tokens = _tokens(batch=2, prefix=1, grid=2, dim=4)
    mask = torch.zeros(2, 4, 4)   # no foreground anywhere
    out = pool_embedding(tokens, 1, "patch", patch_mask=mask)
    assert torch.allclose(out, tokens[:, 1:].mean(dim=1))


def test_pool_mask_with_registers_uses_patch_count():
    """The mask grid must match the PATCH count (prefix tokens excluded)."""
    tokens = _tokens(batch=1, prefix=5, grid=2, dim=4)   # 4 patches after prefix
    mask = torch.ones(1, 4, 4)
    out = pool_embedding(tokens, 5, "patch", patch_mask=mask)
    assert torch.allclose(out, tokens[:, 5:].mean(dim=1))


def test_validate_embed_source_rejects_unknown():
    assert validate_embed_source("cls_patch") == "cls_patch"
    with pytest.raises(MicroMaxError):
        validate_embed_source("mean")


# ---------------------------------------------------------------------------
# EmbedExtractor on a stub ViT-like backbone
# ---------------------------------------------------------------------------

class _StubViT(nn.Module):
    """Minimal ViT-like module: forward_features -> tokens, fc_norm Identity."""

    def __init__(self, prefix, grid, dim):
        super().__init__()
        self.num_prefix_tokens = prefix
        self.embed_dim = dim
        self.num_features = dim
        self.fc_norm = nn.Identity()
        self.dummy = nn.Parameter(torch.zeros(1))   # so it has parameters

    def forward_features(self, x):
        return _tokens(batch=x.shape[0], prefix=self.prefix,
                       grid=self.grid, dim=self.embed_dim, seed=7)


def _stub(prefix=1, grid=3, dim=4):
    m = _StubViT(prefix, grid, dim)
    m.prefix, m.grid = prefix, grid
    return m


def test_extractor_cls_patch_dims_and_values():
    vit = _stub(prefix=1, grid=3, dim=4)
    x = torch.zeros(2, 3, 8, 8)

    for source, dim_out in (("cls", 4), ("patch", 4), ("cls_patch", 8)):
        ex = EmbedExtractor(vit, source=source)
        assert ex.feat_dim == dim_out
        out = ex(x)
        assert out.shape == (2, dim_out)
        tokens = _tokens(batch=2, prefix=1, grid=3, dim=4, seed=7)
        expected = pool_embedding(tokens, 1, source)
        assert torch.allclose(out, expected)


def test_extractor_forwards_mask_only_when_weighted():
    vit = _stub(prefix=1, grid=2, dim=4)
    x = torch.zeros(1, 3, 8, 8)
    mask = torch.zeros(1, 8, 8)
    mask[:, :4, :] = 1.0

    plain = EmbedExtractor(vit, source="patch", mask_weighted=False)
    weighted = EmbedExtractor(vit, source="patch", mask_weighted=True)
    # Without mask_weighted the mask is ignored -> full patch mean.
    assert torch.allclose(plain(x, mask=mask),
                          _tokens(batch=1, prefix=1, grid=2, dim=4, seed=7)[:, 1:].mean(1))
    # With mask_weighted the top half of the 2x2 grid is dropped
    # (patches 0 and 1 = token columns 1 and 2).
    tokens = _tokens(batch=1, prefix=1, grid=2, dim=4, seed=7)
    assert torch.allclose(weighted(x, mask=mask), tokens[:, 1:3].mean(dim=1))


def test_extractor_conv_branch_global_mean_pool():
    class _Conv:
        num_features = 6

        def __call__(self, x):
            return torch.arange(2 * 6 * 2 * 2, dtype=torch.float32).reshape(2, 6, 2, 2)

    ex = EmbedExtractor(_Conv())                 # conv only supports source="patch"
    assert ex.vit is False
    assert ex.feat_dim == 6
    out = ex(torch.zeros(2, 3, 8, 8))
    assert out.shape == (2, 6)
    assert torch.allclose(out, torch.arange(48, dtype=torch.float32).reshape(2, 6, 2, 2).mean(dim=(2, 3)))


def test_conv_forward_features_stubs_route_to_conv_pool():
    """timm ConvNeXt-style backbones ALSO define forward_features (returning
    a (B, C, H, W) feature map, not a token grid) — they must route to the
    conv spatial-mean pool, never the token path that needs
    num_prefix_tokens. cls sources are a hard config error there."""

    class _ConvLike:
        num_features = 3

        def forward_features(self, x):
            b = x.shape[0]
            return torch.arange(b * 3 * 2 * 2,
                                dtype=torch.float32).reshape(b, 3, 2, 2)

        def __call__(self, x):
            return self.forward_features(x).mean(dim=(2, 3))

    conv = _ConvLike()
    ex = EmbedExtractor(conv)                    # default source="patch"
    assert ex.vit is False and ex.feat_dim == 3
    out = ex(torch.zeros(2, 1, 16, 16))
    assert out.shape == (2, 3)
    assert torch.allclose(
        out, conv.forward_features(torch.zeros(2, 1, 16, 16)).mean(dim=(2, 3)))
    with pytest.raises(MicroMaxError):
        EmbedExtractor(conv, source="cls")
    with pytest.raises(MicroMaxError):
        EmbedExtractor(conv, source="cls_patch")


def test_real_convnext_tiny_end_to_end():
    """Regression for the train crash: a REAL timm convnext_tiny (which
    exposes forward_features) must pool to (B, 768) without ever touching
    num_prefix_tokens — and match the backbone's own pooled forward."""
    timm = pytest.importorskip("timm")
    net = timm.create_model("convnext_tiny", pretrained=False,
                            in_chans=1, num_classes=0)
    net.eval()
    ex = EmbedExtractor(net)
    assert ex.vit is False and ex.feat_dim == net.num_features == 768
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        out = ex(x)
        expected = net(x)
    assert out.shape == (2, 768)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Parity with the real timm DINOv3 backbone (random init, no download)
# ---------------------------------------------------------------------------

def test_dinov3_patch_parity_with_timm_forward():
    """EmbedExtractor(source='patch') must be bit-equal to the backbone's own
    forward — the DINOv3/Eva default IS mean-of-patch-tokens + fc_norm."""
    from microModel.backbone import build_dino_vit

    vit = build_dino_vit("vit_small_patch16_dinov3", in_chans=1, pretrained=False)
    vit.eval()
    ex = EmbedExtractor(vit, source="patch")
    ex.eval()
    x = torch.rand(2, 1, 224, 224)
    with torch.no_grad():
        assert torch.allclose(ex(x), vit(x), atol=1e-5)
    assert num_prefix_tokens(vit) == vit.num_prefix_tokens

    concat = EmbedExtractor(vit, source="cls_patch")
    concat.eval()
    with torch.no_grad():
        out = concat(x)
    assert out.shape == (2, 2 * vit.num_features)


# ---------------------------------------------------------------------------
# SSLMultiViewDataset return_mask paths (the pretrain UMAP-check loader)
# ---------------------------------------------------------------------------

def test_ssl_multiview_default_and_mask_paths():
    """Both __getitem__ paths must return the view list / (views, masks)."""
    import numpy as np
    import pandas as pd

    from microModel.dataset import SSLMultiViewDataset

    class _Cells:
        def __init__(self):
            self.metadata = pd.DataFrame({"path": ["a.tif", "b.tif"]})

        def get_cell(self, i):
            rng = np.random.default_rng(i)
            img = (rng.random((32, 32, 1)) * 32768).astype(np.float32)
            img[:4, :, :] = 0.0          # zero-background band
            return img

    spec = [{"LongestMaxSize": {"max_size": 32, "p": 1.0}}]
    args = (_Cells(), [0, 1], [1], [spec])
    kwargs = dict(normalize_method="per_channel", clip_low=0.05,
                  clip_high=99.95, with_masking=False, fixed_reference=False,
                  max_value=65535)

    ds = SSLMultiViewDataset(*args, **kwargs)
    views = ds[0]
    assert isinstance(views, list) and len(views) == 1
    assert views[0].shape == (1, 32, 32) and views[0].dtype == torch.float32

    ds_m = SSLMultiViewDataset(*args, **kwargs, return_mask=True)
    views, masks = ds_m[0]
    assert len(views) == 1 and views[0].shape == (1, 32, 32)
    m0 = masks[0]                       # numpy uint8 HxW (collate -> tensor)
    assert m0.shape == (32, 32)
    assert not m0[:4, :].any() and m0[4:, :].all()


# ---------------------------------------------------------------------------
# WholeImageCellDataset return_mask paths (the whole-image infer loader)
# ---------------------------------------------------------------------------

def test_whole_image_default_and_mask_paths(tmp_path):
    """Both __getitem__ layouts must come back intact: the 4-tuple
    (tensor, label, stem, bbox) when return_mask is off — regression for the
    UnboundLocalError that crashed every whole-image inference without mask
    pooling — and the 5-tuple (+ uint8 foreground mask) when it is on."""
    import numpy as np
    import pandas as pd
    import tifffile
    from torch.utils.data import DataLoader

    from microModel.dataset import WholeImageCellDataset

    # Real labeled mask on disk (labels 1 and 2, equal-size bboxes so the
    # default collate can stack the crops — the real path gets uniform sizes
    # from the bundle's baked resize augmentation).
    mask = np.zeros((24, 24), dtype=np.uint16)
    mask[2:10, 2:10] = 1
    mask[12:20, 12:20] = 2
    mask_path = tmp_path / "field_cp_masks_cell.tif"
    tifffile.imwrite(mask_path, mask)

    class _Images:
        metadata = pd.DataFrame({"well": ["A1"], "mask_cell": [str(mask_path)]})

        @staticmethod
        def get_cropped_cell(row_idx, label, mask_name, padding=4):
            # Zero-background crop (everything outside the cell stays 0) so
            # the return_mask path derives a real foreground mask.
            crop = np.zeros((24, 24, 1), dtype=np.float32)
            crop[mask == label] = 1000.0
            ys, xs = np.where(mask == label)
            y0, x0 = int(ys.min()), int(xs.min())
            h, w = int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1
            return (crop[y0:y0 + h, x0:x0 + w], None,
                    (x0, y0, w, h))

    kwargs = dict(channels=[1], normalize_method="per_channel",
                  clip_low=0.05, clip_high=99.95, with_masking=False,
                  fixed_reference=False, max_value=65535)

    ds = WholeImageCellDataset(_Images(), "mask_cell", **kwargs)
    assert len(ds) == 2
    tensor, label, stem, bbox = ds[0]
    assert tensor.shape == (1, 8, 8) and tensor.dtype == torch.float32
    assert isinstance(label, int) and label == 1
    assert isinstance(stem, str) and stem
    assert len(bbox) == 4
    tb, lb, sb, bb = next(iter(DataLoader(ds, batch_size=2)))
    assert tb.shape == (2, 1, 8, 8) and lb.tolist() == [1, 2]

    ds_m = WholeImageCellDataset(_Images(), "mask_cell", return_mask=True,
                                 **kwargs)
    tensor, label, stem, bbox, m = ds_m[0]
    assert m.shape == (8, 8) and m.dtype == np.uint8 and m.all()
    tb, lb, sb, bb, mb = next(iter(DataLoader(ds_m, batch_size=2)))
    assert mb.shape == (2, 8, 8)


def test_umap_check_accepts_both_loader_layouts(tmp_path):
    """The UMAP-check consumer must unwrap both collated batch layouts —
    plain [view_batch] (return_mask off) and ([view_batch], [mask_batch])
    (return_mask on) — and return the content-dependence index.
    Regression for the "'list' object has no attribute 'to'" crash at
    interval saves."""
    from types import SimpleNamespace

    from microModel.pretrain import _run_umap_check

    class _Teacher:
        def __init__(self):
            self.vit = SimpleNamespace(num_prefix_tokens=1)

        def eval(self):
            return self

        def encode(self, x):
            # One random feature row REPEATED for every image -> identical
            # similarity maps across images -> CDI must be ~1.0 (fully
            # homogenized stub).
            g = torch.Generator().manual_seed(3)
            return torch.randn(1, 1 + 4, 8, generator=g).repeat(x.shape[0], 1, 1)

    model = SimpleNamespace(teacher_backbone=_Teacher())
    model.eval = lambda: None

    n = 8
    views = [torch.rand(n, 1, 32, 32)]
    masks = [torch.ones(n, 32, 32)]
    device = torch.device("cpu")

    p = tmp_path / "umap_mask.pdf"
    cdi = _run_umap_check(model, "dinov3", [(views, masks)], device, 1, 42,
                          str(p), embed_source="patch", mask_weighted=True)
    assert p.exists() and p.stat().st_size > 0
    assert isinstance(cdi, float) and cdi > 0.99

    p2 = tmp_path / "umap_plain.pdf"
    cdi2 = _run_umap_check(model, "dinov3", [views], device, 1, 42,
                           str(p2), embed_source="patch", mask_weighted=False)
    assert p2.exists() and p2.stat().st_size > 0
    assert cdi2 is not None and cdi2 > 0.99
