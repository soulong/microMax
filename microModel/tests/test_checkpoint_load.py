"""DINOv3 iBOT view normalization + checkpoint schema-diff load handling."""

import torch
from torch import nn
import pytest

from microModel.models.dinov3 import iBOTPatchLoss
from microModel.pretrain import _load_checkpoint_state


def test_ibot_loss_normalizes_by_view_count():
    """The masked-patch CE divides by the number of mask VIEWS (official
    DINOv3), not by the total masked-patch count — the latter, combined with
    per-view-normalized weights, shrank the loss by ~patches-per-view."""
    torch.manual_seed(0)
    crit = iBOTPatchLoss()
    # 6 masked patches from 2 views (3 patches each); per-view-normalized
    # weights (1/patches-per-view) sum to the view count, like _make_masks.
    t = torch.randn(6, 4)
    s = torch.randn(6, 4)
    w = torch.full((6,), 1.0 / 3.0)

    loss = crit.forward_masked(s, t, masks_weight=w, n_views=2)

    per_patch = torch.sum(
        t.float() * torch.log_softmax(s.float() / crit.student_temp, dim=-1),
        dim=-1)
    expected = -(per_patch * w).sum() / 2
    assert torch.allclose(loss, expected)
    old_scale = -(per_patch * w).sum() / 6   # the old patch-count denominator
    assert not torch.allclose(loss, old_scale)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)


def _ckpt(state, method="dinov3"):
    return {"state_dict": state, "meta": {"ssl_method": method}}


def test_reverse_gram_bundle_loads_non_strict():
    """A bundle carrying gram keys loads into a model built WITHOUT gram
    (gram disabled in the current config): the gram keys are dropped instead
    of failing the strict load with unexpected keys."""
    model = _TinyModel()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state["gram_backbone.vit.blocks.0.weight"] = torch.zeros(1)
    state["gram_teacher.head.weight"] = torch.zeros(1)

    _load_checkpoint_state(model, _ckpt(state), method="dinov3")

    # The real weights did load.
    assert torch.equal(model.backbone.weight, state["backbone.weight"])


def test_missing_gram_keys_load_non_strict():
    """The pre-existing direction: a phase-1 bundle (no gram state) into a
    model WITH gram modules keeps the gram init and loads the rest."""
    class _GramModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(3, 3)
            self.gram_backbone = nn.Linear(3, 3)

    model = _GramModel()
    state = {"backbone.weight": model.backbone.weight.detach().clone(),
             "backbone.bias": model.backbone.bias.detach().clone()}

    _load_checkpoint_state(model, _ckpt(state), method="dinov3")

    assert torch.equal(model.backbone.weight, state["backbone.weight"])


def test_non_gram_drift_still_fails_loudly():
    """Any non-gram schema diff is real drift and must raise."""
    model = _TinyModel()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state["something_else.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        _load_checkpoint_state(model, _ckpt(state), method="dinov3")


def test_non_dinov3_method_keeps_strict_load():
    model = _TinyModel()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state["gram_backbone.x"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        _load_checkpoint_state(model, _ckpt(state), method="some_ssl")
