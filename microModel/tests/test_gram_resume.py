"""Gram resume update-count arithmetic (floor, not ceil)."""

from types import SimpleNamespace

from microModel.models.dinov3 import DINOv3


def _state(first, freq):
    return SimpleNamespace(
        gram_use_loss=True,
        gram_backbone=object(),
        _gram_it_first_update=first,
        _gram_update_frequency=freq,
        _num_gram_updates=0,
    )


def test_gram_resume_counts_floor_multiple():
    state = _state(0, 10000)
    DINOv3.set_gram_resume_updates(state, 10000)
    assert state._num_gram_updates == 1
    DINOv3.set_gram_resume_updates(state, 20000)
    assert state._num_gram_updates == 2


def test_gram_resume_with_offset_first_update():
    state = _state(10000, 10000)
    DINOv3.set_gram_resume_updates(state, 10000)
    assert state._num_gram_updates == 1
    DINOv3.set_gram_resume_updates(state, 19999)
    assert state._num_gram_updates == 1
    DINOv3.set_gram_resume_updates(state, 20000)
    assert state._num_gram_updates == 2


def test_gram_resume_before_first_update():
    state = _state(10000, 10000)
    DINOv3.set_gram_resume_updates(state, 9999)
    assert state._num_gram_updates == 0
