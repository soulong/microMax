"""Gram refresh schedule: LOCAL time axis of the anchoring phase.

update_frequency / it_first_update / it_load_ema_teacher / max_updates all
count batches of the anchoring run itself (set_gram_origin), independent of
the resumed-from checkpoint's absolute step history; an interrupted run
restores origin + true count from the bundle meta
(_resolve_gram_refresh_state).
"""

from types import SimpleNamespace

from microModel.models.dinov3 import DINOv3
from microModel.pretrain import _resolve_gram_refresh_state


def _state(first=0, freq=10000, origin=0, max_updates=None):
    return SimpleNamespace(
        gram_use_loss=True,
        gram_backbone=object(),
        _gram_it_first_update=first,
        _gram_update_frequency=freq,
        _gram_it_load_ema_teacher=-1,
        _gram_ema_teacher_loads=[],
        _gram_rep_update=True,
        _gram_max_updates=max_updates,
        _gram_step_origin=origin,
        _num_gram_updates=0,
    )


def _run(state, gs_start, gs_stop):
    """Drive maybe_update_gram over [gs_start, gs_stop); return fire steps."""
    fires = []
    holder = {"gs": -1}
    state.update_gram = lambda m: fires.append(holder["gs"])
    for gs in range(gs_start, gs_stop):
        holder["gs"] = gs
        DINOv3.maybe_update_gram(state, gs)
    return fires


def test_gram_refresh_fires_on_local_multiples():
    """With an origin (resumed mid-lineage), the schedule counts batches of
    the anchoring phase: fires at origin + freq - 1, etc. — NOT at absolute
    multiples (the old absolute accounting exhausted max_updates: 1 before
    the run even started)."""
    state = _state(origin=251000, freq=10000, max_updates=1)
    fires = _run(state, 251000, 280000)
    assert fires == [260999]
    assert state._num_gram_updates == 1


def test_gram_refresh_after_max_updates_is_frozen():
    state = _state(origin=251000, freq=10000, max_updates=1)
    _run(state, 251000, 280000)
    assert _run(state, 280000, 320000) == []      # frozen after the one refresh


def test_gram_refresh_unlimited_when_max_updates_null():
    state = _state(origin=0, freq=10000, max_updates=None)
    assert _run(state, 0, 35000) == [9999, 19999, 29999]


def test_gram_refresh_respects_first_update_locally():
    state = _state(first=15000, freq=10000, origin=0)
    fires = _run(state, 0, 35000)
    assert fires == [19999, 29999]                # 15k..19,999/29,999


def test_gram_origin_reset_rearms_schedule():
    state = _state(origin=251000, freq=10000, max_updates=1)
    _run(state, 251000, 280000)
    assert state._num_gram_updates == 1
    DINOv3.set_gram_origin(state, 280000, count=0)
    assert state._gram_step_origin == 280000 and state._num_gram_updates == 0


def test_gram_it_load_ema_teacher_uses_local_step():
    state = _state(origin=251000, freq=10000)
    state._gram_it_load_ema_teacher = 5000
    state.gram_load_ema_teacher = lambda: state._gram_ema_teacher_loads.append(1)
    _run(state, 251000, 256000)                   # local 0..4999: no fire
    _run(state, 256000, 256001)                   # local 5000: fire
    assert state._gram_ema_teacher_loads == [1]


# ---------------------------------------------------------------------------
# _resolve_gram_refresh_state: self-resume restores, new phase resets
# ---------------------------------------------------------------------------

def test_resolve_restores_on_self_resume(tmp_path):
    out = tmp_path / "phase2"
    meta = {"gram_refresh": {"origin": 251000, "count": 1}}
    origin, count = _resolve_gram_refresh_state(
        meta, str(out), str(out / "model_220.pt"), 276100)
    assert (origin, count) == (251000, 1)


def test_resolve_fresh_when_self_resume_lacks_meta(tmp_path):
    out = tmp_path / "phase2"
    origin, count = _resolve_gram_refresh_state(
        {}, str(out), str(out / "model_220.pt"), 276100)
    assert (origin, count) == (276100, 0)


def test_resolve_resets_for_new_anchoring_phase(tmp_path):
    """Resuming from a bundle OUTSIDE the output dir is a NEW anchoring
    phase over a different model — the schedule must start fresh even if
    that bundle somehow carried gram_refresh meta."""
    meta = {"gram_refresh": {"origin": 251000, "count": 1}}
    origin, count = _resolve_gram_refresh_state(
        meta, str(tmp_path / "phase2_new"),
        str(tmp_path / "phase1" / "model.pt"), 0)
    assert (origin, count) == (0, 0)


def test_resolve_fresh_without_resume_path(tmp_path):
    origin, count = _resolve_gram_refresh_state({}, str(tmp_path), None, 0)
    assert (origin, count) == (0, 0)


# ---------------------------------------------------------------------------
# Head-track history alignment (absolute-epoch axis)
# ---------------------------------------------------------------------------

from microModel.pretrain import (_align_history, _append_at,
                                 _is_self_resume)


def test_align_history_pads_to_absolute_epoch():
    """A history that starts mid-run (gram stats from ep201) must land at
    its true epochs after padding, sharing the axis with full histories."""
    assert _align_history([1.0, 2.0], 200)[200 - 2:] == [1.0, 2.0]
    assert len(_align_history([1.0, 2.0], 200)) == 200
    assert _align_history([1.0, 2.0], 2) == [1.0, 2.0]   # nothing to pad
    assert _align_history([], 5) == [None] * 5


def test_append_at_places_sparse_values_at_true_epochs():
    hist = []
    _append_at(hist, 50, 0.7)                 # first CDI at ep50
    _append_at(hist, 100, 0.8)
    assert hist == [None] * 49 + [0.7] + [None] * 49 + [0.8]
    _append_at(hist, 100, 0.9)                # same-epoch re-append overwrites
    assert hist[99] == 0.9


def test_is_self_resume_checks_output_dir(tmp_path):
    out = tmp_path / "phase2"
    assert _is_self_resume(str(out / "model_220.pt"), str(out))
    assert not _is_self_resume(str(tmp_path / "phase1" / "model.pt"), str(out))
    assert not _is_self_resume(None, str(out))
