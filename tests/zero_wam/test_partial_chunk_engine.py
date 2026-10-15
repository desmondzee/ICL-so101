"""Fake-model unit tests for zero_wam.partial_chunk_engine (CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from zero_wam.partial_chunk_engine import install_partial_chunk

OBSERVATION_CACHE_TYPE = 0  # upstream wan_va.modules.icl_model constant


class _FakeTransformer:
    def __init__(self):
        self.calls = []
        self.cleared = 0

    def clear_prediction_cache(self, window, cache_name):
        self.cleared += 1

    def cache_counts(self):
        return {"pos": 11}

    def __call__(self, input_dict, **kwargs):
        self.calls.append({"input": input_dict, "kwargs": kwargs})
        return torch.zeros(1, 8)


class _FakeModel:
    """minimal VA_Server stand-in with the methods the adapter wraps."""

    def __init__(self, frame_chunk_size=4, use_cfg=True):
        self.job_config = SimpleNamespace(
            frame_chunk_size=frame_chunk_size, other_field=42)
        self.use_cfg = use_cfg
        self.chunk_idx = 0
        self.cache_name = "pos"
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.init_latent = torch.zeros(1, 48, 1, 28, 14)
        self.last_predicted_actions = None
        self.transformer = _FakeTransformer()
        self.reset_calls = 0
        self.infer_calls = 0
        self.encoded = []

    def _reset_icl(self, *a, **k):
        self.chunk_idx = 0
        self.last_predicted_actions = None
        self.reset_calls += 1
        return "reset-ok"

    def _infer_icl(self, obs):
        self.infer_calls += 1
        return np.zeros((6, self.job_config.frame_chunk_size, 8),
                        dtype=np.float32)

    def _encode_obs(self, obs):
        self.encoded.append(obs)
        return torch.ones(1, 48, 1, 28, 14)

    def _prepare_icl_stream(self, data, timestep, mode, cache_type):
        n = data.shape[2]
        reps = 2 if self.use_cfg else 1
        base = self.job_config.frame_chunk_size * self.chunk_idx
        # real prepare builds one grid per packed token; CFG repeats it.
        grid = torch.cat(
            [torch.arange(base, base + n, dtype=torch.int)] * reps
        ).unsqueeze(0).expand(3, -1).clone()
        input_dict = {
            ("latent_res_lst" if mode == "video"
             else "action_res_lst"): {
                "noisy_latents": data,
                "cache_type_ids": torch.full([reps * n], cache_type,
                                             dtype=torch.int),
            },
            ("latent_grid_id" if mode == "video"
             else "action_grid_id"): grid,
            "current_frame_ids": torch.full(
                [reps * n], 2 * self.chunk_idx + (1 if mode == "action" else 0)),
        }
        return input_dict


def _obs(i):
    return {"front": np.full((2, 2, 3), i, np.uint8), "state": np.zeros(6)}


def _envelope(images, state=None):
    return {"obs": list(images), "compute_kv_cache": True,
            "state": state if state is not None else np.zeros((6, 2, 8))}


def _actions(temporal):
    return torch.ones(1, 6, temporal, 8)


def test_install_is_instance_bound_and_config_shallow_copied():
    a, b = _FakeModel(), _FakeModel()
    cfg_a, cfg_b = a.job_config, b.job_config
    install_partial_chunk(a, 4)
    assert a.job_config is not cfg_a
    assert cfg_a.frame_chunk_size == 4 and cfg_b.frame_chunk_size == 4
    b.job_config.frame_chunk_size = 99
    assert a.job_config.frame_chunk_size == 4
    assert a.partial_frame_cursor == 0 and a.partial_cache_events == []
    assert b._reset_icl.__func__ is _FakeModel._reset_icl
    with pytest.raises(ValueError):
        install_partial_chunk(_FakeModel(), 3)
    with pytest.raises(RuntimeError):
        install_partial_chunk(a, 4)


def test_reset_wrapper_restores_counters():
    m = install_partial_chunk(_FakeModel(), 4)
    m.partial_frame_cursor = 9
    m.partial_cache_events = [{"x": 1}]
    assert m._reset_icl("p", True, "", "", -1.0, 5.0) == "reset-ok"
    assert m.partial_frame_cursor == 0 and m.partial_cache_events == []
    assert m.reset_calls == 1


def test_grid_rebased_to_cursor_and_frame_ids_unchanged():
    m = install_partial_chunk(_FakeModel(), 4)
    m.chunk_idx = 1
    m.partial_frame_cursor = 2
    # original base F*chunk_idx=4; corrected to cursor=2 over both CFG copies
    out = m._prepare_icl_stream(torch.zeros(1, 48, 1, 28, 14), 0, "video", 2)
    assert out["latent_grid_id"][0].tolist() == [2, 2]
    assert out["current_frame_ids"].tolist() == [2, 2]
    out = m._prepare_icl_stream(torch.zeros(1, 6, 1, 8), 0, "action", 2)
    assert out["action_grid_id"][0].tolist() == [2, 2]
    assert out["current_frame_ids"].tolist() == [3, 3]


def test_cfg_repeated_grid_shifted_and_cache_type_zero():
    m = install_partial_chunk(_FakeModel(), 4)
    m._partial_pending_forecast = True
    m.last_predicted_actions = _actions(2)
    m._compute_icl_kv_cache(_envelope([_obs(i) for i in range(4)]))
    video_call, action_call = m.transformer.calls
    # two committed frames at cursor 0 -> grids [0,1] repeated per CFG branch
    assert video_call["input"]["latent_grid_id"][0].tolist() == [0, 1, 0, 1]
    assert action_call["input"]["action_grid_id"][0].tolist() == [0, 1, 0, 1]
    # upstream OBSERVATION_CACHE_TYPE=0 must reach the transformer payloads
    for call in (video_call, action_call):
        stream = next(v for v in call["input"].values()
                      if isinstance(v, dict) and "cache_type_ids" in v)
        assert (stream["cache_type_ids"] == OBSERVATION_CACHE_TYPE).all()


def test_startup_commit_two_frames_then_one():
    m = install_partial_chunk(_FakeModel(), 4)
    m.init_latent = torch.zeros(1, 48, 1, 28, 14)
    m._partial_pending_forecast = True
    m.last_predicted_actions = _actions(2)
    m._compute_icl_kv_cache(_envelope([_obs(i) for i in range(4)]))
    assert m.partial_frame_cursor == 2 and m.chunk_idx == 1
    ev = m.partial_cache_events[0]
    assert (ev["cursor_before"], ev["committed_frames"],
            ev["obs_count"], ev["chunk_idx"]) == (0, 2, 4, 0)
    assert ev["cache_counts"] == {"pos": 11}
    calls = [c["kwargs"]["mode"] for c in m.transformer.calls]
    assert calls == ["forward_latent_only", "forward_action_only"]
    assert m.transformer.calls[0]["kwargs"]["update_cache"] == 1
    assert m.transformer.cleared == 1
    # full envelope (not the list) went to _encode_obs
    assert isinstance(m.encoded[0], dict) and len(m.encoded[0]["obs"]) == 4
    m._partial_pending_forecast = True
    m.last_predicted_actions = _actions(1)
    m._compute_icl_kv_cache(_envelope([_obs(i) for i in range(4)]))
    assert m.partial_frame_cursor == 3 and m.chunk_idx == 2
    assert m.partial_cache_events[1]["committed_frames"] == 1


def test_rejections():
    m = install_partial_chunk(_FakeModel(), 4)
    with pytest.raises(ValueError):  # not exactly four obs
        m._compute_icl_kv_cache(_envelope([_obs(0)]))
    with pytest.raises(KeyError):   # missing 'obs' envelope key
        m._compute_icl_kv_cache({"compute_kv_cache": True})
    m.chunk_idx = 1
    m._partial_pending_forecast = True
    m.last_predicted_actions = _actions(2)   # len2 after startup -> reject
    with pytest.raises(RuntimeError):
        m._compute_icl_kv_cache(_envelope([_obs(i) for i in range(4)]))
    # sentinel/padded tails (len3+) are rejected outright
    m.last_predicted_actions = _actions(3)
    with pytest.raises(RuntimeError):
        m._compute_icl_kv_cache(_envelope([_obs(i) for i in range(4)]))
    # sentinel rows inside the committed block are preserved as written —
    # length discipline is what rejects unexecuted tails
    m.last_predicted_actions = torch.full((1, 6, 1, 8), -999.0)
    m._compute_icl_kv_cache(_envelope([_obs(i) for i in range(4)]))
    assert m.partial_frame_cursor == 1
    assert m.partial_cache_events[-1]["action_shape"] == [1, 6, 1, 8]
    # commit without a pending forecast is rejected
    m.last_predicted_actions = _actions(1)
    m._partial_pending_forecast = False
    with pytest.raises(RuntimeError):
        m._compute_icl_kv_cache(_envelope([_obs(i) for i in range(4)]))
    # double forecast without commit is rejected
    m._infer_icl({"obs": _obs(0)})
    with pytest.raises(RuntimeError):
        m._infer_icl({"obs": _obs(0)})


def test_forecast_frames_two_accepted():
    m = install_partial_chunk(_FakeModel(), 2)
    assert m.job_config.frame_chunk_size == 2


# ---------------------------------------------------------------------------
# zero_wam.train_scene_replay (frozen-scene reproduction)
# ---------------------------------------------------------------------------

POSES = {
    "ball": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "bowl": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
}


def test_compare_recorded_layout_pass_exact_and_signed_zero():
    from zero_wam.train_scene_replay import compare_recorded_layout

    out = compare_recorded_layout(POSES, POSES)
    assert out["pass"] and out["max_abs_pose_error"] == 0.0
    neg = {"ball": [-0.0, -0.0, -0.0, 1.0, -0.0, 0.0, -0.0],
           "bowl": list(POSES["bowl"])}
    assert compare_recorded_layout(neg, POSES)["pass"]


def test_compare_recorded_layout_membership_and_quaternion():
    from zero_wam.train_scene_replay import compare_recorded_layout

    missing = compare_recorded_layout({"ball": POSES["ball"]}, POSES)
    assert not missing["pass"] and missing["missing"] == ["bowl"]
    extra = compare_recorded_layout({**POSES, "can": POSES["ball"]}, POSES)
    assert not extra["pass"] and extra["extra"] == ["can"]
    badq = {**POSES, "ball": [0, 0, 0, 0.0, 0.0, 0.0, 1.0]}
    out = compare_recorded_layout(badq, POSES)
    assert not out["pass"] and out["max_abs_pose_error"] > 0
    assert not compare_recorded_layout(POSES, {})["pass"]


class _FakeSceneEnv:
    def __init__(self, poses):
        self.poses = poses

    def place_distractors(self):
        self.poses = dict(POSES, ball=[9, 9, 9, 1, 0, 0, 0])

    def reset(self, seed=None):
        self.place_distractors()

    def object_poses(self):
        return dict(self.poses)


def test_reset_recorded_scene_selects_legacy(monkeypatch):
    import sim.val.env as valenv
    from zero_wam.train_scene_replay import reset_recorded_scene

    def legacy(self):
        self.poses = dict(POSES)

    monkeypatch.setattr(valenv.ValEnv, "place_distractors", legacy)
    env = _FakeSceneEnv({})
    out = reset_recorded_scene(env, 100000, POSES)
    assert out["sampler"] == "legacy_permutation"
    assert out["attempts"][0]["sampler"] == "current_weighted"
    assert not out["attempts"][0]["pass"]
    assert env.place_distractors.__func__ is legacy


def test_reset_recorded_scene_raises_when_nothing_matches(monkeypatch):
    import sim.val.env as valenv
    from zero_wam.train_scene_replay import reset_recorded_scene

    def legacy(self):
        self.poses = dict(POSES, ball=[9, 9, 9, 1, 0, 0, 0])

    monkeypatch.setattr(valenv.ValEnv, "place_distractors", legacy)
    env = _FakeSceneEnv({})
    orig = env.place_distractors
    with pytest.raises(ValueError):
        reset_recorded_scene(env, 100000, POSES)
    assert env.place_distractors == orig  # restored on failure


# ---------------------------------------------------------------------------
# zero_wam.batched_partial_chunk_engine (fake-object, CPU only)
# ---------------------------------------------------------------------------

def _mk_partial_engine(frame_chunk_size=2):
    """Minimal instance via __new__; base internals monkeypatched per test."""
    from zero_wam.batched_partial_chunk_engine import BatchedPartialChunkEngine

    eng = object.__new__(BatchedPartialChunkEngine)
    eng.batch_size = 4
    eng.frame_chunk_size = frame_chunk_size
    eng.device = torch.device("cpu")
    eng.dtype = torch.float32
    eng.pending = False
    eng.cache_events = []
    eng.slots = [SimpleNamespace(chunk_idx=0, active=True) for _ in range(4)]
    eng.tf = SimpleNamespace(
        cache_counts=lambda: {"pos": 7},
        clear_prediction_cache=lambda *a, **k: None,
    )
    eng.latent_height, eng.latent_width = 28, 14
    return eng


def test_partial_grid_rebases_to_commit_cursor():
    import zero_wam.batched_static_engine as base_mod

    for f in (2, 4):
        eng = _mk_partial_engine(f)
        calls = []
        orig = base_mod.BatchedStaticEngine._grid
        base_mod.BatchedStaticEngine._grid = lambda self, ft, h, w, t, ci: (
            calls.append(ci) or torch.full((4, ft), f * ci))
        try:
            # chunk0: cursor 0, base 0 -> unchanged
            g = eng._grid(4, 1, 1, 0.0, 0)
            assert g[0].tolist() == [0] * 4
            # chunk k>0: base f*k -> rebased to cursor k+1 = 4
            g = eng._grid(4, 1, 1, 0.0, 3)
            assert g[0].tolist() == [4] * 4
        finally:
            base_mod.BatchedStaticEngine._grid = orig
        assert calls == [0, 3]


def test_partial_pending_invariant_and_validation():
    import zero_wam.batched_static_engine as base_mod
    import zero_wam.batched_partial_chunk_engine as mod

    eng = _mk_partial_engine(2)
    eng.slots[3].active = False   # padding slot
    calls = []
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(base_mod.BatchedStaticEngine, "infer",
                   lambda self, obs_list=None: calls.append("infer") or [0] * 4)
        mp.setattr(base_mod.BatchedStaticEngine, "cache_update",
                   lambda self, o, h: calls.append("cache"))
        # commit without pending forecast -> reject
        with pytest.raises(RuntimeError):
            eng.cache_update([None] * 4, [None] * 4)
        # cache_update refuses wrong list length
        eng.pending = True
        with pytest.raises(ValueError):
            eng.cache_update([None] * 3, [None] * 4)
        with pytest.raises(ValueError):
            eng.cache_update([None] * 4, None)
        eng.pending = False
        # infer -> pending; second infer refused
        eng.infer()
        assert eng.pending
        with pytest.raises(RuntimeError):
            eng.infer()
        # chunk0: history temporal must be 2, obs list 4
        obs4 = {"obs": [None] * 4}
        good_hist = torch.zeros(1, 6, 2, 8)
        hists = [good_hist] * 3 + [None]
        with pytest.raises(ValueError):      # len-1 history at startup
            eng.cache_update([obs4] * 4, [torch.zeros(1, 6, 1, 8)] * 3 + [None])
        eng.cache_update([obs4] * 4, hists)
        assert not eng.pending
        ev = eng.cache_events[0]
        assert (ev["committed_frames"], ev["obs_count"], ev["cursor_before"]) == (2, 4, 0)
        assert len(eng.cache_events) == 3    # inactive slot skipped
        # next cycle: history temporal must be 1
        for s in eng.slots:
            s.chunk_idx += 0  # engine commits bumped slot.chunk_idx? emulate:
        for s in eng.slots[:3]:
            s.chunk_idx = 1
        eng.infer()
        with pytest.raises(ValueError):      # len-2 history after startup
            eng.cache_update([obs4] * 4, [torch.zeros(1, 6, 2, 8)] * 3 + [None])
        eng.cache_update([obs4] * 4,
                         [torch.zeros(1, 6, 1, 8)] * 3 + [None])
        assert eng.cache_events[-1]["committed_frames"] == 1
        # forecast tail history (len>1 post-startup) rejected above; also
        # non-dict/short obs rejected
        eng.infer()
        eng.slots[0].chunk_idx = 2
        with pytest.raises(ValueError):
            eng.cache_update([{"obs": [None] * 3}] * 4,
                             [torch.zeros(1, 6, 1, 8)] * 3 + [None])


def test_partial_encode_temporal_one_assertion():
    import zero_wam.batched_static_engine as base_mod
    import zero_wam.batched_partial_chunk_engine as mod

    eng = _mk_partial_engine(2)
    slot = SimpleNamespace(vae=None)
    good = torch.zeros(1, 48, 1, 28, 14)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(base_mod.BatchedStaticEngine, "_encode_obs_slot",
                   lambda self, s, o: good)
        assert eng._encode_obs_slot(slot, {"obs": [1] * 4}) is good
        mp.setattr(base_mod.BatchedStaticEngine, "_encode_obs_slot",
                   lambda self, s, o: torch.zeros(1, 48, 2, 28, 14))
        with pytest.raises(RuntimeError):
            eng._encode_obs_slot(slot, {"obs": [1] * 4})


def test_partial_ctor_guards():
    from zero_wam.batched_partial_chunk_engine import (
        BatchedPartialChunkEngine,
    )
    server = SimpleNamespace(
        job_config=SimpleNamespace(frame_chunk_size=3))
    with pytest.raises(ValueError):   # F must be 2 or 4
        BatchedPartialChunkEngine(server)
    server.job_config.frame_chunk_size = 4
    with pytest.raises(ValueError):   # batch_size must be 4
        BatchedPartialChunkEngine(server, batch_size=2)
