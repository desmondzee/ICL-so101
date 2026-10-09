"""Resumable training-state checkpoint adapter (world=1, CPU, no CUDA)."""

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import zero_wam.training_checkpoint as tc
import zero_wam.training_primitives as tp


class _ToySub:
    def __init__(self, n, name):
        self.n = n
        self.repo_id = f"/data/train/{name}"
        self.new_metas = [{"episode_index": i} for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


class _ToyMulti:
    def __init__(self, lengths=(10, 5, 7),
                 names=("arm", "block", "cup")):
        self._datasets = [_ToySub(n, names[i])
                          for i, n in enumerate(lengths)]

    def __len__(self):
        return sum(len(d) for d in self._datasets)

    def __getitem__(self, i):
        return i


def _get_opt(model, optim):
    return optim.state_dict()


def _set_opt(model, optim, state):
    optim.load_state_dict(state)


class _FakeTrainer:
    def __init__(self, tmp_path):
        self.transformer = torch.nn.Linear(4, 4)
        self.optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: 1.0)
        self.step = 0
        self.save_dir = tmp_path / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = type("C", (), {"world_size": 1, "rank": 0})()
        self.train_loader = DataLoader(_ToyMulti(), batch_size=1, num_workers=0)
        self.train_loader_iter = None
        tp.install_task_balanced_loader(self)
        self.save_calls = 0

    def save_checkpoint(self):                     # upstream-like: model dir first
        self.save_calls += 1
        (self.save_dir / f"checkpoint_step_{self.step}" / "transformer").mkdir(
            parents=True, exist_ok=True)


def _train_a_bit(t):
    for _ in range(3):                             # fill AdamW moments
        loss = (t.transformer.weight ** 2).sum()
        t.optimizer.zero_grad()
        loss.backward()
        t.optimizer.step()
        t.lr_scheduler.step()
        t.step += 1
    t.optimizer.zero_grad()                        # boundary-clean


def test_rng_roundtrip_exact():
    random_state = tc.capture_rng_state()
    a = (__import__("random").random(), __import__("numpy").random.rand(),
         torch.rand(3))
    tc.restore_rng_state(random_state)
    b = (__import__("random").random(), __import__("numpy").random.rand(),
         torch.rand(3))
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


def test_save_load_roundtrip(tmp_path):
    t = _FakeTrainer(tmp_path)
    _train_a_bit(t)
    it = iter(t.train_loader.sampler)
    [next(it) for _ in range(3)]
    ckpt = t.save_dir / "checkpoint_step_3"
    ckpt.mkdir()
    path = tc.save_training_state(t, ckpt, "cfgsha", "run-1",
                                  get_optimizer_state=_get_opt)
    assert path and Path(path).is_file()
    assert not (ckpt / "training_state.pt.partial").exists()
    # adjacent exposure artifact: exact aggregate ledger == 3 draws on this rank
    exp = json.loads((ckpt / "episode_exposure.json").read_text())
    assert exp["schema"] == 1 and exp["step"] == 3
    assert exp["summary"]["total_draws"] == 3
    assert sum(exp["ledger"].values()) == 3
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["episode_exposure_ledger"] == exp["ledger"]

    # mutate everything
    t.step = 999
    t.lr_scheduler.step()                          # bumps last_epoch past 3
    torch.rand(5)
    it2 = iter(t.train_loader.sampler)
    [next(it2) for _ in range(5)]

    step = tc.load_training_state(t, ckpt, "cfgsha", "run-1",
                                  set_optimizer_state=_set_opt)
    assert step == 3 and t.step == 3
    assert t.lr_scheduler.state_dict()["last_epoch"] == 3
    # sampler resumes at the exact saved suffix (position 3 of a fresh epoch order)
    fresh = tp.TaskBalancedDistributedSampler(_ToyMulti(), 1, 0)
    assert t.train_loader.sampler.position == 3
    assert list(t.train_loader.sampler) == fresh._rank_indices()[3:]
    # optimizer tensors restored (AdamW moments non-zero after 3 real steps)
    assert any("exp_avg_sq" in s and s["exp_avg_sq"].abs().sum() > 0
               for s in t.optimizer.state_dict()["state"].values())
    # RNG restored to saved point: reloading gives an identical next draw
    draw1 = torch.rand(4)
    tc.load_training_state(t, ckpt, "cfgsha", "run-1", set_optimizer_state=_set_opt)
    assert torch.equal(draw1, torch.rand(4))


def test_save_rejects_dirty_gradients(tmp_path):
    t = _FakeTrainer(tmp_path)
    (t.save_dir / "checkpoint_step_0").mkdir()
    t.transformer.weight.grad = torch.ones(4, 4)
    with pytest.raises(RuntimeError, match="optimizer boundary"):
        tc.save_training_state(t, t.save_dir / "checkpoint_step_0", "s", "r",
                               get_optimizer_state=_get_opt)


def test_save_requires_existing_dir(tmp_path):
    t = _FakeTrainer(tmp_path)
    with pytest.raises(RuntimeError, match="does not exist"):
        tc.save_training_state(t, tmp_path / "nope", "s", "r",
                               get_optimizer_state=_get_opt)


def test_load_mismatch_rejection(tmp_path):
    t = _FakeTrainer(tmp_path)
    _train_a_bit(t)
    ckpt = t.save_dir / "checkpoint_step_3"
    ckpt.mkdir()
    tc.save_training_state(t, ckpt, "cfgsha", "run-1", get_optimizer_state=_get_opt)
    for cfg, run in (("other", "run-1"), ("cfgsha", "run-9"), (None, "run-1")):
        with pytest.raises(ValueError):
            tc.load_training_state(t, ckpt, cfg, run, set_optimizer_state=_set_opt)
    # world-size mismatch: fabricate a payload saved under world_size=2
    import torch as _t
    payload = _t.load(ckpt / "training_state.pt", weights_only=False)
    payload["world_size"] = 2
    _t.save(payload, ckpt / "training_state.pt")
    with pytest.raises(ValueError):
        tc.load_training_state(t, ckpt, "cfgsha", "run-1",
                               set_optimizer_state=_set_opt)
    # corrupted stored exposure ledger must reject before mutating trainer
    payload = _t.load(ckpt / "training_state.pt", weights_only=False)
    payload["world_size"] = 1
    key = next(iter(payload["episode_exposure_ledger"]))
    payload["episode_exposure_ledger"][key] += 1
    _t.save(payload, ckpt / "training_state.pt")
    with pytest.raises(ValueError, match="episode_exposure"):
        tc.load_training_state(t, ckpt, "cfgsha", "run-1",
                               set_optimizer_state=_set_opt)


def test_install_wrapper_idempotent_and_writes_state(tmp_path):
    t = _FakeTrainer(tmp_path)
    _train_a_bit(t)
    tc.install_resumable_checkpointing(t, "cfgsha", "run-1")
    tc.install_resumable_checkpointing(t, "cfgsha", "run-1")   # no-op second time
    t.save_checkpoint()
    ckpt = t.save_dir / "checkpoint_step_3"
    assert t.save_calls == 1                          # original called exactly once
    assert (ckpt / "transformer").is_dir()
    assert (ckpt / "training_state.pt").is_file()
    t2 = _FakeTrainer(tmp_path)
    step = tc.load_training_state(t2, ckpt, "cfgsha", "run-1",
                                  set_optimizer_state=_set_opt)
    assert step == 3
