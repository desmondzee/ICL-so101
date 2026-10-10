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


# ---- checkpoint-level episode exposure WandB publishing ----------------------


def _make_ledger(counts):
    """Ledger of `<src>/<task>/episode_<6d>` keys; `counts` is {task: [counts]}."""
    return {
        f"so101_simulated_icl/{task}/episode_{i:06d}": c
        for task, values in counts.items()
        for i, c in enumerate(values)
    }


def test_episode_rows_all_episodes_and_zeros_retained():
    # 61 tasks totalling exactly 1,109 episodes (50x18 + 11x19), zeros included
    counts = {f"task_{i:03d}": [0 if j % 3 == 0 else j % 4
                                for j in range(18 if i < 50 else 19)]
              for i in range(61)}
    ledger = _make_ledger(counts)
    rows = tc.episode_rows_from_ledger(ledger)
    assert len(rows) == 1109
    assert sum(1 for r in rows if r["count"] == 0) == \
        sum(1 for v in ledger.values() if v == 0)
    assert {r["task"] for r in rows} == set(counts)
    r0 = rows[0]
    assert set(r0) == {"task", "episode_index", "sample_id", "count"}
    assert r0["sample_id"].endswith(f"episode_{r0['episode_index']:06d}")
    assert r0["sample_id"].split("/")[1] == r0["task"]


def test_episode_rows_rejects_malformed():
    with pytest.raises(ValueError):
        tc.episode_rows_from_ledger({"not/an_episode_key": 1})
    with pytest.raises(ValueError):
        tc.episode_rows_from_ledger({"src/task/episode_abc": 1})
    with pytest.raises(ValueError):
        tc.episode_rows_from_ledger({"src/task/episode_1": "not_a_count"})
    with pytest.raises(ValueError):
        tc.episode_rows_from_ledger({"src/task/episode_1": 2.5})
    with pytest.raises(ValueError):
        tc.episode_rows_from_ledger({"src//episode_000001": 1})


def test_task_rows_exact_aggregation():
    ledger = _make_ledger({
        "alpha": [0, 1, 3, 0, 2],
        "beta": [5, 0, 0],
    })
    task_rows = tc.task_rows_from_episode_rows(
        tc.episode_rows_from_ledger(ledger))
    by_task = {r["task"]: r for r in task_rows}
    assert by_task["alpha"] == {"task": "alpha", "total_draws": 6,
                              "episodes": 5, "unique_seen": 3,
                              "min": 0, "max": 3, "mean": 6 / 5}
    assert by_task["beta"] == {"task": "beta", "total_draws": 5,
                               "episodes": 3, "unique_seen": 1,
                               "min": 0, "max": 5, "mean": 5 / 3}
    assert sum(r["total_draws"] for r in task_rows) == sum(ledger.values())
    assert sum(r["episodes"] for r in task_rows) == len(ledger)
    assert sum(r["unique_seen"] for r in task_rows) == \
        sum(1 for v in ledger.values() if v > 0)


def _write_exposure(ckpt_dir, counts, step=16):
    ledger = _make_ledger(counts)
    summary = tp.episode_exposure_summary(ledger)
    path = Path(ckpt_dir) / "episode_exposure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"schema": 1, "step": step, "summary": summary, "ledger": ledger}))
    return summary


def test_publish_episode_exposure_wandb_calls(tmp_path):
    import types
    from unittest.mock import MagicMock

    counts = {f"task_{i:03d}": [j % 3 for j in range(18)] for i in range(61)}
    n_eps = 61 * 18
    summary = _write_exposure(tmp_path / "ckpt", counts)
    wandb = MagicMock()
    run_dir = tmp_path / "run-xyz"
    trainer = types.SimpleNamespace(
        wandb=wandb, save_dir=run_dir / "checkpoints", step=16)

    out = tc.publish_episode_exposure(
        trainer, tmp_path / "ckpt", 16, "cfgsha256")
    assert out == {"episodes": n_eps, "tasks": 61}

    # scalars + tables in one commit=False log at the checkpoint step
    wandb.log.assert_called_once()
    metrics, = wandb.log.call_args.args
    kwargs = wandb.log.call_args.kwargs
    assert kwargs == {"step": 16, "commit": False}
    for key in ("total_draws", "unique_seen", "unseen", "min", "median",
                "mean", "p95", "max"):
        assert metrics[f"exposure/{key}"] == summary[key]
    for key in ("exposure/episodes", "exposure/tasks",
                "exposure/task_draws", "exposure/episode_count_distribution"):
        assert key in metrics

    tables = wandb.Table.call_args_list
    assert tables[0].kwargs["columns"] == [
        "task", "episode_index", "sample_id", "count"]
    assert len(tables[0].kwargs["data"]) == n_eps
    assert tables[1].kwargs["columns"] == [
        "task", "total_draws", "episodes", "unique_seen", "min", "max", "mean"]
    assert len(tables[1].kwargs["data"]) == 61

    art = wandb.Artifact.call_args
    assert art.args[0] == "run-xyz-episode-exposure-step-00000016"
    assert art.kwargs["type"] == "episode-exposure"
    assert art.kwargs["metadata"]["config_sha256"] == "cfgsha256"
    assert art.kwargs["metadata"]["optimizer_step"] == 16
    assert art.kwargs["metadata"]["total_draws"] == summary["total_draws"]
    assert art.kwargs["metadata"]["unique_seen"] == summary["unique_seen"]
    wandb.Artifact.return_value.add_file.assert_called_once_with(
        str(tmp_path / "ckpt" / "episode_exposure.json"))
    wandb.log_artifact.assert_called_once()
    assert wandb.log_artifact.call_args.kwargs["aliases"] == [
        "step-16", "latest"]


def test_save_checkpoint_publishes_and_warns_on_failure(tmp_path):
    from unittest.mock import MagicMock

    t = _FakeTrainer(tmp_path)
    _train_a_bit(t)
    t.wandb = MagicMock()
    tc.install_resumable_checkpointing(t, "cfgsha", "run-1")
    t.save_checkpoint()
    ckpt = t.save_dir / "checkpoint_step_3"
    assert (ckpt / "training_state.pt").is_file()
    t.wandb.log.assert_called_once()
    t.wandb.log_artifact.assert_called_once()

    # dashboard failure must not invalidate the local checkpoint
    t2 = _FakeTrainer(tmp_path / "other")
    _train_a_bit(t2)
    t2.wandb = MagicMock()
    t2.wandb.log.side_effect = RuntimeError("wandb down")
    tc.install_resumable_checkpointing(t2, "cfgsha", "run-1")
    t2.save_checkpoint()                            # must not raise
    ckpt2 = t2.save_dir / "checkpoint_step_3"
    assert (ckpt2 / "training_state.pt").is_file()
    assert (ckpt2 / "episode_exposure.json").is_file()
