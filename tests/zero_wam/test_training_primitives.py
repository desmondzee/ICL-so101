"""Production training primitives: every-microstep FSDP sync install +
deterministic task-balanced distributed sampler."""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import zero_wam.training_primitives as tp


class _ToySub:
    def __init__(self, n, name=None, start=0):
        self.n = n
        self.repo_id = f"/data/train/{name}" if name else None
        self.new_metas = [{"episode_index": start + i} for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


class _ToyMulti:
    def __init__(self, lengths, names=None):
        self._datasets = [_ToySub(n, name=(names[i] if names else None))
                          for i, n in enumerate(lengths)]

    def __len__(self):
        return sum(len(d) for d in self._datasets)

    def __getitem__(self, i):
        return i


class _FakeTransformer:
    def __init__(self):
        self.flags = []

    def set_requires_gradient_sync(self, flag):
        self.flags.append(flag)


class _Cfg:
    world_size = 4
    rank = 0


_NAMES = ("arm", "block", "cup")


class _FakeTrainer:
    def __init__(self, lengths=(10, 5, 7), num_workers=0):
        self.transformer = _FakeTransformer()
        self.config = _Cfg()
        self.train_loader = DataLoader(_ToyMulti(lengths, _NAMES), batch_size=1,
                                       num_workers=num_workers, shuffle=True)
        self.train_loader_iter = iter(self.train_loader)


def _mk(lengths=(10, 5, 7), replicas=4, rank=0, seed=42):
    return tp.TaskBalancedDistributedSampler(_ToyMulti(lengths, _NAMES),
                                             replicas, rank, seed=seed)


def test_requires_gradient_sync_truth_table():
    accum = 8
    assert [tp._requires_gradient_sync(i, accum, False) for i in range(8)] == \
        [False] * 7 + [True]
    assert [tp._requires_gradient_sync(i, accum, True) for i in range(8)] == \
        [True] * 8
    # optimizer stepping stays boundary-only: should_step is independent of the flag
    assert [(i + 1) % accum == 0 for i in range(8)] == [False] * 7 + [True]


def test_install_every_microstep_sync():
    t = _FakeTrainer()
    tp.install_every_microstep_gradient_sync(t)
    t.transformer.set_requires_gradient_sync(False)
    t.transformer.set_requires_gradient_sync(False)
    t.transformer.set_requires_gradient_sync(True)
    assert t.transformer.flags == [True, True, True]
    tp.install_every_microstep_gradient_sync(t)        # idempotent
    assert len(t.transformer.flags) == 3
    t.transformer.set_requires_gradient_sync(False)
    assert t.transformer.flags == [True] * 4
    class Bare: pass
    with pytest.raises(ValueError):
        tp.install_every_microstep_gradient_sync(Bare())


def test_sampler_deterministic_and_strided():
    a = list(_mk())
    b = list(_mk())
    assert a == b                                  # same seed -> same order
    assert len(a) == _mk().num_samples
    assert all(0 <= i < 22 for i in a)
    # rank-strided reconstruction across replicas == global order prefix
    ranks = [list(tp.TaskBalancedDistributedSampler(_ToyMulti((10, 5, 7), _NAMES), 4, r,
                                                   seed=42)) for r in range(4)]
    merged = [v for tup in zip(*ranks) for v in tup]
    assert merged == _mk()._global_order(0)[:len(merged)]


def test_sampler_task_balance():
    # heavily skewed task lengths still draw tasks ~uniformly
    ds = _ToyMulti((5000, 10, 10))
    s = tp.TaskBalancedDistributedSampler(ds, num_replicas=1, rank=0)
    counts = [0, 0, 0]
    for i in list(s):
        for t, (lo, hi) in enumerate(zip([0, 5000, 5010], [5000, 5010, 5020])):
            if lo <= i < hi:
                counts[t] += 1
    total = sum(counts)
    for c in counts:
        assert abs(c / total - 1 / 3) < 0.02


def test_sampler_position_resume_exact_suffix():
    s = _mk()
    it = iter(s)
    [next(it) for _ in range(2)]
    state = s.state_dict()
    s2 = _mk()
    s2.load_state_dict(state)
    assert s2.position == 2
    assert len(s2) == s.num_samples - 2    # __len__ returns remaining
    assert list(s2) == s._rank_indices()[2:]


def test_sampler_state_mismatch_rejection():
    s = _mk()
    state = s.state_dict()
    for key, bad in (("seed", 43), ("rank", 1), ("num_replicas", 8),
                     ("position", 10 ** 9), ("version", 1),
                     ("source_name", "other"),
                     ("episode_keys_sha256", "0" * 64)):
        st = dict(state); st[key] = bad
        with pytest.raises(ValueError):
            s.load_state_dict(st)
    st = dict(state); st["task_lengths"] = [9, 9, 9]
    with pytest.raises(ValueError):
        s.load_state_dict(st)
    st = dict(state); st["episode_keys"] = list(reversed(st["episode_keys"]))
    with pytest.raises(ValueError):
        s.load_state_dict(st)
    st = dict(state); st["exposure_counts"] = [-1] * len(st["exposure_counts"])
    with pytest.raises(ValueError):
        s.load_state_dict(st)


def test_sampler_epoch_reset_and_change():
    s = _mk()
    list(iter(s))
    s.set_epoch(0)                          # same epoch -> position preserved
    assert s.position == s.num_samples
    s.set_epoch(1)
    assert s.epoch == 1 and s.position == 0
    assert s._rank_indices() != _mk()._rank_indices()   # new epoch -> new order


def test_sampler_rejects_empty_task():
    with pytest.raises(ValueError):
        _mk((10, 0, 5))


def test_episode_keys_exact_and_unique():
    s = _mk()
    keys = s.episode_keys
    assert keys[0] == "so101_simulated_icl/arm/episode_000000"
    assert keys[9] == "so101_simulated_icl/arm/episode_000009"
    assert keys[10] == "so101_simulated_icl/block/episode_000000"
    assert len(set(keys)) == 22
    # unnamed subdataset falls back to task_XXX / index_NNNNNN
    s2 = tp.TaskBalancedDistributedSampler(
        type("D", (), {"_datasets": [type("S", (),
                     {"repo_id": None, "root": None, "new_metas": None,
                      "__len__": lambda s: 2})()], "__len__": lambda s: 2})(),
        1, 0)
    assert s2.episode_keys == ["so101_simulated_icl/task_000/index_000000",
                               "so101_simulated_icl/task_000/index_000001"]


def test_exposure_counts_and_resume():
    s = _mk()
    it = iter(s)
    got = [next(it) for _ in range(3)]
    ledger = s.episode_exposure_ledger()
    assert sum(ledger.values()) == 3
    for gi in got:
        assert ledger[s.episode_keys[gi]] == 1
    # resume restores counts and continuing increments from saved values
    state = s.state_dict()
    s3 = _mk()
    s3.load_state_dict(state)
    assert s3.exposure_counts == s.exposure_counts
    nxt = next(iter(s3))
    assert s3.exposure_counts[nxt] == s.exposure_counts[nxt] + 1


def test_episode_exposure_summary_and_aggregate():
    ledger = {"a/ep": 0, "b/ep": 1, "c/ep": 2, "d/ep": 5}
    summ = tp.episode_exposure_summary(ledger)
    assert summ["total_draws"] == 8 and summ["dataset_episodes"] == 4
    assert summ["unique_seen"] == 3 and summ["unseen"] == 1
    assert summ["min"] == 0 and summ["max"] == 5
    assert summ["mean"] == 2.0 and summ["median"] == 1.5 and \
        abs(summ["p95"] - 4.55) < 1e-9    # linear interp, k=2.85
    # two rank states sum exactly and require identical key sets
    s_a = _mk(replicas=2, rank=0)
    s_b = _mk(replicas=2, rank=1)
    for _ in range(3):
        next(iter(s_a))
        next(iter(s_b))
    agg = tp.aggregate_episode_exposure_ledgers(
        [{"rank": 0, "sampler": s_a.state_dict()},
         {"rank": 1, "sampler": s_b.state_dict()}])
    assert sum(agg.values()) == 6
    assert list(agg) == sorted(agg)
    bad = tp.TaskBalancedDistributedSampler(
        _ToyMulti((10, 5, 7), _NAMES), 2, 0, source_name="other")
    with pytest.raises(ValueError):
        tp.aggregate_episode_exposure_ledgers(
            [{"rank": 0, "sampler": s_a.state_dict()},
             {"rank": 1, "sampler": bad.state_dict()}])
    truncated = s_a.state_dict()
    truncated["exposure_counts"] = truncated["exposure_counts"][:-1]
    with pytest.raises(ValueError, match="key/count lengths"):
        tp.aggregate_episode_exposure_ledgers(
            [{"rank": 0, "sampler": truncated}])


def test_install_task_balanced_loader_and_state():
    t = _FakeTrainer()
    sampler = tp.install_task_balanced_loader(t)
    assert isinstance(sampler, tp.TaskBalancedDistributedSampler)
    assert t.train_loader_iter is None
    assert t.train_loader.sampler is sampler
    assert t.train_loader.num_workers == 0
    # sampler_state / load_sampler_state round-trip an exact suffix
    it = iter(sampler)
    [next(it) for _ in range(3)]
    st = tp.sampler_state(t)
    assert st["position"] == 3
    s2 = tp.install_task_balanced_loader(t)
    tp.load_sampler_state(t, st)
    assert t.train_loader.sampler.position == 3
    assert list(t.train_loader.sampler) == s2._rank_indices()[3:]


def test_install_task_balanced_loader_rejections():
    class NoDatasets:
        pass
    t = _FakeTrainer()
    t.train_loader = DataLoader(list(range(10)), batch_size=1)   # no _datasets
    with pytest.raises(ValueError):
        tp.install_task_balanced_loader(t)
    t2 = _FakeTrainer(num_workers=2)
    with pytest.raises(ValueError):
        tp.install_task_balanced_loader(t2)


def test_upstream_train_py_unchanged():
    # the pinned submodule retains default no-sync accumulation; we install at runtime
    src = (REPO / "third_party/Zero-WAM/wan_va/train.py").read_text()
    assert "def _requires_gradient_sync" not in src
    assert "set_requires_gradient_sync(False)" in src
