"""Production training primitives layered on the pinned upstream Trainer.

Every-microstep FSDP gradient sync is installed by monkeypatching the transformer's
`set_requires_gradient_sync` so upstream `_train_step`'s accumulation-boundary logic is
untouched — measured feasibility requires reduce-scatter every microstep while the
optimizer still steps only at the boundary. The task-balanced sampler draws task then
episode uniformly, deterministically, with exact-position resume.
"""

from __future__ import annotations

import hashlib
import math
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler


def _requires_gradient_sync(batch_idx, accumulation_steps,
                            sync_every_microstep=False):
    """FSDP gradient-sync policy. Default: sync only at the optimizer boundary
    (upstream behavior). Flag on: reduce-scatter every microstep so accumulated
    gradients stay sharded. Optimizer stepping is unaffected."""
    if sync_every_microstep:
        return True
    return (batch_idx + 1) % accumulation_steps == 0


def _episode_keys(dataset, source_name: str) -> list[str]:
    """One stable key per flattened global index:
    <source_name>/<task>/episode_<episode_index:06d>, falling back to
    /index_<local_idx:06d> when a subdataset has no new_metas."""
    keys = []
    for i, ds in enumerate(dataset._datasets):
        repo_id = getattr(ds, "repo_id", None) or getattr(ds, "root", None)
        task = Path(str(repo_id)).name if repo_id is not None else f"task_{i:03d}"
        metas = getattr(ds, "new_metas", None)
        for local_idx in range(len(ds)):
            ep = None
            if metas is not None:
                ep = metas[local_idx].get("episode_index")
            leaf = (f"episode_{int(ep):06d}" if ep is not None
                    else f"index_{local_idx:06d}")
            keys.append(f"{source_name}/{task}/{leaf}")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate episode keys in dataset")
    return keys


class TaskBalancedDistributedSampler(Sampler):
    """Each draw picks a task uniformly then an episode uniformly within that task —
    unlike a flattened DistributedSampler, which weights tasks by episode count.
    Deterministic per (seed, epoch, rank); exact-position resume via
    state_dict/load_state_dict (requires load_worker=0 so position is not skewed
    by worker prefetch). Tracks per-episode exposure counts (batches delivered)."""

    def __init__(self, dataset, num_replicas: int, rank: int, seed: int = 42,
                 source_name: str = "so101_simulated_icl"):
        sub = getattr(dataset, "_datasets", None)
        if not sub:
            raise ValueError("TaskBalancedDistributedSampler requires dataset._datasets")
        self.source_name = source_name
        self.episode_keys = _episode_keys(dataset, source_name)
        self.keys_sha256 = hashlib.sha256(
            "\n".join(self.episode_keys).encode("utf-8")).hexdigest()
        self.exposure_counts = [0] * len(dataset)
        self.task_lengths = [len(d) for d in sub]
        if any(n == 0 for n in self.task_lengths):
            raise ValueError("empty task dataset")
        self.task_offsets = [0]
        for n in self.task_lengths[:-1]:
            self.task_offsets.append(self.task_offsets[-1] + n)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.position = 0
        self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def _global_order(self, epoch: int) -> list[int]:
        g = torch.Generator().manual_seed(self.seed + int(epoch))
        order = []
        while len(order) < self.total_size:
            task = int(torch.randint(len(self.task_lengths), (1,), generator=g))
            ep = int(torch.randint(self.task_lengths[task], (1,), generator=g))
            order.append(self.task_offsets[task] + ep)
        return order

    def _rank_indices(self) -> list[int]:
        return self._global_order(self.epoch)[self.rank::self.num_replicas]

    def __iter__(self):
        for idx in self._rank_indices()[self.position:]:
            self.exposure_counts[idx] += 1
            self.position += 1
            yield idx

    def __len__(self):
        return self.num_samples - self.position

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self.position = 0

    def state_dict(self) -> dict:
        return {"version": 2, "epoch": self.epoch, "position": self.position,
                "seed": self.seed, "rank": self.rank,
                "num_replicas": self.num_replicas,
                "task_lengths": list(self.task_lengths),
                "source_name": self.source_name,
                "episode_keys": list(self.episode_keys),
                "episode_keys_sha256": self.keys_sha256,
                "exposure_counts": list(self.exposure_counts)}

    def load_state_dict(self, state: dict) -> None:
        if int(state["version"]) != 2:
            raise ValueError(f"unsupported sampler state version {state['version']}")
        for key, want in (("seed", self.seed), ("rank", self.rank),
                          ("num_replicas", self.num_replicas)):
            if int(state[key]) != want:
                raise ValueError(f"sampler state {key}={state[key]} != {want}")
        if list(state["task_lengths"]) != list(self.task_lengths):
            raise ValueError("sampler state task_lengths mismatch")
        if state.get("source_name") != self.source_name:
            raise ValueError("sampler state source_name mismatch")
        if state.get("episode_keys_sha256") != self.keys_sha256 or \
                list(state.get("episode_keys", [])) != list(self.episode_keys):
            raise ValueError("sampler state episode keys mismatch")
        counts = [int(c) for c in state["exposure_counts"]]
        if len(counts) != len(self.episode_keys) or any(c < 0 for c in counts):
            raise ValueError("sampler state exposure_counts invalid")
        pos = int(state["position"])
        if not 0 <= pos <= self.num_samples:
            raise ValueError(f"position {pos} out of range 0..{self.num_samples}")
        self.epoch = int(state["epoch"])
        self.position = pos
        self.exposure_counts = counts

    def episode_exposure_ledger(self) -> dict:
        """Exact key -> delivered-batch count, including zero-exposure episodes."""
        return dict(zip(self.episode_keys, self.exposure_counts))


def episode_exposure_summary(ledger: dict) -> dict:
    """Ledger summary. Percentiles use linear interpolation (numpy 'linear')."""
    vals = sorted(ledger.values())
    n = len(vals)
    total = sum(vals)

    def _pct(p):
        if n == 0:
            return 0.0
        k = (n - 1) * p / 100.0
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

    return {"total_draws": total, "dataset_episodes": n,
            "unique_seen": sum(1 for v in vals if v > 0),
            "unseen": sum(1 for v in vals if v == 0),
            "min": vals[0] if n else 0, "max": vals[-1] if n else 0,
            "mean": total / n if n else 0.0,
            "median": _pct(50), "p95": _pct(95)}


def aggregate_episode_exposure_ledgers(rank_states: list[dict]) -> dict:
    """Sum per-rank exposure counts into one exact key->count mapping. Each rank
    state is the gathered object carrying `sampler` state_dict; key sets must be
    identical across ranks."""
    ledger = {}
    for st in rank_states:
        samp = st["sampler"]
        keys = list(samp["episode_keys"])
        counts = list(samp["exposure_counts"])
        if len(keys) != len(counts):
            raise ValueError("rank state episode key/count lengths differ")
        if ledger and set(ledger) != set(keys):
            raise ValueError("rank states have differing episode key sets")
        for k, c in zip(keys, counts):
            ledger[k] = ledger.get(k, 0) + int(c)
    return dict(sorted(ledger.items()))


# ----- Trainer installs (runtime, no upstream source changes) -----------------------------------------

_SYNC_MARK = "_every_microstep_sync_installed"


def install_every_microstep_gradient_sync(trainer):
    """Wrap transformer.set_requires_gradient_sync so every call enables sync —
    the measured mechanism for batch-64 memory feasibility. The upstream optimizer
    boundary in _train_step is untouched. Idempotent."""
    transformer = getattr(trainer, "transformer", None)
    if transformer is None:
        raise ValueError("trainer has no transformer")
    if getattr(transformer, _SYNC_MARK, False):
        return
    orig = transformer.set_requires_gradient_sync
    if orig is None:
        raise ValueError("transformer lacks set_requires_gradient_sync")

    def always_sync(self, flag):        # upstream calls pass False on non-boundary
        return orig(True)

    transformer.set_requires_gradient_sync = types.MethodType(always_sync, transformer)
    setattr(transformer, _SYNC_MARK, True)


def install_task_balanced_loader(trainer, seed: int = 42):
    """Replace trainer.train_loader with a task-balanced sampled DataLoader.
    Requires single-dataset layout (_datasets) and num_workers=0."""
    loader = trainer.train_loader
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not getattr(dataset, "_datasets", None):
        raise ValueError("task-balanced sampling requires a dataset with _datasets")
    if int(getattr(loader, "num_workers", 0)) != 0:
        raise ValueError(
            "task_balanced_sampling requires load_worker=0 so sampler "
            "position resume stays exact")
    config = trainer.config
    sampler = TaskBalancedDistributedSampler(
        dataset, num_replicas=config.world_size, rank=config.rank, seed=seed)
    trainer.train_loader = DataLoader(
        dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
    )
    trainer.train_loader_iter = None
    return sampler


def sampler_state(trainer) -> dict:
    """State dict of the installed task-balanced sampler."""
    sampler = getattr(getattr(trainer, "train_loader", None), "sampler", None)
    if not isinstance(sampler, TaskBalancedDistributedSampler):
        raise ValueError("no TaskBalancedDistributedSampler installed")
    return sampler.state_dict()


def load_sampler_state(trainer, state: dict) -> None:
    """Load a previously captured task-balanced sampler state."""
    sampler = getattr(getattr(trainer, "train_loader", None), "sampler", None)
    if not isinstance(sampler, TaskBalancedDistributedSampler):
        raise ValueError("no TaskBalancedDistributedSampler installed")
    sampler.load_state_dict(state)
    trainer.train_loader_iter = None
