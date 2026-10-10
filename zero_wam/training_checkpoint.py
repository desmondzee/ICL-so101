"""Resumable training-state checkpoint adapter for the pinned upstream Trainer.

Upstream `save_checkpoint` writes only model safetensors (optimizer state save is
disabled) — this adapter adds, without touching the submodule: full AdamW state via
`torch.distributed.checkpoint` all-rank collectives, LR scheduler state, optimizer
step, per-rank RNG (Python/NumPy/torch CPU/CUDA), and per-rank task-balanced sampler
state. Atomic `training_state.pt` under each `checkpoint_step_<N>` dir.
"""

from __future__ import annotations

import json
import logging
import os
import random
import types
from pathlib import Path

import torch

SCHEMA_VERSION = 1
STATE_FILENAME = "training_state.pt"
_CKPT_MARK = "_resumable_checkpointing_installed"

logger = logging.getLogger(__name__)


def _dist():
    return torch.distributed


def _dist_ready(dist) -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_world(dist) -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if _dist_ready(dist) else (0, 1)


def capture_rng_state(device=None) -> dict:
    """Python/NumPy/torch-CPU RNG plus the current CUDA device RNG (CPU tensors)."""
    import numpy as np

    state = {"python": random.getstate(),
             "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state(),
             "cuda_device": None, "cuda_rng": None}
    if torch.cuda.is_available():
        idx = torch.cuda.current_device() if device is None else int(device)
        state["cuda_device"] = idx
        state["cuda_rng"] = torch.cuda.get_rng_state(idx).clone().cpu()
    return state


def restore_rng_state(state: dict) -> None:
    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng"],
                                 device=state["cuda_device"])


def assert_optimizer_boundary_clean(trainer) -> None:
    """Save/load is only valid right after an optimizer boundary (zero_grad ran);
    any non-None gradient means we are mid-accumulation."""
    for name, p in trainer.transformer.named_parameters():
        if p.grad is not None:
            raise RuntimeError(
                f"parameter {name} carries a gradient — save only at an "
                "optimizer boundary after zero_grad")


def save_training_state(trainer, checkpoint_dir, config_sha256, wandb_run_id,
                        get_optimizer_state=None, dist=None):
    """All ranks must call. Returns the written path on rank 0, None elsewhere."""
    from zero_wam.training_primitives import sampler_state

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"checkpoint dir does not exist: {checkpoint_dir}")
    assert_optimizer_boundary_clean(trainer)
    dist = dist or _dist()
    rank, world = _rank_world(dist)

    if get_optimizer_state is None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, get_optimizer_state_dict)

        def get_optimizer_state(model, optim):
            return get_optimizer_state_dict(
                model, optim,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True))

    optim_state = get_optimizer_state(trainer.transformer, trainer.optimizer)
    local = {"rank": rank,
             "rng": capture_rng_state(),
             "sampler": sampler_state(trainer)}
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    gathered = sorted(gathered, key=lambda s: s["rank"])

    path = None
    if rank == 0:
        from zero_wam.training_primitives import (
            aggregate_episode_exposure_ledgers, episode_exposure_summary)
        ledger = aggregate_episode_exposure_ledgers(gathered)
        summary = episode_exposure_summary(ledger)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "step": int(trainer.step),
            "optimizer_state": optim_state,
            "lr_scheduler": trainer.lr_scheduler.state_dict(),
            "rank_states": gathered,
            "world_size": world,
            "config_sha256": config_sha256,
            "wandb_run_id": wandb_run_id,
            "episode_exposure_ledger": ledger,
            "episode_exposure_summary": summary,
        }
        tmp = checkpoint_dir / f"{STATE_FILENAME}.partial"
        torch.save(payload, tmp)
        path = checkpoint_dir / STATE_FILENAME
        os.replace(tmp, path)
        # compact adjacent artifact for logging — nonsecret
        exp_tmp = checkpoint_dir / "episode_exposure.json.partial"
        exp_tmp.write_text(json.dumps(
            {"schema": 1, "step": int(trainer.step), "summary": summary,
             "ledger": ledger}))
        os.replace(exp_tmp, checkpoint_dir / "episode_exposure.json")
    if world > 1:
        dist.barrier()
    return path


def _validate_schema(payload: dict, checkpoint_dir, expected_config_sha256,
                     expected_wandb_run_id, world) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{checkpoint_dir}: schema {payload.get('schema_version')} != "
            f"{SCHEMA_VERSION}")
    if int(payload.get("world_size", -1)) != world:
        raise ValueError(
            f"{checkpoint_dir}: world_size {payload.get('world_size')} != {world}")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError(f"{checkpoint_dir}: config sha mismatch")
    if payload.get("wandb_run_id") != expected_wandb_run_id:
        raise ValueError(f"{checkpoint_dir}: wandb run id mismatch")
    if len(payload.get("rank_states", [])) != world:
        raise ValueError(f"{checkpoint_dir}: rank_states incomplete")
    from zero_wam.training_primitives import (
        aggregate_episode_exposure_ledgers, episode_exposure_summary)
    ledger = aggregate_episode_exposure_ledgers(payload["rank_states"])
    if payload.get("episode_exposure_ledger") != ledger:
        raise ValueError(
            f"{checkpoint_dir}: stored episode_exposure_ledger does not match "
            "recomputed aggregate")
    if payload.get("episode_exposure_summary") != episode_exposure_summary(ledger):
        raise ValueError(
            f"{checkpoint_dir}: stored episode_exposure_summary does not match "
            "recomputed aggregate")


def load_training_state(trainer, checkpoint_dir, expected_config_sha256,
                        expected_wandb_run_id, set_optimizer_state=None,
                        dist=None):
    """All ranks must call; each restores only its own rank-local state.
    Returns the resumed optimizer step."""
    from zero_wam.training_primitives import load_sampler_state

    dist = dist or _dist()
    rank, world = _rank_world(dist)
    checkpoint_dir = Path(checkpoint_dir)
    payload = torch.load(checkpoint_dir / STATE_FILENAME, map_location="cpu",
                         weights_only=False)
    _validate_schema(payload, checkpoint_dir, expected_config_sha256,
                     expected_wandb_run_id, world)

    if set_optimizer_state is None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, set_optimizer_state_dict)

        def set_optimizer_state(model, optim, state):
            return set_optimizer_state_dict(
                model, optim, optim_state_dict=state,
                options=StateDictOptions(full_state_dict=True, strict=False))

    set_optimizer_state(trainer.transformer, trainer.optimizer,
                        payload["optimizer_state"])
    trainer.lr_scheduler.load_state_dict(payload["lr_scheduler"])
    trainer.step = int(payload["step"])
    mine = payload["rank_states"][rank]
    assert int(mine["rank"]) == rank
    load_sampler_state(trainer, mine["sampler"])       # resets loader iterator
    restore_rng_state(mine["rng"])
    if world > 1:
        dist.barrier()
    return trainer.step


def episode_rows_from_ledger(ledger: dict) -> list[dict]:
    """Exact episode rows from the aggregated ledger.

    Ledger keys are ``<source>/<task>/episode_<index>``; values are integer
    draw counts (zeros included). Raises rather than guessing on any key or
    value that does not carry all four fields.
    """
    rows = []
    for sample_id, count in ledger.items():
        parts = str(sample_id).split("/")
        stem = parts[-1] if len(parts) >= 3 else ""
        if len(parts) < 3 or not stem.startswith("episode_"):
            raise ValueError(
                f"exposure ledger key is not "
                f"'<source>/<task>/episode_<index>': {sample_id!r}")
        task = "/".join(parts[1:-1])
        try:
            episode_index = int(stem.removeprefix("episode_"))
        except ValueError as exc:
            raise ValueError(
                f"exposure ledger key has no integer episode index: "
                f"{sample_id!r}") from exc
        if not task:
            raise ValueError(f"exposure ledger key has empty task: {sample_id!r}")
        if isinstance(count, bool) or not isinstance(count, (int, float)) \
                or not float(count).is_integer():
            raise ValueError(
                f"exposure ledger count is not an integer: "
                f"{sample_id!r}={count!r}")
        rows.append({"task": task, "episode_index": episode_index,
                     "sample_id": str(sample_id), "count": int(count)})
    return rows


def task_rows_from_episode_rows(rows: list[dict]) -> list[dict]:
    """Exact per-task aggregation of episode rows."""
    counts = {}
    for row in rows:
        counts.setdefault(row["task"], []).append(row["count"])
    out = []
    for task in sorted(counts):
        values = counts[task]
        total = sum(values)
        out.append({"task": task, "total_draws": total,
                    "episodes": len(values),
                    "unique_seen": sum(1 for c in values if c > 0),
                    "min": min(values), "max": max(values),
                    "mean": total / len(values)})
    return out


def publish_episode_exposure(trainer, checkpoint_dir, step, config_sha256):
    """Rank-0 WandB publish of the already-written episode_exposure.json.

    Reads the local artifact verbatim — never recomputes sampler state.
    Returns {"episodes", "tasks"} row counts. Raises on malformed payloads;
    the caller decides whether to warn-and-continue.
    """
    exposure_path = Path(checkpoint_dir) / "episode_exposure.json"
    payload = json.loads(exposure_path.read_text())
    summary = payload.get("summary")
    ledger = payload.get("ledger")
    if not isinstance(summary, dict) or not isinstance(ledger, dict):
        raise ValueError(
            f"unexpected episode_exposure.json shape in {exposure_path}: "
            "expected 'summary' and 'ledger' dicts")
    rows = episode_rows_from_ledger(ledger)
    task_rows = task_rows_from_episode_rows(rows)
    if not rows:
        raise ValueError(f"empty exposure ledger in {exposure_path}")

    wandb = trainer.wandb
    metrics = {
        f"exposure/{k}": summary[k]
        for k in ("total_draws", "unique_seen", "unseen", "min", "median",
                  "mean", "p95", "max")
    }
    episode_table = wandb.Table(
        columns=["task", "episode_index", "sample_id", "count"],
        data=[[r["task"], r["episode_index"], r["sample_id"], r["count"]]
              for r in rows])
    task_table = wandb.Table(
        columns=["task", "total_draws", "episodes", "unique_seen", "min",
                 "max", "mean"],
        data=[[t["task"], t["total_draws"], t["episodes"], t["unique_seen"],
               t["min"], t["max"], t["mean"]] for t in task_rows])
    metrics["exposure/episodes"] = episode_table
    metrics["exposure/tasks"] = task_table
    metrics["exposure/task_draws"] = wandb.plot.bar(
        task_table, "task", "total_draws",
        title="Episode draws per task")
    metrics["exposure/episode_count_distribution"] = wandb.plot.histogram(
        episode_table, "count",
        title="Episode draw count distribution")
    wandb.log(metrics, step=step, commit=False)

    run_id = Path(trainer.save_dir).parent.name
    artifact = wandb.Artifact(
        f"{run_id}-episode-exposure-step-{step:08d}",
        type="episode-exposure",
        metadata={"run_id": run_id, "optimizer_step": step,
                  "config_sha256": config_sha256,
                  "total_draws": summary["total_draws"],
                  "unique_seen": summary["unique_seen"]})
    artifact.add_file(str(exposure_path))
    wandb.log_artifact(
        artifact, aliases=[f"step-{step}", "latest"])
    return {"episodes": len(rows), "tasks": len(task_rows)}


def install_resumable_checkpointing(trainer, config_sha256, wandb_run_id):
    """Wrap trainer.save_checkpoint so the existing periodic calls produce both
    upstream model safetensors and full training state. Idempotent."""
    if getattr(trainer, _CKPT_MARK, False):
        return
    orig = getattr(trainer, "save_checkpoint", None)
    if orig is None:
        raise ValueError("trainer lacks save_checkpoint")

    def wrapped(self):
        orig()                                       # upstream model save first
        ckpt_dir = Path(trainer.save_dir) / f"checkpoint_step_{trainer.step}"
        rank, _ = _rank_world(_dist())
        if rank == 0:
            if not (ckpt_dir / "transformer").is_dir():
                raise RuntimeError(
                    f"upstream save produced no transformer dir at {ckpt_dir}")
        save_training_state(trainer, ckpt_dir, config_sha256, wandb_run_id)
        if rank == 0 and getattr(trainer, "wandb", None) is not None:
            try:
                publish_episode_exposure(
                    trainer, ckpt_dir, int(trainer.step), config_sha256)
            except Exception as exc:
                # dashboard failure must never invalidate the local checkpoint
                logger.warning(
                    "episode exposure WandB publish failed at step %s: %s",
                    trainer.step, exc)

    trainer.save_checkpoint = types.MethodType(wrapped, trainer)
    setattr(trainer, _CKPT_MARK, True)
