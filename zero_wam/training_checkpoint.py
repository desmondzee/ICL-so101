"""Resumable training-state checkpoint adapter for the pinned upstream Trainer.

Upstream `save_checkpoint` writes only model safetensors (optimizer state save is
disabled) — this adapter adds, without touching the submodule: full AdamW state via
`torch.distributed.checkpoint` all-rank collectives, LR scheduler state, optimizer
step, per-rank RNG (Python/NumPy/torch CPU/CUDA), and per-rank task-balanced sampler
state. Atomic `training_state.pt` under each `checkpoint_step_<N>` dir.
"""

from __future__ import annotations

import json
import os
import random
import types
from pathlib import Path

import torch

SCHEMA_VERSION = 1
STATE_FILENAME = "training_state.pt"
_CKPT_MARK = "_resumable_checkpointing_installed"


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

    trainer.save_checkpoint = types.MethodType(wrapped, trainer)
    setattr(trainer, _CKPT_MARK, True)
