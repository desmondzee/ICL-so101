"""Deterministic distributed held-out loss evaluation for SO-101 Zero-WAM."""

from __future__ import annotations

import hashlib
import json
import os
import random
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from zero_wam.so101_a100_preflight import atomic_write_json
from zero_wam.training_checkpoint import capture_rng_state, restore_rng_state

VALIDATION_SCHEMA = 1
VALIDATION_SEED = 20261010


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_seed(sample_id: str, base_seed: int = VALIDATION_SEED) -> int:
    payload = f"so101-validation-v1|{base_seed}|{sample_id}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31)


def build_validation_manifest(loader_root, latent_root, base_seed=VALIDATION_SEED):
    loader_root, latent_root = Path(loader_root), Path(latent_root)
    source = loader_root / "val" / "icl_manifest.json"
    payload = json.loads(source.read_text())
    rows = []
    for sample in sorted(payload["samples"], key=lambda row: row["sample_id"]):
        sample_id = sample["sample_id"]
        rows.append({
            "sample_id": sample_id,
            "task": sample["robot_task_name"],
            "episode_index": int(sample_id.rsplit("_", 1)[-1]),
            "random_seed": deterministic_seed(sample_id, base_seed),
        })
    tasks = sorted({row["task"] for row in rows})
    if len(rows) != 50 or len(tasks) != 5:
        raise ValueError(f"expected 50 validation episodes/5 tasks, got {len(rows)}/{len(tasks)}")
    return {
        "schema": VALIDATION_SCHEMA,
        "base_seed": int(base_seed),
        "conditioning": {
            "human_video": True,
            "human_detailed_text": True,
            "short_target_text": False,
            "drop_icl": 0.0,
            "target_text_cfg_prob": 1.0,
        },
        "loader_manifest": str(source.resolve()),
        "loader_manifest_sha256": sha256_file(source),
        "latent_release_sha256": sha256_file(latent_root / "release.json"),
        "samples": rows,
    }


def manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze_validation_manifest(loader_root, latent_root, output_path):
    output_path = Path(output_path)
    manifest = build_validation_manifest(loader_root, latent_root)
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    if output_path.exists():
        current = json.loads(output_path.read_text())
        if current != manifest:
            raise ValueError(f"refusing to replace changed validation manifest: {output_path}")
    else:
        atomic_write_json(output_path, manifest)
    return manifest


def _val_config(train_config, loader_root, latent_root):
    from copy import deepcopy

    cfg = deepcopy(train_config.dataset_sources[0]["config"])
    loader_root, latent_root = Path(loader_root).resolve(), Path(latent_root).resolve()
    cfg.dataset_path = str(loader_root / "val")
    cfg.icl_manifest_path = str(loader_root / "val" / "icl_manifest.json")
    cfg.robot_latent_path = str(latent_root / "val")
    cfg.human_latent_path = str(latent_root / "val" / "human_latents" / "so101")
    cfg.cfg_prob = 1.0
    cfg.rank = train_config.rank
    cfg.local_rank = train_config.local_rank
    cfg.world_size = train_config.world_size
    cfg.init_worker = 1
    cfg.excluded_task_names = []
    cfg.expected_num_train_tasks = 5
    cfg.enable_dataset_index_cache = True
    cfg.rebuild_dataset_index_cache = False
    from wan_va.configs.va_robotwin_cfg import load_robotwin_norm_stat
    cfg.norm_stat = load_robotwin_norm_stat(loader_root / "val" / "meta" / "action_stats.json")
    return cfg


def build_validation_dataset(trainer, loader_root, latent_root):
    from wan_va.dataset.icl_lerobot_latent_dataset import MultiICLLeRobotLatentDataset

    dataset = MultiICLLeRobotLatentDataset(
        config=_val_config(trainer.config, loader_root, latent_root))
    if len(dataset) != 50:
        raise ValueError(f"validation dataset has {len(dataset)} samples, expected 50")
    return dataset


def _dataset_index(dataset, task: str, episode_index: int) -> int:
    offset = 0
    for subdataset in dataset._datasets:
        if Path(str(subdataset.repo_id)).name == task:
            for local_index, meta in enumerate(subdataset.new_metas):
                if int(meta["episode_index"]) == episode_index:
                    return offset + local_index
            break
        offset += len(subdataset)
    raise KeyError(f"validation sample not found: {task}/episode_{episode_index:03d}")


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _batch(item: dict) -> dict:
    return {key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in item.items()}


def _evaluate_one(trainer, item, seed: int):
    _seed_everything(seed)
    with torch.no_grad():
        batch = trainer.convert_input_format(_batch(item))
        input_dict = trainer._prepare_input_dict(batch)
        if input_dict["icl_latent_dict"] is None:
            raise RuntimeError("validation unexpectedly dropped human ICL")
        output = trainer.transformer(input_dict, train_mode=True)
        video, action, mcp = trainer.compute_loss(input_dict, output)
        # compute_loss divides all components by the training accumulation count.
        scale = float(trainer.gradient_accumulation_steps)
        video, action = video * scale, action * scale
        mcp = [loss * scale for loss in mcp]
        weighted_mcp = sum(
            weight * loss for weight, loss in zip(trainer.config.mcp_loss_weights, mcp))
        total = video + action + weighted_mcp
    values = {
        "video_loss": float(video),
        "action_loss": float(action),
        "mcp_losses": [float(loss) for loss in mcp],
        "mcp_weighted_total": float(weighted_mcp),
        "total": float(total),
    }
    if not all(np.isfinite(value) for value in
               [values["video_loss"], values["action_loss"],
                values["mcp_weighted_total"], values["total"], *values["mcp_losses"]]):
        raise FloatingPointError(f"nonfinite validation loss: {values}")
    return values


def _aggregate(rows: list[dict]) -> dict:
    metrics = ("video_loss", "action_loss", "mcp_weighted_total", "total")
    overall = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        overall[metric] = {
            "mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max()),
        }
    per_task = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    for task, task_rows in sorted(grouped.items()):
        per_task[task] = {
            metric: float(np.mean([row[metric] for row in task_rows]))
            for metric in metrics
        }
    macro = {
        metric: float(np.mean([values[metric] for values in per_task.values()]))
        for metric in metrics
    }
    return {"overall": overall, "per_task": per_task, "macro_task_mean": macro}


def evaluate_validation(trainer, dataset, manifest, output_dir, step: int):
    rank, world = dist.get_rank(), dist.get_world_size()
    rng = capture_rng_state()
    was_training = trainer.transformer.training
    scheduler_before = json.dumps(trainer.lr_scheduler.state_dict(), sort_keys=True)
    optimizer_step_before = int(trainer.step)
    drop_icl_before = float(trainer.config.drop_icl)
    trainer.transformer.eval()
    trainer.config.drop_icl = 0.0
    local_rows = []
    try:
        for position in range(rank, len(manifest["samples"]), world):
            spec = manifest["samples"][position]
            item = dataset[_dataset_index(dataset, spec["task"], spec["episode_index"])]
            values = _evaluate_one(trainer, item, spec["random_seed"])
            local_rows.append({**spec, **values, "rank": rank})
    finally:
        trainer.config.drop_icl = drop_icl_before
        trainer.transformer.train(was_training)
        restore_rng_state(rng)
    if int(trainer.step) != optimizer_step_before:
        raise RuntimeError("validation changed optimizer step")
    if json.dumps(trainer.lr_scheduler.state_dict(), sort_keys=True) != scheduler_before:
        raise RuntimeError("validation changed LR scheduler")

    gathered = [None] * world
    dist.all_gather_object(gathered, local_rows)
    rows = sorted([row for shard in gathered for row in shard],
                  key=lambda row: row["sample_id"])
    if len(rows) != 50 or len({row["sample_id"] for row in rows}) != 50:
        raise RuntimeError("validation did not produce exactly 50 unique samples")
    report = None
    if rank == 0:
        report = {
            "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "optimizer_step": int(step), "manifest_sha256": manifest["manifest_sha256"],
            "conditioning": manifest["conditioning"], "samples": rows,
            **_aggregate(rows),
        }
        output = Path(output_dir) / f"step_{int(step):04d}.json"
        atomic_write_json(output, report)
        if getattr(trainer, "wandb", None) is not None:
            metrics = {
                f"validation/{key}": value
                for key, value in report["macro_task_mean"].items()
            }
            trainer.wandb.log(metrics, step=int(step))
    dist.barrier()
    return report


def install_validation(trainer, loader_root, latent_root, run_dir):
    """Run step-0 validation and validate after every checkpoint save."""
    manifest_path = Path(run_dir) / "validation_manifest.json"
    if dist.get_rank() == 0:
        manifest = freeze_validation_manifest(loader_root, latent_root, manifest_path)
    else:
        manifest = None
    box = [manifest]
    dist.broadcast_object_list(box, src=0)
    manifest = box[0]
    dataset = build_validation_dataset(trainer, loader_root, latent_root)
    output_dir = Path(run_dir) / "validation"
    evaluate_validation(trainer, dataset, manifest, output_dir, trainer.step)

    original_save = trainer.save_checkpoint

    def save_and_validate(self):
        original_save()
        evaluate_validation(self, dataset, manifest, output_dir, self.step)

    trainer.save_checkpoint = types.MethodType(save_and_validate, trainer)
    return manifest
