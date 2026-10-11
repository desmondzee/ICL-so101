"""Leakage-safe, single-chunk action-generation validation primitives."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCHEMA = 1
PREFIX_CONTROLS = 8
CHUNK_CONTROLS = 16
OBSERVATION_FRAME_INDICES = (0, 2, 4, 6, 8)
ACTION_NAMES = (
    "shoulder_pan_deg",
    "shoulder_lift_deg",
    "elbow_flex_deg",
    "wrist_flex_deg",
    "wrist_roll_deg",
    "gripper_0_100",
)


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_manifest(loader_root, latent_root, seed=20261010):
    """Freeze one first-steady-state deployment chunk per held-out episode."""
    loader_root, latent_root = Path(loader_root), Path(latent_root)
    source_path = loader_root / "val" / "icl_manifest.json"
    source = json.loads(source_path.read_text())
    samples = []
    for row in sorted(source["samples"], key=lambda item: item["sample_id"]):
        sample_id = row["sample_id"]
        episode = int(sample_id.rsplit("_", 1)[-1])
        task = row["robot_task_name"]
        sample_seed = int(hashlib.sha256(
            f"so101-chunk-causal-v1|{seed}|{sample_id}".encode()
        ).hexdigest()[:8], 16)
        samples.append({
            "sample_id": sample_id,
            "task": task,
            "episode_index": episode,
            "seed": sample_seed,
            "prefix_control_start": 0,
            "prefix_control_end": PREFIX_CONTROLS,
            "target_control_start": PREFIX_CONTROLS,
            "target_control_end": PREFIX_CONTROLS + CHUNK_CONTROLS,
            "observation_frame_indices": list(OBSERVATION_FRAME_INDICES),
            "robot_front_video": str(
                loader_root / "val" / row["robot_video_path"]),
            "robot_wrist_video": str(
                loader_root / "val" / task / "videos" / "chunk-000"
                / "observation.images.wrist"
                / f"episode_{episode:06d}.mp4"),
            "robot_parquet": str(
                loader_root / "val" / task / "data" / "chunk-000"
                / f"episode_{episode:06d}.parquet"),
            "human_latent": str(
                latent_root / "val" / "human_latents"
                / row["human_video_path"].replace(".mp4", ".pth")),
            "human_detailed_text": row["human_local_instruction"],
            "target_text": row["human_text"],
        })
    tasks = sorted({sample["task"] for sample in samples})
    if len(samples) != 50 or len(tasks) != 5:
        raise ValueError(
            f"expected 50 held-out episodes across five tasks, got "
            f"{len(samples)} across {len(tasks)}")
    manifest = {
        "schema": SCHEMA,
        "protocol": "single_chunk_chunk_causal_action_generation",
        "seed": int(seed),
        "action_denoising_steps": 50,
        "action_chunk_controls": CHUNK_CONTROLS,
        "human_icl_scope": "full paired human demonstration",
        "robot_context_scope": (
            "observations and executed actions through control boundary 8 only"),
        "within_chunk_attention": "bidirectional among 16 noisy action tokens",
        "future_robot_context": "withheld",
        "target_visibility": "scorer only after generation",
        "samples": samples,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_manifest_files(manifest):
    for sample in manifest["samples"]:
        for key in (
            "robot_front_video", "robot_wrist_video", "robot_parquet",
            "human_latent",
        ):
            if not Path(sample[key]).is_file():
                raise FileNotFoundError(
                    f"{sample['sample_id']} missing {key}: {sample[key]}")


def inference_payload(sample, observations, prefix_actions):
    """Build the only data allowed to cross into model inference."""
    prefix_actions = np.asarray(prefix_actions, dtype=np.float32)
    if prefix_actions.shape != (PREFIX_CONTROLS, 6):
        raise ValueError(
            f"expected {(PREFIX_CONTROLS, 6)} prefix actions, "
            f"got {prefix_actions.shape}")
    if len(observations) != len(OBSERVATION_FRAME_INDICES):
        raise ValueError("observation prefix length mismatch")
    return {
        "sample_id": sample["sample_id"],
        "seed": int(sample["seed"]),
        "human_latent": sample["human_latent"],
        "human_detailed_text": sample["human_detailed_text"],
        "target_text": sample["target_text"],
        "observations": observations,
        "executed_prefix_actions": prefix_actions,
    }


def assert_payload_has_no_targets(payload):
    forbidden = {
        "target_actions", "future_actions", "future_observations",
        "robot_parquet", "target_control_start", "target_control_end",
    }
    overlap = forbidden.intersection(payload)
    if overlap:
        raise ValueError(f"inference payload leaks scorer fields: {overlap}")


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def score_chunk(predicted, target, persistence):
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    persistence = np.asarray(persistence, dtype=np.float64)
    expected = (CHUNK_CONTROLS, len(ACTION_NAMES))
    if predicted.shape != expected or target.shape != expected:
        raise ValueError(
            f"predicted/target must have shape {expected}, got "
            f"{predicted.shape}/{target.shape}")
    if persistence.shape != expected:
        raise ValueError(f"persistence must have shape {expected}")

    def metrics(value):
        error = value - target
        return {
            "overall_physical_mae": float(np.abs(error).mean()),
            "overall_physical_rmse": float(np.sqrt(np.square(error).mean())),
            "physical_by_channel": {
                name: {
                    "mae": float(np.abs(error[:, index]).mean()),
                    "rmse": float(np.sqrt(np.square(error[:, index]).mean())),
                }
                for index, name in enumerate(ACTION_NAMES)
            },
            "physical_by_control_offset": [
                {
                    "offset": offset,
                    "mae": float(np.abs(error[offset]).mean()),
                    "rmse": float(np.sqrt(np.square(error[offset]).mean())),
                }
                for offset in range(CHUNK_CONTROLS)
            ],
        }
    return {"model": metrics(predicted), "persistence": metrics(persistence)}


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    output = {"num_samples": len(rows), "model": {}, "persistence": {}}
    for predictor in ("model", "persistence"):
        output[predictor]["overall_physical_mae"] = _distribution([
            row["metrics"][predictor]["overall_physical_mae"] for row in rows])
        output[predictor]["overall_physical_rmse"] = _distribution([
            row["metrics"][predictor]["overall_physical_rmse"] for row in rows])
        output[predictor]["physical_by_channel"] = {
            name: {
                metric: _distribution([
                    row["metrics"][predictor]["physical_by_channel"][name][
                        metric]
                    for row in rows
                ])
                for metric in ("mae", "rmse")
            }
            for name in ACTION_NAMES
        }
        output[predictor]["per_task_physical_mae"] = {
            task: float(np.mean([
                row["metrics"][predictor]["overall_physical_mae"]
                for row in task_rows
            ]))
            for task, task_rows in sorted(grouped.items())
        }
    return output
