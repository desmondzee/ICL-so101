"""Sequential (non-packed) replay of one exact training episode (opt-in).

Diagnostic control for the packed replay runner: identical cohort data,
TrainSim, start-pose gate, and artifact schema, but driven through the
original `VA_Server.infer` closed loop (one rollout, dynamic cache) — no
BatchedStaticEngine.  Reuses the generic helpers from
`scripts.run_so101_batched_train_replay` for cohort/sim handling.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from scripts.run_so101_batched_train_replay import (
    TrainSim,
    check_start_poses,
    load_cohort,
    select_episode_rows,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="data/models/checkpoints/v2b_step500")
    p.add_argument("--base-model", default="data/models/zero-wam-pretrain")
    p.add_argument("--output",
                   default="outputs/zero_wam/train_replay_diagnostics/sequential_raw_v2b")
    p.add_argument("--episode-id", action="append",
                  default=["push_cube_into_tape_square/episode_000"])
    p.add_argument("--sample-seed", type=int, default=20261105)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--horizon-times-oracle", type=float, default=2.0)
    p.add_argument("--max-controls", type=int, default=None)
    return p.parse_args()


def run_episode_train(model, episode, out_dir, *, max_controls, sample_seed,
                      video_guidance_scale=-1.0, action_ema_alpha=1.0):
    """run_episode equivalent driving a TrainSim (train env + pose gate)."""
    from zero_wam.so101_closed_loop import (
        FRONT,
        WRIST,
        motion_diagnostics,
        smooth_action,
    )

    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    torch.cuda.manual_seed_all(sample_seed)
    started = time.monotonic()
    sim = TrainSim(episode["task"], episode["seed"], episode["visual_config"])
    check = check_start_poses(sim, episode["object_poses_start"])
    if not check["pass"]:
        raise ValueError(f"start_pose_mismatch: {check}")
    model.infer({
        "reset": True,
        "prompt": episode["instruction"],
        "use_icl": True,
        "icl_latent_path": episode["human_latent"],
        "video_guidance_scale": video_guidance_scale,
    })
    first_obs = sim.observe()
    previous_command = first_obs["observation.state"].copy()
    frames = [np.concatenate([first_obs[FRONT], first_obs[WRIST]], axis=1)]
    actions, telemetry, chunk_seconds, controls, chunk = [], [], [], 0, 0
    target_names = sim.task_objects
    target_starts = {
        name: sim.env.object_pos(name).copy() for name in target_names
    }
    first_success = None
    while controls < max_controls and first_success is None:
        tick = time.monotonic()
        action = np.asarray(model.infer({"obs": first_obs})["action"],
                            dtype=np.float32)
        commanded_history = np.zeros_like(action)
        executed_history = np.zeros_like(action)
        chunk_seconds.append(time.monotonic() - tick)
        stride = action.shape[2] // 4
        keyframes = []
        for i in range(1 if chunk == 0 else 0, action.shape[1]):
            for j in range(action.shape[2]):
                predicted = action[:, i, j]
                command = smooth_action(predicted, previous_command,
                                        action_ema_alpha)
                info = sim.step(command)
                measured = sim.env.lerobot_joints().astype(np.float32)
                commanded_history[:, i, j] = command
                executed_history[:, i, j] = measured
                previous_command = command
                actions.append(predicted.tolist())
                controls += 1
                target_metrics = {}
                tcp = sim.env._get_tcp_pose()[:3]
                for name in target_names:
                    position = sim.env.object_pos(name)
                    target_metrics[name] = {
                        "tcp_distance_m": float(np.linalg.norm(tcp - position)),
                        "displacement_m": float(np.linalg.norm(
                            position - target_starts[name])),
                        "lift_m": float(
                            position[2] - target_starts[name][2]),
                        "grasped": bool(sim.env.is_grasping(name)),
                    }
                telemetry.append({
                    "control": controls,
                    "predicted_action": predicted.astype(float).tolist(),
                    "commanded_action": command.astype(float).tolist(),
                    "executed_state": measured.astype(float).tolist(),
                    "tcp_position_m": tcp.astype(float).tolist(),
                    "targets": target_metrics,
                    "task_info": {
                        key: value for key, value in sim.env.task_info().items()
                        if isinstance(value, (bool, int, float, str))
                    },
                })
                if info["success"] and first_success is None:
                    first_success = controls
                if (j + 1) % stride == 0:
                    obs = sim.observe()
                    keyframes.append(obs)
                    frames.append(np.concatenate(
                        [obs[FRONT], obs[WRIST]], axis=1))
        cached_history = commanded_history
        normalized_history = model.preprocess_action(cached_history)
        normalized_history[:, ~model.action_mask.cpu()] = 0
        if model.chunk_idx == 0:
            normalized_history[:, :, 0] = 0
        model.last_predicted_actions = normalized_history.to(
            model.device, model.dtype)
        model.infer({
            "obs": keyframes,
            "compute_kv_cache": True,
            "state": cached_history,
        })
        chunk += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_dir / "rollout.mp4", frames, fps=15,
                     codec="libx264", quality=7, macro_block_size=8)
    commanded = [row["commanded_action"] for row in telemetry]
    result = {
        "id": episode["id"],
        "task": episode["task"],
        "seed": episode["seed"],
        "sample_seed": sample_seed,
        "instruction": episode["instruction"],
        "success": first_success is not None,
        "first_success_control": first_success,
        "controls": controls,
        "max_controls": max_controls,
        "oracle_controls": episode["robot_frames"],
        "chunks": chunk,
        "mean_chunk_infer_s": float(np.mean(chunk_seconds)),
        "wall_s": round(time.monotonic() - started, 1),
        "final_state_success": bool(sim.env.success()),
        "task_info": {k: v for k, v in sim.env.task_info().items()
                      if isinstance(v, (bool, int, float, str))},
        "wrist_transform": sim.wrist_transform["near_meters"],
        "motion": motion_diagnostics(actions),
        "commanded_motion": motion_diagnostics(commanded),
        "executed_motion": motion_diagnostics([
            row["executed_state"] for row in telemetry]),
        "action_ema_alpha": action_ema_alpha,
        "cache_action_history": "commanded",
        "targets": {
            name: {
                "min_tcp_distance_m": min(
                    row["targets"][name]["tcp_distance_m"]
                    for row in telemetry),
                "max_displacement_m": max(
                    row["targets"][name]["displacement_m"]
                    for row in telemetry),
                "max_lift_m": max(
                    row["targets"][name]["lift_m"] for row in telemetry),
                "grasp_controls": sum(
                    row["targets"][name]["grasped"] for row in telemetry),
            } for name in target_names
        } if telemetry else {},
        "start_pose_check": check,
        "seen_in_training": True,
        "training_provenance": {
            "source_index": "sim_train_v1",
            "episode_json_remote": episode["episode_json_remote"],
            "latent_remote": episode["latent_remote"],
            "visual_config_hash": episode.get("visual_config_hash"),
            "oracle_success": episode.get("oracle_success"),
        },
        "packed_batch": None,
    }
    np.save(out_dir / "actions.npy",
            np.asarray(actions, dtype=np.float32))
    with (out_dir / "telemetry.jsonl").open("w") as sink:
        for row in telemetry:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    args = _parse_args()
    from zero_wam.so101_closed_loop import load_model

    out_root = Path(args.output)
    all_rows = load_cohort()
    selected = select_episode_rows(all_rows, args.episode_id)
    with tempfile.TemporaryDirectory(prefix="so101-train-replay-model-") as temp:
        model, _ = load_model(args.base_model, args.checkpoint, temp, 50, 50)
        rows = []
        for ep in selected:
            max_controls = (
                args.max_controls
                if args.max_controls
                else math.ceil(
                    args.horizon_times_oracle * ep["robot_frames"]))
            print(f"[replay] {ep['id']} seed={ep['seed']} "
                  f"max_controls={max_controls}", flush=True)
            result = run_episode_train(
                model, ep, out_root / ep["id"],
                max_controls=max_controls,
                sample_seed=args.sample_seed,
                video_guidance_scale=-1.0,
                action_ema_alpha=args.alpha,
            )
            rows.append(result)
            print(f"  success={result['success']} "
                  f"controls={result['controls']} "
                  f"jitter={result['motion']['normalized_second_difference_rms']:.5f}",
                  flush=True)
    summary = {
        "schema": 1,
        "episodes": rows,
        "sample_seed": args.sample_seed,
        "alpha": args.alpha,
        "checkpoint": args.checkpoint,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"EXIT_OK episodes={len(rows)}")


if __name__ == "__main__":
    main()
