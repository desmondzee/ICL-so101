from __future__ import annotations

import json
import random
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from sim.rebuild_wrist_v2 import WRIST_CAMERA, clean_wrist_option, configure_clean_wrist
from sim.val.tasks import load as load_task

FRONT = "observation.images.front"
WRIST = "observation.images.wrist"
TASK_TARGETS = {
    "mug_on_plate": ("red_mug",),
    "mugs_in_microwave": ("red_mug", "white_mug"),
    "pan_on_stove": ("pan",),
    "sort_blocks": ("red_block", "blue_block"),
    "stack_bowls": ("white_bowl",),
}


def motion_diagnostics(actions):
    """Summarize command oscillation without conflating it with task success."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"expected [controls, 6] actions, got {actions.shape}")
    velocity = np.diff(actions, axis=0)
    acceleration = np.diff(velocity, axis=0)
    reversals = (
        np.sign(velocity[1:]) * np.sign(velocity[:-1]) < 0
        if len(velocity) > 1 else np.zeros((0, 6), dtype=bool)
    )
    arm_scale = np.array([220, 200, 193.66, 190, 314.42, 100.0])
    normalized_velocity = velocity / arm_scale
    normalized_acceleration = acceleration / arm_scale
    return {
        "action_delta_rms": float(np.sqrt(np.mean(np.square(velocity)))) if velocity.size else 0.0,
        "action_second_difference_rms": float(np.sqrt(np.mean(np.square(acceleration)))) if acceleration.size else 0.0,
        "normalized_delta_rms": float(np.sqrt(np.mean(np.square(normalized_velocity)))) if velocity.size else 0.0,
        "normalized_second_difference_rms": float(np.sqrt(np.mean(np.square(normalized_acceleration)))) if acceleration.size else 0.0,
        "direction_reversals": int(reversals.sum()),
        "direction_reversal_fraction": float(reversals.mean()) if reversals.size else 0.0,
        "per_channel_delta_rms": np.sqrt(np.mean(np.square(velocity), axis=0)).tolist() if velocity.size else [0.0] * 6,
        "per_channel_second_difference_rms": np.sqrt(np.mean(np.square(acceleration), axis=0)).tolist() if acceleration.size else [0.0] * 6,
    }


def smooth_action(target, previous, alpha=1.0):
    if not 0 < alpha <= 1:
        raise ValueError(f"action EMA alpha must be in (0, 1], got {alpha}")
    target = np.asarray(target, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32)
    if target.shape != (6,) or previous.shape != (6,):
        raise ValueError(f"expected two six-channel actions, got {target.shape}/{previous.shape}")
    return previous + alpha * (target - previous)


def load_model(base_model, checkpoint, workdir, video_steps=50, action_steps=50):
    from scripts.run_zero_wam_chunk_causal_validation import _assembled_model, _load_server

    model, load_evidence, config = _load_server(_assembled_model(base_model, checkpoint, workdir), "cuda:0")
    config.num_inference_steps, config.action_num_inference_steps = video_steps, action_steps
    return model, load_evidence


class Sim:
    def __init__(self, task, seed):
        env_cls, _ = load_task(task)
        self.env = env_cls(render_images=False)
        self.wrist_transform = configure_clean_wrist(self.env.model)
        self.option = clean_wrist_option()
        self.env.reset(seed=seed)
        self.success = bool(self.env.success())

    def observe(self):
        env = self.env
        env._renderer_rgb.update_scene(env.data, camera="front", scene_option=self.option)
        front = env._renderer_rgb.render().copy()
        env._renderer_rgb.update_scene(env.data, camera=WRIST_CAMERA, scene_option=self.option)
        wrist = env._renderer_rgb.render().copy()
        return {FRONT: front, WRIST: wrist, "observation.state": env.lerobot_joints().astype(np.float32)}

    def step(self, row):
        _, _, _, _, info = self.env.step(self.env.from_lerobot_joints(row).astype(np.float32))
        self.success = self.success or bool(info["success"])
        return info


def run_episode(
    model, episode, out_dir, *, max_controls, sample_seed,
    video_guidance_scale=-1.0, action_ema_alpha=1.0,
    cache_action_history="commanded",
):
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    torch.cuda.manual_seed_all(sample_seed)
    started = time.monotonic()
    sim = Sim(episode["task"], episode["seed"])
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
    target_names = TASK_TARGETS.get(episode["task"], ())
    target_starts = {
        name: sim.env.object_pos(name).copy() for name in target_names
    }
    first_success = None
    while controls < max_controls and first_success is None:
        tick = time.monotonic()
        action = np.asarray(model.infer({"obs": first_obs})["action"], dtype=np.float32)
        executed_history = np.zeros_like(action)
        commanded_history = np.zeros_like(action)
        chunk_seconds.append(time.monotonic() - tick)
        stride = action.shape[2] // 4
        keyframes = []
        for i in range(1 if chunk == 0 else 0, action.shape[1]):
            for j in range(action.shape[2]):
                predicted = action[:, i, j]
                command = smooth_action(predicted, previous_command, action_ema_alpha)
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
                        "displacement_m": float(np.linalg.norm(position - target_starts[name])),
                        "lift_m": float(position[2] - target_starts[name][2]),
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
                    frames.append(np.concatenate([obs[FRONT], obs[WRIST]], axis=1))
        histories = {
            "commanded": commanded_history,
            "measured": executed_history,
        }
        if cache_action_history not in histories:
            raise ValueError(
                f"cache_action_history must be one of {tuple(histories)}, "
                f"got {cache_action_history!r}"
            )
        cached_history = histories[cache_action_history]
        normalized_history = model.preprocess_action(cached_history)
        normalized_history[:, ~model.action_mask.cpu()] = 0
        if model.chunk_idx == 0:
            normalized_history[:, :, 0] = 0
        model.last_predicted_actions = normalized_history.to(
            model.device, model.dtype
        )
        model.infer({
            "obs": keyframes,
            "compute_kv_cache": True,
            "state": cached_history,
        })
        chunk += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_dir / "rollout.mp4", frames, fps=15, codec="libx264", quality=7, macro_block_size=8)
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
        "task_info": {k: v for k, v in sim.env.task_info().items() if isinstance(v, (bool, int, float, str))},
        "wrist_transform": sim.wrist_transform["near_meters"],
        "motion": motion_diagnostics(actions),
        "executed_motion": motion_diagnostics([
            row["executed_state"] for row in telemetry
        ]),
        "action_ema_alpha": action_ema_alpha,
        "cache_action_history": cache_action_history,
        "targets": {
            name: {
                "min_tcp_distance_m": min(
                    row["targets"][name]["tcp_distance_m"] for row in telemetry
                ),
                "max_displacement_m": max(
                    row["targets"][name]["displacement_m"] for row in telemetry
                ),
                "max_lift_m": max(
                    row["targets"][name]["lift_m"] for row in telemetry
                ),
                "grasp_controls": sum(
                    row["targets"][name]["grasped"] for row in telemetry
                ),
            }
            for name in target_names
        },
    }
    np.save(out_dir / "actions.npy", np.asarray(actions, dtype=np.float32))
    with (out_dir / "telemetry.jsonl").open("w") as sink:
        for row in telemetry:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    sim.env.close()
    return result
