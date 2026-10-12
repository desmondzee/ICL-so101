"""Closed-loop SO-101 sim_val_v2 rollouts of a Zero-WAM checkpoint, conditioned on the paired human video.

Mirrors upstream `evaluation/robotwin/eval_policy_client_openpi.py`: the first chunk executes only its second
latent frame, later chunks execute both; a keyframe is observed every `action_per_frame // 4` controls and fed
back with the executed actions through `compute_kv_cache`. Wrist frames use the val_v3 near-plane transform.
"""

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


def run_episode(model, episode, out_dir, *, max_controls, sample_seed, video_guidance_scale=-1.0):
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
    frames = [np.concatenate([first_obs[FRONT], first_obs[WRIST]], axis=1)]
    actions, chunk_seconds, controls, chunk = [], [], 0, 0
    first_success = None
    while controls < max_controls and first_success is None:
        tick = time.monotonic()
        action = np.asarray(model.infer({"obs": first_obs})["action"], dtype=np.float32)
        chunk_seconds.append(time.monotonic() - tick)
        stride = action.shape[2] // 4
        keyframes = []
        for i in range(1 if chunk == 0 else 0, action.shape[1]):
            for j in range(action.shape[2]):
                row = action[:, i, j]
                info = sim.step(row)
                actions.append(row.tolist())
                controls += 1
                if info["success"] and first_success is None:
                    first_success = controls
                if (j + 1) % stride == 0:
                    obs = sim.observe()
                    keyframes.append(obs)
                    frames.append(np.concatenate([obs[FRONT], obs[WRIST]], axis=1))
        model.infer({"obs": keyframes, "compute_kv_cache": True, "state": action})
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
    }
    np.save(out_dir / "actions.npy", np.asarray(actions, dtype=np.float32))
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    sim.env.close()
    return result

