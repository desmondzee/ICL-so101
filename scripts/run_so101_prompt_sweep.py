#!/usr/bin/env python3
"""Run controlled Zero-WAM prompt contrasts locally on one H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "third_party/Zero-WAM"), str(REPO / "sim")]

from scripts.run_zero_wam_chunk_causal_validation import _assembled_model, _load_server
from zero_wam.prompt_sweep import (
    CheckpointSpec,
    TASK_OBJECTS,
    atomic_json,
    checkpoint_identity,
    condition_manifest,
    prompt_conditions,
    render_report,
    sha256_file,
)

FRONT = "observation.images.front"
WRIST = "observation.images.wrist"
OBJECTS = ("alphabet_soup", "cream_cheese", "tomato_sauce", "ketchup")
CHECKPOINTS = {
    "v1": CheckpointSpec(
        "v1_step500", "so101-sft-a100-b16-lr1e5-v1-500", 500,
        str(REPO / "data/models/checkpoints/v1_step500"),
    ),
    "v2b": CheckpointSpec(
        "v2b_step500", "so101-sft-a100-b32-cosine-v2b-1k-20261011", 500,
        str(REPO / "data/models/checkpoints/v2b_step500"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="v2b", help="comma-separated v1,v2b")
    parser.add_argument("--tasks", default="soup_lift,cheese_lift")
    parser.add_argument("--conditions", default="", help="empty means all eight")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--sample-seed", type=int, default=20261013)
    parser.add_argument("--max-controls", type=int, default=160)
    parser.add_argument("--video-steps", type=int, default=50)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument(
        "--human-video-root",
        default=str(REPO / "data/examples/humangen_so101_libero_basket"),
    )
    parser.add_argument(
        "--output", default=str(REPO / "outputs/zero_wam/prompt_sweep")
    )
    return parser.parse_args()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


class BasketSim:
    def __init__(self, seed: int):
        from libero_basket_env import LiberoBasketEnv, default_config
        from so101_nexus.observations import (
            EndEffectorPose,
            JointPositions,
            OverheadCamera,
            WristCamera,
        )

        config = default_config(288, 224)
        config.observations = [
            JointPositions(), EndEffectorPose(),
            OverheadCamera(width=288, height=224),
            WristCamera(width=288, height=224),
        ]
        self.env = LiberoBasketEnv(
            config=config, control_mode="pd_joint_pos", robot_init_qpos_noise=0
        )
        self.obs, _ = self.env.reset(seed=seed)
        self.starts = {name: self.env.object_pos(name).copy() for name in OBJECTS}

    def observe(self) -> dict:
        qpos = self.env._get_current_qpos().astype(np.float64)
        low, high = self.env._target_low[-1], self.env._target_high[-1]
        state = np.concatenate([
            np.degrees(qpos[:5]),
            [100.0 * (qpos[5] - low) / (high - low)],
        ]).astype(np.float32)
        return {
            FRONT: self.obs["overhead_camera"].copy(),
            WRIST: self.obs["wrist_camera"].copy(),
            "observation.state": state,
        }

    def step(self, lerobot_action: np.ndarray) -> dict:
        low, high = self.env._target_low[-1], self.env._target_high[-1]
        command = np.concatenate([
            np.radians(lerobot_action[:5]),
            [low + lerobot_action[5] / 100.0 * (high - low)],
        ]).astype(np.float32)
        self.obs, _, _, _, _ = self.env.step(command)
        return self.metrics()

    def metrics(self) -> dict:
        tcp = self.env._get_tcp_pose()[:3]
        return {
            name: {
                "position_m": self.env.object_pos(name).tolist(),
                "tcp_distance_m": float(np.linalg.norm(tcp - self.env.object_pos(name))),
                "displacement_m": float(
                    np.linalg.norm(self.env.object_pos(name) - self.starts[name])
                ),
                "lift_m": float(self.env.object_pos(name)[2] - self.starts[name][2]),
                "grasped": bool(self.env.is_grasping(name)),
                "in_basket": bool(self.env.in_basket(name)),
            }
            for name in OBJECTS
        }

    def initial_hashes(self) -> tuple[str, str]:
        state_hash = array_sha256(
            self.env.data.qpos, self.env.data.qvel, self.env.data.ctrl
        )
        observation = self.observe()
        observation_hash = array_sha256(
            observation[FRONT], observation[WRIST], observation["observation.state"]
        )
        return state_hash, observation_hash

    def close(self) -> None:
        self.env.close()


def prepare_video(
    root: Path, seed: int, task: str, shuffled: bool, output_root: Path
) -> dict:
    source = root / f"episode_{seed:03d}" / "video.mp4"
    if not source.is_file():
        raise FileNotFoundError(f"missing human video for environment seed {seed}: {source}")
    reader = imageio.get_reader(source)
    metadata = reader.get_meta_data()
    frames = [frame for frame in reader]
    reader.close()
    if len(frames) < 4:
        raise ValueError(f"human video is too short: {source}")
    midpoint = len(frames) // 2
    selected = frames[:midpoint] if task == "soup_lift" else frames[midpoint:]
    source_fps = float(metadata.get("fps", 30) or 30)
    target_fps = 12.0
    output_count = max(1, int(round(len(selected) / source_fps * target_fps)))
    indices = np.floor(
        np.arange(output_count, dtype=np.float64) * source_fps / target_fps
    ).astype(np.int64)
    selected = [selected[index] for index in np.unique(np.clip(
        indices, 0, len(selected) - 1
    ))]
    # Wan's streaming VAE requires one initial frame plus groups of four.
    selected = selected[: (len(selected) - 1) // 4 * 4 + 1]
    if shuffled:
        order = np.random.default_rng(seed + 731).permutation(len(selected))
        selected = [selected[index] for index in order]
    kind = f"{task}{'_shuffled' if shuffled else ''}"
    target = output_root / "prompt_videos" / f"seed{seed}_{kind}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        target, selected, fps=target_fps, codec="libx264",
        quality=8, macro_block_size=8,
    )
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "temporal_segment": "first_half" if task == "soup_lift" else "second_half",
        "shuffled": shuffled,
        "frame_count": len(selected),
    }


def set_sampling_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selected_object(records: list[dict]) -> str | None:
    for record in records:
        grasped = [
            name for name in ("alphabet_soup", "cream_cheese")
            if record["objects"][name]["grasped"]
        ]
        if grasped:
            return grasped[0]
    if not records:
        return None
    scores = {}
    for name in ("alphabet_soup", "cream_cheese"):
        scores[name] = (
            max(row["objects"][name]["lift_m"] for row in records)
            + max(row["objects"][name]["displacement_m"] for row in records)
            - min(row["objects"][name]["tcp_distance_m"] for row in records)
        )
    best = max(scores, key=scores.get)
    return best if scores[best] != scores[{"alphabet_soup": "cream_cheese",
                                           "cream_cheese": "alphabet_soup"}[best]] else None


def run_rollout(
    model, checkpoint: dict, task: str, condition, environment_seed: int,
    sample_seed: int, max_controls: int, video: dict | None, out: Path,
) -> dict:
    set_sampling_seed(sample_seed)
    sim = BasketSim(environment_seed)
    initial_state_hash, initial_observation_hash = sim.initial_hashes()
    manifest = condition_manifest(
        checkpoint=checkpoint, target_task=task, condition=condition,
        environment_seed=environment_seed, sample_seed=sample_seed,
        initial_state_sha256=initial_state_hash,
        initial_observation_sha256=initial_observation_hash,
        video=video, max_controls=max_controls,
    )
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "manifest.json", manifest)
    reset = condition.reset_payload(video["path"] if video else None)
    model.infer({"reset": True, **reset})
    first_obs = sim.observe()
    frames = [np.concatenate([first_obs[FRONT], first_obs[WRIST]], axis=1)]
    records, raw_chunks, chunk_times = [], [], []
    first = True
    target = TASK_OBJECTS[task]
    first_target_grasp = None
    first_target_success = None
    while len(records) < max_controls and first_target_success is None:
        tick = time.monotonic()
        raw = np.asarray(model.infer({"obs": first_obs})["action"], dtype=np.float32)
        chunk_times.append(time.monotonic() - tick)
        if raw.shape != (6, 2, 8):
            raise ValueError(f"unexpected model action shape: {raw.shape}")
        raw_chunks.append(raw.copy())
        executed = np.zeros_like(raw)
        keyframes = []
        stride = raw.shape[2] // 4
        for frame_index in range(1 if first else 0, raw.shape[1]):
            for control_index in range(raw.shape[2]):
                if len(records) >= max_controls or first_target_success is not None:
                    break
                predicted = raw[:, frame_index, control_index]
                objects = sim.step(predicted)
                measured = sim.observe()["observation.state"]
                executed[:, frame_index, control_index] = measured
                target_grasp = objects[target]["grasped"]
                target_success = target_grasp and objects[target]["lift_m"] >= 0.05
                if target_grasp and first_target_grasp is None:
                    first_target_grasp = len(records) + 1
                if target_success and first_target_success is None:
                    first_target_success = len(records) + 1
                records.append({
                    "control": len(records) + 1,
                    "predicted_action": predicted.tolist(),
                    "executed_state": measured.tolist(),
                    "objects": objects,
                    "target_grasp": target_grasp,
                    "target_success": target_success,
                })
                if (control_index + 1) % stride == 0:
                    observed = sim.observe()
                    keyframes.append(observed)
                    frames.append(np.concatenate([observed[FRONT], observed[WRIST]], axis=1))
            if len(records) >= max_controls or first_target_success is not None:
                break
        first = False
        if len(records) < max_controls and first_target_success is None:
            normalized = model.preprocess_action(executed)
            normalized[:, ~model.action_mask.cpu()] = 0
            if model.chunk_idx == 0:
                normalized[:, :, 0] = 0
            model.last_predicted_actions = normalized.to(model.device, model.dtype)
            model.infer({"obs": keyframes, "compute_kv_cache": True, "state": executed})
        first_obs = sim.observe()
    final_objects = sim.metrics()
    sim.close()
    imageio.mimwrite(
        out / "rollout.mp4", frames, fps=15, codec="libx264",
        quality=7, macro_block_size=8,
    )
    np.savez_compressed(
        out / "trajectory.npz",
        predicted_chunks=np.asarray(raw_chunks, dtype=np.float32),
        executed_states=np.asarray(
            [record["executed_state"] for record in records], dtype=np.float32
        ),
    )
    with (out / "actions.jsonl").open("w") as sink:
        for record in records:
            sink.write(json.dumps(record, sort_keys=True) + "\n")
    result = {
        "schema": 1,
        "checkpoint": checkpoint["name"],
        "checkpoint_config_sha256": checkpoint["config_sha256"],
        "target_task": task,
        "target_object": target,
        "condition": condition.name,
        "environment_seed": environment_seed,
        "sample_seed": sample_seed,
        "initial_state_sha256": initial_state_hash,
        "initial_observation_sha256": initial_observation_hash,
        "controls": len(records),
        "chunks": len(raw_chunks),
        "mean_chunk_infer_s": float(np.mean(chunk_times)) if chunk_times else None,
        "selected_object": selected_object(records),
        "target_grasp_success": first_target_grasp is not None,
        "first_target_grasp_control": first_target_grasp,
        "task_success": first_target_success is not None,
        "first_task_success_control": first_target_success,
        "objects": {
            name: {
                "max_lift_m": max(
                    [row["objects"][name]["lift_m"] for row in records] or [0.0]
                ),
                "max_displacement_m": max(
                    [row["objects"][name]["displacement_m"] for row in records] or [0.0]
                ),
                "min_tcp_distance_m": min(
                    [row["objects"][name]["tcp_distance_m"] for row in records]
                    or [float("inf")]
                ),
                "grasp_controls": sum(
                    row["objects"][name]["grasped"] for row in records
                ),
                "final": final_objects[name],
            }
            for name in OBJECTS
        },
        "manifest_sha256": manifest["manifest_sha256"],
    }
    atomic_json(out / "result.json", result)
    return result


def load_model(spec: CheckpointSpec, video_steps: int, action_steps: int):
    base = REPO / "data/models/zero-wam-pretrain"
    temporary = tempfile.TemporaryDirectory(prefix=f"so101-{spec.name}-")
    assembled = _assembled_model(base, spec.path, temporary.name)
    model, evidence, config = _load_server(assembled, "cuda:0")
    config.num_inference_steps = video_steps
    config.action_num_inference_steps = action_steps
    original_icl_loader = model._load_or_encode_icl

    def load_or_encode_icl(video_path: str, latent_path: str):
        if latent_path:
            return original_icl_loader(video_path, latent_path)
        video = model._read_icl_video(video_path).to(
            device=next(model.vae.parameters()).device, dtype=model.dtype
        )
        video = video / 255.0 * 2.0 - 1.0
        if (video.shape[2] - 1) % 4:
            raise ValueError(
                f"ICL video must contain 1+4k frames, got {video.shape[2]}"
            )
        model.streaming_vae.clear_cache()
        chunks = [model.streaming_vae.encode_chunk(video[:, :, :1])]
        for start in range(1, video.shape[2], 4):
            chunks.append(
                model.streaming_vae.encode_chunk(video[:, :, start:start + 4])
            )
        mu, _ = torch.chunk(torch.cat(chunks, dim=2), 2, dim=1)
        mean = torch.tensor(model.vae.config.latents_mean, device=mu.device)
        std = torch.tensor(model.vae.config.latents_std, device=mu.device)
        latent = model.normalize_latents(mu, mean, 1.0 / std).cpu()
        model.streaming_vae.clear_cache()
        return latent, None

    model._load_or_encode_icl = load_or_encode_icl
    return temporary, model, evidence


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    requested_conditions = set(filter(None, args.conditions.split(",")))
    rows = []
    for checkpoint_name in args.checkpoints.split(","):
        spec = CHECKPOINTS[checkpoint_name]
        identity = checkpoint_identity(spec)
        temporary, model, load_evidence = load_model(
            spec, args.video_steps, args.action_steps
        )
        identity["model_load"] = load_evidence
        atomic_json(output / spec.name / "checkpoint.json", identity)
        try:
            for task in args.tasks.split(","):
                conditions = prompt_conditions(task)
                if requested_conditions:
                    conditions = [
                        item for item in conditions if item.name in requested_conditions
                    ]
                for seed_text in args.seeds.split(","):
                    environment_seed = int(seed_text)
                    for condition in conditions:
                        sample_seed = args.sample_seed + environment_seed
                        video = None
                        if condition.use_video:
                            video = prepare_video(
                                Path(args.human_video_root), environment_seed,
                                condition.video_task, condition.shuffle_video, output,
                            )
                        out = (
                            output / spec.name / task / f"seed{environment_seed}"
                            / condition.name
                        )
                        print(
                            f"{spec.name} {task} seed={environment_seed} "
                            f"{condition.name}", flush=True,
                        )
                        result = run_rollout(
                            model, identity, task, condition, environment_seed,
                            sample_seed, args.max_controls, video, out,
                        )
                        rows.append(result)
                        print(
                            f"  selected={result['selected_object']} "
                            f"grasp={result['target_grasp_success']} "
                            f"success={result['task_success']}", flush=True,
                        )
        finally:
            temporary.cleanup()
            del model
            torch.cuda.empty_cache()
    atomic_json(output / "summary.json", {"schema": 1, "rollouts": rows})
    report_rows = []
    for path in sorted(output.glob("**/result.json")):
        row = json.loads(path.read_text())
        row["_result_path"] = str(path)
        report_rows.append(row)
    report = output / "report.md"
    report.write_text(render_report(output, report_rows))
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
