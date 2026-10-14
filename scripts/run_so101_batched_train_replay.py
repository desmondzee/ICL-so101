"""Batch-4 packed replay of exact TRAINING episodes (opt-in checkpoint probe).

Separate runner — never imported by existing entry points.  Replays six
canonical sim_train_v1 episodes (2 per task: ball_between_cans,
block_in_bowl, push_cube_into_tape_square) through BatchedStaticEngine with
EMA alpha=0.9.  Sim construction mirrors `sim.rebuild_wrist_v2._make_env`
(train split): task-definition env class + frozen `visual_config` + index
seed; start poses are verified against the packaged `object_poses_start`
before any model execution.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

GROUP_SIZE = 4
SAMPLE_SEED_BASE = 20261101
DATA = Path(__file__).resolve().parents[1] / "data/hf_bucket_ICL-so101/train_replay_probe"
COHORT_IDS = (
    "ball_between_cans/episode_000",
    "ball_between_cans/episode_001",
    "block_in_bowl/episode_000",
    "block_in_bowl/episode_001",
    "push_cube_into_tape_square/episode_000",
    "push_cube_into_tape_square/episode_001",
)
START_POSE_TOLERANCE_M = 0.002


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="data/models/checkpoints/v2b_step500")
    p.add_argument("--base-model", default="data/models/zero-wam-pretrain")
    p.add_argument("--output",
                   default="outputs/zero_wam/train_replay_v2b_b4_alpha090_3tasks_2seeds")
    p.add_argument("--max-tokens", type=int, default=98304)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--horizon-times-oracle", type=float, default=2.0)
    p.add_argument("--max-controls", type=int, default=None)
    p.add_argument("--episode-id", action="append", default=[])
    p.add_argument("--safety-fraction", type=float, default=0.85)
    return p.parse_args()


def load_cohort(data_root=DATA):
    manifest = json.loads(
        (Path(data_root) / "cohort_manifest.json").read_text())
    rows = {e["id"]: e for e in manifest["episodes"]}
    return [rows[i] for i in COHORT_IDS]


def select_episode_rows(all_rows, ids):
    if not ids:
        return list(all_rows)
    wanted = set(ids)
    chosen = [ep for ep in all_rows if ep["id"] in wanted]
    if len(chosen) != len(wanted):
        missing = sorted(wanted - {ep["id"] for ep in chosen})
        raise ValueError(f"episode ids not in cohort: {missing}")
    return chosen


class TrainSim:
    """so101_closed_loop.Sim equivalent for sim_train_v1 envs."""

    def __init__(self, task, seed, visual_config):
        from sim.rebuild_wrist_v2 import (
            WRIST_CAMERA,
            clean_wrist_option,
            configure_clean_wrist,
        )
        from sim.train.tasks.catalog import load_train_task
        from sim.train.variation import VisualConfig

        self.task_def = load_train_task(task)
        env_cls, _ = self.task_def.load_classes()
        config = (
            VisualConfig.from_dict(visual_config)
            if isinstance(visual_config, dict)
            else visual_config
        )
        self.env = env_cls(render_images=False, visual_config=config)
        self.wrist_camera = WRIST_CAMERA
        self.wrist_transform = configure_clean_wrist(self.env.model)
        self.option = clean_wrist_option()
        self.env.reset(seed=seed)
        self.success = bool(self.env.success())

    @property
    def task_objects(self):
        return tuple(self.task_def.task_objects)

    def observe(self):
        from zero_wam.so101_closed_loop import FRONT, WRIST

        env = self.env
        env._renderer_rgb.update_scene(
            env.data, camera="front", scene_option=self.option)
        front = env._renderer_rgb.render().copy()
        env._renderer_rgb.update_scene(
            env.data, camera=self.wrist_camera, scene_option=self.option)
        wrist = env._renderer_rgb.render().copy()
        return {
            FRONT: front,
            WRIST: wrist,
            "observation.state": env.lerobot_joints().astype(np.float32),
        }

    def step(self, row):
        _, _, _, _, info = self.env.step(
            self.env.from_lerobot_joints(row).astype(np.float32))
        self.success = self.success or bool(info["success"])
        return info


def check_start_poses(sim, frozen_poses, tolerance=START_POSE_TOLERANCE_M):
    """Max |xyz| error of each task object vs packaged object_poses_start."""
    actual = sim.env.object_poses()
    diffs, missing = {}, []
    for name in sim.task_objects:
        if name not in actual:
            missing.append(name)
            continue
        if name not in frozen_poses:
            continue
        diffs[name] = float(np.max(np.abs(
            np.asarray(actual[name], float)[:3]
            - np.asarray(frozen_poses[name], float)[:3])))
    return {
        "task_objects": list(sim.task_objects),
        "compared": len(diffs),
        "missing": missing,
        "max_abs_error_m": max(diffs.values()) if diffs else None,
        "per_object_m": diffs,
        "pass": bool(diffs) and not missing
        and max(diffs.values()) <= tolerance,
        "tolerance_m": tolerance,
    }


def _write_episode(out_dir, r, ep, sample_seed, alpha, infer_times,
                   group_id, slot_order, wall_s):
    from zero_wam.so101_closed_loop import (
        FRONT,
        WRIST,
        motion_diagnostics,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_dir / "rollout.mp4", r["frames"], fps=15,
                     codec="libx264", quality=7, macro_block_size=8)
    sim = r["sim"]
    target_names = sim.task_objects
    commanded = [row["commanded_action"] for row in r["telemetry"]]
    result = {
        "id": ep["id"],
        "task": ep["task"],
        "seed": ep["seed"],
        "sample_seed": sample_seed,
        "instruction": ep["instruction"],
        "success": r["first_success"] is not None,
        "first_success_control": r["first_success"],
        "controls": r["controls"],
        "max_controls": r["horizon"],
        "oracle_controls": ep["robot_frames"],
        "chunks": r["chunk"],
        "mean_chunk_infer_s": (
            float(np.mean(infer_times)) if infer_times else None),
        "final_state_success": bool(sim.env.success()),
        "task_info": {k: v for k, v in sim.env.task_info().items()
                      if isinstance(v, (bool, int, float, str))},
        "wrist_transform": sim.wrist_transform["near_meters"],
        "motion": motion_diagnostics(r["actions"]),
        "commanded_motion": motion_diagnostics(commanded),
        "executed_motion": motion_diagnostics([
            row["executed_state"] for row in r["telemetry"]
        ]),
        "action_ema_alpha": alpha,
        "cache_action_history": "commanded",
        "targets": {
            name: {
                "min_tcp_distance_m": min(
                    row["targets"][name]["tcp_distance_m"]
                    for row in r["telemetry"]),
                "max_displacement_m": max(
                    row["targets"][name]["displacement_m"]
                    for row in r["telemetry"]),
                "max_lift_m": max(
                    row["targets"][name]["lift_m"]
                    for row in r["telemetry"]),
                "grasp_controls": sum(
                    row["targets"][name]["grasped"]
                    for row in r["telemetry"]),
            }
            for name in target_names
        } if r["telemetry"] else {},
        "start_pose_check": r["start_pose_check"],
        "seen_in_training": True,
        "training_provenance": {
            "source_index": "sim_train_v1",
            "episode_json_remote": ep["episode_json_remote"],
            "latent_remote": ep["latent_remote"],
            "visual_config_hash": ep.get("visual_config_hash"),
            "oracle_success": ep.get("oracle_success"),
        },
        "packed_batch": {
            "batch_size": GROUP_SIZE,
            "group_id": group_id,
            "slot": r["slot"],
            "slot_episode_ids": slot_order,
        },
        "wall_s": wall_s,
    }
    np.save(out_dir / "actions.npy",
            np.asarray(r["actions"], dtype=np.float32))
    with (out_dir / "telemetry.jsonl").open("w") as sink:
        for row in r["telemetry"]:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result


def _existing_results(out_root):
    root = Path(out_root) / "v2b_step500"
    if not root.is_dir():
        return []
    return sorted(
        str(p.parent.relative_to(root)) for p in root.rglob("result.json"))


def main():
    args = _parse_args()
    from scripts.run_zero_wam_chunk_causal_validation import (
        _assembled_model,
        _load_server,
    )
    from zero_wam.batched_static_cache import (
        StaticKVCacheConfig,
        estimate_static_memory,
        static_cache_diagnostics,
    )
    from zero_wam.batched_static_engine import (
        BatchedStaticEngine,
        SlotResetSpec,
    )
    from zero_wam.prompt_sweep import atomic_json
    from zero_wam.so101_closed_loop import FRONT, WRIST, smooth_action

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(*a, flush=True)

    cfg = StaticKVCacheConfig(max_tokens=args.max_tokens)
    estimate = estimate_static_memory(cfg, 30)
    torch.cuda.empty_cache()
    free_b, total_b = torch.cuda.mem_get_info(0)
    preflight = {
        "max_tokens": args.max_tokens,
        "estimate_bytes": estimate,
        "estimated_gib": round(estimate["total_bytes"] / 2**30, 2),
        "free_gib": round(free_b / 2**30, 2),
        "total_gib": round(total_b / 2**30, 2),
    }
    preflight["allowed"] = free_b > estimate["total_bytes"] * 1.05
    log(f"[preflight] {json.dumps(preflight)}")
    if not preflight["allowed"]:
        raise MemoryError("static estimate does not fit free VRAM")

    with tempfile.TemporaryDirectory() as temp:
        assembled = _assembled_model(args.base_model, args.checkpoint, temp)
        model, load_evidence, config = _load_server(assembled, "cuda:0")
    model.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    all_rows = load_cohort()
    selected = select_episode_rows(all_rows, args.episode_id)
    groups = [selected[i:i + GROUP_SIZE]
              for i in range(0, len(selected), GROUP_SIZE)]
    atomic_json(out_root / "episodes.json", all_rows)
    atomic_json(out_root / "cohort.json", selected)

    engine = BatchedStaticEngine(
        model, max_tokens=args.max_tokens, batch_size=GROUP_SIZE,
        video_guidance_scale=-1.0,
        icl_guidance_scale=float(config.icl_guidance_scale),
        use_icl=True, safety_fraction=args.safety_fraction,
    )

    spec_name = Path(args.checkpoint).name
    started = time.monotonic()
    rows, errors, infer_times = [], [], []

    for group_id, group in enumerate(groups):
        global_idx = [all_rows.index(ep) for ep in group]
        specs = [
            SlotResetSpec(
                prompt=ep["instruction"],
                icl_latent_path=ep["human_latent"],
                seed=SAMPLE_SEED_BASE + gidx,
                active=True,
            )
            for ep, gidx in zip(group, global_idx)
        ]
        for _ in range(GROUP_SIZE - len(group)):
            specs.append(SlotResetSpec(
                prompt="pad", icl_latent_path=None, seed=0, active=False))
        log(f"[group {group_id}] real={len(group)} pad={GROUP_SIZE-len(group)}"
            f" episodes={[ep['id'] for ep in group]}")
        model.text_encoder.to("cuda")
        engine.reset(specs)
        model.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        log(f"[group {group_id}] post-reset free "
            f"{torch.cuda.mem_get_info(0)[0]/2**30:.1f} GiB")

        rollouts = []
        for s, (ep, gidx) in enumerate(zip(group, global_idx)):
            sim = TrainSim(ep["task"], ep["seed"], ep["visual_config"])
            check = check_start_poses(sim, ep["object_poses_start"])
            log(f"  [slot {s}] {ep['id']} seed={ep['seed']} "
                f"vc={ep.get('visual_config_hash')} start_poses: "
                f"compared={check['compared']} "
                f"max_err={check['max_abs_error_m']} "
                f"pass={check['pass']}")
            if not check["pass"]:
                errors.append({
                    "id": ep["id"], "group": group_id,
                    "error": "start_pose_mismatch",
                    "check": check,
                })
                sim.env.close()
                continue
            first_obs = sim.observe()
            targets = sim.task_objects
            rollouts.append({
                "episode": ep,
                "slot": s,
                "sample_seed": SAMPLE_SEED_BASE + gidx,
                "sim": sim,
                "start_pose_check": check,
                "prev_command": first_obs["observation.state"].copy(),
                "first_obs": first_obs,
                "frames": [np.concatenate(
                    [first_obs[FRONT], first_obs[WRIST]], axis=1)],
                "actions": [],
                "telemetry": [],
                "controls": 0,
                "chunk": 0,
                "first_success": None,
                "horizon": min(
                    int(math.ceil(
                        args.horizon_times_oracle * ep["robot_frames"])),
                    args.max_controls or 10**9,
                ),
                "active": True,
                "target_starts": {
                    name: sim.env.object_pos(name).copy() for name in targets
                },
            })
        # slots whose start-pose check failed are deactivated before inference
        active_slots = {r["slot"] for r in rollouts}
        for s in range(len(group)):
            if s not in active_slots:
                engine.set_active(s, False)
        if not rollouts:
            continue

        def checkpoint_summary(packed_chunk):
            occ = static_cache_diagnostics(model.transformer)
            atomic_json(out_root / "summary.json", {
                "packed_chunks_completed": packed_chunk,
                "group": group_id,
                "elapsed_s": round(time.monotonic() - started, 1),
                "peak_gib": round(
                    torch.cuda.max_memory_allocated() / 2**30, 2),
                "cache_occupancy_tokens": occ["metadata"]["length"],
                "cache_capacity_tokens": occ["metadata"]["capacity"],
                "episodes_done": sorted(
                    set(_existing_results(out_root))
                    | {r["id"] for r in rows}),
                "slots": [{
                    "episode": r["episode"]["id"], "slot": r["slot"],
                    "controls": r["controls"], "chunks": r["chunk"],
                    "first_success": r["first_success"],
                    "horizon": r["horizon"], "active": r["active"],
                } for r in rollouts],
            })

        torch.cuda.reset_peak_memory_stats()
        obs0 = [None] * GROUP_SIZE
        for r in rollouts:
            obs0[r["slot"]] = {"obs": r["first_obs"]}
        predicted = engine.infer(obs0)
        packed_chunk = 0
        slot_order = [ep["id"] for ep in group] + [None] * (
            GROUP_SIZE - len(group))

        while any(r["active"] for r in rollouts):
            t_chunk = time.monotonic()
            for r in rollouts:
                if not r["active"]:
                    continue
                action = np.asarray(predicted[r["slot"]], dtype=np.float32)
                commanded = np.zeros_like(action)
                r["keyframes"] = []
                stride = action.shape[2] // 4
                sim = r["sim"]
                for i in range(1 if r["chunk"] == 0 else 0, action.shape[1]):
                    for j in range(action.shape[2]):
                        pred = action[:, i, j]
                        command = smooth_action(
                            pred, r["prev_command"], args.alpha)
                        info = sim.step(command)
                        measured = sim.env.lerobot_joints().astype(np.float32)
                        commanded[:, i, j] = command
                        r["prev_command"] = command
                        r["actions"].append(pred.tolist())
                        r["controls"] += 1
                        tcp = sim.env._get_tcp_pose()[:3]
                        target_metrics = {}
                        for name in sim.task_objects:
                            pos = sim.env.object_pos(name)
                            target_metrics[name] = {
                                "tcp_distance_m": float(
                                    np.linalg.norm(tcp - pos)),
                                "displacement_m": float(np.linalg.norm(
                                    pos - r["target_starts"][name])),
                                "lift_m": float(
                                    pos[2] - r["target_starts"][name][2]),
                                "grasped": bool(sim.env.is_grasping(name)),
                            }
                        r["telemetry"].append({
                            "control": r["controls"],
                            "predicted_action": pred.astype(float).tolist(),
                            "commanded_action": command.astype(
                                float).tolist(),
                            "executed_state": measured.astype(float).tolist(),
                            "tcp_position_m": tcp.astype(float).tolist(),
                            "targets": target_metrics,
                            "task_info": {
                                k: v for k, v in sim.env.task_info().items()
                                if isinstance(v, (bool, int, float, str))
                            },
                        })
                        if info["success"] and r["first_success"] is None:
                            r["first_success"] = r["controls"]
                        if (j + 1) % stride == 0:
                            obs = sim.observe()
                            r["keyframes"].append(obs)
                            r["frames"].append(np.concatenate(
                                [obs[FRONT], obs[WRIST]], axis=1))
                r["commanded_hist"] = commanded
                r["chunk"] += 1
                r["active"] = (
                    r["controls"] < r["horizon"]
                    and r["first_success"] is None
                )
            step_s = time.monotonic() - t_chunk

            obs_lists, histories = [], []
            for s in range(GROUP_SIZE):
                r = next((x for x in rollouts if x["slot"] == s), None)
                if r is not None and r["active"]:
                    obs_lists.append({"obs": r["keyframes"]})
                    hist = model.preprocess_action(r["commanded_hist"])
                    hist[:, ~model.action_mask.cpu()] = 0
                    if engine.slots[s].chunk_idx == 0:
                        hist[:, :, 0] = 0
                    histories.append(hist.to(model.device, model.dtype))
                else:
                    obs_lists.append(None)
                    histories.append(None)

            if not any(r["active"] for r in rollouts):
                packed_chunk += 1
                checkpoint_summary(packed_chunk)
                break
            t0 = time.monotonic()
            engine.cache_update(obs_lists, histories)
            for r in rollouts:
                engine.set_active(r["slot"], r["active"])
            t_update = time.monotonic() - t0
            t0 = time.monotonic()
            predicted = engine.infer()
            t_infer = time.monotonic() - t0
            infer_times.append(t_infer)
            packed_chunk += 1
            occ = static_cache_diagnostics(model.transformer)
            log(f"[group {group_id} packed-chunk {packed_chunk}] "
                f"step {step_s:.1f}s update {t_update:.1f}s "
                f"infer {t_infer:.1f}s "
                f"controls={[r['controls'] for r in rollouts]} "
                f"occupancy={occ['metadata']['length']}/"
                f"{occ['metadata']['capacity']} "
                f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
            checkpoint_summary(packed_chunk)

        group_wall = round(time.monotonic() - started, 1)
        for r in rollouts:
            try:
                result = _write_episode(
                    out_root / spec_name / r["episode"]["id"],
                    r, r["episode"], r["sample_seed"], args.alpha,
                    infer_times, group_id, slot_order, group_wall)
                rows.append(result)
                log(f"  done {r['episode']['id']} slot={r['slot']} "
                    f"success={result['success']} "
                    f"controls={result['controls']}")
            except Exception as exc:  # noqa: BLE001
                errors.append({"id": r["episode"]["id"],
                               "group": group_id, "error": repr(exc)})
                log(f"  ERROR {r['episode']['id']}: {exc!r}")
            r["sim"].env.close()
        checkpoint_summary(packed_chunk)

    manifest = {
        "schema": "train_replay_v1",
        "output": str(out_root),
        "checkpoint": args.checkpoint,
        "batch_size": GROUP_SIZE,
        "max_tokens": args.max_tokens,
        "alpha": args.alpha,
        "horizon_times_oracle": args.horizon_times_oracle,
        "max_controls": args.max_controls,
        "groups": [
            {"group": g, "episode_ids": [ep["id"] for ep in grp]}
            for g, grp in enumerate(groups)
        ],
        "sample_seed_base": SAMPLE_SEED_BASE,
        "episodes": [{"id": ep["id"],
                      "sample_seed": SAMPLE_SEED_BASE + i,
                      "seed": ep["seed"],
                      "visual_config_hash": ep.get("visual_config_hash")}
                     for i, ep in enumerate(all_rows)],
        "episodes_done": sorted(
            set(_existing_results(out_root)) | {r["id"] for r in rows}),
        "preflight": preflight,
        "engine_calls": engine.calls,
        "adapter": static_cache_diagnostics(model.transformer),
        "load_evidence": {
            k: (v if isinstance(v, (int, float, str)) else str(v)[:200])
            for k, v in load_evidence.items()
        },
        "wall_s": round(time.monotonic() - started, 1),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "successes": sum(1 for r in rows if r["success"]),
        "errors": errors,
        "seen_in_training": True,
    }
    atomic_json(out_root / "batch_manifest.json", manifest)
    print(f"EXIT_OK wall={manifest['wall_s']}s episodes={len(rows)} "
          f"successes={manifest['successes']} "
          f"peak={manifest['peak_gib']}GiB")


if __name__ == "__main__":
    main()
