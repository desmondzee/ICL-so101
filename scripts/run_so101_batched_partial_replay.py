"""Batch-four counterpart of the frozen partial-chunk training replay protocol."""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_so101_batched_train_replay import (
    DATA, TrainSim, _write_episode, check_start_poses, load_cohort, select_episode_rows,
)
from scripts.run_so101_partial_chunk_replay import blend_at, verify_artifacts
from zero_wam.prompt_sweep import atomic_json, sha256_file
from zero_wam.so101_closed_loop import FRONT, WRIST, load_model

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast-frames", type=int, choices=(2, 4), required=True)
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode-id", action="append", default=[])
    p.add_argument("--max-controls", type=int)
    p.add_argument("--reference-history", type=Path,
                   help="Diagnostic only: slot 0 follows recorded sequential commands.")
    args = p.parse_args()
    if args.max_controls is not None and args.max_controls < 32:
        p.error("bounded runs require >=32 controls")
    if args.reference_history and (args.forecast_frames != 2 or args.ensemble or
                                   not args.max_controls):
        p.error("reference-history diagnostic requires F2, raw, and bounded horizon")
    from zero_wam.batched_partial_chunk_engine import BatchedPartialChunkEngine
    from zero_wam.batched_static_engine import SlotResetSpec
    all_rows = load_cohort()
    selected = select_episode_rows(all_rows, args.episode_id)
    reference = None
    if args.reference_history:
        ref_result = json.loads((args.reference_history / "result.json").read_text())
        if selected[0]["id"] != ref_result["id"]:
            raise ValueError("reference must match first slot episode")
        if not ref_result["scene_replay"]["check"]["pass"]:
            raise ValueError("reference scene failed")
        reference = [json.loads(x)["commanded_action"] for x in
                     (args.reference_history / "telemetry.jsonl").read_text().splitlines()]
        if len(reference) < args.max_controls:
            raise ValueError("reference too short")
    checkpoint = REPO / "data/models/checkpoints/v2b_step500"
    sources = [
        Path(__file__), REPO / "zero_wam/batched_partial_chunk_engine.py",
        REPO / "zero_wam/batched_static_engine.py", REPO / "zero_wam/batched_static_cache.py",
        REPO / "scripts/run_so101_partial_chunk_replay.py",
        REPO / "scripts/run_so101_batched_train_replay.py",
        REPO / "zero_wam/train_scene_replay.py", REPO / "sim/train/tasks/base.py",
        REPO / "sim/val/env.py", REPO / "third_party/Zero-WAM/wan_va/wan_va_server.py",
        REPO / "third_party/Zero-WAM/wan_va/modules/icl_model.py",
    ]
    protocol = {
        "schema": 3, "scene_protocol": "full_recorded_layout_v2",
        "backend": "packed_static_batch4", "batch_size": 4, "max_tokens": 98304,
        "forecast_frames": args.forecast_frames, "execute_controls": 8,
        "ensemble": args.ensemble, "ensemble_age_weight": "3**(-age), normalized",
        "ema_alpha": 1.0, "video_guidance_scale": 5.0, "icl_guidance_scale": 1.0,
        "conditioning": "paired human latent/detailed text plus target text",
        "checkpoint": str(checkpoint), "denoising_steps": [50, 50],
        "cache_action_history": "commanded", "horizon_times_oracle": 2.0,
        "bounded_max_controls": args.max_controls,
        "reference_history": str(args.reference_history) if args.reference_history else None,
        "reference_commands_sha256": sha256_file(args.reference_history / "telemetry.jsonl")
        if args.reference_history else None,
        "sample_seeds": {e["id"]: 20261101 + all_rows.index(e) for e in selected},
        "cohort_sha256": sha256_file(DATA / "cohort_manifest.json"),
        "selected_episodes": selected,
        "human_latent_sha256": {e["id"]: sha256_file(e["human_latent"]) for e in selected},
        "checkpoint_sha256": {f.name: sha256_file(f) for f in sorted(
            (checkpoint / "transformer").iterdir()) if f.is_file()},
        "source_sha256": {str(f.relative_to(REPO)): sha256_file(f) for f in sources},
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise RuntimeError("changed protocol/source: use a new output directory")
    elif list(args.output.iterdir()):
        raise RuntimeError("nonempty output lacks manifest")
    else:
        atomic_json(manifest, protocol)
    rows = []
    with tempfile.TemporaryDirectory(prefix="so101-b4partial-") as temp:
        model, load_evidence = load_model(
            REPO / "data/models/zero-wam-pretrain", checkpoint, temp, 50, 50)
        model.job_config = copy.copy(model.job_config)
        model.job_config.frame_chunk_size = args.forecast_frames
        model.job_config.icl_guidance_scale = 1.0
        model.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        engine = BatchedPartialChunkEngine(
            model, max_tokens=98304, batch_size=4,
            video_guidance_scale=5.0, icl_guidance_scale=1.0,
            use_icl=True, safety_fraction=0.92)
        atomic_json(args.output / "model_load.json", load_evidence)
        for group_id, offset in enumerate(range(0, len(selected), 4)):
            group = selected[offset:offset+4]
            completed = []
            for ep in group:
                directory = args.output / ep["id"]
                if directory.exists():
                    result = json.loads((directory / "result.json").read_text())
                    if result.get("partial_protocol_sha256") != sha256_file(manifest):
                        raise RuntimeError(f"incomplete episode requires separate retry directory: {ep['id']}")
                    verify_artifacts(directory, result)
                    completed.append(result)
            if completed:
                if len(completed) != len(group):
                    raise RuntimeError("partially completed batch: preserve slot layout in separate retry")
                rows.extend(completed)
                continue
            specs = [SlotResetSpec(prompt=e["instruction"], icl_latent_path=e["human_latent"],
                                   seed=protocol["sample_seeds"][e["id"]]) for e in group]
            specs += [SlotResetSpec(prompt="pad", active=False)] * (4-len(specs))
            model.text_encoder.to(model.device)
            engine.reset(specs)
            model.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            rollouts = []
            for slot, ep in enumerate(group):
                sim = TrainSim(ep["task"], ep["seed"], ep["visual_config"],
                               recorded_poses=ep["object_poses_start"])
                obs = sim.observe()
                directory = args.output / ep["id"]
                directory.mkdir(parents=True)
                rollouts.append({
                    "slot": slot, "episode": ep, "sim": sim, "first_obs": obs,
                    "start_pose_check": check_start_poses(sim, ep["object_poses_start"]),
                    "scene_replay": sim.scene_replay, "controls": 0, "chunk": 0,
                    "first_success": None, "actions": [], "telemetry": [],
                    "frames": [np.concatenate([obs[FRONT], obs[WRIST]], axis=1)],
                    "horizon": args.max_controls or math.ceil(2.0*ep["robot_frames"]),
                    "active": True, "forecasts": [], "directory": directory,
                    "target_starts": {n: sim.env.object_pos(n).copy() for n in sim.task_objects},
                    "started": time.monotonic(),
                })
            infer_times = []
            try:
                while any(r["active"] for r in rollouts):
                    first_obs = [None] * 4
                    for r in rollouts:
                        if r["active"] and r["chunk"] == 0:
                            first_obs[r["slot"]] = {"obs": r["first_obs"]}
                    start = time.monotonic()
                    predicted = engine.infer(first_obs)
                    infer_times.append(time.monotonic()-start)
                    observations, histories = [None]*4, [None]*4
                    for r in rollouts:
                        if not r["active"]:
                            continue
                        slot, sim = r["slot"], r["sim"]
                        first = r["chunk"] == 0
                        a = np.asarray(predicted[slot], dtype=np.float32)
                        if a.shape != (6, args.forecast_frames, 8) or not np.isfinite(a).all():
                            raise ValueError("invalid packed forecast")
                        future = a[:, 1 if first else 0:, :].transpose(1, 2, 0).reshape(-1, 6)
                        forecast = {"start": r["controls"], "actions": future.tolist()}
                        r["forecasts"].append(forecast)
                        commands, contributors = blend_at(r["forecasts"], r["controls"], args.ensemble)
                        if reference is not None and slot == 0 and group_id == 0:
                            commands = np.asarray(reference[r["controls"]:r["controls"]+8], dtype=np.float32)
                        with (r["directory"] / "forecasts.jsonl").open("a") as sink:
                            sink.write(json.dumps({
                                **forecast, "selected_commands": commands.tolist(),
                                "contributors": contributors, "initial_context_frame_skipped": first,
                                "reference_commands_forced": reference is not None and slot == 0 and group_id == 0,
                            })+"\n")
                        keyframes = []
                        for j, command in enumerate(commands):
                            info = sim.step(command)
                            r["controls"] += 1
                            r["actions"].append(future[j].tolist())
                            tcp = sim.env._get_tcp_pose()[:3]
                            targets = {}
                            for n in sim.task_objects:
                                pos = sim.env.object_pos(n)
                                targets[n] = {
                                    "tcp_distance_m": float(np.linalg.norm(tcp-pos)),
                                    "displacement_m": float(np.linalg.norm(pos-r["target_starts"][n])),
                                    "lift_m": float(pos[2]-r["target_starts"][n][2]),
                                    "grasped": bool(sim.env.is_grasping(n)),
                                }
                            r["telemetry"].append({
                                "control": r["controls"], "predicted_action": future[j].tolist(),
                                "commanded_action": command.tolist(),
                                "executed_state": sim.env.lerobot_joints().astype(np.float32).astype(float).tolist(),
                                "tcp_position_m": tcp.astype(float).tolist(), "targets": targets,
                                "task_info": {k: v for k, v in sim.env.task_info().items()
                                              if isinstance(v, (bool, int, float, str))},
                            })
                            if info["success"] and r["first_success"] is None:
                                r["first_success"] = r["controls"]
                            if (j+1) % 2 == 0:
                                obs = sim.observe()
                                keyframes.append(obs)
                                r["frames"].append(np.concatenate([obs[FRONT], obs[WRIST]], axis=1))
                        history = np.zeros((6, 2 if first else 1, 8), dtype=np.float32)
                        history[:, -1, :] = commands.T
                        hist = model.preprocess_action(history)
                        hist[:, ~model.action_mask.cpu()] = 0
                        if first:
                            hist[:, :, 0] = 0
                        histories[slot] = hist.to(model.device, model.dtype)
                        observations[slot] = {"obs": keyframes}
                        r["chunk"] += 1
                        r["forecasts"] = [f for f in r["forecasts"]
                                         if f["start"]+len(f["actions"]) > r["controls"]]
                    # Commit all just-executed blocks before deactivating slots.
                    engine.cache_update(observations, histories)
                    for r in rollouts:
                        if not r["active"]:
                            continue
                        if r["controls"] >= r["horizon"] or r["first_success"] is not None:
                            result = _write_episode(
                                r["directory"], r, r["episode"],
                                protocol["sample_seeds"][r["episode"]["id"]], 1.0,
                                infer_times, group_id, [e["id"] for e in group]+[None]*(4-len(group)),
                                time.monotonic()-r["started"])
                            result.update(
                                partial_protocol_sha256=sha256_file(manifest),
                                forecast_frames=args.forecast_frames, execute_controls=8,
                                ensemble=args.ensemble, video_guidance_scale=5.0,
                                reference_history_diagnostic=reference is not None,
                                cache_events=engine.cache_events)
                            verify_artifacts(r["directory"], result)
                            atomic_json(r["directory"] / "result.json", result)
                            rows.append(result)
                            r["active"] = False
                            engine.set_active(r["slot"], False)
                            print(f"[done] {r['episode']['id']} success={result['success']} "
                                  f"controls={r['controls']}", flush=True)
                    atomic_json(args.output / "summary.json", {
                        "episodes": rows, "group": group_id, "diagnostics": engine.diagnostics(),
                        "progress": [{"id": r["episode"]["id"], "controls": r["controls"],
                                      "active": r["active"]} for r in rollouts],
                    })
                    print(f"[batch] group={group_id} controls={[r['controls'] for r in rollouts]} "
                          f"infer_s={infer_times[-1]:.2f}", flush=True)
            finally:
                for r in rollouts:
                    r["sim"].env.close()
    print(f"EXIT_OK episodes={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
