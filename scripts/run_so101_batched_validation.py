"""Closed-loop validation via the packed batch-4 static engine (opt-in).

Entirely separate runner — never imported by existing entry points.  Runs the
same 25 deterministic held-out episodes as `run_so101_validation_batch.py`
(same ordering, same sample seeds 20261014+index), packed in fixed groups of
four slots through one loaded model.  Per-slot stepping, telemetry, EMA, and
artifact schema replicate `zero_wam.so101_closed_loop.run_episode` exactly.
Only transformer execution is packed; simulators step independently per slot
and the final short group uses inactive masked padding slots.
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
SAMPLE_SEED_BASE = 20261014


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="data/models/checkpoints/v2b_step500")
    p.add_argument("--base-model", default="data/models/zero-wam-pretrain")
    p.add_argument("--output",
                   default="outputs/zero_wam/validation_batch_v2b_5x5_full_b4_alpha090")
    p.add_argument("--max-tokens", type=int, default=155648,
                   help="packed static capacity; steady-state obs window "
                        "needs ~145k tokens for 4 slots")
    p.add_argument("--alpha", type=float, default=0.9,
                   help="EMA alpha (command = prev + alpha*(pred - prev))")
    p.add_argument("--horizon-times-oracle", type=float, default=2.0)
    p.add_argument("--max-controls", type=int, default=None,
                   help="Optional absolute per-slot control cap (bounded "
                        "mechanical smoke; omit for the full protocol)")
    p.add_argument("--episode-id", action="append", default=[],
                   help="exact canonical episode id(s) to run; repeatable; "
                        "empty = all 25 in canonical order")
    p.add_argument("--safety-fraction", type=float, default=0.85)
    return p.parse_args()


def select_episode_rows(all_selected, ids):
    """Filter canonical rows to exact ids, preserving order and seeds."""
    if not ids:
        return list(all_selected)
    wanted = set(ids)
    chosen = [ep for ep in all_selected if ep["id"] in wanted]
    if len(chosen) != len(wanted):
        missing = sorted(wanted - {ep["id"] for ep in chosen})
        raise ValueError(f"episode ids not found in index: {missing}")
    return chosen


def _existing_results(out_root):
    """result.json ids already on disk — merge-safe summary membership."""
    root = Path(out_root) / "v2b_step500"
    if not root.is_dir():
        return []
    return sorted(
        str(p.parent.relative_to(root))
        for p in root.rglob("result.json")
    )


def _episode_dir(out_root, spec_name, episode_id):
    return out_root / spec_name / episode_id


def _write_episode(out_dir, r, ep, sample_seed, alpha, infer_times,
                   group_id, slot_order, wall_s):
    """Write the same per-episode artifacts as `run_episode`."""
    from zero_wam.so101_closed_loop import TASK_TARGETS, motion_diagnostics

    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_dir / "rollout.mp4", r["frames"], fps=15,
                     codec="libx264", quality=7, macro_block_size=8)
    sim = r["sim"]
    target_names = TASK_TARGETS.get(ep["task"], ())
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


def main():
    args = _parse_args()
    from scripts.run_so101_validation_batch import episodes
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
    from zero_wam.so101_closed_loop import (
        FRONT,
        TASK_TARGETS,
        WRIST,
        Sim,
        smooth_action,
    )

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(*a, flush=True)

    # ---- admission: static estimate + free VRAM before model load --------------
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
        raise MemoryError(f"static estimate {preflight['estimated_gib']} GiB "
                          f"does not fit in {preflight['free_gib']} GiB free")

    with tempfile.TemporaryDirectory() as temp:
        assembled = _assembled_model(args.base_model, args.checkpoint, temp)
        model, load_evidence, config = _load_server(assembled, "cuda:0")
    # The static budget check at install sees current CUDA allocations; the
    # text encoder (~11 GiB) is only needed inside reset, so park it on CPU
    # and shuttle it back per group for encode_prompt.
    model.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    all_selected = episodes(5)  # 25 deterministic rows, index order
    assert len(all_selected) == 25
    selected = select_episode_rows(all_selected, args.episode_id)
    groups = [selected[i:i + GROUP_SIZE]
            for i in range(0, len(selected), GROUP_SIZE)]
    atomic_json(out_root / "episodes.json", all_selected)

    engine = BatchedStaticEngine(
        model,
        max_tokens=args.max_tokens,
        batch_size=GROUP_SIZE,
        video_guidance_scale=-1.0,
        icl_guidance_scale=float(config.icl_guidance_scale),
        use_icl=True,
        safety_fraction=args.safety_fraction,
    )

    spec_name = Path(args.checkpoint).name
    started = time.monotonic()
    rows, errors = [], []
    infer_times = []

    for group_id, group in enumerate(groups):
        global_idx = [all_selected.index(ep) for ep in group]
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
                prompt="pad", icl_latent_path=None, seed=0,
                active=False))  # masked padding slot
        log(f"[group {group_id}] real={len(group)} pad={GROUP_SIZE-len(group)} "
            f"episodes={[ep['id'] for ep in group]}")
        model.text_encoder.to("cuda")
        engine.reset(specs)
        model.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        log(f"[group {group_id}] post-reset free "
            f"{torch.cuda.mem_get_info(0)[0]/2**30:.1f} GiB")

        # ---- per-slot rollout state (real slots only) ------------------------
        rollouts = []
        for s, (ep, gidx) in enumerate(zip(group, global_idx)):
            sim = Sim(ep["task"], ep["seed"])
            first_obs = sim.observe()
            targets = TASK_TARGETS.get(ep["task"], ())
            rollouts.append({
                "episode": ep,
                "slot": s,
                "sample_seed": SAMPLE_SEED_BASE + gidx,
                "sim": sim,
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
        obs0 = [
            {"obs": r["first_obs"]} if r["active"] else None
            for r in rollouts
        ]
        # obs entries for padding slots (none — engine only reads active)
        while len(obs0) < GROUP_SIZE:
            obs0.append(None)
        predicted = engine.infer(obs0)
        packed_chunk = 0
        slot_order = [ep["id"] for ep in group] + [None] * (
            GROUP_SIZE - len(group))

        group_failed = None
        while any(r["active"] for r in rollouts):
            t_chunk = time.monotonic()
            obs_lists, histories = [], []
            for s, r in enumerate(rollouts):
                if not r["active"]:
                    continue
                action = np.asarray(predicted[s], dtype=np.float32)
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
                        for name in TASK_TARGETS.get(
                                r["episode"]["task"], ()):
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

            for s in range(GROUP_SIZE):
                if s < len(rollouts):
                    obs_lists.append(
                        {"obs": rollouts[s]["keyframes"]}
                        if rollouts[s]["active"] else None
                    )
                    if rollouts[s]["active"]:
                        hist = model.preprocess_action(
                            rollouts[s]["commanded_hist"])
                        hist[:, ~model.action_mask.cpu()] = 0
                        if engine.slots[s].chunk_idx == 0:
                            hist[:, :, 0] = 0
                        histories.append(
                            hist.to(model.device, model.dtype))
                    else:
                        histories.append(None)
                else:
                    obs_lists.append(None)
                    histories.append(None)

            if not any(r["active"] for r in rollouts):
                packed_chunk += 1
                checkpoint_summary(packed_chunk)
                break
            t0 = time.monotonic()
            try:
                engine.cache_update(obs_lists, histories)
            except torch.OutOfMemoryError:
                atomic_json(out_root / "summary.json",
                            {"fatal": "oom_in_cache_update",
                             "group": group_id,
                             "packed_chunks": packed_chunk})
                raise
            for s, r in enumerate(rollouts):
                engine.set_active(s, r["active"])
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
                f"active={[r['active'] for r in rollouts]} "
                f"controls={[r['controls'] for r in rollouts]} "
                f"occupancy={occ['metadata']['length']}/"
                f"{occ['metadata']['capacity']} "
                f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
            checkpoint_summary(packed_chunk)

        # ---- per-episode artifacts -------------------------------------------
        group_wall = round(time.monotonic() - started, 1)
        for r in rollouts:
            try:
                result = _write_episode(
                    _episode_dir(out_root, spec_name, r["episode"]["id"]),
                    r, r["episode"], r["sample_seed"], args.alpha,
                    infer_times, group_id, slot_order, group_wall)
                rows.append(result)
                log(f"  done {r['episode']['id']} slot={r['slot']} "
                    f"success={result['success']} controls={result['controls']}")
            except Exception as exc:  # noqa: BLE001 — record, keep going
                errors.append({"id": r["episode"]["id"],
                               "group": group_id, "error": repr(exc)})
                log(f"  ERROR {r['episode']['id']}: {exc!r}")
            r["sim"].env.close()
        checkpoint_summary(packed_chunk)

    manifest = {
        "schema": "batched_validation_v1",
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
        "episodes": [{"id": ep["id"], "sample_seed": SAMPLE_SEED_BASE + i}
                     for i, ep in enumerate(all_selected)],
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
        "selected_episode_ids": [ep["id"] for ep in selected],
        "errors": errors,
    }
    atomic_json(out_root / "batch_manifest.json", manifest)
    print(f"EXIT_OK wall={manifest['wall_s']}s episodes={len(rows)} "
          f"successes={manifest['successes']} "
          f"peak={manifest['peak_gib']}GiB")


if __name__ == "__main__":
    main()
