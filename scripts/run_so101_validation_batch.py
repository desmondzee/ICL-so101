#!/usr/bin/env python3
"""Run a bounded local-H100 batch over every held-out SO-101 validation task."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "third_party/Zero-WAM")]

from zero_wam.prompt_sweep import CheckpointSpec, atomic_json, checkpoint_identity

DATA = REPO / "data/hf_bucket_ICL-so101"
BASE = REPO / "data/models/zero-wam-pretrain"
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


def episodes(
    per_task: int,
    tasks: tuple[str, ...] | None = None,
    episode_ids: tuple[str, ...] | None = None,
) -> list[dict]:
    index = json.loads((DATA / "val/index.json").read_text())
    manifest_rows = json.loads((DATA / "val/icl_manifest.json").read_text())["samples"]
    manifest = {row["sample_id"]: row for row in manifest_rows}
    selected, counts = [], {}
    for episode in index["episodes"]:
        task = episode["task"]
        if episode_ids is not None and episode["id"] not in episode_ids:
            continue
        if tasks is not None and task not in tasks:
            continue
        if counts.get(task, 0) >= per_task:
            continue
        sample = manifest[episode["id"].replace("/", "__")]
        latent = (
            DATA / "val/human_latents"
            / Path(sample["human_video_path"]).with_suffix(".pth")
        )
        if not latent.is_file():
            raise FileNotFoundError(latent)
        selected.append({
            "id": episode["id"],
            "task": task,
            "seed": int(episode["seed"]),
            "robot_frames": int(episode["robot_frames"]),
            "instruction": sample["human_text"],
            "human_latent": str(latent),
        })
        counts[task] = counts.get(task, 0) + 1
    if episode_ids is not None:
        if {row["id"] for row in selected} != set(episode_ids):
            raise ValueError("one or more requested episode IDs were not found")
        return selected
    expected_tasks = (
        set(tasks) if tasks is not None
        else {row["task"] for row in index["tasks"]}
    )
    if set(counts) != expected_tasks or any(count != per_task for count in counts.values()):
        raise ValueError(f"incomplete per-task selection: {counts}")
    return selected


def summarize(rows: list[dict]) -> dict:
    output = {}
    for task in sorted({row["task"] for row in rows}):
        group = [row for row in rows if row["task"] == task]
        output[task] = {
            "episodes": len(group),
            "successes": sum(row["success"] for row in group),
            "success_rate": sum(row["success"] for row in group) / len(group),
            "mean_normalized_delta_rms": sum(
                row["motion"]["normalized_delta_rms"] for row in group
            ) / len(group),
            "mean_normalized_second_difference_rms": sum(
                row["motion"]["normalized_second_difference_rms"] for row in group
            ) / len(group),
            "mean_direction_reversal_fraction": sum(
                row["motion"]["direction_reversal_fraction"] for row in group
            ) / len(group),
        }
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="v2b")
    parser.add_argument("--per-task", type=int, default=5)
    parser.add_argument("--tasks", default="", help="optional comma-separated task filter")
    parser.add_argument("--episode-ids", default="", help="optional comma-separated exact episode IDs")
    parser.add_argument(
        "--max-controls", type=int, default=0,
        help="fixed control cap; 0 uses --horizon-times-oracle",
    )
    parser.add_argument("--horizon-times-oracle", type=float, default=2.0)
    parser.add_argument("--sample-seed", type=int, default=20261014)
    parser.add_argument("--video-steps", type=int, default=50)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument(
        "--action-ema-alpha", type=float, default=1.0,
        help="1 preserves raw targets; lower values smooth executed joint targets",
    )
    parser.add_argument(
        "--cache-action-history", choices=("commanded", "measured"),
        default="commanded",
    )
    parser.add_argument(
        "--output", default=str(REPO / "outputs/zero_wam/validation_batch")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    from zero_wam.so101_closed_loop import load_model, run_episode

    spec = CHECKPOINTS[args.checkpoint]
    identity = checkpoint_identity(spec)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "checkpoint.json", identity)
    selected = episodes(
        args.per_task,
        tuple(filter(None, args.tasks.split(","))) or None,
        tuple(filter(None, args.episode_ids.split(","))) or None,
    )
    atomic_json(output / "episodes.json", selected)
    rows, errors = [], []
    with tempfile.TemporaryDirectory(prefix="so101-validation-model-") as temporary:
        model, load_evidence = load_model(
            BASE, spec.path, temporary, args.video_steps, args.action_steps
        )
        identity["model_load"] = load_evidence
        atomic_json(output / "checkpoint.json", identity)
        for index, episode in enumerate(selected):
            out = output / spec.name / episode["id"]
            max_controls = (
                args.max_controls
                if args.max_controls > 0
                else math.ceil(args.horizon_times_oracle * episode["robot_frames"])
            )
            print(
                f"[{index + 1}/{len(selected)}] {episode['id']} "
                f"seed={episode['seed']} max_controls={max_controls}", flush=True,
            )
            try:
                result = run_episode(
                    model, episode, out, max_controls=max_controls,
                    sample_seed=args.sample_seed + index,
                    video_guidance_scale=-1.0,
                    action_ema_alpha=args.action_ema_alpha,
                    cache_action_history=args.cache_action_history,
                )
                rows.append(result)
                print(
                    f"  success={result['success']} controls={result['controls']} "
                    f"jitter={result['motion']['normalized_second_difference_rms']:.5f}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({"id": episode["id"], "error": repr(exc)})
                print(f"  ERROR {exc!r}", flush=True)
            atomic_json(
                output / "summary.json",
                {
                    "schema": 1,
                    "checkpoint": identity,
                    "per_task_requested": args.per_task,
                    "max_controls": args.max_controls,
                    "horizon_times_oracle": args.horizon_times_oracle,
                    "video_steps": args.video_steps,
                    "action_steps": args.action_steps,
                    "action_ema_alpha": args.action_ema_alpha,
                    "cache_action_history": args.cache_action_history,
                    "sample_seed_base": args.sample_seed,
                    "episodes": rows,
                    "errors": errors,
                    "aggregate": summarize(rows) if rows else {},
                },
            )
            torch.cuda.empty_cache()
    if errors:
        raise SystemExit(f"{len(errors)} validation episodes failed")


if __name__ == "__main__":
    main()
