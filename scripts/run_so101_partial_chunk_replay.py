"""Controlled 2/4-frame forecasts with 8-control execution on seen train scenes."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.run_so101_batched_train_replay import (
    DATA, load_cohort, select_episode_rows,
)
from scripts.run_so101_sequential_train_replay import run_episode_train
from zero_wam.prompt_sweep import atomic_json, sha256_file

REPO = Path(__file__).resolve().parents[1]


def blend_at(forecasts, start, enabled):
    """Blend only forecasts covering the same eight absolute control times."""
    contributors = [
        f for f in forecasts
        if f["start"] <= start and f["start"] + len(f["actions"]) >= start + 8
    ]
    if not enabled:
        contributors = contributors[-1:]
    if not contributors:
        raise ValueError("no forecast covers execution interval")
    ages = [(start - f["start"]) // 8 for f in contributors]
    weights = np.power(3.0, -np.asarray(ages, dtype=np.float64))
    weights /= weights.sum()
    commands = sum(
        w * np.asarray(f["actions"][start-f["start"]:start-f["start"]+8],
                       dtype=np.float32)
        for w, f in zip(weights, contributors)
    ).astype(np.float32)
    return commands, [
        {"forecast_start": f["start"], "age": age, "weight": float(w)}
        for f, age, w in zip(contributors, ages, weights)
    ]


class ExecutionView:
    """Expose only selected controls to the existing simulator runner.

    Normalized history assignments are forwarded to the real model, so its
    partial cache receives executed commands, including ensemble commands.
    """
    def __init__(self, model, forecast_frames, ensemble, forecast_path):
        self.model = model
        self.forecast_frames = forecast_frames
        self.ensemble = ensemble
        self.forecast_path = forecast_path
        self.forecasts = []
        self.raw_executed = []
        self.controls = 0
        self.pending = False

    def __getattr__(self, key):
        return getattr(self.model, key)

    @property
    def last_predicted_actions(self):
        return self.model.last_predicted_actions

    @last_predicted_actions.setter
    def last_predicted_actions(self, value):
        self.model.last_predicted_actions = value

    @torch.no_grad()
    def infer(self, payload):
        if payload.get("reset"):
            self.forecasts = []
            self.raw_executed = []
            self.controls = 0
            self.pending = False
            return self.model.infer(payload)
        if payload.get("compute_kv_cache"):
            if not self.pending:
                raise RuntimeError("cache commit without pending execution")
            result = self.model.infer(payload)
            self.pending = False
            print(f"[partial] controls={self.controls} "
                  f"cursor={self.model.partial_frame_cursor}", flush=True)
            return result
        if self.pending:
            raise RuntimeError("prediction before previous execution was committed")
        first = self.model.chunk_idx == 0
        action = np.asarray(self.model.infer(payload)["action"], dtype=np.float32)
        if action.shape != (6, self.forecast_frames, 8):
            raise ValueError(f"unexpected forecast shape {action.shape}")
        future = action[:, 1 if first else 0:, :].transpose(1, 2, 0).reshape(-1, 6)
        if not np.isfinite(future).all():
            raise ValueError("nonfinite forecast")
        current = {"start": self.controls, "actions": future.tolist()}
        self.forecasts.append(current)
        commands, contributors = blend_at(self.forecasts, self.controls, self.ensemble)
        record = {**current, "selected_commands": commands.tolist(),
                  "contributors": contributors, "initial_context_frame_skipped": first}
        with self.forecast_path.open("a") as sink:
            sink.write(json.dumps(record) + "\n")
        self.raw_executed.extend(future[:8].tolist())
        self.controls += 8
        self.forecasts = [
            f for f in self.forecasts
            if f["start"] + len(f["actions"]) > self.controls
        ]
        # The unchanged simulator loop skips frame zero only on startup.
        selected = np.zeros((6, 2 if first else 1, 8), dtype=np.float32)
        selected[:, -1, :] = commands.T
        self.pending = True
        return {"action": selected}


def verify_artifacts(directory, result):
    import imageio.v2 as imageio
    controls = result["controls"]
    actions = np.load(directory / "actions.npy")
    telemetry = [json.loads(x) for x in (directory / "telemetry.jsonl").read_text().splitlines()]
    if actions.shape != (controls, 6) or len(telemetry) != controls:
        raise ValueError("action/telemetry length mismatch")
    with imageio.get_reader(directory / "rollout.mp4") as video:
        if video.count_frames() != controls // 2 + 1:
            raise ValueError("video/control length mismatch")
    if not np.isfinite(actions).all():
        raise ValueError("nonfinite actions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-frames", type=int, choices=(2, 4), required=True)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--max-controls", type=int)
    args = parser.parse_args()
    if args.max_controls is not None and args.max_controls < 32:
        parser.error("bounded verification must cover at least four replans")
    from zero_wam.partial_chunk_engine import install_partial_chunk
    from zero_wam.so101_closed_loop import load_model, motion_diagnostics

    all_rows = load_cohort()
    selected = select_episode_rows(all_rows, args.episode_id)
    checkpoint = REPO / "data/models/checkpoints/v2b_step500"
    source_paths = [
        Path(__file__), REPO / "zero_wam/partial_chunk_engine.py",
        REPO / "scripts/run_so101_sequential_train_replay.py",
        REPO / "scripts/run_so101_batched_train_replay.py",
        REPO / "zero_wam/train_scene_replay.py",
        REPO / "sim/train/tasks/base.py",
        REPO / "sim/val/env.py",
        REPO / "third_party/Zero-WAM/wan_va/wan_va_server.py",
        REPO / "third_party/Zero-WAM/wan_va/modules/icl_model.py",
    ]
    protocol = {
        "schema": 2, "scene_protocol": "full_recorded_layout_v2",
        "forecast_frames": args.forecast_frames, "execute_controls": 8,
        "ensemble": args.ensemble, "ensemble_age_weight": "3**(-age), normalized",
        "ema_alpha": 1.0, "video_guidance_scale": 5.0, "icl_guidance_scale": 1.0,
        "conditioning": "paired human latent/detailed text plus target text",
        "checkpoint": str(checkpoint), "denoising_steps": [50, 50],
        "cache_action_history": "commanded", "horizon_times_oracle": 2.0,
        "bounded_max_controls": args.max_controls,
        "sample_seeds": {ep["id"]: 20261101 + all_rows.index(ep) for ep in selected},
        "cohort_sha256": sha256_file(DATA / "cohort_manifest.json"),
        "selected_episodes": selected,
        "human_latent_sha256": {ep["id"]: sha256_file(ep["human_latent"]) for ep in selected},
        "checkpoint_sha256": {
            p.name: sha256_file(p) for p in sorted((checkpoint / "transformer").iterdir())
            if p.is_file()
        },
        "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in source_paths},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != protocol:
            raise RuntimeError("refusing resume with changed protocol or sources")
    else:
        if list(args.output.iterdir()):
            raise RuntimeError("nonempty output lacks manifest")
        atomic_json(manifest_path, protocol)
    rows = []
    with tempfile.TemporaryDirectory(prefix="so101-partial-") as temp:
        model, load_evidence = load_model(
            REPO / "data/models/zero-wam-pretrain", checkpoint, temp, 50, 50)
        install_partial_chunk(model, args.forecast_frames)
        # Explicitly select target text guidance, not human-ICL CFG.
        model.job_config.icl_guidance_scale = 1.0
        atomic_json(args.output / "model_load.json", load_evidence)
        for ep in selected:
            directory = args.output / ep["id"]
            result_path = directory / "result.json"
            if directory.exists():
                result = json.loads(result_path.read_text())
                if result.get("partial_protocol_sha256") != sha256_file(manifest_path):
                    raise RuntimeError(f"incomplete/mismatched episode: {ep['id']}")
                verify_artifacts(directory, result)
                rows.append(result)
                continue
            directory.mkdir(parents=True)
            view = ExecutionView(model, args.forecast_frames, args.ensemble,
                                 directory / "forecasts.jsonl")
            horizon = args.max_controls or math.ceil(2.0 * ep["robot_frames"])
            print(f"[replay] {ep['id']} horizon={horizon} "
                  f"forecast={args.forecast_frames} ensemble={args.ensemble}", flush=True)
            result = run_episode_train(
                view, ep, directory, max_controls=horizon,
                sample_seed=protocol["sample_seeds"][ep["id"]],
                video_guidance_scale=5.0, action_ema_alpha=1.0)
            # Existing runner sees selected commands. Restore the newest raw
            # prediction in the predicted-action artifacts; commands stay intact.
            raw = np.asarray(view.raw_executed, dtype=np.float32)
            np.save(directory / "actions.npy", raw)
            telemetry = [json.loads(x) for x in (directory / "telemetry.jsonl").read_text().splitlines()]
            if len(telemetry) != len(raw):
                raise ValueError("raw prediction/telemetry length mismatch")
            with (directory / "telemetry.jsonl").open("w") as sink:
                for row, prediction in zip(telemetry, raw):
                    row["predicted_action"] = prediction.tolist()
                    sink.write(json.dumps(row) + "\n")
            result.update(
                motion=motion_diagnostics(raw),
                partial_protocol_sha256=sha256_file(manifest_path),
                forecast_frames=args.forecast_frames, execute_controls=8,
                ensemble=args.ensemble, video_guidance_scale=5.0,
                cache_events=model.partial_cache_events)
            verify_artifacts(directory, result)
            atomic_json(result_path, result)
            rows.append(result)
            atomic_json(args.output / "summary.json", {"episodes": rows, "protocol": protocol})
            print(f"[done] {ep['id']} success={result['success']} "
                  f"controls={result['controls']}", flush=True)
    print(f"EXIT_OK episodes={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
