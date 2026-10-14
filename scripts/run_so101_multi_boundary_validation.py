"""Multi-boundary teacher-forced causal action prediction (opt-in).

Deployment-aligned boundaries b = 8 + 16*k; target [b:b+16].  For every
boundary the model is freshly reset with the paired human ICL/detailed
text, then the COMPLETE recorded robot history 0..b is teacher-forced:
the existing first-chunk prefix protocol (obs0 infer, controls 0:8
cached) followed by GT 16-control action blocks cached with robot
observations every 2 controls.  At b, one 16-control chunk is denoised
from seeded noise; targets reach the scorer only after inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys_path_added = False


def _ensure_path():
    global sys_path_added
    if not sys_path_added:
        import sys
        sys.path[:0] = [str(REPO), str(REPO / "third_party" / "Zero-WAM")]
        sys_path_added = True


CHUNK = 16
BOUNDARY_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
PROTOCOL_SEED_SALT = "so101-multi-boundary-tf-v1"


def select_boundaries(num_controls):
    """Five phase-spanning boundaries snapped to the nearest 8+16k."""
    span = num_controls - 24
    picked = {}
    for frac in BOUNDARY_FRACTIONS:
        raw = frac * span
        k = int(round((raw - 8) / CHUNK))
        k = max(k, 0)
        b = 8 + CHUNK * k
        while b + CHUNK > num_controls:
            k -= 1
            b = 8 + CHUNK * k
        if b < 8:
            continue
        picked.setdefault(b, frac)
    boundaries = sorted(picked.items())  # (b, fraction)
    if not boundaries or any(b + CHUNK > num_controls for b, _ in boundaries):
        raise ValueError(f"no valid boundaries for N={num_controls}")
    return boundaries


def boundary_seed(sample_id, boundary):
    return int(hashlib.sha256(
        f"{PROTOCOL_SEED_SALT}|{sample_id}|{boundary}".encode()
    ).hexdigest()[:8], 16)


def build_manifest(data_root):
    """Frozen manifest over the six train-replay cohort episodes."""
    _ensure_path()
    import pyarrow.parquet as pq

    from scripts.run_so101_batched_train_replay import COHORT_IDS
    from zero_wam.chunk_causal_validation import canonical_sha256

    data_root = Path(data_root)
    index = {s["sample_id"]: s for s in json.loads(
        (data_root / "icl_manifest.json").read_text())["samples"]}
    samples = []
    for eid in COHORT_IDS:
        task, ep = eid.split("/")
        sid = eid.replace("/", "__")
        row = index[sid]
        ed = data_root / "episodes" / task / ep
        n_controls = pq.read_table(
            ed / "robot_data.parquet", columns=["action"]).num_rows
        bounds = select_boundaries(n_controls)
        samples.append({
            "sample_id": sid, "task": task,
            "episode_index": int(ep.split("_")[-1]),
            "num_controls": int(n_controls),
            "boundaries": [
                {
                    "boundary": b, "fraction": frac,
                    "target_control_start": b,
                    "target_control_end": b + CHUNK,
                    "seed": boundary_seed(sid, b),
                }
                for b, frac in bounds
            ],
            "robot_front_video": str((ed / "robot_front.mp4").resolve()),
            "robot_wrist_video": str((ed / "robot_wrist.mp4").resolve()),
            "robot_parquet": str((ed / "robot_data.parquet").resolve()),
            "human_latent": str(
                (data_root / "human_latents"
                 / Path(row["human_video_path"]).with_suffix(".pth")
                 ).resolve()),
            "human_detailed_text": row["human_local_instruction"],
            "target_text": row["human_text"],
        })
    manifest = {
        "schema": 1,
        "protocol": "multi_boundary_teacher_forced_action_generation",
        "salt": PROTOCOL_SEED_SALT,
        "boundary_grid": "b = 8 + 16*k, target [b:b+16)",
        "boundary_fractions": list(BOUNDARY_FRACTIONS),
        "action_denoising_steps": 50,
        "observation_cadence_controls": 2,
        "human_icl_scope": "full paired human demonstration",
        "robot_context_scope": "executed GT history 0..b only",
        "future_robot_context": "withheld",
        "target_visibility": "scorer only after generation",
        "cohort": "train_replay_probe_3tasks_2seeds",
        "samples": samples,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="data/models/checkpoints/v2b_step500")
    p.add_argument("--base-model", default="data/models/zero-wam-pretrain")
    p.add_argument("--manifest",
                   default=str(
                       REPO / "data/hf_bucket_ICL-so101/train_replay_probe"
                       / "multi_boundary_manifest.json"))
    p.add_argument("--output", required=True)
    p.add_argument("--video-guidance-scale", type=float, default=-1.0)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def teacher_force_to_boundary(model, obs_by_frame, actions, boundary,
                              preprocess, assert_clean):
    """Cache GT history 0..boundary with the deployment cache semantics."""
    first = obs_by_frame(0)
    assert_clean(first)
    model.infer({"obs": first})
    prefix = actions[:8]
    history = np.zeros((6, 2, 8), dtype=np.float32)
    history[:, 1, :] = prefix.T
    normalized = preprocess(history, zero_first=True)
    model.last_predicted_actions = normalized.to(
        model.device, model.dtype)
    cache_obs = [obs_by_frame(f) for f in (2, 4, 6, 8)]
    assert_clean(cache_obs)
    model.infer({"obs": cache_obs, "compute_kv_cache": True,
                 "state": history})
    n_blocks = (boundary - 8) // CHUNK
    calls = 1
    for block in range(1, n_blocks + 1):
        c0 = 8 + CHUNK * (block - 1)
        block_actions = actions[c0:c0 + CHUNK]          # [16,6]
        hist = np.asarray(block_actions, dtype=np.float32).T.reshape(6, 2, 8)
        normalized = preprocess(hist)
        model.last_predicted_actions = normalized.to(
            model.device, model.dtype)
        frames = [c0 + 2 * m for m in range(1, 9)]      # every 2 controls
        obs_list = [obs_by_frame(f) for f in frames]
        assert_clean(obs_list)
        model.infer({"obs": obs_list, "compute_kv_cache": True,
                     "state": hist})
        calls += 1
    boundary_obs = obs_by_frame(boundary)
    assert_clean(boundary_obs)
    result = model.infer({"obs": boundary_obs})
    return result, calls


def preprocess_gt(model, block, zero_first=False):
    """Deployment action-history normalization.

    Inactive channels are always zeroed; ``zero_first`` additionally zeros
    the first action token, matching the chunk-0 skipped-frame semantics
    used for the initial 0:8 prefix only.
    """
    normalized = model.preprocess_action(block)
    normalized[:, ~model.action_mask.cpu()] = 0
    if zero_first:
        normalized[:, :, 0] = 0
    return normalized


def run_sample(model, sample, video_guidance_scale):
    from scripts.run_zero_wam_chunk_causal_validation import (
        _episode_arrays,
        _video_frames,
        _observation,
        hashlib_array,
    )
    from zero_wam.chunk_causal_validation import (
        CHUNK_CONTROLS,
        score_chunk,
    )

    actions, states = _episode_arrays(sample["robot_parquet"])
    needed = {0, 2, 4, 6, 8}
    for spec in sample["boundaries"]:
        b = spec["boundary"]
        for block in range(1, (b - 8) // CHUNK + 1):
            c0 = 8 + CHUNK * (block - 1)
            needed.update(c0 + 2 * m for m in range(1, 9))
        needed.add(b)
    front = _video_frames(sample["robot_front_video"], sorted(needed))
    wrist = _video_frames(sample["robot_wrist_video"], sorted(needed))

    def obs_by_frame(f):
        return _observation(front[f], wrist[f], states[f])

    leaked = []
    seen_targets = []

    def assert_clean(obj):
        leaked.append(len(obj))

    rows = []
    for spec in sample["boundaries"]:
        b, seed = spec["boundary"], spec["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.infer({
            "reset": True,
            "prompt": sample["target_text"],
            "use_icl": True,
            "icl_latent_path": sample["human_latent"],
            "video_guidance_scale": video_guidance_scale,
        })
        result, n_cache_calls = teacher_force_to_boundary(
            model, obs_by_frame, actions, b,
            lambda h, zero_first=False: preprocess_gt(
                model, h, zero_first), assert_clean)
        generated = np.asarray(result["action"], dtype=np.float32)
        if generated.shape != (6, 2, 8):
            raise ValueError(f"unexpected generated shape {generated.shape}")
        predicted = generated.transpose(1, 2, 0).reshape(CHUNK_CONTROLS, 6)
        # Targets enter only here — scorer side.
        target = actions[b:b + CHUNK_CONTROLS]
        persistence = np.repeat(actions[b - 1:b], CHUNK_CONTROLS, axis=0)
        rows.append({
            "sample_id": sample["sample_id"], "task": sample["task"],
            "boundary": b, "fraction": spec["fraction"],
            "target_control_start": b, "target_control_end": b + CHUNK,
            "seed": seed, "n_gt_cache_calls": n_cache_calls,
            "history_sha256": hashlib_array(actions[:b]),
            "prefix_sha256": hashlib_array(actions[:8]),
            "prediction_sha256": hashlib_array(predicted),
            "target_sha256": hashlib_array(target),
            "metrics": score_chunk(predicted, target, persistence),
            "input_provenance": {
                "obs_frames": sorted(needed & set(range(0, b + 1))),
                "gt_controls_cached": b,
                "no_future_frames_before_score": True,
            },
        })
    return rows


def aggregate_multi(rows):
    by_task = {}
    model_mae, pers_mae = [], []
    per_boundary = []
    beats = 0
    for r in rows:
        model_v = r["metrics"]["model"]["overall_physical_mae"]
        pers_v = r["metrics"]["persistence"]["overall_physical_mae"]
        model_mae.append(model_v)
        pers_mae.append(pers_v)
        beats += model_v < pers_v
        by_task.setdefault(r["task"], []).append(model_v)
        per_boundary.append({
            "sample_id": r["sample_id"],
            "task": r["task"],
            "boundary": r["boundary"],
            "fraction": r["fraction"],
            "model_mae": model_v,
            "persistence_mae": pers_v,
            "model_beats_persistence": model_v < pers_v,
        })
    return {
        "num_boundaries": len(rows),
        "model_overall_physical_mae_mean": float(np.mean(model_mae)),
        "model_overall_physical_rmse_mean": float(np.mean(
            [r["metrics"]["model"]["overall_physical_rmse"] for r in rows])),
        "persistence_overall_physical_mae_mean": float(np.mean(pers_mae)),
        "persistence_overall_physical_rmse_mean": float(np.mean(
            [r["metrics"]["persistence"]["overall_physical_rmse"]
             for r in rows])),
        "model_beats_persistence_count": beats,
        "per_task_model_mae": {
            t: float(np.mean(v)) for t, v in sorted(by_task.items())},
        "per_boundary_model_mae": per_boundary,
    }


def main():
    args = _parse_args()
    _ensure_path()
    from scripts.run_zero_wam_chunk_causal_validation import (
        _assembled_model,
        _load_server,
    )
    from zero_wam.chunk_causal_validation import canonical_sha256
    from zero_wam.so101_a100_preflight import atomic_write_json

    manifest = json.loads(Path(args.manifest).read_text())
    expected_hash = manifest.pop("manifest_sha256")
    if canonical_sha256(manifest) != expected_hash:
        raise ValueError("multi-boundary manifest hash mismatch")
    manifest["manifest_sha256"] = expected_hash
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory(prefix="so101-mb-assembly-") as temp:
        assembled = _assembled_model(args.base_model, args.checkpoint, temp)
        model, load_evidence, config = _load_server(assembled, args.device)
        for i, sample in enumerate(manifest["samples"], 1):
            print(f"[{i}/{len(manifest['samples'])}] {sample['sample_id']} "
                  f"({len(sample['boundaries'])} boundaries)", flush=True)
            rows.extend(run_sample(model, sample,
                                   args.video_guidance_scale))
            torch.cuda.empty_cache()
    report = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": manifest["protocol"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "video_guidance_scale": args.video_guidance_scale,
        "model_load": load_evidence,
        "samples": rows,
        "aggregate": aggregate_multi(rows),
    }
    atomic_write_json(output, report)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
