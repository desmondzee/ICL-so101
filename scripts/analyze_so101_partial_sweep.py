"""Freeze behavioral measurements for the completed corrected batch-four sweep.

Run from the repo root. Motion metrics use 30-Hz commanded and measured joints,
not raw forecasts. Population variance is descriptive motion extent, not noise.
"""
from pathlib import Path
import hashlib
import json
import re

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/results/zero_wam_partial_chunk_replay"
DATA = ROOT / "data/hf_bucket_ICL-so101/train_replay_probe"
ARMS = {
    "F2_raw": "partial8_raw", "F2_ensemble": "partial8_ensemble",
    "F4_raw": "partial8_forecast4_raw", "F4_ensemble": "partial8_forecast4_ensemble",
}
EPS = [f"{task}/episode_{i:03d}" for task in
       ("ball_between_cans", "block_in_bowl", "push_cube_into_tape_square") for i in range(2)]
BOUNDS = ROOT / "data/hf_bucket_ICL-so101/zero_wam_loader_v1/val/meta/action_stats.json"
bounds = json.loads(BOUNDS.read_text())["bounds"]
SPAN = np.asarray(bounds["q99"]) - bounds["q01"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def motion(a):
    a = np.asarray(a, dtype=float)
    arm = a[:, :5] / SPAN[:5]
    d = np.diff(arm, axis=0)
    boundary = np.arange(1, len(a)) % 8 == 0
    return {
        "arm_variance_range2": float(np.var(arm, axis=0).mean()),
        "arm_step_rms_pct_range": 100*rms(d),
        "arm_second_difference_rms_pct_range": 100*rms(np.diff(arm, n=2, axis=0)),
        "arm_boundary_step_rms_pct_range": 100*rms(d[boundary]),
        "arm_within_block_step_rms_pct_range": 100*rms(d[~boundary]),
        "arm_path_range_units": float(np.linalg.norm(d, axis=1).sum()),
        "gripper_step_rms_points": rms(np.diff(a[:, 5])),
        "gripper_second_difference_rms_points": rms(np.diff(a[:, 5], n=2)),
        "gripper_min": float(a[:, 5].min()), "gripper_max": float(a[:, 5].max()),
    }


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def load_arm(name):
    return ROOT / "outputs/zero_wam" / (
        f"train_replay_v2b_{ARMS[name]}_full_conditioning_3tasks_2seeds_scenev2_b4")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = {name: json.loads((load_arm(name)/"manifest.json").read_text()) for name in ARMS}
    common = ("backend", "batch_size", "bounded_max_controls", "cache_action_history",
              "checkpoint_sha256", "cohort_sha256", "conditioning", "denoising_steps",
              "ema_alpha", "execute_controls", "horizon_times_oracle", "human_latent_sha256",
              "icl_guidance_scale", "sample_seeds", "scene_protocol", "selected_episodes",
              "video_guidance_scale", "reference_history")
    for key in common:
        assert all(m[key] == manifests["F2_raw"][key] for m in manifests.values()), key
    records, provenance, oracles = [], {}, {}
    for ep in EPS:
        oracle_path = DATA / "episodes" / ep / "robot_data.parquet"
        table = pd.read_parquet(oracle_path)
        np.testing.assert_allclose(np.diff(table.timestamp), 1/30, atol=2e-6)
        oracle_cmd = np.stack(table.action)
        oracles[ep] = {"controls": len(table), "commanded": motion(oracle_cmd),
                       "executed": motion(np.stack(table["observation.state"])),
                       "sha256": sha(oracle_path)}
        for name in ARMS:
            directory = load_arm(name) / ep
            result = json.loads((directory/"result.json").read_text())
            t, forecasts = rows(directory/"telemetry.jsonl"), rows(directory/"forecasts.jsonl")
            commanded = np.asarray([x["commanded_action"] for x in t])
            executed = np.asarray([x["executed_state"] for x in t])
            raw = np.load(directory/"actions.npy")
            assert len(t) == result["controls"] == len(raw)
            assert len(forecasts)*8 == len(t)
            assert result["partial_protocol_sha256"] == sha(load_arm(name)/"manifest.json")
            assert result["scene_replay"]["check"]["pass"]
            assert result["scene_replay"]["check"]["max_abs_pose_error"] == 0
            assert result["sample_seed"] == manifests[name]["sample_seeds"][ep]
            assert result["controls"] == 8*int(np.ceil(2*len(table)/8))
            np.testing.assert_array_equal(raw, [x["predicted_action"] for x in t])
            np.testing.assert_allclose(commanded, np.concatenate(
                [x["selected_commands"] for x in forecasts]), rtol=0, atol=0)
            # Independently reconstruct the actual ensemble from recorded
            # absolute-time forecasts, including the startup context-frame skip.
            for k, f in enumerate(forecasts):
                assert f["start"] == k*8
                pred = np.asarray(f["actions"])
                assert len(pred) == 8*(manifests[name]["forecast_frames"]-(k==0))
                np.testing.assert_array_equal(raw[k*8:k*8+8], pred[:8])
                eligible = [g for g in forecasts[:k+1]
                            if g["start"]+len(g["actions"]) >= k*8+8]
                if not manifests[name]["ensemble"]:
                    eligible = eligible[-1:]
                weights = np.array([3.**(-(k-g["start"]//8)) for g in eligible])
                weights /= weights.sum()
                expected = sum(w*np.asarray(g["actions"])[k*8-g["start"]:k*8-g["start"]+8]
                               for w, g in zip(weights, eligible))
                np.testing.assert_allclose(commanded[k*8:k*8+8], expected, rtol=1e-6, atol=2e-5)
            target = next(iter(result["targets"]))
            d = np.array([x["targets"][target]["tcp_distance_m"] for x in t])
            displacement = np.array([x["targets"][target]["displacement_m"] for x in t])
            lift = np.array([x["targets"][target]["lift_m"] for x in t])
            grasp = np.array([x["targets"][target]["grasped"] for x in t], dtype=bool)
            tcp = np.asarray([x["tcp_position_m"] for x in t])
            assert int(grasp.sum()) == result["targets"][target]["grasp_controls"]
            np.testing.assert_allclose(d.min(), result["targets"][target]["min_tcp_distance_m"])
            per_slot_events = [e for e in result["cache_events"] if e["slot"] == result["packed_batch"]["slot"]]
            assert len(per_slot_events) == len(forecasts)
            for k, event in enumerate(per_slot_events):
                assert event["chunk_idx"] == k and event["obs_count"] == 4
                assert event["committed_frames"] == (2 if k == 0 else 1)
                assert event["cursor_before"] == (0 if k == 0 else k+1)
                assert event["action_shape"][2] == (2 if k == 0 else 1)
            records.append({
                "arm": name, "episode": ep, "controls": len(t), "oracle_controls": len(table),
                "success": result["success"], "final_success": result["final_state_success"],
                "min_tcp_cm": float(d.min()*100), "closest_control": int(d.argmin()+1),
                "max_displacement_cm": float(displacement.max()*100),
                "final_displacement_cm": float(displacement[-1]*100),
                "max_lift_cm": float(lift.max()*100), "max_lift_control": int(lift.argmax()+1),
                "grasp_controls": int(grasp.sum()),
                "tcp_path_m": float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum()),
                "commanded": motion(commanded), "executed": motion(executed),
                "raw_forecast": motion(raw),
                "first_oracle_horizon_commanded": motion(commanded[:len(table)]),
                "arm_tracking_mae_deg": float(np.abs(commanded[:, :5]-executed[:, :5]).mean()),
                "arm_tracking_p95_deg": float(np.percentile(np.abs(commanded[:, :5]-executed[:, :5]),95)),
                "per_joint_command_variance_deg2": np.var(commanded[:, :5], axis=0).tolist(),
                "directory": str(directory.relative_to(ROOT)),
            })
            provenance[f"{name}/{ep}"] = {file: sha(directory/file) for file in
                ("result.json", "telemetry.jsonl", "actions.npy", "forecasts.jsonl", "rollout.mp4")}
    compute = {}
    for name in ARMS:
        f = "2" if name.startswith("F2") else "4"
        mode = "ensemble" if name.endswith("ensemble") else "raw"
        log = Path(f"/tmp/so101_partial_scenev2_b4_f{f}_{mode}.log")
        text = log.read_text()
        values = np.array([float(x) for x in re.findall(r"\[batch\].*?infer_s=([0-9.]+)", text)])
        # Keep pause-contaminated time visible, and use median/quantiles rather
        # than inventing a corrected mean. These are packed inference times.
        compute[name] = {"log_sha256": sha(log), "logged_batches": len(values),
                         "median_packed_infer_s": float(np.median(values)),
                         "p10_packed_infer_s": float(np.percentile(values,10)),
                         "p90_packed_infer_s": float(np.percentile(values,90)),
                         "max_packed_infer_s": float(values.max())}
        (OUT/f"{name}_runtime.log").write_text("\n".join(
            l for l in text.splitlines() if l.startswith(("[batch]", "[done]", "EXIT_OK")))+"\n")
    snapshot = {"schema": 1, "analysis_source_sha256": sha(Path(__file__)),
                "bounds_sha256": sha(BOUNDS), "common_manifest_fields_verified": common,
                "manifests": {n: sha(load_arm(n)/"manifest.json") for n in ARMS},
                "source_sha256": provenance, "episodes": records, "oracle_motion": oracles,
                "compute": compute}
    (OUT/"measurements.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False)+"\n")
    print("FROZEN",len(records),"episodes at",OUT/"measurements.json")


if __name__ == "__main__":
    main()
