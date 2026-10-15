"""Frozen numerical and rollout checks for switching this sweep to batch four."""
from pathlib import Path
import argparse
import json
import numpy as np

from zero_wam.prompt_sweep import atomic_json, sha256_file


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--mode", choices=("fixed_history", "closed_loop"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    bounds_path = Path("data/hf_bucket_ICL-so101/zero_wam_loader_v1/val/meta/action_stats.json")
    bounds = json.loads(bounds_path.read_text())["bounds"]
    span = np.asarray(bounds["q99"]) - bounds["q01"]
    rr = json.loads((a.reference / "result.json").read_text())
    cr = json.loads((a.candidate / "result.json").read_text())
    for key in ("id", "seed", "sample_seed", "action_ema_alpha",
                "video_guidance_scale", "forecast_frames", "execute_controls", "ensemble"):
        if rr[key] != cr[key]:
            raise ValueError(f"unmatched condition {key}")
    if not cr["scene_replay"]["check"]["pass"] or not rr["scene_replay"]["check"]["pass"]:
        raise ValueError("unmatched scene")
    source_hashes = {
        label: {n: sha256_file(directory/n) for n in
                ("result.json", "forecasts.jsonl", "telemetry.jsonl", "actions.npy")}
        for label, directory in (("reference", a.reference), ("candidate", a.candidate))
    }
    if a.mode == "fixed_history":
        rf, cf = read_rows(a.reference/"forecasts.jsonl"), read_rows(a.candidate/"forecasts.jsonl")
        rt, ct = read_rows(a.reference/"telemetry.jsonl"), read_rows(a.candidate/"telemetry.jsonl")
        if len(cf) < 8:
            raise ValueError("at least eight replans required")
        n = len(ct)
        np.testing.assert_array_equal([x["commanded_action"] for x in rt[:n]],
                                      [x["commanded_action"] for x in ct])
        # Sequential telemetry stores float32; normalize serialization precision
        # before comparing simulator states (the first packed probe stored f64).
        np.testing.assert_array_equal(
            np.asarray([x["executed_state"] for x in rt[:n]], dtype=np.float32),
            np.asarray([x["executed_state"] for x in ct], dtype=np.float32))
        rows = []
        for x, y in zip(rf, cf):
            if x["start"] != y["start"] or not y["reference_commands_forced"]:
                raise ValueError("not paired fixed-history forecasts")
            r, c = np.asarray(x["actions"]), np.asarray(y["actions"])
            if r.shape != c.shape:
                raise ValueError("forecast shape mismatch")
            delta = c-r
            rows.append({
                "start": x["start"], "physical_mae": float(np.abs(delta).mean()),
                "range_normalized_mae": float(np.abs(delta/span).mean()),
                "range_normalized_max": float(np.abs(delta/span).max()),
                "relative_l2": float(np.linalg.norm(delta)/max(np.linalg.norm(r),1e-12)),
            })
        # Declared before observing candidates; this is numerical tolerance,
        # not evidence of identical autonomous behavior.
        passed = all(x["range_normalized_mae"] <= .005 and
                     x["range_normalized_max"] <= .02 and x["relative_l2"] <= .01 for x in rows)
        report = {"mode": a.mode, "pass": passed, "rows": rows,
                  "thresholds": {"range_normalized_mae": .005,
                                 "range_normalized_max": .02, "relative_l2": .01}}
    else:
        r, c = np.load(a.reference/"actions.npy"), np.load(a.candidate/"actions.npy")
        n = min(len(r), len(c))
        rmse = float(np.sqrt(np.mean(((c[:n]-r[:n])/span)**2)))
        target_differences = {}
        for name, target in rr["targets"].items():
            other = cr["targets"][name]
            target_differences[name] = {
                k: abs(other[k]-target[k]) for k in
                ("min_tcp_distance_m", "max_displacement_m", "max_lift_m", "grasp_controls")}
        passed = (len(r)==len(c) and rr["success"]==cr["success"] and
                  rr["final_state_success"]==cr["final_state_success"] and rmse <= .02 and
                  all(d["min_tcp_distance_m"] <= .005 and d["max_displacement_m"] <= .01
                      and d["max_lift_m"] <= .01 and d["grasp_controls"] <= 8
                      for d in target_differences.values()))
        report = {
            "mode": a.mode, "pass": passed, "controls": [len(r),len(c)],
            "success": [rr["success"],cr["success"]],
            "range_normalized_action_rmse": rmse, "target_absolute_differences": target_differences,
            "thresholds": {"range_normalized_action_rmse": .02,
                           "min_tcp_distance_m": .005, "max_displacement_m": .01,
                           "max_lift_m": .01, "grasp_controls": 8,
                           "same_control_count_and_success_required": True},
        }
    report["source_sha256"] = source_hashes
    report["bounds_sha256"] = sha256_file(bounds_path)
    atomic_json(a.output, report)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
