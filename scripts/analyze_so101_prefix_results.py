"""Analyze completed block-prefix handovers, separating oracle and policy steps."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"docs/results/zero_wam_oracle_prefix"
RUN = ROOT/"outputs/zero_wam/block_prefix_handover_scenev2_b4"
DATA = ROOT/"data/hf_bucket_ICL-so101/train_replay_probe/episodes"


def main():
    plan = json.loads((OUT/"prefix_manifest.json").read_text())
    measurements = []
    for condition in plan["conditions"]:
        source, b = condition["source_episode_id"], condition["prefix_controls"]
        directory = RUN/f"{source}__prefix_{b:04d}"
        r = json.loads((directory/"result.json").read_text())
        ts = [json.loads(l) for l in (directory/"telemetry.jsonl").read_text().splitlines()]
        table = pd.read_parquet(DATA/source/"robot_data.parquet")
        gt = np.stack(table.action)
        assert len(ts) == r["controls"]
        assert r["policy_controls"] == len(ts)-b > 0
        assert r["source_episode_id"] == source and r["prefix_controls"] == b
        assert r["partial_protocol_sha256"] == hashlib.sha256((RUN/"manifest.json").read_bytes()).hexdigest()
        assert all(t["control_source"] == ("oracle_prefix" if i < b else "policy")
                   for i,t in enumerate(ts))
        if b:
            np.testing.assert_array_equal([t["commanded_action"] for t in ts[:b]], gt[:b])
        np.testing.assert_array_equal([t["commanded_action"] for t in ts[b:]],
                                      [t["predicted_action"] for t in ts[b:]])
        assert not r["handover"]["success"]
        np.testing.assert_allclose(r["handover"]["state"],condition["expected_handover"]["state"],
                                   atol=1e-5,rtol=0)
        autonomous = ts[b:]
        held = np.array([t["physical_state"]["objects"]["block"]["grasped"] for t in autonomous])
        xyz = np.array([t["physical_state"]["objects"]["block"]["xyz"] for t in autonomous])
        error = np.array([t["physical_state"]["goals"]["block"]["error_xyz"] for t in autonomous])
        zero = next(c for c in plan["conditions"] if c["source_episode_id"]==source and c["prefix_controls"]==0)
        initial_z = zero["expected_handover"]["objects"]["block"]["xyz"][2]
        intervals, begin = [], None
        for j,value in enumerate(np.r_[held,False]):
            if value and begin is None:
                begin = j
            elif not value and begin is not None:
                intervals.append([b+begin+1,b+j])
                begin = None
        oracle_next = gt[b:min(b+8,len(gt))]
        predicted_next = np.array([t["commanded_action"] for t in autonomous[:len(oracle_next)]])
        first_loss = None
        if r["handover"]["objects"]["block"]["grasped"] and (~held).any():
            j = int(np.flatnonzero(~held)[0])
            first_loss = {"control":b+j+1, "seconds_after_handover":(j+1)/30,
                          "goal_error_xyz_m":error[j].tolist(),
                          "height_above_initial_m":float(xyz[j,2]-initial_z)}
        tol = json.loads((DATA/source/"policy.json").read_text())["goals"][0]["position_tolerance"]
        measurements.append({
            "source_episode_id":source, "stage":condition["prefix_stage"],"prefix_controls":b,
            "controls":len(ts),"policy_controls":len(autonomous),
            "success":r["success"],"first_success_control":r["first_success_control"],
            "final_success":r["final_state_success"],
            "policy_grasp_controls":int(held.sum()),"policy_grasp_intervals":intervals,
            "first_loss_after_held_handover":first_loss,
            "max_policy_lift_cm":float((xyz[:,2]-initial_z).max()*100),
            "min_policy_goal_distance_cm":float(np.linalg.norm(error,axis=1).min()*100),
            "final_goal_error_xyz_cm":(error[-1]*100).tolist(),
            "policy_controls_inside_position_tolerance":int((np.abs(error)<=tol).all(axis=1).sum()),
            "first_policy8_arm_mae_vs_oracle_deg":float(np.abs(predicted_next[:,:5]-oracle_next[:,:5]).mean()),
            "first_policy8_gripper_mae_vs_oracle_points":float(np.abs(predicted_next[:,5]-oracle_next[:,5]).mean()),
            "result_sha256":hashlib.sha256((directory/"result.json").read_bytes()).hexdigest(),
            "telemetry_sha256":hashlib.sha256((directory/"telemetry.jsonl").read_bytes()).hexdigest(),
        })
    (OUT/"prefix_measurements.json").write_text(json.dumps({
        "analysis_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions":measurements},indent=2)+"\n")
    for row in measurements:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
