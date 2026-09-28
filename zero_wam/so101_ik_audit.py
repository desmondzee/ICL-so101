"""Replay a pose trace and compare IK convergence independently of physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from zero_wam.so101_eval import make_env
import so101_nexus.mujoco.base_env as base_env


def audit(trial: Path, iterations: int = 80) -> dict:
    manifest = json.loads((trial / "trial.json").read_text())
    assert manifest["variant"]["mode"] == "pose"
    records = [json.loads(line) for line in (trial / "actions.jsonl").read_text().splitlines()]
    env = make_env("pose", cameras=False, task=manifest["task"])
    env.reset(seed=manifest["seed"])
    default_iterations = base_env.EE_IK_ITERATIONS
    errors, replay_errors = [], []
    try:
        for record in records:
            command = np.asarray(record["command"])
            target_rotation = Rotation.from_rotvec(command[3:6])
            row = {}
            for count in (default_iterations, iterations):
                base_env.EE_IK_ITERATIONS = count
                joint_target = env._action_to_ctrl(command)
                data = env._ik_data
                data.qpos[env._arm_qpos_addrs] = joint_target[:5]
                mujoco.mj_forward(env.model, data)
                realized_rotation = Rotation.from_matrix(data.site_xmat[env._tcp_site_id].reshape(3, 3))
                row[str(count)] = {
                    "position_m": float(np.linalg.norm(command[:3] - data.site_xpos[env._tcp_site_id])),
                    "orientation_rad": float((target_rotation.inv() * realized_rotation).magnitude()),
                    "max_penetration_m": max((max(0., -float(c.dist)) for c in data.contact), default=0.),
                }
            errors.append(row)
            base_env.EE_IK_ITERATIONS = default_iterations
            env.step(command)
            replay_errors.append(float(np.linalg.norm(env._get_tcp_pose()[:3] - np.asarray(record["realized_tcp"])[:3])))
    finally:
        base_env.EE_IK_ITERATIONS = default_iterations
        env.close()
    result = {"trial": str(trial), "controls": len(errors), "replay_mean_position_difference_m": float(np.mean(replay_errors)), "ik": {}}
    for count in (default_iterations, iterations):
        pos = np.array([row[str(count)]["position_m"] for row in errors])
        rot = np.array([row[str(count)]["orientation_rad"] for row in errors])
        result["ik"][str(count)] = {"mean_position_m": float(pos.mean()), "median_position_m": float(np.median(pos)), "mean_orientation_rad": float(rot.mean()), "position_within_1cm_fraction": float(np.mean(pos < .01))}
        result["ik"][str(count)]["penetration_over_1mm_fraction"] = float(np.mean([row[str(count)]["max_penetration_m"] > .001 for row in errors]))
    # Open-loop controller intervention: do not claim these are policy trials,
    # since later model predictions were generated from a different trajectory.
    result["open_loop_execution"] = {}
    for hold, fixed_joint in ((1, False), (5, False), (5, True), (25, True)):
        env = make_env("pose", cameras=False, task=manifest["task"])
        env.reset(seed=manifest["seed"])
        positions, orientations, joint_errors, penetrations, grasps = [], [], [], [], 0
        try:
            base_env.EE_IK_ITERATIONS = iterations
            for record in records:
                command = np.asarray(record["command"])
                if fixed_joint:
                    env.data.ctrl[env._actuator_ids] = env._action_to_ctrl(command)
                    mujoco.mj_step(env.model, env.data, nstep=hold * env._N_SUBSTEPS)
                else:
                    for _ in range(hold):
                        env.step(command)
                target_joint = env.data.ctrl[env._actuator_ids].copy()
                joint_errors.append(float(np.linalg.norm(target_joint[:5] - env._get_current_qpos()[:5])))
                penetrations.append(max((max(0., -float(c.dist)) for c in env.data.contact), default=0.))
                tcp = env._get_tcp_pose()
                positions.append(float(np.linalg.norm(command[:3] - tcp[:3])))
                actual_rotation = Rotation.from_quat(tcp[[4, 5, 6, 3]])
                orientations.append(float((Rotation.from_rotvec(command[3:6]).inv() * actual_rotation).magnitude()))
                if manifest["task"] != "cube":
                    grasps += bool(env.is_grasping("alphabet_soup"))
            label = f"ik{iterations}_{'fixed_joint_' if fixed_joint else ''}hold{hold}"
            result["open_loop_execution"][label] = {"mean_position_m": float(np.mean(positions)), "mean_orientation_rad": float(np.mean(orientations)), "mean_arm_joint_target_error_norm_rad": float(np.mean(joint_errors)), "penetration_over_1mm_fraction": float(np.mean(np.asarray(penetrations) > .001)), "soup_grasp_controls": grasps, "control_dt_s": env.control_dt, "seconds_per_prediction": hold * env.control_dt}
        finally:
            base_env.EE_IK_ITERATIONS = default_iterations
            env.close()
    (trial / "ik_convergence_audit.json").write_text(json.dumps({**result, "per_control": errors}, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trial", type=Path)
    parser.add_argument("--iterations", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(audit(args.trial, args.iterations), indent=2))
