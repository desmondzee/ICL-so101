"""Paired local SO-101 cube-lift evaluation against a Modal Zero-WAM worker."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from zero_wam.so101_actions import CHANNELS, active_stats, decode, pack, xyzw_from_wxyz


def make_env(mode, cameras=True):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from so101_nexus.config import PickConfig
    from so101_nexus.mujoco.pick_env import PickLiftEnv
    from so101_nexus.observations import EndEffectorPose, JointPositions, OverheadCamera, WristCamera

    observations = [JointPositions(), EndEffectorPose()]
    if cameras:
        observations += [OverheadCamera(width=288, height=224), WristCamera(width=288, height=224)]
    # The default PickConfig places seed-100's cube near x=0.40 m, beyond a
    # downward-grasp configuration of this five-DOF arm. Use a small reachable
    # spawn region while retaining seed-dependent visual/object variation.
    cfg = PickConfig(
        observations=observations, terminate_on_success=False,
        spawn_center=(.25, 0.), spawn_min_radius=0., spawn_max_radius=.025,
    )
    return PickLiftEnv(config=cfg, control_mode="pd_ee_pose" if mode == "pose" else "pd_joint_pos", robot_init_qpos_noise=0)


def calibrate(seed=101, count=256):
    """Sweep simulator configurations and mix five unrelated basket demonstrations."""
    env = make_env("joint", cameras=False)
    env.reset(seed=seed)
    initial_joint = env._get_current_qpos().copy()
    initial_tcp = env._get_tcp_pose().copy()
    lows, highs = env.action_space.low, env.action_space.high
    rng = np.random.default_rng(seed)
    samples = {"joint": [], "pose": []}
    counts = {"sim": 0, "basket": 0, "contact_rejected": 0}
    targets = [initial_joint.copy()]
    # Smooth targets keep the sweep inside the arm's feasible joint limits.
    for _ in range(count):
        target = np.clip(initial_joint + rng.normal(0, [.65, .65, .65, .6, .8, .35]), lows, highs)
        target[5] = rng.uniform(lows[5], highs[5])
        targets.append(target)
    for target in targets:
        for alpha in np.linspace(0, 1, 8)[1:]:
            desired = np.clip(env._get_current_qpos() + alpha * (target - env._get_current_qpos()), lows, highs)
            env.step(desired)
            # Ignore resting cube/floor contacts; reject arm contacts with the scene.
            arm_contact = False
            for contact in env.data.contact[:env.data.ncon]:
                names = {env.model.body(env.model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)}
                if names - {"world", "pick_slot_0"} and float(contact.dist) < -.001:
                    arm_contact = True
                    break
            if arm_contact:
                counts["contact_rejected"] += 1
                continue
            joint, tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
            pct = 100 * (joint[5] - lows[5]) / (highs[5] - lows[5])
            for mode in samples:
                samples[mode].append(pack(mode, initial_joint, initial_tcp, joint, tcp, pct))
            counts["sim"] += 1
    env.close()
    import pyarrow.parquet as pq

    demo_root = Path(__file__).resolve().parents[1] / "data/examples/so101_libero_basket"
    for path in sorted(demo_root.glob("episode_*/data/chunk-000/file-000.parquet")):
        table = pq.read_table(path, columns=["action", "action.ee", "observation.state", "observation.environment_state"])
        joint0 = np.deg2rad(np.asarray(table["observation.state"][0].as_py(), float)[:5])
        tcp0 = np.asarray(table["observation.environment_state"][0].as_py(), float)[:7]
        for row in range(len(table)):
            joint_action = np.asarray(table["action"][row].as_py(), float)
            ee_action = np.asarray(table["action.ee"][row].as_py(), float)
            target_joint = np.r_[np.deg2rad(joint_action[:5]), ee_action[6]]
            target_tcp = np.r_[ee_action[:3], Rotation.from_rotvec(ee_action[3:6]).as_quat()[[3, 0, 1, 2]]]
            for mode in samples:
                samples[mode].append(pack(mode, joint0, tcp0, target_joint, target_tcp, joint_action[5]))
            counts["basket"] += 1
    return {mode: active_stats(mode, values) for mode, values in samples.items()}, counts


def run_episode(worker, mode, seed, stats, out_dir, max_steps=400):
    import imageio.v2 as imageio

    env = make_env(mode)
    obs, info = env.reset(seed=seed)
    prompt = info.get("task", "Pick up the red cube.")
    initial_joint, initial_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
    low, high = env.action_space.low, env.action_space.high
    # Pose-mode public space is Cartesian; joint limits come from the model targets.
    if mode == "pose":
        low, high = env._target_low, env._target_high
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worker.so101_step.remote({"reset": True, "mode": mode, "stats": stats, "prompt": prompt, "seed": seed})

    def formatted_observation(current_obs):
        measured_joint, measured_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
        pct = 100 * (measured_joint[5] - low[5]) / (high[5] - low[5])
        return {
            "observation.images.top": current_obs["overhead_camera"],
            "observation.images.wrist": current_obs["wrist_camera"],
            "observation.state": pack(mode, initial_joint, initial_tcp, measured_joint, measured_tcp, pct),
            "realized_tcp_pose": measured_tcp,
        }

    frames = [obs["overhead_camera"]]
    records, first = [], True
    success = False
    while len(records) < max_steps and not success:
        model_obs = formatted_observation(obs)
        result = worker.so101_step.remote({"obs": model_obs, "prompt": prompt})
        raw = np.asarray(result["action"])
        assert raw.shape[0] == len(CHANNELS[mode]) and raw.shape[2] % 4 == 0, raw.shape
        history = np.zeros_like(raw)
        keyframes = []
        for f in range(1 if first else 0, raw.shape[1]):
            for h in range(raw.shape[2]):
                if len(records) >= max_steps or success:
                    break
                before_joint, before_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
                cmd, diagnostics = decode(mode, raw[:, f, h], initial_joint, initial_tcp, before_joint, before_tcp, low, high)
                ik_joint_target = env._action_to_ctrl(cmd).copy() if mode == "pose" else None
                obs, _, _, _, info = env.step(cmd)
                after_joint, after_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
                pct = 100 * (after_joint[5] - low[5]) / (high[5] - low[5])
                history[:, f, h] = pack(mode, initial_joint, initial_tcp, after_joint, after_tcp, pct)
                record = {"raw": raw[:, f, h].tolist(), "command": cmd.tolist(), "executed": history[:, f, h].tolist(), "realized_tcp": after_tcp.tolist(), "clipped": bool(diagnostics["clipped"]), "success": bool(info.get("success", False))}
                if mode == "pose":
                    requested_xyz = initial_tcp[:3] + raw[:3, f, h]
                    requested_q = raw[3:7, f, h]
                    requested_q = requested_q / np.linalg.norm(requested_q) if np.all(np.isfinite(requested_q)) and np.linalg.norm(requested_q) > 1e-6 else np.array([0., 0., 0., 1.])
                    requested_rot = Rotation.from_quat(xyzw_from_wxyz(initial_tcp[3:7])) * Rotation.from_quat(requested_q)
                    realized_rot = Rotation.from_quat(xyzw_from_wxyz(after_tcp[3:7]))
                    commanded_rot = Rotation.from_rotvec(cmd[3:6])
                    record.update(ik_joint_target=ik_joint_target.tolist(), requested_position_error_m=float(np.linalg.norm(requested_xyz - after_tcp[:3])), requested_orientation_error_rad=float((requested_rot.inv() * realized_rot).magnitude()), commanded_position_error_m=float(np.linalg.norm(cmd[:3] - after_tcp[:3])), commanded_orientation_error_rad=float((commanded_rot.inv() * realized_rot).magnitude()))
                records.append(record)
                success = bool(info.get("success", False))
                if len(records) % 4 == 0:
                    frames.append(obs["overhead_camera"])
                if (h + 1) % (raw.shape[2] // 4) == 0:
                    keyframes.append(formatted_observation(obs))
            if len(records) >= max_steps or success:
                break
        first = False
        if not success and len(records) < max_steps:
            worker.so101_step.remote({"obs": keyframes, "compute_kv_cache": True, "state": history, "executed_model_actions": history})
    env.close()
    with (out_dir / "actions.jsonl").open("w") as file:
        for row in records:
            file.write(json.dumps(row) + "\n")
    imageio.mimsave(out_dir / "overhead.mp4", frames, fps=12)
    summary = {"mode": mode, "seed": seed, "success": success, "steps": len(records), "clip_fraction": float(np.mean([r["clipped"] for r in records])) if records else 0, "prompt": prompt}
    if mode == "pose" and records:
        for key in ("requested_position_error_m", "requested_orientation_error_rad", "commanded_position_error_m", "commanded_orientation_error_rad"):
            summary[f"mean_{key}"] = float(np.mean([r[key] for r in records]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
