"""Paired local SO-101 cube-lift evaluation against a Modal Zero-WAM worker."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from zero_wam.so101_actions import CHANNELS, active_stats, decode, pack, xyzw_from_wxyz


@dataclass(frozen=True)
class TrialVariant:
    name: str
    mode: str
    channels: tuple[int, ...]
    joint_reference: str = "relative"
    gripper_mapping: str = "normal"
    camera_view: str = "overhead"
    image_width: int = 288
    image_height: int = 224
    normalization: str = "mixed"
    wrist_tilt_deg: float = 0.
    prompt: str | None = None


TRIAL_VARIANTS = {
    "pose": TrialVariant("pose", "pose", tuple(CHANNELS["pose"])),
    "pose_grip_inverted": TrialVariant("pose_grip_inverted", "pose", tuple(CHANNELS["pose"]), gripper_mapping="inverted"),
    "pose_grip_scaled20": TrialVariant("pose_grip_scaled20", "pose", tuple(CHANNELS["pose"]), gripper_mapping="scaled20"),
    "pose_grip_offset20": TrialVariant("pose_grip_offset20", "pose", tuple(CHANNELS["pose"]), gripper_mapping="offset20"),
    "pose_front": TrialVariant("pose_front", "pose", tuple(CHANNELS["pose"]), camera_view="front"),
    "pose_wrist_tilt40": TrialVariant("pose_wrist_tilt40", "pose", tuple(CHANNELS["pose"]), wrist_tilt_deg=40.),
    "pose_native256": TrialVariant("pose_native256", "pose", tuple(CHANNELS["pose"]), image_width=256, image_height=256),
    "pose_sim_stats": TrialVariant("pose_sim_stats", "pose", tuple(CHANNELS["pose"]), normalization="sim"),
    "pose_basket_stats": TrialVariant("pose_basket_stats", "pose", tuple(CHANNELS["pose"]), normalization="basket"),
    "pose_prompt_lift": TrialVariant("pose_prompt_lift", "pose", tuple(CHANNELS["pose"]), prompt="Grasp the red cube and lift it off the table."),
    "joint": TrialVariant("joint", "joint", tuple(CHANNELS["joint"])),
    "joint_grip_inverted": TrialVariant("joint_grip_inverted", "joint", tuple(CHANNELS["joint"]), gripper_mapping="inverted"),
    "joint_grip_scaled20": TrialVariant("joint_grip_scaled20", "joint", tuple(CHANNELS["joint"]), gripper_mapping="scaled20"),
    "joint_absolute": TrialVariant("joint_absolute", "joint", tuple(CHANNELS["joint"]), joint_reference="absolute"),
    "joint_demo_channels": TrialVariant("joint_demo_channels", "joint", (0, 1, 2, 3, 4, 28)),
    "joint_demo_published": TrialVariant("joint_demo_published", "joint", (0, 1, 2, 3, 4, 28), joint_reference="demo_degrees_absolute"),
    "joint_demo_published_inverted": TrialVariant("joint_demo_published_inverted", "joint", (0, 1, 2, 3, 4, 28), joint_reference="demo_degrees_absolute", gripper_mapping="inverted"),
    "joint_front": TrialVariant("joint_front", "joint", tuple(CHANNELS["joint"]), camera_view="front"),
    "joint_wrist_tilt40": TrialVariant("joint_wrist_tilt40", "joint", tuple(CHANNELS["joint"]), wrist_tilt_deg=40.),
    "joint_native256": TrialVariant("joint_native256", "joint", tuple(CHANNELS["joint"]), image_width=256, image_height=256),
    "joint_native256_grip_inverted": TrialVariant("joint_native256_grip_inverted", "joint", tuple(CHANNELS["joint"]), gripper_mapping="inverted", image_width=256, image_height=256),
    "joint_native256_grip_scaled20": TrialVariant("joint_native256_grip_scaled20", "joint", tuple(CHANNELS["joint"]), gripper_mapping="scaled20", image_width=256, image_height=256),
    "joint_native256_front": TrialVariant("joint_native256_front", "joint", tuple(CHANNELS["joint"]), camera_view="front", image_width=256, image_height=256),
    "joint_native256_prompt_lift": TrialVariant("joint_native256_prompt_lift", "joint", tuple(CHANNELS["joint"]), image_width=256, image_height=256, prompt="Grasp the red cube and lift it off the table."),
    "joint_sim_stats": TrialVariant("joint_sim_stats", "joint", tuple(CHANNELS["joint"]), normalization="sim"),
    "joint_basket_stats": TrialVariant("joint_basket_stats", "joint", tuple(CHANNELS["joint"]), normalization="basket"),
}


def variant_stats(variant: TrialVariant, stats: dict, home_joint: np.ndarray) -> dict:
    """Map calibrated physical quantiles to the selected channels/reference."""
    if variant.joint_reference == "demo_degrees_absolute":
        # Exact values from third_party/Zero-WAM/wan_va/configs/va_demo_cfg.py.
        q01, q99 = np.zeros(30), np.zeros(30)
        q01[:5] = [-90.60303497314453, -98.73043060302734, -79.9008560180664, 48.95470428466797, -32.794578552246094]
        q99[:5] = [71.735107421875, 65.89081573486328, 92.87967681884766, 100.0, 22.784151077270508]
        q01[28], q99[28] = 0.8250824809074402, 100.0
        return {"q01": q01.tolist(), "q99": q99.tolist()}
    source = stats[variant.mode]
    q01, q99 = np.zeros(30), np.zeros(30)
    canonical = CHANNELS[variant.mode]
    q01[list(variant.channels)] = np.asarray(source["q01"])[canonical]
    q99[list(variant.channels)] = np.asarray(source["q99"])[canonical]
    if variant.mode == "joint" and variant.joint_reference == "absolute":
        q01[list(variant.channels[:5])] += home_joint[:5]
        q99[list(variant.channels[:5])] += home_joint[:5]
    return {"q01": q01.tolist(), "q99": q99.tolist()}


def physical_gripper(value: float, mapping: str) -> float:
    return {
        "normal": lambda v: v,
        "inverted": lambda v: 100 - v,
        "scaled20": lambda v: 20 * v,
        "offset20": lambda v: 20 * (v + 1),
    }[mapping](value)


def model_gripper(physical_pct: float, mapping: str) -> float:
    return {
        "normal": lambda v: v,
        "inverted": lambda v: 100 - v,
        "scaled20": lambda v: v / 20,
        "offset20": lambda v: v / 20 - 1,
    }[mapping](physical_pct)


def make_env(mode, cameras=True, width=288, height=224):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from so101_nexus.config import PickConfig
    from so101_nexus.mujoco.pick_env import PickLiftEnv
    from so101_nexus.observations import EndEffectorPose, JointPositions, OverheadCamera, WristCamera

    observations = [JointPositions(), EndEffectorPose()]
    if cameras:
        observations += [OverheadCamera(width=width, height=height), WristCamera(width=width, height=height)]
    # The default PickConfig places seed-100's cube near x=0.40 m, beyond a
    # downward-grasp configuration of this five-DOF arm. Use a small reachable
    # spawn region while retaining seed-dependent visual/object variation.
    cfg = PickConfig(
        observations=observations, terminate_on_success=False,
        spawn_center=(.25, 0.), spawn_min_radius=0., spawn_max_radius=.025,
    )
    return PickLiftEnv(config=cfg, control_mode="pd_ee_pose" if mode == "pose" else "pd_joint_pos", robot_init_qpos_noise=0)


def calibrate(seed=101, count=256, source="mixed"):
    """Sweep simulator configurations and mix five unrelated basket demonstrations."""
    env = make_env("joint", cameras=False)
    env.reset(seed=seed)
    initial_joint = env._get_current_qpos().copy()
    initial_tcp = env._get_tcp_pose().copy()
    lows, highs = env.action_space.low, env.action_space.high
    rng = np.random.default_rng(seed)
    if source not in {"mixed", "sim", "basket"}:
        raise ValueError(source)
    samples = {"sim": {"joint": [], "pose": []}, "basket": {"joint": [], "pose": []}}
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
            for mode in ("joint", "pose"):
                samples["sim"][mode].append(pack(mode, initial_joint, initial_tcp, joint, tcp, pct))
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
            for mode in ("joint", "pose"):
                samples["basket"][mode].append(pack(mode, joint0, tcp0, target_joint, target_tcp, joint_action[5]))
            counts["basket"] += 1
    selected = {mode: samples["sim"][mode] + samples["basket"][mode] if source == "mixed" else samples[source][mode] for mode in ("joint", "pose")}
    return {mode: active_stats(mode, values) for mode, values in selected.items()}, counts


def run_episode(worker, mode, seed, stats, out_dir, max_steps=400, variant: TrialVariant | None = None):
    import imageio.v2 as imageio

    variant = variant or TRIAL_VARIANTS[mode]
    if variant.mode != mode:
        raise ValueError(f"Variant {variant.name} is {variant.mode}, not {mode}")
    env = make_env(mode, width=variant.image_width, height=variant.image_height)
    obs, info = env.reset(seed=seed)
    if variant.camera_view == "front":
        camera = env._overhead_obs_cam
        camera.azimuth = 270.
        camera.elevation = -30.
        camera.distance = .65
        camera.lookat[:] = [.22, 0., .025]
    elif variant.camera_view != "overhead":
        raise ValueError(variant.camera_view)
    if variant.wrist_tilt_deg:
        import mujoco

        camera_id = env._wrist_cam_id
        base_wxyz = env.model.cam_quat[camera_id].copy()
        rotated = (Rotation.from_quat(xyzw_from_wxyz(base_wxyz)) * Rotation.from_euler("x", variant.wrist_tilt_deg, degrees=True)).as_quat()
        env.model.cam_quat[camera_id] = rotated[[3, 0, 1, 2]]
        mujoco.mj_forward(env.model, env.data)
    if variant.camera_view != "overhead" or variant.wrist_tilt_deg:
        obs = env._get_obs()
    prompt = variant.prompt or info.get("task", "Pick up the red cube.")
    initial_joint, initial_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
    low, high = env.action_space.low, env.action_space.high
    # Pose-mode public space is Cartesian; joint limits come from the model targets.
    if mode == "pose":
        low, high = env._target_low, env._target_high
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worker.so101_step.remote({"reset": True, "mode": mode, "channels": list(variant.channels), "stats": stats, "prompt": prompt, "seed": seed})

    def encoded_state(measured_joint, measured_tcp):
        pct = 100 * (measured_joint[5] - low[5]) / (high[5] - low[5])
        state = pack(mode, initial_joint, initial_tcp, measured_joint, measured_tcp, model_gripper(pct, variant.gripper_mapping))
        if mode == "joint" and variant.joint_reference == "absolute":
            state[:5] = measured_joint[:5]
        elif mode == "joint" and variant.joint_reference == "demo_degrees_absolute":
            state[:5] = np.rad2deg(measured_joint[:5])
        return state

    def formatted_observation(current_obs):
        measured_joint, measured_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
        return {
            "observation.images.top": current_obs["overhead_camera"],
            "observation.images.wrist": current_obs["wrist_camera"],
            "observation.state": encoded_state(measured_joint, measured_tcp),
            "realized_tcp_pose": measured_tcp,
        }

    frames = [obs["overhead_camera"]]
    records, first = [], True
    success = False
    while len(records) < max_steps and not success:
        model_obs = formatted_observation(obs)
        result = worker.so101_step.remote({"obs": model_obs, "prompt": prompt})
        raw = np.asarray(result["action"])
        assert raw.shape[0] == len(variant.channels) and raw.shape[2] % 4 == 0, raw.shape
        history = np.zeros_like(raw)
        keyframes = []
        for f in range(1 if first else 0, raw.shape[1]):
            for h in range(raw.shape[2]):
                if len(records) >= max_steps or success:
                    break
                before_joint, before_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
                model_action = raw[:, f, h].copy()
                physical_action = model_action.copy()
                physical_action[-1] = physical_gripper(model_action[-1], variant.gripper_mapping)
                reference_joint = initial_joint if variant.joint_reference == "relative" else np.zeros_like(initial_joint)
                if variant.joint_reference == "demo_degrees_absolute":
                    physical_action[:5] = np.deg2rad(physical_action[:5])
                cmd, diagnostics = decode(mode, physical_action, reference_joint, initial_tcp, before_joint, before_tcp, low, high)
                ik_joint_target = env._action_to_ctrl(cmd).copy() if mode == "pose" else None
                obs, _, _, _, info = env.step(cmd)
                after_joint, after_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
                history[:, f, h] = encoded_state(after_joint, after_tcp)
                record = {"raw": raw[:, f, h].tolist(), "physical_action": physical_action.tolist(), "command": cmd.tolist(), "executed": history[:, f, h].tolist(), "realized_tcp": after_tcp.tolist(), "gripper_rad": float(after_joint[5]), "tcp_to_obj_dist_m": float(info["tcp_to_obj_dist"]), "lift_height_m": float(info["lift_height"]), "is_grasped": bool(info["is_grasped"]), "clipped": bool(diagnostics["clipped"]), "diagnostics": diagnostics, "success": bool(info.get("success", False))}
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
    try:
        from zero_wam.contact_sheet import write_sheet

        write_sheet(out_dir / "overhead.mp4", 24, True, out_dir / "overhead.contact.png")
    except Exception as exc:
        print(f"contact sheet failed for {out_dir}: {exc}")
    summary = {"variant": variant.name, "mode": mode, "seed": seed, "success": success, "steps": len(records), "clip_fraction": float(np.mean([r["clipped"] for r in records])) if records else 0, "prompt": prompt}
    if records:
        summary.update(min_tcp_to_object_m=float(min(r["tcp_to_obj_dist_m"] for r in records)), max_lift_m=float(max(r["lift_height_m"] for r in records)), grasp_steps=sum(r["is_grasped"] for r in records), min_tcp_z_m=float(min(r["realized_tcp"][2] for r in records)))
        for key in ("gripper_range_clipped", "gripper_rate_clipped", "arm_range_clipped", "workspace_clipped"):
            if key in records[0]["diagnostics"]:
                summary[f"{key}_fraction"] = float(np.mean([r["diagnostics"][key] for r in records]))
    if mode == "pose" and records:
        for key in ("requested_position_error_m", "requested_orientation_error_rad", "commanded_position_error_m", "commanded_orientation_error_rad"):
            summary[f"mean_{key}"] = float(np.mean([r[key] for r in records]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_trial(worker, variant_name: str, seed: int, max_steps: int, save_root: str, checkpoint: str) -> dict:
    """Run one named one-factor trial with a self-contained manifest."""
    variant = TRIAL_VARIANTS[variant_name]
    root = Path(save_root) / checkpoint / variant.name / f"seed{seed}_steps{max_steps}"
    root.mkdir(parents=True, exist_ok=True)
    all_stats, counts = calibrate(source=variant.normalization)
    home = make_env("joint", cameras=False)
    home.reset(seed=101)
    home_joint = home._get_current_qpos().copy()
    home.close()
    stats = variant_stats(variant, all_stats, home_joint)
    revisions = {
        "posttrain": "07ee865f175d9474a5653e9147698390e47b6143",
        "pretrain": "7040c4195df216c900334ef62d5fdcf05c0601aa",
    }
    manifest = {"checkpoint": checkpoint, "checkpoint_revision": revisions[checkpoint], "variant": asdict(variant), "seed": seed, "max_steps": max_steps, "calibration_counts": counts, "norm_stats": stats}
    (root / "trial.json").write_text(json.dumps(manifest, indent=2))
    return run_episode(worker, variant.mode, seed, stats, root, max_steps, variant)
