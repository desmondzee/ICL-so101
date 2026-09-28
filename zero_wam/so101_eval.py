"""Paired local SO-101 cube-lift evaluation against a Modal Zero-WAM worker."""

from __future__ import annotations

import json
import os
import hashlib
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from zero_wam.so101_actions import CHANNELS, active_stats, decode, pack, xyzw_from_wxyz, robotwin_to_tcp_relative, tcp_to_robotwin_relative

# Freeze provenance before long Modal trials; later worktree edits must not
# silently change the source hashes recorded for this loaded runner.
_REPO = Path(__file__).resolve().parents[1]
_SOURCE_FILES = ("zero_wam/so101_eval.py", "zero_wam/so101_actions.py", "zero_wam/so101_runtime.py", "zero_wam/modal_app.py", "zero_wam/modal_so101_pretrain.py", "sim/libero_basket_env.py")
_SOURCE_SNAPSHOT = {name: (_REPO / name).read_bytes() for name in _SOURCE_FILES}
_CODE_REVISION = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
_CODE_PATCH = subprocess.check_output(["git", "diff", "--", "zero_wam", "sim"], cwd=_REPO)


@dataclass(frozen=True)
class TrialVariant:
    name: str
    mode: str
    channels: tuple[int, ...]
    joint_reference: str = "relative"
    gripper_mapping: str = "normal"
    camera_view: str = "overhead"
    camera_layout: str = "two"
    pose_translation_yaw_deg: float = 0.
    model_channels: tuple[int, ...] = ()
    inference_steps: int = 0
    rate_limits: bool = True
    tool_frame: str = "tcp"
    quaternion_convention: str = "physical"
    native_robotwin_cameras: bool = False
    native_initial_state: bool = False
    head_azimuth_offset_deg: float = 0.
    head_distance_m: float = .75
    image_width: int = 288
    image_height: int = 224
    normalization: str = "mixed"
    calibration_at_initial_state: bool = False
    wrist_tilt_deg: float = 0.
    prompt: str | None = None


TRIAL_VARIANTS = {
    "pose": TrialVariant("pose", "pose", tuple(CHANNELS["pose"])),
    "pose_full_denoise": TrialVariant("pose_full_denoise", "pose", tuple(CHANNELS["pose"]), image_width=256, image_height=256, inference_steps=50),
    "joint_full_denoise": TrialVariant("joint_full_denoise", "joint", tuple(CHANNELS["joint"]), image_width=256, image_height=256, inference_steps=50),
    "pose_full_denoise_no_rate": TrialVariant("pose_full_denoise_no_rate", "pose", tuple(CHANNELS["pose"]), image_width=256, image_height=256, inference_steps=50, rate_limits=False),
    "joint_full_denoise_no_rate": TrialVariant("joint_full_denoise_no_rate", "joint", tuple(CHANNELS["joint"]), image_width=256, image_height=256, inference_steps=50, rate_limits=False),
    "pose_three_full_yaw_no_rate": TrialVariant("pose_three_full_yaw_no_rate", "pose", tuple(CHANNELS["pose"]), camera_layout="three_left_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90., model_channels=tuple(range(14)) + (28, 29), rate_limits=False),
    "joint_three_no_rate": TrialVariant("joint_three_no_rate", "joint", tuple(CHANNELS["joint"]), camera_layout="three_left_blank", rate_limits=False),
    "pose_full_denoise_task_front": TrialVariant("pose_full_denoise_task_front", "pose", tuple(CHANNELS["pose"]), image_width=256, image_height=256, inference_steps=50, camera_view="task_front"),
    "joint_full_denoise_task_front": TrialVariant("joint_full_denoise_task_front", "joint", tuple(CHANNELS["joint"]), image_width=256, image_height=256, inference_steps=50, camera_view="task_front"),
    "pose_three_full_yaw_task_front": TrialVariant("pose_three_full_yaw_task_front", "pose", tuple(CHANNELS["pose"]), camera_layout="three_left_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90., model_channels=tuple(range(14)) + (28, 29), camera_view="task_front"),
    "joint_three_task_front": TrialVariant("joint_three_task_front", "joint", tuple(CHANNELS["joint"]), camera_layout="three_left_blank", camera_view="task_front"),
    "pose_robotwin_tool_left": TrialVariant("pose_robotwin_tool_left", "pose", tuple(CHANNELS["pose"]), camera_layout="three_left_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90., model_channels=tuple(range(14)) + (28, 29), camera_view="task_front", tool_frame="robotwin_ee"),
    "pose_robotwin_tool_right": TrialVariant("pose_robotwin_tool_right", "pose", (7, 8, 9, 10, 11, 12, 13, 29), camera_layout="three_right_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90., model_channels=tuple(range(14)) + (28, 29), camera_view="task_front", tool_frame="robotwin_ee"),
    "pose_robotwin_client_left": TrialVariant("pose_robotwin_client_left", "pose", tuple(CHANNELS["pose"]), camera_layout="three_left_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90., model_channels=tuple(range(14)) + (28, 29), camera_view="task_front", tool_frame="robotwin_ee", quaternion_convention="native_client"),
    "pose_robotwin_client_right": TrialVariant("pose_robotwin_client_right", "pose", (7, 8, 9, 10, 11, 12, 13, 29), camera_layout="three_right_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90., model_channels=tuple(range(14)) + (28, 29), camera_view="task_front", tool_frame="robotwin_ee", quaternion_convention="native_client"),
    "pose_robotwin_cameras_left": TrialVariant("pose_robotwin_cameras_left", "pose", tuple(CHANNELS["pose"]), camera_layout="three_left_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90., model_channels=tuple(range(14)) + (28, 29), camera_view="robotwin_head", tool_frame="robotwin_ee", quaternion_convention="native_client", native_robotwin_cameras=True, image_width=320, image_height=240),
    "pose_robotwin_cameras_right": TrialVariant("pose_robotwin_cameras_right", "pose", (7, 8, 9, 10, 11, 12, 13, 29), camera_layout="three_right_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90., model_channels=tuple(range(14)) + (28, 29), camera_view="robotwin_head", tool_frame="robotwin_ee", quaternion_convention="native_client", native_robotwin_cameras=True, image_width=320, image_height=240),
    "pose_three_left_blank": TrialVariant("pose_three_left_blank", "pose", tuple(CHANNELS["pose"]), camera_layout="three_left_blank"),
    "pose_three_right_blank": TrialVariant("pose_three_right_blank", "pose", (7, 8, 9, 10, 11, 12, 13, 29), camera_layout="three_right_blank"),
    "pose_three_duplicate": TrialVariant("pose_three_duplicate", "pose", tuple(CHANNELS["pose"]), camera_layout="three_duplicate"),
    "pose_three_left_front": TrialVariant("pose_three_left_front", "pose", tuple(CHANNELS["pose"]), camera_view="front", camera_layout="three_left_blank"),
    "pose_three_left_native": TrialVariant("pose_three_left_native", "pose", tuple(CHANNELS["pose"]), camera_layout="three_left_blank", gripper_mapping="unit100", normalization="robotwin"),
    "pose_three_right_native": TrialVariant("pose_three_right_native", "pose", (7, 8, 9, 10, 11, 12, 13, 29), camera_layout="three_right_blank", gripper_mapping="unit100", normalization="robotwin"),
    "pose_three_duplicate_native": TrialVariant("pose_three_duplicate_native", "pose", tuple(CHANNELS["pose"]), camera_layout="three_duplicate", gripper_mapping="unit100", normalization="robotwin"),
    "pose_three_left_native_front": TrialVariant("pose_three_left_native_front", "pose", tuple(CHANNELS["pose"]), camera_view="front", camera_layout="three_left_blank", gripper_mapping="unit100", normalization="robotwin"),
    "pose_three_left_native_yawneg90": TrialVariant("pose_three_left_native_yawneg90", "pose", tuple(CHANNELS["pose"]), camera_layout="three_left_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90.),
    "pose_three_right_native_yawneg90": TrialVariant("pose_three_right_native_yawneg90", "pose", (7, 8, 9, 10, 11, 12, 13, 29), camera_layout="three_right_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90.),
    "pose_three_left_native_full_yawneg90": TrialVariant("pose_three_left_native_full_yawneg90", "pose", tuple(CHANNELS["pose"]), camera_layout="three_left_blank", gripper_mapping="unit100", normalization="robotwin", pose_translation_yaw_deg=-90., model_channels=tuple(range(14)) + (28, 29)),
    "pose_grip_inverted": TrialVariant("pose_grip_inverted", "pose", tuple(CHANNELS["pose"]), gripper_mapping="inverted"),
    "pose_grip_scaled20": TrialVariant("pose_grip_scaled20", "pose", tuple(CHANNELS["pose"]), gripper_mapping="scaled20"),
    "pose_grip_offset20": TrialVariant("pose_grip_offset20", "pose", tuple(CHANNELS["pose"]), gripper_mapping="offset20"),
    "pose_front": TrialVariant("pose_front", "pose", tuple(CHANNELS["pose"]), camera_view="front"),
    "pose_wrist_tilt40": TrialVariant("pose_wrist_tilt40", "pose", tuple(CHANNELS["pose"]), wrist_tilt_deg=40.),
    "pose_native256": TrialVariant("pose_native256", "pose", tuple(CHANNELS["pose"]), image_width=256, image_height=256),
    "pose_native256_front": TrialVariant("pose_native256_front", "pose", tuple(CHANNELS["pose"]), camera_view="front", image_width=256, image_height=256),
    "pose_sim_stats": TrialVariant("pose_sim_stats", "pose", tuple(CHANNELS["pose"]), normalization="sim"),
    "pose_basket_stats": TrialVariant("pose_basket_stats", "pose", tuple(CHANNELS["pose"]), normalization="basket"),
    "pose_prompt_lift": TrialVariant("pose_prompt_lift", "pose", tuple(CHANNELS["pose"]), prompt="Grasp the red cube and lift it off the table."),
    "joint": TrialVariant("joint", "joint", tuple(CHANNELS["joint"])),
    "joint_three_left_blank": TrialVariant("joint_three_left_blank", "joint", tuple(CHANNELS["joint"]), camera_layout="three_left_blank"),
    "joint_grip_inverted": TrialVariant("joint_grip_inverted", "joint", tuple(CHANNELS["joint"]), gripper_mapping="inverted"),
    "joint_grip_scaled20": TrialVariant("joint_grip_scaled20", "joint", tuple(CHANNELS["joint"]), gripper_mapping="scaled20"),
    "joint_absolute": TrialVariant("joint_absolute", "joint", tuple(CHANNELS["joint"]), joint_reference="absolute"),
    "joint_demo_channels": TrialVariant("joint_demo_channels", "joint", (0, 1, 2, 3, 4, 28)),
    "joint_demo_published": TrialVariant("joint_demo_published", "joint", (0, 1, 2, 3, 4, 28), joint_reference="demo_degrees_absolute"),
    "joint_demo_native256_front": TrialVariant("joint_demo_native256_front", "joint", (0, 1, 2, 3, 4, 28), joint_reference="demo_degrees_absolute", camera_view="front", image_width=256, image_height=256),
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
for _side in ("left", "right"):
    _name = f"pose_robotwin_home_{_side}"
    TRIAL_VARIANTS[_name] = replace(TRIAL_VARIANTS[f"pose_robotwin_cameras_{_side}"], name=_name, native_initial_state=True)
TRIAL_VARIANTS["pose_robotwin_home_duplicate"] = replace(TRIAL_VARIANTS["pose_robotwin_home_left"], name="pose_robotwin_home_duplicate", camera_layout="three_duplicate")
for _side in ("left", "right"):
    _name = f"pose_robotwin_home_visible_{_side}"
    TRIAL_VARIANTS[_name] = replace(TRIAL_VARIANTS[f"pose_robotwin_home_{_side}"], name=_name, head_azimuth_offset_deg=20.)
TRIAL_VARIANTS["pose_robotwin_home_visible_zoom_left"] = replace(TRIAL_VARIANTS["pose_robotwin_home_visible_left"], name="pose_robotwin_home_visible_zoom_left", head_distance_m=.65)
TRIAL_VARIANTS["pose_robotwin_home_visible_zoom_right"] = replace(TRIAL_VARIANTS["pose_robotwin_home_visible_right"], name="pose_robotwin_home_visible_zoom_right", head_distance_m=.65)
TRIAL_VARIANTS["pose_robotwin_home_visible_zoom_duplicate"] = replace(TRIAL_VARIANTS["pose_robotwin_home_visible_zoom_left"], name="pose_robotwin_home_visible_zoom_duplicate", camera_layout="three_duplicate")
TRIAL_VARIANTS["pose_robotwin_home_visible_zoom_left_full_denoise"] = replace(
    TRIAL_VARIANTS["pose_robotwin_home_visible_zoom_left"],
    name="pose_robotwin_home_visible_zoom_left_full_denoise", inference_steps=50,
)
for _mode in ("pose", "joint"):
    _name = f"{_mode}_full_denoise_task_front_home"
    TRIAL_VARIANTS[_name] = replace(TRIAL_VARIANTS[f"{_mode}_full_denoise_task_front"], name=_name, native_initial_state=True)
    _rebased = f"{_name}_rebased_stats"
    TRIAL_VARIANTS[_rebased] = replace(TRIAL_VARIANTS[_name], name=_rebased, calibration_at_initial_state=True)


TRIAL_VARIANTS["joint_full_denoise_task_front_home_no_rate"] = replace(TRIAL_VARIANTS["joint_full_denoise_task_front_home"], name="joint_full_denoise_task_front_home_no_rate", rate_limits=False)


TRIAL_VARIANTS["joint_three_task_front_home"] = replace(TRIAL_VARIANTS["joint_three_task_front"], name="joint_three_task_front_home", native_initial_state=True)
TRIAL_VARIANTS["joint_demo_full_denoise_task_front_home"] = replace(
    TRIAL_VARIANTS["joint_demo_native256_front"],
    name="joint_demo_full_denoise_task_front_home", camera_view="task_front",
    native_initial_state=True, inference_steps=50,
)

# Paired with the same scene/config using the default "Pick up" instruction.
for _base in (
    "pose_robotwin_home_visible_zoom_left",
    "joint_three_task_front_home",
    "pose_full_denoise_task_front_home",
    "joint_full_denoise_task_front_home",
):
    _name = f"{_base}_explicit_soup_lift"
    TRIAL_VARIANTS[_name] = replace(
        TRIAL_VARIANTS[_base], name=_name,
        prompt="Grasp the alphabet soup can and lift it off the table.",
    )


MULTITASK_PROMPTS = {
    "soup_lift": "Pick up the alphabet soup can.",
    "cheese_lift": "Pick up the cream cheese box.",
    "soup_place": "Put the alphabet soup can in the basket.",
    "cheese_place": "Put the cream cheese box in the basket.",
    "full_basket": "Put both the alphabet soup and the cream cheese box in the basket.",
}
MULTITASK_OBJECTS = ("alphabet_soup", "cream_cheese", "tomato_sauce", "ketchup")


def variant_stats(variant: TrialVariant, stats: dict, home_joint: np.ndarray) -> dict:
    """Map calibrated physical quantiles to the selected channels/reference."""
    if variant.normalization == "robotwin":
        if variant.mode != "pose":
            raise ValueError("RoboTwin native stats require pose channels")
        path = Path(__file__).resolve().parents[1] / "third_party/Zero-WAM/wan_va/assets/norm_stats/robotwin_icl.json"
        native = json.loads(path.read_text())["norm_stats"]
        hand, gripper = native["action.hand.position"], native["action.effector.position"]
        return {key: hand[key] + [0.] * 14 + gripper[key] for key in ("q01", "q99")}
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
        "unit100": lambda v: 100 * v,
    }[mapping](value)


def model_gripper(physical_pct: float, mapping: str) -> float:
    return {
        "normal": lambda v: v,
        "inverted": lambda v: 100 - v,
        "scaled20": lambda v: v / 20,
        "offset20": lambda v: v / 20 - 1,
        "unit100": lambda v: v / 100,
    }[mapping](physical_pct)


def make_env(mode, cameras=True, width=288, height=224, task="cube"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from so101_nexus.observations import EndEffectorPose, JointPositions, OverheadCamera, WristCamera

    observations = [JointPositions(), EndEffectorPose()]
    if cameras:
        observations += [OverheadCamera(width=width, height=height), WristCamera(width=width, height=height)]
    control_mode = "pd_ee_pose" if mode == "pose" else "pd_joint_pos"
    if task != "cube":
        if task not in MULTITASK_PROMPTS:
            raise ValueError(task)
        import sys

        sim_dir = str(Path(__file__).resolve().parents[1] / "sim")
        if sim_dir not in sys.path:
            sys.path.insert(0, sim_dir)
        from libero_basket_env import LiberoBasketEnv, default_config

        cfg = default_config(width, height)
        cfg.observations = observations
        return LiberoBasketEnv(config=cfg, control_mode=control_mode, robot_init_qpos_noise=0)
    from so101_nexus.config import PickConfig
    from so101_nexus.mujoco.pick_env import PickLiftEnv
    # The default PickConfig places seed-100's cube near x=0.40 m, beyond a
    # downward-grasp configuration of this five-DOF arm. Use a small reachable
    # spawn region while retaining seed-dependent visual/object variation.
    cfg = PickConfig(
        observations=observations, terminate_on_success=False,
        spawn_center=(.25, 0.), spawn_min_radius=0., spawn_max_radius=.025,
    )
    return PickLiftEnv(config=cfg, control_mode=control_mode, robot_init_qpos_noise=0)


def calibrate(seed=101, count=256, source="mixed", reference_joint=None, reference_tcp=None):
    """Sweep simulator configurations and mix five unrelated basket demonstrations."""
    env = make_env("joint", cameras=False)
    env.reset(seed=seed)
    initial_joint = env._get_current_qpos().copy()
    initial_tcp = env._get_tcp_pose().copy()
    if (reference_joint is None) != (reference_tcp is None):
        raise ValueError("Calibration reference needs both joints and TCP")
    pack_joint = initial_joint if reference_joint is None else np.asarray(reference_joint)
    pack_tcp = initial_tcp if reference_tcp is None else np.asarray(reference_tcp)
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
                samples["sim"][mode].append(pack(mode, pack_joint, pack_tcp, joint, tcp, pct))
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
                samples["basket"][mode].append(pack(mode, joint0 if reference_joint is None else pack_joint, tcp0 if reference_tcp is None else pack_tcp, target_joint, target_tcp, joint_action[5]))
            counts["basket"] += 1
    selected = {mode: samples["sim"][mode] + samples["basket"][mode] if source == "mixed" else samples[source][mode] for mode in ("joint", "pose")}
    return {mode: active_stats(mode, values) for mode, values in selected.items()}, counts


def prepare_initial_state(env, obs):
    """Elevated/open home projected onto SO-101; settle before episode reference."""
    import mujoco
    q = np.array([-.06699132819429661, -1.7453345353671794, 1.1399283051152027,
                  .7100030758030357, -1.464626234718075, 1.7453292000408704])
    env.data.qpos[env._qpos_addrs] = q
    env.data.qvel[env._qvel_addrs] = 0.
    env.data.ctrl[env._actuator_ids] = q
    mujoco.mj_forward(env.model, env.data)
    for _ in range(100):
        mujoco.mj_step(env.model, env.data)
    return env._get_obs()


def configure_cameras(env, variant, obs):
    """Apply a recorded camera variant and refresh its rendered observation."""
    if variant.camera_view in {"front", "task_front"}:
        camera = env._overhead_obs_cam
        camera.azimuth = 270. if variant.camera_view == "front" else 180.
        camera.elevation = -30.
        camera.distance = .65
        camera.lookat[:] = [.22, 0., .025]
    elif variant.camera_view == "robotwin_head":
        camera = env._overhead_obs_cam
        camera.azimuth = variant.head_azimuth_offset_deg
        camera.elevation = -np.degrees(np.arctan2(.8, .6))
        camera.distance = variant.head_distance_m
        camera.lookat[:] = [.22, .032, 0.]
        env.model.vis.global_.fovy = 37.
    elif variant.camera_view != "overhead":
        raise ValueError(variant.camera_view)
    if variant.wrist_tilt_deg:
        import mujoco

        camera_id = env._wrist_cam_id
        base_wxyz = env.model.cam_quat[camera_id].copy()
        rotated = (Rotation.from_quat(xyzw_from_wxyz(base_wxyz)) * Rotation.from_euler("x", variant.wrist_tilt_deg, degrees=True)).as_quat()
        env.model.cam_quat[camera_id] = rotated[[3, 0, 1, 2]]
        mujoco.mj_forward(env.model, env.data)
    if variant.native_robotwin_cameras:
        import mujoco
        from zero_wam.so101_actions import ROBOTWIN_TOOL_BASIS

        # Aloha URDF camera: xyz [.07,.032,.065], rpy [0,.4,0] from
        # wrist child link. SAPIEN joint frame includes Rx(pi), and
        # _trans_endpose multiplies by another Rx(pi): these cancel.
        # Verified on the actual native URDF with PhysxCpuSystem.
        global_transform = Rotation.identity()
        sapien_to_mujoco_camera = Rotation.from_matrix([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
        tcp_to_camera = ROBOTWIN_TOOL_BASIS * global_transform * Rotation.from_euler("y", .4) * sapien_to_mujoco_camera
        offset = (ROBOTWIN_TOOL_BASIS * global_transform).apply([.07, .032, .065]) - np.array([0., 0., .12])
        tcp_rotation = Rotation.from_quat(xyzw_from_wxyz(env._get_tcp_pose()[3:7]))
        parent = env.model.cam_bodyid[env._wrist_cam_id]
        parent_rotation = Rotation.from_matrix(env.data.xmat[parent].reshape(3, 3))
        desired_position = env._get_tcp_pose()[:3] + tcp_rotation.apply(offset)
        local_rotation = parent_rotation.inv() * tcp_rotation * tcp_to_camera
        env.model.cam_pos[env._wrist_cam_id] = parent_rotation.inv().apply(desired_position - env.data.xpos[parent])
        env.model.cam_quat[env._wrist_cam_id] = local_rotation.as_quat()[[3, 0, 1, 2]]
        env.model.cam_fovy[env._wrist_cam_id] = 37.
        mujoco.mj_forward(env.model, env.data)
    if variant.camera_view != "overhead" or variant.wrist_tilt_deg:
        obs = env._get_obs()
    return obs


def run_episode(worker, mode, seed, stats, out_dir, max_steps=400, variant: TrialVariant | None = None, task="cube", icl_video_path=None):
    import imageio.v2 as imageio

    variant = variant or TRIAL_VARIANTS[mode]
    if variant.mode != mode:
        raise ValueError(f"Variant {variant.name} is {variant.mode}, not {mode}")
    env = make_env(mode, width=variant.image_width, height=variant.image_height, task=task)
    obs, info = env.reset(seed=seed)
    if variant.native_initial_state:
        obs = prepare_initial_state(env, obs)
    obs = configure_cameras(env, variant, obs)
    prompt = variant.prompt or (MULTITASK_PROMPTS[task] if task != "cube" else info.get("task", "Pick up the red cube."))
    initial_joint, initial_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
    low, high = env.action_space.low, env.action_space.high
    # Pose-mode public space is Cartesian; joint limits come from the model targets.
    if mode == "pose":
        low, high = env._target_low, env._target_high
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_objects = {name: env.object_pos(name).copy() for name in MULTITASK_OBJECTS} if task != "cube" else {}
    if task != "cube":
        start_scene = {name: env.object_pos(name).tolist() for name in (*MULTITASK_OBJECTS, "basket")}
        manifest_path = out_dir / "trial.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["initial_object_positions_m"] = start_scene
            manifest["initial_tcp_wxyz"] = initial_tcp.tolist()
            manifest["initial_joints_rad"] = initial_joint.tolist()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        imageio.imwrite(out_dir / "first_overhead.png", obs["overhead_camera"])
        imageio.imwrite(out_dir / "first_wrist.png", obs["wrist_camera"])
    camera_keys = ["observation.images.top", "observation.images.wrist"] if variant.camera_layout == "two" else ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    model_channels = variant.model_channels or variant.channels
    control_indices = [model_channels.index(c) for c in variant.channels]
    reset_request = {"reset": True, "mode": mode, "channels": list(model_channels), "camera_keys": camera_keys, "stats": stats, "prompt": prompt, "seed": seed}
    if variant.inference_steps:
        reset_request["inference_steps"] = variant.inference_steps
    translation_frame = Rotation.from_euler("z", variant.pose_translation_yaw_deg, degrees=True)
    if icl_video_path is not None:
        reset_request["icl_video_bytes"] = Path(icl_video_path).read_bytes()
    reset_result = worker.so101_step.remote(reset_request)
    if isinstance(reset_result, dict) and "_so101_runtime" in reset_result:
        runtime = reset_result["_so101_runtime"]
        assert runtime["channels"] == list(model_channels)
        assert runtime["camera_keys"] == camera_keys
        assert runtime.get("active_mask_channels", sorted(model_channels)) == sorted(model_channels)
        assert runtime["norm_stats_sha256"] == hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()
        assert runtime.get("prompt") == prompt, "Worker reset used a different task prompt"
        if variant.inference_steps:
            assert runtime["video_inference_steps"] == variant.inference_steps, "Video denoising override was not applied"
            assert runtime["action_inference_steps"] == variant.inference_steps, "Action denoising override was not applied"
        manifest_path = out_dir / "trial.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["worker_runtime"] = runtime
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def encoded_state(measured_joint, measured_tcp):
        pct = 100 * (measured_joint[5] - low[5]) / (high[5] - low[5])
        state = pack(mode, initial_joint, initial_tcp, measured_joint, measured_tcp, model_gripper(pct, variant.gripper_mapping))
        if mode == "pose":
            if variant.tool_frame == "robotwin_ee":
                state = tcp_to_robotwin_relative(initial_tcp, measured_tcp, state[-1], variant.pose_translation_yaw_deg, quaternion_convention=variant.quaternion_convention)
            else:
                state[:3] = translation_frame.inv().apply(state[:3])
        if mode == "joint" and variant.joint_reference == "absolute":
            state[:5] = measured_joint[:5]
        elif mode == "joint" and variant.joint_reference == "demo_degrees_absolute":
            state[:5] = np.rad2deg(measured_joint[:5])
        if variant.model_channels:
            expanded = np.zeros(len(model_channels), dtype=state.dtype)
            # The absent arm is stationary at its initial pose with open gripper.
            for channel in (6, 13, 28, 29):
                if channel in model_channels:
                    expanded[model_channels.index(channel)] = 1.
            expanded[control_indices] = state
            return expanded
        return state

    def formatted_observation(current_obs):
        measured_joint, measured_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
        high, wrist = current_obs["overhead_camera"], current_obs["wrist_camera"]
        if variant.camera_layout == "two":
            images = {"observation.images.top": high, "observation.images.wrist": wrist}
        elif variant.camera_layout in {"three_left_blank", "three_right_blank", "three_duplicate"}:
            blank = np.full_like(wrist, 127)
            left = wrist if variant.camera_layout != "three_right_blank" else blank
            right = wrist if variant.camera_layout != "three_left_blank" else blank
            images = {"observation.images.cam_high": high, "observation.images.cam_left_wrist": left, "observation.images.cam_right_wrist": right}
        else:
            raise ValueError(variant.camera_layout)
        return {**images, "observation.state": encoded_state(measured_joint, measured_tcp), "realized_tcp_pose": measured_tcp}

    if task != "cube":
        first_model_obs = formatted_observation(obs)
        for key in camera_keys:
            imageio.imwrite(out_dir / f"first_{key.rsplit('.', 1)[-1]}.png", first_model_obs[key])

    frames = [obs["overhead_camera"]]
    records, first = [], True
    success = False
    while len(records) < max_steps and not success:
        model_obs = formatted_observation(obs)
        result = worker.so101_step.remote({"obs": model_obs, "prompt": prompt})
        raw = np.asarray(result["action"])
        assert raw.shape[0] == len(model_channels) and raw.shape[2] % 4 == 0, raw.shape
        history = np.zeros_like(raw)
        keyframes = []
        for f in range(1 if first else 0, raw.shape[1]):
            for h in range(raw.shape[2]):
                if len(records) >= max_steps or success:
                    break
                before_joint, before_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
                model_action = raw[control_indices, f, h].copy()
                physical_action = model_action.copy()
                if mode == "pose":
                    if variant.tool_frame == "robotwin_ee":
                        physical_action = robotwin_to_tcp_relative(physical_action, initial_tcp, variant.pose_translation_yaw_deg, quaternion_convention=variant.quaternion_convention)
                    else:
                        physical_action[:3] = translation_frame.apply(physical_action[:3])
                physical_action[-1] = physical_gripper(model_action[-1], variant.gripper_mapping)
                reference_joint = initial_joint if variant.joint_reference == "relative" else np.zeros_like(initial_joint)
                if variant.joint_reference == "demo_degrees_absolute":
                    physical_action[:5] = np.deg2rad(physical_action[:5])
                cmd, diagnostics = decode(mode, physical_action, reference_joint, initial_tcp, before_joint, before_tcp, low, high, rate_limits=variant.rate_limits)
                ik_joint_target = env._action_to_ctrl(cmd).copy() if mode == "pose" else None
                obs, _, _, _, info = env.step(cmd)
                after_joint, after_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
                history[:, f, h] = encoded_state(after_joint, after_tcp)
                record = {"raw": model_action.tolist(), "raw_full": raw[:, f, h].tolist(), "physical_action": physical_action.tolist(), "command": cmd.tolist(), "executed": history[:, f, h].tolist(), "realized_tcp": after_tcp.tolist(), "gripper_rad": float(after_joint[5]), "clipped": bool(diagnostics["clipped"]), "diagnostics": diagnostics}
                if task == "cube":
                    record.update(tcp_to_obj_dist_m=float(info["tcp_to_obj_dist"]), lift_height_m=float(info["lift_height"]), is_grasped=bool(info["is_grasped"]))
                    step_success = bool(info.get("success", False))
                else:
                    object_metrics = {}
                    for name in MULTITASK_OBJECTS:
                        position = env.object_pos(name)
                        object_metrics[name] = {"tcp_distance_m": float(np.linalg.norm(after_tcp[:3] - position)), "displacement_m": float(np.linalg.norm(position - start_objects[name])), "lift_m": float(position[2] - start_objects[name][2]), "grasped": env.is_grasping(name), "in_basket": env.in_basket(name)}
                    target = task.split("_")[0]
                    target = {"soup": "alphabet_soup", "cheese": "cream_cheese"}.get(target)
                    if task == "full_basket":
                        step_success = all(object_metrics[name]["in_basket"] for name in ("alphabet_soup", "cream_cheese"))
                    elif task.endswith("_lift"):
                        step_success = object_metrics[target]["grasped"] and object_metrics[target]["lift_m"] >= .05
                    else:
                        step_success = object_metrics[target]["in_basket"] and object_metrics[target]["displacement_m"] >= .03
                    record["objects"] = object_metrics
                record["success"] = step_success
                if mode == "pose":
                    requested_xyz = initial_tcp[:3] + physical_action[:3]
                    requested_q = physical_action[3:7]
                    requested_q = requested_q / np.linalg.norm(requested_q) if np.all(np.isfinite(requested_q)) and np.linalg.norm(requested_q) > 1e-6 else np.array([0., 0., 0., 1.])
                    requested_rot = Rotation.from_quat(xyzw_from_wxyz(initial_tcp[3:7])) * Rotation.from_quat(requested_q)
                    realized_rot = Rotation.from_quat(xyzw_from_wxyz(after_tcp[3:7]))
                    commanded_rot = Rotation.from_rotvec(cmd[3:6])
                    record.update(ik_joint_target=ik_joint_target.tolist(), requested_position_error_m=float(np.linalg.norm(requested_xyz - after_tcp[:3])), requested_orientation_error_rad=float((requested_rot.inv() * realized_rot).magnitude()), commanded_position_error_m=float(np.linalg.norm(cmd[:3] - after_tcp[:3])), commanded_orientation_error_rad=float((commanded_rot.inv() * realized_rot).magnitude()))
                record["max_contact_penetration_m"] = max([0.] + [-float(c.dist) for c in env.data.contact])
                records.append(record)
                success = step_success
                if len(records) % 4 == 0:
                    frames.append(obs["overhead_camera"])
                if (h + 1) % (raw.shape[2] // 4) == 0:
                    keyframes.append(formatted_observation(obs))
            if len(records) >= max_steps or success:
                break
        first = False
        if not success and len(records) < max_steps:
            cache_result = worker.so101_step.remote({"obs": keyframes, "compute_kv_cache": True, "state": history, "executed_model_actions": history})
            if isinstance(cache_result, dict) and "_so101_runtime" in cache_result:
                cache_runtime = cache_result["_so101_runtime"]
                with (out_dir / "cache_runtime.jsonl").open("a") as runtime_file:
                    runtime_file.write(json.dumps(cache_runtime) + "\n")
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
    summary = {"task": task, "variant": variant.name, "mode": mode, "seed": seed, "success": success, "steps": len(records), "clip_fraction": float(np.mean([r["clipped"] for r in records])) if records else 0, "prompt": prompt}
    if records and task == "cube":
        summary.update(min_tcp_to_object_m=float(min(r["tcp_to_obj_dist_m"] for r in records)), max_lift_m=float(max(r["lift_height_m"] for r in records)), grasp_steps=sum(r["is_grasped"] for r in records), min_tcp_z_m=float(min(r["realized_tcp"][2] for r in records)))
    if records and task != "cube":
        summary["objects"] = {name: {"min_tcp_distance_m": float(min(r["objects"][name]["tcp_distance_m"] for r in records)), "max_lift_m": float(max(r["objects"][name]["lift_m"] for r in records)), "max_displacement_m": float(max(r["objects"][name]["displacement_m"] for r in records)), "grasp_steps": sum(r["objects"][name]["grasped"] for r in records), "ever_in_basket": any(r["objects"][name]["in_basket"] for r in records)} for name in MULTITASK_OBJECTS}
    if records:
        for key in ("gripper_range_clipped", "gripper_rate_clipped", "arm_range_clipped", "workspace_clipped"):
            if key in records[0]["diagnostics"]:
                summary[f"{key}_fraction"] = float(np.mean([r["diagnostics"][key] for r in records]))
    if mode == "pose" and records:
        for key in ("requested_position_error_m", "requested_orientation_error_rad", "commanded_position_error_m", "commanded_orientation_error_rad"):
            summary[f"mean_{key}"] = float(np.mean([r[key] for r in records]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_trial(worker, variant_name: str, seed: int, max_steps: int, save_root: str, checkpoint: str, task="cube", icl_video_path=None) -> dict:
    """Run one named one-factor trial with a self-contained manifest."""
    variant = TRIAL_VARIANTS[variant_name]
    parent = Path(save_root) / (task if task != "cube" else "") / checkpoint / variant.name
    if icl_video_path is not None:
        parent /= f"video_{Path(icl_video_path).stem}"
    root = parent / f"seed{seed}_steps{max_steps}"
    root.mkdir(parents=True, exist_ok=True)
    reference = {}
    if variant.calibration_at_initial_state:
        calibration_env = make_env("joint", cameras=False, task=task)
        calibration_obs, _ = calibration_env.reset(seed=seed)
        if variant.native_initial_state:
            prepare_initial_state(calibration_env, calibration_obs)
        reference = dict(reference_joint=calibration_env._get_current_qpos().copy(), reference_tcp=calibration_env._get_tcp_pose().copy())
        calibration_env.close()
    all_stats, counts = calibrate(source="mixed" if variant.normalization == "robotwin" else variant.normalization, **reference)
    home = make_env("joint", cameras=False)
    home.reset(seed=101)
    home_joint = home._get_current_qpos().copy()
    home.close()
    stats = variant_stats(variant, all_stats, home_joint)
    revisions = {
        "posttrain": "07ee865f175d9474a5653e9147698390e47b6143",
        "pretrain": "7040c4195df216c900334ef62d5fdcf05c0601aa",
    }
    code_revision, code_dirty = _CODE_REVISION, bool(_CODE_PATCH)
    source_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in _SOURCE_SNAPSHOT.items()}
    for name, content in _SOURCE_SNAPSHOT.items():
        snapshot_path = root / "runner_source_snapshot" / name
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(content)
    if code_dirty:
        (root / "code_diff.patch").write_bytes(_CODE_PATCH)
    video = None if icl_video_path is None else {"path": str(icl_video_path), "sha256": hashlib.sha256(Path(icl_video_path).read_bytes()).hexdigest()}
    camera_keys = ["observation.images.top", "observation.images.wrist"] if variant.camera_layout == "two" else ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    manifest = {"checkpoint": checkpoint, "checkpoint_revision": revisions[checkpoint], "task": task, "variant": asdict(variant), "camera_keys": camera_keys, "seed": seed, "max_steps": max_steps, "code_revision": code_revision, "code_dirty": code_dirty, "source_sha256": source_hashes, "source_snapshot_basis": "runner_module_import", "conditioning_video": video, "calibration_counts": counts, "norm_stats": stats}
    if reference:
        manifest["calibration_reference"] = {key: value.tolist() for key, value in reference.items()}
    (root / "trial.json").write_text(json.dumps(manifest, indent=2))
    return run_episode(worker, variant.mode, seed, stats, root, max_steps, variant, task, icl_video_path)
