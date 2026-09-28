"""SO-101 interpretations of Zero-WAM's embodiment-dependent 30 channels."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

CHANNELS = {"pose": [0, 1, 2, 3, 4, 5, 6, 28], "joint": [14, 15, 16, 17, 18, 28]}


def xyzw_from_wxyz(quat):
    return np.asarray(quat)[[1, 2, 3, 0]]


def wxyz_from_xyzw(quat):
    return np.asarray(quat)[[3, 0, 1, 2]]


# RoboTwin virtual tool axes expressed in SO-101 TCP axes: native X forward
# corresponds to TCP Z, native Y jaw separation corresponds to TCP X.
ROBOTWIN_TOOL_BASIS = Rotation.from_matrix([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])


def robotwin_to_tcp_relative(action, initial_tcp, translation_yaw_deg=-90., tool_offset_m=.12, quaternion_convention="physical"):
    """Convert initial-relative RoboTwin wrist targets to SO-101 TCP targets."""
    action = np.asarray(action, dtype=float).copy()
    r0 = Rotation.from_quat(xyzw_from_wxyz(initial_tcp[3:7]))
    q = action[3:7]
    dq = Rotation.from_quat(q / np.linalg.norm(q)) if np.all(np.isfinite(q)) and np.linalg.norm(q) > 1e-6 else Rotation.identity()
    world_frame = Rotation.from_euler('z', translation_yaw_deg, degrees=True)
    if quaternion_convention == "native_client":
        native0 = world_frame.inv() * r0 * ROBOTWIN_TOOL_BASIS
        # Match add_eef_pose: raw WXYZ initial arrays passed to SciPy's XYZW
        # API, and its result interpreted by RoboTwin as raw WXYZ again.
        fake0 = Rotation.from_quat(wxyz_from_xyzw(native0.as_quat()))
        raw_target = (fake0 * dq).as_quat()
        native_target = Rotation.from_quat(xyzw_from_wxyz(raw_target))
        rt = world_frame * native_target * ROBOTWIN_TOOL_BASIS.inv()
        dq_tcp = r0.inv() * rt
    elif quaternion_convention == "physical":
        dq_tcp = ROBOTWIN_TOOL_BASIS * dq * ROBOTWIN_TOOL_BASIS.inv()
        rt = r0 * dq_tcp
    else:
        raise ValueError(quaternion_convention)
    lever = np.array([0., 0., tool_offset_m])
    action[:3] = world_frame.apply(action[:3]) + rt.apply(lever) - r0.apply(lever)
    action[3:7] = dq_tcp.as_quat()
    return action


def tcp_to_robotwin_relative(initial_tcp, measured_tcp, gripper, translation_yaw_deg=-90., tool_offset_m=.12, quaternion_convention="physical"):
    """Encode measured TCP motion in the model's virtual wrist convention."""
    r0 = Rotation.from_quat(xyzw_from_wxyz(initial_tcp[3:7]))
    rt = Rotation.from_quat(xyzw_from_wxyz(measured_tcp[3:7]))
    lever = np.array([0., 0., tool_offset_m])
    delta = np.asarray(measured_tcp[:3]) - np.asarray(initial_tcp[:3]) - rt.apply(lever) + r0.apply(lever)
    world_frame = Rotation.from_euler('z', translation_yaw_deg, degrees=True)
    delta = world_frame.inv().apply(delta)
    if quaternion_convention == "native_client":
        native0 = world_frame.inv() * r0 * ROBOTWIN_TOOL_BASIS
        nativet = world_frame.inv() * rt * ROBOTWIN_TOOL_BASIS
        fake0 = Rotation.from_quat(wxyz_from_xyzw(native0.as_quat()))
        faket = Rotation.from_quat(wxyz_from_xyzw(nativet.as_quat()))
        dq = (fake0.inv() * faket).as_quat()
    elif quaternion_convention == "physical":
        dq = (ROBOTWIN_TOOL_BASIS.inv() * r0.inv() * rt * ROBOTWIN_TOOL_BASIS).as_quat()
    else:
        raise ValueError(quaternion_convention)
    if dq[-1] < 0:
        dq *= -1
    return np.r_[delta, dq, gripper]


def pack(mode: str, initial_joint, initial_tcp, target_joint, target_tcp, gripper_pct):
    """Return active-channel physical action in the same order as CHANNELS."""
    if mode == "joint":
        return np.r_[np.asarray(target_joint)[:5] - np.asarray(initial_joint)[:5], gripper_pct]
    if mode != "pose":
        raise ValueError(mode)
    q0 = Rotation.from_quat(xyzw_from_wxyz(initial_tcp[3:7]))
    qt = Rotation.from_quat(xyzw_from_wxyz(target_tcp[3:7]))
    dq = (q0.inv() * qt).as_quat()
    if dq[3] < 0:
        dq = -dq
    return np.r_[np.asarray(target_tcp)[:3] - np.asarray(initial_tcp)[:3], dq, gripper_pct]


def decode(mode: str, action, initial_joint, initial_tcp, current_joint, current_tcp, joint_low, joint_high, rate_limits: bool = True):
    """Decode, rate-limit and clamp a model action; return sim action and diagnostics."""
    a = np.asarray(action, dtype=float)
    grip = np.clip(a[-1], 0, 100)
    grip_low, grip_high = float(joint_low[5]), float(joint_high[5])
    grip_requested_rad = grip_low + grip / 100 * (grip_high - grip_low)
    joint_step = .12 if rate_limits else np.inf
    position_step = .015 if rate_limits else np.inf
    rotation_step = .15 if rate_limits else np.inf
    grip_rad = np.clip(grip_requested_rad, current_joint[5] - joint_step, current_joint[5] + joint_step)
    rate_clipped = bool(abs(grip_rad - grip_requested_rad) > 1e-9)
    grip_rad = float(np.clip(grip_rad, grip_low, grip_high))
    diagnostics = {"gripper_range_clipped": bool(abs(grip - a[-1]) > 1e-9), "gripper_rate_clipped": rate_clipped}
    if mode == "joint":
        desired = np.clip(np.asarray(initial_joint)[:5] + a[:5], joint_low[:5], joint_high[:5])
        limited = np.clip(desired, np.asarray(current_joint)[:5] - joint_step, np.asarray(current_joint)[:5] + joint_step)
        diagnostics.update(clipped=bool(np.any(np.abs(desired - limited) > 1e-9)), arm_range_clipped=bool(np.any(np.abs(desired - (np.asarray(initial_joint)[:5] + a[:5])) > 1e-9)))
        diagnostics["arm_rate_clipped"] = diagnostics["clipped"]
        diagnostics["clipped"] = any(diagnostics.values())
        return np.r_[limited, grip_rad], diagnostics
    if mode != "pose":
        raise ValueError(mode)
    requested_xyz = np.asarray(initial_tcp)[:3] + a[:3]
    target_xyz = np.clip(requested_xyz, [-.55, -.55, 0], [.55, .55, .55])
    diagnostics["workspace_clipped"] = bool(np.any(np.abs(target_xyz - requested_xyz) > 1e-9))
    delta = target_xyz - np.asarray(current_tcp)[:3]
    distance = np.linalg.norm(delta)
    if distance > position_step:
        target_xyz = np.asarray(current_tcp)[:3] + delta * (position_step / distance)
    quat = a[3:7]
    if not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-6:
        quat = np.array([0., 0., 0., 1.])
    else:
        quat = quat / np.linalg.norm(quat)
    target_rot = Rotation.from_quat(xyzw_from_wxyz(initial_tcp[3:7])) * Rotation.from_quat(quat)
    current_rot = Rotation.from_quat(xyzw_from_wxyz(current_tcp[3:7]))
    delta_rot = (current_rot.inv() * target_rot).as_rotvec()
    angle = np.linalg.norm(delta_rot)
    if angle > rotation_step:
        target_rot = current_rot * Rotation.from_rotvec(delta_rot * (rotation_step / angle))
    diagnostics["position_rate_clipped"] = bool(distance > position_step)
    diagnostics["orientation_rate_clipped"] = bool(angle > rotation_step)
    diagnostics["clipped"] = any(diagnostics.values())
    return np.r_[target_xyz, target_rot.as_rotvec(), grip_rad], diagnostics


def active_stats(mode, samples):
    sample = np.asarray(samples, dtype=float)
    if sample.ndim != 2 or sample.shape[1] != len(CHANNELS[mode]) or len(sample) < 20:
        raise ValueError("Calibration requires at least 20 active-channel samples")
    lo, hi = np.quantile(sample, [.01, .99], axis=0)
    lo[-1], hi[-1] = 0., 100.
    narrow = hi - lo < 1e-3
    lo[narrow] -= .5
    hi[narrow] += .5
    q01, q99 = np.zeros(30), np.zeros(30)
    q01[CHANNELS[mode]], q99[CHANNELS[mode]] = lo, hi
    return {"q01": q01.tolist(), "q99": q99.tolist()}
