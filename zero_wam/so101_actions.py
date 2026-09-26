"""SO-101 interpretations of Zero-WAM's embodiment-dependent 30 channels."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

CHANNELS = {"pose": [0, 1, 2, 3, 4, 5, 6, 28], "joint": [14, 15, 16, 17, 18, 28]}


def xyzw_from_wxyz(quat):
    return np.asarray(quat)[[1, 2, 3, 0]]


def wxyz_from_xyzw(quat):
    return np.asarray(quat)[[3, 0, 1, 2]]


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


def decode(mode: str, action, initial_joint, initial_tcp, current_joint, current_tcp, joint_low, joint_high):
    """Decode, rate-limit and clamp a model action; return sim action and diagnostics."""
    a = np.asarray(action, dtype=float)
    grip = np.clip(a[-1], 0, 100)
    grip_requested_rad = joint_low[5] + grip / 100 * (joint_high[5] - joint_low[5])
    grip_rad = np.clip(grip_requested_rad, current_joint[5] - .12, current_joint[5] + .12)
    grip_rad = float(np.clip(grip_rad, joint_low[5], joint_high[5]))
    diagnostics = {"gripper_range_clipped": bool(abs(grip - a[-1]) > 1e-9), "gripper_rate_clipped": bool(abs(grip_rad - grip_requested_rad) > 1e-9)}
    if mode == "joint":
        desired = np.clip(np.asarray(initial_joint)[:5] + a[:5], joint_low[:5], joint_high[:5])
        limited = np.clip(desired, np.asarray(current_joint)[:5] - .12, np.asarray(current_joint)[:5] + .12)
        diagnostics.update(clipped=bool(np.any(np.abs(desired - limited) > 1e-9)), arm_range_clipped=bool(np.any(np.abs(desired - (np.asarray(initial_joint)[:5] + a[:5])) > 1e-9)))
        return np.r_[limited, grip_rad], diagnostics
    if mode != "pose":
        raise ValueError(mode)
    requested_xyz = np.asarray(initial_tcp)[:3] + a[:3]
    target_xyz = np.clip(requested_xyz, [-.55, -.55, 0], [.55, .55, .55])
    diagnostics["workspace_clipped"] = bool(np.any(np.abs(target_xyz - requested_xyz) > 1e-9))
    delta = target_xyz - np.asarray(current_tcp)[:3]
    distance = np.linalg.norm(delta)
    if distance > .015:
        target_xyz = np.asarray(current_tcp)[:3] + delta * (.015 / distance)
    quat = a[3:7]
    if not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-6:
        quat = np.array([0., 0., 0., 1.])
    else:
        quat = quat / np.linalg.norm(quat)
    target_rot = Rotation.from_quat(xyzw_from_wxyz(initial_tcp[3:7])) * Rotation.from_quat(quat)
    current_rot = Rotation.from_quat(xyzw_from_wxyz(current_tcp[3:7]))
    delta_rot = (current_rot.inv() * target_rot).as_rotvec()
    angle = np.linalg.norm(delta_rot)
    if angle > .15:
        target_rot = current_rot * Rotation.from_rotvec(delta_rot * (.15 / angle))
    diagnostics["clipped"] = bool(distance > .015 or angle > .15)
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
