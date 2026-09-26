"""Trajectory-only screening of SO-101 LeRobot datasets: format, units, quality and calibration."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from curate.io import load, stack
from curate.robot import NAMES, OLD_NAMES, Kinematics, rest_offsets

WRIST = {"wrist", "gripper", "hand", "phone_wrist"}
HOLD_GAP = 1.5
MIN_HOLD_S = 0.3


def units(names: list[str], arm: np.ndarray) -> str | None:
    if names == OLD_NAMES:
        return "old_deg"
    if names != NAMES:
        return None
    return "deg" if np.percentile(np.abs(arm), 99.9) > 101 else "norm"


def grasp_onsets(action_grip: np.ndarray, state_grip: np.ndarray, fps: float) -> list[int]:
    held = np.r_[False, state_grip - action_grip > HOLD_GAP, False]
    start = np.flatnonzero(~held[:-1] & held[1:])
    end = np.flatnonzero(held[:-1] & ~held[1:])
    return [int(s) for s, e in zip(start, end) if e - s >= MIN_HOLD_S * fps]


def screen(root: Path) -> dict:
    src = load(root)
    info = src.info
    names = info["features"]["action"]["names"]
    names = list(names.values())[0] if isinstance(names, dict) else names
    row = {
        "dataset": str(root),
        "robot_type": info.get("robot_type"),
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "fps": info["fps"],
        "dof": len(names),
        "cameras": src.cameras,
        "tasks": sorted(set(src.tasks.values())),
    }
    df = src.frames()
    A, S = stack(df["action"]), stack(df["observation.state"])
    row["units"] = u = units(names, S[:, :5])
    if u is None or A.shape[1] != 6:
        return row | {"reject": "not a 6-DoF SO-101 joint layout"}

    kin = Kinematics()
    A, S = kin.to_degrees(A, u), kin.to_degrees(S, u)
    off = np.zeros(6)
    if u != "norm":
        off = rest_offsets(S)
        if off is None:
            return row | {"reject": "arm never folds to rest; lift/elbow offsets unknown"}
        A, S = A + off, S + off
    row["rest_offset_deg"] = np.round(off[:5], 1).tolist()
    ep = df["episode_index"].to_numpy()
    same = np.r_[False, ep[1:] == ep[:-1]]
    dS = np.abs(np.diff(S[:, :5], axis=0, prepend=S[:1, :5]))[same]
    static = np.r_[False, (np.abs(np.diff(S[:, :5], axis=0)) < 0.2).all(1)] & same
    lens = df.groupby("episode_index").size().to_numpy() / info["fps"]
    lo, hi = kin.range_deg[:, 0] - 5, kin.range_deg[:, 1] + 5
    moving = df.assign(r=np.ptp(S[:, :5], axis=1)).groupby("episode_index")["r"]
    ee = kin.ee(S[::10, :5])

    onsets_z = []
    for e in np.unique(ep):
        m = ep == e
        idx = np.flatnonzero(m)
        for o in grasp_onsets(A[m, 5], S[m, 5], info["fps"]):
            onsets_z.append(kin.ee(S[idx[o] : idx[o] + 1, :5])[0, 2])
    ep_range = [np.ptp(S[ep == e, :5], axis=0).max() for e in np.unique(ep)]

    return row | {
        "nan": float(np.isnan(A).mean() + np.isnan(S).mean()),
        "episode_s": [round(float(np.min(lens)), 1), round(float(np.median(lens)), 1), round(float(np.max(lens)), 1)],
        "still_episodes": int(sum(r < 5 for r in ep_range)),
        "jump_deg": round(float(np.percentile(dS.max(1), 99.99)), 1),
        "leader_offset_deg": np.round(np.median((A - S)[static][:, :5], axis=0), 1).tolist() if static.any() else None,
        "out_of_range": round(float(((S[:, :5] < lo) | (S[:, :5] > hi)).any(1).mean()), 4),
        "ee_z_p1": round(float(np.percentile(ee[:, 2], 1)), 3),
        "ee_reach_p99": round(float(np.percentile(np.linalg.norm(ee[:, :2], axis=1), 99)), 3),
        "grasps": len(onsets_z),
        "grasp_z": np.round(np.percentile(onsets_z, [10, 50, 90]), 3).tolist() if onsets_z else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    with ProcessPoolExecutor(args.workers) as ex:
        rows = []
        for root, row in zip(args.roots, ex.map(_safe, args.roots)):
            rows.append(row)
            print(f"{root}: {row.get('reject') or row.get('error') or 'ok'}", flush=True)
    args.out.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


def _safe(root: Path) -> dict:
    try:
        return screen(root)
    except Exception as e:
        return {"dataset": str(root), "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    sys.exit(main())
