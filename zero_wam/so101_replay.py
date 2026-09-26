"""Fast, open-loop compatibility screens using saved Zero-WAM raw actions.

These are diagnostic replays, not policy success tests: model predictions and
its KV cache are not recomputed after changing the simulator trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zero_wam.so101_actions import decode
from zero_wam.so101_eval import make_env


def remap_gripper(value: float, mapping: str) -> float:
    if mapping == "normal":
        return value
    if mapping == "inverted":
        return 100 - value
    if mapping == "scaled20":
        return 20 * value
    if mapping == "offset20":
        return 20 * (value + 1)
    if mapping == "scaled20_inverted":
        return 100 - 20 * value
    raise ValueError(mapping)


def replay(path: Path, mode: str, seed: int, gripper: str = "normal", joint_reference: str = "relative") -> dict:
    rows = [json.loads(line) for line in path.open()]
    env = make_env(mode, cameras=False)
    _, info = env.reset(seed=seed)
    initial_joint, initial_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
    low, high = (env._target_low, env._target_high) if mode == "pose" else (env.action_space.low, env.action_space.high)
    min_tcp_object_dist = float("inf")
    max_lift = 0.0
    grasp_steps = 0
    success_step = None
    for index, row in enumerate(rows):
        raw = np.asarray(row["raw"], dtype=float).copy()
        raw[-1] = remap_gripper(raw[-1], gripper)
        reference_joint = initial_joint if joint_reference == "relative" else np.zeros_like(initial_joint)
        current_joint, current_tcp = env._get_current_qpos().copy(), env._get_tcp_pose().copy()
        command, _ = decode(mode, raw, reference_joint, initial_tcp, current_joint, current_tcp, low, high)
        _, _, _, _, info = env.step(command)
        min_tcp_object_dist = min(min_tcp_object_dist, float(info["tcp_to_obj_dist"]))
        max_lift = max(max_lift, float(info["lift_height"]))
        grasp_steps += bool(info["is_grasped"])
        if info["success"] and success_step is None:
            success_step = index + 1
    final_tcp = env._get_tcp_pose().copy()
    env.close()
    return {
        "source": str(path), "mode": mode, "seed": seed,
        "gripper": gripper, "joint_reference": joint_reference,
        "success": success_step is not None, "success_step": success_step,
        "min_tcp_object_dist_m": min_tcp_object_dist,
        "max_lift_m": max_lift, "grasp_steps": grasp_steps,
        "final_tcp_xyz": final_tcp[:3].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", default="pose,joint")
    parser.add_argument("--seeds", default="100,101,102")
    args = parser.parse_args()
    results = []
    for mode in args.modes.split(","):
        for seed_text in args.seeds.split(","):
            seed = int(seed_text)
            path = args.source_root / f"{mode}_seed{seed}" / "actions.jsonl"
            for gripper in ("normal", "inverted", "scaled20", "offset20", "scaled20_inverted"):
                results.append(replay(path, mode, seed, gripper=gripper))
            if mode == "joint":
                results.append(replay(path, mode, seed, joint_reference="absolute"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
