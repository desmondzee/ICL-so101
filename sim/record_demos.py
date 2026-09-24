import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.lerobot_dataset import sim_qpos_to_dataset_row

from libero_basket_env import TARGETS, TASK, LiberoBasketEnv
from scripted import PickPlace

ROOT = Path(__file__).resolve().parents[1] / "data" / "so101_libero_basket"
FPS = 30
H, W = 480, 640
ENV_STATE = [f"{p}_{k}" for p in ("tcp", *TARGETS, "basket") for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
FEATURES = {
    "observation.images.wrist": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
    "observation.images.overhead": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in SO101_JOINT_NAMES]},
    "observation.environment_state": {"dtype": "float32", "shape": (len(ENV_STATE),), "names": ENV_STATE},
    "action": {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in SO101_JOINT_NAMES]},
    "action.ee": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "wx", "wy", "wz", "gripper"]},
}


def closing_dirs(env):
    p = env.object_pos("alphabet_soup")[:2]
    r = p / np.linalg.norm(p)
    return {"alphabet_soup": np.array([-r[1], r[0], 0.0]), "cream_cheese": np.array([0.0, 1.0, 0.0])}


def record(episodes, seed):
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ds = LeRobotDataset.create(
        repo_id="local/so101_libero_basket",
        fps=FPS,
        features=FEATURES,
        root=ROOT,
        robot_type="so101_follower",
        use_videos=True,
    )
    env = LiberoBasketEnv()
    limits = (env._target_low[-1], env._target_high[-1])
    summary = []
    s = seed
    while len(summary) < episodes:
        obs, info = env.reset(seed=s)
        sm = PickPlace(env, TARGETS, closing_dirs(env))
        for ee in sm.actions():
            state = sim_qpos_to_dataset_row(env._get_current_qpos(), gripper_limits_rad=limits)
            env_state = np.concatenate([env._get_tcp_pose(), env.object_poses()])
            frame = {
                "observation.images.wrist": obs["wrist_camera"],
                "observation.images.overhead": obs["overhead_camera"],
                "observation.state": state.astype(np.float32),
                "observation.environment_state": env_state.astype(np.float32),
                "action.ee": np.asarray(ee, dtype=np.float32),
            }
            obs, _, _, _, info = env.step(ee)
            target = sim_qpos_to_dataset_row(env.data.ctrl[env._actuator_ids].copy(), gripper_limits_rad=limits)
            ds.add_frame({**frame, "action": target.astype(np.float32), "task": TASK})
        grasps = [entry for entry in sm.log if entry[1] in ("grasp", "lifted", "in_basket")]
        if info["success"]:
            ds.save_episode()
            summary.append({"seed": s, "frames": ds.meta.total_frames, "log": grasps})
            print(f"episode {len(summary) - 1} seed {s}: success")
        else:
            ds.clear_episode_buffer()
            print(f"seed {s}: failed, discarded {grasps}")
        s += 1
    ds.finalize()
    (ROOT / "demo_summary.json").write_text(json.dumps(summary, indent=2, default=bool))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    record(args.episodes, args.seed)
