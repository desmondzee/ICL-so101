"""Record successful oracle episodes of a validation task as a LeRobot v3 dataset in the curated training format.

    uv run python -m sim.val.record --task sort_blocks --episodes 10 --seed 0

Writes data/so101_sim_val/lerobot/<task>/ (30 fps, front + wrist AV1 video, action / observation.state in LeRobot
degrees with gripper 0-100, action.ee / observation.state.ee from curate.robot FK) and, per episode,
data/so101_sim_val/frames/<task>/episode_XXX/{first,last}.png (front camera, robot hidden) and meta.json.
Each seed is first simulated without rendering; only successful seeds are re-run (deterministically) and recorded.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from curate.robot import NAMES, Kinematics
from sim.val.env import FPS, HEIGHT, WIDTH
from sim.val.tasks import load

OUT = Path(__file__).resolve().parents[2] / "data" / "so101_sim_val"
EE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
VIDEO = {"dtype": "video", "shape": (HEIGHT, WIDTH, 3), "names": ["height", "width", "channels"]}
FEATURES = {
    "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES},
    "action.ee": {"dtype": "float32", "shape": (7,), "names": EE_NAMES},
    "observation.state.ee": {"dtype": "float32", "shape": (7,), "names": EE_NAMES},
    "observation.images.front": VIDEO,
    "observation.images.wrist": VIDEO,
}


def rollout(env, oracle_cls, seed, on_step=None):
    """Run the oracle from `env.reset(seed)`. `on_step(obs, action)` sees each pre-step observation and its action."""
    obs, info = env.reset(seed=seed)
    oracle = oracle_cls(env, np.random.default_rng(seed))
    for action in oracle.actions():
        if on_step is not None:
            on_step(obs, action)
        obs, _, _, _, info = env.step(action)
    return info, oracle


def with_ee(kin, joints):
    """[N, 6] LeRobot joints -> [N, 7] gripper-site pose (base frame, rotvec) + gripper, as curate/convert.py."""
    joints = np.asarray(joints, dtype=np.float64)
    return np.hstack([kin.ee(joints[:, :5]), joints[:, 5:6]]).astype(np.float32)


def record(task, episodes, seed, out=OUT, max_seeds=200, seeds=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    env_cls, oracle_cls = load(task)
    env = env_cls(render_images=False)
    kin = Kinematics()
    root, frames_root = out / "lerobot" / task, out / "frames" / task
    for p in (root, frames_root):
        shutil.rmtree(p, ignore_errors=True)
    ds = LeRobotDataset.create(repo_id=f"local/so101_sim_val_{task}", fps=FPS, features=FEATURES, root=root,
                               robot_type="so101_follower", use_videos=True)
    saved, tried, s = 0, [], seed
    queue = list(seeds) if seeds is not None else None  # explicit seeds (e.g. the episodes chosen for the final set)
    if queue is not None:
        episodes, s = len(queue), queue.pop(0)
    while saved < episodes and len(tried) < max_seeds:
        env.render_images = False
        info, _ = rollout(env, oracle_cls, s)
        tried.append((s, bool(info["success"])))
        if not info["success"]:
            print(f"seed {s}: oracle failed, skipped", flush=True)
            if queue is not None:
                raise RuntimeError(f"seed {s} was chosen but the oracle failed")
            s += 1
            continue
        env.render_images = True
        env.reset(seed=s)
        first = env.render_scene_without_robot("front")
        start_poses = env.object_poses()
        count = [0]

        def add(o, a):
            state, action = env.lerobot_joints(o["state"])[None], env.lerobot_joints(a)[None]
            ds.add_frame({
                "action": action[0].astype(np.float32), "observation.state": state[0].astype(np.float32),
                "action.ee": with_ee(kin, action)[0], "observation.state.ee": with_ee(kin, state)[0],
                "observation.images.front": o["front"], "observation.images.wrist": o["wrist"], "task": env.instruction,
            })
            count[0] += 1

        info, oracle = rollout(env, oracle_cls, s, add)
        if not info["success"]:
            ds.clear_episode_buffer()
            print(f"seed {s}: replay diverged, skipped", flush=True)
            if queue is not None:
                raise RuntimeError(f"seed {s} was chosen but its replay diverged")
            s += 1
            continue
        last = env.render_scene_without_robot("front")
        ds.save_episode()
        folder = frames_root / f"episode_{saved:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(first).save(folder / "first.png")
        Image.fromarray(last).save(folder / "last.png")
        meta = {
            "task": task, "instruction": env.instruction, "episode_index": saved, "seed": s, "fps": FPS,
            "frames": count[0], "success": bool(info["success"]), "info": {k: v for k, v in info.items() if isinstance(v, bool)},
            "distractors": env.active_distractors, "object_poses_start": start_poses, "object_poses_end": env.object_poses(),
            "oracle_log": [list(map(str, entry)) for entry in oracle.log],
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"episode {saved}: seed {s}, {count[0]} frames", flush=True)
        saved += 1
        s = queue.pop(0) if queue else s + 1
    ds.finalize()
    shutil.rmtree(root / "images", ignore_errors=True)
    (root / "meta" / "val_episodes.json").write_text(json.dumps({"task": task, "seeds_tried": tried}, indent=2))
    env.close()
    return saved, tried


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--seeds", default="", help="comma-separated seeds to record, in order (overrides --episodes/--seed)")
    a = p.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else None
    saved, tried = record(a.task, a.episodes, a.seed, a.out, seeds=seeds)
    print(f"{saved} episodes saved; oracle success {sum(ok for _, ok in tried)}/{len(tried)} seeds")


if __name__ == "__main__":
    main()
