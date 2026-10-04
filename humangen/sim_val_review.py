"""Review sheets for the simulated validation human demos, in the layout humangen/curate_workflow.js reads.

For every generated video data/so101_sim_val/human/<task>/episode_XXX/seed_<k>/video.mp4 this writes
data/so101_sim_val/review/<task>/episode_XXX_s<k>/ with human.mp4, human_frames.jpg (12 frames), robot_frames.jpg
(8 front-camera frames of the oracle episode, first and last included) and info.json (task, instruction, objects).
Then run the judge and verify workflow with args {"dir": "<repo>/data/so101_sim_val/review", "keys": [...]}.

    uv run python -m humangen.sim_val_review              # sheets for every video without one; prints the keys
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from humangen.curate_dataset import _duration, _frames, _grid
from humangen.sim_val_demos import grasp_order

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data" / "so101_sim_val"
REVIEW = VAL / "review"


def robot_frames(task: str, episode: int, out: Path) -> float:
    root = VAL / "lerobot" / task
    eps = pd.concat(pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").glob("*/*.parquet"))).set_index("episode_index")
    m = eps.loc[episode]
    cam = "observation.images.front"
    src = root / "videos" / cam / f"chunk-{m[f'videos/{cam}/chunk_index']:03d}" / f"file-{m[f'videos/{cam}/file_index']:03d}.mp4"
    t0, t1 = float(m[f"videos/{cam}/from_timestamp"]), float(m[f"videos/{cam}/to_timestamp"])
    ts = [t0 + (t1 - t0 - 0.1) * i / 7 for i in range(8)]
    _grid(_frames(src, ts), [f"robot {t - t0:.1f}s" for t in ts], 4).save(out, quality=85)
    return t1 - t0


def sheet(video: Path) -> str:
    seed_dir, ep_dir, task = video.parent, video.parent.parent, video.parent.parent.parent.name
    episode = int(ep_dir.name.split("_")[1])
    key = f"{task}/{ep_dir.name}_s{seed_dir.name.split('_')[1]}"
    d = REVIEW / key
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(video, d / "human.mp4")
    dur = _duration(d / "human.mp4")
    ht = [(dur - 0.2) * i / 11 for i in range(12)]
    _grid(_frames(d / "human.mp4", ht), [f"human {t:.1f}s" for t in ht], 4).save(d / "human_frames.jpg", quality=85)
    robot_s = robot_frames(task, episode, d / "robot_frames.jpg")
    meta = json.loads((VAL / "frames" / task / ep_dir.name / "meta.json").read_text())
    info = {"key": key, "task": task, "episode": episode, "seed": int(seed_dir.name.split("_")[1]), "instruction": meta["instruction"],
            "steps": [f"pick up the {o} and place it" for o in grasp_order(meta)] or None, "goals": f"The scene ends as in the robot's last frame: {meta['instruction']}",
            "objects": "every object visible in the first frame; the ones named in the instruction are the task objects, the rest are distractors that must not move",
            "note": "The scene is a rendered simulation; a realistic human hand in the rendered scene is expected and fine.",
            "human_duration_s": round(dur, 2), "robot_duration_s": round(robot_s, 2)}
    (d / "info.json").write_text(json.dumps(info, indent=1))
    return key


def main() -> None:
    keys = []
    for video in sorted((VAL / "human").glob("*/episode_*/seed_*/video.mp4")):
        task, ep, seed = video.parts[-4], video.parts[-3], video.parts[-2]
        key = f"{task}/{ep}_s{seed.split('_')[1]}"
        keys.append(key if (REVIEW / key / "info.json").exists() else sheet(video))
    print(json.dumps(keys))


if __name__ == "__main__":
    main()
