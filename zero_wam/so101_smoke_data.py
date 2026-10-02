"""Build the SO-101 smoke-test data in Zero-WAM's released on-disk format (the CPU part).

One collection root per split (train: pbvr test002 export episode 168, val: export episode 84, same task; front and
wrist cameras). The human demos are keyed by curated_episode_index (173 and 85 here), which differs from the export
episode_index in several datasets, so the pairing goes through accepted.json. Each holds one LeRobot v2.1 task
directory, which is what upstream's lerobot==0.3.3 loader reads, plus collection-level meta/action_transform.yaml and
meta/action_stats.json. These map the end-effector columns onto Zero-WAM's 30 action channels through upstream's own
LeRobotActionProcessor: channels 0-6 hold xyz and quaternion relative to the segment start, and channel 28 holds the
absolute gripper. Each root also gets an ICL manifest and its paired human video. The robot and human latents are
encoded on Modal (modal_so101_smoke.py encode), and nothing here changes the upstream code.

    python -m zero_wam.so101_smoke_data
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = "pbvr__so101_test002"
SOURCE = REPO / "data" / "so101_export" / "lerobot" / TASK_DIR
HUMAN = REPO / "data" / "so101_smoke" / "human" / TASK_DIR  # H3 Max Turbo demos from hf://buckets/nikgeo/ICL-so101/robot_removal/demos_h3_max_turbo
OUT = REPO / "data" / "so101_smoke" / "zero_wam" / TASK_DIR
CAMERAS = ["observation.images.front", "observation.images.wrist"]  # concatenated along width in this order, as Robotwin's three
SPLITS = {"train": 168, "val": 84}  # export episode_index
RUN = "run_so101_smoke"
HUMAN_TEXT = "Use a hand to pick up the orange block and put it into the box."
COLUMNS = ["action", "observation.state", "action.ee", "observation.state.ee", "timestamp", "frame_index", "episode_index", "index", "task_index"]

ACTION_TRANSFORM = {
    "states": [
        {"observation.state.hand.position": {"origin_keys": [{"observation.state.ee": {"start": 0, "end": 6}}]}},
        {"observation.state.effector.position": {"origin_keys": [{"observation.state.ee": {"start": 6, "end": 7}}]}},
    ],
    "actions": [
        {"action.hand.position": {"origin_keys": [{"action.ee": {"start": 0, "end": 6}}], "absolute_value": True, "format": "xyza"}},
        {"action.effector.position": {"origin_keys": [{"action.ee": {"start": 6, "end": 7}}], "absolute_value": True, "use_absolute": True}},
    ],
    "images": [{camera: {"origin_keys": camera}} for camera in CAMERAS],
}


def human_episode(episode: int) -> int:
    """The robot-removal and human-demo runs name episodes by curated_episode_index, not the export episode_index."""
    rows = json.loads((REPO / "data" / "so101_export" / "accepted.json").read_text())
    return next(r["curated_episode_index"] for r in rows if r["dataset"] == TASK_DIR and r["episode_index"] == episode)


def episodes_meta() -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted((SOURCE / "meta" / "episodes").glob("*/*.parquet")))


def frames() -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted((SOURCE / "data").glob("*/*.parquet")))


def action_stats(data: pd.DataFrame) -> dict:
    """Quantiles of what the processor normalises, over every episode of the dataset: xyz relative to the episode's
    first state (the latents cover whole episodes, so each segment starts at frame 0) and the absolute gripper, whose
    nominal 0-100 range is far wider than what the datasets use; quaternion over its full range, as in the released
    Robotwin stats."""
    rel = []
    gripper = np.stack(data["action.ee"].to_numpy())[:, 6]
    for _, ep in data.groupby("episode_index"):
        ee = np.stack(ep.sort_values("frame_index")["action.ee"].to_numpy())
        s0 = np.stack(ep.sort_values("frame_index")["observation.state.ee"].to_numpy())[0]
        rel.append(ee[:, :3] - s0[:3])
    rel = np.concatenate(rel)
    q01 = np.quantile(rel, 0.01, axis=0).tolist() + [-1.0] * 4
    q99 = np.quantile(rel, 0.99, axis=0).tolist() + [1.0] * 4
    return {"method": "abs", "window_size": 0, "norm_stats": {
        "action.hand.position": {"q01": q01, "q99": q99},
        "action.effector.position": {"q01": [float(np.quantile(gripper, 0.01))], "q99": [float(np.quantile(gripper, 0.99))]},
    }}


def cut_video(meta: pd.Series, camera: str, out: Path) -> None:
    """The episode's frames from the concatenated v3.0 video file, re-encoded as one H.264 mp4."""
    src = SOURCE / "videos" / camera / f"chunk-{meta[f'videos/{camera}/chunk_index']:03d}" / f"file-{meta[f'videos/{camera}/file_index']:03d}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ss", f"{meta[f'videos/{camera}/from_timestamp']:.6f}",
                    "-frames:v", str(int(meta["length"])), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-an", str(out)], check=True)
    got = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                              "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True).stdout.strip())
    if got != int(meta["length"]):
        raise RuntimeError(f"{out}: {got} frames, expected {meta['length']}")


def feature_stats(values: np.ndarray) -> dict:
    values = values.reshape(len(values), -1).astype(np.float64)
    return {"min": values.min(0).tolist(), "max": values.max(0).tolist(), "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(), "count": [len(values)]}


def video_stats(video: Path) -> dict:
    """Per-channel stats of the frames scaled to [0, 1], shaped (3, 1, 1) as LeRobot stores image stats."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video), "-vf", "fps=2,scale=160:120", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    px = np.frombuffer(raw, np.uint8).reshape(-1, 3) / 255.0
    nest = lambda v: [[[float(x)]] for x in v]
    return {"min": nest(px.min(0)), "max": nest(px.max(0)), "mean": nest(px.mean(0)), "std": nest(px.std(0)), "count": [len(px) // (160 * 120)]}


def build_split(split: str, episode: int, meta: pd.Series, data: pd.DataFrame, info: dict, stats: dict, task: str) -> None:
    root = OUT / split
    if root.exists():
        shutil.rmtree(root)
    task_root = root / TASK_DIR
    (root / "meta").mkdir(parents=True)
    (task_root / "meta").mkdir(parents=True)
    (root / "meta" / "action_transform.yaml").write_text(yaml.safe_dump(ACTION_TRANSFORM, sort_keys=False))
    (root / "meta" / "action_stats.json").write_text(json.dumps(stats, indent=2))

    ep = data[data.episode_index == episode].sort_values("frame_index")[COLUMNS].reset_index(drop=True)
    if len(ep) != int(meta["length"]):
        raise RuntimeError(f"episode {episode}: {len(ep)} rows, expected {meta['length']}")
    ep["index"] = np.arange(len(ep))
    ep["task_index"] = 0
    (task_root / "data" / "chunk-000").mkdir(parents=True)
    ep.to_parquet(task_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet", index=False)
    videos = {camera: task_root / "videos" / "chunk-000" / camera / f"episode_{episode:06d}.mp4" for camera in CAMERAS}
    for camera, video in videos.items():
        cut_video(meta, camera, video)

    features = {k: v for k, v in info["features"].items() if k in COLUMNS}
    for camera in CAMERAS:
        vf = info["features"][camera]
        features[camera] = {"dtype": "video", "shape": vf["shape"], "names": vf.get("names", ["height", "width", "channels"]),
                            "info": {"video.height": vf["shape"][0], "video.width": vf["shape"][1], "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                     "video.is_depth_map": False, "video.fps": info["fps"], "video.channels": 3, "has_audio": False}}
    v21 = {"codebase_version": "v2.1", "robot_type": info.get("robot_type", "so101"), "total_episodes": 1, "total_frames": len(ep), "total_tasks": 1,
           "total_videos": len(CAMERAS), "total_chunks": 1, "chunks_size": 1000, "fps": info["fps"], "splits": {"train": "0:1"},
           "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
           "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": features}
    (task_root / "meta" / "info.json").write_text(json.dumps(v21, indent=4))
    (task_root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    # One action_config segment over the whole episode, as in the released Robotwin metadata.
    (task_root / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": episode, "tasks": [task], "length": len(ep),
                                                                   "action_config": [{"start_frame": 0, "end_frame": len(ep), "action_text": task}]}) + "\n")
    ep_stats = {c: feature_stats(np.stack(ep[c].to_numpy()) if ep[c].dtype == object else ep[c].to_numpy()) for c in COLUMNS}
    for camera, video in videos.items():
        ep_stats[camera] = video_stats(video)
    (task_root / "meta" / "episodes_stats.jsonl").write_text(json.dumps({"episode_index": episode, "stats": ep_stats}) + "\n")

    human_ep = human_episode(episode)
    sample = f"{TASK_DIR}_{human_ep:03d}"
    human = root / "human_data" / "so101" / RUN / "samples" / sample / "generated_video.mp4"
    human.parent.mkdir(parents=True)
    shutil.copy(HUMAN / f"episode_{human_ep:03d}" / "video.mp4", human)
    manifest = {"dataset": "so101_smoke", "samples": [{
        "run": RUN, "sample": sample, "sample_id": sample,
        "robot_video_path": f"{TASK_DIR}/videos/chunk-000/{CAMERAS[0]}/episode_{episode:06d}.mp4",
        "human_video_path": f"so101/{RUN}/samples/{sample}/generated_video.mp4",
        "robot_task_name": task, "human_text": HUMAN_TEXT,
        "source": {"robot": f"data/so101_export/lerobot/{TASK_DIR} episode {episode}",
                   "human": f"hf://buckets/nikgeo/ICL-so101/robot_removal/demos_h3_max_turbo/{TASK_DIR}/episode_{human_ep:03d}/video.mp4"},
    }]}
    (root / "icl_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{split}: episode {episode} (human demo episode_{human_ep:03d}), {len(ep)} frames -> {root}")


def main() -> None:
    info = json.loads((SOURCE / "meta" / "info.json").read_text())
    task = pd.read_parquet(SOURCE / "meta" / "tasks.parquet").index[0]
    episodes = episodes_meta().set_index("episode_index")
    data = frames()
    stats = action_stats(data)
    for split, episode in SPLITS.items():
        build_split(split, episode, episodes.loc[episode], data, info, stats, task)
    # What the Modal side needs to encode the latents.
    (OUT / "smoke.json").write_text(json.dumps({"task_dir": TASK_DIR, "cameras": CAMERAS, "splits": SPLITS, "human_text": HUMAN_TEXT}, indent=2))


if __name__ == "__main__":
    main()
