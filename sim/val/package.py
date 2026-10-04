"""Package the simulated validation set: 10 reviewed human-robot pairs per task.

Takes, per task, the first 10 recorded episodes whose human demo passed the Opus judge and verifier
(data/so101_sim_val/verdicts/approved.json), re-records exactly those seeds into clean LeRobot datasets and writes
data/so101_sim_val_v1/:

    lerobot/<task>/                     front + wrist, 30 fps, the training-data columns (episodes 0-9)
    frames/<task>/episode_XXX/          robot-free first.png / last.png and the oracle meta.json
    episodes/<task>/episode_XXX/        human.mp4 (no audio), robot_front.mp4, robot_wrist.mp4 (H.264), robot_data.parquet,
                                        review.json, source.json, thumb.jpg, robot_thumb.jpg
    index.json, README.md

    uv run python -m sim.val.package [--skip-record]
    uv run --with huggingface_hub hf buckets sync data/so101_sim_val_v1 hf://buckets/akoniti/ICL-so101/sim_val_v1 --delete
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from sim.val.record import record

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "so101_sim_val"
OUT = ROOT / "data" / "so101_sim_val_v1"
BUCKET = "akoniti/ICL-so101/sim_val_v1"
PER_TASK = 10
CAMS = ("front", "wrist")


def selection():
    approved = json.loads((WORK / "verdicts" / "approved.json").read_text())
    out = {}
    for task, eps in sorted(approved.items()):
        chosen = sorted(eps)[:PER_TASK]
        out[task] = [{"source_episode": ep, "gen_seed": eps[ep],
                      "seed": json.loads((WORK / "frames" / task / ep / "meta.json").read_text())["seed"]} for ep in chosen]
    return out


def verdicts():
    out = {}
    for f in sorted((WORK / "verdicts").glob("*.json")):
        if f.name != "approved.json":
            out.update(json.loads(f.read_text()))
    return out


def run(*cmd):
    subprocess.run(cmd, check=True)


def frame_at(video, t, out, size="480:360"):
    run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={size}", str(out))


def duration(video):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                                capture_output=True, text=True, check=True).stdout.strip())


def package_episode(task, i, sel, review, eps_meta, data):
    d = OUT / "episodes" / task / f"episode_{i:03d}"
    d.mkdir(parents=True, exist_ok=True)
    human = WORK / "human" / task / sel["source_episode"] / f"seed_{sel['gen_seed']}" / "video.mp4"
    run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(human), "-an", "-c:v", "copy", "-movflags", "+faststart", str(d / "human.mp4"))
    m = eps_meta.loc[i]
    root = OUT / "lerobot" / task
    for cam in CAMS:
        key = f"observation.images.{cam}"
        src = root / "videos" / key / f"chunk-{m[f'videos/{key}/chunk_index']:03d}" / f"file-{m[f'videos/{key}/file_index']:03d}.mp4"
        run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ss", f"{m[f'videos/{key}/from_timestamp']:.6f}",
            "-frames:v", str(int(m["length"])), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", "-an",
            str(d / f"robot_{cam}.mp4"))
    data[data.episode_index == i].sort_values("frame_index").to_parquet(d / "robot_data.parquet", index=False)
    hd = duration(d / "human.mp4")
    frame_at(d / "human.mp4", hd * 0.45, d / "thumb.jpg")
    frame_at(d / "robot_front.mp4", 0.0, d / "robot_thumb.jpg")
    (d / "review.json").write_text(json.dumps(review, indent=1))
    meta = json.loads((OUT / "frames" / task / f"episode_{i:03d}" / "meta.json").read_text())
    (d / "source.json").write_text(json.dumps({**sel, "human_video": str(human.relative_to(ROOT))}, indent=1))
    j, v = review["judge"], review["verify"]
    return {"id": f"{task}/episode_{i:03d}", "task": task, "episode": i, "seed": sel["seed"], "instruction": meta["instruction"],
            "fps": 30, "robot_frames": int(m["length"]), "robot_duration_s": round(int(m["length"]) / 30, 2), "human_duration_s": round(hd, 2),
            "views": {cam: f"robot_{cam}.mp4" for cam in CAMS}, "human": "human.mp4", "thumb": "thumb.jpg", "robot_thumb": "robot_thumb.jpg",
            "distractors": meta.get("distractors", []),
            "review": {"task_adherence": j["task_adherence"], "physics": j["physics"], "summary": j["summary"], "issues": j["issues"], "verify": v["reason"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-record", action="store_true", help="reuse data/so101_sim_val_v1/lerobot")
    a = ap.parse_args()
    sel, reviews = selection(), verdicts()
    if not a.skip_record:
        for task, rows in sel.items():
            saved, _ = record(task, len(rows), 0, out=OUT, seeds=[r["seed"] for r in rows])
            assert saved == len(rows), (task, saved)
    shutil.rmtree(OUT / "episodes", ignore_errors=True)
    episodes, tasks = [], []
    for task, rows in sel.items():
        root = OUT / "lerobot" / task
        eps_meta = pd.concat(pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").glob("*/*.parquet"))).set_index("episode_index")
        data = pd.concat(pd.read_parquet(f) for f in sorted((root / "data").glob("*/*.parquet")))
        mine = []
        for i, s in enumerate(rows):
            review = reviews[f"{task}/{s['source_episode']}_s{s['gen_seed']}"]
            assert review["final"] == "accept"
            mine.append(package_episode(task, i, s, review, eps_meta, data))
        episodes += mine
        tasks.append({"task": task, "instruction": mine[0]["instruction"], "episodes": len(mine), "views": list(CAMS),
                      "instructions": sorted({e["instruction"] for e in mine})})
    index = {"name": "SO-101 simulated validation pairs", "version": "v1",
             "base_url": f"https://huggingface.co/buckets/{BUCKET.rsplit('/', 1)[0]}/resolve/{BUCKET.rsplit('/', 1)[1]}/",
             "episodes_total": len(episodes), "tasks_total": len(tasks), "tasks": tasks, "episodes": episodes}
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(episodes)} pairs over {len(tasks)} tasks -> {OUT}")


if __name__ == "__main__":
    main()
