"""Curate the SO-101 human-robot pair dataset from the H3 Max Turbo human demos.

Every accepted robot episode has one generated human demo in hf://buckets/nikgeo/ICL-so101/robot_removal/demos_h3_max_turbo,
keyed by curated_episode_index. Up to 10 pairs per task go through review and, if accepted, into the dataset.

    python -m humangen.curate_dataset candidates            # ordered candidate list per task
    python -m humangen.curate_dataset sheets --round 1      # download demos, build review sheets for that round
    (Opus judge + verify workflow: humangen/curate_workflow.js, verdicts in data/so101_curation/verdicts/)
    python -m humangen.curate_dataset build                 # data/so101_curated: videos, actions, pair records, index.json
    python -m humangen.curate_dataset upload                # hf://buckets/akoniti/ICL-so101/curated_humangen_v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "data" / "so101_export"
WORK = REPO / "data" / "so101_curation"
OUT = REPO / "data" / "so101_curated"
DEMO_BUCKET = "nikgeo/ICL-so101/robot_removal/demos_h3_max_turbo"
DEST_BUCKET = "akoniti/ICL-so101/curated_humangen_v1"
PRIOR_REVIEW = WORK / "task_contact_review.json"  # the H3 Max Turbo first-pass review (_run/task_contact_review.json)
TARGET = 10
ROUND_SIZE = {1: 14, 2: 8, 3: 8, 4: 10}  # candidates per task added in each round


def accepted() -> dict:
    return {(r["dataset"], r["curated_episode_index"]): r for r in json.loads((EXPORT / "accepted.json").read_text())}


def demos() -> dict[str, list[int]]:
    """Curated episode indices with a human demo video, per task."""
    out: dict[str, set[int]] = {}
    for f in (WORK / "bucket_files.txt").read_text().split():
        p = f.split("/")
        if p[-1] == "video.mp4":
            out.setdefault(p[2], set()).add(int(p[3].split("_")[1]))
    return {k: sorted(v) for k, v in out.items()}


def spread(items: list[int]) -> list[int]:
    """Order that covers the whole episode range early (bisection order), so the first picks differ in layout."""
    order, seen, n = [], set(), len(items)
    step = n
    while len(order) < n:
        for i in range(0, n, max(1, step)):
            if i not in seen:
                seen.add(i)
                order.append(items[i])
        step //= 2
        if step == 0:
            order += [x for i, x in enumerate(items) if i not in seen]
            break
    return order


def candidates() -> None:
    prior = {}
    if PRIOR_REVIEW.exists():
        for t in json.loads(PRIOR_REVIEW.read_text())["tasks"]:
            for e in t["episodes"]:
                prior[e["key"]] = e["confirmed_success"]
    acc = accepted()
    out = {}
    for task, eps in sorted(demos().items()):
        eps = [e for e in eps if (task, e) in acc]
        first = [e for e in eps if prior.get(f"{task}/episode_{e:03d}")]  # confirmed in the first pass: still re-judged
        rest = [e for e in spread(eps) if e not in first and prior.get(f"{task}/episode_{e:03d}") is not False]
        failed = [e for e in eps if prior.get(f"{task}/episode_{e:03d}") is False]
        out[task] = first + rest + failed  # first-pass failures last
    (WORK / "candidates.json").write_text(json.dumps(out, indent=1))
    print({k: len(v) for k, v in out.items()}, sum(map(len, out.values())))


def adaptive_keys(max_per_task: int = 24) -> list[str]:
    """Next candidates for tasks short of TARGET that have any accepts so far, sized by the task's pass rate
    (with 30% headroom); tasks with no accepts are skipped (their failures are systematic in the generator)."""
    import math
    cands = json.loads((WORK / "candidates.json").read_text())
    verdicts = load_verdicts()
    keys = []
    for task, eps in cands.items():
        done = [e for e in eps if f"{task}/episode_{e:03d}" in verdicts]
        good = sum(verdicts[f"{task}/episode_{e:03d}"]["final"] == "accept" for e in done)
        if good >= TARGET or good == 0:
            continue
        need = math.ceil((TARGET - good) / (good / len(done)) * 1.3)
        todo = [e for e in eps if f"{task}/episode_{e:03d}" not in verdicts][:max(6, min(max_per_task, need))]
        keys += [f"{task}/episode_{e:03d}" for e in todo]
    return keys


def round_keys(n: int) -> list[str]:
    """The candidates of round n: the next ROUND_SIZE[n] per task, only for tasks still short of TARGET accepts."""
    cands = json.loads((WORK / "candidates.json").read_text())
    verdicts = load_verdicts()
    keys = []
    for task, eps in cands.items():
        done = [e for e in eps if f"{task}/episode_{e:03d}" in verdicts]
        good = [e for e in done if verdicts[f"{task}/episode_{e:03d}"]["final"] == "accept"]
        if n > 1 and len(good) >= TARGET:
            continue
        start = sum(ROUND_SIZE[i] for i in range(1, n))
        keys += [f"{task}/episode_{e:03d}" for e in eps[start:start + ROUND_SIZE[n]] if f"{task}/episode_{e:03d}" not in verdicts]
    return keys


def load_verdicts() -> dict:
    out = {}
    for f in sorted((WORK / "verdicts").glob("*.json")):
        for k, v in json.loads(f.read_text()).items():
            out[k] = v  # later rounds supersede
    return out


def export_episode(task: str, curated: int) -> int:
    return accepted()[(task, curated)]["episode_index"]


def _episodes_meta(task: str) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted((EXPORT / "lerobot" / task / "meta" / "episodes").glob("*/*.parquet"))).set_index("episode_index")


def _frames(src: Path, times: list[float], size=(320, 240)) -> list[Image.Image]:
    out = []
    for t in times:
        raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(src), "-frames:v", "1",
                              "-vf", f"scale={size[0]}:{size[1]}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        out.append(Image.frombytes("RGB", size, raw) if len(raw) == size[0] * size[1] * 3 else Image.new("RGB", size))
    return out


def _grid(frames: list[Image.Image], labels: list[str], cols: int) -> Image.Image:
    w, h = frames[0].size
    rows = (len(frames) + cols - 1) // cols
    g = Image.new("RGB", (cols * w, rows * h), "white")
    d = ImageDraw.Draw(g)
    for i, (f, lab) in enumerate(zip(frames, labels)):
        x, y = (i % cols) * w, (i // cols) * h
        g.paste(f, (x, y))
        d.rectangle([x, y, x + 9 * len(lab) + 6, y + 16], fill="black")
        d.text((x + 3, y + 2), lab, fill="white")
    return g


def _duration(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                capture_output=True, text=True, check=True).stdout.strip())


def sheet(key: str, meta_cache: dict) -> str:
    """Review sheet for one pair: 12 human-demo frames, 8 robot front-camera frames (first and last included), task."""
    task, ep = key.split("/")
    curated = int(ep.split("_")[1])
    d = WORK / "review" / task / ep
    d.mkdir(parents=True, exist_ok=True)
    human = d / "human.mp4"
    if not human.exists():
        subprocess.run(["uvx", "--from", "huggingface_hub", "hf", "buckets", "cp", f"hf://buckets/{DEMO_BUCKET}/{key}/video.mp4", str(human)],
                       check=True, capture_output=True)
    dur = _duration(human)
    ht = [(dur - 0.2) * i / 11 for i in range(12)]
    _grid(_frames(human, ht), [f"human {t:.1f}s" for t in ht], 4).save(d / "human_frames.jpg", quality=85)

    export_ep = export_episode(task, curated)
    m = meta_cache.setdefault(task, _episodes_meta(task)).loc[export_ep]
    cam = "observation.images.front"
    src = EXPORT / "lerobot" / task / "videos" / cam / f"chunk-{m[f'videos/{cam}/chunk_index']:03d}" / f"file-{m[f'videos/{cam}/file_index']:03d}.mp4"
    t0, t1 = float(m[f"videos/{cam}/from_timestamp"]), float(m[f"videos/{cam}/to_timestamp"])
    rt = [t0 + (t1 - t0 - 0.1) * i / 7 for i in range(8)]
    _grid(_frames(src, rt), [f"robot {t - t0:.1f}s" for t in rt], 4).save(d / "robot_frames.jpg", quality=85)

    pair = json.loads((EXPORT / accepted()[(task, curated)]["pair"]).read_text())
    t = pair["task"]
    info = {"key": key, "task": task, "curated_episode_index": curated, "export_episode_index": export_ep,
            "instruction": t["instruction"], "robot_caption": t.get("robot_caption"),
            "objects": [f"{r['name']}: {r.get('appearance', '')}" for r in t.get("roles", []) if r["kind"] != "actor"],
            "steps": t.get("steps"), "goals": t.get("goals"), "human_duration_s": round(dur, 2), "robot_duration_s": round(t1 - t0, 2)}
    (d / "info.json").write_text(json.dumps(info, indent=1))
    return key


def sheets(n: int) -> None:
    keys = round_keys(n) if n == 1 else adaptive_keys()
    (WORK / f"round{n}_keys.json").write_text(json.dumps(keys))
    cache: dict = {}
    for task in {k.split("/")[0] for k in keys}:
        cache[task] = _episodes_meta(task)
    with ThreadPoolExecutor(8) as ex:
        done = list(ex.map(lambda k: sheet(k, cache), keys))
    print(f"round {n}: {len(done)} sheets in {WORK / 'review'}")


def select() -> dict[str, list[str]]:
    """Up to TARGET verified pairs per task, in candidate order (spread over the episode range)."""
    cands = json.loads((WORK / "candidates.json").read_text())
    verdicts = load_verdicts()
    out = {}
    for task, eps in cands.items():
        good = [f"{task}/episode_{e:03d}" for e in eps if verdicts.get(f"{task}/episode_{e:03d}", {}).get("final") == "accept"]
        if good:
            out[task] = good[:TARGET]
    return out


def _cut(task: str, export_ep: int, cam: str, out: Path, meta: pd.DataFrame) -> None:
    m = meta.loc[export_ep]
    src = EXPORT / "lerobot" / task / "videos" / cam / f"chunk-{m[f'videos/{cam}/chunk_index']:03d}" / f"file-{m[f'videos/{cam}/file_index']:03d}.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ss", f"{m[f'videos/{cam}/from_timestamp']:.6f}",
                    "-frames:v", str(int(m["length"])), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium",
                    "-movflags", "+faststart", "-an", str(out)], check=True)


def build_episode(key: str, verdict: dict, cache: dict) -> dict:
    task, ep = key.split("/")
    curated = int(ep.split("_")[1])
    row = accepted()[(task, curated)]
    export_ep = row["episode_index"]
    d = OUT / "episodes" / task / ep
    d.mkdir(parents=True, exist_ok=True)
    review = WORK / "review" / task / ep
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(review / "human.mp4"), "-c", "copy", "-movflags", "+faststart", str(d / "human.mp4")], check=True)
    info = json.loads((EXPORT / "lerobot" / task / "meta" / "info.json").read_text())
    cams = [k for k in info["features"] if k.startswith("observation.images.")]
    meta = cache.setdefault(task, _episodes_meta(task))
    views = {}
    for cam in cams:
        name = cam.split(".")[-1]
        out = d / f"robot_{name}.mp4"
        if not out.exists():
            _cut(task, export_ep, cam, out, meta)
        views[name] = out.name
    dur = _duration(d / "human.mp4")
    _frames(d / "human.mp4", [dur * 0.45], (480, 360))[0].save(d / "thumb.jpg", quality=80)
    _frames(d / "robot_front.mp4", [0.0], (480, 360))[0].save(d / "robot_thumb.jpg", quality=80)
    frames = pd.concat(pd.read_parquet(f) for f in sorted((EXPORT / "lerobot" / task / "data").glob("*/*.parquet")))
    frames[frames.episode_index == export_ep].sort_values("frame_index").to_parquet(d / "robot_data.parquet", index=False)
    pair = json.loads((EXPORT / row["pair"]).read_text())
    (d / "pair.json").write_text(json.dumps(pair, indent=1))
    (d / "review.json").write_text(json.dumps(verdict, indent=1))
    j, v = verdict["judge"], verdict["verify"]
    return {"id": key, "task": task, "curated_episode_index": curated, "export_episode_index": export_ep,
            "instruction": pair["task"]["instruction"], "fps": info["fps"], "robot_frames": int(meta.loc[export_ep]["length"]),
            "robot_duration_s": round(int(meta.loc[export_ep]["length"]) / info["fps"], 2), "human_duration_s": round(dur, 2),
            "steps": [f"{st['action'].replace('_', ' ')}: {str(st.get('object') or '').replace('_', ' ')}"
                      + (f" → {st['destination'].replace('_', ' ')}" if st.get("destination") else "") for st in pair["task"].get("steps", [])],
            "views": views, "human": "human.mp4", "thumb": "thumb.jpg", "robot_thumb": "robot_thumb.jpg",
            "review": {"task_adherence": j["task_adherence"], "physics": j["physics"], "summary": j["summary"], "issues": j["issues"],
                       "verify": v["reason"]}}


def build() -> None:
    chosen = select()
    verdicts = load_verdicts()
    cache: dict = {}
    for task in chosen:
        cache[task] = _episodes_meta(task)
    keys = [k for ks in chosen.values() for k in ks]
    with ThreadPoolExecutor(6) as ex:
        eps = list(ex.map(lambda k: build_episode(k, verdicts[k], cache), keys))
    cands = json.loads((WORK / "candidates.json").read_text())
    tasks = []
    for task in sorted(cands):
        reviewed = [k for k in verdicts if k.startswith(task + "/")]
        mine = [e for e in eps if e["task"] == task]
        tasks.append({"task": task, "instruction": mine[0]["instruction"] if mine else
                      json.loads((EXPORT / accepted()[(task, cands[task][0])]["pair"]).read_text())["task"]["instruction"],
                      "episodes": len(mine), "reviewed": len(reviewed), "accepted": sum(verdicts[k]["final"] == "accept" for k in reviewed),
                      "available_demos": len(cands[task]), "views": sorted(mine[0]["views"]) if mine else []})
    index = {"name": "SO-101 HumanGen pairs", "version": "v1", "base_url": f"https://huggingface.co/buckets/{DEST_BUCKET.rsplit('/', 1)[0]}/resolve/{DEST_BUCKET.rsplit('/', 1)[1]}/",
             "episodes_total": len(eps), "tasks_total": sum(t["episodes"] > 0 for t in tasks), "tasks": tasks, "episodes": eps}
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(eps)} episodes over {index['tasks_total']} tasks -> {OUT}")


def upload() -> None:
    subprocess.run(["uvx", "--from", "huggingface_hub", "hf", "buckets", "sync", str(OUT), f"hf://buckets/{DEST_BUCKET}"], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["candidates", "sheets", "build", "upload"])
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "candidates":
        candidates()
    elif a.cmd == "sheets":
        sheets(a.round)
    elif a.cmd == "build":
        build()
    else:
        upload()


if __name__ == "__main__":
    main()
