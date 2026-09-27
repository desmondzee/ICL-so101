"""Convert every dataset in selection.json using the per-episode checks, then merge the front camera views."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from curate.convert import _canonical_features, convert
from curate.episodes import SELECTION
from curate.io import load

EPISODES = Path("data/so101_curated/_episodes")


def _frames(video: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return int(r.stdout.strip() or 0)


def length_mismatch(entry: dict) -> dict[int, str]:
    """v2.1 episodes whose per-episode video frame count differs from the trajectory length."""
    src = load(entry["root"])
    if not src.version.startswith("v2"):
        return {}
    by_name = {v: k for k, v in src.cameras.items()}
    keys = [by_name[c] for c in (entry["front"], entry["wrist"]) if c]
    jobs = [(int(ep), int(n), k) for ep, n in zip(src.episodes.episode_index, src.episodes.length) for k in keys]
    with ThreadPoolExecutor(8) as ex:
        counts = list(ex.map(lambda j: _frames(src.video(j[2], j[0])[0]), jobs))
    return {ep: f"video has {c} frames, data {n}" for (ep, n, _), c in zip(jobs, counts) if abs(c - n) > 2}


def decisions(name: str, review: list[dict]) -> tuple[set[int], dict[int, str], dict]:
    """Camera swaps and dropped episodes from the motion check and the visual review."""
    motion = {int(k): v for k, v in json.loads((EPISODES / "motion" / f"{name}.json").read_text()).items()}
    rows = {r["episode"]: r for r in review if r["sheet"].rsplit("__", 1)[0] == name}
    swap, drop = set(), {}
    for ep, m in motion.items():
        r = rows.get(ep)
        if m.get("video_error") or m.get("video_short"):
            drop[ep] = "video unreadable or shorter than data"
        elif r and r["drop"]:
            drop[ep] = "review: " + ",".join(r["flags"])
        elif r and (r["swap"] or (m.get("swapped") and r["swapped_cameras"])):
            swap.add(ep)
        elif r and bool(m.get("swapped")) != r["swapped_cameras"]:
            drop[ep] = "camera assignment ambiguous"
    summary = {"reviewed": len(rows), "missing_review": sorted(set(motion) - set(rows))}
    return swap, drop, summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("review", type=Path, help="JSON list of per-episode review records")
    p.add_argument("--out", type=Path, default=Path("data/so101_curated"))
    p.add_argument("--only", nargs="*")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args(argv)
    review = json.loads(args.review.read_text())
    entries = [e for e in json.loads(SELECTION.read_text()) if not args.only or e["name"] in args.only]

    for e in entries if not args.merge_only else []:
        out = args.out / e["name"]
        if (out / "meta/curation.json").exists():
            print(f"skip {e['name']}")
            continue
        swap, drop, summary = decisions(e["name"], review)
        drop = length_mismatch(e) | drop
        prov = {"source": e["source"], "family": e["family"], "source_tasks": e["tasks"], "review": summary}
        convert(Path(e["root"]), out, e["front"], e["wrist"], prov, swap, drop)
        print(f"done {e['name']}: {len(swap)} swapped, {len(drop)} dropped by review/video", flush=True)

    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.dataset_tools import remove_feature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    staging = args.out / "_front_only"
    shutil.rmtree(staging, ignore_errors=True)
    roots = []
    for e in json.loads(SELECTION.read_text()):
        root = args.out / e["name"]
        if not (root / "meta/curation.json").exists():
            continue
        if e["wrist"]:
            ds = LeRobotDataset(f"local/{e['name']}", root=root, video_backend="pyav")
            front = staging / e["name"]
            remove_feature(ds, "observation.images.wrist", output_dir=front, repo_id=f"local/{e['name']}")
            _canonical_features(front)
            roots.append(front)
        else:
            roots.append(root)
    merged = args.out / "_merged_front"
    shutil.rmtree(merged, ignore_errors=True)
    aggregate_datasets([f"local/{r.name}" for r in roots], "local/so101_curated_front", roots=roots, aggr_root=merged)
    shutil.rmtree(staging)
    sources = [json.loads((args.out / r.name / "meta/curation.json").read_text()) | {"name": r.name} for r in roots]
    (merged / "meta/curation.json").write_text(json.dumps({
        "family": "merged", "units": "deg", "dropped_episodes": {}, "swapped_camera_episodes": [],
        "sources": [{k: s[k] for k in ("name", "family", "source", "source_tasks")} for s in sources],
        "note": "Front camera only; wrist views are in the per-source datasets.",
    }, indent=2) + "\n")
    print(f"merged {len(roots)} datasets into {merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
