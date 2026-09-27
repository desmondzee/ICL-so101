"""Convert every dataset in selection.json using the per-episode checks, then merge the front camera views."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from curate.convert import _canonical_features, convert
from curate.episodes import SELECTION

EPISODES = Path("data/so101_curated/_episodes")


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
        elif m.get("swapped") != (r["swapped_cameras"] if r else m.get("swapped")):
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
    print(f"merged {len(roots)} datasets into {merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
