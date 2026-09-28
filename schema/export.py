"""Final accepted manifest and a clean export of the accepted pairs.

accept: an episode is accepted when its pair validates, source.outcome is success, and its latest verdict is pass or minor_issues.
The latest verdict is the audit's (Opus fix + Sonnet check) where the episode was audited, else the first Sonnet/Opus judge's.
export: writes LeRobot datasets holding only the accepted episodes (renumbered), their pair.json files (episode_index
renumbered to match), task templates/notes, schemas and the manifest into one folder with relative paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from schema.validate import check_pair

REPO = Path(__file__).resolve().parents[1]


def audit_verdicts(root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(dataset, episode) -> (pair path, final overall) from the audit results."""
    out = {}
    for q, s in json.loads((root / "_logs/audit_results.json").read_text()).items():
        if "fix2" in s and s["fix2"].get("genuine_failure"):
            final = "fail"
        elif "verify2" in s:
            final = s["verify2"]["overall"]
        elif "verify1" in s:
            final = s["verify1"]["overall"]
        else:
            continue
        p = Path(q)
        out[(p.parts[-3], p.parts[-2])] = (q, final)
    return out


def accept(root: Path) -> list[dict]:
    before = {(x["dataset"], x["episode"]): x for x in json.loads((root / "_logs/accepted_before_audit.json").read_text())}  # judge-based manifest before the audit
    audited = audit_verdicts(root)
    result = []
    for key in sorted(set(before) | set(audited)):
        q, judge = audited[key] if key in audited else (before[key]["pair"], before[key]["judge"])
        pair = json.loads(Path(q).read_text())
        if judge != "fail" and pair["source"]["outcome"] == "success" and not check_pair(pair):
            result.append({"dataset": key[0], "episode": key[1], "pair": q, "judge": judge, "audited": key in audited})
    return result


def filter_dataset(src: Path, keep: list[int], out: Path) -> None:
    """Copy a curated LeRobot dataset keeping only the given episodes (renumbered 0..n-1 in order)."""
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"local/{src.name}", root=src, video_backend="pyav")
    drop = sorted(set(range(ds.meta.total_episodes)) - set(keep))
    shutil.rmtree(out, ignore_errors=True)
    if drop:
        delete_episodes(ds, drop, output_dir=out, repo_id=f"local/{out.name}")
    else:
        shutil.copytree(src, out)
    cur = json.loads((src / "meta/curation.json").read_text())
    cur["export"] = {"kept_curated_episodes": keep, "note": "episode i here is curated episode kept_curated_episodes[i]"}
    (out / "meta/curation.json").write_text(json.dumps(cur, indent=1) + "\n")


def export(root: Path, accepted: list[dict], out: Path, curated: Path, datasets: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    by_dataset: dict[str, list[dict]] = {}
    for a in accepted:
        by_dataset.setdefault(a["dataset"], []).append(a)
    for d, rows in sorted(by_dataset.items()):
        pairs = {a["episode"]: json.loads(Path(a["pair"]).read_text()) for a in rows}
        keep = sorted(p["source"]["episode_index"] for p in pairs.values())
        new = {old: i for i, old in enumerate(keep)}
        if datasets:
            filter_dataset(curated / d, keep, out / "lerobot" / d)
        for a in rows:
            pair = pairs[a["episode"]]
            old = pair["source"]["episode_index"]
            pair["source"]["episode_index"] = new[old]
            rel = Path("pairs") / d / f"episode_{new[old]:03d}" / "pair.json"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(json.dumps(pair, indent=1) + "\n")
            manifest.append({"dataset": d, "episode_index": new[old], "pair": str(rel), "lerobot": f"lerobot/{d}",
                             "judge": a["judge"], "audited": a["audited"], "curated_episode_index": old})
    for d in sorted({a["dataset"] for a in accepted}):
        dst = out / "pairs" / d / "tasks"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (root / d / "tasks").iterdir():
            if not f.name.endswith(".old.txt"):
                shutil.copy2(f, dst / f.name)
    (out / "accepted.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "schema").mkdir(exist_ok=True)
    for f in ["common.schema.json", "scene.schema.json", "task.schema.json", "pair.schema.json", "validate.py", "README.md"]:
        shutil.copy2(REPO / "schema" / f, out / "schema" / f)
    shutil.copy2(REPO / "schema" / "export_readme.md", out / "README.md")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="data/so101_pairs")
    p.add_argument("--out", type=Path, help="also export the accepted pairs here")
    p.add_argument("--curated", type=Path, default=Path("data/so101_curated"))
    p.add_argument("--no-datasets", action="store_true", help="export pairs only, skip the filtered LeRobot datasets")
    args = p.parse_args(argv)
    accepted = accept(args.root)
    (args.root / "accepted.json").write_text(json.dumps(accepted, indent=1) + "\n")
    print(f"accepted {len(accepted)} episodes ({sum(a['audited'] for a in accepted)} audited)")
    if args.out:
        export(args.root, accepted, args.out, args.curated, not args.no_datasets)
        print(f"exported to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
