"""Package, publish and read back the accepted simulated training pairs (sim_train_v1).

Builds ``data/so101_sim_train_v1_release/`` from the episode store, taking ONLY episodes in state ``accepted``
(robot physics QA, robot-video review, human-pair review and independent verifier all passed). Every new pair is
loaded through ``EpisodeStore.load`` (re-hashes the frozen manifest artifacts, every state-history and human-attempt
evidence file), and every copied file is re-hashed against the manifest after the copy. Nothing is re-recorded.

Layout (mirrors sim_val_v2, but the LeRobot data is per episode, see below)::

    index.json                       tasks + episodes (gallery schema of site/data/val_index.json, plus family/seed)
    inventory.json                   sha256 + size of every pair file (the release is append-only)
    README.md                        dataset card (regenerated each run from the store, ledger and release)
    episodes/<task>/episode_XXX/
        human.mp4                    accepted generated human attempt, audio stripped (stream copy, faststart)
        robot_front.mp4, robot_wrist.mp4, robot_data.parquet   byte copies of the frozen candidate
        first.png, last.png          robot-free endpoints the human clip was generated from (byte copies)
        episode.json                 scene/visual config, object poses, oracle log (byte copy)
        physics.json, policy.json    substep physics QA result and its thresholds (byte copies)
        review.json                  physics QA + robot review (judge, verifier) + human review (judge, verifier) + attempts
        source.json                  task, seed, family, hashes, fal request id/seed/settings/cost, candidate path
        thumb.jpg, robot_thumb.jpg
    lerobot/<task>/episode_XXX/      the frozen single-episode LeRobot v3 dataset (AV1 front/wrist, joint + EE columns)

LeRobot choice: per-episode v3 datasets, byte-identical to the hash-verified recordings, because the release is
append-only (a merged per-task dataset would have to be rewritten and re-uploaded whenever a pair is added). Merge
locally with ``lerobot.datasets.aggregate.aggregate_datasets`` when one dataset per task is wanted (README).

Incremental and idempotent: episode numbers per task are fixed in index.json; new accepted seeds are appended in
seed order. Existing pair directories are never rewritten (their sizes are checked each run, hashes with --verify).
index.json / inventory.json / README.md are only rewritten when their content changes.

    .venv/bin/python -m sim.train.package              # package only
    .venv/bin/python -m sim.train.package --publish    # package + HF sync + public readback + site/data/sim_index.json
    uv run --with huggingface_hub hf buckets sync data/so101_sim_train_v1_release hf://buckets/akoniti/ICL-so101/sim_train_v1
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import urllib.request

from .model import EpisodeKey, EpisodeState
from .store import EpisodeStore, sha256_file

REPO = Path(__file__).resolve().parents[2]
STORE_ROOT = REPO / "data" / "so101_sim_train_v1"
OUT = REPO / "data" / "so101_sim_train_v1_release"
BUCKET = "akoniti/ICL-so101/sim_train_v1"
BASE_URL = "https://huggingface.co/buckets/akoniti/ICL-so101/resolve/sim_train_v1/"
SITE_INDEX = REPO / "site" / "data" / "sim_index.json"
VAL_TASKS = ("mug_on_plate", "mugs_in_microwave", "pan_on_stove", "sort_blocks", "stack_bowls")
CAMS = ("front", "wrist")
COPIED = ("robot_front.mp4", "robot_wrist.mp4", "robot_data.parquet", "first.png", "last.png",
          "episode.json", "physics.json", "policy.json")


class ReleaseError(RuntimeError):
    pass


# ---------------------------------------------------------------- helpers

def _run(*cmd):
    subprocess.run(cmd, check=True)


def _duration(video: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                                capture_output=True, text=True, check=True).stdout.strip())


def _frame_at(video: Path, t: float, out: Path, size="480:360"):
    _run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={size}", str(out))


def _write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text() == text:
        return False
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return True


def _files(directory: Path) -> dict[str, Path]:
    return {str(p.relative_to(OUT)): p for p in sorted(directory.rglob("*")) if p.is_file()}


def accepted_keys(root: Path = STORE_ROOT) -> list[EpisodeKey]:
    keys = []
    for path in sorted((root / "candidates").glob("*/episode_*/state.json")):
        try:
            if json.loads(path.read_text()).get("state") == EpisodeState.ACCEPTED.value:
                keys.append(EpisodeKey(path.parent.parent.name, int(path.parent.name.split("_")[1])))
        except (OSError, ValueError):
            continue
    return keys


def _history(record, new_state: str) -> dict:
    for entry in record.history:
        if entry["new_state"] == new_state:
            return entry
    raise ReleaseError(f"{record.manifest.key}: no {new_state} transition")


# ---------------------------------------------------------------- one pair

def build_pair(store: EpisodeStore, key: EpisodeKey, number: int) -> tuple[dict, dict]:
    """Copy one accepted pair into OUT. Returns (index entry, inventory rows). Refuses anything not accepted."""
    record = store.load(key)  # verifies manifest, artifacts, history and attempt evidence hashes
    if record.state is not EpisodeState.ACCEPTED:
        raise ReleaseError(f"{key}: state {record.state.value}, not accepted")
    meta, artifacts = record.manifest.metadata, record.manifest.artifacts
    final = record.history[-1]
    if final["new_state"] != EpisodeState.ACCEPTED.value:
        raise ReleaseError(f"{key}: last transition is not acceptance")
    attempt, human_sha = final["evidence"]["attempt"], final["evidence"]["human_sha256"]
    attempts = {a["attempt_id"]: a for a in record.human_attempts}
    complete, review = attempts[f"{attempt}-complete"]["evidence"], attempts[f"{attempt}-review"]["evidence"]
    submitted = attempts.get(f"{attempt}-submitted", {}).get("evidence", {})
    if complete["video_sha256"] != human_sha or review.get("decision") != "accept" or review.get("human_sha256") != human_sha:
        raise ReleaseError(f"{key}: accepted attempt {attempt} records disagree")
    src = store.episode_dir(key)
    if sha256_file(src / complete["video"]) != human_sha:
        raise ReleaseError(f"{key}: human video hash mismatch")
    robot_review = _history(record, EpisodeState.ROBOT_APPROVED.value)["evidence"]
    qa = _history(record, EpisodeState.PHYSICS_APPROVED.value)["evidence"]
    verifier = _history(record, EpisodeState.VERIFIER_APPROVED.value)

    name = f"episode_{number:03d}"
    final_dir, final_lr = OUT / "episodes" / key.task / name, OUT / "lerobot" / key.task / name
    tmp_dir, tmp_lr = final_dir.with_name(f".{name}.tmp"), final_lr.with_name(f".{name}.tmp")
    for d in (tmp_dir, tmp_lr):
        shutil.rmtree(d, ignore_errors=True)
        d.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir()

    for f in COPIED:
        shutil.copyfile(src / f, tmp_dir / f)
        if sha256_file(tmp_dir / f) != artifacts[f]:
            raise ReleaseError(f"{key}: copied {f} does not match the frozen manifest")
    lerobot = {k[len("lerobot/"):]: v for k, v in artifacts.items() if k.startswith("lerobot/")}
    for rel, digest in lerobot.items():
        (tmp_lr / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "lerobot" / rel, tmp_lr / rel)
        if sha256_file(tmp_lr / rel) != digest:
            raise ReleaseError(f"{key}: copied lerobot/{rel} does not match the frozen manifest")
    if not lerobot:
        raise ReleaseError(f"{key}: no LeRobot dataset in manifest")

    _run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src / complete["video"]), "-an", "-c:v", "copy",
         "-movflags", "+faststart", str(tmp_dir / "human.mp4"))
    hd = _duration(tmp_dir / "human.mp4")
    _frame_at(tmp_dir / "human.mp4", hd * 0.45, tmp_dir / "thumb.jpg")
    _frame_at(tmp_dir / "robot_front.mp4", 0.0, tmp_dir / "robot_thumb.jpg")

    judge, verify = review["judge"], review.get("verify") or {}
    rejected = [{"attempt": a["evidence"]["attempt"], "decision": a["evidence"].get("decision"),
                 "reasons": a["evidence"].get("reasons", [])}
                for aid, a in sorted(attempts.items()) if aid.endswith("-review") and aid != f"{attempt}-review"]
    review_doc = {
        "final": "accept", "episode": f"{key.task}/episode_{key.seed}",
        "physics_qa": {"stage": qa.get("stage"), "physics_accepted": qa.get("physics_accepted"),
                       "final_task_success": qa.get("final_task_success"), "terminal_visibility": qa.get("terminal_visibility"),
                       "failed_physics_checks": qa.get("failed_physics_checks", []), "details": "physics.json"},
        "robot_review": {k: robot_review.get(k) for k in ("decision", "problems", "judge", "verify", "request_sha256")},
        "human_review": {"attempt": attempt, "decision": review["decision"], "problems": review.get("problems", []),
                         "judge": judge, "info_sha256": review.get("info_sha256")},
        "verifier": {"attempt": attempt, "verify": verify, "timestamp": verifier["timestamp"]},
        "accepted_at": final["timestamp"],
        "other_attempts": rejected,
    }
    (tmp_dir / "review.json").write_text(json.dumps(review_doc, indent=1) + "\n")
    ep_json = json.loads((src / "episode.json").read_text())
    source = {
        "task": key.task, "seed": key.seed, "family": meta["family"], "instruction": meta["instruction"],
        "action_order": meta["action_order"], "action_text": meta["action_text"], "arena": meta.get("arena"),
        "config_hash": record.manifest.config_hash, "visual_config_hash": record.manifest.visual_config_hash,
        "recorder_hash": meta.get("recorder_hash"), "qualification": meta.get("qualification"), "versions": meta.get("versions"),
        "candidate": f"data/so101_sim_train_v1/candidates/{key.task}/episode_{key.seed}",
        "frozen_sha256": {f: artifacts[f] for f in COPIED},
        "human": {"attempt": attempt, "endpoint": submitted.get("endpoint", "minimax/h3-max-turbo/image-to-video"),
                  "request_id": complete["request_id"], "seed": complete["seed"], "settings": complete["settings"],
                  "cost_usd": complete.get("cost_usd"), "cost_source": complete.get("cost_source"),
                  "price_per_s": submitted.get("price_per_s"), "prompt": submitted.get("prompt"),
                  "prompt_sha256": complete.get("prompt_sha256"), "source_video_sha256": human_sha,
                  "probe": complete.get("probe"), "audio": "stripped (stream copy, -an)"},
    }
    (tmp_dir / "source.json").write_text(json.dumps(source, indent=1) + "\n")

    frames = int(meta["frames"])
    entry = {
        "id": f"{key.task}/{name}", "task": key.task, "family": meta["family"], "episode": number, "seed": key.seed,
        "instruction": meta["instruction"], "fps": int(meta["fps"]), "robot_frames": frames,
        "robot_duration_s": round(frames / meta["fps"], 2), "human_duration_s": round(hd, 2),
        "views": {cam: f"robot_{cam}.mp4" for cam in CAMS}, "human": "human.mp4", "thumb": "thumb.jpg",
        "robot_thumb": "robot_thumb.jpg", "lerobot": f"lerobot/{key.task}/{name}/",
        "arena": ep_json.get("visual_config", {}).get("arena"), "distractors": ep_json.get("distractors", []),
        "review": {"task_adherence": judge.get("task_adherence"), "physics": judge.get("physics"),
                   "summary": judge.get("summary", ""), "issues": judge.get("issues", []),
                   "verify": " ".join(verify.get("reasons", [])),
                   "robot": " ".join((robot_review.get("judge") or {}).get("reasons", [])[:2])},
    }
    os.replace(tmp_lr, final_lr)
    os.replace(tmp_dir, final_dir)
    rows = {rel: {"sha256": sha256_file(p), "size": p.stat().st_size}
            for d in (final_dir, final_lr) for rel, p in _files(d).items()}
    return entry, rows


# ---------------------------------------------------------------- stats and card

def store_stats(root: Path = STORE_ROOT) -> dict:
    """Workflow counts over the whole store (raw JSON reads; release pairs are verified separately)."""
    states, robot, human = collections.Counter(), collections.Counter(), collections.Counter()
    for path in (root / "candidates").glob("*/episode_*/state.json"):
        try:
            s = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        states[s["state"]] += 1
        seen = {h["new_state"] for h in s["history"]}
        if "robot_approved" in seen:
            robot["approved"] += 1
        elif s["state"] == "rejected" and s["history"][-1]["evidence"].get("stage") == "robot_review":
            robot["rejected"] += 1
        elif s["state"] == "rejected":
            robot["physics_rejected"] += 1
        for f in (path.parent / "human_attempts").glob("*-review.json"):
            try:
                human[json.loads(f.read_text())["evidence"].get("decision", "?")] += 1
            except (OSError, ValueError, KeyError):
                pass
        human["generated"] += len(list((path.parent / "human_attempts").glob("*-complete.json")))
    out = {"states": dict(states), "robot_review": dict(robot), "human_review": dict(human)}
    ledger = root / "generation" / "ledger.json"
    if ledger.exists():
        from .budget import totals
        doc = json.loads(ledger.read_text())
        t = totals(doc)
        out["spend"] = {"committed_usd": t["committed_usd"], "unresolved_usd": t["unresolved_usd"],
                        "requests": len(doc["requests"]), "by_status": t["by_status"], "cap_usd": doc.get("absolute_cap_usd")}
    return out


def _task_modules() -> dict[str, str]:
    out = {}
    for path in sorted((REPO / "sim" / "train" / "tasks" / "families").glob("*.py")):
        for name in re.findall(r'define_task\(\s*name\s*=\s*"([^"]+)"', path.read_text()):
            out[name] = path.stem
    return out


def _scanned_assets(tasks: list[str]) -> dict[str, list[str]]:
    """Scanned (GSO/YCB) asset ids referenced by the family modules that define the released tasks."""
    modules, out = _task_modules(), collections.defaultdict(set)
    for module in {modules.get(t) for t in tasks} - {None}:
        text = (REPO / "sim" / "train" / "tasks" / "families" / f"{module}.py").read_text()
        for source, model in re.findall(r'"(gso|ycb)",\s*"([^"]+)"', text):
            out[source].add(model)
    return {k: sorted(v) for k, v in out.items()}


def _rng(values):
    return f"{min(values):.2f}-{max(values):.2f}" if values else "n/a"


def readme(index: dict, stats: dict) -> str:
    eps, tasks = index["episodes"], index["tasks"]
    fam = collections.Counter(e["family"] for e in eps)
    fam_tasks = collections.defaultdict(set)
    for e in eps:
        fam_tasks[e["family"]].add(e["task"])
    mins = sum(e["robot_duration_s"] for e in eps) / 60
    arenas = collections.Counter(e.get("arena") for e in eps)
    lights, fovy = [], []
    for e in eps:
        try:
            vc = json.loads((OUT / "episodes" / e["id"] / "episode.json").read_text())["visual_config"]
            lights += [l["intensity"] for l in vc["lights"] if l.get("name") == "key"]
            fovy.append(vc["front_camera"]["fovy"])
        except (OSError, KeyError, ValueError):
            pass
    rr, hr, sp = stats.get("robot_review", {}), stats.get("human_review", {}), stats.get("spend", {})
    robot_n = rr.get("approved", 0) + rr.get("rejected", 0)
    human_n = hr.get("accept", 0) + hr.get("reject", 0)
    accepted_all = stats.get("states", {}).get("accepted", 0)
    assets = _scanned_assets([t["task"] for t in tasks])
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "n/a"
    lines = [
        f"# {index['name']} (sim_train_v1)", "",
        f"{len(eps)} accepted human-robot pairs over {len(tasks)} task definitions in {len(fam)} manipulation families "
        f"({mins:.0f} min of robot data). Each pair joins a scripted SO-101 episode recorded in MuJoCo (so101-nexus, "
        "LIBERO assets at 0.5x plus a few scanned YCB/GSO objects) with a generated video of a person doing the same task "
        "in the same rendered scene. **This is simulated training data**: it is separate from the real training pairs "
        "(`curated_humangen_v1`) and from the held-out simulated validation set (`sim_val_v2`).", "",
        "Viewer: https://desmondzee.github.io/ICL-so101/#sim (grid) and "
        "https://desmondzee.github.io/ICL-so101/viewer.html#sim/all/1 (one pair at a time)", "",
        f"The release is published incrementally while review continues (index.json `updated`: {index['updated']}). "
        "Existing pairs never change; new accepted pairs are appended with the next episode number of their task.", "",
        "## Counts by family", "", "| Family | Tasks | Pairs |", "|---|---|---|",
        *[f"| `{f}` | {len(fam_tasks[f])} | {n} |" for f, n in sorted(fam.items(), key=lambda x: (-x[1], x[0]))], "",
        "## Counts by task", "", "| Task | Family | Instruction | Pairs |", "|---|---|---|---|",
        *[f"| `{t['task']}` | `{t['family']}` | {t['instruction']} | {t['episodes']} |" for t in tasks], "",
        "## How it was made", "",
        "1. **Robot episodes.** Privileged-state scripted planners (IK, planned arc/joint-space carries, min-jerk timing, "
        "joint speed limits) per task, admitted only after qualification on 50 unseen seeds (>= 43/50 strict success). "
        "Each recorded episode then passed a substep physics audit (finite state, simulator warnings, joint ranges, "
        "penetration, allowed contacts with bounded fingertip-table grasp contact, distractor displacement, released grasp, "
        "support, and the goal holding for a settled interval); see `physics.json` / `policy.json` per pair.",
        "2. **Visual variation.** Per episode, deterministic and frozen before review: arena/background, key/fill light "
        f"colour, intensity and direction, and front-camera pose/FOV. Released coverage: arenas {dict(arenas)}; key-light "
        f"intensity {_rng(lights)}; front-camera fovy {_rng(fovy)} deg. The wrist camera keeps the physical mount.",
        f"3. **Robot-video review** before any paid generation: an Opus judge and an adversarial Opus verifier per episode. "
        f"Store-wide: {rr.get('approved', 0)} approved / {robot_n} reviewed ({pct(rr.get('approved', 0), robot_n)}); "
        f"{rr.get('physics_rejected', 0)} recordings failed the automated physics gate.",
        "4. **Human demos.** fal `minimax/h3-max-turbo/image-to-video`, 480P, 5 s, `prompt_expansion_mode: disabled`, a "
        "fixed per-attempt seed, generated from the robot-free first/last front renders (`first.png`/`last.png`) with a "
        "v5-style prompt naming the robot's step order. Settings, request id, seed and cost are in `source.json`.",
        f"5. **Human-pair review**: an Opus judge (task adherence and physics >= 4) then an independent Opus verifier that "
        f"tries to refute each accept; both bind to the video hash. Store-wide: {hr.get('generated', 0)} clips generated, "
        f"{hr.get('accept', 0)} accepted / {human_n} reviewed ({pct(hr.get('accept', 0), human_n)}). Rejected attempts are "
        "kept outside this release (their reasons are listed under `other_attempts` in a pair's `review.json` when the "
        "pair was accepted on a retry).",
        f"6. **Spend** (fal ledger, all attempts incl. rejected): ${sp.get('committed_usd', 0):.2f} committed over "
        f"{sp.get('requests', 0)} requests (cap ${sp.get('cap_usd', 120):.0f}; unresolved ${sp.get('unresolved_usd', 0):.2f}); "
        f"{hr.get('generated', 0) - human_n} generated clips were still awaiting review when this card was written, so "
        f"spend per accepted pair (currently ${(sp.get('committed_usd', 0) / accepted_all) if accepted_all else 0:.3f}) is an upper bound. "
        "Price: $0.015/s promotional 480P rate (costs estimated from output duration).", "",
        "## Layout", "", "```",
        "index.json                         tasks and episodes (media paths relative to episodes/<task>/<episode>/)",
        "inventory.json                     sha256 and size of every pair file",
        "episodes/<task>/episode_XXX/",
        "  human.mp4                        accepted generated human demo (H.264, 640x480, 24 fps, ~5 s, no audio)",
        "  robot_front.mp4, robot_wrist.mp4 robot views (H.264, 640x480, 30 fps)",
        "  robot_data.parquet               joint action/state (LeRobot degrees, gripper 0-100) and EE action/state (real-data FK)",
        "  first.png, last.png              robot-free endpoints the human clip was generated from",
        "  episode.json                     visual config, object poses, distractors, oracle log",
        "  physics.json, policy.json        physics QA result and thresholds",
        "  review.json                      physics QA, robot review, human review, verifier",
        "  source.json                      seed, family, hashes, fal request id / seed / settings / cost",
        "  thumb.jpg, robot_thumb.jpg",
        "lerobot/<task>/episode_XXX/        single-episode LeRobot v3 dataset (AV1 front + wrist, 30 fps)",
        "```", "",
        "**LeRobot data is per episode** (byte-identical to the hash-verified recording) so the release can grow without "
        "rewriting published files. To build one dataset per task:", "",
        "```python",
        "from pathlib import Path",
        "from lerobot.datasets.aggregate import aggregate_datasets",
        "roots = sorted(Path('lerobot/<task>').glob('episode_*'))",
        "aggregate_datasets(repo_ids=[f'sim_train_v1/{r.name}' for r in roots], roots=roots,",
        "                   aggr_repo_id='sim_train_v1/<task>', aggr_root=Path('merged/<task>'))",
        "```", "",
        "(Pass `video_backend='pyav'` to `LeRobotDataset` if torchcodec cannot load your ffmpeg.) "
        "For closed-loop evaluation, rebuild a task with the `sim.train.tasks` catalog and reset with the pair's `seed` "
        "(the frozen visual config is in `episode.json`).", "",
        "## Held-out validation", "",
        f"The five validation tasks ({', '.join(VAL_TASKS)}) and their scenes, seeds and human clips are excluded from this "
        "release. Several training families (container insertion, placing on a surface, stacking) overlap with the "
        "validation tasks at the family level; separation is by task definition, not by manipulation family.", "",
        "## Realism limits", "",
        "- Rendered MuJoCo scenes with simplified assets: LIBERO objects scaled 0.5x, primitive blocks/mats, scanned meshes "
        "collided as a single convex hull (holes and concavities filled); some thin LIBERO meshes were solidified and the "
        "basket rebuilt. Masses/densities are approximate (YCB masses are set by hand).",
        "- Robot motion is a scripted planner with privileged state, not teleoperation; motion is smooth but stylised.",
        "- Human videos are generated, not recorded: plausibility was checked by model reviewers, not measured. The human "
        "clip (~5 s) is much shorter than the robot episode.",
        "- Visual reviewers cannot certify physics; the numerical gate in `physics.json` is the physics evidence.", "",
        "## Licences and attribution", "",
        "- Robot data, renders and metadata: produced for this project (see repository licence).",
        "- SO-101 / so101-nexus simulation (Apache-2.0); LIBERO object assets (MIT, LIBERO project).",
        "- YCB object models (CC BY 4.0, Calli et al., The YCB Object and Model Set)"
        + (f": {', '.join(assets.get('ycb', []))}." if assets.get("ycb") else "."),
        "- Google Scanned Objects (CC BY 4.0, Downs et al. 2022, Google Research)"
        + (f": {', '.join(assets.get('gso', []))}." if assets.get("gso") else ".")
        + " Meshes are fetched from the public mirrors `johnsutor/gso-so101-nexus` and `ai-habitat/ycb`.",
        "- Human videos generated with MiniMax H3 Max Turbo via fal; subject to the provider's terms.", "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- package

def package(verify: bool = False, limit: int | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    store = EpisodeStore(STORE_ROOT)
    old = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() else {"episodes": []}
    inventory = json.loads((OUT / "inventory.json").read_text()) if (OUT / "inventory.json").exists() else {}
    episodes = list(old["episodes"])
    have = {(e["task"], e["seed"]) for e in episodes}

    # Existing pairs: never rewritten; check presence and sizes (hashes with --verify).
    for e in episodes:
        rows = {k: v for k, v in inventory.items() if k.startswith((f"episodes/{e['id']}/", f"{e['lerobot']}"))}
        if not rows:
            raise ReleaseError(f"{e['id']}: in index.json but not in inventory.json")
        for rel, row in rows.items():
            p = OUT / rel
            if not p.exists() or p.stat().st_size != row["size"] or (verify and sha256_file(p) != row["sha256"]):
                raise ReleaseError(f"published file changed or missing: {rel}")

    # Leftovers of an interrupted build are not published; remove them before building.
    for tmp in list(OUT.glob("episodes/*/.episode_*.tmp")) + list(OUT.glob("lerobot/*/.episode_*.tmp")):
        shutil.rmtree(tmp)
    indexed = {e["id"] for e in episodes}
    for d in OUT.glob("episodes/*/episode_*"):
        if f"{d.parent.name}/{d.name}" not in indexed:
            print(f"removing unindexed partial pair {d}", file=sys.stderr)
            shutil.rmtree(d)
            shutil.rmtree(OUT / "lerobot" / d.parent.name / d.name, ignore_errors=True)

    new = sorted((k for k in accepted_keys() if (k.task, k.seed) not in have), key=lambda k: (k.task, k.seed))
    if limit is not None:
        new = new[:limit]
    next_number = collections.Counter()
    for e in episodes:
        next_number[e["task"]] = max(next_number[e["task"]], e["episode"] + 1)
    added, refused = [], {}
    for key in new:
        if key.task in VAL_TASKS:
            refused[f"{key.task}/{key.seed}"] = "validation task"
            continue
        try:
            entry, rows = build_pair(store, key, next_number[key.task])
        except Exception as exc:  # report and continue; nothing partial is indexed
            refused[f"{key.task}/episode_{key.seed}"] = f"{type(exc).__name__}: {exc}"
            continue
        next_number[key.task] += 1
        episodes.append(entry)
        inventory.update(rows)
        added.append(entry["id"])
        # Persist after every pair so an interruption never orphans a finished pair.
        _write_if_changed(OUT / "inventory.json", json.dumps(dict(sorted(inventory.items())), indent=0) + "\n")
        _write_index(episodes, old, partial=True)
        print(f"+ {entry['id']} (seed {key.seed})", flush=True)

    index = _write_index(episodes, old)
    stats = store_stats()
    _write_if_changed(OUT / "README.md", readme(index, stats))
    return {"added": added, "refused": refused, "episodes_total": index["episodes_total"],
            "tasks_total": index["tasks_total"], "families_total": index["families_total"]}


def _write_index(episodes: list[dict], old: dict, partial: bool = False) -> dict:
    from datetime import datetime, timezone
    episodes = sorted(episodes, key=lambda e: (e["task"], e["episode"]))
    by_task = collections.defaultdict(list)
    for e in episodes:
        by_task[e["task"]].append(e)
    tasks = [{"task": t, "family": eps[0]["family"], "instruction": eps[0]["instruction"], "episodes": len(eps),
              "views": list(CAMS), "instructions": sorted({e["instruction"] for e in eps})}
             for t, eps in sorted(by_task.items())]
    body = {"name": "SO-101 simulated training pairs", "version": "v1", "split": "train", "simulated": True,
            "base_url": BASE_URL, "episodes_total": len(episodes), "tasks_total": len(tasks),
            "families_total": len({e["family"] for e in episodes}),
            "robot_minutes": round(sum(e["robot_duration_s"] for e in episodes) / 60, 1),
            "tasks": tasks, "episodes": episodes}
    same = old.get("episodes") == episodes and {k: v for k, v in old.items() if k != "updated"} == {**body}
    body = {**body, "updated": old.get("updated") if same and old.get("updated") else
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    ordered = {k: body[k] for k in ("name", "version", "split", "simulated", "updated", "base_url", "episodes_total",
                                    "tasks_total", "families_total", "robot_minutes", "tasks", "episodes")}
    _write_if_changed(OUT / "index.json", json.dumps(ordered, indent=1) + "\n")
    return ordered


# ---------------------------------------------------------------- upload and readback

def _env() -> dict:
    env = dict(os.environ)
    for path in (REPO / ".env", REPO.parents[1] / ".env" if len(REPO.parents) > 1 else None):
        if path and path.exists() and "HF_TOKEN" not in env:
            for line in path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    env["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    if "HF_TOKEN" not in env:
        raise ReleaseError("HF_TOKEN not found in the environment or .env")
    return env


def sync():
    _run_env = _env()
    subprocess.run(["uv", "run", "--with", "huggingface_hub", "hf", "buckets", "sync", str(OUT), f"hf://buckets/{BUCKET}",
                    "--exclude", "*.tmp", "--exclude", ".*", "--format", "quiet"], check=True, env=_run_env, cwd=REPO)


def remote_listing() -> dict[str, int]:
    out = subprocess.run(["uv", "run", "--with", "huggingface_hub", "hf", "buckets", "list", BUCKET, "-R", "--json"],
                         capture_output=True, text=True, check=True, env=_env(), cwd=REPO).stdout
    rows = json.loads(out)
    prefix = BUCKET.split("/", 2)[2] + "/"
    listing = {}
    for r in rows:
        path = r.get("path") or r.get("name") or ""
        if r.get("type", "file") == "file" and path.startswith(prefix):
            listing[path[len(prefix):]] = int(r.get("size") or 0)
    return listing


def _fetch_sha(url: str) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sim-train-readback"}), timeout=120) as r:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def readback(sample: int = 8, extra: list[str] | None = None) -> dict:
    """Compare the remote inventory with the local one and hash a sample of files fetched via the public base_url."""
    inventory = json.loads((OUT / "inventory.json").read_text())
    report = {"remote_missing": [], "remote_size_mismatch": [], "checked": [], "mismatch": []}
    listing = remote_listing()
    for rel, row in inventory.items():
        if rel not in listing:
            report["remote_missing"].append(rel)
        elif listing[rel] and listing[rel] != row["size"]:
            report["remote_size_mismatch"].append(rel)
    report["remote_files"], report["local_files"] = len(listing), len(inventory)
    index = json.loads((OUT / "index.json").read_text())
    rng = random.Random(len(index["episodes"]))
    picks = rng.sample(index["episodes"], min(sample, len(index["episodes"])))
    paths = ["index.json", "README.md", "inventory.json"] + (extra or [])
    for e in picks:
        base = f"episodes/{e['id']}/"
        paths += [base + f for f in ("human.mp4", "robot_front.mp4", "robot_wrist.mp4", "robot_data.parquet",
                                     "review.json", "source.json", "thumb.jpg")]
        paths += [e["lerobot"] + "meta/info.json", e["lerobot"] + "data/chunk-000/file-000.parquet"]
    for rel in dict.fromkeys(paths):
        local = OUT / rel
        want = inventory.get(rel, {}).get("sha256") or sha256_file(local)
        got, size = _fetch_sha(BASE_URL + rel)
        (report["checked"] if got == want and size == local.stat().st_size else report["mismatch"]).append(rel)
    # Decode check of one human and one robot video straight from the public URL.
    e = picks[0]
    for f in ("human.mp4", "robot_front.mp4"):
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                "stream=codec_name,width,height,nb_read_packets", "-of", "json",
                                f"{BASE_URL}episodes/{e['id']}/{f}"], capture_output=True, text=True)
        report.setdefault("decoded", {})[f"{e['id']}/{f}"] = (json.loads(probe.stdout or "{}").get("streams") or [None])[0]
    report["ok"] = not (report["remote_missing"] or report["remote_size_mismatch"] or report["mismatch"])
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publish", action="store_true", help="package, sync to HF, read back and update site/data/sim_index.json")
    ap.add_argument("--verify", action="store_true", help="re-hash every already published pair file")
    ap.add_argument("--limit", type=int, help="add at most N new pairs this run")
    ap.add_argument("--readback-only", action="store_true")
    ap.add_argument("--sample", type=int, default=8, help="episodes sampled for readback")
    a = ap.parse_args(argv)
    if not a.readback_only:
        summary = package(verify=a.verify, limit=a.limit)
        print(json.dumps({k: v for k, v in summary.items()}, indent=1))
        if not a.publish:
            return 1 if summary["refused"] else 0
        sync()
        extra = [f"episodes/{i}/human.mp4" for i in summary["added"][:4]]
    else:
        extra = []
    report = readback(a.sample, extra)
    print(json.dumps({k: (v if k not in ("checked",) else len(v)) for k, v in report.items()}, indent=1))
    if not report["ok"]:
        print("readback FAILED; site index not updated", file=sys.stderr)
        return 2
    shutil.copyfile(OUT / "index.json", SITE_INDEX)
    print(f"site index -> {SITE_INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
