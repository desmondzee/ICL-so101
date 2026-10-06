"""Reproducible, non-rendered strict screening; no report means no admission.

  python -m sim.train.tasks.qualify --tasks block_in_bowl            # 50 prescribed seeds
  python -m sim.train.tasks.qualify --family pilot_blocks --jobs 4    # every task in a family module
  python -m sim.train.tasks.qualify --tasks block_in_bowl --seed-start 9000 --count 5 --output /tmp/cal

Calibration (``--seed-start``) uses separate seeds/output and never qualifies a task. Each task's
evidence is keyed by a source hash over the shared kit files plus that task's own family module,
so editing one family never invalidates another family's qualification (editing the kit does).
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from sim.train.physics import audit_trajectory
from sim.train.store import atomic_write_json
from sim.train.telemetry import TelemetryCollector
from sim.train.variation import episode_config_hash
from .catalog import DISCOVERY_ERRORS, TRAIN_TASKS, family_tasks
from .schema import TaskDefinition, validate_catalog

ROOT = Path(__file__).resolve().parents[3]
REQUIRED_RATE = 0.95
FRAME_BUDGET = 3600
KIT_FILES = (*(f"sim/val/{n}" for n in ("env.py", "scene.py", "oracle.py")),
             *(f"sim/train/{n}" for n in ("physics.py", "telemetry.py", "variation.py")),
             *(f"sim/train/tasks/{n}" for n in ("base.py", "schema.py", "qualify.py", "assets.py")))


def _task(task):
    return TRAIN_TASKS[task] if isinstance(task, str) else task


def implementation_hash(task: TaskDefinition | str | None = None) -> str:
    """Source identity of the qualification evidence: shared kit files, plus the task's own modules."""
    paths = [ROOT / p for p in KIT_FILES]
    if task is not None:
        paths += [Path(p) for p in _task(task).source_files()]
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def default_validation_index() -> Path:
    """data/so101_sim_val_v2/index.json in this checkout, else in the main checkout (worktrees)."""
    local = ROOT / "data/so101_sim_val_v2/index.json"
    if local.is_file():
        return local
    try:
        common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], capture_output=True,
                                text=True, check=True).stdout.strip()
        main = (ROOT / common).resolve().parent / "data/so101_sim_val_v2/index.json"
        if main.is_file():
            return main
    except (OSError, subprocess.CalledProcessError):
        pass
    raise FileNotFoundError("validation index not found; pass --validation-index")


def qualification_summary(task, rows):
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("invalid or duplicate qualification seed")
    if any(type(row["accepted"]) is not bool for row in rows):
        raise ValueError("qualification result must be a boolean")
    expected = tuple(task.qualification_seeds)
    complete = len(expected) == 50 and set(seeds) == set(expected)
    accepted = sum(row["accepted"] for row in rows)
    rate = accepted / len(rows) if rows else 0.0
    return {"task": task.name, "qualified": complete and rate >= REQUIRED_RATE,
            "required_seeds": list(expected), "screened_seeds": seeds, "complete": complete,
            "strict_successes": accepted, "strict_success_rate": rate, "required_rate": REQUIRED_RATE,
            "failure_reasons": dict(sorted(Counter(reason for row in rows for reason in row["reasons"]).items())),
            "episodes": rows}


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_evidence(row, directory):
    """Verify retained accepted evidence, never trust a summary flag alone."""
    if row["accepted"]:
        expected = {"telemetry.npz", "physics.json", "policy.json", "episode.json"}
        if set(row.get("artifacts", {})) != expected:
            raise ValueError(f"missing qualification evidence: {directory}")
        for name, digest in row["artifacts"].items():
            if not (directory / name).is_file() or _file_hash(directory / name) != digest:
                raise ValueError(f"corrupt qualification evidence: {directory / name}")
        physics = json.loads((directory / "physics.json").read_text())
        if (physics.get("accepted") is not True or not physics.get("checks")
                or any(ok is not True for ok in physics["checks"].values())
                or physics.get("config_hash") != row.get("config_hash")):
            raise ValueError(f"inconsistent qualification evidence: {directory}")


def require_qualified_task(name, root):
    """Recorder admission boundary; candidate registration is not qualification."""
    task = TRAIN_TASKS[name]
    path = Path(root) / f"{name}.json"
    report = json.loads(path.read_text())
    recomputed = qualification_summary(task, report["episodes"])
    if (report.get("source_hash") != implementation_hash(task) or not report.get("overlap_report", {}).get("accepted")
            or not recomputed["qualified"] or not report.get("qualified")):
        raise ValueError(f"task lacks current complete strict qualification: {name}")
    for row in report["episodes"]:
        directory = Path(root) / name / f"seed_{row['seed']}"
        _verify_evidence(row, directory)
        result = directory / "result.json"
        if not result.is_file() or json.loads(result.read_text()) != row or row.get("source_hash") != report["source_hash"]:
            raise ValueError(f"inconsistent qualification evidence: {directory}")
    return task


def rollout(task, seed, collector_factory=None):
    """Reset with the task's visual variation and run its oracle; returns (env, oracle, config, poses, collector).
    The caller closes ``env``."""
    config = task.sample_visual_config(seed)
    env_class, oracle_class = task.load_classes()
    env = env_class(render_images=False, visual_config=config)
    try:
        env.reset(seed=seed)
        poses = env.object_poses()
        identity = episode_config_hash(config, poses)
        collector = TelemetryCollector(env, config_hash=identity, visual_config_hash=config.config_hash)
        oracle = oracle_class(env, np.random.default_rng(seed))
        for frame, action in enumerate(oracle.actions()):
            if frame >= FRAME_BUDGET:
                raise RuntimeError("oracle exceeded 120-second frame budget")
            env.step(action, substep_observer=collector)
    except BaseException:
        env.close()
        raise
    return env, oracle, config, poses, collector


def screen_seed(task, seed, directory, *, source_hash):
    task = _task(task)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        row = json.loads(result_path.read_text())
        if row.get("source_hash") != source_hash or row.get("seed") != seed:
            raise ValueError(f"stale or conflicting evidence: {directory}; choose a new output directory")
        _verify_evidence(row, directory)
        return row
    env = None
    config = task.sample_visual_config(seed)
    row = {"seed": seed, "source_hash": source_hash, "accepted": False, "reasons": [],
           "visual_config_hash": config.config_hash, "evidence": str(directory)}
    try:
        env, oracle, config, poses, collector = rollout(task, seed)
        policy = task.physics_policy(env)
        telemetry = collector.finish()
        telemetry.save(directory / "telemetry.npz")
        report = audit_trajectory(telemetry, policy)
        identity = telemetry.manifest["config_hash"]
        atomic_write_json(directory / "physics.json", json.loads(report.to_json()), overwrite=False)
        atomic_write_json(directory / "policy.json", policy.to_dict(), overwrite=False)
        atomic_write_json(directory / "episode.json", {"task": asdict(task), "seed": seed,
                          "visual_config": config.to_dict(), "object_poses": poses,
                          "config_hash": identity, "visual_config_hash": config.config_hash}, overwrite=False)
        row["artifacts"] = {name: _file_hash(directory / name) for name in
                            ("telemetry.npz", "physics.json", "policy.json", "episode.json")}
        row.update(accepted=report.accepted, reasons=[name for name, ok in report.checks.items() if not ok],
                   config_hash=identity, maxima=report.maxima, frames=collector.frame,
                   final_task_success=bool(env.success()), oracle_log=oracle.log,
                   limited_frames=getattr(oracle, "limited", None),
                   violation_counts=dict(Counter(v["check"] for v in report.violations)))
        pairs = Counter(tuple(sorted((telemetry.manifest["geom_bodies"][g1], telemetry.manifest["geom_bodies"][g2])))
                        for g1, g2 in telemetry.arrays["contact_geom"])
        row["contact_pairs"] = [{"bodies": list(pair), "samples": count} for pair, count in sorted(pairs.items())]
    except Exception as exc:
        row.update(reasons=[f"runtime_error:{type(exc).__name__}"], error=str(exc))
    finally:
        if env is not None:
            env.close()
    atomic_write_json(result_path, json.loads(json.dumps(row, default=float)), overwrite=False)
    return json.loads(result_path.read_text())


def _screen_job(args):
    name, seed, directory, source_hash = args
    return screen_seed(TRAIN_TASKS[name], seed, Path(directory), source_hash=source_hash)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation-index", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/so101_sim_train_v1/qualification")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TRAIN_TASKS))
    parser.add_argument("--family", nargs="+", help="family module name(s) under sim/train/tasks/families")
    parser.add_argument("--seed-start", type=int, help="calibration seeds (never qualifies a task)")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    args = parser.parse_args(argv)
    if DISCOVERY_ERRORS:
        print(json.dumps({"discovery_errors": DISCOVERY_ERRORS}, indent=1))
    names = list(args.tasks or ())
    for family in args.family or ():
        found = family_tasks(family)
        if not found:
            parser.error(f"no tasks discovered in family module {family!r}")
        names += [n for n in found if n not in names]
    if not names:
        names = list(TRAIN_TASKS)
    if args.count < 1 or args.count > 50 or (args.seed_start is not None and args.seed_start < 0):
        parser.error("use 1-50 nonnegative seeds")
    index_path = args.validation_index or default_validation_index()
    index = json.loads(index_path.read_text())
    tasks = [TRAIN_TASKS[name] for name in names]
    overlap = validate_catalog(list(TRAIN_TASKS.values()), index)
    if not overlap["accepted"]:
        raise ValueError(f"held-out overlap or catalog conflict: {overlap['violations']}")
    validation_seeds = {row["seed"] for row in index["episodes"]}
    jobs, sources = [], {}
    for task in tasks:
        seeds = (tuple(range(args.seed_start, args.seed_start + args.count)) if args.seed_start is not None
                 else task.qualification_seeds[:args.count])
        if set(seeds) & validation_seeds:
            raise ValueError("screen seeds overlap validation")
        sources[task.name] = implementation_hash(task)
        jobs += [(task.name, seed, str(args.output / task.name / f"seed_{seed}"), sources[task.name]) for seed in seeds]
    rows = {task.name: [] for task in tasks}

    def record(job, row):
        name = job[0]
        rows[name].append(row)
        report = qualification_summary(TRAIN_TASKS[name], sorted(rows[name], key=lambda r: r["seed"]))
        report.update(source_hash=sources[name], overlap_report=overlap, rendered=False,
                      calibration=args.seed_start is not None)
        atomic_write_json(args.output / f"{name}.json", report)
        print(json.dumps({"task": name, "seed": row["seed"], "accepted": row["accepted"], "reasons": row["reasons"],
                          "rate": round(report["strict_success_rate"], 3), "n": len(rows[name])}), flush=True)
        return report

    reports = {}
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            for job, row in zip(jobs, pool.map(_screen_job, jobs)):
                reports[job[0]] = record(job, row)
    else:
        for job in jobs:
            reports[job[0]] = record(job, _screen_job(job))
    for name, report in reports.items():
        print(json.dumps({"task": name, "qualified": report["qualified"], "strict_successes": report["strict_successes"],
                          "screened": len(report["episodes"]), "failure_reasons": report["failure_reasons"]}))
    return 0 if all(r["qualified"] for r in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
