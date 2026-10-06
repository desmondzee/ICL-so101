"""Reproducible, non-rendered strict screening; no report means no admission.

Example (the validation index is read only):
  python -m sim.train.tasks.qualify --validation-index /path/to/sim_val_v2/index.json

Calibration uses separate seeds/output: --seed-start 9000 --count 3. Such a
report never qualifies a task. Fixes invalidate previous evidence by source hash.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from sim.train.physics import audit_trajectory
from sim.train.store import atomic_write_json
from sim.train.telemetry import TelemetryCollector
from sim.train.variation import episode_config_hash, sample_visual_config
from .catalog import TRAIN_TASKS
from .schema import validate_catalog


def implementation_hash():
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "sim/train/tasks").glob("*.py"))
    paths += [root / "sim" / group / name for group, names in
              (("val", ("env.py", "scene.py", "oracle.py")),
               ("train", ("physics.py", "telemetry.py", "variation.py"))) for name in names]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
    return {"task": task.name, "qualified": complete and rate >= 0.95,
            "required_seeds": list(expected), "screened_seeds": seeds, "complete": complete,
            "strict_successes": accepted, "strict_success_rate": rate, "required_rate": 0.95,
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
    if (report.get("source_hash") != implementation_hash() or not report.get("overlap_report", {}).get("accepted")
            or not recomputed["qualified"] or not report.get("qualified")):
        raise ValueError(f"task lacks current complete strict qualification: {name}")
    for row in report["episodes"]:
        directory = Path(root) / name / f"seed_{row['seed']}"
        _verify_evidence(row, directory)
        result = directory / "result.json"
        if not result.is_file() or json.loads(result.read_text()) != row or row.get("source_hash") != report["source_hash"]:
            raise ValueError(f"inconsistent qualification evidence: {directory}")
    return task


def screen_seed(task, seed, directory, *, source_hash):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        row = json.loads(result_path.read_text())
        if row.get("source_hash") != source_hash or row.get("seed") != seed:
            raise ValueError(f"stale or conflicting evidence: {directory}; choose a new output directory")
        _verify_evidence(row, directory)
        return row
    env = None
    config = sample_visual_config(task.name, seed)
    row = {"seed": seed, "source_hash": source_hash, "accepted": False, "reasons": [],
           "visual_config_hash": config.config_hash, "evidence": str(directory)}
    try:
        env_class, oracle_class = task.load_classes()
        env = env_class(render_images=False, visual_config=config)
        env.reset(seed=seed)
        poses = env.object_poses()
        identity = episode_config_hash(config, poses)
        policy = task.physics_policy(env)
        collector = TelemetryCollector(env, config_hash=identity, visual_config_hash=config.config_hash)
        oracle = oracle_class(env, np.random.default_rng(seed))
        for frame, action in enumerate(oracle.actions()):
            if frame >= 3600:
                raise RuntimeError("oracle exceeded 120-second frame budget")
            env.step(action, substep_observer=collector)
        telemetry = collector.finish()
        telemetry.save(directory / "telemetry.npz")
        report = audit_trajectory(telemetry, policy)
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
                   violation_counts=dict(Counter(v["check"] for v in report.violations)))
        pairs = Counter(tuple(sorted((telemetry.manifest["geom_bodies"][g1], telemetry.manifest["geom_bodies"][g2])))
                        for g1, g2 in telemetry.arrays["contact_geom"])
        row["contact_pairs"] = [{"bodies": list(pair), "samples": count} for pair, count in sorted(pairs.items())]
    except Exception as exc:
        row.update(reasons=[f"runtime_error:{type(exc).__name__}"], error=str(exc))
    finally:
        if env is not None:
            env.close()
    atomic_write_json(result_path, row, overwrite=False)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/so101_sim_train_v1/qualification"))
    parser.add_argument("--tasks", nargs="+", choices=tuple(TRAIN_TASKS), default=tuple(TRAIN_TASKS))
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    if args.count < 1 or args.count > 50 or (args.seed_start is not None and args.seed_start < 0):
        parser.error("use 1–50 nonnegative seeds")
    index = json.loads(args.validation_index.read_text())
    tasks = [TRAIN_TASKS[name] for name in args.tasks]
    overlap = validate_catalog(tasks, index)
    if not overlap["accepted"]:
        raise ValueError(f"held-out overlap: {overlap}")
    validation_seeds = {row["seed"] for row in index["episodes"]}
    source_hash = implementation_hash()
    all_qualified = True
    for task in tasks:
        seeds = (tuple(range(args.seed_start, args.seed_start + args.count)) if args.seed_start is not None
                 else task.qualification_seeds[:args.count])
        if set(seeds) & validation_seeds:
            raise ValueError("screen seeds overlap validation")
        rows = []
        for seed in seeds:
            row = screen_seed(task, seed, args.output / task.name / f"seed_{seed}", source_hash=source_hash)
            rows.append(row)
            report = qualification_summary(task, rows)
            report.update(source_hash=source_hash, overlap_report=overlap, rendered=False)
            atomic_write_json(args.output / f"{task.name}.json", report)
            print(json.dumps({"task": task.name, "seed": seed, "accepted": row["accepted"],
                              "reasons": row["reasons"], "rate": report["strict_success_rate"]}), flush=True)
        all_qualified &= report["qualified"]
    return 0 if all_qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
