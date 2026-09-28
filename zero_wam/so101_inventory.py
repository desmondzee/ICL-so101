"""Rebuild the SO-101 model-trial inventory from saved episode artifacts."""

import argparse
import json
from collections import Counter
from pathlib import Path


def checkpoint_load_status(manifest: dict) -> str:
    if manifest.get("checkpoint") != "pretrain":
        return "not audited by pretrain loader check"
    runtime = manifest.get("worker_runtime", {})
    if runtime.get("use_icl_model") is False:
        return "invalid: wrong transformer architecture"
    evidence = runtime.get("checkpoint_load", {})
    if evidence.get("class") == "WanICLTransformer3DModel" and "loading_info" in evidence:
        if not any(evidence["loading_info"].values()):
            return "exact checkpoint load verified"
        return "invalid: nonempty checkpoint loading diagnostics"
    return "unverified checkpoint load"


def inventory(base: Path) -> dict:
    trials = []
    for root in sorted(base.glob("multitask*")):
        if not root.is_dir():
            continue
        for summary_path in sorted(root.rglob("summary.json")):
            summary = json.loads(summary_path.read_text())
            directory = summary_path.parent
            manifest_path = directory / "trial.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            cache_path = directory / "cache_runtime.jsonl"
            cache_reports = [json.loads(line) for line in cache_path.read_text().splitlines()] if cache_path.exists() else []
            feedback_status = "unverified runtime feedback"
            if manifest.get("checkpoint") == "pretrain" and root.name == "multitask_pretrain_exact_load":
                feedback_status = "invalid: predicted-target cache before feedback fix"
            elif cache_reports and all(r.get("action_history_source") == "executed_model_actions" for r in cache_reports):
                feedback_status = "executed history verified in recorded cache calls"
            trials.append({
                "root": root.name,
                "path": str(directory),
                "checkpoint": manifest.get("checkpoint"),
                "checkpoint_load_status": checkpoint_load_status(manifest),
                "feedback_status": feedback_status,
                "task": summary["task"],
                "variant": summary["variant"],
                "mode": summary["mode"],
                "success": summary["success"],
                "steps": summary["steps"],
                "grasp_steps": sum(o["grasp_steps"] for o in summary["objects"].values()),
                "missing_files": [name for name in (
                    "trial.json", "actions.jsonl", "overhead.mp4", "overhead.contact.png"
                ) if not (directory / name).exists()],
            })
    main = [trial for trial in trials if trial["root"] != "multitask_repeats"]
    return {
        "scope": "Completed summaries under multitask*; stochastic repeats reported separately; pending trials excluded",
        "main_trials": len(main),
        "repeat_trials": len(trials) - len(main),
        "main_successes": sum(trial["success"] for trial in main),
        "main_grasp_steps": sum(trial["grasp_steps"] for trial in main),
        "by_checkpoint_load_status": dict(Counter(trial["checkpoint_load_status"] for trial in main)),
        "by_root": dict(Counter(trial["root"] for trial in trials)),
        "by_checkpoint_mode": dict(Counter(
            f"{trial['checkpoint']}/{trial['mode']}" for trial in main
        )),
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/zero_wam/so101"))
    args = parser.parse_args()
    result = inventory(args.base)
    output = args.base / "multitask_trial_inventory.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    paired = paired_comparison(args.base)
    (args.base / "full_paired_comparison.json").write_text(json.dumps(paired, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}, indent=2))
    print(f"Full paired episodes completed: {paired['completed']}/{paired['expected']}")


def paired_comparison(base: Path) -> dict:
    """Summarize the fixed paired experiment, preserving unsaved episodes as pending."""
    specs = [
        ("multitask_pose_full_paired", "posttrain", "pose_robotwin_home_visible_zoom_left", "pose"),
        ("multitask_pretrain_valid_full_paired", "pretrain", "pose_full_denoise_task_front_home", "pose"),
        ("multitask_pretrain_valid_full_paired", "pretrain", "joint_full_denoise_task_front_home", "joint"),
        ("multitask_posttrain_joint_full_paired", "posttrain", "joint_three_task_front_home", "joint"),
    ]
    rows = []
    for root, checkpoint, variant, mode in specs:
        for task, target in (("soup_lift", "alphabet_soup"), ("cheese_lift", "cream_cheese")):
            for seed in (1, 7, 8):
                directory = base / root / task / checkpoint / variant / f"seed{seed}_steps400"
                path = directory / "summary.json"
                validity = ("untrained joint block diagnostic" if checkpoint == "posttrain" and mode == "joint"
                            else "empirical SO normalization; require exact loading and executed-cache evidence" if checkpoint == "pretrain"
                            else "native RoboTwin pose adapter")
                row = dict(checkpoint=checkpoint, mode=mode, task=task, seed=seed,
                           path=str(directory), status="completed" if path.exists() else "pending",
                           schema_validity=validity)
                if path.exists():
                    summary = json.loads(path.read_text())
                    obj = summary["objects"][target]
                    row.update(success=summary["success"], steps=summary["steps"],
                               grasp_steps=obj["grasp_steps"], target_min_distance_m=obj["min_tcp_distance_m"],
                               target_max_lift_m=obj["max_lift_m"],
                               commanded_position_error_m=summary.get("mean_commanded_position_error_m"),
                               requested_position_error_m=summary.get("mean_requested_position_error_m"))
                rows.append(row)
    return dict(scope="24 paired 400-control episodes using corrected pretrain root; historical wrong-loader pretrain excluded; pending means no saved summary, not verified running individual episodes",
                completed=sum(row["status"] == "completed" for row in rows), expected=len(rows), rows=rows)


if __name__ == "__main__":
    main()
