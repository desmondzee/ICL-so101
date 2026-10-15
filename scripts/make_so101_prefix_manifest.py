"""Freeze handovers selected after successful original-command simulator replay."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT/"outputs/zero_wam/oracle_command_replay_scenev2"
OUT = ROOT/"docs/results/zero_wam_oracle_prefix"


def main():
    summary = json.loads((ORACLE/"summary.json").read_text())
    assert len(summary["episodes"]) == 6
    assert all(r["final_success"] and r["original_final_success"] for r in summary["episodes"])
    plans = {"prefix_manifest.json": [], "smoke_manifest.json": []}
    for ep, boundaries in (("block_in_bowl/episode_000", [0,96,176,288]),
                           ("block_in_bowl/episode_001", [0,152,232,368])):
        directory = ORACLE/ep
        result = json.loads((directory/"result.json").read_text())
        telemetry = [json.loads(x) for x in (directory/"telemetry.jsonl").read_text().splitlines()]
        for filename in plans:
            if filename.startswith("smoke") and ep.endswith("001"):
                continue
            bs = [0,8,16,24] if filename.startswith("smoke") else boundaries
            for stage, b in zip(("zero", "pre_closure", "held_after_lift", "held_above_bowl"), bs):
                state = telemetry[b-1]["after"] if b else telemetry[0]["before"]
                if filename.startswith("prefix") and stage.startswith("held"):
                    assert state["objects"]["block"]["grasped"]
                    assert state["objects"]["block"]["xyz"][2]-telemetry[0]["before"]["objects"]["block"]["xyz"][2] > .02
                assert not state["success"]
                plans[filename].append({
                    "source_episode_id": ep, "prefix_controls": b,
                    "prefix_stage": "smoke" if filename.startswith("smoke") else stage,
                    "source_parquet_sha256": result["source_parquet_sha256"],
                    "expected_handover": state,
                })
    for filename, conditions in plans.items():
        path = OUT/filename
        if path.exists():
            raise RuntimeError(f"refusing overwrite {path}")
        path.write_text(json.dumps({
            "schema":1, "task":"block_in_bowl", "conditions":conditions,
            "oracle_summary_sha256":hashlib.sha256((ORACLE/"summary.json").read_bytes()).hexdigest(),
            "schedule":"F2 raw, execute8, CFG5, full paired ICL, noEMA; original global2x horizon",
            "prefix_method":"physically execute recorded prefix, infer/discard to preserve noise schedule, cache commanded history",
        },indent=2)+"\n")
        print(path)


if __name__ == "__main__":
    main()
