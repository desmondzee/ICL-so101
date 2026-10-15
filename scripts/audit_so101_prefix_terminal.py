"""Re-execute saved commands without inference to inspect terminal predicates."""
from pathlib import Path
import json
import numpy as np
from scripts.run_so101_batched_train_replay import TrainSim, load_cohort
from zero_wam.prompt_sweep import atomic_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"outputs/zero_wam/block_prefix_handover_scenev2_b4"
OUT = ROOT/"docs/results/zero_wam_oracle_prefix/terminal_predicate_audit.json"


def main():
    episodes = {e["id"]:e for e in load_cohort()}
    result = []
    for path in sorted(RUN.glob("*/*/result.json")):
        r = json.loads(path.read_text())
        t = [json.loads(l) for l in path.with_name("telemetry.jsonl").read_text().splitlines()]
        ep = episodes[r["source_episode_id"]]
        sim = TrainSim(ep["task"],ep["seed"],ep["visual_config"],recorded_poses=ep["object_poses_start"])
        try:
            for row in t:
                sim.step(np.asarray(row["commanded_action"],dtype=np.float32))
            final = r["final_physical_state"]
            state_error = float(np.max(np.abs(sim.env.lerobot_joints()-np.asarray(final["state"]))))
            np.testing.assert_allclose(sim.env.lerobot_joints(),final["state"],atol=1e-5,rtol=0)
            goals = {}
            for goal in sim.env.goals:
                np.testing.assert_allclose(sim.env.object_pos(goal.obj),
                    final["objects"][goal.obj]["xyz"],atol=1e-7,rtol=0)
                error = sim.env.object_pos(goal.obj)-sim.env.goal_target(goal)
                goals[goal.obj] = {
                    "position_pass": bool((np.abs(error)<=goal.tolerance).all()),
                    "position_error_xyz_m":error.tolist(),"position_tolerance_m":list(goal.tolerance),
                    "upright_pass": bool(sim.env.upright(goal.obj,goal.upright_cos))
                    if goal.upright_cos is not None else True,
                    "upright_cos":goal.upright_cos,
                    "released_pass": bool(sim.env.released(goal.obj)),
                    "settled_pass": bool(sim.env.settled(goal.obj)),
                    "support_pass": bool(sim.env.supported_by(goal.obj,goal.support)),
                    "support":goal.support, "goal_holds":bool(sim.env.goal_holds(goal)),
                    "object_pose":sim.env.object_poses()[goal.obj],
                }
            assert bool(sim.env.success()) == r["final_state_success"]
            result.append({"id":r["id"],"state_reproduction_max_abs_error":state_error,
                           "goals":goals,"success":bool(sim.env.success())})
            print(json.dumps(result[-1]),flush=True)
        finally:
            sim.env.close()
    assert len(result)==8
    atomic_json(OUT,{"episodes":result})


if __name__ == "__main__":
    main()
