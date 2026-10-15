"""Replay frozen parquet commands through the deployed TrainSim control path."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio

from scripts.run_so101_batched_train_replay import DATA, TrainSim, load_cohort
from zero_wam.so101_closed_loop import FRONT, WRIST
from zero_wam.prompt_sweep import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/zero_wam/oracle_command_replay_scenev2"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot(sim):
    env = sim.env
    return {
        "state": env.lerobot_joints().astype(float).tolist(),
        "tcp": env._get_tcp_pose()[:3].astype(float).tolist(),
        "objects": {n: {"xyz": env.object_pos(n).astype(float).tolist(),
                        "grasped": bool(env.is_grasping(n))} for n in sim.task_objects},
        "goals": {g.obj: {"xyz": env.goal_target(g).astype(float).tolist(),
                           "error_xyz": (env.object_pos(g.obj)-env.goal_target(g)).astype(float).tolist(),
                           "holds": bool(env.goal_holds(g))} for g in env.goals},
        "success": bool(env.success()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    atomic_json(OUT/"manifest.json", {
        "protocol": "original parquet commands; pre-control row alignment; deployed TrainSim.step",
        "source_sha256": sha(Path(__file__)), "episodes": cohort,
        "control_hz": 30, "early_termination": False,
    })
    results = []
    for ep in cohort:
        source = DATA/"episodes"/ep["id"]
        directory = OUT/ep["id"]
        if directory.exists():
            raise RuntimeError(f"refusing overwrite: {directory}")
        directory.mkdir(parents=True)
        table = pd.read_parquet(source/"robot_data.parquet")
        original = json.loads((source/"episode.json").read_text())
        actions = np.stack(table.action)
        expected = np.stack(table["observation.state"])
        assert actions.shape == expected.shape == (ep["robot_frames"], 6)
        assert np.isfinite(actions).all()
        np.testing.assert_allclose(np.diff(table.timestamp), 1/30, atol=2e-6)
        sim = TrainSim(ep["task"], ep["seed"], ep["visual_config"],
                       recorded_poses=ep["object_poses_start"])
        captures = [cv2.VideoCapture(str(source/f"robot_{cam}.mp4")) for cam in ("front","wrist")]
        telemetry, errors, first_success = [], [], None
        try:
            with imageio.get_writer(directory/"rollout.mp4", fps=30, codec="libx264",
                                    quality=7, macro_block_size=8) as writer, \
                 imageio.get_writer(directory/"comparison.mp4", fps=30, codec="libx264",
                                    quality=7, macro_block_size=8) as comparison:
                for i, command in enumerate(actions):
                    before = snapshot(sim)
                    errors.append(np.asarray(before["state"])-expected[i])
                    obs = sim.observe()
                    live = np.concatenate([obs[FRONT],obs[WRIST]],axis=1)
                    writer.append_data(live)
                    frames = []
                    for cap in captures:
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(f"recorded video shorter than parquet at {i}")
                        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    reference = np.concatenate(frames,axis=1)
                    assert reference.shape == live.shape
                    pair = np.concatenate([reference,live],axis=0)
                    cv2.putText(pair,f"Recorded pre-control {i} | {i/30:.2f}s",(8,22),
                                cv2.FONT_HERSHEY_SIMPLEX,.55,(255,60,60),1)
                    cv2.putText(pair,"Replay, same command index",(8,live.shape[0]+22),
                                cv2.FONT_HERSHEY_SIMPLEX,.55,(255,60,60),1)
                    comparison.append_data(pair)
                    sim.step(command)
                    after = snapshot(sim)
                    if after["success"] and first_success is None:
                        first_success = i+1
                    telemetry.append({"control":i+1,"command":command.astype(float).tolist(),
                                      "before":before,"after":after})
                obs = sim.observe()
                writer.append_data(np.concatenate([obs[FRONT],obs[WRIST]],axis=1))
            error = np.asarray(errors)
            final = snapshot(sim)
            result = {
                "id":ep["id"],"controls":len(actions),"scene_replay":sim.scene_replay,
                "original_final_success":original["final_task_success"],
                "success":first_success is not None,"first_success_control":first_success,
                "final_success":final["success"],"final":final,
                "pre_state_arm_mae_deg":float(np.abs(error[:,:5]).mean()),
                "pre_state_arm_max_abs_deg":float(np.abs(error[:,:5]).max()),
                "pre_state_gripper_mae_points":float(np.abs(error[:,5]).mean()),
                "pre_state_gripper_max_abs_points":float(np.abs(error[:,5]).max()),
                "final_object_xyz_error_m":{n:float(np.linalg.norm(
                    sim.env.object_pos(n)-np.asarray(original["object_poses_end"][n][:3])))
                    for n in sim.task_objects},
                "grasp_controls":{n:sum(t["after"]["objects"][n]["grasped"] for t in telemetry)
                                  for n in sim.task_objects},
                "source_parquet_sha256":sha(source/"robot_data.parquet"),
                "source_episode_sha256":sha(source/"episode.json"),
            }
            with (directory/"telemetry.jsonl").open("w") as sink:
                for t in telemetry:
                    sink.write(json.dumps(t)+"\n")
            np.save(directory/"pre_state_errors.npy",error)
            atomic_json(directory/"result.json",result)
            results.append(result)
            atomic_json(OUT/"summary.json",{"episodes":results})
            print(json.dumps({k:result[k] for k in (
                "id","controls","success","final_success","pre_state_arm_mae_deg",
                "pre_state_arm_max_abs_deg","final_object_xyz_error_m")}),flush=True)
        finally:
            for cap in captures:
                cap.release()
            sim.env.close()


if __name__ == "__main__":
    main()
