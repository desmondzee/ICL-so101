"""Validate named-target basket tasks before scoring Zero-WAM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from zero_wam.so101_eval import MULTITASK_OBJECTS, make_env


def run(seed: int, task: str, mode: str = "pose", video_path: Path | None = None, native_initial_state: bool = False, camera_variant: str | None = None, image_width: int = 448, image_height: int = 320) -> dict:
    sim_dir = str(Path(__file__).resolve().parents[1] / "sim")
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)
    from record_demos import closing_dirs
    from scripted import PickPlace

    env = make_env(mode, cameras=video_path is not None, width=image_width, height=image_height, task=task)
    if mode == "joint":
        import mujoco

        env._ik_data = mujoco.MjData(env.model)
        from so101_nexus.kinematics import rotvec_to_quat
    obs, _ = env.reset(seed=seed)
    if native_initial_state:
        from zero_wam.so101_eval import prepare_initial_state
        obs = prepare_initial_state(env, obs)
    if camera_variant is not None:
        from zero_wam.so101_eval import configure_cameras, TRIAL_VARIANTS
        obs = configure_cameras(env, TRIAL_VARIANTS[camera_variant], obs)
    frames = [obs["overhead_camera"]] if video_path is not None else []
    target = {"soup": "alphabet_soup", "cheese": "cream_cheese"}.get(task.split("_")[0])
    order = [target] if target else ["alphabet_soup", "cream_cheese"]
    starts = {name: env.object_pos(name).copy() for name in MULTITASK_OBJECTS}
    controller = PickPlace(env, order, closing_dirs(env))
    max_lift = {name: 0.0 for name in MULTITASK_OBJECTS}
    grasp_and_lift = {name: False for name in MULTITASK_OBJECTS}
    first_success_step = None
    steps = 0
    for action in controller.actions():
        command = env._solve_ee_ik(action[:3], rotvec_to_quat(action[3:6]), float(action[6])) if mode == "joint" else action
        obs, _, _, _, _ = env.step(command)
        steps += 1
        if video_path is not None and steps % 4 == 0:
            frames.append(obs["overhead_camera"])
        for name in MULTITASK_OBJECTS:
            lift = float(env.object_pos(name)[2] - starts[name][2])
            max_lift[name] = max(max_lift[name], lift)
            grasp_and_lift[name] |= bool(env.is_grasping(name) and lift >= .05)
        if task == "full_basket":
            step_success = all(env.in_basket(name) for name in order)
        elif task.endswith("_lift"):
            step_success = grasp_and_lift[target]
        else:
            step_success = env.in_basket(target) and float(((env.object_pos(target) - starts[target]) ** 2).sum() ** .5) >= .03
        if step_success and first_success_step is None:
            first_success_step = steps
        if steps > 1600:
            raise RuntimeError(f"Scripted baseline exceeded 1600 controls: {task} seed {seed}")
    contained = {name: env.in_basket(name) for name in MULTITASK_OBJECTS}
    displacement = {name: float(((env.object_pos(name) - starts[name]) ** 2).sum() ** .5) for name in MULTITASK_OBJECTS}
    if task == "full_basket":
        success = all(contained[name] for name in order)
    elif task.endswith("_lift"):
        success = grasp_and_lift[target]
    else:
        success = contained[target] and displacement[target] >= .03
    result = {"seed": seed, "task": task, "mode": mode, "steps": steps, "first_success_step": first_success_step, "success": bool(success), "initial_object_positions_m": {n: starts[n].tolist() for n in MULTITASK_OBJECTS}, "max_lift_m": max_lift, "grasp_and_lift": grasp_and_lift, "in_basket": contained, "final_displacement_m": displacement, "controller_log": controller.log}
    if video_path is not None:
        import imageio.v2 as imageio

        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=12)
        video_path.with_suffix(".json").write_text(json.dumps(result, indent=2, default=bool) + "\n")
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,7,8")
    parser.add_argument("--modes", default="pose,joint")
    parser.add_argument("--output", default="outputs/zero_wam/so101/multitask/scripted_baseline.json")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    modes = args.modes.split(",")
    tasks = ("soup_lift", "cheese_lift", "soup_place", "cheese_place", "full_basket")
    rows = [run(seed, task, mode) for mode in modes for seed in seeds for task in tasks]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, default=bool) + "\n")
    for row in rows:
        print(row["mode"], row["seed"], row["task"], row["steps"], row["first_success_step"], row["success"], flush=True)
    if not all(row["success"] for row in rows):
        raise SystemExit("Some scripted controls failed; inspect the report before scoring model trials")


if __name__ == "__main__":
    main()
