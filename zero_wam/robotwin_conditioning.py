"""Same-scene, competing-goal RoboTwin conditioning experiment.

The stock expert gate and scene generator remain unchanged. Only the evaluation
instruction and success order vary; both orders are recorded for every run.
"""
from pathlib import Path
import json

import numpy as np


PROMPTS = {
    "green_on_red": "Stack the green block on top of the red block.",
    "red_on_green": "Stack the red block on top of the green block.",
    "empty": "",
    "neutral": "Stack the blocks.",
}


def stack_outcomes(red, green, grippers_open):
    red, green = np.asarray(red), np.asarray(green)
    eps = np.array([.025, .025, .012])
    return {
        "green_on_red": bool(grippers_open and np.all(np.abs(green - red - [0, 0, .05]) < eps)),
        "red_on_green": bool(grippers_open and np.all(np.abs(red - green - [0, 0, .05]) < eps)),
    }


def run_condition(model, condition, seed, test_num, save_root, video_path=""):
    """Patch only local task evaluation; restore methods even on failure."""
    root = Path(save_root).resolve()
    video_bytes = Path(video_path).read_bytes() if video_path else None
    mode = "video" if video_bytes else "text"
    prompt = "Stack the blocks." if video_bytes else PROMPTS[condition]
    # Upstream initialization sets RoboTwin's required working directory.
    from evaluation.robotwin import eval_policy_client_openpi as client
    from envs.stack_blocks_two import stack_blocks_two
    from zero_wam.modal_client import run

    if condition not in PROMPTS:
        raise ValueError(condition)
    root.mkdir(parents=True, exist_ok=True)
    methods = {name: getattr(stack_blocks_two, name) for name in
               ("setup_demo", "set_instruction", "check_success", "close_env")}
    reports = []
    original_video_selection = client.select_icl_video

    def setup(self, **kwargs):
        self._conditioning_report = None
        self._conditioning_seed = kwargs.get("seed")
        return methods["setup_demo"](self, **kwargs)

    def instruction(self, instruction=None):
        report = {
            "seed": self._conditioning_seed,
            "condition": condition,
            "prompt": prompt,
            "initial_red_position": self.block1.get_pose().p.tolist(),
            "initial_green_position": self.block2.get_pose().p.tolist(),
            "ever": {"green_on_red": False, "red_on_green": False},
            "scoring_goal": "green_on_red" if condition in {"empty", "neutral"} else condition,
        }
        self._conditioning_report = report
        reports.append(report)
        return methods["set_instruction"](self, instruction=prompt)

    def success(self):
        report = getattr(self, "_conditioning_report", None)
        if report is None:
            # Preserve stock expert feasibility gate and hence paired seeds.
            return methods["check_success"](self)
        red, green = self.block1.get_pose().p, self.block2.get_pose().p
        outcomes = stack_outcomes(red, green, self.is_left_gripper_open() and self.is_right_gripper_open())
        report["final_red_position"] = red.tolist()
        report["final_green_position"] = green.tolist()
        report["final_outcomes"] = outcomes
        for name, value in outcomes.items():
            report["ever"][name] |= value
        return outcomes[report["scoring_goal"]]

    def close(self, *args, **kwargs):
        if getattr(self, "_conditioning_report", None) is not None:
            success(self)
            (root / "conditioning_results.json").write_text(json.dumps(reports, indent=2) + "\n")
        return methods["close_env"](self, *args, **kwargs)

    stack_blocks_two.setup_demo = setup
    stack_blocks_two.set_instruction = instruction
    stack_blocks_two.check_success = success
    stack_blocks_two.close_env = close
    # Upstream selects an ICL asset even in text-only mode. This experiment
    # explicitly disables ICL, so it does not need a task video-map entry.
    client.select_icl_video = lambda *args, **kwargs: ""
    try:
        (root / "protocol.json").write_text(json.dumps({
            "task": "stack_blocks_two", "condition": condition,
            "prompt": prompt, "seed_argument": seed, "test_num": test_num,
            "checkpoint": "Robbyant-Research/zero-wam-posttrain-robotwin",
            "revision": "07ee865f175d9474a5653e9147698390e47b6143",
            "mode": mode, "video_path": video_path, "expert_gate": "stock green_on_red",
            "limitations": ["Reverse stacking needs a separate scripted feasibility gate",
                            "Successful stock ordering alone does not establish conditioning"],
        }, indent=2) + "\n")
        run(model, mode, "stack_blocks_two", test_num, seed, str(root), icl_video_bytes=video_bytes)
    finally:
        (root / "conditioning_results.json").write_text(json.dumps(reports, indent=2) + "\n")
        for name, method in methods.items():
            setattr(stack_blocks_two, name, method)
        client.select_icl_video = original_video_selection


def scripted_gate(seeds, save_root, capture_video=False):
    """Use native setup/planners with the expert's pick order reversed."""
    root = Path(save_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from evaluation.robotwin import eval_policy_client_openpi as client
    from zero_wam.modal_client import run
    original = client.eval_policy
    reports = []

    def gate(task_name, env, args, model, st_seed, **kwargs):
        args["eval_mode"] = True
        args["render_freq"] = 0
        for seed in seeds:
            for goal in ("green_on_red", "red_on_green"):
                env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
                red, green = env.block1, env.block2
                initial = {"red": red.get_pose().p.tolist(), "green": green.get_pose().p.tolist()}
                frames = []
                picture = env._take_picture
                original_save_freq = env.save_freq
                if capture_video:
                    def capture():
                        frames.append(env.get_obs()["observation"]["head_camera"]["rgb"].copy())
                    env._take_picture = capture
                    env.save_freq = 20
                    capture()
                try:
                    if goal == "red_on_green":
                        env.block1, env.block2 = green, red
                    env.play_once()
                    env.block1, env.block2 = red, green
                    outcomes = stack_outcomes(red.get_pose().p, green.get_pose().p,
                                              env.is_left_gripper_open() and env.is_right_gripper_open())
                    reports.append(dict(seed=seed, goal=goal, initial=initial,
                                        plan_success=bool(env.plan_success), outcomes=outcomes,
                                        success=bool(env.plan_success and outcomes[goal])))
                    if capture_video:
                        import imageio.v2 as imageio
                        from PIL import Image
                        indices = np.linspace(0, len(frames) - 1, 96).astype(int)
                        selected = [np.asarray(Image.fromarray(frames[i]).resize((320, 240))) for i in indices]
                        video = root / f"seed{seed}_{goal}.mp4"
                        imageio.mimsave(video, selected, fps=12)
                        reports[-1].update(video=str(video), captured_frames=len(frames),
                                           output_frames=96, fps=12,
                                           sampling="uniform over entire successful expert; endpoints included")
                finally:
                    env.block1, env.block2 = red, green
                    env._take_picture = picture
                    env.save_freq = original_save_freq
                    env.close_env()
                    (root / "scripted_gate.json").write_text(json.dumps(reports, indent=2) + "\n")
        return st_seed, sum(all(r["success"] for r in reports if r["seed"] == seed) for seed in seeds)

    client.eval_policy = gate
    try:
        run(None, "text", "stack_blocks_two", len(seeds), 0, str(root))
    finally:
        client.eval_policy = original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-seeds", default="100000,100001,100002")
    parser.add_argument("--save-root", default="outputs/zero_wam/robotwin_conditioning/scripted_gate")
    parser.add_argument("--capture-video", action="store_true")
    options = parser.parse_args()
    scripted_gate([int(s) for s in options.gate_seeds.split(",")], options.save_root, options.capture_video)
