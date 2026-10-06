"""Visual self-check for family authors: one contact sheet per task.

  python -m sim.train.tasks.preview --tasks block_in_bowl --seeds 9000 9001 9002
  python -m sim.train.tasks.preview --family pilot_blocks --count 3      # seeds 9000.. (calibration range)

Each row is one seed: the robot-free first frame (the human-video start image), ``--frames`` front-camera
frames with the robot spread over the rollout, and the robot-free last frame (the end image). The caption
gives seed, arena, task success and the strict physics verdict with any failed checks. Output defaults to
``/tmp/so101_task_previews/<task>.png``; nothing is written under data/.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim.train.physics import audit_trajectory
from .catalog import TRAIN_TASKS, family_tasks
from .qualify import rollout

THUMB = (320, 240)


def _thumb(rgb, caption=None):
    image = Image.fromarray(rgb).resize(THUMB, Image.BILINEAR)
    if caption:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, THUMB[0], 16), fill=(0, 0, 0))
        draw.text((4, 2), caption, fill=(255, 255, 255))
    return image


def preview_seed(task, seed, frames=6):
    """Return the row images for one seed plus a one-line summary."""
    captured = {}

    def capture(env, index):
        captured[index] = env.render_camera("front", robot=True).copy()

    # First pass: physics verdict and frame count (fast, unrendered).
    env, oracle, config, poses, collector = rollout(task, seed)
    try:
        n = collector.frame
        report = audit_trajectory(collector.finish(), task.physics_policy(env))
        success = bool(env.success())
        last = env.render_scene_without_robot("front").copy()
    finally:
        env.close()
    # Second pass: a deterministic replay that renders the chosen frames.
    wanted = set(np.linspace(0, n - 1, frames + 2).round().astype(int)[1:-1])
    env_class, oracle_class = task.load_classes()
    env = env_class(render_images=False, visual_config=config)
    try:
        env.reset(seed=seed)
        first = env.render_scene_without_robot("front").copy()
        oracle = oracle_class(env, np.random.default_rng(seed))
        for frame, action in enumerate(oracle.actions()):
            env.step(action)
            if frame in wanted:
                capture(env, frame)
    finally:
        env.close()
    failed = [k for k, ok in report.checks.items() if not ok]
    verdict = "PASS" if report.accepted else "FAIL " + ",".join(failed)
    row = [_thumb(first, f"seed {seed} {config.arena} first (no robot)")]
    row += [_thumb(captured[i], f"frame {i}/{n}") for i in sorted(captured)]
    row.append(_thumb(last, f"last (no robot) success={success}"))
    return row, f"seed {seed}: {config.arena}, success={success}, physics {verdict}"


def contact_sheet(task, seeds, frames=6):
    rows, lines = [], []
    for seed in seeds:
        row, line = preview_seed(task, seed, frames)
        rows.append(row)
        lines.append(line)
        print(f"{task.name} {line}", flush=True)
    width, height = THUMB[0] * len(rows[0]), THUMB[1] * len(rows) + 40
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), f"{task.name}: {task.instruction}", fill=(0, 0, 0))
    draw.text((6, 20), " | ".join(task.action_text)[:200], fill=(60, 60, 60))
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            sheet.paste(image, (c * THUMB[0], 40 + r * THUMB[1]))
    return sheet, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TRAIN_TASKS))
    parser.add_argument("--family", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--count", type=int, default=3, help="seeds 9000.. when --seeds is not given")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("/tmp/so101_task_previews"))
    args = parser.parse_args(argv)
    names = list(args.tasks or ())
    for family in args.family or ():
        names += [n for n in family_tasks(family) if n not in names]
    if not names:
        parser.error("pass --tasks or --family")
    seeds = args.seeds or list(range(9000, 9000 + args.count))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in names:
        sheet, _ = contact_sheet(TRAIN_TASKS[name], seeds, args.frames)
        path = args.output / f"{name}.png"
        sheet.save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
