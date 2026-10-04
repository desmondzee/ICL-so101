"""Human demonstrations for the simulated validation episodes (H3 Max Turbo on fal).

Each recorded episode has robot-free first and last front-camera frames (data/so101_sim_val/frames/<task>/episode_XXX/);
the generator gets them as start and end images with a v5-style prompt built from the task instruction: one right hand
reaches in empty from the bottom edge, does the task, and withdraws. Videos go to
data/so101_sim_val/human/<task>/episode_XXX/seed_<k>/ (video.mp4, contact.png, meta.json). Re-running resumes queued
requests and skips finished ones; --seed gives a fresh attempt for episodes whose earlier video was rejected.

    uv run python -m humangen.sim_val_demos --tasks sort_blocks --episodes 0,1 --seed 0
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import re
from pathlib import Path

from dotenv import dotenv_values

from humangen.alternative_demos import run

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "so101_sim_val" / "frames"
OUT = ROOT / "data" / "so101_sim_val" / "human"
ENDPOINT = "minimax/h3-max-turbo/image-to-video"
DURATION = 5.0
HOLD_S = 1.0
VERBS = {"put": "puts", "place": "places", "stack": "stacks", "sort": "sorts", "move": "moves", "pick": "picks", "set": "sets"}


def action_text(instruction: str) -> str:
    """'Put the red mug on the plate.' -> 'puts the red mug on the plate' (third person, as the v5 prompts read)."""
    words = instruction.strip().rstrip(".").split(" ")
    words[0] = VERBS.get(words[0].lower(), words[0].lower())
    return re.sub(r"\s+", " ", " ".join(words))


def prompt(instruction: str) -> str:
    """The v5_exclusion prompt (humangen/context.py) with the sim task's action."""
    end = DURATION
    action = (f"A person's single right hand and forearm reaches in empty from the bottom edge, {action_text(instruction)}, "
              "then releases its grip and withdraws empty through the same edge.")
    visual = ("The camera remains fixed. The task uses only the objects already visible in Picture 1, with the same count, "
              "appearance and background throughout. One acting hand and forearm performs the task; the rest of that person "
              f"stays outside the image. {action} The hand completes each manipulation directly at the task locations, with "
              f"continuous physical contact while moving the object. By {end - HOLD_S:.2f} seconds, the task is complete and the "
              "hand is in its ending state; the scene remains still for the final second. No other hands or objects enter the "
              "scene at any time.")
    return ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second "
            f"mark of the target video; Picture 2 (from Shot 1) aligns with the {end:.2f}-second mark of the target video.\n\n"
            "integrated_multimodal_description: [Shot 1] Live-action, beginning with the camera framing, lighting, objects and "
            f"spatial arrangement established by Picture 1. {visual} The action continuously brings the scene into the object "
            "arrangement, hand visibility and composition established by Picture 2 at the end of this single static shot.\n\n"
            "overall_soundscape: a quiet room and the soft sounds of the hand handling the objects.")


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def rows(tasks: list[str], episodes: list[int] | None, seed: int) -> list[tuple[dict, Path]]:
    out = []
    for task in tasks:
        for ep in sorted((FRAMES / task).glob("episode_*")):
            if episodes is not None and int(ep.name.split("_")[1]) not in episodes:
                continue
            meta = json.loads((ep / "meta.json").read_text())
            key = f"{task}/{ep.name}"
            row = {"key": key, "prompt": prompt(meta["instruction"]), "instruction": meta["instruction"], "aspect": "4:3",
                   "start_data": data_url(ep / "first.png"), "end_data": data_url(ep / "last.png"), "seed": seed}
            out.append((row, OUT / task / ep.name / f"seed_{seed}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma-separated task names")
    ap.add_argument("--episodes", default="", help="comma-separated episode numbers (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    token = dotenv_values(ROOT / ".env")["FAL_API_KEY"]
    eps = [int(x) for x in a.episodes.split(",")] if a.episodes else None
    todo = rows(a.tasks.split(","), eps, a.seed)
    print(f"{len(todo)} videos", flush=True)
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda r: run("h3_max_turbo", ENDPOINT, r[0], token, destination=r[1]), todo))


if __name__ == "__main__":
    main()
