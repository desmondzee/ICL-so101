"""Video-to-video pilot: edit the real robot video with Reactor SANA-Streaming, replacing the robot arm with a human
forearm and hand. Object motion, contact and timing come from the real footage, so nothing is invented.

Per episode (under --out/<dataset>/episode_XXX/): source.mp4 (the robot episode's action window, sped up, padded to
16:9), prompt.txt, raw.mp4 (SANA output, 1280x704), video.mp4 (cropped back to 4:3, 640x480, no audio), contact.png,
messages.jsonl (every model message, for debugging), and robot_first.jpg / robot_last.jpg / pair.json for the judges.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from humangen import context
from humangen.pilot import EDGE_TEXT, plan
from schema.source import frame, load_episode, local_source

MODEL = "reactor/sana-streaming"
OUT_W, OUT_H = 1280, 704
SPEEDUP = 2.0
QUIET_S = 15.0  # stop when no new frame arrives for this long after the source length was reached
MAX_WAIT_S = 600.0

PROMPT = ("Replace the {robot} with a person's bare {active} forearm and hand reaching in from the {edge}, "
          "the fingers closing where the gripper closes and holding what it holds. "
          "Natural skin with soft shadows under the same light as the scene. "
          "Preserve every object and its motion, the table, the background, the camera framing and the lighting.")


def cut_source(ep, pair: dict, out: Path, speed: float) -> float:
    """The action window of the robot episode (first segment start to last segment end), sped up, padded to 16:9."""
    video, start = ep.videos[pair["source"]["camera_key"]]
    fps = pair["source"]["fps"]
    segs = pair["segments"]
    grasps = [s["grasp"] for s in segs if s["grasp"] is not None]
    a0 = max(segs[0]["start"], (grasps[0] if grasps else segs[0]["start"]) - int(1.0 * fps))
    a1 = min(pair["source"]["frames"]["end"] - 1, segs[-1]["end"] + int(0.5 * fps))
    t0, dur = start + a0 / fps, (a1 - a0) / fps
    # 640x480 source: pad left/right to the 16:9 canvas so nothing is stretched
    pad_w = round(480 * OUT_W / OUT_H / 2) * 2
    vf = f"setpts=PTS/{speed},fps=24,pad={pad_w}:480:(ow-iw)/2:0:black"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
                    "-vf", vf, "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out)], check=True)
    return dur / speed


class Recorder:
    def __init__(self, out: Path):
        self.out, self.q, self.count, self.last = out, queue.Queue(), 0, time.monotonic()
        self.proc = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def on_video(self, bgra: bytes, width: int, height: int, *_):
        self.count += 1
        self.last = time.monotonic()
        self.q.put((bgra, width, height))

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            bgra, w, h = item
            if self.proc is None:
                self.proc = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
                                              "-s", f"{w}x{h}", "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                                              str(self.out)], stdin=subprocess.PIPE)
            self.proc.stdin.write(bgra)
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()

    def close(self):
        self.q.put(None)
        self.thread.join(timeout=120)


async def edit_video(source: Path, prompt: str, raw: Path, log: Path, api_key: str, seed: int = 0, anchor: int = 4) -> int:
    """Stream the source clip into the model's `camera` track at 24 fps (this deployment has live mode only) and record
    the edited `main_video` until it goes quiet after the last source frame."""
    import numpy as np
    from reactor_sdk import Reactor

    from humangen.generate import _normalize

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(source)],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    dec = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    frames = []
    while True:
        buf = dec.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
    dec.wait()
    rec = Recorder(raw)
    loop = asyncio.get_running_loop()
    with log.open("w") as lf:
        def on_message(m):
            kind, body = _normalize(m)
            loop.call_soon_threadsafe(lf.write, json.dumps({"kind": kind, "body": body}, default=str) + "\n")

        async with Reactor(MODEL, api_key=api_key) as reactor:
            reactor.on("message", on_message)
            reactor.track("main_video").on_raw_frame(rec.on_video)
            await reactor.connect()
            if "main_video" in reactor.paused_tracks:
                await reactor.track("main_video").resume()
            cam = await reactor.publish_track("camera")
            for _ in range(24):  # a second of the first frame so the model has a source before start
                cam.push_frame(frames[0])
                await asyncio.sleep(1 / 24)
            await reactor.send_command("set_seed", {"seed": seed})
            await reactor.send_command("set_anchor_interval", {"chunks": anchor})
            await reactor.send_command("set_prompt", {"prompt": prompt})
            await reactor.send_command("start", {})
            t0 = time.monotonic()
            for i, f in enumerate(frames):
                cam.push_frame(f)
                await asyncio.sleep(max(0.0, t0 + (i + 1) / 24 - time.monotonic()))
            t_end = time.monotonic()
            while time.monotonic() - t_end < 10 and not (rec.count and time.monotonic() - rec.last > 4 and time.monotonic() - t_end > 4):
                cam.push_frame(frames[-1])  # hold the last frame while the model drains its buffer
                await asyncio.sleep(1 / 24)
            await reactor.send_command("pause", {})
            await reactor.disconnect()
    rec.close()
    return rec.count


def finish(raw: Path, out: Path) -> None:
    """Crop the 16:9 output back to the 4:3 source area and scale to 640x480."""
    cw = round(OUT_H * 4 / 3 / 2) * 2
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(raw), "-vf", f"crop={cw}:{OUT_H}:(iw-{cw})/2:0,scale=640:480",
                    "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(out), "-vf", "fps=2,scale=192:-1,tile=6x4",
                    "-frames:v", "1", str(out.with_name("contact.png"))], check=True)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", help="<dataset>/episode_XXX (export episode index)")
    ap.add_argument("--export", type=Path, default=Path("data/so101_export"))
    ap.add_argument("--out", type=Path, default=Path("data/humangen_pilot/sana_v2v"))
    ap.add_argument("--speed", type=float, default=SPEEDUP)
    ap.add_argument("--anchor", type=int, default=4, help="re-ground on the source every N chunks (0 = off)")
    ap.add_argument("--prompt", help="override the edit prompt")
    args = ap.parse_args(argv)
    key = os.environ["REACTOR_API_KEY"]
    for rel in args.episodes:
        dataset, epname = rel.split("/")
        idx = int(epname.split("_")[1])
        d = args.out / rel
        d.mkdir(parents=True, exist_ok=True)
        try:
            pair = json.loads((args.export / "pairs" / dataset / epname / "pair.json").read_text())
            root = args.export / "lerobot" / dataset
            ep = load_episode(root, *local_source(root), idx)
            p = plan(pair, rel)
            robot = next((r for r in pair["task"]["roles"] if r["kind"] == "actor"), None)
            robot_text = (robot["appearance"].rstrip(".").replace("A ", "the ", 1).replace("An ", "the ", 1) if robot else "robot arm")
            prompt = args.prompt or PROMPT.format(robot=robot_text, active=p["active"], edge=EDGE_TEXT[p["edge"]])
            (d / "prompt.txt").write_text(prompt + "\n")
            if not (d / "source.mp4").exists():
                secs = cut_source(ep, pair, d / "source.mp4", args.speed)
                logging.info("%s: source %.1fs", rel, secs)
            cam = pair["source"]["camera_key"]
            frame(ep, cam, 0, d / "robot_first.jpg")
            frame(ep, cam, ep.length - 1, d / "robot_last.jpg")
            (d / "pair.json").write_text(json.dumps(pair, indent=1) + "\n")
            if not (d / "video.mp4").exists():
                n = asyncio.run(edit_video(d / "source.mp4", prompt, d / "raw.mp4", d / "messages.jsonl", key, seed=p["seed"], anchor=args.anchor))
                logging.info("%s: %d frames received", rel, n)
                if n:
                    finish(d / "raw.mp4", d / "video.mp4")
        except Exception as e:
            logging.error("%s: %s", rel, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
