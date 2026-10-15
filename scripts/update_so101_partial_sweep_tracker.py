"""Refresh only the marked live-status section from durable sweep artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "docs/ZERO_WAM_PARTIAL_CHUNK_REPLAY_TRACKER.md"
BEGIN = "<!-- PARTIAL_SWEEP_LIVE_START -->"
END = "<!-- PARTIAL_SWEEP_LIVE_END -->"
ARMS = [
    ("16 controls, no ensemble", "train_replay_v2b_partial8_raw_full_conditioning_3tasks_2seeds_scenev2_b4"),
    ("16 controls, ensemble", "train_replay_v2b_partial8_ensemble_full_conditioning_3tasks_2seeds_scenev2_b4"),
    ("32 controls, no ensemble", "train_replay_v2b_partial8_forecast4_raw_full_conditioning_3tasks_2seeds_scenev2_b4"),
    ("32 controls, ensemble", "train_replay_v2b_partial8_forecast4_ensemble_full_conditioning_3tasks_2seeds_scenev2_b4"),
]
EPISODES = [f"{task}/episode_{i:03d}" for task in
            ("ball_between_cans", "block_in_bowl", "push_cube_into_tape_square")
            for i in range(2)]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def process_state(pid):
    try:
        # The command field in stat may contain spaces.
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state if state != "Z" else None
    except FileNotFoundError:
        return None


def refresh(supervisor):
    alive = process_state(supervisor)
    active = {}
    try:
        children = Path(f"/proc/{supervisor}/task/{supervisor}/children").read_text().split()
    except FileNotFoundError:
        children = []
    for pid in children:
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        except FileNotFoundError:
            continue
        if any(x.endswith("run_so101_batched_partial_replay.py") for x in args) and "--output" in args:
            active[Path(args[args.index("--output")+1]).name] = (pid, process_state(pid))
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [BEGIN, "## Live sweep status", "", f"Last refreshed: **{stamp}**.",
             "Updated automatically every five minutes from completed result markers and live processes.",
             "", "| Case | Complete | Successes | State |",
             "|---|---:|---:|---|"]
    details = []
    total = 0
    for label, name in ARMS:
        root = ROOT / "outputs/zero_wam" / name
        completed = [r for ep in EPISODES if
                     (r := read_json(root / ep / "result.json")).get("partial_protocol_sha256")]
        total += len(completed)
        if len(completed) == 6:
            state = "Completed"
        elif name in active:
            pid, stat = active[name]
            state = f"{'Paused' if stat in ('T', 't') else 'Running'} (PID {pid})"
            summary = read_json(root / "summary.json")
            for row in summary.get("progress", []):
                details.append(f"- {label}: `{row['id']}` — {row['controls']} controls"
                               f" ({'active' if row['active'] else 'finished'}).")
        elif alive:
            state = "Queued"
        else:
            state = "Not running; supervisor absent"
        lines.append(f"| {label} | {len(completed)}/6 | "
                     f"{sum(bool(r['success']) for r in completed)} | {state} |")
    lines += ["", f"Completed episodes: **{total}/24**.",
              f"Continuation supervisor: `{supervisor}` ({alive or 'absent'}).",
              "Suite log: `/tmp/so101_partial_scenev2_b4_suite.log`.",
              "HF sync log: `/tmp/so101_partial_bucket_sync.log`.", ""]
    lines += details
    lines += ["", END]
    block = "\n".join(lines)
    old = TRACKER.read_text()
    if BEGIN in old and END in old:
        start, stop = old.index(BEGIN), old.index(END)+len(END)
        new = old[:start]+block+old[stop:]
    elif BEGIN in old or END in old:
        raise RuntimeError("incomplete status markers; refusing overwrite")
    else:
        heading, rest = old.split("\n", 1)
        new = heading+"\n\n"+block+"\n"+rest
    # Avoid replacing edits made while the status block was being built.
    if TRACKER.read_text() != old:
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=TRACKER.parent, delete=False) as sink:
        sink.write(new)
        temp = Path(sink.name)
    os.chmod(temp, TRACKER.stat().st_mode & 0o777)
    os.replace(temp, TRACKER)
    print(f"{stamp} refreshed tracker: completed={total}/24", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--supervisor-pid", type=int, required=True)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    with open("/tmp/so101_partial_tracker.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("tracker updater already running")
        while True:
            refresh(args.supervisor_pid)
            if args.once:
                break
            time.sleep(300)


if __name__ == "__main__":
    main()
