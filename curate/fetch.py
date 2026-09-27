"""Download metadata, trajectories and the selected cameras' videos for every dataset in selection.json."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

SELECTION = Path(__file__).with_name("selection.json")


def files(entry: dict) -> list[tuple[str, str, str, Path]]:
    src, root = entry["source"], Path(entry["root"])
    if "revision" not in src:
        return []
    info = json.loads((root / "meta/info.json").read_text())
    prefix = src["path"] + "/" if "path" in src else ""
    local = root.parents[len(Path(prefix).parts) - 1] if prefix else root
    bak = root / "meta/info.json.bak"
    names = json.loads(bak.read_text()) if bak.exists() else info
    keys = dict(zip([k.split(".")[-1] for k, f in names["features"].items() if f["dtype"] == "video"],
                    [k for k, f in info["features"].items() if f["dtype"] == "video"]))
    cams = [keys[c] for c in (entry["front"], entry["wrist"]) if c]
    episodes = [json.loads(l)["episode_index"] for l in (root / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for e in episodes:
        chunk = e // info["chunks_size"]
        for c in cams:
            path = info["video_path"].format(episode_chunk=chunk, video_key=c, episode_index=e)
            if not (root / path).exists():
                out.append((src["repo"], prefix + path, src["revision"], local))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    todo = [f for entry in json.loads(SELECTION.read_text()) for f in files(entry)]
    print(f"{len(todo)} files to download", flush=True)

    def get(f):
        try:
            hf_hub_download(f[0], f[1], repo_type="dataset", revision=f[2], local_dir=f[3])
            return None
        except Exception as e:
            return f"{f[1]}: {e}"

    with ThreadPoolExecutor(args.workers) as ex:
        errors = [e for e in ex.map(get, todo) if e]
    for e in errors:
        print(e)
    print(f"done, {len(errors)} failed")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
