"""Judge sheets for drafted pairs, and the accepted manifest once judge verdicts are in.

sheets: one image per pair (first frame with entity boxes, 12 labelled frames with step bars) for an LLM judge.
accept: keep pairs that validate, whose outcome is success, and whose judge verdict is pass or minor_issues.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from schema.source import frame, load_episode, local_source, uniform
from schema.validate import check_pair

CURATED = Path("data/so101_curated")
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
TILE = (320, 240)


def pairs(root: Path) -> list[Path]:
    return sorted(root.glob("*/episode_*/pair.json"))


def sheet(pair_path: Path, out: Path) -> Path:
    name, ep_dir = pair_path.parts[-3], pair_path.parent
    p = json.loads(pair_path.read_text())
    root = CURATED / name
    ep = load_episode(root, *local_source(root), p["source"]["episode_index"])
    first = Image.open(ep_dir / "first.jpg").convert("RGB").resize((640, 480))
    d = ImageDraw.Draw(first)
    for k, e in enumerate(p["scene"]["entities"]):
        if e["box_2d"] and e["kind"] != "surface":
            y0, x0, y1, x1 = [v * s / 1000 for v, s in zip(e["box_2d"], (480, 640, 480, 640))]
            c = COLORS[k % len(COLORS)]
            d.rectangle([x0, y0, x1, y1], outline=c, width=3)
            d.rectangle([x0, y0, x0 + 7 * len(e["name"]) + 6, y0 + 14], fill=c)
            d.text((x0 + 3, y0 + 1), e["name"], fill="white")
    strip = Image.new("RGB", (4 * TILE[0], 3 * (TILE[1] + 22)), "black")
    sd = ImageDraw.Draw(strip)
    for k, i in enumerate(uniform(ep, 12)):
        tile = Image.open(frame(ep, p["source"]["camera_key"], i, ep_dir / "frames" / f"{i:06d}.jpg")).convert("RGB").resize(TILE)
        x, y = (k % 4) * TILE[0], (k // 4) * (TILE[1] + 22)
        strip.paste(tile, (x, y))
        sd.rectangle([x, y, x + 92, y + 16], fill="black")
        sd.text((x + 4, y + 2), f"frame {i}", fill="white")
        for s_i, s in enumerate(p["segments"]):
            if s["start"] <= i < s["end"]:
                sd.rectangle([x, y + TILE[1] + 2, x + TILE[0], y + TILE[1] + 20], fill=COLORS[s_i % len(COLORS)])
                sd.text((x + 4, y + TILE[1] + 5), s["step"][:40], fill="white")
    img = Image.new("RGB", (strip.width, 480 + strip.height), "white")
    img.paste(first, (0, 0))
    info = ImageDraw.Draw(img)
    info.text((660, 10), f"{name} | episode {ep.index} | {p['source']['frames']['end']} frames", fill="black")
    info.text((660, 30), f"task: {p['task']['instruction'][:90]}", fill="black")
    info.text((660, 44), f"outcome: {p['source']['outcome']}", fill="black")
    for k, s in enumerate(p["segments"]):
        info.text((660, 60 + 16 * k), f"{s['step']}: [{s['start']}, {s['end']}) grasp={s['grasp']} release={s['release']} ({s['source']})", fill=COLORS[k % len(COLORS)])
    img.paste(strip, (0, 480))
    path = out / f"{name}__ep{ep.index:03d}.jpg"
    img.save(path, quality=85)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["sheets", "accept"])
    p.add_argument("root", type=Path, help="directory of <dataset>/episode_*/pair.json")
    p.add_argument("--verdicts", type=Path, help="accept: JSON list of judge records with 'pair' and 'overall'")
    args = p.parse_args(argv)
    out = args.root / "_judge"
    out.mkdir(exist_ok=True)
    if args.command == "sheets":
        todo = [q for q in pairs(args.root) if not (out / f"{q.parts[-3]}__{q.parts[-2].replace('episode_', 'ep')}.jpg").exists()]
        with ThreadPoolExecutor(8) as ex:
            made = list(ex.map(lambda q: sheet(q, out), todo))
        index = [f.stem for f in sorted(out.glob("*__ep*.jpg"))]
        (out / "index.json").write_text(json.dumps(index) + "\n")
        print(f"{len(made)} new sheets, {len(index)} total")
        return 0
    verdicts = {str(Path(v["pair"]).resolve()): v for v in json.loads(args.verdicts.read_text())}
    accepted, reasons = [], {}
    for q in pairs(args.root):
        pair = json.loads(q.read_text())
        v = verdicts.get(str(q.resolve()))
        why = ("invalid" if check_pair(pair) else f"outcome {pair['source']['outcome']}" if pair["source"]["outcome"] != "success"
               else "not judged" if v is None else f"judge {v['overall']}" if v["overall"] == "fail" else None)
        if why:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            accepted.append(str(q.relative_to(args.root)))
    (args.root / "accepted.json").write_text(json.dumps(accepted, indent=1) + "\n")
    print(f"accepted {len(accepted)} of {len(pairs(args.root))}; rejected {reasons}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
