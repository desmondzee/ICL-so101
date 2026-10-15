"""Render review sheets from the frozen measurements; no metric recomputation."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/results/zero_wam_partial_chunk_replay"


def main():
    snapshot = json.loads((OUT/"measurements.json").read_text())
    assets = OUT/"assets"
    assets.mkdir(exist_ok=True)
    for episode in snapshot["oracle_motion"]:
        panels = []
        for r in [x for x in snapshot["episodes"] if x["episode"] == episode]:
            cap = cv2.VideoCapture(str(ROOT/r["directory"]/"rollout.mp4"))
            tiles = []
            for label, control in (("quarter oracle", r["oracle_controls"]//4),
                                   ("closest TCP", r["closest_control"]),
                                   ("oracle horizon", r["oracle_controls"]),
                                   ("full horizon", r["controls"])):
                index = control//2
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"missing frame {episode} {r['arm']} {index}")
                tile = np.full((228, 512, 3), 245, dtype=np.uint8)
                tile[36:] = cv2.resize(frame, (512, 192))
                cv2.putText(tile, f"{r['arm']} | {label} | {index/15:.2f}s",
                            (6, 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (20,20,20), 1)
                tiles.append(tile)
            cap.release()
            panels.append(np.concatenate(tiles, axis=1))
        dest = assets/(episode.replace("/", "__")+".jpg")
        cv2.imwrite(str(dest), np.concatenate(panels, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(dest)


if __name__ == "__main__":
    main()
