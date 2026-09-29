"""HTML contact-sheet report for a robot-removal run.

Per episode: original, mask overlay, LaMa reference, pre-composite Qwen output,
native composite, final 480x320, plus coverage, crop bounds, flags, timing.
"""

from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

COLS = ["original.jpg", "mask_overlay.png", "reference_lama.png", "inpaint_native.png", "composite_native.png", "zerowam_480x320.png"]
LABELS = ["original", "mask overlay", "lama reference", "qwen (pre-composite)", "composite native", "Zero-WAM 480x320"]


def _b64(path: Path, max_w: int = 320) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _total_s(met: dict) -> float:
    return sum(met.get(k, 0) for k in ("segment_s", "reference_s", "edit_s", "postprocess_s"))


def build_report(run_dir: Path, out: Path | None = None) -> Path:
    out = out or run_dir / "report.html"
    rows = []
    n_ok = n_flags = 0
    times = []
    for ep in sorted(run_dir.glob("*/*/")):
        meta_p = ep / "metadata.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        met = json.loads((ep / "metrics.json").read_text()) if (ep / "metrics.json").exists() else {}
        flags = meta.get("flags", [])
        real_flags = [f for f in flags if not f.startswith("crop_")]
        n_flags += bool(real_flags)
        n_ok += 1
        times.append(_total_s(met))
        cells = "".join(
            f'<td><img loading="lazy" src="{_b64(ep / c)}"><div>{l}</div></td>'
            for c, l in zip(COLS, LABELS)
            if (ep / c).exists()
        )
        crop = meta.get("crop", {}).get("crop", "")
        rows.append(
            f'<tr><td><b>{meta["key"]}</b><br>cov={meta["mask_coverage"]:.3f}<br>'
            f'crop={crop}<br>flags={flags or "[]"}<br>{_total_s(met):.1f}s</td>{cells}</tr>'
        )
    html = f"""<!doctype html><meta charset="utf-8"><title>robot removal report</title>
<style>body{{font-family:monospace}}td,th{{border:1px solid #ccc;vertical-align:top;padding:4px}}
img{{display:block}}table{{border-collapse:collapse}}</style>
<h1>robot removal — {run_dir.name}</h1>
<p>{n_ok} episodes; flagged (non-crop): {n_flags}; median total {sorted(times)[len(times)//2] if times else 0:.1f}s</p>
<table><tr><th>episode</th>{''.join(f'<th>{l}</th>' for l in LABELS)}</tr>
{''.join(rows)}
</table>"""
    out.write_text(html)
    return out


if __name__ == "__main__":
    print(build_report(Path(sys.argv[1])))
