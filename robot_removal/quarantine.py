"""Build quarantine.json + clean.json for a run.

Quarantined = any non-crop metadata flag (mask size, flat fill, seam step),
a per-dataset coverage outlier (Tukey fences — same dataset shares camera and
robot, so coverage outliers are segmentation misses/floods), or an audit flag
(coverage_miss / robot_fill). Audit coverage may be partial; the list is
regenerated from whatever audit.json currently contains.

Usage: python -m robot_removal.quarantine <run_dir>
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def coverage_outliers(cov: dict[str, float]) -> dict[str, str]:
    """Per-dataset Tukey fences; returns key -> 'coverage_low|coverage_high'."""
    by_ds: dict[str, list[float]] = defaultdict(list)
    for k, c in cov.items():
        by_ds[k.split("/")[0]].append(c)
    bounds = {}
    for ds, vals in by_ds.items():
        if len(vals) < 8:
            continue
        q = statistics.quantiles(vals, n=4)
        iqr = q[2] - q[0]
        bounds[ds] = (max(0.0, q[0] - 1.5 * iqr - 0.01), q[2] + 1.5 * iqr + 0.02)
    out = {}
    for k, c in cov.items():
        b = bounds.get(k.split("/")[0])
        if not b:
            continue
        if c < b[0]:
            out[k] = f"coverage_low:{c:.3f}"
        elif c > b[1]:
            out[k] = f"coverage_high:{c:.3f}"
    return out


def build_quarantine(run_dir: Path) -> dict:
    qual: dict[str, list[str]] = {}
    cov = {}
    for m in sorted(run_dir.glob("*/*/metadata.json")):
        d = json.loads(m.read_text())
        cov[d["key"]] = d["mask_coverage"]
        qf = [f for f in d.get("flags", []) if not f.startswith("crop_")]
        if qf:
            qual[d["key"]] = qf
    for k, f in coverage_outliers(cov).items():
        qual.setdefault(k, []).append(f)
    audit = run_dir / "audit.json"
    audited = 0
    if audit.exists():
        for e in json.loads(audit.read_text()):
            audited += 1
            for k in ("coverage_miss", "robot_fill"):
                if e.get(k):
                    qual.setdefault(e["key"], []).append(k)
    accepted = json.loads((run_dir / "accepted.json").read_text())
    clean = [k for k in accepted if k not in qual]
    (run_dir / "quarantine.json").write_text(json.dumps(qual, indent=1))
    (run_dir / "clean.json").write_text(json.dumps(clean, indent=1))
    return {"accepted": len(accepted), "quarantined": len(qual), "clean": len(clean), "audited": audited}


if __name__ == "__main__":
    print(json.dumps(build_quarantine(Path(sys.argv[1]))))
