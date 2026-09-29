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

    # v4 scalars: fill_outlier_z (dataset-relative) + mask_prior_iou
    stats = defaultdict(lambda: defaultdict(list))  # ds -> stat -> [values]
    metas = {}
    for m in sorted(run_dir.glob("*/*/metadata.json")):
        d = json.loads(m.read_text())
        metas[d["key"]] = d
        for s in ("fill_lum_step", "fill_chroma_drift", "fill_texture_ratio"):
            if s in d:
                stats[d["dataset_id"]][s].append((d["key"], d[s]))
    for ds, ss in stats.items():
        for s, pairs in ss.items():
            vals = [v for _, v in pairs]
            med = statistics.median(vals)
            mad = statistics.median([abs(v - med) for v in vals]) or 1e-6
            for k, v in pairs:
                z = (v - med) / (1.4826 * mad)
                metas[k].setdefault("_z", {})[s] = z
    for k, d in metas.items():
        zmax = max(d.get("_z", {}).values(), default=0.0)
        if zmax > 4.0:
            qual.setdefault(k, []).append(f"fill_outlier_z:{zmax:.1f}")
        iou = d.get("mask_prior_iou")
        if iou is not None and iou < 0.15:
            qual.setdefault(k, []).append(f"mask_prior_iou:{iou:.2f}")

    audit = run_dir / "audit.json"
    audited = 0
    if audit.exists():
        for e in json.loads(audit.read_text()):
            audited += 1
            k = e["key"]
            if e.get("coverage_miss") or e.get("robot_fill"):  # legacy schema
                for f in ("coverage_miss", "robot_fill"):
                    if e.get(f):
                        qual.setdefault(k, []).append(f)
            if e.get("residual_robot", 0) > 0.05:
                qual.setdefault(k, []).append(f"residual_robot:{e['residual_robot']}")
            if e.get("uncovered_px", 0) > 1500:
                qual.setdefault(k, []).append(f"uncovered_px:{e['uncovered_px']}")
            op = e.get("object_preservation")
            if op is not None and op < 0.4:
                qual.setdefault(k, []).append(f"object_preservation:{op}")
    accepted = json.loads((run_dir / "accepted.json").read_text())
    clean = [k for k in accepted if k not in qual]
    (run_dir / "quarantine.json").write_text(json.dumps(qual, indent=1))
    (run_dir / "clean.json").write_text(json.dumps(clean, indent=1))
    return {"accepted": len(accepted), "quarantined": len(qual), "clean": len(clean), "audited": audited}


if __name__ == "__main__":
    print(json.dumps(build_quarantine(Path(sys.argv[1]))))
