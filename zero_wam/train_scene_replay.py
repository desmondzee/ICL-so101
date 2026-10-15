"""Reproduce frozen training layouts across distractor-sampler revisions."""
from __future__ import annotations

import types
import numpy as np


def compare_recorded_layout(actual, expected):
    """Compare all recorded bodies, including reference objects and distractors.

    Episode poses are rounded to five decimal places. Compare these directly,
    allowing only numerical roundoff, and treat signed zero as equal.
    """
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    errors = {}
    for name in sorted(set(expected) & set(actual)):
        a, b = np.asarray(actual[name], float), np.asarray(expected[name], float)
        errors[name] = (
            float(np.max(np.abs(a - b)))
            if a.shape == b.shape == (7,) and np.isfinite(a).all()
            and np.isfinite(b).all() else float("inf")
        )
    maximum = max(errors.values(), default=float("inf"))
    return {
        "pass": bool(expected) and not missing and not extra and maximum <= 1e-9,
        "missing": missing, "extra": extra,
        "max_abs_pose_error": maximum, "per_object_pose_error": errors,
        "compared": len(errors), "scope": "all active bodies; xyz and quaternion",
        "tolerance": 1e-9, "pose_precision_decimals": 5,
    }


def reset_recorded_scene(env, seed, expected):
    """Select a sampler only by exact agreement with frozen full-scene metadata.

    No object pose is teleported after reset: selecting the original sampling
    procedure also reconstructs goals, contacts and settled simulator state.
    The fallback is instance-local and cannot change another environment.
    """
    from sim.val.env import ValEnv
    original = env.place_distractors
    attempts = []
    for mode, sampler in (
        ("current_weighted", original),
        ("legacy_permutation", types.MethodType(ValEnv.place_distractors, env)),
    ):
        env.place_distractors = sampler
        env.reset(seed=seed)
        actual = env.object_poses()
        check = compare_recorded_layout(actual, expected)
        attempts.append({"sampler": mode, **check})
        if check["pass"]:
            return {
                "sampler": mode, "check": check, "attempts": attempts,
                "actual_object_poses": actual, "expected_object_poses": expected,
                "legacy_sampler_source": "sim.val.env.ValEnv.place_distractors",
            }
    env.place_distractors = original
    raise ValueError(f"cannot reproduce recorded scene with known samplers: {attempts}")
