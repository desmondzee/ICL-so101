"""Robot-mask selection: port of humangen/edit.py's robot_mask_sam3 to the
ComfyUI SAM3 node's per-prompt individual-mask outputs.

arm candidates ("robot arm", "robotic arm", "robot") come back score-sorted
per prompt; extras ("cable", "robot base", "shadow") are unioned only where
they touch the arm. The connected region holding the best detection is kept.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _in_box(m: np.ndarray, box: tuple[int, int, int, int] | None) -> float:
    if not box:
        return 0.0
    x, y, w, h = box
    sub = m[y : y + h, x : x + w]
    return sub.sum() / max(1, m.sum())


def select_robot(
    arm_masks: list[np.ndarray],
    extra_masks: list[np.ndarray],
    box_px: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Pick the one robot from scored arm candidates and grow by touching extras."""
    if not arm_masks:
        raise RuntimeError("SAM3 found no robot arm detections")
    # edit.py picks argmax(score + 0.5 * in_box); masks arrive score-ordered
    # per prompt, so rank order within each prompt is the score signal we have.
    def rank_bonus(i):
        return 1.0 / (1.0 + i)

    best = max(enumerate(arm_masks), key=lambda iv: 0.5 * _in_box(iv[1], box_px) + 0.5 * rank_bonus(iv[0]))[1]
    robot = best.copy()
    for m in arm_masks:
        if (m & best).sum() > 0.2 * min(m.sum(), best.sum()):
            robot |= m
    for m in extra_masks:
        # extras must touch the robot and not dwarf it: a real shadow/cable is
        # comparable in size to the arm, while floor/case regions flood it
        if m.sum() > 2.5 * robot.sum():
            continue
        if (m & ndimage.binary_dilation(robot, iterations=6)).any():
            robot |= m
    lab, _ = ndimage.label(ndimage.binary_dilation(robot, iterations=3))
    under = lab[ndimage.binary_dilation(best, iterations=3)]
    under = under[under > 0]
    if under.size == 0:
        return robot
    ids, counts = np.unique(under, return_counts=True)
    main = lab == ids[counts.argmax()]  # the component holding most of `best`
    return robot & main


def disk_dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Exact disk dilation: pixels within `px` of the mask."""
    if px <= 0:
        return mask
    return ndimage.distance_transform_edt(~mask) <= px


def validate_mask(mask: np.ndarray, min_cov: float = 0.005, max_cov: float = 0.45) -> list[str]:
    """Return a list of warning flags (empty list = clean)."""
    flags = []
    cov = float(mask.mean())
    if cov < min_cov:
        flags.append(f"mask_too_small:{cov:.4f}")
    if cov > max_cov:
        flags.append(f"mask_too_large:{cov:.4f}")
    if not mask.any():
        flags.append("mask_empty")
    if mask.all():
        flags.append("mask_full")
    return flags


def mask_file_to_bool(path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 127
