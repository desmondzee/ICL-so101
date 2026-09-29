"""Hard composite, exact preservation assertion, and the Zero-WAM 480x320 transform."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def composite(original: np.ndarray, edited: np.ndarray, mask: np.ndarray, soften_px: int = 0) -> np.ndarray:
    """edited pixels inside mask, original pixels outside. Integer selection.

    soften_px > 0 blends a linear ramp across the first soften_px pixels INSIDE
    the mask edge to hide the polygon boundary; outside stays exactly original.
    """
    assert original.shape == edited.shape == mask.shape[:2] + (3,), (original.shape, edited.shape, mask.shape)
    if soften_px <= 0:
        out = original.copy()
        out[mask] = edited[mask]
        return out
    from scipy import ndimage

    depth = ndimage.distance_transform_edt(mask)  # distance inside to boundary
    alpha = np.clip(depth / soften_px, 0.0, 1.0) * mask
    blend = original.astype(float) * (1 - alpha[..., None]) + edited.astype(float) * alpha[..., None]
    out = original.copy()
    inside = alpha > 0
    out[inside] = np.clip(np.rint(blend[inside]), 0, 255).astype("uint8")
    return out


def assert_preserved(saved_png: Path, original: np.ndarray, mask: np.ndarray) -> dict:
    """Reload the saved composite and require zero diff outside the mask."""
    got = np.asarray(Image.open(saved_png).convert("RGB"))
    assert got.shape == original.shape, (got.shape, original.shape)
    diff = np.abs(got.astype(int) - original.astype(int))
    outside = diff[~mask]
    stats = {
        "outside_max": int(outside.max()) if outside.size else 0,
        "outside_mean_abs": float(outside.mean()) if outside.size else 0.0,
        "outside_diff_pixels": int((diff.max(2) > 0)[~mask].sum()),
    }
    if stats["outside_diff_pixels"] != 0:
        raise AssertionError(f"outside-mask pixels changed: {stats}")
    return stats


def zerowam_transform(image: Image.Image, tw: int = 480, th: int = 320) -> tuple[Image.Image, dict]:
    """Aspect-preserving resize to cover (tw, th), then center crop. Exact spec."""
    w, h = image.size
    scale = max(tw / w, th / h)
    nw, nh = math.ceil(w * scale), math.ceil(h * scale)
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    final = resized.crop((left, top, left + tw, top + th))
    assert final.size == (tw, th), final.size
    return final, {"scale": scale, "resized": [nw, nh], "crop": [left, top, left + tw, top + th]}


def crop_risk(crop: list[int], image_size: tuple[int, int], object_boxes: list[dict], w: int, h: int) -> list[str]:
    """Flag task objects whose 0-1000 box partially leaves the kept region."""
    left, top, right, bottom = crop
    rw, rh = image_size  # resized dims
    flags = []
    for o in object_boxes:
        y0, x0, y1, x1 = o["box_2d"]
        if (y1 - y0) * (x1 - x0) > 600_000:  # covers >60% of frame: background surface
            continue
        # map to resized pixel coords
        px0, py0, px1, py1 = x0 * rw / 1000, y0 * rh / 1000, x1 * rw / 1000, y1 * rh / 1000
        inside = px0 >= left and py0 >= top and px1 <= right and py1 <= bottom
        overlaps = px1 > left and py1 > top and px0 < right and py0 < bottom
        if overlaps and not inside:
            flags.append(f"crop_clips_object:{o['id']}")
        elif not overlaps:
            flags.append(f"crop_removes_object:{o['id']}")
    return flags


def flat_fill_flag(original: np.ndarray, composite_rgb: np.ndarray, mask: np.ndarray) -> str | None:
    """Flag when the filled region is much flatter than the surrounding surface
    (texture-less inpainting artifact on uniform backgrounds)."""
    from scipy import ndimage

    ring = ndimage.binary_dilation(mask, iterations=24) & ~mask
    mi, ri = mask[:-1], ring[:-1]
    if not mi.any() or not ri.any():
        return None
    # local spatial texture: mean per-pixel gradient magnitude inside vs ring
    grad_in = np.abs(np.diff(composite_rgb.astype(float), axis=0)).mean(axis=2)
    grad_out = np.abs(np.diff(original.astype(float), axis=0)).mean(axis=2)
    inside = grad_in[mi].mean()
    outside = grad_out[ri].mean()
    if outside > 0 and inside < 0.35 * outside:
        return f"flat_fill:{inside:.2f}/{outside:.2f}"
    return None


def seam_flag(composite_rgb: np.ndarray, mask: np.ndarray) -> str | None:
    """Mean luminance step across the mask boundary; large steps = visible seam."""
    from scipy import ndimage

    edge_in = mask & ~ndimage.binary_erosion(mask, iterations=2)
    edge_out = ~mask & ndimage.binary_dilation(mask, iterations=2)
    if not edge_in.any() or not edge_out.any():
        return None
    din = ndimage.binary_dilation(edge_in, iterations=2)
    dout = ndimage.binary_erosion(mask, iterations=2)  # inside, just off boundary
    lum = composite_rgb.mean(axis=2)
    a = lum[din & ~mask].mean() if (din & ~mask).any() else np.nan
    b = lum[dout].mean() if dout.any() else np.nan
    if np.isfinite(a) and np.isfinite(b) and abs(a - b) > 45:
        return f"seam_step:{abs(a - b):.0f}"
    return None


def save_png_atomic(image: Image.Image, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".png", delete=False) as tmp:
        tmp_p = Path(tmp.name)
    try:
        image.save(tmp_p)
        tmp_p.replace(out)
    finally:
        tmp_p.unlink(missing_ok=True)
