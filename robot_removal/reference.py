"""Build the robot-free reference image fed to Qwen as image_1.

LaMa prefill makes the masked region plausibly background-colored before the
edit, which stops Qwen from regenerating the robot it saw in the reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
_lama = None


def _model(device: str):
    global _lama
    import torch

    if _lama is None:
        path = Path.home() / ".cache" / "lama" / "big-lama.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(LAMA_URL, str(path))
        _lama = torch.jit.load(str(path), map_location="cpu").eval()
        if device == "cuda":
            _lama = _lama.cuda()
    return _lama


def lama_inpaint(image: Image.Image, mask: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Fill masked pixels with LaMa (big-lama torchscript). Returns RGB uint8."""
    import torch

    m = _model(device)
    a = np.asarray(image.convert("RGB"), np.float32) / 255
    h, w = a.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    t_img = torch.from_numpy(np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="reflect")).permute(2, 0, 1)[None].to(device)
    t_mask = torch.from_numpy(np.pad(mask.astype(np.float32), ((0, ph), (0, pw)), mode="constant"))[None, None].to(device)
    with torch.no_grad():
        out = m(t_img, t_mask)[0].permute(1, 2, 0).float().cpu().numpy()[:h, :w]
    return np.clip(out * 255, 0, 255).astype("uint8")


def diffuse_fill(image: np.ndarray, mask: np.ndarray, iters: int = 300) -> np.ndarray:
    """Harmonic fill fallback: mask pixels relax to neighbor means."""
    from scipy import ndimage

    a = image.astype(np.float32)
    m = mask.astype(bool)
    ring = ndimage.binary_dilation(m, iterations=8) & ~m
    a[m] = np.median(a[ring], axis=0) if ring.any() else 127
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32) / 4
    for _ in range(iters):
        for ch in range(3):
            blur = ndimage.convolve(a[:, :, ch], k, mode="nearest")
            a[:, :, ch] = np.where(m, blur, a[:, :, ch])
    return a.astype("uint8")


def deshadow(ref: np.ndarray, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove low-frequency silhouettes from the filled region.

    LaMa smears the masked object's average color into the fill, leaving a
    robot-shaped luminance bump that Qwen then preserves. Replace the fill's
    low-frequency luminance with a harmonic field interpolated from the
    boundary ring; keep LaMa's high-frequency texture and chroma.
    """
    from scipy import ndimage

    ref = ref.astype(np.float32)
    img = image.astype(np.float32)
    lum_ref = ref.mean(2)
    lum_img = img.mean(2)
    smooth = diffuse_fill(lum_img[..., None], mask, 400)[:, :, 0]
    base = ndimage.gaussian_filter(lum_ref, 12)
    flat = lum_ref - base + np.where(mask, smooth, base)  # inside: smooth field + texture
    delta = np.where(mask, flat - lum_ref, 0.0)
    out = np.clip(ref + delta[..., None], 0, 255)
    return out.astype("uint8")


def build_reference(image: Image.Image, mask: np.ndarray, method: str = "lama", device: str = "cpu") -> np.ndarray:
    if not mask.any():
        return np.asarray(image.convert("RGB"))
    orig = np.asarray(image.convert("RGB"))
    if method == "lama":
        try:
            return deshadow(lama_inpaint(image, mask, device), orig, mask)
        except Exception:
            pass  # fall back to diffusion fill
    return diffuse_fill(orig, mask)
