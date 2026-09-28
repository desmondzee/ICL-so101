"""Keyframe edits that keep the scene fixed: edit one robot frame, then paste the original pixels back outside the robot.

The editor (Nano Banana) redraws the whole image, which can move, duplicate or reframe objects. Here only the robot's
region, found per frame by a VLM, plus a strip to the edge the arm comes from, is taken from the edit; everything else
is the robot frame's own pixels, so objects cannot move or appear except where the robot was.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from schema.annotate import _ask

EDITOR = "gemini-3-pro-image"  # Nano Banana Pro
SIZE = "1K"  # the smallest Pro accepts; outputs are scaled to the 640x480 robot frames
MARGIN = 0.06  # grow the robot box by this share of the frame
FEATHER = 12  # px blur of the mask edge

BOX_SCHEMA = {"type": "object", "required": ["box_2d"], "properties": {"box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}}}
BOX_PROMPT = ("Give the bounding box of the whole robot arm in this image (every part: base, links, gripper, cables and mount), "
              "as box_2d [ymin, xmin, ymax, xmax] in 0-1000 of the image.")


def edit(images: list[Path], prompt: str, out: Path, model: str = EDITOR, size: str = SIZE) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    reply = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=p.read_bytes(), mime_type="image/jpeg") for p in images] + [prompt],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio="4:3", image_size=size)),
    )
    for part in reply.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            tmp = out.with_suffix(".raw")
            tmp.write_bytes(part.inline_data.data)
            Image.open(tmp).convert("RGB").save(out, quality=95)
            tmp.unlink()
            return
    raise RuntimeError("no image returned")


def robot_box(llm, frame_path: Path) -> list[int]:
    def check(d):
        b = d.get("box_2d") or []
        return [] if len(b) == 4 and b[0] < b[2] and b[1] < b[3] else ["box_2d must be [ymin, xmin, ymax, xmax] with min < max"]
    return _ask(llm, [frame_path, BOX_PROMPT], BOX_SCHEMA, check)["box_2d"]


OBJ_SCHEMA = {"type": "object", "required": ["objects"], "properties": {"objects": {"type": "array", "items": {
    "type": "object", "required": ["name", "box_2d"], "properties": {"name": {"type": "string"}, "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}}}}}}


def object_boxes(llm, frame_path: Path, names: list[str]) -> list[list[int]]:
    """Boxes of the listed objects in this frame (not the robot), as box_2d."""
    prompt = ("Give the bounding box of each of these objects in the image, as box_2d [ymin, xmin, ymax, xmax] in 0-1000: "
              + "; ".join(names) + ". Skip an object that is not visible. Do not box the robot or the table.")
    return [o["box_2d"] for o in _ask(llm, [frame_path, prompt], OBJ_SCHEMA, lambda d: [])["objects"]
            if len(o.get("box_2d", [])) == 4 and o["box_2d"][0] < o["box_2d"][2] and o["box_2d"][1] < o["box_2d"][3]]


def mask(size: tuple[int, int], box: list[int], edge: str, keep: list[list[int]] = ()) -> Image.Image:
    """Robot box grown by MARGIN, joined to the entry edge (the forearm comes in from there) except over the objects in
    `keep` (their own pixels stay), feathered."""
    w, h = size
    y0, x0, y1, x1 = box
    m = MARGIN * 1000
    y0, x0, y1, x1 = max(0, y0 - m), max(0, x0 - m), min(1000, y1 + m), min(1000, x1 + m)
    if edge == "far":
        y0 = 0
    elif edge == "near":
        y1 = 1000
    elif edge == "left":
        x0 = 0
    else:
        x1 = 1000
    im = Image.new("L", size, 0)
    dr = ImageDraw.Draw(im)
    dr.rectangle([x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000], fill=255)
    for ky0, kx0, ky1, kx1 in keep:  # objects in the strip keep their pixels
        k = 0.02 * 1000
        dr.rectangle([(kx0 - k) * w / 1000, (ky0 - k) * h / 1000, (kx1 + k) * w / 1000, (ky1 + k) * h / 1000], fill=0)
    by0, bx0, by1, bx1 = [max(0, min(1000, v)) for v in (box[0] - m, box[1] - m, box[2] + m, box[3] + m)]
    dr.rectangle([bx0 * w / 1000, by0 * h / 1000, bx1 * w / 1000, by1 * h / 1000], fill=255)  # the robot itself is always replaced
    return im.filter(ImageFilter.GaussianBlur(FEATHER))


DIFF_THRESHOLD = 45  # colour change that counts as edited
REFRAMED_MEDIAN = 22  # median change outside the robot area above this means the edit shifted or redrew the scene


class Reframed(RuntimeError):
    pass


SAM_MODEL = "facebook/sam2.1-hiera-small"  # Apache 2.0, runs locally (MPS/CPU)
_sam = None


SAM3_MODEL = "facebook/sam3"  # text-prompted segmentation (gated on the Hub; needs HF access)
_sam3 = None


ROBOT_TEXTS = ("robot arm", "cable", "robot base", "shadow")  # arm first; cables and mount only where they touch the arm


def robot_mask_sam3(robot_frame: Path, text: str = "robot arm", keep_score: float = 0.5, box: list[int] | None = None):
    """Pixels of the one robot in the scene, from SAM 3: the best 'robot arm' detection (with the VLM robot box as a
    tie-breaker), grown by any other arm, cable, base or shadow detection that touches it, then only the connected
    region. Arm-like things elsewhere (a desk lamp) are not the robot and are left alone."""
    import numpy as np
    from scipy import ndimage

    arms = [a for t in (text, "robotic arm", "robot") for a in _sam3_instances(robot_frame, t, 0.3, scored=True)]
    if not arms:
        raise RuntimeError(f"SAM 3 found no {text}")
    h, w = arms[0][0].shape

    def in_box(m):
        if not box:
            return 0.0
        y0, x0, y1, x1 = box
        sub = m[int(y0 * h / 1000):int(y1 * h / 1000), int(x0 * w / 1000):int(x1 * w / 1000)]
        return sub.sum() / max(1, m.sum())

    best = max(arms, key=lambda ms: ms[1] + 0.5 * in_box(ms[0]))[0]  # the one robot
    robot = best.copy()
    for m, _ in arms:  # the same robot found under another name ("robot" often covers more of it than "robot arm")
        if (m & best).sum() > 0.2 * min(m.sum(), best.sum()):
            robot |= m
    for extra in ROBOT_TEXTS:
        for m, _ in _sam3_instances(robot_frame, extra, keep_score, scored=True):
            if m is best:
                continue
            if (m & ndimage.binary_dilation(robot, iterations=6)).any():
                robot |= m
    lab, _ = ndimage.label(ndimage.binary_dilation(robot, iterations=3))
    main = lab == lab[ndimage.binary_dilation(best, iterations=3)].max()  # the region that holds the best detection
    return robot & main


def _sam3_text(robot_frame: Path, text: str, keep_score: float, best_if_none: bool):
    import numpy as np

    ms = _sam3_instances(robot_frame, text, keep_score, best_if_none)
    if not ms:
        raise RuntimeError(f"SAM 3 found no {text}")
    return np.any(ms, axis=0)


def _sam3_instances(robot_frame: Path, text: str, keep_score: float, best_if_none: bool = False, scored: bool = False) -> list:
    import numpy as np
    import torch
    from transformers import Sam3Model, Sam3Processor

    global _sam3
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if _sam3 is None:
        _sam3 = (Sam3Processor.from_pretrained(SAM3_MODEL), Sam3Model.from_pretrained(SAM3_MODEL).to(dev))
    proc, model = _sam3
    img = Image.open(robot_frame).convert("RGB")
    inp = proc(images=img, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5, target_sizes=inp["original_sizes"].tolist())[0]
    if not len(res["masks"]):
        return []
    scores = res["scores"].cpu().numpy()
    keep = scores >= keep_score
    if not keep.any():
        if not best_if_none:
            return []
        keep = scores == scores.max()
    out = [(m.cpu().numpy() > 0, float(sc)) for m, sc, k in zip(res["masks"], scores, keep) if k]
    return out if scored else [m for m, _ in out]


def robot_mask(robot_frame: Path, box: list[int], grow: int = 60):
    """Exact robot pixels from SAM 2.1, prompted with the VLM's box grown by `grow` (0-1000 units): the box only has to
    contain the robot; SAM follows its outline, cables and base included. Cached next to the frame."""
    import numpy as np

    cache = robot_frame.with_name(robot_frame.stem + "_robotmask.png")
    if cache.exists() and (m := np.asarray(Image.open(cache)) > 0).mean() > 0.005:  # an empty cache is a failed run
        return m
    try:  # SAM 3 from text first; SAM 2.1 with the VLM box if SAM 3 is unavailable or finds nothing
        m = robot_mask_sam3(robot_frame, box=box)
        if m.mean() < 0.005:
            raise RuntimeError("SAM 3 mask is empty")
        Image.fromarray((m * 255).astype("uint8")).save(cache)
        return m
    except Exception as e:
        import logging
        logging.info("SAM 3 failed on %s (%s); using SAM 2.1 with the box", robot_frame.name, e)
    global _sam
    import torch
    from transformers import Sam2Model, Sam2Processor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if _sam is None:
        _sam = (Sam2Processor.from_pretrained(SAM_MODEL), Sam2Model.from_pretrained(SAM_MODEL).to(dev))
    proc, model = _sam
    img = Image.open(robot_frame).convert("RGB")
    w, h = img.size
    y0, x0, y1, x1 = box
    xyxy = [[[max(0, x0 - grow) * w / 1000, max(0, y0 - grow) * h / 1000, min(1000, x1 + grow) * w / 1000, min(1000, y1 + grow) * h / 1000]]]
    inp = proc(images=img, input_boxes=xyxy, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    m = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0][0, 0].numpy() > 0
    Image.fromarray((m * 255).astype("uint8")).save(cache)
    return m


def change_mask(robot_frame: Path, edited: Path, box: list[int], keep: list[list[int]] = ()) -> Image.Image:
    """Where the edit changed the robot frame, keeping the changed regions connected to the robot (the arm and hand,
    wherever it was drawn), plus the robot box; object boxes outside the robot box keep their own pixels."""
    import numpy as np
    from scipy import ndimage

    base = Image.open(robot_frame).convert("RGB")
    new = Image.open(edited).convert("RGB").resize(base.size, Image.LANCZOS)
    w, h = base.size
    diff = np.abs(np.asarray(base, float) - np.asarray(new, float)).max(axis=2)
    changed = ndimage.binary_opening(diff > DIFF_THRESHOLD, iterations=3)  # drop texture noise
    changed = ndimage.binary_closing(changed, iterations=2)
    m = MARGIN * 1000
    y0, x0, y1, x1 = [int(v) for v in (max(0, box[0] - m) * h / 1000, max(0, box[1] - m) * w / 1000, min(1000, box[2] + m) * h / 1000, min(1000, box[3] + m) * w / 1000)]
    try:  # exact robot pixels (SAM), widened; the rectangle is the fallback
        robot = ndimage.binary_dilation(robot_mask(robot_frame, box), iterations=8)
    except Exception:
        robot = np.zeros((h, w), bool)
        robot[y0:y1, x0:x1] = True
    outside = ~ndimage.binary_dilation(robot | changed, iterations=10)
    if outside.any() and np.median(diff[outside]) > REFRAMED_MEDIAN:
        raise Reframed(f"median change outside the robot is {np.median(diff[outside]):.0f} (reframed or redrawn)")
    labels, _ = ndimage.label(changed | robot)
    region = np.isin(labels, np.unique(labels[robot]))
    for ky0, kx0, ky1, kx1 in keep:
        k = 20
        obj = np.zeros((h, w), bool)
        obj[int(max(0, ky0 - k) * h / 1000):int(min(1000, ky1 + k) * h / 1000), int(max(0, kx0 - k) * w / 1000):int(min(1000, kx1 + k) * w / 1000)] = True
        region &= ~(obj & ~robot)
    region = ndimage.binary_dilation(region, iterations=4)
    return Image.fromarray((region * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(FEATHER / 2))


def hand_crop(human_frame: Path, box: list[int], edge: str, out: Path) -> None:
    """The forearm and hand of an edited frame (its robot mask area), for use as the hand reference."""
    im = Image.open(human_frame).convert("RGB")
    mfile = human_frame.with_name(human_frame.stem + "_mask.png")
    bbox = (Image.open(mfile) if mfile.exists() else mask(im.size, box, edge)).getbbox()
    im.crop(bbox).save(out, quality=95)


def composite(robot_frame: Path, edited: Path, box: list[int], edge: str, out: Path, keep: list[list[int]] = ()) -> None:
    base = Image.open(robot_frame).convert("RGB")
    new = Image.open(edited).convert("RGB").resize(base.size, Image.LANCZOS)
    m = change_mask(robot_frame, edited, box, keep)
    m.save(out.with_name(out.stem + "_mask.png"))
    Image.composite(new, base, m).save(out, quality=95)


LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"  # LaMa (Apache 2.0), TorchScript
_lama = None


def inpaint(image: Image.Image, mask_bool) -> Image.Image:
    """Fill the masked pixels with what is plausibly behind them (LaMa, local, free)."""
    global _lama
    import numpy as np
    import torch

    if _lama is None:
        path = Path.home() / ".cache" / "lama" / "big-lama.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(LAMA_URL, str(path))
        _lama = torch.jit.load(str(path), map_location="cpu").eval()
    img = np.asarray(image.convert("RGB"), np.float32) / 255
    h, w = img.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    t_img = torch.from_numpy(np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")).permute(2, 0, 1)[None]
    t_mask = torch.from_numpy(np.pad(mask_bool.astype(np.float32), ((0, ph), (0, pw)), mode="constant"))[None, None]
    with torch.no_grad():
        out = _lama(t_img, t_mask)[0].permute(1, 2, 0).numpy()[:h, :w]
    return Image.fromarray(np.clip(out * 255, 0, 255).astype("uint8"))


def remove_robot(robot_frame: Path, box: list[int], out: Path, dilate: int = 16) -> None:
    """The robot frame with the robot removed: SAM robot mask, widened, filled by LaMa. No generative edit, no hand."""
    from scipy import ndimage

    m = ndimage.binary_dilation(robot_mask(robot_frame, box), iterations=dilate)
    Image.fromarray((m * 255).astype("uint8")).save(out.with_name(out.stem + "_mask.png"))
    inpaint(Image.open(robot_frame), m).save(out, quality=95)
