"""API-format ComfyUI workflow builder for robot removal.

One submission per input image:
  LoadImage -> SAM3_Detect x2 (arm prompts / extra prompts, individual masks)
          -> SaveImage mask candidates
  LoadImage -> TextEncodeQwenImage21 (resolution=0, images.image_1=original)
          -> KSampler (full denoise) -> VAEDecode -> SaveImage qwen output

Mask selection, dilation, compositing and the 480x320 transform happen
in Python (see masks.py / postprocess.py) so preservation is exact.
"""

from __future__ import annotations

import re


def safe_id(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", key)


def build_seg_prompt(
    image_name: str,
    prefix: str,
    cfg: dict,
    robot_box_px: tuple[int, int, int, int] | None = None,
) -> dict:
    """Phase A: SAM3 candidate masks -> SaveImage. robot_box_px = (x,y,w,h) or None."""
    sam3, models = cfg["sam3"], cfg["models"]
    p = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": models["sam3"]}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": sam3["arm_prompt"]}},
        "5": {
            "class_type": "SAM3_Detect",
            "inputs": {
                "model": ["1", 0],
                "image": ["2", 0],
                "conditioning": ["3", 0],
                "threshold": sam3["arm_threshold"],
                "refine_iterations": sam3["refine_iterations"],
                "individual_masks": True,
            },
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": sam3["extra_prompt"]}},
        "7": {
            "class_type": "SAM3_Detect",
            "inputs": {
                "model": ["1", 0],
                "image": ["2", 0],
                "conditioning": ["6", 0],
                "threshold": sam3["extra_threshold"],
                "refine_iterations": sam3["refine_iterations"],
                "individual_masks": True,
            },
        },
        "8": {"class_type": "MaskToImage", "inputs": {"mask": ["5", 0]}},
        "9": {"class_type": "MaskToImage", "inputs": {"mask": ["7", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": f"{prefix}_arm"}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": f"{prefix}_ext"}},
    }
    if robot_box_px is not None:
        x, y, w, h = robot_box_px
        p["4"] = {"class_type": "PrimitiveBoundingBox", "inputs": {"x": x, "y": y, "width": w, "height": h}}
        p["5"]["inputs"]["bboxes"] = ["4", 0]
        p["7"]["inputs"]["bboxes"] = ["4", 0]
    return p


def build_edit_prompt(image_name: str, prefix: str, cfg: dict, seed: int | None = None) -> dict:
    """Phase B: Qwen full-denoise edit conditioned on the robot-free reference."""
    qwen, models = cfg["qwen"], cfg["models"]
    return {
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "20": {"class_type": "UNETLoader", "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}},
        "21": {"class_type": "CLIPLoader", "inputs": {"clip_name": models["clip"], "type": "qwen_image", "device": "default"}},
        "22": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
        "23": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["21", 0],
                "prompt": qwen["prompt"],
                "negative_prompt": qwen["negative_prompt"],
                "vae": ["22", 0],
                "resolution": qwen["resolution"],
                "images.image_1": ["2", 0],
            },
        },
        "24": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["20", 0],
                "positive": ["23", 0],
                "negative": ["23", 1],
                "latent_image": ["23", 2],
                "seed": seed if seed is not None else qwen["seed"],
                "control_after_generate": "fixed",
                "steps": qwen["steps"],
                "cfg": qwen["cfg"],
                "sampler_name": qwen["sampler"],
                "scheduler": qwen["scheduler"],
                "denoise": qwen["denoise"],
            },
        },
        "25": {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": ["22", 0]}},
        "26": {"class_type": "SaveImage", "inputs": {"images": ["25", 0], "filename_prefix": f"{prefix}_qwen"}},
    }


def box_2d_to_px(box_2d: list[int] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    """[ymin, xmin, ymax, xmax] in 0-1000 -> (x, y, w, h) pixels."""
    if not box_2d:
        return None
    y0, x0, y1, x1 = box_2d
    x, y = int(x0 * width / 1000), int(y0 * height / 1000)
    return x, y, max(1, int(x1 * width / 1000) - x), max(1, int(y1 * height / 1000) - y)
