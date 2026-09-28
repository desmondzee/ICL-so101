"""Small runtime evidence returned by the remote SO-101 inference workers."""
import hashlib
import json
import time


def finish_call(model, request, result, started):
    cfg = model.job_config
    kind = "reset" if request.get("reset") else "cache" if request.get("compute_kv_cache") else "infer"
    runtime = {
        "kind": kind,
        "use_icl_model": bool(getattr(model, "use_icl_model", False)),
        "use_icl": bool(getattr(model, "use_icl", False)),
        "icl_guidance_scale": getattr(model, "icl_guidance_scale", None),
        "video_guidance_scale": getattr(model, "video_guidance_scale", None),
        "configured_text_guidance_scale": getattr(cfg, "guidance_scale", None),
        "configured_action_guidance_scale": getattr(cfg, "action_guidance_scale", None),
        "target_text_cfg_active": getattr(model, "target_text_cfg_active", None),
        "use_cfg": getattr(model, "use_cfg", None),
        "elapsed_s": round(time.monotonic() - started, 3),
        "camera_keys": list(cfg.obs_cam_keys),
        "channels": list(cfg.used_action_channel_ids),
        "height": cfg.height,
        "width": cfg.width,
        "video_inference_steps": cfg.num_inference_steps,
        "action_inference_steps": cfg.action_num_inference_steps,
        "action_per_frame": cfg.action_per_frame,
        "frame_chunk_size": cfg.frame_chunk_size,
        "norm_stats_sha256": hashlib.sha256(json.dumps(cfg.norm_stat, sort_keys=True).encode()).hexdigest(),
    }
    if request.get("reset"):
        runtime["prompt"] = request.get("prompt")
    if isinstance(result, dict) and "action" in result:
        runtime["action_shape"] = list(result["action"].shape)
    mask = getattr(model, "action_mask", None)
    if mask is not None:
        runtime["active_mask_channels"] = mask.nonzero().flatten().tolist()
    print("[so101-runtime] " + json.dumps(runtime), flush=True)
    return {**(result or {}), "_so101_runtime": runtime}
