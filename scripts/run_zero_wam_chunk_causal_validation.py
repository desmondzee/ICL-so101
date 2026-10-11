#!/usr/bin/env python3
"""Run exact deployment-path, one-chunk validation for one checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
ZERO_WAM = REPO / "third_party" / "Zero-WAM"
sys.path[:0] = [str(REPO), str(ZERO_WAM)]

from zero_wam.chunk_causal_validation import (  # noqa: E402
    CHUNK_CONTROLS,
    PREFIX_CONTROLS,
    aggregate,
    assert_payload_has_no_targets,
    build_manifest,
    canonical_sha256,
    inference_payload,
    score_chunk,
    validate_manifest_files,
)
from zero_wam.so101_a100_preflight import atomic_write_json  # noqa: E402
from zero_wam.so101_a100_train import load_so101_norm_stat  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--loader-root", required=True)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _assembled_model(base_model, checkpoint, temporary_root):
    base_model, checkpoint = Path(base_model), Path(checkpoint)
    transformer = checkpoint / "transformer"
    if not transformer.is_dir():
        transformer = checkpoint
    if not (transformer / "config.json").is_file():
        raise FileNotFoundError(f"missing transformer checkpoint: {transformer}")
    assembled = Path(temporary_root) / "model"
    assembled.mkdir()
    for child in base_model.iterdir():
        if child.name != "transformer":
            (assembled / child.name).symlink_to(child.resolve(),
                                                 target_is_directory=child.is_dir())
    (assembled / "transformer").symlink_to(
        transformer.resolve(), target_is_directory=True)
    return assembled


def _load_server(model_path, device):
    if device != "cuda:0":
        raise ValueError(
            "use CUDA_VISIBLE_DEVICES to select a GPU; runtime device must be cuda:0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.modules.icl_model import WanICLTransformer3DModel
    import wan_va.wan_va_server as server_module

    load_evidence = {}

    def checked_transformer(path, torch_dtype, torch_device, **kwargs):
        del kwargs
        model, info = WanICLTransformer3DModel.from_pretrained(
            path, torch_dtype=torch_dtype, output_loading_info=True)
        problems = {key: value for key, value in info.items() if value}
        if problems:
            raise RuntimeError(f"inexact checkpoint load: {problems}")
        load_evidence.update(
            model_class=type(model).__name__, loading_info=info)
        return model.to(device=torch_device, dtype=torch_dtype)

    server_module.load_transformer = checked_transformer
    config = VA_CONFIGS["robotwin"]
    config.model_path = str(model_path)
    config.rank = config.local_rank = 0
    config.world_size = 1
    config.save_root = tempfile.mkdtemp(prefix="so101-chunk-validation-")
    config.obs_cam_keys = [
        "observation.images.front", "observation.images.wrist"]
    config.height, config.width = 224, 288
    config.icl_width = 448
    config.frame_chunk_size = 2
    config.action_per_frame = 8
    config.action_dim = 30
    config.num_inference_steps = 50
    config.action_num_inference_steps = 50
    config.used_action_channel_ids = [0, 1, 2, 3, 4, 28]
    inverse = [len(config.used_action_channel_ids)] * config.action_dim
    for index, channel in enumerate(config.used_action_channel_ids):
        inverse[channel] = index
    config.inverse_used_action_channel_ids = inverse
    config.norm_stat = load_so101_norm_stat(
        REPO / "data/hf_bucket_ICL-so101/zero_wam_loader_v1"
        / "val/meta/action_stats.json")
    init_distributed(1, 0, 0)
    model = server_module.VA_Server(config)
    return model, load_evidence, config


def _video_frames(path, indices):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    wanted, output = set(indices), {}
    position = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if position in wanted:
            output[position] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            wanted.remove(position)
        position += 1
    capture.release()
    if wanted:
        raise ValueError(f"{path} missing frames {sorted(wanted)}")
    return output


def _episode_arrays(path):
    table = pq.read_table(path, columns=["action", "observation.state"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(
        table["observation.state"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 6 or states.shape != actions.shape:
        raise ValueError(f"unexpected episode arrays: {actions.shape}/{states.shape}")
    return actions, states


def _observation(front, wrist, state):
    return {
        "observation.images.front": front,
        "observation.images.wrist": wrist,
        "observation.state": np.asarray(state, dtype=np.float32),
    }


def _predict_one(model, sample):
    actions, states = _episode_arrays(sample["robot_parquet"])
    if len(actions) < PREFIX_CONTROLS + CHUNK_CONTROLS:
        raise ValueError(f"episode too short: {sample['sample_id']}")
    indices = sample["observation_frame_indices"]
    front = _video_frames(sample["robot_front_video"], indices)
    wrist = _video_frames(sample["robot_wrist_video"], indices)
    observations = [
        _observation(front[index], wrist[index], states[index])
        for index in indices
    ]
    prefix = actions[:PREFIX_CONTROLS].copy()
    payload = inference_payload(sample, observations, prefix)
    assert_payload_has_no_targets(payload)

    seed = payload["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.infer({
        "reset": True,
        "prompt": payload["target_text"],
        "use_icl": True,
        "icl_latent_path": payload["human_latent"],
        "video_guidance_scale": 5.0,
    })

    # Deployment's first chunk observes frame zero and executes only its second
    # eight-control frame. Its prediction is intentionally discarded; recorded
    # executed actions replace it before the prefix is cached.
    model.infer({"obs": payload["observations"][0]})
    history = np.zeros((6, 2, 8), dtype=np.float32)
    history[:, 1, :] = prefix.T
    normalized_history = model.preprocess_action(history)
    normalized_history[:, ~model.action_mask.cpu()] = 0
    normalized_history[:, :, 0] = 0
    model.last_predicted_actions = normalized_history.to(
        model.device, model.dtype)
    model.infer({
        "obs": payload["observations"][1:],
        "compute_kv_cache": True,
        "state": history,
    })

    result = model.infer({"obs": payload["observations"][-1]})
    generated = np.asarray(result["action"], dtype=np.float32)
    if generated.shape != (6, 2, 8):
        raise ValueError(f"unexpected generated action shape {generated.shape}")
    predicted = generated.transpose(1, 2, 0).reshape(CHUNK_CONTROLS, 6)

    # Targets enter only here, after inference is complete.
    start, end = sample["target_control_start"], sample["target_control_end"]
    target = actions[start:end]
    persistence = np.repeat(prefix[-1:], CHUNK_CONTROLS, axis=0)
    return {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "seed": seed,
        "prefix_sha256": hashlib_array(prefix),
        "prediction_sha256": hashlib_array(predicted),
        "target_sha256": hashlib_array(target),
        "metrics": score_chunk(predicted, target, persistence),
    }


def hashlib_array(value):
    value = np.ascontiguousarray(value)
    return __import__("hashlib").sha256(value.view(np.uint8)).hexdigest()


def main():
    args = parse_args()
    manifest = (
        json.loads(Path(args.manifest).read_text())
        if args.manifest else
        build_manifest(args.loader_root, args.latent_root)
    )
    expected_hash = manifest.pop("manifest_sha256")
    actual_hash = canonical_sha256(manifest)
    manifest["manifest_sha256"] = expected_hash
    if actual_hash != expected_hash:
        raise ValueError("chunk-validation manifest hash mismatch")
    validate_manifest_files(manifest)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    selected = manifest["samples"][args.sample_offset:]
    if args.max_samples is not None:
        selected = selected[:args.max_samples]
    with tempfile.TemporaryDirectory(prefix="so101-model-assembly-") as temp:
        assembled = _assembled_model(args.base_model, args.checkpoint, temp)
        model, load_evidence, config = _load_server(assembled, args.device)
        rows = []
        for index, sample in enumerate(selected, 1):
            print(
                f"[{index}/{len(selected)}] {sample['sample_id']}",
                flush=True)
            rows.append(_predict_one(model, sample))
            torch.cuda.empty_cache()
    report = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": manifest["protocol"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_transformer_config_sha256": __import__("hashlib").sha256(
            (Path(args.checkpoint) / "transformer/config.json").read_bytes()
        ).hexdigest(),
        "model_load": load_evidence,
        "inference": {
            "video_steps": int(config.num_inference_steps),
            "action_steps": int(config.action_num_inference_steps),
            "frame_chunk_size": int(config.frame_chunk_size),
            "action_per_frame": int(config.action_per_frame),
            "used_action_channels": list(config.used_action_channel_ids),
            "target_visibility": "scorer only after generation",
        },
        "sample_offset": args.sample_offset,
        "num_selected_samples": len(selected),
        "samples": rows,
        "aggregate": aggregate(rows),
    }
    atomic_write_json(output, report)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
