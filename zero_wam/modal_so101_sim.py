"""Zero-WAM latent encoding for the converted SO-101 sim corpus on Modal.

Uploads `data/zero_wam_sim` to the `zero-wam-so101-sim` volume at /sim/data, then encodes
robot front+wrist latents (stride-2, 256x256, Wan VAE streaming, mu only, bf16) and paired
human-demo latents (12 fps, 320x480) in the released .pth layout, sharded across 8 H100
workers by deterministic task partition. Followed by a CPU latent inventory and a
loader-smoke pass instantiating upstream `MultiICLLeRobotLatentDataset` per split.

    modal run zero_wam/modal_so101_sim.py::prepare        # volume upload + preflight
    modal run zero_wam/modal_so101_sim.py::encode         # 8-shard latent encoding
    modal run zero_wam/modal_so101_sim.py::verify         # /sim/reports/latent_inventory.json
    modal run zero_wam/modal_so101_sim.py::loader_smoke   # /sim/reports/loader_smoke.json

Secrets: Modal token from environment (MODAL_TOKEN_ID/MODAL_TOKEN_SECRET). The checkpoint at
/weights/zero-wam-pretrain on volume `zero-wam-weights` is required and never downloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "zero-wam-so101-sim"
LOCAL_REPO = Path(__file__).resolve().parents[1] / "third_party" / "Zero-WAM"
LOCAL_DATA = Path(__file__).resolve().parents[1] / "data" / "zero_wam_sim"
REMOTE_REPO = "/opt/zero-wam"
WEIGHTS = "/weights"
MODEL_DIR = f"{WEIGHTS}/zero-wam-pretrain"
SIM = "/sim"
DATA_DIR = f"{SIM}/data"
REPORTS = f"{SIM}/reports"

SPLITS = ("train", "val")
CAMERAS = ("observation.images.front", "observation.images.wrist")
NUM_SHARDS = 8
FRAME_STRIDE = 2          # robot: every 2nd source frame -> 8 controls per latent frame
HEIGHT, WIDTH = 256, 256  # robot latent source geometry (va_demo_cfg / settled contract)
ICL_HEIGHT, ICL_WIDTH, ICL_FPS = 320, 480, 12  # pinned robotwin ICL geometry
VAE_SPATIAL = 16          # Wan 2.2 VAE spatial downsample (upstream robotwin: 320x448 -> 20x28)
VAE_TEMPORAL = 4          # latent temporal frames = (len(ids)-1)//4 + 1
VAE_CHANNELS = 48

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name("zero-wam-weights")
data_volume = modal.Volume.from_name("zero-wam-so101-sim", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel ninja packaging",
        "python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 "
        "--index-url https://download.pytorch.org/whl/cu126",
    )
    .add_local_file(str(LOCAL_REPO / "requirements.txt"),
                    "/tmp/zero-wam-requirements.txt", copy=True)
    .run_commands(
        "sed -e '/^lerobot==/d' -e '/^flash_attn$/d' /tmp/zero-wam-requirements.txt "
        "> /tmp/req.txt",
        "MAX_JOBS=4 python -m pip install -r /tmp/req.txt --no-build-isolation",
        "python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/"
        "releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-"
        "cp312-linux_x86_64.whl'",
        "python -m pip install --no-deps lerobot==0.3.3",
        "python -m pip install 'datasets<3.7' 'huggingface-hub<1.0' jsonlines av "
        "decord==0.6.0 pyarrow pandas",
    )
    .add_local_dir(str(LOCAL_REPO), remote_path=REMOTE_REPO, copy=True,
                   ignore=[".git", ".git/**", "data/**", "checkpoints/**", "results/**",
                           "**/__pycache__/**", "**/*.pyc"])
    .env({"PYTHONPATH": REMOTE_REPO, "MODEL_PATH": MODEL_DIR,
          "HF_HOME": f"{SIM}/hf_home", "TOKENIZERS_PARALLELISM": "false"})
)


# ----- pure helpers (unit-testable without Modal/GPU) -------------------------------------------------


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_partition(tasks: list[str], num_shards: int = NUM_SHARDS) -> dict[int, list[str]]:
    """Deterministic balanced partition: sorted task names round-robin over shards."""
    out = {i: [] for i in range(num_shards)}
    for i, task in enumerate(sorted(tasks)):
        out[i % num_shards].append(task)
    return out


def robot_frame_ids(length: int) -> list[int]:
    """Every FRAME_STRIDE-th source frame of a whole episode, trimmed to 4k+1 for the VAE."""
    ids = list(range(0, int(length), FRAME_STRIDE))
    return ids[: (len(ids) - 1) // VAE_TEMPORAL * VAE_TEMPORAL + 1]


def human_frame_ids(num_frames: int, src_fps: float, icl_fps: int = ICL_FPS) -> list[int]:
    """ICL resample to icl_fps by source fps, trimmed to 4k+1."""
    import numpy as np

    n_out = max(1, int(round(num_frames / src_fps * icl_fps)))
    ids = np.unique(np.clip(np.floor(np.arange(n_out) * src_fps / icl_fps).astype(np.int64),
                            0, num_frames - 1)).tolist()
    return ids[: (len(ids) - 1) // VAE_TEMPORAL * VAE_TEMPORAL + 1]


def latent_frames(num_ids: int) -> int:
    return (num_ids - 1) // VAE_TEMPORAL + 1


def robot_latent_relpath(task: str, episode: int, length: int, camera: str) -> str:
    return (f"{task}/latents/chunk-000/{camera}/"
            f"episode_{episode:06d}_0_{length}.pth")


def human_latent_relpath(split: str, task: str, episode: int) -> str:
    sample = f"{task}__episode_{episode:03d}"
    return f"human_latents/so101/run_{split}/samples/{sample}/generated_video.pth"


def robot_video_sha(task_root, camera: str, episode: int) -> str:
    return sha256_file(
        Path(task_root) / "videos" / "chunk-000" / camera / f"episode_{episode:06d}.mp4")


def validate_robot_payload(payload: dict, frame_ids: list[int], source_sha: str) -> str | None:
    """Returns None if valid else a reason string — used both for skip and for verify."""
    required = ("latent", "latent_num_frames", "latent_height", "latent_width",
                "video_num_frames", "video_height", "video_width", "task",
                "local_instruction", "local_instruction_emb", "text", "frame_ids",
                "start_frame", "end_frame", "fps", "ori_fps", "source_video_sha256")
    missing = [k for k in required if k not in payload]
    if missing:
        return f"missing keys {missing}"
    if list(payload["frame_ids"]) != list(frame_ids):
        return "frame_ids mismatch"
    if payload.get("source_video_sha256") != source_sha:
        return "source_video_sha256 mismatch"
    f, h, w = (int(payload["latent_num_frames"]), int(payload["latent_height"]),
               int(payload["latent_width"]))
    lat = payload["latent"]
    if tuple(lat.shape) != (f * h * w, VAE_CHANNELS):
        return f"latent shape {tuple(lat.shape)} != {(f * h * w, VAE_CHANNELS)}"
    if h != HEIGHT // VAE_SPATIAL or w != WIDTH // VAE_SPATIAL:
        return f"latent geometry {h}x{w} != {HEIGHT // VAE_SPATIAL}x{WIDTH // VAE_SPATIAL}"
    if f != latent_frames(len(frame_ids)):
        return f"latent_num_frames {f} != {latent_frames(len(frame_ids))}"
    if str(lat.dtype) != "torch.bfloat16":
        return f"latent dtype {lat.dtype}"
    import torch
    if not torch.isfinite(lat.float()).all():
        return "latent not finite"
    return None


def validate_human_payload(payload: dict, frame_ids: list[int], source_sha: str,
                           local_instruction: str | None = None,
                           local_instruction_sha256: str | None = None) -> str | None:
    required = ("latent", "latent_num_frames", "latent_height", "latent_width",
                "text_emb", "text", "local_instruction", "local_instruction_sha256",
                "frame_ids", "fps", "ori_fps", "source_video_sha256")
    missing = [k for k in required if k not in payload]
    if missing:
        return f"missing keys {missing}"
    if list(payload["frame_ids"]) != list(frame_ids):
        return "frame_ids mismatch"
    if payload.get("source_video_sha256") != source_sha:
        return "source_video_sha256 mismatch"
    if local_instruction is not None and \
            payload.get("local_instruction") != local_instruction:
        return "local_instruction text mismatch"
    if local_instruction_sha256 is not None and \
            payload.get("local_instruction_sha256") != local_instruction_sha256:
        return "local_instruction_sha256 mismatch"
    f, h, w = (int(payload["latent_num_frames"]), int(payload["latent_height"]),
               int(payload["latent_width"]))
    lat = payload["latent"]
    if tuple(lat.shape) != (f * h * w, VAE_CHANNELS):
        return f"latent shape {tuple(lat.shape)} != {(f * h * w, VAE_CHANNELS)}"
    if h != ICL_HEIGHT // VAE_SPATIAL or w != ICL_WIDTH // VAE_SPATIAL:
        return f"latent geometry {h}x{w} != {ICL_HEIGHT // VAE_SPATIAL}x{ICL_WIDTH // VAE_SPATIAL}"
    if f != latent_frames(len(frame_ids)):
        return f"latent_num_frames {f} != {latent_frames(len(frame_ids))}"
    if str(lat.dtype) != "torch.bfloat16":
        return f"latent dtype {lat.dtype}"
    emb = payload["text_emb"]
    if tuple(emb.shape) != (512, 4096):
        return f"text_emb shape {tuple(emb.shape)}"
    import torch
    if not torch.isfinite(lat.float()).all() or not torch.isfinite(emb.float()).all():
        return "not finite"
    return None


def atomic_torch_save(payload: dict, path: Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ----- Modal functions ---------------------------------------------------------------------------------


@app.function(image=image, volumes={SIM: data_volume, WEIGHTS: weights_volume.read_only()},
              timeout=3600)
def preflight() -> dict:
    """Fail fast if the pinned checkpoint is absent or the upload is incomplete."""
    missing = [d for d in ("vae", "tokenizer", "text_encoder")
               if not (Path(MODEL_DIR) / d).is_dir()]
    if missing:
        raise RuntimeError(f"checkpoint incomplete at {MODEL_DIR}: missing {missing} — "
                           "not downloading a different revision")
    counts = {}
    for split in SPLITS:
        root = Path(DATA_DIR) / split
        tasks = sorted(p.name for p in root.iterdir()
                       if p.is_dir() and p.name not in ("meta", "human_data")
                       and not p.name.endswith(".partial"))
        counts[split] = {"tasks": len(tasks)}
        manifest = root / "icl_manifest.json"
        counts[split]["manifest_samples"] = (
            len(json.loads(manifest.read_text())["samples"]) if manifest.is_file() else None)
    return {"model_dir": MODEL_DIR, "counts": counts}


def _video_frames(path: str):
    import numpy as np
    from decord import VideoReader, cpu

    reader = VideoReader(path, ctx=cpu(0))
    return reader, float(reader.get_avg_fps() or 0), np.arange(len(reader))


def _encode(vae, streaming, frames, height, width):
    """Wan VAE streaming encode exactly as modal_so101_smoke._encode: resize, scale to [-1,1],
    first frame then 4-frame chunks, keep mu, normalise with VAE latents mean/std."""
    import torch
    import torch.nn.functional as F

    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    video = video.permute(1, 0, 2, 3).unsqueeze(0) / 255.0 * 2.0 - 1.0
    video = video.to("cuda", torch.bfloat16)
    streaming.clear_cache()
    with torch.no_grad():
        chunks = [streaming.encode_chunk(video[:, :, :1])]
        for start in range(1, video.shape[2], 4):
            chunks.append(streaming.encode_chunk(video[:, :, start:start + 4]))
    mu, _ = torch.chunk(torch.cat(chunks, dim=2), 2, dim=1)
    mean = torch.tensor(vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=mu.device).view(1, -1, 1, 1, 1)
    streaming.clear_cache()
    return ((mu.float() - mean) * inv_std).to(torch.bfloat16)[0].cpu()


def _text_emb(tokenizer, text_encoder, text):
    """UMT5 embedding padded to 512 tokens, as wan_va_server._get_t5_prompt_embeds."""
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    inputs = tokenizer([prompt_clean(text)], padding="max_length", max_length=512,
                       truncation=True, add_special_tokens=True, return_attention_mask=True,
                       return_tensors="pt")
    with torch.no_grad():
        emb = text_encoder(inputs.input_ids.cuda(), inputs.attention_mask.cuda()).last_hidden_state[0]
    n = int(inputs.attention_mask.sum())
    return torch.cat([emb[:n], emb.new_zeros(512 - n, emb.shape[1])]).to(torch.bfloat16).cpu()


def _encode_episode(vae, streaming, tokenizer, text_encoder, split: str, task: str,
                    meta: dict, instruction: str, task_emb, sample: dict) -> dict:
    """One episode: both robot cameras + the paired human video. Idempotent."""
    import numpy as np
    import torch
    from einops import rearrange

    task_root = Path(DATA_DIR) / split / task
    ep = int(meta["episode_index"])
    length = int(meta["length"])
    ids = robot_frame_ids(length)
    out_record = {"task": task, "episode": ep, "robot": {}, "human": None,
                  "skipped": []}

    for camera in CAMERAS:
        video = task_root / "videos" / "chunk-000" / camera / f"episode_{ep:06d}.mp4"
        sha = sha256_file(video)
        out = Path(DATA_DIR) / split / robot_latent_relpath(task, ep, length, camera)
        if out.is_file():
            payload = torch.load(out, map_location="cpu", weights_only=False)
            reason = validate_robot_payload(payload, ids, sha)
            if reason is None:
                out_record["robot"][camera] = [str(out), "skipped"]
                out_record["skipped"].append(camera)
                continue
            print(f"{split}/{task}/ep{ep:03d} {camera}: re-encoding ({reason})", flush=True)
            out.unlink()
        reader, _, _ = _video_frames(str(video))
        if len(reader) != length:
            raise RuntimeError(f"{split}/{task}/ep{ep:03d} {camera}: video has "
                               f"{len(reader)} frames, episodes.jsonl {length}")
        lat = _encode(vae, streaming, reader.get_batch(ids).asnumpy(), HEIGHT, WIDTH)
        c, f, h, w = lat.shape
        payload = {"latent": rearrange(lat, "c f h w -> (f h w) c").contiguous(),
                   "latent_num_frames": f, "latent_height": h, "latent_width": w,
                   "video_num_frames": len(ids), "video_height": HEIGHT, "video_width": WIDTH,
                   "task": instruction, "local_instruction": instruction,
                   "local_instruction_emb": task_emb, "text": instruction,
                   "frame_ids": ids, "start_frame": 0, "end_frame": length,
                   "fps": 30 / FRAME_STRIDE, "ori_fps": 30,
                   "source_video_sha256": sha,
                   "source_episode_id": f"{split}:{task}/episode_{ep:03d}",
                   "transform_version": "absolute-near-plane-v3"}
        atomic_torch_save(payload, out)
        out_record["robot"][camera] = [str(out), [c, f, h, w]]

    # paired human demo via the manifest sample passed in by the shard loop
    video = Path(DATA_DIR) / split / "human_data" / sample["human_video_path"]
    sha = sha256_file(video)
    hout = Path(DATA_DIR) / split / human_latent_relpath(split, task, ep)
    reader, src_fps, _ = _video_frames(str(video))
    hids = human_frame_ids(len(reader), src_fps)
    detailed = sample["human_local_instruction"]
    detailed_sha = sample["human_local_instruction_sha256"]
    if hout.is_file():
        payload = torch.load(hout, map_location="cpu", weights_only=False)
        if validate_human_payload(payload, hids, sha, detailed, detailed_sha) is None:
            out_record["human"] = [str(hout), "skipped"]
            out_record["skipped"].append("human")
            return out_record
        hout.unlink()
    hlat = _encode(vae, streaming, reader.get_batch(hids).asnumpy(), ICL_HEIGHT, ICL_WIDTH)
    hc, hf, hh, hw = hlat.shape
    payload = {"latent": rearrange(hlat, "c f h w -> (f h w) c").contiguous(),
               "latent_num_frames": hf, "latent_height": hh, "latent_width": hw,
               "text_emb": _text_emb(tokenizer, text_encoder, detailed),
               "text": sample["human_text"], "local_instruction": detailed,
               "local_instruction_sha256": detailed_sha, "frame_ids": hids,
               "fps": ICL_FPS, "ori_fps": src_fps,
               "source_video_sha256": sha, "icl_sample_id": sample["sample"]}
    atomic_torch_save(payload, hout)
    out_record["human"] = [str(hout), [hc, hf, hh, hw]]
    return out_record


@app.function(image=image, gpu="H100", timeout=12 * 3600, memory=131072,
              volumes={SIM: data_volume, WEIGHTS: weights_volume.read_only()})
def encode_shard(shard: int) -> dict:
    """Encode one deterministic task shard. Commits after each task so interruption resumes."""
    import torch
    from wan_va.modules.utils import (WanVAEStreamingWrapper, load_text_encoder,
                                      load_tokenizer, load_vae)

    assert 0 <= shard < NUM_SHARDS
    vae = load_vae(f"{MODEL_DIR}/vae", torch_dtype=torch.bfloat16, torch_device="cuda")
    streaming = WanVAEStreamingWrapper(vae)
    tokenizer = load_tokenizer(f"{MODEL_DIR}/tokenizer")
    text_encoder = load_text_encoder(f"{MODEL_DIR}/text_encoder",
                                     torch_dtype=torch.bfloat16, torch_device="cuda")

    done, skipped, encoded = [], 0, 0
    for split in SPLITS:
        split_root = Path(DATA_DIR) / split
        manifest = json.loads((split_root / "icl_manifest.json").read_text())
        samples_by_name = {s["sample"]: s for s in manifest["samples"]}
        # Task names come from the manifest — never from a directory listing, which would
        # pick up generated dirs like human_latents created during the encode itself.
        tasks = sorted({s["robot_task_name"] for s in manifest["samples"]})
        for task in shard_partition(tasks).get(shard, []):
            task_root = split_root / task
            metas = [json.loads(l) for l in
                     (task_root / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
            emb_by_instruction = {}
            for meta in metas:
                instruction = meta["tasks"][0]
                if instruction not in emb_by_instruction:
                    emb_by_instruction[instruction] = _text_emb(tokenizer, text_encoder,
                                                                instruction)
                rec = _encode_episode(vae, streaming, tokenizer, text_encoder, split,
                                      task, meta, instruction, emb_by_instruction[instruction],
                                      samples_by_name[f"{task}__episode_{meta['episode_index']:03d}"])
                n_skip = len(rec["skipped"])
                skipped += n_skip
                encoded += 3 - n_skip
            data_volume.commit()
            done.append(f"{split}/{task}")
            print(f"[shard {shard}] done {split}/{task}", flush=True)
    return {"shard": shard, "tasks": done, "encoded": encoded, "skipped": skipped}


@app.function(image=image, cpu=8, timeout=6 * 3600, memory=65536,
              volumes={SIM: data_volume, WEIGHTS: weights_volume.read_only()})
def inventory_latents() -> dict:
    """Enumerate every expected episode from the manifests; require 2 robot + 1 human latent
    each with validated payload (fields, frame_ids, source sha, shape, dtype, finite). Writes
    /sim/reports/latent_inventory.json atomically."""
    import torch

    failures, files, expected = [], {}, 0
    expected_paths = set()
    for split in SPLITS:
        split_root = Path(DATA_DIR) / split
        manifest = json.loads((split_root / "icl_manifest.json").read_text())
        by_task = {}
        for sample in manifest["samples"]:
            by_task.setdefault(sample["robot_task_name"], []).append(sample)
        for task, samples in by_task.items():
            task_root = split_root / task
            metas = {int(m["episode_index"]): m for m in (
                json.loads(l) for l in (task_root / "meta" / "episodes.jsonl").read_text().splitlines()
                if l.strip())}
            for sample in sorted(samples, key=lambda s: s["sample"]):
                ep = int(sample["sample"].rsplit("episode_", 1)[1])
                length = int(metas[ep]["length"])
                expected += 1
                for camera in CAMERAS:
                    path = split_root / robot_latent_relpath(task, ep, length, camera)
                    expected_paths.add(str(path))
                    if not path.is_file():
                        failures.append(f"missing {path}")
                        continue
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    sha = sha256_file(task_root / "videos" / "chunk-000" / camera /
                                      f"episode_{ep:06d}.mp4")
                    reason = validate_robot_payload(payload, robot_frame_ids(length), sha)
                    if reason:
                        failures.append(f"{path}: {reason}")
                    files[str(path)] = {"bytes": path.stat().st_size,
                                        "sha256": sha256_file(path)}
                hpath = split_root / human_latent_relpath(split, task, ep)
                expected_paths.add(str(hpath))
                if not hpath.is_file():
                    failures.append(f"missing {hpath}")
                    continue
                hpayload = torch.load(hpath, map_location="cpu", weights_only=False)
                hvideo = split_root / "human_data" / sample["human_video_path"]
                hsha = sha256_file(hvideo)
                reader, src_fps, _ = _video_frames(str(hvideo))
                reason = validate_human_payload(
                    hpayload, human_frame_ids(len(reader), src_fps), hsha,
                    sample.get("human_local_instruction"),
                    sample.get("human_local_instruction_sha256"))
                if reason:
                    failures.append(f"{hpath}: {reason}")
                files[str(hpath)] = {"bytes": hpath.stat().st_size,
                                     "sha256": sha256_file(hpath)}
        # no extras
        for pth in split_root.rglob("*.pth"):
            if str(pth) not in expected_paths:
                failures.append(f"unexpected {pth}")
        for pth in split_root.rglob("*.partial"):
            failures.append(f"stale partial {pth}")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "expected_episodes": expected, "files": len(files), "failures": failures,
              "total_bytes": sum(f["bytes"] for f in files.values()), "ok": not failures}
    out = Path(REPORTS) / "latent_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.partial")
    tmp.write_text(json.dumps({"summary": {k: v for k, v in report.items() if k != "failures"},
                               "files": files, "failures": failures}, indent=1))
    os.replace(tmp, out)
    data_volume.commit()
    report["report"] = str(out)
    report["report_sha256"] = sha256_file(out)
    return report


@app.function(image=image, cpu=32, timeout=4 * 3600, memory=131072,
              volumes={SIM: data_volume, WEIGHTS: weights_volume.read_only()})
def loader_smoke_remote() -> dict:
    """Instantiate pinned MultiICLLeRobotLatentDataset per split and retrieve deterministic
    samples: shortest/longest train, five additional task families, one per val task."""
    import numpy as np
    import torch
    from easydict import EasyDict
    from wan_va.dataset.icl_lerobot_latent_dataset import MultiICLLeRobotLatentDataset

    def cfg_for(split: str):
        cfg = EasyDict()
        cfg.dataset_path = f"{DATA_DIR}/{split}"
        cfg.icl_manifest_path = f"{DATA_DIR}/{split}/icl_manifest.json"
        cfg.human_latent_path = f"{DATA_DIR}/{split}/human_latents/so101"
        cfg.robot_latent_path = f"{DATA_DIR}/{split}"
        cfg.env_type = "none"
        cfg.height = HEIGHT
        cfg.width = WIDTH
        cfg.action_dim = 30
        cfg.action_per_frame = 2 * VAE_TEMPORAL
        cfg.obs_cam_keys = list(CAMERAS)
        cfg.text_encoder_type = "umt_dense"
        cfg.empty_emb_path = f"{REMOTE_REPO}/wan_va/assets/empty_text_emb.pt"
        cfg.cfg_prob = 0.0
        cfg.init_worker = 8
        cfg.rank = 0
        cfg.local_rank = 0
        cfg.world_size = 1
        cfg.excluded_task_names = []
        cfg.expected_num_train_tasks = 61 if split == "train" else 5
        cfg.enable_dataset_index_cache = False
        cfg.rebuild_dataset_index_cache = False
        return cfg

    def find_idx(ds, task: str, ep: int) -> int:
        offset = 0
        for sub in ds._datasets:
            if sub.repo_id and Path(sub.repo_id).name == task:
                for i, meta in enumerate(sub.new_metas):
                    if int(meta["episode_index"]) == ep:
                        return offset + i
                raise KeyError(f"{task}: episode {ep} not in valid metas")
            offset += len(sub)
        raise KeyError(f"task {task} not found")

    train_ds = MultiICLLeRobotLatentDataset(config=cfg_for("train"))
    assert len(train_ds) == 1109, len(train_ds)
    metas_flat = []
    off = 0
    for sub in train_ds._datasets:
        for meta in sub.new_metas:
            metas_flat.append((off, meta, Path(sub.repo_id).name))
            off += 1
    by_len = sorted(metas_flat, key=lambda t: int(t[1]["end_frame"]) - int(t[1]["start_frame"]))
    picks = [("shortest", by_len[0][0]), ("longest", by_len[-1][0])]
    tasks_sorted = [Path(s.repo_id).name for s in train_ds._datasets]
    for pos in (0, 15, 30, 45, 60):
        task = tasks_sorted[pos]
        picks.append((task, find_idx(train_ds, task, 0)))

    val_tasks = sorted(p.name for p in Path(f"{DATA_DIR}/val").iterdir()
                       if p.is_dir() and (p / "meta" / "episodes.jsonl").is_file())
    report_samples = []
    for tag, idx in picks:
        report_samples.append(_sample_report(train_ds, tag, idx))

    val_ds = MultiICLLeRobotLatentDataset(config=cfg_for("val"))
    assert len(val_ds) == 50, len(val_ds)
    for task in val_tasks:
        idx = find_idx(val_ds, task, 0)
        report_samples.append(_sample_report(val_ds, f"val:{task}", idx))

    report = {"train_len": len(train_ds), "val_len": len(val_ds),
              "samples": report_samples, "ok": True}
    out = Path(REPORTS) / "loader_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.partial")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out)
    data_volume.commit()
    report["report"] = str(out)
    report["report_sha256"] = sha256_file(out)
    return report


def _sample_report(ds, tag, idx) -> dict:
    """Fetch the same item twice; assert deterministic equality and check the contract."""
    import numpy as np
    import torch

    item, item2 = ds[idx], ds[idx]
    problems = []
    for k in ("latents", "actions", "actions_mask", "text_emb", "icl_latents",
              "icl_text_emb"):
        if not torch.equal(item[k], item2[k]):
            problems.append(f"{k} not deterministic")
    lat = item["latents"]
    if lat.shape[0] != VAE_CHANNELS or lat.shape[2] != HEIGHT // VAE_SPATIAL or \
            lat.shape[3] != 2 * (WIDTH // VAE_SPATIAL):
        problems.append(f"latents shape {tuple(lat.shape)}")
    mask = item["actions_mask"]
    active = sorted(np.nonzero(mask[..., 0].any(dim=(1, 2)).cpu().numpy())[0].tolist())
    if active != [0, 1, 2, 3, 4, 28]:
        problems.append(f"action mask channels {active}")
    if item["actions"].shape[0] != 30 or item["actions"].shape[2] != 2 * VAE_TEMPORAL:
        problems.append(f"actions shape {tuple(item['actions'].shape)}")
    icl = item["icl_latents"]
    if icl.shape[0] != VAE_CHANNELS or \
            tuple(icl.shape[2:]) != (ICL_HEIGHT // VAE_SPATIAL, ICL_WIDTH // VAE_SPATIAL):
        problems.append(f"icl geometry {tuple(icl.shape)}")
    for k in ("latents", "actions", "text_emb", "icl_latents", "icl_text_emb"):
        if not torch.isfinite(item[k].float()).all():
            problems.append(f"{k} not finite")
    tensors = {k: {"shape": list(item[k].shape),
                   "sha256": hashlib.sha256(item[k].float().contiguous().numpy().tobytes()).hexdigest()}
               for k in ("latents", "actions", "actions_mask", "text_emb", "icl_latents",
                         "icl_text_emb")}
    return {"tag": tag, "idx": int(idx), "icl_sample_id": item.get("icl_sample_id"),
            "icl_human_latent_path": item.get("icl_human_latent_path"),
            "problems": problems, "tensors": tensors}


# ----- topology probe -----------------------------------------------------------------------------------

_DIST_PROBE = '''
import json, os, socket, time
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
world = [None] * dist.get_world_size()
dist.all_gather_object(world, [dist.get_rank(), socket.gethostname(), local_rank,
                               torch.cuda.current_device(), torch.cuda.get_device_name(local_rank)])
result = {"world": world}
for tag, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    n = 64 * 1024 * 1024 // (4 if dtype == torch.float32 else 2)
    t = torch.ones(n, dtype=dtype, device="cuda")
    for _ in range(5):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        dist.all_reduce(t)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    result[tag] = {"iters": len(times), "min_s": min(times),
                   "mean_s": sum(times) / len(times), "max_s": max(times)}
if dist.get_rank() == 0:
    print("PROBE_JSON" + json.dumps(result))
dist.destroy_process_group()
'''


def _run_cmd(argv: list[str], timeout: int = 300) -> dict:
    """Run argv without a shell; retain everything for the report."""
    import subprocess

    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "returncode": p.returncode,
                "stdout": p.stdout[-131072:], "stderr": p.stderr[-131072:]}
    except Exception as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": str(exc)}


def _probe_marker(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("PROBE_JSON"):
            return json.loads(line[len("PROBE_JSON"):])
    return None


def _single_host_unique_devices(dist_result: dict | None, world_size: int = 8) -> bool:
    if not dist_result or not dist_result.get("world"):
        return False
    hosts = {g[1] for g in dist_result["world"]}
    devices = sorted(g[3] for g in dist_result["world"])
    return len(hosts) == 1 and devices == list(range(world_size))


@app.function(image=image, gpu="H100:8", timeout=10 * 60, memory=32768,
              volumes={SIM: data_volume})
def topology_probe() -> dict:
    """Single-node 8-GPU topology + NCCL sanity probe; report committed before any raise."""
    import tempfile

    commands = []
    for argv in (["hostname"],
                 ["nvidia-smi", "-L"],
                 ["nvidia-smi", "topo", "-m"],
                 ["nvidia-smi",
                  "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
                  "--format=csv,noheader"]):
        commands.append(_run_cmd(argv))

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "probe.py"
        script.write_text(_DIST_PROBE)
        commands.append(_run_cmd(
            ["python", "-m", "torch.distributed.run", "--standalone",
             "--nproc_per_node=8", str(script)], timeout=540))
    dist_result = _probe_marker(commands[-1]["stdout"])
    single_host = _single_host_unique_devices(dist_result)
    ok = all(c["returncode"] == 0 for c in commands) and single_host
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "commands": commands, "distributed": dist_result,
              "single_host_unique_devices": single_host, "ok": ok}
    out = Path(REPORTS) / "topology.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.partial")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out)
    data_volume.commit()
    if not ok:
        raise RuntimeError(f"topology probe failed; report committed to {out}")
    return {"ok": ok, "report": str(out), "single_host_unique_devices": single_host,
            "devices": len(dist_result["world"]) if dist_result else 0}


@app.local_entrypoint()
def topology() -> None:
    print(json.dumps(topology_probe.remote(), indent=2))


# ----- WandB data-prep tracking (local, names/config only — never logs secrets) -----------------------

WANDB_PROJECT = "zero-wam-so101"
WANDB_GROUP = "sim-train-v1"
WANDB_JOB_TYPE = "data-prep"
WANDB_RUN_ID = "sim-train-v1-data-prep"


def _git_rev(path) -> tuple[str, bool]:
    import subprocess

    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return head, dirty


def _wandb_config() -> dict:
    repo = Path(__file__).resolve().parents[1]
    rev, dirty = _git_rev(repo)
    upstream, upstream_dirty = _git_rev(LOCAL_REPO)
    bucket = repo / "data" / "hf_bucket_ICL-so101"
    out = repo / "data" / "zero_wam_sim"
    return {
        "repo_rev": rev, "repo_dirty": dirty,
        "zerowam_rev": upstream, "zerowam_dirty": upstream_dirty,
        "train_source_index_sha256": sha256_file(bucket / "sim_train_v1/index.json"),
        "val_source_index_sha256": sha256_file(bucket / "sim_val_v2/index.json"),
        "train_overlay_index_sha256": sha256_file(bucket / "train_v2/index.json"),
        "val_overlay_index_sha256": sha256_file(bucket / "val_v3/index.json"),
        "audit_sha256": sha256_file(out / "audit.json"),
        "action_stats_file_sha256": sha256_file(out / "train/meta/action_stats.json"),
        "app": APP_NAME, "weights_volume": "zero-wam-weights", "data_volume": "zero-wam-so101-sim",
        "cameras": list(CAMERAS), "robot_geometry": f"{HEIGHT}x{WIDTH}", "robot_stride": FRAME_STRIDE,
        "icl_geometry": f"{ICL_HEIGHT}x{ICL_WIDTH}@{ICL_FPS}fps",
        "action_channels": [0, 1, 2, 3, 4, 28], "num_shards": NUM_SHARDS,
        "transform_version": "absolute-near-plane-v3",
    }


def _wandb_run():
    import wandb

    return wandb.init(project=WANDB_PROJECT, group=WANDB_GROUP, job_type=WANDB_JOB_TYPE,
                    id=WANDB_RUN_ID, resume="allow", config=_wandb_config())


def _wandb_artifact(run, artifact_files: dict, name: str = "sim-train-v1-data-prep",
                    type_: str = "data-prep") -> None:
    """Attach compact JSON artifacts (reports/audit/stats — never latent tensors/videos)."""
    import wandb

    artifact = wandb.Artifact(name, type=type_)
    for arc_name, path in artifact_files.items():
        artifact.add_file(str(path), name=arc_name)
    run.log_artifact(artifact)


def _volume_download(remote_path: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        data_volume.read_file_into_fileobj(remote_path, f)
    return dest


# ----- local entrypoints --------------------------------------------------------------------------------


@app.local_entrypoint()
def prepare() -> None:
    """Upload the whole converted corpus to /sim/data, then preflight volume+checkpoint."""
    with data_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_DATA), "/data")
    print(json.dumps(preflight.remote(), indent=2))


@app.local_entrypoint()
def encode(wait: bool = True) -> None:
    """Spawn the 8 shard encoders. Prints Modal call IDs; re-run resumes via payload checks."""
    run = _wandb_run()
    print(json.dumps(preflight.remote(), indent=2))
    calls = [encode_shard.spawn(i) for i in range(NUM_SHARDS)]
    ids = [c.object_id for c in calls]
    print("shard call ids:", ids, flush=True)
    run.log({"shard_call_ids": ids, "shards_spawned": NUM_SHARDS})
    if not wait:
        run.finish()
        return
    totals = {"encoded": 0, "skipped": 0, "tasks": 0}
    for shard, call in enumerate(calls):
        result = call.get(timeout=12 * 3600)
        print(json.dumps(result, indent=2), flush=True)
        totals["encoded"] += result["encoded"]
        totals["skipped"] += result["skipped"]
        totals["tasks"] += len(result["tasks"])
        run.log({"encoded": totals["encoded"], "skipped": totals["skipped"],
                 "tasks_completed": totals["tasks"]})
    run.finish()


@app.local_entrypoint()
def verify() -> None:
    run = _wandb_run()
    result = inventory_latents.remote()
    print(json.dumps(result, indent=2))
    run.log({"latent_inventory_ok": result["ok"], "expected_episodes": result["expected_episodes"],
             "latent_files": result["files"], "latent_total_bytes": result["total_bytes"],
             "latent_inventory_sha256": result["report_sha256"],
             "latent_failures": len(result["failures"])})
    if result["ok"]:
        local = _volume_download("/reports/latent_inventory.json",
                                 Path("/tmp/sim_reports/latent_inventory.json"))
        _wandb_artifact(run, {
            "latent_inventory.json": local,
            "audit.json": Path(__file__).resolve().parents[1] / "data/zero_wam_sim/audit.json",
            "action_stats.json": Path(__file__).resolve().parents[1] /
                                 "data/zero_wam_sim/train/meta/action_stats.json"})
    run.finish()


@app.local_entrypoint()
def loader_smoke() -> None:
    run = _wandb_run()
    result = loader_smoke_remote.remote()
    print(json.dumps(result, indent=2))
    run.log({"loader_smoke_ok": result["ok"], "train_len": result["train_len"],
             "val_len": result["val_len"], "loader_smoke_sha256": result["report_sha256"],
             "loader_smoke_samples": len(result["samples"])})
    if result["ok"]:
        local = _volume_download("/reports/loader_smoke.json",
                                 Path("/tmp/sim_reports/loader_smoke.json"))
        _wandb_artifact(run, {"loader_smoke.json": local}, name="sim-train-v1-loader-smoke",
                        type_="loader-smoke")
    run.finish()
