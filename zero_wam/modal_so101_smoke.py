"""SO-101 overfit smoke test of Zero-WAM post-training on Modal: one robot episode (front + wrist cameras), one paired
human demo; a second episode of the same task is held out for validation.

Upstream training runs as released: the robotwin_train config and the exact torchrun command of script/train.sh,
8 GPUs, from zero-wam-pretrain. Only the input data is ours. It goes through the oxe source key, which copies the
robotwin_train settings but has no 43-task RoboTwin isolation check and, like our data, is single-camera and
single-arm. WandB is switched on through zero_wam/so101_wandb_train.py.

    uv run python -m zero_wam.so101_smoke_data           # local: LeRobot v2.1 dirs, action metadata, manifests
    modal run zero_wam/modal_so101_smoke.py::prepare      # checkpoint to the volume, upload data, encode latents (1 GPU)
    modal run --detach zero_wam/modal_so101_smoke.py::smoke  # train (8x H100), then held-out and train-episode losses

Secrets: Modal secret `so101-smoke-wandb` holding WANDB_API_KEY, WANDB_TEAM_NAME (and optionally WANDB_BASE_URL).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "zero-wam-so101-smoke"
MODEL_ID = "robbyant-research/zero-wam-pretrain"
MODEL_REVISION = "7040c4195df216c900334ef62d5fdcf05c0601aa"
LOCAL_REPO = Path(__file__).resolve().parents[1] / "third_party" / "Zero-WAM"
LOCAL_DATA = Path(__file__).resolve().parents[1] / "data" / "so101_smoke" / "zero_wam" / "pbvr__so101_test002"
REMOTE_REPO = "/opt/zero-wam"
VOL = "/vol"
MODEL_DIR = f"{VOL}/zero-wam-pretrain"
DATA_DIR = f"{VOL}/so101_smoke/{LOCAL_DATA.name}"  # smoke.json there says which task, cameras and episodes
RUNS_DIR = f"{VOL}/runs"
FRAME_STRIDE = 4  # 16 actions per latent frame (action_per_frame=16), as in the released Robotwin latents
HEIGHT, WIDTH = 224, 288  # robot camera latent geometry of the robotwin config
ICL_HEIGHT, ICL_WIDTH, ICL_FPS = 320, 448, 12  # geometry of the released human latents

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("zero-wam-so101-smoke", create_if_missing=True)
wandb_secret = modal.Secret.from_name("so101-smoke-wandb")

download_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel ninja packaging",
        "python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126",
    )
    .add_local_file(str(LOCAL_REPO / "requirements.txt"), "/tmp/zero-wam-requirements.txt", copy=True)
    .run_commands(
        # Upstream requirements as pinned; flash_attn from its prebuilt wheel, lerobot without deps (it pins torch<2.8).
        "sed -e '/^lerobot==/d' -e '/^flash_attn$/d' /tmp/zero-wam-requirements.txt > /tmp/req.txt",
        "MAX_JOBS=4 python -m pip install -r /tmp/req.txt --no-build-isolation",
        "python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'",
        "python -m pip install --no-deps lerobot==0.3.3",
        "python -m pip install datasets jsonlines av decord==0.6.0 pyarrow pandas",
    )
    .add_local_dir(str(LOCAL_REPO), remote_path=REMOTE_REPO, copy=True,
                   ignore=[".git", ".git/**", "data/**", "checkpoints/**", "results/**", "**/__pycache__/**", "**/*.pyc"])
    .add_local_file(str(Path(__file__).with_name("so101_wandb_train.py")), f"{REMOTE_REPO}/so101_wandb_train.py", copy=True)
    .env({"PYTHONPATH": REMOTE_REPO, "MODEL_PATH": MODEL_DIR, "HF_HOME": f"{VOL}/hf_home", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(image=download_image, volumes={VOL: volume}, timeout=3 * 3600)
def prepare_checkpoint() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR)
    volume.commit()


def _video_frames(path: str):
    import numpy as np
    from decord import VideoReader, cpu

    reader = VideoReader(path, ctx=cpu(0))
    return reader, float(reader.get_avg_fps() or 0), np.arange(len(reader))


def _encode(vae, streaming, frames, height, width):
    """Wan VAE streaming encode exactly as wan_va_server: resize, scale to [-1, 1], first frame then 4-frame chunks,
    keep mu, normalise with the VAE latents mean and std. Returns [48, F, h, w]."""
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

    inputs = tokenizer([prompt_clean(text)], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                       return_attention_mask=True, return_tensors="pt")
    with torch.no_grad():
        emb = text_encoder(inputs.input_ids.cuda(), inputs.attention_mask.cuda()).last_hidden_state[0]
    n = int(inputs.attention_mask.sum())
    return torch.cat([emb[:n], emb.new_zeros(512 - n, emb.shape[1])]).to(torch.bfloat16).cpu()


@app.function(image=image, gpu="H100", volumes={VOL: volume}, timeout=3600, memory=65536)
def encode() -> dict:
    """Robot-camera and human-demo latents, with their text embeddings, in the released .pth layout."""
    import numpy as np
    import torch
    from einops import rearrange
    from wan_va.modules.utils import WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_vae

    vae = load_vae(f"{MODEL_DIR}/vae", torch_dtype=torch.bfloat16, torch_device="cuda")
    streaming = WanVAEStreamingWrapper(vae)
    tokenizer = load_tokenizer(f"{MODEL_DIR}/tokenizer")
    text_encoder = load_text_encoder(f"{MODEL_DIR}/text_encoder", torch_dtype=torch.bfloat16, torch_device="cuda")
    smoke = json.loads((Path(DATA_DIR) / "smoke.json").read_text())
    report = {}
    for split, episode in smoke["splits"].items():
        root = Path(DATA_DIR) / split
        task_root = root / smoke["task_dir"]
        meta = json.loads((task_root / "meta" / "episodes.jsonl").read_text().splitlines()[0])
        task, length = meta["tasks"][0], int(meta["length"])
        fps = json.loads((task_root / "meta" / "info.json").read_text())["fps"]

        # Robot: whole episode, every 4th frame, trimmed to 4k+1 frames (the released Robotwin latents do the same);
        # one file per camera, which the loader concatenates along width.
        ids = list(range(0, length, FRAME_STRIDE))
        ids = ids[: (len(ids) - 1) // 4 * 4 + 1]
        task_emb = _text_emb(tokenizer, text_encoder, task)
        robot = {}
        for camera in smoke["cameras"]:
            reader, _, _ = _video_frames(str(task_root / "videos" / "chunk-000" / camera / f"episode_{episode:06d}.mp4"))
            if len(reader) != length:
                raise RuntimeError(f"{split} {camera}: video has {len(reader)} frames, parquet {length}")
            lat = _encode(vae, streaming, reader.get_batch(ids).asnumpy(), HEIGHT, WIDTH)
            c, f, h, w = lat.shape
            out = task_root / "latents" / "chunk-000" / camera / f"episode_{episode:06d}_0_{length}.pth"
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"latent": rearrange(lat, "c f h w -> (f h w) c").contiguous(), "latent_num_frames": f, "latent_height": h, "latent_width": w,
                        "video_num_frames": len(ids), "video_height": HEIGHT, "video_width": WIDTH, "task": task, "local_instruction": task,
                        "local_instruction_emb": task_emb, "text": task, "frame_ids": ids,
                        "start_frame": 0, "end_frame": length, "fps": fps / FRAME_STRIDE, "ori_fps": fps}, out)
            robot[camera] = [str(out), [c, f, h, w]]

        # Human: 12 fps as wan_va_server._sample_icl_frame_indices, 320x448, trimmed to 4k+1 frames.
        sample = json.loads((root / "icl_manifest.json").read_text())["samples"][0]
        video = root / "human_data" / sample["human_video_path"]
        hreader, src_fps, _ = _video_frames(str(video))
        n_out = max(1, int(round(len(hreader) / src_fps * ICL_FPS)))
        hids = np.unique(np.clip(np.floor(np.arange(n_out) * src_fps / ICL_FPS).astype(np.int64), 0, len(hreader) - 1))
        hids = hids[: (len(hids) - 1) // 4 * 4 + 1]
        hlat = _encode(vae, streaming, hreader.get_batch(hids.tolist()).asnumpy(), ICL_HEIGHT, ICL_WIDTH)
        hc, hf, hh, hw = hlat.shape
        run_rel = sample["human_video_path"].split("/", 1)[1]  # drop the dataset folder: <run_*>/samples/<sample>/generated_video.mp4
        hout = root / "human_latents" / "so101" / Path(run_rel).with_suffix(".pth")
        hout.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"latent": rearrange(hlat, "c f h w -> (f h w) c").contiguous(), "latent_num_frames": hf, "latent_height": hh, "latent_width": hw,
                    "text_emb": _text_emb(tokenizer, text_encoder, smoke["human_text"]), "text": smoke["human_text"], "frame_ids": hids.tolist(),
                    "fps": ICL_FPS, "ori_fps": src_fps}, hout)
        report[split] = {"robot": robot, "human": [str(hout), [hc, hf, hh, hw]]}
        print(split, report[split], flush=True)
    volume.commit()
    return report


def _torchrun(args: list[str], env: dict, log: Path) -> None:
    """script/train.sh's command (NGPU=8, port 29501, rank-0 logs, --tee 3), with the module swapped for the
    WandB launcher, which then runs wan_va.train itself."""
    cmd = ["python", "-m", "torch.distributed.run", "--nproc_per_node=8", "--local-ranks-filter=0", "--master_port", "29501", "--tee", "3",
           "-m", "so101_wandb_train", "--config-name", "robotwin_train", "--datasets", "oxe:1.0", *args]
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TORCHFT_LIGHTHOUSE": "http://localhost:29510",
           "CONFIG_NAME": "robotwin_train", "WANDB_BASE_URL": os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai"),
           "WANDB_PROJECT": "zero-wam-so101", **env}
    log.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd), flush=True)
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=REMOTE_REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line)
            print(line, end="", flush=True)
    volume.commit()
    if p.wait():
        raise RuntimeError(f"torchrun exited {p.returncode}; log {log}")


def _split_args(split: str) -> list[str]:
    root = f"{DATA_DIR}/{split}"
    return ["--dataset-path", root, "--icl-manifest-path", f"{root}/icl_manifest.json", "--human-latent-path", f"{root}/human_latents/so101",
            "--init-worker", "1", "--load-worker", "1"]


@app.function(image=image, gpu="H100:8", volumes={VOL: volume}, secrets=[wandb_secret], timeout=6 * 3600, memory=480 * 1024, cpu=32)
def train(run: str = "smoke", num_steps: int = 300, warmup_steps: int = 0, cosine: bool = False) -> str:
    """Overfit on the train episode with the preset's optimiser and losses; one checkpoint at the end. warmup_steps > 0
    overrides the preset's 200, and cosine decays the learning rate to 0 by the last step (see so101_wandb_train.py)."""
    save_root = Path(RUNS_DIR) / run
    losses = save_root / "train_losses.jsonl"
    losses.unlink(missing_ok=True)
    env = {"WANDB_NAME": f"so101-smoke-{run}-train", "SO101_LOSS_LOG": str(losses), "ZERO_WAM_SAVE_ROOT": str(save_root)}
    if warmup_steps:
        env["SO101_WARMUP_STEPS"] = str(warmup_steps)
    if cosine:
        env["SO101_COSINE_TOTAL_STEPS"] = str(num_steps)
    _torchrun(_split_args("train") + ["--save-root", str(save_root), "--num-steps", str(num_steps), "--save-interval", str(num_steps)],
              env, save_root / "train.log")
    return str(save_root / "checkpoints" / f"checkpoint_step_{num_steps}")


@app.function(image=image, gpu="H100:8", volumes={VOL: volume}, secrets=[wandb_secret], timeout=3 * 3600, memory=480 * 1024, cpu=32)
def evaluate(model_path: str, split: str, tag: str, run: str = "smoke", steps: int = 20) -> dict:
    """Loss of a checkpoint on one split through the same training step with learning rate 0 (AdamW's decoupled
    weight decay scales with the learning rate, so the weights do not change), conditioned on both text and the human
    video (drop_icl 0, droptext 0), averaged over `steps` steps x 8 ranks of random timesteps and noise."""
    out = Path(RUNS_DIR) / run / f"eval_{tag}_{split}.jsonl"
    out.unlink(missing_ok=True)
    _torchrun(_split_args(split) + ["--model-path", model_path, "--save-root", f"/tmp/eval_{tag}_{split}", "--learning-rate", "0",
                                    "--drop-icl", "0", "--droptext-target", "0", "--num-steps", str(steps), "--save-interval", "1000000"],
              {"WANDB_NAME": f"so101-smoke-{run}-eval-{tag}-{split}", "SO101_LOSS_LOG": str(out)}, out.with_suffix(".log"))
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    keys = ["loss_metrics/global_avg_video_loss", "loss_metrics/global_avg_action_loss", "loss_metrics/mcp_weighted_total"]
    return {"tag": tag, "split": split, "steps": len(rows), **{k.split("/")[-1]: sum(r[k] for r in rows) / len(rows) for k in keys if rows and k in rows[0]}}


@app.local_entrypoint()
def prepare(checkpoint: bool = True, encode_latents: bool = True) -> None:
    """Checkpoint, data upload and latents; --no-encode-latents re-uploads only (e.g. new action stats; the latents
    already on the volume stay)."""
    if checkpoint:
        prepare_checkpoint.remote()
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_DATA), f"/so101_smoke/{LOCAL_DATA.name}")
    if encode_latents:
        print(json.dumps(encode.remote(), indent=2))


@app.local_entrypoint()
def evals(run: str = "smoke", num_steps: int = 300, eval_steps: int = 20, skip: str = "") -> None:
    """The validation passes alone, for a run whose training already finished; skip e.g. 'pretrain:train'."""
    done = set(filter(None, skip.split(",")))
    checkpoint = f"{RUNS_DIR}/{run}/checkpoints/checkpoint_step_{num_steps}"
    for tag, model in (("pretrain", MODEL_DIR), (f"step{num_steps}", checkpoint)):
        for split in ("train", "val"):
            if f"{tag}:{split}" not in done:
                print(json.dumps(evaluate.remote(model, split, tag, run, eval_steps)), flush=True)


@app.function(image=download_image, volumes={VOL: volume}, timeout=10 * 3600)
def pipeline(run: str, num_steps: int, eval_steps: int, warmup_steps: int, cosine: bool) -> dict:
    """Train, then the four validation passes, chained on Modal so a dropped local connection cannot stop it midway
    (launch with `modal run --detach`). The summary is also written to the volume."""
    checkpoint = train.remote(run, num_steps, warmup_steps, cosine)
    results = []
    for tag, model in (("pretrain", MODEL_DIR), (f"step{num_steps}", checkpoint)):
        for split in ("train", "val"):
            results.append(evaluate.remote(model, split, tag, run, eval_steps))
            print(json.dumps(results[-1]), flush=True)
    summary = {"checkpoint": checkpoint, "results": results}
    volume.reload()
    (Path(RUNS_DIR) / run / "eval.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


@app.local_entrypoint()
def smoke(run: str = "smoke", num_steps: int = 300, eval_steps: int = 20, warmup_steps: int = 0, cosine: bool = False) -> None:
    summary = pipeline.remote(run, num_steps, eval_steps, warmup_steps, cosine)
    out = Path(__file__).resolve().parents[1] / "outputs" / "zero_wam" / "so101_smoke" / run / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
