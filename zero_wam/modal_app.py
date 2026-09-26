"""Single-container Zero-WAM inference and asset preparation on Modal."""

from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "zero-wam-robotwin"
MODEL_ID = "robbyant-research/zero-wam-posttrain-robotwin"
MODEL_REVISION = "07ee865f175d9474a5653e9147698390e47b6143"
DATASET_ID = "Robbyant-Research/HumanGen"
DATASET_REVISION = "971d12e942bf101c0d47716f928b1c044d02ac71"
HUMAN_SAMPLE = "273_robotwin_Use_an_arm_to_place_the_empty_cup_on_the_coaster"
HUMAN_RUN = "run_robotwin_20260728_013720_robotwin_n1000"
HUMAN_RELATIVE = f"robotwin/{HUMAN_RUN}/samples/{HUMAN_SAMPLE}/generated_video_kling-v3"
HUMAN_LATENT_FILE = f"human_latents/{HUMAN_RELATIVE}.pth"
HUMAN_VIDEO_MEMBER = f"human_data/{HUMAN_RELATIVE}.mp4"
HUMAN_ARCHIVE_FILE = "human_data/part-00006.tar.zst"

LOCAL_REPO = Path(__file__).resolve().parents[1] / "third_party" / "Zero-WAM"
REMOTE_REPO = "/opt/zero-wam"
MODEL_DIR = "/models/zero-wam-posttrain-robotwin"
LATENT_PATH = f"/models/{HUMAN_LATENT_FILE}"
VIDEO_PATH = f"/models/{HUMAN_VIDEO_MEMBER}"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("zero-wam-weights", create_if_missing=True)

download_image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "zstd"
).pip_install("huggingface_hub")

inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel ninja packaging",
        "python -m pip install torch==2.9.0 torchvision==0.24.0 "
        "torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126",
    )
    .add_local_dir(
        str(LOCAL_REPO),
        remote_path=REMOTE_REPO,
        copy=True,
        ignore=[
            ".git", ".git/**", "data/**", "checkpoints/**", "results/**",
            "**/__pycache__/**", "**/*.pyc", "**/.pytest_cache/**",
        ],
    )
    .run_commands(
        f"sed -e '/^lerobot==/d' -e '/^wandb$/d' -e '/^flash_attn$/d' "
        f"{REMOTE_REPO}/requirements.txt "
        "> /tmp/zero-wam-inference-requirements.txt",
        "MAX_JOBS=4 python -m pip install "
        "-r /tmp/zero-wam-inference-requirements.txt --no-build-isolation",
        "python -m pip install --no-deps "
        "'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'",
        "python -m pip install decord==0.6.0",
    )
    .env({"PYTHONPATH": REMOTE_REPO, "MODEL_PATH": MODEL_DIR, "ICL_CFG": "5"})
)


@app.function(image=download_image, volumes={"/models": volume}, timeout=7200)
def prepare_checkpoint() -> None:
    """Populate the shared Volume without baking weights into the image."""
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
    )
    hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=HUMAN_LATENT_FILE,
        local_dir="/models",
    )
    volume.commit()


@app.function(image=download_image, volumes={"/models": volume}, timeout=7200)
def prepare_human_video() -> str:
    """Extract only the manifest-selected MP4 from the public HumanGen shard."""
    import shutil
    import subprocess
    import tarfile

    from huggingface_hub import hf_hub_download

    target = Path(VIDEO_PATH)
    if target.is_file():
        return str(target)

    archive = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=HUMAN_ARCHIVE_FILE,
        cache_dir="/tmp/human-video-cache",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    decoder = subprocess.Popen(["zstd", "-dc", archive], stdout=subprocess.PIPE)
    try:
        assert decoder.stdout is not None
        with tarfile.open(fileobj=decoder.stdout, mode="r|") as members:
            for member in members:
                if member.name.lstrip("./") != HUMAN_VIDEO_MEMBER:
                    continue
                source = members.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Archive member has no content: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                break
            else:
                raise FileNotFoundError(f"{HUMAN_VIDEO_MEMBER} absent from {archive}")
    finally:
        if decoder.stdout is not None:
            decoder.stdout.close()
        decoder.terminate()
        decoder.wait()
    if target.stat().st_size == 0:
        raise RuntimeError(f"Extracted video is empty: {target}")
    volume.commit()
    return str(target)


@app.cls(
    image=inference_image,
    gpu="A100-80GB",
    volumes={"/models": volume},
    max_containers=1,
    scaledown_window=600,
    timeout=3600,
    startup_timeout=1800,
    memory=65536,
)
class ZeroWAM:
    @modal.enter()
    def load(self) -> None:
        import sys

        if not Path(MODEL_DIR).is_dir():
            raise FileNotFoundError(
                f"Checkpoint absent at {MODEL_DIR}; run prepare_checkpoint first"
            )
        sys.path.insert(0, REMOTE_REPO)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29561")
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        from wan_va.configs import VA_CONFIGS
        from wan_va.distributed.util import init_distributed
        from wan_va.wan_va_server import VA_Server

        config = VA_CONFIGS["robotwin"]
        config.model_path = MODEL_DIR
        # The released sample metadata records 320x448 frames / 20x28 latents.
        # Match that geometry when encoding the raw MP4.
        config.icl_width = 448
        config.rank = config.local_rank = 0
        config.world_size = 1
        config.save_root = "/tmp/zero-wam"
        init_distributed(1, 0, 0)
        self.model = VA_Server(config)
        original_icl_loader = self.model._load_or_encode_icl

        def load_or_encode_icl(video_path: str, latent_path: str):
            if latent_path and Path(latent_path).is_file():
                return original_icl_loader(video_path, latent_path)
            if not video_path or not Path(video_path).is_file():
                return original_icl_loader(video_path, latent_path)

            import torch

            self.model.streaming_vae.clear_cache()
            video = self.model._read_icl_video(video_path).to(
                device=next(self.model.vae.parameters()).device,
                dtype=self.model.dtype,
            )
            video = video / 255.0 * 2.0 - 1.0
            trailing = (video.shape[2] - 1) % 4
            if trailing:
                pad = 4 - trailing
                video = torch.cat(
                    [video, video[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2
                )
            chunks = [self.model.streaming_vae.encode_chunk(video[:, :, :1])]
            for start in range(1, video.shape[2], 4):
                chunks.append(
                    self.model.streaming_vae.encode_chunk(video[:, :, start:start + 4])
                )
            mu, _ = torch.chunk(torch.cat(chunks, dim=2), 2, dim=1)
            mean = torch.tensor(self.model.vae.config.latents_mean, device=mu.device)
            std = torch.tensor(self.model.vae.config.latents_std, device=mu.device)
            latent = self.model.normalize_latents(mu, mean, 1.0 / std).cpu()
            self.model.streaming_vae.clear_cache()
            return latent, None

        self.model._load_or_encode_icl = load_or_encode_icl
        self._log_gpu("post-load")

    def _log_gpu(self, tag: str) -> dict:
        """VRAM/util snapshot printed so `modal run` streams it to the local log."""
        import subprocess

        import torch

        stats = {
            "vram_alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            "vram_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
            "vram_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            util, used, total = [v.strip() for v in out.stdout.split(",")]
            stats.update(
                gpu_util_pct=int(util),
                gpu_mem_gib=round(int(used) / 1024, 2),
                gpu_total_gib=round(int(total) / 1024, 2),
            )
        except Exception as exc:
            stats["nvidia_smi_error"] = str(exc)
        self._last_stats = stats
        print(f"[zw-metrics] {tag} {stats}", flush=True)
        return stats

    @modal.method()
    def stats(self) -> dict:
        return self._log_gpu("stats")

    @modal.method()
    def step(self, obs: dict) -> dict:
        import time

        import torch

        t0 = time.time()
        result = self.model.infer(obs)
        kind = (
            "reset"
            if obs.get("reset")
            else "kv" if obs.get("compute_kv_cache") else "infer"
        )
        print(
            f"[zw-metrics] {kind} {time.time() - t0:.2f}s "
            f"vram_alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
            f"vram_peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
            flush=True,
        )
        return result

    @modal.method()
    def so101_step(self, request: dict) -> dict:
        """Configure one SO-101 embodiment and keep its cache on this worker."""
        import copy
        import numpy as np
        import torch

        if request.get("reset"):
            mode = request["mode"]
            channels = request.get("channels") or {"pose": [0, 1, 2, 3, 4, 5, 6, 28], "joint": [14, 15, 16, 17, 18, 28]}[mode]
            cfg = self.model.job_config
            cfg.obs_cam_keys = ["observation.images.top", "observation.images.wrist"]
            cfg.used_action_channel_ids = channels
            inverse = [len(channels)] * 30
            for i, channel in enumerate(channels):
                inverse[channel] = i
            cfg.inverse_used_action_channel_ids = inverse
            cfg.norm_stat = copy.deepcopy(request["stats"])
            torch.manual_seed(int(request["seed"]))
            np.random.seed(int(request["seed"]))
            return self.model.infer({"reset": True, "prompt": request["prompt"], "use_icl": False, "video_guidance_scale": 5.0})
        if request.get("compute_kv_cache") and self.model.use_icl_model:
            # Cache the actions actually executed after simulator safety limits.
            executed = np.asarray(request["executed_model_actions"], dtype=np.float32)
            normalized = self.model.preprocess_action(executed)
            normalized[:, ~self.model.action_mask.cpu()] = 0
            if self.model.chunk_idx == 0:
                normalized[:, :, 0] = 0
            self.model.last_predicted_actions = normalized.to(self.model.device, self.model.dtype)
        return self.model.infer(request)

    @modal.method()
    def inspect_human_prompt(self) -> dict:
        """Compare the published latent with an encoding of its paired MP4."""
        import torch

        video_latent, _ = self.model._load_or_encode_icl(VIDEO_PATH, "")
        saved_latent, _ = self.model._load_or_encode_icl("", LATENT_PATH)
        result = {
            "video_shape": tuple(video_latent.shape),
            "saved_shape": tuple(saved_latent.shape),
            "same_shape": video_latent.shape == saved_latent.shape,
        }
        if result["same_shape"]:
            diff = video_latent.float() - saved_latent.float()
            result["cosine_similarity"] = torch.nn.functional.cosine_similarity(
                video_latent.float().flatten(), saved_latent.float().flatten(), dim=0
            ).item()
            result["max_abs_diff"] = diff.abs().max().item()
            result["rel_l2_diff"] = (
                diff.norm() / saved_latent.float().norm()
            ).item()
            # Latent layout is (1, C=48, T=16, H=20, W=28); cosine per frame.
            vid_frames = video_latent.float().reshape(48, 16, -1)
            ref_frames = saved_latent.float().reshape(48, 16, -1)
            result["per_frame_cosine"] = [
                torch.nn.functional.cosine_similarity(
                    vid_frames[:, t].flatten(), ref_frames[:, t].flatten(), dim=0
                ).item()
                for t in range(vid_frames.shape[1])
            ]
        return result


@app.function(image=inference_image, timeout=300)
def check_runtime() -> dict:
    """Verify the packaged inference imports before loading model weights."""
    import flash_attn
    import torch
    from wan_va.configs import VA_CONFIGS
    from wan_va.wan_va_server import VA_Server

    assert VA_Server is not None
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "flash_attn": flash_attn.__version__,
        "robotwin_icl": VA_CONFIGS["robotwin"].use_icl_model,
    }


@app.local_entrypoint()
def smoke(mode: str = "text") -> None:
    """Exercise reset and one action without starting RoboTwin."""
    import numpy as np

    if mode not in {"text", "latent", "video"}:
        raise ValueError("mode must be text, latent, or video")
    model = ZeroWAM()
    reset = {
        "reset": True,
        "prompt": "place the empty cup on the coaster",
        "use_icl": mode != "text",
        "video_guidance_scale": 5.0 if mode == "text" else -1.0,
        "icl_guidance_scale": 5.0,
        "icl_latent_path": LATENT_PATH if mode == "latent" else "",
        "icl_video_path": VIDEO_PATH if mode == "video" else "",
    }
    print(model.step.remote(reset))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = {key: frame for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )}
    obs["observation.state"] = np.zeros(14, dtype=np.float32)
    obs["task"] = reset["prompt"]
    action = model.step.remote({"obs": obs, "prompt": reset["prompt"]})["action"]
    if action.shape != (16, 2, 16):
        raise AssertionError(f"Unexpected action shape: {action.shape}")
    print(f"action shape: {action.shape}")


@app.local_entrypoint()
def rollout(
    mode: str = "text",
    task: str = "place_empty_cup",
    test_num: int = 1,
    seed: int = 0,
    save_root: str = "outputs/zero_wam",
) -> None:
    """Run a local RoboTwin episode with an ephemeral Modal GPU worker."""
    from zero_wam.modal_client import run

    run(ZeroWAM(), mode, task, test_num, seed, save_root)


@app.local_entrypoint()
def verify_human_prompt() -> None:
    """Compare HumanGen MP4 encoding with its published latent on the GPU."""
    print(ZeroWAM().inspect_human_prompt.remote())


@app.local_entrypoint()
def so101_compare(
    seeds: str = "100,101,102",
    max_steps: int = 400,
    save_root: str = "outputs/zero_wam/so101",
) -> None:
    """Calibrate and run paired zero-shot cube-lift episodes on one GPU host."""
    import json

    from zero_wam.so101_eval import calibrate, run_episode

    root = Path(save_root)
    root.mkdir(parents=True, exist_ok=True)
    stats, counts = calibrate()
    (root / "calibration.json").write_text(json.dumps({"stats": stats, "counts": counts, "seed": 101}, indent=2))
    model = ZeroWAM()
    results = []
    for seed_text in seeds.split(","):
        seed = int(seed_text)
        for mode in ("pose", "joint"):
            result = run_episode(model, mode, seed, stats[mode], root / f"{mode}_seed{seed}", max_steps)
            results.append(result)
            print(result, flush=True)
            (root / "results.json").write_text(json.dumps(results, indent=2))


@app.local_entrypoint()
def so101_trial(
    variant: str = "pose",
    seed: int = 100,
    max_steps: int = 128,
    save_root: str = "outputs/zero_wam/so101/trials",
) -> None:
    """Run one named compatibility hypothesis with a saved manifest."""
    from zero_wam.so101_eval import run_trial

    print(run_trial(ZeroWAM(), variant, seed, max_steps, save_root, "posttrain"), flush=True)


@app.local_entrypoint()
def so101_screen(
    variant: str,
    seeds: str = "100,101,102",
    max_steps: int = 128,
    save_root: str = "outputs/zero_wam/so101/trials",
) -> None:
    from zero_wam.so101_eval import run_trial

    worker = ZeroWAM()
    for seed_text in seeds.split(","):
        print(run_trial(worker, variant, int(seed_text), max_steps, save_root, "posttrain"), flush=True)


@app.local_entrypoint()
def so101_screen_variants(
    variants: str,
    seed: int = 100,
    max_steps: int = 128,
    save_root: str = "outputs/zero_wam/so101/trials",
) -> None:
    from zero_wam.so101_eval import run_trial

    worker = ZeroWAM()
    for variant in variants.split(","):
        print(run_trial(worker, variant, seed, max_steps, save_root, "posttrain"), flush=True)
