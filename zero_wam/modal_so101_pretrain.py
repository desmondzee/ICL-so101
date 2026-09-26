"""Pretrain fallback for the paired SO-101 action-space experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

MODEL_ID = "Robbyant-Research/zero-wam-pretrain"
MODEL_REVISION = "7040c4195df216c900334ef62d5fdcf05c0601aa"
MODEL_DIR = "/models/zero-wam-pretrain"
LOCAL_REPO = Path(__file__).resolve().parents[1] / "third_party" / "Zero-WAM"
REMOTE_REPO = "/opt/zero-wam"
app = modal.App("zero-wam-so101-pretrain")
volume = modal.Volume.from_name("zero-wam-weights", create_if_missing=True)
inference_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel ninja packaging",
        "python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126",
    )
    .add_local_dir(str(LOCAL_REPO), remote_path=REMOTE_REPO, copy=True, ignore=[".git", ".git/**", "data/**", "checkpoints/**", "results/**", "**/__pycache__/**", "**/*.pyc", "**/.pytest_cache/**"])
    .run_commands(
        f"sed -e '/^lerobot==/d' -e '/^wandb$/d' -e '/^flash_attn$/d' {REMOTE_REPO}/requirements.txt > /tmp/zero-wam-inference-requirements.txt",
        "MAX_JOBS=4 python -m pip install -r /tmp/zero-wam-inference-requirements.txt --no-build-isolation",
        "python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'",
        "python -m pip install decord==0.6.0",
    )
    .env({"PYTHONPATH": REMOTE_REPO, "MODEL_PATH": MODEL_DIR})
)


@app.function(image=modal.Image.debian_slim(python_version="3.10").pip_install("huggingface_hub"), volumes={"/models": volume}, timeout=7200)
def prepare_pretrain():
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR)
    volume.commit()


@app.cls(image=inference_image, gpu="A100-80GB", volumes={"/models": volume}, max_containers=1, scaledown_window=600, timeout=3600, startup_timeout=1800, memory=65536)
class PretrainSO101:
    @modal.enter()
    def load(self):
        import sys

        if not Path(MODEL_DIR).is_dir():
            raise FileNotFoundError(f"Run prepare_pretrain first: {MODEL_DIR}")
        sys.path.insert(0, REMOTE_REPO)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29562")
        os.environ["RANK"] = os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        from wan_va.configs import VA_CONFIGS
        from wan_va.distributed.util import init_distributed
        from wan_va.wan_va_server import VA_Server

        config = VA_CONFIGS["demo"]
        config.model_path = MODEL_DIR
        config.rank = config.local_rank = 0
        config.world_size = 1
        config.save_root = "/tmp/zero-wam-so101-pretrain"
        init_distributed(1, 0, 0)
        self.model = VA_Server(config)

    @modal.method()
    def so101_step(self, request):
        import copy
        import numpy as np
        import torch

        if request.get("reset"):
            channels = {"pose": [0, 1, 2, 3, 4, 5, 6, 28], "joint": [14, 15, 16, 17, 18, 28]}[request["mode"]]
            config = self.model.job_config
            config.used_action_channel_ids = channels
            config.obs_cam_keys = ["observation.images.top", "observation.images.wrist"]
            inverse = [len(channels)] * 30
            for index, channel in enumerate(channels):
                inverse[channel] = index
            config.inverse_used_action_channel_ids = inverse
            config.norm_stat = copy.deepcopy(request["stats"])
            torch.manual_seed(int(request["seed"]))
            np.random.seed(int(request["seed"]))
            return self.model.infer({"reset": True, "prompt": request["prompt"]})
        return self.model.infer(request)


@app.local_entrypoint()
def so101_compare(seeds: str = "100,101,102", max_steps: int = 400, save_root: str = "outputs/zero_wam/so101/pretrain"):
    from zero_wam.so101_eval import calibrate, run_episode

    root = Path(save_root)
    root.mkdir(parents=True, exist_ok=True)
    stats, counts = calibrate()
    (root / "calibration.json").write_text(json.dumps({"stats": stats, "counts": counts, "seed": 101}, indent=2))
    worker = PretrainSO101()
    results = []
    for seed_text in seeds.split(","):
        for mode in ("pose", "joint"):
            result = run_episode(worker, mode, int(seed_text), stats[mode], root / f"{mode}_seed{seed_text}", max_steps)
            results.append(result)
            (root / "results.json").write_text(json.dumps(results, indent=2))
            print(result, flush=True)
