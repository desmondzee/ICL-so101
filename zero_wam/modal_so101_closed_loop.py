"""Closed-loop sim_val_v2 evaluation of an SO-101 Zero-WAM checkpoint on Modal H100s (sim + model in one container).

Weights and val data stream from Hugging Face into the `icl-so101-closed-loop` volume; nothing is downloaded locally.

    MODAL_PROFILE=desmond-zee modal run zero_wam/modal_so101_closed_loop.py::prepare
    MODAL_PROFILE=desmond-zee modal run zero_wam/modal_so101_closed_loop.py --limit 1          # smoke
    MODAL_PROFILE=desmond-zee modal run zero_wam/modal_so101_closed_loop.py                    # all 50
    MODAL_PROFILE=desmond-zee modal volume get icl-so101-closed-loop results/<name> outputs/zero_wam/closed_loop/

`--video-guidance-scale -1` (default) conditions on the human video and its detailed text only; a value > 1
adds the short task instruction with text CFG, as in the chunk-causal validation.
"""

import json
import math
import os
from pathlib import Path

import modal

APP_NAME = "icl-so101-closed-loop"
ROOT = Path(__file__).resolve().parents[1]
REPO = "/opt/icl-so101"
VOL = "/vol"
RUN = "so101-sft-a100-b32-cosine-v2b-1k-20261011"
STEP = 500
CKPT_BUCKET = "hf://buckets/nikgeo/icl-so101-weights/runs"
DATA_BUCKET = "hf://buckets/akoniti/ICL-so101"
BASE_ID = "Robbyant-Research/zero-wam-pretrain"
BASE_REVISION = "7040c4195df216c900334ef62d5fdcf05c0601aa"
BASE_DIR = f"{VOL}/models/zero-wam-pretrain"
DATA = f"{VOL}/data"
RESULTS = f"{VOL}/results"
SAMPLE_SEED = 20261012

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("icl-so101-closed-loop", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

download_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet,cli]==1.6.0", "click", "typer")

_IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/*.png", "**/.pytest_cache/**"]
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0", "libegl1", "libosmesa6")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel ninja packaging",
        "python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 "
        "--index-url https://download.pytorch.org/whl/cu126",
    )
    .add_local_file(str(ROOT / "third_party/Zero-WAM/requirements.txt"), "/tmp/zero-wam-requirements.txt", copy=True)
    .run_commands(
        "sed -e '/^lerobot==/d' -e '/^wandb$/d' -e '/^flash_attn$/d' /tmp/zero-wam-requirements.txt "
        "> /tmp/zero-wam-inference-requirements.txt",
        "MAX_JOBS=4 python -m pip install -r /tmp/zero-wam-inference-requirements.txt --no-build-isolation",
        "python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'",
        "python -m pip install decord==0.6.0 so101-nexus==0.7.0 mujoco==3.14.0 pyarrow==17.0.0 numpy==1.26.4",
    )
    .env({"PYTHONPATH": f"{REPO}:{REPO}/third_party/Zero-WAM", "MUJOCO_GL": "egl", "ICL_CFG": "5"})
    .add_local_dir(str(ROOT / "third_party/Zero-WAM"), f"{REPO}/third_party/Zero-WAM",
                   ignore=[".git", ".git/**", "data/**", "checkpoints/**", "results/**", *_IGNORE])
    .add_local_dir(str(ROOT / "third_party/LIBERO/libero/libero/assets"), f"{REPO}/third_party/LIBERO/libero/libero/assets")
    .add_local_dir(str(ROOT / "sim"), f"{REPO}/sim", ignore=[*_IGNORE, "**/*.mp4"])
    .add_local_dir(str(ROOT / "zero_wam"), f"{REPO}/zero_wam", ignore=_IGNORE)
    .add_local_file(str(ROOT / "scripts/run_zero_wam_chunk_causal_validation.py"),
                    f"{REPO}/scripts/run_zero_wam_chunk_causal_validation.py")
)


def _ckpt(run, step):
    return f"{VOL}/checkpoints/{run}/checkpoint_step_{step}"


@app.function(image=download_image, volumes={VOL: volume}, secrets=[hf_secret], timeout=3 * 3600, cpu=8)
def prepare(run: str = RUN, step: int = STEP):
    import subprocess

    from huggingface_hub import snapshot_download

    def sync(src, dst, *flags):
        subprocess.run(["hf", "buckets", "sync", src, dst, *flags], check=True)

    sync(f"{CKPT_BUCKET}/{run}/checkpoints/checkpoint_step_{step}/transformer", f"{_ckpt(run, step)}/transformer")
    snapshot_download(BASE_ID, revision=BASE_REVISION, local_dir=BASE_DIR, ignore_patterns=["transformer/*"])
    sync(f"{DATA_BUCKET}/zero_wam_latents_v1/val/human_latents", f"{DATA}/val/human_latents")
    sync(f"{DATA_BUCKET}/zero_wam_loader_v1/val/meta", f"{DATA}/zero_wam_loader_v1/val/meta")
    for src, dst in (("zero_wam_loader_v1/val/icl_manifest.json", "val/icl_manifest.json"),
                     ("sim_val_v2/index.json", "val/index.json")):
        subprocess.run(["hf", "buckets", "cp", f"{DATA_BUCKET}/{src}", f"{DATA}/{dst}"], check=True)
    volume.commit()
    files = sorted(str(p.relative_to(VOL)) for p in Path(VOL).rglob("*") if p.is_file())
    print(f"{len(files)} files; transformer shards:", [f for f in files if "safetensors" in f and run in f])


@app.function(image=download_image, volumes={VOL: volume})
def episodes() -> list[dict]:
    index = json.loads(Path(f"{DATA}/val/index.json").read_text())
    manifest = {s["sample_id"]: s for s in json.loads(Path(f"{DATA}/val/icl_manifest.json").read_text())["samples"]}
    out = []
    for e in index["episodes"]:
        sample = manifest[e["id"].replace("/", "__")]
        latent = Path(f"{DATA}/val/human_latents") / Path(sample["human_video_path"]).with_suffix(".pth")
        if not latent.is_file():
            raise FileNotFoundError(latent)
        out.append({"id": e["id"], "task": e["task"], "seed": e["seed"], "robot_frames": e["robot_frames"],
                    "instruction": sample["human_text"], "human_latent": str(latent)})
    return out


@app.cls(image=image, gpu="H100", volumes={VOL: volume}, timeout=6 * 3600, max_containers=8)
class Rollout:
    run: str = modal.parameter(default=RUN)
    step: int = modal.parameter(default=STEP)
    video_steps: int = modal.parameter(default=50)
    action_steps: int = modal.parameter(default=50)

    @modal.enter()
    def load(self):
        import tempfile

        link = Path(REPO) / "data/hf_bucket_ICL-so101"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(DATA)
        os.chdir(REPO)
        from zero_wam.so101_closed_loop import load_model

        self.workdir = tempfile.mkdtemp()
        self.model, self.load_evidence = load_model(BASE_DIR, _ckpt(self.run, self.step), self.workdir,
                                                    self.video_steps, self.action_steps)

    @modal.method()
    def rollout(self, episode: dict, name: str, sample_seed: int, horizon: float, video_guidance_scale: float) -> dict:
        from zero_wam.so101_closed_loop import run_episode

        out = Path(RESULTS) / name / episode["id"]
        result = run_episode(self.model, episode, out, max_controls=math.ceil(horizon * episode["robot_frames"]),
                             sample_seed=sample_seed, video_guidance_scale=video_guidance_scale)
        volume.commit()
        return result


@app.function(image=download_image, volumes={VOL: volume})
def write_summary(name: str, payload: dict):
    path = Path(RESULTS) / name / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    volume.commit()


def summarize(results):
    tasks = {}
    for r in results:
        tasks.setdefault(r["task"], []).append(r["success"])
    per_task = {t: {"success": sum(v), "episodes": len(v), "rate": sum(v) / len(v)} for t, v in sorted(tasks.items())}
    n = len(results)
    wins = sum(r["success"] for r in results)
    return {"overall": {"success": wins, "episodes": n, "rate": wins / max(n, 1)}, "per_task": per_task}


@app.local_entrypoint()
def main(name: str = "", run: str = RUN, step: int = STEP, limit: int = 0, tasks: str = "",
         video_steps: int = 50, action_steps: int = 50, horizon: float = 2.0, video_guidance_scale: float = -1.0):
    name = name or f"{run}-step{step}-v{video_steps}a{action_steps}-tcfg{video_guidance_scale:g}"
    todo = [{**e, "sample_seed": SAMPLE_SEED + i} for i, e in enumerate(episodes.remote())]
    if tasks:
        todo = [e for e in todo if e["task"] in tasks.split(",")]
    if limit:
        todo = todo[:limit]
    worker = Rollout(run=run, step=step, video_steps=video_steps, action_steps=action_steps)
    args = [(e, name, e["sample_seed"], horizon, video_guidance_scale) for e in todo]
    results, errors = [], []
    for e, r in zip(todo, worker.rollout.starmap(args, return_exceptions=True)):
        if isinstance(r, BaseException):
            errors.append({"id": e["id"], "error": repr(r)})
            print(f"ERROR {e['id']}: {r!r}")
        else:
            results.append(r)
            print(f"{r['id']}: success={r['success']} controls={r['controls']}/{r['max_controls']} "
                  f"chunk={r['mean_chunk_infer_s']:.2f}s wall={r['wall_s']}s")
    payload = {"name": name, "run": run, "step": step, "video_steps": video_steps, "action_steps": action_steps,
               "horizon_x_oracle": horizon, "video_guidance_scale": video_guidance_scale, "icl_guidance_scale": 5.0,
               "sample_seed_base": SAMPLE_SEED, "summary": summarize(results), "errors": errors,
               "episodes": sorted(results, key=lambda r: r["id"])}
    write_summary.remote(name, payload)
    local = ROOT / "outputs/zero_wam/closed_loop" / name / "summary.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["summary"], indent=1))
