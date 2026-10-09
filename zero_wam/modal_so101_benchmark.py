"""Modal wrapper for the SO-101 full-episode batch-64 feasibility benchmark.

Self-contained (imports stdlib + modal only): one `H100:8` function call, one
container, torchrun nproc 8 over `zero_wam/so101_full_episode_benchmark.py` copied to
`/opt/bench/`. Software pin matches `zero_wam/modal_so101_sim.py` exactly. Outputs land
under `/sim/reports/full_episode_batch64/` on the `zero-wam-so101-sim` volume.

    modal run zero_wam/modal_so101_benchmark.py::benchmark
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = "zero-wam-so101-sim-benchmark"
LOCAL_REPO = Path(__file__).resolve().parents[1] / "third_party" / "Zero-WAM"
BENCH_MODULE = Path(__file__).resolve().parent / "so101_full_episode_benchmark.py"
REMOTE_REPO = "/opt/zero-wam"
REMOTE_BENCH = "/opt/bench/so101_full_episode_benchmark.py"
WEIGHTS = "/weights"
SIM = "/sim"
DATA_DIR = f"{SIM}/data"
REPORT_DIR = f"{SIM}/reports/full_episode_batch64"

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
    .add_local_file(str(BENCH_MODULE), REMOTE_BENCH, copy=True)
    .env({"PYTHONPATH": f"{REMOTE_REPO}:/opt/bench", "MODEL_PATH":
          f"{WEIGHTS}/zero-wam-pretrain", "HF_HOME": f"{SIM}/hf_home",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
)


@app.function(image=image, cpu=32, timeout=2 * 3600,
              memory=128 * 1024, max_containers=1,
              volumes={SIM: data_volume})
def prepare_indexes() -> dict:
    """CPU-only durable dataset-index preflight; must succeed before any H100 work."""
    import subprocess

    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(REPORT_DIR) / "index_preflight.log"
    env = dict(os.environ, PYTHONPATH="/opt/bench:/opt/zero-wam",
               PYTORCH_ALLOC_CONF="expandable_segments:True")
    argv = ["python", REMOTE_BENCH, "--prepare-indexes",
            "--data-root", DATA_DIR, "--report-dir", REPORT_DIR]
    proc = subprocess.run(argv, capture_output=True, text=True,
                          timeout=100 * 60, env=env)
    log_path.write_text(proc.stdout + "\n===== STDERR =====\n" + proc.stderr)
    print(proc.stdout, end="", flush=True)
    print(proc.stderr, end="", flush=True)
    data_volume.commit()
    summary = {"returncode": proc.returncode,
               "report": f"{REPORT_DIR}/index_preflight.json",
               "log": str(log_path)}
    if proc.returncode != 0:
        raise RuntimeError(
            f"index preflight exited {proc.returncode}; see {log_path}")
    return summary


@app.function(image=image, gpu="H100:8", timeout=2 * 3600,
              memory=480 * 1024, max_containers=1,
              volumes={SIM: data_volume, WEIGHTS: weights_volume.read_only()})
def run_benchmark() -> dict:
    """torchrun the 8-rank benchmark; commit report+log even when the child fails."""
    import subprocess
    import threading

    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(REPORT_DIR) / "torchrun.log"
    env = dict(os.environ, PYTHONPATH="/opt/bench:/opt/zero-wam")
    argv = ["python", "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=8", REMOTE_BENCH,
            "--data-root", DATA_DIR, "--weights", f"{WEIGHTS}/zero-wam-pretrain",
            "--report-dir", REPORT_DIR]
    rc, timed_out = None, False
    log = open(log_path, "w")
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=env)

        def _tee():
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

        reader = threading.Thread(target=_tee, daemon=True)
        reader.start()
        try:
            rc = proc.wait(timeout=90 * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
        reader.join(timeout=30)
    except Exception as exc:
        rc = -1
        log.write(f"launcher error: {exc!r}\n")
    finally:
        if timed_out:
            log.write("=== killed after 90min wait ===\n")
            rc = -1
        log.close()
    data_volume.commit()
    summary = {"returncode": rc, "report": f"{REPORT_DIR}/benchmark_report.json",
               "log": str(log_path)}
    print(json.dumps(summary, indent=2))
    if rc != 0:
        raise RuntimeError(f"torchrun exited {rc}; log+report committed to {REPORT_DIR}")
    return summary


@app.local_entrypoint()
def benchmark() -> None:
    # CPU index preflight gates any H100 allocation
    print(json.dumps(prepare_indexes.remote(), indent=2))
    print(json.dumps(run_benchmark.remote(), indent=2))
