"""Local A100 host + Zero-WAM loader preflight for the ICL-so101 launch gate.

Subcommands:
    host    collect host inventory and assert the 8xA100 NV12 topology contract
    loader  smoke the released MultiICLLeRobotLatentDataset on the downloaded
            loader + latent prefixes with deterministic double-fetch checks
    all     run host then loader

Pure helpers in this module (GPU query parsing, topology parsing, atomic JSON
writes, action-mask channel extraction) are import- and test-safe without torch
or the upstream Zero-WAM package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_GPU_COUNT = 8
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
MIN_GPU_MEMORY_MIB = 81920
EXPECTED_LINK_TYPE = "NV12"
GPU_QUERY_FIELDS = (
    "index,name,memory.total,compute_cap,driver_version,pci.bus_id"
)

EXPECTED_TRAIN_LEN = 1109
EXPECTED_VAL_LEN = 50
EXPECTED_TRAIN_TASKS = 61
EXPECTED_VAL_TASKS = 5
EXPECTED_ACTIVE_CHANNELS = [0, 1, 2, 3, 4, 28]

CAMERAS = ("observation.images.front", "observation.images.wrist")
HEIGHT, WIDTH = 224, 288
ACTION_DIM = 30
ACTION_PER_FRAME = 8
ROBOT_CHANNELS, ROBOT_H, ROBOT_W = 48, 14, 36
HUMAN_CHANNELS, HUMAN_H, HUMAN_W = 48, 20, 28
TENSOR_KEYS = (
    "latents",
    "actions",
    "actions_mask",
    "text_emb",
    "icl_latents",
    "icl_text_emb",
)


# ---------------------------------------------------------------------------
# pure helpers (no GPU / upstream / torch required)
# ---------------------------------------------------------------------------


def atomic_write_json(path, payload):
    """Write JSON atomically via a sibling temp file + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(argv, timeout=60):
    """Run a command and capture raw outputs; never raises."""
    record = {"cmd": list(argv)}
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        record.update(returncode=None, stdout="", stderr="", error=str(exc))
        return record
    record.update(
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
    )
    return record


_MEMORY_RE = re.compile(r"(\d+)\s*MiB", re.IGNORECASE)
_GPU_LABEL_RE = re.compile(r"^GPU\d+$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_gpu_query_csv(text):
    """Parse `nvidia-smi --query-gpu=... --format=csv` output.

    Tolerates an optional header row. Returns a list of dicts with keys
    index/name/memory_mib/compute_cap/driver_version/pci_bus_id.
    """
    rows = []
    for fields in csv.reader(io.StringIO(text.strip())):
        if not fields or not any(cell.strip() for cell in fields):
            continue
        cells = [cell.strip() for cell in fields]
        if cells[0].lower() == "index":
            continue
        if len(cells) < 6:
            raise ValueError(f"Malformed GPU query row: {cells!r}")
        memory_match = _MEMORY_RE.search(cells[2])
        rows.append(
            {
                "index": int(cells[0]),
                "name": cells[1],
                "memory_mib": int(memory_match.group(1))
                if memory_match
                else None,
                "compute_cap": cells[3],
                "driver_version": cells[4],
                "pci_bus_id": cells[5],
            }
        )
    return rows


def check_gpu_rows(rows):
    """Return a list of contract problems for parsed GPU query rows."""
    problems = []
    if len(rows) != EXPECTED_GPU_COUNT:
        problems.append(
            f"expected {EXPECTED_GPU_COUNT} GPUs, found {len(rows)}"
        )
        return problems
    indices = [row["index"] for row in rows]
    if sorted(indices) != list(range(EXPECTED_GPU_COUNT)) or len(set(indices)) != len(
        indices
    ):
        problems.append(f"GPU indices not unique 0..7: {sorted(indices)}")
    for row in rows:
        if row["name"] != EXPECTED_GPU_NAME:
            problems.append(
                f"GPU {row['index']} name {row['name']!r} != {EXPECTED_GPU_NAME!r}"
            )
        if row["memory_mib"] is None or row["memory_mib"] < MIN_GPU_MEMORY_MIB:
            problems.append(
                f"GPU {row['index']} memory {row['memory_mib']} MiB "
                f"< {MIN_GPU_MEMORY_MIB} MiB"
            )
    return problems


def parse_topo_links(text):
    """Parse `nvidia-smi topo -m` into {(row, col): link_type}.

    Only GPU<->GPU cells are retained; trailing legend columns
    (CPU Affinity, NUMA Affinity, GPU NUMA ID) are ignored.
    """
    text = _ANSI_RE.sub("", text)
    columns = []
    links = {}
    for line in text.splitlines():
        tokens = line.split()
        if not tokens or not _GPU_LABEL_RE.match(tokens[0]):
            continue
        if len(tokens) > 1 and _GPU_LABEL_RE.match(tokens[1]):
            columns = tokens[:]
            columns = [tok for tok in tokens if _GPU_LABEL_RE.match(tok)]
            continue
        row = tokens[0]
        for col, link in zip(columns, tokens[1 : 1 + len(columns)]):
            links[(row, col)] = link
    return links


def check_nv12_topology(text):
    """Return problems if any GPU pair in the topo matrix is not NV12."""
    links = parse_topo_links(text)
    gpu_labels = sorted({label for pair in links for label in pair})
    problems = []
    if not gpu_labels:
        return ["no GPU<->GPU topology parsed from nvidia-smi topo -m output"]
    for row in gpu_labels:
        for col in gpu_labels:
            if row == col:
                continue
            link = links.get((row, col))
            if link != EXPECTED_LINK_TYPE:
                problems.append(
                    f"{row}->{col} link {link!r} != {EXPECTED_LINK_TYPE!r}"
                )
    return problems


def active_mask_channels(mask):
    """Active channel indices of an action mask shaped [C, F, N, 1]."""
    arr = np.asarray(mask)[..., 0]
    return sorted(np.nonzero(np.asarray(arr).any(axis=(1, 2)))[0].tolist())


def collect_package_versions():
    """Installed distribution versions; contains no env vars or secrets."""
    import importlib.metadata

    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            versions[name.lower()] = dist.version
    return dict(sorted(versions.items()))


# ---------------------------------------------------------------------------
# host subcommand
# ---------------------------------------------------------------------------


def host_report(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    git_rev = run_command(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]
    )
    git_status = run_command(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"]
    )
    uv_lock = REPO_ROOT / "uv.lock"

    try:
        import psutil

        memory = {
            "total_bytes": psutil.virtual_memory().total,
            "available_bytes": psutil.virtual_memory().available,
        }
    except ImportError:
        memory = {"total_bytes": None, "available_bytes": None}

    disk = shutil.disk_usage(str(output_root))

    list_gpus = run_command(["nvidia-smi", "-L"])
    topo = run_command(["nvidia-smi", "topo", "-m"])
    query = run_command(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY_FIELDS}",
            "--format=csv",
        ]
    )

    problems = []
    gpu_rows = []
    if query["returncode"] != 0:
        problems.append(
            f"nvidia-smi GPU query failed rc={query['returncode']}: "
            f"{query.get('error') or query['stderr'].strip()}"
        )
    else:
        try:
            gpu_rows = parse_gpu_query_csv(query["stdout"])
        except ValueError as exc:
            problems.append(f"failed to parse GPU query output: {exc}")
        else:
            problems.extend(check_gpu_rows(gpu_rows))
    if topo["returncode"] != 0:
        problems.append(
            f"nvidia-smi topo -m failed rc={topo['returncode']}: "
            f"{topo.get('error') or topo['stderr'].strip()}"
        )
    else:
        problems.extend(check_nv12_topology(topo["stdout"]))

    report = {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "disk": {
            "path": str(output_root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "git": {
            "repo_root": str(REPO_ROOT),
            "revision": git_rev["stdout"].strip() or None,
            "revision_returncode": git_rev["returncode"],
            "dirty": bool(git_status["stdout"].strip()),
            "status_returncode": git_status["returncode"],
        },
        "uv_lock_sha256": sha256_file(uv_lock) if uv_lock.is_file() else None,
        "packages": collect_package_versions(),
        "nvidia_smi": {
            "list": list_gpus,
            "topo": topo,
            "query": query,
        },
        "gpu_contract": {
            "expected_count": EXPECTED_GPU_COUNT,
            "expected_name": EXPECTED_GPU_NAME,
            "min_memory_mib": MIN_GPU_MEMORY_MIB,
            "expected_link": EXPECTED_LINK_TYPE,
            "parsed_gpus": gpu_rows,
            "problems": problems,
            "ok": not problems,
        },
    }
    out = output_root / "host_inventory.json"
    atomic_write_json(out, report)
    report["report"] = str(out)
    return report


# ---------------------------------------------------------------------------
# loader subcommand
# ---------------------------------------------------------------------------


def _ensure_easydict():
    """Return `EasyDict`, installing a minimal shim if the package is missing."""
    try:
        from easydict import EasyDict

        return EasyDict
    except ImportError:
        pass
    import types

    class EasyDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

    module = types.ModuleType("easydict")
    module.EasyDict = EasyDict
    sys.modules.setdefault("easydict", module)
    return EasyDict


def _v21_get_episode_data_index(episode_dicts, episodes=None):
    """v2.1 `get_episode_data_index`: cumulative from/to frame offsets."""
    import itertools

    import torch

    episode_lengths = {
        ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()
    }
    if episodes is not None:
        episode_lengths = {
            ep_idx: episode_lengths[ep_idx] for ep_idx in episodes
        }
    cumulative = list(itertools.accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative[:-1]),
        "to": torch.LongTensor(cumulative),
    }


class _V21DatasetMetadata:
    """v2.1 dataset metadata: meta/info.json + meta/episodes.jsonl.

    Module-level (not function-local) so constructed dataset objects remain
    picklable across the upstream multiprocessing init pool.
    """

    def __init__(self, repo_id, root=None, revision=None, force_cache_sync=False):
        del revision, force_cache_sync
        self.repo_id = repo_id
        if root is not None:
            self.root = Path(root)
        else:
            self.root = _v21_hf_lerobot_home() / str(repo_id)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.episodes = {}
        with (self.root / "meta" / "episodes.jsonl").open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                episode = json.loads(line)
                self.episodes[int(episode["episode_index"])] = episode

    def get_episode_chunk(self, episode_index):
        return int(episode_index) // int(self.info["chunks_size"])

    def get_data_file_path(self, ep_index):
        return Path(
            self.info["data_path"].format(
                episode_chunk=self.get_episode_chunk(ep_index),
                episode_index=int(ep_index),
            )
        )


class _V21LeRobotDataset:
    """Base-class shim providing v2.1 `load_hf_dataset` (ordered concat of
    the selected episodes' parquet files)."""

    def load_hf_dataset(self):
        import datasets as hf_datasets

        data_paths = [
            str(self.root / self.meta.get_data_file_path(ep))
            for ep in self.episodes
        ]
        return hf_datasets.Dataset.from_parquet(data_paths)


def _v21_hf_lerobot_home():
    return Path(
        os.environ.get(
            "HF_LEROBOT_HOME",
            Path.home() / ".cache" / "huggingface" / "lerobot",
        )
    )


def _install_lerobot_v21_compat():
    """Provide the lerobot==0.3.3 (v2.1) surface the upstream loader imports.

    Upstream's requirements pin lerobot==0.3.3, but this environment carries
    lerobot 0.5.1 (v3.0 dataset format), where `lerobot.constants`,
    `get_episode_data_index`, and v2.1 `episodes.jsonl` metadata no longer
    exist. The loader path only needs four symbols, reproduced at module level
    (must stay picklable for the upstream init worker pool) with the original
    v2.1 semantics:

      * HF_LEROBOT_HOME                  (Path; repo_id is absolute so it only
                                          needs to be a Path prefix)
      * get_episode_data_index           (cumulative from/to frame offsets)
      * LeRobotDatasetMetadata           (meta/episodes.jsonl + info.json)
      * LeRobotDataset.load_hf_dataset   (per-episode parquet concat, in the
                                          dataset's episode order)
    """
    import types

    stubs = {
        "lerobot": {"__path__": []},
        "lerobot.constants": {"HF_LEROBOT_HOME": _v21_hf_lerobot_home()},
        "lerobot.datasets": {"__path__": []},
        "lerobot.datasets.utils": {
            "get_episode_data_index": _v21_get_episode_data_index
        },
        "lerobot.datasets.lerobot_dataset": {
            "LeRobotDataset": _V21LeRobotDataset,
            "LeRobotDatasetMetadata": _V21DatasetMetadata,
        },
    }
    for dotted, attrs in stubs.items():
        if dotted in sys.modules:
            continue
        module = types.ModuleType(dotted)
        for name, value in attrs.items():
            setattr(module, name, value)
        sys.modules[dotted] = module


def _import_upstream(zero_wam_root):
    zero_wam_root = Path(zero_wam_root).resolve()
    sys.path.insert(0, str(zero_wam_root))
    _ensure_easydict()
    _install_lerobot_v21_compat()

    # wan_va/__init__.py eagerly imports configs/distributed/modules, which
    # require packages (e.g. transformers) not needed by the latent dataset
    # path. Register path-only package handles so the two submodule files we
    # need load directly, without executing the heavyweight package __init__s.
    import types

    for dotted, subdir in (
        ("wan_va", "wan_va"),
        ("wan_va.configs", "wan_va/configs"),
        ("wan_va.dataset", "wan_va/dataset"),
    ):
        if dotted in sys.modules:
            continue
        stub = types.ModuleType(dotted)
        stub.__path__ = [str(zero_wam_root / subdir)]
        sys.modules[dotted] = stub

    from wan_va.configs.va_robotwin_cfg import load_robotwin_norm_stat
    from wan_va.dataset.icl_lerobot_latent_dataset import (
        MultiICLLeRobotLatentDataset,
    )

    return MultiICLLeRobotLatentDataset, load_robotwin_norm_stat


def _split_config(
    split,
    loader_root,
    latent_root,
    zero_wam_root,
    load_norm_stat,
    easydict_cls,
):
    cfg = easydict_cls()
    cfg.dataset_path = str(Path(loader_root) / split)
    cfg.icl_manifest_path = str(Path(loader_root) / split / "icl_manifest.json")
    cfg.robot_latent_path = str(Path(latent_root) / split)
    cfg.human_latent_path = str(Path(latent_root) / split / "human_latents" / "so101")
    cfg.env_type = "none"
    cfg.height = HEIGHT
    cfg.width = WIDTH
    cfg.action_dim = ACTION_DIM
    cfg.action_per_frame = ACTION_PER_FRAME
    cfg.obs_cam_keys = list(CAMERAS)
    cfg.text_encoder_type = "umt_dense"
    cfg.empty_emb_path = str(
        Path(zero_wam_root) / "wan_va" / "assets" / "empty_text_emb.pt"
    )
    cfg.cfg_prob = 0
    cfg.init_worker = 8
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.excluded_task_names = []
    cfg.expected_num_train_tasks = (
        EXPECTED_TRAIN_TASKS if split == "train" else EXPECTED_VAL_TASKS
    )
    cfg.enable_dataset_index_cache = True
    cfg.rebuild_dataset_index_cache = False
    cfg.norm_stat = load_norm_stat(
        Path(loader_root) / split / "meta" / "action_stats.json"
    )
    return cfg


def _find_episode_idx(dataset, task, episode_index):
    """Global index of (task, episode_index) inside a MultiICL dataset."""
    offset = 0
    for sub in dataset._datasets:
        if sub.repo_id and Path(sub.repo_id).name == task:
            for i, meta in enumerate(sub.new_metas):
                if int(meta["episode_index"]) == episode_index:
                    return offset + i
            raise KeyError(f"{task}: episode {episode_index} not in valid metas")
        offset += len(sub)
    raise KeyError(f"task {task} not found in dataset")


def _tensor_sha256(tensor):
    arr = np.asarray(
        tensor.detach().float().cpu().contiguous().numpy()
    )
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sample_report(ds, tag, idx):
    """Fetch one item twice; assert determinism and the tensor contract."""
    import torch

    item, item2 = ds[idx], ds[idx]
    problems = []
    for key in TENSOR_KEYS:
        if not torch.equal(item[key], item2[key]):
            problems.append(f"{key} not deterministic across repeated fetch")

    latents = item["latents"]
    if latents.ndim != 4 or (
        latents.shape[0],
        latents.shape[2],
        latents.shape[3],
    ) != (ROBOT_CHANNELS, ROBOT_H, ROBOT_W):
        problems.append(
            f"latents shape {tuple(latents.shape)} != [48,F,{ROBOT_H},{ROBOT_W}]"
        )
    robot_frames = latents.shape[1] if latents.ndim == 4 else None

    icl = item["icl_latents"]
    if icl.ndim != 4 or (icl.shape[0], icl.shape[2], icl.shape[3]) != (
        HUMAN_CHANNELS,
        HUMAN_H,
        HUMAN_W,
    ):
        problems.append(
            f"icl_latents shape {tuple(icl.shape)} != [48,T,{HUMAN_H},{HUMAN_W}]"
        )

    actions = item["actions"]
    if actions.ndim != 4 or actions.shape[0] != ACTION_DIM or (
        actions.shape[2],
        actions.shape[3],
    ) != (ACTION_PER_FRAME, 1):
        problems.append(
            f"actions shape {tuple(actions.shape)} != [30,F,{ACTION_PER_FRAME},1]"
        )
    elif robot_frames is not None and actions.shape[1] != robot_frames:
        problems.append(
            f"actions F={actions.shape[1]} != latents F={robot_frames}"
        )

    active = active_mask_channels(
        item["actions_mask"].detach().cpu().numpy()
    )
    if active != EXPECTED_ACTIVE_CHANNELS:
        problems.append(f"action mask active channels {active}")

    for key in ("latents", "actions", "text_emb", "icl_latents", "icl_text_emb"):
        if not torch.isfinite(item[key].float()).all():
            problems.append(f"{key} not finite")

    tensors = {
        key: {
            "shape": list(item[key].shape),
            "sha256": _tensor_sha256(item[key]),
        }
        for key in TENSOR_KEYS
    }
    return {
        "tag": tag,
        "idx": int(idx),
        "icl_sample_id": item.get("icl_sample_id"),
        "icl_human_latent_path": item.get("icl_human_latent_path"),
        "problems": problems,
        "tensors": tensors,
    }


def loader_report(loader_root, latent_root, zero_wam_root, output_root):
    # upstream resolves dataset roots as HF_LEROBOT_HOME / repo_id, which only
    # lands correctly when repo_id (derived from dataset_path) is absolute
    loader_root = Path(loader_root).resolve()
    latent_root = Path(latent_root).resolve()
    zero_wam_root = Path(zero_wam_root).resolve()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timings = {}
    t_start = time.monotonic()

    easydict_cls = _ensure_easydict()
    MultiICLLeRobotLatentDataset, load_norm_stat = _import_upstream(
        zero_wam_root
    )

    t0 = time.monotonic()
    train_ds = MultiICLLeRobotLatentDataset(
        config=_split_config(
            "train", loader_root, latent_root, zero_wam_root,
            load_norm_stat, easydict_cls,
        )
    )
    timings["train_construct_s"] = round(time.monotonic() - t0, 3)

    t0 = time.monotonic()
    val_ds = MultiICLLeRobotLatentDataset(
        config=_split_config(
            "val", loader_root, latent_root, zero_wam_root,
            load_norm_stat, easydict_cls,
        )
    )
    timings["val_construct_s"] = round(time.monotonic() - t0, 3)

    problems = []
    if len(train_ds) != EXPECTED_TRAIN_LEN:
        problems.append(f"train len {len(train_ds)} != {EXPECTED_TRAIN_LEN}")
    if len(val_ds) != EXPECTED_VAL_LEN:
        problems.append(f"val len {len(val_ds)} != {EXPECTED_VAL_LEN}")

    picks = []
    metas_flat = []
    offset = 0
    for sub in train_ds._datasets:
        for meta in sub.new_metas:
            metas_flat.append((offset, meta))
            offset += 1
    by_len = sorted(
        metas_flat,
        key=lambda pair: int(pair[1]["end_frame"]) - int(pair[1]["start_frame"]),
    )
    picks.append(("train:shortest", by_len[0][0]))
    picks.append(("train:longest", by_len[-1][0]))

    tasks_sorted = sorted(
        Path(sub.repo_id).name for sub in train_ds._datasets
    )
    for pos in (0, 15, 30, 45, 60):
        task = tasks_sorted[pos]
        picks.append((f"train:{task}", _find_episode_idx(train_ds, task, 0)))

    val_tasks = sorted(Path(sub.repo_id).name for sub in val_ds._datasets)
    for task in val_tasks:
        picks.append(
            (f"val:{task}", _find_episode_idx(val_ds, task, 0))
        )

    samples = []
    t0 = time.monotonic()
    for tag, idx in picks:
        ds = val_ds if tag.startswith("val:") else train_ds
        samples.append(_sample_report(ds, tag, idx))
    timings["fetch_s"] = round(time.monotonic() - t0, 3)
    timings["total_s"] = round(time.monotonic() - t_start, 3)

    for sample in samples:
        problems.extend(
            f"{sample['tag']}[{sample['idx']}]: {problem}"
            for problem in sample["problems"]
        )

    report = {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "loader_root": str(loader_root),
        "latent_root": str(latent_root),
        "zero_wam_root": str(zero_wam_root),
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "train_tasks": len(train_ds._datasets),
        "val_tasks": len(val_ds._datasets),
        "train_index_cache": {
            "hits": train_ds.index_cache_hits,
            "misses": train_ds.index_cache_misses,
            "hf_cache_hits": train_ds.hf_cache_hits,
        },
        "val_index_cache": {
            "hits": val_ds.index_cache_hits,
            "misses": val_ds.index_cache_misses,
            "hf_cache_hits": val_ds.hf_cache_hits,
        },
        "train_excluded_roots": [
            str(path) for path in train_ds.excluded_dataset_roots
        ],
        "val_excluded_roots": [
            str(path) for path in val_ds.excluded_dataset_roots
        ],
        "timings": timings,
        "samples": samples,
        "problems": problems,
        "ok": not problems,
    }
    out = output_root / "loader_smoke.json"
    atomic_write_json(out, report)
    report["report"] = str(out)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_loader_args(parser):
    parser.add_argument("--loader-root", required=True)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--zero-wam-root", required=True)
    parser.add_argument("--output-root", required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zero_wam.so101_a100_preflight")
    sub = parser.add_subparsers(dest="command", required=True)

    host_p = sub.add_parser("host", help="host inventory + GPU contract")
    host_p.add_argument("--output-root", required=True)

    loader_p = sub.add_parser("loader", help="dataset loader smoke test")
    _add_loader_args(loader_p)

    all_p = sub.add_parser("all", help="host then loader")
    _add_loader_args(all_p)

    args = parser.parse_args(argv)

    if args.command == "host":
        report = host_report(args.output_root)
        ok = report["gpu_contract"]["ok"]
        print(f"host_inventory: {report['report']} ok={ok}")
        for problem in report["gpu_contract"]["problems"]:
            print(f"  PROBLEM: {problem}")
        return 0 if ok else 1

    if args.command in ("loader", "all"):
        host_ok = True
        if args.command == "all":
            report = host_report(args.output_root)
            host_ok = report["gpu_contract"]["ok"]
            print(f"host_inventory: {report['report']} ok={host_ok}")
        report = loader_report(
            args.loader_root,
            args.latent_root,
            args.zero_wam_root,
            args.output_root,
        )
        print(f"loader_smoke: {report['report']} ok={report['ok']}")
        print(f"timings: {json.dumps(report['timings'])}")
        for problem in report["problems"]:
            print(f"  PROBLEM: {problem}")
        return 0 if (report["ok"] and host_ok) else 1

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
