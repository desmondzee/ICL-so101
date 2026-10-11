"""Non-Modal 8xA100 production launcher for Zero-WAM SO-101 ICL sim training.

Runs under `torchrun --standalone --nproc_per_node=8 -m zero_wam.so101_a100_train`
using the pinned upstream stack in `.venv-zero-wam`. Wraps the upstream
`wan_va.train.Trainer` with the repo's training primitives (every-microstep
gradient sync, task-balanced loader, resumable checkpointing) — no upstream
source changes.

All upstream/torch-heavy imports stay inside functions so the pure manifest,
hashing, and run-directory helpers remain unit-testable in any environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from zero_wam.so101_a100_preflight import atomic_write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
ZERO_WAM_ROOT = REPO_ROOT / "third_party" / "Zero-WAM"
PREFLIGHT_ARTIFACTS = REPO_ROOT / "outputs" / "zero_wam" / "a100_preflight"

EXPECTED_WORLD_SIZE = 8
EXPECTED_TRAIN_TASKS = 61
DATASET_NAME = "so101_simulated_icl"
CAMERAS = ("observation.images.front", "observation.images.wrist")
HEIGHT, WIDTH, ACTION_DIM, ACTION_PER_FRAME = 224, 288, 30, 8
ACTION_LAYOUT = (14, 14, 2)

def lr_multiplier(step, *, scheduler_type, warmup_steps, total_steps,
                  peak_lr, final_lr):
    """One-indexed warmup + constant/cosine decay multiplier for LambdaLR.

    Warmup update `w` (LambdaLR step w-1) reaches w/warmup_steps; cosine
    progress 0 starts at the first post-warmup step and 1 at total_steps-1.
    """
    step = int(step)
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    if not 0 <= final_lr <= peak_lr or peak_lr <= 0:
        raise ValueError(
            f"need 0 <= final_lr <= peak_lr and peak_lr > 0, got "
            f"final_lr={final_lr} peak_lr={peak_lr}")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            f"need 0 <= warmup_steps < total_steps, got "
            f"warmup_steps={warmup_steps} total_steps={total_steps}")
    if scheduler_type not in {"constant", "cosine"}:
        raise ValueError(f"unknown scheduler_type: {scheduler_type!r}")
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if scheduler_type == "constant":
        return 1.0
    progress = min(
        max((step - warmup_steps + 1) / (total_steps - warmup_steps), 0.0),
        1.0)
    minimum = final_lr / peak_lr
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(
        math.pi * progress))


def parse_validation_extra_steps(value):
    if not value:
        return []
    steps = sorted({int(item.strip()) for item in str(value).split(",")})
    if any(step <= 0 for step in steps):
        raise ValueError("validation extra steps must be positive")
    return steps


# run-dir files the wrapper script legitimately creates before python starts
_INFRA_FILES = {"run.log", "run.pid"}
_RUN_SENTINELS = {
    "run_manifest.json",
    "wandb_run_id.txt",
    "completed.json",
    "failed.json",
}


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_sha256(root):
    """SHA256 of `<root>/release.json` — immutable input identity."""
    path = Path(root) / "release.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing release manifest: {path}")
    return sha256_file(path)


def load_so101_norm_stat(path):
    """Map five SO-101 joints and one gripper into the released 30-D layout."""
    payload = json.loads(Path(path).read_text())
    stats = payload["norm_stats"]
    hand = stats["action.hand.position"]
    effector = stats["action.effector.position"]
    if len(hand["q01"]) != 5 or len(effector["q01"]) != 1:
        raise ValueError("expected five SO-101 joints and one gripper channel")

    def padded(which):
        return (
            list(hand[which])
            + [0.0] * (ACTION_LAYOUT[0] - len(hand[which]))
            + [0.0] * ACTION_LAYOUT[1]
            + list(effector[which])
            + [0.0] * (ACTION_LAYOUT[2] - len(effector[which]))
        )

    return {"q01": padded("q01"), "q99": padded("q99")}


def tree_manifest_sha256(root):
    """Content hash over every file under `root` (relpath + size + sha256)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"missing model tree: {root}")
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files under {root}")
    for path in files:
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode())
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def git_code_fingerprint(repo_root=REPO_ROOT):
    """Git HEAD revision plus a hash of working-tree diff + status."""
    import subprocess

    def _git(*argv):
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *argv],
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout

    rc, head = _git("rev-parse", "HEAD")
    revision = head.strip() if rc == 0 else None
    _, diff = _git("diff", "HEAD")
    _, status = _git("status", "--porcelain")
    dirty_diff_sha256 = hashlib.sha256(
        (diff + "\n--status--\n" + status).encode()
    ).hexdigest()
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "dirty_diff_sha256": dirty_diff_sha256,
    }


def config_identity_payload(
    *,
    loader_root,
    latent_root,
    model_path,
    world_size,
    seed,
    gradient_accumulation_steps,
    learning_rate=1e-4,
    warmup_steps=200,
    validation_interval=500,
    lr_scheduler_type="constant",
    final_learning_rate=0.0,
    validation_extra_steps=(),
    num_steps=None,
    save_interval=None,
    repo_root=REPO_ROOT,
):
    """Canonical dict covering scientific settings + immutable input identity.

    `num_steps` is excluded for constant-LR runs so they can be extended. It is
    part of cosine-schedule identity because changing it changes every decay LR.
    """
    schedule_steps = (
        int(num_steps) if str(lr_scheduler_type) == "cosine" else None)
    return {
        "schema": 1,
        "inputs": {
            "loader_root": str(loader_root),
            "loader_release_sha256": release_sha256(loader_root),
            "latent_root": str(latent_root),
            "latent_release_sha256": release_sha256(latent_root),
            "model_path": str(model_path),
            "model_tree_sha256": tree_manifest_sha256(model_path),
        },
        "dataset": {
            "name": DATASET_NAME,
            "weight": 1.0,
            "split": "train",
            "obs_cam_keys": list(CAMERAS),
            "height": HEIGHT,
            "width": WIDTH,
            "action_dim": ACTION_DIM,
            "action_per_frame": ACTION_PER_FRAME,
            "env_type": "none",
            "text_encoder_type": "umt_dense",
            "empty_emb_path": "wan_va/assets/empty_text_emb.pt",
            "excluded_task_names": [],
            "expected_num_train_tasks": EXPECTED_TRAIN_TASKS,
            "norm_stat_source": "meta/action_stats.json",
            "enable_dataset_index_cache": True,
            "rebuild_dataset_index_cache": False,
            "init_worker": 1,
        },
        "hyperparameters": {
            "learning_rate": float(learning_rate),
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "warmup_steps": int(warmup_steps),
            "batch_size": 1,
            "drop_icl": 0.1,
            "droptext_target": 0.4,
            "cfg_prob": 0.4,
            "load_worker": 0,
            "mcp_loss_weights": [0.5, 0.25, 0.15, 0.1],
            "max_frame_chunk_size": 4,
            "frame_chunk_size": 0,
            "param_dtype": "bfloat16",
            "max_norm": 1.0,
            "skip_step_grad_norm_multiplier": 20.0,
            "gc_interval": 50,
            "save_interval": save_interval,
            "validation_interval": int(validation_interval),
            "validation_extra_steps": [
                int(step) for step in validation_extra_steps],
            "lr_scheduler_type": str(lr_scheduler_type),
            "final_learning_rate": float(final_learning_rate),
            "lr_schedule_steps": schedule_steps,
            "seed": seed,
            "world_size": world_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch": world_size * gradient_accumulation_steps,
            "sync_every_microstep": True,
            "task_balanced_sampler": True,
        },
        "code": git_code_fingerprint(repo_root),
    }


def config_sha256(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assert_fresh_run_dir(run_dir):
    """Refuse to overwrite a nonempty existing run.

    `run.log`/`run.pid` (wrapper-script infrastructure) do not count as run
    content; any prior-run sentinel or other file aborts.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    contents = {p.name for p in run_dir.iterdir()}
    leftovers = contents - _INFRA_FILES
    if contents & _RUN_SENTINELS or leftovers:
        raise FileExistsError(
            f"refusing fresh run over existing content in {run_dir}: "
            f"{sorted(leftovers | (contents & _RUN_SENTINELS))}"
        )


def load_resume_metadata(run_dir, config_sha):
    """Read prior run manifest + wandb id; require matching config hash."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    wid_path = run_dir / "wandb_run_id.txt"
    if not manifest_path.is_file() or not wid_path.is_file():
        raise FileNotFoundError(
            f"resume requires run_manifest.json and wandb_run_id.txt "
            f"under {run_dir}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != config_sha:
        raise ValueError(
            "resume config hash mismatch: "
            f"{manifest.get('config_sha256')} != {config_sha}"
        )
    wandb_run_id = wid_path.read_text().strip()
    if not wandb_run_id:
        raise ValueError(f"empty wandb run id in {wid_path}")
    return manifest, wandb_run_id


def build_run_manifest(
    *, args, config_sha, wandb_run_id, scientific_config=None,
    preflight_dir=PREFLIGHT_ARTIFACTS
):
    """Nonsecret launch manifest persisted under the run root."""
    return {
        "schema": 1,
        "run_id": args.run_id,
        "created_utc": _utcnow(),
        "config_sha256": config_sha,
        "scientific_config": scientific_config,
        "wandb_run_id": wandb_run_id,
        "wandb": {
            "project": args.wandb_project,
            "group": args.wandb_group,
            "job_type": "train",
            "disabled": bool(args.disable_wandb),
        },
        "num_steps": int(args.num_steps),
        "save_interval": int(args.save_interval),
        "validation_interval": int(args.validation_interval),
        "validation_extra_steps": parse_validation_extra_steps(
            args.validation_extra_steps),
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "lr_scheduler_type": str(args.lr_scheduler),
        "final_learning_rate": float(args.final_learning_rate),
        "lr_schedule_steps": (
            int(args.num_steps) if args.lr_scheduler == "cosine" else None),
        "seed": int(args.seed),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch": EXPECTED_WORLD_SIZE
        * int(args.gradient_accumulation_steps),
        "world_size": EXPECTED_WORLD_SIZE,
        "paths": {
            "loader_root": str(args.loader_root),
            "latent_root": str(args.latent_root),
            "model_path": str(args.model_path),
            "run_root": str(args.run_root),
        },
        "preflight_artifacts": {
            "dir": str(preflight_dir),
            "host_inventory": str(preflight_dir / "host_inventory.json"),
            "loader_smoke": str(preflight_dir / "loader_smoke.json"),
        },
        "code": git_code_fingerprint(),
    }


def build_completed_payload(
    *, run_id, config_sha, final_step, checkpoint_path, exposure_path
):
    return {
        "schema": 1,
        "success": True,
        "run_id": run_id,
        "completed_utc": _utcnow(),
        "final_step": int(final_step),
        "config_sha256": config_sha,
        "checkpoint_path": str(checkpoint_path),
        "episode_exposure_path": str(exposure_path),
    }


def build_failed_payload(*, run_id, config_sha, step, error):
    return {
        "schema": 1,
        "success": False,
        "run_id": run_id,
        "failed_utc": _utcnow(),
        "step": step,
        "config_sha256": config_sha,
        "error": f"{type(error).__name__}: {error}",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="zero_wam.so101_a100_train",
        description="8xA100 Zero-WAM SO-101 ICL sim training launcher",
    )
    parser.add_argument("--loader-root", required=True)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-extra-steps", default="")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="micro-batches per optimizer step; effective batch is "
        "world_size * this value",
    )
    parser.add_argument(
        "--lr-scheduler", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--final-learning-rate", type=float, default=0.0)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="zero-wam-so101")
    parser.add_argument("--wandb-group", default="sim-train-v1-a100")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args(argv)


def _rank_env():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"WORLD_SIZE must be {EXPECTED_WORLD_SIZE}, got {world}")
    if not 0 <= local_rank < EXPECTED_WORLD_SIZE or rank != local_rank:
        raise RuntimeError(
            f"expected single-node local ranks 0..7, got RANK={rank} "
            f"LOCAL_RANK={local_rank}"
        )
    return rank, local_rank, world


def _import_training_stack():
    """Import the upstream training stack from the pinned submodule checkout."""
    if str(ZERO_WAM_ROOT) not in sys.path:
        sys.path.insert(0, str(ZERO_WAM_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from wan_va.configs.va_robotwin_cfg import load_robotwin_norm_stat
    from wan_va.configs.va_robotwin_train_cfg import va_robotwin_train_cfg
    from wan_va.distributed.util import init_distributed
    from wan_va.train import Trainer
    from wan_va.utils import init_logger

    return {
        "Trainer": Trainer,
        "init_distributed": init_distributed,
        "init_logger": init_logger,
        "load_robotwin_norm_stat": load_robotwin_norm_stat,
        "base_config": va_robotwin_train_cfg,
    }


def _build_config(args, rank, local_rank, world, stack):
    """Deepcopy va_robotwin_train_cfg with the frozen SO-101 launch settings."""
    import easydict
    import torch

    config = deepcopy(stack["base_config"])
    config.model_path = str(Path(args.model_path).resolve())
    config.save_root = str(Path(args.run_root).resolve() / args.run_id)
    config.resume_from = args.resume_from
    config.num_steps = int(args.num_steps)
    config.save_interval = int(args.save_interval)
    config.learning_rate = float(args.learning_rate)
    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.01
    config.warmup_steps = int(args.warmup_steps)
    config.validation_interval = int(args.validation_interval)
    config.lr_scheduler_type = args.lr_scheduler
    config.final_learning_rate = float(args.final_learning_rate)
    config.batch_size = 1
    config.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    config.drop_icl = 0.1
    config.droptext_target = 0.4
    config.cfg_prob = config.droptext_target
    config.load_worker = 0
    config.init_worker = 1
    config.mcp_loss_weights = [0.5, 0.25, 0.15, 0.1]
    config.max_frame_chunk_size = 4
    config.frame_chunk_size = 0
    config.param_dtype = torch.bfloat16
    config.max_norm = 1.0
    config.skip_step_grad_norm_multiplier = 20.0
    config.gc_interval = 50
    config.enable_dataset_index_cache = True
    config.rebuild_dataset_index_cache = False
    config.enable_wandb = False  # attached manually on rank 0 after construction
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world

    loader_root = Path(args.loader_root).resolve()
    latent_root = Path(args.latent_root).resolve()
    ds = easydict.EasyDict()
    ds.dataset_path = str(loader_root / "train")
    ds.icl_manifest_path = str(loader_root / "train" / "icl_manifest.json")
    ds.robot_latent_path = str(latent_root / "train")
    ds.human_latent_path = str(
        latent_root / "train" / "human_latents" / "so101"
    )
    ds.env_type = "none"
    ds.height = HEIGHT
    ds.width = WIDTH
    ds.action_dim = ACTION_DIM
    ds.action_per_frame = ACTION_PER_FRAME
    ds.obs_cam_keys = list(CAMERAS)
    ds.text_encoder_type = "umt_dense"
    ds.empty_emb_path = str(
        ZERO_WAM_ROOT / "wan_va" / "assets" / "empty_text_emb.pt"
    )
    ds.cfg_prob = config.droptext_target
    ds.init_worker = config.init_worker
    ds.rank = rank
    ds.local_rank = local_rank
    ds.world_size = world
    ds.excluded_task_names = []
    ds.expected_num_train_tasks = EXPECTED_TRAIN_TASKS
    ds.enable_dataset_index_cache = True
    ds.rebuild_dataset_index_cache = False
    ds.norm_stat = load_so101_norm_stat(
        loader_root / "train" / "meta" / "action_stats.json")
    config.norm_stat = deepcopy(ds.norm_stat)
    config.dataset_sources = [
        {"name": DATASET_NAME, "weight": 1.0, "config": ds}
    ]
    return config


def _new_wandb_run_id():
    try:
        from wandb.util import generate_id

        return generate_id()
    except Exception:
        import secrets

        return secrets.token_hex(4)


def _latest_checkpoint_dir(save_dir):
    save_dir = Path(save_dir)
    candidates = sorted(
        save_dir.glob("checkpoint_step_*"),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    return candidates[-1] if candidates else None


class LocalMetricsWandbProxy:
    """Rank-0 wandb proxy: every log() call is first appended to a local
    JSONL under the run dir, then delegated with the original arguments.

    All non-log attributes (Table, Artifact, plot, log_artifact, finish, run)
    delegate untouched to the wrapped module.
    """

    def __init__(self, module, jsonl_path):
        self._module = module
        self._jsonl_path = Path(jsonl_path)
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def jsonl_path(self):
        return self._jsonl_path

    def __getattr__(self, name):
        return getattr(self._module, name)

    def log(self, data, step=None, commit=None, **kwargs):
        scalars = {}
        non_scalar = []
        for key, value in data.items():
            if value is None or isinstance(value, (bool, int, str)) or (
                    isinstance(value, float) and math.isfinite(value)):
                scalars[str(key)] = value
            else:
                non_scalar.append(str(key))
        record = {
            "schema": 1,
            "ts_utc": _utcnow(),
            "step": int(step) if step is not None else None,
            "commit": commit,
            "data": scalars,
        }
        if non_scalar:
            record["non_scalar_keys"] = sorted(non_scalar)
        line = json.dumps(record, sort_keys=True) + "\n"
        # durable local metrics are required — any write failure propagates
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return self._module.log(data, step=step, commit=commit, **kwargs)


def _wandb_attach_rank0(trainer, args, manifest, wandb_run_id, resuming):
    """Rank-0-only wandb init + attach; other ranks keep enable_wandb False."""
    import wandb

    entity = os.environ.get("WANDB_TEAM_NAME") or None
    wandb.init(
        entity=entity,
        project=args.wandb_project,
        group=args.wandb_group,
        job_type="train",
        id=wandb_run_id,
        resume="must" if resuming else "never",
        config=manifest,
    )
    run_dir = Path(args.run_root) / args.run_id
    trainer.wandb = LocalMetricsWandbProxy(
        wandb, run_dir / "metrics.jsonl")
    trainer.config.enable_wandb = True


def main(argv=None):
    args = parse_args(argv)
    if int(args.gradient_accumulation_steps) <= 0:
        raise SystemExit(
            "--gradient-accumulation-steps must be a positive integer"
        )
    args.loader_root = str(Path(args.loader_root).resolve())
    args.latent_root = str(Path(args.latent_root).resolve())
    args.model_path = str(Path(args.model_path).resolve())
    args.run_root = str(Path(args.run_root).resolve())
    if args.resume_from:
        args.resume_from = str(Path(args.resume_from).resolve())

    rank, local_rank, world = _rank_env()
    run_dir = Path(args.run_root) / args.run_id
    resuming = bool(args.resume_from)

    # .env carries WANDB_* etc. — loaded into the process, never printed
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    stack = _import_training_stack()
    stack["init_logger"]()
    config = _build_config(args, rank, local_rank, world, stack)
    payload = config_identity_payload(
        loader_root=args.loader_root,
        latent_root=args.latent_root,
        model_path=args.model_path,
        world_size=world,
        seed=args.seed,
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        save_interval=int(args.save_interval),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        validation_interval=int(args.validation_interval),
        validation_extra_steps=parse_validation_extra_steps(
            args.validation_extra_steps),
        lr_scheduler_type=str(args.lr_scheduler),
        final_learning_rate=float(args.final_learning_rate),
    )
    config_sha = config_sha256(payload)

    wandb_run_id = None
    if resuming:
        _, wandb_run_id = load_resume_metadata(run_dir, config_sha)
    else:
        assert_fresh_run_dir(run_dir)

    stack["init_distributed"](world, local_rank, rank)
    import torch.distributed as dist

    trainer = None
    try:
        # every rank computes the hash independently; all must agree with rank 0
        box = [config_sha]
        dist.broadcast_object_list(box, src=0)
        if box[0] != config_sha:
            raise RuntimeError("config hash disagreement across ranks")

        if not resuming:
            if rank == 0:
                run_dir.mkdir(parents=True, exist_ok=True)
                wandb_run_id_local = _new_wandb_run_id()
                manifest = build_run_manifest(
                    args=args, config_sha=config_sha,
                    wandb_run_id=wandb_run_id_local,
                    scientific_config=payload,
                )
                atomic_write_json(run_dir / "run_manifest.json", manifest)
                (run_dir / "wandb_run_id.txt").write_text(
                    wandb_run_id_local + "\n"
                )
                box = [wandb_run_id_local]
            else:
                box = [None]
            dist.broadcast_object_list(box, src=0)
            wandb_run_id = box[0]
            dist.barrier()

        trainer = stack["Trainer"](config)

        # replace upstream scheduler unconditionally: LambdaLR with the frozen
        # one-indexed warmup + constant/cosine schedule (before any resume load)
        from torch.optim.lr_scheduler import LambdaLR

        peak_lr = float(config.learning_rate)
        final_lr = float(args.final_learning_rate)
        scheduler_type = str(args.lr_scheduler)
        warmup = int(config.warmup_steps)
        total = int(config.num_steps)
        trainer.lr_scheduler = LambdaLR(
            trainer.optimizer,
            lr_lambda=lambda step: lr_multiplier(
                step, scheduler_type=scheduler_type, warmup_steps=warmup,
                total_steps=total, peak_lr=peak_lr, final_lr=final_lr))

        from zero_wam.training_checkpoint import (
            install_resumable_checkpointing,
            load_training_state,
        )
        from zero_wam.training_primitives import (
            install_every_microstep_gradient_sync,
            install_task_balanced_loader,
        )

        install_every_microstep_gradient_sync(trainer)
        install_task_balanced_loader(trainer, seed=args.seed)
        install_resumable_checkpointing(trainer, config_sha, wandb_run_id)

        if resuming:
            resumed_step = load_training_state(
                trainer, args.resume_from, config_sha, wandb_run_id
            )
            if rank == 0:
                print(f"resumed at optimizer step {resumed_step}", flush=True)

        if rank == 0 and not args.disable_wandb:
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text()
            )
            _wandb_attach_rank0(
                trainer, args, manifest, wandb_run_id, resuming
            )

        from zero_wam.so101_validation import install_validation
        install_validation(
            trainer,
            loader_root=args.loader_root,
            latent_root=args.latent_root,
            run_dir=run_dir,
            validation_interval=args.validation_interval,
            validation_extra_steps=parse_validation_extra_steps(
                args.validation_extra_steps),
        )

        trainer.train()

        if trainer.step % int(config.save_interval) != 0:
            if rank == 0:
                print(
                    f"final step {trainer.step} not at save interval; "
                    "writing final checkpoint",
                    flush=True,
                )
            trainer.save_checkpoint()

        dist.barrier()
        if rank == 0:
            ckpt_dir = _latest_checkpoint_dir(trainer.save_dir)
            atomic_write_json(
                run_dir / "completed.json",
                build_completed_payload(
                    run_id=args.run_id,
                    config_sha=config_sha,
                    final_step=trainer.step,
                    checkpoint_path=ckpt_dir,
                    exposure_path=(
                        ckpt_dir / "episode_exposure.json"
                        if ckpt_dir is not None
                        else None
                    ),
                ),
            )
        return 0
    except Exception as exc:
        if rank == 0:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(
                    run_dir / "failed.json",
                    build_failed_payload(
                        run_id=args.run_id,
                        config_sha=config_sha,
                        step=getattr(trainer, "step", None)
                        if trainer is not None
                        else None,
                        error=exc,
                    ),
                )
            except Exception:
                pass
        raise
    finally:
        if rank == 0 and trainer is not None:
            wandb_mod = getattr(trainer, "wandb", None)
            if wandb_mod is not None:
                try:
                    wandb_mod.finish()
                except Exception:
                    pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
