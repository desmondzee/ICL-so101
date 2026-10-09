"""Bounded full-episode feasibility benchmark for the SO-101 Zero-WAM train corpus.

Runs under torchrun with 8 ranks (one H100 each): builds the frozen top-64 longest
episodes by source `length`, resolves them through the upstream
`MultiICLLeRobotLatentDataset`, adapts tensor SHAPES in memory to the final geometry
(robot [48,F,14,36], human ICL [48,T,20,28]) — provisional latents are NOT re-encoded —
then executes one exact effective-batch-64 optimizer boundary: microsteps 0..7 of
`Trainer._train_step` at learning rate 0 (real AdamW step still allocates optimizer
state; weights are preserved).

    torchrun --standalone --nproc_per_node=8 so101_full_episode_benchmark.py \
        --data-root /sim/data --weights /weights/zero-wam-pretrain \
        --report-dir /sim/reports/full_episode_batch64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ----- pure helpers (unit-testable) -------------------------------------------------------------------

TOP_N = 64
MICROSTEPS = 8            # microstep*8 + rank covers 0..63
ROBOT_LATENT_HW = (14, 36)   # two cameras, each 224x288 -> 14x18, concatenated along W
ICL_LATENT_HW = (20, 28)     # human 320x448 -> 20x28
MODEL_ACTION_DIM = 30
ACTION_H = 8
ACTIVE_CHANNELS = (0, 1, 2, 3, 4, 28)
SMI_QUERY = ("timestamp,index,utilization.gpu,utilization.memory,"
             "memory.used,memory.total,power.draw")


def top64_rows(rows: list[dict]) -> list[tuple[str, int, int]]:
    """All train episode rows -> top-N selection (task, episode, length) sorted by
    length desc, then task, then episode index."""
    entries = [(int(r["episode_index"]), str(r["task"]), int(r["length"]))
               for r in rows]
    entries.sort(key=lambda t: (-t[2], t[1], t[0]))
    return [(t, e, n) for e, t, n in entries[:TOP_N]]


def top64_hash(selection: list[tuple[str, int, int]]) -> str:
    return hashlib.sha256(json.dumps(selection).encode()).hexdigest()


def selected_tasks(selection: list[tuple[str, int, int]]) -> list[str]:
    """Sorted unique task names covered by the frozen selection."""
    return sorted({t for t, _, _ in selection})


def robot_latent_frames(source_length: int, stride: int = 2, temporal: int = 4) -> int:
    """stride-2 frame ids trimmed to 4k+1 -> VAE latent frames (1 + floor((n-1)/4))."""
    n = len(range(0, int(source_length), stride))
    n = (n - 1) // temporal * temporal + 1
    return (n - 1) // temporal + 1


def adapt_robot_latent(latent, hw=ROBOT_LATENT_HW):
    """[C,T,H,W] or [B,C,T,H,W] -> same ranks with H,W -> hw. Spatial bilinear
    resize only; B, C, T and dtype preserved. In-memory only."""
    import torch.nn.functional as F

    if latent.dim() == 4:
        t = latent.unsqueeze(0)
    elif latent.dim() == 5:
        t = latent
    else:
        raise ValueError(f"expected rank-4 or rank-5 latent, got {tuple(latent.shape)}")
    B, C, T = t.shape[:3]
    out = F.interpolate(t.permute(0, 2, 1, 3, 4).reshape(B * T, C, *t.shape[-2:]).float(),
                        size=hw, mode="bilinear", align_corners=False)
    out = out.reshape(B, T, C, *hw).permute(0, 2, 1, 3, 4).to(latent.dtype)
    return out[0] if latent.dim() == 4 else out


def adapt_icl_latent(latent, hw=ICL_LATENT_HW):
    """[C,T,H,W] or [B,C,T,H,W] -> [*,C,T,*hw]; preserves B/C/T."""
    return adapt_robot_latent(latent, hw)


def synth_robot_tensors(template_item: dict, f_target: int) -> dict:
    """Finite zero robot/action tensors at the target latent frame count, with the
    action mask carrying exactly the active channels of the template item."""
    import torch

    mask = torch.zeros(MODEL_ACTION_DIM, f_target, ACTION_H, 1,
                       dtype=template_item["actions_mask"].dtype)
    mask[list(ACTIVE_CHANNELS)] = 1
    return {"latents": torch.zeros(template_item["latents"].shape[0], f_target,
                                   *ROBOT_LATENT_HW,
                                   dtype=template_item["latents"].dtype),
            "actions": torch.zeros(MODEL_ACTION_DIM, f_target, ACTION_H, 1,
                                   dtype=template_item["actions"].dtype),
            "actions_mask": mask}


# ----- remote-side runtime ----------------------------------------------------------------------------


def _smi_sampler_stop(proc, fileobj):
    """Stop the nvidia-smi -lms process writing into `fileobj`, close it, and return
    the collected CSV text. stdout goes to a file, never an unread PIPE."""
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    fileobj.close()
    return Path(fileobj.name).read_text()


def _smi_summary(csv_text: str) -> dict:
    import numpy as np

    per_gpu = {}
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            _, idx, util, mem_util, used, total, power = parts
            per_gpu.setdefault(int(idx), []).append(
                (float(util), float(mem_util), float(used), float(total), float(power)))
        except ValueError:
            continue
    out = {}
    for idx, rows in per_gpu.items():
        arr = np.asarray(rows)
        out[idx] = {"samples": len(rows),
                    "utilization_gpu_mean": float(arr[:, 0].mean()),
                    "utilization_gpu_p50": float(np.percentile(arr[:, 0], 50)),
                    "utilization_gpu_p95": float(np.percentile(arr[:, 0], 95)),
                    "memory_used_max_mb": float(arr[:, 2].max()),
                    "memory_total_mb": float(arr[:, 3].max()),
                    "power_draw_w_max": float(arr[:, 4].max())}
    return out


def _find_item(dataset, task: str, ep: int) -> tuple[int, dict] | None:
    offset = 0
    for sub in dataset._datasets:
        if Path(str(sub.repo_id)).name == task:
            for i, meta in enumerate(sub.new_metas):
                if int(meta["episode_index"]) == ep:
                    return offset + i, meta
            return None
        offset += len(sub)
    return None


def _collect_selection(data_root: str):
    """Frozen top-64 selection from the collection's own episodes.jsonl rows —
    callable before any distributed init."""
    rows = []
    for task_dir in sorted(Path(data_root, "train").iterdir()):
        meta_file = task_dir / "meta" / "episodes.jsonl"
        if not meta_file.is_file():
            continue
        for line in meta_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({"task": task_dir.name,
                             "episode_index": r["episode_index"],
                             "length": r["length"]})
    assert len(rows) == 1109, len(rows)
    selection = top64_rows(rows)
    assert len(selection) == TOP_N
    return selection, top64_hash(selection)


def _all_task_names(data_root: str) -> list[str]:
    return sorted(p.name for p in Path(data_root, "train").iterdir()
                  if p.is_dir() and (p / "meta" / "episodes.jsonl").is_file())


def _artifact_complete(row: dict) -> bool:
    """True iff every artifact dict reachable under row['artifacts'] (or inline
    exists-bearing dicts) reports exists: true."""
    def _walk(node):
        if isinstance(node, dict):
            if "exists" in node:
                yield bool(node["exists"])
            else:
                for v in node.values():
                    yield from _walk(v)

    arts = row.get("artifacts")
    flags = list(_walk(arts if arts is not None else
                       {k: v for k, v in row.items() if isinstance(v, dict)}))
    return bool(flags) and all(flags)


def loader_tasks_from_inventory(selection: list[tuple[str, int, int]],
                                inventory: dict) -> tuple[list[str], list[str]]:
    """Validate the durable top-64 latent inventory (one row per selected
    (task, episode), complete iff all artifacts exist) and return
    (sorted tasks with >=1 complete selected episode, sorted missing episode ids)."""
    rows = None
    for key in ("episodes", "items", "rows", "selection", "episodes_inventory"):
        if isinstance(inventory, dict) and isinstance(inventory.get(key), list):
            rows = inventory[key]
            break
    if rows is None and isinstance(inventory, list):
        rows = inventory
    if rows is None or len(rows) != TOP_N:
        raise ValueError(
            f"inventory must contain exactly {TOP_N} rows, got "
            f"{None if rows is None else len(rows)}")

    import re
    seen, complete_tasks, missing = set(), set(), []
    for row in rows:
        task = row.get("task") or row.get("task_name")
        ep = row.get("episode_index", row.get("episode"))
        if task is None or ep is None:
            raise ValueError(f"inventory row lacks task/episode: {row!r}")
        m = re.search(r"(\d+)$", str(ep))
        key = (str(task), int(m.group(1)))
        if key in seen:
            raise ValueError(f"duplicate inventory row for {key}")
        seen.add(key)
        if _artifact_complete(row):
            complete_tasks.add(key[0])
        else:
            missing.append(f"{key[0]}/episode_{key[1]:06d}")

    expected = {(t, e) for t, e, _ in selection}
    if seen != expected:
        raise ValueError(
            f"inventory rows do not match selection: missing "
            f"{sorted(expected - seen)}, extra {sorted(seen - expected)}")
    return sorted(complete_tasks), sorted(missing)


def _build_ds_config(args, excluded: list[str], expected_tasks: int,
                     init_worker: int):
    from easydict import EasyDict
    from wan_va.configs.va_robotwin_cfg import load_robotwin_norm_stat

    split_root = Path(args.data_root) / "train"
    ds = EasyDict()
    ds.dataset_path = str(split_root)
    ds.icl_manifest_path = str(split_root / "icl_manifest.json")
    ds.human_latent_path = str(split_root / "human_latents" / "so101")
    ds.robot_latent_path = str(split_root)
    ds.env_type = "none"
    ds.height = 256
    ds.width = 256
    ds.action_dim = MODEL_ACTION_DIM
    ds.action_per_frame = ACTION_H
    ds.obs_cam_keys = ["observation.images.front", "observation.images.wrist"]
    ds.text_encoder_type = "umt_dense"
    ds.empty_emb_path = "/opt/zero-wam/wan_va/assets/empty_text_emb.pt"
    ds.cfg_prob = 0.0
    ds.norm_stat = load_robotwin_norm_stat(str(split_root / "meta" / "action_stats.json"))
    ds.excluded_task_names = list(excluded)
    ds.expected_num_train_tasks = expected_tasks
    ds.enable_dataset_index_cache = True
    ds.rebuild_dataset_index_cache = False
    ds.init_worker = init_worker
    return ds


def _build_config(args, selected_task_names: list[str], all_task_names: list[str],
                  rank: int, local_rank: int, world_size: int):
    from copy import deepcopy
    from wan_va.configs.va_robotwin_train_cfg import va_robotwin_train_cfg

    config = deepcopy(va_robotwin_train_cfg)
    config.model_path = args.weights
    ds = _build_ds_config(
        args, excluded=sorted(set(all_task_names) - set(selected_task_names)),
        expected_tasks=len(selected_task_names), init_worker=1)
    config.dataset_sources = [{"name": "so101_sim", "weight": 1.0, "config": ds}]
    config.learning_rate = 0.0          # LR 0 preserves weights; AdamW still allocates state
    config.batch_size = 1
    config.gradient_accumulation_steps = MICROSTEPS
    config.num_steps = 1
    config.save_interval = 1_000_000_000
    config.save_root = args.scratch
    config.enable_wandb = False
    config.drop_icl = 0.0
    config.droptext_target = 0.0
    config.cfg_prob = 0.0
    config.enable_dataset_index_cache = True
    config.init_worker = 1
    config.load_worker = 0
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    return config


def _run_prepare_indexes(args) -> int:
    """CPU-only dataset-cache preflight: builds the top-64 selection, constructs the
    upstream MultiICLLeRobotLatentDataset for the 8 selected tasks (excluding the other
    53), and asserts the durable index caches are then complete. No NCCL/model init."""
    def stage(name: str, detail: str = "") -> None:
        print(f"[preflight] {name} {detail}", flush=True)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "index_preflight.json"
    t0 = time.perf_counter()
    try:
        stage("selection_start")
        selection, selection_sha = _collect_selection(args.data_root)
        tasks = selected_tasks(selection)
        assert len(tasks) == 8, f"expected 8 selected tasks, got {len(tasks)}"
        all_tasks = _all_task_names(args.data_root)
        assert len(all_tasks) == 61, len(all_tasks)
        stage("selection_ready", f"n={len(selection)} sha={selection_sha[:16]}")

        # loader tasks = selected tasks with at least one complete latent episode
        inventory_path = Path(args.report_dir).parent / "top64_longest_inventory.json"
        inventory = json.loads(inventory_path.read_text())
        loader_tasks, missing_ids = loader_tasks_from_inventory(selection, inventory)
        assert loader_tasks and set(loader_tasks) <= set(tasks)
        stage("inventory_ready",
              f"loader_tasks={len(loader_tasks)} missing={len(missing_ids)}")

        stage("dataset_init_start")
        from wan_va.dataset.icl_lerobot_latent_dataset import (
            MultiICLLeRobotLatentDataset, icl_dataset_indexes_ready)
        ds = _build_ds_config(
            args, excluded=sorted(set(all_tasks) - set(loader_tasks)),
            expected_tasks=len(loader_tasks), init_worker=8)
        dataset = MultiICLLeRobotLatentDataset(config=ds)
        assert len(dataset._datasets) == len(loader_tasks), len(dataset._datasets)
        assert len(dataset) > 0
        ready = icl_dataset_indexes_ready(ds)
        assert ready, "index caches not complete after dataset construction"
        report = {"success": True,
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "selection_sha256": selection_sha,
                  "selected_tasks": tasks,
                  "loader_tasks": loader_tasks,
                  "missing_episode_ids": missing_ids,
                  "inventory_path": str(inventory_path),
                  "samples": len(dataset),
                  "subdatasets": len(dataset._datasets),
                  "index_cache_hits": getattr(dataset, "index_cache_hits", None),
                  "index_cache_misses": getattr(dataset, "index_cache_misses", None),
                  "hf_cache_hits": getattr(dataset, "hf_cache_hits", None),
                  "elapsed_s": time.perf_counter() - t0}
        stage("dataset_init_end", f"{report['elapsed_s']:.1f}s")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report = {"success": False,
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "error": repr(exc), "elapsed_s": time.perf_counter() - t0}
    tmp = out.with_suffix(".json.partial")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out)
    stage("report_written", str(out))
    return 0 if report.get("success") else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/sim/data")
    parser.add_argument("--weights", default="/weights/zero-wam-pretrain")
    parser.add_argument("--report-dir", default="/sim/reports/full_episode_batch64")
    parser.add_argument("--scratch", default="/tmp/so101_bench_ckpt_scratch")
    parser.add_argument("--prepare-indexes", action="store_true")
    args = parser.parse_args()
    if args.prepare_indexes:
        return _run_prepare_indexes(args)

    import socket
    import numpy as np
    import torch
    import torch.distributed as dist
    from wan_va.distributed.util import init_distributed
    from wan_va.train import Trainer

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8, world_size
    report_dir = Path(args.report_dir)
    if rank == 0:
        report_dir.mkdir(parents=True, exist_ok=True)

    def stage(name: str, detail: str = "") -> None:
        print(f"[rank{rank}] {name} {detail}", flush=True)

    errors = []
    selection_sha = None
    preflight = None
    trainer_init_s = None
    item_resolution_s = None
    compute_wall_s = None
    result = {"rank": rank, "hostname": socket.gethostname(), "device": local_rank,
              "items": [], "microstep_s": [], "peak_alloc_bytes": None,
              "peak_reserved_bytes": None, "end_alloc_bytes": None,
              "end_reserved_bytes": None, "grad_norm": None, "grad_finite": None,
              "optimizer_step_skipped": None,
              "trainer_init_s": None, "item_resolution_s": None, "error": None}
    smi_proc = smi_file = None
    t_main = time.perf_counter()
    try:
        # frozen top-64 by source length + CPU preflight check BEFORE any NCCL init
        selection, selection_sha = _collect_selection(args.data_root)
        stage("selection_ready", f"n={len(selection)} sha={selection_sha[:16]}")
        tasks = selected_tasks(selection)
        assert len(tasks) == 8, f"expected 8 selected tasks, got {len(tasks)}"
        all_tasks = _all_task_names(args.data_root)
        assert len(all_tasks) == 61, len(all_tasks)
        if rank == 0:
            (report_dir / "top64_selection.json").write_text(json.dumps(
                {"sha256": selection_sha, "selection": selection}, indent=1))
        inventory = json.loads(
            (report_dir.parent / "top64_longest_inventory.json").read_text())
        loader_tasks, missing_ids = loader_tasks_from_inventory(selection, inventory)
        assert loader_tasks and set(loader_tasks) <= set(tasks)
        preflight_path = report_dir / "index_preflight.json"
        preflight = json.loads(preflight_path.read_text())
        if not (preflight.get("success")
                and preflight.get("selection_sha256") == selection_sha
                and preflight.get("selected_tasks") == tasks
                and preflight.get("loader_tasks") == loader_tasks
                and preflight.get("missing_episode_ids") == missing_ids):
            raise RuntimeError(
                f"stale or failed index preflight at {preflight_path}: "
                "run --prepare-indexes first")
        stage("preflight_verified",
              f"loader_tasks={len(loader_tasks)} missing={len(missing_ids)}")

        init_distributed(world_size, local_rank, rank)
        stage("trainer_init_start")
        t0 = time.perf_counter()
        config = _build_config(args, loader_tasks, all_tasks,
                               rank, local_rank, world_size)
        trainer = Trainer(config)
        original_set_gradient_sync = trainer.transformer.set_requires_gradient_sync

        def force_sharded_gradient_accumulation(_requires_sync):
            original_set_gradient_sync(True)

        trainer.transformer.set_requires_gradient_sync = (
            force_sharded_gradient_accumulation
        )
        trainer_init_s = time.perf_counter() - t0
        result["trainer_init_s"] = trainer_init_s
        stage("trainer_init_end", f"{trainer_init_s:.1f}s")
        dataset = trainer.train_loader.dataset

        stage("selected_items_resolution_start")
        t0 = time.perf_counter()
        template = None          # first resolvable real item, reused for synthetic fill
        per_rank_items = []
        for microstep in range(MICROSTEPS):
            pos = microstep * world_size + rank
            task, ep, length = selection[pos]
            found = _find_item(dataset, task, ep)
            per_rank_items.append((pos, task, ep, length, found))
        item_resolution_s = time.perf_counter() - t0
        result["item_resolution_s"] = item_resolution_s
        stage("selected_items_resolution_end", f"{item_resolution_s:.1f}s")

        torch.cuda.reset_peak_memory_stats()
        stage("compute_start")
        if rank == 0:
            smi_file = open(report_dir / "nvidia_smi_samples.csv.partial", "w")
            smi_proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={SMI_QUERY}",
                 "--format=csv,noheader,nounits", "-lms", "200"],
                stdout=smi_file, stderr=subprocess.DEVNULL)

        t_compute = time.perf_counter()
        for microstep, (pos, task, ep, length, found) in enumerate(per_rank_items):
            item_id = f"{task}/episode_{ep:03d}"
            synthetic = False
            if found is not None:
                item = dataset[found[0]]
            else:
                if template is None:
                    # locate the first resolvable selected item as the template
                    for _, tt, ee, _, ff in per_rank_items:
                        if ff is not None:
                            template = dataset[ff[0]]
                            break
                if template is None:
                    raise RuntimeError("no resolvable selected item for template")
                item = dict(template)
                item.update(synth_robot_tensors(item, robot_latent_frames(length)))
                synthetic = True
                item_id += "+synthetic"
            if template is None:
                template = item
            batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
                     for k, v in item.items()}
            batch["latents"] = adapt_robot_latent(batch["latents"])
            batch["icl_latents"] = adapt_icl_latent(batch["icl_latents"])
            stage(f"microstep{microstep}_start", f"item={item_id} len={length}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            losses = trainer._train_step(batch, microstep)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            result["microstep_s"].append(dt)
            stage(f"microstep{microstep}_end", f"{dt:.1f}s")
            result["items"].append({"pos": pos, "id": item_id,
                                    "source_length": length,
                                    "latent_frames": int(batch["latents"].shape[2]),
                                    "icl_frames": int(batch["icl_latents"].shape[2]),
                                    "synthetic": synthetic})
            if microstep == MICROSTEPS - 1:
                gn = losses.get("total_norm")
                if gn is not None:
                    result["grad_norm"] = float(gn)
                    result["grad_finite"] = bool(torch.isfinite(gn).item())
                result["optimizer_step_skipped"] = bool(
                    losses.get("optimizer_step_skipped"))
        compute_wall_s = time.perf_counter() - t_compute
        stage("compute_end", f"{compute_wall_s:.1f}s")
        result["peak_alloc_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["end_alloc_bytes"] = torch.cuda.memory_allocated()
        result["end_reserved_bytes"] = torch.cuda.memory_reserved()
    except Exception as exc:
        result["error"] = repr(exc)
        errors.append(result["error"])
        import traceback
        traceback.print_exc()
    finally:
        if smi_proc is not None:
            smi_csv = _smi_sampler_stop(smi_proc, smi_file)
            os.replace(report_dir / "nvidia_smi_samples.csv.partial",
                       report_dir / "nvidia_smi_samples.csv")
            result["smi"] = _smi_summary(smi_csv)
        elif smi_file is not None:
            smi_file.close()
    result["wall_s"] = time.perf_counter() - t_main
    result["compute_wall_s"] = compute_wall_s

    gathered = [None] * world_size
    try:
        dist.all_gather_object(gathered, result)
    except Exception as exc:                       # best effort; still write partial report
        gathered = [result]
        errors.append(f"gather failed: {exc!r}")

    if rank == 0:
        stage("gather_report_end")
        total_latent_elems = sum(
            i["latent_frames"] * 48 * 14 * 36 + i["icl_frames"] * 48 * 20 * 28
            for r in gathered for i in r["items"])
        robot_elems = sum(i["latent_frames"] * 48 * 14 * 36
                          for r in gathered for i in r["items"])
        wall_max = max((r.get("compute_wall_s") or 0 for r in gathered), default=0)
        report = {
            "success": not errors and all(r["error"] is None for r in gathered),
            "created_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "world_size": world_size, "microsteps": MICROSTEPS,
            "profiler_enabled": False,
            "gradient_sync_every_microstep": True,
            "selection_sha256": selection_sha,
            "index_preflight": preflight,
            "wall_s": max((r["wall_s"] or 0 for r in gathered), default=0),
            "compute_wall_s": wall_max,
            "trainer_init_s_per_rank": [r["trainer_init_s"] for r in gathered],
            "item_resolution_s_per_rank": [r["item_resolution_s"] for r in gathered],
            "samples_per_s": (64 / wall_max) if wall_max else None,
            "robot_latent_elements_per_s": (robot_elems / wall_max) if wall_max else None,
            "total_latent_elements_per_s":
                (total_latent_elems / wall_max) if wall_max else None,
            "smi": result.get("smi"),
            "topology": {r["rank"]: {"hostname": r["hostname"],
                                     "device": r["device"]} for r in gathered},
            "peak_alloc_bytes_per_rank": [r["peak_alloc_bytes"] for r in gathered],
            "peak_reserved_bytes_per_rank": [r["peak_reserved_bytes"] for r in gathered],
            "end_alloc_bytes_per_rank": [r["end_alloc_bytes"] for r in gathered],
            "grad_norm": [r["grad_norm"] for r in gathered],
            "optimizer_step_skipped": [r["optimizer_step_skipped"] for r in gathered],
            "microstep_s_per_rank": [r["microstep_s"] for r in gathered],
            "items_per_rank": [r["items"] for r in gathered],
            "errors": errors + [r["error"] for r in gathered if r["error"]],
            "caveats": [
                "tensor shapes adapted in memory to final geometry (robot 14x36, "
                "ICL 20x28); provisional latents 256x256/320x480 not re-encoded",
                "LR=0 AdamW step: optimizer state allocated and step executed, "
                "weights unchanged",
                "FSDP reduce-scatter forced after every microstep so accumulated "
                "gradients remain sharded; this adds communication but preserves "
                "the effective-batch objective",
                "unprofiled pass: memory/wall/utilization only; NCCL and operator "
                "attribution deferred to a separate lightweight profiler pass",
                f"{len(missing_ids) if missing_ids else 'five'} missing top-64 "
                "episodes synthesized as zero tensors at exact target latent frame "
                "count; loader restricted to tasks with complete latents; not a "
                "loss-correctness test",
            ],
        }
        out = report_dir / "benchmark_report.json"
        tmp = out.with_suffix(".json.partial")
        tmp.write_text(json.dumps(report, indent=1))
        os.replace(tmp, out)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
