"""Pure seams of the full-episode benchmark: selection, frame formula, shape adaptation."""

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import zero_wam.so101_full_episode_benchmark as bench
import zero_wam.modal_so101_sim as ms


def _rows(n=1109, tasks=61):
    rows = []
    for i in range(n):
        task = f"task_{i % tasks}"
        rows.append({"task": task, "episode_index": i // tasks,
                     "length": 100 + (i * 37 % 800)})
    return rows


def test_top64_ordering_and_hash():
    rows = _rows()
    sel = bench.top64_rows(rows)
    assert len(sel) == 64
    lengths = [n for _, _, n in sel]
    assert lengths == sorted(lengths, reverse=True)
    assert bench.top64_hash(sel) == bench.top64_hash(bench.top64_rows(rows))
    assert bench.top64_hash(sel) != bench.top64_hash(sel[::-1])
    # ties break by (task, episode); assert exact result under the helper contract
    tied = [{"task": f"t{k}", "episode_index": e, "length": 100 + (k % 3)}
            for k in range(80) for e in range(2)]
    s = bench.top64_rows(tied)
    expected = sorted(tied, key=lambda r: (-r["length"], r["task"], r["episode_index"]))
    assert s == [(r["task"], r["episode_index"], r["length"]) for r in expected[:64]]


def test_selected_tasks_sorted_unique():
    sel = [("b_task", 0, 500), ("a_task", 1, 400), ("b_task", 2, 300),
           ("a_task", 0, 200)]
    assert bench.selected_tasks(sel) == ["a_task", "b_task"]
    assert bench.selected_tasks(sel) == bench.selected_tasks(list(reversed(sel)))


def test_no_profiler_symbols_or_claims():
    import inspect
    src = inspect.getsource(bench)
    for sym in ("torch.profiler", "ProfilerActivity", "export_chrome_trace",
                "key_averages", "nccl_per_rank", "cuda_totals"):
        assert sym not in src, sym
    assert '"profiler_enabled": False' in src
    mod = __import__("zero_wam.modal_so101_benchmark",
                     fromlist=["run_benchmark"])
    run_src = inspect.getsource(mod.run_benchmark.get_raw_f())
    assert "capture_output" not in run_src            # GPU run must stream, not buffer
    assert "subprocess.Popen" in run_src and "STDOUT" in run_src


def test_wrapper_streaming_is_threaded_with_wait_timeout():
    from pathlib import Path as P
    import inspect
    wsrc = P(inspect.getfile(bench)).parent.joinpath(
        "modal_so101_benchmark.py").read_text()
    assert "threading.Thread" in wsrc and "daemon=True" in wsrc
    assert "proc.wait(timeout=90 * 60)" in wsrc
    assert "reader.join" in wsrc
    assert "deadline" not in wsrc                       # no manual deadline bookkeeping
    # stdout tee happens inside the reader thread, not a blocking main-thread loop
    func_src = inspect.getsource(
        __import__("zero_wam.modal_so101_benchmark",
                   fromlist=["run_benchmark"]).run_benchmark.get_raw_f())
    assert "for line in proc.stdout" in func_src        # inside _tee only
    assert func_src.count("for line in proc.stdout") == 1


def test_prepare_indexes_has_no_distributed_or_trainer():
    import inspect
    src = inspect.getsource(bench._run_prepare_indexes)
    for sym in ("init_distributed", "Trainer(", "init_process_group", "H100"):
        assert sym not in src, sym
    for sym in ("MultiICLLeRobotLatentDataset", "icl_dataset_indexes_ready",
                "index_preflight.json", "excluded"):
        assert sym in src, sym


def test_preflight_gate_precedes_distributed_init():
    import inspect
    src = inspect.getsource(bench.main)
    assert "args.prepare_indexes" in src
    i_pre = src.find("index_preflight.json")
    i_dist = src.find("init_distributed(")
    assert 0 < i_pre < i_dist        # preflight verified before NCCL init on every rank


def test_collect_selection_on_fixture(tmp_path):
    tasks = 61
    rows = 1109
    root = tmp_path / "data" / "train"
    per, extra = divmod(rows, tasks)
    for k in range(tasks):
        meta = root / f"task_{k:02d}" / "meta"
        meta.mkdir(parents=True)
        n = per + (1 if k < extra else 0)
        meta.joinpath("episodes.jsonl").write_text(
            "".join(json.dumps({"episode_index": e, "length": 100 + k * 1000 + e})
                    + "\n" for e in range(n)))
    selection, sha = bench._collect_selection(str(tmp_path / "data"))
    assert len(selection) == 64 and sha == bench.top64_hash(selection)
    assert bench._all_task_names(str(tmp_path / "data")) == \
        [f"task_{k:02d}" for k in range(tasks)]
    # longest rows all come from the highest-length task
    assert all(t == "task_60" for t, _, _ in selection[:18])


def test_loader_tasks_from_inventory():
    import pytest
    sel = [("task_a", i, 1000 - i) for i in range(40)]
    sel += [("task_b", i, 500 - i) for i in range(23)]
    sel += [("task_c", 0, 100)]
    assert len(sel) == 64
    # task_a: 2 incomplete; task_b: all complete; task_c: only row incomplete
    incomplete = {("task_a", 1), ("task_a", 2), ("task_c", 0)}
    rows = [{"task": t, "episode_index": e,
             "artifacts": {"robot_latent": {"exists": (t, e) not in incomplete},
                           "icl_latent": {"exists": True}}}
            for t, e, _ in sel]
    lt, miss = bench.loader_tasks_from_inventory(sel, {"episodes": rows})
    assert lt == ["task_a", "task_b"]          # task_c drops out entirely
    assert miss == ["task_a/episode_000001", "task_a/episode_000002",
                    "task_c/episode_000000"]

    with pytest.raises(ValueError):            # duplicate row
        bench.loader_tasks_from_inventory(sel, {"episodes": rows[:-1] + [rows[0]]})
    with pytest.raises(ValueError):            # row not in selection
        bad = rows[:-1] + [{"task": "task_z", "episode_index": 999,
                            "artifacts": {"x": {"exists": True}}}]
        bench.loader_tasks_from_inventory(sel, {"episodes": bad})
    with pytest.raises(ValueError):            # wrong row count
        bench.loader_tasks_from_inventory(sel, {"episodes": rows[:63]})


def test_wrapper_preflight_ordering():
    import inspect
    wrap = Path(inspect.getfile(bench)).parent / "modal_so101_benchmark.py"
    wsrc = wrap.read_text()
    i_fn = wsrc.find("def prepare_indexes")
    assert i_fn > 0
    dec = wsrc[wsrc.rfind("@app.function", 0, i_fn):i_fn]
    assert "cpu=32" in dec and "gpu=" not in dec   # CPU-only function
    i_pre = wsrc.find("prepare_indexes.remote()")
    i_gpu = wsrc.find("run_benchmark.remote()")
    assert 0 < i_pre < i_gpu        # entrypoint calls preflight first


def test_smi_output_goes_to_file_not_pipe():
    import inspect
    src = inspect.getsource(bench.main)
    assert "stdout=subprocess.PIPE" not in src
    assert "stdout=smi_file" in src
    assert "stderr=subprocess.DEVNULL" in src
    stop_src = inspect.getsource(bench._smi_sampler_stop)
    assert "communicate" not in stop_src
    assert "fileobj.close()" in stop_src


def test_latent_frame_formula_matches_encoder():
    for length in (1, 2, 8, 9, 10, 41, 530, 1243):
        ids = ms.robot_frame_ids(length)
        assert bench.robot_latent_frames(length) == ms.latent_frames(len(ids))
    assert bench.robot_latent_frames(9) == 2   # ids [0,2,4,6,8] -> 2 latent frames
    assert bench.robot_latent_frames(10) == 2


def test_shape_adaptation():
    robot = torch.randn(48, 5, 16, 32, dtype=torch.bfloat16)
    out = bench.adapt_robot_latent(robot)
    assert tuple(out.shape) == (48, 5, 14, 36) and out.dtype == torch.bfloat16
    icl = torch.randn(48, 3, 20, 30)
    out2 = bench.adapt_icl_latent(icl)
    assert tuple(out2.shape) == (48, 3, 20, 28)
    # runtime calls are batched: [B,C,T,H,W]
    b_robot = robot.unsqueeze(0)
    out_b = bench.adapt_robot_latent(b_robot)
    assert tuple(out_b.shape) == (1, 48, 5, 14, 36) and out_b.dtype == torch.bfloat16
    assert torch.equal(out_b[0], out)
    b_icl = icl.unsqueeze(0)
    out_b2 = bench.adapt_icl_latent(b_icl)
    assert tuple(out_b2.shape) == (1, 48, 3, 20, 28) and out_b2.dtype == icl.dtype
    import pytest
    with pytest.raises(ValueError):
        bench.adapt_robot_latent(torch.zeros(48, 16, 32))


def test_synth_robot_tensors():
    tmpl = {"latents": torch.zeros(48, 3, 16, 32, dtype=torch.bfloat16),
            "actions": torch.zeros(30, 3, 8, 1),
            "actions_mask": torch.zeros(30, 3, 8, 1, dtype=torch.bool)}
    for c in bench.ACTIVE_CHANNELS:
        tmpl["actions_mask"][c] = True
    out = bench.synth_robot_tensors(tmpl, f_target=7)
    assert tuple(out["latents"].shape) == (48, 7, 14, 36)
    assert tuple(out["actions"].shape) == (30, 7, 8, 1)
    mask = out["actions_mask"]
    assert tuple(mask.shape) == (30, 7, 8, 1)
    active = sorted(mask[..., 0].any(dim=(1, 2)).nonzero().flatten().tolist())
    assert active == [0, 1, 2, 3, 4, 28]
    assert torch.isfinite(out["latents"].float()).all() and \
        torch.isfinite(out["actions"]).all()


def test_runtime_order_synth_unsqueeze_adapt():
    # mirrors the runtime path: template -> synth -> collate batch dim -> adapters
    tmpl = {"latents": torch.zeros(48, 3, 16, 32, dtype=torch.bfloat16),
            "icl_latents": torch.zeros(48, 2, 20, 30, dtype=torch.bfloat16),
            "actions": torch.zeros(30, 3, 8, 1),
            "actions_mask": torch.zeros(30, 3, 8, 1, dtype=torch.bool)}
    for c in bench.ACTIVE_CHANNELS:
        tmpl["actions_mask"][c] = True
    item = dict(tmpl)
    item.update(bench.synth_robot_tensors(item, f_target=7))
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in item.items()}
    batch["latents"] = bench.adapt_robot_latent(batch["latents"])
    batch["icl_latents"] = bench.adapt_icl_latent(batch["icl_latents"])
    assert tuple(batch["latents"].shape) == (1, 48, 7, 14, 36)
    assert tuple(batch["icl_latents"].shape) == (1, 48, 2, 20, 28)
    assert tuple(batch["actions"].shape) == (1, 30, 7, 8, 1)
    assert tuple(batch["actions_mask"].shape) == (1, 30, 7, 8, 1)
