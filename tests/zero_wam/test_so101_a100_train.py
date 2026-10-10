"""Unit tests for pure helpers in zero_wam.so101_a100_train plus the launcher
shell script's static contract. No GPU, no torchrun, no upstream import.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import zero_wam.so101_a100_train as tr


SCRIPT = REPO / "scripts" / "run_zero_wam_so101_a100.sh"


def _ns(**kw):
    base = dict(
        loader_root="/l", latent_root="/z", model_path="/m",
        run_root="/r", run_id="rid", num_steps=4000, save_interval=500,
        resume_from=None, seed=42, wandb_project="p", wandb_group="g",
        disable_wandb=False, gradient_accumulation_steps=8,
    )
    base.update(kw)
    import argparse

    return argparse.Namespace(**base)


# ---- hashing / identity ------------------------------------------------------


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert tr.sha256_file(f) == hashlib.sha256(b"abc").hexdigest()


def test_release_sha256(tmp_path):
    (tmp_path / "release.json").write_text('{"a": 1}')
    assert tr.release_sha256(tmp_path) == hashlib.sha256(b'{"a": 1}').hexdigest()
    with pytest.raises(FileNotFoundError):
        tr.release_sha256(tmp_path / "nope")


def test_tree_manifest_sha256_stable_and_sensitive(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "transformer").mkdir(parents=True)
        (root / "transformer" / "config.json").write_text("{}")
        (root / "x.bin").write_bytes(b"weights")
    assert tr.tree_manifest_sha256(a) == tr.tree_manifest_sha256(b)
    (b / "x.bin").write_bytes(b"weights2")
    assert tr.tree_manifest_sha256(a) != tr.tree_manifest_sha256(b)
    with pytest.raises(FileNotFoundError):
        tr.tree_manifest_sha256(tmp_path / "missing")


def _identity(tmp_path, loader_release='{"release": 1}', **kw):
    loader = tmp_path / "loader"
    latent = tmp_path / "latent"
    model = tmp_path / "model"
    loader.mkdir(exist_ok=True)
    latent.mkdir(exist_ok=True)
    (loader / "release.json").write_text(loader_release)
    (latent / "release.json").write_text('{"release": 1}')
    model.mkdir(exist_ok=True)
    (model / "f").write_text("w")
    base = dict(
        loader_root=loader, latent_root=latent, model_path=model,
        world_size=8, seed=42, num_steps=4000, save_interval=500,
        gradient_accumulation_steps=8,
    )
    base.update(kw)
    return tr.config_identity_payload(**base)


def test_config_identity_excludes_num_steps(tmp_path):
    a = tr.config_sha256(_identity(tmp_path, num_steps=4000))
    b = tr.config_sha256(_identity(tmp_path, num_steps=8000))
    assert a == b


def test_config_identity_changes_on_inputs(tmp_path):
    a = tr.config_sha256(_identity(tmp_path))
    b = tr.config_sha256(_identity(tmp_path, loader_release='{"release": 2}'))
    assert a != b


def test_config_identity_changes_on_seed(tmp_path):
    a = tr.config_sha256(_identity(tmp_path, seed=42))
    b = tr.config_sha256(_identity(tmp_path, seed=43))
    assert a != b


def test_config_identity_changes_on_accumulation(tmp_path):
    a = tr.config_sha256(
        _identity(tmp_path, gradient_accumulation_steps=8)
    )
    b = tr.config_sha256(
        _identity(tmp_path, gradient_accumulation_steps=2)
    )
    assert a != b


def test_config_identity_records_effective_batch(tmp_path):
    p = _identity(tmp_path, gradient_accumulation_steps=2)
    assert p["hyperparameters"]["gradient_accumulation_steps"] == 2
    assert p["hyperparameters"]["effective_batch"] == 16


# ---- run-directory gating -----------------------------------------------------


def test_fresh_run_dir_rules(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    tr.assert_fresh_run_dir(empty)  # empty dir ok

    infra = tmp_path / "infra"
    infra.mkdir()
    (infra / "run.log").write_text("log")
    (infra / "run.pid").write_text("123")
    tr.assert_fresh_run_dir(infra)  # script-owned files ok

    (infra / "run_manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        tr.assert_fresh_run_dir(infra)


def test_fresh_run_dir_refuses_other_content(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "checkpoints").mkdir()
    with pytest.raises(FileExistsError):
        tr.assert_fresh_run_dir(d)


def test_resume_metadata_validation(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        tr.load_resume_metadata(d, "abc")

    (d / "run_manifest.json").write_text(
        json.dumps({"config_sha256": "abc"})
    )
    (d / "wandb_run_id.txt").write_text("wbrun123\n")
    manifest, wid = tr.load_resume_metadata(d, "abc")
    assert manifest["config_sha256"] == "abc"
    assert wid == "wbrun123"

    with pytest.raises(ValueError):
        tr.load_resume_metadata(d, "different")


# ---- payloads -----------------------------------------------------------------


def test_completed_payload(tmp_path):
    p = tr.build_completed_payload(
        run_id="r1", config_sha="deadbeef", final_step=4000,
        checkpoint_path=tmp_path / "checkpoints" / "checkpoint_step_4000",
        exposure_path=tmp_path / "checkpoints" / "checkpoint_step_4000"
        / "episode_exposure.json",
    )
    assert p["success"] is True and p["final_step"] == 4000
    assert p["config_sha256"] == "deadbeef"
    assert p["completed_utc"].endswith("+00:00")
    assert "checkpoint_path" in p and "episode_exposure_path" in p


def test_failed_payload():
    p = tr.build_failed_payload(
        run_id="r1", config_sha="c", step=17, error=ValueError("boom")
    )
    assert p["success"] is False and p["step"] == 17
    assert "ValueError" in p["error"]


def test_run_manifest_accumulation_values(tmp_path):
    manifest = tr.build_run_manifest(
        args=_ns(gradient_accumulation_steps=2), config_sha="c",
        wandb_run_id="w1", preflight_dir=Path("outputs/zero_wam"),
    )
    assert manifest["gradient_accumulation_steps"] == 2
    assert manifest["effective_batch"] == 16


def test_run_manifest_nonsecret(tmp_path):
    manifest = tr.build_run_manifest(
        args=_ns(), config_sha="c", wandb_run_id="w1",
        preflight_dir=Path("outputs/zero_wam/a100_preflight"),
    )
    text = json.dumps(manifest)
    assert manifest["config_sha256"] == "c"
    assert manifest["wandb_run_id"] == "w1"
    assert manifest["gradient_accumulation_steps"] == 8
    assert manifest["effective_batch"] == 64
    for marker in ("WANDB_API_KEY", "api_key", "token", "secret"):
        assert marker not in text
    assert manifest["preflight_artifacts"]["host_inventory"].endswith(
        "host_inventory.json"
    )


# ---- shell script static contract ---------------------------------------------


def _script():
    return SCRIPT.read_text()


def test_script_exists_and_contract():
    s = _script()
    assert s.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in s
    assert "--standalone --nproc_per_node=8" in s
    assert '--gradient-accumulation-steps "$accum_steps"' in s
    assert 'accum_steps="${3:-8}"' in s
    assert "[NUM_STEPS] [ACCUM_STEPS]" in s
    assert "expandable_segments:True" in s
    assert "TOKENIZERS_PARALLELISM=false" in s
    assert "nohup" in s and "run.pid" in s
    assert "run.log" in s
    # canonical local roots
    assert "zero_wam_loader_v1" in s
    assert "zero_wam_latents_v1" in s
    assert "zero-wam-pretrain" in s
    assert "zero_wam.so101_a100_train" in s
    # .env sourced with set -a
    assert "set -a" in s and "source" in s and ".env" in s
    # subcommands
    for sub in ("start", "status", "stop"):
        assert f"{sub})" in s or f"{sub} " in s
    # stop is graceful only
    assert "kill -TERM" in s
    assert "kill -9" not in s and "SIGKILL" not in s and "-KILL" not in s
    # no secret literals
    for marker in ("WANDB_API_KEY=", "api_key=", "password", "token="):
        assert marker not in s


def test_script_preflight_gate():
    s = _script()
    assert "host_inventory.json" in s and "loader_smoke.json" in s


# ---- LocalMetricsWandbProxy ---------------------------------------------------


class _FakeWandbModule:
    def __init__(self):
        self.log_calls = []
        self.Table = object()
        self.Artifact = object()
        self.finish = "finish-attr"
        self.run = "run-attr"

    def log(self, data, step=None, commit=None, **kwargs):
        self.log_calls.append((data, step, commit, kwargs))
        return "logged"


def test_jsonl_proxy_append_only_multiple_rows(tmp_path):
    mod = _FakeWandbModule()
    proxy = tr.LocalMetricsWandbProxy(mod, tmp_path / "run" / "metrics.jsonl")
    proxy.log({"loss": 1.5, "name": "s"}, step=3)
    proxy.log({"loss": 1.2}, step=4, commit=False, extra="x")
    proxy.log({"m": 1}, step=None)
    lines = proxy.jsonl_path.read_text().splitlines()
    assert len(lines) == 3
    rec0 = json.loads(lines[0])
    assert rec0["schema"] == 1 and rec0["step"] == 3
    assert rec0["commit"] is None and rec0["data"] == {"loss": 1.5, "name": "s"}
    assert json.loads(lines[1])["commit"] is False
    assert json.loads(lines[2])["step"] is None
    for line in lines:
        assert json.loads(line)["ts_utc"].endswith("+00:00")


def test_jsonl_proxy_scalar_filtering_and_non_scalar_keys(tmp_path):
    proxy = tr.LocalMetricsWandbProxy(
        _FakeWandbModule(), tmp_path / "run" / "metrics.jsonl")
    sentinel_table = object()
    proxy.log({"exposure/z_table": sentinel_table,
               "exposure/a_chart": object(),
               "exposure/total_draws": 1024,
               "exposure/mean": 0.92,
               "exposure/tag": "x",
               "exposure/null_v": None,
               "exposure/bad_nan": float("nan")})
    rec = json.loads(proxy.jsonl_path.read_text().splitlines()[0])
    assert rec["data"] == {
        "exposure/total_draws": 1024,
        "exposure/mean": 0.92,
        "exposure/tag": "x",
        "exposure/null_v": None,
    }
    assert rec["non_scalar_keys"] == [
        "exposure/a_chart", "exposure/bad_nan", "exposure/z_table"]


def test_jsonl_proxy_fsync_real_file(tmp_path, monkeypatch):
    fsynced = []
    real_fsync = os.fsync

    def spy(fd):
        fsynced.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    proxy = tr.LocalMetricsWandbProxy(
        _FakeWandbModule(), tmp_path / "run" / "metrics.jsonl")
    proxy.log({"x": 1}, step=0)
    assert len(fsynced) == 1
    # durable without any explicit close: a fresh read sees the line
    assert json.loads(open(proxy.jsonl_path).read().splitlines()[0])[
        "data"] == {"x": 1}
    proxy.log({"y": 2}, step=1)
    assert len(fsynced) == 2
    assert len(proxy.jsonl_path.read_text().splitlines()) == 2


def test_jsonl_proxy_exact_delegation_and_attrs(tmp_path):
    mod = _FakeWandbModule()
    proxy = tr.LocalMetricsWandbProxy(mod, tmp_path / "run" / "metrics.jsonl")
    data = {"loss": 0.1}
    out = proxy.log(data, step=7, commit=False, sync=False)
    assert out == "logged"
    (d, step, commit, kwargs), = mod.log_calls
    assert d is data and step == 7 and commit is False
    assert kwargs == {"sync": False}
    # non-log attributes delegate unchanged
    assert proxy.Table is mod.Table
    assert proxy.Artifact is mod.Artifact
    assert proxy.finish == mod.finish
    # path lives under the run root
    assert proxy.jsonl_path == tmp_path / "run" / "metrics.jsonl"
    assert proxy.jsonl_path.parent == tmp_path / "run"


def test_jsonl_proxy_write_failure_raises(tmp_path):
    proxy = tr.LocalMetricsWandbProxy(
        _FakeWandbModule(), tmp_path / "no_such_dir" / "run")
    # constructor creates the parent; remove it to force a write failure
    import shutil

    shutil.rmtree(proxy.jsonl_path.parent)
    with pytest.raises(OSError):
        proxy.log({"x": 1})


def test_script_resume_interface():
    s = _script()
    # usage + dispatch
    assert "resume RUN_ID CHECKPOINT_DIR [NUM_STEPS] [ACCUM_STEPS]" in s
    assert "resume)" in s and "cmd_resume" in s
    # required run + checkpoint artifacts
    assert "run_manifest.json" in s
    assert "wandb_run_id.txt" in s
    assert "transformer/config.json" in s
    assert "training_state.pt" in s
    # passes --resume-from through to the trainer
    assert '--resume-from "$checkpoint_dir"' in s
    # live-pid + preflight gates shared with start
    assert "live_pid" in s and "preflight_ok" in s
    # completed.json archived to completed.step_<final_step>.json via python
    assert "completed.step_" in s and "final_step" in s
    # failed.json archived with UTC-compact suffix
    assert "failed.resume_" in s
    # canonical env identical to start (shared helper)
    assert "launch_torchrun" in s
    assert 'source "$REPO/.env"' in s
    assert 'PYTHONPATH="$REPO:$REPO/third_party/Zero-WAM"' in s
    assert "expandable_segments:True" in s
    # no secret literals
    for marker in ("WANDB_API_KEY=", "api_key=", "password", "token="):
        assert marker not in s
