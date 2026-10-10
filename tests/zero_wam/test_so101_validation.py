import json

import pytest

from zero_wam.so101_validation import (
    _aggregate,
    build_validation_manifest,
    deterministic_seed,
    distributed_validation_schedule,
    freeze_validation_manifest,
    manifest_sha256,
)


def _release_roots(tmp_path):
    loader, latent = tmp_path / "loader", tmp_path / "latent"
    (loader / "val").mkdir(parents=True)
    latent.mkdir()
    (latent / "release.json").write_text('{"release":"latents"}')
    samples = []
    for task_index in range(5):
        task = f"task_{task_index}"
        for episode in range(10):
            sample_id = f"{task}__episode_{episode:03d}"
            samples.append({
                "sample_id": sample_id, "sample": sample_id,
                "robot_task_name": task,
            })
    (loader / "val" / "icl_manifest.json").write_text(
        json.dumps({"samples": samples}))
    return loader, latent


def test_manifest_is_complete_stable_and_task_disjoint(tmp_path):
    loader, latent = _release_roots(tmp_path)
    first = build_validation_manifest(loader, latent)
    second = build_validation_manifest(loader, latent)
    assert first == second
    assert len(first["samples"]) == 50
    assert len({row["task"] for row in first["samples"]}) == 5
    assert len({row["random_seed"] for row in first["samples"]}) == 50
    assert first["conditioning"] == {
        "human_video": True,
        "human_detailed_text": True,
        "short_target_text": False,
        "drop_icl": 0.0,
        "target_text_cfg_prob": 1.0,
    }


def test_deterministic_seed_changes_by_sample():
    assert deterministic_seed("a") == deterministic_seed("a")
    assert deterministic_seed("a") != deterministic_seed("b")


def test_freeze_refuses_drift(tmp_path):
    loader, latent = _release_roots(tmp_path)
    output = tmp_path / "frozen.json"
    frozen = freeze_validation_manifest(loader, latent, output)
    assert json.loads(output.read_text()) == frozen
    assert frozen["manifest_sha256"] == manifest_sha256(
        {key: value for key, value in frozen.items() if key != "manifest_sha256"})
    payload = json.loads(output.read_text())
    payload["samples"][0]["random_seed"] += 1
    output.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="refusing to replace"):
        freeze_validation_manifest(loader, latent, output)


def test_aggregate_macro_is_equal_task_weighted():
    rows = []
    for task, values in {"a": [1.0] * 10, "b": [3.0] * 10}.items():
        for index, value in enumerate(values):
            rows.append({
                "sample_id": f"{task}-{index}", "task": task,
                "video_loss": value, "action_loss": value + 1,
                "mcp_weighted_total": value + 2, "total": value + 3,
            })
    result = _aggregate(rows)
    assert result["macro_task_mean"]["video_loss"] == 2.0
    assert result["overall"]["video_loss"]["mean"] == 2.0
    assert result["per_task"]["a"]["total"] == 4.0
    assert result["per_task"]["b"]["total"] == 6.0


def test_distributed_schedule_pads_equal_fsdp_forward_counts():
    schedules = [distributed_validation_schedule(50, 8, rank) for rank in range(8)]
    assert {len(schedule) for schedule in schedules} == {7}
    real = [slot["source_position"] for schedule in schedules
            for slot in schedule if not slot["padding"]]
    assert sorted(real) == list(range(50))
    assert sum(slot["padding"] for schedule in schedules for slot in schedule) == 6
