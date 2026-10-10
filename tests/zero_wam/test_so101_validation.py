import json

import pytest
import torch

from zero_wam.so101_validation import (
    _action_endpoint_metrics,
    _aggregate,
    _gripper_binary_metrics,
    build_validation_manifest,
    deterministic_seed,
    distributed_validation_schedule,
    freeze_validation_manifest,
    manifest_sha256,
    validation_due,
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
                "action_endpoint": {
                    "normalized": {"mae": value / 10, "rmse": value / 5},
                    "gripper_binary": {
                        "closed_tp": 4, "closed_fp": 1,
                        "closed_tn": 4, "closed_fn": 1,
                    },
                    "physical_by_channel": {
                        name: {"mae": value, "rmse": value + 1}
                        for name in (
                            "shoulder_pan_deg", "shoulder_lift_deg",
                            "elbow_flex_deg", "wrist_flex_deg",
                            "wrist_roll_deg", "gripper_0_100",
                        )
                    },
                    "normalized_by_control_offset": [
                        {"offset": offset, "mae": value, "rmse": value + 1}
                        for offset in range(16)
                    ],
                },
            })
    result = _aggregate(rows)
    assert result["macro_task_mean"]["video_loss"] == 2.0
    assert result["overall"]["video_loss"]["mean"] == 2.0
    assert result["per_task"]["a"]["total"] == 4.0
    assert result["per_task"]["b"]["total"] == 6.0
    assert result["overall"]["video_loss"]["q05"] == 1.0
    assert result["overall"]["video_loss"]["q95"] == 3.0
    assert result["action_endpoint"]["normalized_mae"]["median"] == 0.2
    assert result["action_endpoint"]["gripper_binary"][
        "matthews_correlation"] == pytest.approx(0.6)


def test_action_endpoint_metrics_physical_units_offsets_and_gripper():
    error = torch.zeros(1, 30, 2, 8, 1)
    clean = torch.zeros_like(error)
    mask = torch.zeros_like(error, dtype=torch.bool)
    for channel in (0, 1, 2, 3, 4, 28):
        mask[:, channel] = True
        error[:, channel] = 0.1
    clean[:, 28] = -0.05
    stats = {
        "q01": [-100.0] * 28 + [0.0, 0.0],
        "q99": [100.0] * 28 + [100.0, 0.0],
    }
    result = _action_endpoint_metrics(
        endpoint_error=error,
        clean_action=clean,
        action_mask=mask,
        norm_stat=stats,
    )
    assert result["normalized"]["mae"] == pytest.approx(0.1)
    assert result["physical_by_channel"]["shoulder_pan_deg"]["mae"] == pytest.approx(10)
    assert result["physical_by_channel"]["gripper_0_100"]["mae"] == pytest.approx(5)
    assert len(result["normalized_by_control_offset"]) == 16
    assert result["gripper_binary"]["accuracy"] == 0.0
    assert result["gripper_binary"]["closed_fn"] == 16


def test_gripper_mcc_exposes_majority_accuracy():
    result = _gripper_binary_metrics({
        "closed_tp": 0, "closed_fp": 0, "closed_tn": 97, "closed_fn": 3,
    })
    assert result["accuracy"] == pytest.approx(0.97)
    assert result["majority_baseline_accuracy"] == pytest.approx(0.97)
    assert result["closed_recall"] == 0.0
    assert result["macro_f1"] < 0.5
    assert result["matthews_correlation"] == 0.0


def test_distributed_schedule_pads_equal_fsdp_forward_counts():
    schedules = [distributed_validation_schedule(50, 8, rank) for rank in range(8)]
    assert {len(schedule) for schedule in schedules} == {7}
    real = [slot["source_position"] for schedule in schedules
            for slot in schedule if not slot["padding"]]
    assert sorted(real) == list(range(50))
    assert sum(slot["padding"] for schedule in schedules for slot in schedule) == 6


def test_validation_due_interval_and_checkpoint_deduplication():
    validated = {0}
    assert not validation_due(99, 100, validated)
    assert validation_due(100, 100, validated)
    validated.add(100)
    assert not validation_due(100, 100, validated)
    assert validation_due(500, 100, validated)
