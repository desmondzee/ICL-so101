import json
from pathlib import Path

import pytest
import numpy as np

from zero_wam.prompt_sweep import (
    CheckpointSpec,
    ConditionSpec,
    NULL_PROMPT,
    atomic_json,
    checkpoint_identity,
    condition_manifest,
    prompt_conditions,
    render_report,
)
from zero_wam.so101_closed_loop import motion_diagnostics, smooth_action


def test_prompt_conditions_cover_required_causal_controls():
    conditions = {item.name: item for item in prompt_conditions("soup_lift")}
    assert set(conditions) == {
        "no_conditioning",
        "correct_text",
        "incorrect_text",
        "correct_human_video",
        "mismatched_human_video",
        "shuffled_human_video",
        "matching_text_video",
        "conflicting_text_video",
    }
    assert conditions["correct_text"].text_task == "soup_lift"
    assert conditions["incorrect_text"].text_task == "cheese_lift"
    assert conditions["conflicting_text_video"].video_task == "cheese_lift"
    assert conditions["shuffled_human_video"].shuffle_video is True


def test_modality_disable_uses_empty_embedding_and_no_icl_cache():
    no_conditioning = prompt_conditions("soup_lift")[0].reset_payload()
    assert no_conditioning == {
        "prompt": NULL_PROMPT,
        "use_icl": False,
        "icl_video_path": "",
        "icl_latent_path": "",
        "video_guidance_scale": -1.0,
        "icl_guidance_scale": 1.0,
    }
    text = prompt_conditions("soup_lift")[1].reset_payload()
    assert text["use_icl"] is False
    assert text["video_guidance_scale"] == 5.0
    video = prompt_conditions("soup_lift")[3].reset_payload("/video.mp4")
    assert video["prompt"] == NULL_PROMPT
    assert video["use_icl"] is True
    assert video["icl_guidance_scale"] == 5.0
    with pytest.raises(ValueError):
        prompt_conditions("soup_lift")[3].reset_payload()


def test_manifest_freezes_paired_reproducibility_inputs():
    condition = prompt_conditions("soup_lift")[1]
    kwargs = dict(
        checkpoint={"name": "v2b", "config_sha256": "c"},
        target_task="soup_lift",
        condition=condition,
        environment_seed=7,
        sample_seed=99,
        initial_state_sha256="state",
        initial_observation_sha256="obs",
        video=None,
        max_controls=160,
    )
    first = condition_manifest(**kwargs)
    second = condition_manifest(**kwargs)
    assert first == second
    assert first["environment_seed"] == 7
    assert first["sample_seed"] == 99
    assert first["initial_state_sha256"] == "state"
    assert first["initial_observation_sha256"] == "obs"
    assert first["closed_loop"]["kv_cache_action_history"] == \
        "executed_lerobot_joint_targets"


def _fake_checkpoint(root: Path):
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text('{"_class_name":"WanICLTransformer3DModel"}')
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text("{}")
    (transformer / "diffusion_pytorch_model-00001-of-00001.safetensors").write_bytes(b"x")


def test_checkpoint_identity_is_explicit_and_rejects_robotwin(tmp_path):
    checkpoint = tmp_path / "v1_step500"
    _fake_checkpoint(checkpoint)
    identity = checkpoint_identity(
        CheckpointSpec("v1_step500", "so101-v1", 500, str(checkpoint))
    )
    assert identity["name"] == "v1_step500"
    assert identity["run"] == "so101-v1"
    assert identity["step"] == 500
    assert len(identity["shards"]) == 1
    robotwin = tmp_path / "zero-wam-posttrain-robotwin"
    _fake_checkpoint(robotwin)
    with pytest.raises(ValueError, match="RoboTwin"):
        checkpoint_identity(
            CheckpointSpec("bad", "robotwin", 0, str(robotwin))
        )


def test_result_serialization_and_report_do_not_run_inference(tmp_path):
    result = {
        "checkpoint": "v2b_step500",
        "target_task": "soup_lift",
        "target_object": "alphabet_soup",
        "condition": "correct_text",
        "environment_seed": 0,
        "selected_object": "alphabet_soup",
        "task_success": True,
        "target_grasp_success": True,
        "controls": 42,
    }
    path = tmp_path / "v2b/soup/seed0/correct_text/result.json"
    atomic_json(path, result)
    assert json.loads(path.read_text()) == result
    row = {**result, "_result_path": str(path)}
    report = render_report(tmp_path, [row])
    assert "correct_text" in report
    assert "Action prediction/execution" in report
    assert "Genuine ICL" in report
    assert "not measured" in report


def test_noncanonical_condition_is_rejected():
    bad = ConditionSpec("correct_text", "cheese_lift", None)
    with pytest.raises(ValueError, match="non-canonical"):
        condition_manifest(
            checkpoint={"name": "v2b"},
            target_task="soup_lift",
            condition=bad,
            environment_seed=0,
            sample_seed=0,
            initial_state_sha256="a",
            initial_observation_sha256="b",
            video=None,
            max_controls=8,
        )


def test_motion_diagnostics_distinguishes_smooth_motion_from_shaking():
    smooth = np.column_stack([
        np.linspace(0, 10, 20),
        np.zeros((20, 5)),
    ])
    shaking = smooth.copy()
    shaking[:, 0] = np.tile([-10.0, 10.0], 10)
    smooth_metrics = motion_diagnostics(smooth)
    shaking_metrics = motion_diagnostics(shaking)
    assert smooth_metrics["direction_reversals"] == 0
    assert shaking_metrics["direction_reversal_fraction"] > 0
    assert (
        shaking_metrics["normalized_second_difference_rms"]
        > smooth_metrics["normalized_second_difference_rms"]
    )


def test_action_smoothing_is_bounded_and_optional():
    previous = np.zeros(6, dtype=np.float32)
    target = np.full(6, 10.0, dtype=np.float32)
    assert np.array_equal(smooth_action(target, previous, 1.0), target)
    assert np.array_equal(
        smooth_action(target, previous, 0.25), np.full(6, 2.5, dtype=np.float32)
    )
    with pytest.raises(ValueError):
        smooth_action(target, previous, 0)
