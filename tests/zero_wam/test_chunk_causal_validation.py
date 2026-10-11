import numpy as np
import pytest

from zero_wam.chunk_causal_validation import (
    ACTION_NAMES,
    CHUNK_CONTROLS,
    assert_payload_has_no_targets,
    inference_payload,
    score_chunk,
)


def _sample():
    return {
        "sample_id": "task__episode_000",
        "seed": 7,
        "human_latent": "/human.pth",
        "human_detailed_text": "human",
        "target_text": "target",
    }


def test_inference_payload_contains_prefix_but_never_targets():
    observations = [{"frame": index} for index in range(5)]
    payload = inference_payload(
        _sample(), observations, np.zeros((8, 6), dtype=np.float32))
    assert payload["observations"] == observations
    assert payload["executed_prefix_actions"].shape == (8, 6)
    assert "target_actions" not in payload
    assert "future_observations" not in payload
    assert_payload_has_no_targets(payload)


def test_inference_payload_rejects_wrong_prefix_and_leakage():
    with pytest.raises(ValueError):
        inference_payload(_sample(), [{}] * 5, np.zeros((16, 6)))
    with pytest.raises(ValueError):
        inference_payload(_sample(), [{}] * 4, np.zeros((8, 6)))
    with pytest.raises(ValueError):
        assert_payload_has_no_targets({"target_actions": np.zeros((16, 6))})


def test_score_chunk_physical_metrics_and_persistence():
    target = np.zeros((CHUNK_CONTROLS, len(ACTION_NAMES)))
    predicted = np.ones_like(target)
    persistence = np.full_like(target, 2.0)
    result = score_chunk(predicted, target, persistence)
    assert result["model"]["overall_physical_mae"] == 1.0
    assert result["model"]["overall_physical_rmse"] == 1.0
    assert result["persistence"]["overall_physical_mae"] == 2.0
    assert len(result["model"]["physical_by_control_offset"]) == 16
    assert set(result["model"]["physical_by_channel"]) == set(ACTION_NAMES)


def test_score_chunk_rejects_non_chunk_shapes():
    with pytest.raises(ValueError):
        score_chunk(np.zeros((8, 6)), np.zeros((8, 6)), np.zeros((8, 6)))
