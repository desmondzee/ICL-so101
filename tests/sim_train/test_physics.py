"""Physics gates reject transient success and retain real substep evidence."""

from dataclasses import replace
import json

import numpy as np
import pytest

from sim.train.physics import GoalRule, PhysicsPolicy, audit_trajectory
from sim.train.telemetry import Telemetry, TelemetryCollector


def fixture():
    n = 218  # initial sample plus 31 complete seven-substep control frames
    pose = np.zeros((n, 3, 7))
    pose[:, :, 3] = 1
    pose[:, 0, :3] = [0.2, 0, 0.02]
    pose[:, 1, :3] = [0.2, 0, 0]
    pose[:, 2, :3] = [0.3, 0.1, 0.02]
    arrays = dict(
        time=np.arange(n) / 210, contact_time=np.maximum(0, np.arange(n) - 1) / 210,
        frame=np.maximum(0, (np.arange(n) - 1) // 7),
        frame_end=(np.arange(n) % 7 == 0) & (np.arange(n) > 0),
        joint_qpos=np.zeros((n, 2)), joint_qvel=np.zeros((n, 2)),
        joint_target=np.zeros((n, 2)), actuator_force=np.zeros((n, 2)),
        qpos=np.zeros((n, 2)), qvel=np.zeros((n, 2)), qacc=np.zeros((n, 2)),
        body_pose=pose, body_velocity=np.zeros((n, 3, 6)),
        success=np.ones(n, dtype=bool), grasp=np.zeros((n, 3), dtype=bool),
        warnings=np.zeros((n, 8), dtype=int),
        contact_offsets=np.arange(n + 1), contact_geom=np.tile([0, 1], (n, 1)),
        contact_distance=np.full(n, -0.0001), contact_force=np.tile([1., 0, 0, 0, 0, 0], (n, 1)),
        contact_normal=np.tile([0., 0, -1], (n, 1)), contact_position=np.tile([0.2, 0, 0], (n, 1)),
    )
    manifest = dict(schema_version=1, config_hash="a" * 64, visual_config_hash="b" * 64,
                    body_names=["target", "support", "distractor"],
                    geom_names=["target_geom", "support_geom", "finger", "fixture"],
                    geom_bodies=["target", "support", "gripper", "fixture"],
                    joint_names=["j0", "j1"], joint_ranges=[[-1, 1], [-1, 1]],
                    timestep=1 / 210, substeps=7)
    policy = PhysicsPolicy(task_objects=("target",), distractors=("distractor",),
                           support_bodies=(("target", ("support",)),),
                           allowed_contacts=(("target", "support"), ("gripper", "target")),
                           goals=(GoalRule("target", reference="support", offset=(0, 0, 0.02),
                                           position_tolerance=(0.01, 0.01, 0.005), upright_cos=0.98),))
    return Telemetry(arrays, manifest), policy


def test_settled_supported_placement_with_manipulation_contacts_is_accepted():
    telemetry, policy = fixture()
    telemetry.arrays["contact_geom"][1] = [2, 0]
    telemetry.arrays["success"][:8] = False
    telemetry.arrays["grasp"][1, 0] = True
    report = audit_trajectory(telemetry, policy)
    assert report.accepted
    assert report.config_hash == "a" * 64
    assert report.maxima["settled_success_frames"] == 30
    assert all(report.checks.values())
    assert json.loads(report.to_json())["accepted"] is True


@pytest.mark.parametrize("field,index,value,check", [
    ("joint_qpos", (17, 0), np.nan, "finite"),
    ("contact_force", (17, 0), np.inf, "finite"),
    ("joint_qpos", (17, 0), 1.1, "joint_ranges"),
    ("joint_target", (17, 0), -1.1, "joint_ranges"),
    ("joint_qvel", (17, 0), 20, "joint_speed"),
    ("body_velocity", (17, 0, 0), 10, "linear_speed"),
    ("body_velocity", (17, 0, 3), 100, "angular_speed"),
    ("contact_distance", 17, -0.02, "penetration"),
    ("warnings", (17, 0), 1, "simulator_warnings"),
])
def test_rejects_bad_substep_with_exact_timestamp(field, index, value, check):
    telemetry, policy = fixture()
    telemetry.arrays[field][index] = value
    report = audit_trajectory(telemetry, policy)
    assert not report.accepted
    assert not report.checks[check]
    assert any(v["check"] == check and v["timestamp"] == 17 / 210 for v in report.violations)
    json.loads(report.to_json())  # reports stay strict JSON even for nonfinite input


def test_one_frame_success_is_insufficient_and_30_substeps_are_not_30_frames():
    telemetry, policy = fixture()
    telemetry.arrays["success"][:-30] = False
    report = audit_trajectory(telemetry, policy)
    assert not report.accepted
    assert not report.checks["settled_success"]


@pytest.mark.parametrize("change,check", [
    ("unsupported", "support"), ("unreleased", "release"),
    ("distractor", "distractor_displacement"), ("rotation", "distractor_rotation"),
    ("collision", "allowed_contacts"), ("goal", "goal"),
    ("acceleration", "linear_acceleration"), ("joint_acceleration", "joint_acceleration"),
    ("angular_acceleration", "angular_acceleration"),
])
def test_rejects_physical_and_goal_failures(change, check):
    telemetry, policy = fixture()
    if change == "unsupported":
        telemetry.arrays["contact_force"][-1] = 0
    elif change == "unreleased":
        telemetry.arrays["grasp"][-1, 0] = True
    elif change == "distractor":
        telemetry.arrays["body_pose"][17, 2, 0] += 0.02
    elif change == "rotation":
        telemetry.arrays["body_pose"][17, 2, 3:] = [0.98, 0, 0, 0.199]
    elif change == "collision":
        telemetry.arrays["contact_geom"][17] = [2, 3]
    elif change == "goal":
        telemetry.arrays["body_pose"][-1, 0, 0] += 0.1
    elif change == "acceleration":
        telemetry.arrays["body_velocity"][17, 0, 0] = 0.5
        policy = replace(policy, max_linear_acceleration=10)
    elif change == "angular_acceleration":
        telemetry.arrays["body_velocity"][17, 0, 3] = 1
        policy = replace(policy, max_angular_acceleration=10)
    else:
        telemetry.arrays["joint_qvel"][17, 0] = 0.5
        policy = replace(policy, max_joint_acceleration=10)
    report = audit_trajectory(telemetry, policy)
    assert not report.accepted
    assert not report.checks[check]


@pytest.mark.parametrize("change", ["empty", "time", "offset", "geom", "shape", "quaternion", "flag", "hash", "missing"])
def test_corrupt_telemetry_fails_closed(change):
    telemetry, policy = fixture()
    if change == "empty": telemetry.arrays["time"] = np.array([])
    elif change == "time": telemetry.arrays["time"][17] = telemetry.arrays["time"][16]
    elif change == "offset": telemetry.arrays["contact_offsets"][17] = -1
    elif change == "geom": telemetry.arrays["contact_geom"][17, 0] = 999
    elif change == "shape": telemetry.arrays["body_velocity"] = np.zeros((1, 3, 6))
    elif change == "quaternion": telemetry.arrays["body_pose"][17, 0, 3:] = 0
    elif change == "flag": telemetry.arrays["success"] = np.full(218, 2.)
    elif change == "hash": del telemetry.manifest["visual_config_hash"]
    else: del telemetry.arrays["qacc"]
    report = audit_trajectory(telemetry, policy)
    assert not report.accepted
    assert not report.checks["telemetry_integrity"]


def test_roundtrip_is_pickle_free_and_audit_is_deterministic(tmp_path):
    telemetry, policy = fixture()
    path = tmp_path / "telemetry.npz"
    telemetry.save(path)
    with pytest.raises(FileExistsError):
        telemetry.save(path)
    restored = Telemetry.load(path)
    assert restored.manifest == telemetry.manifest
    assert audit_trajectory(restored, policy).to_json() == audit_trajectory(telemetry, policy).to_json()
    with np.load(path, allow_pickle=False) as archive:
        assert json.loads(str(archive["manifest"]))["visual_config_hash"] == "b" * 64


def test_missing_task_policy_body_rejects():
    telemetry, policy = fixture()
    report = audit_trajectory(telemetry, replace(policy, task_objects=("absent",),
                                               support_bodies=(("absent", ("support",)),), goals=()))
    assert not report.accepted
    assert not report.checks["telemetry_integrity"]


def test_side_contact_does_not_count_as_placement_support():
    telemetry, policy = fixture()
    telemetry.arrays["contact_normal"][-1] = [1, 0, 0]
    report = audit_trajectory(telemetry, policy)
    assert not report.accepted
    assert not report.checks["support"]


def test_corrupt_episode_hash_still_produces_strict_rejection_json():
    telemetry, policy = fixture()
    telemetry.manifest["config_hash"] = float("nan")
    report = audit_trajectory(telemetry, policy)
    assert not report.accepted
    json.loads(report.to_json())


def test_finite_values_that_overflow_speed_computation_fail_closed():
    telemetry, policy = fixture()
    telemetry.arrays["body_velocity"][17, 0, 0] = 1e308
    report = audit_trajectory(telemetry, policy)
    assert not report.accepted
    json.loads(report.to_json())


def test_contact_evidence_requires_unit_normals():
    telemetry, policy = fixture()
    telemetry.arrays["contact_normal"][17] = 0
    report = audit_trajectory(telemetry, policy)
    assert not report.accepted
    assert not report.checks["telemetry_integrity"]


def test_unknown_scalar_array_cannot_bypass_schema_or_crash_audit():
    telemetry, policy = fixture()
    telemetry.arrays["unknown"] = np.array(np.nan)
    report = audit_trajectory(telemetry, policy)
    assert not report.accepted
    assert not report.checks["telemetry_integrity"]


def test_audit_preserves_telemetry_evidence():
    telemetry, policy = fixture()
    original = {key: value.copy() for key, value in telemetry.arrays.items()}
    audit_trajectory(telemetry, policy)
    for key, expected in original.items():
        np.testing.assert_array_equal(telemetry.arrays[key], expected)


def test_observer_captures_all_substeps_without_changing_validation_state():
    from sim.val.tasks.mug_on_plate import MugOnPlateEnv, MUG
    env = MugOnPlateEnv(render_images=False, robot_init_qpos_noise=0)
    try:
        env.reset(seed=0)
        action = env._get_current_qpos().copy()
        expected = env.step(action)
        expected_qpos = env.data.qpos.copy()
        expected_qvel = env.data.qvel.copy()
        env.reset(seed=0)
        collector = TelemetryCollector(env, config_hash="a" * 64, visual_config_hash="b" * 64)
        actual = env.step(action, substep_observer=collector)
        telemetry = collector.finish()
        np.testing.assert_array_equal(env.data.qpos, expected_qpos)
        np.testing.assert_array_equal(env.data.qvel, expected_qvel)
        np.testing.assert_array_equal(actual[0]["state"], expected[0]["state"])
        assert actual[1:] == expected[1:]
        assert len(telemetry.arrays["time"]) == 8
        np.testing.assert_allclose(np.diff(telemetry.arrays["time"]), 1 / 210)
        assert telemetry.arrays["frame_end"].tolist() == [False] * 7 + [True]
        assert telemetry.arrays["contact_offsets"][-1] > 0
        assert np.max(telemetry.arrays["contact_force"][:, 0]) > 0
        index = telemetry.manifest["body_names"].index(MUG)
        np.testing.assert_array_equal(telemetry.arrays["body_pose"][-1, index],
                                      env.data.qpos[env._qadr[MUG]:env._qadr[MUG] + 7])
    finally:
        env.close()
