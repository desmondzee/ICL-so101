"""Variation must be replayable and reject actual framing/occlusion failures."""

from dataclasses import FrozenInstanceError, replace
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from sim.train.model import EpisodeKey, EpisodeManifest
from sim.train.store import EpisodeStore
from sim.train.variation import (
    DEFAULT_POLICY, GoalRegion, VisualConfig, episode_config_hash,
    sample_visual_config, screen_visual_config, save_screening_report,
)
from sim.val.env import ValEnv, FRONT_CAM, WRIST_CAM_POS, WRIST_CAM_QUAT, WRIST_CAM_FOVY
from sim.val.scene import Block, SceneSpec, build_scene_xml, camera_xml


class VisibilityEnv(ValEnv):
    n_distractors = (0, 0)

    def make_scene(self):
        return SceneSpec(arena="kitchen", objects=[Block("target"), Block("goal")],
                         distractors=[Block("occluder", half=(0.045, 0.045, 0.045))])

    def layout(self):
        self.set_object_pose("target", (0.20, -0.06))
        self.set_object_pose("goal", (0.20, 0.07))
        self.set_object_pose("occluder", (20, 0))

    def success(self):
        return False


@pytest.fixture
def env():
    instance = VisibilityEnv(render_images=False, robot_init_qpos_noise=0,
                             visual_config=sample_visual_config("put_can_in_basket", 11))
    instance.reset(seed=19)
    yield instance
    instance.close()


def goal_region(env):
    p = env.object_pos("goal")
    return GoalRegion("goal", tuple(tuple(p + [x, y, 0.014])
                                    for x in (-0.008, 0, 0.008)
                                    for y in (-0.008, 0, 0.008)))


def test_variation_is_deterministic_and_changes_across_seeds():
    a = sample_visual_config("put_can_in_basket", 11, DEFAULT_POLICY)
    assert a == sample_visual_config("put_can_in_basket", 11, DEFAULT_POLICY)
    assert a.config_hash != sample_visual_config("put_can_in_basket", 12, DEFAULT_POLICY).config_hash
    assert a != sample_visual_config("other_task", 11, DEFAULT_POLICY)


def test_hundred_seeds_cover_arenas_and_vary_all_sampled_dimensions():
    configs = [sample_visual_config("put_can_in_basket", i) for i in range(100)]
    assert len({c.config_hash for c in configs}) == 100
    assert {c.arena for c in configs} == {"living_room", "kitchen"}
    assert len({c.front_camera for c in configs}) == 100
    assert len({c.lights for c in configs}) == 100
    assert len({c.background for c in configs}) == 100
    for c in configs:
        for values, bounds in ((c.front_camera.pos, DEFAULT_POLICY.front_pos),
                               (c.front_camera.lookat, DEFAULT_POLICY.front_lookat)):
            assert all(low <= value <= high for value, (low, high) in zip(values, bounds))
        assert 44 <= c.front_camera.fovy <= 54
        for light, bounds, intensity in zip(c.lights, (DEFAULT_POLICY.key_pos, DEFAULT_POLICY.fill_pos),
                                            (DEFAULT_POLICY.key_intensity, DEFAULT_POLICY.fill_intensity)):
            assert all(low <= value <= high for value, (low, high) in zip(light.pos, bounds))
            assert intensity[0] <= light.intensity <= intensity[1]
            assert 0.90 <= light.color[2] <= light.color[1] <= light.color[0] == 1.0


def test_config_is_immutable_json_roundtrips_and_detects_tampering():
    config = sample_visual_config("put_can_in_basket", 11)
    assert VisualConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(FrozenInstanceError):
        config.front_camera.fovy = 99
    changed = config.to_dict()
    changed["front_camera"]["fovy"] += 1
    with pytest.raises(ValueError, match="hash"):
        VisualConfig.from_dict(changed)
    # Seeds are provenance, not an escape hatch from visual duplicate detection.
    assert replace(config, variation_seed=999).config_hash == config.config_hash


def test_resample_index_is_replayable_and_changes_configuration():
    a = sample_visual_config("put_can_in_basket", 11, resample_index=1)
    assert a == sample_visual_config("put_can_in_basket", 11, resample_index=1)
    assert a.config_hash != sample_visual_config("put_can_in_basket", 11).config_hash
    with pytest.raises(ValueError):
        sample_visual_config("put_can_in_basket", -1)
    with pytest.raises(ValueError):
        sample_visual_config("put_can_in_basket", 0, resample_index=-1)
    with pytest.raises(ValueError):
        replace(DEFAULT_POLICY, arenas=("unqualified",))


def test_scene_injection_is_stable_and_preserves_default_xml():
    spec = SceneSpec(objects=[Block("target")])
    camera = camera_xml("front", **FRONT_CAM)
    default = build_scene_xml(spec, [camera])
    assert default == build_scene_xml(spec, [camera], visual_config=None)
    for seed in (11, 12):
        config = sample_visual_config("put_can_in_basket", seed)
        xml = build_scene_xml(spec, visual_config=config)
        assert xml == build_scene_xml(spec, visual_config=config)
        root = ET.fromstring(xml)
        front = root.find("worldbody/camera[@name='front']")
        np.testing.assert_allclose(np.fromstring(front.get("pos"), sep=" "), config.front_camera.pos, rtol=1e-5)
        assert float(front.get("fovy")) == config.front_camera.fovy
        for light in config.lights:
            el = root.find(f"worldbody/light[@name='{light.name}']")
            np.testing.assert_allclose(np.fromstring(el.get("diffuse"), sep=" "),
                                       np.array(light.color) * light.intensity, rtol=1e-5)
        assert spec.arena == "living_room"  # Scene injection does not mutate task declarations.


def test_optional_env_configuration_preserves_wrist_and_task_reset(env):
    config = sample_visual_config("put_can_in_basket", 11)
    varied = VisibilityEnv(render_images=False, robot_init_qpos_noise=0, visual_config=config)
    try:
        varied.reset(seed=19)
        np.testing.assert_allclose(varied.model.cam_pos[varied._front_cam_id], config.front_camera.pos, rtol=1e-5)
        assert varied.scene.arena == config.arena
        for name in ("target", "goal"):
            np.testing.assert_allclose(varied.object_pos(name), env.object_pos(name), atol=1e-5)
        for instance in (env, varied):
            np.testing.assert_allclose(instance.model.cam_pos[instance._wrist_cam_id], WRIST_CAM_POS)
            np.testing.assert_allclose(instance.model.cam_quat[instance._wrist_cam_id], WRIST_CAM_QUAT)
            assert instance.model.cam_fovy[instance._wrist_cam_id] == WRIST_CAM_FOVY
        varied.reset(seed=19)
        assert varied.visual_config == config
    finally:
        varied.close()


def test_episode_identity_includes_task_owned_layout():
    config = sample_visual_config("put_can_in_basket", 11)
    poses = {"can": [0.2, 0.1, 0.02, 1, 0, 0, 0]}
    a = episode_config_hash(config, poses)
    assert a == episode_config_hash(config, json.loads(json.dumps(poses)))
    assert a != episode_config_hash(config, {"can": [0.21, 0.1, 0.02, 1, 0, 0, 0]})


def test_visibility_accepts_clear_calibration_scene_and_restores_renderer(env):
    config = sample_visual_config("put_can_in_basket", 11)
    result = screen_visual_config(config, env, ("target", "goal"), (goal_region(env),))
    assert result.accepted, result.reasons
    assert all(area >= 100 for _, area in result.object_pixels)
    assert result.goal_visible_fraction == 1
    assert env.render_camera().shape == (480, 640, 3)


def test_visibility_rejects_evidence_for_a_different_visual_config(env):
    config = sample_visual_config("put_can_in_basket", 12)
    result = screen_visual_config(config, env, ("target", "goal"), (goal_region(env),))
    assert not result.accepted
    assert "config_mismatch" in result.reasons


def test_goal_surface_hidden_by_its_own_support_is_rejected(env):
    p = env.object_pos("goal")
    underside = GoalRegion("goal", tuple(tuple(p + [x, y, -0.014])
                                       for x in (-0.008, 0, 0.008)
                                       for y in (-0.008, 0, 0.008)))
    result = screen_visual_config(env.visual_config, env, ("target", "goal"), (underside,))
    assert not result.accepted
    assert "goal_occluded" in result.reasons


@pytest.mark.parametrize("case,reason", [
    ("outside", "object_outside_frame"), ("tiny", "object_too_small"),
    ("occluded", "goal_occluded"), ("missing_goal", "missing_goal_region"),
])
def test_visibility_rejects_real_geometric_failures(env, case, reason):
    config = sample_visual_config("put_can_in_basket", 11)
    goals = (goal_region(env),)
    if case == "outside":
        env.set_object_pose("target", (2, 0))
    elif case == "tiny":
        env.model.geom_size[env.model.geom("target_geom").id] = [0.0005] * 3
    elif case == "occluded":
        p = env.object_pos("goal")
        camera = env.data.cam_xpos[env._front_cam_id]
        blocker = p + 0.35 * (camera - p)
        env.set_object_pose("occluder", blocker[:2], z=blocker[2] - 0.045)
    else:
        goals = ()
    mujoco.mj_forward(env.model, env.data)
    result = screen_visual_config(config, env, ("target", "goal"), goals)
    assert not result.accepted
    assert reason in result.reasons
    assert result.next_resample_index == 1


def test_duplicate_store_screen_and_persisted_rejection_resume(env, tmp_path):
    store = EpisodeStore(tmp_path)
    config = sample_visual_config("put_can_in_basket", 11)
    key = EpisodeKey("put_can_in_basket", 11)
    store.create_candidate(EpisodeManifest(key, config.config_hash))
    result = screen_visual_config(config, env, ("target", "goal"), (goal_region(env),), store=store)
    assert not result.accepted
    assert "duplicate_config" in result.reasons
    path = save_screening_report(store, key, config, result)
    assert save_screening_report(store, key, config, result) == path
    saved = json.loads(path.read_text())
    assert saved["report"]["next_resample_index"] == 1
    assert saved["report"]["reasons"] == ["duplicate_config"]
    assert saved["visual_config"] == config.to_dict()
