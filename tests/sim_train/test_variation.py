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
    DEFAULT_POLICY, LIGHT_BOX, GoalRegion, VisualConfig, episode_config_hash, exposure_reasons, exposure_stats,
    kelvin_rgb, sample_visual_config, screen_visual_config, save_screening_report,
)
from sim.val.env import ValEnv, FRONT_CAM, WRIST_CAM_POS, WRIST_CAM_QUAT, WRIST_CAM_FOVY
from sim.val.scene import ARENAS, TABLE_SURFACES, Block, SceneSpec, build_scene_xml, camera_xml


# A sampled config whose calibration scene passes framing and exposure screening.
WELL_EXPOSED_SEED = 14


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
                             visual_config=sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED))
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


def _azimuth_elevation(camera):
    offset = np.subtract(camera.pos, camera.lookat)
    return (np.degrees(np.arctan2(offset[1], offset[0])),
            np.degrees(np.arctan2(offset[2], np.hypot(offset[0], offset[1]))))


def test_hundred_seeds_cover_arenas_and_vary_all_sampled_dimensions():
    configs = [sample_visual_config("put_can_in_basket", i) for i in range(300)]
    assert len({c.config_hash for c in configs}) == 300
    assert {c.arena for c in configs} == set(ARENAS) == set(DEFAULT_POLICY.arenas)
    assert len(ARENAS) >= 12
    assert {c.table_surface for c in configs} == set(TABLE_SURFACES)
    for name in ("front_camera", "lights", "background", "ambient", "table_tint", "wall_tint", "floor_tint"):
        assert len({getattr(c, name) for c in configs}) == 300, name
    policy = DEFAULT_POLICY
    azimuths, elevations = zip(*(_azimuth_elevation(c.front_camera) for c in configs))
    assert policy.front_azimuth[0] - 1e-6 <= min(azimuths) < policy.front_azimuth[0] + 10
    assert policy.front_azimuth[1] - 10 < max(azimuths) <= policy.front_azimuth[1] + 1e-6
    assert policy.front_elevation[0] - 1e-6 <= min(elevations) and max(elevations) <= policy.front_elevation[1] + 1e-6
    for c in configs:
        camera = c.front_camera
        assert all(low <= v <= high for v, (low, high) in zip(camera.lookat, policy.front_lookat))
        assert policy.front_fovy[0] <= camera.fovy <= policy.front_fovy[1]
        assert policy.front_roll[0] <= camera.roll <= policy.front_roll[1]
        half_width = np.linalg.norm(np.subtract(camera.pos, camera.lookat)) * np.tan(np.radians(camera.fovy) / 2)
        assert policy.front_half_width[0] - 1e-9 <= half_width <= policy.front_half_width[1] + 1e-9
        names = [light.name for light in c.lights]
        assert names in (["key", "fill", "ceiling"], ["key", "fill", "ceiling", "rim"])
        key = c.lights[0]
        assert policy.key_intensity[0] <= key.intensity <= policy.key_intensity[1]
        assert max(key.color) == 1.0
        for light in c.lights:
            assert all(low - 1e-9 <= v <= high + 1e-9 for v, (low, high) in zip(light.pos, LIGHT_BOX))
            assert light.pos[2] > 0.4  # always above the table, inside the room
    keys = [c.lights[0] for c in configs]
    assert min(k.intensity for k in keys) < 0.4 and max(k.intensity for k in keys) > 1.0
    assert min(k.color[2] for k in keys) < 0.45  # warm (~2700 K)
    assert any(k.color[0] < 1.0 for k in keys)  # cool (> 6600 K)
    assert {k.castshadow for k in keys} == {True, False}
    assert {len(c.lights) for c in configs} == {3, 4}
    key_azimuths = [np.degrees(np.arctan2(k.pos[1], k.pos[0] - 0.18)) for k in keys]
    assert min(key_azimuths) < -150 and max(key_azimuths) > 150


def test_kelvin_colours_run_from_warm_to_cool():
    warm, neutral, cool = kelvin_rgb(2700), kelvin_rgb(6600), kelvin_rgb(9000)
    assert warm[0] == 1.0 and warm[2] < 0.4
    assert min(neutral) > 0.95
    assert cool[2] == 1.0 and cool[0] < 0.85


def test_legacy_configurations_keep_their_hash():
    """Configs recorded before the optional dimensions existed must still verify."""
    legacy = {
        "arena": "kitchen", "variation_seed": 5, "resample_index": 0,
        "front_camera": {"pos": [0.48, -0.13, 0.38], "lookat": [0.15, -0.02, 0.0], "fovy": 48.0},
        "lights": [{"name": "key", "pos": [0.7, 0.8, 1.6], "intensity": 0.6, "color": [1.0, 0.98, 0.95]},
                   {"name": "fill", "pos": [-0.4, -1.2, 1.3], "intensity": 0.3, "color": [1.0, 1.0, 1.0]}],
        "background": {"skybox_top": [0.9, 0.9, 1.0], "skybox_bottom": [0.2, 0.3, 0.4]},
    }
    import hashlib
    value = {k: v for k, v in legacy.items() if k not in ("variation_seed", "resample_index")}
    expected = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = VisualConfig.from_dict({**legacy, "config_hash": expected})
    assert config.config_hash == expected and config.ambient is None and config.front_camera.roll is None


def test_rooms_restyle_visuals_only_and_keep_base_physics():
    """Every training room compiles to exactly its base arena's collision geometry and table height."""
    from sim.val.scene import arena_base, table_top_z

    import tempfile
    from so101_nexus import get_so101_mujoco_model_dir

    def collision(arena):
        with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=get_so101_mujoco_model_dir()) as f:
            f.write(build_scene_xml(SceneSpec(arena=arena, objects=[Block("target")])))
            f.flush()
            model = mujoco.MjModel.from_xml_path(f.name)
        rows = [(model.geom(i).name, model.body(int(model.geom_bodyid[i])).name, int(model.geom_type[i]),
                 tuple(np.round(model.geom_size[i], 9)), tuple(np.round(model.geom_pos[i], 9)),
                 tuple(np.round(model.geom_quat[i], 9)), int(model.geom_contype[i]), int(model.geom_conaffinity[i]))
                for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
        return sorted(rows), [tuple(np.round(model.body_pos[i], 9)) for i in range(model.nbody)][:2]

    bases = {base: collision(base) for base in ("kitchen", "living_room")}
    for arena in ARENAS:
        base = arena_base(arena)
        assert table_top_z(arena) == table_top_z(base)
        assert collision(arena) == bases[base], arena


def test_visual_config_applies_surfaces_lights_roll_and_ambient():
    from sim.val.scene import ASSETS, TABLE_MATERIALS, arena_base
    for seed in range(40):
        config = sample_visual_config("put_can_in_basket", seed)
        if config.table_surface != "default" and len(config.lights) == 4:
            break
    root = ET.fromstring(build_scene_xml(SceneSpec(objects=[Block("target")]), visual_config=config))
    texture, material = TABLE_MATERIALS[arena_base(config.arena)]
    assert root.find(f"asset/texture[@name='{texture}']").get("file") == str(
        (ASSETS / TABLE_SURFACES[config.table_surface]).resolve())
    np.testing.assert_allclose(np.fromstring(root.find(f"asset/material[@name='{material}']").get("rgba"), sep=" "),
                               [*config.table_tint, 1.0], rtol=1e-5)
    np.testing.assert_allclose(np.fromstring(root.find("visual/headlight").get("ambient"), sep=" "),
                               [config.ambient] * 3, rtol=1e-5)
    rim = root.find("worldbody/light[@name='rim']")
    np.testing.assert_allclose(np.fromstring(rim.get("diffuse"), sep=" "),
                               np.array(config.lights[3].color) * config.lights[3].intensity, rtol=1e-5)
    assert root.find("worldbody/light[@name='key']").get("castshadow") == str(config.lights[0].castshadow).lower()
    level = ET.fromstring(camera_xml("c", (0.5, -0.1, 0.4), (0.15, 0, 0), 50))
    rolled = ET.fromstring(camera_xml("c", (0.5, -0.1, 0.4), (0.15, 0, 0), 50, 10.0))
    a, b = (np.fromstring(el.get("xyaxes"), sep=" ") for el in (level, rolled))
    assert np.degrees(np.arccos(np.clip(a[:3] @ b[:3], -1, 1))) == pytest.approx(10.0, abs=1e-3)


def test_exposure_screen_rejects_dark_and_blown_out_frames():
    policy = DEFAULT_POLICY
    assert exposure_reasons(exposure_stats(np.full((48, 64, 3), 120, np.uint8)), policy) == []
    assert exposure_reasons(exposure_stats(np.full((48, 64, 3), 15, np.uint8)), policy) == ["underexposed"]
    assert exposure_reasons(exposure_stats(np.full((48, 64, 3), 255, np.uint8)), policy) == ["overexposed"]
    clipped = np.full((48, 64, 3), 120, np.uint8)
    clipped[:8] = 255
    assert exposure_reasons(exposure_stats(clipped), policy) == ["overexposed"]


def test_task_arena_restriction_expands_to_rooms_on_that_table():
    from sim.val.scene import arena_base
    policy = replace(DEFAULT_POLICY, arenas=("kitchen",))
    assert len(policy.arenas) >= 6 and {arena_base(a) for a in policy.arenas} == {"kitchen"}
    assert replace(DEFAULT_POLICY, arenas=("study",)).arenas == ("study",)


def test_config_is_immutable_json_roundtrips_and_detects_tampering():
    config = sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED)
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
    assert a.config_hash != sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED).config_hash
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
    config = sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED)
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
    config = sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED)
    poses = {"can": [0.2, 0.1, 0.02, 1, 0, 0, 0]}
    a = episode_config_hash(config, poses)
    assert a == episode_config_hash(config, json.loads(json.dumps(poses)))
    assert a != episode_config_hash(config, {"can": [0.21, 0.1, 0.02, 1, 0, 0, 0]})


def test_layout_inclusive_manifest_cannot_omit_visual_identity(tmp_path):
    config = sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED)
    poses = {"can": [0.2, 0.1, 0.02, 1, 0, 0, 0]}
    store = EpisodeStore(tmp_path)
    key = EpisodeKey("put_can_in_basket", 11)
    # This exact previously legal call silently bypassed duplicate screening.
    with pytest.raises(TypeError, match="visual_config_hash"):
        store.create_candidate(EpisodeManifest(key=key, config_hash=episode_config_hash(config, poses), metadata={}))
    assert not store.manifest_path(key).exists()


def test_duplicate_visual_identity_survives_layout_inclusive_manifest(env, tmp_path):
    config = env.visual_config
    store = EpisodeStore(tmp_path)
    key = EpisodeKey("put_can_in_basket", 11)
    full_hash = episode_config_hash(config, env.object_poses())
    assert full_hash != config.config_hash
    store.create_candidate(EpisodeManifest(key, full_hash, metadata={},
                                           visual_config_hash=config.config_hash))
    persisted = json.loads(store.manifest_path(key).read_text())
    assert persisted["visual_config_hash"] == config.config_hash
    record = EpisodeStore(tmp_path).load(key)
    assert record.manifest.config_hash == full_hash
    assert record.manifest.visual_config_hash == config.config_hash
    result = screen_visual_config(config, env, ("target", "goal"), (goal_region(env),), store=store)
    assert not result.accepted
    assert "duplicate_config" in result.reasons


def test_visibility_accepts_clear_calibration_scene_and_restores_renderer(env):
    config = sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED)
    result = screen_visual_config(config, env, ("target", "goal"), (goal_region(env),))
    assert result.accepted, result.reasons
    assert all(area >= 100 for _, area in result.object_pixels)
    assert result.goal_visible_fraction == 1
    assert env.render_camera().shape == (480, 640, 3)


def test_visibility_rejects_evidence_for_a_different_visual_config(env):
    config = sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED + 1)
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
    config = sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED)
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
    config = sample_visual_config("put_can_in_basket", WELL_EXPOSED_SEED)
    key = EpisodeKey("put_can_in_basket", 11)
    store.create_candidate(EpisodeManifest(key, config.config_hash, visual_config_hash=config.config_hash))
    result = screen_visual_config(config, env, ("target", "goal"), (goal_region(env),), store=store)
    assert not result.accepted
    assert "duplicate_config" in result.reasons
    path = save_screening_report(store, key, config, result)
    assert save_screening_report(store, key, config, result) == path
    saved = json.loads(path.read_text())
    assert saved["report"]["next_resample_index"] == 1
    assert saved["report"]["reasons"] == ["duplicate_config"]
    assert saved["visual_config"] == config.to_dict()


def test_viewpoint_worded_tasks_keep_the_original_camera_azimuth_band():
    from sim.train.tasks.catalog import TRAIN_TASKS
    from sim.train.variation import VIEW_ALIGNED
    aligned = {n for n, t in TRAIN_TASKS.items() if t.view_aligned}
    assert {"milk_left_of_bowl", "block_behind_bowl", "turn_book_cover_to_camera", "bar_crosswise_on_mat"} <= aligned
    for name in sorted(aligned):
        for seed in range(30):
            config = TRAIN_TASKS[name].sample_visual_config(seed)
            azimuth, _ = _azimuth_elevation(config.front_camera)
            assert VIEW_ALIGNED["front_azimuth"][0] - 1e-6 <= azimuth <= VIEW_ALIGNED["front_azimuth"][1] + 1e-6
            assert abs(config.front_camera.roll) <= 3 + 1e-9
            assert 38 - 1e-6 <= _azimuth_elevation(config.front_camera)[1] <= 62 + 1e-6
    wide = [_azimuth_elevation(TRAIN_TASKS["block_in_bowl"].sample_visual_config(s).front_camera)[0] for s in range(200)]
    assert min(wide) < -50 and max(wide) > 20


def test_extra_distractors_respect_task_words_family_exclusions_and_heights():
    from sim.train.tasks.assets import EXTRA_DISTRACTORS, eligible_extra_distractors
    from sim.train.tasks.base import LOW_DISTRACTORS
    assert len(set(LOW_DISTRACTORS) | set(EXTRA_DISTRACTORS)) >= 30
    full = eligible_extra_distractors(["Put the block in the basket."], LOW_DISTRACTORS)
    assert "blue_plate" in full and "fork" in full
    assert "black_book_flat" not in eligible_extra_distractors(["Put the can on the book."], LOW_DISTRACTORS)
    assert "fork" not in eligible_extra_distractors(["Put the forks in the cutlery tray."], LOW_DISTRACTORS)
    no_plate = tuple(n for n in LOW_DISTRACTORS if n != "plate")
    assert "blue_plate" not in eligible_extra_distractors(["Center the cube on the coaster."], no_plate)
    short = ("cream_cheese", "butter", "plate")
    assert all(EXTRA_DISTRACTORS[n][2] <= 0.9 for n in eligible_extra_distractors(["Cover the ball."], short))


def test_task_scene_gets_extras_and_recolours_unnamed_primitives_only():
    from sim.train.tasks.catalog import TRAIN_TASKS
    plain = TRAIN_TASKS["block_between_plates"]       # "Put the block between the two plates."
    config = plain.sample_visual_config(5)
    assert config.distractor_extras and set(config.distractor_extras) <= set(plain.variation_policy().extra_distractors)
    assert VisualConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    env_class, _ = plain.load_classes()
    env = env_class(render_images=False, visual_config=config)
    try:
        env.reset(seed=5)
        assert set(config.distractor_extras) <= set(env.distractor_names)
        block = next(o for o in env.scene.objects if o.name == "block")
        assert block.rgba[:3] == config.primitive_palette[0]
    finally:
        env.close()
    named = TRAIN_TASKS["block_behind_bowl"]          # "Put the red block behind the bowl."
    env_class, _ = named.load_classes()
    env = env_class(render_images=False, visual_config=named.sample_visual_config(5))
    try:
        authored = {o.name: o.rgba for o in env.scene_objects() if hasattr(o, "rgba")}
        assert all(o.rgba == authored[o.name] for o in env.scene.objects if o.name in authored)
    finally:
        env.close()
