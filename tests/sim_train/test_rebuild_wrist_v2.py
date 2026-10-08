"""wrist_v2 rebuilder seams: geom selection, near-plane transform, resume metadata, path mapping.

Uses the compiled menagerie model and tiny fake episode dirs (no rollouts, no EGL renders).
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from so101_nexus import get_so101_mujoco_model_path

import sim.rebuild_wrist_v2 as rw


@pytest.fixture()
def model():
    return mujoco.MjModel.from_xml_path(str(get_so101_mujoco_model_path()))


def _geom_id(model):
    return rw.camera_mount_visual_geom(model)[0]


def _mount_body_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rw.CAMERA_MOUNT_BODY)


# ----- geom selection -----------------------------------------------------------------------------


def test_camera_mount_geom_found(model):
    geom_id, mesh = rw.camera_mount_visual_geom(model)
    assert mesh == rw.CAMERA_MOUNT_MESH == "wrist_roll_follower_so101_camera_mount"
    assert model.geom_bodyid[geom_id] == _mount_body_id(model)
    assert int(model.geom_group[geom_id]) == rw.ROBOT_VISUAL_GROUP
    assert model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH


def test_camera_mount_geom_rejects_zero(model):
    model.geom_group[_geom_id(model)] = 1
    with pytest.raises(ValueError, match="exactly one"):
        rw.camera_mount_visual_geom(model)


def test_camera_mount_geom_rejects_multiple(model):
    mount = _mount_body_id(model)
    other = next(g for g in range(model.ngeom)
                 if int(model.geom_group[g]) == rw.ROBOT_VISUAL_GROUP
                 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and model.geom_bodyid[g] != mount)
    model.geom_bodyid[other] = mount
    with pytest.raises(ValueError, match="exactly one"):
        rw.camera_mount_visual_geom(model)


def test_camera_mount_geom_rejects_wrong_mesh(model):
    geom_id = _geom_id(model)
    other_mesh = next(i for i in range(model.nmesh) if i != int(model.geom_dataid[geom_id]))
    model.geom_dataid[geom_id] = other_mesh
    with pytest.raises(ValueError, match="mesh"):
        rw.camera_mount_visual_geom(model)


# ----- near-plane transform ------------------------------------------------------------------------


def test_configure_sets_absolute_near(model):
    before = np.array(model.geom_group)
    extent = float(model.stat.extent)
    original_factor = float(model.vis.map.znear)
    provenance = rw.configure_clean_wrist(model)
    # no geom group changes anywhere; the mount keeps group 2
    assert np.array_equal(np.array(model.geom_group), before)
    assert int(model.geom_group[provenance["geom_id"]]) == rw.ROBOT_VISUAL_GROUP
    assert provenance["geom_group"] == rw.ROBOT_VISUAL_GROUP
    # absolute near hits the target within float tolerance
    assert model.vis.map.znear * extent == pytest.approx(rw.TARGET_NEAR_METERS, abs=1e-9)
    assert provenance["near_meters"] == rw.TARGET_NEAR_METERS == 0.02425
    assert provenance["znear_factor"] == pytest.approx(rw.TARGET_NEAR_METERS / extent)
    assert provenance["original_znear_factor"] == pytest.approx(original_factor)
    assert provenance["original_near_meters"] == pytest.approx(original_factor * extent)
    assert provenance["extent"] == pytest.approx(extent)
    assert provenance["mesh_name"] == rw.CAMERA_MOUNT_MESH
    assert model.geom_contype[provenance["geom_id"]] == 0  # still collision-free
    camera = provenance["camera"]
    assert camera["name"] == rw.WRIST_CAMERA
    assert len(camera["pos"]) == 3 and len(camera["quat"]) == 4
    assert Path(provenance["model_path"]).is_file()
    assert len(provenance["model_sha256"]) == 64


def test_absolute_near_invariant_under_different_extent(model):
    """znear is a factor of model.stat.extent: a different extent must still land on 0.02425 m."""
    model.stat.extent = float(model.stat.extent) * 1.37
    extent = float(model.stat.extent)
    rw.configure_clean_wrist(model)
    assert model.vis.map.znear * extent == pytest.approx(rw.TARGET_NEAR_METERS, abs=1e-9)


def test_clean_wrist_option_matches_mujoco_defaults():
    """Default option: scene/visual groups 0-2 on, collision groups 3/4 off, group 5 unused."""
    option = rw.clean_wrist_option()
    assert option.geomgroup.tolist() == [1, 1, 1, 0, 0, 0]


# ----- fake episode fixture -------------------------------------------------------------------------

TASK, EPISODE, SEED, N_ACTIONS = "fake_task", "episode_000", 7, 3


def _fake_source(root: Path) -> Path:
    source = root / "src" / "episodes" / TASK / EPISODE
    source.mkdir(parents=True)
    (source / "episode.json").write_text(json.dumps({"task": TASK, "seed": SEED}))
    (source / "source.json").write_text(json.dumps({"seed": SEED}))
    pq.write_table(pa.table({"action": [[0.0] * 6 for _ in range(N_ACTIONS)]}),
                   source / rw.PARQUET_NAME)
    return source


def _valid_output(source: Path, output: Path) -> dict:
    """Encode a real tiny mp4 and write a wrist_v2.json matching the fake source."""
    output.mkdir(parents=True, exist_ok=True)
    video = output / rw.VIDEO_NAME
    enc = rw._VideoEncoder(video, "libx264")
    for _ in range(N_ACTIONS):
        enc.write(np.full((rw.HEIGHT, rw.WIDTH, 3), 80, np.uint8))
    enc.close()
    meta = rw._episode_meta(source, "train")
    doc = {"schema": "wrist_v2/1", "transform_version": rw.TRANSFORM_VERSION, "split": "train",
           "task": TASK, "episode": EPISODE, "seed": SEED, **rw._expected(meta),
           "output": {"file": rw.VIDEO_NAME, "sha256": rw._sha256_file(video),
                      "frames": N_ACTIONS, "fps": rw.FPS, "width": rw.WIDTH, "height": rw.HEIGHT,
                      "pix_fmt": "yuv420p", "encoder": "libx264"}}
    (output / rw.META_NAME).write_text(json.dumps(doc))
    return doc


def _rewrite_meta(output_dir: Path, **changes):
    doc = json.loads((output_dir / rw.META_NAME).read_text())
    for key, value in changes.items():
        if key == "output":
            doc["output"].update(value)
        else:
            doc[key] = value
    (output_dir / rw.META_NAME).write_text(json.dumps(doc))


def test_verify_accepts_valid_output(tmp_path):
    source = _fake_source(tmp_path)
    output = tmp_path / "out" / "episodes" / TASK / EPISODE
    _valid_output(source, output)
    report = rw.verify_output(source, output)
    assert report["ok"], report["errors"]
    assert report["checks"]["ffprobe"]["nb_read_frames"] == N_ACTIONS


@pytest.mark.parametrize("mutation", [
    "source_parquet", "episode_json", "source_json", "transform_version", "output_sha",
    "frame_count"])
def test_verify_rejects_stale_output(tmp_path, mutation):
    source = _fake_source(tmp_path)
    output = tmp_path / "out" / "episodes" / TASK / EPISODE
    _valid_output(source, output)
    if mutation == "source_parquet":
        pq.write_table(pa.table({"action": [[1.0] * 6 for _ in range(N_ACTIONS)]}),
                       source / rw.PARQUET_NAME)
    elif mutation == "episode_json":
        (source / "episode.json").write_text(json.dumps({"task": TASK, "seed": SEED, "x": 1}))
    elif mutation == "source_json":
        (source / "source.json").write_text(json.dumps({"seed": SEED, "x": 1}))
    elif mutation == "transform_version":
        _rewrite_meta(output, transform_version="hide-camera-mount-v2")  # superseded transform
    elif mutation == "output_sha":
        _rewrite_meta(output, output={"sha256": "0" * 64})
    elif mutation == "frame_count":
        _rewrite_meta(output, output={"frames": N_ACTIONS + 1})
    report = rw.verify_output(source, output)
    assert not report["ok"]
    assert report["errors"]


def test_verify_rejects_missing_and_corrupt(tmp_path):
    source = _fake_source(tmp_path)
    output = tmp_path / "out" / "episodes" / TASK / EPISODE
    report = rw.verify_output(source, output)
    assert not report["ok"] and any("missing" in e for e in report["errors"])
    _valid_output(source, output)
    (output / rw.VIDEO_NAME).write_bytes(b"not a video")
    report = rw.verify_output(source, output)
    assert not report["ok"]


def test_replay_skips_valid_output_without_env(tmp_path):
    """A complete valid output must never be rebuilt — no env is even constructed."""
    source = _fake_source(tmp_path)
    output = tmp_path / "out" / "episodes" / TASK / EPISODE
    _valid_output(source, output)
    before = (output / rw.VIDEO_NAME).stat().st_mtime_ns
    result = rw.replay_episode(source, output, split="train")
    assert result["status"] == "skipped"
    assert (output / rw.VIDEO_NAME).stat().st_mtime_ns == before


def test_output_path_mapping(tmp_path):
    root = tmp_path / "train_v2"
    assert (rw.output_dir_for(root, "train", "push_cube_into_tape_square", "episode_005")
            == root / "episodes" / "push_cube_into_tape_square" / "episode_005")
    assert (rw.output_dir_for(root, "val", "sort_blocks", "episode_005")
            == root / "validation_probe" / "sort_blocks" / "episode_005")


def test_episode_meta_reads_seed_and_hashes(tmp_path):
    source = _fake_source(tmp_path)
    meta = rw._episode_meta(source, "train")
    assert meta["task"] == TASK and meta["seed"] == SEED
    assert meta["parquet_sha256"] == rw._sha256_file(source / rw.PARQUET_NAME)
    assert set(meta["metadata_sha256"]) == {"episode.json", "source.json"}


def test_legacy_distractors_detection():
    old_schema = {"arena": "kitchen", "variation_seed": 1, "resample_index": 0,
                  "front_camera": {}, "lights": [], "background": {}, "config_hash": "x"}
    new_schema = {**old_schema, "distractor_extras": None, "primitive_palette": None}
    assert rw._legacy_distractors({"split": "train", "visual_config": old_schema})
    assert not rw._legacy_distractors({"split": "train", "visual_config": new_schema})
    assert not rw._legacy_distractors({"split": "val", "visual_config": None})


def test_encoder_overwrites_stale_tmp(tmp_path):
    """A leftover temp file from a killed worker must not fail the rebuild (ffmpeg -y)."""
    path = tmp_path / "robot_wrist.mp4.tmp.mp4"
    path.write_bytes(b"stale")
    enc = rw._VideoEncoder(path, "libx264")
    enc.write(np.full((rw.HEIGHT, rw.WIDTH, 3), 40, np.uint8))
    enc.close()
    assert path.read_bytes() != b"stale" and path.stat().st_size > 100


def test_resolve_encoder_fallback():
    assert rw._resolve_encoder("libx264") == "libx264"
    assert rw._resolve_encoder("auto") in ("nvenc", "libx264")
    with pytest.raises(ValueError):
        rw._resolve_encoder("vp9")


# ----- freeze ------------------------------------------------------------------------------------


def _fake_bucket(tmp_path):
    """Tiny bucket root: 2-episode source index and matching valid train_v2 outputs."""
    bucket = tmp_path / "bucket"
    src_root = bucket / rw.TRAIN_SOURCE_DIR / "episodes"
    ids = [f"{TASK}/episode_00{i}" for i in range(2)]
    for i, rel in enumerate(ids):
        source = src_root / rel
        source.mkdir(parents=True)
        (source / "episode.json").write_text(json.dumps({"task": TASK, "seed": 10 + i}))
        (source / "source.json").write_text(json.dumps({"seed": 10 + i}))
        pq.write_table(pa.table({"action": [[0.0] * 6 for _ in range(N_ACTIONS)]}),
                       source / rw.PARQUET_NAME)
        output = bucket / rw.OUTPUT_DIRNAME / "episodes" / rel
        output.mkdir(parents=True)
        video = output / rw.VIDEO_NAME
        enc = rw._VideoEncoder(video, "libx264")
        for _ in range(N_ACTIONS):
            enc.write(np.full((rw.HEIGHT, rw.WIDTH, 3), 80, np.uint8))
        enc.close()
        doc = {"schema": "wrist_v2/1", "split": "train", "task": TASK,
               "episode": rel.split("/")[1], "seed": 10 + i, **rw._expected(rw._episode_meta(source, "train")),
               "output": {"file": rw.VIDEO_NAME, "sha256": rw._sha256_file(video),
                          "frames": N_ACTIONS, "fps": rw.FPS, "width": rw.WIDTH,
                          "height": rw.HEIGHT, "pix_fmt": "yuv420p", "encoder": "libx264"}}
        (output / rw.META_NAME).write_text(json.dumps(doc))
    (bucket / rw.TRAIN_SOURCE_DIR / "index.json").write_text(json.dumps(
        {"episodes_total": 2, "tasks_total": 1, "families_total": 1,
         "episodes": [{"id": rel} for rel in ids]}))
    return bucket, ids


def test_freeze_refuses_on_incomplete_verify(tmp_path, monkeypatch):
    bucket, ids = _fake_bucket(tmp_path)
    monkeypatch.setattr(rw, "_verify_all",
                        lambda root: {"checked": 2, "ok": 1,
                                      "failures": [{"id": ids[1], "errors": ["stale"]}]})
    with pytest.raises(RuntimeError, match="freeze refused"):
        rw.freeze_overlay(bucket)
    assert not (bucket / rw.OUTPUT_DIRNAME / "index.json").exists()


def test_freeze_writes_index_readme_inventory(tmp_path, monkeypatch):
    bucket, ids = _fake_bucket(tmp_path)
    monkeypatch.setattr(rw, "_verify_all",
                        lambda root: {"checked": 2, "ok": 2, "failures": []})
    result = rw.freeze_overlay(bucket)
    assert result["ok"] and result["episodes"] == 2

    root = bucket / rw.OUTPUT_DIRNAME
    index = json.loads((root / "index.json").read_text())
    assert index["schema"] == "wrist_v2/index/1"
    assert index["version"] == "train_v2"
    assert index["transform_version"] == rw.TRANSFORM_VERSION
    assert index["source_prefix"] == rw.TRAIN_SOURCE_DIR
    assert index["source_index_sha256"] == rw._sha256_file(
        bucket / rw.TRAIN_SOURCE_DIR / "index.json")
    assert index["model_sha256"] == rw._sha256_file(rw.get_so101_mujoco_model_path())
    assert (index["episodes_total"], index["tasks_total"], index["families_total"]) == (2, 1, 1)
    assert [e["id"] for e in index["episodes"]] == ids
    ep = index["episodes"][0]
    assert ep["video"] == f"episodes/{ids[0]}/{rw.VIDEO_NAME}"
    assert ep["metadata"] == f"episodes/{ids[0]}/{rw.META_NAME}"
    assert ep["source"] == f"{rw.TRAIN_SOURCE_DIR}/episodes/{ids[0]}"
    assert ep["sha256"] == rw._sha256_file(root / ep["video"])
    assert (ep["frames"], ep["fps"], ep["width"], ep["height"], ep["encoder"]) == (3, 30, 640, 480, "libx264")
    assert ep["seed"] == 10 and ep["legacy_distractors"] is False
    assert "overlay" in (root / "README.md").read_text()

    inventory = json.loads((root / "inventory.json").read_text())
    assert inventory["schema"] == "wrist_v2/inventory/1"
    paths = [f["path"] for f in inventory["files"]]
    assert paths == sorted(paths) and len(paths) == len(set(paths))
    assert "inventory.json" not in paths  # no self-hash paradox
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert set(paths) == on_disk - {"inventory.json"}
    for entry in inventory["files"]:
        path = root / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert rw._sha256_file(path) == entry["sha256"]
    assert inventory["totals"] == {"files": len(paths),
                                   "bytes": sum(f["bytes"] for f in inventory["files"])}
    assert {"index.json", "README.md"} <= set(paths)
