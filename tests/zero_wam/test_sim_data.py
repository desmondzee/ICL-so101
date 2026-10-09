"""Converter seams: v2.1 layout, action transform, manifest matching, stats, verify.

Uses tiny synthetic buckets — fake source/overlay releases with real H.264 mp4s and parquet —
plus upstream's pinned `lerobot_action` module loaded directly by path (its package __init__
pulls optional training deps that aren't installed here).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import sim.rebuild_wrist_v2 as rw
import zero_wam.sim_data as sd


def _load_lerobot_action():
    spec = importlib.util.spec_from_file_location(
        "lerobot_action", REPO / "third_party" / "Zero-WAM" / "wan_va" / "dataset" / "lerobot_action.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lerobot_action = _load_lerobot_action()

N_FRAMES = 8
TASKS = {"train": ["arm_task_a", "arm_task_b"], "val": ["held_out_task"]}
INSTRUCTIONS = {"arm_task_a": "do thing A.", "arm_task_b": "do thing B.",
                "held_out_task": "do held-out thing."}


def _tiny_video(path: Path, frames=N_FRAMES, shade=80):
    enc = rw._VideoEncoder(path, "libx264")
    for i in range(frames):
        enc.write(np.full((rw.HEIGHT, rw.WIDTH, 3), shade + i, np.uint8))
    enc.close()


def _write_parquet(path: Path, seed: int, frames=N_FRAMES):
    rng = np.random.default_rng(seed)
    action = rng.uniform(-5, 5, (frames, 6)).round(4)
    action[:, 5] = rng.uniform(0, 100, frames).round(2)
    state = action + rng.normal(0, 0.01, (frames, 6)).round(4)
    pq.write_table(pa.table({
        "action": action.tolist(), "observation.state": state.tolist(),
        "timestamp": (np.arange(frames) / 30).tolist()}), path)
    return action, state


def _human_prompt(task: str, ep: int) -> str:
    return ("header lines\n\n"
            "integrated_multimodal_description: [Shot 1] Live-action, beginning with the "
            f"camera framing for {task} episode {ep}. The hand grasps the object.\n\n"
            "overall_soundscape: a quiet room and soft handling sounds\n")


def _fake_bucket(tmp_path):
    bucket = tmp_path / "bucket"
    for split, cfg in sd.SPLITS.items():
        src_ep_root = bucket / cfg["source"] / "episodes"
        wrist_root = bucket / cfg["wrist"] / "episodes"
        entries = []
        for task in TASKS[split]:
            n_eps = 2 if split == "train" else 1
            for ep in range(n_eps):
                rel_dir = src_ep_root / task / f"episode_{ep:03d}"
                rel_dir.mkdir(parents=True)
                (wrist_root / task / f"episode_{ep:03d}").mkdir(parents=True)
                _write_parquet(rel_dir / "robot_data.parquet", seed=ep + len(task))
                _tiny_video(rel_dir / "robot_front.mp4")
                _tiny_video(rel_dir / "human.mp4", frames=N_FRAMES + 2)
                _tiny_video(wrist_root / task / f"episode_{ep:03d}" / "robot_wrist.mp4", shade=20)
                if split == "train":
                    (rel_dir / "source.json").write_text(json.dumps(
                        {"human": {"prompt": _human_prompt(task, ep)}}))
                else:
                    work_prompt = (bucket / "work" / "so101_sim_val" / "human" / task /
                                   f"episode_{ep:03d}" / "seed_0")
                    work_prompt.mkdir(parents=True, exist_ok=True)
                    (work_prompt / "prompt.txt").write_text(_human_prompt(task, ep))
                    (rel_dir / "source.json").write_text(json.dumps({
                        "human_video": f"data/so101_sim_val/human/{task}/"
                                       f"episode_{ep:03d}/seed_0/video.mp4"}))
                entries.append({"id": f"{task}/episode_{ep:03d}", "task": task, "episode": ep,
                                "instruction": INSTRUCTIONS[task], "robot_frames": N_FRAMES})
        (bucket / cfg["source"] / "index.json").parent.mkdir(parents=True, exist_ok=True)
        (bucket / cfg["source"] / "index.json").write_text(json.dumps(
            {"episodes": entries, "episodes_total": len(entries),
             "tasks_total": len(TASKS[split])}))
    return bucket


@pytest.fixture(scope="module")
def bucket_out(tmp_path_factory):
    bucket = _fake_bucket(tmp_path_factory.mktemp("bucket_parent"))
    out = bucket / "out"
    sd.build(bucket, out, workers=1, verify=False)
    return bucket, out


def test_build_structure_and_linked_bytes(bucket_out):
    bucket, out = bucket_out
    train = out / "train"
    assert (train / "meta" / "action_transform.yaml").is_file()
    assert (train / "meta" / "action_stats.json").is_file()
    assert (train / "icl_manifest.json").is_file()
    for task in TASKS["train"]:
        root = train / task
        for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
            assert (root / "meta" / name).is_file(), name
        for ep in range(2):
            assert (root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet").is_file()
            for cam in sd.CAMERAS:
                assert (root / "videos" / "chunk-000" / cam / f"episode_{ep:06d}.mp4").is_file()
    info = json.loads((train / TASKS["train"][0] / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["fps"] == 30
    assert "observation.images.front" in info["features"]
    # linked output bytes equal selected source/overlay bytes
    task = TASKS["train"][0]
    front = train / task / "videos/chunk-000" / sd.FRONT_KEY / "episode_000000.mp4"
    assert rw._sha256_file(front) == rw._sha256_file(
        bucket / "sim_train_v1/episodes" / task / "episode_000" / "robot_front.mp4")
    wrist = train / task / "videos/chunk-000" / sd.WRIST_KEY / "episode_000000.mp4"
    assert rw._sha256_file(wrist) == rw._sha256_file(
        bucket / "train_v2/episodes" / task / "episode_000" / "robot_wrist.mp4")
    human = (train / "human_data/so101/run_train/samples" /
             f"{task}__episode_000/generated_video.mp4")
    assert rw._sha256_file(human) == rw._sha256_file(
        bucket / "sim_train_v1/episodes" / task / "episode_000" / "human.mp4")


def test_manifest_resolves_through_loader_matching(bucket_out):
    _, out = bucket_out
    manifest = json.loads((out / "train" / "icl_manifest.json").read_text())
    assert len(manifest["samples"]) == 4
    for sample in manifest["samples"]:
        # upstream loader's candidate key for episode_index/chunk 0
        ep = int(sample["sample"].rsplit("episode_", 1)[1])
        expected_key = (f"{sample['robot_task_name']}/videos/chunk-000/"
                        f"episode_{ep:06d}.mp4")
        # _video_key_without_view strips the <view> dir from videos/chunk-X/<view>/episode_*.mp4
        assert f"/{sd.FRONT_KEY}/" in sample["robot_video_path"]
        norm = sample["robot_video_path"].replace(f"/{sd.FRONT_KEY}/", "/")
        assert norm == expected_key
        # human path must contain a run_ segment for _human_latent_candidate
        assert f"so101/run_train/samples/{sample['sample']}/generated_video.mp4" == \
            sample["human_video_path"]


def test_manifest_carries_detailed_human_instruction(bucket_out):
    _, out = bucket_out
    manifest = json.loads((out / "train" / "icl_manifest.json").read_text())
    assert manifest["manifest_schema"] == "icl_manifest/2"
    assert manifest["human_text_field"] == "human_local_instruction"
    for sample in manifest["samples"]:
        detailed = sample["human_local_instruction"]
        assert detailed.startswith("Live-action") and "[Shot 1]" not in detailed
        assert "overall_soundscape" not in detailed
        assert sample["human_local_instruction_sha256"] == \
            sd.human_local_instruction_sha256(detailed)
        assert sample["human_text"] == INSTRUCTIONS[sample["robot_task_name"]]


def test_human_local_instruction_extraction():
    body = sd.human_local_instruction(_human_prompt("task_x", 3))
    assert body == ("Live-action, beginning with the camera framing for task_x episode 3. "
                    "The hand grasps the object.")
    with pytest.raises(ValueError, match="markers"):
        sd.human_local_instruction("no markers here")
    with pytest.raises(ValueError, match="markers"):
        sd.human_local_instruction("overall_soundscape: a\nintegrated_multimodal_description: x")
    with pytest.raises(ValueError, match="empty"):
        sd.human_local_instruction(
            "integrated_multimodal_description: [Shot 1] \n\noverall_soundscape: x")


def test_action_transform_and_processor(bucket_out):
    _, out = bucket_out
    task_root = out / "train" / TASKS["train"][0]
    processor = lerobot_action.LeRobotActionProcessor(task_root)
    assert processor.video_keys == list(sd.CAMERAS)
    # batch: raw parquet columns, N_FRAMES rows
    batch = {"action": np.zeros((N_FRAMES, 6)),
             "observation.state": np.zeros((N_FRAMES, 6))}
    table = pq.read_table(task_root / "data/chunk-000/episode_000000.parquet")
    batch["action"] = np.asarray(table.column("action").to_pylist())
    batch["observation.state"] = np.asarray(table.column("observation.state").to_pylist())
    action, mask, action_h = processor.process(
        batch, source_start_frame=0, latent_frame_ids=list(range(9)))
    active = sorted(np.nonzero(mask[0])[0].tolist())
    assert active == [0, 1, 2, 3, 4, 28]
    assert np.isfinite(action).all()
    # loader prepends action_h zero rows; denormalize what follows to recover packaged targets
    stats = json.loads((out / "train/meta/action_stats.json").read_text())
    arm = stats["norm_stats"]["action.hand.position"]
    rows = action[action_h:action_h + N_FRAMES]
    arm_denorm = (rows[:, :5] + 1.0) / 2.0 * (np.asarray(arm["q99"]) - arm["q01"] + 1e-6) + arm["q01"]
    np.testing.assert_allclose(arm_denorm, batch["action"][:, :5], atol=1e-4)
    eff = stats["norm_stats"]["action.effector.position"]
    eff_denorm = (rows[:, 28] + 1.0) / 2.0 * (eff["q99"][0] - eff["q01"][0] + 1e-6) + eff["q01"][0]
    np.testing.assert_allclose(eff_denorm, batch["action"][:, 5], atol=1e-4)
    np.testing.assert_array_equal(action[:action_h], 0.0)  # history padding


def test_stats_are_frozen_physical_bounds_not_quantiles(bucket_out):
    bucket, out = bucket_out
    expected = sd._physical_action_stats()
    for split in ("train", "val"):
        got = json.loads((out / split / "meta/action_stats.json").read_text())
        assert got == expected
    # identical bytes on both splits
    assert rw._sha256_file(out / "train/meta/action_stats.json") == \
        rw._sha256_file(out / "val/meta/action_stats.json")
    ns = expected["norm_stats"]
    assert ns["action.hand.position"]["q01"] == sd.ACTION_Q01[:5]
    assert ns["action.hand.position"]["q99"] == sd.ACTION_Q99[:5]
    assert ns["action.effector.position"]["q01"] == [0.0]
    assert ns["action.effector.position"]["q99"] == [100.0]
    assert expected["method"] == "abs" and expected["window_size"] == 0
    assert expected["source_method"] == "physical_actuator_bounds"
    assert "model_sha256" in expected["provenance"]
    # fixed bounds: independent of the (synthetic, in-range) data distribution
    actions = sd._all_source_actions(bucket, "sim_train_v1",
                                     json.loads((bucket / "sim_train_v1/index.json").read_text()))
    assert not np.allclose(np.quantile(actions[:, :5], 0.01, axis=0),
                           ns["action.hand.position"]["q01"])


def test_out_of_physical_range_source_fails_closed(tmp_path):
    bucket = _fake_bucket(tmp_path)
    bad = bucket / "sim_train_v1/episodes" / TASKS["train"][0] / "episode_000" / "robot_data.parquet"
    action, state = _write_parquet(bad, seed=99)
    action[0, 0] = 200.0  # shoulder_pan beyond the physical ±110-degree bound
    pq.write_table(pa.table({
        "action": action.tolist(), "observation.state": state.tolist(),
        "timestamp": (np.arange(len(action)) / 30).tolist()}), bad)
    with pytest.raises(RuntimeError, match="physical bounds"):
        sd.build(bucket, tmp_path / "out", workers=1, verify=False)


def test_verify_ok_and_tamper(tmp_path, bucket_out):
    bucket, out = bucket_out
    report = sd.verify_conversion(bucket, out)
    assert report["ok"] and report["checked"] == 5 and not report["failures"]
    # tamper: rewrite one output video with different bytes
    victim = (out / "train" / TASKS["train"][0] / "videos/chunk-000" /
              sd.FRONT_KEY / "episode_000000.mp4")
    os_v = victim.with_suffix(".orig.mp4")
    victim.rename(os_v)
    _tiny_video(victim, shade=200)
    try:
        report2 = sd.verify_conversion(bucket, out)
        assert not report2["ok"] and any("hash" in e or "frames" in e for e in report2["failures"])
    finally:
        victim.unlink()
        os_v.rename(victim)


def test_leakage_blocks_build(tmp_path):
    bucket = _fake_bucket(tmp_path)
    index_path = bucket / "sim_val_v2/index.json"
    index = json.loads(index_path.read_text())
    index["episodes"][0]["task"] = TASKS["train"][0]  # force overlap
    index_path.write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="held-out"):
        sd.build(bucket, tmp_path / "out", workers=1, verify=False)
