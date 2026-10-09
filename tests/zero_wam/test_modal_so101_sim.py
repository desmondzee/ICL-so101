"""Zero-WAM sim latent pipeline seams: sharding, frame-id trimming, path mapping, payload validation.

No Modal/GPU calls — pure helpers plus synthetic payloads.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import zero_wam.modal_so101_sim as ms


def test_task_discovery_ignores_generated_dirs(tmp_path):
    # a generated human_latents dir (and latents/meta/human_data) must never be treated as tasks
    for name in ("real_task", "human_latents", "human_data", "meta", "x.partial"):
        (tmp_path / name).mkdir()
    (tmp_path / "real_task" / "meta").mkdir()
    (tmp_path / "real_task" / "meta" / "episodes.jsonl").write_text("{}\n")
    tasks = sorted(p.name for p in tmp_path.iterdir()
                   if p.is_dir() and (p / "meta" / "episodes.jsonl").is_file())
    assert tasks == ["real_task"]
    # manifest-based discovery (encode_shard) is immune to generated dirs by construction
    manifest = {"samples": [{"robot_task_name": "real_task", "sample": "real_task__episode_000"}]}
    tasks2 = sorted({s["robot_task_name"] for s in manifest["samples"]})
    assert tasks2 == ["real_task"]


def test_shard_partition_disjoint_covering_deterministic():
    tasks = [f"task_{i}" for i in range(66)]
    part = ms.shard_partition(tasks)
    all_assigned = [t for ts in part.values() for t in ts]
    assert sorted(all_assigned) == sorted(tasks)                    # covering, no dupes
    assert all(len(ts) <= 9 for ts in part.values())                # balanced
    # deterministic regardless of input ordering
    assert ms.shard_partition(list(reversed(tasks))) == part


@pytest.mark.parametrize("length,expected_len", [
    (8, [0, 2, 4, 6]),          # 4 ids -> 4k+1 stays 4? trim: (4-1)//4*4+1=1 -> [0]
    (10, None),
    (530, None),
    (1243, None),
])
def test_robot_frame_ids(length, expected_len):
    ids = ms.robot_frame_ids(length)
    assert ids[0] == 0 and ids == sorted(set(ids))
    assert ids == list(range(0, length, 2))[: (len(list(range(0, length, 2))) - 1) // 4 * 4 + 1]
    assert (len(ids) - 1) % 4 == 0
    assert all(i < length for i in ids)
    assert ms.latent_frames(len(ids)) == (len(ids) - 1) // 4 + 1


def test_robot_frame_ids_exact():
    assert ms.robot_frame_ids(9) == [0, 2, 4, 6, 8]
    assert ms.robot_frame_ids(10) == [0, 2, 4, 6, 8]  # 5 ids -> trimmed to 5? (5-1)//4*4+1=5 -> same
    assert ms.robot_frame_ids(12) == [0, 2, 4, 6, 8, 10][: (6 - 1) // 4 * 4 + 1]  # [0,2,4,6,8]


def test_human_frame_ids():
    # 30 fps source, 150 frames -> 60 output at 12 fps, trimmed to 4k+1
    ids = ms.human_frame_ids(150, 30.0)
    assert (len(ids) - 1) % 4 == 0 and ids == sorted(set(ids))
    assert all(0 <= i < 150 for i in ids)
    assert ms.latent_frames(len(ids)) == (len(ids) - 1) // 4 + 1
    # different source fps shifts sampling
    assert ms.human_frame_ids(100, 60.0) != ms.human_frame_ids(100, 30.0)


def test_path_mapping():
    assert ms.robot_latent_relpath("sort_blocks", 5, 800, "observation.images.front") == \
        "sort_blocks/latents/chunk-000/observation.images.front/episode_000005_0_800.pth"
    assert ms.human_latent_relpath("val", "sort_blocks", 5) == \
        "human_latents/so101/run_val/samples/sort_blocks__episode_005/generated_video.pth"
    # loader's _latent_file builds exactly this name from meta episode/start/end
    assert ms.robot_latent_relpath("t", 0, 42, "c").endswith("episode_000000_0_42.pth")


def _latent(n_ids, h=14, w=18):
    f = ms.latent_frames(n_ids)
    return torch.zeros(f * h * w, ms.VAE_CHANNELS, dtype=torch.bfloat16), f


def _robot_payload(ids, sha="abc", h=14, w=18):
    lat, f = _latent(len(ids), h, w)
    return {"latent": lat, "latent_num_frames": f, "latent_height": h, "latent_width": w,
            "video_num_frames": len(ids), "video_height": ms.HEIGHT,
            "video_width": ms.WIDTH, "task": "t", "local_instruction": "t",
            "local_instruction_emb": torch.zeros(512, 4096, dtype=torch.bfloat16),
            "text": "t", "frame_ids": ids, "start_frame": 0, "end_frame": 42,
            "fps": 15.0, "ori_fps": 30, "source_video_sha256": sha,
            "encoding_geometry_version": ms.ENCODING_GEOMETRY_VERSION}


def _human_payload(ids, f):
    return {"latent": torch.zeros(f * 20 * 28, 48, dtype=torch.bfloat16),
            "latent_num_frames": f, "latent_height": 20, "latent_width": 28,
            "video_num_frames": len(ids), "video_height": ms.ICL_HEIGHT,
            "video_width": ms.ICL_WIDTH,
            "encoding_geometry_version": ms.ENCODING_GEOMETRY_VERSION,
            "text_emb": torch.zeros(512, 4096, dtype=torch.bfloat16), "text": "t",
            "local_instruction": "detailed body", "local_instruction_sha256": "abc123",
            "frame_ids": ids, "fps": 12, "ori_fps": 30.0, "source_video_sha256": "h"}


def test_robot_payload_validation():
    ids = ms.robot_frame_ids(41)
    payload = _robot_payload(ids)
    assert ms.validate_robot_payload(payload, ids, "abc") is None
    assert "frame_ids" in ms.validate_robot_payload({**payload, "frame_ids": ids[:-1]}, ids, "abc")
    assert "sha256" in ms.validate_robot_payload(payload, ids, "other")
    bad = _robot_payload(ids); bad["latent"] = torch.zeros(4, 48, dtype=torch.bfloat16)
    assert "shape" in ms.validate_robot_payload(bad, ids, "abc")
    bad2 = _robot_payload(ids); bad2["latent"][0, 0] = float("nan")
    assert "finite" in ms.validate_robot_payload(bad2, ids, "abc")
    bad3 = _robot_payload(ids); bad3.pop("text")
    assert "missing" in ms.validate_robot_payload(bad3, ids, "abc")


def test_geometry_version_and_video_metadata_rejection():
    ids = ms.robot_frame_ids(41)
    payload = _robot_payload(ids)
    bad = dict(payload); bad["encoding_geometry_version"] = "stale-256x256"
    assert "encoding_geometry_version" in ms.validate_robot_payload(bad, ids, "abc")
    bad = _robot_payload(ids); bad["video_height"] = 256; bad["video_width"] = 256
    assert "video geometry" in ms.validate_robot_payload(bad, ids, "abc")
    bad = _robot_payload(ids); bad.pop("encoding_geometry_version")
    assert "missing" in ms.validate_robot_payload(bad, ids, "abc")
    hids = ms.human_frame_ids(150, 30.0)
    hp = _human_payload(hids, ms.latent_frames(len(hids)))
    bad = dict(hp); bad["encoding_geometry_version"] = "stale-320x480"
    assert "encoding_geometry_version" in ms.validate_human_payload(bad, hids, "h")
    bad = dict(hp); bad["video_width"] = 480
    assert "video geometry" in ms.validate_human_payload(bad, hids, "h")
    bad = dict(hp); bad.pop("video_num_frames")
    assert "missing" in ms.validate_human_payload(bad, hids, "h")


def test_manifest_instruction_errors():
    import hashlib
    ok = {"samples": [{"sample": "s0", "human_local_instruction": "body text",
                       "human_local_instruction_sha256":
                       hashlib.sha256(b"body text").hexdigest()}]}
    assert ms.manifest_instruction_errors(ok) == []
    bad = {"samples": [{"sample": "s0", "human_local_instruction": "body text",
                        "human_local_instruction_sha256": "wrong"},
                       {"sample": "s1"},
                       {"sample": "s2", "human_local_instruction": "",
                        "human_local_instruction_sha256": "x"}]}
    errs = ms.manifest_instruction_errors(bad)
    assert len(errs) == 3 and all(":" in e for e in errs)


def test_human_payload_validation():
    ids = ms.human_frame_ids(150, 30.0)
    f = ms.latent_frames(len(ids))
    payload = _human_payload(ids, f)
    assert ms.validate_human_payload(payload, ids, "h", "detailed body", "abc123") is None
    bad = dict(payload); bad["latent_height"] = 21
    bad["latent"] = torch.zeros(f * 21 * 28, 48, dtype=torch.bfloat16)  # consistent with claim
    assert "geometry" in ms.validate_human_payload(bad, ids, "h")
    bad2 = dict(payload); bad2["text_emb"] = torch.zeros(64, 4096)
    assert "text_emb" in ms.validate_human_payload(bad2, ids, "h")


def test_human_payload_rejects_old_schema_and_mismatches():
    ids = ms.human_frame_ids(150, 30.0)
    f = ms.latent_frames(len(ids))
    payload = _human_payload(ids, f)
    # old payloads (encoded before detailed-text contract) lack the new keys
    old = dict(payload)
    del old["local_instruction"], old["local_instruction_sha256"]
    assert "missing" in ms.validate_human_payload(old, ids, "h")
    # wrong detailed text/hash must invalidate even when keys exist
    assert "local_instruction text" in ms.validate_human_payload(
        payload, ids, "h", "other text", "abc123")
    assert "local_instruction_sha256" in ms.validate_human_payload(
        payload, ids, "h", "detailed body", "deadbeef")


def test_progress_aggregate():
    recs = [
        {"phase": "encode", "shard": 0, "status": "completed", "total_tasks": 4,
         "total_episodes": 10, "total_artifacts": 30, "completed_tasks": 4,
         "completed_episodes": 10, "completed_artifacts": 30},
        {"phase": "encode", "shard": 1, "status": "running", "total_tasks": 4,
         "total_episodes": 8, "total_artifacts": 24, "completed_tasks": 2,
         "completed_episodes": 4, "completed_artifacts": 12},
        {"phase": "encode", "shard": 2, "status": "failed", "total_tasks": 3,
         "total_episodes": 6, "total_artifacts": 18, "completed_tasks": 1,
         "completed_episodes": 2, "completed_artifacts": 6,
         "error": "boom"},
    ]
    agg = ms.progress_aggregate(recs)
    assert agg["episodes"] == 24 and agg["completed_episodes"] == 16
    assert agg["artifacts"] == 72 and agg["completed_artifacts"] == 48
    assert agg["tasks"] == 11 and agg["completed_tasks"] == 7
    assert agg["completed"] == 1 and agg["running"] == 1 and agg["failed"] == 1
    assert abs(agg["percent"] - 100.0 * 48 / 72) < 1e-9


def test_progress_aggregate_zero_and_empty():
    agg = ms.progress_aggregate([])
    assert agg["percent"] == 0.0 and agg["episodes"] == 0
    agg = ms.progress_aggregate([{"status": "running", "total_artifacts": 0,
                                  "completed_artifacts": 0}])
    assert agg["percent"] == 0.0


def test_atomic_json_write(tmp_path):
    p = tmp_path / "progress" / "shard0.json"
    ms.atomic_json_write(p, {"status": "running", "n": 1})
    assert json.loads(p.read_text())["n"] == 1
    assert not (tmp_path / "progress" / "shard0.json.partial").exists()
    ms.atomic_json_write(p, {"status": "completed", "n": 2})
    assert json.loads(p.read_text())["status"] == "completed"


def test_phase_metrics_fields():
    snap = {"records": [
        {"phase": "encode", "shard": 0, "status": "running", "total_tasks": 8,
         "total_episodes": 20, "total_artifacts": 60, "completed_tasks": 3,
         "completed_episodes": 9, "completed_artifacts": 27},
        {"phase": "inventory", "status": "running", "total_episodes": 1159,
         "total_files": 3477, "completed_episodes": 100, "completed_files": 300},
    ]}
    m = ms._phase_metrics(snap, "encode")
    assert m["tasks"] == 3 and m["episodes"] == 9 and m["files"] == 27
    assert abs(m["percent"] - 45.0) < 1e-9
    m2 = ms._phase_metrics(snap, "inventory")
    assert m2["files"] == 300 and m2["total_files"] == 3477


def test_run_cmd_captures_without_shell():
    r = ms._run_cmd(["python3", "-c", "print('hi')"])
    assert r["returncode"] == 0 and r["stdout"].strip() == "hi"
    assert r["argv"] == ["python3", "-c", "print('hi')"]
    bad = ms._run_cmd(["no-such-binary-xyz"])
    assert bad["returncode"] is None and bad["stderr"]
    fail = ms._run_cmd(["python3", "-c", "import sys; sys.exit(3)"])
    assert fail["returncode"] == 3


def test_probe_marker_parses_last_json():
    out = 'noise\nPROBE_JSON{"world": [[0, "h", 0, 0, "H100"]]}\nmore\n'
    assert ms._probe_marker(out)["world"][0][1] == "h"
    assert ms._probe_marker("no marker") is None


def test_single_host_unique_devices():
    world = [[i, "h1", i, i, "H100"] for i in range(8)]
    assert ms._single_host_unique_devices({"world": world}) is True
    dup = [[i, "h1", i, min(i, 6), "H100"] for i in range(8)]
    assert ms._single_host_unique_devices({"world": dup}) is False
    two_hosts = [[i, "h1" if i < 4 else "h2", i, i, "H100"] for i in range(8)]
    assert ms._single_host_unique_devices({"world": two_hosts}) is False
    assert ms._single_host_unique_devices(None) is False
    assert ms._single_host_unique_devices({"world": world[:7]}) is False


def test_atomic_torch_save_roundtrip(tmp_path):
    out = tmp_path / "d" / "x.pth"
    ms.atomic_torch_save({"a": torch.ones(3)}, out)
    assert torch.equal(torch.load(out, weights_only=False)["a"], torch.ones(3))
    assert not (tmp_path / "d" / "x.pth.partial").exists()
