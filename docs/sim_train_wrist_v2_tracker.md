# sim_train wrist-video v2 rebuilder tracker

Updated: 2026-10-08

## Objective / scope

Rebuild the wrist video of every released `sim_train_v1` episode so the wrist camera no longer
shows the `camera_mount` housing (the circular clipping artifact), while preserving the exact
recorded robot trajectory. The v2 release is a **slim wrist-video overlay**: per episode it
carries only `robot_wrist.mp4` (regenerated) and `wrist_v2.json` (provenance/resume record).
No front/human/parquet content is duplicated.

Scope of this stage: implement the rebuilder, cover its seams with unit tests, and run the pilot
(one validation probe + one train episode). The 1,109-episode bulk rebuild, full verify, sampled
visual audit, inventory freeze and upload are explicitly **not** in this stage.

## Roots

- Source (immutable, downloaded read-only): `/workspace/ICL-so101/data/hf_bucket_ICL-so101/sim_train_v1`
  (`episodes/<task>/episode_NNN/`, 1,109 index episodes in `index.json`)
- Validation probe source: `.../sim_val_v2/episodes/sort_blocks/episode_005`
- Output root: `/workspace/ICL-so101/data/hf_bucket_ICL-so101/train_v2`
  - rebuilt episodes → `train_v2/episodes/<task>/episode_NNN/`
  - validation probe → `train_v2/validation_probe/sort_blocks/episode_005/`
- Code: `sim/rebuild_wrist_v2.py` (`python -m sim.rebuild_wrist_v2 pilot|rebuild|verify`)
- Tests: `tests/sim_train/test_rebuild_wrist_v2.py`

## Diagnosis

- The wrist camera `wrist_cam` rides inside the `camera_mount` body. That body owns one visual
  geom (group 2, `contype="0"`, mesh `wrist_roll_follower_so101_camera_mount`): the camera
  housing. It occludes ~78% of the 640x480 wrist frame; the scene shows only through a
  circular-ish aperture (the "circular clipping" in v1 wrist videos).
- Verified pixel-exact on the validation probe: a from-scratch render of the seed-6 reset with
  the mount visible matches packaged `sim_val_v2` `robot_wrist.mp4` frame 0 to encoder noise
  (mean |diff| 1.68, 0.18% px>30); the same reset rendered with the mount hidden matches the v2
  output (mean |diff| 0.85, 0.09%).
- The mount geom is collision-free. `absolute-near-plane-v3` sets the wrist camera's absolute
  near plane to 0.02425 m (`model.vis.map.znear = 0.02425 / model.stat.extent`), clipping the
  camera-adjacent mount housing out of direct view while the mount stays in the scene and shadow
  pass — no physics, no collision geoms, no camera pose, no geom group changes.
- **Release-era divergence:** commit `7f8ce42` ("diverse scenes ... second half of sim_train_v1")
  landed mid-release. Pre-`7f8ce42` episodes used the inherited `ValEnv.place_distractors`
  (uniform permutation); later ones use the extras-weighted `TrainEnv.place_distractors`.
  Marker: a recorded `episode.json.visual_config` lacking all of `distractor_extras`,
  `primitive_palette`, `ambient`, `table_surface`, `table_tint`, `wall_tint`, `floor_tint` is a
  legacy-schema config. `replay_episode` binds `ValEnv.place_distractors` for those episodes
  (`legacy_distractors` is recorded in `wrist_v2.json` and enforced on resume). Verified on the
  train pilot: with the legacy selector, post-reset `object_poses()` matches
  `episode.json.object_poses_start` exactly (6/6 poses, max_abs_diff 0.0); with the current
  selector the task objects matched but distractor picks diverged (max_abs_diff 1.38).
- **Validation seed:** `sim_val_v2/episodes/sort_blocks/episode_005` was generated with seed 6
  (index.json and source.json agree; "seed 5" in earlier notes was the episode number, not the
  seed). Verified: seed-6 front render matches packaged `robot_front.mp4` f0 (mean |diff| 2.09,
  0.19% px>30) while seed-5 does not (mean 19.16, 13.7%). Replays read the seed from source
  metadata, not the episode number.
- **Trajectory fidelity:** replaying packaged `action` rows through `env.from_lerobot_joints`
  reproduces the packaged `observation.state` joint rows to max 0.168 deg (mean 0.002 deg) over
  all 743 rows of the validation probe — the replay is deterministic, not a re-plan.

## Immutable-source rule

`sim_train_v1`/`sim_val_v2` are never written. Every output is encoded to a sibling temp file
(`robot_wrist.mp4.tmp.mp4`) and `os.replace`d; `wrist_v2.json`/sheets likewise write-then-rename.
A completed, hash-valid output is never overwritten; a stale/partial one is rebuilt atomically.
No worker deletes anything outside its own temp file.

## GPU / batching / resume plan

- Rendering: MuJoCo EGL (`MUJOCO_GL=egl`), `render_images=False` so `env.step` skips obs renders;
  exactly one wrist render per parquet row (render-then-step → frame count == row count).
- Encoding: H.264 640x480@30 yuv420p via ffmpeg stdin pipe. `h264_nvenc` when a real-size probe
  encode succeeds (nvenc rejects tiny frames, so the probe encodes 640x480), else libx264
  (`--encoder auto|nvenc|libx264`; `nvenc` falls back with a warning and records the codec used).
- Batch: `rebuild --workers N` maps the 1,109 index episodes across spawned processes
  (`ProcessPoolExecutor`, spawn context, chunksize 1); `workers <= 1` stays in-process.
- Resume: an output is skipped only when `wrist_v2.json` matches source parquet SHA256, source
  metadata SHA256 (`episode.json`/`source.json`), model XML SHA256, `transform_version`,
  `legacy_distractors`, output SHA256 + frame count, and ffprobe decodes the expected geometry
  and exact frame count. Any mismatch rebuilds atomically.

## Acceptance gates

- `camera_mount_visual_geom` fails closed unless exactly one group-2/contype-0 mesh geom on body
  `camera_mount` exists and its mesh is `wrist_roll_follower_so101_camera_mount`.
- `configure_clean_wrist` identifies the mount geom as a fail-closed identity check, then sets
  `model.vis.map.znear = TARGET_NEAR_METERS / model.stat.extent` and asserts the geom's group is
  unchanged; provenance records the mount geom/mesh/body, both near factors, and model hashes.
- `clean_wrist_option` returns the MuJoCo-default `MjvOption` (`[1,1,1,0,0,0]` asserted); no
  geomgroup entry is altered, so collision groups 3/4 stay hidden.
- Output frames == packaged `action` rows; ffprobe reports h264, 640x480, 30/1, yuv420p, and the
  same decoded frame count.
- `verify` exits nonzero on any mismatch or missing output.

## Caught defects

- **Group-hidden mount removed its shadow (2026-10-08, superseded):** the `hide-camera-mount-v2`
  transform moved the mount geom to render group 5, which removed the mount **including its cast
  shadow** — the mount is physically present in the scene, so a shadow-less render is wrong.
  Superseded by `absolute-near-plane-v3`: `model.vis.map.znear = 0.02425 / model.stat.extent`
  sets an absolute near plane of 0.02425 m that clips the camera-adjacent mount housing out of
  direct view while the mount stays in the scene and shadow pass. Measured directly-rendered
  mount pixels vs absolute near: **29 at 0.02375 m, 2 at 0.02400 m, 0 at ≥0.02425 m** — so
  `TARGET_NEAR_METERS = 0.02425`, the smallest tested 0.25 mm-grid value at zero. Baselines: train
  near was 0.0212196 m, validation 0.0236620 m (66 mount px); swapping those values swapped the
  measured train/val segmentation bounds exactly, proving the near plane is the mechanism.
- **Worker OOM during bulk (2026-10-08, fixed):** each episode's env + EGL renderer leaks
  ~100-150 MB of native-side memory despite `env.close()`+gc (measured RSS curve). The first
  bulk pass reached 579/1109 before a worker was OOM-killed (24 GB cgroup, `oom_kill=1`),
  breaking the pool (`BrokenProcessPool`). Fix: a fresh `ProcessPoolExecutor` per batch of
  `workers * 15` jobs bounds worker lifetime, and a `BrokenProcessPool` restart loop retries the
  remaining jobs (resume is idempotent). `max_tasks_per_child` was tried first but deadlocked
  worker spawn under this Python 3.12.3 build — replaced by explicit batch pools.
- **Stale temp file:** the OOM kill stranded two `robot_wrist.mp4.tmp.mp4` temps; plain ffmpeg
  refused to overwrite them (`BrokenPipeError` on write, 2 failed jobs). The encoder now passes
  `-y`; both episodes rebuilt cleanly on the resume pass.
- **Geom-group visibility (2026-10-08, fixed before bulk):** `mujoco.MjvOption()` defaults
  `geomgroup` to `[1,1,1,0,0,0]` — collision groups 3 (`collision_gripper`, red) and 4
  (`collision_gripper_mesh`, faceted proxies) are OFF by default, which is why packaged videos
  show gray visual jaws and opaque objects. The first `clean_wrist_option()` forced
  `[1,1,1,1,1,0]`, exposing collision jaws as red and collision meshes as translucent facets in
  the regenerated sheets. `clean_wrist_option()` now asserts the MuJoCo default and only zeroes
  group 5; `TRANSFORM_VERSION` bumped to `hide-camera-mount-v2` so every v1-transform output
  fails resume and rebuilds. Post-fix frame-0 checks: regenerated outputs are pixel-matches of
  the intended clean renders (val mean |diff| 1.34 / 0.02% px>30; train 1.84 / 0.01%).
- nvenc probe: a 64x64 probe frame is below nvenc's minimum frame dimension and reported the GPU
  encoder as unavailable; the probe now encodes at the real 640x480.
- Sheet temp file: PIL refused the `.tmp` extension; sheet temps are `*.tmp.jpg`.

## Pilot results — v3 `absolute-near-plane` (2026-10-08)

Commands run:

```bash
MUJOCO_GL=egl uv run pytest tests/sim_train/test_rebuild_wrist_v2.py -q        # 23 passed
MUJOCO_GL=egl uv run python -m sim.rebuild_wrist_v2 pilot \
    --bucket-root /workspace/ICL-so101/data/hf_bucket_ICL-so101 --encoder nvenc
MUJOCO_GL=egl uv run python -m sim.rebuild_wrist_v2 verify \
    --bucket-root /workspace/ICL-so101/data/hf_bucket_ICL-so101                # exit 1: checked 1109, ok 1 (expected pre-v3-bulk)
```

- `validation_probe/sort_blocks/episode_005` (seed 6): rebuilt under `absolute-near-plane-v3`,
  **743 frames** (= 743 rows), 24.77 s, h264 yuv420p 640x480@30 via h264_nvenc,
  `output_sha256=bf1d43e5…69f4d6`. Original near 0.0236620 m → 0.02425 m
  (znear factor 0.004000→0.004099 at extent 5.9155).
- `episodes/push_cube_into_tape_square/episode_005` (seed 100008): rebuilt under
  `absolute-near-plane-v3`, **530 frames** (= 530 rows), 17.67 s, h264 yuv420p 640x480@30 via
  h264_nvenc, `output_sha256=093740bb…6261ad`; `layout_check` 6/6 poses exact, max_abs_diff 0.0.
  Original near 0.0212196 m → 0.02425 m (znear factor 0.004000→0.004571 at extent 5.3049).
- Segmentation proof (MuJoCo `render` in seg mode, frame 0, default option): geom 36
  `wrist_roll_follower_so101_camera_mount` renders **0 pixels in both pilots** while its geom
  group remains **2**; absolute near measured 0.02425 m in both.
- Shadow proof (train f0): normal v3 render vs identical render with the mount geom hidden
  (diagnostic-only group move, same 0.02425 m near): **1,556 changed px** (>30), a coherent blob
  at x∈[0,141], y∈[250,479] — the mount's cast shadow on the table, present only when the mount
  remains in the scene. 12,091 px differ >10 including the soft-shadow falloff.
- Near-pure-red pixel fraction matches packaged levels (v3 val 0.52% vs packaged 0.54%; v3 train
  1.96% vs packaged 1.96%) — no collision-group content; gray visual jaws and task objects
  visible in both sheets.
- Geom provenance (both): geom_id 36, mesh `wrist_roll_follower_so101_camera_mount`, body
  `camera_mount`, group **2 unchanged**; camera pos `[0.0025, 0.06157, -0.01877]`, quat
  `[0.98481,-0.17365,0,0]`, fovy 65.0; model `so101.xml` sha256 `038b7983…313b1`.
- Stale `train_v2/index.json`, `README.md`, `inventory.json` from the v2 freeze were removed —
  they referenced v2 hashes superseded by v3 outputs.

## Pilot results — v2 `hide-camera-mount` (superseded 2026-10-08)

Commands run:

```bash
MUJOCO_GL=egl uv run pytest tests/sim_train/test_rebuild_wrist_v2.py -q        # 21 passed
MUJOCO_GL=egl uv run python -m sim.rebuild_wrist_v2 pilot \
    --bucket-root /workspace/ICL-so101/data/hf_bucket_ICL-so101 --encoder nvenc
MUJOCO_GL=egl uv run python -m sim.rebuild_wrist_v2 verify \
    --bucket-root /workspace/ICL-so101/data/hf_bucket_ICL-so101                # exit 0: checked=ok=1109 (post-bulk)
```

- `validation_probe/sort_blocks/episode_005` (seed 6): rebuilt under
  `hide-camera-mount-v2`, **743 frames** (= 743 rows), 24.77 s, h264 yuv420p 640x480@30 via
  h264_nvenc, `output_sha256=b5624a15…697d33`. Clipping-free; gray visual jaws and opaque task
  objects/plates as in the packaged view. Regenerated frame 0 vs packaged `robot_wrist.mp4`
  frame 0: mean absolute pixel difference **9.32** (the removed mount aperture + encoder noise;
  outside the mount footprint it is encoder noise only).
- `episodes/push_cube_into_tape_square/episode_005` (seed 100008): rebuilt under
  `hide-camera-mount-v2`, **530 frames** (= 530 rows), 17.67 s, h264 yuv420p 640x480@30 via
  h264_nvenc, `output_sha256=b7ee8e95…ba0b44`; `layout_check` 6/6 poses exact. With-mount render
  reproduces packaged v1 frame 0 to mean |diff| 2.01, proving the packaged episode carries the
  mount; regenerated output has the circular mount aperture removed while the gray jaw/task
  scene remains (sheet right-half texture std 12-20 in the living-room lighting).
- Near-pure-red (collision-jaw) pixel fraction now matches packaged levels: val 0.52% vs
  packaged 0.54%; train 2.01% vs packaged 1.96% (residual is scene content, not group-3 jaws).
- Geom provenance (both): geom_id 36, mesh `wrist_roll_follower_so101_camera_mount`, body
  `camera_mount`, group 2→5; camera pos `[0.0025, 0.06157, -0.01877]`, quat `[0.98481,-0.17365,0,0]`,
  fovy 65.0; model `so101.xml` sha256 `038b7983…313b1`.
- Train resume/provenance now hashes both `episode.json` and `source.json`
  (`TRAIN_META_FILES`); mutating either invalidates the output (tested).

## Work queue

- [x] Implement `sim/rebuild_wrist_v2.py` (geom seam, clean-wrist render, replay, spawn workers,
      resume metadata, verify, CLI).
- [x] Unit tests (v3): geom selection (zero/multiple/wrong mesh), absolute near = 0.02425 m
      incl. differing extent, geom group stays 2, default visibility unchanged, resume-hash
      rejection (incl. superseded `hide-camera-mount-v2`), stale-temp overwrite, path mapping,
      freeze gate/artifacts — 23 passed.
- [x] Pilot (v3): validation probe (sort_blocks/episode_005, seed 6) + train pilot
      (push_cube_into_tape_square/episode_005, seed 100008) rebuilt with sheets; ffprobe clean;
      0 mount pixels by segmentation; mount shadow preserved.
- [x] Bulk rebuild under v3 (`rebuild --workers 3 --encoder nvenc`, 2026-10-08):
      **1,108 rebuilt + 1 skipped (v3 pilot), 0 failed**; 103m28s; all v2 metadata failed resume
      on transform mismatch and was atomically replaced. Zero pool restarts, zero EGL/NVENC or
      process errors in the log; all 1,109 outputs are `absolute-near-plane-v3` + h264_nvenc;
      436 episodes used the legacy distractor selector; `layout_check` passed everywhere.
      v2 record superseded (530 rebuilt + 579 skipped, 0 failed; 45m49s + 3m53s).
- [x] Full `verify` green under v3: **checked=1109, ok=1109, failures=0**, exit 0, 9m24s.
      Output tree: `train_v2/` = 3.0 GB, 2,222 files (1,109 episode mp4+json + pilot sheet +
      3 validation-probe files), zero stray temp files. v2 result superseded (was 1109/1109).
- [x] Sampled visual audit under v3 (2026-10-08): `/tmp/sim_train_wrist_v3_audit/overview.jpg`,
      one episode from each of all 29 families, 12 uniformly spaced frames each. Result: **29/29**
      show no direct mount clipping or circular aperture; the mount's cast shadow remains visible
      (typically square/ring-shaped); normal gray jaws and task objects remain; no collision
      groups exposed. Audit files are temporary and are not in the release.
      v2 result superseded (was 29/29 families clean).
- [x] Freeze under v3 (2026-10-08): `python -m sim.rebuild_wrist_v2 freeze --bucket-root
      /workspace/ICL-so101/data/hf_bucket_ICL-so101` (9m20s, gated on verify checked=ok=1109).
      `index.json` sha256 `e9734148…61c858` (schema `wrist_v2/index/1`, 1,109 episodes, source
      order); `README.md` sha256 `44db4fef…d58293`; `inventory.json` sha256 `6eda5dc6…188793`
      (schema `wrist_v2/inventory/1`, **2,224 files / 3,149,130,920 bytes**). Independent check:
      index ids unique + order/set match source; all indexed video/meta hashes match; every
      episode meta = `absolute-near-plane-v3`, near_meters 0.02425, geom_group 2, nvenc;
      inventory sorted/unique, all sizes+hashes verified, no undeclared files, totals exact.
      v2 record superseded: `7f657bc8…f15622` / `433ab537…3144370` / `79a56709…157994`
      (2,224 files / 3,023,202,648 bytes).
- [x] Upload `train_v2` to `hf://buckets/akoniti/ICL-so101/train_v2` without `--delete` and verify readback (2026-10-08). Initial sync uploaded 2,225 files; a post-upload dry run classified every path as identical. Downloaded `README.md`, `index.json`, `inventory.json` and representative first/middle/last-task videos (`ball_between_cans/episode_000`, `push_cube_into_tape_square/episode_005`, `two_cubes_on_plate_in_order/episode_014`); all sizes and SHA-256 hashes matched local files, and all three videos decoded as H.264 640x480 at 30 fps.

## Resolved bulk question

- Pre-`7f8ce42` legacy episodes in the nine family modules that commit touched were a divergence
  risk for `layout()` beyond distractors. The bulk run's hard `layout_check` gate (fail when
  post-reset poses differ from packaged `object_poses_start` by >1e-3 or any recorded pose is
  missing) passed on every episode — 0 failures across all 1,109 — so the legacy selector fully
  covers the pre-diversification layout behavior.
