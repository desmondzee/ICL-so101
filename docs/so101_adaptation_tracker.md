# SO-101 adaptation tracker

Updated: 2026-09-26

## Goal

Test whether released Zero-WAM pretrain or RoboTwin posttrain checkpoints can zero-shot control the local SO-101 cube-lift task through a Modal GPU host.

| Variant | Active channels | Physical meaning | Simulator control |
| --- | --- | --- | --- |
| Pose | `0–6, 28` | Initial-TCP-relative XYZ (m), initial-TCP-relative XYZW quaternion, gripper percent | `pd_ee_pose` IK |
| Joint | `14–18, 28` | Initial-joint-relative 5 arm angles (rad), gripper percent | `pd_joint_pos` |

The 30 channels are capacity blocks. Channel 28 is the single SO-101 gripper; other channels are masked. Each trial changes a named factor from the baseline, including its matching normalization and action-history convention where needed.

## Implementation status

- [x] Confirmed `so101_nexus.mujoco.pick_env.PickLiftEnv` runs headlessly with `MUJOCO_GL=egl`, and the task prompt is `Pick up the red cube.`
- [x] Added SO-101 channel packing, decoding, quaternion composition, target clamping and control-rate limits in `zero_wam/so101_actions.py`.
- [x] Added a simulator calibration sweep, local episode runner, raw/commanded/executed action logs, success summaries, and overhead MP4s in `zero_wam/so101_eval.py`.
- [x] Added stateful SO-101 configuration and executed-action cache updates to the existing single-container Modal worker in `zero_wam/modal_app.py`.
- [x] For pose actions, action history is repacked from the **measured TCP pose after the simulator step**. Each formatted observation also carries that measured TCP and its active-channel state. The cache therefore receives the realized initial-state-relative motion, rather than the unreachable target. The current Zero-WAM visual encoder reads cameras while its action cache carries the numeric motion; the extra observation state is retained for inspection. Logs include requested-to-realized and commanded-to-realized position/orientation errors plus the IK joint target.
- [x] Completed a 16-step smoke rollout for both variants on seed 100. Both inferred, stepped the simulator, and saved artifacts. Neither succeeded in 16 steps; pose rate/space clipping occurred on all 16 steps, and joint rate clipping on 8 of 16.
- [x] Finished a three-seed paired 64-step post-trained comparison. Both variants completed multi-chunk inference and cache updates, but neither lifted the cube in any seed. This run used the earlier simulator-only calibration and cached `-1` on inactive channels; the code now zeroes inactive channels and mixes in basket calibration, so a revised run is required before comparing modes conclusively.
- [x] Finished full 400-step episodes on the reachable scene for both checkpoints and both action variants, paired across seeds 100–102. No policy episode succeeded; the scripted baseline succeeds on all three seeds. Runs are in separate `posttrain_reachable/` and `pretrain_reachable/` folders.
- [x] Added the five existing basket demonstrations to calibration stats, while keeping cube-lift target demonstrations excluded. The simulator sweep now rejects arm/scene contacts deeper than 1 mm. This change applies to the next run; the currently running 64-step trial began with the earlier sweep-only stats.
- [x] Verified a scripted successful cube lift in the exact evaluation environment for seeds 100, 101, and 102. The reachable spawn region is centered near 25 cm; the default seed-100 spawn near 40 cm was unreachable for a downward grasp. The baseline succeeds in 340 controls with 8.3–8.8 cm lift.
- [x] Downloaded the released pretrain checkpoint into the Modal volume and added a separate `zero_wam/modal_so101_pretrain.py` worker for the same paired runner.
- [x] Completed pretrain 16-step smoke for both modes (no success in either short run). An attempted 64-step comparison was stopped when the default cube spawn was found to be unsuitable.
- [x] Completed pretrain paired evaluation on the reachable scene: both modes failed all three 400-step episodes. Every action step hit at least one safety/rate clamp. Pose ended well above the cube in all three seeds; joint ended closer to its XY position but did not grasp.
- [x] Replaced the proposed revised 64-step comparison with full 400-step post-trained runs using mixed calibration and corrected inactive-channel cache values.

## Post-trained pilot results (64 steps, provisional calibration)

| Mode | Seed 100 | Seed 101 | Seed 102 | Mean clipping | Mean requested pose error |
| --- | --- | --- | --- | --- | --- |
| Pose | fail | fail | fail | 99.5% | 5.6 cm, 0.27 rad |
| Joint | fail | fail | fail | 22.9% | — |

The pose error columns compare the raw model target with the measured TCP after stepping the simulator. The logs separately contain commanded-to-realized errors, which isolate IK/dynamics error from the safety clamp. These short episodes show no task success and do not rank final policy quality.

The first 64-step comparison used the `PickConfig` default cube spawn. It is retained as a plumbing diagnostic, not an adaptation result. Subsequent evaluations use `spawn_center=(0.25, 0)`, `spawn_min_radius=0`, and `spawn_max_radius=0.025`, where a scripted grasp succeeds for all three paired seeds. See `zero_wam/so101_baseline.py`.

## Reachable-scene pretrain results (400 steps)

| Mode | Success (seeds 100/101/102) | Safety/rate clamp | Final TCP XYZ range |
| --- | --- | --- | --- |
| Pose | 0/3 | 100% of steps | x 0.17–0.20 m, z 0.16–0.21 m |
| Joint | 0/3 | 100% of steps | x 0.27–0.31 m, z 0.04–0.07 m |

The cube is near x 0.25–0.27 m at z 0.012 m. These figures describe zero-shot behavior with the released pretrain checkpoint, not an SO-101 fine-tuned policy. See `outputs/zero_wam/so101/pretrain_reachable/` for per-step actions and videos.

## Reachable-scene posttrain results (400 steps)

| Mode | Success (seeds 100/101/102) | Rate-limit clip by seed | Median TCP height by seed |
| --- | --- | --- | --- |
| Pose | 0/3 | 40.5% / 43.0% / 54.8% | 0.229 / 0.190 / 0.217 m |
| Joint | 0/3 | 4.3% / 4.0% / 6.8% | 0.223 / 0.232 / 0.236 m |

All episodes reached the 400-control limit. The cube remained at z ≈ 0.012 m; these median heights show that neither policy settled into a grasp. See `outputs/zero_wam/so101/posttrain_reachable/` for per-step actions and videos.

## Diagnostics from contact sheets and action logs

Contact sheets (`overhead.contact.png`, now auto-generated per episode) plus per-step `actions.jsonl` were reviewed for the reachable-scene runs.

### Clipping decomposition (posttrain, seed 100)

The historical `clipped` flag in `decode()` counts arm/TCP rate-limit violations (15 mm/step position, 0.15 rad/step rotation, ±0.12 rad/step joint). New runs also log gripper range/rate, arm joint bounds, and workspace clipping separately in `actions.jsonl` and `summary.json` (`zero_wam/so101_actions.py`, `zero_wam/so101_eval.py`). Older artifacts lack those extra fields; quantile-range saturation was measured separately.

| Mode | Rate-limit clip | Workspace-bound clip | Output outside [q01, q99] |
| --- | --- | --- | --- |
| Pose | 41% | 0% | ~0% on all channels **except gripper ch28: 44% below q01** |
| Joint | 4% | — | ~0% (gripper 8% below q01) |

- Pose XYZ/quaternion outputs sit inside the chosen calibration quantiles in seed 100; the 41% `clipped` flag reflects the rate cap. That does not independently prove the calibration matches the checkpoint's learned semantics. Realized TCP descends to z ≈ 0.064 m near the cube (z ≈ 0.012 m).
- In pose seed 100, the gripper is the channel outside the calibration interval: raw values −6 to +5% vs calibration q01 ≈ 0.8%, below the floor on ~half of steps. Under the current mapping (0% → −0.175 rad = closed), the gripper stays near closed. The contact sheets show an approach without an open-gripper phase. Other seeds have different gripper trajectories, so a convention mismatch remains a hypothesis rather than a conclusion.
- Joint seed 100 emits relatively small deltas and spends much of the rollout at an elevated TCP height (median z ≈ 0.223 m). The low rate-limit clip fraction does not establish whether the checkpoint expects absolute or relative joint targets; that remains a testable hypothesis.
- Pretrain saturates ~5–22% per channel on both tails — its raw outputs lie far outside the calibrated quantile bands. Consistent with the demo packing placing joints on channels 0–4 + 28 rather than 14–18, or with no zero-shot transfer to this embodiment.

### Camera viewpoint and resolution

- Reference `third_party/Zero-WAM/example/demo/observation.images.top.png` shows a **front-elevated** table view (arm side-on, workspace horizontal). Our `OverheadCamera` is near **top-down** — a large viewpoint shift for a model trained on front-elevated priors (RoboTwin `cam_high` is also front-elevated).
- Resolution: posttrain (robotwin cfg) expects 288×224 — our cameras match exactly. Pretrain (`va_demo_cfg`) expects 256×256 — our 288×224 frames are bilinearly squashed ~29% horizontally in `_encode_obs`.
- The wrist feed was checked at seed 100: local `/tmp/so101_wrist_100.png` shows both yellow gripper fingers framing the red cube, while `third_party/Zero-WAM/example/demo/observation.images.wrist.png` shows mostly tabletop and only a small dark tool edge. This confirms a substantial wrist-view shift; it does not establish which view the released checkpoints learned.

### Hypothesis test log

| Hypothesis | Evidence / screening | Status |
| --- | --- | --- |
| Gripper direction or scale | `zero_wam/so101_replay.py` replayed raw actions from both checkpoints for normal, inverted, ×20, (+1)×20, and inverted ×20 mappings over seeds 100–102; all 60 replays had zero grasp steps. Live 128-step posttrain pose inverted, (+1)×20, and ×20 screens stayed ≥6.9 cm from the object; pretrain joint inverted and ×20 reached 2.4 and 2.2 cm, with zero grasp in instrumented runs. The pretrain ×20 gripper clipped on 99% of controls. See `outputs/zero_wam/so101/{posttrain,pretrain}_reachable/open_loop_replay.json` and `outputs/zero_wam/so101/trials/`. | Simple mapping changes did not produce a live grasp. |
| Absolute joint targets | Open-loop replay treated saved joint outputs as absolute radians: no grasp or success. Live seed-100 128-step tests shifted quantiles by home joint angles, decoded absolute radians, and cached measured absolute joints (`zero_wam/so101_eval.py`, `joint_absolute`). Pretrain minimum TCP-to-object distance was 2.8 cm versus about 2.4 cm for relative-joint baseline; posttrain was 6.8 cm versus about 6.8 cm baseline. Zero grasp steps in both. Artifacts: `outputs/zero_wam/so101/trials/{pretrain,posttrain}/joint_absolute/seed100_steps128/`. | Absolute-radian semantics alone did not improve either checkpoint. |
| Top camera viewpoint | `OverheadCamera` uses azimuth 0°, elevation −90°. Rendered azimuths 0/90/180/270° at elevation −30°; 270° gives a side view with the cube left of the robot, closest in layout to `example/demo/observation.images.top.png`. Live seed-100 128-step front-view screens failed: posttrain pose minimum TCP-to-object distance 6.9 cm, pretrain joint 3.3 cm, zero grasp in both (baseline first-128 distances about 6.9 and 2.4 cm). See `zero_wam/so101_eval.py` and `outputs/zero_wam/so101/trials/{posttrain/pose_front,pretrain/joint_front}/seed100_steps128/`. | Front view alone did not improve either checkpoint's short screen. |
| Wrist view | Seed-100 sim wrist image and released demo wrist image differ markedly (paths above). Rotating the simulated wrist camera +40° around its local X axis moves the large yellow fingers mostly out of frame while retaining the cube (`/tmp/wrist_x_40.png`). Live seed-100 128-step wrist-tilt screens failed: posttrain pose minimum TCP-to-object distance 6.9 cm, pretrain joint 3.4 cm, zero grasp steps in both. Artifacts: `outputs/zero_wam/so101/trials/{posttrain/pose_wrist_tilt40,pretrain/joint_wrist_tilt40}/seed100_steps128/`. | Wrist tilt alone did not improve either checkpoint. |
| Native resolution | Pretrain config is 256×256 (`third_party/Zero-WAM/wan_va/configs/va_demo_cfg.py`); baseline sim renders 288×224 and server resizes it. Live pretrain joint 128-step native-256 trials (`joint_native256`) reached minimum TCP-to-object distances 4.6/12.6/5.5 mm on seeds 100/101/102, versus 24.1/20.7/4.7 mm from first-128-step baseline raw-action replays. No grasp in any short trial. Full 400-step native-256 trials also failed 0/3 with zero grasp steps and maximum lifts 1.0–1.2 cm; minimum TCP-to-object distances were 18.8/21.5/5.9 mm. Two seed-100 runs with identical manifests matched the first 24 controls, then diverged after the first cache update. Artifacts: `outputs/zero_wam/so101/trials/pretrain/joint_native256/`. | Native resolution alone does not yield control; apparent early reach gains are unstable across repeated runs. |
| Demo channels `0–4,28` | Released demo config selects them; baseline joint variant uses `14–18,28`. Pretrain 128-step channel-only screen reached 3.7 cm; exact published degree-valued absolute quantiles reached 2.9 cm, and with inverted gripper 2.5 cm. Posttrain channel-only reached 6.5 cm. None succeeded; the two published-demo runs logged zero grasp steps. See `third_party/Zero-WAM/wan_va/configs/va_demo_cfg.py` and `outputs/zero_wam/so101/trials/{pretrain,posttrain}/joint_demo*/seed100_steps128/`. | Neither channel-only nor coherent published-demo decoding produced a lift. |
| Normalization | Current quantiles mix 1,725 contact-filtered sim samples and 3,293 unrelated basket actions. Simulator-only 128-step screens failed: posttrain pose reached 6.9 cm, pretrain joint 5.5 cm. Posttrain basket-only pose also reached 6.9 cm. Exact published demo quantiles were tested above. See `outputs/zero_wam/so101/trials/{posttrain/pose_sim_stats,posttrain/pose_basket_stats,pretrain/joint_sim_stats}/seed100_steps128/`. | Neither simulator-only nor basket-only quantiles rescued the screened checkpoint/mode. |
| Text wording | Baseline prompt is `Pick up the red cube.`. `joint_native256_prompt_lift` changes only text to `Grasp the red cube and lift it off the table.` while keeping native-resolution pretrain joint controls (`zero_wam/so101_eval.py`). The live 128-step seed-100 run failed: minimum TCP-to-object distance 1.8 cm, maximum lift 0.5 cm, zero grasp (`outputs/zero_wam/so101/trials/pretrain/joint_native256_prompt_lift/seed100_steps128/`). | No grasp or clear reach improvement from wording alone. |

Combination screens on pretrain joint seed 100: `joint_native256_grip_inverted` failed with zero grasp steps and 3.1 cm minimum distance. `joint_native256_grip_scaled20` reached 4.4 mm but no grasp over 128 controls; its full 400-step runs failed 0/3 with zero grasp steps and 7.8/10.5/5.5 mm minimum distances. Gripper commands exceeded range on 94–97% of steps. `joint_native256_front` also failed its 128-step screen, with 2.7 cm minimum distance and zero grasp. Artifacts: `outputs/zero_wam/so101/trials/pretrain/{joint_native256_grip_inverted,joint_native256_grip_scaled20,joint_native256_front}/`. These combinations follow the single-factor screens above; they do not replace them.

Open-loop replay is only a screen: changed motion changes later camera observations and action history. Named variants, matching quantile/channel remapping, and per-trial manifests (checkpoint revision, stats, cameras, channels, seed, horizon) are implemented in `zero_wam/so101_eval.py` and exposed through `so101_trial`/`so101_screen` in both Modal apps. Each of 31 live trials saves raw and executed actions, video, contact sheet, and summary under `outputs/zero_wam/so101/trials/`; two early runs lack explicit grasp-step telemetry, though neither succeeded. Local mock-worker checks passed for gripper rescaling, front view, native resolution, absolute joints, and demo channels.

### Current conclusion

The scripted controller succeeds 3/3 on the paired reachable scene, but neither released checkpoint succeeds in its baseline pose or joint interpretation (0/3 each, 400 controls). The closest full-horizon variant is pretrain `joint_native256_grip_scaled20`: minimum TCP-to-object distances 7.8/10.5/5.5 mm for seeds 100/101/102, yet 0/3 grasp and lift; its gripper saturates on 94–97% of controls. Native-256 without gripper scaling is also 0/3, with no grasp. Short one-factor screens above find no credible single compatibility fix. These results support SO-101 demonstrations or posttraining as the next step; they do not rule out every untested combination or ICL video conditioning. Close approach alone is not task success.

## Artifacts and reproduction

The experiment writes `outputs/zero_wam/so101/calibration.json`, `results.json`, and per-mode/per-seed `actions.jsonl`, `summary.json`, `overhead.mp4`, and `overhead.contact.png` (24-frame contact sheet, generated automatically by `run_episode`; `run_robotwin.sh` also regenerates sheets for RoboTwin runs after each rollout). Local dependencies are in `.venv`; the Modal CLI and SO-101 simulator are available there.

For a fresh checkout, install the optional local runner dependencies with `uv sync --extra zero-wam`.

```bash
MUJOCO_GL=egl .venv/bin/python -m zero_wam.modal_cli run \
  zero_wam/modal_app.py::so101_compare --seeds 100,101,102 --max-steps 64
```

For a full trial, set `--max-steps 400`. The CLI reads Modal credentials from `.env`; no persistent endpoint is needed.

Pretrain fallback:

```bash
MUJOCO_GL=egl .venv/bin/python -m zero_wam.modal_cli run \
  zero_wam/modal_so101_pretrain.py::so101_compare \
  --seeds 100,101,102 --max-steps 64
```

## Interpretation limits

The simulator sweep uses smooth, bounded joint targets and rejects arm/scene contact states. It mixes those samples with unrelated basket demonstrations; their action distribution may still differ from cube lift. The 16-step smoke confirms the inference/control plumbing, not task competence. Pose orientation residuals are especially important because SO-101 has only five arm DOF; a reachable XYZ target may retain a substantial rotation error.
