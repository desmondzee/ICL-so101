# SO-101 adaptation tracker

Updated: 2026-09-26

## Goal

Compare two semantically aligned interpretations of Zero-WAM's 30-channel output on a local SO-101 cube-lift task, using a Modal GPU host for inference. This is a zero-shot transfer test of the released RoboTwin post-trained checkpoint; it does not claim that the checkpoint was trained for SO-101.

| Variant | Active channels | Physical meaning | Simulator control |
| --- | --- | --- | --- |
| Pose | `0–6, 28` | Initial-TCP-relative XYZ (m), initial-TCP-relative XYZW quaternion, gripper percent | `pd_ee_pose` IK |
| Joint | `14–18, 28` | Initial-joint-relative 5 arm angles (rad), gripper percent | `pd_joint_pos` |

The 30 channels are capacity blocks. Channel 28 is the single SO-101 gripper; other channels are masked. Both variants use the same text prompt, overhead and wrist camera observations, checkpoint, temporal settings, and episode seeds.

## Implementation status

- [x] Confirmed `so101_nexus.mujoco.pick_env.PickLiftEnv` runs headlessly with `MUJOCO_GL=egl`, and the task prompt is `Pick up the red cube.`
- [x] Added SO-101 channel packing, decoding, quaternion composition, target clamping and control-rate limits in `zero_wam/so101_actions.py`.
- [x] Added a simulator calibration sweep, local episode runner, raw/commanded/executed action logs, success summaries, and overhead MP4s in `zero_wam/so101_eval.py`.
- [x] Added stateful SO-101 configuration and executed-action cache updates to the existing single-container Modal worker in `zero_wam/modal_app.py`.
- [x] For pose actions, action history is repacked from the **measured TCP pose after the simulator step**. Each formatted observation also carries that measured TCP and its active-channel state. The cache therefore receives the realized initial-state-relative motion, rather than the unreachable target. The current Zero-WAM visual encoder reads cameras while its action cache carries the numeric motion; the extra observation state is retained for inspection. Logs include requested-to-realized and commanded-to-realized position/orientation errors plus the IK joint target.
- [x] Completed a 16-step smoke rollout for both variants on seed 100. Both inferred, stepped the simulator, and saved artifacts. Neither succeeded in 16 steps; pose rate/space clipping occurred on all 16 steps, and joint rate clipping on 8 of 16.
- [x] Finished a three-seed paired 64-step post-trained comparison. Both variants completed multi-chunk inference and cache updates, but neither lifted the cube in any seed. This run used the earlier simulator-only calibration and cached `-1` on inactive channels; the code now zeroes inactive channels and mixes in basket calibration, so a revised run is required before comparing modes conclusively.
- [ ] Finish full 400-step episodes on the reachable scene. Both posttrain and pretrain paired runs for seeds 100–102 have started; they write to separate `posttrain_reachable/` and `pretrain_reachable/` folders.
- [x] Added the five existing basket demonstrations to calibration stats, while keeping cube-lift target demonstrations excluded. The simulator sweep now rejects arm/scene contacts deeper than 1 mm. This change applies to the next run; the currently running 64-step trial began with the earlier sweep-only stats.
- [x] Verified a scripted successful cube lift in the exact evaluation environment for seeds 100, 101, and 102. The reachable spawn region is centered near 25 cm; the default seed-100 spawn near 40 cm was unreachable for a downward grasp. The baseline succeeds in 340 controls with 8.3–8.8 cm lift.
- [x] Downloaded the released pretrain checkpoint into the Modal volume and added a separate `zero_wam/modal_so101_pretrain.py` worker for the same paired runner.
- [x] Completed pretrain 16-step smoke for both modes (no success in either short run). An attempted 64-step comparison was stopped when the default cube spawn was found to be unsuitable.
- [x] Completed pretrain paired evaluation on the reachable scene: both modes failed all three 400-step episodes. Every action step hit at least one safety/rate clamp. Pose ended well above the cube in all three seeds; joint ended closer to its XY position but did not grasp.
- [ ] Finish the revised post-trained 64-step comparison with mixed calibration and corrected inactive-channel cache values.

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

## Artifacts and reproduction

The experiment writes `outputs/zero_wam/so101/calibration.json`, `results.json`, and per-mode/per-seed `actions.jsonl`, `summary.json`, and `overhead.mp4`. Local dependencies are in `.venv`; the Modal CLI and SO-101 simulator are available there.

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
