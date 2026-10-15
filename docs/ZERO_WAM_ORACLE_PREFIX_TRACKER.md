# SO-101 oracle replay and trajectory-prefix diagnosis

## Scope

First validate the current simulator and deployed control conversion using the
original recorded oracle commands on all six corrected training scenes.
Visualize and analyze that result before interpreting policy failures.

Then restrict policy-prefix experiments to `block_in_bowl` (cube pickup and
placement), episodes 000 and 001. Use task-phase boundaries from the recorded
oracle events, confirmed against the new replay's grasp/lift telemetry.
Do not choose or launch a focused retraining experiment yet; discuss its task
and design with the user after these diagnostics.

## Stage 1: original oracle commands

- [x] Inspect recording alignment: row i stores state and camera observations
  **before** executing action i (`sim/train/record.py`).
- [x] Replay all six complete command sequences through `TrainSim.step`, with
  the recorded seeds, visual configuration and complete-layout reset.
- [x] Save front/wrist videos, recorded-versus-replay comparisons, per-control
  pre/post joint states, object xyz, goal errors, grasp flags and success.
- [x] Compare pre-control states with the same parquet rows, final task success
  with the original episode metadata, and final object positions.
- [x] Inspect visual comparisons and report any divergence, including the
  float32 dataset-unit → simulator-radian conversion used during deployment.

Output: `outputs/zero_wam/oracle_command_replay_scenev2/`.
No learned model, smoothing, replanning, fresh oracle generation, or teleporting
to recorded intermediate states. Execute every original command once at 30 Hz.
Do not stop early at success.

Result: **6/6 successes, 6/6 final successes**. All original scene layouts pass.
Mean arm-state error is below 0.00046 degrees in every episode; final task-object
position errors are below 0.4 mm. Videos and comparison sheets reproduce the
successful grasp/carry/place and push trajectories. The current simulator/control
path can execute the original demonstrations successfully.

Detailed analysis:
[oracle/prefix report](results/zero_wam_oracle_prefix/report.md).

## Stage 2: block pickup with trajectory prefixes

- [x] Proceed after stage 1 establishes that original commands solve the block
  task through this simulator path; investigate failures first if it does not.
- [x] Select pre-closure, verified post-grasp/lift, and pre-placement
  handovers from recorded phase events and measured replay states.
- [x] Freeze an explicit manifest with boundaries aligned to eight controls.
- [x] Execute the oracle prefix physically while warming the model with only
  prefix observations and commanded actions; no future trajectory leakage.
- [x] Continue autonomously with V2b, full text/human-video conditioning, CFG 5,
  50/50 denoising, F2 raw, eight-control execution, no EMA.
- [x] Include a zero-prefix control under the same backend and seed convention.
- [x] Save videos and assess grasp retention, lift, transport, placement,
  success, tracking error and divergence after handover.

Prefix warmup must preserve model noise and cache lifecycle explicitly.
This is a test of continuation from demonstrated states, not an ICL claim.

Stage-1 evidence now fixes four conditions per scene: zero prefix, pre-closure,
held after lift, held above bowl. Block 000 boundaries: 0/96/176/288.
Block 001: 0/152/232/368. Source-episode canonical seeds are shared across all
four conditions within a batch. The oracle prefix executes physically while
inference continues and forecasts are discarded, preserving RNG draw cadence.
Only executed commands and live observations enter the cache.

Verification: 16 partial-engine tests pass. Four-slot 32-control smoke passed
prefixes 0/8/16/24; lead checked exact forced commands, correct handover states,
source switching, and policy-command equality after each handover.

Full output: `outputs/zero_wam/block_prefix_handover_scenev2_b4/`.
Log: `/tmp/so101_block_prefix_handover.log`.

Completed: **8/8 rollouts**, with strict success **2/8**, both above-bowl
handovers. Pre-closure prefixes enable learned grasp and approximately 8 cm lift
in both scenes, but the policy opens the gripper before reaching the bowl.
Post-lift prefixes retain the grasp for about 4.7 seconds, then release short.
Above-bowl handovers succeed after 41/42 policy controls (about 1.4 seconds).

The new zero-prefix block-000 run does pick up and place into the bowl: position,
release, settling and bowl support pass, but the strict upright criterion fails
(local-axis tilt about 100.7 degrees vs a 10-degree limit). Block-001 baseline
never grasps. Batch composition changed versus the earlier baseline; initial
predictions differ slightly and later behavior diverges. Do not attribute the
new zero-prefix pickup to a nonexistent prefix.

Validation: all 8 prefix artifact sets align; all 6 oracle comparison videos
have the expected N pre-control frames. Replaying saved policy commands
reproduces final joint states exactly for all 8 conditions and confirms the
success-predicate breakdown. See the report and frozen measurement JSONs.

## Stage 3: discussion

Discuss which task and failure mode deserve a focused training experiment.
No retraining is authorized by this tracker.
