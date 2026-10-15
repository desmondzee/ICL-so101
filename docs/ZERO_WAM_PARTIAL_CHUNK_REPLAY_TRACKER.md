# Partial-chunk training replay and temporal ensembling

<!-- PARTIAL_SWEEP_LIVE_START -->
## Live sweep status

Last refreshed: **2026-10-15T16:05:07+00:00**.
Final completed snapshot. The automatic updater was stopped for the final analysis.

| Case | Complete | Successes | State |
|---|---:|---:|---|
| 16 controls, no ensemble | 6/6 | 0 | Completed |
| 16 controls, ensemble | 6/6 | 0 | Completed |
| 32 controls, no ensemble | 6/6 | 0 | Completed |
| 32 controls, ensemble | 6/6 | 0 | Completed |

Completed episodes: **24/24**.
Continuation supervisor: `1881026` (absent).
Suite log: `/tmp/so101_partial_scenev2_b4_suite.log`.
HF sync log: `/tmp/so101_partial_bucket_sync.log`.


<!-- PARTIAL_SWEEP_LIVE_END -->

## Completed analysis

All four corrected batch-four cohorts completed: 24/24 rollouts, 0/24 successes.
The final report, frozen measurements, runtime extracts and six visual comparison
sheets are in
[the detailed report](results/zero_wam_partial_chunk_replay/report.md).

F4 ensembling reduced commanded arm jitter by 23.8% and measured joint jitter
by 27.9% versus F4 raw, without successful manipulation. F2 ensembling increased
mean trajectory variance by 52.1%; it produced the only brief opposing-finger
pinch (two nine-control intervals on block-000), with no successful carry.
Neither pushing episode moved its cube meaningfully in any mode.

Final checks: 95 focused tests passed; all 24 video/action/telemetry artifact
sets align. Analysis independently verified ensemble arithmetic, per-slot cache
commit schedules, horizons, seeds and complete recorded layouts.
Prioritize exact oracle-command replay through the deployed simulator/control
path, then phase-specific handovers and a small closed-loop memorization test.

The completed experiments use `_scenev2_b4` output suffixes. Earlier launch,
pause and planning records below are retained as historical provenance.

## Authorization to proceed after reference review

The user reviewed the small initial prediction differences and increasing
closed-loop divergence, then explicitly authorized proceeding with the batch-four
sweep. Keep the failed reference comparison unchanged; this is a consistent
batch-four experiment, not a claim of sequential trajectory equivalence.
Resume the paused F2/raw cohort and run F2/ensemble, F4/raw, F4/ensemble after
it completes. All other protocol settings and `_scenev2_b4` outputs stay fixed.
The continuation launcher requires `--accept-reviewed-closed-loop-drift` to
record this explicit decision when the comparison is failed.

## Batch-four transition requested

Finish the first corrected sequential F2/raw rollout with seed 20261101 and
retain it as the reference. Stop the sequential suite after that episode.
Batch four must retain the four-arm protocol, exact reconstructed scenes,
target-text CFG 5, paired human ICL/detailed text, 50/50 denoising, eight executed
controls per update, no EMA, and canonical per-episode seeds.

- [x] Implement separate `BatchedPartialChunkEngine` with the same committed
  temporal cursor and actual-command-only history as the sequential adapter.
- [x] Author `scripts/run_so101_batched_partial_replay.py` and a reference
  comparison harness. Preserve existing scene provenance.
- [x] First reference rollout completes and sequential suite stops.
- [x] Four active slots, fixed-reference-history comparison at eight replans
  (64 controls). Slot zero executes reference commands so prediction differences
  are measured without autonomous state divergence.
- [x] Full autonomous reference-episode comparison with matched seed (FAILED).
- [x] F4/ensemble and padding GPU check (two real + two padding, 40 controls).
- [x] Switch the remaining sweep to batch four after user review of measured drift.

Numerical gates, fixed before observing candidates: every fixed-history forecast
must have relative L2 <=1%, mean absolute difference <=0.5% of per-channel
actuator range, and maximum difference <=2% of range. These are practical
numerical tolerances, not bitwise equivalence or proof of policy equivalence.
Reference commands and measured states must match in the fixed-history probe.

The autonomous comparison reports full action trajectory, task success and
object interaction. Initial acceptance requires equal control count and success
states; normalized action RMSE <=2% of actuator range; closest-distance change
<=5 mm, displacement/lift changes <=1 cm, and grasp-duration change <=8 controls.
A failure must be reported rather than loosening thresholds silently or
claiming a match from equal task failure alone.

Packed artifacts must remain separate from sequential artifacts and retain
batch size, slot membership, per-slot seeds and reference-check evidence.

Sequential reference: corrected `ball_between_cans/episode_000`, seed 20261101,
1,104 controls, success false, no grasp, closest TCP 0.0397811 m. The completed
reference is under the original `_scenev2` F2/raw directory.

Fixed-history batch-four check passed all eight replans: per-forecast relative
L2 differences 0.153–0.517%, mean range-normalized absolute differences
0.0081–0.0387%, maximum 0.391%. Actual simulator states are exactly equal after
normalizing telemetry to float32 (initial packed telemetry serialized float64;
the runner now uses float32 like the sequential path). No numerical acceptance
threshold was relaxed. Evidence:
`outputs/zero_wam/partial_batch4_reference_history/comparison.json`.
Focused suite: 95 passed.

Full packed runs append `_b4` to corrected `_scenev2` directory names.
The autonomous F2/raw reference comparison is the next gate; fixed-history
agreement alone is not proof of closed-loop trajectory equivalence.

Autonomous comparison completed and failed the declared gate. Both runs have
1,104 controls and success false, but range-normalized action RMSE is 8.618%
(limit 2%). Closest TCP-to-ball distance is 3.978 cm sequential versus 7.075 cm
packed (difference 3.097 cm; limit 0.5 cm). Neither grasped or moved the ball.
Their full recorded scene checks pass.

The first eight executed predictions differ by only 0.0399% range-normalized
RMSE, but controls 8–63 already differ by 2.338%. Fixed-history numerical
agreement alongside autonomous divergence is consistent with sensitivity to
small numerical perturbations; it does not prove a specific floating-point
kernel mechanism or exclude later cache differences.

Evidence:
`outputs/zero_wam/train_replay_v2b_partial8_raw_full_conditioning_3tasks_2seeds_scenev2_b4/reference_comparison.json`.
Python PID 368540 was paused with SIGSTOP after the reference result; other
three simulator states remain in memory. Remaining sweep modes have not been
queued. `scripts/run_so101_remaining_batched_partial_sweep.sh` refuses automatic
continuation on this failed comparison. HF syncing includes batch artifacts and
both bounded checks. Batch four is operational but not established as an
equivalent replacement for this sequential rollout.

## Scene mismatch correction

The first suite was stopped after discovering that seed-based resets did not
reproduce the original distractors. Commit `7f8ce42` changed the sampler from
permutation to weighted ranking. The earlier gate verified only manipulated
objects, leaving references and distractors unchecked.

`zero_wam/train_scene_replay.py` now selects the current or legacy sampler only
when every active object's identity and full seven-component pose matches the
recorded metadata. Otherwise reset fails. No post-reset pose teleportation is
used. All six cohort episodes match exactly with the legacy sampler (rounded
recorded poses; signed zero treated equally), and visual config hashes match.
Front/wrist comparison images were inspected for all six.

Evidence: `outputs/zero_wam/train_replay_scene_audit/` (audit JSON and six JPGs).
Focused suite after correction: 91 passed. Corrected runs use the same four
directory names below **with `_scenev2` appended**. Old outputs are preserved
as mismatched-scene diagnostics and must not be pooled with corrected results.
New per-arm logs: `/tmp/so101_partial_scenev2_f{2,4}_{raw,ensemble}.log`.
The original PID/progress records below describe the stopped first suite.

## Objective and current status

Compare predicting 16 controls and executing eight, with and without temporal
ensembling, on exact seen SO-101 training episodes. Determine whether earlier
visual feedback improves grasp/contact accuracy and whether overlapping forecasts
reduce roughness without delaying corrections.

- [x] Stop the sequential full-conditioning, 16-control execution run.
  Python PID 3730958 was terminated on user request. Its completed
  `ball_between_cans/episode_000` artifacts remain under
  `outputs/zero_wam/train_replay_v2b_sequential_raw_full_conditioning_3tasks_2seeds/`.
  The interrupted episode is not a completed evaluation.
- [x] Freeze the comparison design below.
- [x] Implement a separate opt-in partial-chunk sequential inference adapter.
- [x] Verify cache positions, observation cadence, and actual-command history.
- [x] Run bounded GPU checks for all four modes.
- [x] Finish all four full-horizon six-episode cohorts (24/24 complete).
- [x] Analyze matched results, save report and manifests, and update this tracker.

Implementation: `zero_wam/partial_chunk_engine.py`, with replay runner
`scripts/run_so101_partial_chunk_replay.py`. Existing sequential and packed
inference paths retain their behavior. The adapter is installed per instance.

Verification: 87 focused tests pass. All four 32-control GPU preflights pass;
an additional 40-control four-frame ensemble preflight verifies four overlapping
contributors with weights 0.025, 0.075, 0.225 and 0.675 from oldest to newest.
Independent artifact checks confirm actual command arithmetic, startup equality
between raw/ensemble at each forecast length, and temporal cursor progression
0→2→3→4→5→6. Observation history is cache type 0; generated predictions are
removed before commits, and the human ICL cache remains type 2.

Bounded outputs are under `outputs/zero_wam/partial_chunk_preflight/` and are
not full-horizon evaluation results.

Full-suite launcher: `scripts/run_so101_partial_replay_suite.sh`, in arm order
A, B, C, D. Per-arm logs: `/tmp/so101_partial_full_f{2,4}_{raw,ensemble}.log`.
Each arm runs all six episodes. A failure stops the suite rather than silently
skipping a condition.

Suite launched with supervisor PID 3947550; initial Python worker PID 3947555.
Arm A (`forecast_frames=2`, ensemble off) was verified through 120 controls
and committed-frame cursor 16 on `ball_between_cans/episode_000`.
Suite log: `/tmp/so101_partial_replay_suite.log`. These are launch records,
not completion evidence.

## Original experimental protocol (later scene and backend amendments above)

Use V2b step 500 and the existing exact training replay cohort:

1. `ball_between_cans/episode_000`
2. `ball_between_cans/episode_001`
3. `block_in_bowl/episode_000`
4. `block_in_bowl/episode_001`
5. `push_cube_into_tape_square/episode_000`
6. `push_cube_into_tape_square/episode_001`

Common settings:

- Original sequential model backend, with an opt-in partial-chunk adapter.
- Exact packaged training scene seeds and frozen visual configuration.
- Existing manipulated-object start-position gate: maximum error <= 0.002 m.
- Paired human-video ICL latent and its detailed text.
- Short task text enabled, target-text CFG 5.0, as in the stopped replay.
  CFG 5.0 is a deployment setting; do not describe guidance itself as training.
- Live front and corrected wrist-camera observations.
- 50 video and 50 action denoising steps.
- No EMA: alpha 1.0. No additional rate limiting or trajectory filter.
- Feed actually commanded actions into action-history tokens, including blended
  commands when temporal ensembling is enabled.
- Model-noise seed = 20261101 + canonical cohort index, unchanged by filtering.
- Full requested horizon = ceil(2 * recorded robot controls), existing
  first-success termination semantics, and at most seven controls of horizon
  overshoot to finish an eight-control execution block.
- No return-home termination in this comparison.

| Arm | Forecast | Execute before observing | Temporal ensemble |
|---|---|---|---|
| A | 16 controls | 8 controls | off; use newest forecast |
| B | 16 controls | 8 controls | on; same-time overlap only |
| C | 32 controls | 8 controls | off; use newest forecast |
| D | 32 controls | 8 controls | on; up to four overlapping forecasts |

Arms C/D extend the forecast to four latent/action frames, within the configured
1–4 training grouping range. This does not establish that four-frame deployment
will be better. Run all four arms to separate forecast length from ensembling.

Output directories:

- `outputs/zero_wam/train_replay_v2b_partial8_raw_full_conditioning_3tasks_2seeds`
- `outputs/zero_wam/train_replay_v2b_partial8_ensemble_full_conditioning_3tasks_2seeds`
- `outputs/zero_wam/train_replay_v2b_partial8_forecast4_raw_full_conditioning_3tasks_2seeds`
- `outputs/zero_wam/train_replay_v2b_partial8_forecast4_ensemble_full_conditioning_3tasks_2seeds`

The completed full-execution ball episode is contextual baseline evidence.
It is not a full six-episode baseline. Paired A/B comparisons are the primary
experiment. Matching seeds does not imply identical noise draws versus the old
16-control execution schedule.

## Temporal ensemble definition

At control boundary t, generate absolute targets for [t, t+16).
Execute only [t, t+8). Retain the forecast tail for [t+8, t+16).
After observing at t+8, generate [t+8, t+24).

For a control time present in both forecasts:

    command = 0.75 * newest_prediction + 0.25 * previous_prediction

Use the newest forecast alone when no earlier forecast covers that exact time.
For four-frame forecasts use normalized weights proportional to `3**(-age)`,
where age is the number of eight-control replans since generation. With four
contributors the newest-to-oldest weights are 27/40, 9/40, 3/40 and 1/40.
With two contributors this is exactly 75/25. Startup includes one observed
frame, leaving three executable forecast groups for four-frame mode; execute
only the first, and retain the other two as forecasts. Never execute the initial
context action frame. Each later four-frame inference predicts 32 controls.
Apply the fixed weights to all six active channels for this first comparison,
including the gripper. Record gripper timing explicitly: blending may blur
closure/release and is a limitation to test, not an assumed improvement.
This is a deliberately recent-weighted overlap ensemble, not a claim of using
ACT's exact weighting convention. Do not average different control times or
blend against the previous executed command.

Record each raw forecast with its absolute control start, selected commands,
contributing forecast ages and weights, and measured states. Both arms must
consume the same-shaped noise draws at the same replanning boundaries.

## Partial-cache design requirements

The current server couples position to
`frame_chunk_size * chunk_idx` and increments `chunk_idx` after committing two
frames. Merely slicing the action output would leave incorrect cache positions
and possibly commit an unexecuted future.

Before coding, trace `_infer_icl`, `_prepare_icl_stream`, `_pack_icl_cfg`,
`_compute_icl_kv_cache`, streaming VAE state, and cache pruning in
`third_party/Zero-WAM/wan_va/wan_va_server.py`. Resolve and document:

- Separate prediction length (two temporal frames) from committed length (one
  temporal frame after startup), using an explicit absolute temporal cursor.
- Startup retains the existing observed first frame and executes only the
  second eight-control group. Commit initial context plus that executed group.
  There is no unused future tail to ensemble at this special first step.
- Subsequently commit one frame of actual observations and eight commands;
  advance the temporal cursor by one, while the next forecast spans two frames.
  In four-frame mode the next forecast spans four frames, with the same commit.
- Remove generated-video prediction-cache entries before replacing them with
  observed history. Never commit the unexecuted action tail.
- Preserve temporal positions, modality/CFG sequence isolation, clean history
  masks, observation-window pruning, and initial zero-action-frame semantics.
  RoPE uses the explicit committed-frame cursor. Attention/window metadata
  retains upstream inference-cycle IDs, so history-window extent in controls
  differs from 16-control execution; account for this in baseline comparisons.
- Feed four new camera observations at two-control intervals after each
  eight-control execution. Preserve each observation's true timestamp.
- Preserve streaming VAE continuity without replaying or duplicating frames.
- Retained ensemble tails are forecast buffers, never executed-history tokens.

Use a separate adapter/module and separate replay option or runner. Reuse
existing scene initialization, conditioning, telemetry, and artifact helpers.
Do not silently change global server configuration to a one-frame forecast.

## Required verification

- [ ] Startup: eight executable controls, correct clean initial action token,
  observation sequence and cache positions.
- [ ] At least three later replans: position advances by one frame, never two;
  each cache update contains only eight actual commands.
- [ ] Distinct sentinel actions in unexecuted tails never enter history.
- [ ] Prediction-cache replacement and window pruning remain correct.
- [ ] VAE consumes each observed frame exactly once in chronological order.
- [ ] Ensemble alignment uses absolute control timestamps; unmatched times
  use only the newest forecast.
- [ ] Ensemble arithmetic and alpha 1.0 produce the specified commands; cached
  actions equal those commands, not raw predictions or measured joint states.
- [ ] Arm A and B have identical startup behavior and matching inference
  schedule/seeds. Divergence begins only when overlap blending becomes possible.
- [ ] Existing sequential, batching, prompt and causal tests remain green.
- [ ] Bounded GPU runs cover startup plus at least three replans for each mode,
  with finite outputs, stable cache growth, and matching telemetry/action/video
  lengths. Store these separately from full-horizon results.
- [ ] Launch full cohorts only after the above checks pass.

## Analysis and acceptance

Report each episode, paired A/B, without equating smoother motion with success:

- Task success, first-success control, final-state success, controls and horizon.
- Closest TCP distance, grasp controls, object lift/displacement, and contact
  evidence where available. TCP-to-object-center distance alone is not grasp
  clearance.
- Video review of approach, closure, transport and release. Distinguish empty
  gripper transport from an actual placement attempt.
- Whether behavior changes after a miss; avoid claiming the model's internal
  awareness from motion alone.
- Predicted, commanded and measured action roughness separately, including
  second differences, reversals, boundary jumps and per-channel behavior.
- Gripper closure/release timing and whether ensembling delays corrections.
- Inference calls, wall time, simulated time, GPU memory and cache occupancy.
  Earlier feedback is expected to increase wall-clock cost; it does not change
  the 30 Hz simulated control rate.

Keep predictions, commands, measured states, video, telemetry, results, immutable
run configuration and cohort manifest. Record checkpoint/input hashes, seeds,
conditioning, prediction/execution lengths, startup convention, ensemble
weights, cache semantics and code revision. Preserve completed episodes across
interruptions and refuse resume with mismatched configuration.

This cohort measures execution on seen scenes. It cannot establish genuine ICL
or semantic prompt causality without controlled conditioning counterfactuals.
Faster feedback and smoother forecasts may improve precision but do not by
themselves train failed-grasp detection or recovery.
