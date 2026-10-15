# Corrected SO-101 training replay: partial execution and temporal ensembling

Date: 2026-10-15. Checkpoint: V2b step 500. Completed: 24/24 autonomous rollouts.

## Findings

All four settings failed all six episodes. Faster feedback with eight-control
execution, longer forecasts, and temporal ensembling did not produce a completed
task on these seen training scenes. They did change the behavior substantially:

- **F4 ensembling reduced arm jitter:** commanded second-difference RMS fell
  23.8% relative to F4 raw, and measured joint second-difference RMS fell 27.9%.
  Commanded jitter fell in all six paired episodes. This did not improve
  approach accuracy consistently; several approaches became worse.
- **F2 ensembling increased movement rather than consistently stabilizing it:**
  mean arm trajectory variance rose 52.1%, TCP path length rose from 2.05 to
  2.48 m, and commanded jitter was essentially unchanged (+1.7%).
- **The only detected pinch was F2 ensemble, block episode 000:** 18 controls
  total, in two nine-control intervals. There was no sustained lift or successful
  transport. This is a small intermediate improvement, not a successful grasp
  and placement.
- **The pushing task failed before meaningful object manipulation in every
  mode.** Both cubes remained effectively stationary. This rules out an
  explanation confined to gripper closure or holding an object.
- **Smoother does not mean more correct.** Even the smoothest arm setting
  remained much more jagged than the recorded oracle commands, and often moved
  less than the oracle or repeated motion without changing the task state.

These are descriptive results for six scenes and one noise seed per scene/mode,
not a statistically powered ranking. They support further investigation of
visual/state grounding, precise control, and closed-loop stability. They do not
establish a single training failure mechanism or evidence of genuine ICL.

## Experimental design and controls

| Name | Forecast after startup | Controls executed before re-observation | Ensemble |
|---|---:|---:|---|
| F2 raw | 16 | 8 | None |
| F2 ensemble | 16 | 8 | Newest 75%, previous 25% when both cover the interval |
| F4 raw | 32 | 8 | None |
| F4 ensemble | 32 | 8 | Up to four forecasts: newest 67.5%, then 22.5%, 7.5%, 2.5% |

Weights are proportional to `3**(-age)` and normalized over forecasts covering
the same absolute control times. Startup discards the action frame aligned with
current observed context: F2 initially has eight future controls and F4 has 24.
Ensembling ramps up as overlapping forecasts become available.

All four arms use packed batch four, V2b step 500, 50 video and 50 action
denoising steps, target text CFG 5.0, human ICL guidance 1.0, the paired human
latent and its detailed text, and **no EMA**. All six channels, including the
gripper, are averaged by the current ensemble; there is no special gripper
transition-preservation rule. That is an implementation limitation worth testing.

The six episodes are 000 and 001 from `ball_between_cans`, `block_in_bowl`, and
`push_cube_into_tape_square`. Canonical model seeds are 20261101–20261106.
Each horizon is twice the recorded oracle control count, rounded up to the
next eight-control boundary. There were 30,304 executed controls in total.
The simulator advances at 30 Hz; observations are collected every two controls
and replanning occurs every eight, approximately 0.267 simulated seconds.
Inference latency pauses wall-clock execution, not simulated time.

Only commanded actions enter history. Cache events show two committed frames
at startup and one thereafter, four fresh observations per commit, and temporal
positions advancing with executed history rather than unexecuted forecasts.
Forecast records were independently reconstructed to verify the ensemble
weights and selected commands for every update in all 24 episodes.

Manifests agree on checkpoint hashes, cohort, episodes, scenes, human-latent
hashes, seeds, conditioning, denoising, horizons and execution cadence.
Identical integer seeds across F2 and F4 do **not** imply identical per-token
noise: forecast tensor shapes differ. Closed-loop observations also diverge
after actions differ.
The runs were launched from an uncommitted working tree based on `e30f25d`;
their recorded Git revision alone is not a complete code identifier. Use the
per-file source hashes in the manifests alongside the subsequently committed
replay implementation.

### Scene correction and sequential reference

The earlier replay directories without the corrected scene protocol used
different distractors, despite matching task-object positions. Commit `7f8ce42`
changed distractor sampling from permutation to weighted ranking. The corrected
reset selects the original sampler only when all recorded object identities,
positions and orientations agree. All 24 runs pass with zero pose error at the
recording's five-decimal precision. The previous report's autonomous exact-scene
claims are explicitly superseded; its recorded-input teacher-forced findings
remain relevant.

The corrected sequential F2/raw ball-000 reference also failed. Batch four was
accepted by the user after reviewing its numerical differences:

- Fixed reference history, eight replans: forecast relative L2 differences
  0.153–0.517%; simulator states match after normalizing telemetry precision.
- First autonomous eight controls: 46 of 48 scalar targets are identical;
  differences are one wrist target (0.614 degrees) and one gripper target
  (0.195 points).
- Autonomous divergence grows: full trajectory range-normalized action RMSE
  8.618%; closest TCP distance 3.978 cm sequential versus 7.075 cm packed.

This is consistent with amplification of small perturbations. It does not prove
the precise numerical source or exclude differences in later cache behavior.
The four-arm comparison uses batch four throughout; it is not a mixture of
sequential and packed outcomes.

## Measurement definitions

The frozen source-hashed measurements are in [measurements.json](measurements.json).
The executable analysis is `scripts/analyze_so101_partial_sweep.py`.

- **Arm variance:** population temporal variance of each commanded arm joint
  divided by its physical actuator range, averaged across five joints. It
  measures motion extent; a stationary arm scores low. It is not stochastic
  prediction variance or a standalone quality measure.
- **Step RMS:** RMS of successive differences in those five normalized joints,
  expressed as percent of actuator range per control.
- **Second-difference RMS:** RMS of `q[t+2] - 2*q[t+1] + q[t]`, with the same
  normalization. This is a discrete jaggedness measure, not physical jerk.
  No factors of 30 or 30² are applied.
- **Commanded versus executed:** commands are the actual selected/blended
  targets; executed joints are simulator measurements after each control.
  `actions.npy` contains the newest unblended forecasts, so using it alone
  would mismeasure the ensemble's actual control behavior.
- **Boundary steps:** transitions into controls with zero-based index divisible
  by eight, versus transitions inside a block.
- **Object measures:** closest TCP-to-object-origin distance, maximum
  displacement from initialization, maximum vertical displacement, and controls
  with the simulator's grasp flag. Distance is not collision distance or proof
  of contact. Displacement alone does not establish movement toward the goal.
- **Grasp flag:** both finger sets must exert sufficient normal force on the
  object with opposing contact normals. A transient flag does not prove a
  load-bearing grasp survived transport.

Aggregate tables give an equal-weight mean of six episode-level metrics; longer
episodes do not dominate. Oracle motion metrics use the original 30-Hz parquet
commands/states. The oracle succeeds over its original horizon, whereas replay
receives twice that horizon, so oracle comparisons are diagnostic references,
not aligned imitation-error scores.

## Motion variance and smoothness

| Setting | Command variance, range² | Command step RMS, % | Command second difference, % | Executed second difference, % | TCP path, m |
|---|---:|---:|---:|---:|---:|
| F2 raw | 0.008770 | 0.2095 | 0.2062 | 0.2766 | 2.052 |
| F2 ensemble | 0.013342 | 0.2436 | 0.2097 | 0.2799 | 2.483 |
| F4 raw | 0.016468 | 0.2460 | 0.2504 | 0.3371 | 2.557 |
| F4 ensemble | 0.017237 | 0.2099 | 0.1908 | 0.2431 | 2.138 |
| Recorded oracle | 0.024749 | 0.2920 | 0.0249 | 0.0560 | — |

F4 raw has the highest arm jitter and longest average TCP path. F4 ensemble
reduces its commanded jitter in 6/6 scenes and its average TCP travel by 16.4%.
Its mean variance nevertheless increases 4.7%; four scenes have lower variance
and two have higher variance. Smoother local motion can coexist with broader
excursions.

F2 ensemble raises variance in all six scenes. Its jitter decreases in only
three, leaving the aggregate nearly unchanged. Thus the two-frame version
does not reliably make the autonomous trajectory smoother.

Compared with the oracle, replay commands have lower average step size but
roughly 7.7–10.1 times greater second-difference RMS (mean of per-episode
ratios). The measured joints also remain substantially more jagged. The
difference is not merely dilution by the replay's extra horizon: command
second-difference RMS over only the first oracle-length interval is
0.2175%, 0.2036%, 0.2685%, 0.1960% for the four modes, respectively.

### Does the ensemble itself smooth its inputs?

Yes. Comparing selected commands with the newest raw forecast **within each
ensemble rollout's own observation history**, F2 reduces arm second-difference
RMS to 79.1–80.7% of raw; F4 reduces it to 70.2–75.3%.
This confirms the averaging operation works. The raw forecasts in this
comparison are not the separate raw-mode rollout: once the controller changes
actions, future observations change too. That feedback explains why local
smoothing does not guarantee a smoother or more successful autonomous run.

### Are replan boundaries causing most of the jitter?

| Setting | Boundary command step RMS, % | Within-block command step RMS, % |
|---|---:|---:|
| F2 raw | 0.2020 | 0.2103 |
| F2 ensemble | 0.2277 | 0.2457 |
| F4 raw | 0.2275 | 0.2484 |
| F4 ensemble | 0.1941 | 0.2119 |

Boundary steps are not larger on average. Jitter is not confined to chunk
stitching, so smoothing only the joins is unlikely to address the whole problem.
These means do not rule out isolated bad boundary events.

### Gripper and tracking exceptions

Commanded gripper second-difference RMS averages 0.181, 0.154, 0.203 and 0.477
points on the 0–100 scale. The F4-ensemble average is dominated by push-000:
at control 885 its command jumps from 9.61 to 38.83, then falls to 14.57 at
886. Measured gripper state goes approximately 9.84 → 11.27 → 14.01 instead.
The actuator filters that brief command spike; arm smoothing statistics would
miss it. F4-ensemble push-000 gripper command second-difference RMS is 2.042,
while the other five episodes are about 0.124–0.179.

Mean arm command-to-state absolute error is 0.210, 0.153, 1.142 and 0.691
degrees. There are larger localized discrepancies: F4 raw block-001 has
3.019-degree mean error and 15.531-degree absolute-error p95; F4 ensemble
block-000 has 2.413 and 14.345 degrees. These warrant checking contacts, limits,
and feasibility. The current telemetry does not identify which of these caused
the discrepancy, so it would be premature to call it a controller defect.

## Correctness and object interaction

No episode reaches task success, either at any time or in the final state.
The following tables preserve scene-level differences hidden by that total.
Values are centimeters; entries below 0.001 cm are shown as approximately zero.

### Closest TCP approach

| Episode | F2 raw | F2 ensemble | F4 raw | F4 ensemble |
|---|---:|---:|---:|---:|
| Ball 000 | 7.075 | 6.753 | 3.571 | 6.063 |
| Ball 001 | 2.522 | 12.682 | 3.029 | 5.573 |
| Block 000 | 1.594 | 0.675 | 0.896 | 2.168 |
| Block 001 | 8.981 | 6.456 | 5.268 | 16.302 |
| Push 000 | 7.296 | 6.171 | 4.452 | 7.237 |
| Push 001 | 12.586 | 9.801 | 12.760 | 12.738 |

F2 ensemble improves this distance in five scenes but dramatically worsens
ball-001. F4 ensemble worsens it in five scenes relative to F4 raw; the tiny
improvement on push-001 has no manipulation benefit. Closest distance alone
cannot establish correct wrist orientation, finger alignment, or target
identification.

### Maximum object displacement / maximum lift

| Episode | F2 raw | F2 ensemble | F4 raw | F4 ensemble |
|---|---:|---:|---:|---:|
| Ball 000 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 |
| Ball 001 | 4.895 / 0.208 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 |
| Block 000 | 2.881 / 0.658 | 2.786 / 0.662 | 3.341 / 0.726 | 1.202 / 0.334 |
| Block 001 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 |
| Push 000 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 |
| Push 001 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 |

**Ball 000:** all modes leave the ball stationary. F4 raw gets closest, but
does not secure a pinch. The earlier observation that the fingers appear to
touch the ball is understandable visually, but the saved telemetry establishes
neither a grasp nor consequential movement. Repeated near approaches should
not be described as confirmed physical recovery attempts.

**Ball 001:** only F2 raw produces substantial movement (4.895 cm), without a
pinch or meaningful lift. The F2 ensemble trajectory instead stays much farther
away. This is a concrete example of averaging changing the attempted behavior
unfavorably, despite a small local smoothing benefit.

**Block 000:** every mode moves the object somewhat. F2 ensemble registers
opposing-finger contact at controls 97–105 and 114–122 (about 3.23–3.50 and
3.80–4.07 seconds): two 0.3-second pinch intervals, 0.6 seconds total.
Its maximum lift is only 6.62 mm, comparable to other block-000 runs.
There is no evidence of successful carry or placement. F4 raw moves the block
most but never registers a pinch; F4 ensemble reduces both movement and lift.
Nudging/tipping and repeatedly approaching the object are not equivalent to
progress through pick → lift → carry → place.

**Block 001:** no mode moves the object meaningfully. F4 raw comes closest;
F4 ensemble stays at least 16.3 cm from the object origin. The two scenes
therefore expose very different behavior even within the same task.

**Push 000 and 001:** the target cube does not move meaningfully under any
setting. Sampled front/wrist images show arm/wrist repositioning around the
scene without a productive push into the marked square. F2 ensemble roughly
doubles TCP travel relative to F2 raw (0.804 → 1.623 m and 1.545 → 2.992 m),
yet does not move either cube. Low variance in the other pushing runs reflects
limited motion, not accurate execution.

### What the visual evidence can and cannot establish

Review sheets show every mode at one-quarter oracle time, closest approach,
oracle horizon, and full replay horizon. The closest-approach column is
event-aligned and therefore uses different timestamps across modes. Other
columns use matched control horizons for that scene. Frames are sampled at
15 Hz, using the latest frame at or before the requested control.

- [Ball 000](assets/ball_between_cans__episode_000.jpg)
- [Ball 001](assets/ball_between_cans__episode_001.jpg)
- [Block 000](assets/block_in_bowl__episode_000.jpg)
- [Block 001](assets/block_in_bowl__episode_001.jpg)
- [Push 000](assets/push_cube_into_tape_square__episode_000.jpg)
- [Push 001](assets/push_cube_into_tape_square__episode_001.jpg)

The measurements and sampled images support approach errors, intermittent
object interaction, and inadequate sustained manipulation. They do not prove
whether the policy identifies the wrong object, misestimates pose, fails to
recognize the miss, or lacks a recovery policy. Repeated motion alone cannot
distinguish deliberate retry from an oscillation or a learned repeated sequence.
Object xyz trajectories, goal-distance trajectories, finger geometry and
per-geom contact forces were not logged, so direction-of-progress and exact
contact classification are not reconstructed here.

## Cost and validation

Median packed inference time is 14.41 s (F2 raw), 14.43 s (F2 ensemble),
20.35 s (F4 raw), and 20.19 s (F4 ensemble). F4 is about 40–41% slower by this
measure, without a success benefit. These are per-packed-call times across
batches with changing active-slot counts; they exclude stepping and cache
updates and are not a uniform four-active-slot throughput benchmark.
Ensembling adds little inference cost at the same forecast size.

F2 raw contains a 5,366.42-second logged inference interval contaminated by
SIGSTOP during the reference review. Its mean inference time and wall-clock
duration must not be presented as compute cost. The frozen runtime extracts
retain that interval; medians are used rather than silently deleting it.
The continuation supervisor initially stopped before advancing; after restart,
the other arms completed at 04:42, 07:32 and 10:22 UTC on October 15.

Validation:

- 95 focused tests passed across partial execution, static batch/cache,
  prompt-sweep and causal-action utilities.
- All 24 artifact sets pass: action rows = telemetry rows = controls;
  video frames = controls/2 + 1, with finite actions.
- Analysis checks every forecast's selected-command reconstruction, source
  manifest hash, episode seed, full horizon, recorded scene and per-slot cache
  commit sequence.
- Small six-scene cohort, no independent repeats: no confidence intervals,
  significance claims or generalization claims are warranted.
- There is no corrected full-cohort 16-execute baseline in this comparison, so
  this sweep does not isolate the benefit of eight-control execution itself.
- All modes retain the same prompts. This is a control-policy comparison,
  not a causal test of text/video following or in-context learning.

## Recommended next experiments

1. **Prove that the complete recorded robot trajectory replays successfully
   through this exact simulator/control path.** Replay the parquet commands on
   these corrected scenes, without the learned model, recording measured joints,
   object poses, contacts and success. Compare with the recorded robot video and
   state trajectory. Low teacher-forced loss plus matching initial poses does
   not validate the entire actuator/physics path. If oracle replay fails, fix
   that before interpreting policy failures as training defects.

2. **Test recovery from phase-specific oracle states.** Warm the model's
   observation/action cache with real oracle prefixes, then hand over near
   approach, just before closure, after a verified grasp, and before placement.
   Compare unperturbed handovers with small measured pose perturbations.
   This can separate inability to perform a phase from failure to reach it or
   to recover after a miss. Use the corrected scene protocol and record contacts.

3. **Make a very small training memorization test decisive.** Train/overfit one
   or a few exact episodes and measure closed-loop success alongside local
   action loss. Validate complete episode coverage, especially contact/lift
   phases, using the exact deployed history convention. If even that cannot
   succeed after oracle replay is validated, prioritize representation, timing,
   objective and data alignment over scaling the dataset or controller sweep.

4. **Investigate control feasibility and contact precision.** Log commanded
   versus achieved joints, actuator limits/forces, TCP orientation, finger
   locations, object xyz, goal distance and bilateral-contact forces. Start
   with F2-ensemble block-000's brief pinch and the F4 runs with large tracking
   errors. Evaluate gripper handling separately from arm averaging; avoid
   averaging away useful closures or importing old forecasts that assume a
   successful grasp.

5. **Train and evaluate recovery explicitly if the control path checks out.**
   Add demonstrations or oracle corrections for failed grasps, displaced
   objects and missed pushes; use perturbed states and on-policy errors.
   Evaluate sustained grasp and progress toward the actual goal, not only
   visual approach or action MSE. Do not conclude that more denoising or a longer
   forecast solves this: F4 already costs more and showed no completed tasks.

6. **Return to the original causal prompt experiment after establishing a
   competent execution baseline.** Cross correct, absent and mismatched text
   and human-video conditions on frozen scenes/noise schedules. Distinguish
   prompt sensitivity from semantically appropriate action and from novel
   task adaptation. The current all-zero success sweep cannot establish ICL.

For immediate diagnosis, retain F2 raw as the simplest baseline and F2 ensemble
as the interesting brief-pinch case. Retain F4 ensemble as a smoothness control,
not as a recommended task-performing controller.

## Artifacts and reproduction

All authoritative rollout directories are under `outputs/zero_wam/`:

```text
train_replay_v2b_partial8_raw_full_conditioning_3tasks_2seeds_scenev2_b4
train_replay_v2b_partial8_ensemble_full_conditioning_3tasks_2seeds_scenev2_b4
train_replay_v2b_partial8_forecast4_raw_full_conditioning_3tasks_2seeds_scenev2_b4
train_replay_v2b_partial8_forecast4_ensemble_full_conditioning_3tasks_2seeds_scenev2_b4
```

Each contains its manifest and six episode directories with video, raw
predictions, selected commands/telemetry, overlapping forecasts and results.
HF keeps these same names beneath:
`hf://buckets/akoniti/ICL-so101/zero_wam_evaluation_results/2026-10-14/`.
This report, measurements, runtime extracts and review sheets are synchronized
under `partial_chunk_replay_analysis` at that prefix.

To recompute on the original local artifacts:

```bash
.venv-zero-wam/bin/python scripts/analyze_so101_partial_sweep.py
.venv-zero-wam/bin/python scripts/render_so101_partial_sweep.py
```

The analysis requires the original `/tmp/so101_partial_scenev2_b4_f{2,4}_{raw,ensemble}.log`
files for timing extraction. Frozen filtered runtime logs are included alongside
the report for review after those temporary originals expire. Rendering reads
the frozen measurements and does not rerun inference or recalculate metrics.
