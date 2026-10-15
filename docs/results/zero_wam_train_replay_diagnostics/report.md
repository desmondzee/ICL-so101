# Zero-WAM SO-101 training-replay diagnosis

> **Correction: original autonomous replays did not reproduce the full training
> scenes.** Investigation after visual review found that commit `7f8ce42`
> changed distractor sampling from `ValEnv`'s permutation to `TrainEnv`'s
> weighted-ranking procedure. Reusing the seed matched manipulated objects but
> changed distractor identities/poses in all six episodes. The old start-position
> gate checked only manipulated objects. Consequently, statements below that
> describe these autonomous rollouts as exact-scene replays, rule out scene
> mismatch, or attribute their failure primarily to distribution drift are
> superseded. The teacher-forced measurements use recorded inputs and are not
> invalidated by this simulator reset issue.
>
> The corrected replay selects a sampler only if every active object's recorded
> identity, xyz and quaternion match. All six episodes reproduce their rounded
> recorded poses with zero numerical error using the legacy sampler, with
> matching visual-configuration hashes. Front/wrist comparison sheets confirm
> the corrected layouts; recorded MP4 frames remain subject to compression and
> first-frame timing differences. Evidence:
> `outputs/zero_wam/train_replay_scene_audit/audit.json` and adjacent JPGs.
> Corrected autonomous outputs use the `_scenev2` suffix and must be evaluated
> before making a new causal diagnosis of task failure.
>
> **Completed follow-up (2026-10-15):** the corrected four-arm partial-execution
> sweep finished 24/24 rollouts, with 0/24 successes. Temporal ensembling changes
> motion smoothness and object interaction but does not yield completed tasks.
> See [the detailed corrected-scene report](../zero_wam_partial_chunk_replay/report.md)
> for scene-level results, limitations and the recommended oracle replay check.

Date: 2026-10-14

## Executive conclusion

V2b step 500 fails autonomous closed-loop execution on all six tested episodes
drawn from its own training set, including two simple pushing episodes. This is
not explained by a new scene, a held-out task, batch-4 inference, EMA smoothing,
an unexposed sample, the base model failing to acquire SO-101 actions, or any of
the 15 machine-checkable training/inference configuration fields audited here.

At the same time, the checkpoint predicts recorded actions accurately when it is
given the recorded robot history. Across 30 phase-spanning, teacher-forced
boundaries, its next-16-control MAE is 0.429 versus 2.947 for persistence. It
beats persistence at 25 of 30 boundaries. The base checkpoint's first-boundary
MAE is 12.986, versus 0.296 after SO-101 post-training.

The evidence therefore supports a narrower diagnosis: post-training acquired
local action-continuation capability, but not a robust autonomous policy. Errors
in the first few generated chunks move the robot off the demonstration
distribution; action and state errors then grow rapidly, and the policy does not
recover. This is consistent with behavior-cloning covariate shift and limited
closed-loop robustness. It does not prove that covariate shift is the only
cause: cold-start generation, stochastic chunk-to-chunk drift, and insufficient
optimization remain plausible contributors.

## Questions and outcomes

| Question | Result |
|---|---|
| Does V2b succeed on exact seen tasks and scenes? | No: 0/6 batch-4 replays. |
| Is batch-4 numerical drift the sole cause? | No: the original sequential path also fails the exact push episode. |
| Is EMA smoothing the sole cause? | No: sequential raw execution with alpha 1.0 also fails. |
| Were scene initialization and task identity exact? | Yes for the frozen metadata checked: seed and visual configuration match, and manipulated-object start-position error is 0.0 m in all six episodes. |
| Did the checkpoint see these samples? | Yes: 7-28 draws per episode; all 1,109 training episodes were seen by step 500. |
| Did post-training learn SO-101 action prediction? | Yes: V2b teacher-forced MAE is far below both base initialization and persistence. |
| Does adding short target text repair prediction? | No measurable benefit at the first boundary: 0.2958 MAE at CFG -1 versus 0.2970 at CFG 5. |
| Is action prediction good throughout recorded episodes? | Usually, but not uniformly: V2b beats persistence at 25/30 boundaries and degrades around the middle manipulation phases. |
| Is a basic camera/action/normalization mismatch evident? | No: all 15 audited fields pass. |

## Evaluated cohort

The cohort deliberately spans three skills, with two exact training episodes per
task:

| Task | Episodes | Skill |
|---|---|---|
| `ball_between_cans` | 000, 001 | grasp, transport, release |
| `block_in_bowl` | 000, 001 | grasp and container insertion |
| `push_cube_into_tape_square` | 000, 001 | pushing without a required grasp |

Each replay uses the packaged training seed, frozen `VisualConfig`, paired human
latent and detailed text, V2b step 500, 50 video and 50 action denoising steps,
commanded-action history, and a horizon of `ceil(2 * robot_frames)`. The
start-pose gate compares the live manipulated-object position to
`object_poses_start` and aborts above 2 mm. All six errors are exactly 0.0 m.

This establishes equality for the metadata-controlled simulator state. It is not
a pixel-for-pixel proof that every rendered frame is identical to the original
encoded MP4, although the same renderer configuration and corrected wrist
transform are used.

## Closed-loop results

### Batch-4, EMA alpha 0.9

| Episode | Success | Controls | Closest TCP | Object displacement | Grasp controls |
|---|---:|---:|---:|---:|---:|
| ball 000 | no | 1,112 | 7.12 cm | ~0 | 0 |
| ball 001 | no | 1,352 | 5.10 cm | ~0 | 0 |
| block 000 | no | 1,160 | 2.49 cm | 1.41 cm | 0 |
| block 001 | no | 1,288 | 11.05 cm | ~0 | 0 |
| push 000 | no | 1,016 | 5.31 cm | ~0 | 0 |
| push 001 | no | 1,672 | 5.54 cm | ~0 | 0 |

The videos show mostly reaching, hovering, and wandering. `block_in_bowl/000`
approaches and nudges the block but never grasps it. The two ball episodes never
move the ball. Unexpectedly, neither batch-4 pushing episode materially moves
the cube even though pushing does not require gripper timing.

### Sequential raw control

`push_cube_into_tape_square/episode_000` was repeated through the original
single-episode `VA_Server` path, without packed inference and with EMA disabled:

| Metric | Result |
|---|---:|
| Success | no |
| Controls / requested horizon | 1,016 / 1,014 |
| Chunks | 64 |
| Cube displacement | 2.12 cm |
| Closest TCP distance | 3.85 cm |
| Mean chunk inference | 10.31 s |
| Normalized second-difference RMS | 0.00226 |

Sequential execution produces somewhat more useful contact than batch 4, so the
known packed-versus-sequential numerical drift can change behavior. It does not
change the task-level conclusion: raw sequential inference still fails on the
exact seen scene over a full horizon.

## Where closed-loop behavior diverges

Time-aligned comparison against the recorded oracle shows that the generated
trajectory begins relatively close and then diverges quickly:

| Episode | First-8 action MAE | First-64 action MAE | Full overlap action MAE |
|---|---:|---:|---:|
| ball 000 | 1.664 | 7.189 | 18.785 |
| ball 001 | 0.895 | 6.726 | 42.763 |
| block 000 | 1.044 | 2.548 | 7.265 |
| block 001 | 0.549 | 3.591 | 22.399 |
| push 000 | 0.608 | 2.174 | 17.889 |
| push 001 | 0.611 | 3.407 | 20.668 |

These batch-4 comparisons are time-aligned rather than dynamic-time-warped, so
some later error reflects phase lag as well as incorrect motion. The early trend
is still informative: error increases several-fold within 64 controls, before
most demonstrations reach manipulation.

The sequential push replay shows the same pattern with a different model-noise
seed: action MAE is 0.628 over controls 0-7, 0.962 over 0-15, 2.958 over 0-31,
and 4.425 over 0-63. Thus the divergence is not introduced only by packing or
EMA.

There is also a meaningful gripper-distribution symptom. In the recorded push
episodes the gripper remains approximately 0.24-0.91, while the batch-4 policy
commands ranges up to 25.0 and 33.6 despite the task requiring a push. This is
consistent with a policy leaving the demonstrated phase/task trajectory after
its early errors; it is not evidence of a unit mismatch because normalization
audits pass and teacher-forced gripper prediction is accurate.

## Teacher-forced causal action prediction

### First deployment boundary

The leakage-safe protocol caches recorded observations and the first eight
executed controls, then generates controls 8-23 from seeded noise. Future
actions enter only the scorer.

| Model / conditioning | MAE | RMSE | Persistence MAE | Persistence RMSE |
|---|---:|---:|---:|---:|
| V2b, CFG -1 | **0.2958** | **0.4373** | 0.6835 | 1.2145 |
| V2b, CFG 5 | **0.2970** | **0.4342** | 0.6835 | 1.2145 |
| Base initialization, CFG -1 | 12.9857 | 20.9874 | 0.6835 | 1.2145 |

Prefix and target hashes are identical across the three runs. The large
base-to-V2b improvement proves that the selected checkpoint contains learned
SO-101 action capability. The negligible CFG difference indicates that adding
the short instruction does not repair the failure in this cohort; the paired
human ICL bundle and its detailed text remain present in both V2b conditions.

### Phase-spanning boundaries

For each episode, five boundaries are chosen near 10%, 30%, 50%, 70%, and 90%
of its recorded trajectory and snapped to deployment boundaries
`8 + 16*k`. Each boundary starts from a fresh reset and caches the complete
recorded history through that boundary before predicting the next 16 controls.
There are 30 independent predictions.

| Boundary fraction | V2b MAE | Persistence MAE |
|---:|---:|---:|
| 0.1 | 0.275 | 4.439 |
| 0.3 | **0.275** | 0.567 |
| 0.5 | 0.632 | 2.854 |
| 0.7 | 0.625 | 1.616 |
| 0.9 | 0.340 | 5.258 |
| Overall | **0.429** | 2.947 |

By task, model MAE is 0.461 for ball placement, 0.370 for block insertion, and
0.458 for pushing. By channel:

| Channel | V2b MAE | Persistence MAE |
|---|---:|---:|
| shoulder pan | 0.299 | 3.160 |
| shoulder lift | 0.449 | 4.770 |
| elbow flex | 0.607 | 4.677 |
| wrist flex | 0.406 | 1.562 |
| wrist roll | 0.682 | 2.914 |
| gripper | **0.134** | 0.597 |

V2b beats persistence at 25 of 30 boundaries, not all 30. Its worst errors
cluster around the dynamic middle phases: ball episode 001 at boundary 328 has
MAE 1.616, while push episode 000 and block episode 000 also have their largest
errors in the 50-70% region. Five relatively static boundaries favor
persistence, including ball episode 001 at boundaries 200 and 328.

This changes the interpretation from “the model only learned the initial
reach.” With correct history, it predicts every broad phase substantially better
than a global persistence baseline on average, including retreat. Accuracy is
weakest around grasp/transport/placement transitions, where closed-loop errors
are also most consequential.

## Training and exposure evidence

At optimizer step 500:

- all 1,109 training episodes had been drawn;
- total draws were 16,000;
- mean exposure was 14.43, median 11, range 1-286;
- cohort exposures were 7, 13, 11, 14, 28, and 25 draws.

The deterministic validation metrics changed as follows:

| Metric | Step 0 | Step 500 | Reduction |
|---|---:|---:|---:|
| action loss | 0.008772 | 0.002408 | 72.5% |
| video loss | 0.185875 | 0.158089 | 14.9% |

The exposure and loss evidence rules out “these files were never sampled” and
“the checkpoint remained equivalent to initialization.” It does not establish
that 7-28 exposures are enough to learn recovery around each trajectory.

## Configuration audit

All 15 machine-checkable checks pass:

- front then wrist camera order;
- 224x288 inference resize from recorded 640x480 images;
- two latent frames and eight actions per latent frame;
- 30-dimensional action space with active channels `[0,1,2,3,4,28]`;
- the packaged physical-bound normalization statistics;
- corrected wrist near plane of 0.02425 m;
- human latent width 448;
- short target text absent at CFG -1 and detailed human text present;
- commanded-action cache history.

The checkpoint loads exactly, with no missing, unexpected, or mismatched keys.
These results substantially reduce the likelihood of a gross camera, action,
normalization, checkpoint, or conditioning setup error. Remaining pixel-level,
VAE-state, and cold-start semantic mismatches are not mathematically excluded.

## Diagnosis

### Best-supported mechanism: autonomous distribution drift

Three observations jointly support this mechanism:

1. next-chunk prediction is accurate with recorded history throughout the
   episode;
2. autonomous action error grows rapidly over the first 32-64 controls; and
3. once displaced, the policy wanders, emits phase-inappropriate commands, and
   does not recover.

The training objective is teacher-forced imitation. It rewards predicting local
continuations on the demonstration manifold but does not directly train
recovery from model-induced states. Diffusion noise augmentation is not the
same as state-distribution or observation-history augmentation.

### Contributing factors

- **Transition sensitivity.** Teacher-forced errors approximately double at
  the 50-70% boundaries, which often cover grasp, transport, release, or the
  central push.
- **Cold-start and early accumulation.** Initial action errors are modest but
  exceed later teacher-forced first-boundary error, and trajectory error grows
  within a few chunks.
- **Open-loop chunk execution.** Sixteen controls are executed before
  replanning. There is no overlapping temporal ensemble or velocity/acceleration
  constraint.
- **Limited checkpoint maturity.** Step 500 represents 16,000 draws across
  1,109 episodes. Local imitation has improved strongly, but recovery and
  transition precision may need more optimization or different data.
- **Packed numerical drift.** Batch 4 changes realized behavior and should not
  be used to claim bitwise equivalence. The sequential failure proves it is not
  the root task-level explanation.

### Explanations weakened by evidence

- held-out-task generalization failure;
- unseen scene initialization;
- EMA over-smoothing;
- batch-4 inference alone;
- absent training exposure;
- wrong checkpoint or unchanged base model;
- gross camera order, action channel, normalization, wrist transform, or ICL
  width mismatch;
- lack of short-task text at inference.

## Recommended next experiments

1. **Cold-start causal grid.** Score the actual first executable eight controls
   over multiple fixed noise seeds, then score boundaries 8, 24, 40, and 56.
   This separates initial-generation weakness from later state drift.
2. **Closed-loop recovery augmentation.** Collect DAgger-style corrective
   trajectories from states reached by V2b, or perturb demonstration states and
   train recovery actions. This directly targets the best-supported failure.
3. **Replan every eight controls.** Test half-chunk execution with correct
   partial-cache semantics. It halves the open-loop interval; compare against
   the same scenes and noise seeds.
4. **Transition-weighted evaluation and training.** Use oracle event metadata to
   oversample pregrasp, contact, release, and central-push boundaries, where
   teacher-forced error is highest.
5. **Longer-checkpoint comparison.** Evaluate a later checkpoint with the same
   frozen replay and multi-boundary manifests. Do not infer that more steps help
   without this controlled comparison.
6. **Sequential scientific baseline.** Keep packed inference for throughput,
   but use sequential results for acceptance until a task-level equivalence
   tolerance is established.

## Artifacts

- Batch-4 replay:
  `outputs/zero_wam/train_replay_v2b_b4_alpha090_3tasks_2seeds/`
- Sequential raw replay:
  `outputs/zero_wam/train_replay_diagnostics/sequential_raw_v2b/`
- First-boundary causal reports:
  `outputs/zero_wam/train_replay_diagnostics/chunk_causal/`
- Multi-boundary causal report:
  `outputs/zero_wam/train_replay_diagnostics/multi_boundary_v2b_cfgneg1.json`
- Exposure audit:
  `outputs/zero_wam/train_replay_diagnostics/exposure_audit.json`
- Configuration audit:
  `outputs/zero_wam/train_replay_diagnostics/config_audit.json`
- Frozen probe manifests and selected training inputs:
  `data/hf_bucket_ICL-so101/train_replay_probe/`

## Limitations

- Six episodes across three of 61 training tasks are a diagnostic cohort, not a
  training-set success estimate.
- Only one episode was rerun sequentially without smoothing.
- Time-aligned autonomous/oracle MAE conflates wrong actions with phase drift;
  it is used to locate divergence, not as a standalone policy score.
- Teacher-forced next-chunk accuracy does not test autonomous observation
  histories or recovery.
- Batch-4 and sequential predictions are numerically close but not identical,
  and their task behavior can differ.
- No later checkpoint was available in this local analysis.
