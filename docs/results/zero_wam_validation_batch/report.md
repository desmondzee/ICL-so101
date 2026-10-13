# SO-101 held-out validation batch

## Status

V2b step 500 completed **25/25 bounded rollouts**: all five `sim_val_v2`
tasks and five deterministic dataset episodes per task. No episode succeeded or
grasped a target during the evaluated prefix.

This run deliberately evaluates only the first nominal 64 controls to make the
requested five-task, five-seed diagnostic tractable. Because the existing
closed-loop executor finishes the current generated chunk, every episode recorded
72 controls. These are early-behavior and failure-mode results, not full-horizon
task-success estimates.

## Protocol

| Item | Value |
|---|---|
| Checkpoint | V2b step 500 |
| Checkpoint config SHA256 | `f30afc3dc6393199678e485580151eeb932cc8d8c7223b0f988c2651fe7a7d52` |
| Tasks | `mug_on_plate`, `mugs_in_microwave`, `pan_on_stove`, `sort_blocks`, `stack_bowls` |
| Episodes | First five deterministic index entries per task; 25 total |
| Conditioning | Paired generated-human latent and its detailed text; short target text disabled |
| ICL guidance | 5 |
| Denoising | 50 video steps, 50 action steps |
| Controls | Requested 64; recorded 72 due to final-chunk completion |
| Feedback | Closed-loop stride-2 observations and executed joint targets |
| Errors | 0 |

## Aggregate Results

| Task | Episodes | Success | Target grasp | Normalized action delta RMS | Normalized second-difference RMS | Direction reversal fraction |
|---|---:|---:|---:|---:|---:|---:|
| Mug on plate | 5 | 0 | 0 | 0.00620 | 0.00338 | 0.063 |
| Mugs in microwave | 5 | 0 | 0 | 0.00617 | 0.00345 | 0.091 |
| Pan on stove | 5 | 0 | 0 | 0.00524 | 0.00323 | 0.075 |
| Sort blocks | 5 | 0 | 0 | 0.00602 | 0.00344 | 0.065 |
| Stack bowls | 5 | 0 | 0 | 0.00563 | 0.00339 | 0.084 |

All native task predicates remained false. All five mug-on-plate episodes kept
the mug upright; all stack-bowls episodes kept distractors undisturbed.

## Target Approach

| Task | Closest target TCP distance across five episodes |
|---|---|
| Mug on plate | 7.6-11.9 cm to the red mug |
| Mugs in microwave | 9.6-16.4 cm to the red mug; 10.2-16.3 cm to the white mug |
| Pan on stove | 7.3-16.1 cm to the pan |
| Sort blocks | 8.6-13.5 cm to the red block; 9.8-18.3 cm to the blue block |
| Stack bowls | 7.6-12.5 cm to the white bowl |

The TCP moved 23.5-34.4 cm per rollout, but net displacement was only
6.2-20.2 cm. Path/net ratios ranged from 1.6 to 5.3, showing curved or
backtracking motion rather than a direct reach.

## Behavioral Analysis

### Shaking and oscillation

- There is **no strong aggregate evidence of severe high-frequency shaking** in
  this prefix. Normalized second-difference RMS is tightly grouped at
  `0.00303-0.00387`, and mean reversal fractions are `6.3%-9.1%`.
- The clearest oscillatory outlier is `mugs_in_microwave/episode_002`, with a
  `15.0%` direction-reversal fraction and a path/net ratio of `5.3`.
- Sparse rollout video records every two controls, so sub-frame vibration can be
  underestimated. The action telemetry is the stronger shaking diagnostic.
- Joint/gripper saturation is absent in most episodes, but appears for portions
  of several runs: microwave episode 004 (`30.6%`), pan episodes 001/004
  (`26.4%`), sort-blocks episode 002 (`29.2%`), and stack-bowls episode 003
  (`25.0%`). Saturation may contribute to stalled or repetitive motion.

### Target identification

- The model shows partial visual target localization. Pan rollouts visibly bring
  the pan into the wrist-camera center, mug-on-plate rollouts focus on the red
  mug/plate area, and stack-bowls rollouts move toward the bowl pair.
- Localization is not precise enough for a grasp in this prefix: no target gets
  closer than 7.3 cm to the TCP, and there are zero grasp controls across all 25
  episodes.
- `mugs_in_microwave` is the least selective visually. Wrist views alternate
  among the two mugs and nearby packaged distractors; both mug distances often
  remain above 10 cm. The model does not demonstrate a committed first-mug
  selection.
- Sort-block rollouts generally move into the blocks/plates workspace but do not
  settle over either small block. Four of five are closer to red than blue; one
  favors blue. This is compatible with coarse color/object localization but not
  reliable target acquisition.

### Gripper behavior

- The commanded gripper range starts near `0.4-1.6` and rises only to
  approximately `13-29` during these prefixes. There is no completed close-lift
  sequence and no simulator-confirmed grasp.
- The failure is therefore upstream of placement/release: the run does not yet
  test whether the model can transport or place an acquired target.

## Rollouts And Data

- [Batch summary](../../../outputs/zero_wam/validation_batch_v2b_5x5_c64/summary.json)
- [Selected episodes](../../../outputs/zero_wam/validation_batch_v2b_5x5_c64/episodes.json)
- Per-episode directories under
  `outputs/zero_wam/validation_batch_v2b_5x5_c64/v2b_step500/` contain
  `rollout.mp4`, `actions.npy`, `telemetry.jsonl`, and `result.json`.

## Conclusions

- The 25 prefixes provide no task-success or grasp evidence.
- Motion is generally active and target-region directed rather than frozen, and
  severe shaking is not the dominant failure mode.
- The dominant observed failure is **coarse approach without precise target
  acquisition or a grasp attempt**. Microwave scenes additionally show target
  ambiguity among two mugs and distractors.
- Full oracle-relative horizons are still required before concluding that the
  policy cannot eventually grasp or complete these tasks. The current bounded
  run should be treated as a broad diagnostic screen.
