# SO-101 chunk-causal action validation

## Protocol

This report replaces the earlier teacher-forced endpoint reconstruction as the
primary offline action metric. Each of the 50 held-out episodes contributes one
first-steady-state deployment boundary.

- The full paired human ICL demonstration is visible.
- Robot observations and the eight actions executed before the boundary are cached.
- No robot observation after the boundary enters inference.
- The next 16 action tokens start from seeded Gaussian noise.
- Those 16 tokens are denoised jointly for 50 steps using the deployment inference
  and cache path; bidirectional attention within the noisy chunk is allowed.
- The 16 demonstration targets enter only the scorer after generation.
- A deployment-executable baseline repeats the final executed prefix action for all
  16 controls.

Manifest SHA256:
`87fe453cc14a6d5f40085a6f234863ccb6c3d39ccb0f5aa003c76a9a296da937`.

Cross-run checks confirmed identical target hashes, prefix hashes, persistence
metrics, sample ordering, boundaries, and noise seeds for all four checkpoints.

## Overall results

Physical MAE averages all six active action channels after computing each episode's
metric. Joint channels are in degrees and the gripper is in `[0, 100]`.

| Checkpoint | Mean MAE | Median | q05 | q95 | Mean RMSE |
|---|---:|---:|---:|---:|---:|
| Zero-WAM pretrain initialization | 12.1911 | 12.1519 | 9.2554 | 15.7102 | 19.2962 |
| V0 step 500 (`1e-4`, batch 16) | 1.0829 | 1.0753 | 0.6939 | 1.4496 | 1.6625 |
| V1 step 500 (`1e-5`, batch 16) | 0.6750 | 0.6561 | 0.4537 | 0.9769 | 1.0292 |
| V2b step 500 (`1e-5`, batch 32, cosine) | **0.6720** | 0.6667 | 0.4561 | 0.9265 | **1.0252** |
| 16-control persistence baseline | 1.4205 | - | - | - | - |

V0, V1, and V2b all beat persistence. V1 and V2b are effectively tied overall:
the paired V2b-minus-V1 mean MAE difference is `-0.00306`, with a deterministic
100,000-resample episode bootstrap 95% interval of `[-0.02212, 0.01553]`. V2b is
better on 23 of 50 episodes.

## Per-channel MAE

| Channel | Initialization | V0 step 500 | V1 step 500 | V2b step 500 |
|---|---:|---:|---:|---:|
| Shoulder pan (degrees) | 3.8065 | 0.6170 | 0.3654 | **0.3395** |
| Shoulder lift (degrees) | 9.8318 | 0.8417 | 0.8502 | **0.7279** |
| Elbow flex (degrees) | 12.0408 | 0.6323 | 0.5477 | **0.4771** |
| Wrist flex (degrees) | 6.3321 | 0.4744 | 0.3874 | **0.3684** |
| Wrist roll (degrees) | 3.0036 | 0.9869 | **0.6565** | 0.6853 |
| Gripper (`0-100`) | 38.1318 | 2.9452 | **1.2429** | 1.4335 |

Relative to V1, V2b improves shoulder pan by 7.1%, shoulder lift by 14.4%, elbow
flex by 12.9%, and wrist flex by 4.9%. It regresses wrist roll by 4.4% and gripper
MAE by 15.3%.

## Per-task MAE

| Held-out task | Initialization | V0 step 500 | V1 step 500 | V2b step 500 |
|---|---:|---:|---:|---:|
| `mug_on_plate` | 11.7178 | 1.0606 | 0.7076 | **0.6998** |
| `mugs_in_microwave` | 12.3342 | 1.2367 | 0.9006 | **0.8282** |
| `pan_on_stove` | 12.9936 | 1.0345 | **0.6344** | 0.6391 |
| `sort_blocks` | 12.1097 | 0.9699 | **0.5292** | 0.5666 |
| `stack_bowls` | 11.8002 | 1.1129 | **0.6033** | 0.6261 |

## Interpretation

Target-only SO-101 post-training produces a large genuine improvement under
pure-noise, chunk-causal generation. This improvement is not an artifact of exposing
future target actions or observations.

The corrected metric strongly favors both lower-LR recipes over V0. It does not
establish a clear aggregate winner between V1 and V2b at step 500. V2b is better on
four arm joints and the two tasks where V1 had the largest errors, while V1 retains
better wrist-roll and gripper accuracy. Closed-loop evaluation remains required for
final controller selection.
