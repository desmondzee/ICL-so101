# Zero-WAM -> SO-101 controlled prompt sweep

## Status

**Pilot complete; genuine ICL not established.** V2b step 500 was evaluated with
full 50-step video and action denoising for 16 paired closed-loop rollouts: two
mutually exclusive goals, eight prompt conditions, one identical scene seed, and
a 32-control horizon.

The prompts caused deterministic action and trajectory differences, but every
condition approached the alphabet-soup region. No rollout grasped or lifted
either target. Prompt-switch accuracy was therefore zero for both text and video.

## Protocol

| Item | Value |
|---|---|
| Checkpoint | `so101-sft-a100-b32-cosine-v2b-1k-20261011`, step 500 |
| Transformer config SHA256 | `f30afc3dc6393199678e485580151eeb932cc8d8c7223b0f988c2651fe7a7d52` |
| Scene | Existing SO-101 LIBERO basket scene |
| Goals | Lift alphabet soup / lift cream cheese |
| Environment seed | 0 |
| Sampling seed | 20261013 |
| Horizon | 32 controls (three generated chunks) |
| Denoising | 50 video steps, 50 action steps |
| Feedback | Executed LeRobot joint targets and stride-2 observations enter the KV cache |
| Human prompts | Deterministic soup/cheese halves of the existing generated human demonstration |

All 16 rollouts have the same initial simulator-state hash
`3dfb35bce651...c55c` and initial-observation hash
`c3632c847c15...2726`. Payload-identical cross-task controls produced
bit-identical trajectories and rollout videos, confirming paired determinism.

## Outcomes

| Target | Condition | Selected object | Target grasp | Task success | Min soup distance (m) | Min cheese distance (m) |
|---|---|---|---:|---:|---:|---:|
| Soup | No conditioning | alphabet soup | 0 | 0 | 0.1531 | 0.1968 |
| Soup | Correct text | alphabet soup | 0 | 0 | 0.1530 | 0.1967 |
| Soup | Incorrect text | alphabet soup | 0 | 0 | 0.1518 | 0.1966 |
| Soup | Correct human video | alphabet soup | 0 | 0 | 0.1533 | 0.1906 |
| Soup | Mismatched human video | alphabet soup | 0 | 0 | 0.1542 | 0.1895 |
| Soup | Shuffled human video | alphabet soup | 0 | 0 | 0.1519 | 0.1963 |
| Soup | Matching text + video | alphabet soup | 0 | 0 | 0.1527 | 0.1857 |
| Soup | Conflicting text + video | alphabet soup | 0 | 0 | 0.1521 | 0.1962 |
| Cheese | No conditioning | alphabet soup | 0 | 0 | 0.1531 | 0.1968 |
| Cheese | Correct text | alphabet soup | 0 | 0 | 0.1518 | 0.1966 |
| Cheese | Incorrect text | alphabet soup | 0 | 0 | 0.1530 | 0.1967 |
| Cheese | Correct human video | alphabet soup | 0 | 0 | 0.1542 | 0.1895 |
| Cheese | Mismatched human video | alphabet soup | 0 | 0 | 0.1533 | 0.1906 |
| Cheese | Shuffled human video | alphabet soup | 0 | 0 | 0.1499 | 0.1964 |
| Cheese | Matching text + video | alphabet soup | 0 | 0 | 0.1502 | 0.1964 |
| Cheese | Conflicting text + video | alphabet soup | 0 | 0 | 0.1536 | 0.1951 |

- Text prompt-switch accuracy: **0/2** paired target cohorts.
- Human-video prompt-switch accuracy: **0/2** paired target cohorts.
- Task success: **0/16**; target grasp success: **0/16**.
- Maximum target lift and displacement were effectively zero (approximately
  `1e-6 m`) in every rollout.

## Behavioral Differences

Relative to no conditioning, all prompt conditions changed the generated actions
and executed trajectories; no within-task pair was bit-identical. For the soup
target, executed-state RMS differences were `0.898` (correct text), `1.049`
(incorrect text), `2.103` (correct video), `1.447` (mismatched video), `6.055`
(shuffled video), `2.102` (matching text + video), and `1.510` (conflicting text
+ video). The shuffled-video response was the largest difference, not the most
semantically correct response.

These differences demonstrate **prompt sensitivity**, not successful prompt
following: the arm remained closest to the soup region under cheese-directed
prompts as well.

## Rollouts

- [Generated pilot report and all 16 rollout links](../../../outputs/zero_wam/prompt_sweep_pilot_v2b_seed0/report.md)
- [Pilot summary JSON](../../../outputs/zero_wam/prompt_sweep_pilot_v2b_seed0/summary.json)
- [Eight-condition reduced-denoising modality smoke](../../../outputs/zero_wam/prompt_sweep_modality_smoke/report.md)

## Failures And Limitations

- The first raw-video smoke exposed two encoding issues: ICL clips must contain
  `1+4k` frames, and Wan VAE streaming must encode one initial frame followed by
  four-frame chunks. Both are fixed and the complete modality smoke passes.
- The 32-control pilot is too short to measure eventual grasp/task success.
- Only V2b and one environment seed were run at full denoising. V1, additional
  seeds, longer horizons, and held-out task expansion remain unexecuted.
- The soup and cheese prompts are temporal halves of one generated human video.
  The cheese half starts after soup motion, so the two clips are not perfectly
  scene-matched demonstrations.
- Rollout frames were checked for decodability, camera layout, and visible arm
  motion; a complete semantic human review of every frame was not performed.

## Conclusions

- **Action prediction/execution:** the model generates stable closed-loop motion,
  but this pilot shows no grasp or task completion.
- **Prompt sensitivity:** established at the action and trajectory level under
  identical state and noise, including exact replay for identical payloads.
- **Genuine ICL:** not established. Correct video did not switch object selection
  or outperform mismatched/shuffled controls. Expanding this protocol is not
  justified until a longer-horizon pilot demonstrates semantic object switching.
