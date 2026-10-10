# Zero-WAM and LingBot-VA training hyperparameter report

Research date: 2026-10-10

## Sources and scope

- Zero-WAM paper: arXiv 2608.26103v2 and the repository copy in
  `docs/2608.26103.txt`.
- Official Zero-WAM repository at commit
  `08e2c4ae41e2b63573a299825cebe6753481407c`.
- LingBot-VA paper, *Causal World Modeling for Robot Control*: arXiv 2601.21998.
- Official LingBot-VA repository at commit
  `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`.
- The active SO-101 run manifest and launcher at launch revision
  `9c9dc6eaa66896c4687edbe47d924c0f247d6a32`.

Paper values and released-code defaults are reported separately. They are not always
the same, and unspecified values are left unspecified rather than inferred.

## Executive comparison

| Setting | Zero-WAM paper | LingBot-VA paper | Current SO-101 |
|---|---:|---:|---:|
| Stage | RoboTwin post-training | preferred post-training | post-training |
| Initialization | Zero-WAM pretrain | LingBot-VA pretrain | Zero-WAM pretrain |
| Optimizer | AdamW | AdamW | AdamW |
| Peak LR | `1e-4` | `1e-5` | `1e-4` |
| Schedule | not stated in paper | reduced LR; released code is warmup + constant | 200-step warmup + constant |
| Weight decay | `0.01` | post-training paper does not restate it | `0.01` |
| Steps | 4,000 | 3,000 preferred | 4,000 |
| Parallelism/data per step | 64 GPUs, packed to at most 160K tokens/GPU | packed sequences; benchmark-specific | 8 GPUs, 2 full episodes/GPU/step, no packing |
| Nominal episode batch | not reported; token packing is authoritative | not reported in paper | 16 |
| Data mixture | Task-diverse VA : HumanGen : RoboTwin = `2:10:3` | target-domain data | SO-101 simulation only |

The current LR and 4,000-step duration exactly match the Zero-WAM paper, but the
batching and data regime do not. The paper used 64 GPUs with multi-episode token
packing and replayed broad pre-training data. The current run uses a global batch of
16 full episodes on eight GPUs and no replay mixture. LingBot-VA's preferred
post-training recipe is ten times lower at `1e-5`; it explicitly reports that
`1e-4` for 1,000 steps is faster but slightly inferior.

## Zero-WAM paper

### Pre-training

- Architecture initialized from Wan-2.2-TI2V-5B.
- AdamW, peak LR `1e-4`, weight decay `0.01`.
- Task-diverse VA : HumanGen sampling ratio `1:5`.
- Non-ICL language dropout `0.1`.
- ICL human-video dropout `0.1`.
- ICL language dropout `0.4`.
- Robot-video chunk size sampled uniformly from 1 through 4.
- Multiple variable-length samples packed per GPU, isolated by attention masks.
- Maximum 160K tokens per GPU.
- 15,360 GPU-hours.
- Pre-training step count, global batch, Adam betas, epsilon, warmup length,
  clipping threshold, and LR decay are not stated in the paper.

### RoboTwin post-training

- Starts from the Zero-WAM pre-trained checkpoint.
- 64 GPUs for 4,000 optimizer steps.
- Task-diverse VA : HumanGen : RoboTwin sampling ratio `2:10:3`.
- Multi-sample packing remains capped at 160K tokens per GPU.
- The implementation section gives AdamW `1e-4`, weight decay `0.01`; it does not
  separately state a lower post-training LR.
- Seven evaluation tasks are held out; 43 RoboTwin tasks are used for training.

### Released Zero-WAM code

The released RoboTwin training configuration uses:

- AdamW LR `1e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`.
- Weight decay `0.01`.
- 200-step linear warmup followed by constant LR.
- Gradient clipping at `1.0`; skip threshold `20 x` the clipping threshold.
- BF16, FSDP, activation checkpointing.
- Batch size 1 per rank, gradient accumulation 1, default 50,000 steps.
- Save interval 1,000.
- ICL dropout `0.1`, target-language dropout `0.4`.
- Video/action/MCP flow shifts `5/1/10`.
- Video and MCP timestep reweighting; action loss unweighted.
- Noisy conditioning probability `0.5`.
- Four one-block MCP/IFP modules at layers `[3, 11, 19, 29]`.
- Released MCP weights `[0.5, 0.25, 0.15, 0.1]`.
- Future-chunk stride 2, random frame chunk 1-4, maximum attention window 64.

The released default of 50,000 steps is a generic Robotwin-only launcher default,
not the paper's 4,000-step 64-GPU co-training recipe.

### Paper/code discrepancies

- The paper prints MCP/IFP weights `[0.5, 0.25, 0.15, 0.15]`; released code uses
  `[0.5, 0.25, 0.15, 0.1]`.
- The paper states human-video height RoPE offset 32; released code config uses 24.
- The paper describes packed training; the public loader/trainer enforces one sample
  per rank and does not reproduce the paper's 160K-token packing regime.

## LingBot-VA paper

### Pre-training

- Wan2.2-5B video backbone plus action stream, approximately 5.3B parameters.
- 1.4T training tokens from approximately 16K hours of robot manipulation data.
- AdamW with peak LR `1e-4`, weight decay `0.01`.
- Cosine annealing with linear warmup.
- BF16 and gradient clipping at `2.0`.
- Text dropout `0.1`; inverse-dynamics loss weight `1`.
- Uniform sampling across data sources.
- Uniform flow/SNR sampler for video and action.
- Random chunk size 1-4.
- Noise augmentation probability `0.5`, augmentation level uniform in `[0.5, 1]`.
- Multiple episodes packed up to 10K tokens with isolation masks.
- The paper does not report global batch, GPU count, warmup length, Adam betas, or
  epsilon.

### General post-training guidance

- Preferred: LR `1e-5` for 3,000 steps.
- Faster alternative: LR `1e-4` for 1,000 steps, reported as reasonable but
  slightly inferior.
- The paper reports that as few as 50 demonstrations can be sufficient.

### Reported downstream adaptations

- Six real-world tasks, 50 demonstrations: LR `1e-4`, 500 steps, sequence length
  150K.
- RoboTwin: 2,500 clean plus 25,000 randomized demonstrations, LR `1e-5`,
  50,000 steps.
- Controlled LingBot-VA versus Wan initialization ablation: 50 task-specific
  demonstrations, LR `1e-5`, 3,000 steps.
- LIBERO suites: 500 demonstrations per suite, LR `1e-5`, 4,000 steps,
  sequence length 100K.

### Released LingBot-VA post-training code

Common implementation:

- Full-parameter FSDP training in BF16 with activation checkpointing.
- AdamW betas `(0.9, 0.95)`, epsilon `1e-8`.
- Linear warmup followed by constant LR.
- Gradient clipping `2.0`.
- Text/CFG dropout `0.1`.
- README recommends global batch 32 or 64, using gradient accumulation if needed.

RoboTwin config:

- LR `1e-5`, weight decay `0.1`, warmup 10.
- Per-rank batch 1, accumulation 1, 50,000 steps, save every 1,000.
- Video/action flow shifts `5/1`.

LIBERO config:

- LR `1e-5`, weight decay `0.1`, warmup 10.
- Per-rank batch 1, accumulation 10, 5,000 steps, save every 200.
- Video/action flow shifts `5/0.05`.

Generic demo config:

- LR `1e-4`, weight decay `0.1`, warmup 10.
- Per-rank batch 1, accumulation 8, 2,000 steps, save every 50.

The released code and paper differ in schedule and, for LIBERO, nominal step count:
the paper reports cosine decay for pre-training and 4,000 LIBERO steps; public
post-training code uses warmup-constant and defaults to 5,000 LIBERO steps.

## Current SO-101 run

- Full-parameter Zero-WAM post-training from `zero-wam-pretrain`.
- 8 A100 GPUs, per-rank batch 1, accumulation 2, effective batch 16.
- One variable-length full episode per microstep; no multi-episode token packing.
- 4,000 optimizer steps, or 64,000 episode draws.
- AdamW LR `1e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`.
- Weight decay `0.01`.
- 200-step linear warmup followed by constant LR.
- BF16, clip `1.0`, skip threshold `20`.
- Only the 1,109 paired SO-101 simulation episodes across 61 tasks.
- Task-balanced sampling; no Task-diverse VA or HumanGen replay data.
- ICL/video dropout `0.1`, target-language dropout `0.4`.
- Flow shifts, loss reweighting, MCP structure/weights, noisy conditioning, chunk
  randomization, and attention-window randomization match released Zero-WAM code.
- Deterministic validation on 50 task-disjoint episodes.

## Implications

1. `1e-4` is not an invented or obviously invalid LR: it exactly matches Zero-WAM
   pre-training, the Zero-WAM paper's reported optimizer setting, and LingBot-VA
   pre-training.
2. It is not directly validated for the current small-batch, target-only regime.
   Zero-WAM's paper recipe couples `1e-4` to 64-GPU packed training and broad replay.
3. LingBot-VA is stronger evidence for conservative target-domain adaptation:
   its preferred and benchmark post-training LR is `1e-5`; `1e-4` is explicitly
   described as a shorter, slightly inferior alternative.
4. The current 200-step warmup is more conservative than LingBot-VA's released
   10-step post-training warmup, but both released trainers hold LR constant after
   warmup. The LingBot-VA pre-training paper instead uses cosine decay.
5. The largest scientific divergence is not merely LR. It is the combination of
   effective batch 16, no token packing, and no replay mixture. These change both
   gradient noise and forgetting risk.
6. A literature-grounded continuation comparison should include `1e-5`, not only
   nearby values around `1e-4`. A geometric three-point set is
   `{1e-4, 3e-5, 1e-5}`.

## Recommended bounded decision

- Preserve the active run through its step-500 atomic checkpoint and validation.
- If validation improves clearly and all tasks remain healthy, continuing `1e-4`
  is supported by the Zero-WAM paper.
- If validation is flat, noisy, or task-imbalanced, fork the exact step-500 state
  into short continuation branches at `{1e-4, 3e-5, 1e-5}`.
- Compare deterministic validation at equal additional exposure, ideally steps 600
  and 700, while preserving identical sampler/RNG state.
- Treat this as a continuation/decay decision. It does not prove which LR would have
  been optimal from initialization; answering that requires fresh controlled runs.
- Do not tune flow shifts, loss weights, dropout, chunking, and LR simultaneously.
  Those settings match released Zero-WAM and changing them would confound the LR
  result.
