# SO-101 Zero-WAM A100 v2 training tracker

Purpose: track the proposed 1,000-step, batch-32 production candidate following the
batch-16 v1 experiment. V2 starts fresh from the same released Zero-WAM pre-training
checkpoint; it is not a resume of v1.

## Reference runs

### V0: high-LR reference

- Run ID: `so101-sft-a100-b16-4k-20261010`
- Retained boundary: step 500
- Effective batch: 16
- Peak/constant LR: `1e-4`
- Warmup: 200 steps
- Episode draws at step 500: 8,000
- Step-500 legacy-protocol macro total/action:
  `0.489130 / 0.003601`

### V1: lower-LR reference

- Run ID: `so101-sft-a100-b16-lr1e5-v1-500`
- WandB run ID: `3ed7a335`
- Effective batch: 16
- Peak/constant LR: `1e-5`
- Warmup: 10 steps
- Target: 500 steps / 8,000 episode draws
- Validation interval: 100 steps
- Checkpoint interval: 500 steps
- V1-protocol macro totals:
  - Step 0: `0.558893`
  - Step 100: `0.495299`
  - Step 200: `0.489184`
  - Step 300: `0.487234`
- V1 action endpoint normalized MAE:
  - Step 0: `0.036992`
  - Step 100: `0.011626`
  - Step 200: `0.011620`
  - Step 300: `0.011699`

## Frozen v2 proposal

- Start from: `data/models/zero-wam-pretrain`
- GPUs: 8 A100-SXM4-80GB
- Per-rank batch: 1
- Gradient accumulation: 4 microsteps
- Effective batch: 32
- Optimizer: AdamW
- Peak LR: `1e-5`
- Betas: `(0.9, 0.95)`
- Epsilon: `1e-8`
- Weight decay: `0.01`
- Gradient clipping: `1.0`
- Warmup: 10 linear steps
- Post-warmup schedule: cosine decay through step 1,000
- Final LR: `1e-6`
- Total optimizer steps: 1,000
- Total episode draws: 32,000
- Mean draws per 1,109-episode dataset: approximately 28.9
- Validation interval: 100 optimizer steps
- Full checkpoint interval: 500 optimizer steps
- Seed: 42
- Precision: BF16
- Sampling: unchanged deterministic task-balanced sampler
- Conditioning, loss definitions, flow shifts, MCP weights, chunking, dropout, and
  dataset releases: unchanged from v1

## Exposure-matched comparisons

V2 must be compared with v1 by episode exposure, not only optimizer step:

| Episode draws | V1 batch 16 | V2 batch 32 |
|---:|---:|---:|
| 8,000 | step 500 | step 250 |
| 16,000 | step 1,000 equivalent | step 500 |
| 32,000 | step 2,000 equivalent | step 1,000 |

- [ ] Run an additional deterministic validation at v2 step 250 for the exact
      equal-exposure comparison with v1 step 500.
- [ ] Keep scheduled validation at steps 0, 100, 200, 300, ..., 1,000.
- [ ] Compare training-loss variance at equal exposure using detrended CV.
- [ ] Compare macro validation total, each objective, per-task losses, action endpoint
      errors, and per-joint physical errors.
- [ ] Do not infer a batch-size effect solely from v1/v2 differences because v2 also
      introduces cosine decay.

## Required implementation

- [x] Add an explicit scheduler selection argument.
- [x] Preserve `linear_warmup_then_constant` as the default for new constant-LR runs.
- [x] Implement `linear_warmup_then_cosine` for v2.
- [x] Add an explicit final/minimum LR argument.
- [x] Define the exact schedule:
      - Steps 1-10: linear increase to `1e-5`.
      - Steps 11-1,000: cosine decay from `1e-5` to `1e-6`.
- [x] Include scheduler type, total schedule steps, warmup, peak LR, and final LR in
      config identity, local run manifest, checkpoint metadata, and WandB config.
- [x] Reject resume when scheduler identity differs.
- [x] Unit-test LR values at initialization, warmup boundary, representative midpoint,
      penultimate step, and final step.
- [ ] Unit-test scheduler state round-trip through a full checkpoint.
- [x] Ensure step indexing has no one-step warmup or cosine endpoint shift.

## Validation and action metrics

- [ ] Retain the frozen 50-episode, five-task validation manifest and deterministic
      per-episode seeds.
- [ ] Retain q05, median, q95, mean, standard deviation, min, and max for all losses.
- [ ] Retain one-forward clean-action endpoint MAE/RMSE only as a teacher-forced
      flow-denoising diagnostic. It receives noisy target actions and a clean
      target-action conditioning stream, so it is not the primary controller metric.
- [ ] Retain per-joint MAE/RMSE in degrees for all five joints.
- [ ] Retain gripper MAE/RMSE in physical 0-100 units.
- [ ] Retain action error by control offset 0-15.
- [ ] Publish per-channel mean/q05/median/q95 metrics to WandB.
- [ ] Recompute the updated continuous physical metrics for v1 step 500 before final
      v1-v2 comparison.
- [ ] Before checkpoint selection, run recorded-observation generative action
      validation through the exact deployment cache/inference path: seeded Gaussian
      action noise, 50 action-denoising steps, and no target actions beyond previously
      executed history available at the chunk boundary.
- [ ] Match deployment chunk-causal attention exactly:
      - keep the full, separate human ICL demonstration visible;
      - cache only held-out robot observations and executed actions strictly before the
        scoring boundary;
      - hide all held-out robot observations and actions at or after the boundary;
      - initialize the next 16 action tokens from Gaussian noise;
      - allow those 16 noisy tokens to attend bidirectionally within the jointly
        denoised chunk;
      - reveal the 16 target actions only after generation, for metric computation.
- [ ] Treat each frozen boundary as an independent single-chunk, open-loop action
      validation: generate and score exactly one 16-control chunk, then stop. Do not
      execute the actions, advance the environment, reveal the next observation, or
      interpret this metric as rollout success.
- [ ] Reuse the human ICL cache and each boundary's robot-prefix cache across all 50
      action-denoising steps. Parallelize independent boundaries where possible without
      changing the deployed cache or attention semantics.
- [ ] Freeze sample/chunk anchors and noise seeds in an immutable manifest. Compare
      each generated 16-control chunk against the held-out demonstration and report
      normalized and per-channel physical MAE/RMSE, offsets 0-15, per-task aggregates,
      and the executable 16-control persistence baseline.
- [ ] Evaluate initialization and every candidate checkpoint with the same generative
      manifest before selecting a checkpoint.
- [ ] Treat closed-loop rollout success as deferred; do not equate validation loss
      with task success.

## Launch gates

- [x] V1 reaches step 500 and writes its atomic full checkpoint.
- [x] V1 step-500 deterministic validation completes.
- [x] V1 checkpoint sync completes under
      `hf://buckets/nikgeo/icl-so101-weights/runs/`.
- [x] Scheduler implementation and focused tests pass: 48 focused tests.
- [x] Batch-32 memory gate is covered by the stricter prior batch-64 qualification.
- [x] Batch-32 optimizer gate is covered by the stricter prior batch-64 qualification.
- [ ] Step-0 validation smoke completes with the v2 config.
- [x] GPUs are free of unrelated processes.
- [x] Disk capacity is recorded: approximately 19 TiB available.
- [x] Git worktree is clean and launch revision is pushed.
- [x] Full scientific config is visible in WandB.

## Operational plan

- [ ] Use a new run ID and WandB ID; never overwrite or resume v1 as v2.
- [ ] Save full checkpoints at steps 500 and 1,000.
- [ ] Sync each completed checkpoint asynchronously to
      `hf://buckets/nikgeo/icl-so101-weights/runs/<v2-run-id>/`.
- [ ] Keep local fsync-backed `metrics.jsonl`.
- [ ] Keep exact episode exposure ledgers at each checkpoint.
- [ ] Confirm no checkpoint is written at validation-only boundaries.
- [ ] Monitor NaN/Inf, skipped steps, clipping, throughput, VRAM, validation tail
      metrics, and per-task regressions.

## Decision rules

- At step 250, compare v2 against v1 step 500 at 8,000 episode draws.
- Continue through step 500 unless v2 has nonfinite updates, broad per-task
  validation regression, or materially worse action endpoint errors.
- At step 500, compare the value of another 16,000 draws against the observed
  validation slope.
- Continue to step 1,000 only while deterministic validation or physical action
  metrics show useful progress.
- Select only a retained full checkpoint; do not select a validation-only step unless
  a separately approved model-snapshot policy is implemented.

## V2 launch record

- Run ID: `so101-sft-a100-b32-cosine-v2-1k-20261011`
- WandB ID/URL: `7b954ad5` /
  `https://wandb.ai/wamh3/zero-wam-so101/runs/7b954ad5`
- Launch Git revision: `b239790` (cosine implementation at `644794e`, equal-exposure
  validation at `131e32c`, launch-identity fix at `b239790`)
- Config SHA256:
  `e0d1e63ae04824771a0ea5552dc1d2437ae97b1a46de9a0cf3e079a3c9e2c962`
- Initialization model hash: `_______________________________________________`
- Launch command:
  `scripts/run_zero_wam_so101_a100.sh start so101-sft-a100-b32-cosine-v2-1k-20261011 1000 4 1e-5 10 100 cosine 1e-6 250`
- Launch UTC: `2026-10-11T02:36:58Z`
- Preflight/smoke references: `________________________________________________`
- Bucket destination:
  `hf://buckets/nikgeo/icl-so101-weights/runs/so101-sft-a100-b32-cosine-v2-1k-20261011`

## V2 result record

- First launch incident: rank 1 OOM after optimizer step 58 while constructing a
  long-sequence flex-attention mask. PyTorch reported 59.47 GiB allocated,
  13.26 GiB reserved-but-unallocated, and a failed 9.38 GiB allocation. No checkpoint
  existed, so the first launch is not resumable.
- Mitigation: release the CUDA allocator's unused cached blocks before every
  variable-length training microstep. This is memory hygiene only; it does not change
  samples, losses, optimizer math, effective batch, or LR schedule.
- Step-250 equal-exposure comparison: `_______________________________________`
- Step-500 checkpoint/validation: `___________________________________________`
- Step-1,000 checkpoint/validation: `_________________________________________`
- Best retained checkpoint: `_________________________________________________`
- Exact episode exposure summary: `___________________________________________`
- Final checkpoint hashes: `__________________________________________________`
- Bucket verification: `______________________________________________________`
- Decision and rationale: `__________________________________________________`
