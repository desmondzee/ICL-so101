# SO-101 Zero-WAM A100 v1 continuation tracker

Purpose: track the bounded changes made after the first 500 optimizer steps of
`so101-sft-a100-b16-4k-20261010`. The current process remains unchanged through its
first atomic checkpoint and deterministic validation at step 500. The v1 comparison
will then start fresh from the same released initialization checkpoint; it will not
resume the step-500 optimizer state because warmup and peak LR are changing.

## Current v0 run

- Run ID: `so101-sft-a100-b16-4k-20261010`
- WandB run ID: `83281560`
- Launch code revision: `9c9dc6eaa66896c4687edbe47d924c0f247d6a32`
- Frozen config SHA256:
  `e489e7fc9788cf1814946838dc56d5e55c394d9bb915244124b9c81ed361f61f`
- Effective batch: 16 (8 ranks x 2 accumulation microsteps)
- Optimizer: AdamW, peak/constant LR `1e-4`, 200-step linear warmup
- Full checkpoint interval: 500 optimizer steps
- Deterministic validation interval in the loaded process: 500 optimizer steps
- Step-0 validation macro total: `0.5452174925804137`
- Decision: do not interrupt before the step-500 checkpoint; stopping earlier would
  discard all optimizer, scheduler, sampler, RNG, and exposure state since launch.

## v1 change boundary

- [x] Reach optimizer step 500 with finite losses and no repeated skipped steps.
- [x] Complete `checkpoint_step_500`, including transformer, AdamW, scheduler,
      rank-local RNG, task-balanced sampler, exact 1,109-episode exposure ledger,
      config hash, and WandB run ID.
- [x] Complete deterministic validation at step 500.
- [ ] Verify checkpoint atomicity and exact-load the transformer.
- [x] Record step-500 validation overall, macro-task, and per-task metrics.
- [x] Compare step 500 with the frozen step-0 baseline: macro total `0.545217` to
      `0.489130` (-10.3%); action `0.009799` to `0.003601` (-63.3%).
- [x] Record the decision: preserve v0 as the high-LR reference and launch a fresh
      lower-LR v1 comparison.
- [x] Stop the v0 process only after the checkpoint and validation reports are durable.

## Planned v1 changes

The user froze the following v1 optimizer changes on 2026-10-10:

- Peak LR: `1e-5` rather than `1e-4`.
- Linear warmup: 10 steps rather than 200.
- Start from the same `zero-wam-pretrain` initialization, seed, and initial sampler
  state. Do not apply a new warmup to the v0 step-500 optimizer state.
- Keep effective batch, data, losses, conditioning, flow schedules, dropout, MCP,
  clipping, and task-balanced sampling unchanged so the optimizer change remains
  interpretable.

The v1 run also changes observability as follows:

- [x] Add explicit `--validation-interval`, default 500.
- [x] Set the v1 validation interval to 100 optimizer steps, including steps 0, 100,
      200, 300, 400, and 500.
- [x] Keep full resumable checkpoints at steps 500, 1000, ..., 4000.
- [x] Run validation independently of checkpoint writing every 100 steps through
      4000.
- [x] Keep v0 and v1 validation reports under separate immutable run roots.
- [x] Continue using the same frozen 50-episode validation manifest, per-episode seeds,
      conditioning, aggregation, and manifest hash.
- [ ] Prove each validation pass restores model mode, RNG, scheduler, sampler, loader,
      and exposure state exactly.
- [x] Log each validation row to both `metrics.jsonl` and the new v1 WandB run.
- [x] Keep full per-episode validation JSON reports append-only under `validation/`.
- [ ] Keep checkpoint selection restricted to full 500-step checkpoints unless a
      separate model snapshot policy is explicitly approved.
- [x] Report mean, standard deviation, q05, median, q95, minimum, and maximum for
      total, video, action, and MCP validation losses.
- [x] Fix validation to the deployment chunk of two latent frames x eight controls
      (16 absolute controls, approximately 0.53 seconds), rather than sampling a
      random training chunk length.
- [x] Reconstruct the clean absolute-action endpoint from each fixed-noise flow
      prediction and report normalized MAE/RMSE distributions.
- [x] Denormalize endpoint errors and report per-channel MAE/RMSE in joint degrees
      and gripper 0-100 units.
- [x] Report normalized MAE/RMSE separately for control offsets 0-15.
- [x] Use continuous gripper MAE/RMSE in physical 0-100 actuator units rather than
      thresholding the potentially continuous target into arbitrary open/closed
      classes. Publish mean, q05, median, and q95 to WandB.
- [ ] Recompute continuous gripper MAE/RMSE for the retained v1 checkpoint; the
      already-running process loaded the earlier evaluator before this change.
- [x] Publish the principal action endpoint and loss quantiles to local JSONL and
      WandB; retain the full per-episode/channel/offset detail in validation JSON.
- [x] Treat one-step clean-action reconstruction as a diagnostic, not a substitute
      for periodic full action-flow denoising or eventual closed-loop rollout.
- [ ] Add fixed-subset full 50-step action denoising metrics after the v1 launch if
      they can be implemented without delaying or perturbing the core comparison.

## Comparison identity

- [x] Create a new run ID and WandB run linked to v0 as an LR/warmup comparison.
- [x] Require a new config hash; never treat v1 as a strict resume of v0.
- [ ] Exact-load the same initialization checkpoint used by v0.
- [ ] Prove initial model hash, sampler seed/order, and validation manifest are
      identical to v0. The v1 step-0 scalar loss is intentionally not numerically
      identical because v1 fixes validation chunking to the deployment horizon; use
      the v1 step-0 report as the new-protocol baseline.
- [ ] Record only these intended scientific differences: peak LR `1e-5`, warmup 10,
      and validation interval 100.
- [ ] Compare v0 and v1 at equal optimizer steps and equal episode exposure.

## v1 validation gates

- [x] Unit-test interval scheduling: step 0, every requested validation interval, and
      no duplicate execution at checkpoint boundaries.
- [x] Unit-test that checkpoint interval remains 500 when validation interval is 100.
- [ ] Unit-test exact state restoration around non-checkpoint validation.
- [x] Run an eight-rank fresh-run smoke with peak LR `1e-5` and warmup 10:
      `a100-v1-validation-action-smoke-v3`, WandB `419959e8`.
- [ ] Confirm one non-checkpoint validation report and WandB row.
- [ ] Confirm exact exposure counts are unchanged by validation.
- [ ] Confirm no full checkpoint is written at a validation-only boundary.
- [x] Re-run focused trainer, validation, checkpoint, and launcher tests: 71 passed.

## Step-500 decision record

- Step reached UTC: `2026-10-10T19:20Z`
- Checkpoint path: `outputs/zero_wam/sim_training/so101-sft-a100-b16-4k-20261010/checkpoints/checkpoint_step_500`
- Checkpoint/training-state SHA256:
  `6c03814689ed0fa9290c5a3d20a526971b9aba9cb543de47ec5bd638c687fe6a`
- Exposure total/unique/unseen: `8000 / 1102 / 7`
- Step-500 validation manifest SHA256:
  `d309785c3cd73e721c81537a2ac39c19d3d618a5f87de8ff0f1dc84260324cdf`
- Macro total/video/action/MCP:
  `0.489130 / 0.159244 / 0.003601 / 0.326286`
- Change from step 0: `-10.3% / -10.8% / -63.3% / -8.6%`
- NaN/Inf/skipped-step audit: `none`
- Throughput and projected completion: `_____________________________________`
- Decision and rationale: `launch fresh v1 at 1e-5/10-step warmup; preserve v0`

## v1 launch record

- Code revision: `9fc4cd9b710b0d0f1082004f4b730bc54cef2598`
- New config SHA256:
  `3a2f2949c1b20ae6f299b86f12629490ede4fee49b158e6c3e1c23bc9fb74c2f`
- Initialization checkpoint/hash: `data/models/zero-wam-pretrain`; immutable model
  tree hash is included in the config identity.
- Fresh launch command:
  `scripts/run_zero_wam_so101_a100.sh start so101-sft-a100-b16-lr1e5-v1-500 500 2 1e-5 10 100`
- UTC resume time: `2026-10-10T20:26:19Z`
- New run/WandB IDs: `so101-sft-a100-b16-lr1e5-v1-500` / `3ed7a335`
- Initial sample IDs/LR verification: `_______________________________________`
- First validation-only step/report: `_______________________________________`
