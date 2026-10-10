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

- [ ] Reach optimizer step 500 with finite losses and no repeated skipped steps.
- [ ] Complete `checkpoint_step_500`, including transformer, AdamW, scheduler,
      rank-local RNG, task-balanced sampler, exact 1,109-episode exposure ledger,
      config hash, and WandB run ID.
- [ ] Complete deterministic validation at step 500.
- [ ] Verify checkpoint atomicity and exact-load the transformer.
- [ ] Record step-500 validation overall, macro-task, and per-task metrics.
- [ ] Compare step 500 with the frozen step-0 baseline.
- [ ] Record the continue/stop decision before changing the launcher.
- [ ] Stop the v0 process only after the checkpoint and validation reports are durable.

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

- [ ] Add explicit `--validation-interval`, default 500.
- [ ] Set the v1 validation interval to 100 optimizer steps, including steps 0, 100,
      200, 300, 400, and 500.
- [ ] Keep full resumable checkpoints at steps 500, 1000, ..., 4000.
- [ ] Run validation independently of checkpoint writing every 100 steps through
      4000.
- [ ] Keep v0 and v1 validation reports under separate immutable run roots.
- [ ] Continue using the same frozen 50-episode validation manifest, per-episode seeds,
      conditioning, aggregation, and manifest hash.
- [ ] Prove each validation pass restores model mode, RNG, scheduler, sampler, loader,
      and exposure state exactly.
- [ ] Log each validation row to both `metrics.jsonl` and the existing WandB run.
- [ ] Keep full per-episode validation JSON reports append-only under `validation/`.
- [ ] Keep checkpoint selection restricted to full 500-step checkpoints unless a
      separate model snapshot policy is explicitly approved.

## Comparison identity

- [ ] Create a new run ID and WandB run linked to v0 as an LR/warmup comparison.
- [ ] Require a new config hash; never treat v1 as a strict resume of v0.
- [ ] Exact-load the same initialization checkpoint used by v0.
- [ ] Prove initial model hash, sampler seed/order, validation manifest, and step-0
      validation are identical to v0.
- [ ] Record only these intended scientific differences: peak LR `1e-5`, warmup 10,
      and validation interval 100.
- [ ] Compare v0 and v1 at equal optimizer steps and equal episode exposure.

## v1 validation gates

- [ ] Unit-test interval scheduling: step 0, every requested validation interval, and
      no duplicate execution at checkpoint boundaries.
- [ ] Unit-test that checkpoint interval remains 500 when validation interval is 100.
- [ ] Unit-test exact state restoration around non-checkpoint validation.
- [ ] Run an eight-rank fresh-run smoke with peak LR `1e-5` and warmup 10.
- [ ] Confirm one non-checkpoint validation report and WandB row.
- [ ] Confirm exact exposure counts are unchanged by validation.
- [ ] Confirm no full checkpoint is written at a validation-only boundary.
- [ ] Re-run focused trainer, validation, checkpoint, and launcher tests.

## Step-500 decision record

- Step reached UTC: `__________________________________________________________`
- Checkpoint path: `___________________________________________________________`
- Checkpoint/training-state SHA256: `__________________________________________`
- Exposure total/unique/unseen: `_____________________________________________`
- Step-500 validation manifest SHA256: `______________________________________`
- Macro total/video/action/MCP: `_____________________________________________`
- Change from step 0: `_______________________________________________________`
- NaN/Inf/skipped-step audit: `_______________________________________________`
- Throughput and projected completion: `_____________________________________`
- Decision and rationale: `__________________________________________________`

## v1 launch record

- Code revision: `____________________________________________________________`
- New config SHA256: `________________________________________________________`
- Initialization checkpoint/hash: `___________________________________________`
- Fresh launch command: `_____________________________________________________`
- UTC resume time: `__________________________________________________________`
- New run/WandB IDs: `________________________________________________________`
- Initial sample IDs/LR verification: `_______________________________________`
- First validation-only step/report: `_______________________________________`
