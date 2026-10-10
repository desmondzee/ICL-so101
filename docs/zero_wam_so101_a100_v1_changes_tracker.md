# SO-101 Zero-WAM A100 v1 continuation tracker

Purpose: track the bounded changes made after the first 500 optimizer steps of
`so101-sft-a100-b16-4k-20261010`. The current process remains unchanged through its
first atomic checkpoint and deterministic validation at step 500. This document
defines the continuation work; it does not retroactively change steps 1-500.

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

The v1 continuation changes observability only. It must not alter model parameters,
optimizer state, LR/scheduler state, sampled-data sequence, RNG state, exposure
counts, loss definitions, conditioning, effective batch, or total target step.

- [ ] Add explicit `--validation-interval`, default 500.
- [ ] Set the v1 continuation validation interval to 100 optimizer steps.
- [ ] Keep full resumable checkpoints at steps 500, 1000, ..., 4000.
- [ ] Run validation independently of checkpoint writing at steps
      600, 700, 800, 900, then every 100 steps through 4000.
- [ ] Do not rewrite or duplicate the already-published step-0 or step-500 reports.
- [ ] Continue using the same frozen 50-episode validation manifest, per-episode seeds,
      conditioning, aggregation, and manifest hash.
- [ ] Prove each validation pass restores model mode, RNG, scheduler, sampler, loader,
      and exposure state exactly.
- [ ] Log each validation row to both `metrics.jsonl` and the existing WandB run.
- [ ] Keep full per-episode validation JSON reports append-only under `validation/`.
- [ ] Keep checkpoint selection restricted to full 500-step checkpoints unless a
      separate model snapshot policy is explicitly approved.

## Resume compatibility

The current config identity includes the Git revision and dirty-diff hash. A code
change therefore intentionally fails strict resume unless it is handled as an audited
compatible migration.

- [ ] Separate the immutable scientific/training identity from the operational-code
      identity without weakening checks on data, model, optimizer, scheduler,
      conditioning, sampler, batch, LR, or losses.
- [ ] Define a one-time migration record containing old/new Git revisions, old/new
      config hashes, checkpoint path, training-state hash, and the exact allowed
      observability-only diff.
- [ ] Require an explicit CLI migration flag naming that record; never silently accept
      config drift.
- [ ] Load and verify step-500 model/optimizer/scheduler/RNG/sampler/exposure state.
- [ ] Prove the next sampled episode IDs and LR match the uninterrupted expectation.
- [ ] Resume the same WandB ID with `resume="must"`.
- [ ] Preserve the original run ID or record an explicit linked continuation ID; do
      not create an unlinked metric series.

## v1 validation gates

- [ ] Unit-test interval scheduling: step 0, every requested validation interval, and
      no duplicate execution at checkpoint boundaries.
- [ ] Unit-test that checkpoint interval remains 500 when validation interval is 100.
- [ ] Unit-test exact state restoration around non-checkpoint validation.
- [ ] Run an eight-rank continuation smoke from a disposable checkpoint.
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

- Old/new Git revisions: `____________________________________________________`
- Migration record path/SHA256: `____________________________________________`
- Resume checkpoint: `_______________________________________________________`
- Continuation command: `____________________________________________________`
- UTC resume time: `__________________________________________________________`
- Run/WandB IDs: `____________________________________________________________`
- First resumed sample IDs/LR verification: `_______________________________`
- First validation-only step/report: `_______________________________________`
