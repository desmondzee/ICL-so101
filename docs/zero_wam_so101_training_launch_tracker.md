# SO-101 Zero-WAM training launch tracker

Purpose: operational checklist for launching and monitoring the finalized 4,000-step
SO-101 Zero-WAM run. The canonical settings are in
`docs/zero_wam_so101_final_training_config.md`; historical data/evaluation work remains in
`docs/sim_zero_wam_training_tracker.md`. This tracker is the concise launch status.

**Current state: preparation only — optimizer training has not started.**

- [ ] **BLOCKED for Modal-only gates:** raise/reset the Modal workspace spend limit. The final-geometry
      encoders completed 3,477/3,477 artifacts, but Modal rejected the independent
      inventory launch before app creation with `Workspace ... has exceeded its spend
      limit`. No Modal job is currently active; remaining work may run on external
      on-demand compute.

## Frozen decisions and evidence

- [x] Canonical configuration committed to `main` at `e38c50b`.
- [x] Start checkpoint pinned to `Robbyant-Research/zero-wam-pretrain` revision
      `7040c4195df216c900334ef62d5fdcf05c0601aa`.
- [x] Zero-WAM source pinned to revision
      `08e2c4ae41e2b63573a299825cebe6753481407c`.
- [x] Dataset split frozen: 1,109 training episodes / 61 tasks and 50 validation
      episodes / 5 task-disjoint tasks.
- [x] Robot preprocessing frozen: each 640×480 camera directly resized to 224×288,
      latent 14×18 per camera, front then wrist.
- [x] Human ICL preprocessing frozen: 320×448, latent 20×28, 12 FPS, detailed
      human-video text.
- [x] Action interface frozen: absolute targets, channels `[0,1,2,3,4,28]`, eight
      controls per latent frame, physical-bound normalization.
- [x] Training objective frozen: full episodes, random block-causal chunk size 1–4,
      four MCP/IFP modules, weights `[0.5,0.25,0.15,0.1]`.
- [x] Batch frozen: 1 episode/rank × 8 ranks × 8 accumulation microsteps = 64.
- [x] One-host topology verified: one Modal `H100:8` container, ranks/devices 0–7;
      64 MiB all-reduce ≈0.360 ms FP32 / ≈0.342 ms BF16.
- [x] Longest-64 full-episode update passed with every-microstep reduce-scatter and
      `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- [x] User approved the finalized settings and preparation for training, contingent
      on the mandatory gates below.

## Production training implementation

- [ ] Implement every-microstep FSDP gradient reduce-scatter in the production trainer;
      do not use upstream no-sync accumulation.
- [ ] Regression-test gradient equivalence against averaged eight-microstep accumulation.
- [x] Implement deterministic distributed task-balanced sampling: uniform task, then
      uniform episode within task.
- [x] Persist and restore sampler RNG/state exactly.
- [x] Track exact cumulative exposure count for every source/task/episode; persist
      per-rank counters in resumable state and publish a checkpoint-level aggregate
      ledger plus compact WandB summaries.
- [ ] Save resumable model, AdamW optimizer, LR scheduler, optimizer step, and sampler
      state at every checkpoint.
- [x] Pass a CPU checkpoint round-trip test proving the next sampled IDs, LR, and optimizer
      state match an uninterrupted run.
- [ ] Implement inline eight-GPU deterministic SFT validation at steps
      `0,500,1000,…,4000`.
- [ ] Prove inline validation performs no optimizer/scheduler update and restores model
      training mode plus training RNG/sampler state.
- [ ] Implement rank-0-only WandB logging with distributed metric reduction.
- [ ] Persist the WandB run ID in checkpoints and test `resume="must"`.
- [ ] Implement detached/persistent Modal execution with `max_containers=1`, an
      eight-H100 hard ceiling, durable logs, and an available stop command.

## Final latent data

- [x] Regenerate all training robot latents at 224×288 → 14×18 per camera.
- [x] Regenerate all validation robot latents at the same geometry.
- [x] Regenerate all human latents at 320×448 → 20×28 using detailed human-video text.
- [x] Verify exactly 1,109 training and 50 validation episodes each resolve to two robot
      camera latents, one human latent, actions, masks, and text embeddings.
- [x] Verify latent frame IDs, temporal boundaries, camera order, hashes, finite values,
      and expected action horizon of eight.
- [ ] Pass deterministic loader smoke across multiple tasks and short/median/longest
      episodes using complete trajectories.
- [ ] Verify training, validation, and online rendering use identical robot preprocessing.

## Deterministic validation and rollout baseline

- [ ] Freeze the 50-episode SFT-loss manifest: episode IDs, segments, timesteps, and noise
      seeds.
- [ ] Verify primary SFT conditioning: human video + detailed human text enabled; short
      target-task text disabled.
- [ ] Implement and verify the closed-loop runner for all five held-out task classes.
- [ ] Freeze the 50-rollout manifest with simulator/model seeds, horizons, prompts,
      human-video hashes, and success predicates.
- [ ] Verify packaged validation trajectories can reach the configured success predicate.
- [ ] Exact-load audit the initialization checkpoint.
- [ ] Run and publish initialization deterministic SFT validation loss at optimizer step 0.
- [ ] Run and publish all 50 initialization closed-loop rollouts.
- [ ] Freeze initialization summaries, per-episode artifacts, hashes, and WandB run IDs.

## Final preflight

- [ ] Run a short representative task-balanced unprofiled throughput benchmark.
- [ ] Record warm optimizer steps/s, samples/s, GPU utilization, peak memory, and skipped
      steps.
- [ ] Project H100-hours and expected wall time through steps 500 and 4,000.
- [ ] Confirm projected cost/throughput is acceptable.
- [ ] Confirm all gradients/losses are finite and no unexpected clipping/skips occur.
- [ ] Confirm WandB charts and durable JSON artifacts agree.
- [ ] Confirm repository revision and config hash are recorded and working tree changes do
      not affect mounted training code.
- [ ] Confirm no other Modal GPU app is active immediately before launch.
- [ ] Recheck the one-container/eight-H100 resource ceiling.

## Launch record

- [ ] All mandatory gates above passed.
- [ ] Final launch command recorded:
      `____________________________________________________________`
- [ ] Git revision recorded: `____________________________`
- [ ] Frozen config SHA256 recorded: `________________________________________`
- [ ] Modal app ID recorded: `____________________________`
- [ ] Modal function-call ID recorded: `____________________________`
- [ ] WandB train run ID recorded: `____________________________`
- [ ] Durable run root recorded: `____________________________________________`
- [ ] Optimizer training started.

## Live monitoring

- [ ] Confirm first optimizer step and checkpoint writes without changing the frozen config.
- [ ] Monitor training/video/action/IFP/total losses, LR, gradient norm, skipped steps,
      throughput, GPU utilization, and allocated/reserved memory.
- [ ] At step 500, wait for checkpoint plus inline SFT validation to complete.
- [ ] Review step-500 training/validation trends, finite gradients, skipped steps,
      throughput/cost, task sampling, loader health, and conditioning.
- [ ] Stop the active Modal run immediately if unhealthy; otherwise record approval to
      continue without terminating the job.
- [ ] Record the step-500 decision and evidence:
      `________________________________________________________________________`

## Completion

- [ ] Evaluate deterministic SFT loss at every retained checkpoint.
- [ ] Select the checkpoint with the lowest finite macro-task SFT validation total under
      the frozen conditioning protocol.
- [ ] Run the same 50 closed-loop rollouts on the selected checkpoint.
- [ ] Publish overall, macro-task, per-task, confidence-interval, transition, and exact
      McNemar results.
- [ ] Publish final checkpoint hash and Hugging Face/WandB/durable-artifact references.

## Durable evidence roots

- Hugging Face latent release:
  `hf://buckets/akoniti/ICL-so101/zero_wam_latents_v1`
- Latent release: 3,477 `.pth` files / 20,562,195,257 bytes; 3,477/3,477 source-video
  hashes verified against local immutable inputs.
- Episode manifest SHA256:
  `f1acb7656b811644b1c6c3c429cbd86b78310a9d99fe5dac5f52d5747c601b98`
- Topology: `/sim/reports/topology.json`
- FSDP feasibility: `/sim/reports/full_episode_batch64/`
- Training run: `/sim/runs/<run_id>/`
- Local compact summaries: `outputs/zero_wam/sim_training/<run_id>/`
- Rollouts: `outputs/zero_wam/sim_val/<eval_id>/`
