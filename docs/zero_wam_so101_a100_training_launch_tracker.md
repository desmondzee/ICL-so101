# SO-101 Zero-WAM 8xA100 training launch tracker

Purpose: operational checklist for preparing, launching, monitoring, resuming, and
evaluating the finalized 4,000-step SO-101 Zero-WAM run on the current on-demand
single-host 8x NVIDIA A100-SXM4-80GB instance.

The canonical model, data, objective, and evaluation settings remain in
`docs/zero_wam_so101_final_training_config.md`. The earlier
`docs/zero_wam_so101_training_launch_tracker.md` records the H100/Modal preparation
history and is not the launch checklist for this host.

**Current state (2026-10-10): A100 data, environment, loader, production training,
checkpoint, resume, WandB, and deterministic held-out-loss paths are implemented.
Batch-64 and batch-16 qualification runs completed successfully. Validation smoke v1
exposed and fixed unequal FSDP forward counts; v2 is running with equal padded rounds.
The 4,000-step run has not started. Closed-loop initialization
rollouts remain launch-blocking.**

## Frozen experiment contract

- [x] Start checkpoint:
      `Robbyant-Research/zero-wam-pretrain@7040c4195df216c900334ef62d5fdcf05c0601aa`.
- [x] Zero-WAM source:
      `08e2c4ae41e2b63573a299825cebe6753481407c`.
- [x] Training split: 1,109 episodes, 61 tasks, 29 skill families.
- [x] Validation split: 50 episodes, five task-disjoint tasks, ten episodes per task.
- [x] Robot inputs: front then corrected wrist; direct bilinear 640x480 to 288x224
      (WxH); separate 18x14 latents.
- [x] Human ICL input: 448x320 (WxH), 12 FPS, 28x20 latent, detailed human-video text.
- [x] Actions: absolute joint targets in channels `[0,1,2,3,4]`, absolute gripper in
      channel `28`, fixed physical-bound normalization.
- [x] Objective: full episodes, random block-causal chunk size 1-4, video/action losses,
      and four MCP/IFP terms weighted `[0.5,0.25,0.15,0.1]`.
- [x] Optimizer protocol: AdamW, BF16 parameters/activations, FP32 loss reductions,
      LR `1e-4`, 200 optimizer-step warmup, constant LR, maximum 4,000 optimizer steps.
- [ ] Select the final effective batch. Batch 64 (eight ranks x eight accumulation
      microsteps) and batch 16 (eight ranks x two microsteps) both passed 16-step
      qualification; comparative noise evidence is recorded below.
- [x] Sampling: uniform over tasks then uniform over episodes within a task; no length
      bucketing and no temporal segmentation.
- [x] Checkpoints and inline validation: optimizer steps
      `0,500,1000,...,4000`; final selection by the lowest finite macro-task validation
      MCP/IFP-weighted total under the frozen conditioning protocol.
- [x] Primary validation conditioning: human video and detailed human text enabled;
      short target-task text disabled.

Changes to this contract require an explicit recorded decision before optimizer
training. Hardware-specific throughput, memory, paths, and launch mechanics are not
part of the frozen scientific contract.

## A100 host inventory

- [x] Verify exactly eight visible GPUs:
      `NVIDIA A100-SXM4-80GB`, compute capability 8.0, 81,920 MiB each.
- [x] Verify one host with CUDA device IDs 0-7.
- [x] Record `nvidia-smi topo -m`: every GPU pair reports `NV12`; the host NIC is
      reached through `PHB`.
- [x] Save compact machine evidence in
      `outputs/zero_wam/a100_preflight/host_inventory.json`: hostname, UTC time,
      platform, CPU/RAM/disk, Git state, `uv.lock` hash, package versions, GPU query,
      and topology.
- [ ] Save immutable machine evidence under the run root: hostname, UTC time,
      `nvidia-smi -q`, topology, driver, CUDA runtime, PyTorch, NCCL, FlashAttention,
      CPU, RAM, mount capacity, and Git revision/dirty state.
- [ ] Run an eight-rank NCCL sanity test and record FP32 and BF16 64 MiB all-reduce
      latency/bandwidth on this host.
- [ ] Confirm no unrelated process uses GPU memory or compute immediately before each
      benchmark and launch.
- [ ] Record the provider instance ID/type, hourly price, interruption policy, and
      persistent-storage behavior without storing credentials.

The prior H100 feasibility and topology results are historical evidence only. They do
not satisfy any A100 benchmark or launch gate.

## Data acquisition and layout

Expected immutable source prefixes:

| Purpose | Hugging Face bucket prefix | Required local path |
| --- | --- | --- |
| Training source | `hf://buckets/akoniti/ICL-so101/sim_train_v1` | `data/hf_bucket_ICL-so101/sim_train_v1` |
| Corrected training wrist | `hf://buckets/akoniti/ICL-so101/train_v2` | `data/hf_bucket_ICL-so101/train_v2` |
| Validation source | `hf://buckets/akoniti/ICL-so101/sim_val_v2` | `data/hf_bucket_ICL-so101/sim_val_v2` |
| Corrected validation wrist | `hf://buckets/akoniti/ICL-so101/val_v3` | `data/hf_bucket_ICL-so101/val_v3` |
| Final latents | `hf://buckets/akoniti/ICL-so101/zero_wam_latents_v1` | `data/hf_bucket_ICL-so101/zero_wam_latents_v1` |

- [x] Complete the resumable, non-deleting `sim_train_v1` download (2026-10-10):
      22,183 files / 21,415,571,323 bytes from the remote listing; sync exited 0.
- [x] Complete the resumable, non-deleting `zero_wam_latents_v1` download
      (2026-10-10): sync exited 0.
- [x] Download `train_v2`, `sim_val_v2`, and `val_v3`; do not silently substitute old
      wrist media.
- [x] Download and verify `zero_wam_loader_v1`: 4,974 files /
      5,671,305,086 bytes, 1,109 train + 50 validation samples, 61 + 5 tasks, and zero
      train/validation task overlap.
- [x] Verify `train_v2`, `sim_val_v2`, and `val_v3` are locally available for
      closed-loop evaluation; do not silently substitute old wrist media.
- [x] Verify downloaded loader/latent releases against release, audit, manifest, and
      remote file/byte counts.
- [x] Verify the local latent release contains exactly 3,477 `.pth` files totaling
      20,562,195,257 bytes: 2,318 robot-camera and 1,159 paired human-video artifacts.
- [x] Verify local latent manifest SHA256
      `f1acb7656b811644b1c6c3c429cbd86b78310a9d99fe5dac5f52d5747c601b98`.
- [x] Verify every one of 1,109 train and 50 validation episodes resolves to two robot
      latents, one human latent, actions, masks, task text, and detailed human text.
- [x] Confirm the published release records 3,477/3,477 source-video hash checks against
      immutable release inputs.
- [x] Download the Zero-WAM LeRobot v2.1 collection metadata/actions required by the
      loader; the latent bucket alone is not treated as a complete training dataset.
- [x] Keep downloads and generated caches outside Git; record exact paths and hashes in
      the run manifest.

## Checkpoint and software environment

- [x] Download the pinned initialization checkpoint to
      `data/models/zero-wam-pretrain` (23 files, approximately 35 GB).
- [x] Hash the complete model tree into each run's immutable config identity.
- [ ] Exact-load audit the initialization checkpoint; require no missing, unexpected,
      or mismatched model keys.
- [x] Pin the executable training environment in `.venv-zero-wam`: Python 3.12,
      PyTorch 2.9.0+cu126, Transformers 4.55.2, Diffusers 0.36.0,
      FlashAttention 2.8.3, LeRobot 0.3.3, and Datasets 3.6.0.
- [x] Record `uv.lock` and installed package
      versions, CUDA extensions, and environment variables.
- [x] Confirm PyTorch BF16, FSDP, NCCL training, and the pinned FlashAttention build work on
      A100 compute capability 8.0.
- [x] Set `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- [x] Set and record deterministic sampler and validation seeds.
- [ ] Confirm all run outputs, checkpoints, logs, and WandB metadata are written to
      storage that survives instance termination.

## A100 production implementation

- [x] Add `zero_wam/so101_a100_train.py`, a non-Modal production entry point for one-host
      `torchrun --standalone --nproc_per_node=8`.
- [x] Make model, converted data, latent, run, accumulation, and resume paths explicit
      CLI arguments;
      reject missing or inconsistent roots before allocating model memory.
- [x] Install deterministic task-balanced sampling in the production trainer.
- [x] Install every-microstep FSDP gradient synchronization; do not use upstream
      `no_sync` accumulation.
- [ ] Regression-test every-microstep reduce-scatter against correctly averaged
      eight-microstep gradients.
- [x] Integrate resumable checkpoints containing model, AdamW optimizer, LR scheduler,
      optimizer step, all rank RNG states, sampler state, exposure ledger, config hash,
      and WandB run ID.
- [x] Save checkpoints at optimizer boundaries; the CPU checkpoint round-trip proves
      matching next sample IDs, LR, optimizer, scheduler, RNG, and sampler state.
- [ ] Prove an actual eight-rank interrupted
      run produces the same next sample IDs, LR, and optimizer state after resume.
- [x] Implement rank-0-only WandB logging with distributed metric reduction and
      `resume="must"`.
- [ ] Implement append-only local JSONL metrics and tee stdout/stderr to durable logs.
- [x] Implement inline eight-rank deterministic validation at steps
      `0,500,1000,...,4000`.
- [ ] Complete the live validation smoke proving validation performs no optimizer or
      scheduler update and restores model mode, RNG, sampler, and loader state exactly.
- [ ] Add signal-aware shutdown: finish or discard the current microstep safely, save
      only at a clean optimizer boundary, and record the termination reason.
- [x] Provide explicit start, status, stop, and staged resume commands suitable for an
      on-demand instance and local-shell disconnect.
- [x] Remove Modal/H100 resource assumptions from the new launcher while leaving
      historical Modal utilities intact.

## Loader and objective gates

- [x] Run deterministic loader smoke tests over multiple tasks and short, median, and
      longest complete episodes.
- [x] Decode sampled IDs back to source task/episode IDs and confirm task-balanced
      sampling empirically.
- [x] Verify robot latent camera order and geometry: front then wrist, each 14x18.
- [x] Verify human latent geometry 20x28 and detailed-text embedding identity.
- [x] Verify active action mask is exactly `[0,1,2,3,4,28]`; all other channels must be
      masked from loss.
- [x] Verify eight controls per robot latent frame and complete-trajectory boundaries.
- [x] Verify training and validation preprocessing and normalization are identical.
- [x] Run 16-step multi-task qualification gates: finite forward/backward, finite gradients,
      expected active losses, and decreasing train loss without changing production
      hyperparameters except the bounded sample/step count.

## A100 memory and throughput gates

- [x] Run a complete effective-batch-64 optimizer update and a sustained 16-step
      batch-64 qualification on 8xA100 with the full MCP/IFP objective.
- [x] Require zero OOMs, finite losses/gradients, no skipped optimizer step, and memory
      headroom on every rank.
- [x] Observe 51-70 GiB per GPU during batch-64 qualification, with all eight A100s
      active and no OOM.
- [x] Run representative task-balanced unprofiled batch-64 and batch-16 qualifications.
- [x] Record optimizer-step timing, GPU utilization/memory, losses, gradient norm, LR,
      and exact episode exposure.
- [ ] Add durable automated samples/s, tokens/s, GPU power,
      host I/O, data wait, and communication time.
- [ ] Compare A100 results with the historical H100 evidence only as context; do not
      reuse H100 throughput or cost projections.
- [ ] Project wall time and A100-hours through steps 500 and 4,000 using warm steady-state
      measurements.
- [ ] Project total cost using the recorded provider price, including inline validation
      and checkpoint overhead.
- [ ] Obtain explicit approval of measured throughput, projected duration, and cost
      before optimizer training.

### Batch qualification results

Both comparisons used eight ranks, seed 42, the same initialization, task-balanced
sampler, optimizer, 200-step warmup, and 16 optimizer steps. They differ only in
accumulation and therefore episode exposure.

| Effective batch | Accumulation | Episodes | Wall/compute evidence | Result | WandB |
| ---: | ---: | ---: | --- | --- | --- |
| 64 | 8 | 1,024 | 16 steps completed; steady steps approximately 170-190 s | finite, no skips; action `0.0070 -> 0.0032` | `352ef90f` |
| 16 | 2 | 256 | 16 steps completed; steady steps approximately 32-60 s | finite, no skips; action `0.0067 -> 0.0037` | `66709412` |

Detrended coefficient of variation, batch 64 versus batch 16:

| Metric | Batch 64 | Batch 16 |
| --- | ---: | ---: |
| Video loss | 2.75% | 8.09% |
| Action loss | 9.00% | 13.02% |
| MCP weighted loss | 1.38% | 3.78% |
| Gradient norm | 11.75% | 17.77% |

Batch 16 is materially noisier but remained stable. Mean rank-max gaps increased from
15.8% to 32.0% for video loss and from 20.4% to 37.9% for action loss. Final batch
selection must account for the changed exposure budget: 4,000 steps process 256,000
episodes at batch 64 but 64,000 at batch 16.

If the longest episodes do not fit, stop for an explicit scientific-protocol decision.
Do not silently reduce the effective batch, shorten episodes, disable MCP/IFP, enable
length bucketing, or change precision.

## Deterministic validation and baseline

- [x] Implement immutable generation of the 50-episode validation-loss manifest:
      episode/task IDs and deterministic per-episode diffusion/noise seeds; refuse
      manifest drift on rerun.
- [x] Configure released-faithful conditioning: human latent and detailed human text on;
      short target-task text off.
- [ ] Complete the active `a100-validation-smoke` run and publish initialization
      validation loss at optimizer step 0 with overall,
      macro-task, and per-task metrics with the frozen manifest hash.
- [ ] Implement and verify closed-loop evaluation for all five held-out task classes.
- [ ] Freeze the 50-rollout manifest with simulator/model seeds, horizons, prompts,
      source hashes, and success predicates.
- [ ] Replay packaged validation actions to prove each configured success predicate is
      reachable.
- [ ] Run all 50 initialization closed-loop rollouts before training.
- [ ] Freeze initialization summaries and artifact hashes before allowing optimizer
      step 1.

## Final preflight

- [ ] All data, latent, loader, checkpoint, topology, memory, throughput, validation,
      logging, persistence, stop, and resume gates above pass.
- [ ] Working tree state and all code/config hashes are recorded.
- [ ] No GPU processes other than the launch are active.
- [ ] Sufficient durable disk remains for checkpoints, validation artifacts, logs, and
      temporary atomic writes.
- [x] Background downloads and verification jobs have completed successfully.
- [x] WandB connectivity is verified; strict same-ID `resume="must"` is implemented but
      still needs an eight-rank live resume test.
- [x] A detached `nohup` torchrun process owns each run; training does not depend on an
      SSH session.
- [ ] The user explicitly approves the measured A100 launch configuration and cost.

## Launch record

### Qualification run history

| Run | Effective batch | Outcome | Evidence |
| --- | ---: | --- | --- |
| `a100-smoke-1step` | 64 | Failed before training: Arrow cache produced by Datasets 4.8.5 was incompatible with pinned 3.6.0 | Preserved `failed.json`; caches regenerated under `.venv-zero-wam` |
| `a100-smoke-1step-v2` | 64 | Success: one optimizer step, full checkpoint/state, WandB | run `c8fa34cd` |
| `a100-multihour-16step` | 64 | Success: 16/16 steps, no nonfinite/skipped step, full resumable checkpoint | run `352ef90f` |
| `a100-multihour-64step` | 64 | Intentionally stopped around step 7 to avoid redundant qualification compute; no checkpoint | run `8663a6b8` |
| `a100-batch16-16step` | 16 | Success: 16/16 steps, no nonfinite/skipped step, full resumable checkpoint | run `66709412` |
| `a100-validation-smoke` | 16 | Stopped after diagnosing an NCCL timeout: 50 samples assigned 7 calls to two ranks and 6 to six ranks, violating equal FSDP collective counts | run `a3518ee8` |
| `a100-validation-smoke-v2` | 16 | Running with seven forward calls per rank; six duplicate padding calls are excluded from the 50-sample report | run ID pending |

- [ ] Launch command:
      `________________________________________________________________________`
- [ ] UTC launch time: `________________________________________________________`
- [ ] Provider instance ID/type: `______________________________________________`
- [ ] Git revision and dirty-state hash: `______________________________________`
- [ ] Frozen config SHA256: `__________________________________________________`
- [ ] Validation manifest SHA256: `____________________________________________`
- [ ] Run root: `______________________________________________________________`
- [ ] Log path: `______________________________________________________________`
- [ ] Supervisor/session/service ID: `_________________________________________`
- [ ] WandB run ID: `__________________________________________________________`
- [ ] Initialization checkpoint audit: `______________________________________`
- [ ] Optimizer training started.

## Live monitoring

- [ ] Confirm the first optimizer step, exposure counters, metrics, and atomic
      checkpoint path without changing the frozen configuration.
- [ ] Monitor losses, LR, gradient norm, skipped/nonfinite steps, throughput, per-rank
      memory/utilization, data wait, disk capacity, and estimated remaining cost.
- [ ] Confirm checkpoints and logs are visible from durable storage independently of
      the training process.
- [ ] At step 500, require checkpoint completion and inline validation completion.
- [ ] Review step-500 training/validation trends, finite gradients, skipped steps,
      throughput, cost, loader health, conditioning, and resume artifacts.
- [ ] Stop immediately on nonfinite state, repeated skipped steps, corrupted artifacts,
      conditioning/data mismatch, uncontrolled cost, or loss of durable persistence.
- [ ] Record the step-500 continue/stop decision and evidence:
      `________________________________________________________________________`

## Completion

- [ ] Complete or explicitly account for optimizer steps 1-4,000.
- [ ] Verify every retained checkpoint and resumable state artifact.
- [ ] Evaluate deterministic validation loss at every retained checkpoint.
- [ ] Select the checkpoint with the lowest finite macro-task MCP/IFP-weighted
      validation total under the frozen conditioning protocol.
- [ ] Exact-load audit the selected checkpoint.
- [ ] Run the same frozen 50 closed-loop trials on the selected checkpoint.
- [ ] Publish overall, macro-task, and per-task success with confidence intervals.
- [ ] Publish paired fail-to-success and success-to-fail transitions and exact McNemar
      result.
- [ ] Publish final checkpoint/config/manifest hashes, WandB reference, A100-hours,
      total cost, interruptions/resumes, and durable artifact locations.

## A100 evidence roots

```text
data/hf_bucket_ICL-so101/                         immutable downloaded releases
outputs/zero_wam/a100_preflight/                  compact host/data/loader evidence
outputs/zero_wam/sim_training/<run_id>/           compact run summaries
outputs/zero_wam/sim_val/<eval_id>/               validation and rollout summaries
<durable-run-root>/reports/                       full topology/benchmark reports
<durable-run-root>/runs/<run_id>/                 checkpoints, state, metrics, logs
```

Large datasets, latents, model weights, checkpoints, and rollout media remain outside
Git. Commit only code, compact manifests, reports, and this tracker.
