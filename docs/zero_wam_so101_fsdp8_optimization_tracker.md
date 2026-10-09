# Zero-WAM SO-101 8×H100 FSDP2 optimization tracker

## Objective

Review and optimize full-episode SO-101 Zero-WAM training on one eight-GPU Modal host while preserving model behavior, loss definitions, conditioning, task-balanced sampling, learning rate `1e-4`, and 4,000 optimizer steps.

The required batch contract is one episode per GPU, eight GPUs, and eight gradient-accumulation microsteps: effective batch size `1 × 8 × 8 = 64` episodes per optimizer step.

No paid benchmark or optimizer run may start until its exact command, maximum duration, output location, and stop conditions are recorded here. The hard resource limit is one Modal function container with `gpu="H100:8"`; eight independent GPU containers are forbidden for distributed training.

## Current assessment

| Area | Current implementation | Status / risk |
|---|---|---|
| Modal topology | Smoke train/eval functions request `gpu="H100:8"` and launch `torchrun --nproc_per_node=8` inside that one container. | Correct shape for single-node training, but topology and NVLink/NVSwitch have not been measured on the actual allocation. The full simulated-data driver does not yet expose a training function. |
| Process topology | Eight local ranks, one process per GPU, NCCL process group. | Structurally correct; verify rank→GPU affinity and NCCL transport. |
| FSDP generation | PyTorch composable `fully_shard` (FSDP2), nested on attention, cross-attention, FFN, block, MCP blocks and root model. | Confirm no redundant collectives from over-wrapping by profiler trace. |
| Parameter precision | `MixedPrecisionPolicy(param_dtype=torch.bfloat16)`. | BF16 parameter all-gathers expected; verify trace and NCCL payloads. |
| Gradient reduction | `reduce_dtype=torch.float32`. | FP32 reduce-scatter is numerically conservative but doubles gradient communication bytes versus BF16. Keep as baseline; only test BF16 reduction as an explicit equivalence experiment. |
| Resharding | `reshard_after_forward=True` at every FSDP unit. | Memory-efficient, but forces repeated parameter all-gathers across eight accumulation microsteps. Highest-priority FSDP tuning candidate after memory measurement. |
| Activation checkpointing | Every main Transformer block and every MCP block is checkpoint wrapped with `preserve_rng_state=False`. | Preserves memory but recomputes block forward in backward. Measure recompute cost; do not remove before full-episode memory is proven. |
| Gradient accumulation | Trainer divides every loss by accumulation count and calls `set_requires_gradient_sync(False)` on microsteps 1–7, then `True` on microstep 8. | Correctly suppresses gradient synchronization/reduce-scatter on non-final microsteps. Parameter all-gathers are still required and must not be suppressed. Verify one reduce-scatter phase per optimizer step in trace. |
| Current accumulation default | Upstream preset uses 1. | Must be frozen to 8 for this run. |
| Dataset unit | SO-101 converter writes one `action_config` covering each complete episode; latent loader metadata therefore resolves to complete trajectories. | Full-episode representation is already the default. Verify loader does not crop or segment downstream. |
| Episode lengths | Variable; no length-aware distributed batching is currently evident. | Long-rank stragglers can idle the other seven GPUs at each collective. Add deterministic task-balanced length bucketing only after baseline measurement. |
| Sampling | Intended uniform task then uniform episode within task. | Must be verified against the actual distributed sampler; equal weighting of task datasets is not sufficient evidence by itself. |
| Logging | Loss/WandB hooks exist; no profiler or topology artifact is frozen. | Add durable topology, profiler, memory, utilization, samples/s and tokens/s reports. |
| Checkpoint resume | Model checkpoint save is active; optimizer-state save is commented out upstream. | Blocking reliability issue for a long 4,000-step run. Restore resumable sharded optimizer/training state without changing model math. |

## Frozen target configuration

```yaml
modal:
  instances: 1
  gpu_request: H100:8
  distributed_nodes: 1
  processes_per_node: 8
  backend: nccl

batch:
  episodes_per_gpu_microstep: 1
  world_size: 8
  gradient_accumulation_steps: 8
  effective_episodes_per_optimizer_step: 64

optimizer:
  type: AdamW
  learning_rate: 1.0e-4
  optimizer_steps: 4000
  warmup_steps: 200
  gradient_max_norm: 1.0

precision:
  parameter_compute: bfloat16
  gradient_reduce_baseline: float32

sequence:
  preferred_unit: complete_episode
  robot_source_rate_hz: 30
  robot_rgb_stride: 2
  actions_per_vae_latent_frame: 8
  inference_setting_is_out_of_scope: true
```

At this configuration, training consumes `64 × 4,000 = 256,000` episode samples across all ranks, before accounting for skipped optimizer steps.

## Phase 1 — topology and NCCL transport

- [ ] Add a topology-only `gpu="H100:8"` Modal function; do not allocate eight separate containers.
- [ ] Save full `nvidia-smi -L` output.
- [ ] Save full `nvidia-smi topo -m` output.
- [ ] Save GPU model, memory size, driver and CUDA runtime.
- [ ] Record whether all GPU pairs report NVLink/NVSwitch-class connectivity (`NV#`/`NVSwitch`) rather than `PIX`, `PHB`, `NODE` or `SYS` paths.
- [ ] Record GPU-to-NIC affinity and visible NICs.
- [ ] Run `NCCL_DEBUG=INFO` initialization and save selected transports.
- [ ] Confirm one hostname/container ID for all eight ranks.
- [ ] Confirm local ranks 0–7 bind uniquely to CUDA devices 0–7.
- [ ] Run an eight-rank all-reduce microbenchmark for FP32 and BF16 payloads representative of gradient buckets.
- [ ] Record bus bandwidth and algorithm bandwidth; compare with topology-appropriate expectations.

### Acceptance gate

Training may proceed only if all ranks share one host and NCCL uses the fastest available intra-host path. If the host lacks NVLink/NVSwitch, record the measured bandwidth and expected training penalty before deciding whether to accept that allocation.

## Phase 2 — FSDP2 communication audit

- [x] Confirm composable FSDP2 via `torch.distributed.fsdp.fully_shard`.
- [x] Confirm BF16 mixed-precision parameter policy.
- [x] Confirm baseline FP32 gradient reduction policy.
- [x] Confirm activation checkpointing on main and MCP Transformer blocks.
- [x] Confirm non-final accumulation microsteps call `set_requires_gradient_sync(False)`.
- [x] Confirm final accumulation microstep reenables gradient synchronization before backward.
- [ ] Trace one complete eight-microstep optimizer update.
- [ ] Count and time parameter all-gather collectives per microstep.
- [ ] Count and time gradient reduce-scatter collectives; require none on microsteps 1–7 and exactly the expected final-microstep reductions.
- [ ] Verify suppressed gradient synchronization does not suppress parameter gathering.
- [ ] Measure communication bytes in BF16 parameter all-gathers and FP32 gradient reduction.
- [ ] Detect redundant collectives caused by nested attention/FFN/block/root wrapping.
- [ ] Verify activation-checkpoint recomputation does not unexpectedly duplicate communication beyond FSDP2 requirements.

### FSDP experiments, in order

1. **Baseline:** `reshard_after_forward=True`, FP32 reduction, activation checkpointing enabled.
2. **Accumulation-aware resharding:** retain full parameters between forward/backward or across accumulation where FSDP2 safely permits it; profile memory before accepting.
3. **FSDP wrap simplification:** only if traces prove nested wrapping creates avoidable collectives.
4. **BF16 gradient reduction:** only after topology and reshard optimizations, and only with finite-gradient/loss equivalence evidence. This is not the default because it changes reduction precision.
5. **Selective activation checkpointing:** only if full episodes have sufficient memory headroom and recomputation is a dominant bottleneck.

## Phase 3 — full-episode memory feasibility

### Bounded batch-64 benchmark — implemented, pending execution

`zero_wam/so101_full_episode_benchmark.py` (torchrun, 8 ranks) +
`zero_wam/modal_so101_benchmark.py` (one `H100:8` Modal function, `max_containers=1`,
2 h timeout, 480 GiB). Protocol:

- Frozen top-64 selection: all 1,109 train `meta/episodes.jsonl` rows sorted by source
  `length` desc, ties by (task, episode); `selection_sha256` recorded in the report.
- Upstream `Trainer` on the `va_robotwin_train_cfg` contract with SO-101 dataset paths,
  batch 1, gradient accumulation 8, LR 0 (real AdamW step executes for optimizer-state
  allocation; weights preserved), FSDP2/MCP/IFP/activation-checkpointing/FP32 reduction
  unchanged, `drop_icl=0`, `droptext=0`, `num_steps=1`, checkpoint saving unused.
- In-memory shape adaptation only: robot latent → `[48, F, 14, 36]`, human ICL →
  `[48, T, 20, 28]`; provisional 256×256/320×480 latents are not regenerated or saved.
- Item assignment `microstep*8 + rank` so each concurrent group has adjacent sorted
  lengths; unresolved selected episodes (five missing top-64 per
  `/sim/reports/top64_longest_inventory.json`) are synthesized as zero tensors at the
  exact stride-2 / 1+4k latent frame count with mask channels `[0,1,2,3,4,28]`, marked
  `synthetic`.
- One real `_train_step(batch, microstep)` per microstep 0–7 (sync suppressed until the
  8th, real optimizer boundary), timed per microstep. **Unprofiled pass** — no
  `torch.profiler`; `profiler_enabled: false` in the report and no NCCL/operator claims.
- Rank-local stage logging (selection, trainer init, item resolution, per-microstep,
  gather) with per-rank `trainer_init_s` / `item_resolution_s` and separate total vs
  compute wall time; rank-0 `nvidia-smi` 200 ms sampling starts only at compute start
  so utilization excludes initialization.
- Dataset construction restricted to the loader tasks — selected tasks having at
  least one complete latent episode per the durable
  `/sim/reports/top64_longest_inventory.json` (currently 7; missing selected
  episodes/tasks are synthesized at exact target shape) — by excluding all other
  task names through the upstream loader; dataset index cache enabled.
- **CPU index preflight first** (`--prepare-indexes`, `cpu=32`, 128 GiB, data volume
  only): builds the frozen selection, validates the top-64 inventory (one row per
  selected episode, complete iff all artifacts exist), constructs the upstream
  `MultiICLLeRobotLatentDataset` for the loader tasks, writes durable caches and an
  atomic `index_preflight.json` (selected tasks, loader tasks, missing episode IDs,
  inventory path). The `benchmark` entrypoint runs this synchronously before the H100
  function — zero GPU allocation if it fails. The GPU run re-derives loader tasks and
  missing IDs from the inventory and verifies them against the preflight report
  *before* `init_distributed`; no rank-zero cache build or barrier.
- The wrapper streams torchrun output line-by-line to Modal stdout and `torchrun.log`
  (no RAM buffering), enforces a 90 min deadline (terminate then kill), and always
  commits the volume.
- Outputs under `/sim/reports/full_episode_batch64/`: `benchmark_report.json`
  (atomic), `top64_selection.json`, `nvidia_smi_samples.csv`, `torchrun.log`; volume
  committed even on failure.
- **Follow-up (only after feasibility succeeds):** a separate lightweight profiler
  pass profiling the final sync microstep only — `record_shapes=False`,
  `profile_memory=False`, aggregated operator/NCCL table, no Chrome trace initially.

Observed no-sync baseline on one Modal `H100:8` host:

- CPU cache preflight completed; GPU trainer initialization was approximately 41 seconds.
- With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the first concurrent group
  of the eight longest episodes completed forward/backward in 205.3–205.4 seconds.
- Microstep 2 OOMed during MCP forward while gradients from microstep 1 were retained:
  GPUs reported approximately 76.7–77.5 GiB in use, 73.3–73.9 GiB allocated, and failed
  requests of 2.84–2.93 GiB.
- Therefore accumulation 8 with FSDP gradient synchronization disabled on microsteps
  1–7 is not memory-feasible for the longest episodes.

Next test: preserve effective batch 64 but force reduce-scatter after every microstep so
only sharded gradients are retained. This trades additional intra-host communication for
memory and preserves the averaged objective.

Synchronized-accumulation result:

- **Passed:** one complete effective-batch-64 optimizer boundary over the frozen 64
  longest episodes on one Modal `H100:8` host.
- Selection SHA256:
  `fe7f63304ba8fa1fd627a2e3588a8b9e5c9c26f6890ff6479bc9a300aa0a5339`.
- Eight ranks, eight microsteps, reduce-scatter after every microstep, LR 0 with a real
  AdamW step and optimizer-state allocation; no step skipped and all gradient norms finite.
- Trainer initialization: approximately 35.9 seconds/rank.
- Compute wall time: **351.4 seconds**; total benchmark wall time **392.2 seconds**;
  **0.182 samples/s** for this cold worst-case update.
- First longest microstep: approximately 214.5–215.0 seconds; later microsteps ranged
  approximately 11.3–39.3 seconds, with the final optimizer boundary 27.1–27.9 seconds.
- Peak allocated: **50.4–55.7 GiB/rank**. Peak reserved: **62.2–67.4 GiB/rank**.
  End allocated after optimizer step: approximately **14.8 GiB/rank**.
- Sampled mean GPU utilization: **32.7–39.3%** by GPU; p50 0%, p95 100%. The workload is
  bursty and does not sustain high utilization despite fast intra-host collectives.
- Five unavailable selected episodes were represented by exact-length synthetic tensors;
  the other 59 used real latent/action/ICL payloads. This is a shape/memory/throughput test,
  not a loss-correctness result.
- Durable report:
  `/sim/reports/full_episode_batch64/benchmark_report.json`.

Conclusion: effective batch 64 with full episodes is memory-feasible **only** when FSDP
reduce-scatter runs on every accumulation microstep and expandable CUDA allocator segments
are enabled. The production trainer must not suppress gradient synchronization for this
configuration. Status: **memory feasibility passed; throughput optimization pending**.

Build a frozen episode-length inventory from latent token shapes and actions. Select at minimum:

- [ ] Median-length episode.
- [ ] Representative episode near p90.
- [ ] Representative episode near p99.
- [ ] Longest training episode.
- [ ] Longest episode from a distinct task family if the global longest is atypical.

For each episode, run one forward/backward microstep at batch 1 per rank and record:

- [ ] Robot latent tokens by camera and combined.
- [ ] Human ICL tokens.
- [ ] Action tokens.
- [ ] Total attention tokens.
- [ ] Allocated and reserved memory before forward, after forward and after backward.
- [ ] Peak allocated and peak reserved memory.
- [ ] Forward, backward and optimizer-boundary duration.
- [ ] Whether all ranks finish without OOM.

Then run one complete eight-microstep accumulation update for representative and longest cases. Reset CUDA peak-memory statistics between cases.

### Memory gate

- Prefer complete trajectories.
- Require a safety margin after the longest successful case; do not treat a near-capacity single pass as production-safe.
- If only extreme tails fail, first evaluate deterministic length-aware batching and accumulation-aware reshard tradeoffs.
- Introduce temporal segmentation only if complete episodes remain infeasible after topology/FSDP/memory optimization. Any segmentation proposal requires separate approval because it changes training examples.

## Phase 4 — task-balanced length bucketing

- [ ] Measure episode and token-length distributions by task.
- [ ] Measure baseline per-rank length spread and collective idle time.
- [ ] Design deterministic distributed batches whose eight simultaneous rank samples have similar token lengths.
- [ ] Preserve uniform task probability first, then sample an episode from a compatible length bucket within the selected task.
- [ ] Preserve one unique sample per rank where possible.
- [ ] Make sampler state checkpointable and seed-controlled.
- [ ] Verify observed task frequencies against uniform expectation.
- [ ] Verify no validation episode enters training.
- [ ] Compare padding, rank skew and samples/s against the unbucketed baseline.

Bucketing must not silently change task weights. If perfect task balance conflicts with tight length matching, report the tradeoff rather than changing the distribution implicitly.

## Phase 5 — profiler benchmark

Use a bounded `torch.profiler` schedule that excludes startup/model loading and captures representative steady-state microsteps plus one optimizer boundary. Export Chrome traces and aggregated tables to the durable Modal volume.

Required metrics:

- [ ] Forward wall time per microstep.
- [ ] Backward wall time per microstep.
- [ ] Optimizer-boundary wall time.
- [ ] NCCL all-gather time and bytes.
- [ ] NCCL reduce-scatter/all-reduce time and bytes.
- [ ] Activation-checkpoint recomputation time.
- [ ] Data-loading and host-to-device time.
- [ ] Per-rank peak allocated/reserved memory.
- [ ] Per-GPU utilization and memory utilization sampled during the benchmark.
- [ ] Episodes/s and optimizer steps/s.
- [ ] Robot, human, action and total tokens/s.
- [ ] Rank skew: fastest versus slowest forward/backward time.

Benchmark matrix:

| Case | Episode lengths | Accumulation | Purpose |
|---|---|---:|---|
| A | Representative, matched across ranks | 8 | Stable baseline |
| B | Longest feasible, matched across ranks | 8 | Memory and worst-case compute |
| C | Naturally sampled, unbucketed | 8 | Real rank-skew baseline |
| D | Naturally sampled, length-bucketed | 8 | Bucketing improvement |
| E | Representative with selected FSDP optimization | 8 | Communication optimization |

## Ranked optimization hypotheses

These rankings are provisional until profiler evidence exists.

| Rank | Optimization | Expected effect | Risk |
|---:|---|---|---|
| 1 | One `H100:8` host with verified NVLink/NVSwitch and NCCL transport | Largest communication improvement versus eight networked containers; required architecture | Low |
| 2 | Length-bucket simultaneous rank samples while retaining task balance | Reduces straggler/collective idle time for variable full episodes | Low–medium; sampler correctness |
| 3 | Accumulation-aware FSDP reshard policy | Avoids repeated parameter all-gathers across eight microsteps | Medium–high memory cost; must profile longest episodes |
| 4 | Remove demonstrably redundant nested FSDP wrapping | Fewer small collectives and launch overhead | Medium; can alter memory scheduling |
| 5 | Data-loader/cache/prefetch tuning after communication stalls are resolved | Hides latent and parquet I/O | Low |
| 6 | Selective activation checkpointing | Reduces recomputation if memory headroom exists | Medium–high memory cost |
| 7 | BF16 gradient reduce-scatter | Roughly halves gradient communication bytes | Medium numerical risk; requires equivalence gate |
| 8 | Temporal segmentation | Reduces memory substantially | High behavior change; last resort and out of current scope |

## Deliverables

- [ ] `topology.txt`: `nvidia-smi topo -m`, device and host identity.
- [ ] `nccl_info.log` and collective bandwidth results.
- [ ] FSDP2 collective audit for one eight-microstep update.
- [ ] Episode token/length inventory.
- [ ] Representative and longest-episode memory feasibility report.
- [ ] Baseline and optimized profiler traces.
- [ ] Task-balance and length-bucketing audit.
- [ ] Ranked bottleneck report with measured expected throughput improvements.
- [ ] Final launch configuration and projected compute consumption.
- [ ] Resumable checkpoint proof including optimizer, scheduler, sampler and optimizer-step state.

## Current blockers

1. Human latent text embeddings must be regenerated from detailed human-video descriptions before final training data are frozen.
2. Latent coverage is incomplete and has not passed final inventory verification.
3. The full simulated-data Modal training entrypoint and bounded profiler entrypoint are not yet implemented.
4. Actual Modal H100 topology has not been measured.
5. Longest full-episode memory feasibility is unknown.
6. Task-balanced distributed length bucketing is not implemented.
7. Upstream optimizer-state checkpoint save is disabled and must be restored for a reliable 4,000-step run.
