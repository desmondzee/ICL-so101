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
