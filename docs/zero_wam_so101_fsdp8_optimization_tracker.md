# SO-101 Zero-WAM FSDP8 optimization tracker

Objective: train on one Modal host (`gpu="H100:8"`) over full episodes with
effective batch 64 and learning rate `1e-4` for 4,000 optimizer steps, preserving the
upstream loss/objective/conditioning. Canonical configuration lives in
`docs/zero_wam_so101_final_training_config.md`; this file records measured evidence,
results, and the remaining launch gates.

## Final decisions

```yaml
host: 1                                    # single Modal container, world across GPUs
world_size: 8                              # 8 ranks, torchrun
batch_size_per_rank: 1                     # upstream Trainer constraint
gradient_accumulation_steps: 8             # effective batch 64
precision:
  parameters_and_activations: bfloat16
  fsdp_gradient_reduce_dtype: float32
fsdp_reduce_scatter_each_microstep: true   # measured memory requirement
suppress_gradient_sync_during_accumulation: false
reshard_after_forward: true                # FSDP2 default, retained
activation_checkpointing: true             # main blocks + MCP blocks
allocator_env: "PYTORCH_ALLOC_CONF=expandable_segments:True"
length_bucketing: false
full_episodes: true                        # no temporal segmentation needed for memory
```

## Verified topology and FSDP behavior

- Single hostname; CUDA devices 0–7, each H100 (80 GB), per the
  `/sim/reports/topology.json` probe. 64 MiB all-reduce measured ~0.360 ms FP32 /
  ~0.342 ms BF16 — intra-host class. `nvidia-smi topo -m` device labels unavailable
  on the host; NVLink/NVSwitch matrix not enumerated.
- Upstream `Trainer` shards with FSDP2 `fully_shard`, BF16 params, FP32 gradient
  reduction, activation checkpointing. Parameter all-gather still happens every
  microstep.
- Upstream `set_requires_gradient_sync(False)` accumulation behavior was measured
  (below) and identified as a memory blocker: grads held unreduced across microsteps
  peak past device capacity.

## Longest-64 feasibility run

Frozen selection: all 1,109 train `meta/episodes.jsonl` rows sorted by source
`length` descending (ties: task, episode), top 64 — covering 8 tasks. Exact list
and `selection_sha256` recorded in
`/sim/reports/full_episode_batch64/top64_selection.json` and `benchmark_report.json`.

- 59 real episodes + 5 exact-shape synthetic (missing latents per
  `/sim/reports/top64_longest_inventory.json`; loader tasks = 7 with complete
  samples).
- Tensor shapes adapted in memory to final geometry: robot `[48, F, 14, 36]`,
  human ICL `[48, T, 20, 28]`; provisional 256×256/320×480 latents not re-encoded.
- LR 0 with a real AdamW boundary (optimizer state allocated, weights preserved);
  microstep `i` gets sorted item `i*8 + rank`; not a loss-correctness test.

## Results

| Variant | Outcome | Memory (GiB, per rank) | Timing |
|---|---|---|---|
| No-sync accumulation (upstream) | **OOM on microstep 2** | in use 76.7–77.5; allocated 73.3–73.9 | step1 205.3–205.4 s; step2 OOM |
| Reduce-scatter every microstep + expandable segments | **passed, all 64 items, optimizer step executed** | peak alloc 50.4–55.7; peak reserved 62.2–67.4; end ~14.8 | init 35.9 s; compute 351.4 s; total 392.2 s; 0.182 samples/s |

Passing run details: cold first microstep 214.5–215 s; subsequent 11.3–39.3 s;
final (optimizer boundary) 27.1–27.9 s. Gradient norms finite (0.261–0.412),
no optimizer step skipped. GPU utilization mean 32.7–39.3%, p50 0 / p95 100 —
bursty.

## Conclusions

- Effective-batch-64 full episodes are memory-feasible **only** with
  every-microstep gradient sync/reduce-scatter plus `expandable_segments`.
- Communication cost of per-step reduction is acceptable at intra-host bandwidth.
- Utilization is poor/bursty; throughput (0.182 samples/s) is the remaining
  concern, not memory.
- Full episodes retained; no temporal segmentation required on memory evidence;
  no length bucketing (per final config).

## Required production changes (before launch)

- Implement every-microstep gradient sync/reduce-scatter in the trainer path and
  test gradient equivalence versus accumulated-then-reduced.
- Set `PYTORCH_ALLOC_CONF=expandable_segments:True` in the run environment.
- Resumable checkpoint covering model, AdamW optimizer, LR scheduler, optimizer
  step, and sampler RNG/state (upstream optimizer save is currently disabled —
  launch-blocking).
- Deterministic distributed task-balanced sampler (uniform task → uniform
  episode); the flattened `DistributedSampler` is non-compliant.
- Re-encode and re-inventory final robot (224×288) and human (320×448) latents.
- Validate with real artifacts and a short bounded production smoke run.
- Optional: a lightweight final-sync-microstep profiler pass for throughput
  analysis — not a launch prerequisite unless decided.

## Durable artifacts

- Topology: `/sim/reports/topology.json`.
- Benchmark: `/sim/reports/full_episode_batch64/{benchmark_report.json,
  index_preflight.json, index_preflight.log, top64_selection.json,
  nvidia_smi_samples.csv, torchrun.log}`.
- Source: `zero_wam/so101_full_episode_benchmark.py`,
  `zero_wam/modal_so101_benchmark.py`, tests under
  `tests/zero_wam/test_so101_full_episode_benchmark.py`.
- Canonical config: `docs/zero_wam_so101_final_training_config.md`.

## Remaining launch gates

- [x] Single-host topology measured (unique devices 0–7).
- [x] Longest-episode memory feasibility (with required sync/allocator change).
- [ ] Every-microstep gradient sync implemented + equivalence-tested.
- [ ] Resumable checkpoint (model + optimizer + scheduler + step + sampler RNG).
- [ ] Deterministic task-balanced sampler.
- [ ] Final-geometry latent regeneration + inventory verification.
- [ ] Loader smoke on final artifacts.
- [ ] Init-loss validation / short bounded rollouts.
- [ ] Explicit approval before any optimizer launch.
