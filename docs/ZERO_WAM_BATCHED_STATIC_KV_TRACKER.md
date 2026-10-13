# Zero-WAM Batched Static KV-Cache Tracker

## Goal

Adapt local Zero-WAM inference to run a fixed batch of independent closed-loop
rollouts through one shared model. Preallocate per-layer KV storage for a declared
maximum episode length so inference does not repeatedly concatenate, prune, or
reallocate cache tensors.

The implementation must preserve single-rollout outputs before it is used for
evaluation. Throughput improvement is secondary to cache isolation and numerical
equivalence.

## Initial Decisions

- [x] Keep one shared copy of model weights and batch independent rollout states.
- [x] Start with a fixed physical batch size of two on the 80 GB H100.
- [x] Require common camera geometry, action layout, chunk size, and denoising-step count within a batch.
- [x] Declare maximum episode controls before reset and derive static video/action token capacities from it.
- [x] Allocate separate cache regions for every physical batch slot, CFG branch, modality, layer, and attention stream.
- [x] Keep simulator environments independent; only model inference is batched.
- [x] Preserve deterministic per-rollout noise streams independent of batch ordering.
- [x] Use active masks for completed or failed batch slots; never let one slot's early termination alter another slot's cache positions.
- [ ] Select the final maximum episode length after measuring all intended validation horizons and cache memory.
- [ ] Decide whether a completed slot remains masked until the whole batch ends or can be safely reset and reused.

## Investigation

- [x] Trace every cache allocation, append, prediction-cache replacement, pruning, and clear operation in `WanICLTransformer3DModel`.
- [x] Inventory cache tensor shapes, dtypes, layer counts, modalities, and memory consumption for one SO-101 rollout.
- [x] Document how `ICL_CACHE_TYPE`, `OBSERVATION_CACHE_TYPE`, and `PREDICTION_CACHE_TYPE` occupy cache positions.
- [x] Trace CFG packing: conditional/unconditional sequence IDs, temporal duplication, and cache visibility.
- [x] Identify hard-coded batch-one assumptions such as `"1 (f n) c"` reshapes and scalar `chunk_idx`.
- [ ] Trace VAE streaming caches separately; determine whether they need batch support or should remain per-slot/sequential.
- [ ] Measure current dynamic-cache allocation count, bytes allocated, peak VRAM, fragmentation, and per-chunk latency.
- [ ] Confirm the maximum controls/chunks required by `2x robot_frames` for every target evaluation set.
- [x] Determine whether all batch members can share timestep schedules without changing numerical results.
- [x] Verify whether the existing training packed-sequence machinery offers reusable sequence-isolation utilities.

## Static Cache Design

- [x] Define a `StaticKVCacheConfig` — implemented with token-level fields (`max_tokens`, `batch_slots`, `heads`, `head_dim`, `dtype`, `align`); per-chunk token geometry is derived at install time, not stored in the config.
- [x] Define immutable slot IDs and map `(slot, cfg_branch)` to unique sequence IDs (`PackedSlot` frozen dataclass; `slot_seq_id = slot*2 + cfg_branch`).
- [x] Preallocate K/V tensors once at model initialization or first compatible reset (lazy allocation at first `update_cache` append; backing tensors never resize afterward).
- [x] Preallocate cache metadata: type/seq/frame/cache-type IDs in `StaticMetadataCache` with `-1` invalid fill.
- [ ] Maintain per-slot write pointers for ICL prefix, committed observation history, committed action history, and replaceable prediction tokens — NOT implemented; the prototype keeps a contiguous prefix store and uses mask-compaction instead of named regions.
- [x] Replace concatenation with indexed writes into preallocated ranges (on the `update_cache=1` path; `update_cache=0` still builds a transient `cat(view, new)` attention operand, same as the dynamic path's transient concat).
- [x] Replace dynamic prediction-cache deletion with compaction: index-select into reusable scratch, copy-back into the same backing tensors (data_ptrs stable), stale tail invalidated.
- [x] Replace window pruning with deterministic keep-mask compaction preserving `clear_prediction_cache` semantics exactly (ICL + 64-frame observation window retained, prediction dropped) — keep-mask was chosen over a ring buffer to preserve token order and identical semantics.
- [x] Prevent stale-token exposure: attention only ever receives `[:length]` prefix views; compacted tails are zeroed/`-1`-invalidated.
- [x] Reject writes that would exceed configured capacity — bounds check runs before any mutation (`CacheOverflowError`, verified atomic).
- [x] Reject incompatible geometry/configuration — heads/head_dim checked at install; per-append K/V shape checks; second cache name rejected.
- [x] Expose cache capacity, occupancy, pointers, append/compact counts, and allocated bytes through `static_cache_diagnostics`.

## Batched Inference State

- [ ] Introduce per-slot state for prompt embeddings, ICL use, guidance scales, action mask, normalization stats, initial latent, last predicted actions, and chunk index.
- [ ] Accept a fixed-size reset batch and validate every slot before mutating model state.
- [ ] Batch text embeddings with explicit padding masks; do not rely on equal semantic token lengths.
- [ ] Batch human-video latents while preserving separate ICL prefix cache ownership.
- [ ] Use one `torch.Generator` per slot so results do not depend on batch composition or order.
- [ ] Generate video/action noise per slot from its own generator, then stack tensors for model execution.
- [ ] Batch initial observations and subsequent keyframe groups without mixing camera or temporal axes.
- [ ] Preserve the first-chunk rule that skips the first action frame independently per slot.
- [ ] Track active, completed, failed, and padding slots explicitly.
- [ ] Mask inactive slots from transformer outputs and cache writes.
- [ ] Keep success detection and simulator stepping outside the model batch.

## Model Changes

- [ ] Remove hard-coded batch-one `einops` patterns from ICL video and action inference.
- [ ] Make `_prepare_icl_stream` accept batched chunk indices and slot/CFG sequence IDs.
- [ ] Make `_pack_icl_cfg` preserve slot isolation for both ICL CFG and target-text CFG.
- [ ] Make action postprocessing operate per slot with its own normalization statistics and channel mask.
- [ ] Make `_compute_icl_kv_cache` replace each slot's prediction region with its own encoded observations/actions.
- [ ] Ensure attention masks cannot cross slot or CFG-branch boundaries.
- [ ] Preserve the action Transformer's intended lack of direct human-video attention.
- [ ] Keep single-slot execution as a supported batch-size-one configuration of the same implementation.
- [ ] Avoid changes to checkpoint parameters or checkpoint-loading semantics.

## Scheduler

- [ ] Build a batch coordinator that owns two simulator environments and one batched model.
- [ ] Reset both environments before the first batched inference call.
- [ ] Step each simulator with only its corresponding action output.
- [ ] Collect stride-2 observations and commanded action history independently.
- [ ] Perform one batched cache update after both active slots finish a generated chunk.
- [ ] Mask a completed slot while remaining slots continue.
- [ ] Record queue wait, simulator time, VAE time, video denoising time, action denoising time, cache-update time, and serialization time.
- [ ] Write independent per-episode artifacts using the existing result schema.
- [ ] Write a batch manifest recording slot assignment and all per-slot seeds.

## Correctness Tests

- [x] Unit-test static allocation shapes, byte estimates, pointers, bounds, rollback, and clear semantics (`test_batched_static_cache.py`).
- [x] Unit-test that an over-capacity write fails before modifying cache contents.
- [x] Unit-test that stale entries are never exposed (prefix views + invalidated tails).
- [x] Unit-test slot and CFG sequence-ID uniqueness.
- [x] Unit-test inactive-slot masking metadata (all fields INVALID_ID; no cross-boundary attention mass) — runtime one-completed/one-continuing behaviour still needs the batch coordinator.
- [x] Unit-test independent per-slot RNG streams under reversed batch order.
- [x] Unit-test batch-one outputs against the existing dynamic-cache implementation — bit-identical outputs, cache, and metadata on a tiny CPU transformer (2 layers, heads=4, head_dim=24), plus WanICLAttention-level equivalence. Real-scale equivalence remains unverified.
- [ ] Unit-test batch-two outputs against two sequential runs with identical seeds and inputs.
- [ ] Compare every generated action tensor before simulator execution, not only aggregate metrics.
- [ ] Compare cache tensors or stable cache hashes after reset, inference, and observation replacement.
- [ ] Test heterogeneous ICL latent lengths if supported; otherwise verify explicit rejection.
- [ ] Test different episode horizons within one fixed batch.
- [ ] Test one slot failing/resetting without corrupting another.
- [ ] Test CFG-off, target-text CFG, and ICL CFG independently.
- [ ] Test maximum-capacity final write and one-token/chunk overflow.

## Integration Validation

- [ ] Run a synthetic two-slot, two-chunk GPU smoke test.
- [ ] Run two identical slots and verify outputs remain identical.
- [ ] Run two different prompts/scenes and verify outputs differ without cross-slot contamination.
- [ ] Reverse slot order and verify each episode's artifacts are bit-identical or within a documented numerical tolerance.
- [ ] Compare sequential versus batched closed-loop rollouts on fixed simulator/model seeds.
- [ ] Verify early success in one slot does not change the other slot's outputs.
- [x] Verify checkpoint identity and model loading remain exact — model-smoke loads through the existing `checked_transformer` (any inexact load raises) with `loading_info` recorded.
- [ ] Verify videos, actions, telemetry, manifests, success predicates, and trajectories serialize independently.
- [x] Run the existing prompt-sweep and chunk-causal tests unchanged (43 passed incl. the new file).

## Performance Validation

- [ ] Record baseline batch-one dynamic-cache latency and allocator statistics after warmup.
- [ ] Record batch-one static-cache latency to isolate allocation savings from batching gains.
- [ ] Record batch-two static-cache latency, throughput, peak VRAM, and GPU utilization.
- [ ] Measure reset, first-chunk, steady-state chunk, and cache-update latency separately.
- [ ] Confirm static allocation eliminates steady-state cache tensor reallocations.
- [ ] Confirm peak VRAM leaves a safety margin for the longest supported ICL latent and observation batch.
- [ ] Measure performance with one inactive slot to quantify padding/masking waste.
- [ ] Report episodes/hour and controls/second, not only per-forward latency.
- [ ] Require no statistically meaningful task-success regression before adopting batched inference.
- [ ] Set a throughput acceptance target only after the batch-one static-cache baseline is measured.

## Rollout Experiments

- [ ] Start with two short deterministic simulator rollouts.
- [ ] Repeat the successful `pan_on_stove/episode_004` alongside one failed pan episode.
- [ ] Compare batch-two results against frozen sequential artifacts.
- [ ] Run all five pan episodes with smoothing only after numerical equivalence is established.
- [ ] Expand to the five-task validation batch only after cache isolation and artifact equivalence pass.
- [ ] Keep static-cache/batching effects separate from EMA, rate limiting, or temporal-ensemble experiments.

## Documentation And Results

- [ ] Document the static-cache memory formula and selected capacity.
- [ ] Document supported and rejected batch heterogeneity.
- [ ] Document deterministic seed and slot-order guarantees.
- [ ] Record exact commands, hardware, revisions, checkpoints, and environment.
- [ ] Generate a Markdown performance report with sequential/static/batched tables.
- [ ] Record failures, numerical differences, allocator findings, and unresolved risks.
- [ ] Update this tracker continuously; never mark an unverified item complete.

## Current Findings

- [x] The current SO-101 model uses about 38.5 GiB during batch-one inference on an 80 GB H100.
- [x] A second independent worker would nearly duplicate model weights and is not the preferred batching design.
- [x] The current server is stateful and batch-one: mutable cache, scalar chunk index, and hard-coded batch-one reshapes must be changed.
- [x] Validation episodes share compatible robot observation geometry and action layout, making fixed-size batching plausible.
- [x] The paper's packed training sequences and training-only IFP do not provide deployment batching or a static inference cache directly.
- [x] Dynamic inference appends K/V with `torch.cat`, appends metadata with `torch.cat`, and prunes by boolean-index allocation across all 30 self-attention blocks.
- [x] Each self-attention cache uses `[1, tokens, 24, 128]` bf16 K and V tensors. Cross-attention is not cached.
- [x] SO-101 robot video chunks contribute 504 unpadded tokens and action chunks 16 tokens before 128-token alignment; ICL prefix size depends on human latent length.
- [x] The model already isolates packed sequences with `seq_ids`; a batch-two prototype can pack slots into one physical batch dimension of one and assign disjoint sequence IDs, avoiding checkpoint/model changes.
- [x] Both slots can share video/action diffusion timestep schedules when geometry and denoising-step counts match; independent `torch.Generator` instances preserve per-slot noise.
- [x] Full unwindowed episode cache capacity would be wasteful. The static implementation will retain the ICL prefix plus a fixed ring/compaction region sized from the existing 64-frame attention window and declared maximum chunk geometry.
- [x] No batched/static-cache code has been benchmarked yet — still true; only correctness diagnostics exist (2026-10-13).

## Prototype Implementation Status (2026-10-13)

Added files only — no existing source, submodule, script, or test file was
modified (`git status` shows only new `??` entries from this work).

### Files added

- `zero_wam/batched_static_cache.py` — `StaticKVCacheConfig`, `StaticKVCache`,
  `StaticMetadataCache`, `install_static_kv_cache` / `uninstall_static_kv_cache`
  (instance-bound, no class mutation), `static_cache_diagnostics`,
  `CacheOverflowError`.
- `zero_wam/packed_batch.py` — `PackedSlot`, `slot_seq_id` (`slot*2+branch`),
  `slot_generator`, `validate_slots`, `pack_tokens`, `slot_metadata`
  (cfg-branch aware, inactive→INVALID_ID everywhere), `pack_cfg_metadata`,
  `pack_metadata`, `pack_text_embeddings` (encoder ids via `slot_seq_id`),
  `split_packed_output`, `check_compatible`.
- `scripts/run_so101_batched_static_smoke.py` — separate option; `--dry-run`
  (default), `--model-smoke`, `--batch-size 2` (only 2 accepted),
  `--max-chunks`, `--max-tokens`, `--checkpoint`, `--base-model`, `--output`.
- `tests/zero_wam/test_batched_static_cache.py` — 31 tests.

### Commands run and results

```bash
pytest -q tests/zero_wam/test_batched_static_cache.py \
        tests/zero_wam/test_prompt_sweep.py \
        tests/zero_wam/test_chunk_causal_validation.py
# 43 passed (31 new + existing suite unchanged)

python scripts/run_so101_batched_static_smoke.py --dry-run --batch-size 2 \
       --max-chunks 128
# CPU: static/dynamic equivalence on a tiny WanICLTransformer3DModel —
# out1/out2 bit-identical, cache bit-identical, metadata bit-identical,
# prune lengths equal, cache_counts equal, uninstall restores originals.
# (dry-run substitutes eager flex_attention/create_block_mask in-process:
# inductor-compiled flex fails on this CPU host — eager_flex_fallback flag.)

PYTHONPATH=... MUJOCO_GL=egl python scripts/run_so101_batched_static_smoke.py \
    --model-smoke --batch-size 2 --max-chunks 128 --max-tokens 16384 \
    --checkpoint data/models/checkpoints/v2b_step500 \
    --base-model data/models/zero-wam-pretrain
# GPU (H100, ran after the smoothing pilot freed the device):
#   real V2b weights loaded via the existing checked loader;
#   adapter installed on all 30 blocks (30 × [1,16384,24,128] bf16 K/V +
#   scratch ≈ 12.08 GiB allocated);
#   ONE true packed transformer forward observed: two synthetic same-geometry
#   slot streams (504 tokens each) concatenated at batch=1 through
#   forward_latent_only update_cache=1 → output [1,4032,48], 20.2s wall
#   (first-call compile); clear_prediction_cache + clear_cache lifecycle ran;
#   then the script raised the intended NotImplementedError (exit 1) because
#   the server-level batched lifecycle is not wired.  Diagnostics:
#   outputs/zero_wam/batched_static_smoke_model/diagnostics.json
```

### Verified

- Static stores allocate once, never move (stable `data_ptr` across appends
  and compactions), overflow is atomic, prune preserves
  `clear_prediction_cache` semantics exactly, uninstall hands contents back.
- Packed-seq-id isolation rule matches `_pack_icl_cfg`: slot s uses ids
  `2s`/`2s+1` for cond/uncond; text-encoder ids use the same function;
  inactive slots emit `-1` everywhere and can neither attend nor be attended.
- Existing sequential path byte-for-byte untouched; adapter binds instance
  methods only (two-instance test proves no global monkeypatch).

### Remaining gap to true closed-loop batching

- `VA_Server.infer` is single-slot: scalar `chunk_idx`, `_pack_icl_cfg`
  hardcodes CFG seq ids 0/1, one ICL prefix — a batch coordinator owning two
  sims + packed reset/denoise/cache-update calls is not implemented
  (`NotImplementedError` where it would be needed).
- No per-slot named cache regions (ICL prefix / obs / action / prediction
  write pointers) — contiguous prefix + mask compaction only.
- VAE streaming caches remain per-slot/sequential (untraced for batching).
- No batch-two-vs-sequential equivalence, no throughput/latency benchmarks,
  no closed-loop artifacts — none claimed.

## Engine Lifecycle + Warm Benchmark (2026-10-13, second pass)

New files changed only; no existing source/script/test modified.
`git status` remains: new `??` entries plus the pre-existing `M
zero_wam/so101_closed_loop.py` from the earlier validation-telemetry work
(not touched by this milestone).

### What changed since the first pass

- `StaticKVCache` compaction now uses ONE adapter-owned shared K/V workspace
  (lazily allocated per (device,dtype)) instead of per-layer scratch;
  `allocated_bytes` counts workspace once.  Store-level `allocated_bytes`
  counts only its own backing (+private scratch when no workspace).
- `estimate_static_memory()` + `check_static_budget()` (pre-allocation
  admission control; install refuses when estimate + current CUDA allocation
  exceeds 0.85×total on CUDA).
- `packed_batch.py` generalized from exactly-2 to fixed sizes {2,4}:
  `validate_slots(slots, batch_size)`, variadic `pack_tokens(*ts)`,
  sequence-based `check_compatible(...)`.  Seq IDs remain `2s`/`2s+1`
  (batch4 → 0..7); inactive slots still mask to `-1` everywhere.
- `zero_wam/batched_static_engine.py` — NEW `BatchedStaticEngine` around one
  loaded `VA_Server`: per-slot prompt/negative/target embeds, ICL latent/text,
  chunk idx, init latent, last predicted latents/actions, normalization state,
  independent CUDA `torch.Generator`s, and independent `FlowMatchScheduler`s.
  All transformer work is ONE packed `forward_*_only` call per timestep —
  no silent sequential fallback.  VAE encodes stay sequential per slot with
  `streaming_vae.clear_cache()` between slots.
- `scripts/run_so101_batched_static_smoke.py` — `--batch-size {2,4}`,
  `--reverse-check`, `--sequential-check`, `--benchmark`; memory precheck
  printed before allocation; `--max-tokens` default 40960 (batch4 floor
  auto-raised to 49152 for the 2-chunk workload).

### Measured evidence (H100 80GB, V2b checkpoint, real pan validation episodes)

Deterministic set: batch2 = `pan_on_stove/episode_004` + `episode_000`;
batch4 adds `episode_001`, `episode_002`.  Sample seeds 20261028/24/25/26.
Sims observed once for the initial frame; cache-update keyframes are the
initial frame ×1 (fresh 1-frame encode — the streaming VAE's continuation
path needs its cached tail and ≥3 frames; sims are never stepped).

```bash
# packed lifecycle, batch2 (2 chunks each slot)
.../run_so101_batched_static_smoke.py --model-smoke --batch-size 2 \
    --max-tokens 40960 ...
# reset 11.18s | infer chunk0 16.15s | cache_update 0.96s | infer chunk1 12.95s
# packed calls: icl_pack=1, video=102 (51/chunk), action=100, obs_video=1,
# obs_action=1, prune=1; actions [6,2,8]/slot; peak 60.1 GiB
# diagnostics: outputs/zero_wam/batched_static_smoke_model/diagnostics.json

# warm benchmark (one packed infer = one chunk for the whole batch;
# sequential baseline = unmodified server, adapter uninstalled)
.../run_so101_batched_static_smoke.py --model-smoke --benchmark ...
# diagnostics: outputs/zero_wam/batched_static_bench/diagnostics.json
```

| config | warm chunk total | s / episode-chunk | peak VRAM | vs same-set sequential |
|---|---|---|---|---|
| sequential batch1 (4 eps) | 43.43 s | 10.86 | ~60 GiB baseline | 1.00× |
| packed batch2 (2 eps) | 11.92 s | 5.96 | 60.1 GiB | **1.806×** |
| packed batch4 (4 eps) | 11.85 s | 2.96 | 75.86 GiB | **3.665×** |

batch4 static estimate 17.44 GiB (49152 tokens incl. alignment pad) preflighted
before allocation; an earlier 61440-token config was correctly rejected after
OOM during T5 encode and the floor was lowered to what the 2-chunk workload
actually needs (~45k tokens).

### Numerical isolation — recorded as NOT bit-identical

- Slot-order reversal (b2 and b4): per-episode action hashes differ, but
  relative L2 stays 0.24–0.59% on postprocessed actions — consistent with
  bf16 reduction-order drift under different packed layouts, not cross-slot
  leakage (leakage would produce O(1) divergence).
- Sequential-vs-packed, identical seeds/inputs (Philox-matched draws):
  chunk0 rel-L2 0.34–0.59%, chunk1 0.13–0.49%, max abs ≤1.51 on the
  normalized action range.  NOT bit-identical → do NOT use for rollout
  artifacts until the lead accepts this tolerance or a root-cause is found.
- Per-slot packed call counts identical across orders; independent generators
  give deterministic per-episode noise.

### Failure modes fixed during implementation

- `n_uncond` sliced the batch dim instead of the token dim (empty tensor).
- `scheduler.timesteps` is CPU — trailing zero timestep must match its device.
- Single-keyframe `_encode_obs` on a populated streaming cache fails
  (temporal kernel 3 vs padded 2); multi-frame fresh encodes fail at
  `WanResample` (a(8) vs b(4) shortcut mismatch).  Smoke uses fresh 1-frame
  encodes; true per-slot streaming state is still open.
- CUDA OOM at batch4/61440 inside `encode_prompt` — lazy-loaded T5 (~10 GiB)
  arrives after install-time admission; floor lowered to 49152.

### Tests (this pass)

`tests/zero_wam/test_batched_static_cache.py` — 43 tests now:
shared-workspace compaction correctness + pointer stability + byte
accounting, `estimate_static_memory` formula, `check_static_budget`
rejection, batch4 seq-id/CFG ordering (0..7), inactive masking at 4 slots,
4-way pack/split roundtrip, RNG independence, `BatchedStaticEngine`
validation (batch-size rejection) and a full fake-server lifecycle
(reset → packed ICL → 4 video + 2 action packed calls → packed obs writes →
second infer) through the real tiny `WanICLTransformer3DModel`.

```bash
pytest -q tests/zero_wam/test_batched_static_cache.py \
        tests/zero_wam/test_prompt_sweep.py \
        tests/zero_wam/test_chunk_causal_validation.py
# 55 passed
python scripts/run_so101_batched_static_smoke.py --dry-run --batch-size 2 \
       --max-chunks 128   # ok
```

### Remaining gap to true closed-loop batching

- No simulator-coordinated closed-loop rollout yet — the engine lifecycle is
  proven on real model weights/episode inputs, but `Sim.step` is never called
  from the engine path; a coordinator owning two environments + the artifact
  writer is still pending.
- VAE streaming caches remain sequential per slot and are cleared between
  slots — real per-slot temporal continuation across chunks is unresolved
  (packing semantics of the VAE itself need investigation or a second
  streaming-VAE instance).
- Sequential↔packed equivalence is ~0.3–0.6% rel-L2, not bit-identical —
  pending acceptance or root-cause work (likely bf16 attention reduction
  order under different packed layouts).
- No per-slot named cache regions; contiguous prefix + mask compaction only.
- No early-finish slot masking exercised end-to-end at model level.
- Closed-loop artifacts (videos/telemetry/success) from a batched run: none.
