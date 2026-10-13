#!/usr/bin/env python
"""Standalone smoke for the opt-in batched/static-KV prototype.

Two modes:

* ``--dry-run`` (the safe default): exercises the pure packing helpers and the
  static cache against a *tiny* WanICLTransformer3DModel on CPU, including a
  dynamic-vs-static output-equivalence check.  On CPU the inductor-compiled
  flex-attention path is unavailable, so dry-run substitutes the eager
  ``flex_attention``/``create_block_mask`` in-process only (recorded in the
  diagnostics as ``eager_flex_fallback``).

* ``--model-smoke``: loads the real V2b transformer, installs the adapter,
  packs two synthetic same-geometry streams and runs one packed
  ``forward_latent_only`` + a clear_prediction_cache lifecycle through the
  real self-attention static stores.  The *server-level* two-slot
  reset/infer lifecycle is not wired in this prototype — reaching it raises
  ``NotImplementedError`` rather than silently running sequentially.

This file is a separate option: existing runners never import it.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU helper validation (default when no --model-smoke)")
    parser.add_argument("--batch-size", type=int, default=2, choices=(2, 4))
    parser.add_argument("--max-chunks", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=40960,
                        help="logical static-cache capacity (pre-alignment)")
    parser.add_argument("--checkpoint", default="data/models/checkpoints/v2b_step500")
    parser.add_argument("--base-model", default="data/models/zero-wam-pretrain")
    parser.add_argument("--model-smoke", action="store_true",
                        help="load the real model on cuda:0 and run the packed forward")
    parser.add_argument("--reverse-check", action="store_true",
                        help="model-smoke only: rerun the lifecycle with slots swapped and compare per-episode action hashes")
    parser.add_argument("--sequential-check", action="store_true",
                        help="model-smoke only: run each episode through the unmodified sequential server with the same seeds and compare actions")
    parser.add_argument("--benchmark", action="store_true",
                        help="model-smoke only: warm one-chunk benchmark — batch1 sequential vs packed batch2/batch4")
    parser.add_argument("--output", default="outputs/zero_wam/batched_static_smoke")
    return parser.parse_args()


def _eager_flex_fallback():
    """In-process substitute for the inductor-compiled flex path on CPU."""
    import wan_va.modules.icl_model as icl
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    saved = {
        "_compiled_flex_attention": icl._compiled_flex_attention,
        "_compiled_create_block_mask": icl._compiled_create_block_mask,
    }
    icl._compiled_flex_attention = (
        lambda q, k, v, **kw: flex_attention(
            q, k, v, block_mask=kw.get("block_mask")
        )
    )
    icl._compiled_create_block_mask = create_block_mask
    return icl, saved


def _tiny_model():
    from wan_va.modules.icl_model import WanICLTransformer3DModel

    torch.manual_seed(0)
    return WanICLTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=4, attention_head_dim=24,
        in_channels=4, out_channels=4, action_dim=8, text_dim=32,
        freq_dim=64, ffn_dim=128, num_layers=2, action_inner_dim=96,
        action_ffn_dim=128, rope_max_seq_len=2048, pos_embed_seq_len=64,
        enable_mcp=False, attn_window=64,
    ).to(torch.bfloat16)


def _stream_input(frames=2, h=4, w=4, channels=4, text_dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = frames * (h // 2) * (w // 2)
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(frames), torch.arange(h // 2), torch.arange(w // 2),
            indexing="ij",
        )
    ).reshape(3, -1)
    return {
        "latent_res_lst": {
            "noisy_latents": torch.randn(
                1, channels, frames, h, w, generator=g, dtype=torch.bfloat16
            ),
            "timesteps": torch.full((1, frames), 500.0),
        },
        "latent_grid_id": grid,
        "text_emb": torch.randn(1, 6, text_dim, generator=g, dtype=torch.bfloat16),
        "encoder_seq_ids": torch.zeros(6, dtype=torch.int),
        "cache_type_id": 1,
        "_tokens": tokens,
    }


def _dry_run(args, diag):
    from zero_wam.batched_static_cache import (
        StaticKVCache,
        StaticKVCacheConfig,
        StaticMetadataCache,
        install_static_kv_cache,
        static_cache_diagnostics,
        uninstall_static_kv_cache,
    )
    from zero_wam.packed_batch import PackedSlot

    icl, saved = _eager_flex_fallback()
    diag["eager_flex_fallback"] = True
    try:
        # --- pure static stores -------------------------------------------
        kv = StaticKVCache(args.max_tokens, 4, 24, torch.float32, "cpu")
        kv.append(torch.randn(1, 32, 4, 24), torch.randn(1, 32, 4, 24))
        kv.compact(torch.tensor([True] * 20 + [False] * 12))
        diag["static_kv"] = kv.diagnostics()
        meta = StaticMetadataCache(args.max_tokens, "cpu")
        meta.append(*(torch.ones(8, dtype=torch.int) * i for i in range(4)))
        diag["static_meta"] = meta.diagnostics()

        # --- dynamic vs static equivalence on a tiny model -----------------
        model_dyn = _tiny_model()
        model_stat = _tiny_model()
        model_stat.load_state_dict(model_dyn.state_dict())
        install_static_kv_cache(
            model_stat,
            StaticKVCacheConfig(max_tokens=1024, heads=4, head_dim=24),
        )
        inp = _stream_input()
        tokens = inp.pop("_tokens")
        with torch.no_grad():
            out_d1 = model_dyn(dict(inp), update_cache=1, mode="forward_latent_only")
            out_s1 = model_stat(dict(inp), update_cache=1, mode="forward_latent_only")
            out_d2 = model_dyn(dict(inp), update_cache=1, mode="forward_latent_only")
            out_s2 = model_stat(dict(inp), update_cache=1, mode="forward_latent_only")
        dyn_cache = model_dyn.blocks[0].attn1.attn_caches["pos"]["k"]
        stat_store = model_stat._static_kv_state.kv_by_attn[
            id(model_stat.blocks[0].attn1)]["pos"]
        diag["equivalence"] = {
            "tokens_per_forward": tokens,
            "out1_bit_identical": bool(torch.equal(out_d1, out_s1)),
            "out2_bit_identical": bool(torch.equal(out_d2, out_s2)),
            "cache_bit_identical": bool(
                torch.equal(stat_store.view()["k"], dyn_cache)
            ),
            "meta_bit_identical": bool(
                torch.equal(model_stat.seq_ids_cache, model_dyn.seq_ids_cache)
            ),
        }
        # prune + clear through the real code path
        model_dyn.clear_prediction_cache()
        model_stat.clear_prediction_cache()
        dyn_len = model_dyn.blocks[0].attn1.attn_caches["pos"]["k"].shape[1]
        stat_len = stat_store.length
        diag["prune"] = {
            "dynamic_length": int(dyn_len),
            "static_length": int(stat_len),
            "equal": dyn_len == stat_len,
            "counts_dyn": model_dyn.cache_counts(),
            "counts_stat": model_stat.cache_counts(),
        }
        diag["adapter"] = static_cache_diagnostics(model_stat)
        uninstall_static_kv_cache(model_stat)
        diag["uninstall_restored"] = not hasattr(model_stat, "_static_kv_state")

        # --- packed helpers -------------------------------------------------
        from zero_wam.packed_batch import (
            check_compatible,
            pack_tokens,
            slot_generator,
        )

        a = torch.randn(1, 8, 16)
        b = torch.randn(1, 8, 16)
        packed = pack_tokens(a, b)
        diag["packing"] = {
            "packed_shape": list(packed.shape),
            "slots_validated": 2,
            "rng_stream0_equal": bool(
                torch.equal(
                    torch.randn(4, generator=slot_generator(
                        PackedSlot(0, True, 0, 99), "cpu")),
                    torch.randn(4, generator=slot_generator(
                        PackedSlot(0, True, 0, 99), "cpu")),
                )
            ),
        }
        check_compatible([(2, 4, 4)] * 4,
                         [torch.arange(3.0)] * 4, [50] * 4, [50] * 4)
    finally:
        for name, fn in saved.items():
            setattr(icl, name, fn)
    return diag


def _model_smoke(args, diag):
    """Real-weights GPU path: packed lifecycle + optional benchmark.

    Default lifecycle: batch=args.batch_size slots over deterministic
    pan_on_stove validation episodes with real manifest text + human latents
    and per-episode sample seeds from the sequential batch run.  Sims are
    observed once for the initial frame — never stepped; cache-update
    keyframes are the initial frame repeated x8 (matching the
    real ~8-keyframe chunk; the temporal VAE resampler needs >=4 frames; stepping the sim is out of scope for this smoke).

    --benchmark instead runs warm one-chunk action generation for:
      batch1  — the unmodified sequential VA_Server per episode,
      batch2  — the packed engine with 2 slots,
      batch4  — the packed engine with 4 slots (only if its memory
                preflight passes; otherwise recorded as rejected).
    Timings exclude model load and the compile-heavy first infer: chunk 0
    warms kernels, chunk 1 is the timed warm chunk.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("--model-smoke requires CUDA")
    from scripts.run_so101_validation_batch import episodes
    from scripts.run_zero_wam_chunk_causal_validation import (
        _assembled_model,
        _load_server,
    )
    from zero_wam.batched_static_cache import (
        StaticKVCacheConfig,
        estimate_static_memory,
        static_cache_diagnostics,
        uninstall_static_kv_cache,
    )
    from zero_wam.batched_static_engine import (
        BatchedStaticEngine,
        SlotResetSpec,
        action_sha256,
    )
    from zero_wam.so101_closed_loop import Sim

    # deterministic pan episodes + sequential-batch sample seeds
    # (base 20261014 + row index; pan rows are indices 10..14)
    EP_DEFS = {
        "pan_on_stove/episode_000": 20261024,
        "pan_on_stove/episode_001": 20261025,
        "pan_on_stove/episode_002": 20261026,
        "pan_on_stove/episode_003": 20261027,
        "pan_on_stove/episode_004": 20261028,
    }
    EPISODE_SETS = {
        2: ["pan_on_stove/episode_004", "pan_on_stove/episode_000"],
        4: ["pan_on_stove/episode_004", "pan_on_stove/episode_000",
            "pan_on_stove/episode_001", "pan_on_stove/episode_002"],
    }

    def _max_tokens_for(batch_size):
        # per-slot ICL ~8,960 tokens; obs/prediction writes add ~9k at b4,
        # so 49,152 covers a 2-chunk smoke (~45k) with margin.
        floor = {2: 40960, 4: 49152}[batch_size]
        return max(args.max_tokens, floor) if args.max_tokens else floor

    def _preflight(batch_size, max_tokens):
        cfg = StaticKVCacheConfig(max_tokens=max_tokens)
        estimate = estimate_static_memory(cfg, 30)
        torch.cuda.empty_cache()  # release prior config's freed blocks first
        free_b, total_b = torch.cuda.mem_get_info(0)
        entry = {
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "estimate_bytes": estimate,
            "estimated_gib": round(estimate["total_bytes"] / 2**30, 2),
            "free_gib": round(free_b / 2**30, 2),
            "total_gib": round(total_b / 2**30, 2),
        }
        entry["allowed"] = free_b > estimate["total_bytes"] * 1.2
        return entry

    rows = {r["id"]: r for r in episodes(5)}

    def run_lifecycle(episode_ids, batch_size, order=None):
        """Packed lifecycle; slots = `order` (permutation of range(B))."""
        if order is None:
            order = tuple(range(batch_size))
        max_tokens = _max_tokens_for(batch_size)
        pf = _preflight(batch_size, max_tokens)
        if not pf["allowed"]:
            return {"skipped": "memory_preflight_failed", "preflight": pf}
        engine = BatchedStaticEngine(
            model,
            max_tokens=max_tokens,
            batch_size=batch_size,
            video_guidance_scale=-1.0,
            icl_guidance_scale=float(config.icl_guidance_scale),
            use_icl=True,
        )
        eps = [rows[eid] for eid in episode_ids]
        specs = [
            SlotResetSpec(
                prompt=eps[i]["instruction"],
                icl_latent_path=eps[i]["human_latent"],
                seed=EP_DEFS[episode_ids[i]],
            )
            for i in order
        ]
        sims = [Sim("pan_on_stove", eps[i]["seed"]) for i in order]
        obs = [sim.observe() for sim in sims]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        engine.reset(specs)
        t_reset = time.monotonic() - t0
        t0 = time.monotonic()
        actions0 = engine.infer([{"obs": o} for o in obs])
        t_infer0 = time.monotonic() - t0
        t0 = time.monotonic()
        engine.cache_update([{"obs": [o]} for o in obs])
        t_update = time.monotonic() - t0
        t0 = time.monotonic()
        actions1 = engine.infer()
        t_infer1 = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated()
        for sim in sims:
            sim.env.close()
        return {
            "order": [episode_ids[i] for i in order],
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "preflight": pf,
            "reset_s": round(t_reset, 2),
            "infer0_s": round(t_infer0, 2),
            "cache_update_s": round(t_update, 2),
            "infer1_s": round(t_infer1, 2),
            "peak_gib": round(peak / 2**30, 2),
            "actions": {
                f"slot{s}_{episode_ids[i].replace('/', '_')}": {
                    "chunk0_sha256": action_sha256(actions0[s]),
                    "chunk1_sha256": action_sha256(actions1[s]),
                    "chunk0_shape": list(np.asarray(actions0[s]).shape),
                    "chunk1_shape": list(np.asarray(actions1[s]).shape),
                }
                for s, i in enumerate(order)
            },
            "engine_calls": engine.calls,
            "adapter": static_cache_diagnostics(model.transformer),
            "_raw": (actions0, actions1),  # in-memory only, never serialized
        }

    def run_sequential(episode_id):
        """Warm two-chunk sequential lifecycle for one episode (real path)."""
        ep = rows[episode_id]
        sim = Sim("pan_on_stove", ep["seed"])
        obs0 = sim.observe()
        model.infer({
            "reset": True,
            "prompt": ep["instruction"],
            "icl_latent_path": ep["human_latent"],
            "use_icl": True,
            "video_guidance_scale": -1.0,
            "icl_guidance_scale": float(config.icl_guidance_scale),
        })
        torch.manual_seed(EP_DEFS[episode_id])
        t0 = time.monotonic()
        a0 = np.asarray(model.infer({"obs": obs0})["action"])
        t_cold = time.monotonic() - t0
        # match the engine's obs semantics exactly: fresh 1-frame encode
        # (clearing the streaming VAE first), so seq-vs-packed numerics compare
        # like-for-like on identical cached content.
        model.streaming_vae.clear_cache()
        model.infer({"obs": [obs0], "compute_kv_cache": True})
        t0 = time.monotonic()
        a1 = np.asarray(model.infer({"obs": obs0})["action"])
        t_warm = time.monotonic() - t0
        sim.env.close()
        return {
            "episode": episode_id,
            "cold_chunk_s": round(t_cold, 2),
            "warm_chunk_s": round(t_warm, 2),
            "actions": (a0, a1),
            "chunk0_sha256": action_sha256(a0),
            "chunk1_sha256": action_sha256(a1),
        }

    with tempfile.TemporaryDirectory() as temp:
        assembled = _assembled_model(args.base_model, args.checkpoint, temp)
        model, load_evidence, config = _load_server(assembled, "cuda:0")
    diag["model_class"] = type(model.transformer).__name__
    diag["load_evidence"] = {
        k: (v if isinstance(v, (int, float, str)) else str(v)[:200])
        for k, v in load_evidence.items()
    }

    if args.benchmark:
        # warm one-chunk benchmark: batch1 sequential vs packed 2 vs packed 4
        bench = {"warm_chunk": "chunk1 (2nd infer; compile/load excluded)",
                 "steps": "video 50 / action 50", "configs": {}}
        for bs in (2, 4):
            episode_ids = EPISODE_SETS[bs]
            pf = _preflight(bs, _max_tokens_for(bs))
            if not pf["allowed"]:
                bench["configs"][f"packed_batch{bs}"] = {
                    "skipped": "memory_preflight_failed", "preflight": pf}
                continue
            run = run_lifecycle(episode_ids, bs)
            # keep packed raw for numerical checks later
            raw = run.pop("_raw")
            diag.setdefault("_bench_raw", {})[bs] = (run, raw)
            bench["configs"][f"packed_batch{bs}"] = {
                "preflight": pf,
                "warm_chunk_total_s": run["infer1_s"],
                "s_per_episode_chunk": round(run["infer1_s"] / bs, 2),
                "peak_gib": run["peak_gib"],
                "engine_calls": run["engine_calls"],
            }
            uninstall_static_kv_cache(model.transformer)
        # sequential batch1 baseline over all 4 benchmark episodes
        seq_eps = EPISODE_SETS[4]
        seq_results = {}
        for eid in seq_eps:
            seq_results[eid] = run_sequential(eid)
        seq_total = sum(r["warm_chunk_s"] for r in seq_results.values())
        bench["configs"]["sequential_batch1"] = {
            "episodes": seq_eps,
            "warm_chunk_total_s": round(seq_total, 2),
            "s_per_episode_chunk": round(seq_total / len(seq_eps), 2),
            "per_episode": {
                eid: {k: v for k, v in r.items() if k != "actions"}
                for eid, r in seq_results.items()
            },
        }
        for bs, (run, raw) in diag.pop("_bench_raw", {}).items():
            eps = run["order"]
            seq_sub = [seq_results[e]["warm_chunk_s"] for e in eps]
            seq_sub_total = sum(seq_sub)
            bench["configs"][f"packed_batch{bs}"]["sequential_same_set_s"] = (
                round(seq_sub_total, 2))
            bench["configs"][f"packed_batch{bs}"][
                "throughput_ratio_seq_over_packed"] = round(
                seq_sub_total / run["infer1_s"], 3)
            bench["configs"][f"packed_batch{bs}"]["is_speedup"] = (
                run["infer1_s"] < seq_sub_total)
            # sequential-vs-packed numerical comparison
            cmp_ = {}
            for s, eid in enumerate(eps):
                a0, a1 = seq_results[eid]["actions"]
                d0 = np.asarray(raw[0][s]) - a0
                d1 = np.asarray(raw[1][s]) - a1
                cmp_[eid] = {
                    "chunk0_sha_equal": seq_results[eid]["chunk0_sha256"]
                    == run["actions"][f"slot{s}_{eid.replace('/', '_')}"]["chunk0_sha256"],
                    "chunk1_sha_equal": seq_results[eid]["chunk1_sha256"]
                    == run["actions"][f"slot{s}_{eid.replace('/', '_')}"]["chunk1_sha256"],
                    "chunk0_max_abs_diff": float(np.abs(d0).max()),
                    "chunk0_rel_l2": float(
                        np.linalg.norm(d0) / (np.linalg.norm(a0) + 1e-12)),
                    "chunk1_max_abs_diff": float(np.abs(d1).max()),
                    "chunk1_rel_l2": float(
                        np.linalg.norm(d1) / (np.linalg.norm(a1) + 1e-12)),
                }
            bench["configs"][f"packed_batch{bs}"]["vs_sequential"] = cmp_
            bench["configs"][f"packed_batch{bs}"]["actions"] = run["actions"]
        diag["benchmark"] = bench
        diag["closed_loop_batched"] = True
        return

    # ---- single-config lifecycle (default mode) ---------------------------
    episode_ids = EPISODE_SETS[args.batch_size]
    diag["memory_precheck"] = _preflight(
        args.batch_size, _max_tokens_for(args.batch_size))
    diag["run_ab"] = run_lifecycle(episode_ids, args.batch_size)
    if "skipped" in diag["run_ab"]:
        raise MemoryError("memory preflight rejected the packed config")

    if args.reverse_check:
        uninstall_static_kv_cache(model.transformer)
        rev_order = tuple(reversed(range(args.batch_size)))
        diag["run_ba"] = run_lifecycle(episode_ids, args.batch_size, rev_order)
        reversal = {}
        raw_ab, raw_ba = diag["run_ab"]["_raw"], diag["run_ba"]["_raw"]
        for i in range(args.batch_size):
            j = args.batch_size - 1 - i
            key = episode_ids[i]
            tag = key.replace("/", "_")
            d0 = np.asarray(raw_ab[0][i]) - np.asarray(raw_ba[0][j])
            d1 = np.asarray(raw_ab[1][i]) - np.asarray(raw_ba[1][j])
            reversal[key] = {
                "chunk0_sha_equal": diag["run_ab"]["actions"][
                    f"slot{i}_{tag}"]["chunk0_sha256"]
                == diag["run_ba"]["actions"][f"slot{j}_{tag}"]["chunk0_sha256"],
                "chunk1_sha_equal": diag["run_ab"]["actions"][
                    f"slot{i}_{tag}"]["chunk1_sha256"]
                == diag["run_ba"]["actions"][f"slot{j}_{tag}"]["chunk1_sha256"],
                "chunk0_max_abs_diff": float(np.abs(d0).max()),
                "chunk0_rel_l2": float(
                    np.linalg.norm(d0) / (np.linalg.norm(raw_ab[0][i]) + 1e-12)),
                "chunk1_max_abs_diff": float(np.abs(d1).max()),
                "chunk1_rel_l2": float(
                    np.linalg.norm(d1) / (np.linalg.norm(raw_ab[1][i]) + 1e-12)),
            }
        diag["reversal"] = reversal

    if args.sequential_check:
        if getattr(model.transformer, "_static_kv_state", None) is not None:
            uninstall_static_kv_cache(model.transformer)
        raw_ab = diag["run_ab"]["_raw"]
        seq_cmp = {}
        for i, eid in enumerate(episode_ids):
            ref = run_sequential(eid)
            d0 = np.asarray(raw_ab[0][i]) - ref["actions"][0]
            d1 = np.asarray(raw_ab[1][i]) - ref["actions"][1]
            seq_cmp[eid] = {
                "chunk0_sha_equal": ref["chunk0_sha256"]
                == diag["run_ab"]["actions"][
                    f"slot{i}_{eid.replace('/', '_')}"]["chunk0_sha256"],
                "chunk1_sha_equal": ref["chunk1_sha256"]
                == diag["run_ab"]["actions"][
                    f"slot{i}_{eid.replace('/', '_')}"]["chunk1_sha256"],
                "chunk0_max_abs_diff": float(np.abs(d0).max()),
                "chunk0_rel_l2": float(
                    np.linalg.norm(d0)
                    / (np.linalg.norm(ref["actions"][0]) + 1e-12)),
                "chunk1_max_abs_diff": float(np.abs(d1).max()),
                "chunk1_rel_l2": float(
                    np.linalg.norm(d1)
                    / (np.linalg.norm(ref["actions"][1]) + 1e-12)),
                "seq_warm_chunk_s": ref["warm_chunk_s"],
            }
        diag["sequential_check"] = seq_cmp

    for run_key in ("run_ab", "run_ba"):
        if run_key in diag:
            diag[run_key].pop("_raw", None)
    diag["closed_loop_batched"] = True


def main():
    args = _parse_args()
    if args.batch_size not in (2, 4):
        raise ValueError(f"--batch-size must be 2 or 4, got {args.batch_size}")
    diag = {
        "mode": "model-smoke" if args.model_smoke else "dry-run",
        "batch_size": args.batch_size,
        "max_chunks": args.max_chunks,
        "max_tokens": args.max_tokens,
        "checkpoint": args.checkpoint if args.model_smoke else None,
    }
    try:
        if args.model_smoke:
            _model_smoke(args, diag)
        else:
            _dry_run(args, diag)
    finally:
        # Diagnostics are evidence; write them even when the smoke legitimately
        # ends in NotImplementedError (unwired server-level batched lifecycle).
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=1))
        print(json.dumps(diag, indent=1))
        print(f"diagnostics -> {out_dir / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
