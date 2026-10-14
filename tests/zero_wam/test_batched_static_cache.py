"""Unit tests for the opt-in static-KV prototype (new files only)."""

from __future__ import annotations

import types

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from zero_wam.batched_static_cache import (
    CACHE_NAME,
    CacheOverflowError,
    StaticKVCache,
    StaticKVCacheConfig,
    StaticMetadataCache,
    estimate_static_memory,
    install_static_kv_cache,
    static_cache_diagnostics,
    uninstall_static_kv_cache,
)
from zero_wam.packed_batch import (
    COND_BRANCH,
    INVALID_ID,
    UNCOND_BRANCH,
    PackedSlot,
    check_compatible,
    pack_cfg_metadata,
    pack_metadata,
    pack_text_embeddings,
    pack_tokens,
    slot_generator,
    slot_metadata,
    slot_seq_id,
    split_packed_output,
    validate_slots,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# StaticKVCache
# ---------------------------------------------------------------------------

def _kv(n, heads=4, dim=8, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, n, heads, dim, generator=g, dtype=dtype),
        torch.randn(1, n, heads, dim, generator=g, dtype=dtype),
    )


def test_static_kv_append_and_view():
    cache = StaticKVCache(64, 4, 8, torch.float32, DEVICE)
    k, v = _kv(10)
    start = cache.append(k, v)
    assert start == 0 and cache.length == 10
    view = cache.view()
    assert view["k"].shape == (1, 10, 4, 8)
    assert torch.equal(view["k"], k)
    assert torch.equal(view["v"], v)
    k2, v2 = _kv(5, seed=1)
    cache.append(k2, v2)
    assert cache.length == 15
    assert torch.equal(cache.view()["k"][:, 10:], k2)


def test_static_kv_capacity_includes_alignment_padding():
    # 130 logical tokens round up to 256 physical (align=128).
    cache = StaticKVCache(130, 4, 8, torch.float32, DEVICE)
    assert cache.capacity == 256
    cfg = StaticKVCacheConfig(max_tokens=130)
    assert cfg.capacity == 256


def test_static_kv_overflow_is_atomic():
    cache = StaticKVCache(16, 4, 8, torch.float32, DEVICE, align=1)
    k, v = _kv(10)
    cache.append(k, v)
    before = cache.view()["k"].clone()
    with pytest.raises(CacheOverflowError):
        cache.append(*_kv(7))  # 10 + 7 > 16
    assert cache.length == 10
    assert torch.equal(cache.view()["k"], before)


def test_static_kv_exact_capacity_boundary():
    cache = StaticKVCache(16, 4, 8, torch.float32, DEVICE, align=1)
    cache.append(*_kv(16))
    with pytest.raises(CacheOverflowError):
        cache.append(*_kv(1))


def test_static_kv_compact_preserves_order_and_pointer():
    cache = StaticKVCache(64, 4, 8, torch.float32, DEVICE)
    k, v = _kv(6)
    cache.append(k, v)
    ptr_k, ptr_v = cache.k.data_ptr(), cache.v.data_ptr()
    keep = torch.tensor([True, False, True, False, True, True])
    new_len = cache.compact(keep)
    assert new_len == 4
    expected = k[:, [0, 2, 4, 5]]
    assert torch.equal(cache.view()["k"], expected)
    assert cache.k.data_ptr() == ptr_k and cache.v.data_ptr() == ptr_v
    # vacated tail zeroed
    assert cache.k[:, 4:6].abs().sum() == 0


def test_static_kv_compact_index_mask_and_wrong_length():
    cache = StaticKVCache(64, 4, 8, torch.float32, DEVICE)
    cache.append(*_kv(5))
    assert cache.compact(torch.tensor([4, 2, 0])) == 3  # index reorder allowed
    with pytest.raises(ValueError):
        cache.compact(torch.ones(4, dtype=torch.bool))


def test_static_kv_stale_region_never_exposed():
    cache = StaticKVCache(64, 4, 8, torch.float32, DEVICE)
    cache.append(*_kv(8))
    cache.compact(torch.tensor([True] * 4 + [False] * 4))
    assert cache.length == 4
    view = cache.view()
    assert view["k"].shape[1] == 4  # stale tokens unreachable
    cache.clear()
    assert cache.view()["k"].shape[1] == 0


def test_static_kv_rollback_restores_length():
    cache = StaticKVCache(64, 4, 8, torch.float32, DEVICE)
    cache.append(*_kv(4))
    mark = cache.append(*_kv(3))
    assert mark == 4 and cache.length == 7
    cache.rollback(mark)
    assert cache.length == 4
    with pytest.raises(ValueError):
        cache.rollback(9)


def test_static_kv_shape_rejection():
    cache = StaticKVCache(64, 4, 8, torch.float32, DEVICE)
    with pytest.raises(ValueError):
        cache.append(torch.randn(2, 4, 4, 8), torch.randn(2, 4, 4, 8))  # B!=1
    with pytest.raises(ValueError):
        cache.append(torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8))  # heads
    with pytest.raises(ValueError):
        cache.append(*_kv(4, heads=4, dim=16))  # head_dim


# ---------------------------------------------------------------------------
# StaticMetadataCache
# ---------------------------------------------------------------------------

def _meta_rows(n, base=0):
    return (
        torch.arange(base, base + n, dtype=torch.int),          # type
        torch.arange(100 + base, 100 + base + n),               # seq
        torch.arange(200 + base, 200 + base + n),               # frame
        torch.arange(300 + base, 300 + base + n),               # cache_type
    )


def test_static_metadata_append_view_invalid_fill():
    meta = StaticMetadataCache(64, DEVICE)
    assert (meta._store["seq_ids_cache"] == -1).all()  # -1 fill
    t, s, f, c = _meta_rows(5)
    meta.append(t, s, f, c)
    assert meta.length == 5
    assert torch.equal(meta.view("seq_ids_cache"), s)
    # region beyond length stays -1
    assert (meta._store["seq_ids_cache"][5:] == -1).all()


def test_static_metadata_overflow_atomic():
    meta = StaticMetadataCache(8, DEVICE, align=1)
    meta.append(*_meta_rows(6))
    before = meta.view("frame_ids_cache").clone()
    with pytest.raises(CacheOverflowError):
        meta.append(*_meta_rows(3, base=10))  # 6 + 3 > 8
    assert meta.length == 6
    assert torch.equal(meta.view("frame_ids_cache"), before)


def test_static_metadata_compact_via_scratch_order_and_fill():
    meta = StaticMetadataCache(64, DEVICE)
    meta.append(*_meta_rows(6))
    keep = torch.tensor([True, False, True, True, False, True])
    assert meta.compact(keep) == 4
    assert torch.equal(
        meta.view("seq_ids_cache"),
        torch.tensor([100, 102, 103, 105]),
    )
    # tail re-invalidated to -1
    assert (meta._store["seq_ids_cache"][4:6] == -1).all()


def test_static_metadata_clear():
    meta = StaticMetadataCache(64, DEVICE)
    meta.append(*_meta_rows(4))
    meta.clear()
    assert meta.length == 0
    assert (meta._store["type_ids_cache"] == -1).all()


def test_static_metadata_append_length_mismatch():
    meta = StaticMetadataCache(64, DEVICE)
    t, s, f, c = _meta_rows(4)
    with pytest.raises(ValueError):
        meta.append(t, s[:3], f, c)


# ---------------------------------------------------------------------------
# Adapter install / uninstall / migration on fake transformers
# ---------------------------------------------------------------------------

def _fake_transformer(layers=2, heads=2, head_dim=12):
    """Duck-typed WanICLTransformer3DModel stand-in bound to real methods."""
    from wan_va.modules.icl_model import (
        WanICLAttention,
        WanICLTransformer3DModel,
    )

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = WanICLAttention(
                dim=heads * head_dim, heads=heads, dim_head=head_dim,
                eps=1e-6, action_inner_dim=heads * head_dim,
            )

    class _Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_Block() for _ in range(layers)])
            self.attn_window = 64
            self.type_ids_cache = None
            self.seq_ids_cache = None
            self.frame_ids_cache = None
            self.cache_type_ids_cache = None
            for name in (
                "clear_cache", "_append_metadata", "_prune_cache",
                "clear_prediction_cache", "cache_counts",
            ):
                setattr(
                    self, name,
                    types.MethodType(getattr(WanICLTransformer3DModel, name), self),
                )

    return _Fake()


def test_install_binds_instance_methods_only():
    from wan_va.modules.icl_model import WanICLTransformer3DModel

    a, b = _fake_transformer(), _fake_transformer()
    cfg = StaticKVCacheConfig(max_tokens=128, heads=2, head_dim=12)
    orig_cls_forward = type(a.blocks[0].attn1).forward
    install_static_kv_cache(a, cfg)
    # bound on the instance
    assert a.clear_cache.__func__ is not WanICLTransformer3DModel.clear_cache
    assert a.blocks[0].attn1.forward.__func__ is not orig_cls_forward
    # sibling instance untouched (no global monkeypatch)
    assert b.blocks[0].attn1.forward == orig_cls_forward.__get__(
        b.blocks[0].attn1)
    # class definitions untouched
    assert type(a.blocks[0].attn1).forward is orig_cls_forward
    assert (
        WanICLTransformer3DModel.__dict__["clear_cache"].__qualname__
        == "WanICLTransformer3DModel.clear_cache"
    )


def test_install_rejects_second_install_and_bad_names():
    a = _fake_transformer()
    cfg = StaticKVCacheConfig(max_tokens=128, heads=2, head_dim=12)
    install_static_kv_cache(a, cfg)
    with pytest.raises(RuntimeError):
        install_static_kv_cache(a, cfg)
    with pytest.raises(ValueError):
        a.clear_cache("other")  # second cache name rejected
    with pytest.raises(ValueError):
        a._prune_cache(torch.ones(0, dtype=torch.bool), "other")


def test_install_rejects_incompatible_heads():
    a = _fake_transformer(heads=2, head_dim=12)
    with pytest.raises(ValueError):
        install_static_kv_cache(
            a, StaticKVCacheConfig(max_tokens=128, heads=4, head_dim=12)
        )
    with pytest.raises(ValueError):
        install_static_kv_cache(
            a, StaticKVCacheConfig(max_tokens=128, heads=2, head_dim=16)
        )


def test_metadata_append_and_prune_through_adapter():
    a = _fake_transformer()
    install_static_kv_cache(
        a, StaticKVCacheConfig(max_tokens=128, heads=2, head_dim=12)
    )
    # simulate _append_metadata call exactly as _forward_stream does
    qtype = torch.zeros(5, dtype=torch.int)
    qseq = torch.zeros(5, dtype=torch.int)
    qframe = torch.arange(5, dtype=torch.int)
    ctype = torch.ones(5, dtype=torch.int)  # PREDICTION
    a._append_metadata(qtype, qseq, qframe, ctype)
    assert a.seq_ids_cache is not None and len(a.seq_ids_cache) == 5
    # prune to first 3 through the adapter
    keep = torch.tensor([True, True, True, False, False])
    a._prune_cache(keep)
    assert len(a.seq_ids_cache) == 3
    assert a.cache_counts() == {0: 0, 1: 3, 2: 0}
    a.clear_cache()
    assert a.seq_ids_cache is None and a.cache_counts() == {}


def test_install_migrates_dynamic_cache_contents():
    a = _fake_transformer()
    k, v = _kv(7, heads=2, dim=12)
    for block in a.blocks:
        block.attn1.attn_caches[CACHE_NAME] = {"k": k.clone(), "v": v.clone()}
    t, s, f, c = _meta_rows(7)
    a.type_ids_cache, a.seq_ids_cache = t.clone(), s.clone()
    a.frame_ids_cache, a.cache_type_ids_cache = f.clone(), c.clone()

    install_static_kv_cache(
        a, StaticKVCacheConfig(max_tokens=128, heads=2, head_dim=12)
    )
    state = a._static_kv_state
    for block in a.blocks:
        store = state.kv_by_attn[id(block.attn1)][CACHE_NAME]
        assert store.length == 7
        assert torch.equal(store.view()["k"], k)
    assert len(a.seq_ids_cache) == 7


def test_uninstall_restores_methods_and_contents():
    a = _fake_transformer()
    orig = a.blocks[0].attn1.forward
    install_static_kv_cache(
        a, StaticKVCacheConfig(max_tokens=128, heads=2, head_dim=12)
    )
    a._append_metadata(*_meta_rows(4))
    uninstall_static_kv_cache(a)
    assert not hasattr(a, "_static_kv_state")
    assert a.blocks[0].attn1.forward == orig
    # metadata survived as plain tensors
    assert isinstance(a.seq_ids_cache, torch.Tensor)
    assert len(a.seq_ids_cache) == 4
    # block caches returned to plain dict form
    assert isinstance(a.blocks[0].attn1.attn_caches, dict)


def test_uninstall_requires_install():
    a = _fake_transformer()
    with pytest.raises(RuntimeError):
        uninstall_static_kv_cache(a)


def test_diagnostics_fields():
    a = _fake_transformer(layers=2)
    assert static_cache_diagnostics(a) == {"installed": False}
    install_static_kv_cache(
        a, StaticKVCacheConfig(max_tokens=128, heads=2, head_dim=12)
    )
    diag = static_cache_diagnostics(a)
    assert diag["installed"] and diag["cache_name"] == "pos"
    assert diag["config"]["capacity"] == 128
    assert diag["metadata"]["allocated_bytes"] > 0
    assert diag["allocated_bytes"] >= diag["metadata"]["allocated_bytes"]


# ---------------------------------------------------------------------------
# packed_batch
# ---------------------------------------------------------------------------

def test_slot_seq_ids_disjoint_across_slots_and_branches():
    ids = {slot_seq_id(s, b) for s in (0, 1) for b in (0, 1)}
    assert ids == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        slot_seq_id(0, 2)
    with pytest.raises(ValueError):
        slot_seq_id(-1, 0)


def test_validate_slots():
    s0 = PackedSlot(0, True, 0, 11)
    s1 = PackedSlot(1, True, 0, 22)
    assert validate_slots([s0, s1]) == (s0, s1)
    with pytest.raises(ValueError):
        validate_slots([s0])
    with pytest.raises(ValueError):
        validate_slots([s0, PackedSlot(0, True, 0, 33)])


def test_pack_tokens_concat_and_reject():
    a = torch.randn(1, 8, 16)
    b = torch.randn(1, 8, 16)
    packed = pack_tokens(a, b)
    assert packed.shape == (1, 16, 16)
    assert torch.equal(packed[:, :8], a) and torch.equal(packed[:, 8:], b)
    with pytest.raises(ValueError):
        pack_tokens(torch.randn(2, 8, 16), b)  # batch != 1
    with pytest.raises(ValueError):
        pack_tokens(a, torch.randn(1, 8, 32))  # geometry mismatch
    with pytest.raises(ValueError):
        pack_tokens(a, b.float() if b.dtype != torch.float32 else b.double())


def test_pack_split_roundtrip():
    a = torch.randn(1, 6, 4)
    b = torch.randn(1, 9, 4)
    packed = pack_tokens(a, b)
    sa, sb = split_packed_output(packed, [6, 9])
    assert torch.equal(sa, a) and torch.equal(sb, b)
    with pytest.raises(ValueError):
        split_packed_output(packed, [6, 8])


def test_slot_metadata_active_inactive_masking():
    slot_a = PackedSlot(0, True, 3, 42)
    slot_b = PackedSlot(1, False, 3, 43)
    frames = torch.full((10,), 7, dtype=torch.int)
    m0 = slot_metadata(slot_a, 10, frames, 0, 1, DEVICE, COND_BRANCH)
    assert (m0["seq_ids"] == slot_seq_id(0, COND_BRANCH)).all()
    assert (m0["frame_ids"] == 7).all()
    m0u = slot_metadata(slot_a, 10, frames, 0, 1, DEVICE, UNCOND_BRANCH)
    assert (m0u["seq_ids"] == slot_seq_id(0, UNCOND_BRANCH)).all()
    m1 = slot_metadata(slot_b, 10, frames, 0, 1, DEVICE)
    for key in ("seq_ids", "frame_ids", "type_ids", "cache_type_ids"):
        assert (m1[key] == INVALID_ID).all()
    # inactive slot cannot leak: emulate the real mask's key-validity check
    packed = pack_metadata(m0, m0u, m1)
    valid_keys = packed["type_ids"] != -1
    assert valid_keys.sum() == 20  # only slot-0 cond+uncond tokens are valid
    inactive_query_seq = m1["seq_ids"][0].item()
    assert not (packed["seq_ids"][valid_keys] == inactive_query_seq).any()


def test_pack_text_embeddings_uses_slot_seq_ids():
    e0 = torch.randn(1, 5, 16)
    e1 = torch.randn(1, 7, 16)
    s0 = PackedSlot(0, True, 0, 1)
    s1 = PackedSlot(1, True, 0, 2)
    packed, enc = pack_text_embeddings(
        [(e0, s0, COND_BRANCH), (e1, s1, COND_BRANCH)]
    )
    assert packed.shape == (1, 12, 16)
    assert enc.tolist() == [0] * 5 + [2] * 7
    # uncond text of slot 1 -> id 3
    packed2, enc2 = pack_text_embeddings(
        [(e0, s0, COND_BRANCH), (e1, s1, UNCOND_BRANCH)]
    )
    assert enc2.tolist() == [0] * 5 + [3] * 7
    # inactive slot text masked
    s1_off = PackedSlot(1, False, 0, 2)
    _, enc3 = pack_text_embeddings([(e0, s0, COND_BRANCH), (e1, s1_off, COND_BRANCH)])
    assert (enc3[5:] == INVALID_ID).all()


def test_per_slot_rng_deterministic_independent_of_order():
    s0 = PackedSlot(0, True, 0, 1234)
    s1 = PackedSlot(1, True, 0, 5678)
    # forward order
    n0 = torch.randn(4, generator=slot_generator(s0, "cpu"))
    n1 = torch.randn(4, generator=slot_generator(s1, "cpu"))
    # reversed order — slot 1 first
    n1b = torch.randn(4, generator=slot_generator(s1, "cpu"))
    n0b = torch.randn(4, generator=slot_generator(s0, "cpu"))
    assert torch.equal(n0, n0b) and torch.equal(n1, n1b)
    assert not torch.equal(n0, n1)


def test_check_compatible_rejections():
    g = (2, 4, 4)
    ts = torch.linspace(1000, 0, 5)
    check_compatible([g, g], [ts, ts], [50, 50], [50, 50])
    with pytest.raises(ValueError):
        check_compatible([g, (2, 4, 5)], [ts, ts], [50, 50], [50, 50])
    with pytest.raises(ValueError):
        check_compatible([g, g], [ts, ts + 1], [50, 50], [50, 50])
    with pytest.raises(ValueError):
        check_compatible([g, g], [ts, ts], [50, 25], [50, 50])
    check_compatible([g] * 4, [ts] * 4, [50] * 4, [50] * 4)  # batch4


def test_batch4_slot_ids_and_validation():
    """seq ids 2s/2s+1 for s in 0..3; validation accepts exactly {2,4}."""
    slots = [PackedSlot(i, True, 0, 10 + i) for i in range(4)]
    assert validate_slots(slots, batch_size=4) == tuple(slots)
    seqs = {slot_seq_id(i, br) for i in range(4) for br in (0, 1)}
    assert seqs == set(range(8))
    with pytest.raises(ValueError):
        validate_slots(slots[:3], batch_size=4)
    with pytest.raises(ValueError):
        validate_slots(slots, batch_size=3)
    with pytest.raises(ValueError):
        validate_slots(
            [PackedSlot(0, True, 0, 1)] * 4, batch_size=4
        )


def test_batch4_pack_split_roundtrip_and_inactive():
    from zero_wam.packed_batch import pack_metadata, slot_metadata

    ts = [torch.randn(1, 6, 4) + i for i in range(4)]
    packed = pack_tokens(*ts, dim=1)
    assert packed.shape == (1, 24, 4)
    parts = split_packed_output(packed, [6, 6, 6, 6], dim=1)
    for i, p in enumerate(parts):
        assert torch.equal(p, ts[i])
    # inactive slot → all metadata invalid; no leakage either direction
    metas = [
        slot_metadata(
            PackedSlot(i, active=(i != 2), chunk_idx=0, seed=i),
            8, torch.zeros(8, dtype=torch.int), 0, 0, "cpu",
            cfg_branch=0,
        )
        for i in range(4)
    ]
    packed_meta = pack_metadata(*metas)
    assert (packed_meta["seq_ids"][16:24] == -1).all()
    assert packed_meta["seq_ids"].unique().tolist() == [-1, 0, 2, 6]


def test_unsafe_memory_rejection():
    from zero_wam.batched_static_cache import check_static_budget

    with pytest.raises(MemoryError, match="refusing"):
        check_static_budget(
            estimate_bytes=60 * 2**30,
            device=None,
            allocated_bytes=20 * 2**30,
            total_bytes=80 * 2**30,
        )
    ok = check_static_budget(
        estimate_bytes=10 * 2**30,
        device=None,
        allocated_bytes=20 * 2**30,
        total_bytes=80 * 2**30,
    )
    assert ok["allowed"] is True


def test_engine_rejects_bad_batch_size():
    from zero_wam.batched_static_engine import BatchedStaticEngine

    server = _fake_server()
    with pytest.raises(ValueError, match="batch_size"):
        BatchedStaticEngine(server, max_tokens=4096, batch_size=3)


def test_engine_batch4_pack_ordering():
    """4 slots × 2 CFG branches → seq order 0..7 in packed stream."""
    from zero_wam.batched_static_engine import BatchedStaticEngine, SlotResetSpec

    server = _fake_server()
    engine = BatchedStaticEngine(
        server, max_tokens=8192, batch_size=4,
        video_guidance_scale=-1.0, icl_guidance_scale=5.0,
    )
    for i, s in enumerate(engine.slots):
        s.spec = SlotResetSpec("x", "y", 100 + i)
        s.target_prompt_embeds = torch.zeros(1, 4, 32)
        s.negative_prompt_embeds = torch.zeros(1, 4, 32)
    lat = torch.zeros(1, 48, 2, 4, 12, dtype=torch.bfloat16)
    d, toks = engine._pack_video_stream(
        [lat] * 4, 500.0, PREDICTION_TYPE := 1
    )
    seqs = d["current_seq_ids"].tolist()
    per = toks[0]
    for i in range(4):
        assert set(seqs[i * 2 * per:(i * 2 + 1) * per]) == {2 * i}
        assert set(seqs[(i * 2 + 1) * per:(i * 2 + 2) * per]) == {2 * i + 1}
    enc = d["encoder_seq_ids"].unique().tolist()
    assert enc == list(range(8))


# ---------------------------------------------------------------------------
# WanICLAttention-level adapter equivalence (eager flex fallback on CPU)
# ---------------------------------------------------------------------------

def test_attention_forward_equivalence_static_vs_dynamic(monkeypatch):
    """Same weights + inputs: adapter path must match the dynamic path."""
    import wan_va.modules.icl_model as icl
    from torch.nn.attention.flex_attention import flex_attention

    # Inductor-compiled flex_attention fails on this CPU host; use the eager
    # path inside this test only (monkeypatch restores it automatically).
    monkeypatch.setattr(
        icl, "_compiled_flex_attention",
        lambda q, k, v, **kw: flex_attention(q, k, v, block_mask=kw.get("block_mask")),
    )
    torch.manual_seed(7)
    heads, dim_head = 2, 12
    attn_dyn = icl.WanICLAttention(
        dim=heads * dim_head, heads=heads, dim_head=dim_head, eps=1e-6,
        action_inner_dim=heads * dim_head,
    ).to(torch.bfloat16)
    attn_stat = icl.WanICLAttention(
        dim=heads * dim_head, heads=heads, dim_head=dim_head, eps=1e-6,
        action_inner_dim=heads * dim_head,
    ).to(torch.bfloat16)
    attn_stat.load_state_dict(attn_dyn.state_dict())

    # minimal fake transformer carrying the adapter state for attn_stat
    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = attn_stat

    class _Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_Block()])
            self.attn_window = 64
            self.type_ids_cache = None
            self.seq_ids_cache = None
            self.frame_ids_cache = None
            self.cache_type_ids_cache = None
            for name in ("clear_cache", "_append_metadata", "_prune_cache",
                         "clear_prediction_cache", "cache_counts"):
                setattr(self, name, types.MethodType(
                    getattr(icl.WanICLTransformer3DModel, name), self))

    fake = _Fake()
    install_static_kv_cache(
        fake, StaticKVCacheConfig(max_tokens=128, heads=heads, head_dim=dim_head)
    )

    def _build_mask(q_len, kv_len):
        # sized like _forward_stream builds it per call (kv grows with history)
        zq = torch.zeros(q_len, dtype=torch.int)
        zk = torch.zeros(kv_len, dtype=torch.int)
        icl.ICLAttentionBackend.build_self_mask(
            zq, zk, zq, zk, zq, zk, 64, DEVICE, compile_mask=False,
        )

    q = 8
    hs = torch.randn(1, q, heads * dim_head, dtype=torch.bfloat16)
    pad = torch.zeros(1, 0, heads * dim_head, dtype=torch.bfloat16)

    _build_mask(q, q)
    out_d, _ = attn_dyn(hs, None, pad, None, None, update_cache=1,
                        cache_name="pos")
    out_s, _ = attn_stat(hs, None, pad, None, None, update_cache=1,
                         cache_name="pos")
    assert torch.equal(out_d, out_s)

    # second update_cache=1 forward: history appended statically must equal
    # the dynamic torch.cat result
    _build_mask(q, 2 * q)
    out_d2, _ = attn_dyn(hs, None, pad, None, None, update_cache=1,
                         cache_name="pos")
    out_s2, _ = attn_stat(hs, None, pad, None, None, update_cache=1,
                          cache_name="pos")
    assert torch.equal(out_d2, out_s2)
    dyn_k = attn_dyn.attn_caches["pos"]["k"]
    store = fake._static_kv_state.kv_by_attn[id(attn_stat)]["pos"]
    assert torch.equal(store.view()["k"], dyn_k)
    # backing storage pointer stable across both appends
    assert store.k.shape[1] == store.capacity


# ---------------------------------------------------------------------------
# Shared workspace + memory accounting
# ---------------------------------------------------------------------------

def test_shared_workspace_compaction_two_stores():
    """Two stores share one workspace; sequential compaction stays correct."""
    ws = {
        "k": torch.empty(1, 64, 4, 8),
        "v": torch.empty(1, 64, 4, 8),
    }
    a = StaticKVCache(64, 4, 8, torch.float32, DEVICE, shared_workspace=ws)
    b = StaticKVCache(64, 4, 8, torch.float32, DEVICE, shared_workspace=ws)
    ka, va = _kv(6, seed=1)
    kb, vb = _kv(6, seed=2)
    a.append(ka, va)
    b.append(kb, vb)
    pa, pb = a.k.data_ptr(), b.k.data_ptr()
    keep = torch.tensor([True, False, True, True, False, True])
    a.compact(keep)
    b.compact(keep)  # same workspace, sequential — must not corrupt a
    assert torch.equal(a.view()["k"], ka[:, [0, 2, 3, 5]])
    assert torch.equal(b.view()["k"], kb[:, [0, 2, 3, 5]])
    assert a.k.data_ptr() == pa and b.k.data_ptr() == pb
    assert a.diagnostics()["uses_shared_workspace"] is True
    # stores count only their own backing
    own = a.k.numel() * 4 + a.v.numel() * 4
    assert a.diagnostics()["allocated_bytes"] == own


def test_estimate_static_memory_formula():
    cfg = StaticKVCacheConfig(max_tokens=40960, heads=24, head_dim=128)
    est = estimate_static_memory(cfg, 30)
    kv = 40960 * 24 * 128 * 2
    assert est["per_block_bytes"] == 2 * kv
    assert est["workspace_bytes"] == 2 * kv
    assert est["total_bytes"] == 30 * 2 * kv + 2 * kv + 8 * 40960 * 4
    # ~15.6 GiB target fits with headroom on an 80GB card
    assert est["total_bytes"] < 17 * 2**30


def test_adapter_shared_workspace_not_double_counted():
    a = _fake_transformer(layers=2)
    install_static_kv_cache(
        a, StaticKVCacheConfig(max_tokens=128, heads=2, head_dim=12)
    )
    # force stores + one workspace to exist
    attn = a.blocks[0].attn1
    state = a._static_kv_state
    k, v = _kv(4, heads=2, dim=12)
    from zero_wam.batched_static_cache import _get_or_create_store

    store0 = _get_or_create_store(state, attn, k)
    store0.append(k, v)
    diag = static_cache_diagnostics(a)
    ws_bytes = diag["workspace_bytes"]
    assert ws_bytes == 2 * 128 * 2 * 12 * 4
    block_bytes = sum(b["allocated_bytes"] for b in diag["blocks"])
    # allocated = block own + workspace once + metadata once
    assert diag["allocated_bytes"] == (
        block_bytes + ws_bytes + diag["metadata"]["allocated_bytes"]
    )
    assert diag["memory_estimate"]["total_bytes"] > 0


# ---------------------------------------------------------------------------
# Four-seq-id pruning / slot isolation through the adapter
# ---------------------------------------------------------------------------

def test_prune_preserves_four_packed_seq_ids():
    a = _fake_transformer()
    install_static_kv_cache(
        a, StaticKVCacheConfig(max_tokens=256, heads=2, head_dim=12)
    )
    # emulate packed metadata: [slot0 cond|uncond|slot1 cond|uncond] obs + pred
    seq = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4, dtype=torch.int)
    typ = torch.cat([torch.zeros(8, dtype=torch.int),   # video obs
                     torch.ones(8, dtype=torch.int)])   # action obs
    typ = torch.zeros(16, dtype=torch.int)
    frame = torch.arange(16, dtype=torch.int) % 8
    ctype = torch.zeros(16, dtype=torch.int)  # OBSERVATION
    a._append_metadata(typ, seq, frame, ctype)
    # add prediction tokens for seq 0 and 2
    pseq = torch.tensor([0] * 4 + [2] * 4, dtype=torch.int)
    ptype = torch.zeros(8, dtype=torch.int)
    pframe = torch.full((8,), 99, dtype=torch.int)
    pctype = torch.ones(8, dtype=torch.int)  # PREDICTION
    a._append_metadata(ptype, pseq, pframe, pctype)
    assert len(a.seq_ids_cache) == 24
    a.clear_prediction_cache(-1)  # drop all prediction entries, keep obs
    assert len(a.seq_ids_cache) == 16
    assert sorted(set(a.seq_ids_cache.tolist())) == [0, 1, 2, 3]
    assert (a.cache_type_ids_cache == 0).all()


# ---------------------------------------------------------------------------
# Engine validation with fake server components
# ---------------------------------------------------------------------------

class _FakeEncoder(nn.Module):
    """Deterministic encoder: [B,3,T,H,W] -> [B,96,T,H/16,W/16] (48ch mu/var)."""

    def forward(self, x, feat_cache=None, feat_idx=None):
        import torch.nn.functional as F
        pooled = F.avg_pool3d(x.float(), (1, 16, 16)).to(x.dtype)
        base = pooled.mean(1, keepdim=True)  # [B,1,T,h,w]
        return torch.cat([base, base * 0.5], dim=1).repeat(
            1, 48, 1, 1, 1)  # [B,96,T,h,w] -> chunk(2) gives [B,48,...]


class _FakeVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("c", (), {"latents_mean": [0.0] * 48,
                                     "latents_std": [1.0] * 48,
                                     "patch_size": None})()
        self.encoder = _FakeEncoder()
        self.quant_conv = nn.Identity()
        self.dtype = torch.bfloat16
        self._p = nn.Parameter(torch.zeros(1))


class _FakeStreamingVae:
    def __init__(self):
        self.vae = _FakeVae()
        self.cleared = 0

    def clear_cache(self):
        self.cleared += 1


def _fake_server(heads=4, head_dim=24):
    """Tiny real transformer + canned server surfaces for engine tests."""
    from types import SimpleNamespace
    from wan_va.modules.icl_model import WanICLTransformer3DModel

    torch.manual_seed(3)
    transformer = WanICLTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=heads,
        attention_head_dim=head_dim, in_channels=48, out_channels=48,
        action_dim=6, text_dim=32, freq_dim=64, ffn_dim=128, num_layers=2,
        action_inner_dim=heads * head_dim, action_ffn_dim=128,
        rope_max_seq_len=4096, pos_embed_seq_len=64, enable_mcp=False,
        attn_window=64,
    ).to(torch.bfloat16)
    jc = SimpleNamespace(
        action_per_frame=2, height=64, width=96, obs_cam_keys=["cam_a", "cam_b"],
        action_dim=6, used_action_channel_ids=[0, 1, 2],
        inverse_used_action_channel_ids=[0, 1, 2, 3, 3, 3],
        norm_stat={"q01": [0.0] * 6, "q99": [1.0] * 6},
        action_norm_method="quantiles", num_inference_steps=3,
        action_num_inference_steps=2, icl_rope_h=8, snr_shift=1.0,
        action_snr_shift=1.0, frame_chunk_size=2, guidance_scale=-1.0,
        icl_guidance_scale=5.0,
    )
    server = SimpleNamespace(
        use_icl_model=True,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        transformer=transformer,
        job_config=jc,
        streaming_vae=_FakeStreamingVae(),
        streaming_vae_half=None,
        empty_text_emb=None,
        vae=_FakeVae(),
    )
    g = torch.Generator().manual_seed(5)
    server.encode_prompt = lambda **kw: (
        torch.randn(1, 4, 32, generator=g, dtype=torch.float32).to(torch.bfloat16),
        torch.randn(1, 4, 32, generator=g, dtype=torch.float32).to(torch.bfloat16),
    )
    # ICL latent: [1,4,F,h,w] at the latent resolution the engine derives
    # latent geometry: channels=48, H=height//16=4, W=width//16*2cams=12
    server._load_or_encode_icl = lambda vp, lp: (
        torch.randn(1, 48, 2, 4, 12, generator=g, dtype=torch.float32).to(torch.bfloat16),
        None,
    )
    server._encode_initial_obs = lambda obs: torch.randn(
        1, 48, 1, 4, 12, generator=g, dtype=torch.float32
    ).to(torch.bfloat16)
    server._encode_obs = lambda obs: torch.randn(
        1, 48, 1, 4, 12, generator=g, dtype=torch.float32
    ).to(torch.bfloat16)
    server.postprocess_action = lambda a: a.detach().float().cpu().numpy()
    server.normalize_latents = lambda mu, mean, std: (
        (mu.float() - mean.view(1, -1, 1, 1, 1)) * std.view(1, -1, 1, 1, 1)
    ).to(mu.dtype)
    return server


def _fake_obs(h=64, w=96):
    return {"cam_a": np.zeros((h, w, 3), dtype=np.uint8),
            "cam_b": np.zeros((h, w, 3), dtype=np.uint8)}


def test_engine_lifecycle_with_fake_server(monkeypatch):
    """reset → infer → cache_update → infer through the real tiny transformer."""
    import wan_va.modules.icl_model as icl
    from torch.nn.attention.flex_attention import flex_attention
    from zero_wam.batched_static_engine import (
        BatchedStaticEngine,
        SlotResetSpec,
    )

    monkeypatch.setattr(
        icl, "_compiled_flex_attention",
        lambda q, k, v, **kw: flex_attention(
            q, k, v, block_mask=kw.get("block_mask")
        ),
    )
    monkeypatch.setattr(
        icl, "_compiled_create_block_mask", icl.create_block_mask
        if hasattr(icl, "create_block_mask")
        else __import__(
            "torch.nn.attention.flex_attention", fromlist=["create_block_mask"]
        ).create_block_mask,
    )
    server = _fake_server()
    engine = BatchedStaticEngine(
        server, max_tokens=4096,
        video_guidance_scale=-1.0, icl_guidance_scale=5.0, use_icl=True,
    )
    spec_a = SlotResetSpec("pick the pan", "lat_a.pth", 101)
    spec_b = SlotResetSpec("pick the pan", "lat_b.pth", 202)
    engine.reset([spec_a, spec_b])
    assert engine.calls["icl_pack"] == 1
    # ICL prefixes cached under seq ids 0 (slot0) and 2 (slot1) only;
    # -1 entries are the 128-token alignment padding the runtime caches
    seqs = set(server.transformer.seq_ids_cache.tolist())
    assert seqs <= {0, 2, -1} and {0, 2} <= seqs
    cts = set(server.transformer.cache_type_ids_cache.tolist())
    assert cts == {2}  # ICL_CACHE_TYPE only

    out = engine.infer([{"obs": _fake_obs()}, {"obs": _fake_obs()}])
    assert len(out) == 2
    assert engine.calls["video"] == 4  # 3 steps + final
    assert engine.calls["action"] == 2
    engine.cache_update([{"obs": [_fake_obs()]}, {"obs": [_fake_obs()]}])
    assert engine.calls["obs_video"] == 1 and engine.calls["obs_action"] == 1
    out2 = engine.infer()
    assert len(out2) == 2
    assert engine.slots[0].chunk_idx == 1


def test_engine_packed_seq_ids_match_cfg_convention(monkeypatch):
    """Prediction streams must use 0/1 (slot0) and 2/3 (slot1)."""
    server = _fake_server()
    from zero_wam.batched_static_engine import BatchedStaticEngine, SlotResetSpec

    engine = BatchedStaticEngine(
        server, max_tokens=4096,
        video_guidance_scale=-1.0, icl_guidance_scale=5.0, use_icl=True,
    )
    # fake slot state so _pack_video_stream can run standalone
    for i, s in enumerate(engine.slots):
        s.spec = SlotResetSpec("x", "y", 100 + i)
        s.target_prompt_embeds = torch.zeros(1, 4, 32)
        s.negative_prompt_embeds = torch.zeros(1, 4, 32)
    lat = torch.zeros(1, 48, 2, 4, 12, dtype=torch.bfloat16)
    d, toks = engine._pack_video_stream([lat, lat], 500.0, 1)
    seqs = d["current_seq_ids"]
    assert seqs.unique().tolist() == [0, 1, 2, 3]
    enc = d["encoder_seq_ids"]
    assert enc.unique().tolist() == [0, 1, 2, 3]
    # order: s0-cond | s0-uncond | s1-cond | s1-uncond
    per = toks[0]
    assert (seqs[:per] == 0).all() and (seqs[per:2*per] == 1).all()
    assert (seqs[2*per:3*per] == 2).all() and (seqs[3*per:] == 3).all()


def test_engine_rejects_non_icl_server():
    from zero_wam.batched_static_engine import BatchedStaticEngine
    server = _fake_server()
    server.use_icl_model = False
    with pytest.raises(ValueError):
        BatchedStaticEngine(server, max_tokens=4096)


# ---------------------------------------------------------------------------
# Per-slot VAE wrappers, inactive slots, commanded history
# ---------------------------------------------------------------------------

def test_per_slot_vae_wrappers_share_modules_but_not_cache():
    from wan_va.modules.utils import WanVAEStreamingWrapper

    vae = _FakeVae()
    a = WanVAEStreamingWrapper(vae)
    b = WanVAEStreamingWrapper(vae)
    assert a.encoder is b.encoder is vae.encoder
    assert a.quant_conv is b.quant_conv is vae.quant_conv
    assert a.feat_cache is not b.feat_cache
    if a.feat_cache:
        a.feat_cache[0] = torch.zeros(1)
        assert b.feat_cache[0] is None


def test_engine_creates_independent_slot_vae_caches():
    from zero_wam.batched_static_engine import BatchedStaticEngine

    server = _fake_server()
    engine = BatchedStaticEngine(server, max_tokens=4096)
    assert engine.slots[0].vae is not engine.slots[1].vae
    assert engine.slots[0].vae.encoder is engine.slots[1].vae.encoder
    assert engine.slots[0].vae.feat_cache is not engine.slots[1].vae.feat_cache


def test_engine_inactive_slot_continues_through_lifecycle(monkeypatch):
    """Deactivate slot 1 mid-lifecycle; slot 0 still produces finite output."""
    import wan_va.modules.icl_model as icl
    from torch.nn.attention.flex_attention import flex_attention
    from zero_wam.batched_static_engine import (
        BatchedStaticEngine,
        SlotResetSpec,
    )

    monkeypatch.setattr(
        icl, "_compiled_flex_attention",
        lambda q, k, v, **kw: flex_attention(
            q, k, v, block_mask=kw.get("block_mask")))
    server = _fake_server()
    engine = BatchedStaticEngine(
        server, max_tokens=4096,
        video_guidance_scale=-1.0, icl_guidance_scale=5.0)
    engine.reset([SlotResetSpec("x", "a", 1), SlotResetSpec("x", "b", 2)])
    out0 = engine.infer([{"obs": _fake_obs()}, {"obs": _fake_obs()}])
    engine.set_active(1, False)
    assert engine.slots[1].active is False
    ci = engine.slots[1].chunk_idx
    engine.cache_update(
        [{"obs": [_fake_obs()]}, None],
        [None, None],
    )
    assert engine.slots[1].chunk_idx == ci  # frozen
    # packed stream masks inactive slot to -1
    lat = torch.zeros(1, 48, 2, 4, 12, dtype=torch.bfloat16)
    d, _ = engine._pack_video_stream([lat, lat], 500.0, 1)
    seqs = d["current_seq_ids"].tolist()
    assert all(v in (0, 1, -1) for v in seqs)
    out1 = engine.infer()
    assert len(out1) == 2
    assert np.isfinite(out1[0]).all()
    # inactive slot's chunk index did not advance; active did
    assert engine.slots[0].chunk_idx == 1
    assert engine.slots[1].chunk_idx == 0


def test_commanded_history_cache_write_shapes(monkeypatch):
    """cache_update accepts [1,C,F+1,H,1] normalized commanded histories."""
    import wan_va.modules.icl_model as icl
    from torch.nn.attention.flex_attention import flex_attention
    from zero_wam.batched_static_engine import (
        BatchedStaticEngine,
        SlotResetSpec,
    )

    monkeypatch.setattr(
        icl, "_compiled_flex_attention",
        lambda q, k, v, **kw: flex_attention(
            q, k, v, block_mask=kw.get("block_mask")))
    server = _fake_server()
    engine = BatchedStaticEngine(
        server, max_tokens=4096,
        video_guidance_scale=-1.0, icl_guidance_scale=5.0)
    engine.reset([SlotResetSpec("x", "a", 1), SlotResetSpec("x", "b", 2)])
    engine.infer([{"obs": _fake_obs()}, {"obs": _fake_obs()}])
    hist = [torch.zeros(1, 6, 3, 2, 1, dtype=torch.bfloat16),
            torch.ones(1, 6, 3, 2, 1, dtype=torch.bfloat16)]
    engine.cache_update([{"obs": [_fake_obs()]}, {"obs": [_fake_obs()]}], hist)
    assert engine.calls["obs_video"] == 1 and engine.calls["obs_action"] == 1
    # action tokens for the commanded history: F+1=3 frames x apf=2 = 6/slot
    meta = server.transformer.cache_type_ids_cache
    assert (meta == 0).any()  # OBSERVATION entries present


def test_ema_alpha09_formula():
    from zero_wam.so101_closed_loop import smooth_action

    prev = np.zeros(6)
    pred = np.ones(6)
    got = smooth_action(pred, prev, 0.9)
    assert np.allclose(got, 0.9)
    got2 = smooth_action(pred, got, 0.9)
    assert np.allclose(got2, 0.9 + 0.09)
    with pytest.raises(ValueError):
        smooth_action(pred, prev, 0.0)


def test_pack_metadata_masks_inactive_in_batch4():
    from zero_wam.packed_batch import pack_metadata, slot_metadata

    metas = [
        slot_metadata(
            PackedSlot(i, active=(i != 3), chunk_idx=0, seed=i),
            4, torch.zeros(4, dtype=torch.int), 0, 0, "cpu", cfg_branch=0)
        for i in range(4)
    ]
    m = pack_metadata(*metas)
    assert m["seq_ids"].tolist() == [0] * 4 + [2] * 4 + [4] * 4 + [-1] * 4


def test_batched_validation_runner_schema():
    """The separate runner must import, pin the group protocol, and not be
    imported by existing runners."""
    import importlib

    mod = importlib.import_module("scripts.run_so101_batched_validation")
    assert mod.GROUP_SIZE == 4
    assert mod.SAMPLE_SEED_BASE == 20261014
    # 25 episodes -> 7 groups of 4 with one padded group
    assert math.ceil(25 / mod.GROUP_SIZE) == 7
    # runner must not be imported by existing entry points
    from pathlib import Path

    seq_src = (
        Path("scripts/run_so101_validation_batch.py").read_text()
        + Path("zero_wam/so101_closed_loop.py").read_text()
    )
    assert "run_so101_batched_validation" not in seq_src
    assert "batched_static" not in seq_src


def test_reset_with_padding_slot(monkeypatch):
    """active=False spec -> masked ICL/encoder ids, no prompt/ICL encode."""
    import wan_va.modules.icl_model as icl
    from torch.nn.attention.flex_attention import flex_attention
    from zero_wam.batched_static_engine import (
        BatchedStaticEngine,
        SlotResetSpec,
    )

    monkeypatch.setattr(
        icl, "_compiled_flex_attention",
        lambda q, k, v, **kw: flex_attention(
            q, k, v, block_mask=kw.get("block_mask")))
    server = _fake_server()
    engine = BatchedStaticEngine(
        server, max_tokens=4096,
        video_guidance_scale=-1.0, icl_guidance_scale=5.0)
    specs = [
        SlotResetSpec("pick", "lat_a.pth", 101, active=True),
        SlotResetSpec("pad", None, 0, active=False),
    ]
    engine.reset(specs)
    assert engine.slots[1].active is False
    assert engine.slots[1].prompt_embeds is None
    meta = server.transformer.seq_ids_cache
    # padding slot tokens masked to -1; only slot0 seq 0 appears
    assert set(meta.tolist()) <= {0, -1}
    assert (meta == 0).any()


def test_episode_selector_preserves_canonical_order_and_seeds():
    from scripts.run_so101_batched_validation import select_episode_rows

    rows = [{"id": f"task/ep_{i:03d}"} for i in range(25)]
    assert select_episode_rows(rows, []) == rows
    chosen = select_episode_rows(rows, ["task/ep_024", "task/ep_003"])
    assert [r["id"] for r in chosen] == ["task/ep_003", "task/ep_024"]
    with pytest.raises(ValueError):
        select_episode_rows(rows, ["task/ep_999"])


def test_engine_one_active_three_padding_slots(monkeypatch):
    """1 real + 3 inactive padding slots: infer/cache_update never deref
    None prompt embeds; packed metadata masks pads to -1."""
    import wan_va.modules.icl_model as icl
    from torch.nn.attention.flex_attention import flex_attention
    from zero_wam.batched_static_engine import (
        BatchedStaticEngine,
        SlotResetSpec,
    )

    monkeypatch.setattr(
        icl, "_compiled_flex_attention",
        lambda q, k, v, **kw: flex_attention(
            q, k, v, block_mask=kw.get("block_mask")))
    server = _fake_server()
    engine = BatchedStaticEngine(
        server, max_tokens=4096, batch_size=4,
        video_guidance_scale=-1.0, icl_guidance_scale=5.0)
    specs = [
        SlotResetSpec("pick", "lat_a.pth", 101, active=True),
        SlotResetSpec("pad", None, 0, active=False),
        SlotResetSpec("pad", None, 0, active=False),
        SlotResetSpec("pad", None, 0, active=False),
    ]
    engine.reset(specs)
    assert [s.active for s in engine.slots] == [True, False, False, False]
    assert engine.slots[1].prompt_embeds is None
    obs0 = [{"obs": _fake_obs()}, None, None, None]
    out = engine.infer(obs0)
    assert len(out) == 4 and np.isfinite(out[0]).all()
    engine.cache_update([{"obs": [_fake_obs()]}, None, None, None],
                        [None, None, None, None])
    out2 = engine.infer()
    assert np.isfinite(out2[0]).all()
    # all pad tokens masked
    seqs = set(server.transformer.seq_ids_cache.tolist())
    assert seqs <= {0, 1, -1}


# ---------------------------------------------------------------------------
# Train-replay coordinator
# ---------------------------------------------------------------------------

def test_train_replay_cohort_loading():
    from scripts.run_so101_batched_train_replay import load_cohort

    rows = load_cohort()
    assert [r["id"] for r in rows] == [
        "ball_between_cans/episode_000", "ball_between_cans/episode_001",
        "block_in_bowl/episode_000", "block_in_bowl/episode_001",
        "push_cube_into_tape_square/episode_000",
        "push_cube_into_tape_square/episode_001",
    ]
    assert [r["seed"] for r in rows] == [
        100000, 100001, 100000, 100001, 100000, 100003]
    for r in rows:
        assert Path(r["human_latent"]).is_file()
        assert r["visual_config"]["config_hash"]
        assert r["object_poses_start"]


def test_train_replay_selector():
    from scripts.run_so101_batched_train_replay import (
        COHORT_IDS, load_cohort, select_episode_rows)

    rows = load_cohort()
    assert select_episode_rows(rows, []) == rows
    sel = select_episode_rows(
        rows, [COHORT_IDS[4], COHORT_IDS[0]])
    assert [r["id"] for r in sel] == [COHORT_IDS[0], COHORT_IDS[4]]
    with pytest.raises(ValueError):
        select_episode_rows(rows, ["nope/episode_999"])


def test_train_replay_groups_of_four_with_padding():
    import scripts.run_so101_batched_train_replay as mod
    from scripts.run_so101_batched_train_replay import GROUP_SIZE

    rows = mod.load_cohort()
    groups = [rows[i:i + GROUP_SIZE] for i in range(0, len(rows), GROUP_SIZE)]
    assert len(groups) == 2 and len(groups[0]) == 4 and len(groups[1]) == 2
    # padding spec construction is valid and masked
    from zero_wam.batched_static_engine import SlotResetSpec
    pad = SlotResetSpec(prompt="pad", icl_latent_path=None, seed=0,
                        active=False)
    assert pad.active is False and pad.icl_latent_path is None


class _FakeTrainEnv:
    """records ctor kwargs/reset seed; provides the ValEnv API surface."""

    instances = []

    def __init__(self, render_images=False, visual_config=None):
        self.render_images = render_images
        self.visual_config = visual_config
        self.reset_seed = None
        self.objects = {"ball": np.array([0.1, 0.2, 0.03]),
                        "can_a": np.array([0.4, 0.0, 0.03])}
        self.model = None
        _FakeTrainEnv.instances.append(self)

    def reset(self, seed=None):
        self.reset_seed = seed

    def success(self):
        return False

    def object_pos(self, name):
        return self.objects[name]

    def object_poses(self):
        return {k: [float(v[0]), float(v[1]), float(v[2]), 1, 0, 0, 0]
                for k, v in self.objects.items()}


def test_train_sim_propagates_visual_config_and_seed(monkeypatch):
    import scripts.run_so101_batched_train_replay as mod

    fake_def = type("TD", (), {
        "task_objects": ("ball",),
        "load_classes": lambda self: (_FakeTrainEnv, None),
    })()
    monkeypatch.setattr(
        "sim.train.tasks.catalog.load_train_task", lambda name: fake_def)
    monkeypatch.setattr(
        "sim.rebuild_wrist_v2.configure_clean_wrist",
        lambda model: {"near_meters": 0.0})
    monkeypatch.setattr(
        "sim.rebuild_wrist_v2.clean_wrist_option", lambda: None)

    _FakeTrainEnv.instances.clear()
    vc = types.SimpleNamespace(config_hash="abc")  # pre-built VisualConfig
    sim = mod.TrainSim("ball_between_cans", 100000, vc)
    env = _FakeTrainEnv.instances[-1]
    assert env.reset_seed == 100000
    assert env.render_images is False
    assert getattr(env.visual_config, "config_hash", None) == "abc"
    assert sim.task_objects == ("ball",)


def test_start_pose_check(monkeypatch):
    import scripts.run_so101_batched_train_replay as mod

    fake_def = type("TD", (), {
        "task_objects": ("ball",),
        "load_classes": lambda self: (_FakeTrainEnv, None),
    })()
    monkeypatch.setattr(
        "sim.train.tasks.catalog.load_train_task", lambda name: fake_def)
    monkeypatch.setattr(
        "sim.rebuild_wrist_v2.configure_clean_wrist",
        lambda model: {"near_meters": 0.0})
    monkeypatch.setattr(
        "sim.rebuild_wrist_v2.clean_wrist_option", lambda: None)

    sim = mod.TrainSim("ball_between_cans", 100000,
                       types.SimpleNamespace(config_hash="x"))
    frozen = {"ball": [0.1, 0.2, 0.03, 1, 0, 0, 0]}
    ok = mod.check_start_poses(sim, frozen)
    assert ok["pass"] and ok["max_abs_error_m"] == 0.0
    bad = mod.check_start_poses(sim, {"ball": [0.11, 0.2, 0.03, 1, 0, 0, 0]})
    assert not bad["pass"] and abs(bad["max_abs_error_m"] - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# Sequential train-replay + causal-guidance flag
# ---------------------------------------------------------------------------

def test_sequential_replay_runner_import_and_defaults():
    import importlib

    mod = importlib.import_module("scripts.run_so101_sequential_train_replay")
    assert mod.TrainSim is not None
    assert callable(mod.run_episode_train)
    # isolation: existing runners must not import the sequential replay
    from pathlib import Path
    src = Path("zero_wam/so101_closed_loop.py").read_text()
    assert "train_replay" not in src


def test_chunk_causal_guidance_flag_and_custom_manifest():
    import importlib
    import json as _json

    mod = importlib.import_module(
        "scripts.run_zero_wam_chunk_causal_validation")
    # default preserves existing behavior
    import sys
    old = sys.argv
    try:
        sys.argv = ["x", "--checkpoint", "c", "--base-model", "b",
                    "--loader-root", "l", "--latent-root", "lt",
                    "--output", "o"]
        args = mod.parse_args()
        assert args.video_guidance_scale == 5.0
        sys.argv.append("--video-guidance-scale"); sys.argv.append("-1.0")
        assert mod.parse_args().video_guidance_scale == -1.0
    finally:
        sys.argv = old
    # custom manifest validates with canonical hash + files
    from zero_wam.chunk_causal_validation import (
        canonical_sha256, validate_manifest_files)
    m = _json.loads(Path(
        "data/hf_bucket_ICL-so101/train_replay_probe/chunk_causal_manifest.json"
    ).read_text())
    expected = m.pop("manifest_sha256")
    assert canonical_sha256(m) == expected
    validate_manifest_files(m)
    assert len(m["samples"]) == 6
    assert all(s["target_control_start"] == 8 and s["target_control_end"] == 24
               for s in m["samples"])
    assert all(s["observation_frame_indices"] == [0, 2, 4, 6, 8]
               for s in m["samples"])


# ---------------------------------------------------------------------------
# Multi-boundary teacher-forced protocol
# ---------------------------------------------------------------------------

def test_boundary_selection_and_seed():
    from scripts.run_so101_multi_boundary_validation import (
        boundary_seed, select_boundaries)

    bounds = select_boundaries(507)
    assert [b for b, _ in bounds] == [56, 152, 248, 344, 440]
    assert all((b - 8) % 16 == 0 and b + 16 <= 507 for b, _ in bounds)
    assert len({b for b, _ in bounds}) == len(bounds)
    s1 = boundary_seed("task__ep", 56)
    s2 = boundary_seed("task__ep", 56)
    s3 = boundary_seed("task__ep", 72)
    assert s1 == s2 != s3


class _FakeTFModel:
    """records infer payloads to assert full-history caching, no leakage."""

    def __init__(self):
        self.calls = []
        self.last_predicted_actions = None
        self.normalized_writes = []  # last_predicted_actions per cache step
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.action_mask = torch.zeros(1, 30, dtype=torch.bool)
        self.action_mask[0, :6] = True

    @property
    def _last(self):
        return self._last_pa

    @_last.setter
    def _last(self, v):
        self._last_pa = v

    def preprocess_action(self, block):
        out = torch.zeros(1, 30, 2, 8, dtype=torch.float32)
        out[:, :6] = torch.tensor(np.asarray(block, dtype=np.float32))
        return out

    def infer(self, payload):
        self.calls.append(dict(payload))
        if payload.get("reset"):
            return {}
        if payload.get("compute_kv_cache"):
            return {}
        return {"action": np.zeros((6, 2, 8), dtype=np.float32)}


def test_multi_boundary_full_history_teacher_force():
    import importlib

    mod = importlib.import_module(
        "scripts.run_so101_multi_boundary_validation")
    model = _FakeTFModel()
    actions = np.arange(300 * 6, dtype=np.float32).reshape(300, 6)
    states = np.zeros((300, 6), dtype=np.float32)
    captured = {}

    def obs_by_frame(f):
        return {"observation.images.front": np.full((2, 2, 3), f, np.uint8),
                "observation.images.wrist": np.full((2, 2, 3), f, np.uint8),
                "observation.state": np.asarray(states[f], np.float32)}

    seen = []
    boundary = 120  # 8 + 16*7
    writes = []
    def record(block, zero_first=False):
        norm = model.preprocess_action(block)
        norm[:, ~model.action_mask[0]] = 0
        if zero_first:
            norm[:, :, 0] = 0
        writes.append((norm.clone(), zero_first))
        return norm
    result, n_calls = mod.teacher_force_to_boundary(
        model, obs_by_frame, actions, boundary, record, seen.append)

    cache_calls = [c for c in model.calls if c.get("compute_kv_cache")]
    # 1 prefix call + (120-8)//16 = 7 GT blocks
    assert n_calls == 8 and len(cache_calls) == 8
    # prefix call: 4 obs at frames 2,4,6,8
    assert len(cache_calls[0]["obs"]) == 4
    # deployment chunk-0 semantics: first normalized token zeroed only on
    # the initial 0:8 prefix block; GT blocks are never zero_first
    assert len(writes) == 8
    assert writes[0][1] is True
    assert torch.all(writes[0][0][:, :, 0] == 0)
    for j, (norm, zf) in enumerate(writes[1:], start=1):
        assert zf is False
        c0 = 8 + 16 * (j - 1)
        np.testing.assert_array_equal(
            norm[0, :6].numpy().reshape(6, 16).T,
            actions[c0:c0 + 16])
    # GT block j=1..7: 8 keyframes every 2 controls starting at c0+2
    for j, call in enumerate(cache_calls[1:], start=1):
        c0 = 8 + 16 * (j - 1)
        states_seen = [o["observation.state"][0] for o in call["obs"]]
        frames_expected = [c0 + 2 * m for m in range(1, 9)]
        # state values are all zeros here; verify obs count + cadence via
        # the frames baked into pixel values
        pix = [int(o["observation.images.front"][0, 0, 0]) for o in call["obs"]]
        assert pix == frames_expected
        assert len(call["obs"]) == 8
        # teacher-forced action block = actions[c0:c0+16]
        hist = call["state"]
        np.testing.assert_array_equal(
            np.asarray(hist).reshape(6, 16).T,
            actions[c0:c0 + 16])
    # final infer obs = boundary frame 120
    last = model.calls[-1]
    assert not last.get("compute_kv_cache") and not last.get("reset")
    assert int(last["obs"]["observation.images.front"][0, 0, 0]) == 120
    # no scorer fields ever cross into inference payloads
    forbidden = {"target_actions", "future_actions", "target_control_start",
                 "target_control_end", "metrics", "target"}
    for call in model.calls:
        assert not (forbidden & set(call.keys()))
        obs_list = call.get("obs")
        if isinstance(obs_list, list):
            for o in obs_list:
                assert not (forbidden & set(o.keys()))


def test_multi_boundary_manifest_hash_roundtrip():
    import json as _json

    from zero_wam.chunk_causal_validation import (
        canonical_sha256, validate_manifest_files)
    m = _json.loads(Path(
        "data/hf_bucket_ICL-so101/train_replay_probe/"
        "multi_boundary_manifest.json").read_text())
    expected = m.pop("manifest_sha256")
    assert canonical_sha256(m) == expected
    validate_manifest_files(m)
    assert sum(len(s["boundaries"]) for s in m["samples"]) == 30
    for s in m["samples"]:
        for b in s["boundaries"]:
            assert (b["boundary"] - 8) % 16 == 0
            assert b["target_control_end"] <= s["num_controls"]


def test_multi_boundary_aggregate_no_collision():
    from scripts.run_so101_multi_boundary_validation import aggregate_multi

    def row(sid, b, model, pers):
        return {"sample_id": sid, "task": sid.split("__")[0],
                "boundary": b, "fraction": 0.5,
                "metrics": {
                    "model": {"overall_physical_mae": model,
                              "overall_physical_rmse": model},
                    "persistence": {"overall_physical_mae": pers,
                                    "overall_physical_rmse": pers}}}

    # same numeric boundary in two episodes must not collide
    rows = [row("task_a__ep0", 56, 0.3, 0.5), row("task_b__ep0", 56, 1.0, 0.2),
            row("task_a__ep0", 72, 0.1, 0.9)]
    agg = aggregate_multi(rows)
    assert agg["num_boundaries"] == 3
    assert agg["model_beats_persistence_count"] == 2
    pb = agg["per_boundary_model_mae"]
    assert len(pb) == 3 and {(x["sample_id"], x["boundary"]) for x in pb} == {
        ("task_a__ep0", 56), ("task_b__ep0", 56), ("task_a__ep0", 72)}
    keyed = {f"{x['sample_id']}@{x['boundary']}": x for x in pb}
    assert keyed["task_b__ep0@56"]["model_mae"] == 1.0
    assert not keyed["task_b__ep0@56"]["model_beats_persistence"]
