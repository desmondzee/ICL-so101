"""Unit tests for the opt-in static-KV prototype (new files only)."""

from __future__ import annotations

import types

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

class _FakeVae:
    def __init__(self):
        self.config = type("c", (), {"latents_mean": [0.0] * 48,
                                     "latents_std": [1.0] * 48})()
        self.cleared = 0

    def clear_cache(self):
        self.cleared += 1


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
    return server


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

    out = engine.infer([{"obs": {}}, {"obs": {}}])
    assert len(out) == 2
    assert engine.calls["video"] == 4  # 3 steps + final
    assert engine.calls["action"] == 2
    engine.cache_update([{"obs": [{}]}, {"obs": [{}]}])
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
