"""Opt-in static KV cache for WanICLTransformer3DModel inference.

New-file-only prototype: installs per-instance replacement methods so the
existing dynamic ``torch.cat`` append / boolean-prune path remains byte-for-byte
untouched.  Static caches keep the *same* semantics as the dynamic path:

* one self-attention K/V store per ``block.attn1`` and cache name
* transformer-level metadata (type/seq/frame/cache-type IDs) kept in lockstep
* ``clear_prediction_cache`` retains ICL + the 64-frame observation window and
  drops prediction entries, exactly like the dynamic implementation

Only the ``"pos"`` cache name is supported; any other name raises.  Cross
attention (``attn2``) is never cached upstream and is left completely alone.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Optional

import torch

CACHE_NAME = "pos"
ICL_CACHE_TYPE = 2
_METADATA_FIELDS = (
    "type_ids_cache",
    "seq_ids_cache",
    "frame_ids_cache",
    "cache_type_ids_cache",
)


class CacheOverflowError(RuntimeError):
    """Raised before any mutation when a write would exceed capacity."""


def _aligned_capacity(max_tokens: int, align: int) -> int:
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if align <= 0:
        raise ValueError(f"align must be positive, got {align}")
    # Physical capacity includes the 128-token alignment padding every stream
    # chunk contributes, so `max_tokens` of logical payload always fits.
    return math.ceil(max_tokens / align) * align


@dataclass(frozen=True)
class StaticKVCacheConfig:
    """Sizing for static per-block K/V + metadata stores."""

    max_tokens: int
    batch_slots: int = 2
    heads: int = 24
    head_dim: int = 128
    dtype: torch.dtype = torch.bfloat16
    align: int = 128

    @property
    def capacity(self) -> int:
        return _aligned_capacity(self.max_tokens, self.align)


class StaticKVCache:
    """Contiguous-prefix K/V store; backing tensors never move or resize.

    `shared_workspace` may hold ``{"k", "v"}`` scratch tensors of at least this
    store's capacity — compaction is sequential across blocks, so all stores
    may safely share one adapter-owned workspace.  Without it each store owns
    private scratch (used by the unit tests).
    """

    def __init__(
        self,
        max_tokens: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        align: int = 128,
        shared_workspace: Optional[dict] = None,
    ):
        self.capacity = _aligned_capacity(max_tokens, align)
        self.max_tokens = max_tokens
        self.heads = heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.k = torch.empty(
            (1, self.capacity, heads, head_dim), dtype=dtype, device=device
        )
        self.v = torch.empty_like(self.k)
        self._scratch = shared_workspace  # None → private, allocated lazily
        self._scratch_k = None
        self._scratch_v = None
        self.length = 0
        self.appends = 0
        self.compacts = 0

    @property
    def scratch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._scratch is not None:
            return self._scratch["k"], self._scratch["v"]
        if self._scratch_k is None:
            self._scratch_k = torch.empty_like(self.k)
            self._scratch_v = torch.empty_like(self.v)
        return self._scratch_k, self._scratch_v

    def _check_keys(self, key: torch.Tensor, value: torch.Tensor) -> int:
        if key.shape != value.shape:
            raise ValueError(f"k/v shape mismatch: {key.shape} vs {value.shape}")
        if key.ndim != 4 or key.shape[0] != 1:
            raise ValueError(f"expected [1, n, heads, head_dim], got {key.shape}")
        if key.shape[2] != self.heads or key.shape[3] != self.head_dim:
            raise ValueError(
                f"incompatible heads/head_dim {tuple(key.shape[2:])}; "
                f"expected ({self.heads}, {self.head_dim})"
            )
        return key.shape[1]

    def can_append(self, n: int) -> bool:
        return self.length + n <= self.capacity

    def append(self, key: torch.Tensor, value: torch.Tensor) -> int:
        """Bounds-checked atomic append; returns the start offset.

        The capacity check runs before any write, so a rejected append leaves
        the store bit-identical.
        """
        n = self._check_keys(key, value)
        start = self.length
        if not self.can_append(n):
            raise CacheOverflowError(
                f"append of {n} tokens exceeds capacity "
                f"{self.capacity} (occupied {self.length})"
            )
        self.k[:, start : start + n] = key
        self.v[:, start : start + n] = value
        self.length = start + n
        self.appends += 1
        return start

    def rollback(self, start: int) -> None:
        """Drop tokens appended at/after `start` (append-failure recovery)."""
        if not 0 <= start <= self.length:
            raise ValueError(f"rollback marker {start} outside [0, {self.length}]")
        self.length = start

    def view(self) -> dict[str, torch.Tensor]:
        """Prefix views; attention only ever sees ``[:length]``."""
        return {"k": self.k[:, : self.length], "v": self.v[:, : self.length]}

    def compact(self, keep: torch.Tensor) -> int:
        """Keep-mask compaction via index-select into scratch, then copy-back.

        `keep` may be a boolean mask of length `self.length` or an index tensor.
        Backing tensors, scratch, and their data_ptrs are never replaced.
        """
        if keep.dtype == torch.bool:
            if keep.numel() != self.length:
                raise ValueError(
                    f"keep mask has {keep.numel()} entries for {self.length} tokens"
                )
            index = keep.to(self.device).nonzero(as_tuple=False).flatten()
        else:
            index = keep.to(self.device).long()
        new_length = int(index.numel())
        # Gather through reusable scratch so backing storage is untouched by
        # any partially-overlapping read/write.  Scratch may be a shared
        # adapter workspace — compaction is sequential across blocks.
        scratch_k, scratch_v = self.scratch
        scratch_k[:, :new_length] = torch.index_select(
            self.k[:, : self.length], 1, index
        )
        scratch_v[:, :new_length] = torch.index_select(
            self.v[:, : self.length], 1, index
        )
        self.k[:, :new_length] = scratch_k[:, :new_length]
        self.v[:, :new_length] = scratch_v[:, :new_length]
        # Invalidate the vacated tail so stale tokens are never readable.
        self.k[:, new_length : self.length] = 0
        self.v[:, new_length : self.length] = 0
        self.length = new_length
        self.compacts += 1
        return new_length

    def clear(self) -> None:
        self.length = 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "max_tokens": self.max_tokens,
            "length": self.length,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "dtype": str(self.dtype),
            "k_data_ptr": self.k.data_ptr(),
            "v_data_ptr": self.v.data_ptr(),
            "appends": self.appends,
            "compacts": self.compacts,
            "uses_shared_workspace": self._scratch is not None,
            # Shared workspace bytes are reported once at adapter level; a
            # store only counts the scratch it actually owns.
            "allocated_bytes": int(
                self.k.numel() * self.k.element_size()
                + self.v.numel() * self.v.element_size()
                + (
                    0
                    if self._scratch is not None or self._scratch_k is None
                    else self._scratch_k.numel() * self._scratch_k.element_size()
                    + self._scratch_v.numel() * self._scratch_v.element_size()
                )
            ),
        }


class StaticMetadataCache:
    """Static store for the four transformer metadata vectors (-1 fill)."""

    FIELDS = _METADATA_FIELDS

    def __init__(self, max_tokens: int, device: torch.device, align: int = 128):
        self.capacity = _aligned_capacity(max_tokens, align)
        self.max_tokens = max_tokens
        self.device = device
        self._store = {
            name: torch.full(
                (self.capacity,), -1, dtype=torch.int, device=device
            )
            for name in self.FIELDS
        }
        self._scratch = {
            name: torch.empty(
                (self.capacity,), dtype=torch.int, device=device
            )
            for name in self.FIELDS
        }
        self.length = 0
        self.appends = 0
        self.compacts = 0

    def append(
        self,
        type_ids: torch.Tensor,
        seq_ids: torch.Tensor,
        frame_ids: torch.Tensor,
        cache_type_ids: torch.Tensor,
    ) -> int:
        n = type_ids.numel()
        for name, value in zip(
            self.FIELDS, (type_ids, seq_ids, frame_ids, cache_type_ids)
        ):
            if value.numel() != n:
                raise ValueError(
                    f"metadata length mismatch: {name}={value.numel()} vs {n}"
                )
        start = self.length
        if start + n > self.capacity:
            raise CacheOverflowError(
                f"metadata append of {n} exceeds capacity {self.capacity} "
                f"(occupied {self.length})"
            )
        for name, value in zip(
            self.FIELDS, (type_ids, seq_ids, frame_ids, cache_type_ids)
        ):
            self._store[name][start : start + n] = value.to(
                device=self.device, dtype=torch.int
            )
        self.length = start + n
        self.appends += 1
        return start

    def compact(self, keep: torch.Tensor) -> int:
        if keep.dtype == torch.bool:
            if keep.numel() != self.length:
                raise ValueError(
                    f"keep mask has {keep.numel()} entries for {self.length} tokens"
                )
            index = keep.to(self.device).nonzero(as_tuple=False).flatten()
        else:
            index = keep.to(self.device).long()
        new_length = int(index.numel())
        for name in self.FIELDS:
            store = self._store[name]
            scratch = self._scratch[name]
            # index-select into the reusable scratch, then copy back — the
            # backing store is never read and written in the same operation.
            scratch[:new_length] = torch.index_select(
                store[: self.length], 0, index
            )
            store[:new_length] = scratch[:new_length]
            store[new_length : self.length] = -1
        self.length = new_length
        self.compacts += 1
        return new_length

    def clear(self) -> None:
        for store in self._store.values():
            store[: self.length] = -1
        self.length = 0

    def view(self, field: str) -> torch.Tensor:
        return self._store[field][: self.length]

    def views(self) -> dict[str, torch.Tensor]:
        return {name: self.view(name) for name in self.FIELDS}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "length": self.length,
            "appends": self.appends,
            "compacts": self.compacts,
            "allocated_bytes": int(
                sum(
                    s.numel() * s.element_size()
                    for s in (*self._store.values(), *self._scratch.values())
                )
            ),
            "data_ptrs": {
                name: self._store[name].data_ptr() for name in self.FIELDS
            },
        }


def estimate_static_memory(
    config: StaticKVCacheConfig, num_blocks: int
) -> dict[str, int]:
    """Bytes the adapter will allocate: per-block K/V backings, ONE shared
    compaction workspace per (device, dtype), and the metadata store."""
    kv = config.capacity * config.heads * config.head_dim * 2  # bf16 elements→B
    per_block = 2 * kv  # k + v backing
    workspace = 2 * kv  # shared k + v scratch, counted once
    metadata = 8 * config.capacity * 4  # 4 int32 stores + 4 int32 scratch
    return {
        "per_block_bytes": per_block,
        "num_blocks": num_blocks,
        "workspace_bytes": workspace,
        "metadata_bytes": metadata,
        "total_bytes": per_block * num_blocks + workspace + metadata,
    }


class _StaticAdapterState:
    """Per-transformer adapter state; removed wholesale on uninstall."""

    def __init__(self, transformer, config: StaticKVCacheConfig):
        self.config = config
        self.cache_name = CACHE_NAME
        self.metadata: Optional[StaticMetadataCache] = None
        self.kv_by_attn: dict[int, dict[str, StaticKVCache]] = {}
        self.workspaces: dict[tuple, dict[str, torch.Tensor]] = {}
        self.saved_model_methods: dict[str, Any] = {}
        self.saved_attn_methods: dict[int, dict[str, Any]] = {}
        self.transformer = transformer
        self.memory_estimate: dict[str, int] = {}

    def workspace(self, device, dtype) -> dict[str, torch.Tensor]:
        key = (str(device), str(dtype))
        ws = self.workspaces.get(key)
        if ws is None:
            shape = (
                1,
                self.config.capacity,
                self.config.heads,
                self.config.head_dim,
            )
            ws = self.workspaces[key] = {
                "k": torch.empty(shape, dtype=dtype, device=device),
                "v": torch.empty(shape, dtype=dtype, device=device),
            }
        return ws


def _name_check(state: _StaticAdapterState, cache_name: str) -> None:
    if cache_name != state.cache_name:
        raise ValueError(
            f"static adapter supports only cache_name={state.cache_name!r}; "
            f"got {cache_name!r}"
        )


def _set_metadata_views(transformer, meta: StaticMetadataCache) -> None:
    """Mirror the dynamic path's post-mutation state: ``[:length]`` prefix
    tensors, including empty tensors after a prune-to-zero (the dynamic path
    leaves empty tensors, not ``None``)."""
    for name in _METADATA_FIELDS:
        setattr(transformer, name, meta.view(name))


def _clear_metadata_attrs(transformer) -> None:
    """Mirror ``clear_cache``: the four attributes become ``None``."""
    for name in _METADATA_FIELDS:
        setattr(transformer, name, None)


def _static_append_metadata(self, query_types, query_seq_ids, query_frame_ids,
                            cache_type_ids):
    state: _StaticAdapterState = self._static_kv_state
    stored_types = query_types.clone()
    stored_types[cache_type_ids == ICL_CACHE_TYPE] = ICL_CACHE_TYPE
    state.metadata.append(
        stored_types, query_seq_ids, query_frame_ids, cache_type_ids
    )
    _set_metadata_views(self, state.metadata)


def _static_prune_cache(self, keep: torch.Tensor, cache_name: str = "pos"):
    state: _StaticAdapterState = self._static_kv_state
    _name_check(state, cache_name)
    if self.type_ids_cache is None:
        return
    state.metadata.compact(keep)
    _set_metadata_views(self, state.metadata)
    for block in self.blocks:
        static = state.kv_by_attn.get(id(block.attn1), {}).get(cache_name)
        if static is not None:
            static.compact(keep)


def _static_clear_cache(self, cache_name: str = "pos"):
    state: _StaticAdapterState = self._static_kv_state
    _name_check(state, cache_name)
    state.metadata.clear()
    _clear_metadata_attrs(self)
    if hasattr(self, "blocks"):
        for block in self.blocks:
            entry = state.kv_by_attn.get(id(block.attn1), {}).get(cache_name)
            if entry is not None:
                entry.clear()


def _get_or_create_store(state: _StaticAdapterState, attn,
                         key: torch.Tensor) -> StaticKVCache:
    slot = state.kv_by_attn[id(attn)]
    static = slot.get(state.cache_name)
    if static is None:
        static = slot[state.cache_name] = StaticKVCache(
            state.config.max_tokens,
            state.config.heads,
            state.config.head_dim,
            key.dtype,
            key.device,
            state.config.align,
            shared_workspace=state.workspace(key.device, key.dtype),
        )
    return static


def _static_attn_forward(
    self,
    hs_latent: Optional[torch.Tensor],
    hs_action: Optional[torch.Tensor],
    hs_pad: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor],
    rotary_emb: Optional[torch.Tensor],
    update_cache: int,
    cache_name: str,
):
    state: _StaticAdapterState = self._static_kv_adapter_state
    if encoder_hidden_states is not None:
        # Cross attention is never cached; run the untouched original.
        return state.saved_attn_methods[id(self)]["forward"](
            hs_latent, hs_action, hs_pad, encoder_hidden_states,
            rotary_emb, update_cache, cache_name,
        )
    _name_check(state, cache_name)

    from wan_va.modules.icl_model import (
        ICLAttentionBackend,
        _apply_rotary_emb,
    )

    query, key, value, latent_length, action_length = self._project(
        hs_latent, hs_action, hs_pad, None
    )
    if rotary_emb is not None:
        query = _apply_rotary_emb(query, rotary_emb)
        key = _apply_rotary_emb(key, rotary_emb)

    static = state.kv_by_attn[id(self)].get(cache_name)
    if update_cache:
        static = _get_or_create_store(state, self, key)
        start = static.length
        # Append first, then attend over the [:length] prefix view — identical
        # to the dynamic path's cat-then-attend-then-store.  On failure the
        # append rolls back so the store is exactly as before the call.
        static.append(key, value)
        try:
            hidden_states = ICLAttentionBackend.apply(
                query,
                static.view()["k"],
                static.view()["v"],
                cross_attention=False,
                block_mask=self.self_block_mask,
            ).flatten(2, 3)
        except Exception:
            static.rollback(start)
            raise
    else:
        if static is not None and static.length:
            key = torch.cat([static.view()["k"], key], dim=1)
            value = torch.cat([static.view()["v"], value], dim=1)
        hidden_states = ICLAttentionBackend.apply(
            query,
            key,
            value,
            cross_attention=False,
            block_mask=self.self_block_mask,
        ).flatten(2, 3)

    out_latent = None
    out_action = None
    if latent_length:
        out_latent = self.to_out[1](
            self.to_out[0](hidden_states[:, :latent_length])
        )
    if action_length:
        action_start = latent_length
        out_action = self.action_to_out[1](
            self.action_to_out[0](
                hidden_states[:, action_start : action_start + action_length]
            )
        )
    return out_latent, out_action


def _static_attn_prune_cache(self, cache_name: str, keep: torch.Tensor) -> None:
    state: _StaticAdapterState = self._static_kv_adapter_state
    _name_check(state, cache_name)
    static = state.kv_by_attn.get(id(self), {}).get(cache_name)
    if static is not None:
        static.compact(keep)


def _static_attn_clear_cache(self, cache_name: str) -> None:
    state: _StaticAdapterState = self._static_kv_adapter_state
    _name_check(state, cache_name)
    static = state.kv_by_attn.get(id(self), {}).get(cache_name)
    if static is not None:
        static.clear()


_MODEL_METHODS = ("clear_cache", "_append_metadata", "_prune_cache")
_ATTN_METHODS = ("forward", "prune_cache", "clear_cache")


def check_static_budget(
    estimate_bytes: int,
    device,
    safety_fraction: float = 0.85,
    allocated_bytes: Optional[int] = None,
    total_bytes: Optional[int] = None,
) -> dict:
    """Reject a static-cache plan before GPU allocation.

    Returns the budget report; raises MemoryError when `allocated + estimate`
    exceeds `safety_fraction` of total device memory.
    """
    if allocated_bytes is None:
        allocated_bytes = torch.cuda.memory_allocated(device)
    if total_bytes is None:
        total_bytes = torch.cuda.get_device_properties(device).total_memory
    projected = allocated_bytes + estimate_bytes
    report = {
        "estimate_bytes": int(estimate_bytes),
        "allocated_bytes": int(allocated_bytes),
        "total_bytes": int(total_bytes),
        "projected_bytes": int(projected),
        "safety_fraction": safety_fraction,
        "allowed": projected <= total_bytes * safety_fraction,
    }
    if not report["allowed"]:
        raise MemoryError(
            f"static cache estimate {estimate_bytes / 2**30:.1f} GiB + current "
            f"{allocated_bytes / 2**30:.1f} GiB would exceed "
            f"{safety_fraction:.0%} of {total_bytes / 2**30:.1f} GiB — "
            "refusing to allocate"
        )
    return report


def install_static_kv_cache(transformer, config: StaticKVCacheConfig,
                            safety_fraction: float = 0.85):
    """Bind static-cache methods onto this transformer instance + its attn1s.

    Only instance attributes are set; class definitions and every sibling
    model are untouched.  Existing dynamic cache contents are migrated into
    the static stores (append order preserved).  On CUDA the full estimated
    allocation is checked against `safety_fraction` of total device memory
    BEFORE any static tensor is allocated.
    """
    if getattr(transformer, "_static_kv_state", None) is not None:
        raise RuntimeError("static KV adapter already installed")
    if not hasattr(transformer, "blocks") or not hasattr(transformer, "attn_window"):
        raise TypeError("expected a WanICLTransformer3DModel-like module")
    if len(list(transformer.blocks)) == 0:
        raise ValueError("transformer has no blocks")

    first_attn = transformer.blocks[0].attn1
    if first_attn.heads != config.heads:
        raise ValueError(
            f"config.heads={config.heads} != module heads={first_attn.heads}"
        )
    if first_attn.inner_dim // first_attn.heads != config.head_dim:
        raise ValueError(
            f"config.head_dim={config.head_dim} != module "
            f"{first_attn.inner_dim // first_attn.heads}"
        )

    param = next(transformer.parameters())
    estimate = estimate_static_memory(config, len(list(transformer.blocks)))
    if param.is_cuda:
        check_static_budget(
            estimate["total_bytes"], param.device, safety_fraction
        )
    state = _StaticAdapterState(transformer, config)
    state.memory_estimate = estimate
    state.metadata = StaticMetadataCache(
        config.max_tokens, param.device, config.align
    )

    for name in _MODEL_METHODS:
        state.saved_model_methods[name] = getattr(transformer, name)
    transformer._static_kv_state = state
    transformer.clear_cache = types.MethodType(_static_clear_cache, transformer)
    transformer._append_metadata = types.MethodType(
        _static_append_metadata, transformer
    )
    transformer._prune_cache = types.MethodType(_static_prune_cache, transformer)

    for block in transformer.blocks:
        attn = block.attn1
        saved = {}
        for name in _ATTN_METHODS:
            saved[name] = getattr(attn, name)
        state.saved_attn_methods[id(attn)] = saved
        state.kv_by_attn[id(attn)] = {}
        attn._static_kv_adapter_state = state
        attn.forward = types.MethodType(_static_attn_forward, attn)
        attn.prune_cache = types.MethodType(_static_attn_prune_cache, attn)
        attn.clear_cache = types.MethodType(_static_attn_clear_cache, attn)

    # Migrate any live dynamic cache contents into the static stores, then
    # empty the dynamic dicts (their state now lives in the static buffers).
    if transformer.type_ids_cache is not None:
        state.metadata.append(
            transformer.type_ids_cache,
            transformer.seq_ids_cache,
            transformer.frame_ids_cache,
            transformer.cache_type_ids_cache,
        )
        _set_metadata_views(transformer, state.metadata)
    for block in transformer.blocks:
        attn = block.attn1
        for name, cache in list(getattr(attn, "attn_caches", {}).items()):
            _name_check(state, name)
            store = _get_or_create_store(state, attn, cache["k"])
            store.append(cache["k"], cache["v"])
        attn.attn_caches = {}
    return state


def uninstall_static_kv_cache(transformer) -> None:
    """Restore original bound methods; move static contents back to dynamic."""
    state = getattr(transformer, "_static_kv_state", None)
    if state is None:
        raise RuntimeError("static KV adapter is not installed")

    meta = state.metadata
    if meta.length:
        for name in _METADATA_FIELDS:
            setattr(transformer, name, meta.view(name).clone())
    else:
        for name in _METADATA_FIELDS:
            setattr(transformer, name, None)
    for block in transformer.blocks:
        attn = block.attn1
        stores = state.kv_by_attn.get(id(attn), {})
        attn.attn_caches = {
            name: {"k": s.view()["k"].clone(), "v": s.view()["v"].clone()}
            for name, s in stores.items()
        }
        saved = state.saved_attn_methods[id(attn)]
        for name, fn in saved.items():
            setattr(attn, name, fn)
        del attn._static_kv_adapter_state

    for name, fn in state.saved_model_methods.items():
        setattr(transformer, name, fn)
    del transformer._static_kv_state


def static_cache_diagnostics(transformer) -> dict[str, Any]:
    state = getattr(transformer, "_static_kv_state", None)
    if state is None:
        return {"installed": False}
    per_block = []
    total_bytes = 0
    for i, block in enumerate(transformer.blocks):
        for name, store in state.kv_by_attn.get(id(block.attn1), {}).items():
            d = store.diagnostics()
            d["block"] = i
            d["cache_name"] = name
            per_block.append(d)
            total_bytes += d["allocated_bytes"]
    meta_diag = state.metadata.diagnostics()
    workspace_bytes = int(
        sum(
            ws["k"].numel() * ws["k"].element_size()
            + ws["v"].numel() * ws["v"].element_size()
            for ws in state.workspaces.values()
        )
    )
    return {
        "installed": True,
        "cache_name": state.cache_name,
        "config": {
            "max_tokens": state.config.max_tokens,
            "capacity": state.config.capacity,
            "batch_slots": state.config.batch_slots,
            "heads": state.config.heads,
            "head_dim": state.config.head_dim,
            "dtype": str(state.config.dtype),
            "align": state.config.align,
        },
        "memory_estimate": state.memory_estimate,
        "metadata": meta_diag,
        "blocks": per_block,
        "workspaces": {
            str(key): {
                "k_data_ptr": ws["k"].data_ptr(),
                "v_data_ptr": ws["v"].data_ptr(),
            }
            for key, ws in state.workspaces.items()
        },
        "workspace_bytes": workspace_bytes,
        # shared workspace counted once, never per block
        "allocated_bytes": total_bytes + meta_diag["allocated_bytes"] + workspace_bytes,
    }
