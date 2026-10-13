"""Pure packed-sequence batching helpers for the two-slot prototype.

This is packed-sequence batching: the physical batch dimension stays 1 and the
two slots are concatenated along the token dimension, isolated by disjoint
sequence IDs exactly like the model's existing seq-id masks.  No checkpoint or
model changes are required.

Sequence-ID convention mirrors ``VA_Server._pack_icl_cfg``: the conditional
stream of slot ``s`` uses ``s*2 + 0``, the unconditional stream ``s*2 + 1``.
Text-encoder tokens use the same ``slot_seq_id`` values, so the existing
cross-mask (``query_seq_ids == encoder_seq_ids``) cannot cross slot or branch
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

BATCH_SIZES = (2, 4)
BATCH_SLOTS = 2  # default slot count
COND_BRANCH = 0
UNCOND_BRANCH = 1
INVALID_ID = -1


@dataclass(frozen=True)
class PackedSlot:
    """One independent rollout stream inside a packed batch."""

    slot_id: int
    active: bool
    chunk_idx: int
    seed: int


def slot_seq_id(slot_id: int, cfg_branch: int) -> int:
    """Disjoint sequence ID per (slot, cfg_branch): ``slot*2 + cfg_branch``."""
    if slot_id < 0:
        raise ValueError(f"slot_id must be >= 0, got {slot_id}")
    if cfg_branch not in (COND_BRANCH, UNCOND_BRANCH):
        raise ValueError(f"cfg_branch must be 0 or 1, got {cfg_branch}")
    return slot_id * 2 + cfg_branch


def slot_generator(slot: PackedSlot, device) -> torch.Generator:
    """Deterministic per-slot RNG, independent of batch order."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(slot.seed))
    return generator


def validate_slots(
    slots: Sequence[PackedSlot], batch_size: int = BATCH_SLOTS
) -> tuple[PackedSlot, ...]:
    """`batch_size` ∈ {2,4} fixed slots with distinct ids 0..batch_size-1."""
    if batch_size not in BATCH_SIZES:
        raise ValueError(f"batch_size must be one of {BATCH_SIZES}, got {batch_size}")
    if len(slots) != batch_size:
        raise ValueError(f"expected {batch_size} slots, got {len(slots)}")
    ids = sorted(slot.slot_id for slot in slots)
    if ids != list(range(batch_size)):
        raise ValueError(
            f"slot ids must be exactly {list(range(batch_size))}, got {ids}"
        )
    return tuple(slots)


def pack_tokens(*tensors: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Concatenate same-shaped slot chunks along the token dimension.

    `dim` is the temporal token-producing axis (1 for [1,T,C] tokens, 2 for
    [1,C,F,H,W] latents).  Physical batch dim 0 stays 1.
    """
    if len(tensors) < 2:
        raise ValueError("pack_tokens requires at least two slot tensors")
    a = tensors[0]
    for t in tensors:
        if t.ndim != a.ndim or t.shape[0] != 1:
            raise ValueError(f"expected [1, ...] tensors, got {t.shape}")
        if dim <= 0 or dim >= t.ndim:
            raise ValueError(f"token dim {dim} out of range for shape {t.shape}")
        for axis, (sa, st) in enumerate(zip(a.shape, t.shape)):
            if axis != dim and sa != st:
                raise ValueError(
                    f"mismatched slot geometry at dim {axis}: "
                    f"{a.shape} vs {t.shape}"
                )
        if a.dtype != t.dtype or a.device != t.device:
            raise ValueError("dtype/device mismatch between slots")
    return torch.cat(list(tensors), dim=dim)


def _invalid_vector(token_count: int, device) -> torch.Tensor:
    return torch.full((token_count,), INVALID_ID, dtype=torch.int, device=device)


def slot_metadata(
    slot: PackedSlot,
    token_count: int,
    frame_ids: torch.Tensor,
    type_id: int,
    cache_type_id: int,
    device,
    cfg_branch: int = COND_BRANCH,
) -> dict[str, torch.Tensor]:
    """Per-slot, per-CFG-branch metadata vectors for `token_count` tokens.

    `frame_ids` is the slot's own frame-id vector (length `token_count`).
    Seq ids come from ``slot_seq_id(slot_id, cfg_branch)``.  Inactive slots get
    INVALID_ID on every field: as keys they fail ``key_type_ids != -1``, and as
    queries their ``seq_ids == -1`` can never match a valid key seq id — so an
    inactive slot contributes no attention mass in either direction.
    """
    if frame_ids.numel() != token_count:
        raise ValueError(
            f"frame_ids has {frame_ids.numel()} entries for {token_count} tokens"
        )
    if not slot.active:
        return {
            name: _invalid_vector(token_count, device)
            for name in ("seq_ids", "frame_ids", "type_ids", "cache_type_ids")
        }
    return {
        "seq_ids": torch.full(
            (token_count,),
            slot_seq_id(slot.slot_id, cfg_branch),
            dtype=torch.int,
            device=device,
        ),
        "frame_ids": frame_ids.to(device=device, dtype=torch.int),
        "type_ids": torch.full(
            (token_count,), type_id, dtype=torch.int, device=device
        ),
        "cache_type_ids": torch.full(
            (token_count,), cache_type_id, dtype=torch.int, device=device
        ),
    }


def pack_cfg_metadata(
    cond_meta: dict[str, torch.Tensor],
    uncond_meta: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Concatenate a slot's cond+uncond branch metadata (CFG duplication)."""
    return pack_metadata(cond_meta, uncond_meta)


def pack_metadata(*slot_metas: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concatenate per-slot/branch metadata dicts; keys must match exactly."""
    keys = set(slot_metas[0])
    for meta in slot_metas[1:]:
        if set(meta) != keys:
            raise ValueError("metadata field mismatch between slots")
    return {
        key: torch.cat([meta[key] for meta in slot_metas]) for key in keys
    }


def pack_text_embeddings(
    entries: Sequence[tuple[torch.Tensor, PackedSlot, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate [1, S, C] text embeddings along sequence length.

    Each entry is ``(embedding, slot, cfg_branch)``; encoder sequence IDs are
    ``slot_seq_id(slot.slot_id, cfg_branch)`` so cond/uncond branches of both
    slots stay distinct.  Inactive slots emit INVALID_ID encoder ids — the
    cross-mask cannot route any query to them.
    """
    if not entries:
        raise ValueError("no text embeddings to pack")
    embeds = [e for e, _, _ in entries]
    shape = embeds[0].shape
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(f"expected [1, S, C] embeddings, got {shape}")
    for emb in embeds[1:]:
        if emb.shape[2] != shape[2] or emb.shape[0] != 1:
            raise ValueError("text embedding geometry mismatch")
    device = embeds[0].device
    packed = torch.cat(embeds, dim=1)
    encoder_seq_ids = torch.cat(
        [
            torch.full(
                (emb.shape[1],),
                slot_seq_id(slot.slot_id, branch) if slot.active else INVALID_ID,
                dtype=torch.int,
                device=device,
            )
            for emb, slot, branch in entries
        ]
    )
    return packed, encoder_seq_ids


def split_packed_output(
    packed: torch.Tensor, lengths: Sequence[int], dim: int = 1
) -> list[torch.Tensor]:
    """Slice a packed output back into per-slot views (no copies)."""
    if sum(lengths) != packed.shape[dim]:
        raise ValueError(
            f"split lengths {list(lengths)} do not cover dim {dim} "
            f"size {packed.shape[dim]}"
        )
    out, offset = [], 0
    for length in lengths:
        out.append(packed.narrow(dim, offset, length))
        offset += length
    return out


def check_compatible(
    grid_shapes: Sequence[Sequence[int]],
    timesteps_list: Sequence[torch.Tensor],
    video_steps: Sequence[int],
    action_steps: Sequence[int],
) -> None:
    """Reject mismatched geometry, timestep schedule, or step counts
    across all packed slots (sequences of equal length)."""
    n = len(grid_shapes)
    if not (len(timesteps_list) == len(video_steps) == len(action_steps) == n):
        raise ValueError("compatibility inputs must have equal length")
    for i in range(1, n):
        if tuple(grid_shapes[i]) != tuple(grid_shapes[0]):
            raise ValueError(
                f"grid geometry mismatch: {grid_shapes[0]} vs {grid_shapes[i]}"
            )
        if not torch.equal(timesteps_list[i], timesteps_list[0]):
            raise ValueError("timestep schedules differ between slots")
        if video_steps[i] != video_steps[0] or action_steps[i] != action_steps[0]:
            raise ValueError(
                "denoising-step counts differ: "
                f"({video_steps[0]},{action_steps[0]}) vs "
                f"({video_steps[i]},{action_steps[i]})"
            )
