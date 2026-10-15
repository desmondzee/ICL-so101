"""Partial-chunk variant of the opt-in packed batch engine.

Subclasses ``BatchedStaticEngine`` (batch_size 4, forecast frames 2 or 4)
so each replan cycle forecasts F latent frames but commits exactly one
latent frame of real observations per active slot per cache update (two
on startup: init frame + first obs).  Packed slot positions therefore
advance under an explicit commit cursor, while ``current_frame_ids``
remain the per-cycle ids used by window pruning and cache masks.

Per-slot cursor at chunk_idx k is ``0 if k == 0 else k + 1`` — each live
slot commits exactly once per cycle and ``slot.chunk_idx`` is preserved
on deactivation, so no per-slot cursor bookkeeping is needed.
"""

from __future__ import annotations

import torch

from zero_wam.batched_static_engine import BatchedStaticEngine


class BatchedPartialChunkEngine(BatchedStaticEngine):
    def __init__(self, server, *, batch_size=4, **kwargs):
        if batch_size != 4:
            raise ValueError(
                "BatchedPartialChunkEngine requires batch_size 4, "
                f"got {batch_size}")
        f = getattr(server.job_config, "frame_chunk_size", None)
        if f not in (2, 4):
            raise ValueError(
                f"forecast frame_chunk_size must be 2 or 4, got {f}")
        super().__init__(server, batch_size=batch_size, **kwargs)
        self.pending = False
        self.cache_events = []

    # ------------------------------------------------------------------
    # packed grid rebased to the committed-frame cursor
    # ------------------------------------------------------------------

    def _grid(self, f_tokens, h_tok, w_tok, t, chunk_idx):
        grid = super()._grid(f_tokens, h_tok, w_tok, t, chunk_idx)
        cursor = 0 if chunk_idx == 0 else chunk_idx + 1
        grid[0] += cursor - self.frame_chunk_size * chunk_idx
        return grid

    # ------------------------------------------------------------------
    # forecast / commit lifecycle
    # ------------------------------------------------------------------

    def reset(self, specs):
        result = super().reset(specs)
        self.pending = False
        self.cache_events = []
        return result

    @torch.no_grad()
    def infer(self, obs_list=None):
        if self.pending:
            raise RuntimeError(
                "partial-chunk: a packed forecast is still pending — call "
                "cache_update before the next inference")
        result = super().infer(obs_list)
        self.pending = True
        return result

    @torch.no_grad()
    def cache_update(self, obs_list, commanded_histories=None):
        if not self.pending:
            raise RuntimeError(
                "partial-chunk: no pending forecast to commit; infer must "
                "run first")
        if len(obs_list) != self.batch_size:
            raise ValueError(
                f"obs_list must have {self.batch_size} entries, "
                f"got {len(obs_list)}")
        if commanded_histories is None or (
            len(commanded_histories) != self.batch_size
        ):
            raise ValueError(
                "partial-chunk cache_update requires one commanded-history "
                f"entry per slot (got "
                f"{None if commanded_histories is None else len(commanded_histories)})")
        events_before = []
        for i in range(self.batch_size):
            slot = self.slots[i]
            if not slot.active:
                # padding/deactivated slots accept None placeholders
                continue
            obs = obs_list[i]
            images = obs["obs"] if isinstance(obs, dict) else None
            if not isinstance(images, list) or len(images) != 4:
                raise ValueError(
                    f"partial-chunk slot {i}: exactly 4 observations "
                    "required per commit, got "
                    f"{0 if not isinstance(images, list) else len(images)}")
            hist = commanded_histories[i]
            expected = 2 if slot.chunk_idx == 0 else 1
            if hist is None or hist.shape[2] != expected:
                raise ValueError(
                    f"partial-chunk slot {i}: commanded history temporal "
                    f"length must be {expected} at chunk_idx "
                    f"{slot.chunk_idx}, got "
                    f"{None if hist is None else hist.shape[2]}")
            events_before.append({
                "slot": i,
                "chunk_idx": int(slot.chunk_idx),
                "cursor_before": 0 if slot.chunk_idx == 0
                else slot.chunk_idx + 1,
                "committed_frames": 2 if slot.chunk_idx == 0 else 1,
                "obs_count": len(images),
                "action_shape": list(hist.shape),
            })
        super().cache_update(obs_list, commanded_histories)
        for ev in events_before:
            ev["cache_counts"] = {
                str(k): int(v) for k, v in
                self.tf.cache_counts().items()
            }
            self.cache_events.append(ev)
        self.pending = False

    # ------------------------------------------------------------------
    # VAE continuation must commit exactly one fresh latent frame
    # ------------------------------------------------------------------

    def _encode_obs_slot(self, slot, obs):
        lat = super()._encode_obs_slot(slot, obs)
        if lat is None or lat.shape[2] != 1:
            raise RuntimeError(
                f"partial-chunk: encoded continuation must be temporal 1, "
                f"got {None if lat is None else lat.shape[2]}")
        return lat
