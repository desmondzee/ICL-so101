"""Opt-in partial-chunk inference adapter for ``VA_Server``.

Experimental: enables forecast horizon ``frame_chunk_size`` in {2, 4}
while the committed observation cache advances one latent frame per
cycle (two on startup).  All changes are instance-bound
(``types.MethodType``) — no class or global monkeypatch — and
``model.job_config`` is shallow-copied so the shared config object is
never mutated.

RoPE vs cycle-frame-id distinction
----------------------------------
``latent_grid_id``/``action_grid_id`` positions are corrected from the
denoising formula ``frame_chunk_size * chunk_idx`` to the explicit
``partial_frame_cursor`` — the count of actually-committed latent
frames.  ``current_frame_ids`` deliberately remains the original
``2*chunk_idx + offset`` cycle id: window pruning and cache masks group
by this id, and switching it to absolute frame positions would change
attention grouping semantics.
"""

from __future__ import annotations

import copy
import types

import torch


def _serializable_cache_counts(transformer):
    try:
        counts = transformer.cache_counts()
    except Exception:  # noqa: BLE001
        return None
    if isinstance(counts, dict):
        return {str(k): int(v) for k, v in counts.items()}
    return counts


def install_partial_chunk(model, forecast_frames):
    """Bind partial-chunk methods onto ``model``; returns ``model``.

    After ``install_partial_chunk``, one inference cycle is: one
    ``_infer_icl`` forecast (``forecast_frames`` ahead) followed by one
    ``_compute_icl_kv_cache`` commit of four fresh observations / one
    latent frame of actual executed history.  At most one pending
    forecast is allowed between commits.
    """
    if forecast_frames not in (2, 4):
        raise ValueError(
            f"forecast_frames must be 2 or 4, got {forecast_frames}")
    if getattr(model, "_partial_chunk_installed", False):
        raise RuntimeError("partial-chunk adapter already installed")
    # Upstream constant: observation cache type is 0 (2 is persistent ICL).
    from wan_va.modules.icl_model import OBSERVATION_CACHE_TYPE

    # Instance-scoped config: the shared job_config object is never edited.
    model.job_config = copy.copy(model.job_config)
    model.job_config.frame_chunk_size = forecast_frames

    orig_reset_icl = model._reset_icl
    orig_prepare = model._prepare_icl_stream
    orig_infer_icl = model._infer_icl

    model.partial_frame_cursor = 0
    model.partial_cache_events = []
    model._partial_pending_forecast = False
    model._partial_chunk_installed = True

    def _reset_icl_partial(self, *args, **kwargs):
        result = orig_reset_icl(*args, **kwargs)
        self.partial_frame_cursor = 0
        self.partial_cache_events = []
        self._partial_pending_forecast = False
        return result

    def _infer_icl_partial(self, obs):
        if self._partial_pending_forecast:
            raise RuntimeError(
                "partial-chunk: a forecast is still pending — call "
                "_compute_icl_kv_cache before the next inference")
        action = orig_infer_icl(obs)
        self._partial_pending_forecast = True
        return action

    def _prepare_icl_stream_partial(self, data, timestep, mode, cache_type):
        input_dict = orig_prepare(data, timestep, mode, cache_type)
        grid_key = "latent_grid_id" if mode == "video" else "action_grid_id"
        input_dict[grid_key][0] += (
            self.partial_frame_cursor
            - self.job_config.frame_chunk_size * self.chunk_idx
        )
        return input_dict

    @torch.no_grad()
    def _compute_icl_kv_cache_partial(self, obs):
        """Commit four fresh observations + the executed action block."""
        images = obs["obs"]
        if not isinstance(images, list) or len(images) != 4:
            raise ValueError(
                f"partial-chunk cache requires exactly 4 observations, "
                f"got {0 if not isinstance(images, list) else len(images)}")
        expected_len = 2 if self.chunk_idx == 0 else 1
        last = self.last_predicted_actions
        if last is None or last.shape[2] != expected_len:
            raise RuntimeError(
                f"partial-chunk: last_predicted_actions temporal length "
                f"must be {expected_len} at chunk_idx {self.chunk_idx}, "
                f"got {None if last is None else last.shape[2]}")
        if not self._partial_pending_forecast:
            raise RuntimeError(
                "partial-chunk: no pending forecast to commit; "
                "_infer_icl must run first")
        self.transformer.clear_prediction_cache(-1, self.cache_name)
        latent_input = self._encode_obs(obs)
        if latent_input is None or latent_input.shape[2] != 1:
            raise RuntimeError(
                "partial-chunk: encoded observation must have temporal "
                f"length 1, got "
                f"{None if latent_input is None else latent_input.shape[2]}")
        if self.chunk_idx == 0:
            latent_input = torch.cat([self.init_latent, latent_input], dim=2)
        action_input = self.last_predicted_actions.to(latent_input)
        if action_input.shape[2] != latent_input.shape[2]:
            raise RuntimeError(
                f"partial-chunk: action/video temporal mismatch "
                f"{action_input.shape[2]} != {latent_input.shape[2]}")

        cursor_before = self.partial_frame_cursor
        video_input = self._prepare_icl_stream(
            latent_input, 0, "video", OBSERVATION_CACHE_TYPE)
        self.transformer(
            video_input,
            update_cache=1,
            cache_name=self.cache_name,
            mode="forward_latent_only",
        )
        action_dict = self._prepare_icl_stream(
            action_input, 0, "action", OBSERVATION_CACHE_TYPE)
        self.transformer(
            action_dict,
            update_cache=1,
            cache_name=self.cache_name,
            mode="forward_action_only",
            clean_window_cache=True,
        )
        committed = int(latent_input.shape[2])
        self.partial_cache_events.append({
            "chunk_idx": int(self.chunk_idx),
            "cursor_before": int(cursor_before),
            "committed_frames": committed,
            "obs_count": len(images),
            "action_shape": list(self.last_predicted_actions.shape),
            "cache_counts": _serializable_cache_counts(self.transformer),
        })
        self.partial_frame_cursor += committed
        self.chunk_idx += 1
        self._partial_pending_forecast = False

    model._reset_icl = types.MethodType(_reset_icl_partial, model)
    model._infer_icl = types.MethodType(_infer_icl_partial, model)
    model._prepare_icl_stream = types.MethodType(
        _prepare_icl_stream_partial, model)
    model._compute_icl_kv_cache = types.MethodType(
        _compute_icl_kv_cache_partial, model)
    return model
