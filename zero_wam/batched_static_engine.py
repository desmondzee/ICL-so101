"""Two-slot packed-sequence inference engine around one VA_Server.

Entirely separate from the sequential runtime: this module is only used by
``scripts/run_so101_batched_static_smoke.py``.  Per-slot state (prompt/negative
embeds, ICL latent/text, chunk index, initial latent, last predicted
latents/actions, normalization stats, schedulers, RNG generators) lives here;
the transformer is shared and driven through ONE packed ``forward_*_only``
call per timestep with disjoint sequence IDs ``0/1`` (slot 0 cond/uncond) and
``2/3`` (slot 1 cond/uncond).  VAE encoding stays sequential per slot with
``streaming_vae.clear_cache()`` between slots — no simulator stepping here.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch
from einops import rearrange

from zero_wam.batched_static_cache import (
    CACHE_NAME,
    StaticKVCacheConfig,
    install_static_kv_cache,
    static_cache_diagnostics,
)
from zero_wam.packed_batch import (
    BATCH_SIZES,
    COND_BRANCH,
    INVALID_ID,
    UNCOND_BRANCH,
    slot_seq_id,
)

# mirror wan_va/modules/icl_model.py
ICL_CACHE_TYPE = 2
PREDICTION_CACHE_TYPE = 1
OBSERVATION_CACHE_TYPE = 0


@dataclass(frozen=True)
class SlotResetSpec:
    """One rollout's reset inputs; geometry/steps must match across slots."""

    prompt: str
    icl_latent_path: Optional[str] = None
    seed: int = 0
    icl_video_path: str = ""
    active: bool = True


@dataclass
class _SlotState:
    spec: Optional[SlotResetSpec] = None
    generator: Optional[torch.Generator] = None
    scheduler: Any = None
    action_scheduler: Any = None
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    target_prompt_embeds: Optional[torch.Tensor] = None
    vae: Any = None  # per-slot WanVAEStreamingWrapper (own feat_cache)
    chunk_idx: int = 0
    init_latent: Optional[torch.Tensor] = None
    last_predicted_latents: Optional[torch.Tensor] = None
    last_predicted_actions: Optional[torch.Tensor] = None
    active: bool = True


class BatchedStaticEngine:
    """Fixed `batch_size`-slot packed engine over a loaded VA_Server.

    `batch_size` ∈ {2,4}; slot s owns sequence IDs 2s (cond) / 2s+1 (uncond).
    """

    def __init__(
        self,
        server,
        *,
        max_tokens: int,
        batch_size: int = 2,
        video_guidance_scale: float = -1.0,
        icl_guidance_scale: float = 5.0,
        use_icl: bool = True,
        safety_fraction: float = 0.85,
    ):
        if batch_size not in BATCH_SIZES:
            raise ValueError(
                f"batch_size must be one of {BATCH_SIZES}, got {batch_size}"
            )
        self.batch_size = batch_size
        self.server = server
        self.tf = server.transformer
        jc = server.job_config
        if not getattr(server, "use_icl_model", False):
            raise ValueError("BatchedStaticEngine requires an ICL server")
        self.jc = jc
        self.device = server.device
        self.dtype = server.dtype
        self.use_icl = use_icl
        self.video_guidance_scale = float(video_guidance_scale)
        self.icl_guidance_scale = float(icl_guidance_scale)
        self.use_icl_cfg = use_icl and self.icl_guidance_scale > 1.0
        self.target_text_cfg_active = (
            not self.use_icl_cfg and self.video_guidance_scale > 1.0
        )

        # geometry identical to VA_Server._reset_icl
        self.action_per_frame = jc.action_per_frame
        self.height, self.width = jc.height, jc.width
        self.latent_height = self.height // 16
        self.latent_width = self.width // 16 * len(jc.obs_cam_keys)
        self.frame_chunk_size = jc.frame_chunk_size
        self.patch_size = self.tf.patch_size

        self.config = StaticKVCacheConfig(
            max_tokens=max_tokens,
            batch_slots=batch_size,
            heads=self.tf.num_attention_heads,
            head_dim=self.tf.attention_head_dim,
        )
        install_static_kv_cache(
            self.tf, self.config, safety_fraction=safety_fraction
        )
        self.slots = [_SlotState() for _ in range(batch_size)]
        # Per-slot streaming-VAE state: wrappers share the immutable
        # encoder/quant_conv modules but own independent feat_caches, so
        # temporal continuation is correct per slot and no slot's cache is
        # ever cleared by another slot's encode.
        from wan_va.modules.utils import WanVAEStreamingWrapper

        for slot in self.slots:
            slot.vae = WanVAEStreamingWrapper(server.streaming_vae.vae)
        self.calls = {
            "icl_pack": 0,
            "video": 0,
            "action": 0,
            "obs_video": 0,
            "obs_action": 0,
            "prune": 0,
        }

    # ------------------------------------------------------------------
    # per-slot helpers (shared server geometry/action state set at reset)
    # ------------------------------------------------------------------

    def _shared_reset_state(self):
        jc = self.jc
        s = self.server
        s.action_per_frame = jc.action_per_frame
        s.height, s.width = jc.height, jc.width
        s.latent_height = self.latent_height
        s.latent_width = self.latent_width
        s.action_mask = torch.zeros(
            jc.action_dim, dtype=torch.bool, device=self.device
        )
        s.action_mask[jc.used_action_channel_ids] = True
        s.actions_q01 = torch.tensor(
            jc.norm_stat["q01"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        s.actions_q99 = torch.tensor(
            jc.norm_stat["q99"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        s.action_norm_method = jc.action_norm_method
        self.action_mask = s.action_mask

    def _make_schedulers(self):
        from wan_va.utils.scheduler import FlowMatchScheduler

        sched = FlowMatchScheduler(
            shift=self.jc.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        action = FlowMatchScheduler(
            shift=self.jc.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        return sched, action

    def _grid(self, f_tokens, h_tok, w_tok, t, chunk_idx):
        """Replicates _prepare_icl_stream's get_mesh_id call ([4, T])."""
        from wan_va.utils.utils import get_mesh_id

        return get_mesh_id(
            f_tokens,
            h_tok,
            w_tok,
            t,
            1,
            self.frame_chunk_size * chunk_idx,
            action=False,
        ).to(self.device)

    def set_active(self, slot_index: int, active: bool) -> None:
        """Deactivate/reactivate a slot; masked regions use INVALID_ID."""
        self.slots[slot_index].active = bool(active)

    # ------------------------------------------------------------------
    # per-slot VAE encoders (each slot's own feat_cache; sharing the same
    # immutable encoder modules through WanVAEStreamingWrapper)
    # ------------------------------------------------------------------

    def _encode_initial_obs_slot(self, slot, obs):
        """_encode_initial_obs semantics on slot.vae (fresh 1-frame encode)."""
        import torch.nn.functional as F

        image = obs["obs"]
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError(
                    "The initial observation must contain exactly one frame"
                )
            image = image[0]
        slot.vae.clear_cache()  # fresh stream — only this slot's cache
        videos = []
        for key in self.jc.obs_cam_keys:
            current = (
                torch.from_numpy(image[key].copy())
                .float()
                .to(self.device)
            )
            current = current / 255.0 * 2.0 - 1.0
            current = current.permute(2, 0, 1).unsqueeze(0)
            current = F.interpolate(
                current, size=(self.height, self.width),
                mode="bilinear", align_corners=False,
            )
            videos.append(current.unsqueeze(2))
        video = torch.cat(videos, dim=0).to(self.server.vae.dtype)
        encoded = slot.vae.encode_chunk(video)
        mu, _ = torch.chunk(encoded, 2, dim=1)
        mean = torch.tensor(
            self.server.vae.config.latents_mean, device=mu.device
        )
        std = torch.tensor(
            self.server.vae.config.latents_std, device=mu.device
        )
        mu = self.server.normalize_latents(mu, mean, 1.0 / std)
        return torch.cat(mu.split(1, dim=0), dim=-1).to(self.device)

    def _encode_obs_slot(self, slot, obs):
        """_encode_obs semantics on slot.vae (continuation; no clear)."""
        import torch.nn.functional as F
        import numpy as np

        images = obs["obs"]
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        # Encode both cams together (feat_cache tails are batch=2) in <=4-frame
        # temporal pieces: streaming continuation stays identical while each
        # encode_chunk's transient activation stays small enough for batch-4
        # VRAM headroom.
        vae_device = next(slot.vae.vae.parameters()).device
        piece = 4
        videos = []
        for k in self.jc.obs_cam_keys:
            history = torch.from_numpy(
                np.stack([each[k] for each in images])
            ).float().permute(3, 0, 1, 2)
            history = F.interpolate(
                history, size=(self.height, self.width),
                mode="bilinear", align_corners=False,
            ).unsqueeze(0)
            videos.append(history)
        videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
        mus = []
        for t0 in range(0, videos.shape[2], piece):
            chunk = videos[:, :, t0 : t0 + piece].to(vae_device).to(self.dtype)
            enc_out = slot.vae.encode_chunk(chunk)
            mu, _ = torch.chunk(enc_out, 2, dim=1)
            mus.append(mu)
        mu_all = torch.cat(mus, dim=2)
        mean = torch.tensor(
            self.server.vae.config.latents_mean, device=mu_all.device
        )
        std = torch.tensor(
            self.server.vae.config.latents_std, device=mu_all.device
        )
        mu_norm = self.server.normalize_latents(mu_all, mean, 1.0 / std)
        return torch.cat(mu_norm.split(1, dim=0), dim=-1).to(self.device)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset(self, specs: Sequence[SlotResetSpec]):
        """Multi-slot reset: sequential text/ICL encode, ONE packed ICL write."""
        if len(specs) != self.batch_size:
            raise ValueError(
                f"reset requires {self.batch_size} specs, got {len(specs)}"
            )
        if any(not s.prompt for s in specs if s.active):
            raise ValueError("every active slot requires a prompt")
        self._shared_reset_state()
        self.tf.clear_cache(CACHE_NAME)
        self.server.streaming_vae.clear_cache()
        if self.server.streaming_vae_half is not None:
            self.server.streaming_vae_half.clear_cache()

        icl_parts = []
        for i, spec in enumerate(specs):
            slot = self.slots[i]
            slot.spec = spec
            slot.chunk_idx = 0
            slot.init_latent = None
            slot.last_predicted_latents = None
            slot.last_predicted_actions = None
            slot.active = spec.active
            slot.generator = torch.Generator(device=self.device)
            slot.generator.manual_seed(int(spec.seed))
            slot.scheduler, slot.action_scheduler = self._make_schedulers()

            if not slot.active:
                # Padding slot: masked (-1) placeholders only; nothing is
                # encoded and nothing of it is attendable.
                slot.prompt_embeds = None
                slot.negative_prompt_embeds = None
                slot.target_prompt_embeds = None
                icl_parts.append(None)  # filled after real latents are known
                continue

            prompt_emb, neg_emb = self.server.encode_prompt(
                prompt=spec.prompt,
                negative_prompt="",
                do_classifier_free_guidance=True,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )
            if self.server.empty_text_emb is not None:
                neg_emb = self.server.empty_text_emb.clone()
            slot.prompt_embeds = prompt_emb
            slot.negative_prompt_embeds = neg_emb
            slot.target_prompt_embeds = (
                neg_emb if self.video_guidance_scale < 0 else prompt_emb
            )

            latent, icl_text = self.server._load_or_encode_icl(
                spec.icl_video_path, spec.icl_latent_path
            )
            expected = len(self.server.vae.config.latents_mean)
            if latent.shape[1] != expected:
                raise ValueError(
                    f"slot {i} ICL latent has {latent.shape[1]} channels; "
                    f"expected {expected}"
                )
            latent = latent.to(device=self.device, dtype=self.dtype)
            icl_text = (
                slot.prompt_embeds if icl_text is None else icl_text.to(
                    self.device
                )
            )
            self.server.streaming_vae.clear_cache()
            icl_parts.append((latent, icl_text))
        if self.server.streaming_vae_half is not None:
            self.server.streaming_vae_half.clear_cache()

        ref = next(p for p in icl_parts if p is not None)
        text_dim = (
            self.server.empty_text_emb.shape[-1]
            if self.server.empty_text_emb is not None
            else ref[1].shape[-1]
        )
        for i, part in enumerate(icl_parts):
            if part is None:  # padding slot — masked zeros, same geometry
                icl_parts[i] = (
                    torch.zeros(
                        1, ref[0].shape[1], 1,
                        ref[0].shape[3], ref[0].shape[4],
                        device=self.device, dtype=self.dtype,
                    ),
                    torch.zeros(
                        1, 1, text_dim, device=self.device,
                        dtype=self.dtype,
                    ),
                )

        # One packed ICL-prefix cache write: slot s under seq id 2s
        # (cond branch; uncond branches see no ICL — same as sequential
        # _pack_icl_cfg where the uncond stream keeps seq id 1 without ICL).
        latents = torch.cat([p[0] for p in icl_parts], dim=2)
        grids = []
        for latent, _ in icl_parts:
            grid_id = self._grid(
                latent.shape[-3] // self.patch_size[0],
                latent.shape[-2] // self.patch_size[1],
                latent.shape[-1] // self.patch_size[2],
                0,
                0,
            )
            grid_id[1] += int(self.jc.icl_rope_h)
            grids.append(grid_id)
        grid = torch.cat(grids, dim=1)
        t_counts = [g.shape[1] for g in grids]
        text_emb = torch.cat(
            [p[1].to(self.dtype) for p in icl_parts], dim=1
        )
        total = sum(t_counts)
        seq_ids = torch.cat(
            [
                torch.full(
                    (n,),
                    slot_seq_id(i, COND_BRANCH)
                    if self.slots[i].active else INVALID_ID,
                    dtype=torch.int, device=self.device,
                )
                for i, n in enumerate(t_counts)
            ]
        )
        input_dict = {
            "latent_res_lst": {
                "noisy_latents": latents,
                "timesteps": torch.zeros(
                    latents.shape[2], device=self.device, dtype=torch.float32
                ),
                "cache_type_ids": torch.full(
                    [total], ICL_CACHE_TYPE, device=self.device, dtype=torch.int
                ),
            },
            "latent_grid_id": grid,
            "current_seq_ids": seq_ids,
            "current_frame_ids": torch.zeros(
                total, device=self.device, dtype=torch.int
            ),
            "encoder_seq_ids": torch.cat(
                [
                    torch.full(
                        (icl_parts[i][1].shape[1],),
                        slot_seq_id(i, COND_BRANCH)
                        if self.slots[i].active else INVALID_ID,
                        dtype=torch.int, device=self.device,
                    )
                    for i in range(self.batch_size)
                ]
            ),
            "text_emb": text_emb,
        }
        with torch.no_grad():
            self.tf(input_dict, update_cache=1, cache_name=CACHE_NAME,
                    mode="forward_latent_only")
        self.calls["icl_pack"] += 1
        return dict()

    # ------------------------------------------------------------------
    # packed stream builders
    # ------------------------------------------------------------------


    def _slot_emb(self, slot, key):
        emb = getattr(slot, key)
        if emb is None:  # inactive padding slot — masked, shape only
            sibling = next(
                getattr(s, key) for s in self.slots
                if getattr(s, key) is not None
            )
            emb = torch.zeros(
                1, 1, sibling.shape[-1],
                device=self.device, dtype=self.dtype,
            )
        return emb

    def _pack_video_stream(self, latents, timestep, cache_type):
        """Pack per-slot [cond, uncond] video chunks (seq 2s / 2s+1)."""
        cfg = self.use_icl_cfg or self.target_text_cfg_active
        latent_parts, grid_parts, ts_parts = [], [], []
        seq_ids, frame_ids, ctype = [], [], []
        text_parts, enc_ids = [], []
        tok_per_part = []
        for i, lat in enumerate(latents):
            slot = self.slots[i]
            F = lat.shape[2]
            gh = lat.shape[-2] // self.patch_size[1]
            gw = lat.shape[-1] // self.patch_size[2]
            grid = self._grid(lat.shape[-3] // self.patch_size[0], gh, gw, 0,
                              slot.chunk_idx)
            ts = torch.full(
                [F], float(timestep), device=self.device, dtype=torch.float32
            )
            branches = (COND_BRANCH, UNCOND_BRANCH) if cfg else (COND_BRANCH,)
            for branch in branches:
                latent_parts.append(lat)
                grid_parts.append(grid)
                ts_parts.append(ts)
                n = grid.shape[1]
                tok_per_part.append(n)
                seq_ids.append(
                    torch.full(
                        [n],
                        slot_seq_id(i, branch) if slot.active else INVALID_ID,
                        dtype=torch.int, device=self.device,
                    )
                )
                frame_ids.append(
                    torch.full(
                        [n], 2 * slot.chunk_idx, dtype=torch.int,
                        device=self.device,
                    )
                )
                ctype.append(
                    torch.full(
                        [n], cache_type, dtype=torch.int, device=self.device
                    )
                )
                # _pack_icl_cfg duplicates text per branch for every cache
                # type (the uncond branch's text depends on the CFG mode).
                emb = (
                    self._slot_emb(slot, "target_prompt_embeds")
                    if branch == COND_BRANCH
                    else (
                        self._slot_emb(slot, "target_prompt_embeds")
                        if self.use_icl_cfg
                        else self._slot_emb(slot, "negative_prompt_embeds")
                    )
                )
                text_parts.append(emb.to(self.dtype))
                enc_ids.append(
                    torch.full(
                        [emb.shape[1]],
                        slot_seq_id(i, branch)
                        if slot.active else INVALID_ID,
                        dtype=torch.int, device=self.device,
                    )
                )
        input_dict = {
            "latent_res_lst": {
                "noisy_latents": torch.cat(latent_parts, dim=2),
                "timesteps": torch.cat(ts_parts),
                "cache_type_ids": torch.cat(ctype),
            },
            "latent_grid_id": torch.cat(grid_parts, dim=1),
            "current_seq_ids": torch.cat(seq_ids),
            "current_frame_ids": torch.cat(frame_ids),
            "text_emb": torch.cat(text_parts, dim=1),
            "encoder_seq_ids": torch.cat(enc_ids),
        }
        return input_dict, tok_per_part

    def _pack_action_stream(self, actions, timestep, cache_type):
        """Pack all slots' action chunks (no CFG on the action stream)."""
        latent_parts, grid_parts, ts_parts = [], [], []
        seq_ids, frame_ids, ctype = [], [], []
        tok_per_part = []
        for i, act in enumerate(actions):
            slot = self.slots[i]
            grid = self._grid(
                act.shape[-3],
                act.shape[-2],
                act.shape[-1],
                1,
                slot.chunk_idx,
            )
            n = grid.shape[1]
            tok_per_part.append(n)
            latent_parts.append(act)
            grid_parts.append(grid)
            ts_parts.append(
                torch.full(
                    [act.shape[2]], float(timestep),
                    device=self.device, dtype=torch.float32,
                )
            )
            seq_ids.append(
                torch.full(
                    [n],
                    slot_seq_id(i, COND_BRANCH) if slot.active else INVALID_ID,
                    dtype=torch.int, device=self.device,
                )
            )
            frame_ids.append(
                torch.full(
                    [n], 2 * slot.chunk_idx + 1, dtype=torch.int,
                    device=self.device,
                )
            )
            ctype.append(
                torch.full([n], cache_type, dtype=torch.int, device=self.device)
            )
        text_parts = [
            self._slot_emb(self.slots[i], "target_prompt_embeds").to(
                self.dtype
            )
            for i in range(self.batch_size)
        ]
        input_dict = {
            "action_res_lst": {
                "noisy_latents": torch.cat(latent_parts, dim=2),
                "timesteps": torch.cat(ts_parts),
                "cache_type_ids": torch.cat(ctype),
            },
            "action_grid_id": torch.cat(grid_parts, dim=1),
            "current_seq_ids": torch.cat(seq_ids),
            "current_frame_ids": torch.cat(frame_ids),
            "text_emb": torch.cat(text_parts, dim=1),
            "encoder_seq_ids": torch.cat(
                [
                    torch.full(
                        [
                            self._slot_emb(
                                self.slots[i], "target_prompt_embeds"
                            ).shape[1]
                        ],
                        slot_seq_id(i, COND_BRANCH)
                        if self.slots[i].active else INVALID_ID,
                        dtype=torch.int, device=self.device,
                    )
                    for i in range(self.batch_size)
                ]
            ),
        }
        return input_dict, tok_per_part

    # ------------------------------------------------------------------
    # inference (one packed transformer call per timestep)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(self, obs_list=None):
        """Packed inference for all slots; returns a list of action arrays."""
        jc = self.jc
        if obs_list is None:
            obs_list = [None] * self.batch_size
        if len(obs_list) != self.batch_size:
            raise ValueError(
                f"obs_list must have {self.batch_size} entries, got {len(obs_list)}"
            )
        for i, obs in enumerate(obs_list):
            slot = self.slots[i]
            if slot.chunk_idx == 0 and slot.active:
                slot.init_latent = self._encode_initial_obs_slot(slot, obs)

        latents = []
        for slot in self.slots:
            if slot.active:
                latents.append(torch.randn(
                    1, 48, self.frame_chunk_size,
                    self.latent_height, self.latent_width,
                    generator=slot.generator, device=self.device,
                    dtype=self.dtype,
                ))
            else:
                latents.append(torch.zeros(
                    1, 48, self.frame_chunk_size,
                    self.latent_height, self.latent_width,
                    device=self.device, dtype=self.dtype,
                ))
        actions = []
        for slot in self.slots:
            if slot.active:
                actions.append(torch.randn(
                    1, jc.action_dim, self.frame_chunk_size,
                    self.action_per_frame, 1,
                    generator=slot.generator, device=self.device,
                    dtype=self.dtype,
                ))
            else:
                actions.append(torch.zeros(
                    1, jc.action_dim, self.frame_chunk_size,
                    self.action_per_frame, 1,
                    device=self.device, dtype=self.dtype,
                ))
        # Consume inactive-channel noise per slot exactly like the sequential
        # path (seeded determinism across chunks).
        for i, slot in enumerate(self.slots):
            if not slot.active:
                continue
            _ = torch.randn(
                actions[i].shape, generator=slot.generator,
                device=self.device, dtype=self.dtype,
            )[:, ~self.action_mask]

        for slot in self.slots:
            slot.scheduler.set_timesteps(jc.num_inference_steps)
            slot.action_scheduler.set_timesteps(jc.action_num_inference_steps)

        video_timesteps = torch.cat(
            [
                self.slots[0].scheduler.timesteps,
                torch.zeros(
                    1,
                    device=self.slots[0].scheduler.timesteps.device,
                    dtype=self.slots[0].scheduler.timesteps.dtype,
                ),
            ]
        )
        for step, timestep in enumerate(video_timesteps):
            last_step = step == len(video_timesteps) - 1
            latent_inputs = []
            for i, slot in enumerate(self.slots):
                li = latents[i].clone()
                if slot.chunk_idx == 0 and slot.init_latent is not None:
                    li[:, :, :1] = slot.init_latent[:, :, :1]
                latent_inputs.append(li)
            input_dict, _ = self._pack_video_stream(
                latent_inputs, timestep, PREDICTION_CACHE_TYPE,
            )
            if any(s.chunk_idx == 0 for s in self.slots):
                input_dict["latent_res_lst"]["timesteps"][
                    :: self.frame_chunk_size
                ] = 0
            with torch.no_grad():
                noise = self.tf(
                    input_dict,
                    update_cache=int(last_step),
                    cache_name=CACHE_NAME,
                    mode="forward_latent_only",
                )
            self.calls["video"] += 1
            if last_step:
                continue
            # split noise: per part token counts are equal per branch
            cfg = self.use_icl_cfg or self.target_text_cfg_active
            branch_tokens = noise.shape[1] // (
                2 * self.batch_size if cfg else self.batch_size
            )
            for i, slot in enumerate(self.slots):
                if not slot.active:
                    continue
                if cfg:
                    base = i * 2 * branch_tokens
                    n_cond = noise[:, base : base + branch_tokens]
                    n_uncond = noise[
                        :, base + branch_tokens : base + 2 * branch_tokens
                    ]
                    scale = (
                        self.icl_guidance_scale
                        if self.use_icl_cfg
                        else self.video_guidance_scale
                    )
                    n = n_uncond + scale * (n_cond - n_uncond)
                else:
                    n = noise[:, i * branch_tokens : (i + 1) * branch_tokens]
                n = self._seq_to_patch(n)
                latents[i] = slot.scheduler.step(
                    n, timestep, latents[i], return_dict=False
                )
                if slot.chunk_idx == 0 and slot.init_latent is not None:
                    latents[i][:, :, :1] = slot.init_latent[:, :, :1]

        for timestep in self.slots[0].action_scheduler.timesteps:
            action_inputs = []
            for act in actions:
                ai = act.clone()
                ai[:, ~self.action_mask] = 0
                action_inputs.append(ai)
            input_dict, _ = self._pack_action_stream(
                action_inputs, timestep, PREDICTION_CACHE_TYPE,
            )
            with torch.no_grad():
                action_noise = self.tf(
                    input_dict,
                    update_cache=0,
                    cache_name=CACHE_NAME,
                    mode="forward_action_only",
                )
            self.calls["action"] += 1
            tok = action_noise.shape[1] // self.batch_size
            for i, slot in enumerate(self.slots):
                if not slot.active:
                    continue
                n = action_noise[:, i * tok : (i + 1) * tok]
                n = rearrange(
                    n, "1 (f n) c -> 1 c f n 1", f=self.frame_chunk_size
                )
                actions[i] = slot.action_scheduler.step(
                    n, timestep, actions[i], return_dict=False
                )

        outputs = []
        for i, slot in enumerate(self.slots):
            actions[i][:, ~self.action_mask] = 0
            cached = actions[i].clone()
            if slot.chunk_idx == 0:
                cached[:, :, 0] = 0
            slot.last_predicted_latents = latents[i]
            slot.last_predicted_actions = cached
            outputs.append(self.server.postprocess_action(actions[i]))
        return outputs

    def _seq_to_patch(self, seq):
        from wan_va.utils.utils import data_seq_to_patch

        return data_seq_to_patch(
            self.patch_size,
            seq,
            self.frame_chunk_size,
            self.latent_height,
            self.latent_width,
        )

    # ------------------------------------------------------------------
    # cache update (one packed video write + one packed action write)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cache_update(self, obs_list, commanded_histories=None):
        """Packed obs/action cache write for all slots in one call per stream.

        `obs_list[i]` is slot i's keyframe payload (``{"obs": [frames]}``) or
        None for inactive/finished slots, which get masked zero placeholders
        without touching their VAE state.  `commanded_histories[i]` is the
        slot's normalized commanded-action history `[1,C,F+1,H,1]` (the same
        shape `preprocess_action` produces); None → `last_predicted_actions`.
        """
        self.tf.clear_prediction_cache(-1, CACHE_NAME)
        self.calls["prune"] += 1
        if len(obs_list) != self.batch_size:
            raise ValueError(
                f"obs_list must have {self.batch_size} entries, got {len(obs_list)}"
            )
        if commanded_histories is not None and (
            len(commanded_histories) != self.batch_size
        ):
            raise ValueError("commanded_histories length mismatch")
        latent_inputs = []
        for i, obs in enumerate(obs_list):
            slot = self.slots[i]
            if not slot.active or obs is None:
                # masked placeholder: never attendable (seq -1), minimal size
                latent_inputs.append(torch.zeros(
                    1, 48, 1, self.latent_height, self.latent_width,
                    device=self.device, dtype=self.dtype,
                ))
                continue
            lat = self._encode_obs_slot(slot, obs)  # continuation on slot.vae
            if slot.chunk_idx == 0:
                lat = torch.cat([slot.init_latent, lat], dim=2)
            latent_inputs.append(lat)
        action_inputs = []
        for i, slot in enumerate(self.slots):
            hist = (
                commanded_histories[i]
                if commanded_histories is not None
                else None
            )
            if hist is None:
                hist = slot.last_predicted_actions
            if hist is None:
                if slot.active:
                    raise RuntimeError(
                        f"slot {i}: action history requested before first inference"
                    )
                hist = torch.zeros(
                    1, self.jc.action_dim, 1, self.action_per_frame, 1,
                    device=self.device, dtype=self.dtype,
                )
            action_inputs.append(hist)
        video_dict, _ = self._pack_video_stream(
            latent_inputs, 0, OBSERVATION_CACHE_TYPE
        )
        with torch.no_grad():
            self.tf(video_dict, update_cache=1, cache_name=CACHE_NAME,
                    mode="forward_latent_only")
        self.calls["obs_video"] += 1
        action_dict, _ = self._pack_action_stream(
            [a.to(latent_inputs[i]) for i, a in enumerate(action_inputs)],
            0,
            OBSERVATION_CACHE_TYPE,
        )
        with torch.no_grad():
            self.tf(action_dict, update_cache=1, cache_name=CACHE_NAME,
                    mode="forward_action_only", clean_window_cache=True)
        self.calls["obs_action"] += 1
        for slot in self.slots:
            if slot.active:
                slot.chunk_idx += 1
        return dict()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "adapter": static_cache_diagnostics(self.tf),
            "slots": [
                {
                    "chunk_idx": s.chunk_idx,
                    "seed": s.spec.seed if s.spec else None,
                    "active": s.active,
                }
                for s in self.slots
            ],
        }


def action_sha256(actions) -> str:
    arr = np.ascontiguousarray(np.asarray(actions))
    return hashlib.sha256(arr.tobytes()).hexdigest()
