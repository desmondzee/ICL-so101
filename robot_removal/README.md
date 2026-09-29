# Robot removal for Zero-WAM references

Removes the SO-101 robot arm from human-generation reference images and emits
exact 480×320 outputs, preserving original pixels outside the expanded robot
mask.

## Pipeline

Two ComfyUI phases per input (`first.jpg` + `pair.json` per accepted episode),
run as two passes over the manifest so each model family stays resident:

**Pass A — segment + reference**

1. `SAM3_Detect` ×2 (ComfyUI native): arm prompts `robot arm/robotic arm/robot`
   at threshold 0.3 and extras `cable/robot base/shadow` at 0.5, with the
   `pair.json` robot `box_2d` as a bounding-box hint. Individual masks are saved
   and selected in Python (`masks.py`, a port of `humangen/edit.py`'s
   `robot_mask_sam3`: best arm by score-rank + box overlap, union overlapping
   arms, union extras that touch the arm and are ≤2.5× its size, keep the
   connected component holding the best detection).
2. Robot mask is disk-dilated 8 px (`mask.dilate_px`).
3. Robot-free reference image: LaMa inpaint of the expanded mask, then
   `reference.deshadow` replaces the fill's low-frequency luminance with a
   harmonic field interpolated from the boundary ring — this removes the
   robot-shaped brightness silhouette that Qwen otherwise reproduces.

**Pass B — edit + composite**

4. `TextEncodeQwenImage21` full-denoise edit at native resolution
   (`resolution=0`, `images.image_1` = reference, plan prompt, steps=25, cfg=1,
   euler/simple, seed=42). The mask is NOT fed to the sampler — we found
   latent-masked sampling produces boundary artifacts; instead the Qwen output
   is composited inside the expanded mask only.
5. `composite_native.png` = Qwen inside mask (with a 6 px inward alpha ramp,
   `mask.soften_px`), original outside; reloaded and asserted to have zero
   differing pixels outside the mask.
6. Aspect-cover resize + center crop to exactly 480×320 (`postprocess.py`).
   Crop bounds and clipped task objects are recorded in metadata.

## Layout

```
robot_removal/
  config.yaml      prompts, thresholds, sampler settings, output size
  manifest.py      build hashed input JSONL from data/robot_removal/full
  workflow.py      API prompt graph builder (+ box_2d → px)
  comfy.py         /upload/image, /prompt, /history, /view client
  masks.py         candidate selection + disk dilation + coverage flags
  reference.py     LaMa/diffusion robot-free reference + deshadow
  postprocess.py   composite (with soften ramp), preservation assert, 480x320
  batch.py         resumable two-pass runner (lock, retries, .stage, quarantine)
  audit.py         post-run SAM3 remnant/ghost flags (review, not proof)
  report.py        HTML contact sheet
  ui_export.py     exports workflow.json (UI) + workflow.api.json
```

## Commands

```bash
# setup check (env, GPU, model SHA-256 vs data/robot_removal/model_manifest.json)
bash scripts/setup_robot_removal.sh

# ComfyUI runs via supervisor at 127.0.0.1:18188. IMPORTANT: model residency
# requires the LRU output cache — loader patchers get GC'd under the default
# RAM-pressure cache and every prompt reloads ~17 GB of weights (=> ~20 s/img).
# With --cache-lru both model families stay resident (~4.5 s/img total).
COMFYUI_ARGS="--disable-auto-launch --disable-xformers --cache-lru 200 --port 18188 --enable-cors-header"
# (set in /etc/environment on this instance, consumed by /opt/supervisor-scripts/comfyui.sh;
#  do NOT use --highvram — the sam3.pt checkpoint requires dynamic weight loading)

# rebuild manifests
/venv/main/bin/python -m robot_removal.manifest data/robot_removal/full data/robot_removal/full_inputs.jsonl

# benchmark
bash scripts/run_robot_removal.sh \
  --manifest data/robot_removal/benchmark_inputs.jsonl \
  --output-root outputs/robot_removal/benchmark_v2

# staging (100) and full (1960)
bash scripts/run_robot_removal.sh \
  --manifest data/robot_removal/staging_inputs.jsonl \
  --output-root outputs/robot_removal/staging
bash scripts/run_robot_removal.sh \
  --manifest data/robot_removal/full_inputs.jsonl \
  --output-root outputs/robot_removal/full

# report + audit
/venv/main/bin/python -m robot_removal.report outputs/robot_removal/<run>
/venv/main/bin/python -m robot_removal.audit outputs/robot_removal/<run> --sample 0.25
```

Resume is automatic: episodes with a `complete.json` matching input SHA-256 +
config hash + workflow version are skipped; phase A state survives in
`.stage/state.json` so a crash mid-pass-B resumes at the edit. `--force`
starts a new attempt. Failed episodes get `failed.json` + are listed in
`<run>/failed.json` (exit 1).

## Artifacts per episode

```
outputs/robot_removal/<run>/<dataset>/<episode>/
  original.jpg  original_rgb.png  raw_robot_mask.png  expanded_robot_mask.png
  mask_overlay.png  reference_lama.png  inpaint_native.png  composite_native.png
  zerowam_480x320.png  metadata.json  metrics.json  complete.json  .stage/
```

## Flags (metadata.json / report)

- `mask_too_small|too_large` — coverage outside [0.005, 0.45]
- `flat_fill:<in>/<out>` — gradient inside fill <35% of ring (texture-less fill)
- `seam_step:<n>` — mean luminance step across mask boundary >45
- `crop_clips_object:<id>` / `crop_removes_object:<id>` — task object cut by crop
- audit (review flags, not proof): `coverage_miss` (robot detected outside the
  mask on the original), `robot_fill` (robot-like detection inside the
  composite's mask — fires on faint luminance ghosts AND real remnants)

## Known artifacts / decisions

- **v3 (pending confirmation, see plan §8):** mask-free editing — `image_1` =
  original image, removal positive + `robot/cables/cast shadow` negative @
  cfg=4 — beat the LaMa-reference pipeline on every diagnosed failure (regen,
  ghosts, flood repaint, swallowed objects). `config.yaml` already carries
  the cfg4 + negative; `batch.py` still builds the LaMa reference — switch
  phase B to feed `original` directly once confirmed.
- Feeding the *original* image at cfg=1 with no negative made Qwen regenerate
  the robot (~15% of scenes); cfg4 + robot-negative removed it in all tested
  cases. Scene-only positives reproduce the robot at any cfg — the negative
  needs the concept active in the positive.
- LaMa's fill retains an arm-shaped smear that Qwen "completes" into a
  repainted arm (grabtissue_138), plus luminance ghosts on flat surfaces
  (svla_037/038). Deshadow reduced but didn't eliminate them.
- Residual: faint luminance ghosts may survive (flagged `robot_fill`/
  `seam_step`); quarantine worst cases via audit.
- Full-denoise edit + image-space composite replaced latent masked sampling
  (boundary dot artifacts on packed 64ch latents).
- 57 episodes lack `robot_box_2d`; SAM3 runs text-only there.
- ComfyUI deadlocked once mid-seg job (transient); `client.wait` posts
  /interrupt on timeout and the runner retries with fresh attempts.
