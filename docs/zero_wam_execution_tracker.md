# Zero-WAM inference execution tracker

## Goal prompt

Implement and validate Zero-WAM inference with the released RoboTwin post-trained checkpoint. Run RoboTwin locally and the stateful model on one Modal GPU. First prove a complete `place_empty_cup` rollout with text conditioning and record whether it succeeds. Then validate video conditioning using the published HumanGen `place_empty_cup` human video and its matching latent, including the MP4 encoding path. Keep upstream Zero-WAM unchanged, keep separate Python environments, and record commands, evidence, failures, and outputs here. Stop after video-conditioning validation; integration with our own HumanGen pipeline is a later goal. Continue through routine setup and fixes autonomously, but report any missing external credentials or services precisely.

## Current state

- [x] Read the plan and paper; pin upstream Zero-WAM as `third_party/Zero-WAM` at `08e2c4a`.
- [x] Identify the published `place_empty_cup` HumanGen pairing: sample `273_robotwin_Use_an_arm_to_place_the_empty_cup_on_the_coaster`, with a public MP4 archive and matching `.pth` latent.
- [x] Clone RoboTwin locally at revision `2eeec322` (`/workspace/Robotwin`).
- [x] Add the local Modal adapter and client; Python compilation, shell syntax, and module import checks pass.
- [x] Install RoboTwin dependencies and assets; render and local expert success checks pass.
- [x] Load the model on a Modal GPU and validate reset, action shape, and KV-cache updates; text rollout reached at least 48 control steps.
- [x] Build the Modal Python 3.12 inference image and import Zero-WAM, PyTorch, and FlashAttention on Modal. The image uses the official `flash_attn-2.8.3+cu12torch2.9` wheel.
- [x] Run a text-conditioned `place_empty_cup` rollout; seed `100000` succeeded after 132 control steps, with video, metrics, and log saved.
- [x] Extract the matching public human MP4 and verify its GPU encoding against the published latent.
- [x] Run latent-conditioned and MP4-conditioned closed-loop rollouts; save videos, metrics, seeds, and logs. (Latent succeeded; MP4 rollout ran all 500 steps with task-consistent behavior but failed the success check.)

## Evidence and results

| Check | Result | Evidence path or note |
| --- | --- | --- |
| RoboTwin render | Pass | `/workspace/Robotwin/.venv/bin/python script/test_render.py` printed `Render Well` |
| RoboTwin expert | Pass | `place_empty_cup`, seed `100000`: `plan_success=True`, `task_success=True` |
| Modal model load | Pass | `modal run zero_wam/modal_app.py::smoke --mode text` loaded all 15 transformer shards on A100 |
| Modal runtime imports | Pass | `modal run zero_wam/modal_app.py::check_runtime` completed successfully |
| Modal checkpoint assets | Pass | `prepare_checkpoint` fetched 30 model files and the HumanGen latent; `modal volume ls zero-wam-weights` confirmed model directories and latent |
| Modal action shape | Pass | Text smoke returned `(16, 2, 16)` |
| Text rollout | Pass | `outputs/zero_wam/text/run.log`; `outputs/zero_wam/text/stseed-100000/metrics/place_empty_cup/res.json` reports `1/1`; 37-frame video under `outputs/zero_wam/text/stseed-100000/visualization/place_empty_cup/` |
| Human MP4 and latent mapping | Pass | Sample `273_robotwin_Use_an_arm_to_place_the_empty_cup_on_the_coaster`; MP4 is 121 frames, 1108×828, 24 fps. GPU encoding and saved latent both shape `(1, 48, 16, 20, 28)`; cosine similarity `0.9985909462` |
| Encode equivalence (sharpened) | Near-equal, not identical | `verify_human_prompt` reports cosine `0.9985909`, rel L2 diff `0.0523`, max abs diff `1.086`. Per-latent-frame cosine is `~0.9993` for frames 0–14 but `0.9897` on the last frame — the streaming-VAE tail differs most from the released `.pth` |
| Latent rollout | Pass | `outputs/zero_wam/latent/run.log`; `outputs/zero_wam/latent/stseed-100000/metrics/place_empty_cup/res.json` reports `1/1`; video under `outputs/zero_wam/latent/stseed-100000/visualization/place_empty_cup/` |
| Latent rollout, same seed rerun | Fail (0/1) | `outputs/zero_wam/latent_run2/run.log`; `res.json` reports `0/1` at seed `100000`, ran all 500 steps; contact sheet shows task-directed grasp-and-carry behavior. Same task, scene, and published latent as the passing run — outcome flipped anyway |
| MP4 rollout | Fail (0/1) | `outputs/zero_wam/video/run.log`; `outputs/zero_wam/video/stseed-100000/metrics/place_empty_cup/res.json` reports `0/1`; seed `100000`, ran all 500 steps. Video shows the arm grasp the cup and move over the coaster — task-inferring behavior, final state did not satisfy the success check. |

## Setup notes

- The root `uv` environment is Python 3.12. RoboTwin uses a separate local Python 3.10 environment. The Modal inference image uses Python 3.12 with PyTorch 2.9 and an official matching FlashAttention wheel; this avoids a long Python 3.10 source build while keeping the released PyTorch/CUDA stack.
- `.env` contains `MODAL_API_KEY`, `MODAL_SECRET_TOKEN`, and `HF_TOKEN`; the CLI wrapper maps the aliases to Modal's expected environment variables. `modal token info` authenticated successfully. Never copy secret values into this document or logs.
- `run_robotwin.sh` uses `modal run` to keep the GPU worker alive only for the local RoboTwin process. It does not deploy an endpoint.
- The upstream requirements combine inference and post-training packages. Modal filters out LeRobot and Weights & Biases from the inference image because LeRobot 0.3.3 declares `torch<2.8`, conflicting with the released checkpoint's tested PyTorch 2.9 runtime. Neither package is imported by the inference server.
- The upstream raw-MP4 fallback calls `list(reader)` when `imageio` reports an unbounded frame count, causing `MemoryError`. Modal installs `decord` to select finite frames. Its streaming VAE also needs the first frame followed by four-frame chunks; the adapter applies that only for raw-video encoding. The resulting MP4 latent nearly matches the released `.pth`.
- HumanGen's `human_data/` contains compressed shards; the `place_empty_cup` latent is a loose file. Extract only the matching MP4, not the complete dataset.
- The released latent file is 5.1 MB and contains a `(8960, 48)` bfloat16 tensor with 16 latent frames at 20×28, representing 320×448 video. The raw MP4 encoder is configured to match that geometry.
- The local GPU is an 8 GB RTX 3070 Laptop; it is for SAPIEN rendering, while Zero-WAM runs on Modal.
- Text conditioning is disabled in ICL modes by upstream design: `video_guidance_scale=-1` makes `_reset_icl` use the shipped `empty_text_emb.pt` for every transformer branch, so `latent`/`video` rollouts are video-only tests. The prompt string must still be non-empty (`_reset_icl` raises otherwise) but its embedding is unused; the per-step `task` field is not read after reset.
- Rollout outcomes are stochastic, not a deterministic function of task+seed+latent: nothing seeds the Modal container's torch RNG, so each `modal run` samples fresh denoising noise. Evidence: the same-seed latent rerun flipped pass→fail with a byte-identical ICL latent. Compare modes by success rate over N episodes, not by single-rollout equality; RoboTwin's own protocol is 100 rollouts × 3 seeds.
- `run_robotwin.sh` now resolves a relative `SAVE_ROOT` against the caller's cwd; the eval client chdirs into `ROBOTWIN_ROOT`, so a relative path would otherwise write under `/workspace/Robotwin/outputs/`.
- GPU self-reporting added to `modal_app.py`: a `[zw-metrics] post-load` VRAM/util line at container start, a per-call `[zw-metrics] reset|infer|kv <t>s vram_alloc=… vram_peak=…` line per `step`, and a `stats` method for on-demand snapshots (`torch.cuda` counters plus `nvidia-smi` util/mem). Prints stream into the local `run.log` via `modal run`.
