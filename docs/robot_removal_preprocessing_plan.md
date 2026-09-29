# SO-101 robot-removal preprocessing plan

Prepare approximately 2,000 human-generation reference images on a Vast.ai RTX 5090 32GB using a ComfyUI template. Remove robots at native resolution and reconstruct plausible hidden background inside the expanded robot mask. Preserve original pixels outside that mask, then resize/crop to exactly 480×320 for Zero-WAM.

## Progress record

Check an item only after its work is complete. Record the date and evidence (commit, output path, or report) below; leave unverified work unchecked.

| Date | Completed item | Evidence / notes |
| --- | --- | --- |
| | | |

## 1. Inspect inputs and reuse existing code

- [ ] List the pilot bucket and inspect originals, cached masks, boxes, and logs.
- [ ] Identify the full dataset input root and build a hashed input manifest.
- [ ] Download inputs through official HF bucket APIs without changing sources.
- [ ] Review code entry points and validate cached masks before reuse.

List the [pilot bucket](https://huggingface.co/buckets/akoniti/ICL-so101/tree/work/humangen_pilot/fast_h3_v5_nohand_strict) through the official HF bucket API. Inspect originals, cached `robot_first_robotmask.png`, `robot_boxes.json`, and existing logs before changing the pipeline. Build an input manifest with dataset/episode IDs, source paths, dimensions, and SHA256 hashes. Identify the full dataset input root separately from the pilot.

Download originals and relevant masks/metadata through `HfApi.download_bucket_files` or filtered bucket sync; use the official [HF bucket transfer support](https://huggingface.co/docs/huggingface_hub/guides/buckets), including Xet. Never overwrite source files or upload into the source prefix.

```bash
hf buckets sync \
  hf://buckets/akoniti/ICL-so101/work/humangen_pilot/fast_h3_v5_nohand_strict \
  /workspace/ICL-so101/data/robot_removal/pilot \
  --include '*.jpg' --include '*.png' --include '*.json' --include '*.log'
```

Review these code entry points:

| Code | Planned use |
| --- | --- |
| [curate/selection.json](../curate/selection.json#L1) | Upstream repositories and external-camera assignments |
| [curate/io.py:37](../curate/io.py#L37) | Resolve episode video path/start time for LeRobot v2.1 or v3 |
| [curate/io.py:47](../curate/io.py#L47) | Load upstream metadata; support v2.1 without conversion |
| [humangen/edit.py:111](../humangen/edit.py#L111) | Reuse SAM3 robot-instance selection and box hints |
| [humangen/edit.py:155](../humangen/edit.py#L155) | SAM3 inference; explicitly select CUDA when adapting |
| [humangen/edit.py:182](../humangen/edit.py#L182) | Reuse cached-mask lookup; verify provenance because this helper can fall back to SAM2.1 |
| [humangen/edit.py:294](../humangen/edit.py#L294) | Existing removal entry point; replace its LaMa generation path with the new Qwen workflow |

Validate cached masks against their originals: dimensions, polarity, coverage, and source hash. Reuse approved masks; rerun SAM3 otherwise. Do not treat unreviewed cached masks as ground truth.

## 2. Set up the destination ComfyUI template

- [ ] Locate template paths, Python environment, and supervisor service.
- [ ] Validate CUDA execution on the RTX 5090.
- [ ] Install and pin compatible ComfyUI/SAM3 dependencies.
- [ ] Download and hash Qwen diffusion, Qwen3-VL encoder, VAE, and SAM3 models.
- [ ] Validate model loading and select a stable residency/offload strategy.
- [ ] Bind to localhost and verify SSH access with no unauthenticated public route.

Use the new host's existing ComfyUI installation and Python environment. Locate its launch wrapper, model directories, and supervisor service; create reproducible setup/start wrappers around those paths.

- Validate RTX 5090 CUDA execution. Use a Blackwell-compatible PyTorch build with CUDA 12.8 or newer; retain the template stack if it works. Do not replace the host driver.
- Pin tested ComfyUI/custom-node commits, Python dependencies, model revisions, and file hashes.
- Prefer [native ComfyUI Qwen nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_qwen.py) and [official workflow templates](https://github.com/Comfy-Org/workflow_templates). Inspect existing editing/inpainting workflows before writing adapters.
- Evaluate [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) and its example workflows. Review its installer/environment dependencies before installation. Obtain [SAM3 checkpoint access](https://huggingface.co/facebook/sam3) if required.
- Download models using HF Hub APIs with pinned revisions. Use the [official Qwen 2.1 repository](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).

| Component | Preferred model | ComfyUI directory |
| --- | --- | --- |
| Diffusion | `qwen_image_2.1_int8_convrot.safetensors` | `models/diffusion_models/` |
| Text encoder | `qwen3vl_8b_int8_convrot.safetensors` | `models/text_encoders/` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` |
| SAM3 | Checkpoint supported by the pinned integration | Integration's documented directory |

Keep a long-lived server and resident models where VRAM permits. If combined residency is unstable, segment the inputs first, release SAM3 once, and then run Qwen with controlled offload. Preserve native resolution and VAE quality.

Bind ComfyUI to `127.0.0.1:8188`; use SSH tunneling. Disable any unauthenticated public template route. Integrate the launch wrapper with the template's supervisor service.

## 3. Build and validate the workflow

- [ ] Build SAM3 segmentation and save raw/expanded masks.
- [ ] Validate mask polarity, task-object protection, and configurable dilation.
- [ ] Prove genuine mask-constrained Qwen sampling on synthetic and real inputs.
- [ ] Validate native-resolution generation and padding/unpadding.
- [ ] Implement hard compositing and exact saved-output preservation checks.
- [ ] Export and smoke-test both UI and API workflow JSON.

```text
Load Image
  -> load approved cached mask or run SAM3
  -> save binary raw robot mask
  -> grow mask by configurable 8 px
  -> Qwen 2.1 genuinely masked editing at native resolution
  -> save decoded candidate before compositing
  -> hard composite with original
  -> preservation validation
  -> aspect-preserving resize / center crop
  -> save PNGs and metrics
```

### Segmentation

Use robot/arm/gripper prompts and available box hints. Include robot base, attached wires, and fringes conservatively. Preserve held objects, handles, and task cables. Start with a disk-shaped 8 px dilation; compare 5/8/10 px on the pilot. White means editable pixels.

Flag uncertain object overlaps, incomplete robot coverage, and shadows outside the expanded mask. Additional shadow removal requires an explicitly reviewed mask. Exact preservation applies outside the mask; hidden content inside it cannot be recovered exactly.

### Mask-constrained Qwen editing

Use this prompt:

> Remove the robot arm/gripper and reconstruct the occluded background naturally. Preserve all other objects, geometry, camera viewpoint, lighting, colors, and scene contents unchanged.

Start with existing ComfyUI masked-editing nodes. Run a short compatibility smoke test before processing the pilot; native image-reference conditioning alone does not supply the mask:

1. Inspect compatible native editing nodes and inpainting workflows.
2. Test original-image VAE encoding plus `SetLatentNoiseMask` and masked sampling with Qwen 2.1's actual latent layout. Replace any empty target latent with the encoded original; wire the mask into sampling.
3. Verify mask polarity, alignment, and unmasked latent restoration using a synthetic rectangle and three real scenes. Instrument the sampler if needed.
4. If native masked sampling is incompatible, use a tested adapter that restores unmasked latents at each step. Do not substitute prompt-only removal or silently change model versions.

Set native reference sizing (`resolution=0` where supported). For dimensions incompatible with model alignment, pad deterministically and unpad after decoding. Do not resize the image to 480×320 before removal.

Start from official editing sampler defaults. Tune steps/denoise on pilot images. Record actual precision, sampler, scheduler, steps, CFG, attention backend, seeds, and versions. Avoid acceleration LoRAs until baseline quality passes.

### Hard composite and validation

Decode the original into canonical RGB and preserve the original file separately. Use integer selection and save the composite as lossless PNG:

```python
final = original_rgb.copy()
final[expanded_mask] = qwen_rgb[expanded_mask]
```

Default feather width is zero. Any optional feather should remain inside the expanded mask; an outside seam band must be explicitly recorded.

Reload the saved composite and assert zero differing pixels, zero maximum channel difference, and zero mean absolute difference outside the permitted mask/seam region. Reject invalid dimensions, malformed masks, nonfinite outputs, and incomplete artifacts. Compare decoded RGB pixels, not JPEG file bytes.

## 4. Apply the exact Zero-WAM transform

- [ ] Implement deterministic aspect-preserving resize and center crop.
- [ ] Test 640×480 → 480×360 → 480×320 and other source dimensions.
- [ ] Assert final dimensions and record crop bounds/task-object crop risks.

Use pinned Pillow/OpenCV processing after native-resolution compositing:

```python
scale = max(480 / width, 320 / height)
new_width = math.ceil(width * scale)
new_height = math.ceil(height * scale)
resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
left = (new_width - 480) // 2
top = (new_height - 320) // 2
final = resized.crop((left, top, left + 480, top + 320))
assert final.size == (480, 320)
```

The required 640×480 case is exactly 480×360 followed by crop `(0, 20, 480, 340)`. Use one aspect-preserving scale with deterministic integer rounding for other dimensions.

Test the exact transform and synthetic landmarks. Display crop bounds in the report and flag task objects cut off by the crop. Do not silently change the agreed center crop. Pixel equality is checked at native resolution; downstream resampling blends neighboring pixels.

## 5. Implement resumable batch execution

- [ ] Implement directory/manifest input and non-interactive ComfyUI submission.
- [ ] Save all per-episode artifacts, metadata, and metrics.
- [ ] Implement atomic completion, hash-aware skipping, and versioned force reruns.
- [ ] Implement locking, interrupted-job reconciliation, and clean resume.
- [ ] Implement separate failure logs, bounded retries, and OOM handling.
- [ ] Create config and test interruption/resume without source changes.

Create `robot_removal.batch`, accepting either a recursive input directory or JSONL manifest. Submit API-format workflows to local ComfyUI `/prompt`; monitor jobs through WebSocket/history and check node compatibility via `/object_info`. Keep model-loading nodes stable between jobs for caching. Deliver both UI and API workflow JSON.

Save per episode:

```text
outputs/robot_removal/<run>/<dataset>/episode_XXX/
  original.jpg
  original_rgb.png
  raw_robot_mask.png
  expanded_robot_mask.png
  mask_overlay.png
  inpaint_native.png
  composite_native.png
  zerowam_480x320.png
  metadata.json
  metrics.json
  attempts/
  complete.json
```

- Never write into source directories.
- Stage outputs and publish atomically; write `complete.json` last after validation.
- Skip only complete valid outputs matching input/config/workflow/model hashes. `--force` creates a new attempt.
- Record job IDs and reconcile interrupted jobs before resubmission. Lock episode outputs against concurrent writers.
- Log failures separately with stage, exception, seed, settings, and retry history. Continue unrelated inputs and return nonzero for unresolved failures.
- Allow two configured retries by default. Quarantine ambiguous segmentation, remnants, artifacts, and crop failures; never silently accept them.
- Handle OOM through the validated offload/residency strategy, retaining native resolution.

Config: model/template paths, input/output roots, prompts, mask thresholds/dilation, feather width, Qwen sampling settings, seeds, target dimensions, retries, quality thresholds, and residency strategy.

## 6. Validate quality and benchmark

- [ ] Freeze representative benchmark/tuning/holdout manifests.
- [ ] Run a 30–50-image benchmark on the destination RTX 5090.
- [ ] Review outputs and publish the HTML/contact-sheet report.
- [ ] Report precision, peak VRAM, stage timings, throughput, yield, and retries.
- [ ] Estimate time and remaining work for 2,000 accepted images.
- [ ] Pass validation gates and complete a 100-image staging run.
- [ ] Process the full manifest and reconcile accepted/failed/pending inputs.

Benchmark 30–50 representative unique pilot inputs across datasets, robot sizes, lighting, drawers, handles, cables, held objects, and occlusions. Use a separate tuning subset and fixed holdout manifest with input hashes. Review all benchmark outputs visually.

Quality checks:

- Flag implausibly small/large masks and protected-object overlap; calibrate thresholds on the pilot.
- Detect likely robot remnants using a second segmentation pass, plus gross blank/texture/seam artifacts. Treat detectors as flags rather than proof of success.
- Require exact native outside-mask preservation and exact 480×320 output dimensions.
- Generate an HTML/contact-sheet report with original, mask overlay, pre-composite Qwen output, native composite, final Zero-WAM image, crop bounds, flags, and retries.

Measure warm uncached processing separately from startup/downloads and cached-mask runs. Record synchronized GPU stage timings and end-to-end wall time, including validation, saving, and retries:

| Metric | Report |
| --- | --- |
| Hardware | GPU model, VRAM, driver, Torch/CUDA |
| Model settings | Actual precision/quantization, versions, steps, sampler |
| Memory | Peak allocated/reserved and device-wide VRAM, including node subprocesses |
| Latency | SAM3, Qwen conditioning/sampling/decode, total time/image; median and p95 |
| Throughput | Completed and accepted images/hour |
| Quality | Manually reviewed first-pass success rate, final yield, retries, unresolved failures |
| Production estimate | Startup + `2000 / accepted_images_per_hour` hours, with assumptions |

If yield is below 100%, report additional candidates/manual work needed to obtain 2,000 accepted images. Do not assume several seconds per image.

Run preservation/masked-sampling tests, repeat/resume tests, pilot review, and a 100-image staging run before the full dataset. Stop on systematic scene damage or preservation failures. Account for every input as accepted, failed, or pending review.

## 7. Deliverables and run interface

- [ ] Deliver tested workflow JSON, setup/start/batch scripts, and config.
- [ ] Deliver pinned environment/model manifest and exact README commands.
- [ ] Deliver sample outputs, benchmark results, report, and failure-mode notes.
- [ ] Record full-run output location, measured throughput, and remaining decisions.

Deliver:

- [ ] Working UI/API ComfyUI workflow JSON.
- [ ] Idempotent setup/start scripts and pinned environment/model manifest.
- [ ] Batch runner, config, and README with exact tested commands.
- [ ] Pilot outputs, HTML report, benchmark results, and failure-mode notes.
- [ ] Full-run accepted/failure manifests and final reconciliation.

Target command interface below; implement and test these scripts before publishing the final README:

```bash
cd /workspace/ICL-so101
bash scripts/setup_robot_removal.sh --config robot_removal/config.yaml
bash scripts/start_comfyui.sh --listen 127.0.0.1 --port 8188
```

From the user's computer:

```bash
ssh -p <VAST_SSH_PORT> -L 8188:127.0.0.1:8188 root@<VAST_HOST>
```

The setup script must provide a batch wrapper using the template's verified Python environment:

```bash
# Pilot benchmark
bash scripts/run_robot_removal.sh \
  --manifest data/robot_removal/benchmark_inputs.jsonl \
  --config robot_removal/config.yaml \
  --output-root outputs/robot_removal/pilot_benchmark --benchmark --force

# Full dataset; resume and skip valid completed outputs by default
bash scripts/run_robot_removal.sh \
  --manifest data/robot_removal/full_inputs.jsonl \
  --config robot_removal/config.yaml \
  --output-root outputs/robot_removal/full
```

Completion report: exact ComfyUI/full-dataset commands, output location, measured throughput and 2,000-image estimate, unresolved failures, and remaining decisions about mask/shadow policy or model compatibility.
