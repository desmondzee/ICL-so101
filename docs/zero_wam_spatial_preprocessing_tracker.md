# Zero-WAM robot-video spatial preprocessing investigation

## Scope and constraints

Determine the robot-video preprocessing and latent geometry used by released Zero-WAM training data, then recommend a compatible transform for two 640×480 SO-101 cameras.

This is investigation only. Do not regenerate SO-101 latents, change training configuration, or launch training. Minimize downloads; record inaccessible artifacts instead of inferring their contents.

## Sources

- HumanGen dataset: `https://huggingface.co/datasets/Robbyant-Research/HumanGen`
- Zero-WAM source: `https://github.com/robbyant-research/Zero-WAM`
- Pinned local source: `third_party/Zero-WAM` at `08e2c4ae41e2b63573a299825cebe6753481407c`

## Evidence labels

- **Confirmed:** directly established by artifact metadata, tensor inspection, executable source, or decoded comparison.
- **Inferred:** best explanation supported by evidence but not independently proven.
- **Unknown:** evidence is absent, inaccessible, or conflicting.

No latent canvas alone proves crop, stretch, or letterboxing.

## Primary questions

- [x] What spatial resolutions encoded the released robot-video latents?
- [x] Was the sampled RoboTwin raw video directly resized, cropped, padded, letterboxed, or transformed another way?
- [x] Did AgiBot, RoboCOIN, RoboMIND, InternData-A1 and OXE/Bridge use the same geometry as RoboTwin?
- [x] What geometry and transform should two 640×480 SO-101 camera streams use?

## Current findings

| Finding | Status | Evidence |
|---|---|---|
| Released runtime observation preprocessing directly calls bilinear interpolation to configured `(height, width)` with no crop/pad in that path. | Confirmed for online runtime only | `third_party/Zero-WAM/wan_va/wan_va_server.py`; must not yet be generalized to offline latent generation. |
| Published configs declare Demo 256×256, Franka 224×320 and RoboTwin 224×288 robot canvases. | Confirmed config values | `wan_va/configs/va_{demo,franka,robotwin}_cfg.py`; values do not prove released latent preprocessing. |
| AgiBot, RoboCOIN, RoboMIND, InternData-A1 and OXE training configs inherit the RoboTwin training config and do not override height/width; their declared robot canvas is therefore 224×288. | Confirmed config inheritance; artifact verification pending | `wan_va/configs/va_{agibot,robocoin,robomind,interna1,oxe}_train_cfg.py`. |
| Two released RoboTwin episodes across high, left-wrist and right-wrist cameras use native MP4 480×640 at 50 fps, sampled every four frames to 12.5 fps and encoded at 224×288 into 14×18 VAE latents. | Confirmed artifact metadata | HumanGen revision `971d12e…`; six MP4/latent pairs inspected. |
| All six sampled RoboTwin latent tensors have flattened shape `[2268,48]` = `9×14×18×48`, BF16, with 33 sampled video frames and 9 latent frames. | Confirmed artifact metadata | Episodes 42 and 77 from `adjust_bottle-demo_clean_collect_200-1000`. |
| Released AgiBot, RoboCOIN, RoboMIND and OXE samples all record encoded video 224×288 and latent 14×18. | Confirmed artifact metadata | Smallest shard from each source; ≥2 latent samples per available/configured camera. |
| Released InternData-A1 samples record encoded video 192×352 and latent 12×22 for head and both hand cameras. | Confirmed artifact metadata; contradicts inherited current config | Smallest InternData-A1 shard; two episodes × three cameras. |
| The five external sources therefore did **not** all use the same encoded resolution, despite current public train configs inheriting RoboTwin's 224×288 fields. | Confirmed artifact conclusion | Four sources sampled at 224×288; InternData-A1 sampled at 192×352. |
| Sampled native video geometry varies: 640×480 for most cameras, 1280×720 for RoboCOIN high, and 640×360 for InternData head. | Confirmed sampled MP4 metadata | Two raw videos per available camera in selected task shards. |
| Released repository contains runtime encoding but no offline robot-latent generation program. Offline crop/resize/pad behavior cannot be established from the public source alone. | Confirmed repository limitation | Complete source grep at pinned public revision; only initial public commit exists. |
| Decoding all 33 frames of a released RoboTwin high-camera latent with the released Wan VAE matches direct bilinear resize of the exact raw frames by a large margin: MSE `0.000253` / PSNR `35.97 dB`, versus center crop `0.006299` / `22.01 dB`, and black letterbox `0.031002` / `15.09 dB`. | Confirmed for sampled RoboTwin episode/camera | HumanGen `adjust_bottle…` episode 42 high camera; released `zero-wam-pretrain` VAE; report `/sim/reports/spatial_decode_robotwin_ep42.json`. |
| The sampled RoboTwin offline latent was produced from a direct resize, not the tested aspect-preserving center crop or letterbox transforms. | Confirmed for sampled RoboTwin episode/camera | Direct-resize reconstruction error is ~25× lower than center crop and ~122× lower than black letterbox. |
| A more discriminating RoboCOIN high-camera sample changes aspect ratio from native 1280×720 (16:9) to 288×224 (~9:7). Across all 365 reconstructed frames, direct resize gives MSE `0.0000565` / PSNR `42.48 dB`, versus center crop `0.006924` / `21.60 dB`, and black letterbox `0.046088` / `13.36 dB`. | Confirmed for sampled RoboCOIN episode/camera | `Split_aloha_scoop_coffee_beans` source episode 345; report `/sim/reports/spatial_decode_robocoin_high345.json`. |
| The large-aspect-change RoboCOIN latent was directly resized: direct-resize error is ~123× lower than center crop and ~816× lower than black letterbox. | Confirmed | Released latent, exact raw MP4 and released Zero-WAM VAE. |
| One released RoboTwin human-ICL latent records video 320×448 and latent 20×28 at 12 fps, despite current Robotwin config declaring ICL width 480. | Confirmed artifact/config discrepancy | Representative released human latent inspected locally; this concerns human ICL, not robot-video geometry. |
| SO-101 source robot videos are 640×480 at 30 fps. | Confirmed | Dataset indexes and complete MP4 scan; recommendation remains open. |

## Dataset artifact inventory and download budget

| Dataset | Robot latent distribution | Raw/source distribution | Candidate samples | Downloaded bytes | Access status |
|---|---|---|---|---:|---|
| RoboTwin | Individually accessible `.pth` | Individual MP4s and metadata | Episodes 42/77, high/left/right | 20,788,269 | Sample complete |
| AgiBot | `tar.zst` shards | Same shard | Smallest shard `part-00005`, 3,732 entries | 2,570,915,419 | Sample extracted |
| RoboCOIN | `tar.zst` shards | Same shard | Smallest shard `part-00009`, 1,950 entries | 3,613,178,830 | Sample extracted |
| RoboMIND | `tar.zst` shards | Same shard | Smallest shard `part-00004`, 2,919 entries | 952,076,702 | Sample extracted |
| InternData-A1 | `tar.zst` shards | Same shard | Smallest shard `part-00011`, 1,848 entries | 511,812,551 | Sample extracted |
| OXE / Bridge | `tar.zst` shards | Same shard | Smallest shard `part-00002`, 2,162 entries | 1,028,475,626 | Sample extracted |

## Phase 1 — public tree and source-code trace

- [x] Freeze HumanGen repository revision `971d12e942bf101c0d47716f928b1c044d02ac71` and enumerate files/sizes without downloading archives.
- [x] Locate individually accessible RoboTwin robot latent files.
- [x] Locate the smallest representative shard for each external source.
- [x] Identify RoboTwin camera names and episode/sample mappings.
- [ ] Trace latent-generation entrypoints and exact decoder→transform→VAE path.
- [x] Search released source for resize, center/random crop, letterbox, padding, aspect-ratio buckets and camera concatenation.
- [ ] Trace linked upstream preprocessing repositories where public.
- [x] Separate online inference preprocessing from offline latent-generation preprocessing: online direct bilinear resize is present; offline encoder is not released.

## Phase 2 — RoboTwin first

For at least two episodes and every available camera:

- [x] Download individual robot latent `.pth` files for two episodes and all three cameras, plus matching small MP4s.
- [x] Record `video_height`, `video_width`, `latent_height`, `latent_width`, `latent_num_frames`, `frame_ids`, tensor shape/dtype and text fields.
- [x] Inspect corresponding original video metadata and native camera dimensions.
- [x] Compare metadata across episodes and cameras.
- [x] Establish transform behavior by decoded comparison when offline transformation metadata/source is absent.
- [x] Decode all 33 frames of one representative RoboTwin high-camera latent with the released Wan VAE.
- [x] Compare reconstruction against direct bilinear resize, aspect-preserving center crop and black letterbox transforms; direct resize wins decisively.

## Phase 3 — five external pretraining sources

For each source, select the smallest practical shard or range-capable artifact. Do not download a whole corpus.

- [x] AgiBot: two episodes across head, left-hand and right-hand.
- [x] RoboCOIN: two samples across high, left-wrist and right-wrist cameras; `cam_head_rgb` exists elsewhere in the shard but is not used by the released training config and was not present in the selected task.
- [x] RoboMIND: two episodes from the sole top camera.
- [x] InternData-A1: two episodes across head, left-hand and right-hand.
- [x] OXE/Bridge: two episodes from the sole configured image camera.
- [x] Record shard names, total sizes, downloaded bytes and extraction method.
- [x] Report archive limitation: `tar.zst` is not seekable here, so each smallest shard required a full download and sequential scan.
- [x] Extract the same latent and native-video metadata fields as RoboTwin.
- [x] Determine geometry compatibility: four sources match RoboTwin 224×288; InternData-A1 uses 192×352. Transform method remains unknown.

## Artifact inspection schema

Record one row per dataset/camera/episode:

| Dataset | Camera | Episode/sample | Native H×W | Encoded video H×W | Latent F×H×W×C or stored shape | Frame IDs | Verified transform | Evidence level |
|---|---|---|---:|---:|---|---|---|---|
| RoboTwin | high | adjust_bottle ep42/77 | 480×640 @ 50 fps | 224×288 @ 12.5 fps | `[2268,48]` = 9×14×18×48 BF16 | 0,4,…,128 | Direct bilinear resize confirmed on ep42 by 33-frame VAE reconstruction | Confirmed |
| RoboTwin | left wrist | adjust_bottle ep42/77 | 480×640 @ 50 fps | 224×288 @ 12.5 fps | `[2268,48]` = 9×14×18×48 BF16 | 0,4,…,128 | Geometry confirmed; transform method unknown | Confirmed/Unknown |
| RoboTwin | right wrist | adjust_bottle ep42/77 | 480×640 @ 50 fps | 224×288 @ 12.5 fps | `[2268,48]` = 9×14×18×48 BF16 | 0,4,…,128 | Geometry confirmed; transform method unknown | Confirmed/Unknown |
| AgiBot | head/left hand/right hand | task_714 source ep7/8 | sampled raw cameras 480×640 @ 30 fps | 224×288 @ 10 fps | 44–49×14×18×48 BF16 | stride 3 over source ranges | Geometry confirmed; transform method unknown | Confirmed/Unknown |
| RoboCOIN | high | scoop_coffee_beans source ep345/347 | sampled high 720×1280 @ 30 fps | 224×288 @ 15 fps | 83–92×14×18×48 BF16 | stride 2 | Direct bilinear resize confirmed on ep345 by 365-frame VAE reconstruction | Confirmed |
| RoboCOIN | left/right wrist | scoop_coffee_beans source ep1/8 | sampled wrists 480×640 @ 30 fps | 224×288 @ 15 fps | 96–110×14×18×48 BF16 | stride 2 | Geometry confirmed; transform method unknown | Confirmed/Unknown |
| RoboMIND | camera top | flowers source ep160/167 | sampled raw 480×640 @ 30 fps | 224×288 @ 15 fps | 25–30×14×18×48 BF16 | stride 2 | Geometry confirmed; transform method unknown | Confirmed/Unknown |
| InternData-A1 | head | boxed_beverage source ep38/81 | sampled head 360×640 @ 30 fps | 192×352 @ 10 fps | 16–17×12×22×48 BF16 | stride 3 | Geometry confirmed; transform method unknown | Confirmed/Unknown |
| InternData-A1 | left/right hand | boxed_beverage source ep38/81 | sampled hands 480×640 @ 30 fps | 192×352 @ 10 fps | 16–17×12×22×48 BF16 | stride 3 | Geometry confirmed; transform method unknown | Confirmed/Unknown |
| OXE / Bridge | image | part_7 source ep264/271 | sampled raw 480×640 @ 5 fps | 224×288 @ 5 fps | 9–12×14×18×48 BF16 | stride 1 | Geometry confirmed; transform method unknown | Confirmed/Unknown |

## Candidate SO-101 policies — unresolved

| Candidate | Benefit | Compatibility risk | Status |
|---|---|---|---|
| Direct resize 640×480 → 256×256 | Matches Demo canvas and released runtime transform style | Strong 4:3→1:1 geometric distortion; may alter gripper/object geometry | Pending |
| Aspect-preserving center crop → 256×256 | No geometric distortion; matches Demo canvas | Removes horizontal context and may crop task-relevant content | Pending |
| Direct resize 640×480 → 224×288 | Matches RoboTwin native source geometry, encoded canvas, released online transform and reconstructed offline transform | Small 4:3→9:7 geometric distortion remains, but it is pretrained behavior | **Recommended** |
| Aspect-preserving crop → 224×288 | Preserves geometry and Robotwin canvas | Crops vertical content from 4:3 source; offline precedent unconfirmed | Pending |
| Letterbox/pad → selected canvas | Preserves full field of view and geometry | Padding may be out-of-distribution if absent in pretraining | Pending |
| Another source-specific geometry | Could match the most relevant pretrained source | Requires evidence and identical train/validation/online implementation | Pending |

## Decision criteria

A recommendation requires:

1. Artifact-confirmed encoded geometry for RoboTwin and all five external sources.
2. Source- or metadata-confirmed transform type; latent dimensions alone are insufficient.
3. Camera-layout compatibility with two SO-101 streams concatenated by Zero-WAM.
4. A field-of-view audit on representative front and wrist frames.
5. Identical preprocessing for training, validation loss and closed-loop inference.
6. Explicit accounting of pretrained spatial-distribution mismatch.

## Current recommendation

Use a direct bilinear resize from each SO-101 640×480 camera frame to **288×224
(W×H)**, then encode each camera separately to a 18×14 latent and concatenate
camera latents using the existing Zero-WAM camera layout.

Evidence:

1. SO-101 and sampled RoboTwin cameras share native 640×480 geometry.
2. Released RoboTwin robot latents record 224×288 and 14×18.
3. The released online encoder directly bilinear-resizes to configured H×W.
4. Released-VAE reconstruction of 33 exact RoboTwin source frames decisively
   identifies direct resize over center crop or letterbox.
5. Released-VAE reconstruction of 365 RoboCOIN frames confirms direct resize
   even for a large 16:9→9:7 aspect-ratio change.
6. Four of five sampled external pretraining sources also use 224×288.

Compatibility risk is lower than 256×256 because 224×288 is the dominant
HumanGen robot canvas and exact RoboTwin canvas. It intentionally preserves the
small geometric distortion learned during RoboTwin preprocessing rather than
introducing unseen crop boundaries or padding. InternData-A1 demonstrates that
the model can accept another latent geometry, but it does not make 192×352 the
appropriate choice for 4:3 SO-101 cameras.

## Final deliverables

- [ ] Completed cross-dataset camera/resolution/latent/preprocessing table.
- [ ] Exact local source line references and public source URLs.
- [ ] Confirmed/inferred/unknown labels on every conclusion.
- [ ] Minimal-download ledger and inaccessible-artifact report.
- [x] Representative 33-frame decoded-latent comparison.
- [x] SO-101 recommendation: direct bilinear resize to 224×288.
- [x] Compatibility-risk analysis.
- [ ] Update the pending training config only after separate user approval; not as part of this investigation.
