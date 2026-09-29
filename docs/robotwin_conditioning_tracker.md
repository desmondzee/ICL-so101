# RoboTwin conditioning discrimination tracker

## Result: complete, 11 policy episodes

The post-trained checkpoint can complete **opposite goals in the same scene using opposite video demonstrations**: green-on-red and red-on-green both succeeded with identical neutral text. Text-only reversal did not succeed. This supports video-conditioned goal selection for this paired scene; it does not establish general text-following or benchmark-level reliability.

| Conditioning | Episodes | Green-on-red ever | Red-on-green ever | Requested goal success |
| --- | ---: | ---: | ---: | ---: |
| Text: green on red | 3 | 2 | 0 | 2/3 |
| Text: red on green | 3 | 2 | 0 | 0/3 |
| Text: “Stack the blocks.” | 3 | 2 | 0 | Unspecified order; 2/3 stock order |
| Green-on-red video + neutral text | 1 | 1 | 0 | 1/1 |
| Red-on-green video + neutral text | 1 | 0 | 1 | 1/1 |

“Ever” requires native XYZ stacking tolerances and both grippers open. Reverse-text seed100002 briefly achieved green-on-red, then dismantled it; only seed100000 ended in the wrong stack. All requested video successes also held at the final state. Native metrics agree with the assigned-goal scorer.

## Protocol and validation

Native `stack_blocks_two`, `demo_clean`, Aloha embodiment/cameras/pose actions, full 800-control horizon. Fixed posttrain revision `07ee865f175d9474a5653e9147698390e47b6143`; exact checkpoint load checked. Text conditions share expert-valid seeds100000–100002; video conditions share seed100000. Initial red/green positions match **exactly** between conditions and expert gates. Native instruction reference `/workspace/Robotwin/description/task_instruction/stack_blocks_two.json` always describes green-on-red.

Bidirectional expert gate passed **6/6** (both orders on all three seeds), so reverse order is physically feasible. Robot expert clips start in the same scene, capture244 frames each, and uniformly resample the full successful rollout to96 frames at12fps,320×240. Both demonstrations and both resulting policy videos were visually checked. They are robot demonstrations, not human footage.

Code: `zero_wam/robotwin_conditioning.py:run_condition,stack_outcomes,scripted_gate`; `zero_wam/modal_app.py:conditioning_test,ZeroWAM.step`; `zero_wam/modal_client.py:run`. Prompt override reaches upstream `eval_policy_client_openpi.py:582,615` and inference. Uploaded MP4 bytes are materialized on the Modal GPU host during reset. Text experiments disable the upstream unconditional ICL-video lookup; methods are restored on exit. Syntax and scorer checks passed.

## Hypotheses

| Hypothesis | Assessment / evidence |
| --- | --- |
| H0: scene alone fixes the stock order | Consistent with text results; insufficient to explain opposite successful video goals. Neutral text and stock text both succeed on seeds100001/100002. |
| H1: text selects either stack order | Not validated: reverse0/3, stock2/3; reverse trials achieved the wrong stock order in2/3. Reverse wording is outside the stock instruction templates, so this does not prove text is universally ignored. |
| H2: reverse order is physically feasible | Confirmed by expert6/6 and reverse-video policy1/1. |
| H3: video selects stack order | Supported by the opposite-goal pair with identical scene/text: each clip produces its requested order. Limited to one pair; diffusion noise was not explicitly paired. |

These trials demonstrate bidirectional video-conditioned control, not a statistical causal estimate. No pretrain comparison, human-video transfer, multi-task benchmark, or SO-101 success is claimed.

## Artifacts

- Summary: `outputs/zero_wam/robotwin_conditioning/summary.json`.
- Expert feasibility: `outputs/zero_wam/robotwin_conditioning/scripted_gate/scripted_gate.json`.
- Demonstrations: `outputs/zero_wam/robotwin_conditioning/scripted_videos/seed100000_{green_on_red,red_on_green}.mp4`.
- Text reports: `outputs/zero_wam/robotwin_conditioning_text/{green_on_red,red_on_green,neutral}/conditioning_results.json`.
- Video reports: `outputs/zero_wam/robotwin_conditioning_video/{green_on_red,red_on_green}/conditioning_results.json`.
- Green-on-red policy video: `outputs/zero_wam/robotwin_conditioning_video/green_on_red/stseed-100000/visualization/stack_blocks_two/0_Stack_the_blocks._True.mp4`.
- Red-on-green policy video: `outputs/zero_wam/robotwin_conditioning_video/red_on_green/stseed-100000/visualization/stack_blocks_two/0_Stack_the_blocks._True.mp4`.
- Each condition also saves `protocol.json`, native metrics, MP4s and contact sheets.

## Execution exceptions (excluded from policy outcomes)

Initial text attempt failed before reset because the upstream ICL map lacks this task; corrected by disabling lookup for text-only. Early parallel launches failed during local Curobo initialization with CUDA OOM; two simulators fit after expert capture ended. Empty prompt failed before inference (`Robotwin ICL inference requires a prompt`); replaced with valid neutral text. These failed attempts are preserved separately and excluded.

Video and neutral cohorts exited normally. Original text launcher was intentionally interrupted **after all six explicit-text episodes and metrics were saved**, preventing a duplicate ablation block. All eleven planned valid episodes are complete; no pending trial is counted as failure. Logs: `/tmp/robotwin_conditioning_{text,video,neutral}.log`.

## Reproduction

Use the existing RoboTwin Python environment with `ROBOTWIN_ROOT`, repo/upstream/Robotwin `PYTHONPATH`, and existing Modal credentials. Text entry: `python -m zero_wam.modal_cli run zero_wam/modal_app.py::conditioning_test --test-num 3 --save-root ABS_ROOT` (default conditions: green_on_red,red_on_green,neutral). Video entry: same command with `--conditions green_on_red,red_on_green --test-num 1 --video-root ABS_SCRIPTED_VIDEO_ROOT`; these clips correspond to `--seed 0` / actual seed100000. Use separate save roots for text and video.
