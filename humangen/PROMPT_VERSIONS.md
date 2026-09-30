# Whole-task prompt versions

`humangen.context.task_prompt(..., version=...)` selects the template.
Every demo metadata record stores the version, exact rendered text, SHA-256,
template input, requested duration, effective duration, seed and source frames.

| Version | Change |
| --- | --- |
| v1 | Original plain prose template, before motion constraints. Existing rendered prompts survive in demo archives. |
| v2_motion | Added direct task motion, appearance continuity and final-second stillness. Available through `version="v2_motion"`. |
| v3_fl2va | Explicit Picture 1/Picture 2 time alignment; `[Shot 1]` and `integrated_multimodal_description`; separate `overall_soundscape`; explicit convergence to Picture 2. Music section omitted at user request. Motion constraints retained to compare format changes. |

The 5-second validation requests align to 5.167 seconds (124 frames at 24 fps);
pass `effective_duration=124 / 24` for their endpoint instruction.
The 5-second duration change is a run setting, rather than a template version.

Source: [MiniMax H3 base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
This guide covers base H3; the demos use Reactor's distilled FastH3.

Before regeneration, archive `video.mp4`, `contact.png` and `meta.json` under
`versions/<prompt_version>/`. Keep archived files intact. Latest outputs remain
at the episode root for browsing.

## v4_inventory experiment

Opt in with `version="v4_inventory"`; the default remains v3 pending comparison.
Retains the official FL2VA structure, removes the long generic motion block,
and explicitly preserves the initial object inventory and any visible bystanders.
Only the acting hand and forearm is introduced, performs the action and retreats.
Uses the effective duration for the final-second hold. First tested on puzzle
`episode_000` at five seconds, seed zero, using unchanged endpoint images.

Research context (anecdotal, not a verified fix):

- [MiniMax H3 issue 81](https://github.com/MiniMax-AI/MiniMax-H3/issues/81):
  reporter says dialogue content manifests visually and breaks first-frame subject consistency.
- [H3 problems discussion](https://www.reddit.com/r/StableDiffusion/comments/1vg65hr/what_problems_have_you_found_with_minimax_h3/):
  reports of unwanted interpretation, endpoint freezes and excess-duration invention.
- Reactor's current hosted API documents starting/ending frames, while its public
  `infinite-livestream` checkout at `c2326e49ca29149d67fe3b97fb9908d31bb6716f`
  is an older text-only implementation. It cannot establish the hosted conditioning mechanism.

Endpoint audit artifacts are in `outputs/robot_removal/demo_experiments/conditioning_audit/`.
The v3 recordings contain 105–121 of the expected 124 frames. The client calls
`recorder.finish()` immediately on the control-channel `clip_finished` event,
then deliberately drops trailing media frames. Thus recorded last frames are
not reliable evidence that the service ignored ending-image conditioning.

### v4 result

Puzzle episode 000 still introduces extra hands and red/pink pieces, although
the bowl of spare pieces visible in v3 is absent. This is a failure of inventory
preservation. The first recording was interrupted (42 frames, broken pipe);
`interrupted_video.mp4` preserves it. The successful retry records 122 frames
and decodes successfully. Neither run changes the default version.

## v5_exclusion experiment

Opt in with `version="v5_exclusion"`. Same as v4, with one appended sentence:
“No other hands or objects enter the scene at any time.” This is an ordinary
text instruction, not a dedicated negative-conditioning field. Test uses the
same puzzle episode, seed, duration and input frames.

### v5 result and stopping point

Episode 000 still introduces extra hands and a red/pink piece despite the
explicit exclusion. Recorded 122 frames; MP4 decodes successfully. The visible
inventory failure remains. User requested stopping after this final test;
no further generations were run. Latest episode-root files remain v3; v4 and
v5 experiments are preserved separately under `versions/` for comparison.

## v6_one_person experiment

Same as v5, with explicit actor identity and entry direction: exactly one person
uses only their right hand and forearm; their other hand and body remain outside
the frame; visible bystanders remain still and do not participate. The actor
stands beyond the selected entry edge, and the same forearm and hand enter and
withdraw through that edge. Tested on puzzle episode 000 from the top edge,
with unchanged source frames, five-second duration and seed zero.

### v6 result

The episode 000 contact sheet still shows multiple participating hands and
additional yellow and star-shaped pieces, despite the explicit one-person and
top-edge entry constraints. The acting arm also enters from the left side in
several sampled frames. Recorded 122 frames; MP4 decodes successfully. This
single test does not support promoting v6 over v5. Episode-root files remain
v5; the v6 test is preserved under `versions/v6_one_person/`.

## v7_minimal experiment

User-supplied prompt, used verbatim without a template:

> a man moves with one hand the green piece to the correct hole

Tested on puzzle episode 000 with unchanged frames, seed zero and five-second
request. Contact sheet shows multiple participating hands and duplicate green
pieces during the action. The correct piece is in the recess at the end, but
inventory and actor continuity still fail. Recorded 122 frames; MP4 decodes.
Saved separately under `versions/v7_minimal/`; episode-root files remain v5.

## v8_existing_piece experiment

Prompt without a template:

> the green piece on the table is picked up by one hand and moved into the correct hole

Same episode 000 frames, seed zero and five-second request. Contact sheet shows
multiple hands and duplicate green pieces; a foreground close-up also introduces
an additional yellow labelled shape. Existing-object wording did not resolve
the additions in this test. Recorded 122 frames; MP4 decodes successfully.
Saved under `versions/v8_existing_piece/`; episode-root files remain v5.

## Bulk run selected by the user

The bulk run uses the unchanged `v5_exclusion` template, five-second requests,
seed zero, and both `result_native.png` endpoint images. Its scope is all 1,960
matched episodes across 28 tasks. Fifteen pilot episodes retain their exact
rendered prompts; the remaining episodes use their full `pair.json` annotations.
All videos are regenerated after the transport boundary repair described below.
The manifest and template snapshot are saved in
`outputs/robot_removal/demos/_run/`.

Earlier experiments and version directories have been moved to
`outputs/robot_removal/demo_experiments/`, preserving their task/episode paths.
Historical `versions/` paths above describe their locations at the time of each
experiment. Current demo episode roots contain the selected video, contact sheet,
and metadata; active recordings are staged in `.pending` and published after
frame-count and full decode checks.

The recorder now waits for expected media frames after `clip_finished`, then
waits for the media track to become quiet before starting the next clip. Generation
can run ahead, while playback and recording remain serial. No clip-continuation
fields are sent. These recording changes do not change the prompt template.

The current API also exposes `set_flush_on_clip_end`: disabling it holds the
last frame instead of flushing the stream at clip end. Initial bulk recordings
with the default flush included truncated tails and black final frames, even
in some 124-frame captures. Those recordings were archived under each episode's
`recording_before_hold/` directory in `demo_experiments`. The attempt was stopped.
The resumed full batch explicitly disables both flush and autoplay. It uploads
each clip's endpoints as that clip enters the generation queue, keeping exact
episode-to-frame mapping while overlapping later uploads with playback. Its state
confirms both settings are false. Publication requires at least 120 frames (five seconds
at 24 fps), and the final audit checks for a black tail when the supplied ending
image is visible. Prompt text and conditioning images remain unchanged. The first
seven corrected recordings decode successfully (five with 124 frames and two
with 123); the sampled first/last comparisons preserve their own input scenes and
no longer end in black. Uploads retry transient network errors, enqueue timeouts
recover from the authoritative queue before resending, and the batch resumes
after session errors without regenerating published clips.

### Recording revision v2_hold_apad

The prompt remains `v5_exclusion`. A subsequent check found that FFmpeg's
`-shortest` could trim video to a shorter audio track even when all 124 video
frames had arrived. Padding audio with `-af apad -shortest` preserves the video
length. A recorder integration check preserved 124 frames (5.167 seconds) with
4.8 seconds of audio. Each new capture records received and expected frame counts;
publication requires encoded frames to equal received frames. Of 609 published
captures, 174 full captures were retained and 435 shorter captures were archived
outside demos for regeneration. A three-clip simulation also verified distinct
start/end references, serial recording, and retention of delayed final frames
without frames crossing episode boundaries.

The audio writer permits a broken pipe only during shutdown, when video EOF
causes FFmpeg to stop before trailing audio finishes draining. Encoder exit status
and received-versus-encoded video counts are still checked. Integration tests with
both 4.8-second and 8-second audio preserve all 124 video frames.
