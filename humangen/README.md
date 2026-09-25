# humangen

Generate reference-guided clips with Reactor [H3 Reference Turbo Realtime](https://docs.reactor.inc/model-api-reference/h3-reference-to-video-turbo-realtime/overview). One command opens one session, queues every task in a folder, and writes an MP4 plus a contact sheet into each task directory.

Set `REACTOR_API_KEY` in the environment or in `.env`. `ffmpeg` must be on `PATH`.

```sh
uv sync
uv run python -m humangen example/tasks --aspect 1:1
uv run python -m humangen example/nexus --aspect 4:3
```

The session is billed while it is connected. A folder is one session: clips build while the current one plays, and the client disconnects after the last file is written. A task that already has `video.mp4` is skipped. `--force` regenerates it.

## Reference still

A robot view is turned into the reference still in Gemini chat before it goes in `frames/`. The prompt is:

```text
Convert this image into a photorealistic first-person human view. Remove the robot and add realistic human forearms and hands. Preserve the objects, background, and scene layout. Show both hands in a natural pre-action ready pose. No robot parts, extra limbs, duplicated objects, text, or overlays.
```

## Layout

Stills live once in a `frames` directory next to the task folder, or inside the batch folder. Each task directory holds `meta.json` and, after a run, `video.mp4` and `contact.png`.

```text
example/
  frames/
    human_in_libero.jpg
    human_top_down_so101_nexus.jpeg
  tasks/
    01_open_top_drawer/
      meta.json
      video.mp4
      contact.png
  nexus/
    01_biscuit_in_bin/
      meta.json
      video.mp4
      contact.png
```

`meta.json`:

```json
{
  "prompt": "The person in Picture 1 ...",
  "duration": 5,
  "seed": 0,
  "frame": "human_in_libero.jpg"
}
```

`frame` is a file name only, not a path. JPEG, PNG, and WebP are accepted. `duration` is seconds, from 5.0 to 15.084. `seed` is optional.

`contact.png` is a 2 Hz sample of the clip, scaled and tiled, so the take can be checked without opening the video.

## Prompt

Name the still `Picture 1`. The [prompt guide](https://docs.reactor.inc/model-api-reference/h3-reference-to-video-turbo-realtime/prompt-guide) treats the reference as appearance, not as a locked first frame. One action per clip. Say which hand moves, that the other hand stays down, that the camera stays locked, and that the scene holds still once the action is done. Put the sound in the same text, and say when there is no dialogue and no music.

A requested duration of 5 is aligned up to 5.167 seconds. End the last beat at that time.

## One clip

```sh
uv run python -m humangen \
  --prompt "The person in Picture 1 ..." \
  --image example/frames/human_in_libero.jpg \
  --duration 5 \
  --seed 0 \
  --aspect 1:1 \
  --output out.mp4
```

`--prompt`, `--image`, and `--duration` are required together. Do not pass them with a batch directory.

## Canvas

One aspect per session, chosen to match the still. The default is `16:9`.

| `--aspect` | Resolution |
| --- | --- |
| `16:9` | 1344 × 768 |
| `1:1` | 768 × 768 |
| `9:16` | 768 × 1344 |
| `4:3` | 1024 × 768 |

Output is 24 fps H.264 with 48 kHz mono AAC. The square kitchen still uses `1:1`. The top-down table still uses `4:3`.

## From Python

`generate_videos` takes a list of `VideoRequest` and returns one MP4 path per item, in order. `generate_video` is the one-clip wrapper. Both have `_sync` variants. `load_batch` reads a folder the same way the command does.

```python
from humangen import VideoRequest, generate_videos_sync, load_batch

generate_videos_sync(load_batch("example/nexus"), aspect="4:3")
```

A failed clip raises `BatchError` with the task index. MP4s already written stay on disk.
