"""One H3 session that turns a series of prompts and reference images into MP4s."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from reactor_sdk import Reactor

logger = logging.getLogger(__name__)

MODEL = "reactor/h3-reference-to-video-turbo-realtime"
ASPECTS = ("16:9", "1:1", "9:16", "4:3")
Aspect = Literal["16:9", "1:1", "9:16", "4:3"]

DURATION_MIN = 5.0
DURATION_MAX = 15.084
DEFAULT_GENERATION_CAPACITY = 20
GENERATE_WAIT_SECONDS = 180.0
PLAY_GRACE_SECONDS = 10.0
TRACK_QUIET_SECONDS = 0.2
TRACK_DRAIN_SECONDS = 2.0

OUTPUT_NAME = "video.mp4"
CONTACT_NAME = "contact.png"
FRAME_RATE = 24
CONTACT_HZ = 2
CONTACT_STRIDE = FRAME_RATE // CONTACT_HZ
CONTACT_CELL = 192
CONTACT_COLS = 5
FRAME_NAME = "frame.png"
META_NAME = "meta.json"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class BatchError(RuntimeError):
    """A clip failed or a command was refused. Earlier MP4s stay on disk."""

    def __init__(self, index: int, reason: str):
        self.index = index
        self.reason = reason
        super().__init__(f"clip {index} failed: {reason}")


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    reference_image: str | Path | bytes
    duration: float
    seed: int | None = None
    output_path: str | Path | None = None


def load_batch(folder: str | Path, *, force: bool = False) -> list[VideoRequest]:
    """Load clip directories under ``folder``.

    Each clip has ``meta.json`` with ``prompt``, ``duration``, optional ``seed``,
    and ``frame`` (a file name). Stills live in a ``frames`` directory beside the
    task folder, or inside the batch folder. ``frames`` itself is not a task.
    Directories that already contain ``video.mp4`` are skipped unless ``force``.
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    frames = _frames_dir(root)
    clip_dirs = _clip_dirs(root)
    if not clip_dirs:
        raise ValueError(f"no clip directories under {root}")

    requests: list[VideoRequest] = []
    for clip_dir in clip_dirs:
        output = clip_dir / OUTPUT_NAME
        if output.is_file() and not force:
            logger.info("skip %s (%s exists)", clip_dir, OUTPUT_NAME)
            continue
        meta = _read_meta(clip_dir / META_NAME)
        frame = frames / meta["frame"]
        if not frame.is_file():
            raise ValueError(f"{clip_dir}: frame not found: {frame}")
        requests.append(
            VideoRequest(
                prompt=meta["prompt"],
                reference_image=frame,
                duration=float(meta["duration"]),
                seed=meta.get("seed"),
                output_path=output,
            )
        )
    return requests


async def generate_videos(
    items: Sequence[VideoRequest],
    *,
    output_dir: str | Path | None = None,
    aspect: Aspect = "16:9",
    api_key: str | None = None,
) -> list[Path]:
    """Open one H3 session and return one MP4 path per item, in input order."""
    batch = list(items)
    if not batch:
        raise ValueError("items is empty")
    if aspect not in ASPECTS:
        raise ValueError(f"aspect must be one of {', '.join(ASPECTS)}")
    for index, item in enumerate(batch):
        _validate_item(item, index)

    key = api_key or os.environ.get("REACTOR_API_KEY")
    if not key:
        raise RuntimeError("REACTOR_API_KEY is not set")
    _require_ffmpeg()

    outputs = [_resolve_output(item, index, output_dir) for index, item in enumerate(batch)]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    messages: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_message(message: Any) -> None:
        loop.call_soon_threadsafe(messages.put_nowait, message)

    async with Reactor(MODEL, api_key=key) as reactor:
        reactor.on("message", on_message)
        frames = _subscribe(reactor, recorder := _LiveRecorder())
        try:
            await reactor.connect()
            await _resume_outputs(reactor)
            _raise_if_error(await reactor.send_command("set_canvas", {"aspect": aspect}))
            state = _body(await reactor.send_command("get_state", {}))
            capacity = int(state.get("generation_capacity") or DEFAULT_GENERATION_CAPACITY)
            refs = await _upload_unique(reactor, batch)
            return await _play_all(
                reactor, batch, refs, outputs, messages, capacity, frames, recorder
            )
        finally:
            reactor.off("message", on_message)
            await reactor.disconnect()


async def generate_video(
    prompt: str,
    reference_image: str | Path | bytes,
    duration: float,
    *,
    seed: int | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    aspect: Aspect = "16:9",
    api_key: str | None = None,
) -> Path:
    """Generate a single clip. Wrapper around :func:`generate_videos`."""
    paths = await generate_videos(
        [
            VideoRequest(
                prompt=prompt,
                reference_image=reference_image,
                duration=duration,
                seed=seed,
                output_path=output_path,
            )
        ],
        output_dir=output_dir,
        aspect=aspect,
        api_key=api_key,
    )
    return paths[0]


def generate_videos_sync(
    items: Sequence[VideoRequest],
    *,
    output_dir: str | Path | None = None,
    aspect: Aspect = "16:9",
    api_key: str | None = None,
) -> list[Path]:
    return asyncio.run(
        generate_videos(items, output_dir=output_dir, aspect=aspect, api_key=api_key)
    )


def generate_video_sync(
    prompt: str,
    reference_image: str | Path | bytes,
    duration: float,
    *,
    seed: int | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    aspect: Aspect = "16:9",
    api_key: str | None = None,
) -> Path:
    return asyncio.run(
        generate_video(
            prompt,
            reference_image,
            duration,
            seed=seed,
            output_path=output_path,
            output_dir=output_dir,
            aspect=aspect,
            api_key=api_key,
        )
    )


async def _play_all(
    reactor: Reactor,
    items: list[VideoRequest],
    refs: list[Any],
    outputs: list[Path],
    messages: asyncio.Queue[Any],
    capacity: int,
    frames: dict[str, int],
    recorder: "_LiveRecorder",
) -> list[Path]:
    next_index = 0
    generating = 0
    ready: deque[str] = deque()
    playing: str | None = None
    play_started = 0.0
    index_by_id: dict[str, int] = {}
    seconds_by_id: dict[str, float] = {}
    generated: set[str] = set()
    finished: list[Path | None] = [None] * len(items)

    async def enqueue_next() -> None:
        nonlocal next_index, generating
        index = next_index
        next_index += 1
        payload: dict[str, Any] = {
            "prompt": items[index].prompt,
            "reference_image": refs[index],
            "seconds": items[index].duration,
            "metadata": str(index),
        }
        if items[index].seed is not None:
            payload["seed"] = items[index].seed
        reply = await reactor.send_command("enqueue", payload)
        _raise_if_error(reply, index)
        clip = _clip(_body(reply))
        if not clip.get("clip_id"):
            clip = _take_queued(messages, index)
        if not clip.get("clip_id"):
            raise BatchError(index, "enqueue did not return a clip id")
        _remember(clip, index)
        generating += 1

    def _remember(clip: dict[str, Any], index: int) -> None:
        clip_id = str(clip["clip_id"])
        index_by_id[clip_id] = index
        seconds_by_id[clip_id] = float(clip["seconds"])
        logger.info(
            "queued index=%s clip_id=%s seconds=%s frames=%s",
            index,
            clip_id,
            clip.get("seconds"),
            clip.get("frames"),
        )

    async def top_up() -> None:
        while generating < capacity and next_index < len(items):
            await enqueue_next()

    async def maybe_play() -> None:
        nonlocal playing, play_started
        if playing is not None or not ready:
            return
        clip_id = ready.popleft()
        playing = clip_id
        play_started = time.monotonic()
        logger.info(
            "playing index=%s clip_id=%s video_frames=%s audio_frames=%s",
            index_by_id.get(clip_id),
            clip_id,
            frames["main_video"],
            frames["main_audio"],
        )
        await _wait_for_quiet(frames)
        recorder.start(outputs[index_by_id[clip_id]])
        reply = await reactor.send_command("play", {"clip_id": clip_id})
        _raise_if_error(reply, index_by_id.get(clip_id, -1))

    await top_up()
    while any(path is None for path in finished):
        if playing is None:
            timeout = GENERATE_WAIT_SECONDS
        else:
            timeout = max(
                0.1,
                seconds_by_id[playing] + PLAY_GRACE_SECONDS - (time.monotonic() - play_started),
            )
        try:
            message = await asyncio.wait_for(messages.get(), timeout=timeout)
        except asyncio.TimeoutError:
            if playing is None:
                raise BatchError(next_index, "timed out waiting for a clip to generate") from None
            index = index_by_id.get(playing, -1)
            logger.error(
                "no clip_finished for index=%s after %.1fs video_frames=%s audio_frames=%s",
                index,
                time.monotonic() - play_started,
                frames["main_video"],
                frames["main_audio"],
            )
            await reactor.send_command("stop", {})
            raise BatchError(index, "playback did not finish") from None
        kind, body = _normalize(message)
        if kind == "clip_queued":
            clip = _clip(body)
            metadata = str(clip.get("metadata", ""))
            if metadata.isdigit() and clip.get("clip_id"):
                _remember(clip, int(metadata))
            continue
        if kind == "clip_generated":
            clip = _clip(body)
            clip_id = str(clip.get("clip_id", ""))
            if not clip_id or clip_id in generated:
                continue
            generated.add(clip_id)
            logger.info("generated index=%s clip_id=%s", index_by_id.get(clip_id), clip_id)
            generating = max(0, generating - 1)
            ready.append(clip_id)
            await top_up()
            await maybe_play()
            continue
        if kind == "clip_finished":
            clip = _clip(body)
            clip_id = str(clip.get("clip_id", ""))
            index = index_by_id.get(clip_id)
            if index is None or finished[index] is not None:
                continue
            logger.info(
                "clip_finished index=%s video_frames=%s audio_frames=%s",
                index,
                frames["main_video"],
                frames["main_audio"],
            )
            recorder.finish()
            finished[index] = outputs[index]
            if playing == clip_id:
                playing = None
            await maybe_play()
            continue
        if kind == "clip_failed":
            clip = _clip(body)
            index = index_by_id.get(str(clip.get("clip_id", "")), -1)
            raise BatchError(index, str(body.get("reason") or "clip_failed"))
        if kind == "command_error":
            command = body.get("command") or "command"
            reason = body.get("reason") or "refused"
            raise BatchError(-1, f"{command}: {reason}")
        if kind not in {"queue_update", "state_update", "clip_started"}:
            logger.info("message %s", kind)

    return [path for path in finished if path is not None]


async def _wait_for_quiet(counts: dict[str, int]) -> None:
    """Drop frames still arriving from the clip that just finished."""
    started = counts["main_video"]
    deadline = time.monotonic() + TRACK_DRAIN_SECONDS
    while time.monotonic() < deadline:
        seen = counts["main_video"]
        await asyncio.sleep(TRACK_QUIET_SECONDS)
        if counts["main_video"] == seen:
            dropped = counts["main_video"] - started
            if dropped:
                logger.info("dropped %s trailing video frames", dropped)
            return
    logger.info("video track still running after %.1fs", TRACK_DRAIN_SECONDS)


def _subscribe(reactor: Reactor, recorder: "_LiveRecorder") -> dict[str, int]:
    """Consume both output tracks and keep the raw frames for a local recording."""
    reactor.track("main_video").on_raw_frame(recorder.on_video)
    reactor.track("main_audio").on_raw_frame(recorder.on_audio)
    return recorder.counts


class _LiveRecorder:
    """Encode the live tracks with ffmpeg as each frame arrives.

    Callbacks only enqueue. Separate threads feed the two ffmpeg pipes so a
    full video pipe cannot stall the audio callback on the media thread.
    """

    def __init__(self) -> None:
        self.counts = {"main_video": 0, "main_audio": 0}
        self._recording = False
        self._video_q: queue.Queue[bytes | None] = queue.Queue()
        self._audio_q: queue.Queue[bytes | None] = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._audio_fd: int | None = None
        self._width = 0
        self._height = 0
        self._rate = 48000
        self._channels = 1
        self._output: Path | None = None
        self._threads: list[threading.Thread] = []
        self._samples: list[bytes] = []
        self._sample_index = 0

    def start(self, output: Path) -> None:
        self._recording = False
        self._output = output
        self._width = 0
        self._height = 0
        self._error = None
        self._proc = None
        self._audio_fd = None
        self._ready.clear()
        self._video_q = queue.Queue()
        self._audio_q = queue.Queue()
        self._samples = []
        self._sample_index = 0
        self._recording = True
        self._threads = [
            threading.Thread(target=self._write_video, name="humangen-video", daemon=True),
            threading.Thread(target=self._write_audio, name="humangen-audio", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        logger.info("recording %s", output)

    def finish(self) -> None:
        self._recording = False
        self._video_q.put(None)
        self._audio_q.put(None)
        for thread in self._threads:
            thread.join(timeout=60)
        if self._error is not None:
            raise BatchError(-1, f"recording failed: {self._error}")
        proc = self._proc
        if proc is None:
            raise BatchError(-1, "no video frames were recorded")
        if proc.wait(timeout=30) != 0:
            raise BatchError(-1, "ffmpeg failed while writing the clip")
        logger.info("wrote %s", self._output)
        if self._output is not None and self._samples:
            sheet = self._output.with_name(CONTACT_NAME)
            _write_contact_sheet(self._samples, self._width, self._height, sheet)
            logger.info("wrote %s", sheet)

    def on_video(
        self,
        bgra: bytes,
        width: int,
        height: int,
        _frame_id: int,
        _timestamp_us: int,
        _user_data: bytes,
    ) -> None:
        self.counts["main_video"] += 1
        if not self._recording:
            return
        if not self._width:
            self._width = width
            self._height = height
        self._video_q.put(bgra)

    def on_audio(self, pcm: bytes, _num_samples: int, sample_rate: int, num_channels: int) -> None:
        self.counts["main_audio"] += 1
        if not self._recording:
            return
        self._rate = sample_rate
        self._channels = num_channels
        self._audio_q.put(pcm)

    def _write_video(self) -> None:
        try:
            while True:
                frame = self._video_q.get()
                if frame is None:
                    if self._proc is not None and self._proc.stdin is not None:
                        self._proc.stdin.close()
                    self._ready.set()
                    return
                if self._proc is None:
                    self._open_ffmpeg()
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(frame)
                if self._sample_index % CONTACT_STRIDE == 0:
                    self._samples.append(frame)
                self._sample_index += 1
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def _write_audio(self) -> None:
        try:
            if not self._ready.wait(timeout=30):
                raise TimeoutError("ffmpeg did not start")
            audio_fd = self._audio_fd
            if audio_fd is None:
                return
            while True:
                pcm = self._audio_q.get()
                if pcm is None:
                    os.close(audio_fd)
                    return
                _write_all(audio_fd, pcm)
        except BaseException as exc:
            self._error = exc

    def _open_ffmpeg(self) -> None:
        if self._output is None:
            raise BatchError(-1, "recording was not started")
        audio_r, audio_w = os.pipe()
        self._audio_fd = audio_w
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgra",
                "-s",
                f"{self._width}x{self._height}",
                "-r",
                "24",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-ar",
                str(self._rate),
                "-ac",
                str(self._channels),
                "-i",
                f"/dev/fd/{audio_r}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(self._output),
            ],
            stdin=subprocess.PIPE,
            pass_fds=(audio_r,),
        )
        os.close(audio_r)
        self._ready.set()


def _contact_grid(count: int) -> tuple[int, int]:
    """Column count from 4 to 6 that leaves the fewest empty cells."""
    if count <= 6:
        return count, 1
    best_cols = min(CONTACT_COLS, count)
    best_pad = math.ceil(count / best_cols) * best_cols - count
    for cols in range(4, min(6, count) + 1):
        pad = math.ceil(count / cols) * cols - count
        if pad < best_pad or (pad == best_pad and cols > best_cols):
            best_cols, best_pad = cols, pad
    return best_cols, math.ceil(count / best_cols)


def _write_contact_sheet(samples: list[bytes], width: int, height: int, output: Path) -> None:
    """Tile raw BGRA frames, sampled at 2 Hz, into one PNG."""
    count = len(samples)
    cols, rows = _contact_grid(count)
    blank = b"\x00\x00\x00\xff" * (width * height)
    payload = b"".join(samples) + blank * (rows * cols - count)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgra",
            "-s",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-vf",
            f"scale={CONTACT_CELL}:{CONTACT_CELL},tile={cols}x{rows}:padding=4:margin=4:color=white",
            "-frames:v",
            "1",
            str(output),
        ],
        input=payload,
        check=True,
        capture_output=True,
    )


def _write_all(fd: int, data: bytes | bytearray) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


async def _resume_outputs(reactor: Reactor) -> None:
    for name in ("main_video", "main_audio"):
        if name in reactor.paused_tracks:
            await reactor.track(name).resume()


def _take_queued(messages: asyncio.Queue[Any], index: int) -> dict[str, Any]:
    """Pull a matching ``clip_queued`` already sitting on the event queue."""
    buffered: list[Any] = []
    while True:
        try:
            buffered.append(messages.get_nowait())
        except asyncio.QueueEmpty:
            break
    found: dict[str, Any] = {}
    for message in buffered:
        kind, body = _normalize(message)
        clip = _clip(body)
        if (
            not found
            and kind == "clip_queued"
            and str(clip.get("metadata", "")) == str(index)
            and clip.get("clip_id")
        ):
            found = clip
            continue
        messages.put_nowait(message)
    return found


async def _upload_unique(reactor: Reactor, items: list[VideoRequest]) -> list[Any]:
    cache: dict[str, Any] = {}
    refs: list[Any] = []
    for item in items:
        key = _image_key(item.reference_image)
        if key not in cache:
            cache[key] = await _upload_image(reactor, item.reference_image)
        refs.append(cache[key])
    return refs


async def _upload_image(reactor: Reactor, image: str | Path | bytes) -> Any:
    if isinstance(image, bytes):
        mime, name = _mime_from_bytes(image)
        return await reactor.upload_file(image, name=name, mime_type=mime)
    return await reactor.upload_file(Path(image))


def _image_key(image: str | Path | bytes) -> str:
    if isinstance(image, bytes):
        return hashlib.sha256(image).hexdigest()
    return str(Path(image).resolve())


def _mime_from_bytes(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "reference.jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "reference.png"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp", "reference.webp"
    raise ValueError("reference image bytes must be JPEG, PNG, or WebP")


def _validate_item(item: VideoRequest, index: int) -> None:
    if not str(item.prompt).strip():
        raise ValueError(f"clip {index}: prompt is empty")
    if item.reference_image is None or item.reference_image == b"" or item.reference_image == "":
        raise ValueError(f"clip {index}: reference image is missing")
    try:
        duration = float(item.duration)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"clip {index}: duration must be a number") from exc
    if duration < DURATION_MIN or duration > DURATION_MAX:
        raise ValueError(
            f"clip {index}: duration {duration} is outside {DURATION_MIN}–{DURATION_MAX} seconds"
        )
    if item.seed is not None and item.seed < 0:
        raise ValueError(f"clip {index}: seed must be nonnegative")


def _resolve_output(item: VideoRequest, index: int, output_dir: str | Path | None) -> Path:
    if item.output_path is not None:
        return Path(item.output_path)
    directory = Path(output_dir) if output_dir is not None else Path("outputs")
    return directory / f"clip_{index:03d}.mp4"


def _clip_dirs(root: Path) -> list[Path]:
    if (root / META_NAME).is_file():
        return [root]
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "frames"
    )


def _frames_dir(root: Path) -> Path:
    if (root / META_NAME).is_file():
        root = root.parent
    for candidate in (root / "frames", root.parent / "frames"):
        if candidate.is_dir():
            return candidate
    raise ValueError(f"no frames directory next to {root}")


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {path}")
    meta = json.loads(path.read_text())
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: expected an object")
    prompt = meta.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path}: prompt is required")
    if "duration" not in meta:
        raise ValueError(f"{path}: duration is required")
    duration = float(meta["duration"])
    seed = meta.get("seed")
    if seed is not None:
        seed = int(seed)
    frame = meta.get("frame")
    if not isinstance(frame, str) or not frame.strip() or Path(frame).name != frame:
        raise ValueError(f"{path}: frame must be a file name in the frames directory")
    return {"prompt": prompt, "duration": duration, "seed": seed, "frame": frame}


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required on PATH")


def _normalize(message: Any) -> tuple[str | None, dict[str, Any]]:
    if not isinstance(message, dict):
        return None, {}
    kind = message.get("type")
    data = message.get("data")
    if isinstance(data, dict):
        return kind if isinstance(kind, str) else None, data
    body = {key: value for key, value in message.items() if key != "type"}
    return kind if isinstance(kind, str) else None, body


def _body(reply: Any) -> dict[str, Any]:
    _kind, body = _normalize(reply)
    return body


def _clip(body: dict[str, Any]) -> dict[str, Any]:
    clip = body.get("clip")
    if isinstance(clip, dict):
        return clip
    if "clip_id" in body:
        return body
    return {}


def _raise_if_error(reply: Any, index: int = -1) -> None:
    kind, body = _normalize(reply)
    if kind == "command_error":
        command = body.get("command") or "command"
        reason = body.get("reason") or "refused"
        raise BatchError(index, f"{command}: {reason}")
