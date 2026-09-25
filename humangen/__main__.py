"""Command line for one clip or a folder of clip directories."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from humangen.generate import generate_video_sync, generate_videos_sync, load_batch


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate H3 reference videos as MP4s.")
    parser.add_argument("batch", nargs="?", type=Path, help="Clip directory or a folder of them")
    parser.add_argument("--aspect", default="16:9", choices=["16:9", "1:1", "9:16", "4:3"])
    parser.add_argument("--force", action="store_true", help="Regenerate clips that already have video.mp4")
    parser.add_argument("--prompt")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    single = any(value is not None for value in (args.prompt, args.image, args.duration, args.output))
    if args.batch is not None and single:
        parser.error("pass a batch directory or a single --prompt/--image/--duration, not both")
    if args.batch is not None:
        requests = load_batch(args.batch, force=args.force)
        if not requests:
            print(f"nothing to generate under {args.batch}")
            return 0
        paths = generate_videos_sync(requests, aspect=args.aspect)
    else:
        if not args.prompt or args.image is None or args.duration is None:
            parser.error("a batch directory, or --prompt, --image, and --duration, is required")
        paths = [
            generate_video_sync(
                args.prompt,
                args.image,
                args.duration,
                seed=args.seed,
                output_path=args.output,
                aspect=args.aspect,
            )
        ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
