"""Audit the complete demo manifest and assemble contact sheets for review."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def check_tail(video: Path, ending_image: Path) -> None:
    """Reject a transport cut to black when the supplied ending image is visible."""
    with Image.open(ending_image) as image:
        pixels = image.convert('L').resize((32, 32)).tobytes()
    expected_luma = sum(pixels) / len(pixels)
    raw = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-sseof', '-0.15', '-i', str(video),
        '-vf', 'scale=32:32', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1',
    ])
    if len(raw) < 3072:
        raise RuntimeError('could not decode the final video frame')
    tail_luma = sum(raw[-3072:]) / 3072
    if expected_luma > 5 and tail_luma < 2:
        raise RuntimeError('recording ends in black despite a visible supplied ending image')


def check(row: dict, decode: bool) -> dict:
    dest = Path(row['output_dir'])
    result = {'key': row['key'], 'errors': [], 'warnings': []}
    missing = [name for name in ('video.mp4', 'contact.png', 'meta.json') if not (dest / name).is_file()]
    if missing:
        result['errors'].append('missing: ' + ', '.join(missing))
        return result
    try:
        meta = json.loads((dest / 'meta.json').read_text())
        if hashlib.sha256(meta['prompt'].encode()).hexdigest() != row['prompt_sha256']:
            result['errors'].append('rendered prompt differs from manifest')
        if meta.get('prompt_sha256') != row['prompt_sha256']:
            result['errors'].append('prompt hash differs from manifest')
        for key in ('prompt_version', 'seed', 'duration', 'effective_duration', 'aspect',
                    'model', 'starting_frame', 'ending_frame', 'pair_json', 'pair_sha256',
                    'starting_frame_sha256', 'ending_frame_sha256', 'template_source_sha256'):
            if meta.get(key) != row[key]:
                result['errors'].append(f'{key} differs from manifest')
        for path_key, hash_key in (('starting_frame', 'starting_frame_sha256'),
                                   ('ending_frame', 'ending_frame_sha256'),
                                   ('pair_json', 'pair_sha256')):
            if hashlib.sha256(Path(row[path_key]).read_bytes()).hexdigest() != row[hash_key]:
                result['errors'].append(f'{path_key} source changed since preparation')
        unexpected = sorted(p.name for p in dest.iterdir()
                            if p.name not in ('video.mp4', 'contact.png', 'meta.json', '.pending'))
        if unexpected:
            result['errors'].append('unexpected episode artifacts: ' + ', '.join(unexpected))
        with Image.open(dest / 'contact.png') as sheet:
            sheet.verify()
        info = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,width,height,r_frame_rate', '-of', 'json',
            str(dest / 'video.mp4'),
        ]))['streams'][0]
        result['video'] = info
        frames = int(info['nb_frames'])
        if meta.get('recorded_frames') != frames:
            result['errors'].append('recorded_frames metadata differs from encoded video')
        capture = meta.get('capture')
        if capture and capture.get('received_video_frames') != frames:
            result['errors'].append('encoder did not preserve all received video frames')
        if not capture and frames < 124:
            result['errors'].append('legacy capture may have an audio-trimmed video tail')
        if frames < 120:
            result['errors'].append(f'only {frames} frames')
        elif frames != 124:
            result['warnings'].append(f'{frames} frames; request expected 124')
        if info['width'] * 3 != info['height'] * 4:
            result['errors'].append('video aspect differs from requested 4:3')
        if info['r_frame_rate'] != '24/1':
            result['errors'].append('video rate differs from requested 24 fps')
        check_tail(dest / 'video.mp4', Path(row['ending_frame']))
        if decode:
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(dest / 'video.mp4'),
                            '-f', 'null', '-'], check=True, capture_output=True)
            result['decoded'] = True
    except Exception as exc:
        result['errors'].append(str(exc))
    return result


def review_sheets(rows: list[dict], run: Path) -> list[dict]:
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['key'].split('/')[0]].append(row)
    review = run / 'review'
    review.mkdir(exist_ok=True)
    selected = []
    for task, episodes in grouped.items():
        chosen = [episodes[i] for i in sorted({0, len(episodes) // 2, len(episodes) - 1})]
        images = []
        keys = []
        for row in chosen:
            path = Path(row['output_dir']) / 'contact.png'
            if not path.is_file():
                continue
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
            keys.append(row['key'])
        if not images:
            continue
        canvas = Image.new('RGB', (max(i.width for i in images), sum(i.height + 28 for i in images)), 'white')
        draw = ImageDraw.Draw(canvas)
        y = 0
        for key, image in zip(keys, images):
            draw.text((6, y + 6), key, fill='black')
            canvas.paste(image, (0, y + 28))
            y += image.height + 28
        path = review / f'{task}.png'
        canvas.save(path)
        selected.append({'task': task, 'episodes': keys, 'sheet': str(path.resolve())})
    (review / 'index.json').write_text(json.dumps(selected, indent=2) + '\n')
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--decode', action='store_true', help='fully decode every video')
    args = parser.parse_args()
    base = args.root / 'outputs/robot_removal'
    run = base / 'demos/_run'
    rows = [json.loads(line) for line in (run / 'manifest.jsonl').read_text().splitlines()]
    expected = {str(p.parent.relative_to(base / 'v4')) for p in (base / 'v4').glob('*/episode_*/result_native.png')}
    manifest = {row['key'] for row in rows}
    actual = {str(p.parent.relative_to(base / 'demos')) for p in (base / 'demos').glob('*/episode_*/video.mp4')}
    artifact_keys = {str(p.parent.relative_to(base / 'demos'))
                     for name in ('video.mp4', 'contact.png', 'meta.json')
                     for p in (base / 'demos').glob(f'*/episode_*/{name}')}
    endings = {str(p.parent.relative_to(base / 'v4_last'))
               for p in (base / 'v4_last').glob('*/episode_*/result_native.png')}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(lambda row: check(row, args.decode), rows))
    selected = review_sheets(rows, run)
    report = {
        'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'expected_episodes': len(expected), 'manifest_episodes': len(rows),
        'tasks': len({key.split('/')[0] for key in expected}),
        'manifest_missing': sorted(expected - manifest), 'manifest_extra': sorted(manifest - expected),
        'missing_videos': sorted(expected - actual), 'extra_videos': sorted(actual - expected),
        'extra_episode_artifacts': sorted(artifact_keys - expected),
        'missing_ending_frames': sorted(expected - endings),
        'duplicate_manifest_keys': len(rows) - len(manifest),
        'valid_episodes': sum(not result['errors'] for result in checks),
        'frame_counts': dict(collections.Counter(result.get('video', {}).get('nb_frames', 'missing') for result in checks)),
        'full_decode': args.decode, 'checks': checks,
        'review_tasks': len(selected), 'review_episodes': sum(len(item['episodes']) for item in selected),
        'pending_directories': [str(path.relative_to(base / 'demos')) for path in (base / 'demos').glob('*/episode_*/.pending')],
    }
    (run / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('checks', 'pending_directories')}, indent=2))
    if (expected != manifest or actual != expected or artifact_keys - expected or expected - endings
            or report['duplicate_manifest_keys']
            or report['pending_directories'] or any(r['errors'] for r in checks)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
