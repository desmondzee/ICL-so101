"""Resumable FastH3 demos from robot-removal start/end images and pair annotations.

Prepare: python -m humangen.bulk_demos --prepare
Generate: python -m humangen.bulk_demos --one-go
The full batch resumes automatically after session errors. Optional worker
arguments support partitioned runs. No image edits are performed.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import datetime
from pathlib import Path
from dotenv import load_dotenv
from humangen.context import task_prompt
from humangen.generate import FAST_MODEL, CAPTURE_NAME, RECORDING_VERSION, VideoRequest, generate_videos, generate_videos_sync
from humangen.audit_demos import check_tail

VERSION = 'v5_exclusion'
DURATION = 5.0
EFFECTIVE = 124 / 24


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entry(scene: dict) -> tuple[str, str]:
    """Same robot-edge rule as pilot.entry_edge, without VLM dependencies."""
    robot = next((e for e in scene['entities'] if e['kind'] == 'actor' and e.get('box_2d')), None)
    edge = 'right'
    if robot:
        y0, x0, y1, x1 = robot['box_2d']
        edge = min({'top': y0, 'bottom': 1000-y1, 'left': x0, 'right': 1000-x1},
                   key=lambda k: {'top': y0, 'bottom': 1000-y1, 'left': x0, 'right': 1000-x1}[k])
    return ('left' if edge == 'left' else 'right'), edge+' edge'


def prepare(root: Path) -> list[dict]:
    base = root/'outputs/robot_removal'
    records=[]
    for start in sorted((base/'v4').glob('*/episode_*/result_native.png')):
        relative = start.relative_to(base/'v4')
        end=base/'v4_last'/relative
        pair_path=root/'data/robot_removal/full'/relative.parent/'pair.json'
        if not end.is_file():
            raise FileNotFoundError(end)
        pair=json.loads(pair_path.read_text())
        hand,edge=entry(pair['scene'])
        prompt=task_prompt(pair,hand,edge,DURATION,True,effective_duration=EFFECTIVE,version=VERSION)
        dest=base/'demos'/relative.parent
        # Retain the 15 already approved/generated v5 demos and their exact prompts.
        old=dest/'meta.json'
        reused=False
        if old.exists() and (dest/'video.mp4').exists() and (dest/'contact.png').exists():
            previous=json.loads(old.read_text())
            if previous.get('prompt_version')==VERSION and previous.get('duration')==DURATION:
                prompt=previous['prompt'];reused=True
        records.append(dict(key=str(relative.parent),prompt=prompt,prompt_version=VERSION,
            prompt_sha256=digest(prompt.encode()),starting_frame=str(start.resolve()),
            ending_frame=str(end.resolve()),pair_json=str(pair_path.resolve()),
            pair_sha256=digest(pair_path.read_bytes()),starting_frame_sha256=digest(start.read_bytes()),
            ending_frame_sha256=digest(end.read_bytes()),duration=DURATION,effective_duration=EFFECTIVE,
            seed=0,aspect='4:3',model=FAST_MODEL,hand=hand,entry_edge=edge,reused=reused,
            prompt_template='humangen.context.task_prompt',template_source_sha256=digest((root/'humangen/context.py').read_bytes()),
            output_dir=str(dest.resolve())))
    run=base/'demos/_run';run.mkdir(parents=True,exist_ok=True)
    (run/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (run/'template_source.py').write_bytes((root/'humangen/context.py').read_bytes())
    (run/'summary.json').write_text(json.dumps({'total':len(records),'datasets':len({r['key'].split('/')[0] for r in records}),
        'prompt_version':VERSION,'reused':sum(r['reused'] for r in records)},indent=2)+'\n')
    return records


def complete(row: dict) -> bool:
    d=Path(row['output_dir'])
    if not all((d/f).is_file() for f in ('video.mp4','contact.png','meta.json')):
        return False
    m=json.loads((d/'meta.json').read_text())
    frames=m.get('recorded_frames', 0)
    capture=m.get('capture', {})
    preserved=(frames >= 124 or (m.get('recording_version')==RECORDING_VERSION
               and capture.get('received_video_frames')==frames))
    return (m.get('prompt_sha256')==row['prompt_sha256'] and m.get('prompt_version')==VERSION
            and frames >= 120 and preserved)


def publish(row: dict) -> None:
    dest=Path(row['output_dir']);pending=dest/'.pending'
    video=pending/'video.mp4'
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,width,height','-of','json',str(video)]))['streams'][0]
    if int(info['nb_frames']) < 120:
        raise RuntimeError(f"too few recorded frames: {info['nb_frames']}")
    capture=json.loads((pending/CAPTURE_NAME).read_text())
    if capture['received_video_frames'] != int(info['nb_frames']):
        raise RuntimeError('encoder did not preserve all received video frames')
    check_tail(video, Path(row['ending_frame']))
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    meta=dict(row,recorded_frames=int(info['nb_frames']),capture=capture,
              recording_version=capture['recording_version'])
    (pending/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    for name in ('video.mp4','contact.png','meta.json'):
        (pending/name).replace(dest/name)
    (pending/CAPTURE_NAME).unlink()
    pending.rmdir()


def worker(root: Path,index: int,workers: int,chunk_size: int) -> None:
    manifest=root/'outputs/robot_removal/demos/_run/manifest.jsonl'
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    assigned=[r for i,r in enumerate(rows) if i % workers==index and not complete(r)]
    logging.info('worker %s: %s pending',index,len(assigned))
    failures=[]
    for offset in range(0,len(assigned),chunk_size):
        chunk=assigned[offset:offset+chunk_size]
        for attempt in range(3):
            unfinished=[]
            for row in chunk:
                if complete(row): continue
                pending=Path(row['output_dir'])/'.pending';pending.mkdir(parents=True,exist_ok=True)
                if (pending/'video.mp4').exists() and (pending/'contact.png').exists():
                    try:
                        publish(row);logging.info('completed %s',row['key']);continue
                    except Exception as exc:logging.warning('invalid partial %s: %s',row['key'],exc)
                for f in ('video.mp4','contact.png','meta.json',CAPTURE_NAME):(pending/f).unlink(missing_ok=True)
                unfinished.append(row)
            if not unfinished:break
            reqs=[VideoRequest(prompt=r['prompt'],reference_image=r['starting_frame'],ending_image=r['ending_frame'],
                duration=DURATION,seed=r['seed'],output_path=Path(r['output_dir'])/'.pending/video.mp4') for r in unfinished]
            try:generate_videos_sync(reqs,aspect='4:3',model=FAST_MODEL)
            except Exception:logging.exception('worker %s batch %s attempt %s',index,offset,attempt+1)
        # Publish any files written by the last attempt, including partial-batch successes.
        for row in chunk:
            if complete(row):continue
            try:publish(row);logging.info('completed %s',row['key'])
            except Exception as exc:
                failures.append(dict(key=row['key'],error=str(exc)))
                logging.error('failed %s: %s',row['key'],exc)
        path=root/f'outputs/robot_removal/demos/_run/worker_{index}_status.json'
        path.write_text(json.dumps({'assigned':len(assigned),'processed':min(offset+chunk_size,len(assigned)),
            'failures':failures},indent=2)+'\n')
    if failures:raise RuntimeError(f'{len(failures)} episodes failed; rerun to retry')


async def one_go(root: Path) -> None:
    """One generation session, publishing finished clips as they arrive."""
    run=root/'outputs/robot_removal/demos/_run'
    rows=[json.loads(line) for line in (run/'manifest.jsonl').read_text().splitlines()]
    todo=[row for row in rows if not complete(row)]
    requests=[]
    for row in todo:
        pending=Path(row['output_dir'])/'.pending';pending.mkdir(parents=True,exist_ok=True)
        for name in ('video.mp4','contact.png','meta.json',CAPTURE_NAME):(pending/name).unlink(missing_ok=True)
        requests.append(VideoRequest(prompt=row['prompt'],reference_image=row['starting_frame'],
            ending_image=row['ending_frame'],duration=DURATION,seed=row['seed'],output_path=pending/'video.mp4'))
    logging.info('one session: %s remaining, %s already complete',len(todo),len(rows)-len(todo))
    if not requests:return
    task=asyncio.create_task(generate_videos(requests,aspect='4:3',model=FAST_MODEL))
    outstanding=list(todo);failures=[]
    try:
        while True:
            # Generation is played in queue order; inspect a small advancing window.
            for row in outstanding[:20]:
                marker=Path(row['output_dir'])/'.pending/contact.png'
                if not marker.is_file():continue
                try:
                    await asyncio.to_thread(publish,row)
                    logging.info('completed %s',row['key'])
                except Exception as exc:
                    failures.append(dict(key=row['key'],error=str(exc)))
                    logging.exception('publish failed %s',row['key'])
                outstanding.remove(row)
            status=dict(total=len(rows),complete=len(rows)-len(outstanding)-len(failures),remaining=len(outstanding)+len(failures),
                        failed=failures,prompt_version=VERSION)
            (run/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            if task.done() and not any((Path(r['output_dir'])/'.pending/contact.png').exists() for r in outstanding):
                break
            await asyncio.sleep(1)
        await task
        if failures or outstanding:raise RuntimeError('generation or publication incomplete; see status.json')
    finally:
        if not task.done():
            task.cancel()
            try:await task
            except asyncio.CancelledError:pass


async def resumable_one_go(root: Path) -> None:
    """Keep the entire batch in one invocation, resuming after session errors."""
    run=root/'outputs/robot_removal/demos/_run'
    for attempt in range(1, 6):
        try:
            await one_go(root)
            return
        except Exception as exc:
            # Log a bounded diagnostic; SDK errors may carry signed URLs.
            code=getattr(exc, 'code', None)
            reason=f'{type(exc).__name__}: {code}' if code else str(exc).split('https://')[0][:300]
            logging.error('session attempt %s/5 failed: %s', attempt, reason)
            record=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt=attempt,error=reason)
            with (run/'session_retries.jsonl').open('a') as file:file.write(json.dumps(record)+'\n')
            if attempt==5:raise RuntimeError('five session attempts failed; see session_retries.jsonl') from None
            await asyncio.sleep(5)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--one-go',action='store_true',help='use one continuous Reactor session')
    ap.add_argument('--worker-index',type=int,default=0)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--chunk-size',type=int,default=10)
    args=ap.parse_args();load_dotenv(args.root/'.env')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.prepare:
        rows=prepare(args.root);print(f'Prepared {len(rows)} episodes across {len({r["key"].split("/")[0] for r in rows})} tasks')
    elif args.one_go:asyncio.run(resumable_one_go(args.root))
    else:worker(args.root,args.worker_index,args.workers,args.chunk_size)

if __name__=='__main__':main()
