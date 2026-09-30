"""Full-batch H3 Max Turbo generation, preserving original episode prompts.

Run: python -m humangen.fal_bulk_demos --workers 8
Resumes saved fal queue requests; uncertain submissions are never repeated.
"""
import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import json
import os
import time
from pathlib import Path
from dotenv import dotenv_values
from humangen.alternative_demos import ROOT, run, save

ENDPOINT = 'minimax/h3-max-turbo/image-to-video'
DEST = ROOT / 'outputs/robot_removal/demos_h3_max_turbo'
RUN = DEST / '_run'
FIELDS = ('key','prompt','prompt_version','prompt_sha256','starting_frame','ending_frame',
          'starting_frame_sha256','ending_frame_sha256','aspect')


def prepare():
    baseline = ROOT / 'outputs/robot_removal/demos'
    manifest = [json.loads(line) for line in (baseline/'_run/manifest.jsonl').read_text().splitlines()]
    rows = []
    for original in manifest:
        meta = json.loads((baseline/original['key']/'meta.json').read_text())
        row = {field:meta[field] for field in FIELDS}
        assert hashlib.sha256(row['prompt'].encode()).hexdigest() == row['prompt_sha256']
        for field in ('starting_frame','ending_frame'):
            assert hashlib.sha256(Path(row[field]).read_bytes()).hexdigest() == row[field+'_sha256']
        row.update(duration=5, resolution='480P', seed=0, prompt_expansion_mode='disabled',
                   source_meta=str((baseline/row['key']/'meta.json').resolve()),
                   output_dir=str((DEST/row['key']).resolve()))
        rows.append(row)
    assert len({r['key'] for r in rows}) == len(rows) == 1960
    assert len({r['key'].split('/')[0] for r in rows}) == 28
    RUN.mkdir(parents=True,exist_ok=True)
    path=RUN/'manifest.jsonl'
    if path.exists():
        assert [json.loads(line) for line in path.read_text().splitlines()] == rows
    else:
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return rows


def generate(row, token):
    dest = DEST/row['key']
    meta_path = dest/'meta.json'
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        if old.get('status')=='complete':
            assert all((dest/name).exists() for name in ('video.mp4','contact.png','probe.json'))
            assert old['prompt_sha256']==row['prompt_sha256'] and old['model']==ENDPOINT
            return 'complete'
    encoded=dict(row)
    for side,field in [('start','starting_frame'),('end','ending_frame')]:
        encoded[side+'_data']='data:image/png;base64,'+base64.b64encode(Path(row[field]).read_bytes()).decode()
    for attempt in range(3):
        run('h3_max_turbo',ENDPOINT,encoded,token,destination=dest)
        meta=json.loads(meta_path.read_text())
        if meta.get('status')=='complete':return 'complete'
        # Retrying retrieval is free of duplicate generation; never repeat uncertain POST.
        if 'queue' not in meta or meta.get('http_status') in (400,401,402,403,404,422):break
        time.sleep(2*(attempt+1))
    return 'needs_attention'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    token=dotenv_values(ROOT/'.env').get('FAL_API_KEY')
    if not token:raise RuntimeError('FAL_API_KEY missing')
    rows=prepare()
    (RUN/'pid').write_text(str(os.getpid())+'\n')
    state=dict(total=len(rows),tasks=28,model=ENDPOINT,duration=5,resolution='480P',
               prompt_version='v5_exclusion',workers=args.workers,complete=0,needs_attention=[],remaining=len(rows),
               started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),status='running')
    save(RUN/'status.json',state)
    print('Starting',len(rows),'episodes, 28 tasks, workers',args.workers,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(generate,row,token):row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row=futures[future]
            try:result=future.result()
            except Exception as exc:
                result='needs_attention';print('worker error',row['key'],type(exc).__name__,flush=True)
            if result=='complete':state['complete']+=1
            else:state['needs_attention'].append(row['key'])
            state['remaining']=state['total']-state['complete']-len(state['needs_attention'])
            state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            save(RUN/'status.json',state)
            print('Progress',state['complete'],'complete,',state['remaining'],'remaining,',len(state['needs_attention']),'need attention',flush=True)
    state['status']='complete' if not state['needs_attention'] else 'needs_attention'
    state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(RUN/'status.json',state)

if __name__=='__main__':main()
