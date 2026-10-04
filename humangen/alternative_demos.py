"""Frozen six-transition fal comparison; resumes existing queue requests."""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from dotenv import dotenv_values
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/robot_removal/alternative_demos/validation_controls_v5'
KEYS = [
    'ricky0526__so101_pick_toy_to_plate_v3/episode_000',
    'lerobot__svla_so101_pickplace/episode_000',
    'chirag1701__can/episode_000',
]

MODELS = {
    'seedance_1_5_pro': 'fal-ai/bytedance/seedance/v1.5/pro/image-to-video',
    'h3_max_turbo': 'minimax/h3-max-turbo/image-to-video',
    'h3_max': 'minimax/h3-max/image-to-video',
}


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def request(url, token=None, payload=None):
    headers = {'Authorization': 'Key ' + token} if token else {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=120) as response:
        return response.read()


def run(model, endpoint, row, token, destination=None):
    dest = destination if destination is not None else OUT / model / row['key']
    dest.mkdir(parents=True, exist_ok=True)
    meta_path = dest / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(row, model=endpoint, status='prepared', attempt=1)
    if meta['status'] == 'complete':
        return
    try:
        payload = dict(prompt=row['prompt'], image_url=row['start_data'], end_image_url=row['end_data'], seed=row.get('seed', 0))
        if model == 'seedance_1_5_pro':
            payload.update(resolution='480p', duration='5', generate_audio=False, camera_fixed=True, aspect_ratio=row['aspect'])
        else:
            payload.update(resolution='480P', duration=5, prompt_expansion_mode='disabled')
        meta.pop('start_data', None)
        meta.pop('end_data', None)
        meta['settings'] = {k:v for k,v in payload.items() if k not in ('image_url', 'end_image_url', 'prompt')}
        (dest / 'prompt.txt').write_text(row['prompt'])
        if 'queue' not in meta:
            # Save an ambiguous state before POST; never silently resubmit after uncertain delivery.
            if meta['status'] == 'submitting':
                raise RuntimeError('Prior submission delivery unknown; inspect before resubmitting')
            meta.update(status='submitting', submitted_at=time.time())
            save(meta_path, meta)
            meta['queue'] = json.loads(request('https://queue.fal.run/' + endpoint, token, payload))
            meta['status'] = 'queued'
            save(meta_path, meta)
            print('submitted', model, row['key'], flush=True)
        queue = meta['queue']
        deadline = time.time() + 1800
        while time.time() < deadline:
            status = json.loads(request(queue['status_url'], token))
            meta['queue_status'] = status.get('status')
            save(meta_path, meta)
            if status.get('status') == 'COMPLETED':
                break
            time.sleep(5)
        else:
            raise TimeoutError('Queue wait exceeded 30 minutes; resume this same request')
        result_path = dest / 'response.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = json.loads(request(queue['response_url'], token))
            save(result_path, result)
        video = dest / 'video.mp4'
        if not video.exists():
            video.write_bytes(request(result['video']['url']))
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
        save(dest / 'probe.json', probe)
        subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True)
        frames = dest / '_frames'
        frames.mkdir(exist_ok=True)
        subprocess.run(['ffmpeg','-v','error','-y','-threads','1','-i',str(video),'-filter_threads','1','-vf','fps=2,scale=320:-1',str(frames/'%03d.png')],check=True)
        samples = [Image.open(p).convert('RGB') for p in sorted(frames.glob('*.png'))]
        w,h = samples[0].size
        canvas = Image.new('RGB',(6*w,((len(samples)+5)//6)*h),'black')
        for i,im in enumerate(samples):
            canvas.paste(im,((i%6)*w,(i//6)*h))
        canvas.save(dest / 'contact.png')
        for im in samples: im.close()
        for p in frames.glob('*.png'): p.unlink()
        frames.rmdir()
        meta.update(status='complete', completed_at=time.time(), elapsed_seconds=time.time()-meta['submitted_at'], actual_duration=probe['format']['duration'], billed_cost=None)
        save(meta_path,meta)
        print('complete',model,row['key'],flush=True)
    except Exception as exc:
        # Print only type; provider errors can include request details.
        meta['error_type'] = type(exc).__name__
        if hasattr(exc, 'code'): meta['http_status'] = exc.code
        save(meta_path,meta)
        print('error',model,row['key'],type(exc).__name__,flush=True)


def main():
    global OUT, KEYS
    parser = argparse.ArgumentParser()
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--shape', action='store_true')
    parser.add_argument('--shape-more', action='store_true')
    args = parser.parse_args()
    if args.hard:
        OUT = ROOT / 'outputs/robot_removal/alternative_demos/first_panel_v5'
        KEYS = ['ReubenLim__so101_tape_in_square/episode_030',
                'aiden-li__so101-close-lower-drawer/episode_000',
                'sattgle__clean2-test/episode_006']
    if args.extended:
        OUT = ROOT / 'outputs/robot_removal/alternative_demos/extended_five_v5'
        KEYS = ['Rorschach4153__so101_30_fold/episode_005',
                'LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid/episode_000',
                'fbeltrao__so101_unplug_cable_4/episode_005',
                'tenkau__SO101-Stack3Blocks/episode_004',
                'stsqitx__clean/episode_005']
    if args.shape:
        OUT = ROOT / 'outputs/robot_removal/alternative_demos/shape_green_v5'
        KEYS = ['LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_000']
    if args.shape_more:
        OUT = ROOT / 'outputs/robot_removal/alternative_demos/shape_green_more_v5'
        KEYS = ['LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place/episode_' + n
                for n in ['001', '003', '005', '007', '009']]
    token = dotenv_values(ROOT / '.env').get('FAL_API_KEY')
    if not token: raise RuntimeError('FAL_API_KEY missing')
    rows=[]
    for key in KEYS:
        m=json.loads((ROOT/'outputs/robot_removal/demos'/key/'meta.json').read_text())
        row={k:m[k] for k in ('key','prompt','prompt_version','prompt_sha256','starting_frame','ending_frame','starting_frame_sha256','ending_frame_sha256','aspect')}
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        for side,field in [('start','starting_frame'),('end','ending_frame')]:
            raw=Path(row[field]).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==row[field+'_sha256']
            row[side+'_data']='data:image/png;base64,'+base64.b64encode(raw).decode()
        rows.append(row)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'manifest.json'
    frozen=[{k:v for k,v in row.items() if not k.endswith('_data')} for row in rows]
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        assert [row for row in existing if row['key'] in KEYS] == frozen
    else: save(manifest,frozen)
    # Sequential model priority; up to three independent episode requests per model.
    for model,endpoint in MODELS.items():
        if (args.shape or args.shape_more) and model != 'h3_max_turbo': continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(run,model,endpoint,row,token) for row in rows]
            for f in futures: f.result()

if __name__=='__main__': main()
