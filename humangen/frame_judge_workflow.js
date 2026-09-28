export const meta = {
  name: 'humangen-frame-judge',
  description: 'Opus (low effort) judges edited human keyframes against their robot frames, five episodes per agent, before any video is generated',
  phases: [{ title: 'Judge frames', detail: 'one Opus agent per 5 episodes' }],
}

const REPO = '/Users/desmondzee/Hardware/ICL-so101'
const FRAME = {
  type: 'object',
  properties: {
    frame: { type: 'string' },
    ok: { type: 'boolean' },
    objects_ok: { type: 'boolean' }, nothing_added: { type: 'boolean' }, states_ok: { type: 'boolean' },
    camera_ok: { type: 'boolean' }, one_hand: { type: 'boolean' }, hand_plausible: { type: 'boolean' }, same_hand_as_start: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['frame', 'ok', 'objects_ok', 'nothing_added', 'states_ok', 'camera_ok', 'one_hand', 'hand_plausible', 'same_hand_as_start', 'issues'],
}
const SCHEMA = {
  type: 'object',
  properties: { episodes: { type: 'array', items: { type: 'object', properties: { dir: { type: 'string' }, frames: { type: 'array', items: FRAME } }, required: ['dir', 'frames'] } } },
  required: ['episodes'],
}

const PROMPT = (dirs) => `You check edited images for a dataset that turns robot videos into human videos. For each episode folder below, compare every edited frame with its robot frame:
frame.jpg vs robot_first.jpg, and kf1.jpg vs robot_kf1.jpg, kf2.jpg vs robot_kf2.jpg, ... (list the folder to see which exist). pair.json has the scene objects (scene.entities) and task.
Each edited frame must show the robot frame's scene with the robot arm replaced by exactly one person's forearm and hand (the other hand out of frame):
objects_ok (every object present exactly once, same place), nothing_added (no new objects, clutter, text or robot parts), states_ok (open/closed, stacked, inside, upright as in the robot frame), camera_ok (same framing; ignore black letterbox bars),
one_hand (exactly one hand and forearm, no other limbs), hand_plausible (natural anatomy and size), same_hand_as_start (same skin, sleeve and size as in frame.jpg; true for frame.jpg itself). ok = all true.
Open the images yourself with the Read tool; zoom by cropping with PIL if needed (write crops only under each folder's judge_frames/). Be strict on duplicates, missing objects and extra hands, do not invent problems. Keep it brief.
Folders:
${dirs.map(d => `- ${d}`).join('\n')}`

const B = args.batch || 5
const chunks = []
for (let i = 0; i < args.dirs.length; i += B) chunks.push(args.dirs.slice(i, i + B))
const results = await parallel(chunks.map((c, i) => () =>
  agent(PROMPT(c), { label: `frames:${i}`, phase: 'Judge frames', schema: SCHEMA, model: 'opus', effort: 'low' })))
const eps = results.filter(Boolean).flatMap(r => r.episodes)
log(`${eps.length} episodes judged, ${eps.filter(e => e.frames.every(f => f.ok)).length} with every frame ok`)
return eps
