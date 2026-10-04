export const meta = {
  name: 'so101-humangen-curation',
  description: 'Opus judges each human-robot pair (5 per agent); every accept is re-checked by an independent Opus verifier told to refute it',
  phases: [{ title: 'Judge', detail: 'Opus, 5 pairs per agent' }, { title: 'Verify', detail: 'Opus, up to 3 accepted pairs per agent, adversarial' }],
}

const REPO = '/Users/desmondzee/Hardware/ICL-so101'
const DIR = args.dir || `${REPO}/data/so101_curation/review`  // args.dir: another review folder in the same layout (e.g. the sim val set)

const RULES = `A pair is a generated human demonstration video (human.mp4) and the real SO-101 robot episode it was generated from. The human video must show a person doing the same task as the robot, so a policy can learn the task from it. Accept only if ALL of these hold:
1. Task adherence: the person does the task in info.json (instruction, steps, goals) on the same objects, in the same order, to the same destinations, and the human video's last frames show the same end state as the robot's last frame (object positions and states: inside, on top, open/closed, folded, upright...). Partial completion, a different destination, or a different end state rejects.
2. Physical plausibility: objects move only while a hand touches them; no object appears, vanishes, duplicates, morphs, teleports, passes through another object or floats; contact and grasps are believable; drawers and lids move along their real axis; deformables (cloth, cable, tissue) deform plausibly.
3. Hands: one acting hand. A second hand of the same person may only steady or hold a task object; it must not bring anything in or do the task. A hand that enters already holding a task object (a copy of it) rejects. A visible person, sleeve or idle hand at the frame edge is fine.
4. Scene consistency: the human video's first frame matches the robot's first frame with the robot removed (same objects, layout, camera); no new objects, text, captions or graphic overlays at any time; no scene change or cut.
Minor rendering softness, slight camera drift, or small lighting changes are fine. Do not invent problems, and do not excuse real ones.`

const LOOK = (key) => `Pair ${key}, in ${DIR}/${key}/: info.json (task), human_frames.jpg (12 frames of the human video, timestamps on each), robot_frames.jpg (8 robot front-camera frames, first and last included), human.mp4. Open both images and info.json. When the 12 frames leave any doubt (a fast grasp, a possible copy appearing, the exact end state), extract more frames, e.g. mkdir -p ${DIR}/${key}/more && ffmpeg -nostdin -loglevel error -y -i ${DIR}/${key}/human.mp4 -vf fps=6,scale=480:-1 ${DIR}/${key}/more/f_%03d.jpg, and look at the ones you need.`

const PAIR = {
  type: 'object',
  properties: {
    key: { type: 'string' },
    verdict: { enum: ['accept', 'reject'] },
    task_adherence: { type: 'integer', minimum: 1, maximum: 5 },
    physics: { type: 'integer', minimum: 1, maximum: 5 },
    end_state_match: { type: 'boolean' },
    hands_ok: { type: 'boolean' },
    scene_ok: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string', description: 'one sentence: what the person does and why it passes or fails' },
  },
  required: ['key', 'verdict', 'task_adherence', 'physics', 'end_state_match', 'hands_ok', 'scene_ok', 'issues', 'summary'],
}
const JUDGE = { type: 'object', properties: { pairs: { type: 'array', items: PAIR } }, required: ['pairs'] }
const VERIFY = {
  type: 'object',
  properties: { pairs: { type: 'array', items: { type: 'object', properties: {
    key: { type: 'string' }, confirmed: { type: 'boolean' }, reason: { type: 'string' } }, required: ['key', 'confirmed', 'reason'] } } },
  required: ['pairs'],
}

const judgePrompt = (keys) => `You review human-robot pairs for a robot-learning dataset. Work in ${REPO}; do not modify any file outside ${DIR}.
${RULES}
Review each of these ${keys.length} pairs independently and return one entry per pair, with its key exactly as given:
${keys.map(LOOK).join('\n')}
Scores: task_adherence and physics 1-5 (5 = flawless). Accept only if every rule holds; an accept needs task_adherence >= 4 and physics >= 4. In issues, cite the frame timestamps you saw.`

const verifyPrompt = (items) => `You are an independent second reviewer. Another reviewer accepted the pairs below for a robot-learning dataset; your job is to find any reason they should be rejected. Work in ${REPO}; do not modify any file outside ${DIR}.
${RULES}
For each pair, look at the evidence yourself; do not trust the first review. Always extract extra frames from human.mp4 (fps=6 as shown) and check: the end state against the robot's last frame, whether any object appears, duplicates or moves without contact, whether a second hand does work, and whether the first frame matches the robot's first frame. Confirm only if you are confident every rule holds; if in doubt, do not confirm.
${items.map(p => `${LOOK(p.key)}\nFirst review's summary: ${p.summary}`).join('\n')}
Return one entry per pair with its key exactly as given; reason cites the frames you checked.`

const chunk = (xs, n) => Array.from({ length: Math.ceil(xs.length / n) }, (_, i) => xs.slice(i * n, i * n + n))
const batches = chunk(args.keys, 5)
log(`${args.keys.length} pairs, ${batches.length} judge agents`)

const results = await pipeline(
  batches,
  (keys, _b, i) => agent(judgePrompt(keys), { label: `judge:${i}`, phase: 'Judge', schema: JUDGE, model: 'opus', effort: 'medium' }),
  async (judged, keys, i) => {
    const byKey = Object.fromEntries((judged?.pairs || []).filter(p => keys.includes(p.key)).map(p => [p.key, p]))
    const accepted = keys.map(k => byKey[k]).filter(p => p && p.verdict === 'accept' && p.task_adherence >= 4 && p.physics >= 4)
    const verified = (await parallel(chunk(accepted, 3).map((items, j) => () =>
      agent(verifyPrompt(items), { label: `verify:${i}.${j}`, phase: 'Verify', schema: VERIFY, model: 'opus', effort: 'medium' }))))
      .filter(Boolean).flatMap(v => v.pairs)
    const vByKey = Object.fromEntries(verified.map(v => [v.key, v]))
    return keys.map(k => {
      const j = byKey[k] || null
      const v = vByKey[k] || null
      const final = j && j.verdict === 'accept' && j.task_adherence >= 4 && j.physics >= 4 && v && v.confirmed ? 'accept'
        : (j ? 'reject' : 'missing')
      return { key: k, final, judge: j, verify: v }
    })
  },
)
const flat = results.filter(Boolean).flat()
const acc = flat.filter(r => r.final === 'accept').length
const judgedAcc = flat.filter(r => r.judge && r.judge.verdict === 'accept').length
log(`${flat.length} pairs: judge accepted ${judgedAcc}, verified ${acc}, missing ${flat.filter(r => r.final === 'missing').length}`)
return Object.fromEntries(flat.map(r => [r.key, r]))
