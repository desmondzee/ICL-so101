export const meta = {
  name: 'so101-sim-train-human-review',
  description: 'Opus judges each generated human video against its simulated robot episode (5 per agent); an independent Opus verifier tries to refute every accept',
  whenToUse: 'After `python -m sim.train.human_review queue --out args.json`; pass that JSON as args. Save the return value and run `python -m sim.train.human_review import <file>`.',
  phases: [{ title: 'Judge', detail: 'Opus, 5 pairs per agent' }, { title: 'Verify', detail: 'Opus, up to 3 accepted pairs per agent, adversarial' }],
}

// Same sheet layout and rules as humangen/curate_workflow.js, adapted to the simulated
// training corpus: reviewer identities are stamped on every verdict, reasons and inspected
// evidence are required, and each result echoes the human.mp4 / info.json hashes it was
// issued for so the import can bind the verdict to exact files.
// args: {dir, keys, items: [{key, human_sha256, info_sha256}]}. Optional: judgeBatch (5), verifyBatch (3), effort ('medium').
const DIR = args.dir
const ITEMS = args.items || (args.keys || []).map(key => ({ key }))
const byKey = Object.fromEntries(ITEMS.map(it => [it.key, it]))
const SCRATCH = `${DIR}/_scratch`

const RULES = `A pair is a generated human demonstration video (human.mp4) and the simulated SO-101 robot episode it was generated from. Both are in the same rendered simulation scene (MuJoCo); the robot was removed from the start/end images given to the video model. The human video must show a person doing the same task as the robot, in the same order, so a policy can learn the task from it. Accept only if ALL of these hold:
1. Task adherence: the person does the task in info.json (instruction, ordered steps, goals) on the same objects, one object at a time in exactly the listed order, to the same destinations, and the human video's last frames show the same end state as the robot's last frame (object positions and states: inside, on, beside with a gap, orientation...). Partial completion, a wrong object, a different destination, a wrong order or a different end state rejects.
2. Physical plausibility: objects move only while a hand touches them; no object appears, vanishes, duplicates, morphs, teleports, passes through another object or floats; contact and grasps are believable; released objects rest stably.
3. Hands: one acting right hand, entering empty from the image edge and leaving empty. A hand that enters already holding a task object (a copy of it) rejects. A second hand doing work rejects.
4. Scene consistency: the human video's first frame matches the robot's first frame with the robot removed (same objects, layout, lighting, camera); distractors and fixtures do not move; no new objects, text, captions or graphic overlays at any time; no scene change or cut.
The scene is rendered; rendering softness, a realistic hand in the rendered scene, slight camera drift or small lighting changes are fine. Do not invent problems, and do not excuse real ones.`

const LOOK = (key) => `Pair ${key}, in ${DIR}/${key}/ (read-only): info.json (task, ordered steps, goals), human_frames.jpg (12 frames of the human video, timestamps on each), robot_frames.jpg (8 robot front-camera frames, first and last included), human.mp4. Open both images and info.json. When the 12 frames leave any doubt (a fast grasp, the order of objects, a possible copy appearing, the exact end state), extract more frames, e.g. mkdir -p ${SCRATCH}/${key} && ffmpeg -nostdin -loglevel error -y -i ${DIR}/${key}/human.mp4 -vf fps=6,scale=480:-1 ${SCRATCH}/${key}/f_%03d.jpg, and look at the ones you need. Write only under ${SCRATCH}.`

const PAIR = {
  type: 'object',
  properties: {
    key: { type: 'string' },
    verdict: { enum: ['accept', 'reject'] },
    task_adherence: { type: 'integer', minimum: 1, maximum: 5 },
    physics: { type: 'integer', minimum: 1, maximum: 5 },
    order_ok: { type: 'boolean' },
    end_state_match: { type: 'boolean' },
    hands_ok: { type: 'boolean' },
    scene_ok: { type: 'boolean' },
    reasons: { type: 'array', items: { type: 'string' }, minItems: 1, description: 'why it passes or fails, citing frame timestamps' },
    inspected_evidence: { type: 'array', items: { type: 'string' }, minItems: 1, description: 'files and frame ranges actually opened' },
    summary: { type: 'string', description: 'one sentence: what the person does and why it passes or fails' },
  },
  required: ['key', 'verdict', 'task_adherence', 'physics', 'order_ok', 'end_state_match', 'hands_ok', 'scene_ok', 'reasons', 'inspected_evidence', 'summary'],
}
const JUDGE = { type: 'object', properties: { pairs: { type: 'array', items: PAIR } }, required: ['pairs'] }
const VERIFY = {
  type: 'object',
  properties: { pairs: { type: 'array', items: { type: 'object', properties: {
    key: { type: 'string' }, confirmed: { type: 'boolean' },
    reasons: { type: 'array', items: { type: 'string' }, minItems: 1 },
    inspected_evidence: { type: 'array', items: { type: 'string' }, minItems: 1 },
  }, required: ['key', 'confirmed', 'reasons', 'inspected_evidence'] } } },
  required: ['pairs'],
}

const judgePrompt = (keys) => `You review human-robot pairs for a robot-learning dataset. Do not modify any file except under ${SCRATCH}.
${RULES}
Review each of these ${keys.length} pairs independently and return one entry per pair, with its key exactly as given:
${keys.map(LOOK).join('\n')}
Scores: task_adherence and physics 1-5 (5 = flawless). Accept only if every rule holds; an accept needs task_adherence >= 4 and physics >= 4. In reasons, cite the frame timestamps you saw.`

const verifyPrompt = (items) => `You are an independent second reviewer. Another reviewer accepted the pairs below for a robot-learning dataset; your job is to find any reason they should be rejected. Do not modify any file except under ${SCRATCH}.
${RULES}
For each pair, look at the evidence yourself; do not trust the first review. Always extract extra frames from human.mp4 (fps=6 as shown) and check: the order in which objects are handled, the end state against the robot's last frame, whether any object appears, duplicates or moves without contact, whether a second hand does work, and whether the first frame matches the robot's first frame. Confirm only if you are confident every rule holds; if in doubt, do not confirm.
${items.map(p => `${LOOK(p.key)}\nFirst review's summary: ${p.summary}`).join('\n')}
Return one entry per pair with its key exactly as given; reasons cite the frames you checked.`

const chunk = (xs, n) => Array.from({ length: Math.ceil(xs.length / n) }, (_, i) => xs.slice(i * n, i * n + n))
const JB = args.judgeBatch || 5, VB = args.verifyBatch || 3, EFFORT = args.effort || 'medium'
const stamp = (review, label, role) => review && { ...review, reviewer: { label, model: 'opus', role } }
const batches = chunk(ITEMS.map(it => it.key), JB)
log(`${ITEMS.length} pairs, ${batches.length} judge agents`)

const results = await pipeline(
  batches,
  async (keys, _b, i) => {
    const label = `human-judge:${i}`
    const judged = await agent(judgePrompt(keys), { label, phase: 'Judge', schema: JUDGE, model: 'opus', effort: EFFORT })
    return Object.fromEntries((judged?.pairs || []).filter(p => keys.includes(p.key)).map(p => [p.key, stamp(p, label, 'judge')]))
  },
  async (found, keys, i) => {
    const accepted = keys.map(k => found[k]).filter(p => p && p.verdict === 'accept' && p.task_adherence >= 4 && p.physics >= 4)
    const verified = {}
    await parallel(chunk(accepted, VB).map((items, j) => async () => {
      const label = `human-verify:${i}.${j}`
      const v = await agent(verifyPrompt(items), { label, phase: 'Verify', schema: VERIFY, model: 'opus', effort: EFFORT })
      const ks = items.map(p => p.key)
      for (const e of (v?.pairs || []).filter(e => ks.includes(e.key))) verified[e.key] = stamp(e, label, 'verifier')
    }))
    return keys.map(k => {
      const j = found[k] || null
      const v = verified[k] || null
      const passes = j && j.verdict === 'accept' && j.task_adherence >= 4 && j.physics >= 4
      const final = !j ? 'pending' : !passes ? 'reject' : !v ? 'pending' : v.confirmed ? 'accept' : 'reject'
      return { key: k, human_sha256: byKey[k].human_sha256, info_sha256: byKey[k].info_sha256, final, judge: j, verify: v }
    })
  },
)
const flat = results.filter(Boolean).flat()
const count = (f) => flat.filter(r => r.final === f).length
log(`${flat.length} pairs: accept ${count('accept')}, reject ${count('reject')}, pending ${count('pending')}`)
return Object.fromEntries(flat.map(r => [r.key, r]))
