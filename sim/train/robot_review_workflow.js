export const meta = {
  name: 'so101-sim-train-robot-review',
  description: 'Opus judges each physics-approved simulated robot episode (5 per agent), re-checks needs_detail with dense frames, and an independent Opus verifier tries to refute every accept',
  whenToUse: 'After `python -m sim.train.review queue --out args.json`; pass that JSON as args. Save the return value and run `python -m sim.train.review import <file>`.',
  phases: [
    { title: 'Judge', detail: 'Opus, 5 episodes per agent' },
    { title: 'Detail', detail: 'Opus, needs_detail episodes one at a time with contiguous frames' },
    { title: 'Verify', detail: 'Opus, up to 3 accepted episodes per agent, adversarial' },
  ],
}

// args: {root, items: [{key, dir, request_sha256}]} from `python -m sim.train.review queue`, or the compact {root, pairs: [[key, request_sha256]]}.
// Optional: judgeBatch (5), verifyBatch (3), judgeEffort ('low'), verifyEffort ('low').
const ROOT = args.root
const ITEMS = (args.items || (args.pairs || []).map(([key, request_sha256]) => ({ key, request_sha256 })))
  .map(it => ({ ...it, dir: it.dir || `${args.root}/candidates/${it.key}` }))
const SCRATCH = `${ROOT}/review_scratch`
const byKey = Object.fromEntries(ITEMS.map(it => [it.key, it]))

const RULES = `Each item is one simulated SO-101 robot episode recorded by a scripted oracle in MuJoCo. It already passed a strict numeric physics audit (physics.json); your job is the visual review that numbers cannot do, before money is spent generating a human video from it. Accept only if ALL hold:
1. Task completion and order: the robot performs the instruction in robot_review_request.json, handling the objects in exactly the listed ordered_actions order, and the final frames show the instructed end state (inside / on / beside with a gap / long side left-to-right ...). Partial completion, the wrong object, wrong destination or wrong order rejects.
2. Physical plausibility: objects move only while the gripper holds or touches them; no teleporting, popping, sinking into or passing through the table, fixtures or other objects; no floating; grasps look like real two-finger grasps; release is clean and the object settles (no sliding, rolling or tipping after release in the final second).
3. Scene integrity: distractors and fixtures do not move unless the task says so; no arm, wrist or gripper collisions with the table, fixtures or the basket rim beyond the fingertips gently touching near a grasp; no wild arm swings, snaps, jitter or oscillation.
4. Visual quality: in the front view the task objects and goal are visible and identifiable for the whole episode (not hidden behind the arm at the moment that matters, not out of frame, not too dark or blown out); the wrist view is coherent. first.png / last.png are robot-free renders: first.png must match the start layout and last.png must show the same final object arrangement as the last video frames, with no robot.
Do not invent problems, and do not excuse real ones. Physics numbers in the request are context; they cannot make a visibly wrong episode acceptable.`

const LOOK = (it) => `Episode ${it.key}, frozen evidence in ${it.dir}/ (read-only, never modify it):
- robot_review_request.json: instruction, ordered_actions, oracle_events (frame indices of skills, gripper commands, grasp/release), variation, physics summary, artifact hashes.
- review/front_NN.jpg and review/wrist_NN.jpg: contact sheets; each tile is labelled with camera, frame index, time and why it was chosen (start, skill, grasp, release, violation, final_second, filler). review/frames.json lists them.
- first.png, last.png (robot-free endpoints), robot_front.mp4 and robot_wrist.mp4 (30 fps).
Open the request and EVERY sheet page. Whenever a grasp, release, contact or the final settling is ambiguous, extract contiguous frames yourself instead of guessing, e.g. mkdir -p ${SCRATCH}/${it.key} && ffmpeg -nostdin -loglevel error -y -ss <t0> -t <seconds> -i ${it.dir}/robot_front.mp4 -vf fps=15,scale=480:-1 ${SCRATCH}/${it.key}/front_%03d.jpg (same for robot_wrist.mp4), and look at them. Write only under ${SCRATCH}.`

const REVIEWER = { type: 'object', properties: { label: { type: 'string' }, model: { type: 'string' }, role: { type: 'string' } }, required: ['label', 'model', 'role'] }
const EPISODE = {
  type: 'object',
  properties: {
    key: { type: 'string' },
    verdict: { enum: ['accept', 'reject', 'needs_detail'] },
    task_completed_in_order: { type: 'boolean' },
    physics_plausible: { type: 'boolean' },
    scene_ok: { type: 'boolean' },
    visibility_ok: { type: 'boolean' },
    endpoints_ok: { type: 'boolean' },
    reasons: { type: 'array', items: { type: 'string' }, minItems: 1, description: 'why it passes or fails, citing frame indices/times' },
    inspected_evidence: { type: 'array', items: { type: 'string' }, minItems: 1, description: 'files and frame ranges actually opened, e.g. review/front_02.jpg, robot_front.mp4 3.0-4.5s @15fps' },
    detail_request: { type: 'string', description: 'for needs_detail: which time window and camera need contiguous frames and why' },
  },
  required: ['key', 'verdict', 'task_completed_in_order', 'physics_plausible', 'scene_ok', 'visibility_ok', 'endpoints_ok', 'reasons', 'inspected_evidence'],
}
const JUDGE = { type: 'object', properties: { episodes: { type: 'array', items: EPISODE } }, required: ['episodes'] }
const VERIFY = {
  type: 'object',
  properties: { episodes: { type: 'array', items: { type: 'object', properties: {
    key: { type: 'string' }, confirmed: { type: 'boolean' },
    reasons: { type: 'array', items: { type: 'string' }, minItems: 1 },
    inspected_evidence: { type: 'array', items: { type: 'string' }, minItems: 1 },
  }, required: ['key', 'confirmed', 'reasons', 'inspected_evidence'] } } },
  required: ['episodes'],
}

const judgePrompt = (items) => `You review simulated robot episodes for a robot-learning dataset. Work in ${ROOT}; do not modify anything except under ${SCRATCH}.
${RULES}
Review each of these ${items.length} episodes independently and return one entry per episode with its key exactly as given:
${items.map(LOOK).join('\n\n')}
Use needs_detail only if the evidence you could extract still leaves the decision open; say what is missing in detail_request.`

const detailPrompt = (it, first) => `You are the detail reviewer for one simulated robot episode a first reviewer could not decide. Work in ${ROOT}; do not modify anything except under ${SCRATCH}.
${RULES}
${LOOK(it)}
First reviewer's open question: ${first.detail_request || first.reasons.join('; ')}
Extract contiguous frames (fps=15 or the full 30 fps) over every window in question from BOTH robot_front.mp4 and robot_wrist.mp4 and decide accept or reject. Return needs_detail only if the videos themselves cannot answer it.`

const verifyPrompt = (pairs) => `You are an independent second reviewer. Another reviewer accepted the simulated robot episodes below; your job is to find any reason they should be rejected. Work in ${ROOT}; do not modify anything except under ${SCRATCH}.
${RULES}
For each episode, look at the evidence yourself; do not trust the first review. Always extract contiguous frames (fps=15) around every grasp and release and over the final second from robot_front.mp4, and check the wrist view at the grasps. Compare last.png with the final video frame. Confirm only if you are confident every rule holds; if in doubt, do not confirm.
${pairs.map(p => `${LOOK(p.item)}\nFirst review: ${p.judged.reasons.join(' ')}`).join('\n\n')}
Return one entry per episode with its key exactly as given; reasons cite the frames you checked.`

const chunk = (xs, n) => Array.from({ length: Math.ceil(xs.length / n) }, (_, i) => xs.slice(i * n, i * n + n))
const JB = args.judgeBatch || 5, VB = args.verifyBatch || 3
const JE = args.judgeEffort || 'low', VE = args.verifyEffort || 'low'
const batches = chunk(ITEMS, JB)
log(`${ITEMS.length} robot episodes, ${batches.length} judge agents`)

const stamp = (review, label, role) => review && { ...review, reviewer: { label, model: 'opus', role } }

const results = await pipeline(
  batches,
  async (items, _b, i) => {
    const label = `robot-judge:${i}`
    const judged = await agent(judgePrompt(items), { label, phase: 'Judge', schema: JUDGE, model: 'opus', effort: JE })
    const keys = items.map(it => it.key)
    const found = Object.fromEntries((judged?.episodes || []).filter(e => keys.includes(e.key)).map(e => [e.key, stamp(e, label, 'judge')]))
    // needs_detail: one dense single-episode pass replaces the batched verdict.
    await parallel(items.filter(it => found[it.key]?.verdict === 'needs_detail').map((it, j) => async () => {
      const dl = `robot-detail:${i}.${j}`
      const d = await agent(detailPrompt(it, found[it.key]), { label: dl, phase: 'Detail', schema: EPISODE, model: 'opus', effort: 'medium' })
      if (d && d.key === it.key) found[it.key] = stamp(d, dl, 'judge')
    }))
    return found
  },
  async (found, items, i) => {
    const accepted = items.filter(it => found[it.key]?.verdict === 'accept').map(it => ({ item: it, judged: found[it.key] }))
    const verified = {}
    await parallel(chunk(accepted, VB).map((pairs, j) => async () => {
      const label = `robot-verify:${i}.${j}`
      const v = await agent(verifyPrompt(pairs), { label, phase: 'Verify', schema: VERIFY, model: 'opus', effort: VE })
      const keys = pairs.map(p => p.item.key)
      for (const e of (v?.episodes || []).filter(e => keys.includes(e.key))) verified[e.key] = stamp(e, label, 'verifier')
    }))
    return items.map(it => {
      const j = found[it.key] || null
      const v = verified[it.key] || null
      const final = !j ? 'pending'
        : j.verdict === 'reject' ? 'reject'
        : j.verdict === 'needs_detail' ? 'pending'
        : !v ? 'pending'
        : v.confirmed ? 'accept' : 'reject'
      return { key: it.key, request_sha256: it.request_sha256, final, judge: j, verify: v }
    })
  },
)
const flat = results.filter(Boolean).flat()
const missing = ITEMS.filter(it => !flat.some(r => r.key === it.key)).map(it => it.key)
if (missing.length) log(`no result for ${missing.length} episodes (they stay pending): ${missing.join(', ')}`)
const count = (f) => flat.filter(r => r.final === f).length
log(`${flat.length} episodes: accept ${count('accept')}, reject ${count('reject')}, pending ${count('pending')}`)
return Object.fromEntries(flat.map(r => [r.key, r]))
