export const meta = {
  name: 'so101-pair-audit',
  description: 'Per SO-101 pair: Opus (medium) fixes it against the frames, Sonnet (low) verifies it independently; a Sonnet fail gets one more fix and verify',
  phases: [
    { title: 'Fix', detail: 'one Opus agent per pair edits pair.json and validates it' },
    { title: 'Verify', detail: 'one Sonnet agent per pair judges the result independently' },
  ],
}

const REPO = '/Users/desmondzee/Hardware/ICL-so101'
const PAIRS = `${REPO}/data/so101_pairs`
const PART = { type: 'object', properties: { verdict: { enum: ['correct', 'partial', 'wrong'] }, issues: { type: 'array', items: { type: 'string' } } }, required: ['verdict', 'issues'] }
const PARTS = ['task_steps', 'task_goals', 'roles', 'segments', 'scene_entities', 'initial_state', 'boxes', 'bindings', 'outcome']
const VERDICT = {
  type: 'object',
  properties: { pair: { type: 'string' }, ...Object.fromEntries(PARTS.map(p => [p, PART])), overall: { enum: ['pass', 'minor_issues', 'fail'] }, notes: { type: 'string' } },
  required: ['pair', ...PARTS, 'overall', 'notes'],
}
const FIX = {
  type: 'object',
  properties: { pair: { type: 'string' }, changed: { type: 'boolean' }, fixed_parts: { type: 'array', items: { type: 'string' } }, genuine_failure: { type: 'boolean' }, validated: { type: 'boolean' }, notes: { type: 'string' } },
  required: ['pair', 'changed', 'fixed_parts', 'genuine_failure', 'validated', 'notes'],
}

const ITEMS = args.file
const CONTEXT = (it) => `Read ${ITEMS} and take element [${it.i}]; below, <pair>, <dir>, <name>, <ep> are its fields pair, dir, name, ep.
Files:
- Pair JSON: <pair> (task.instruction/roles/steps/goals, scene.entities/initial_state, bindings, segments, source.outcome/frames/fps/camera_key)
- Episode images in <dir>: first.jpg, middle.jpg, last.jpg, task_strip.jpg, other strips if present, frames/*.jpg (named by frame number)
- Dataset task notes (authoritative description of this dataset's task, recently corrected): ${PAIRS}/<name>/tasks/*.notes.txt (ignore *.old.txt)
- Schemas: ${REPO}/schema/common.schema.json, scene.schema.json, task.schema.json, pair.schema.json
- Extract any frame from the source video (the images above may be stale):
  cd ${REPO} && uv run --extra schema python -c "from pathlib import Path; from schema.source import load_episode, local_source, frame; r=Path('data/so101_curated/<name>'); ep=load_episode(r,*local_source(r),<ep>); print(frame(ep,'observation.images.front',<frame>,Path('<dir>/frames/<frame>.jpg')))"
- Draw the current annotation as a sheet (first frame with boxes, 12 frames with step bars):
  cd ${REPO} && uv run --extra schema python -c "from pathlib import Path; from schema.judge import sheet; print(sheet(Path('<pair>'), Path('<dir>')))"`

const FIX_PROMPT = (it, issues, why) => `You correct one automatically drafted annotation of a robot episode (single SO-101 arm) so it is true to the video. Work in ${REPO}.
${why ? `Why this pair is being checked: ${why}\n` : ''}${issues ? `Issues a judge reported (verify each against the frames; judges can be wrong):\n- ${issues.join('\n- ')}` : 'Why the pair is being checked and any issues a judge reported are in the element\'s fields why and issues (verify each issue against the frames; judges can be wrong). If issues is empty, check every part against the frames and the dataset notes.'}

${CONTEXT(it)}

Steps:
1. Copy pair.json to pair.orig.json in the same folder, only if pair.orig.json does not exist yet.
2. Look at the frames yourself (always, whatever else you have been told about effort) and fix every real error in the pair: task steps/goals/roles (follow the dataset notes), segments (start/end/grasp/release in frames; segments in step order, non-overlapping, grasp/release inside the segment or null), scene entities/names/attributes/box_2d ([ymin,xmin,ymax,xmax] 0-1000 of the first frame), initial_state (first frame only), bindings (the entity actually manipulated; a bound entity's name and kind equal its role's), and source.outcome (success only if the task's goals hold at the end). Keep ids stable where possible. Change no source field other than outcome; keep the evidence list (you may append one item describing your check). If everything is already right, change nothing.
3. If the robot genuinely fails the task, keep or set outcome "failure" and set genuine_failure=true; never force a success.
4. Validate: cd ${REPO} && uv run --extra schema python -m schema.validate <pair>   (must print ok; fix until it does).
Return what you changed.`

const VERIFY_PROMPT = (it) => `You independently judge one automatically drafted annotation of a robot episode (single SO-101 arm) against its video. Do not edit any file except new frame images or the sheet.

${CONTEXT(it)}

You must draw the sheet and look at it and at the frames yourself; a verdict from the JSON alone, or one that defers to earlier checks, is not acceptable, whatever else you have been told about effort.
Draw the sheet first, then check the frames; extract extra frames where timing or the end state matters.
Judge each part (correct / partial / wrong, concrete issues):
- task_steps: one step per object manipulation, right order/action/object/destination, as the dataset notes describe.
- task_goals: the intended end state (not whether this episode achieved it).
- roles: every participant, sensible kind, short colour+type name.
- segments: each step's [start, end) covers when it actually happens; order right; about 1/12 of the episode of slop allowed.
- scene_entities: task objects, distractors, surface, robot; each separately movable object separate; nothing hallucinated.
- initial_state: facts about the first frame only.
- boxes: boxes enclose their objects in the first frame (hidden objects may lack boxes).
- bindings: each role maps to the entity actually manipulated.
- outcome: source.outcome matches whether the goals hold at the end of the episode.
overall: pass (all correct), minor_issues (only partials that would not mislead a video-generation prompt), fail (anything wrong or misleading). Be strict and do not invent problems.`

const issuesOf = (v) => PARTS.flatMap(p => v[p].verdict === 'correct' ? [] : v[p].issues.map(i => `${p}: ${i}`))
const tag = (it) => `${it.i}`
const fix = (it, issues, why, round) => agent(FIX_PROMPT(it, issues, why), { label: `fix${round}:${tag(it)}`, phase: 'Fix', schema: FIX, model: 'opus', effort: 'medium' })
const verify = (it, round) => agent(VERIFY_PROMPT(it), { label: `verify${round}:${tag(it)}`, phase: 'Verify', schema: VERDICT, model: 'sonnet', effort: 'low' })

const items = Array.from({ length: args.end - args.start }, (_, k) => ({ i: args.start + k, mode: args.start + k < args.nfix ? 'fix' : 'verify' }))
const results = await pipeline(items,
  async (it) => ({ fix1: it.mode === 'fix' ? await fix(it, null, null, 1) : null }),
  async (r, it) => ({ ...r, verify1: await verify(it, 1) }),
  async (r, it) => {
    if (!r.verify1 || r.verify1.overall !== 'fail') return r
    const fix2 = await fix(it, issuesOf(r.verify1), 'an independent judge rated the current annotation fail', 2)
    if (fix2 && fix2.genuine_failure) return { ...r, fix2 }
    return { ...r, fix2, verify2: await verify(it, 2) }
  },
)
const out = results.map((r, i) => r && { i: items[i].i, mode: items[i].mode, ...r, final: r.verify2 || r.verify1 })
const done = out.filter(Boolean)
const c = (v) => done.filter(r => r.final && r.final.overall === v).length
log(`${done.length}/${items.length} done: final pass ${c('pass')}, minor ${c('minor_issues')}, fail ${c('fail')}; second round ${done.filter(r => r.fix2).length}`)
return out
