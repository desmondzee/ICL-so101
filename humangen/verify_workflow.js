export const meta = {
  name: 'humangen-clip-verify',
  description: 'Second opinion on each kept human video: one Opus agent per episode re-judges it independently against the robot frames and pair.json and looks for problems the first judge missed',
  phases: [{ title: 'Verify', detail: 'one Opus agent per kept video' }],
}

const REPO = '/Users/desmondzee/Hardware/ICL-so101'
const STATUS = { type: 'object', properties: { status: { enum: ['pass', 'fail', 'unknown'] }, detail: { type: 'string' } }, required: ['status', 'detail'] }
const SCORE = { type: 'object', properties: { score: { type: 'integer', minimum: 1, maximum: 5 }, detail: { type: 'string' } }, required: ['score', 'detail'] }
const SCHEMA = {
  type: 'object',
  properties: {
    dir: { type: 'string' },
    reference_preservation: STATUS, first_frame_visible: STATUS, task_consistency: STATUS,
    semantic: { type: 'object', properties: { score: { type: 'integer', minimum: 1, maximum: 5 }, goals: { type: 'array', items: { enum: ['pass', 'fail', 'unknown'] } }, detail: { type: 'string' } }, required: ['score', 'goals', 'detail'] },
    physics: SCORE,
    agrees_with_first_judge: { type: 'boolean' },
    disagreements: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['dir', 'reference_preservation', 'first_frame_visible', 'task_consistency', 'semantic', 'physics', 'agrees_with_first_judge', 'disagreements', 'notes'],
}

const PROMPT = (dir) => `You are the second, independent verifier of one generated human video that must show a person doing the same task a robot did. Work in ${REPO}. Do not edit any file except extracted frames under ${dir}/verify/.
Files in ${dir}: robot_first.jpg and robot_kf*.jpg (robot frames at the start and after each step), frame.jpg and kf*.jpg (edited human frames at the same moments), video.mp4 (the video, 24 fps), pair.json (task.instruction, task.steps, task.goals, generation.idle_hand), judge.json (the first judge's verdict).
Look at the video yourself: extract frames densely, e.g. mkdir -p ${dir}/verify && ffmpeg -nostdin -loglevel error -y -i ${dir}/video.mp4 -vf fps=4 ${dir}/verify/f_%03d.jpg, and view them, zooming into hands and task objects.
Try hard to find problems: extra or changing hands (if generation.idle_hand says the other hand is out of frame, exactly one hand may ever be visible), objects appearing, vanishing, morphing, duplicating or moving without contact, passing through each other, surfaces or lighting changing, jumps at the joins between step clips, a wrong end state compared with the robot's last frame, steps in the wrong order or on the wrong object.
Judge: reference_preservation, first_frame_visible, task_consistency (pass/fail); semantic 1-5 with goals pass/fail/unknown per task.goals entry; physics 1-5 (more hands than allowed at any point caps physics at 2 and fails task_consistency).
Then compare with judge.json: agrees_with_first_judge = same accept decision (semantic 5, physics >= 3, the three checks pass) and no material disagreement; list concrete disagreements, citing frames. Do not invent problems.`

const dirs = args.dirs
const results = await parallel(dirs.map((d, i) => () =>
  agent(PROMPT(d), { label: `verify:${i}`, phase: 'Verify', schema: SCHEMA, model: 'opus', effort: 'medium' })))
const ok = results.filter(Boolean)
const acc = (r) => r.semantic.score === 5 && r.physics.score >= 3 && ['reference_preservation', 'first_frame_visible', 'task_consistency'].every(k => r[k].status === 'pass')
log(`${ok.length}/${dirs.length} verified, ${ok.filter(acc).length} accepted, ${ok.filter(r => r.agrees_with_first_judge).length} agree with the first judge`)
return ok
