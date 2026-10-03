export const meta = {
  name: 'so101-petbottle-cap-check',
  description: 'Opus checks each accepted stand-the-bottle-up pair for the bottle cap swapping ends (bottom morphing into the cap)',
  phases: [{ title: 'Check', detail: 'Opus, 3 pairs per agent' }],
}

const REPO = '/Users/desmondzee/Hardware/ICL-so101'
const DIR = `${REPO}/data/so101_curation/review`
const SCHEMA = {
  type: 'object',
  properties: { pairs: { type: 'array', items: { type: 'object', properties: {
    key: { type: 'string' },
    cap_consistent: { type: 'boolean', description: 'the cap stays on the same physical end of the bottle for the whole video' },
    detail: { type: 'string', description: 'where the cap is at the start (lying bottle) and at the end (standing bottle), and any frame where it swaps or morphs' },
  }, required: ['key', 'cap_consistent', 'detail'] } } },
  required: ['pairs'],
}

const prompt = (keys) => `You check generated human videos of a person picking up a lying plastic bottle and standing it upright. Work in ${REPO}; only write under ${DIR}.
A known failure of the video generator: the bottle's cap does not stay on one end. While the bottle is lifted and turned, the bottom morphs into a cap (or the cap end turns into a flat bottom), so the cap ends up on the "wrong" physical end, or the bottle has a cap at both ends at some point.
For each pair below, decide whether the cap stays on the same physical end of the bottle for the whole video: track that end from the first frame (bottle lying, note which side the cap points to) through the lift and rotation to the upright bottle (cap must be the top end that was the cap end before). Extract frames yourself:
  mkdir -p ${DIR}/<key>/cap && ffmpeg -nostdin -loglevel error -y -i ${DIR}/<key>/human.mp4 -vf fps=10,scale=640:-1 ${DIR}/<key>/cap/f_%03d.jpg
and look closely (crop or enlarge if needed) at the frames around the lift and turn. Also see robot_frames.jpg for the real bottle. Say cap_consistent=false only if you actually see the swap or morph; give the frame numbers.
Pairs (key = folder under ${DIR}): ${keys.join(', ')}`

const chunk = (xs, n) => Array.from({ length: Math.ceil(xs.length / n) }, (_, i) => xs.slice(i * n, i * n + n))
const res = await parallel(chunk(args.keys, 3).map((keys, i) => () =>
  agent(prompt(keys), { label: `cap:${i}`, phase: 'Check', schema: SCHEMA, model: 'opus', effort: 'medium' })))
const flat = res.filter(Boolean).flatMap((r) => r.pairs).filter((p) => args.keys.includes(p.key))
log(`${flat.length}/${args.keys.length} checked, ${flat.filter((p) => !p.cap_consistent).length} with the cap swapping ends`)
return Object.fromEntries(flat.map((p) => [p.key, p]))
