// One pair at a time: pick a category, click (or use ← →) through its pairs. The position is kept in the URL hash:
// #<task>/<n> or #all/<n> for training, #val/<task>/<n> or #val/all/<n> for validation, #sim/<task>/<n> or #sim/all/<n>
// for simulated training. Videos stream from the HF bucket.

const SETS = {}
const $ = (id) => document.getElementById(id)
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
const cap = (s) => String(s || '').charAt(0).toUpperCase() + String(s || '').slice(1)
const tasksOf = (d) => d.tasks.filter((t) => t.episodes > 0).sort((a, b) => b.episodes - a.episodes || a.task.localeCompare(b.task))
const pairsOf = (d, task) => task === 'all'
  ? tasksOf(d).flatMap((t) => d.episodes.filter((e) => e.task === t.task))
  : d.episodes.filter((e) => e.task === task)

function state() {
  let h = decodeURIComponent(location.hash.slice(1))
  const pre = h.split('/')[0]
  const set = (pre === 'val' || pre === 'sim') && SETS[pre] ? pre : 'train'
  if (pre === 'val' || pre === 'sim') h = h.slice(4)
  const [task, n] = h.split('/')
  const d = SETS[set]
  const valid = task === 'all' || tasksOf(d).some((t) => t.task === task)
  return { set, task: valid ? task : 'all', n: Math.max(0, parseInt(n || '1', 10) - 1) }
}

function go(set, task, n) {
  const len = pairsOf(SETS[set], task).length
  location.hash = `${set === 'train' ? '' : `${set}/`}${task}/${((n % len) + len) % len + 1}`
}

function render() {
  const { set, task, n } = state()
  const d = SETS[set]
  const eps = pairsOf(d, task)
  const i = Math.min(n, eps.length - 1)
  const ep = eps[i]
  const url = (file) => `${d.base_url}episodes/${ep.id}/${file}`
  $('task').value = `${set}:${task}`
  $('count').textContent = `${i + 1} / ${eps.length}`
  $('title').textContent = cap(ep.instruction)
  const views = Object.keys(ep.views)
  const switcher = views.length > 1
    ? `<div class="views">${views.map((v, k) => `<button aria-pressed="${k === 0}" data-file="${esc(ep.views[v])}">${esc(v)}</button>`).join('')}</div>` : ''
  $('videos').innerHTML = `
    <div class="clip"><span class="tag">Human demonstration</span>
      <video controls playsinline preload="metadata" poster="${url(ep.thumb)}" src="${url(ep.human)}"></video></div>
    <div class="clip"><span class="tag">${set === 'train' ? 'Robot video' : 'Robot video (simulation)'}</span>${switcher}
      <video controls muted playsinline preload="metadata" poster="${url(ep.robot_thumb)}" src="${url(ep.views[views[0]])}"></video></div>`
  const robot = $('videos').querySelectorAll('video')[1]
  $('videos').querySelectorAll('.views button').forEach((b, _, all) => b.addEventListener('click', () => {
    all.forEach((x) => x.setAttribute('aria-pressed', x === b))
    robot.src = url(b.dataset.file)
    robot.play().catch(() => {})
  }))
  $('caption').textContent = `${{ val: 'Validation (simulation) · ', sim: `Training (simulation) · ${(ep.family || '').replace(/_/g, ' ')} · ` }[set] || ''}Episode ${ep.curated_episode_index ?? ep.episode} · human ${ep.human_duration_s.toFixed(1)} s · robot ${ep.robot_duration_s.toFixed(1)} s`
}

const load = (path) => fetch(path).then((r) => (r.ok ? r.json() : null)).catch(() => null)

Promise.all([load('data/index.json'), load('data/val_index.json'), load('data/sim_index.json')]).then(([train, val, sim]) => {
  if (!train) { $('title').textContent = 'Could not load the dataset index.'; return }
  SETS.train = train
  if (val) SETS.val = val
  if (sim && sim.episodes && sim.episodes.length) SETS.sim = sim
  const NOUN = { train: 'training', val: 'validation', sim: 'simulated training' }
  const options = (set, d, label) => `<optgroup label="${label}"><option value="${set}:all">All ${NOUN[set]} pairs (${d.episodes_total})</option>` +
    tasksOf(d).map((t) => `<option value="${set}:${esc(t.task)}">${esc(cap(t.instruction))} (${t.episodes})</option>`).join('') + '</optgroup>'
  $('task').innerHTML = options('train', train, 'Training (real)') + (val ? options('val', val, 'Validation (simulation)') : '') +
    (SETS.sim ? options('sim', sim, 'Training (simulation)') : '')
  $('task').addEventListener('change', (e) => { const [set, task] = e.target.value.split(':'); go(set, task, 0) })
  $('prev').addEventListener('click', () => { const s = state(); go(s.set, s.task, s.n - 1) })
  $('next').addEventListener('click', () => { const s = state(); go(s.set, s.task, s.n + 1) })
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.metaKey || e.ctrlKey) return
    if (e.key === 'ArrowLeft') { const s = state(); go(s.set, s.task, s.n - 1) }
    if (e.key === 'ArrowRight') { const s = state(); go(s.set, s.task, s.n + 1) }
  })
  window.addEventListener('hashchange', render)
  render()
})
