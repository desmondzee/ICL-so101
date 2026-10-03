// One-page viewer for the curated SO-101 HumanGen pairs. The index ships with the site; videos and thumbnails stream from
// the public HF bucket (index.base_url). The selected task is kept in the URL hash (#<task>).

const PAGE = 6  // pairs shown before "Show more" (three rows of two)
let DATA = null
let shown = PAGE

const $ = (id) => document.getElementById(id)
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
const cap = (s) => String(s || '').charAt(0).toUpperCase() + String(s || '').slice(1)
// Short tab names; the full instruction is shown above each task's pairs.
const LABELS = {
  'aiden-li__so101-close-upper-drawer': 'Close drawer', 'aiden-li__so101-open-upper-drawer': 'Open drawer',
  'aiden-li__so101-close-lower-drawer': 'Close lower drawer', 'aiden-li__so101-open-lower-drawer': 'Open lower drawer',
  'aiden-li__so101-grabtissue': 'Grab tissue', 'b3rnd__record-50-episodes': 'Stack cube', 'chirag1701__can': 'Can in box',
  'chirag1701__sandwich': 'Sandwich in box', 'EverNorif__so101-table-cleanup': 'Pens in holder', 'k1000dai__standup_petbottle': 'Stand bottle up',
  'lerobot__svla_so101_pickplace': 'Lego in box', 'pbvr__so101_test002': 'Block in box', 'pbvr__so101_test005': 'Three blocks in box',
  'psg777__combinedtape': 'Move tape roll', 'ReubenLim__so101_tape_in_square': 'Push to square', 'ricky0526__so101_pick_toy_to_plate_v3': 'Toy on plate',
  'sattgle__clean-test': 'Slippers on shelf', 'sattgle__clean2-test': 'Two pairs of slippers', 'stsqitx__clean': 'Wipe table',
  'tenkau__SO101-Stack3Blocks': 'Stack three blocks', 'un1c0rnio__so101_sock_stowing_3pair2': 'Stow sock', 'Cornito__so101_tea2': 'Make tea',
  'tenkau__so101_color_block': 'Stack by colour', 'Tear4Pixelation__lego2': 'Stack lego', 'Rorschach4153__so101_30_fold': 'Fold cloth',
  'fbeltrao__so101_unplug_cable_4': 'Unplug cable', 'LeRobot-worldwide-hackathon__27-AI_Learners-Shape_Pick_and_Place': 'Shape sorter',
  'LeRobot-worldwide-hackathon__91-AM-PM-pouring-liquid': 'Pour liquid',
}
const short = (task) => LABELS[task] || task.replace(/^[^_]+__/, '').replace(/[_-]+/g, ' ')
const url = (ep, file) => `${DATA.base_url}episodes/${ep.id}/${file}`
const tasks = () => DATA.tasks.filter((t) => t.episodes > 0).sort((a, b) => b.episodes - a.episodes || a.task.localeCompare(b.task))

function stats() {
  const mins = DATA.episodes.reduce((s, e) => s + e.robot_duration_s, 0) / 60
  const wrist = DATA.episodes.filter((e) => e.views.wrist).length
  $('stats').textContent = `${DATA.episodes_total} pairs · ${tasks().length} tasks · ${Math.round(mins)} min of robot data · ` +
    `front camera on every pair, wrist camera on ${wrist}`
}

function tabs(current) {
  $('tabs').innerHTML = tasks().map((t) =>
    `<button class="tab" role="tab" aria-selected="${t.task === current}" data-task="${esc(t.task)}">${esc(short(t.task))}<span class="n">${t.episodes}</span></button>`).join('')
  $('tabs').querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => { location.hash = b.dataset.task }))
  const sel = $('tabs').querySelector('[aria-selected="true"]')  // keep it in view when the row scrolls (phones)
  if (sel) $('tabs').scrollLeft = sel.offsetLeft - $('tabs').offsetLeft - 16
}

const eps_index = (ep) => DATA.episodes.filter((e) => e.task === ep.task).indexOf(ep) + 1

function pairHTML(ep) {
  const views = Object.keys(ep.views)
  const switcher = views.length > 1
    ? `<div class="views">${views.map((v, i) => `<button aria-pressed="${i === 0}" data-file="${esc(ep.views[v])}">${esc(v)}</button>`).join('')}</div>` : ''
  return `<article class="pair" data-id="${esc(ep.id)}">
    <div class="videos">
      <div class="clip"><span class="tag">Human demonstration</span>
        <video controls playsinline preload="none" poster="${url(ep, ep.thumb)}" src="${url(ep, ep.human)}"></video></div>
      <div class="clip"><span class="tag">Robot video</span>${switcher}
        <video controls muted playsinline preload="none" poster="${url(ep, ep.robot_thumb)}" src="${url(ep, ep.views[views[0]])}"></video></div>
    </div>
    <div class="caption"><span>Episode ${ep.curated_episode_index} · human ${ep.human_duration_s.toFixed(1)} s · robot ${ep.robot_duration_s.toFixed(1)} s</span>
      <a href="viewer.html#${esc(ep.task)}/${eps_index(ep)}">open in viewer</a></div>
  </article>`
}

function show(task) {
  const t = DATA.tasks.find((x) => x.task === task && x.episodes > 0) || tasks()[0]
  const eps = DATA.episodes.filter((e) => e.task === t.task)
  tabs(t.task)
  $('task-head').innerHTML = `<h3>${esc(cap(t.instruction))}</h3><span class="meta">${eps.length} pairs · robot views: ${esc(t.views.join(', '))}</span>`
  $('pairs').innerHTML = eps.slice(0, shown).map(pairHTML).join('')
  $('more').innerHTML = eps.length > shown ? `<button id="more-btn">Show ${eps.length - shown} more</button>` : ''
  if (eps.length > shown) $('more-btn').addEventListener('click', () => { shown = eps.length; show(t.task) })
  // Front / wrist switch: swap the robot video's source and keep playing.
  $('pairs').querySelectorAll('.pair').forEach((card) => {
    const ep = eps.find((e) => e.id === card.dataset.id)
    const video = card.querySelectorAll('video')[1]
    card.querySelectorAll('.views button').forEach((b, _, all) => b.addEventListener('click', () => {
      all.forEach((x) => x.setAttribute('aria-pressed', x === b))
      video.src = url(ep, b.dataset.file)
      video.play().catch(() => {})
    }))
  })
}

function table() {
  const rows = [...DATA.tasks].sort((a, b) => b.episodes - a.episodes || a.task.localeCompare(b.task))
  $('table').innerHTML = `<thead><tr><th>Task</th><th>Pairs</th><th>Accepted / reviewed</th><th>Generated</th></tr></thead><tbody>` +
    rows.map((t) => `<tr class="${t.episodes ? '' : 'empty'}"><td>${esc(cap(t.instruction))}${t.excluded ? ' <span class="note">(being regenerated)</span>' : ''}</td><td class="num">${t.episodes}</td>` +
      `<td class="num">${t.accepted} / ${t.reviewed}</td><td class="num">${t.available_demos}</td></tr>`).join('') + '</tbody>'
}

function route() { shown = PAGE; show(decodeURIComponent(location.hash.slice(1))) }

fetch('data/index.json').then((r) => r.json()).then((d) => {
  DATA = d
  stats(); table(); route()
  window.addEventListener('hashchange', route)
}).catch(() => { $('pairs').innerHTML = '<p>Could not load the dataset index.</p>' })
