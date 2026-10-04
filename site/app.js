// One-page viewer for the SO-101 HumanGen pairs: the real training set and the simulated validation set. The indexes ship
// with the site; videos and thumbnails stream from the public HF bucket (each index's base_url). The selected task of
// each gallery is kept in the URL hash (#<task> for training, #val/<task> for validation).

const PAGE = 6  // pairs shown before "Show more" (three rows of two)

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
  sort_blocks: 'Sort blocks', stack_bowls: 'Stack bowls', mug_on_plate: 'Mug on plate', mugs_in_microwave: 'Mugs in microwave', pan_on_stove: 'Pan on stove',
}
const short = (task) => LABELS[task] || task.replace(/^[^_]+__/, '').replace(/[_-]+/g, ' ')

// A gallery: task tabs, a heading and the pairs of the selected task. prefix is '' (training) or 'val/' (validation).
function gallery(data, ids, prefix) {
  let shown = PAGE
  const url = (ep, file) => `${data.base_url}episodes/${ep.id}/${file}`
  const tasks = data.tasks.filter((t) => t.episodes > 0).sort((a, b) => b.episodes - a.episodes || a.task.localeCompare(b.task))
  const viewerIndex = (ep) => data.episodes.filter((e) => e.task === ep.task).indexOf(ep) + 1
  const label = (ep) => ep.curated_episode_index ?? ep.episode

  const pairHTML = (ep) => {
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
      <div class="caption"><span>Episode ${label(ep)} · human ${ep.human_duration_s.toFixed(1)} s · robot ${ep.robot_duration_s.toFixed(1)} s</span>
        <a href="viewer.html#${prefix}${esc(ep.task)}/${viewerIndex(ep)}">open in viewer</a></div>
    </article>`
  }

  const show = (task) => {
    const t = tasks.find((x) => x.task === task) || tasks[0]
    const eps = data.episodes.filter((e) => e.task === t.task)
    $(ids.tabs).innerHTML = tasks.map((x) =>
      `<button class="tab" role="tab" aria-selected="${x.task === t.task}" data-task="${esc(x.task)}">${esc(short(x.task))}<span class="n">${x.episodes}</span></button>`).join('')
    $(ids.tabs).querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => { location.hash = prefix + b.dataset.task }))
    const sel = $(ids.tabs).querySelector('[aria-selected="true"]')  // keep it in view when the row scrolls (phones)
    if (sel) $(ids.tabs).scrollLeft = sel.offsetLeft - $(ids.tabs).offsetLeft - 16
    const instr = t.instructions && t.instructions.length > 1 ? `${t.instruction} (the plate varies per episode)` : t.instruction
    $(ids.head).innerHTML = `<h3>${esc(cap(instr))}</h3><span class="meta">${eps.length} pairs · robot views: ${esc(t.views.join(', '))}</span>`
    $(ids.pairs).innerHTML = eps.slice(0, shown).map(pairHTML).join('')
    $(ids.more).innerHTML = eps.length > shown ? `<button>Show ${eps.length - shown} more</button>` : ''
    if (eps.length > shown) $(ids.more).querySelector('button').addEventListener('click', () => { shown = eps.length; show(t.task) })
    // Front / wrist switch: swap the robot video's source and keep playing.
    $(ids.pairs).querySelectorAll('.pair').forEach((card) => {
      const ep = eps.find((e) => e.id === card.dataset.id)
      const video = card.querySelectorAll('video')[1]
      card.querySelectorAll('.views button').forEach((b, _, all) => b.addEventListener('click', () => {
        all.forEach((x) => x.setAttribute('aria-pressed', x === b))
        video.src = url(ep, b.dataset.file)
        video.play().catch(() => {})
      }))
    })
  }
  return { show, reset: () => { shown = PAGE } }
}

function stats(train, val) {
  const mins = train.episodes.reduce((s, e) => s + e.robot_duration_s, 0) / 60
  const wrist = train.episodes.filter((e) => e.views.wrist).length
  $('stats').textContent = `${train.episodes_total} training pairs · ${train.tasks_total} tasks · ${Math.round(mins)} min of robot data · ` +
    `front camera on every pair, wrist camera on ${wrist}` + (val ? ` · ${val.episodes_total} validation pairs over ${val.tasks_total} simulated tasks` : '')
}

function table(data) {
  const rows = [...data.tasks].sort((a, b) => b.episodes - a.episodes || a.task.localeCompare(b.task))
  $('table').innerHTML = `<thead><tr><th>Task</th><th>Pairs</th><th>Accepted / reviewed</th><th>Generated</th></tr></thead><tbody>` +
    rows.map((t) => `<tr class="${t.episodes ? '' : 'empty'}"><td>${esc(cap(t.instruction))}${t.excluded ? ' <span class="note">(being regenerated)</span>' : ''}</td><td class="num">${t.episodes}</td>` +
      `<td class="num">${t.accepted} / ${t.reviewed}</td><td class="num">${t.available_demos}</td></tr>`).join('') + '</tbody>'
}

const load = (path) => fetch(path).then((r) => (r.ok ? r.json() : null)).catch(() => null)

Promise.all([load('data/index.json'), load('data/val_index.json')]).then(([train, val]) => {
  if (!train) { $('pairs').innerHTML = '<p>Could not load the dataset index.</p>'; return }
  const g = gallery(train, { tabs: 'tabs', head: 'task-head', pairs: 'pairs', more: 'more' }, '')
  const v = val ? gallery(val, { tabs: 'val-tabs', head: 'val-head', pairs: 'val-pairs', more: 'val-more' }, 'val/') : null
  if (!val) $('validation').hidden = true
  stats(train, val); table(train)
  const route = () => {
    const h = decodeURIComponent(location.hash.slice(1))
    g.reset(); if (v) v.reset()
    g.show(h.startsWith('val/') ? '' : h)
    if (v) v.show(h.startsWith('val/') ? h.slice(4) : '')
    if (h.startsWith('val/')) $('validation').scrollIntoView()
  }
  route()
  window.addEventListener('hashchange', route)
})
