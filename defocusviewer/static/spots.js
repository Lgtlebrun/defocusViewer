// The Spots subtab: full field of view, filters, timeline and table of the spots on the left, one spot (metadata,
// fit, images) on the right.
import { h, api, post, q, toast, fmt, getHash, setHash, cssVar } from './ui.js';
import { VIEWS, loadOptions, arrays, viewValues, range, heatmap, zoomGroup, viewToggles } from './views.js';
import { addPin } from './compare.js';
import { jointFitModal } from './analyses.js';

export const ROUTINE_ORDER = ['train_burnin', 'train', 'train2', 'stability', 'detection', 'positioner', 'train_end'];
const SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const SYMBOLS = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'pentagon', 'hexagon', 'star'];
const MAX_ROWS = 400;

let table = null;        // {run, rows: [{img, fiber, ...}], fits, joint_fit, ghost_masks}
let filters = { fibers: new Set(), routines: new Set(), fit: 'all' };
let sort = { column: 'timestamp', descending: false };
let els = {};
const options = loadOptions();

export function routineStyle(routine, all) {
  const known = ROUTINE_ORDER.indexOf(routine);
  const extra = all.filter((r) => !ROUTINE_ORDER.includes(r)).sort();
  const slot = known >= 0 ? known : ROUTINE_ORDER.length + extra.indexOf(routine);
  return slot < SERIES_COLORS.length ? { color: SERIES_COLORS[slot], symbol: SYMBOLS[slot] } : { color: '#9b9a95', symbol: 'circle' };
}

export function ghostSetting() {
  try { return localStorage.getItem('ghost') || 'on'; } catch { return 'on'; }
}

function setGhostSetting(value) {
  try { localStorage.setItem('ghost', value); } catch { /* storage blocked */ }
}

export async function renderSpots(body, info) {
  if (body) {
    els = { body };
    if (!table || table.run !== info.key) {
      body.replaceChildren(h('p', { class: 'pad muted' }, 'loading spots…'));
      const data = await api(`/api/run/spots?${q({ run: info.key })}`);
      const rows = data.rows.map((r) => Object.fromEntries(data.columns.map((c, i) => [c, r[i]])));
      table = { run: info.key, info, rows, fits: data.fits, joint_fit: data.joint_fit, ghost_masks: data.ghost_masks };
      filters = { fibers: new Set(), routines: new Set(), fit: 'all' };
    }
    els.left = h('div', { class: 'spots-left' });
    els.right = h('div', { class: 'spots-right' });
    body.replaceChildren(h('div', { class: 'spots' }, els.left, els.right));
    renderLeft();
  }
  const img = getHash().img;
  if (img) showSpot(img);
  else els.right.replaceChildren(h('p', { class: 'muted' }, 'Click a spot in the timeline or the table (arrow keys move through the filtered list).'));
}

function filtered() {
  const out = table.rows.filter((r) => (!filters.fibers.size || filters.fibers.has(r.fiber))
    && (!filters.routines.size || filters.routines.has(r.routine))
    && (filters.fit === 'all' || (filters.fit === 'fitted') === (table.fits[r.img]?.length > 0)));
  const { column, descending } = sort;
  out.sort((a, b) => (a[column] < b[column] ? -1 : a[column] > b[column] ? 1 : 0) * (descending ? -1 : 1));
  return out;
}

function chips(values, selected, onToggle, style) {
  return h('div', { class: 'chips' }, values.map((value) => h('button', {
    class: `chip${selected.has(value) ? ' on' : ''}`,
    style: style ? `border-left: 4px solid ${style(value).color}` : null,
    onclick: () => { selected.has(value) ? selected.delete(value) : selected.add(value); onToggle(); },
  }, value)));
}

function renderLeft() {
  const fibers = [...new Set(table.rows.map((r) => r.fiber))].sort();
  const rank = (r) => (ROUTINE_ORDER.includes(r) ? ROUTINE_ORDER.indexOf(r) : 100);
  const routines = [...new Set(table.rows.map((r) => r.routine))].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  const refresh = () => { renderLeft(); };
  const fitSelect = h('select', {}, ['all', 'fitted', 'not fitted'].map((v) => h('option', { value: v, selected: filters.fit === v }, v)));
  fitSelect.onchange = () => { filters.fit = fitSelect.value === 'not fitted' ? 'unfitted' : fitSelect.value; refresh(); };
  const list = filtered();
  els.timeline = h('div', { style: 'height:230px' });
  els.tableWrap = h('div', { class: 'table-wrap' });
  els.left.replaceChildren(
    fovView(fibers),
    h('div', {}, h('div', { class: 'small muted' }, 'fibers'), chips(fibers, filters.fibers, refresh)),
    h('div', {}, h('div', { class: 'small muted' }, 'routines'), chips(routines, filters.routines, refresh, (r) => routineStyle(r, routines))),
    h('div', { class: 'row small' }, 'fit ', fitSelect, h('span', { class: 'spacer' }),
      h('span', { class: 'muted' }, `${list.length} of ${table.rows.length} spots`),
      table.joint_fit ? null : h('span', { class: 'badge warn' }, 'no joint fit')),
    els.timeline, els.tableWrap);
  drawTimeline(list, routines);
  drawTable(list);
}

function fovView(fibers) {
  const box = h('div', { class: 'fov' });
  const img = h('img', { src: `/api/run/fov.png?${q({ run: table.run })}`, alt: 'full field of view' });
  img.onerror = () => box.replaceChildren(h('div', { class: 'small muted' }, 'no full-frame image (results/equalizer_raw_image.npy)'));
  box.append(img);
  api(`/api/run/fov?${q({ run: table.run })}`).then((fov) => {
    if (!fov.shape) return;
    const [height, width] = fov.shape;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    for (const f of fov.fibers) {
      const on = filters.fibers.has(f.fiber);
      const toggle = () => { on ? filters.fibers.delete(f.fiber) : filters.fibers.add(f.fiber); renderLeft(); };
      const circle = document.createElementNS(svg.namespaceURI, 'circle');
      Object.entries({ cx: f.x, cy: f.y, r: 70, class: on ? 'on' : '' }).forEach(([k, v]) => circle.setAttribute(k, v));
      circle.addEventListener('click', toggle);
      const label = document.createElementNS(svg.namespaceURI, 'text');
      Object.entries({ x: f.x + 85, y: f.y + 20, 'font-size': 90 }).forEach(([k, v]) => label.setAttribute(k, v));
      label.textContent = f.fiber;
      label.addEventListener('click', toggle);
      svg.append(circle, label);
    }
    box.append(svg);
  }).catch(() => {});
  return box;
}

function drawTimeline(list, routines) {
  const byRoutine = {};
  for (const r of list) (byRoutine[r.routine] ||= []).push(r);
  const traces = routines.filter((r) => byRoutine[r]).map((routine) => {
    const rs = byRoutine[routine];
    const style = routineStyle(routine, routines);
    return {
      type: 'scattergl', mode: 'markers', name: routine, x: rs.map((r) => r.timestamp), y: rs.map((r) => r.fiber),
      customdata: rs.map((r) => [r.img, r.x_position, (table.fits[r.img] || []).join(', ') || 'not fitted']),
      marker: { color: style.color, size: 8, symbol: rs.map((r) => (table.fits[r.img]?.length ? style.symbol : `${style.symbol}-open`)),
        line: { width: 1, color: style.color } },
      hovertemplate: '%{customdata[0]}<br>x = %{customdata[1]} um<br>%{x}<br>%{customdata[2]}<extra>' + routine + '</extra>',
    };
  });
  const axis = { color: cssVar('--text-secondary'), gridcolor: cssVar('--border'), zeroline: false };
  Plotly.react(els.timeline, traces, {
    margin: { l: 50, r: 8, t: 26, b: 30 }, paper_bgcolor: cssVar('--surface-0'), plot_bgcolor: cssVar('--surface-1'),
    font: { size: 11, color: cssVar('--text-secondary') }, legend: { orientation: 'h', y: 1.15, x: 0 },
    xaxis: { ...axis, type: 'date' }, yaxis: { ...axis, type: 'category', categoryorder: 'category descending' },
  }, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['select2d', 'lasso2d'] });
  els.timeline.on('plotly_click', (event) => {
    const img = event.points?.[0]?.customdata?.[0];
    if (img) openSpot(img);
  });
}

const COLUMNS = [['fiber', 'fiber'], ['routine', 'routine'], ['x_position', 'x (um)'], ['timestamp', 'time'], ['texp', 'texp (us)'], ['fit', 'fit']];

function drawTable(list) {
  const current = getHash().img;
  const head = h('tr', {}, COLUMNS.map(([key, label]) => h('th', {
    onclick: () => { sort = { column: key, descending: sort.column === key ? !sort.descending : false }; renderLeft(); },
  }, label, sort.column === key ? (sort.descending ? ' ▾' : ' ▴') : '')));
  const rows = list.slice(0, MAX_ROWS).map((r) => h('tr', { class: r.img === current ? 'selected' : '', 'data-img': r.img, onclick: () => openSpot(r.img) },
    h('td', {}, r.fiber), h('td', {}, r.routine), h('td', {}, r.x_position), h('td', { title: r.timestamp }, (r.timestamp || '').slice(11, 19)),
    h('td', {}, r.texp), h('td', { class: 'small' }, (table.fits[r.img] || []).join(', '))));
  els.tableWrap.replaceChildren(h('table', { class: 'grid' }, h('thead', {}, head), h('tbody', {}, rows)));
  if (list.length > MAX_ROWS) els.tableWrap.append(h('div', { class: 'small muted pad' }, `${list.length - MAX_ROWS} more: narrow the filters, or use the timeline.`));
  els.tableWrap.querySelector('tr.selected')?.scrollIntoView({ block: 'nearest' });
}

function openSpot(img) {
  setHash({ img });
  for (const tr of els.tableWrap.querySelectorAll('tr.selected')) tr.classList.remove('selected');
  els.tableWrap.querySelector(`tr[data-img="${CSS.escape(img)}"]`)?.classList.add('selected');
  showSpot(img);
}

function step(offset) {
  const list = filtered();
  const i = list.findIndex((r) => r.img === getHash().img);
  const next = list[Math.min(list.length - 1, Math.max(0, i + offset))];
  if (next) {
    openSpot(next.img);
    els.tableWrap.querySelector('tr.selected')?.scrollIntoView({ block: 'nearest' });
  }
}

document.addEventListener('keydown', (event) => {
  if (!els.right?.isConnected || ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) return;
  if (event.key === 'ArrowDown' || event.key === 'ArrowRight') { step(1); event.preventDefault(); }
  if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') { step(-1); event.preventDefault(); }
});

let request = 0;

async function showSpot(img) {
  const mine = ++request;
  const ghost = ghostSetting();
  els.right.replaceChildren(h('p', { class: 'muted' }, h('span', { class: 'spinner' }), ' loading ', img));
  let detail;
  try {
    detail = await api(`/api/spot?${q({ run: table.run, img, ghost })}`);
  } catch (e) {
    els.right.replaceChildren(h('p', { class: 'banner warn' }, e.message));
    return;
  }
  if (mine !== request) return;
  renderSpot(detail);
  if (detail.needs === 'refit') refit(detail, mine);
}

async function refit(detail, mine) {
  try {
    const fitted = await post('/api/spot/refit', { run: table.run, img: detail.img, ghost: detail.ghost });
    if (!table.fits[detail.img]?.includes('viewer')) (table.fits[detail.img] ||= []).push('viewer');
    if (mine === request) renderSpot(fitted);
  } catch (e) {
    if (mine === request) toast(`refit of ${detail.img}: ${e.message}`, true);
  }
}

function fitBanner(detail) {
  if (detail.needs === 'joint_fit') {
    return h('div', { class: 'banner warn' }, 'No joint fit on this run yet: theta and the pupil scales it fits are needed to model any spot. ',
      h('button', { class: 'primary', onclick: () => jointFitModal(table.info, ghostSetting()) }, 'Run the joint fit…'));
  }
  if (detail.needs === 'refit') {
    return h('div', { class: 'banner' }, h('span', { class: 'spinner' }), ` No saved fit of this spot with the ghost mask ${detail.ghost}: fitting it now with the joint fit's theta and pupil scale (a few seconds).`);
  }
  const fit = detail.fit;
  const ghostNote = fit.ghost_inferred ? ' (inferred from log.log)' : '';
  return h('div', { class: 'banner' }, h('b', {}, fit.source), ` · ghost mask ${fit.ghost}${ghostNote}`,
    fit.note ? h('div', { class: 'small muted' }, fit.note) : null);
}

function paramTable(detail) {
  const fits = [...detail.saved.map((f) => ({ ...f, label: `${f.source} (ghost ${f.ghost})` }))];
  if (detail.viewer_fit) fits.push({ ...detail.viewer_fit, label: `viewer fit (ghost ${detail.viewer_fit.ghost})` });
  if (!fits.length) return h('div', { class: 'small muted' }, 'no fit');
  const noll = [...new Set(fits.flatMap((f) => f.noll.slice(1)))].sort((a, b) => a - b);
  const columns = ['r', 'x0', 'y0', 'c4', ...noll.map((j) => `c${j}`)];
  const value = (f, c) => {
    const i = c.startsWith('c') && c !== 'c4' ? 4 + f.noll.slice(1).indexOf(Number(c.slice(1))) : ['r', 'x0', 'y0', 'c4'].indexOf(c);
    return i >= 0 && (i < 4 || f.noll.slice(1).includes(Number(c.slice(1)))) ? fmt(f.params[i]) : '';
  };
  return h('div', { style: 'overflow:auto' }, h('table', { class: 'grid' },
    h('thead', {}, h('tr', {}, h('th', {}, 'fit'), columns.map((c) => h('th', {}, c)), h('th', {}, 'residual_frac'), h('th', {}, 'delta_z (um)'))),
    h('tbody', {}, fits.map((f) => h('tr', { class: detail.fit && f.source === detail.fit.source && f.ghost === detail.fit.ghost ? 'selected' : '' },
      h('td', {}, f.label), columns.map((c) => h('td', { class: 'mono' }, value(f, c))),
      h('td', { class: 'mono' }, fmt(f.residual_frac)), h('td', { class: 'mono' }, f.delta_z !== undefined ? `${fmt(f.delta_z)} ± ${fmt(f.sigma_delta_z, 2)}` : ''))))));
}

function metadata(detail) {
  const m = detail.meta;
  const rows = [
    ['fiber', m.fiber], ['routine', m.routine], ['timestamp', m.timestamp], ['f-number', m.fnum], ['exposure', `${m.texp} us`],
    ['frames averaged', m.n_frames], ['stage position', `${m.x_position} um (${m.stage_counts} encoder counts)`], ['crop centre (sensor px)', `${m.mom_center_x}, ${m.mom_center_y}`],
    ['spot centre (sensor px)', detail.absolute_center ? detail.absolute_center.map((v) => v.toFixed(2)).join(', ') : 'needs a fit'],
    ['crop side', `${m.crop_side} px`], ['device', m.device], ['image', detail.img],
    ['pupil scale', detail.pupil_scale !== undefined ? fmt(detail.pupil_scale, 5) : ''],
    ['ghost box', detail.ghost_box ? `x ${detail.ghost_box.x0}, y ${detail.ghost_box.y0}, side ${detail.ghost_box.side} px` : 'none'],
    ['fit available', detail.saved.length ? detail.saved.map((f) => `${f.source} (ghost ${f.ghost})`).join('; ') : 'no pipeline fit'],
  ];
  return h('dl', { class: 'kv' }, rows.map(([k, v]) => [h('dt', {}, k), h('dd', {}, v ?? '')]));
}

function foldable(key, title, content, openByDefault) {
  let open = openByDefault;
  try { open = JSON.parse(localStorage.getItem(`fold-${key}`) ?? String(openByDefault)); } catch { /* storage blocked */ }
  const details = h('details', { open }, h('summary', {}, title), content);
  details.addEventListener('toggle', () => { try { localStorage.setItem(`fold-${key}`, details.open); } catch { /* blocked */ } });
  return details;
}

function renderSpot(detail) {
  const views = h('div', { class: 'views' });
  const ghostSelect = h('select', { disabled: !detail.ghost_masks, title: detail.ghost_masks ? '' : 'this run has no ghost masks (ghost_mask step)' },
    h('option', { value: 'on', selected: detail.ghost === 'on' }, 'on'), h('option', { value: 'off', selected: detail.ghost === 'off' }, 'off'));
  ghostSelect.onchange = () => { setGhostSetting(ghostSelect.value); showSpot(detail.img); };
  const pin = () => { addPin({ run: table.run, img: detail.img, label: `${detail.meta.fiber} ${detail.meta.routine} x=${detail.meta.x_position} ${(detail.meta.timestamp || '').slice(11, 19)}` }); toast('Pinned: see the Compare tab'); };
  const refitButton = h('button', { disabled: !detail.joint_fit, title: 'fit this spot again here, with the ghost setting above',
    onclick: () => { refitButton.disabled = true; refitButton.replaceChildren(h('span', { class: 'spinner' }), ' fitting'); refit(detail, request); } }, 'Refit here');
  els.right.replaceChildren(
    h('div', { class: 'viewer-head' }, h('h3', {}, detail.img), h('span', { class: 'spacer' }),
      h('button', { onclick: () => step(-1), title: 'previous (↑)' }, '◀'), h('button', { onclick: () => step(1), title: 'next (↓)' }, '▶'),
      h('button', { onclick: pin }, 'Pin to compare'), refitButton),
    h('div', { class: 'row small', style: 'margin-top:4px' }, 'ghost mask in the fit ', ghostSelect,
      detail.ghost_masks ? null : h('span', { class: 'muted' }, 'no ghost masks on this run (run the ghost_mask step in Analyses)'),
      detail.residual_frac !== undefined ? h('span', { class: 'muted' }, `residual_frac ${fmt(detail.residual_frac)}`) : null),
    fitBanner(detail),
    foldable('meta', 'Metadata', metadata(detail), false),
    foldable('fit', 'Fit parameters', paramTable(detail), true),
    h('div', { class: 'card', style: 'margin-top:6px;padding:6px 10px' }, viewToggles(options, () => drawViews(detail, views))),
    views);
  drawViews(detail, views);
}

function drawViews(detail, container) {
  const images = arrays(detail);
  const group = zoomGroup();
  group.enabled = options.syncZoom;
  const drawn = [];
  for (const view of VIEWS.filter((v) => options.views.includes(v.key))) {
    const values = viewValues(view, images, options, detail.ghost_box);
    if (values) drawn.push({ view, values, div: h('div', { class: 'plot' }) });
  }
  container.replaceChildren(...drawn.map(({ view, div }) => h('div', { class: 'panel' }, h('div', { class: 'title' }, view.label), div)));
  if (!drawn.length) container.append(h('p', { class: 'muted' }, detail.needs ? 'Only the data until the spot is fitted.' : 'Tick a view above.'));
  for (const { view, values, div } of drawn) {
    heatmap(div, view, values, range(values.values, values.diverging), detail.ghost_box, options).then(() => group.add(div, view.domain));
  }
}
