// The Fibers subtab: the fitted quantities of every spot against time or stage position, drawn by the server with
// matplotlib (house style): one row per quantity, one colour per fiber, one marker per routine.
import { h, api, q, EXPORT_DPI } from './ui.js';

const DEFAULTS = { quantities: ['cx', 'cy', 'c4', 'delta_z', 'dz_pooled', 'residual_frac'], xAxis: 'time', relative: true,
  average: 1, errors: false, hiddenFibers: [], hiddenRoutines: ['train_burnin'] };
const FIBER_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];

function loadState() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('fiberView') || '{}') }; } catch { return { ...DEFAULTS }; }
}

function saveState(state) {
  try { localStorage.setItem('fiberView', JSON.stringify(state)); } catch { /* storage blocked */ }
}

export async function renderFibers(body, info) {
  body.replaceChildren(h('p', { class: 'pad muted' }, 'loading…'));
  const options = await api(`/api/run/fibers/options?${q({ run: info.key })}`);
  const state = loadState();
  const controls = h('div', { class: 'card' });
  const status = h('span', { class: 'muted small' });
  const figure = h('img', { alt: 'fitted quantities per fiber', style: 'max-width:100%;background:#fff;border-radius:6px;transition:opacity 0.2s' });
  const actions = h('div', { class: 'row' });
  body.replaceChildren(h('div', { class: 'pad', style: 'display:flex;flex-direction:column;gap:10px' }, controls,
    h('div', { class: 'row' }, actions, h('span', { class: 'spacer' }), status), figure));
  if (!options.quantities.length) {
    controls.replaceChildren(h('p', { class: 'muted' }, 'No fitted quantity yet: run the joint fit (Analyses) first.'));
    return;
  }
  if (state.xAxis === 'x_position' && !options.has_x_position) state.xAxis = 'time';
  let timer = null;

  const change = (mutate) => () => { mutate(); saveState(state); renderControls(); schedule(); };
  const toggleIn = (list, value) => { const i = list.indexOf(value); if (i >= 0) list.splice(i, 1); else list.push(value); };
  const chip = (label, on, toggle, color) => h('button', { class: `chip${on ? ' on' : ''}`, style: color ? `border-left: 4px solid ${color}` : null,
    onclick: change(toggle) }, label);

  function params(extra) {
    const quantities = options.quantities.map(([key]) => key).filter((key) => state.quantities.includes(key));
    return q({ run: info.key, quantities: quantities.join(','), x: state.xAxis, relative: state.relative, average: state.average,
      errors: state.errors, fibers: options.fibers.filter((f) => !state.hiddenFibers.includes(f)).join(','),
      routines: options.routines.filter((r) => !state.hiddenRoutines.includes(r)).join(','), ...extra });
  }

  function renderControls() {
    const xSelect = h('select', {}, Object.entries(options.x_axes).filter(([k]) => k !== 'x_position' || options.has_x_position)
      .map(([k, label]) => h('option', { value: k, selected: state.xAxis === k }, label)));
    xSelect.onchange = change(() => { state.xAxis = xSelect.value; });
    const average = h('input', { type: 'number', min: 1, max: 501, step: 2, value: state.average, style: 'width:60px' });
    average.onchange = change(() => { state.average = Math.max(1, Math.min(501, Number(average.value) || 1)); });
    const flag = (key, label) => {
      const box = h('input', { type: 'checkbox', checked: state[key] });
      box.onchange = change(() => { state[key] = box.checked; });
      return h('label', {}, box, ' ', label);
    };
    controls.replaceChildren(
      h('div', { class: 'row' }, h('b', {}, 'fibers'), options.fibers.map((f, i) => chip(f, !state.hiddenFibers.includes(f),
        () => toggleIn(state.hiddenFibers, f), FIBER_COLORS[i % FIBER_COLORS.length]))),
      h('div', { class: 'row', style: 'margin-top:4px' }, h('b', {}, 'routines'), options.routines.map((r) => chip(r,
        !state.hiddenRoutines.includes(r), () => toggleIn(state.hiddenRoutines, r)))),
      h('div', { class: 'toggles', style: 'margin-top:6px' }, h('b', {}, 'quantities'), options.quantities.map(([key, label]) => {
        const box = h('input', { type: 'checkbox', checked: state.quantities.includes(key) });
        box.onchange = change(() => { toggleIn(state.quantities, key); });
        return h('label', {}, box, ' ', label);
      })),
      h('div', { class: 'toggles', style: 'margin-top:6px' }, h('label', {}, 'x axis ', xSelect), flag('relative', 'each fiber minus its median'),
        h('label', {}, 'moving average over ', average, ' spots'), flag('errors', '±1 sigma where fitted')));
    actions.replaceChildren(
      h('a', { href: `/api/run/fibers/figure?${params({ format: 'pdf', dpi: EXPORT_DPI })}` }, h('button', {}, 'Save PDF')),
      h('a', { href: `/api/run/fibers/figure?${params({ format: 'png', dpi: EXPORT_DPI })}` }, h('button', {}, `Save PNG (${EXPORT_DPI} dpi)`)),
      h('a', { href: `/api/run/fibers.csv?${q({ run: info.key })}` }, h('button', {}, 'Save data (CSV)')));
  }

  function schedule() {
    // a redraw after the clicks settle: each figure takes the server a second or two
    clearTimeout(timer);
    const empty = !state.quantities.length || state.hiddenFibers.length === options.fibers.length
      || state.hiddenRoutines.length === options.routines.length;
    if (empty) { figure.removeAttribute('src'); status.textContent = 'Pick at least one quantity, fiber and routine.'; return; }
    status.textContent = 'drawing…';
    figure.style.opacity = 0.4;
    timer = setTimeout(() => { figure.src = `/api/run/fibers/figure?${params({ format: 'png', dpi: 150 })}`; }, 250);
  }

  figure.onload = () => { figure.style.opacity = 1; status.textContent = ''; };
  figure.onerror = () => { figure.style.opacity = 1; status.textContent = 'the figure could not be drawn (see the server log)'; };
  renderControls();
  schedule();
}
