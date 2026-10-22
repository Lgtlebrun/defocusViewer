// The Fibers subtab: the fitted quantities of every spot against time or stage position, one row per quantity, fibers
// overlaid (one colour each), routines told apart by their marker; a click opens the spot.
import { h, api, q, setHash, cssVar } from './ui.js';
import { ROUTINE_ORDER, routineStyle } from './spots.js';

const QUANTITIES = [
  ['cx', 'spot centre x on the sensor', 'px'], ['cy', 'spot centre y on the sensor', 'px'],
  ['x0', 'centre x in the crop', 'px'], ['y0', 'centre y in the crop', 'px'],
  ['c4', 'c4 (defocus)', 'waves', 'sigma_c4'], ['delta_z', 'delta_z, raw', 'um', 'sigma_delta_z'],
  ['dz_pooled', 'delta_z calibrated (pooled)', 'um'], ['dz_free', 'delta_z calibrated (free)', 'um'],
  ['x_obj_pooled', 'object x (pooled calibration)', 'um'], ['y_obj_pooled', 'object y (pooled calibration)', 'um'],
  ['x_obj_free', 'object x (free calibration)', 'um'], ['y_obj_free', 'object y (free calibration)', 'um'],
  ['r', 'source radius r', 'px'], ['c5', 'c5', 'waves'], ['c6', 'c6', 'waves'], ['c7', 'c7', 'waves'], ['c8', 'c8', 'waves'],
  ['residual_frac', 'residual_frac', ''], ['texp', 'exposure', 'us'],
];
const X_AXES = { time: 'time', minutes: 'minutes since the first spot', x_position: 'stage position (um)', index: 'spot index' };
const FIBER_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const DEFAULTS = { quantities: ['cx', 'cy', 'c4', 'delta_z', 'dz_pooled', 'residual_frac'], xAxis: 'time', relative: false,
  average: 1, errors: false, hiddenFibers: [], hiddenRoutines: ['train_burnin'] };

function loadState() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('fiberView') || '{}') }; } catch { return { ...DEFAULTS }; }
}

function saveState(state) {
  try { localStorage.setItem('fiberView', JSON.stringify(state)); } catch { /* storage blocked */ }
}

function movingAverage(values, window) {
  if (window <= 1) return values;
  const half = Math.floor(window / 2);
  return values.map((_, i) => {
    let sum = 0, n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j++) if (values[j] !== null) { sum += values[j]; n++; }
    return n ? sum / n : null;
  });
}

function median(values) {
  const v = values.filter((x) => x !== null && Number.isFinite(x)).sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)] : 0;
}

export async function renderFibers(body, info) {
  body.replaceChildren(h('p', { class: 'pad muted' }, 'loading…'));
  const data = await api(`/api/run/fibers?${q({ run: info.key })}`);
  const rows = data.rows.map((r, index) => ({ ...Object.fromEntries(data.columns.map((c, i) => [c, r[i]])), index }));
  const fibers = [...new Set(rows.map((r) => r.fiber))].sort();
  const routines = [...new Set(rows.map((r) => r.routine))].sort((a, b) => (ROUTINE_ORDER.indexOf(a) + 1 || 99) - (ROUTINE_ORDER.indexOf(b) + 1 || 99));
  const available = QUANTITIES.filter(([key]) => data.columns.includes(key) && rows.some((r) => r[key] !== null && r[key] !== undefined));
  const t0 = Math.min(...rows.map((r) => Date.parse(r.timestamp)).filter(Number.isFinite));
  const state = loadState();
  const controls = h('div', { class: 'card' });
  const plot = h('div', { style: 'width:100%' });
  const root = h('div', { class: 'pad', style: 'display:flex;flex-direction:column;gap:10px' }, controls, plot);
  body.replaceChildren(root);
  if (!available.length) {
    plot.replaceChildren(h('p', { class: 'muted' }, 'No fitted quantity yet: run the joint fit (Analyses) first.'));
  }

  const change = (mutate) => () => { mutate(); saveState(state); draw(); };
  function chip(label, on, toggle, color) {
    return h('button', { class: `chip${on ? ' on' : ''}`, style: color ? `border-left: 4px solid ${color}` : null, onclick: change(toggle) }, label);
  }
  function toggleIn(list, value) {
    const i = list.indexOf(value);
    if (i >= 0) list.splice(i, 1); else list.push(value);
  }

  function renderControls() {
    const xSelect = h('select', {}, Object.entries(X_AXES).map(([k, label]) => h('option', { value: k, selected: state.xAxis === k }, label)));
    xSelect.onchange = change(() => { state.xAxis = xSelect.value; });
    const average = h('input', { type: 'number', min: 1, max: 201, step: 2, value: state.average, style: 'width:60px' });
    average.onchange = change(() => { state.average = Math.max(1, Number(average.value) || 1); });
    const flag = (key, label) => {
      const box = h('input', { type: 'checkbox', checked: state[key] });
      box.onchange = change(() => { state[key] = box.checked; });
      return h('label', {}, box, ' ', label);
    };
    controls.replaceChildren(
      h('div', { class: 'row' }, h('b', {}, 'fibers'), fibers.map((f, i) => chip(f, !state.hiddenFibers.includes(f), () => toggleIn(state.hiddenFibers, f), FIBER_COLORS[i % FIBER_COLORS.length]))),
      h('div', { class: 'row', style: 'margin-top:4px' }, h('b', {}, 'routines'), routines.map((r) => chip(r, !state.hiddenRoutines.includes(r), () => toggleIn(state.hiddenRoutines, r)))),
      h('div', { class: 'toggles', style: 'margin-top:6px' }, h('b', {}, 'quantities'), available.map(([key, label]) => {
        const box = h('input', { type: 'checkbox', checked: state.quantities.includes(key) });
        box.onchange = change(() => { state.quantities = available.map(([k]) => k).filter((k) => (k === key ? box.checked : state.quantities.includes(k))); });
        return h('label', {}, box, ' ', label);
      })),
      h('div', { class: 'toggles', style: 'margin-top:6px' }, h('label', {}, 'x axis ', xSelect),
        flag('relative', 'minus each fiber\'s median'), h('label', {}, 'moving average over ', average, ' points'),
        flag('errors', '±1 sigma where fitted')));
  }

  function xOf(r, i) {
    if (state.xAxis === 'time') return r.timestamp;
    if (state.xAxis === 'minutes') return (Date.parse(r.timestamp) - t0) / 60000;
    if (state.xAxis === 'x_position') return r.x_position;
    return i;
  }

  function draw() {
    renderControls();
    const quantities = available.filter(([key]) => state.quantities.includes(key));
    if (!quantities.length) { Plotly.purge(plot); plot.replaceChildren(h('p', { class: 'muted' }, 'Tick a quantity.')); return; }
    const n = quantities.length, gap = 0.02, height = (1 - gap * (n - 1)) / n;
    const traces = [], layout = {
      height: 80 + 190 * n, margin: { l: 70, r: 10, t: 30, b: 40 }, paper_bgcolor: cssVar('--surface-0'), plot_bgcolor: cssVar('--surface-1'),
      font: { size: 11, color: cssVar('--text-secondary') }, showlegend: true, legend: { orientation: 'h', y: 1.0, yanchor: 'bottom' },
      hovermode: 'closest',
    };
    const axis = { color: cssVar('--text-secondary'), gridcolor: cssVar('--border'), zeroline: false };
    quantities.forEach(([key, label, unit, sigma], qi) => {
      const yname = qi ? `yaxis${qi + 1}` : 'yaxis';
      layout[yname] = { ...axis, anchor: 'x', domain: [1 - (qi + 1) * height - qi * gap, 1 - qi * height - qi * gap], title: { text: `${label}${unit ? ` (${unit})` : ''}`, font: { size: 10 } } };
      fibers.forEach((fiber, fi) => {
        if (state.hiddenFibers.includes(fiber)) return;
        const ofFiber = rows.filter((r) => r.fiber === fiber && !state.hiddenRoutines.includes(r.routine) && r[key] !== null && r[key] !== undefined);
        const offset = state.relative ? median(ofFiber.map((r) => r[key])) : 0;
        routines.forEach((routine) => {
          if (state.hiddenRoutines.includes(routine)) return;
          const rs = ofFiber.filter((r) => r.routine === routine);
          if (!rs.length) return;
          const indexed = rs.map((r) => [xOf(r, r.index), r]).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
          const y = movingAverage(indexed.map(([, r]) => r[key] - offset), state.average);
          traces.push({
            type: 'scattergl', mode: rs.length > 1 ? 'lines+markers' : 'markers', name: fiber, legendgroup: fiber,
            showlegend: qi === 0 && routine === routines.find((rt) => ofFiber.some((r) => r.routine === rt)),
            x: indexed.map(([x]) => x), y, xaxis: 'x', yaxis: qi ? `y${qi + 1}` : 'y',
            customdata: indexed.map(([, r]) => [r.img, r.routine, r.source || '']),
            marker: { color: FIBER_COLORS[fi % FIBER_COLORS.length], size: 5, symbol: routineStyle(routine, routines).symbol },
            line: { color: FIBER_COLORS[fi % FIBER_COLORS.length], width: 1 },
            error_y: state.errors && sigma ? { type: 'data', array: indexed.map(([, r]) => r[sigma]), visible: true, thickness: 0.6, width: 0 } : undefined,
            hovertemplate: `${fiber} %{customdata[1]}<br>%{customdata[0]}<br>${label}: %{y:.5g}<br>%{x}<extra>%{customdata[2]}</extra>`,
          });
        });
      });
    });
    layout.xaxis = { ...axis, anchor: quantities.length > 1 ? `y${quantities.length}` : 'y', title: { text: X_AXES[state.xAxis], font: { size: 10 } },
      type: state.xAxis === 'time' ? 'date' : 'linear' };
    plot.replaceChildren();
    Plotly.react(plot, traces, layout, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['select2d', 'lasso2d'] });
    plot.on('plotly_click', (event) => {
      const img = event.points?.[0]?.customdata?.[0];
      if (img) { setHash({ sub: null, img }); window.dispatchEvent(new PopStateEvent('popstate')); }
    });
  }
  if (available.length) draw(); else renderControls();
}
