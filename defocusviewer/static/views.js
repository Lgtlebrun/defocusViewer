// The images of a spot as Plotly heatmaps: which views exist, their colour scales, the ghost box, and zoom shared
// between the panels of one domain (image pixels, or FFT frequencies).
import { h, cssVar } from './ui.js';

export const VIEWS = [
  { key: 'data', label: 'Data', needs: 'data', domain: 'image', kind: 'image' },
  { key: 'model', label: 'Model', needs: 'model', domain: 'image', kind: 'image' },
  { key: 'residual', label: 'Residual (model - data)', needs: 'residual', domain: 'image', kind: 'diverging' },
  { key: 'logres', label: 'log10 |residual|', needs: 'residual', domain: 'image', kind: 'sequential' },
  { key: 'relres', label: 'Residual / data peak', needs: 'residual', domain: 'image', kind: 'diverging' },
  { key: 'fft_data', label: 'log(1 + |FFT data|)', needs: 'fft_data', domain: 'fft', kind: 'sequential' },
  { key: 'fft_model', label: 'log(1 + |FFT model|)', needs: 'fft_model', domain: 'fft', kind: 'sequential' },
  { key: 'fft_residual', label: 'log(1 + |FFT residual|)', needs: 'fft_residual', domain: 'fft', kind: 'sequential' },
  { key: 'fftp_data', label: '|FFT data|^p', needs: 'fft_data', domain: 'fft', kind: 'sequential' },
  { key: 'fftp_residual', label: '|FFT residual|^p', needs: 'fft_residual', domain: 'fft', kind: 'sequential' },
];

const DEFAULT_OPTIONS = { views: ['data', 'model', 'residual', 'fft_residual'], scale: 'linear', power: 0.3,
  syncZoom: true, showGhost: true, blankGhost: false, sharedRange: false, diffFirst: false };

export function loadOptions() {
  try { return { ...DEFAULT_OPTIONS, ...JSON.parse(localStorage.getItem('viewOptions') || '{}') }; } catch { return { ...DEFAULT_OPTIONS }; }
}

export function saveOptions(options) {
  try { localStorage.setItem('viewOptions', JSON.stringify(options)); } catch { /* storage blocked */ }
}

const decoded = new WeakMap();

export function arrays(detail) {
  // {name: {rows, cols, values}} of the base64 float32 images of a spot detail (decoded once)
  if (decoded.has(detail)) return decoded.get(detail);
  const out = {};
  for (const [name, { shape, data }] of Object.entries(detail.images)) {
    const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
    out[name] = { rows: shape[0], cols: shape[1], values: new Float32Array(bytes.buffer) };
  }
  decoded.set(detail, out);
  return out;
}

function dark() {
  const theme = document.documentElement.dataset.theme;
  return theme ? theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
}

function colorscale(kind) {
  if (kind === 'diverging') {
    const mid = dark() ? '#383835' : '#f0efec';
    return [[0, '#184f95'], [0.25, '#5598e7'], [0.5, mid], [0.75, '#e86f6e'], [1, '#a3201f']];
  }
  return dark()
    ? [[0, '#1a1a19'], [0.25, '#104281'], [0.5, '#256abf'], [0.75, '#6da7ec'], [1, '#cde2fb']]
    : [[0, '#fcfcfb'], [0.25, '#9ec5f4'], [0.5, '#3987e5'], [0.75, '#1c5cab'], [1, '#0d366b']];
}

function maxOf(values) {
  let m = -Infinity;
  for (const x of values) if (x > m) m = x;
  return m;
}

function inBox(box, x, y) {
  return box && x >= box.x0 && x < box.x0 + box.side && y >= box.y0 && y < box.y0 + box.side;
}

export function viewValues(view, images, options, ghostBox, reference) {
  // Float32Array of a view (row-major) or null when the spot lacks the image it needs
  const base = images[view.needs];
  if (!base) return null;
  const { rows, cols } = base;
  let v = Float32Array.from(base.values);
  if (reference && options.diffFirst && view.domain === 'image' && ['data', 'model'].includes(view.key) && reference[view.needs]) {
    const ref = reference[view.needs].values;
    const sum = (a) => a.reduce((s, x) => s + x, 0);
    const k = sum(base.values) / sum(ref);
    v = v.map((x, i) => x - k * ref[i]);
    return { rows, cols, values: v, diverging: true };
  }
  const peak = images.data ? maxOf(images.data.values) : 1;
  switch (view.key) {
    case 'data': case 'model': {
      const max = maxOf(v);
      if (options.scale === 'log') v = v.map((x) => Math.log10(Math.max(x, max * 1e-4)));
      else if (options.scale === 'asinh') v = v.map((x) => Math.asinh(x / (0.02 * max)));
      break;
    }
    case 'logres': v = v.map((x) => Math.log10(Math.abs(x) + 1e-12)); break;
    case 'relres': v = v.map((x) => x / peak); break;
    case 'fft_data': case 'fft_model': case 'fft_residual': v = v.map((x) => Math.log1p(x)); break;
    case 'fftp_data': case 'fftp_residual': v = v.map((x) => Math.pow(x, options.power)); break;
    default: break;
  }
  if (options.blankGhost && ghostBox && view.domain === 'image' && view.needs === 'residual') {
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) if (inBox(ghostBox, x, y)) v[y * cols + x] = NaN;
  }
  return { rows, cols, values: v, diverging: view.kind === 'diverging' };
}

export function range(values, diverging) {
  let lo = Infinity, hi = -Infinity;
  for (const x of values) if (Number.isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; }
  if (diverging) { const m = Math.max(Math.abs(lo), Math.abs(hi)); return [-m, m]; }
  return [lo, hi];
}

function rows2d({ rows, cols, values }) {
  const z = new Array(rows);
  for (let y = 0; y < rows; y++) z[y] = Array.from(values.subarray(y * cols, (y + 1) * cols));
  return z;
}

function frequencies(n) {
  return Array.from({ length: n }, (_, i) => (i - Math.floor(n / 2)) / n);
}

// One zoom group per spot panel (or per compare grid): relayout of one plot is copied to the plots of its domain
export function zoomGroup() {
  const plots = [];
  let syncing = false;
  return {
    add(div, domain) {
      plots.push({ div, domain });
      div.on('plotly_relayout', (event) => {
        if (syncing || !this.enabled) return;
        const update = {};
        for (const k of ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]', 'xaxis.autorange', 'yaxis.autorange']) {
          if (k in event) update[k] = event[k];
        }
        if (!Object.keys(update).length) return;
        syncing = true;
        Promise.all(plots.filter((p) => p.div !== div && p.domain === domain && p.div.isConnected)
          .map((p) => Plotly.relayout(p.div, update))).finally(() => { syncing = false; });
      });
    },
    enabled: true,
  };
}

export function heatmap(div, view, values, zrange, ghostBox, options) {
  // Draws one view into `div` (Plotly), with the ghost box outlined on image views when asked
  const fft = view.domain === 'fft';
  const axis = { showgrid: false, zeroline: false, ticks: 'outside', ticklen: 3, color: cssVar('--text-secondary') };
  const trace = {
    type: 'heatmap', z: rows2d(values), zmin: zrange[0], zmax: zrange[1], colorscale: colorscale(values.diverging ? 'diverging' : view.kind),
    colorbar: { thickness: 10, outlinewidth: 0, tickfont: { size: 10 } },
    hovertemplate: fft ? 'fx %{x:.3f}, fy %{y:.3f} c/px<br>%{z:.4g}<extra></extra>' : 'x %{x}, y %{y}<br>%{z:.4g}<extra></extra>',
  };
  if (fft) { trace.x = frequencies(values.cols); trace.y = frequencies(values.rows); }
  const shapes = [], extra = [];
  if (options.showGhost && ghostBox && !fft) {
    shapes.push({ type: 'rect', x0: ghostBox.x0 - 0.5, x1: ghostBox.x0 + ghostBox.side - 0.5, y0: ghostBox.y0 - 0.5,
      y1: ghostBox.y0 + ghostBox.side - 0.5, line: { color: '#d03b3b', width: 1.5, dash: 'dot' } });
    extra.push({ type: 'scatter', mode: 'markers', x: [ghostBox.gx], y: [ghostBox.gy], marker: { symbol: 'x-thin', size: 9, line: { width: 2, color: '#d03b3b' } },
      hovertemplate: 'ghost centre %{x:.1f}, %{y:.1f}<extra></extra>', showlegend: false });
  }
  const layout = {
    margin: { l: 34, r: 4, t: 4, b: 26 }, paper_bgcolor: cssVar('--surface-1'), plot_bgcolor: cssVar('--surface-1'),
    font: { size: 10, color: cssVar('--text-secondary') }, shapes,
    xaxis: { ...axis, constrain: 'domain' }, yaxis: { ...axis, autorange: 'reversed', scaleanchor: 'x', constrain: 'domain' },
  };
  return Plotly.react(div, [trace, ...extra], layout, { displaylogo: false, responsive: true,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'] });
}

export function viewToggles(options, onChange) {
  // Checkboxes of the views and the display options, shared by the spot panel and the Compare tab
  const change = (mutate) => () => { mutate(); saveOptions(options); onChange(); };
  const boxes = VIEWS.map((view) => {
    const box = h('input', { type: 'checkbox', checked: options.views.includes(view.key) });
    box.onchange = change(() => {
      options.views = VIEWS.map((v) => v.key).filter((k) => (k === view.key ? box.checked : options.views.includes(k)));
    });
    return h('label', {}, box, ' ', view.label);
  });
  const scale = h('select', {}, ['linear', 'log', 'asinh'].map((s) => h('option', { value: s, selected: options.scale === s }, s)));
  scale.onchange = change(() => { options.scale = scale.value; });
  const power = h('input', { type: 'range', min: 0.05, max: 1, step: 0.05, value: options.power, style: 'width:110px;vertical-align:middle' });
  const powerLabel = h('span', { class: 'mono' }, options.power.toFixed(2));
  power.oninput = () => { powerLabel.textContent = Number(power.value).toFixed(2); };
  power.onchange = change(() => { options.power = Number(power.value); });
  const flag = (key, label, title) => {
    const box = h('input', { type: 'checkbox', checked: options[key] });
    box.onchange = change(() => { options[key] = box.checked; });
    return h('label', { title }, box, ' ', label);
  };
  return h('div', {},
    h('div', { class: 'toggles' }, boxes),
    h('div', { class: 'toggles', style: 'margin-top:4px' },
      h('label', {}, 'data / model scale ', scale),
      h('label', {}, 'p ', power, ' ', powerLabel),
      flag('syncZoom', 'shared zoom'),
      flag('showGhost', 'outline ghost box'),
      flag('blankGhost', 'blank ghost box in residuals', 'the pixels the masked fits ignore')));
}
