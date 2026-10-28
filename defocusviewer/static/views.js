// The images of a spot as Plotly heatmaps, laid out as a matrix: rows are quantities (data, model, residual, PSF,
// source, pupil, wavefront), columns are representations (image or |FFT|, and a scale). Zoom is shared between the
// panels of one domain (image pixels, source pixels, pupil radii, and their frequencies).
import { h, cssVar, PLOT_CONFIG } from './ui.js';

export const QUANTITIES = {
  data: { label: 'Data', domain: 'image', ghost: true },
  model: { label: 'Model', domain: 'image', ghost: true },
  residual: { label: 'Residual (model - data)', domain: 'image', signed: true, ghost: true },
  psf: { label: 'PSF', domain: 'image' },
  source: { label: 'Source', domain: 'source' },
  pupil: { label: 'Pupil amplitude', domain: 'pupil' },
  wavefront: { label: 'Wavefront (waves)', domain: 'pupil', signed: true },
};
export const SCALES = { linear: 'linear', log10: 'log10 |x|', log1p: 'log(1 + |x|)', asinh: 'asinh', power: '|x|^p',
  peak: '÷ data peak' };
const SPACES = { image: 'image', fft: '|FFT|' };
const SOURCE_ZOOM = 12;    // half-width (px) shown at first on source panels: the source is a few px wide

const DEFAULT_ROWS = ['data', 'model', 'residual'];
const DEFAULT_COLUMNS = [{ space: 'image', scale: 'linear', p: 0.3 }, { space: 'fft', scale: 'log1p', p: 0.3 }];
export const ZOOM_SHARING = { column: 'same column', row: 'same row', all: 'everything', none: 'off' };
const DEFAULT_OPTIONS = { rows: DEFAULT_ROWS, columns: DEFAULT_COLUMNS, zoomShare: 'column', showGhost: true, blankGhost: false,
  sharedRange: false, diffFirst: false, imageMap: 'viridis', divergingMap: 'blue-red', reverseMaps: false };

// Colour maps of the unsigned panels and of the signed ones (residual, wavefront, differences)
const IMAGE_MAPS = {
  viridis: 'Viridis', cividis: 'Cividis',
  inferno: [[0, '#000004'], [0.25, '#57106e'], [0.5, '#bc3754'], [0.75, '#f98e09'], [1, '#fcffa4']],
  magma: [[0, '#000004'], [0.25, '#51127c'], [0.5, '#b73779'], [0.75, '#fc8961'], [1, '#fcfdbf']],
  blues: null,          // the theme ramp, see colorscale()
  gray: [[0, '#000000'], [1, '#ffffff']], hot: 'Hot',
};
const DIVERGING_MAPS = {
  'blue-red': null,     // the theme pair, see colorscale()
  'purple-green': [[0, '#40004b'], [0.25, '#9970ab'], [0.5, '#f7f7f7'], [0.75, '#5aae61'], [1, '#00441b']],
  'brown-teal': [[0, '#543005'], [0.25, '#bf812d'], [0.5, '#f5f5f5'], [0.75, '#35978f'], [1, '#003c30']],
};

export function loadOptions() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('viewOptions') || '{}'); } catch { /* storage blocked */ }
  const options = { ...DEFAULT_OPTIONS, ...saved };
  if (!ZOOM_SHARING[options.zoomShare]) options.zoomShare = 'column';
  if (!Array.isArray(options.rows) || !options.rows.every((q) => QUANTITIES[q])) options.rows = [...DEFAULT_ROWS];
  if (!Array.isArray(options.columns) || !options.columns.every((c) => SPACES[c.space] && SCALES[c.scale])) {
    options.columns = DEFAULT_COLUMNS.map((c) => ({ ...c }));
  }
  return options;
}

export function saveOptions(options) {
  try { localStorage.setItem('viewOptions', JSON.stringify(options)); } catch { /* storage blocked */ }
}

export function cells(options) {
  // The panels of the matrix, row by row: {q, space, scale, p}
  return options.rows.flatMap((q) => options.columns.map((c) => ({ q, ...c })));
}

const decoded = new WeakMap();

export function arrays(detail) {
  // {name: {rows, cols, values, coords}} of the base64 float32 images of a spot detail (decoded once)
  if (decoded.has(detail)) return decoded.get(detail);
  const out = {};
  for (const [name, { shape, data, coords }] of Object.entries(detail.images)) {
    const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
    out[name] = { rows: shape[0], cols: shape[1], values: new Float32Array(bytes.buffer), coords };
  }
  decoded.set(detail, out);
  return out;
}

const twiddles = new Map();

function twiddle(n) {
  if (!twiddles.has(n)) {
    const cos = new Float64Array(n * n), sin = new Float64Array(n * n);
    for (let k = 0; k < n; k++) for (let x = 0; x < n; x++) {
      const a = (-2 * Math.PI * ((k * x) % n)) / n;
      cos[k * n + x] = Math.cos(a);
      sin[k * n + x] = Math.sin(a);
    }
    twiddles.set(n, [cos, sin]);
  }
  return twiddles.get(n);
}

function fftMagnitude({ rows, cols, values, coords }) {
  // |DFT| of an image (NaN as 0), zero frequency at the centre (as numpy's fftshift), frequencies in 1 / axis unit
  const [cx, sx] = twiddle(cols), [cy, sy] = twiddle(rows);
  const re = new Float64Array(rows * cols), im = new Float64Array(rows * cols);
  for (let y = 0; y < rows; y++) for (let k = 0; k < cols; k++) {
    let r = 0, i = 0;
    for (let x = 0; x < cols; x++) {
      const v = values[y * cols + x];
      if (v === v) { r += v * cx[k * cols + x]; i += v * sx[k * cols + x]; }
    }
    re[y * cols + k] = r;
    im[y * cols + k] = i;
  }
  const out = new Float32Array(rows * cols);
  const shiftY = Math.floor(rows / 2), shiftX = Math.floor(cols / 2);
  for (let ky = 0; ky < rows; ky++) for (let kx = 0; kx < cols; kx++) {
    let r = 0, i = 0;
    for (let y = 0; y < rows; y++) {
      const c = cy[ky * rows + y], s = sy[ky * rows + y], a = re[y * cols + kx], b = im[y * cols + kx];
      r += a * c - b * s;
      i += a * s + b * c;
    }
    out[((ky + shiftY) % rows) * cols + ((kx + shiftX) % cols)] = Math.hypot(r, i);
  }
  const step = coords ? coords[1] - coords[0] : 1;
  return { values: out, coords: Array.from({ length: cols }, (_, j) => (j - shiftX) / (cols * step)) };
}

const fftCache = new WeakMap();

function cachedFft(image, key) {
  // the |FFT| of an image of a spot is computed once per (image, variant)
  if (!fftCache.has(image)) fftCache.set(image, new Map());
  const cache = fftCache.get(image);
  if (!cache.has(key.id)) cache.set(key.id, fftMagnitude({ ...image, values: key.values }));
  return cache.get(key.id);
}

function dark() {
  const theme = document.documentElement.dataset.theme;
  return theme ? theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
}

function colorscale(signed, options) {
  const chosen = signed ? DIVERGING_MAPS[options.divergingMap] : IMAGE_MAPS[options.imageMap];
  if (chosen) return chosen;
  if (signed) {
    const mid = dark() ? '#383835' : '#f0efec';
    return [[0, '#184f95'], [0.25, '#5598e7'], [0.5, mid], [0.75, '#e86f6e'], [1, '#a3201f']];
  }
  return dark()
    ? [[0, '#1a1a19'], [0.25, '#104281'], [0.5, '#256abf'], [0.75, '#6da7ec'], [1, '#cde2fb']]
    : [[0, '#fcfcfb'], [0.25, '#9ec5f4'], [0.5, '#3987e5'], [0.75, '#1c5cab'], [1, '#0d366b']];
}

function maxAbs(values) {
  let m = 0;
  for (const x of values) if (Math.abs(x) > m) m = Math.abs(x);
  return m;
}

function inBox(box, x, y) {
  return box && x >= box.x0 && x < box.x0 + box.side && y >= box.y0 && y < box.y0 + box.side;
}

export function columnLabel(column) {
  const scale = column.scale === 'power' ? `|x|^${column.p}` : SCALES[column.scale];
  return `${SPACES[column.space]}, ${scale}`;
}

export function panelLabel(panel) {
  return `${QUANTITIES[panel.q].label}: ${columnLabel(panel)}`;
}

export function domainOf(panel) {
  const domain = QUANTITIES[panel.q].domain;
  return panel.space === 'fft' ? `fft-${domain === 'pupil' ? 'pupil' : 'image'}` : domain;
}

export function panelValues(panel, images, options, ghostBox, reference) {
  // {rows, cols, values, coords, signed} of a panel, null when the spot lacks the image it needs
  const quantity = QUANTITIES[panel.q];
  const base = images[panel.q];
  if (!base) return null;
  let signed = Boolean(quantity.signed);
  let values = base.values, variant = 'plain';
  if (reference?.[panel.q] && reference[panel.q] !== base && options.diffFirst && ['data', 'model'].includes(panel.q)) {
    const ref = reference[panel.q].values;
    let sum = 0, sumRef = 0;
    for (let i = 0; i < values.length; i++) { sum += values[i]; sumRef += ref[i]; }
    values = values.map((x, i) => x - (sum / sumRef) * ref[i]);
    signed = true;
    variant = 'diff';
  }
  if (options.blankGhost && panel.q === 'residual' && ghostBox) {
    values = Float32Array.from(values);
    for (let y = 0; y < base.rows; y++) for (let x = 0; x < base.cols; x++) if (inBox(ghostBox, x, y)) values[y * base.cols + x] = NaN;
    variant += '-blank';
  }
  let coords = base.coords;
  let raw = values;
  if (panel.space === 'fft') {
    // differences depend on the reference spot: never cached
    ({ values, coords } = variant.startsWith('diff') ? fftMagnitude({ ...base, values }) : cachedFft(base, { id: variant, values }));
    raw = values;
    signed = false;
  }
  const peak = images.data ? maxAbs(images.data.values) : 1;
  switch (panel.scale) {
    case 'log10': { const floor = maxAbs(values) * 1e-9 + 1e-30; values = values.map((x) => Math.log10(Math.abs(x) + floor)); signed = false; break; }
    case 'log1p': values = values.map((x) => Math.log1p(Math.abs(x))); signed = false; break;
    case 'asinh': { const m = maxAbs(values) || 1; values = values.map((x) => Math.asinh(x / (0.02 * m))); break; }
    case 'power': values = values.map((x) => Math.pow(Math.abs(x), panel.p)); signed = false; break;
    case 'peak': values = values.map((x) => x / peak); break;
    default: values = Float32Array.from(values); break;
  }
  return { rows: base.rows, cols: base.cols, values, raw, coords, signed };
}

export const AXES = {
  image: ['x (px)', 'y (px)'], source: ['dx (px)', 'dy (px)'], pupil: ['u (pupil radii)', 'v (pupil radii)'],
  'fft-image': ['fx (cycles/px)', 'fy (cycles/px)'], 'fft-pupil': ['fu (cycles/pupil radius)', 'fv (cycles/pupil radius)'],
};

export function range(values, signed) {
  let lo = Infinity, hi = -Infinity;
  for (const x of values) if (Number.isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; }
  if (signed) { const m = Math.max(Math.abs(lo), Math.abs(hi)); return [-m, m]; }
  return [lo, hi];
}

function pairKey(panel) {
  return ['data', 'model'].includes(panel.q) ? `${panel.space}|${panel.scale}|${panel.scale === 'power' ? panel.p : ''}` : null;
}

export function pairRanges(panels, ranges) {
  // Data and model in the same representation get one colour range (the union of theirs), so they compare by eye
  const union = new Map();
  panels.forEach((panel, j) => {
    const key = pairKey(panel), r = ranges[j];
    if (!key || !r) return;
    const u = union.get(key);
    union.set(key, u ? [Math.min(u[0], r[0]), Math.max(u[1], r[1])] : r);
  });
  return ranges.map((r, j) => (r && pairKey(panels[j]) ? union.get(pairKey(panels[j])) : r));
}

function rows2d({ rows, cols, values }) {
  const z = new Array(rows);
  for (let y = 0; y < rows; y++) z[y] = Array.from(values.subarray(y * cols, (y + 1) * cols));
  return z;
}

export function zoomKey(options, panel, row, column) {
  // Panels with the same key zoom together: same axes (domain) and, per options.zoomShare, same column or row
  const domain = domainOf(panel);
  return { column: `${domain}|c${column}`, row: `${domain}|r${row}`, all: domain, none: `${domain}|${row}|${column}` }[options.zoomShare];
}

// One zoom group per spot (or per compare grid): relayout of one plot is copied to the plots of the same key
export function zoomGroup() {
  const plots = [];
  let syncing = false;
  return {
    add(div, domain) {
      plots.push({ div, domain });
      div.on('plotly_relayout', (event) => {
        if (syncing) return;
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
  };
}

const HOVER = {
  image: 'x %{x}, y %{y} px<br>%{z:.4g}', source: 'dx %{x}, dy %{y} px<br>%{z:.4g}',
  pupil: 'u %{x:.3f}, v %{y:.3f} (pupil radii)<br>%{z:.4g}', 'fft-image': 'fx %{x:.3f}, fy %{y:.3f} cycles/px<br>%{z:.4g}',
  'fft-pupil': 'fu %{x:.2f}, fv %{y:.2f} cycles/pupil radius<br>%{z:.4g}',
};

export function heatmap(div, panel, values, zrange, ghostBox, options) {
  // Draws one panel into `div` (Plotly), with the ghost box outlined on data / model / residual images when asked
  const domain = domainOf(panel);
  const axis = { showgrid: false, zeroline: false, ticks: 'outside', ticklen: 3, color: cssVar('--text-secondary') };
  const trace = {
    type: 'heatmap', z: rows2d(values), zmin: zrange[0], zmax: zrange[1], reversescale: options.reverseMaps,
    colorscale: colorscale(values.signed, options), colorbar: { thickness: 10, outlinewidth: 0, tickfont: { size: 10 } },
    hovertemplate: `${HOVER[domain]}<extra></extra>`,
  };
  if (values.coords) { trace.x = values.coords; trace.y = values.coords; }
  const shapes = [], extra = [];
  if (options.showGhost && ghostBox && panel.space === 'image' && QUANTITIES[panel.q].ghost) {
    shapes.push({ type: 'rect', x0: ghostBox.x0 - 0.5, x1: ghostBox.x0 + ghostBox.side - 0.5, y0: ghostBox.y0 - 0.5,
      y1: ghostBox.y0 + ghostBox.side - 0.5, line: { color: '#d03b3b', width: 1.5, dash: 'dot' } });
    if (ghostBox.gx !== undefined) {
      extra.push({ type: 'scatter', mode: 'markers', x: [ghostBox.gx], y: [ghostBox.gy], showlegend: false,
        marker: { symbol: 'x-thin', size: 9, line: { width: 2, color: '#d03b3b' } }, hovertemplate: 'ghost centre %{x:.1f}, %{y:.1f}<extra></extra>' });
    }
  }
  const zoomed = domain === 'source';
  const layout = {
    margin: { l: 34, r: 4, t: 4, b: 26 }, paper_bgcolor: cssVar('--surface-1'), plot_bgcolor: cssVar('--surface-1'),
    font: { size: 10, color: cssVar('--text-secondary') }, shapes,
    xaxis: { ...axis, constrain: 'domain', ...(zoomed ? { range: [-SOURCE_ZOOM - 0.5, SOURCE_ZOOM + 0.5] } : {}) },
    yaxis: { ...axis, scaleanchor: 'x', constrain: 'domain',
      ...(zoomed ? { range: [SOURCE_ZOOM + 0.5, -SOURCE_ZOOM - 0.5] } : { autorange: 'reversed' }) },
  };
  return Plotly.react(div, [trace, ...extra], layout, { ...PLOT_CONFIG,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'] });
}

function columnControls(column, onChange, onRemove) {
  // The drop-downs of one column: space, scale (and p for the power), and a remove button
  const select = (choices, key) => {
    const el = h('select', { class: 'small' }, Object.entries(choices).map(([value, label]) =>
      h('option', { value, selected: column[key] === value }, label)));
    el.onchange = () => {
      column[key] = el.value;
      if (key === 'space' && el.value === 'fft' && column.scale === 'linear') column.scale = 'log1p';
      onChange();
    };
    return el;
  };
  const p = h('input', { type: 'number', min: 0.05, max: 2, step: 0.05, value: column.p ?? 0.3, style: 'width:56px', title: 'exponent p' });
  p.onchange = () => { column.p = Number(p.value) || 0.3; onChange(); };
  return h('span', { class: 'column-editor' }, select(SPACES, 'space'), select(SCALES, 'scale'),
    column.scale === 'power' ? p : null, h('button', { class: 'small', title: 'remove this column', onclick: onRemove }, '×'));
}

export function viewToggles(options, onChange) {
  // Rows (quantities), columns (representations) and the display options shared by every panel
  const change = (mutate) => () => { mutate(); saveOptions(options); onChange(); };
  const flag = (key, label, title) => {
    const box = h('input', { type: 'checkbox', checked: options[key] });
    box.onchange = change(() => { options[key] = box.checked; });
    return h('label', { title }, box, ' ', label);
  };
  const mapSelect = (key, maps) => {
    const select = h('select', {}, Object.entries(maps).map(([m, label]) =>
      h('option', { value: m, selected: options[key] === m }, typeof label === 'string' && key === 'zoomShare' ? label : m)));
    select.onchange = change(() => { options[key] = select.value; });
    return select;
  };
  const rowBoxes = Object.entries(QUANTITIES).map(([q, { label }]) => {
    const box = h('input', { type: 'checkbox', checked: options.rows.includes(q) });
    box.onchange = change(() => { options.rows = Object.keys(QUANTITIES).filter((k) => (k === q ? box.checked : options.rows.includes(k))); });
    return h('label', {}, box, ' ', label);
  });
  const columnEditors = options.columns.map((column, i) => columnControls(column, change(() => {}),
    change(() => { options.columns.splice(i, 1); })));
  return h('div', {},
    h('div', { class: 'toggles' }, h('b', {}, 'rows'), rowBoxes),
    h('div', { class: 'toggles', style: 'margin-top:4px' }, h('b', {}, 'columns'), columnEditors,
      h('button', { class: 'small', onclick: change(() => { options.columns.push({ space: 'image', scale: 'log10', p: 0.3 }); }) }, '+ column'),
      h('button', { class: 'small', onclick: change(() => { options.rows = [...DEFAULT_ROWS]; options.columns = DEFAULT_COLUMNS.map((c) => ({ ...c })); }) }, 'reset')),
    h('div', { class: 'toggles', style: 'margin-top:4px' },
      h('label', {}, 'colour map ', mapSelect('imageMap', IMAGE_MAPS)),
      h('label', {}, 'signed map ', mapSelect('divergingMap', DIVERGING_MAPS)),
      flag('reverseMaps', 'reversed'),
      h('label', {}, 'zoom together ', mapSelect('zoomShare', ZOOM_SHARING)),
      flag('showGhost', 'outline ghost box'),
      flag('blankGhost', 'blank ghost box in residuals', 'the pixels the masked fits ignore')));
}
