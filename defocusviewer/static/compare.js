// The Compare tab: pinned spots (from any run, kept in this browser) side by side, same views, shared zoom.
import { h, api, post, q, toast, fmt } from './ui.js';
import { attachMenu } from './menu.js';
import { loadOptions, arrays, cells, panelValues, panelLabel, range, heatmap, zoomGroup, zoomKey, viewToggles, saveOptions } from './views.js';

export function pins() {
  try { return JSON.parse(localStorage.getItem('pins') || '[]'); } catch { return []; }
}

function savePins(list) {
  try { localStorage.setItem('pins', JSON.stringify(list)); } catch { toast('This browser blocks storage: pins are not kept', true); }
  window.dispatchEvent(new Event('pinschange'));
}

export const pinCount = () => pins().length;

export function addPin(pin) {
  savePins([...pins().filter((p) => !(p.run === pin.run && p.img === pin.img)), pin]);
}

function storedGhost() {
  try { return localStorage.getItem('ghost') || 'on'; } catch { return 'on'; }
}

async function load(pin, ghost) {
  const detail = await api(`/api/spot?${q({ run: pin.run, img: pin.img, ghost })}`);
  return detail.needs === 'refit' ? post('/api/spot/refit', { run: pin.run, img: pin.img, ghost }) : detail;
}

export async function renderCompare() {
  const root = document.getElementById('tab-compare');
  const list = pins();
  if (!list.length) {
    root.replaceChildren(h('p', { class: 'muted' }, 'No pinned spots: use "Pin to compare" on a spot (Runs tab).'));
    return;
  }
  const options = loadOptions();
  const ghost = storedGhost();
  const grid = h('div', { style: 'display:flex;flex-direction:column;gap:8px' }, h('p', { class: 'muted' }, h('span', { class: 'spinner' }), ` loading ${list.length} spots (fitting those without a fit)…`));
  const flag = (key, label, title) => {
    const box = h('input', { type: 'checkbox', checked: options[key] });
    box.onchange = () => { options[key] = box.checked; saveOptions(options); draw(); };
    return h('label', { title }, box, ' ', label);
  };
  const toolbar = h('div', { style: 'margin-top:6px' });
  const ghostSelect = h('select', {}, ['on', 'off'].map((g) => h('option', { value: g, selected: g === ghost }, `ghost mask ${g}`)));
  ghostSelect.onchange = () => { try { localStorage.setItem('ghost', ghostSelect.value); } catch { /* blocked */ } renderCompare(); };
  root.replaceChildren(
    h('div', { class: 'card' }, h('div', { class: 'row' }, h('b', {}, `${list.length} pinned spots`), ghostSelect,
      flag('sharedRange', 'same colour range on every row'),
      flag('diffFirst', 'data/model: minus the first row (scaled to the same flux)', 'shows how each spot differs from the first'),
      h('span', { class: 'spacer' }), h('button', { onclick: () => { savePins([]); renderCompare(); } }, 'Unpin all')),
      toolbar),
    grid);
  const refresh = () => { toolbar.replaceChildren(viewToggles(options, refresh)); draw(); };
  const details = await Promise.all(list.map((pin) => load(pin, ghost).catch((e) => ({ error: e.message }))));

  function draw() {
    // rows: the pinned spots, columns: every (quantity, representation) cell of the spot panel's matrix
    const panels = cells(options);
    const reference = details[0]?.images ? arrays(details[0]) : null;
    const values = details.map((d) => (d.images ? panels.map((panel) => panelValues(panel, arrays(d), options, d.ghost_box, reference)) : []));
    const shared = panels.map((_, j) => {
      const all = values.map((row) => row[j]).filter(Boolean);
      if (!all.length) return null;
      const ranges = all.map((v) => range(v.values, v.signed));
      return [Math.min(...ranges.map((r) => r[0])), Math.max(...ranges.map((r) => r[1]))];
    });
    const group = zoomGroup();
    const columns = `200px repeat(${panels.length}, minmax(220px, 1fr))`;
    const header = h('div', { style: `display:grid;grid-template-columns:${columns};gap:6px` }, h('div'), panels.map((p) => h('div', { class: 'small muted' }, panelLabel(p))));
    const toDraw = [];
    const rows = details.map((d, i) => {
      const pin = list[i];
      const remove = h('button', { class: 'small', onclick: () => { savePins(pins().filter((p) => !(p.run === pin.run && p.img === pin.img))); renderCompare(); } }, 'unpin');
      const label = h('div', { class: 'small' }, h('b', {}, pin.label), h('div', { class: 'muted', style: 'word-break:break-all' }, pin.run.split('/').pop()),
        h('div', { class: 'mono' }, pin.img),
        d.error ? h('div', { class: 'state-failed' }, d.error) : h('div', { class: 'muted' }, d.fit ? `${d.fit.source}, residual_frac ${fmt(d.residual_frac, 3)}` : 'no fit'),
        h('a', { href: `#${new URLSearchParams({ run: pin.run, img: pin.img })}` }, 'open'), ' ', remove);
      const cellsOfRow = panels.map((panel, j) => {
        const v = values[i][j];
        if (!v) return h('div', { class: 'panel muted small' }, 'n/a');
        const div = h('div', { class: 'plot', style: 'height:250px' });
        const own = range(v.values, v.signed);
        const shownRange = options.sharedRange && !(options.diffFirst && i > 0 && ['data', 'model'].includes(panel.q)) ? shared[j] : own;
        const stem = `${pin.run.split('/').pop()}_${pin.img.replace(/\.npy$/, '')}`;
        toDraw.push(() => heatmap(div, panel, v, shownRange, d.ghost_box, options).then(() => group.add(div, zoomKey(options, panel, i, j))));
        attachMenu(div, { panel, values: v, zrange: shownRange, ghostBox: d.ghost_box, options, title: `${stem}: ${panelLabel(panel)}`,
          name: `${stem}_${panel.q}_${panel.space}_${panel.scale}${panel.scale === 'power' ? panel.p : ''}` });
        return h('div', { class: 'panel' }, div);
      });
      return h('div', { style: `display:grid;grid-template-columns:${columns};gap:6px;align-items:start` }, label, cellsOfRow);
    });
    grid.replaceChildren(header, ...rows);
    for (const f of toDraw) f();
  }
  refresh();
}
