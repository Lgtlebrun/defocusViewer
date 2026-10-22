// Small DOM and API helpers shared by the tabs.

export const EXPORT_DPI = 400;
// Plotly's PNG export (camera button, right-click copy / save): the plot's CSS pixels are 96 dpi
export const PLOT_CONFIG = { displaylogo: false, responsive: true, toImageButtonOptions: { format: 'png', scale: EXPORT_DPI / 96 } };

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k in el && typeof v !== 'string') el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === undefined || c === null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export async function api(path, options = {}) {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  if (response.status === 401) {
    toast('Not logged in: open the URL printed by `defocusviewer url` on farm-gpu1.', true);
    throw new Error('not logged in');
  }
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail; } catch { /* not JSON */ }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  const type = response.headers.get('content-type') || '';
  return type.includes('json') ? response.json() : response.text();
}

export const post = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body || {}) });
export const q = (params) => new URLSearchParams(params).toString();

let toastTimer;
export function toast(message, error = false) {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.className = error ? 'error' : '';
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, error ? 9000 : 4000);
}

export function modal(content) {
  const root = document.getElementById('modal');
  root.replaceChildren(h('div', { class: 'dialog' }, content));
  root.hidden = false;
  root.onclick = (e) => { if (e.target === root) closeModal(); };
  return root;
}

export function closeModal() {
  document.getElementById('modal').hidden = true;
}

export function fmt(value, digits = 4) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(digits);
  return String(value);
}

export function ago(seconds) {
  if (!seconds) return 'never';
  const s = Math.round(Date.now() / 1000 - seconds);
  return s < 90 ? `${s} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
}

// The URL hash holds the view: #run=<key>&sub=<subtab>&img=<spot>
export function getHash() {
  return Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
}

export function setHash(update, replace = false) {
  const next = { ...getHash(), ...update };
  for (const k of Object.keys(next)) if (next[k] === null || next[k] === undefined || next[k] === '') delete next[k];
  const hash = '#' + new URLSearchParams(next).toString();
  if (hash !== location.hash) (replace ? history.replaceState(null, '', hash) : history.pushState(null, '', hash));
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
