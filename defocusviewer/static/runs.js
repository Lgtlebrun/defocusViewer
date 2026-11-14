// The Runs tab: the run list (local and pos-control), the run header and its subtabs.
import { h, api, post, q, toast, getHash, setHash, ago } from './ui.js';
import { renderSpots } from './spots.js';
import { renderFibers } from './fibers.js';
import { renderPlots } from './plots.js';
import { renderAnalyses } from './analyses.js';

const SUBTABS = { spots: 'Spots', fibers: 'Fibers', plots: 'Plots', analyses: 'Analyses', info: 'Info' };
const DEVICE_ORDER = ['Orbray6', 'MPS6', 'calibration_stand'];
let runs = [];
let pinned = new Set();
const FILTERS_KEY = 'runFilters';      // where / analysed / tags of the run list, kept per browser
const WHERE = { all: 'all', local: 'fetched', rig: 'rig only' };
const ANALYSED = { all: 'all', yes: 'analysed', no: 'not analysed' };
const FNUM_SOURCES = { config: 'fit.fnum of meta/config.yaml', results: 'results/fnum.npy (viewer joint fit)', 'run name': 'read from the run name' };
let tagWords = [];
let listState = { search: '', updated: null, error: null, ...savedFilters() };

function savedFilters() {
  const out = { where: 'all', analysed: 'all', tags: [] };
  try {
    const saved = JSON.parse(localStorage.getItem(FILTERS_KEY)) || {};
    if (saved.where in WHERE) out.where = saved.where;
    if (saved.analysed in ANALYSED) out.analysed = saved.analysed;
    if (Array.isArray(saved.tags)) out.tags = saved.tags.filter((t) => typeof t === 'string');
  } catch { /* storage blocked or bad JSON */ }
  return out;
}

function saveFilters() {
  const { where, analysed, tags } = listState;
  try { localStorage.setItem(FILTERS_KEY, JSON.stringify({ where, analysed, tags })); } catch { /* storage blocked */ }
}
let shown = { run: null, sub: null };

export const runByKey = (key) => runs.find((r) => r.key === key);

export async function initRuns() {
  await loadRuns();
  setInterval(loadRuns, 30000);
  window.addEventListener('runschange', () => { loadRuns(); });
}

export async function loadRuns() {
  const [data, pins] = await Promise.all([api('/api/runs'), api('/api/pinned')]);
  pinned = new Set(pins);
  runs = data.runs;
  tagWords = data.tag_words || [];
  listState.updated = data.remote_updated;
  listState.error = data.remote_error;
  renderList();
}

function prettyName(name) {
  const m = name.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})\d{2}_(.*)$/);
  return m ? { date: `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}`, slug: m[6] } : { date: '', slug: name };
}

function groupOrder(group) {
  const device = group.split('/')[0];
  const i = DEVICE_ORDER.indexOf(device);
  return i < 0 ? 100 : i;
}

function openGroups() {
  try { return JSON.parse(localStorage.getItem('openGroups')) || {}; } catch { return {}; }
}

function badges(run) {
  const out = [];
  if (!run.local) out.push(h('span', { class: 'badge' }, 'on rig'));
  if (!run.finished) out.push(h('span', { class: 'badge warn' }, 'unfinished'));
  if (run.flow) out.push(h('span', { class: 'badge' }, run.flow));
  const fnum = (run.tags || []).find(isFnumTag);
  if (fnum) out.push(h('span', { class: 'badge', title: `f-number: ${FNUM_SOURCES[run.fnum_source] || run.fnum_source}` }, fnum));
  if (run.local && run.flow && !run.joint_fit) out.push(h('span', { class: 'badge warn' }, 'no joint fit'));
  if (run.steps_done?.length) out.push(h('span', { class: 'badge ok', title: run.steps_done.join(', ') }, `${run.steps_done.length} steps`));
  if (run.steps_done?.includes('ghost_mask')) out.push(h('span', { class: 'badge' }, 'ghost masks'));
  if (run.viewer_fit) out.push(h('span', { class: 'badge ok', title: 'joint fit made by the viewer (results/viewer_joint_fit.json)' }, 'viewer fit'));
  if (run.data_link) out.push(h('span', { class: 'badge', title: 'data/ is a symlink: an analysis copy' }, 'copy'));
  if (run.key.startsWith('calibration_stand/')) out.push(h('span', { class: 'badge warn', title: 'no fiducial table in DefocusTest/devices.py: analyses may fail' }, 'no fiducials'));
  return out;
}

function withVariants(items, runItem) {
  // the runs of a group, each original followed by its variants (local runs named `<original>_<suffix>`), indented
  const local = items.filter((r) => r.local);
  const parentOf = (run) => local.filter((p) => p !== run && run.local && run.name.startsWith(`${p.name}_`))
    .sort((a, b) => b.name.length - a.name.length)[0];
  const children = new Map();
  const top = [];
  for (const run of items) {
    const parent = parentOf(run);
    if (parent) (children.get(parent) || children.set(parent, []).get(parent)).push(run);
    else top.push(run);
  }
  return top.flatMap((run) => [runItem(run), ...(children.get(run) || []).sort((a, b) => a.name.localeCompare(b.name)).map((v) => runItem(v, run))]);
}

function matchesSearch(run, terms) {
  // every word of the filter must appear in "<device or label> <run name>", so "mps6 fnum8" finds the runs of that device
  const text = `${run.group.split('/')[0]} ${run.name}`.toLowerCase();
  return terms.every((t) => text.includes(t));
}

const isFnumTag = (tag) => tag.startsWith('f/');
const facetOf = (tag) => (isFnumTag(tag) ? 'fnum' : 'kind');

function passesStatus(run) {
  if ((listState.where === 'local' && !run.local) || (listState.where === 'rig' && run.local)) return false;
  return listState.analysed === 'all' || (listState.analysed === 'yes') === Boolean(run.analysed);
}

function passesTags(run, skipFacet = null) {
  // selected tags: any of a facet (kind of test, f-number), and both facets when both have some
  for (const facet of ['kind', 'fnum']) {
    if (facet === skipFacet) continue;
    const wanted = listState.tags.filter((t) => facetOf(t) === facet);
    if (wanted.length && !wanted.some((t) => (run.tags || []).includes(t))) return false;
  }
  return true;
}

function passes(run, terms, skipFacet = null) {
  return passesStatus(run) && passesTags(run, skipFacet) && (!terms.length || matchesSearch(run, terms));
}

function segmented(choices, value, onPick) {
  return h('span', { class: 'seg' }, Object.entries(choices).map(([key, label]) =>
    h('button', { class: key === value ? 'on' : '', onclick: () => onPick(key) }, label)));
}

function filterBar(terms) {
  // tag chips (counts: the runs the other filters keep that have the tag), then where / analysed
  const present = new Set(runs.flatMap((r) => r.tags || []));
  const fnums = [...present].filter(isFnumTag).sort((a, b) => parseFloat(a.slice(2)) - parseFloat(b.slice(2)));
  const kinds = [...tagWords.filter((t) => present.has(t)), ...[...present].filter((t) => !isFnumTag(t) && !tagWords.includes(t)).sort()];
  const chip = (tag) => {
    const on = listState.tags.includes(tag);
    const n = runs.filter((r) => (r.tags || []).includes(tag) && passes(r, terms, facetOf(tag))).length;
    return h('button', {
      class: `chip${on ? ' on' : ''}`, disabled: !n && !on,
      title: isFnumTag(tag) ? `f-number ${tag.slice(2)} (local runs: fit.fnum of the config; runs on the rig: their name)` : `"${tag}" in the run name or its hold routine`,
      onclick: () => {
        listState.tags = on ? listState.tags.filter((t) => t !== tag) : [...listState.tags, tag];
        saveFilters();
        renderList();
      },
    }, tag, ' ', h('span', { class: 'muted' }, n));
  };
  const set = (field) => (value) => { listState[field] = value; saveFilters(); renderList(); };
  const active = listState.tags.length || listState.where !== 'all' || listState.analysed !== 'all';
  const clear = active ? h('button', { class: 'small', onclick: () => {
    Object.assign(listState, { where: 'all', analysed: 'all', tags: [] });
    saveFilters();
    renderList();
  } }, 'clear filters') : null;
  return h('div', { class: 'run-filters' },
    h('div', { class: 'chips' }, kinds.map(chip)),
    h('div', { class: 'chips' }, fnums.map(chip)),
    h('div', { class: 'row', style: 'gap:6px' }, segmented(WHERE, listState.where, set('where')), clear),
    h('div', { class: 'row', style: 'gap:6px' }, segmented(ANALYSED, listState.analysed, set('analysed'))));
}

function renderList() {
  const aside = document.getElementById('run-list');
  const opened = openGroups();
  const search = listState.search.trim().toLowerCase();
  const terms = search.split(/\s+/).filter(Boolean);
  const forceOpen = terms.length > 0 || listState.tags.length > 0;     // a search or a tag shows its matches, whatever the folding
  const groups = {};
  for (const run of runs) {
    if (passes(run, terms)) (groups[run.group] ||= []).push(run);
  }
  const oldBox = aside.querySelector('input[type=text]');
  const keepFocus = oldBox && document.activeElement === oldBox ? [oldBox.selectionStart, oldBox.selectionEnd] : null;
  const selected = getHash().run;
  const runItem = (run, parent = null) => {
    const { date, slug } = prettyName(run.name);
    const label = parent ? [h('span', { class: 'muted' }, '↳ '), run.name.slice(parent.name.length + 1)] : [h('span', { class: 'muted' }, date), ' ', slug];
    const star = h('button', { class: 'star', title: pinned.has(run.key) ? 'unpin' : 'pin on top of the list', onclick: async (e) => {
      e.stopPropagation();
      pinned = new Set(await post('/api/pinned', { run: run.key, pinned: !pinned.has(run.key) }));
      renderList();
    } }, pinned.has(run.key) ? '★' : '☆');
    return h('div', {
      class: `run-item${run.local ? '' : ' remote'}${run.key === selected ? ' selected' : ''}${parent ? ' variant' : ''}`, title: run.key,
      onclick: () => { setHash({ run: run.key, sub: null, img: null, tab: null }); showRunFromHash(); renderList(); },
    }, h('div', { class: 'row', style: 'gap:4px;flex-wrap:nowrap;align-items:flex-start' },
      h('span', { class: 'name', style: 'flex:1' }, label), star),
    h('div', { class: 'badges' }, badges(run)));
  };
  const pinnedRuns = runs.filter((r) => pinned.has(r.key) && passes(r, terms));
  const pinnedGroup = pinnedRuns.length ? h('details', { class: 'group', open: true },
    h('summary', {}, '★ Pinned ', h('span', { class: 'muted small' }, `${pinnedRuns.length}`)),
    pinnedRuns.sort((a, b) => b.name.localeCompare(a.name)).map((r) => runItem(r))) : null;
  const children = Object.keys(groups).sort((a, b) => groupOrder(a) - groupOrder(b) || a.localeCompare(b)).map((group) => {
    const items = groups[group].sort((a, b) => b.name.localeCompare(a.name));
    const nLocal = items.filter((r) => r.local).length;
    const isOpen = forceOpen || (opened[group] ?? (nLocal > 0));
    const details = h('details', { class: 'group', open: isOpen },
      h('summary', {}, `${group} `, h('span', { class: 'muted small' }, `${nLocal} local, ${items.length - nLocal} on rig`)),
      withVariants(items, runItem));
    details.addEventListener('toggle', () => {
      if (forceOpen) return;     // opened by the filter, not by the user
      const state = openGroups();
      state[group] = details.open;
      try { localStorage.setItem('openGroups', JSON.stringify(state)); } catch { /* storage blocked */ }
    });
    return details;
  });
  const searchBox = h('input', { type: 'text', placeholder: 'filter runs', value: listState.search, style: 'width:100%' });
  searchBox.oninput = () => { listState.search = searchBox.value; renderList(); };
  const refresh = h('button', { class: 'small', onclick: async () => { await post('/api/remote/refresh'); toast('Listing pos-control again (a few seconds)'); setTimeout(loadRuns, 5000); } }, 'refresh');
  const nShown = Object.values(groups).reduce((n, items) => n + items.length, 0);
  aside.replaceChildren(
    searchBox,
    filterBar(terms),
    h('div', { class: 'row small muted', style: 'margin:6px 0' },
      h('span', {}, listState.error ? `pos-control: ${listState.error}` : `pos-control listed ${ago(listState.updated)}`),
      h('span', { class: 'spacer' }), refresh),
    pinnedGroup, ...children,
    nShown ? null : h('p', { class: 'small muted' }, 'No run matches the filters.'));
  if (keepFocus) {      // the box is rebuilt with the list: put the caret back where it was (also on the 30 s refresh)
    searchBox.focus();
    searchBox.setSelectionRange(...keepFocus);
  }
}

export function showRunFromHash() {
  const { run, sub } = getHash();
  const view = document.getElementById('run-view');
  if (!run) return;
  const info = runByKey(run);
  if (!info) {
    view.replaceChildren(h('p', { class: 'pad muted' }, `Unknown run ${run}`));
    return;
  }
  if (!info.local) {
    shown = { run: null, sub: null };
    view.replaceChildren(remotePanel(info));
    return;
  }
  const subtab = SUBTABS[sub] ? sub : 'spots';
  if (shown.run !== run || shown.sub !== subtab) {
    shown = { run, sub: subtab };
    const body = h('div', { class: 'run-body' });
    view.replaceChildren(runHeader(info, subtab), body);
    ({ spots: renderSpots, fibers: renderFibers, plots: renderPlots, analyses: renderAnalyses, info: renderInfo })[subtab](body, info);
  } else if (subtab === 'spots') {
    renderSpots(null, info);      // only the spot of the hash changed
  }
}

function variantSelector(info) {
  // the run and its variants (other source / aperture models, copies): picking one shows it in the same tab and spot
  const holder = h('span', { class: 'row', style: 'gap:4px;margin-left:8px' });
  api(`/api/variants?${q({ run: info.key })}`).then((v) => {
    if (v.family.length < 2) return;
    const select = h('select', {}, v.family.map((m) => h('option', { value: m.key, selected: m.key === info.key },
      `${m.suffix ?? 'original'} · ${m.model || '?'}`)));
    select.onchange = () => { setHash({ run: select.value }); showRunFromHash(); renderList(); };
    holder.replaceChildren(h('span', { class: 'muted small' }, 'analysis'), select);
  }).catch(() => {});
  return holder;
}

function runHeader(info, subtab) {
  const { date, slug } = prettyName(info.name);
  return h('div', { class: 'run-header' },
    h('h2', { class: 'row', style: 'gap:6px' }, slug, h('span', { class: 'muted small' }, `${date} · ${info.group}`), variantSelector(info)),
    h('div', { class: 'badges', style: 'margin-top:4px' }, badges(info)),
    h('div', { class: 'subtabs' }, Object.entries(SUBTABS).map(([key, label]) =>
      h('button', { class: key === subtab ? 'active' : '', onclick: () => { setHash({ sub: key === 'spots' ? null : key }); showRunFromHash(); } }, label))));
}

function remotePanel(info) {
  const ghost = h('select', {}, h('option', { value: 'on' }, 'ghost mask on'), h('option', { value: 'off' }, 'ghost mask off'));
  const fetchRun = async (analyse) => {
    try {
      const job = await post('/api/fetch', { run: info.key, analyse, ghost: ghost.value });
      toast(`Started job ${job.id}: ${job.label}. See the Jobs tab.`);
    } catch (e) { toast(e.message, true); }
  };
  return h('div', { class: 'pad' },
    h('h2', {}, info.name),
    h('p', {}, `Only on pos-control (${info.group}), ${info.finished ? 'finished' : 'not finished: the copy takes what is there now'}.`),
    h('p', { class: 'muted' }, 'Fetch copies it with rsync into the local data root (never deletes anything). Fetch + analyse then runs the default analysis of its flow: on up to 2 idle GPUs if there are any, else on up to 32 cores of the CPU host with the most idle ones (over SSH), else as a Slurm job of 10 cores.'),
    h('div', { class: 'row' },
      h('button', { onclick: () => fetchRun(false) }, 'Fetch'),
      h('button', { class: 'primary', onclick: () => fetchRun(true) }, 'Fetch + analyse'), ghost));
}

async function renderInfo(body, info) {
  body.replaceChildren(h('p', { class: 'pad muted' }, 'loading…'));
  const data = await api(`/api/run/info?${q({ run: info.key })}`);
  const log = h('pre', { style: 'max-height:420px' }, data.log || '(no log.log)');
  const cards = (data.summary || []).map((section) => h('div', { class: 'card' }, h('h3', {}, section.title),
    h('dl', { class: 'kv' }, section.rows.map(([label, value]) => [h('dt', {}, label), h('dd', { style: 'font-family:var(--font);font-size:12.5px' }, value)]))));
  body.replaceChildren(h('div', { class: 'pad', style: 'display:flex;flex-direction:column;gap:12px' },
    h('div', { class: 'small muted mono' }, data.path),
    h('div', { class: 'summary-grid' }, cards),
    h('details', {}, h('summary', {}, 'meta/config.yaml'), h('pre', { style: 'max-height:420px' }, data.config || '(none)')),
    h('details', {}, h('summary', {}, `results/ (${data.results.length} files)`),
      h('table', { class: 'grid' }, h('tbody', {}, data.results.map(([name, size]) =>
        h('tr', {}, h('td', { class: 'mono' }, name), h('td', {}, `${(size / 1024).toFixed(1)} kB`)))))),
    h('details', { open: true }, h('summary', {}, 'log.log (end)'), log)));
  log.scrollTop = log.scrollHeight;
}
