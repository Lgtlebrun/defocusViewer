// The Runs tab: the run list (local and pos-control), the run header and its subtabs.
import { h, api, post, q, toast, getHash, setHash, ago } from './ui.js';
import { renderSpots } from './spots.js';
import { renderPlots } from './plots.js';
import { renderAnalyses } from './analyses.js';

const SUBTABS = { spots: 'Spots', plots: 'Plots', analyses: 'Analyses', info: 'Info' };
const DEVICE_ORDER = ['Orbray6', 'MPS6', 'calibration_stand'];
let runs = [];
let listState = { search: '', remote: true, updated: null, error: null };
let shown = { run: null, sub: null };

export const runByKey = (key) => runs.find((r) => r.key === key);

export async function initRuns() {
  await loadRuns();
  setInterval(loadRuns, 30000);
}

export async function loadRuns() {
  const data = await api('/api/runs');
  runs = data.runs;
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
  if (run.local && run.flow && !run.joint_fit) out.push(h('span', { class: 'badge warn' }, 'no joint fit'));
  if (run.steps_done?.length) out.push(h('span', { class: 'badge ok', title: run.steps_done.join(', ') }, `${run.steps_done.length} steps`));
  if (run.steps_done?.includes('ghost_mask')) out.push(h('span', { class: 'badge' }, 'ghost masks'));
  if (run.viewer_fit) out.push(h('span', { class: 'badge ok', title: 'joint fit made by the viewer (results/viewer_joint_fit.json)' }, 'viewer fit'));
  if (run.data_link) out.push(h('span', { class: 'badge', title: 'data/ is a symlink: an analysis copy' }, 'copy'));
  if (run.key.startsWith('calibration_stand/')) out.push(h('span', { class: 'badge warn', title: 'no fiducial table in DefocusTest/devices.py: analyses may fail' }, 'no fiducials'));
  return out;
}

function renderList() {
  const aside = document.getElementById('run-list');
  const opened = openGroups();
  const search = listState.search.toLowerCase();
  const groups = {};
  for (const run of runs) {
    if (!run.local && !listState.remote) continue;
    if (search && !run.name.toLowerCase().includes(search)) continue;
    (groups[run.group] ||= []).push(run);
  }
  const selected = getHash().run;
  const children = Object.keys(groups).sort((a, b) => groupOrder(a) - groupOrder(b) || a.localeCompare(b)).map((group) => {
    const items = groups[group].sort((a, b) => b.name.localeCompare(a.name));
    const nLocal = items.filter((r) => r.local).length;
    const isOpen = opened[group] ?? (nLocal > 0);
    const details = h('details', { class: 'group', open: isOpen },
      h('summary', {}, `${group} `, h('span', { class: 'muted small' }, `${nLocal} local, ${items.length - nLocal} on rig`)),
      items.map((run) => {
        const { date, slug } = prettyName(run.name);
        return h('div', {
          class: `run-item${run.local ? '' : ' remote'}${run.key === selected ? ' selected' : ''}`, title: run.key,
          onclick: () => { setHash({ run: run.key, sub: null, img: null, tab: null }); showRunFromHash(); renderList(); },
        }, h('span', { class: 'name' }, h('span', { class: 'muted' }, date), ' ', slug), h('div', { class: 'badges' }, badges(run)));
      }));
    details.addEventListener('toggle', () => {
      const state = openGroups();
      state[group] = details.open;
      try { localStorage.setItem('openGroups', JSON.stringify(state)); } catch { /* storage blocked */ }
    });
    return details;
  });
  const searchBox = h('input', { type: 'text', placeholder: 'filter runs', value: listState.search, style: 'width:100%' });
  searchBox.oninput = () => { listState.search = searchBox.value; renderList(); document.querySelector('#run-list input[type=text]').focus(); };
  const remoteToggle = h('input', { type: 'checkbox', checked: listState.remote });
  remoteToggle.onchange = () => { listState.remote = remoteToggle.checked; renderList(); };
  const refresh = h('button', { class: 'small', onclick: async () => { await post('/api/remote/refresh'); toast('Listing pos-control again (a few seconds)'); setTimeout(loadRuns, 5000); } }, 'refresh');
  aside.replaceChildren(
    searchBox,
    h('div', { class: 'row small muted', style: 'margin:6px 0' },
      h('label', {}, remoteToggle, ' runs on pos-control'), h('span', { class: 'spacer' }), refresh),
    h('div', { class: 'small muted', style: 'margin-bottom:6px' },
      listState.error ? `pos-control: ${listState.error}` : `pos-control listed ${ago(listState.updated)}`),
    ...children);
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
    ({ spots: renderSpots, plots: renderPlots, analyses: renderAnalyses, info: renderInfo })[subtab](body, info);
  } else if (subtab === 'spots') {
    renderSpots(null, info);      // only the spot of the hash changed
  }
}

function runHeader(info, subtab) {
  const { date, slug } = prettyName(info.name);
  return h('div', { class: 'run-header' },
    h('h2', {}, slug, ' ', h('span', { class: 'muted small' }, `${date} · ${info.group}`)),
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
    h('p', { class: 'muted' }, 'Fetch copies it with rsync into the local data root (never deletes anything). Fetch + analyse then runs the default analysis of its flow: on up to 2 idle GPUs if there are any, else as a Slurm job of 10 cores.'),
    h('div', { class: 'row' },
      h('button', { onclick: () => fetchRun(false) }, 'Fetch'),
      h('button', { class: 'primary', onclick: () => fetchRun(true) }, 'Fetch + analyse'), ghost));
}

async function renderInfo(body, info) {
  body.replaceChildren(h('p', { class: 'pad muted' }, 'loading…'));
  const data = await api(`/api/run/info?${q({ run: info.key })}`);
  const log = h('pre', { style: 'max-height:420px' }, data.log || '(no log.log)');
  body.replaceChildren(h('div', { class: 'pad', style: 'display:flex;flex-direction:column;gap:12px' },
    h('div', { class: 'small muted mono' }, data.path),
    h('details', { open: true }, h('summary', {}, 'meta/config.yaml'), h('pre', { style: 'max-height:420px' }, data.config || '(none)')),
    h('details', {}, h('summary', {}, `results/ (${data.results.length} files)`),
      h('table', { class: 'grid' }, h('tbody', {}, data.results.map(([name, size]) =>
        h('tr', {}, h('td', { class: 'mono' }, name), h('td', {}, `${(size / 1024).toFixed(1)} kB`)))))),
    h('details', { open: true }, h('summary', {}, 'log.log (end)'), log)));
  log.scrollTop = log.scrollHeight;
}
