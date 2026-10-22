// Entry point: tabs, theme, URL hash, and the periodic header counters.
import { api, getHash, setHash } from './ui.js';
import { initRuns, showRunFromHash } from './runs.js';
import { renderCompare, pinCount } from './compare.js';
import { renderJobs, runningCount } from './jobs.js';
import { renderCluster, clusterSummary } from './cluster.js';

const TABS = ['runs', 'compare', 'jobs', 'cluster'];
const renderers = { compare: renderCompare, jobs: renderJobs, cluster: renderCluster };
let current = null;

function showTab(tab) {
  if (!TABS.includes(tab)) tab = 'runs';
  current = tab;
  for (const t of TABS) {
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab);
    document.querySelector(`#tabs button[data-tab=${t}]`).classList.toggle('active', t === tab);
  }
  if (renderers[tab]) renderers[tab]();
}

export function currentTab() { return current; }

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem('theme'); } catch { /* storage blocked */ }
  applyTheme(saved);
  document.getElementById('theme').onclick = () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    const next = dark ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem('theme', next); } catch { /* storage blocked */ }
    window.dispatchEvent(new Event('themechange'));
  };
}

async function tick() {
  document.getElementById('pin-count').textContent = pinCount() || '';
  try {
    const n = await runningCount();
    document.getElementById('job-count').textContent = n || '';
    document.getElementById('cluster-summary').textContent = await clusterSummary();
  } catch { /* shown on the tabs themselves */ }
}

function route() {
  const hash = getHash();
  showTab(hash.tab || 'runs');
  if ((hash.tab || 'runs') === 'runs') showRunFromHash();
}

initTheme();
for (const button of document.querySelectorAll('#tabs button')) {
  button.onclick = () => { setHash({ tab: button.dataset.tab === 'runs' ? null : button.dataset.tab }); route(); };
}
window.addEventListener('popstate', route);
window.addEventListener('pinschange', () => { document.getElementById('pin-count').textContent = pinCount() || ''; });
await api('/api/config').then((config) => { window.viewerConfig = config; });
await initRuns();
route();
tick();
setInterval(tick, 10000);
