// The Jobs tab: the jobs started from the viewer, their state and log, cancel.
import { h, api, post, toast, getHash, setHash } from './ui.js';

let timer = null;

export async function runningCount() {
  const jobs = await api('/api/jobs');
  return jobs.filter((j) => j.state === 'running' || j.state === 'pending').length;
}

function resourcesText(job) {
  const r = job.resources || {};
  if (r.mode === 'gpu') return `${r.n_gpus} GPU(s): ${r.gpu_hosts.join(', ')}, on ${r.launch_host}`;
  if (r.mode === 'slurm') return `Slurm ${job.slurm_id}: ${r.cpus} cores, ${r.partition}`;
  if (r.mode === 'cpu') return `${r.cpus} cores on ${r.host}`;
  return `on ${job.host}`;
}

export async function renderJobs() {
  const root = document.getElementById('tab-jobs');
  const jobs = await api('/api/jobs');
  const selected = getHash().job || jobs[0]?.id;
  const job = jobs.find((j) => j.id === selected);
  const logText = job ? await api(`/api/jobs/${job.id}/log`) || '(no output yet)' : '';
  // The refresh rebuilds the tab: keep the log's and the table's scroll unless the log was followed at its end.
  const oldLog = root.querySelector('pre.job-log'), oldTable = root.querySelector('.table-wrap');
  const keepLog = oldLog && oldLog.dataset.job === selected && oldLog.scrollHeight - oldLog.scrollTop - oldLog.clientHeight > 4;
  const logTop = oldLog?.scrollTop, tableTop = oldTable?.scrollTop;
  const logBox = h('pre', { class: 'job-log', 'data-job': selected, style: 'max-height:55vh;min-height:200px' }, logText);
  const rows = jobs.map((j) => h('tr', { class: j.id === selected ? 'selected' : '', onclick: () => { setHash({ job: j.id }, true); renderJobs(); } },
    h('td', {}, j.started.replace('T', ' ')),
    h('td', { class: `state-${j.state}` }, j.state, j.exit_code !== undefined && j.exit_code !== null ? ` (${j.exit_code})` : ''),
    h('td', {}, h('a', { href: `#${new URLSearchParams({ run: j.run, sub: 'analyses' })}`, onclick: (e) => e.stopPropagation() }, j.run.split('/').pop())),
    h('td', {}, j.label), h('td', { class: 'muted' }, resourcesText(j)),
    h('td', {}, j.mode === 'lab' ? h('a', { href: '#tab=lab', onclick: (e) => e.stopPropagation() }, 'Lab tab') : (j.state === 'running' || j.state === 'pending') ? h('button', { class: 'small', onclick: async (e) => {
      e.stopPropagation();
      if (!confirm(`Cancel job ${j.id} (${j.label})?`)) return;
      await post(`/api/jobs/${j.id}/cancel`);
      toast(`Cancel sent to ${j.id}`);
      setTimeout(renderJobs, 1500);
    } }, 'cancel') : null)));
  root.replaceChildren(
    h('div', { class: 'card' }, h('h3', {}, 'Jobs'), jobs.length
      ? h('div', { class: 'table-wrap', style: 'max-height:32vh' }, h('table', { class: 'grid' },
        h('thead', {}, h('tr', {}, ['started', 'state', 'run', 'analysis', 'resources', ''].map((c) => h('th', {}, c)))), h('tbody', {}, rows)))
      : h('p', { class: 'muted' }, 'No job started from the viewer yet.')),
    job ? h('div', { class: 'card' }, h('h3', {}, `Log of ${job.id}`), h('div', { class: 'small muted mono', style: 'margin-bottom:6px;word-break:break-all' }, job.argv.join(' ')), logBox) : '');
  logBox.scrollTop = keepLog ? logTop : logBox.scrollHeight;
  const table = root.querySelector('.table-wrap');
  if (tableTop && table) table.scrollTop = tableTop;
  clearTimeout(timer);
  if (jobs.some((j) => j.state === 'running' || j.state === 'pending')) {
    timer = setTimeout(() => { if (root.classList.contains('active')) renderJobs(); }, 30000);
  }
}
