// The Cluster tab: GPUs and cores of the GPU hosts, Slurm partitions and nodes, our Slurm jobs.
import { h, api, ago } from './ui.js';

let timer = null;

function bar(fraction, text) {
  const pct = Math.max(0, Math.min(1, fraction)) * 100;
  return h('div', { class: 'bar', title: text }, h('span', { style: `width:${pct.toFixed(1)}%` }), h('em', {}, text));
}

export async function clusterSummary() {
  const c = await api('/api/cluster');
  const free = c.hosts.filter((host) => window.viewerConfig.job_gpu_hosts.includes(host.host))
    .reduce((n, host) => n + (host.free_gpus?.length || 0), 0);
  const usable = c.partitions.find((p) => p.partition === 'standard')?.usable ?? 0;
  return `${free} GPU(s) free for us · ${usable} Slurm cores usable now (standard)`;
}

function hostCard(host) {
  if (!host.reachable) {
    return h('div', { class: 'card' }, h('h3', {}, host.host), h('p', { class: 'state-failed' }, host.reachable === null ? 'not sampled yet' : 'unreachable'),
      host.hint ? h('p', { class: 'small mono' }, host.hint) : null);
  }
  const used = Math.min(host.cores, host.load1);
  const gpus = host.gpus.length ? host.gpus.map((g) => {
    const owner = g.processes ? `${g.processes} process(es), ${g.own_processes} ours` : 'idle';
    const free = host.free_gpus.includes(g.index);
    return h('div', { class: 'gpu-row' },
      h('span', {}, `GPU ${g.index} `, h('span', { class: 'muted' }, g.name.replace('NVIDIA ', '')), free ? h('span', { class: 'badge ok', style: 'margin-left:4px' }, 'free for us') : null),
      bar(g.utilization / 100, `${g.utilization} % busy`),
      bar(g.memory_used_mb / g.memory_total_mb, `${(g.memory_used_mb / 1024).toFixed(1)} / ${(g.memory_total_mb / 1024).toFixed(0)} GB · ${owner}`));
  }) : [h('p', { class: 'state-failed small' }, 'no GPU reported (nvidia-smi failed?)')];
  return h('div', { class: 'card' },
    h('h3', {}, host.host, host.cap ? h('span', { class: 'muted small' }, ` (at most ${host.cap} GPU of ours at once)`) : null,
      window.viewerConfig.job_gpu_hosts.includes(host.host) ? null
        : h('span', { class: 'badge warn', style: 'margin-left:6px', title: 'not in job_gpu_hosts of ~/.config/defocusViewer/config.toml' }, 'not used by job defaults')),
    h('div', { class: 'gpu-row' }, h('span', {}, `${host.cores} cores`), bar(used / host.cores, `load ${host.load1.toFixed(1)}`),
      bar(1 - host.mem_available_gb / host.mem_total_gb, `${(host.mem_total_gb - host.mem_available_gb).toFixed(0)} / ${host.mem_total_gb.toFixed(0)} GB RAM`)),
    gpus);
}

export async function renderCluster() {
  const root = document.getElementById('tab-cluster');
  const c = await api('/api/cluster');
  const nodes = [...c.nodes].sort((a, b) => b.usable - a.usable || a.node.localeCompare(b.node));
  const jobs = Object.entries(c.jobs);
  root.replaceChildren(
    h('div', { class: 'small muted' }, `GPU hosts sampled ${ago(c.updated.hosts)}, Slurm ${ago(c.updated.slurm)}. "Usable" cores: idle cores on nodes with ${(c.mem_per_cpu_mb / 1024).toFixed(1)} GB of unreserved memory per core (Slurm here reserves memory, which often runs out first).`),
    h('div', { class: 'hosts' }, c.hosts.map(hostCard)),
    h('div', { class: 'card' }, h('h3', {}, 'Slurm partitions'), h('table', { class: 'grid' },
      h('thead', {}, h('tr', {}, ['partition', 'nodes', 'cores allocated', 'idle', 'usable now'].map((x) => h('th', {}, x)))),
      h('tbody', {}, c.partitions.map((p) => h('tr', {}, h('td', {}, p.partition), h('td', {}, p.nodes),
        h('td', {}, bar(p.cpus_allocated / p.cpus_total, `${p.cpus_allocated} / ${p.cpus_total}`)), h('td', {}, p.cpus_idle), h('td', {}, h('b', {}, p.usable))))))),
    h('div', { class: 'card' }, h('h3', {}, `Our Slurm jobs (${c.user})`), jobs.length ? h('table', { class: 'grid' },
      h('thead', {}, h('tr', {}, ['job', 'name', 'partition', 'states', 'nodes', 'elapsed / limit', 'cores', 'memory'].map((x) => h('th', {}, x)))),
      h('tbody', {}, jobs.map(([id, j]) => h('tr', {}, h('td', {}, id), h('td', {}, j.name), h('td', {}, j.part),
        h('td', {}, Object.entries(j.states).map(([s, n]) => `${s} ${n}`).join(', '), j.reason ? ` (${j.reason})` : ''),
        h('td', {}, j.nodes.join(', ')), h('td', {}, `${j.elapsed} / ${j.limit}`), h('td', {}, j.cpus), h('td', {}, j.mem))))) : h('p', { class: 'muted small' }, 'none')),
    h('details', { class: 'card' }, h('summary', {}, `Slurm nodes (${nodes.length})`), h('table', { class: 'grid' },
      h('thead', {}, h('tr', {}, ['node', 'state', 'cores allocated', 'idle', 'memory reserved', 'usable now'].map((x) => h('th', {}, x)))),
      h('tbody', {}, nodes.map((n) => h('tr', {}, h('td', {}, n.node), h('td', {}, n.state),
        h('td', {}, bar(n.cpus_allocated / n.cpus_total, `${n.cpus_allocated} / ${n.cpus_total}`)), h('td', {}, n.cpus_idle),
        h('td', {}, bar(n.memory_allocated_mb / n.memory_mb, `${(n.memory_allocated_mb / 1024).toFixed(0)} / ${(n.memory_mb / 1024).toFixed(0)} GB`)),
        h('td', {}, n.usable)))))));
  clearTimeout(timer);
  timer = setTimeout(() => { if (root.classList.contains('active')) renderCluster(); }, 10000);
}
