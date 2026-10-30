// The Lab tab: runs on the rig (pos-control) with their phase and progress, the stage and the mount, who holds which
// device, launching a routine there and stopping it. Everything is read from files and processes: the viewer never
// opens rig hardware.
import { h, api, post, toast, modal, closeModal, ago, cssVar, PLOT_CONFIG, getHash, setHash } from './ui.js';

let timer = null;
let form = null;            // the launch form survives the 10 s refreshes

const when = (t) => (t ? new Date(t * 1000).toLocaleString('en-GB', { dateStyle: 'short', timeStyle: 'short' }) : '');
const pct = (f) => (f === null || f === undefined ? '' : `${(100 * f).toFixed(f < 0.1 ? 1 : 0)} %`);

export async function labSummary() {
  const s = await api('/api/lab');
  return (s.runs || []).filter((r) => r.state === 'running').length;
}

function bar(fraction, text) {
  const f = Math.max(0, Math.min(1, fraction || 0)) * 100;
  return h('div', { class: 'bar', title: text }, h('span', { style: `width:${f.toFixed(1)}%` }), h('em', {}, text));
}

function progressText(c) {
  if (!c) return '';
  if (c.kind === 'sweep') return `${c.name}: sweep ${c.done} / ${c.total} positions`;
  if (c.kind === 'hold') return `${c.name}: hold ${pct(c.fraction)}, ${c.stacks} stacks, ends ${when(c.end)}`;
  if (c.kind === 'burn-in') return `${c.name}: burn-in ${pct(c.fraction)}, ends ${when(c.end)}`;
  return `${c.name}`;
}

const STATE_CLASS = { running: 'state-running', done: 'state-done', stopped: 'state-done', failed: 'state-failed', died: 'state-failed' };

function runCard(run, jobsByRun, snapshot) {
  const job = jobsByRun[run.rundir];
  const running = run.state === 'running';
  const phases = h('div', { class: 'badges' }, run.phases.map((p) => h('span', {
    class: `badge ${p.state === 'done' ? 'ok' : p.state === 'running' ? 'run' : 'warn'}`,
    title: `${when(p.start)}${p.end ? ` → ${when(p.end)}` : ''}`,
  }, p.name)));
  const stop = running ? h('button', {
    disabled: !run.hook || run.stop_requested,
    title: run.hook ? 'ends the current acquisition between two stacks, skips the rest, closes stage and camera, switches the LEDs off'
      : 'this run was started without the stop hook (no meta/status.json): it can only be killed',
    onclick: async () => {
      if (!confirm(`Stop ${run.name}?\n\nThe current routine ends after its stack, the rest of the sequence is skipped.`)) return;
      await post('/api/lab/stop', { rundir: run.rundir });
      toast(`Stop requested for ${run.name}`);
      setTimeout(renderLab, 1500);
    },
  }, run.stop_requested ? 'stopping…' : 'Stop') : null;
  const kill = running && job && job.state === 'running' ? h('button', {
    class: 'small', title: 'SIGTERM to the process group: no clean-up, the LEDs stay on, no meta/_finished',
    onclick: async () => {
      if (!confirm(`Kill ${run.name} (pid ${job.pid})?\n\nNo clean-up: the PWM LEDs stay on and no meta/_finished is written. Use Stop unless the run hangs.`)) return;
      await post('/api/lab/kill', { job: job.id });
      toast(`Kill sent to ${job.id}`);
      setTimeout(renderLab, 1500);
    },
  }, 'Kill') : null;
  const logButton = h('button', { class: 'small', onclick: () => showLog(run) }, 'log');
  const status = run.status || {};
  return h('div', { class: 'card' },
    h('h3', {}, run.name, ' ', h('span', { class: STATE_CLASS[run.state] || 'muted' }, run.state),
      run.stalled ? h('span', { class: 'badge warn', style: 'margin-left:6px' }, `no log line for ${ago(run.last_line)}`) : null,
      job ? h('a', { class: 'small', style: 'margin-left:8px', href: `#tab=jobs&job=${job.id}` }, `viewer job ${job.id}`) : null),
    h('div', { class: 'small muted' }, `${run.device} · started ${when(run.started)} · last log line ${ago(run.last_line)}`,
      run.config['stability.duration_s'] ? ` · hold ${(run.config['stability.duration_s'] / 3600).toFixed(1)} h` : '',
      run.config['train.camera.burn_in_s'] ? ` · burn-in ${(run.config['train.camera.burn_in_s'] / 3600).toFixed(1)} h` : ''),
    phases,
    run.current ? h('div', { style: 'margin:6px 0' }, bar(run.current.fraction, progressText(run.current))) : null,
    h('div', { class: 'row small' },
      run.stage ? h('span', {}, `stage ${run.stage.x_um.toFixed(1)} µm `, h('span', { class: 'muted' }, `(${ago(run.stage.t)})`)) : null,
      status.texp ? h('span', { class: 'muted' }, `texp ${status.texp} µs`) : null,
      run.hook ? null : h('span', { class: 'badge warn', title: 'started before the routines wrote meta/status.json and checked meta/_stop_requested' }, 'no stop hook'),
      run.finished ? h('span', { class: 'badge ok' }, '_finished') : null,
      job?.lab?.analysis_job ? h('a', { href: `#tab=jobs&job=${job.lab.analysis_job}` }, 'analysis job') : null,
      job?.lab?.analysis_error ? h('span', { class: 'state-failed', title: job.lab.analysis_error }, 'analysis not started') : null,
      h('a', { href: `#${new URLSearchParams({ run: run.key })}`, title: 'the local copy, once fetched' }, 'open run'),
      logButton, stop, kill),
    run.errors.length ? h('details', {}, h('summary', { class: 'state-failed small' }, `${run.errors.length} error line(s)`),
      h('pre', { style: 'max-height:200px' }, run.errors.slice(-20).join('\n'))) : null);
}

async function showLog(run) {
  const pre = h('pre', { style: 'max-height:70vh;min-height:300px;white-space:pre-wrap' }, 'loading…');
  modal([h('h3', {}, `log.log of ${run.name} (last 64 kB)`), pre, h('div', { class: 'row', style: 'margin-top:8px' }, h('button', { onclick: closeModal }, 'close'))]);
  document.querySelector('#modal .dialog').style.maxWidth = '1100px';
  try {
    pre.textContent = await api(`/api/lab/log?${new URLSearchParams({ rundir: run.rundir })}`);
    pre.scrollTop = pre.scrollHeight;
  } catch (e) {
    pre.textContent = `could not read the log: ${e.message}`;
  }
}

function stageCard(run) {
  const div = h('div', { style: 'height:220px' });
  const card = h('div', { class: 'card' }, h('h3', {}, 'Stage', run ? h('span', { class: 'muted small' }, ` (${run.name})`) : null),
    run?.stage ? h('div', {}, h('b', {}, `${run.stage.x_um.toFixed(2)} µm`), h('span', { class: 'muted small' },
      ` ${run.state === 'running' ? 'last reported' : 'last known'} ${ago(run.stage.t)} (${run.hook ? 'status.json readback / ' : ''}log "Arrived to position")`))
      : h('p', { class: 'muted small' }, 'no position logged by a followed run'),
    div);
  if (run?.stage_trace?.length) {
    const axis = { color: cssVar('--text-secondary'), gridcolor: cssVar('--border'), zeroline: false };
    requestAnimationFrame(() => Plotly.react(div, [{
      x: run.stage_trace.map(([t]) => new Date(t * 1000)), y: run.stage_trace.map(([, x]) => x),
      mode: 'lines+markers', line: { shape: 'hv', width: 1.5 }, marker: { size: 4 }, name: 'stage',
    }], {
      margin: { l: 60, r: 8, t: 8, b: 30 }, paper_bgcolor: cssVar('--surface-1'), plot_bgcolor: cssVar('--surface-1'),
      font: { size: 11, color: cssVar('--text-secondary') }, xaxis: axis, yaxis: { ...axis, title: 'µm' }, showlegend: false,
    }, PLOT_CONFIG));
  } else {
    div.style.height = '0';
  }
  return card;
}

function mountCard(snapshot, run) {
  const dev = snapshot.devices?.mount;
  const status = run?.status || {};
  const m = run?.mount;
  let position;
  if (status.mount_alt_deg !== undefined) {
    position = h('div', {}, h('b', {}, `alt ${status.mount_alt_deg.toFixed(3)}° · az ${status.mount_az_deg.toFixed(3)}°`), h('span', { class: 'muted small' }, ` (${status.t})`));
  } else if (m?.t) {
    position = h('div', {}, h('b', {}, `alt ${m.alt_deg.toFixed(4)}° · az ${m.az_deg.toFixed(4)}°`),
      h('span', { class: 'muted small' }, ` ${run.state === 'running' ? 'last reported' : 'last known'} ${ago(m.t)} (log of ${run.name})`));
  } else {
    position = h('p', { class: 'muted small' }, 'alt / az unknown: no followed routine logs the mount, and the viewer does not open the device');
  }
  return h('div', { class: 'card' }, h('h3', {}, 'Mount (CEM120)'),
    dev ? h('div', { class: 'small' }, h('span', { class: dev.present ? 'state-done' : 'muted' }, dev.present ? 'connected' : 'not connected'),
      ` · ${dev.path}`, dev.holders.length ? ` · held by pid ${dev.holders.join(', ')}` : ' · not open') : h('p', { class: 'muted small' }, 'not sampled'),
    position,
    m?.state ? h('div', { class: 'small' }, `${m.state} since ${when(m.since)}`,
      m.target ? ` · target alt ${m.target[0]}° · az ${m.target[1]}°` : '') : null);
}

function rigCard(snapshot) {
  const procs = Object.fromEntries((snapshot.processes || []).map((p) => [p.pid, p]));
  const services = snapshot.services ? new RegExp(snapshot.services) : null;
  const holder = (pid) => {
    const p = procs[pid];
    const service = p && services && services.test(p.args);
    return h('span', { title: p?.args || '', class: service ? 'muted' : '' }, `${pid}${service ? ' (service)' : ''}`);
  };
  return h('details', { class: 'card', open: !!snapshot.busy?.length && !(snapshot.runs || []).some((r) => r.state === 'running') },
    h('summary', {}, `Rig: devices and processes (${snapshot.host})`),
    h('table', { class: 'grid' }, h('thead', {}, h('tr', {}, ['device', 'path', '', 'held by'].map((x) => h('th', {}, x)))),
      h('tbody', {}, Object.entries(snapshot.devices || {}).map(([name, d]) => h('tr', {}, h('td', {}, name),
        h('td', { class: 'mono small' }, d.path === d.real ? d.path : `${d.path} → ${d.real}`),
        h('td', { class: d.present ? 'state-done' : 'muted' }, d.present ? 'present' : 'absent'),
        h('td', {}, d.holders.length ? d.holders.flatMap((pid, i) => [i ? ', ' : '', holder(pid)]) : h('span', { class: 'muted' }, 'free')))))),
    h('table', { class: 'grid', style: 'margin-top:8px' }, h('thead', {}, h('tr', {}, ['pid', 'running since', 'process', 'cwd'].map((x) => h('th', {}, x)))),
      h('tbody', {}, (snapshot.processes || []).map((p) => h('tr', {}, h('td', {}, p.pid), h('td', {}, when(p.started)),
        h('td', { class: 'mono small', style: 'max-width:520px;overflow:hidden;text-overflow:ellipsis', title: p.args }, p.args),
        h('td', { class: 'mono small muted' }, p.cwd || ''))))));
}

function fieldInput(f) {
  const show = (v) => (Array.isArray(v) ? v.join(', ') : v ?? '');
  if (f.kind === 'bool') return h('input', { type: 'checkbox', checked: !!f.value });
  if (f.kind === 'choice') {
    return h('select', {}, f.choices.map((c, i) => h('option', { value: i, selected: c === f.value }, String(c))));
  }
  if (f.kind === 'int_list') return h('input', { type: 'text', value: show(f.value), style: 'width:100%' });
  if (f.kind === 'str') return h('input', { type: 'text', value: show(f.value), pattern: f.pattern });
  return h('input', { type: 'number', value: show(f.value), min: f.lo, max: f.hi, step: f.kind === 'int' ? 1 : 'any', style: 'width:12em' });
}

function readInput(f, el) {
  if (f.kind === 'bool') return el.checked;
  if (f.kind === 'choice') return f.choices[Number(el.value)];
  if (f.kind === 'int_list') return el.value.split(/[\s,]+/).filter(Boolean).map(Number);
  if (f.kind === 'str') return el.value;
  return el.value === '' ? null : Number(el.value);
}

async function buildForm(routine) {
  const body = h('div', {}, h('p', { class: 'muted small' }, 'reading the config on the rig…'));
  form = { routine, el: h('div', { class: 'card' }, h('h3', {}, 'Launch on the rig'), body) };
  let spec;
  try {
    spec = await api(`/api/lab/routine?${new URLSearchParams({ name: routine })}`);
  } catch (e) {
    body.replaceChildren(h('p', { class: 'state-failed small' }, `could not read the routine's config: ${e.message}`),
      h('button', { onclick: () => { form = null; renderLab(); } }, 'retry'));
    return form;
  }
  const inputs = spec.fields.map((f) => [f, fieldInput(f)]);
  const analyse = h('input', { type: 'checkbox', checked: true });
  const overrides = () => {
    const out = {};
    for (const [f, el] of inputs) {
      const v = readInput(f, el);
      if (JSON.stringify(v) !== JSON.stringify(f.value)) out[f.key] = v;
    }
    return out;
  };
  const launch = async () => {
    const request = { routine, overrides: overrides(), analyse_after: analyse.checked };
    let preview;
    try {
      preview = await post('/api/lab/launch', request);
    } catch (e) {
      toast(e.message, true);
      return;
    }
    const go = h('button', { class: 'primary', disabled: preview.busy.length > 0, onclick: async () => {
      go.disabled = true;
      try {
        const job = await post('/api/lab/launch', { ...request, confirm: true });
        closeModal();
        toast(`${job.label} started on the rig (pid ${job.pid})`);
        setTimeout(renderLab, 3000);
      } catch (e) {
        toast(e.message, true);
        go.disabled = false;
      }
    } }, 'Launch');
    modal([
      h('h3', {}, `Launch ${routine} on the rig?`),
      h('p', { class: 'small muted' }, `base: ${spec.config}`),
      preview.changes.length ? h('table', { class: 'grid' }, h('thead', {}, h('tr', {}, ['setting', 'base', 'this run'].map((x) => h('th', {}, x)))),
        h('tbody', {}, preview.changes.map((c) => h('tr', {}, h('td', {}, c.key), h('td', { class: 'muted', style: 'white-space:normal' }, JSON.stringify(c.old)),
          h('td', { style: 'white-space:normal' }, h('b', {}, JSON.stringify(c.new)))))))
        : h('p', { class: 'small' }, 'No change: the base config as it is.'),
      preview.busy.length ? h('div', { class: 'state-failed small', style: 'margin-top:8px' }, h('b', {}, 'The rig is busy:'), h('ul', {}, preview.busy.map((b) => h('li', {}, b))))
        : h('p', { class: 'small state-done' }, 'Rig free: no routine running, stage, cameras and LED arrays not held.'),
      h('details', {}, h('summary', { class: 'small' }, 'full config'), h('pre', { style: 'max-height:300px' }, preview.yaml)),
      h('div', { class: 'row', style: 'margin-top:10px' }, go, h('button', { onclick: closeModal }, 'cancel'))]);
  };
  body.replaceChildren(
    h('p', { class: 'small muted' }, `Values from ${spec.config} on the rig; only the changed ones are sent. `,
      'The run then appears above. Preview first: nothing starts before you confirm.'),
    h('table', { class: 'grid lab-form' }, h('tbody', {}, inputs.map(([f, el]) => h('tr', {},
      h('td', { class: 'mono small' }, f.key), h('td', {}, el), h('td', { class: 'muted small', style: 'white-space:normal' }, f.label))))),
    h('div', { class: 'row', style: 'margin-top:8px' },
      h('label', { class: 'small' }, analyse, ' fetch + analyse on the cluster when it finishes'),
      h('button', { class: 'primary', onclick: launch }, 'Preview launch…'),
      h('button', { onclick: () => { form = null; renderLab(); } }, 'reset to base')));
  return form;
}

export async function renderLab() {
  const root = document.getElementById('tab-lab');
  let snapshot;
  try {
    snapshot = await api('/api/lab');
  } catch (e) {
    root.replaceChildren(h('p', { class: 'state-failed pad' }, `Lab state unavailable: ${e.message}`));
    return;
  }
  const runs = [...(snapshot.runs || [])].sort((a, b) => (b.state === 'running') - (a.state === 'running') || (b.started || 0) - (a.started || 0));
  const jobsByRun = Object.fromEntries((snapshot.jobs || []).filter((j) => j.lab?.rundir).map((j) => [j.lab.rundir, j]));
  const selected = runs.find((r) => r.rundir === getHash().labrun) || runs[0];
  const pending = (snapshot.jobs || []).filter((j) => j.state === 'running' && !j.lab?.rundir);
  if (!form) await buildForm(snapshot.routines?.[0] || 'StabilityTest_new');
  const monitor = h('div', { class: 'lab-monitor' },
    h('div', { class: 'small muted' }, `${snapshot.host} sampled ${ago(snapshot.updated)}`,
      snapshot.error ? h('span', { class: 'state-failed' }, ` · ${snapshot.error}`) : null,
      ' · positions and progress come from the runs\' files (log.log, meta/status.json), never from the devices'),
    pending.map((j) => h('div', { class: 'card small' }, `${j.label}: started ${j.started.replace('T', ' ')}, waiting for its rundir… `,
      h('a', { href: `#tab=jobs&job=${j.id}` }, 'output'))),
    runs.length ? runs.map((r) => runCard(r, jobsByRun, snapshot)) : h('div', { class: 'card muted small' }, 'No routine running, and no recent run launched from the viewer.'),
    h('div', { class: 'hosts' }, stageCard(selected), mountCard(snapshot, selected)),
    runs.length > 1 ? h('div', { class: 'row small' }, 'stage of: ', h('select', { onchange: (e) => { setHash({ labrun: e.target.value }, true); renderLab(); } },
      runs.map((r) => h('option', { value: r.rundir, selected: r === selected }, r.name)))) : null,
    rigCard(snapshot));
  root.replaceChildren(monitor, form.el);
  clearTimeout(timer);
  timer = setTimeout(() => { if (root.classList.contains('active')) renderLab(); }, 10000);
}
