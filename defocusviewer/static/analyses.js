// The Analyses subtab, the joint-fit prompt of the spot panel, and the resources editor ("Advanced settings").
import { h, api, post, q, toast, modal, closeModal } from './ui.js';

const GPU_STEPS = ['fit', 'ghost_mask', 'refit', 'series'];
const STEP_HELP = {
  fit: 'joint fit of the train sweep (theta, pupil scales, per-image r/x0/y0/Zernikes)',
  ghost_mask: 'one ghost box per fiber, found in the residuals of the fit',
  refit: 'the joint fit again, with weight 0 on the ghost boxes',
  joint_checks: 'joint-fit healthcheck figures',
  series: 'per-frame fits of the hold, and the train_end fits',
  model_fits: 'redraw the data / model / residual figures (opt-in)',
  overlay: 'train vs train_end curves',
  relative: 'stability relative to each fiber',
  calibration: 'fiducial calibration',
  temperature: 'temperature plots',
};

function storedGhost() {
  try { return localStorage.getItem('ghost') || 'on'; } catch { return 'on'; }
}

export function resourcesEditor(defaults, freeGpus, cpuDefaults) {
  // <details> "Advanced settings" with a one-line summary; value() returns the resources of the job
  const config = window.viewerConfig;
  const gpuDefaults = defaults.mode === 'gpu' ? defaults : { n_gpus: 2, gpu_hosts: config.job_gpu_hosts, cpu_workers: 4, launch_host: config.launch_hosts[0] };
  const slurmDefaults = defaults.mode === 'slurm' ? defaults : cpuDefaults;
  const mode = { value: defaults.mode, chosen: false };      // chosen: the user picked it, the step boxes no longer change it
  const group = `mode-${Math.random().toString(36).slice(2)}`;   // one radio group per editor
  const radio = (value, label) => {
    const input = h('input', { type: 'radio', name: group, value, checked: mode.value === value });
    input.onchange = () => { mode.value = value; mode.chosen = true; update(); };
    return h('label', {}, input, ' ', label);
  };
  const nGpus = h('input', { type: 'number', min: 1, max: 8, value: gpuDefaults.n_gpus });
  const hostBoxes = config.gpu_hosts.map((host) => {
    const box = h('input', { type: 'checkbox', checked: gpuDefaults.gpu_hosts.includes(host), value: host });
    const free = freeGpus?.[host];
    return h('label', {}, box, ` ${host} `, h('span', { class: 'muted' }, free ? `(${free.length} free)` : '(none free)'));
  });
  const workers = h('input', { type: 'number', min: 1, max: 32, value: gpuDefaults.cpu_workers });
  const launchHost = h('select', {}, config.launch_hosts.map((host) => h('option', { value: host, selected: host === gpuDefaults.launch_host }, host)));
  const cpus = h('input', { type: 'number', min: 1, max: 128, value: slurmDefaults.cpus });
  const partition = h('select', {}, config.partitions.map((p) => h('option', { value: p, selected: p === slurmDefaults.partition }, p)));
  const memory = h('input', { type: 'text', value: slurmDefaults.mem_per_cpu, size: 6 });
  const time = h('input', { type: 'text', value: slurmDefaults.time || '', placeholder: 'partition default', size: 12 });
  const gpuSet = h('fieldset', {}, h('legend', {}, 'GPUs (no Slurm: the job runs detached on the launch host)'),
    h('div', { class: 'row' }, 'GPUs ', nGpus, ' CPU workers ', workers, ' launch host ', launchHost),
    h('div', { class: 'row' }, 'hosts, in order: ', hostBoxes));
  const slurmSet = h('fieldset', {}, h('legend', {}, 'Slurm (CPU only)'),
    h('div', { class: 'row' }, 'cores ', cpus, ' partition ', partition, ' memory per core ', memory, ' time limit ', time));
  const summary = h('span', { class: 'muted' });
  const value = () => (mode.value === 'gpu'
    ? { mode: 'gpu', n_gpus: Number(nGpus.value), gpu_hosts: hostBoxes.map((l) => l.querySelector('input')).filter((b) => b.checked).map((b) => b.value),
        cpu_workers: Number(workers.value), launch_host: launchHost.value }
    : { mode: 'slurm', cpus: Number(cpus.value), partition: partition.value, mem_per_cpu: memory.value, time: time.value || null });
  function update() {
    gpuSet.hidden = mode.value !== 'gpu';
    slurmSet.hidden = mode.value !== 'slurm';
    const v = value();
    summary.textContent = v.mode === 'gpu'
      ? `${v.n_gpus} GPU(s) from ${v.gpu_hosts.join(', ') || 'no host'}, ${v.cpu_workers} CPU workers on ${v.launch_host}`
      : `Slurm job: ${v.cpus} cores on ${v.partition}, ${v.mem_per_cpu} per core`;
  }
  const el = h('details', {}, h('summary', {}, 'Resources: ', summary, h('span', { class: 'small muted' }, '  (advanced settings)')),
    h('div', { class: 'row' }, radio('gpu', 'GPUs'), radio('slurm', 'Slurm CPU job')), gpuSet, slurmSet);
  for (const input of [nGpus, workers, launchHost, cpus, partition, memory, time, ...hostBoxes.map((l) => l.querySelector('input'))]) input.addEventListener('change', update);
  update();
  const setMode = (m) => {
    if (mode.chosen) return;
    mode.value = m;
    for (const input of el.querySelectorAll('input[type=radio]')) input.checked = input.value === m;
    update();
  };
  return { el, value, setMode };
}

export async function launch(request, onDone) {
  // POST the job; on a conflict (another job on the run) ask before launching anyway
  try {
    const job = await post('/api/jobs', request);
    toast(`Started job ${job.id} (${job.mode}): ${job.label}`);
    onDone?.(job);
  } catch (e) {
    if (e.status !== 409) { toast(e.message, true); return; }
    modal(h('div', {},
      h('h3', {}, 'Something already works on this run'),
      h('ul', {}, e.message.split('; ').map((line) => h('li', { class: 'small mono' }, line))),
      h('p', {}, 'Two analyses on one rundir overwrite each other\'s results.'),
      h('div', { class: 'row' }, h('span', { class: 'spacer' }), h('button', { onclick: closeModal }, 'Cancel'),
        h('button', { class: 'primary', onclick: () => { closeModal(); launch({ ...request, force: true }, onDone); } }, 'Launch anyway'))));
  }
}

export async function viewerFitModal(info, ghost, onDone) {
  // The viewer's joint fit of a run the pipeline cannot analyse (jointfit.py): mode, reference, f-number, ghost, resources
  const o = await api(`/api/jointfit/options?${q({ run: info.key })}`);
  const refLabel = (r) => `${r.key.split('/').pop()} · f/${r.fnum ?? '?'}${r.hours_apart !== null ? ` · ${r.hours_apart} h apart` : ''}${r.viewer_made ? ' · viewer fit' : ''}`;
  const reference = h('select', {}, o.references.map((r) => h('option', { value: r.key }, refLabel(r))));
  const mode = h('select', {},
    h('option', { value: 'seeded', selected: o.default_mode === 'seeded', disabled: !o.references.length }, 'seeded: theta from the reference, wavelength per fiber refitted'),
    h('option', { value: 'reuse', disabled: !o.references.length }, 'reuse as is: copy theta and wavelengths, fit spots when opened (instant)'),
    h('option', { value: 'scratch', selected: o.default_mode === 'scratch' }, 'from scratch: search theta and wavelengths'));
  const fnum = h('input', { type: 'number', min: 0.5, max: 32, step: 0.1, value: o.fnum ?? '' });
  const ghostSelect = h('select', {}, ['on', 'off'].map((g) => h('option', { value: g, selected: g === ghost }, `ghost mask ${g}`)));
  const fitTheta = h('input', { type: 'checkbox' });
  const pupil = h('select', {}, ['hexagon', 'circle'].map((s) => h('option', { value: s, selected: s === o.default_pupil_shape }, s)));
  const editor = resourcesEditor(o.default_resources, o.free_gpus, o.cpu_resources);
  const matched = Object.entries(o.matched || {}).filter(([f, r]) => f !== r).map(([f, r]) => `${f} → ${r ?? 'none (median wavelength)'}`);
  const warnings = [
    !o.has_x_position && o.n_images <= 3 * o.fibers.length ? 'few stacks per fiber at one stage position: the from-scratch mode is weakly constrained, prefer seeded' : null,
    /circular/.test(info.name) && o.default_pupil_shape !== 'circle' ? 'the run name says circular: choose the circle aperture' : null,
    o.pipeline_fit ? 'this run already has a pipeline joint fit: it would be overwritten' : null,
    o.existing ? `replaces the viewer fit of ${o.existing.made} (${o.existing.mode})` : null,
  ].filter(Boolean);
  modal(h('div', {},
    h('h3', {}, 'Fit this run'),
    h('p', { class: 'small' }, `${o.n_images} spots of ${o.fibers.length} fibers, ${o.image_side} px. The result goes into the run's results/ under the pipeline's names (+ viewer_joint_fit.json).`),
    h('dl', { class: 'kv' },
      h('dt', {}, 'mode'), h('dd', {}, mode),
      h('dt', {}, 'reference'), h('dd', {}, o.references.length ? reference : 'none with a joint fit on this device'),
      h('dt', {}, 'f-number'), h('dd', {}, fnum, h('span', { class: 'muted small' }, ` from ${o.fnum_source}`)),
      h('dt', {}, 'ghost'), h('dd', {}, ghostSelect, h('span', { class: 'muted small' }, ' on: unmasked fit, boxes from its residuals, masked fit')),
      h('dt', {}, 'aperture'), h('dd', {}, pupil, h('span', { class: 'muted small' }, ' circle: theta held at 0 (a disc has no orientation)')),
      h('dt', {}, 'theta'), h('dd', {}, h('label', {}, fitTheta, ' search theta around the reference\'s (seeded, hexagon)')),
      matched.length ? [h('dt', {}, 'fibers matched'), h('dd', {}, matched.join(', '))] : null),
    warnings.map((w) => h('p', { class: 'banner warn small' }, w)),
    h('div', { style: 'margin:8px 0' }, editor.el),
    h('p', { class: 'muted small' }, `About ${o.n_images * 19 * 2 * 2} least-squares fits (seeded, ghost on): seconds on GPUs, minutes on 10 cores.`),
    h('div', { class: 'row' }, h('span', { class: 'spacer' }), h('button', { onclick: closeModal }, 'Cancel'),
      h('button', { class: 'primary', onclick: async () => {
        if (!Number(fnum.value)) { toast('Give the f-number', true); return; }
        const request = { run: info.key, mode: mode.value, reference: mode.value === 'scratch' ? null : reference.value || null,
          fnum: Number(fnum.value), ghost: ghostSelect.value, fit_theta: fitTheta.checked, pupil_shape: pupil.value,
          resources: editor.value() };
        closeModal();
        try {
          const out = await post('/api/jointfit', request);
          toast(out.done ? 'Theta and wavelengths copied: open a spot to fit it' : `Started job ${out.id}: ${out.label}`);
          onDone?.(out);
        } catch (e) {
          if (e.status === 409 && confirm(`Something already works on this run:\n${e.message}\n\nLaunch anyway?`)) {
            post('/api/jointfit', { ...request, force: true }).then((out) => toast(`Started job ${out.id}`)).catch((err) => toast(err.message, true));
          } else toast(e.message, true);
        }
      } }, 'Run'))));
}

export async function jointFitModal(info, ghost) {
  const data = await api(`/api/analyses?${q({ run: info.key, ghost })}`);
  if (data.viewer_joint_fit) { viewerFitModal(info, ghost); return; }
  if (data.reason || !data.device) { toast(data.reason || 'this run is not under the data root: it cannot be analysed here', true); return; }
  const steps = ['fit', ...(ghost === 'on' ? ['ghost_mask', 'refit'] : [])].filter((s) => data.steps.includes(s));
  const withSeries = h('input', { type: 'checkbox' });
  const editor = resourcesEditor(data.default_resources, data.free_gpus, data.cpu_resources);
  modal(h('div', {},
    h('h3', {}, 'Run the joint fit?'),
    h('p', {}, `${info.name} has no joint fit. Every spot model needs its theta and pupil scales. This runs the steps `,
      h('b', {}, steps.join(' → ')), ghost === 'on' ? ' (unmasked fit, ghost boxes from its residuals, masked refit).' : '.'),
    h('label', {}, withSeries, ' also fit every hold frame (series step); otherwise the viewer fits the spots you open, one by one'),
    h('div', { style: 'margin:8px 0' }, editor.el),
    h('div', { class: 'row' }, h('span', { class: 'spacer' }), h('button', { onclick: closeModal }, 'Cancel'),
      h('button', { class: 'primary', onclick: () => {
        closeModal();
        launch({ run: info.key, analysis: 'flow', steps: [...steps, ...(withSeries.checked ? ['series'] : [])], ghost, resources: editor.value() });
      } }, 'Run'))));
}

export async function renderAnalyses(body, info) {
  const ghost = storedGhost();
  body.replaceChildren(h('p', { class: 'pad muted' }, 'loading…'));
  const data = await api(`/api/analyses?${q({ run: info.key, ghost })}`);
  const ghostSelect = h('select', {}, ['on', 'off'].map((g) => h('option', { value: g, selected: g === ghost }, `ghost mask ${g}`)));
  ghostSelect.onchange = () => { try { localStorage.setItem('ghost', ghostSelect.value); } catch { /* blocked */ } renderAnalyses(body, info); };
  const root = h('div', { class: 'pad', style: 'display:flex;flex-direction:column;gap:12px;max-width:1100px' });
  body.replaceChildren(root);
  if (data.reason) {
    root.append(h('p', { class: 'banner warn' }, data.reason));
    if (data.viewer_joint_fit) {
      root.append(h('div', { class: 'card' }, h('h3', {}, 'Viewer joint fit'),
        h('p', { class: 'small' }, 'Fits every spot of this run jointly (theta, wavelength per fiber, per-spot r, centre, Zernikes), seeded by an analysed run of the same device, with the pipeline\'s ghost hunt. The spots tab then shows models and residuals.'),
        info.viewer_fit ? h('p', { class: 'small muted' }, 'This run already has a viewer joint fit: running again replaces it.') : null,
        h('button', { class: 'primary', onclick: () => viewerFitModal(info, ghost) }, 'Fit this run…')));
    }
    return;
  }
  if (!data.device) { root.append(h('p', { class: 'banner warn' }, 'This run is not under the data root: the analysis entry points cannot find it.')); return; }

  const boxes = data.steps.map((step) => {
    const box = h('input', { type: 'checkbox', checked: data.default_steps.includes(step) && !info.steps_done.includes(step), value: step });
    return h('label', { title: STEP_HELP[step] || '' }, box, ' ', step, info.steps_done.includes(step) ? h('span', { class: 'badge ok', style: 'margin-left:4px' }, 'done') : null);
  });
  const selected = () => boxes.map((l) => l.querySelector('input')).filter((b) => b.checked).map((b) => b.value);
  const editor = resourcesEditor(data.default_resources, data.free_gpus, data.cpu_resources);
  const syncMode = () => editor.setMode(selected().some((s) => GPU_STEPS.includes(s)) && data.default_resources.mode === 'gpu' ? 'gpu' : 'slurm');
  for (const l of boxes) l.querySelector('input').addEventListener('change', syncMode);
  const setAll = (pick) => () => { for (const l of boxes) { const b = l.querySelector('input'); b.checked = pick(b.value); } syncMode(); };
  syncMode();
  const jobsList = h('div');
  const refreshJobs = async () => {
    const jobs = (await api('/api/jobs')).filter((j) => j.run === info.key).slice(0, 10);
    jobsList.replaceChildren(jobs.length ? h('table', { class: 'grid' }, h('tbody', {}, jobs.map((j) => h('tr', {},
      h('td', {}, j.started.replace('T', ' ')), h('td', { class: `state-${j.state}` }, j.state), h('td', {}, j.label), h('td', { class: 'muted' }, j.mode),
      h('td', {}, h('a', { href: `#tab=jobs&job=${j.id}` }, 'log')))))) : h('span', { class: 'muted small' }, 'none yet'));
  };
  root.append(
    h('div', { class: 'row' }, h('b', {}, `${data.flow} run`), ghostSelect, h('span', { class: 'muted small' },
      'ghost mask on adds ghost_mask and refit after fit, and the later fits use the masks')),
    h('div', { class: 'card' }, h('h3', {}, `Steps of run_${data.flow}`),
      h('div', { class: 'steps' }, boxes),
      h('div', { class: 'row small', style: 'margin:6px 0' }, 'select: ',
        h('button', { onclick: setAll((s) => data.default_steps.includes(s)) }, 'full run'),
        h('button', { onclick: setAll((s) => ['fit', 'ghost_mask', 'refit'].includes(s)) }, 'joint fit'),
        h('button', { onclick: setAll((s) => !info.steps_done.includes(s) && data.default_steps.includes(s)) }, 'missing'),
        h('button', { onclick: setAll(() => false) }, 'none')),
      editor.el,
      h('div', { class: 'row', style: 'margin-top:8px' }, h('span', { class: 'spacer' }), h('button', { class: 'primary', onclick: () => {
        const steps = selected();
        if (!steps.length) { toast('Pick at least one step', true); return; }
        launch({ run: info.key, analysis: 'flow', steps, ghost, resources: editor.value() }, refreshJobs);
      } }, 'Launch'))),
    h('div', { class: 'card' }, h('h3', {}, 'Other analyses'),
      h('div', { class: 'row' }, data.extras.map((extra) => h('button', {
        onclick: () => launch({ run: info.key, analysis: extra, ghost, resources: data.cpu_resources }, refreshJobs),
        title: `Slurm job of ${data.cpu_resources.cpus} cores`,
      }, `run_${extra}`))),
      h('div', { class: 'small muted', style: 'margin-top:4px' }, `Each runs as a Slurm job of ${data.cpu_resources.cpus} cores.`)),
    h('div', { class: 'card' }, h('h3', {}, 'Jobs of this run'), jobsList));
  refreshJobs();
}
