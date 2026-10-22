// Right-click menu of an image panel: copy or save it as PNG, save it as a PDF figure, save its values as .npy.
import { h, toast, EXPORT_DPI } from './ui.js';
import { AXES, domainOf, panelLabel } from './views.js';

let open = null;

function close() {
  open?.remove();
  open = null;
}

document.addEventListener('click', close);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
window.addEventListener('blur', close);

function download(blob, filename) {
  const a = h('a', { href: URL.createObjectURL(blob), download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

export function npyBlob(values, rows, cols) {
  // .npy (format 1.0) of a float32 image, readable by numpy.load
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${rows}, ${cols}), }`;
  header += ' '.repeat((64 - ((10 + header.length + 1) % 64)) % 64) + '\n';
  const prefix = new Uint8Array(10 + header.length);
  prefix.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(prefix.buffer).setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) prefix[10 + i] = header.charCodeAt(i);
  return new Blob([prefix, Float32Array.from(values)], { type: 'application/octet-stream' });
}

async function png(div) {
  const url = await Plotly.toImage(div, { format: 'png', scale: EXPORT_DPI / 96, width: div.clientWidth, height: div.clientHeight });
  return (await fetch(url)).blob();
}

async function copyPng(div, name) {
  const blob = await png(div);
  if (window.isSecureContext && navigator.clipboard?.write && window.ClipboardItem) {
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
    toast('Image copied to the clipboard');
  } else {
    download(blob, `${name}.png`);
    toast('Browsers copy images only on HTTPS or localhost pages (this one is plain HTTP): saved as a PNG file instead.', true);
  }
}

async function pdf(info) {
  const { panel, values, zrange, ghostBox, options, name, title } = info;
  const [xlabel, ylabel] = AXES[domainOf(panel)];
  const bytes = new Uint8Array(Float32Array.from(values.values).buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  const response = await fetch('/api/figure.pdf', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      shape: [values.rows, values.cols], data: btoa(binary), coords: values.coords, zrange, title, xlabel, ylabel,
      colour_map: values.signed ? options.divergingMap : options.imageMap, reverse: options.reverseMaps, filename: name,
      ghost_box: options.showGhost && panel.space === 'image' && ['data', 'model', 'residual'].includes(panel.q) ? ghostBox : null,
    }),
  });
  if (!response.ok) throw new Error(`PDF: ${response.status}`);
  download(await response.blob(), `${name}.pdf`);
}

export function attachMenu(div, info) {
  // info: {panel, values (panelValues), zrange, ghostBox, options, name (file stem), title}
  div.addEventListener('contextmenu', (event) => {
    event.preventDefault();
    close();
    const run = (action) => async () => {
      close();
      try { await action(); } catch (e) { toast(e.message, true); }
    };
    const { values, name } = info;
    open = h('div', { class: 'context-menu', style: `left:${event.clientX}px;top:${event.clientY}px` },
      h('div', { class: 'small muted', style: 'padding:2px 10px' }, panelLabel(info.panel)),
      h('button', { onclick: run(() => copyPng(div, name)) }, 'Copy image (PNG)'),
      h('button', { onclick: run(async () => download(await png(div), `${name}.png`)) }, 'Save image (PNG)'),
      h('button', { onclick: run(() => pdf(info)) }, 'Save figure (PDF)'),
      h('button', { onclick: run(() => download(npyBlob(values.values, values.rows, values.cols), `${name}.npy`)) }, 'Save shown values (.npy)'),
      h('button', { onclick: run(() => download(npyBlob(values.raw, values.rows, values.cols), `${name}_raw.npy`)) },
        info.panel.space === 'fft' ? 'Save |FFT| before scale (.npy)' : 'Save values before scale (.npy)'));
    document.body.append(open);
    const box = open.getBoundingClientRect();
    if (box.right > innerWidth) open.style.left = `${innerWidth - box.width - 4}px`;
    if (box.bottom > innerHeight) open.style.top = `${innerHeight - box.height - 4}px`;
  });
}
