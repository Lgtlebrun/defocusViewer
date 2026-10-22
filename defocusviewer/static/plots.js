// The Plots subtab: the plot/ tree of a run on the left, the chosen figure (or a folder's gallery) on the right.
import { h, api, q, getHash, setHash } from './ui.js';

function buildTree(files) {
  const root = { name: '', path: '', folders: new Map(), files: [] };
  for (const file of files) {
    const parts = file.path.split('/');
    let node = root;
    for (const part of parts.slice(0, -1)) {
      if (!node.folders.has(part)) node.folders.set(part, { name: part, path: node.path ? `${node.path}/${part}` : part, folders: new Map(), files: [] });
      node = node.folders.get(part);
    }
    node.files.push(file);
  }
  return root;
}

function count(node) {
  let n = node.files.length;
  for (const child of node.folders.values()) n += count(child);
  return n;
}

export async function renderPlots(body, info) {
  body.replaceChildren(h('p', { class: 'pad muted' }, 'loading…'));
  const files = await api(`/api/run/plots?${q({ run: info.key })}`);
  if (!files.length) {
    body.replaceChildren(h('p', { class: 'pad muted' }, 'No figures in plot/ yet.'));
    return;
  }
  const tree = buildTree(files);
  const main = h('div', { class: 'plots-main' });
  const url = (kind, path) => `/api/run/${kind}?${q({ run: info.key, path })}`;
  const selectedPath = () => getHash().plot;

  function mark() {
    for (const el of side.querySelectorAll('.selected')) el.classList.remove('selected');
    const sel = side.querySelector(`[data-path="${CSS.escape(selectedPath() || '')}"]`);
    if (sel) sel.classList.add('selected');
  }

  function showFile(path) {
    setHash({ plot: path }, true);
    mark();
    const head = h('div', { class: 'row pad', style: 'padding:6px 10px' }, h('span', { class: 'mono' }, path), h('span', { class: 'spacer' }),
      h('a', { href: url('file', path), target: '_blank' }, 'open in new tab'));
    let content;
    if (path.endsWith('.pdf')) content = h('iframe', { src: `${url('file', path)}#view=FitH` });
    else if (/\.(png|jpe?g|svg)$/i.test(path)) content = h('div', { style: 'overflow:auto;flex:1' }, h('img', { src: url('file', path), style: 'max-width:100%' }));
    else {
      content = h('pre', { style: 'flex:1' }, 'loading…');
      fetch(url('file', path)).then((r) => r.text()).then((t) => { content.textContent = t; });
    }
    main.replaceChildren(head, content);
  }

  function showGallery(node) {
    setHash({ plot: `${node.path}/` }, true);
    mark();
    const below = (n) => [...n.folders.values()].flatMap((f) => [...f.files, ...below(f)]);
    const items = [...node.files, ...below(node)].slice(0, 120);
    main.replaceChildren(
      h('div', { class: 'row pad', style: 'padding:6px 10px' }, h('b', {}, node.path || 'plot/'), h('span', { class: 'muted' }, `${items.length} figures`)),
      h('div', { class: 'gallery' }, items.map((file) => h('figure', { onclick: () => showFile(file.path) },
        h('img', { loading: 'lazy', src: url('thumb', file.path), alt: file.path }),
        h('figcaption', {}, file.path.split('/').pop())))));
  }

  function folderView(node, depth) {
    const opened = depth === 0 || (selectedPath() || '').startsWith(`${node.path}/`);
    const details = h('details', { open: opened, style: `margin-left:${depth ? 10 : 0}px` },
      h('summary', { class: 'folder', 'data-path': `${node.path}/`, onclick: () => showGallery(node) },
        node.name || 'plot', ' ', h('span', { class: 'muted' }, `(${count(node)})`)),
      [...node.folders.values()].sort((a, b) => a.name.localeCompare(b.name)).map((child) => folderView(child, depth + 1)),
      node.files.map((file) => h('div', { class: 'file', 'data-path': file.path, onclick: () => showFile(file.path) }, file.path.split('/').pop())));
    return details;
  }

  const side = h('div', { class: 'plots-tree' }, folderView(tree, 0));
  body.replaceChildren(h('div', { class: 'plots' }, side, main));
  const start = selectedPath();
  if (start && !start.endsWith('/') && files.some((f) => f.path === start)) showFile(start);
  else showGallery(tree);
}
