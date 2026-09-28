const $ = id => document.getElementById(id);
const endpoint = '/api/v1/annotation/sequence';
let current = null, queue = [], pending = {}, saving = null, timer = null;
let navigation = Promise.resolve();
let anchor = null, selected = null, editingEntity = null, dragging = false, moved = false, dragOrigin = null;

function error(message) { $('annotation-error').textContent = message; $('annotation-error').hidden = !message; }
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error?.message || (typeof value.detail === 'string' ? value.detail : JSON.stringify(value.detail)) || '请求失败');
  return value;
}
function run(action) { Promise.resolve().then(action).catch(e => error(e.message)); }
function navigate(action) {
  navigation = navigation.then(async () => {
    $('annotation-editor').inert = true;
    try { await action(); } finally { $('annotation-editor').inert = false; }
  }).catch(e => error(e.message));
  return navigation;
}
function button(text, action, className = 'outline small') {
  const node = document.createElement('button'); node.textContent = text; node.className = className;
  node.addEventListener('click', () => run(action)); return node;
}
function characters() { return Array.from(current.text); }
function tokens(boundaries) {
  const chars = characters(), cuts = [0, ...boundaries, chars.length];
  return cuts.slice(0, -1).map((start, i) => chars.slice(start, cuts[i + 1]).join(''));
}
function tags(words) {
  return words.flatMap(word => { const n = Array.from(word).length; return n === 1 ? ['S'] : ['B', ...Array(n - 2).fill('M'), 'E']; });
}
function previewCws() {
  const words = current.boundaries === null ? null : tokens(current.boundaries);
  current.gold_tokens = words;
  $('gold-tokens').textContent = words === null ? '未标注；需人工确认' : JSON.stringify(words);
  $('gold-bmes').textContent = words === null ? '未标注' : tags(words).join(' ');
  $('cws-characters').querySelectorAll('[data-boundary]').forEach(node => node.setAttribute('aria-pressed', String((current.boundaries || []).includes(Number(node.dataset.boundary)))));
}
function change(value) {
  Object.assign(pending, value); $('save-state').textContent = '有编辑，等待自动保存…'; error('');
  $('review-confirm').checked = false;
  clearTimeout(timer); timer = setTimeout(() => run(flush), 450);
}
function query() {
  const params = new URLSearchParams();
  for (const [id, key] of [['filter-label','label'], ['filter-source','source_type'], ['filter-status','status']]) if ($(id).value) params.set(key, $(id).value);
  return params.toString();
}
async function refreshQueue() {
  const result = await api(`${endpoint}?${query()}`); queue = result.items;
  $('annotation-progress').textContent = `${result.is_demo ? 'TOY 测试队列 · 不可正式导出 · ' : ''}总计 ${result.total} · pending ${result.progress.pending} · in_progress ${result.progress.in_progress} · done ${result.progress.done} · reviewed ${result.progress.reviewed}`;
  $('queue-count').textContent = `当前筛选 ${result.filtered_count} 条`;
  for (const [id, values, label] of [['filter-label',result.labels,'全部主题'],['filter-source',result.source_types,'全部体裁']]) {
    const value = $(id).value; $(id).replaceChildren(new Option(label, ''), ...values.map(x => new Option(x, x))); $(id).value = value;
  }
  $('annotation-queue').replaceChildren(...queue.map(item => {
    const node = button(`${item.sample_id} · ${item.status}`, () => navigate(() => loadSample(item.sample_id)), 'queue-item');
    node.setAttribute('aria-current', String(current?.sample_id === item.sample_id)); return node;
  }));
  const index = queue.findIndex(x => x.sample_id === current?.sample_id);
  $('previous-sample').disabled = index <= 0; $('next-sample').disabled = index < 0 || index >= queue.length - 1;
}
function applySaved(saved) {
  const preserve = {boundaries: current.boundaries, entities: current.entities, notes: current.notes};
  current = {...saved};
  for (const key of Object.keys(preserve)) if (Object.hasOwn(pending, key)) current[key] = preserve[key];
  previewCws(); renderNer(); renderStatus();
}
async function flush() {
  clearTimeout(timer);
  if (saving) { await saving; return flush(); }
  if (!current || !Object.keys(pending).length) return;
  const changes = structuredClone(pending); pending = {};
  const id = current.sample_id, version = current.draft_version;
  $('save-state').textContent = '保存中…';
  saving = (async () => {
    try {
      const saved = await api(`${endpoint}/${id}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({...changes, expected_version:version})});
      applySaved(saved); $('save-state').textContent = '已原子保存'; await refreshQueue();
    } catch (e) {
      pending = {...changes, ...pending}; $('save-state').textContent = '保存失败，编辑仍保留；请重试'; throw e;
    }
  })();
  try { await saving; } finally { saving = null; }
  if (Object.keys(pending).length) return flush();
}
function glyph(char) { return char === ' ' ? '␠' : char === '\n' ? '↵' : char === '\r' ? '␍' : char === '\t' ? '⇥' : char; }
function characterNode(char, index, clickable) {
  const node = document.createElement(clickable ? 'button' : 'span'); node.className = clickable ? 'cp-character ner-character' : 'cp-character';
  const content = document.createElement('span'); content.textContent = glyph(char);
  const sub = document.createElement('span'); sub.textContent = index; sub.className = 'cp-index';
  node.append(content, sub); return node;
}
function renderCws() {
  const nodes = [], chars = characters();
  chars.forEach((char, index) => {
    nodes.push(characterNode(char, index, false));
    if (index < chars.length - 1) {
      const position = index + 1;
      const cut = button('|', () => {
        const boundaries = new Set(current.boundaries || []);
        if (boundaries.has(position)) boundaries.delete(position); else boundaries.add(position);
        current.boundaries = [...boundaries].sort((a,b) => a-b); previewCws(); change({boundaries:current.boundaries});
      }, 'boundary');
      cut.dataset.boundary = position; cut.setAttribute('aria-label', `切词边界 ${position}`); nodes.push(cut);
    }
  });
  $('cws-characters').replaceChildren(...nodes); previewCws();
}
function selectRange(start, end) {
  selected = [start, end]; $('entity-start').value = start; $('entity-end').value = end;
  $('entity-selection').textContent = `选择 [${start},${end})：${characters().slice(start,end).join('')}`;
  $('ner-characters').querySelectorAll('[data-index]').forEach(node => node.classList.toggle('selected', Number(node.dataset.index) >= start && Number(node.dataset.index) < end));
}
function clearSelection() {
  anchor = selected = editingEntity = null; $('entity-start').value = ''; $('entity-end').value = '';
  $('entity-selection').textContent = ''; $('save-entity').textContent = '添加实体';
  $('ner-characters').querySelectorAll('.selected').forEach(node => node.classList.remove('selected'));
}
function renderNer() {
  const nodes = characters().map((char,index) => {
    const node = characterNode(char,index,true); node.dataset.index = index; node.setAttribute('aria-label', `实体字符 ${index} ${glyph(char)}`);
    const entity = (current.entities || []).find(e => e.start <= index && index < e.end);
    if (entity) node.classList.add(`entity-${entity.type}`);
    node.addEventListener('pointerdown', () => { dragging = true; moved = false; dragOrigin = index; });
    node.addEventListener('pointerenter', () => { if (dragging && dragOrigin !== null) { moved = true; selectRange(Math.min(dragOrigin,index),Math.max(dragOrigin,index)+1); } });
    node.addEventListener('click', () => {
      if (moved) { moved = false; anchor = null; return; }
      if (anchor === null) { anchor = index; selectRange(index,index+1); }
      else { selectRange(Math.min(anchor,index),Math.max(anchor,index)+1); anchor = null; }
    });
    return node;
  });
  $('ner-characters').replaceChildren(...nodes);
  if (selected) selectRange(...selected);
  $('ner-confirmation').textContent = current.entities === null ? 'NER 尚未人工确认' : `已标注 ${current.entities.length} 个实体${!current.entities.length ? '（人工确认无实体）' : ''}`;
  $('entity-list').replaceChildren(...(current.entities || []).map((entity,index) => {
    const row = document.createElement('div'); row.className = 'entity-row';
    const text = document.createElement('span'); text.textContent = `${entity.type} [${entity.start},${entity.end}) ${entity.text}`;
    row.append(text, button('修改', () => { editingEntity = index; selectRange(entity.start,entity.end); $('entity-type').value = entity.type; $('save-entity').textContent = '保存实体修改'; }),
      button('删除', () => { current.entities = current.entities.filter((_,i) => i !== index); clearSelection(); renderNer(); change({entities:current.entities}); })); return row;
  }));
}
function renderStatus() {
  $('sample-heading').textContent = `${current.sample_id} · ${current.status}`;
  $('review-count').textContent = `明确第二遍复核次数 ${current.revision_count} · 草稿版本 ${current.draft_version}`;
  $('mark-reviewed').disabled = current.status !== 'done';
}
async function loadSample(id) {
  await flush(); error(''); current = await api(`${endpoint}/${id}`); pending = {}; clearSelection();
  $('annotation-editor').hidden = false;
  $('original-text').textContent = current.text;
  $('sample-metadata').textContent = `${current.doc_id} · ${current.primary_label} / ${current.source_type} · source [${current.source_start},${current.source_end}) · ${characters().length} code points`;
  $('source-link').textContent = current.source_title;
  const url = new URL(current.source_url, location.href);
  $('source-link').href = ['http:','https:'].includes(url.protocol) ? url.href : '#';
  $('annotation-notes').value = current.notes; $('review-confirm').checked = false;
  $('reference-suggestions').open = false; $('suggestion-content').textContent = '展开后加载参考建议。';
  $('save-state').textContent = current.draft_version ? '已保存草稿' : '尚无草稿；Gold 未标注';
  renderCws(); renderNer(); renderStatus(); await refreshQueue();
}
async function markStatus(status) {
  await flush();
  if (status === 'reviewed' && !$('review-confirm').checked) throw new Error('请先完成第二遍复核并勾选确认。');
  const saved = await api(`${endpoint}/${current.sample_id}`, {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_version:current.draft_version,status,review_confirmation:status === 'reviewed'})});
  applySaved(saved); $('save-state').textContent = '已保存状态'; $('review-confirm').checked = false; await refreshQueue();
}
for (const id of ['filter-label','filter-source','filter-status']) $(id).addEventListener('change', () => navigate(async () => { await flush(); await refreshQueue(); if (queue.length) await loadSample(queue[0].sample_id); else $('annotation-editor').hidden = true; }));
for (const [id,offset] of [['previous-sample',-1],['next-sample',1]]) $(id).addEventListener('click', () => navigate(async () => { await flush(); const index = queue.findIndex(x => x.sample_id === current.sample_id); if (queue[index+offset]) await loadSample(queue[index+offset].sample_id); }));
$('confirm-cws').addEventListener('click', () => { if (!current) return; current.boundaries ||= []; previewCws(); change({boundaries:current.boundaries}); });
$('annotation-notes').addEventListener('input', () => { current.notes = $('annotation-notes').value; change({notes:current.notes}); });
$('no-entities').addEventListener('click', () => { current.entities = []; clearSelection(); renderNer(); change({entities:[]}); });
$('cancel-entity').addEventListener('click', clearSelection);
for (const id of ['entity-start','entity-end']) $(id).addEventListener('input', () => { const start = Number($('entity-start').value), end = Number($('entity-end').value); if (Number.isInteger(start) && Number.isInteger(end) && start >= 0 && start < end && end <= characters().length) selectRange(start,end); });
$('save-entity').addEventListener('click', () => run(async () => {
  const start = Number($('entity-start').value), end = Number($('entity-end').value);
  if (!selected || !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || start >= end || end > characters().length) throw new Error('请选择有效的连续 code-point span。');
  const entities = (current.entities || []).filter((_,i) => i !== editingEntity);
  if (entities.some(e => start < e.end && e.start < end)) throw new Error('实体跨度重叠；请先修改或删除已有实体。');
  entities.push({text:characters().slice(start,end).join(''),type:$('entity-type').value,start,end});
  current.entities = entities.sort((a,b) => a.start-b.start); clearSelection(); renderNer(); change({entities:current.entities});
}));
$('retry-save').addEventListener('click', () => navigate(flush));
for (const [id,status] of [['resume-annotation','in_progress'],['mark-done','done'],['mark-reviewed','reviewed']]) $(id).addEventListener('click', () => navigate(() => markStatus(status)));
$('reference-suggestions').addEventListener('toggle', () => { if (!$('reference-suggestions').open || !current) return; const id=current.sample_id; run(async () => { const result=await api(`${endpoint}/${id}/suggestions`); if(current.sample_id===id) $('suggestion-content').textContent=JSON.stringify(result,null,2); }); });
document.addEventListener('pointerup', () => { dragging = false; });
window.addEventListener('beforeunload', event => { if (Object.keys(pending).length || saving) { event.preventDefault(); event.returnValue=''; } });
navigate(async () => { await refreshQueue(); if (queue.length) await loadSample(queue[0].sample_id); else $('annotation-progress').textContent += ' · 当前筛选无候选'; });
