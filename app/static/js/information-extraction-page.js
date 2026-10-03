import { api, node } from './api.js';

const $ = (id) => document.getElementById(id);
const input = $('ie-input');
let requestNumber = 0;
let timer;
let graph;
let currentValue;

function entityLabel(entity) {
  return `${entity.text} · ${entity.type} [${entity.start},${entity.end})`;
}

function renderHighlight(text, entities, relation = null) {
  const box = $('ie-highlight');
  const chars = Array.from(text);
  box.replaceChildren();
  const points = new Set([0, chars.length]);
  for (const item of [...entities, ...(relation ? [relation.evidence, relation.trigger] : [])]) {
    points.add(item.start); points.add(item.end);
  }
  const offsets = [...points].sort((a, b) => a - b);
  for (let i = 0; i < offsets.length - 1; i++) {
    const start = offsets[i], end = offsets[i + 1];
    const entity = entities.find((item) => item.start <= start && item.end >= end);
    const inEvidence = relation && start >= relation.evidence.start && end <= relation.evidence.end;
    const inTrigger = relation && start >= relation.trigger.start && end <= relation.trigger.end;
    const part = node(entity ? 'mark' : 'span', chars.slice(start, end).join(''),
      `${entity ? `entity ${entity.type.toLowerCase()}` : ''}${inEvidence ? ' ie-evidence-focus' : ''}${inTrigger ? ' ie-trigger-focus' : ''}`);
    if (entity) part.title = entityLabel(entity);
    box.append(part);
  }
}

function selectRelation(index) {
  const relation = currentValue?.relations[index];
  if (!relation) return;
  renderHighlight(currentValue.text, currentValue.entities, relation);
  $('ie-selection-hint').textContent = `当前证据 [${relation.evidence.start},${relation.evidence.end}) · 下划线标出触发词“${relation.trigger.text}”`;
  document.querySelectorAll('.ie-relation-card').forEach((card) => {
    const selected = Number(card.dataset.relationIndex) === index;
    card.classList.toggle('is-selected', selected);
    card.setAttribute('aria-pressed', String(selected));
  });
  graph?.dispatchAction({ type: 'downplay', seriesIndex: 0 });
  graph?.dispatchAction({ type: 'highlight', seriesIndex: 0, dataType: 'edge', dataIndex: index });
}

function relationCard(relation, index) {
  const card = node('article', '', 'ie-relation-card');
  card.dataset.relationIndex = index;
  card.tabIndex = 0; card.setAttribute('role', 'button'); card.setAttribute('aria-pressed', 'false');
  card.setAttribute('aria-label', `${relation.subject.text} ${relation.predicate} ${relation.object.text}，查看原文证据`);
  card.addEventListener('click', () => selectRelation(index));
  card.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectRelation(index); }
  });
  const triple = node('div', '', 'ie-triple');
  triple.append(
    node('span', relation.subject.text, `ie-node ${relation.subject.type.toLowerCase()}`),
    node('span', `${relation.predicate} →`, 'ie-predicate'),
    node('span', relation.object.text, `ie-node ${relation.object.type.toLowerCase()}`),
  );
  const evidence = node('p', '', 'ie-card-evidence');
  evidence.append(node('span', '原文证据'), node('q', relation.evidence.text));
  card.append(triple, evidence,
    node('small', `触发：${relation.trigger.text} · [${relation.evidence.start},${relation.evidence.end}) · 点击定位`));
  return card;
}

function renderGraph(entities, relations) {
  const target = $('ie-graph');
  if (graph) { graph.dispose(); graph = null; }
  target.replaceChildren();
  if (!relations.length) {
    target.append(node('p', '当前文本没有满足规则的显式关系；不根据实体共现补画关系边。', 'ie-empty'));
    return;
  }
  if (!window.echarts) {
    target.append(node('p', '图表资源不可用；上方关系卡仍显示完整结果。', 'ie-empty'));
    return;
  }
  const involved = new Map();
  const key = (item) => `${item.type}:${item.start}:${item.end}`;
  for (const relation of relations) {
    involved.set(key(relation.subject), relation.subject);
    involved.set(key(relation.object), relation.object);
  }
  const colors = { PER: ['#e0f5ed', '#096a58'], LOC: ['#e2efff', '#0b67a0'], ORG: ['#fff0cd', '#8d5b05'] };
  const subjects = new Set(relations.map((relation) => key(relation.subject)));
  const objects = new Set(relations.map((relation) => key(relation.object)));
  const columns = [[], [], []];
  for (const [id, item] of involved) columns[subjects.has(id) ? (objects.has(id) ? 1 : 0) : 2].push([id, item]);
  const nodes = columns.flatMap((column, col) => column.map(([id, item], row) => {
    const chars = Array.from(item.text), lines = [];
    for (let i = 0; i < chars.length; i += 8) lines.push(chars.slice(i, i + 8).join(''));
    return {
      id, name: item.text, value: item.type, x: col * 260, y: (row + .5) * 260 / column.length,
      symbol: 'roundRect', symbolSize: [Math.max(96, Math.min(chars.length, 8) * 15 + 28), lines.length * 20 + 36],
      itemStyle: { color: colors[item.type][0], borderColor: colors[item.type][1], borderWidth: 1.2 },
      label: { show: true, formatter: `${lines.join('\n')}\n${item.type}`, color: colors[item.type][1], fontWeight: 'bold', lineHeight: 20 },
    };
  }));
  // ECharts view 坐标不能使用高度为零的边界，否则 symbol 会被纵向拉伸。
  if (nodes.length > 1 && nodes.every((item) => item.y === nodes[0].y)) {
    nodes.forEach((item, index) => { item.y += index * 160; });
  }
  const links = relations.map((relation) => ({
    source: key(relation.subject), target: key(relation.object), value: relation.predicate,
    label: { show: true, formatter: relation.predicate, color: '#075c6d', fontSize: 12, backgroundColor: '#fff', padding: [4, 7], borderRadius: 4 },
    lineStyle: { color: '#46a8b2', width: 2.5, opacity: .85, curveness: .05 },
  }));
  graph = window.echarts.init(target);
  graph.setOption({
    animationDuration: 350, tooltip: { trigger: 'item', renderMode: 'richText', formatter: (params) => params.dataType === 'edge' ? params.data.value : entityLabel(involved.get(params.data.id)) },
    series: [{ type: 'graph', layout: 'none', left: '17%', right: '17%', top: '22%', bottom: '22%', roam: true, draggable: true,
      data: nodes, links, emphasis: { focus: 'adjacency', lineStyle: { width: 4, color: '#087f8c', opacity: 1 } },
      edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 10 }],
  });
  graph.on('click', (params) => { if (params.dataType === 'edge') selectRelation(params.dataIndex); });
}

function render(value) {
  currentValue = value;
  const { text, entities, relations } = value;
  $('ie-char-count').textContent = Array.from(text).length;
  $('ie-entity-count').textContent = value.status === 'ok' ? entities.length : '—';
  $('ie-result-count').textContent = value.status === 'ok' ? relations.length : '—';
  $('ie-selection-hint').textContent = '选择下方三元组，定位原文中的证据与触发词。';
  if (value.status !== 'ok') {
    $('ie-status').textContent = `NER 不可用：${value.error}`;
    $('ie-highlight').textContent = text;
    $('ie-summary').textContent = 'NER 资源不可用；没有执行关系抽取。';
    $('ie-relation-count').textContent = '0 条';
    $('ie-relation-cards').textContent = 'NER 不可用，暂无关系结果。';
    $('ie-entities').textContent = 'NER 不可用。';
    $('ie-relation-details').textContent = 'NER 不可用。';
    $('ie-json').textContent = JSON.stringify(value, null, 2);
    renderGraph([], []);
    return;
  }
  $('ie-status').textContent = `已分析 ${Array.from(text).length} 字符 · NER: ${value.ner_model}`;
  $('ie-summary').textContent = `${entities.length} 个实体 · ${relations.length} 条显式关系 · NER: ${value.ner_model}`;
  renderHighlight(text, entities);
  $('ie-relation-count').textContent = `${relations.length} 条`;
  for (const id of ['ie-relation-cards', 'ie-relation-details']) {
    const box = $(id); box.replaceChildren();
    if (!relations.length) box.append(node('p', '未抽取到符合类型与触发词约束的关系。', 'ie-empty'));
    else relations.forEach((relation, index) => box.append(relationCard(relation, index)));
  }
  const list = $('ie-entities'); list.replaceChildren();
  if (!entities.length) list.append(node('p', '当前文本未识别出 PER / LOC / ORG 实体。', 'ie-empty'));
  else for (const entity of entities) list.append(node('div', entityLabel(entity), `ie-entity-row ${entity.type.toLowerCase()}`));
  $('ie-json').textContent = JSON.stringify({ text, ner_model: value.ner_model, entities, relations }, null, 2);
  renderGraph(entities, relations);
}

async function analyze() {
  const text = input.value.trim();
  const current = ++requestNumber;
  if (!text) {
    $('ie-status').textContent = '请输入文本。';
    return;
  }
  $('ie-status').textContent = '正在识别实体与显式关系…';
  try {
    const value = await api('/api/v1/information-extraction/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
    });
    if (current === requestNumber) render(value);
  } catch (error) {
    if (current === requestNumber) $('ie-status').textContent = `抽取失败：${error.message}`;
  }
}

$('ie-analyze').addEventListener('click', analyze);
$('ie-reset-view').addEventListener('click', () => {
  if (currentValue?.status === 'ok') renderGraph(currentValue.entities, currentValue.relations);
});
input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(analyze, 550); });
document.querySelectorAll('[data-ie-example]').forEach((button) => button.addEventListener('click', () => {
  input.value = button.dataset.ieExample; analyze();
}));
$('ie-copy-json').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('ie-json').textContent);
  $('ie-copy-json').textContent = '已复制';
  setTimeout(() => { $('ie-copy-json').textContent = '复制 JSON'; }, 1400);
});
window.addEventListener('resize', () => graph?.resize());
document.addEventListener('module-tab', () => {
  if (!$('interactive').hidden) requestAnimationFrame(() => graph?.resize());
});
analyze();
