import { api, node } from './api.js';

const $ = (id) => document.getElementById(id);
const input = $('ie-input');
let requestNumber = 0;
let timer;
let graph;

function entityLabel(entity) {
  return `${entity.text} · ${entity.type} [${entity.start},${entity.end})`;
}

function renderHighlight(text, entities) {
  const box = $('ie-highlight');
  const chars = Array.from(text);
  box.replaceChildren();
  let offset = 0;
  for (const entity of [...entities].sort((a, b) => a.start - b.start)) {
    if (entity.start < offset || entity.end > chars.length) continue;
    box.append(node('span', chars.slice(offset, entity.start).join('')));
    const mark = node('mark', chars.slice(entity.start, entity.end).join(''), `entity ${entity.type.toLowerCase()}`);
    mark.title = entityLabel(entity);
    box.append(mark);
    offset = entity.end;
  }
  box.append(node('span', chars.slice(offset).join('')));
}

function relationCard(relation) {
  const card = node('article', '', 'ie-relation-card');
  const triple = node('div', '', 'ie-triple');
  triple.append(
    node('span', relation.subject.text, `ie-node ${relation.subject.type.toLowerCase()}`),
    node('span', `— ${relation.predicate} →`, 'ie-predicate'),
    node('span', relation.object.text, `ie-node ${relation.object.type.toLowerCase()}`),
  );
  card.append(triple, node('p', `证据：“${relation.evidence.text}”`),
    node('small', `触发词：${relation.trigger.text} [${relation.trigger.start},${relation.trigger.end}) · 原文证据 [${relation.evidence.start},${relation.evidence.end})`));
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
  const colors = { PER: '#278b75', LOC: '#3686bf', ORG: '#b78532' };
  const nodes = [...involved].map(([id, item]) => ({
    id, name: item.text, value: item.type, symbolSize: 66,
    itemStyle: { color: colors[item.type] }, label: { show: true, color: '#fff', fontWeight: 'bold' },
    tooltip: { formatter: entityLabel(item) },
  }));
  const links = relations.map((relation) => ({
    source: key(relation.subject), target: key(relation.object), value: relation.predicate,
    label: { show: true, formatter: relation.predicate, color: '#173b52' },
    lineStyle: { color: '#46a8b2', width: 2.5, curveness: .08 },
  }));
  graph = window.echarts.init(target);
  graph.setOption({
    animationDuration: 350, tooltip: { trigger: 'item' },
    series: [{ type: 'graph', layout: 'force', roam: true, draggable: true,
      force: { repulsion: 280, edgeLength: 150 }, data: nodes, links,
      emphasis: { focus: 'adjacency' }, edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 9 }],
  });
}

function render(value) {
  const { text, entities, relations } = value;
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
    else for (const relation of relations) box.append(relationCard(relation));
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
analyze();
