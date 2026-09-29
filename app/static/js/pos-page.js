import { api, node } from './api.js';
import { activeTab } from './layout.js';

const $ = (id) => document.getElementById(id);
const CATEGORY_LABELS = {
  noun: '名词', verb: '动词', adjective: '形容词',
  pronoun: '代词', adverb: '副词', other: '其它'
};
let request = 0;
let loaded = false;
let timer;

function put(host, tag, text, className = '') {
  const element = node(tag, text, className);
  host.append(element);
  return element;
}

function render(value) {
  const original = $('pos-original');
  const result = $('pos-result');
  original.replaceChildren();
  result.replaceChildren();
  const chars = Array.from(value.text);
  let cursor = 0;
  for (const item of value.items) {
    if (item.start > cursor) original.append(document.createTextNode(chars.slice(cursor, item.start).join('')));
    const mark = put(original, 'mark', item.word, `pos-mark pos-${item.category}`);
    mark.title = `${item.pos} · [${item.start},${item.end})`;
    const chip = put(result, 'span', '', `pos-token pos-${item.category}`);
    put(chip, 'strong', item.word);
    put(chip, 'small', item.pos);
    cursor = item.end;
  }
  if (cursor < chars.length) original.append(document.createTextNode(chars.slice(cursor).join('')));
  if (!value.items.length) result.textContent = '当前文本没有可标注的词语。';
  const counts = $('pos-counts');
  counts.replaceChildren();
  const total = value.items.length;
  for (const [category, count] of Object.entries(value.counts)) {
    const line = put(counts, 'div', '', `pos-count pos-${category}`);
    put(line, 'strong', CATEGORY_LABELS[category] || category);
    put(line, 'span', String(count));
    const rail = put(line, 'i', '');
    rail.style.width = `${Math.round(100 * count / total)}%`;
  }
  const explanations = $('pos-explanations');
  explanations.replaceChildren();
  for (const [tag, explanation] of Object.entries(value.labels)) {
    const line = put(explanations, 'div', '', 'pos-explanation');
    put(line, 'strong', tag);
    put(line, 'span', explanation);
  }
  const table = document.createElement('table');
  const head = document.createElement('tr');
  for (const label of ['词语', '原始 POS', '辅助粗类', 'start', 'end']) put(head, 'th', label);
  table.append(head);
  for (const item of value.items) {
    const tr = document.createElement('tr');
    for (const label of [item.word, item.pos, CATEGORY_LABELS[item.category], item.start, item.end]) {
      put(tr, 'td', String(label));
    }
    table.append(tr);
  }
  $('pos-table').replaceChildren(table);
  $('pos-summary').textContent = `${value.model} · ${value.tokenizer} · ${chars.length} 字符 · ${total} 词 · ${value.elapsed_ms} ms。交互结果不构成 POS benchmark。`;
}

async function analyze() {
  const text = $('pos-input').value.trim();
  if (!text) { $('pos-status').textContent = '请输入中文文本。'; return; }
  const id = ++request;
  $('pos-status').textContent = '正在分词并识别词性…';
  try {
    const value = await api('/api/v1/course/pos', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (request !== id) return;
    render(value);
    loaded = true;
    $('pos-status').textContent = `已完成 · ${value.items.length} 词 · ${value.elapsed_ms} ms`;
  } catch (error) {
    if (request !== id) return;
    $('pos-status').textContent = `POS 暂不可用：${error.message}`;
  }
}

$('pos-input').addEventListener('input', () => {
  clearTimeout(timer); timer = setTimeout(analyze, 450);
});
$('pos-analyze').addEventListener('click', analyze);
for (const button of document.querySelectorAll('[data-pos-example]')) {
  button.addEventListener('click', () => {
    $('pos-input').value = button.dataset.posExample;
    analyze();
  });
}
document.addEventListener('module-tab', () => {
  if (activeTab() === 'pos' && !loaded) analyze();
});
if (activeTab() === 'pos') analyze();
