import { api, node } from './api.js';
import { activeTab } from './layout.js';

const STATES = ['B', 'M', 'E', 'S'];
const values = new Map();
const requestIds = new Map();
const $ = (id) => document.getElementById(id);
const fmt = (value) => value == null ? '−∞' : Number(value).toFixed(3);

function row(parent, tag, label, className = '') {
  const item = node(tag, label, className);
  parent.append(item);
  return item;
}

function renderCurrent(model, index) {
  const value = values.get(model);
  if (!value) return;
  const text = Array.from(value.text);
  const position = Math.max(0, Math.min(index, text.length - 1));
  const tag = value.tags[position];
  $(`${model}-focus-char`).textContent = text[position];
  $(`${model}-focus-index`).textContent = `第 ${position + 1} / ${text.length} 个字符`;
  $(`${model}-focus-tag`).textContent = `${tag} · ${model === 'crf' ? '整句最优标签' : 'Viterbi 最优标签'}`;
  $(`${model}-focus-detail`).textContent = model === 'crf'
    ? `边缘概率 ${fmt(value.marginals[position][STATES.indexOf(tag)])}`
    : `累计线性得分 ${fmt(value.scores[position][STATES.indexOf(tag)])}`;
  for (const button of $(`${model}-path`).querySelectorAll('button')) {
    button.classList.toggle('is-active', Number(button.dataset.index) === position);
  }
  const grid = $(`${model}-state-grid`);
  grid.replaceChildren();
  STATES.forEach((state, i) => {
    const cell = row(grid, 'div', '', `course-state-card ${state === tag ? 'is-path' : ''}`);
    row(cell, 'strong', state);
    row(cell, 'span', model === 'crf'
      ? `P(yₜ | X) ${fmt(value.marginals[position][i])}`
      : `${fmt(value.scores[position][i])} 分`);
    if (model === 'perceptron' && position > 0 && value.backpointer[position][i] >= 0) {
      row(cell, 'small', `前驱 ${STATES[value.backpointer[position][i]]}`);
    }
  });
  const features = $(`${model}-features`);
  features.replaceChildren();
  const items = model === 'crf' ? value.features[position] : value.contributions[position];
  for (const [name, weight] of items) {
    const feature = row(features, 'div', '', 'course-feature-row');
    row(feature, 'span', name);
    row(feature, 'strong', `${weight >= 0 ? '+' : ''}${fmt(weight)}`);
  }
}

function details(model, value) {
  const host = $(`${model}-details`);
  const table = document.createElement('table');
  const head = document.createElement('tr');
  ['位置', '字符', '路径', ...STATES].forEach((label) => row(head, 'th', label));
  table.append(head);
  Array.from(value.text).forEach((char, index) => {
    const tr = document.createElement('tr');
    [String(index), char, value.tags[index],
      ...(model === 'crf' ? value.marginals[index] : value.scores[index]).map(fmt)
    ].forEach((label) => row(tr, 'td', label));
    table.append(tr);
  });
  host.replaceChildren(table);
}

function renderUpdate(update) {
  const host = $('perceptron-update');
  host.replaceChildren();
  if (!update) { row(host, 'p', '该训练记录没有发生预测差异。'); return; }
  row(host, 'p', update.text, 'course-update-text');
  const paths = row(host, 'div', '', 'course-update-paths');
  row(paths, 'span', `Gold　${update.gold_tags.join(' → ')}`);
  row(paths, 'span', `预测　${update.predicted_tags.join(' → ')}`);
  const diff = row(host, 'div', '', 'course-update-diff');
  for (const [label, items, cls] of [
    ['+1/次 · Gold 特征', update.positive_features, 'positive'],
    ['−1/次 · 预测特征', update.negative_features, 'negative']
  ]) {
    const group = row(diff, 'div', '', cls);
    row(group, 'strong', label);
    for (const [feature, count] of items) row(group, 'span', `${feature} (${count})`);
  }
  row(host, 'small', '列出前 8 项差分；同一特征多次出现时权重变化会累计。', 'muted');
}

function renderTransitions(weights) {
  const host = $('crf-transitions');
  host.replaceChildren();
  const table = document.createElement('table');
  const head = document.createElement('tr');
  ['From / To', ...STATES].forEach((label) => row(head, 'th', label));
  table.append(head);
  weights.forEach((line, index) => {
    const tr = document.createElement('tr');
    row(tr, 'th', STATES[index]);
    line.forEach((weight) => row(tr, 'td', weight == null ? '未学习' : fmt(weight)));
    table.append(tr);
  });
  host.append(table);
  row(host, 'p', '表格是模型实际学习到的转移权重；“未学习”表示训练子集中未出现相应转移。', 'muted');
}

function render(model, value) {
  values.set(model, value);
  const tokens = $(`${model}-tokens`);
  tokens.replaceChildren();
  for (const token of value.tokens) row(tokens, 'span', token, 'result-token');
  const path = $(`${model}-path`);
  path.replaceChildren();
  Array.from(value.text).forEach((char, index) => {
    const button = row(path, 'button', '', 'course-path-step');
    button.type = 'button';
    button.dataset.index = String(index);
    button.append(node('strong', char), node('small', value.tags[index]));
    button.addEventListener('click', () => renderCurrent(model, index));
  });
  details(model, value);
  renderCurrent(model, 0);
  const meta = value.metadata || {};
  const summary = $(`${model}-summary`);
  summary.replaceChildren();
  row(summary, 'h3', '当前运行摘要');
  row(summary, 'p', `${value.tokens.length} 词 · ${Array.from(value.text).length} 字符 · ${value.elapsed_ms} ms`);
  if (meta.train_size) row(summary, 'p', `课程训练子集 ${meta.train_size} 句 · dev ${meta.dev_size} 句 · seed ${meta.seed}`);
  row(summary, 'small', '交互文本推理摘要。');
  if (model === 'perceptron') renderUpdate(value.update_example);
  else renderTransitions(value.transitions);
}

async function run(model) {
  const text = $(`${model}-input`).value.trim();
  if (!text) { $(`${model}-status`).textContent = '请输入文本。'; return; }
  const id = (requestIds.get(model) || 0) + 1;
  requestIds.set(model, id);
  $(`${model}-status`).textContent = '正在计算当前输入的 BMES 路径…';
  try {
    const value = await api('/api/v1/course/sequence', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model })
    });
    if (requestIds.get(model) !== id) return;
    render(model, value);
    $(`${model}-status`).textContent = `已完成 · ${value.model} · ${value.elapsed_ms} ms`;
  } catch (error) {
    if (requestIds.get(model) !== id) return;
    $(`${model}-status`).textContent = `当前模型不可用：${error.message}`;
  }
}

let comparisonTimer;
async function observePerceptronContext() {
  const button = $('perceptron-observe');
  const host = $('perceptron-observation');
  button.disabled = true;
  host.textContent = '正在用已保存模型计算两个上下文…';
  const sentences = ['北京大学欢迎你。', '北方的天气很好。'];
  try {
    const results = await Promise.all(sentences.map((text) => api('/api/v1/course/sequence', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text,model:'perceptron'})
    })));
    host.replaceChildren();
    results.forEach((value, index) => {
      const position = Array.from(value.text).indexOf('北');
      const card = row(host, 'article', '', 'course-observation-card');
      row(card, 'strong', sentences[index]);
      row(card, 'p', `“北”的 BMES：${value.tags[position]} · 分词：${value.tokens.join(' / ')}`);
      row(card, 'small', STATES.map((state, i) => `${state} ${fmt(value.scores[position][i])}`).join('　'));
    });
  } catch (error) {
    host.textContent = `对照暂不可用：${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function compare() {
  const text = $('comparison-input').value.trim();
  if (!text) return;
  const host = $('comparison-results');
  host.textContent = '正在对同一句子调用三个已保存模型…';
  const requests = [
    api('/api/v1/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text,tasks:['segmentation'],run_id:'latest',segmentation_model:'hmm1-smooth'})}),
    ...['perceptron','crf'].map((model) => api('/api/v1/course/sequence', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,model})}))
  ];
  const results = await Promise.allSettled(requests);
  if ($('comparison-input').value.trim() !== text) return;
  host.replaceChildren();
  ['HMM-1 Smooth','平均结构化感知机','线性链 CRF'].forEach((label,index) => {
    const card = row(host, 'article', '', 'comparison-result-card');
    row(card, 'h3', label);
    const result = results[index];
    if (result.status !== 'fulfilled') { row(card, 'p', `不可用：${result.reason.message}`); return; }
    const value = index === 0 ? result.value.segmentation : result.value;
    row(card, 'p', `BMES　${value.tags.join(' → ')}`, 'course-comparison-tags');
    row(card, 'p', `分词　${value.tokens.join(' / ')}`);
    row(card, 'small', `本次推理 ${result.value.elapsed_ms} ms`, 'muted');
  });
}

for (const model of ['perceptron','crf']) {
  const input = $(`${model}-input`);
  let timer;
  input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => run(model), 400); });
  $(`${model}-analyze`).addEventListener('click', () => run(model));
}
$('comparison-input').addEventListener('input', () => {
  clearTimeout(comparisonTimer); comparisonTimer = setTimeout(compare, 400);
});
$('perceptron-observe').addEventListener('click', observePerceptronContext);
document.addEventListener('module-tab', () => {
  const tab = activeTab();
  document.querySelector('.hmm-only-selector')?.classList.toggle('is-hidden', tab !== 'hmm');
  if (['perceptron','crf'].includes(tab) && !values.has(tab)) run(tab);
  if (tab === 'comparison') compare();
});
document.querySelector('.hmm-only-selector')?.classList.toggle('is-hidden', activeTab() !== 'hmm');
if (['perceptron','crf'].includes(activeTab())) run(activeTab());
if (activeTab() === 'comparison') compare();
