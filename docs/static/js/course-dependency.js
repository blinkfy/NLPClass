import {retainViewFocus} from './course-view-focus.js';
import { api, node } from './api.js';

const $ = (id) => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const state = { result:null, sentenceIndex:0, wordId:null, processWordId:null, candidateHead:null, evaluation:null, evaluationSample:0, evaluationWord:null };
let parseController = null;
let evaluationController = null;
let parseVersion = 0;
let evaluationVersion = 0;
let treeInstance = 0;

function el(tag, text = '', className = '') {
  const value = node(tag, text, className);
  return value;
}

function setStatus(element, message, error = false) {
  element.textContent = message;
  element.classList.toggle('is-error', error);
}

function fmt(value, digits = 4) {
  if (value == null || !Number.isFinite(Number(value))) return '未提供';
  return Number(value).toFixed(digits);
}

function metric(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  const number = Number(value);
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
}

function currentSentence() {
  return state.result?.sentences?.[state.sentenceIndex] || null;
}

function currentItem() {
  return currentSentence()?.items?.find((item) => Number(item.id) === Number(state.wordId)) || null;
}

function clearVisuals(message = '等待分析文本。') {
  for (const id of ['cm-source','cm-words','cm-tree-mini','cm-tree-large','cm-inspector','cm-tree-inspector','cm-tree-source','cm-candidates','cm-process-words']) $(id).replaceChildren(el('p', message, 'cm-empty'));
  for (const id of ['cm-tree-caption','cm-tree-large-caption']) $(id).textContent = '分析后可选择词项与依存弧。';
  $('cm-tree-checks').textContent = message;
  $('cm-conll').textContent = message;
  $('cm-decoder-note').textContent = '等待真实 decoder 元数据。';
  $('cm-run-meta').textContent = '等待解析';
  document.querySelectorAll('[data-cm-sentence-select]').forEach((select) => {
    select.replaceChildren(new Option('尚无解析结果', ''));
    select.disabled = true;
  });
  document.querySelectorAll('[data-cm-export]').forEach((button) => { button.disabled = true; });
}

function codepointToUtf16(text, offset) {
  return Array.from(text).slice(0, Math.max(0, Number(offset) || 0)).join('').length;
}

function selectWord(id) {
  const sentence = currentSentence();
  if (!sentence?.items?.some((item) => Number(item.id) === Number(id))) return;
  state.wordId = Number(id);
  state.processWordId = Number(id);
  state.candidateHead = sentence.items.find((item) => Number(item.id) === Number(id))?.head ?? null;
  renderCurrent();
}

function relationText(item) {
  return item?.explanation ? `${item.relation} · ${item.explanation}` : (item?.relation || '未提供');
}

function readableProtocol(protocol) {
  if (typeof protocol === 'string') return protocol;
  if (!protocol || typeof protocol !== 'object') return '固定 Gold-token parser evaluation；小型教学诊断。';
  const labels = {dataset_scope:'数据范围',benchmark_scope:'基准范围',tokenization:'分词协议',head_indexing:'HEAD 索引',relation_inventory:'关系标签体系',punctuation:'标点协议',provenance:'Gold 来源'};
  return Object.entries(labels).filter(([key]) => protocol[key] != null).map(([key,label]) => `${label}：${protocol[key]}`).join('；');
}

function addSource(host, text, sentence, selectedId, onSelect) {
  host.replaceChildren();
  const start = codepointToUtf16(text, sentence.start);
  const end = codepointToUtf16(text, sentence.end);
  let cursor = start;
  for (const item of sentence.items || []) {
    const itemStart = codepointToUtf16(text, item.start);
    const itemEnd = codepointToUtf16(text, item.end);
    if (itemStart < cursor || itemEnd < itemStart || itemEnd > end) continue;
    if (itemStart > cursor) host.append(el('span', text.slice(cursor, itemStart), 'cm-source-gap'));
    const isSelected = Number(item.id) === Number(selectedId);
    const button = el('button', item.form, `cm-token cm-button cm-source-word${isSelected ? ' is-selected cm-selected' : ''}`);
    button.type = 'button';
    button.dataset.cmWord = String(item.id);
    button.setAttribute('aria-pressed', String(Number(item.id) === Number(selectedId)));
    button.title = `第 ${item.id} 词，原文位置 ${item.start}–${item.end}`;
    button.addEventListener('click', () => onSelect(item.id));
    host.append(button);
    cursor = itemEnd;
  }
  if (cursor < end) host.append(el('span', text.slice(cursor, end), 'cm-source-gap'));
}

function drawTree(host, items, selectedId, onWord, onArc) {
  host.replaceChildren();
  if (!Array.isArray(items) || !items.length) {
    host.append(el('p', '当前句没有可显示的词项。', 'cm-empty'));
    return;
  }
  const widths = items.map((item) => Math.min(150, Math.max(58, Array.from(String(item.form || '')).length * 13 + 18)));
  const centers = [];
  let cursor = 34;
  for (const width of widths) { centers.push(cursor + width / 2); cursor += width + 28; }
  const width = Math.max(440, cursor + 12);
  const positions = new Map(items.map((item, index) => [Number(item.id), centers[index]]));
  const maxSpan = Math.max(1, ...items.map((item) => Math.abs(Number(item.id) - Number(item.head || 0))));
  const baseY = Math.max(145, 48 + maxSpan * 17);
  const height = baseY + 59;
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('width', String(width)); svg.setAttribute('height', String(height)); svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('role', 'group'); svg.setAttribute('aria-label', '真实预测依存树，箭头由 head 指向 dependent');
  const defs = document.createElementNS(SVG, 'defs');
  const markerId = `cm-arc-arrow-${++treeInstance}`;
  const marker = document.createElementNS(SVG, 'marker');
  marker.setAttribute('id', markerId); marker.setAttribute('markerWidth', '7'); marker.setAttribute('markerHeight', '7'); marker.setAttribute('refX', '6'); marker.setAttribute('refY', '3.5'); marker.setAttribute('orient', 'auto'); marker.setAttribute('markerUnits', 'strokeWidth');
  const arrow = document.createElementNS(SVG, 'path'); arrow.setAttribute('d', 'M0,0 L7,3.5 L0,7 z'); arrow.setAttribute('fill', '#4a8f82'); marker.append(arrow); defs.append(marker); svg.append(defs);
  const arcs = document.createElementNS(SVG, 'g'); arcs.setAttribute('aria-label', '依存弧');
  for (const item of items) {
    const dep = Number(item.id), head = Number(item.head), depX = positions.get(dep);
    if (!Number.isFinite(depX) || !Number.isFinite(head)) continue;
    const group = document.createElementNS(SVG, 'g');
    group.setAttribute('class', 'cm-svg-arc-group'); group.setAttribute('role', 'button'); group.setAttribute('tabindex', '0');
    group.setAttribute('data-cm-arc', String(dep)); group.setAttribute('aria-label', `${head === 0 ? 'ROOT' : `词 ${head}`} 指向词 ${dep} ${item.form}，关系 ${item.relation}`);
    let d, labelX, labelY;
    if (head === 0) {
      d = `M ${depX} ${baseY - 64} Q ${depX} ${baseY - 42} ${depX} ${baseY - 22}`;
      labelX = depX; labelY = baseY - 67;
    } else {
      const headX = positions.get(head);
      if (!Number.isFinite(headX)) continue;
      const span = Math.abs(dep - head), peak = baseY - 22 - (20 + span * 13);
      d = `M ${headX} ${baseY - 22} Q ${(headX + depX) / 2} ${peak} ${depX} ${baseY - 22}`;
      labelX = (headX + depX) / 2; labelY = peak + 12;
    }
    const hit = document.createElementNS(SVG, 'path'); hit.setAttribute('d', d); hit.setAttribute('class', 'cm-svg-arc-hit');
    const path = document.createElementNS(SVG, 'path'); path.setAttribute('d', d); path.setAttribute('class', `cm-svg-arc cm-token${Number(item.id) === Number(selectedId) ? ' is-selected cm-selected' : ''}`); path.setAttribute('marker-end', `url(#${markerId})`);
    const label = document.createElementNS(SVG, 'text'); label.setAttribute('x', String(labelX)); label.setAttribute('y', String(labelY)); label.setAttribute('text-anchor', 'middle'); label.setAttribute('class', head === 0 ? 'cm-svg-root' : 'cm-svg-label'); label.textContent = head === 0 ? 'ROOT' : String(item.relation || 'dep');
    group.append(hit, path, label);
    group.addEventListener('click', () => onArc(dep));
    group.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onArc(dep); } });
    arcs.append(group);
  }
  svg.append(arcs);
  const nodes = document.createElementNS(SVG, 'g'); nodes.setAttribute('aria-label', '依存词项');
  items.forEach((item, index) => {
    const x = centers[index], w = widths[index];
    const isSelected = Number(item.id) === Number(selectedId);
    const group = document.createElementNS(SVG, 'g'); group.setAttribute('class', `cm-svg-node cm-token${isSelected ? ' is-selected cm-selected' : ''}`); group.setAttribute('role', 'button'); group.setAttribute('tabindex', '0'); group.setAttribute('data-cm-word', String(item.id)); group.setAttribute('aria-label', `选择词 ${item.id} ${item.form}，head ${item.head}，${item.relation}`);
    const rect = document.createElementNS(SVG, 'rect'); rect.setAttribute('x', String(x - w / 2)); rect.setAttribute('y', String(baseY - 22)); rect.setAttribute('width', String(w)); rect.setAttribute('height', '34'); rect.setAttribute('rx', '7'); rect.setAttribute('class', 'cm-svg-token');
    const form = document.createElementNS(SVG, 'text'); form.setAttribute('x', String(x)); form.setAttribute('y', String(baseY - 7)); form.setAttribute('text-anchor', 'middle'); form.setAttribute('class', 'cm-svg-form'); form.textContent = String(item.form || '');
    const meta = document.createElementNS(SVG, 'text'); meta.setAttribute('x', String(x)); meta.setAttribute('y', String(baseY + 28)); meta.setAttribute('text-anchor', 'middle'); meta.setAttribute('class', 'cm-svg-meta'); meta.textContent = `${item.id} · head ${item.head}`;
    group.append(rect, form, meta); group.addEventListener('click', () => onWord(item.id));
    group.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onWord(item.id); } });
    nodes.append(group);
  });
  svg.append(nodes); host.append(svg);
}

function renderWords(sentence) {
  const host = $('cm-words'); host.replaceChildren();
  sentence.items.forEach((item) => {
    const selected = Number(item.id) === Number(state.wordId);
    const button = el('button', '', `cm-row cm-token cm-button cm-word-row${selected ? ' is-selected cm-selected' : ''}`);
    button.type = 'button'; button.dataset.cmWord = String(item.id); button.setAttribute('aria-pressed', String(Number(item.id) === Number(state.wordId)));
    button.append(el('small', String(item.id)), el('strong', String(item.form)), el('span', `${item.head === 0 ? 'ROOT' : `h${item.head}`} · ${item.relation}`));
    button.addEventListener('click', () => selectWord(item.id)); host.append(button);
  });
}

function renderInspector(sentence, hostId = 'cm-inspector') {
  const host = $(hostId); host.replaceChildren();
  const item = currentItem();
  if (!item) { host.append(el('p', '选择一个词项或依存弧。', 'cm-empty')); return; }
  host.append(el('div', String(item.form), 'cm-inspector-title'));
  const values = el('div', '', 'cm-inspector-grid');
  for (const [label, value] of [['词 ID',item.id],['原文位置',`${item.start}–${item.end}`],['HEAD',item.head === 0 ? 'ROOT (0)' : `${item.head} · ${sentence.items.find((x) => Number(x.id) === Number(item.head))?.form || '词项'}`],['关系',item.relation]]) {
    const cell = el('div', '', 'cm-inspector-value'); cell.append(el('small', label), el('strong', String(value))); values.append(cell);
  }
  host.append(values, el('p', relationText(item), 'cm-explanation'));
  const note = el('p', `原文字符偏移 ${item.start}–${item.end}；依存弧由 ${item.head === 0 ? 'ROOT' : `词 ${item.head}`} 指向当前词。`, 'cm-tree-caption'); host.append(note);
  const provenance = el('details', '', 'cm-disclosure'); provenance.append(el('summary', '模型来源与运行环境'));
  const source = state.result.source ?? {};
  const details = {source, environment:state.result.environment ?? null, tokenizer:state.result.tokenizer ?? null, tokenizer_run:state.result.tokenizer_run ?? null, elapsed_ms:state.result.elapsed_ms ?? null};
  provenance.append(el('pre', JSON.stringify(details, null, 2))); host.append(provenance);
}

function updateSentenceSelectors() {
  document.querySelectorAll('[data-cm-sentence-select]').forEach((select) => {
    const previous = Number(select.value);
    select.replaceChildren(...state.result.sentences.map((sentence, index) => new Option(`${index + 1}. ${sentence.text}`, String(index))));
    select.value = String(Number.isInteger(previous) && previous === state.sentenceIndex ? previous : state.sentenceIndex);
    select.disabled = state.result.sentences.length < 2;
  });
}

function renderProcess(sentence) { return retainViewFocus(document.querySelector('.cm-page'),()=>renderProcessView(sentence)); }
function renderProcessView(sentence) {
  const itemHost = $('cm-process-words'); itemHost.replaceChildren();
  sentence.items.forEach((item) => {
    const selectedWord = Number(item.id) === Number(state.processWordId);
    const button = el('button', `${item.id} · ${item.form}`, `cm-token cm-button cm-process-word${selectedWord ? ' is-selected cm-selected' : ''}`);
    button.type = 'button'; button.dataset.cmProcessWord = String(item.id); button.setAttribute('aria-pressed', String(Number(item.id) === Number(state.processWordId)));
    button.title = `真实预测 head：${item.head === 0 ? 'ROOT' : item.head}`;
    button.addEventListener('click', () => selectWord(item.id)); itemHost.append(button);
  });
  const selected = sentence.items.find((item) => Number(item.id) === Number(state.processWordId)) || sentence.items[0];
  if (!selected) return;
  const scores = sentence.arc_scores;
  const row = Array.isArray(scores) ? scores[Number(selected.id)] : null;
  const candidates = [];
  if (Array.isArray(row)) {
    for (let head = 0; head <= sentence.items.length; head++) {
      if (head === Number(selected.id) || row[head] == null || !Number.isFinite(Number(row[head]))) continue;
      candidates.push({head, score:Number(row[head])});
    }
  }
  candidates.sort((a,b) => b.score - a.score || a.head - b.head);
  const host = $('cm-candidates'); host.replaceChildren();
  host.append(el('p', `当前 dependent ${selected.id} · ${selected.form}。以下为每个候选 head 的原始 biaffine arc logit；实际树 head 为 ${selected.head === 0 ? 'ROOT (0)' : `${selected.head} · ${sentence.items.find((x) => Number(x.id) === Number(selected.head))?.form || ''}`}。`, 'cm-decoder-note'));
  if (!candidates.length) {
    host.append(el('p', '此响应没有可用的弧分数矩阵；不会用规则或示例数值代替。', 'cm-empty'));
  } else {
    candidates.forEach((candidate, rank) => {
      const actual = Number(candidate.head) === Number(selected.head), explored = Number(candidate.head) === Number(state.candidateHead);
      const button = el('button', '', `cm-row cm-button cm-candidate${explored ? ' is-selected cm-selected' : ''}${actual ? ' is-actual' : ''}`);
      button.type = 'button'; button.dataset.cmCandidate = String(candidate.head); button.setAttribute('aria-pressed', String(explored));
      const head = candidate.head === 0 ? 'ROOT' : `${candidate.head} · ${sentence.items.find((x) => Number(x.id) === candidate.head)?.form || ''}`;
      const title=el('span', `${rank + 1}. head ${head}`, 'cm-candidate-name');
      const track=el('span','','cm-score-track'),bar=el('i','','cm-score-bar');
      const scale=Math.max(...candidates.map(c=>Math.abs(c.score)),1e-12);
      bar.style.width=`${Math.abs(candidate.score)/scale*50}%`;
      bar.classList.add(candidate.score<0?'is-negative':'is-positive');track.append(bar);track.setAttribute('aria-hidden','true');
      button.append(title,track,el('strong',fmt(candidate.score)),el('small',actual?'实际树 head':explored?'候选观察中':'候选'));
      button.addEventListener('click', () => { state.candidateHead = candidate.head; renderProcess(sentence); }); host.append(button);
    });
  }
  if(candidates.length)host.append(el('p','条长表示原始 logit 的绝对大小；中线为 0，左右区分负 / 正。不是概率。','cm-score-legend'));
  host.append(el('h3', `最终 head → 关系标签：${selected.relation}`));
  const relation = sentence.relation_scores?.find(row=>Number(row.dependent)===Number(selected.id));
  if(relation?.verified && Number(relation.head)===Number(selected.head)) {
    host.append(el('p', `以下 relation logits 条件为实际树 head ${selected.head}，不是正在观察的其他 head；最高分已与 parser 最终标签核验。`, 'cm-note'));
    const list=el('div','','cm-relation-candidates');
    relation.candidates.slice(0,5).forEach(candidate=>{
      const row=el('div','','cm-row');
      row.append(el('strong',candidate.label),el('span',fmt(candidate.score)),el('small',candidate.label===selected.relation?'最终关系':'候选'));
      list.append(row);
    });
    host.append(list,el('p','关系 logit 是未归一化分数，不是概率。关系标签是在已选 head 下预测，再与 head 一起形成有标签依存树。','cm-note'));
  } else host.append(el('p','此历史响应未暴露 relation score；只展示真实最终标签，不补造候选数值。','cm-note'));
  const decoder = sentence.decoder ?? state.result.decoder;
  $('cm-decoder-note').textContent = `实际 decoder：${typeof decoder === 'string' ? decoder : decoder == null ? '服务未提供说明' : JSON.stringify(decoder)}。先检查局部最高分的 head 是否组成有效树；若不满足树约束，上游使用整体解码。因此局部第一名不保证是最终 head；点选候选不改写树。`;
}

function renderCurrent() { return retainViewFocus(document.querySelector('.cm-page'),()=>renderCurrentView()); }
function renderCurrentView() {
  if (!state.result) { clearVisuals(); return; }
  const sentence = currentSentence();
  if (!sentence) { clearVisuals('当前没有句子结果。'); return; }
  if (!sentence.items.some((item) => Number(item.id) === Number(state.wordId))) state.wordId = Number(sentence.items[0]?.id) || null;
  if (!sentence.items.some((item) => Number(item.id) === Number(state.processWordId))) state.processWordId = state.wordId;
  if (state.candidateHead == null && currentItem()) state.candidateHead = currentItem().head;
  updateSentenceSelectors();
  const text = state.result.text || '';
  addSource($('cm-source'), text, sentence, state.wordId, selectWord);
  addSource($('cm-tree-source'), text, sentence, state.wordId, selectWord);
  renderWords(sentence);
  drawTree($('cm-tree-mini'), sentence.items, state.wordId, selectWord, selectWord);
  drawTree($('cm-tree-large'), sentence.items, state.wordId, selectWord, selectWord);
  renderInspector(sentence);
  renderInspector(sentence, 'cm-tree-inspector');
  renderProcess(sentence);
  const checks = $('cm-tree-checks'); checks.replaceChildren();
  for (const [key, label] of [['single_root','恰有一个 ROOT'],['acyclic','无环'],['root_reachable','所有词可达 ROOT']]) {
    const row = el('li', sentence.checks?.[key] ? label : `${label} · 未通过`, ''); checks.append(row);
  }
  const details = sentence.checks ? `${sentence.checks.node_count ?? sentence.items.length} 个词项 · root / cycle / reachability 校验通过` : '服务未提供树校验记录';
  $('cm-tree-caption').textContent = `${sentence.items.length} 词 · ${details}。点击弧或词项查看关联信息。`;
  $('cm-tree-large-caption').textContent = `${sentence.text} · ${sentence.items.length} 词 · 箭头指向 dependent。左右滑动查看完整依存树。`;
  $('cm-conll').textContent = sentence.conll || '服务未提供 CoNLL 行。';
  document.querySelectorAll('[data-cm-export]').forEach((button) => { button.disabled = !sentence.conll; });
  const source = state.result.source;
  const sourceName = typeof source === 'string' ? source : source?.model || '本地预训练依存模型';
  $('cm-run-meta').textContent = `${state.result.tokenizer || '冻结 HMM 分词'} · ${sourceName} · ${state.result.elapsed_ms ?? '—'} ms`;
}

function clearParseOutput(message = '等待分析文本。') {
  state.result = null; state.sentenceIndex = 0; state.wordId = null; state.processWordId = null; state.candidateHead = null;
  clearVisuals(message);
}

async function analyze() {
  const text = $('cm-input').value;
  parseController?.abort();
  const version = ++parseVersion;
  if (!text.trim()) { clearParseOutput('请输入非空文本后再分析。'); setStatus($('cm-parse-status'), '请先输入一段非空文本。', true); return; }
  clearParseOutput('正在调用本地模型…');
  parseController = new AbortController();
  $('cm-analyze').disabled = true;
  setStatus($('cm-parse-status'), '正在进行冻结分词与依存解析…');
  try {
    const started = Date.now();
    while (true) {
      const resources = await api('/api/v1/course/modules/resources', {signal:parseController.signal});
      if (!resources.busy) break;
      if (Date.now() - started > 120000) throw new Error('本地 CPU 持续忙碌；请稍后重试。');
      setStatus($('cm-parse-status'), '正在等待前一个本地 CPU 任务结束；不会显示旧结果。');
      await new Promise(resolve => setTimeout(resolve, 750));
      if (version !== parseVersion) throw new DOMException('Aborted', 'AbortError');
    }
    const result = await api('/api/v1/course/dependency/parse', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text}), signal:parseController.signal});
    if (version !== parseVersion) return;
    if (!Array.isArray(result.sentences) || !result.sentences.length) throw new Error('解析响应没有有效句子。');
    state.result = result; state.sentenceIndex = 0; state.wordId = Number(result.sentences[0].items?.[0]?.id) || null; state.processWordId = state.wordId; state.candidateHead = result.sentences[0].items?.[0]?.head ?? null;
    renderCurrent();
    setStatus($('cm-parse-status'), `解析完成 · ${result.sentences.length} 句 · ${result.environment?.device || 'CPU'} · ${Math.round(result.elapsed_ms || 0)} ms。选择词项继续探索。`);
  } catch (error) {
    if (error.name === 'AbortError' || version !== parseVersion) return;
    clearParseOutput('解析失败，当前没有展示上一条结果。');
    setStatus($('cm-parse-status'), error.message || '依存解析失败。', true);
  } finally {
    if (version === parseVersion) $('cm-analyze').disabled = false;
  }
}

function exportConll() {
  const sentence = currentSentence();
  if (!sentence?.conll) return;
  const body = `# text = ${sentence.text}\n${sentence.conll}\n`;
  const url = URL.createObjectURL(new Blob([body], {type:'text/plain;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = `dependency-${state.sentenceIndex + 1}.conllu`; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderMetricCards(host, metrics) {
  const cards = el('div', '', 'cm-eval-metrics cm-metrics');
  for (const [key, label] of [['support','计分词数'],['uas','UAS'],['las','LAS']]) {
    const card = el('div', '', 'cm-eval-metric'); card.append(el('small', label), el('strong', key === 'support' ? String(metrics?.[key] ?? '—') : metric(metrics?.[key]))); cards.append(card);
  }
  host.append(cards);
}

function renderEvaluationAggregate(metrics) {
  const host = $('cm-evaluation-aggregate'); host.replaceChildren();
  host.append(el('p', '排除标点 · 聚合主报告', 'cm-eval-protocol'));
  renderMetricCards(host, metrics || {});
  host.append(el('p', '包含标点 · 协议对照', 'cm-eval-protocol'));
  renderMetricCards(host, state.evaluation?.including_punctuation || {});
  host.setAttribute('aria-label', '教学样本聚合指标');
}

function renderEvaluationSample() { return retainViewFocus(document.querySelector('.cm-page'),renderEvaluationSampleView); }
function renderEvaluationSampleView() {
  const data = state.evaluation;
  if (!data?.samples?.length) return;
  const sample = data.samples[state.evaluationSample] || data.samples[0];
  const output = $('cm-evaluation-output'); output.replaceChildren();
  const sidebar = el('section', '', 'panel'); sidebar.append(el('h2', 'Gold 句子', 'cm-tree-detail-heading'));
  const list = el('div', '', 'cm-eval-samples');
  data.samples.forEach((record, index) => {
    const button = el('button', `${record.id || `样本 ${index + 1}`} · ${record.text}`, `cm-eval-sample${index === state.evaluationSample ? ' is-selected' : ''}`);
    button.type = 'button'; button.dataset.cmEvalSample=String(index); button.setAttribute('aria-pressed', String(index === state.evaluationSample)); button.addEventListener('click', () => { state.evaluationSample = index; state.evaluationWord = null; renderEvaluationSample(); }); list.append(button);
  });
  sidebar.append(list);
  const detail = el('section', '', 'panel');
  detail.append(el('h2', `${sample.id || `样本 ${state.evaluationSample + 1}`} · 固定 Gold 分词`, 'cm-tree-detail-heading'));
  detail.append(el('p', '排除标点', 'cm-eval-protocol'));
  renderMetricCards(detail, sample.metrics || {});
  detail.append(el('p', '包含标点', 'cm-eval-protocol'));
  renderMetricCards(detail, sample.protocol_metrics?.include_punctuation || {});
  const tree = el('div', '', 'cm-tree-scroll');
  const predItems = (sample.prediction?.items || []).map((item, index) => ({...item, id:Number(item.id ?? index + 1), form:item.form ?? sample.words?.[index] ?? '', head:Number(item.head), relation:item.relation ?? ''}));
  const selected = state.evaluationWord ?? predItems[0]?.id;
  drawTree(tree, predItems, selected, (id) => { state.evaluationWord = Number(id); renderEvaluationSample(); }, (id) => { state.evaluationWord = Number(id); renderEvaluationSample(); });
  detail.append(tree);
  const tableWrap = el('div', '', 'cm-eval-table-wrap'); const table = el('table', '', 'cm-eval-table');
  const thead = el('thead'); const headRow = el('tr'); ['词项','Gold HEAD / 关系','模型 HEAD / 关系','弧'].forEach((label) => headRow.append(el('th', label))); thead.append(headRow); table.append(thead);
  const tbody = el('tbody');
  (sample.words || []).forEach((word, index) => {
    const item = predItems[index] || {}, id = Number(item.id ?? index + 1), goldHead = sample.gold_heads?.[index], goldRel = sample.gold_relations?.[index], predictedHead = item.head, predictedRel = item.relation;
    const headCorrect = Number(predictedHead) === Number(goldHead), labelCorrect = headCorrect && predictedRel === goldRel;
    const tr = el('tr', '', `${headCorrect && labelCorrect ? '' : 'is-error'}${id === Number(selected) ? ' is-selected' : ''}`);
    const wordCell = el('td'); const wordButton = el('button', String(word)); wordButton.type = 'button'; wordButton.dataset.cmEvalWord=String(id); wordButton.addEventListener('click', () => { state.evaluationWord = id; renderEvaluationSample(); }); wordCell.append(wordButton);
    tr.append(wordCell, el('td', `${goldHead === 0 ? 'ROOT' : goldHead} · ${goldRel ?? '—'}`), el('td', `${predictedHead === 0 ? 'ROOT' : predictedHead ?? '—'} · ${predictedRel ?? '—'}`), el('td', labelCorrect ? 'LAS ✓' : headCorrect ? 'HEAD ✓ · 标签 ✕' : 'HEAD ✕'));
    tr.addEventListener('click', (event) => { if (event.target !== wordButton) { state.evaluationWord = id; renderEvaluationSample(); } }); tbody.append(tr);
  });
  table.append(tbody); tableWrap.append(table); detail.append(tableWrap);
  output.append(sidebar, detail);
}

async function loadEvaluation() {
  evaluationController?.abort();
  const version = ++evaluationVersion;
  evaluationController = new AbortController();
  $('cm-load-evaluation').disabled = true;
  $('cm-evaluation-output').replaceChildren(el('div', '正在读取固定教学 Gold 与模型预测…', 'panel cm-empty'));
  $('cm-evaluation-aggregate').replaceChildren(el('p', '正在读取聚合指标…', 'cm-empty'));
  $('cm-evaluation-provenance').textContent = '正在读取来源与运行环境…';
  setStatus($('cm-evaluation-status'), '正在读取独立教学评测；不会访问分类 Test。');
  try {
    const result = await api('/api/v1/course/dependency/evaluation', {signal:evaluationController.signal});
    if (version !== evaluationVersion) return;
    if (!Array.isArray(result.samples) || !result.samples.length) throw new Error('评测响应缺少有效教学样本。');
    state.evaluation = result; state.evaluationSample = 0; state.evaluationWord = null;
    const protocol = readableProtocol(result.protocol);
    $('cm-evaluation-protocol').textContent = protocol || '固定 Gold-token parser evaluation；小型教学诊断。';
    $('cm-evaluation-provenance').textContent = JSON.stringify({source:result.source, environment:result.environment, elapsed_ms:result.elapsed_ms}, null, 2);
    renderEvaluationAggregate(result.aggregate);
    renderEvaluationSample();
    setStatus($('cm-evaluation-status'), `已读取 ${result.samples.length} 条 Gold-token 教学样本。`);
  } catch (error) {
    if (error.name === 'AbortError' || version !== evaluationVersion) return;
    state.evaluation = null;
    $('cm-evaluation-output').replaceChildren(el('div', '评测读取失败。', 'panel cm-empty'));
    $('cm-evaluation-protocol').textContent = '评测协议尚未读取。';
    $('cm-evaluation-aggregate').replaceChildren(el('p', '评测失败，未显示旧指标。', 'cm-empty'));
    $('cm-evaluation-provenance').textContent = '评测来源未读取。';
    setStatus($('cm-evaluation-status'), error.message || '教学评测暂不可用。', true);
  } finally {
    if (version === evaluationVersion) $('cm-load-evaluation').disabled = false;
  }
}

function selectSentence(index) {
  if (!state.result || !Number.isInteger(index) || !state.result.sentences[index]) return;
  state.sentenceIndex = index;
  const sentence = currentSentence();
  state.wordId = Number(sentence.items?.[0]?.id) || null;
  state.processWordId = state.wordId;
  state.candidateHead = sentence.items?.[0]?.head ?? null;
  renderCurrent();
}

async function loadResources() {
  try {
    const value = await api('/api/v1/course/modules/resources');
    const ready = value.dependency_ready;
    $('cm-resource-status').textContent = ready
      ? `HanLP 依存模型已就绪 · ${value.device || 'CPU'} · 冻结 HMM-1 Smooth 分词 · 点击分析后运行。`
      : `依存模型资源尚未准备。${value.policy ? ` ${value.policy}` : ''} 输入仍可编辑；点击分析时会显示明确的资源错误。`;
    $('cm-resource-status').classList.toggle('is-unavailable', !ready);
  } catch (error) {
    $('cm-resource-status').textContent = `资源状态暂不可用：${error.message}。你仍可查看课程说明。`;
  }
}

$('cm-analyze').addEventListener('click', analyze);
$('cm-input').addEventListener('input', () => {
  parseController?.abort(); ++parseVersion; $('cm-analyze').disabled = false;
  clearParseOutput('文本已更改；请重新点击分析。');
  setStatus($('cm-parse-status'), '文本已更改；上一条结果已清除。');
});
$('cm-clear').addEventListener('click', () => {
  parseController?.abort(); ++parseVersion; $('cm-input').value = ''; $('cm-analyze').disabled = false;
  clearParseOutput(); setStatus($('cm-parse-status'), '输入已清空。'); $('cm-input').focus();
});
document.querySelectorAll('[data-cm-example]').forEach((button) => button.addEventListener('click', () => {
  parseController?.abort(); ++parseVersion; $('cm-input').value = button.dataset.cmExample; $('cm-analyze').disabled = false;
  clearParseOutput('示例已填入；点击“分析文本”后才会运行模型。'); setStatus($('cm-parse-status'), '示例已填入；点击“分析文本”后才会运行模型。');
}));
document.querySelectorAll('[data-cm-sentence-select]').forEach((select) => select.addEventListener('change', () => selectSentence(Number(select.value))));
document.querySelectorAll('[data-cm-export]').forEach((button) => button.addEventListener('click', exportConll));
$('cm-load-evaluation').addEventListener('click', loadEvaluation);
loadResources();
