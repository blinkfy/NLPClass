import { api, node, notice } from './api.js';
import { ClusteringWorkbench } from './clustering-workbench.js?v=20261003-calc3';
import { Scatter3D, boundedTooltipPosition } from './clustering-scatter3d.js';

const $ = (id) => document.getElementById(id);
const sections = ['data', 'clustering', 'visualization', 'evaluation'];
const colors = ['#087f8c', '#397dca', '#bd8028', '#8b68b5', '#d2656c', '#46a078', '#a86937', '#607b96', '#b75896', '#6c9b3e'];
const charts = new Map();
const tooltipElements = new Map();
let scatter3d = null;
const state = { overview: null, result: null, run: null, k: null, request: 0, summary: null, projection3d: null, projectionRequest: 0, inspection: null, comparison: null, document: null, documentError: null, selectedId: null, selectedCluster: null, docRequest: 0, documentViews: null };
const number = (value, digits = 2) => Number.isFinite(Number(value)) && value !== null && value !== '' ? Number(value).toLocaleString('zh-CN', { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
const integer = (value) => Number.isFinite(Number(value)) && value !== null && value !== '' ? Number(value).toLocaleString('zh-CN') : '—';
const plain = (value) => value == null || value === '' ? '—' : String(value);
const htmlSafe = (value) => plain(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

const workbench = new ClusteringWorkbench({state, $, number, integer, colors, selectDocument, selectCluster, selectCentroid, selectRepresentation, selectK});

async function selectDocument(id) {
  if (!state.result || !id) return;
  const request = ++state.docRequest, run = state.run, k = state.k;
  state.selectedId = id; state.document = null; state.documentViews = null; state.documentError = null;
  const assignment = state.result.assignments.find((item) => item.id === id);
  if (assignment) state.selectedCluster = assignment.cluster_id;
  workbench.renderInspectors(); workbench.renderFlow(); workbench.renderExplorer(); workbench.renderCenterButtons(); afterRender();
  try {
    const runIds=[run,...(state.summary?.items || []).filter(item=>item.run_id!==run).map(item=>item.run_id)];
    const documentUrl=viewRun=>`/api/v1/experiments/2/document?run_id=${encodeURIComponent(viewRun)}&k=${k}&document_id=${encodeURIComponent(id)}`;
    const doc=await api(documentUrl(run));
    if (request !== state.docRequest || run !== state.run || k !== state.k) return;
    state.document = doc; state.documentViews = { [run]: doc };
    workbench.renderInspectors(); workbench.renderFlow(); workbench.renderExplorer(); afterRender();
    // Cross-representation examples are supplementary; never make the active
    // document wait for the other frozen Runs to respond.
    const comparisonRuns=runIds.slice(1);
    if(comparisonRuns.length){
      const views=await Promise.allSettled(comparisonRuns.map(viewRun=>api(documentUrl(viewRun))));
      if(request!==state.docRequest||run!==state.run||k!==state.k)return;
      views.forEach((view,index)=>{if(view.status==='fulfilled')state.documentViews[comparisonRuns[index]]=view.value;});
      workbench.renderEvolution();
    }
  } catch (error) {
    if (request !== state.docRequest || run !== state.run || k !== state.k) return;
    state.documentError = `文档检视加载失败：${error.message}`;
    workbench.renderInspectors(); workbench.renderFlow();
  }
}

function selectK(k) {
  if(k===state.k)return;
  $('clustering-k').value=String(k);loadResult();
}

function selectCluster(id) {
  const cluster = state.result?.clusters.find((item) => item.cluster_id === id);
  if (!cluster) return;
  state.selectedCluster = id;
  workbench.renderExplorer(); workbench.renderCenterButtons();
  if (cluster.representatives[0]) selectDocument(cluster.representatives[0].id);
}

function selectCentroid(id) {
  state.selectedCluster = id;
  workbench.renderExplorer(); workbench.renderCenterButtons(); workbench.renderInspectors();
  for (const profile of document.querySelectorAll('.clustering-inspected-centroid')) profile.open = true;
  afterRender();
}

function selectPoint(point) {
  if (point?.kind === 'centroid') selectCentroid(point.cluster);
  else if (point?.id) selectDocument(point.id);
}

function setSectionState(title, detail, kind = '') {
  for (const key of sections) {
    const section = $(key);
    $(`clustering-${{ data: 'data', clustering: 'clusters', visualization: 'visual', evaluation: 'evaluation' }[key]}-content`).hidden = true;
    let box = section.querySelector('.clustering-state');
    if (!box) { box = node('div', '', 'clustering-state'); section.prepend(box); }
    box.className = `clustering-state ${kind}`;
    box.replaceChildren(node('h2', title), node('p', detail));
    box.hidden = false;
  }
}

function showSection(key, contentId) {
  const section = $(key);
  section.querySelector('.clustering-state')?.setAttribute('hidden', '');
  $(contentId).hidden = false;
}

function clearCharts() {
  scatter3d?.dispose(); scatter3d = null;
  for (const chart of charts.values()) chart.dispose();
  charts.clear();
  tooltipElements.clear();
}

function representationName(item) {
  return item.representation_id === 'R2' ? 'TF-IDF → LSA → L2' : item.representation_id === 'R1' ? 'TF-IDF + bigram / DF filtering' : 'Word unigram TF-IDF';
}

function selectRepresentation(runId) {
  if (!runId || runId === state.run) return;
  $('clustering-run').value = runId;
  loadOverview();
}

function renderEvolution() {
  const summary = state.summary;
  const items = summary?.items || [];
  const box = $('clustering-representation-evolution'); box.replaceChildren();
  const select = $('clustering-representation'); select.replaceChildren();
  for (const item of items) {
    const stats = item.representation;
    const dimensions = `${integer(stats.feature_dimension ?? stats.vocabulary_size)} 词维 → ${integer(stats.representation_dimension ?? stats.dimension)} 聚类维`;
    const button = node('button', '', `clustering-evolution-step${item.run_id === state.run ? ' is-current' : ''}`);
    button.type = 'button'; button.setAttribute('aria-pressed', String(item.run_id === state.run));
    button.append(node('strong', `${item.representation_id}${item.representation_id === summary.selection?.primary_representation ? ' · 主表示' : ''}`), node('span', representationName(item)), node('small', dimensions));
    button.addEventListener('click', () => selectRepresentation(item.run_id)); box.append(button);
    select.append(new Option(`${item.representation_id} · ${representationName(item)}`, item.run_id));
  }
  select.disabled = !items.length;
  select.value = state.run;
  $('clustering-final-rationale').textContent = summary?.selection?.selection_rationale || '当前没有最终主表示记录。';
}

function factList(rows) {
  const list = node('dl', '', 'clustering-facts');
  for (const [label, value] of rows) list.append(node('dt', label), node('dd', plain(value)));
  return list;
}

function stat(label, value, detail = '') {
  const item = node('div', '', 'clustering-stat');
  item.append(node('span', label), node('strong', plain(value)));
  if (detail) item.append(node('small', detail));
  return item;
}

function displayValue(value) {
  if (value == null) return '—';
  if (Array.isArray(value)) return value.length ? value.map(plain).join('、') : '无';
  if (typeof value === 'object') return Object.entries(value).map(([key, item]) => `${key}: ${displayValue(item)}`).join('；');
  if (typeof value === 'boolean') return value ? '是' : '否';
  return String(value);
}

function renderConfigGroup(parent, title, data, labels = {}) {
  if (!data || typeof data !== 'object') return;
  const rows = Object.entries(data).map(([key, value]) => [labels[key] || key.replaceAll('_', ' '), displayValue(value)]);
  if (!rows.length) return;
  const group = node('div', '', 'clustering-config-group');
  group.append(node('h4', title), factList(rows));
  parent.append(group);
}

function renderData() {
  const { run = {}, dataset = {}, config = {}, representation = {} } = state.overview;
  const currentCount = dataset.count ?? representation.document_count;
  $('clustering-data-count').textContent = `${integer(currentCount)} 篇文档`;
  $('clustering-data-stats').replaceChildren(
    stat('读取文档', integer(currentCount)),
    stat('词表大小', integer(representation.vocabulary_size)),
    stat('向量维度', integer(representation.dimension)),
    stat('非零特征', integer(representation.nnz)),
  );
  $('clustering-representation-facts').replaceChildren(factList([
    ['词表 / 原词空间', `${integer(representation.vocabulary_size)} / ${integer(representation.feature_dimension ?? representation.dimension)} 维`],
    ['实际聚类输入', `${integer(representation.dimension)} 维 · ${representation.space ?? 'tfidf_l2'}`],
    ['SVD 用途', representation.lsa ? `${integer(representation.dimension)}D LSA 为实际聚类输入；2D / 3D SVD 仅用于可视化` : '2D / 3D SVD 仅用于可视化'],
    ...(representation.lsa ? [['LSA 解释方差', `${number(representation.lsa.explained_variance_sum * 100, 2)}%`]] : []),
    ['预处理', 'NFKC → 英文小写 → HMM-1 Smooth 分词 → 最短词长过滤'],
  ]));
  renderEvolution();
  const configBox = $('clustering-config'); configBox.replaceChildren();
  renderConfigGroup(configBox, '文本预处理', config.preprocessing, { lowercase: '小写化', stop_words: '停用词', min_token_length: '最短词长', tokenizer: '分词器' });
  renderConfigGroup(configBox, '向量表示', config.representation, { method: '方法', max_features: '最大特征数', min_df: '最小文档频率', max_df: '最大文档频率', ngram_range: 'N-gram 范围', norm: '归一化' });
  renderConfigGroup(configBox, 'LSA · 实际聚类表示', config.lsa, { n_components: '潜在维度', n_iter: '迭代次数', random_state: '随机种子', algorithm: '算法' });
  renderConfigGroup(configBox, 'K-Means', config.kmeans, { init: '初始化', n_init: '重复初始化次数', max_iter: '最大迭代', tol: '收敛阈值', algorithm: '算法', random_state: '随机种子' });
  renderConfigGroup(configBox, '比较范围', { k_values: config.k_values, stability_seeds: config.stability_seeds }, { k_values: 'K 值', stability_seeds: '稳定性种子' });
  renderConfigGroup(configBox, '二维投影', config.projection, { method: '方法', n_components: '维度', n_iter: '迭代次数', random_state: '随机种子', purpose: '用途' });
  if (!configBox.children.length) configBox.textContent = '此 Run 未提供配置详情。';
  workbench.renderFlow();
  const source = $('clustering-provenance'); source.replaceChildren(factList([
    ['数据集', run.dataset], ['创建时间', run.created_at],
    ['输入字段', displayValue(dataset.input_fields)], ['输入内容 SHA-256', dataset.input_sha256],
    ['文档 ID SHA-256', dataset.ids_sha256], ['分词器', representation.tokenizer?.name],
    ['分词器 Run', representation.tokenizer?.run_id ?? config.tokenizer_run_id],
    ['文本 token 总数', integer(representation.token_count)], ['零向量文档', integer(representation.zero_vector_count)],
    ['稀疏度', representation.sparsity == null ? '—' : `${number(representation.sparsity * 100, 1)}%`],
    ['拟合次数', integer(run.fit_count)],
    ['词空间维度', integer(representation.feature_dimension ?? representation.vocabulary_size)],
    ['实际聚类空间', representation.space ?? 'tfidf_l2'],
    ['Top terms 来源', representation.top_terms_method ?? 'TF-IDF 簇中心权重'],
    ...(representation.lsa ? [['LSA 累计解释方差', `${number(representation.lsa.explained_variance_sum * 100, 2)}%`], ['词空间稀疏度', `${number(representation.word_space_sparsity * 100, 2)}%`]] : []),
  ]));
  showSection('data', 'clustering-data-content');
}

function renderClusters() {
  const result = state.result;
  const clusters = result.clusters || [];
  const total = clusters.reduce((sum, item) => sum + (Number(item.size) || 0), 0);
  $('clustering-role').textContent = result.role === 'known-topic-reference' ? `K = ${result.k} · 5主题参考配置` : `K = ${result.k} · 参数扫描`;
  $('clustering-cluster-summary').replaceChildren(
    stat('当前簇数 K', integer(result.k), result.role === 'known-topic-reference' ? '5主题参考配置，不代表最佳 K' : '参数扫描中的一个取值'),
    stat('已分配文档', integer(total)),
    stat('最小簇', integer(Math.min(...clusters.map((item) => item.size)))),
    stat('最大簇', integer(Math.max(...clusters.map((item) => item.size)))),
  );
  workbench.renderExplorer();
  showSection('clustering', 'clustering-clusters-content');
}

function makeTable(headers, rows, currentIndex = -1) {
  const table = node('table'); const head = node('thead'); const hr = node('tr');
  for (const label of headers) hr.append(node('th', label));
  head.append(hr); table.append(head);
  const body = node('tbody');
  rows.forEach((values, index) => {
    const row = node('tr', '', index === currentIndex ? 'is-current' : '');
    for (const value of values) row.append(node('td', plain(value)));
    body.append(row);
  });
  table.append(body);
  return table;
}

function renderVisualization() {
  const sweep = state.overview.sweep || [];
  updateProjectionCaption(); workbench.renderKContext(); workbench.renderCenterButtons();
  const filter = $('clustering-cluster-filter');
  const retained = filter.value;
  filter.replaceChildren(new Option('全部簇', 'all'));
  for (const cluster of state.result.clusters || []) filter.append(new Option(`簇 ${plain(cluster.cluster_id)}`, String(cluster.cluster_id)));
  filter.value = [...filter.options].some((option) => option.value === retained) ? retained : 'all';
  const rows = sweep.map((entry) => [integer(entry.k), number(entry.inertia, 1), number(entry.silhouette, 3), number(entry.calinski_harabasz, 1), number(entry.davies_bouldin, 3), `${integer(entry.min_cluster_size)}–${integer(entry.max_cluster_size)}`, number(entry.stability?.mean_ari, 3)]);
  $('clustering-sweep-table').replaceChildren(makeTable(['K', '惯性 ↓', '轮廓系数 ↑', 'CH ↑', 'DB ↓', '簇规模范围', '稳定性 ARI ↑'], rows, sweep.findIndex((entry) => Number(entry.k) === Number(state.k))));
  showSection('visualization', 'clustering-visual-content');
}

function metric(label, value, detail = '') {
  const item = node('div', '', 'clustering-metric');
  item.append(node('small', label), node('strong', value));
  if (detail) item.append(node('span', detail));
  return item;
}

function renderEvaluation() {
  const internal = state.result.internal || {};
  const external = state.result.external || {};
  $('clustering-internal-metrics').replaceChildren(
    metric('轮廓系数 ↑', number(internal.silhouette, 3), '簇内凝聚与簇间分离'),
    metric('惯性 ↓', number(internal.inertia, 1), '样本到所属中心的平方距离和'),
    metric('Calinski-Harabasz ↑', number(internal.calinski_harabasz, 1)),
    metric('Davies-Bouldin ↓', number(internal.davies_bouldin, 3)),
    metric('迭代次数', integer(internal.n_iter)),
    metric('seed stability ARI', number((state.overview.sweep || []).find((item) => item.k === state.k)?.stability?.mean_ari, 3), '不同种子分配之间的内部比较'),
  );
  renderComparison(); workbench.renderStructureComparison();
  $('clustering-external-metrics').replaceChildren(
    metric('ARI', number(external.ari, 3), '与已知类别的一致性'),
    metric('NMI', number(external.nmi, 3)),
    metric('V-measure', number(external.v_measure, 3)),
    metric('同质性', number(external.homogeneity, 3)),
    metric('完整性', number(external.completeness, 3)),
  );
  const categories = state.result.category_order || [];
  const clusters = state.result.clusters || [];
  const rows = clusters.map((cluster) => [`簇 ${plain(cluster.cluster_id)}`, ...categories.map((category) => integer(cluster.category_counts?.[category] ?? 0)), integer(cluster.size)]);
  $('clustering-matrix-table').replaceChildren(makeTable(['簇 / 类别', ...categories, '合计'], rows));
  const current = (state.overview.sweep || []).find((entry) => entry.k === state.result.k);
  const largest = [...clusters].sort((a, b) => b.size - a.size)[0];
  const variance = (state.overview.projection?.explained_variance_ratio || []).reduce((sum, value) => sum + value, 0);
  const interpretation = $('clustering-interpretation');
  interpretation.replaceChildren(
    node('p', `当前 K 的簇规模范围为 ${integer(current?.min_cluster_size)}–${integer(current?.max_cluster_size)}，最大/最小簇比为 ${number(current?.imbalance_ratio)}。请结合高权重词、代表文本和规模判断是否形成可解释的分组。`),
    node('p', `不同随机种子分配的稳定性 ARI：均值 ${number(current?.stability?.mean_ari, 3)}，最小值 ${number(current?.stability?.min_ari, 3)}。这是聚类之间的比较，不使用真实类别。`),
    node('p', largest ? `最大簇 ${largest.cluster_id} 的事后类别组成：${Object.entries(largest.category_counts || {}).map(([category, count]) => `${category} ${integer(count)} 篇`).join('、')}。类别组成仅帮助解释；簇编号不是类别编号，也不是分类准确率。` : '暂无簇组成。'),
    node('p', `二维投影累计解释方差为 ${number(variance * 100, 2)}%，图上重叠不能直接当作实际聚类空间的距离。内部指标与代表距离在 ${state.overview.representation?.space ?? 'tfidf_l2'} 空间计算；关键词来自词空间。本轮保留完整 K sweep；K=5 是已知主题参考，不自动选唯一最佳 K。`),
  );
  showSection('evaluation', 'clustering-evaluation-content');
}

function renderComparison() {
  const items = state.summary?.items || [];
  const rows = items.map((item) => [item.representation_id, representationName(item), integer(item.representation.dimension), number(item.internal.silhouette, 4), number(item.internal.calinski_harabasz, 3), number(item.internal.davies_bouldin, 3), number(item.stability.mean_ari, 4), number(item.external.ari, 4), number(item.external.nmi, 4), `${integer(item.min_cluster_size)}–${integer(item.max_cluster_size)}`]);
  $('clustering-comparison-table').replaceChildren(makeTable(['表示', '方法', '维度', '内部 silhouette', '内部 CH', '内部 DB', '内部 stability', '事后 ARI', '事后 NMI', '簇规模范围'], rows, items.findIndex((item) => item.run_id === state.run)));
  $('clustering-comparison-role').textContent = `K=${state.k}${state.k === 5 ? ' · 5主题参考配置' : ' · 参数扫描对照'}`;
  const box = $('clustering-optimization'); box.replaceChildren();
  if (state.summary?.selection) box.append(node('p', state.summary.selection.reference_rationale));
  for (const item of items) {
    const text = item.representation_id === 'R0' ? `原词空间高维且稀疏；部分 Top terms 出现“自己、一个、表示”等泛化词。` : item.representation_id === 'R1' ? 'DF filtering / bigram 改善部分关键词，并扩大了科技主题簇，但几何质量与稳定性未单调改善。' : 'LSA 将聚类空间压缩到 100 维并增强稳定性；仍存在消费电子子主题分裂与娱乐混合。';
    box.append(node('p', `${item.representation_id}：${text} 当前 K silhouette ${number(item.internal.silhouette, 4)}，seed stability ${number(item.stability.mean_ari, 4)}。`));
  }
  if (state.summary?.selection) {
    box.append(node('p', '这些优化观察来自固定 K=5 对照；其他 K 可在表格和完整 sweep 中查看。THUCNews 人工栏目与词汇主题结构并不完全一致，人物、媒体和互联网新闻与娱乐文本共享叙事词汇，大混合簇也揭示了这种交叉。'));
    box.append(node('p', state.summary.selection.limitations.join('；') + '。'));
  }
}

function showChart(id, option) {
  const element = $(id);
  if (!element || !element.clientWidth || !element.clientHeight) return;
  if (element.closest('[role="tabpanel"]')?.hidden || element.closest('.clustering-content')?.hidden) return;
  if (!window.echarts) { element.textContent = '图表资源不可用，请查看下方数据表。'; return; }
  let chart = charts.get(id);
  if (!chart) { chart = window.echarts.init(element); charts.set(id, chart); if (id === 'clustering-scatter-chart') chart.on('click', (event) => selectPoint(event.data)); }
  chart.setOption(option, true);
  chart.resize();
}

function scatterOption(groupByGold = false) {
  const projection = state.overview?.projection || {};
  const assignments = new Map((state.result?.assignments || []).map((item) => [String(item.id), item]));
  const filtered = $('clustering-cluster-filter').value;
  const groups = new Map();
  for (const point of projection.points || []) {
    const assignment = assignments.get(String(point.id));
    if (!assignment) continue;
    if (!groupByGold && filtered !== 'all' && String(assignment.cluster_id) !== filtered) continue;
    const label = groupByGold ? plain(assignment.gold_category) : `簇 ${plain(assignment.cluster_id)}`;
    if (!groups.has(label)) groups.set(label, []);
    const selected = assignments.get(String(state.selectedId));
    const isSelected = point.id === state.selectedId;
    groups.get(label).push({ value: [Number(point.x), Number(point.y)], kind: 'document', id: point.id, snippet: assignment.snippet, cluster: assignment.cluster_id, category: assignment.gold_category, symbolSize: isSelected ? 19 : 5, itemStyle: { opacity: !groupByGold && selected && selected.cluster_id !== assignment.cluster_id ? .23 : .9, borderColor: isSelected ? '#173b52' : 'transparent', borderWidth: isSelected ? 3 : 0, shadowBlur: isSelected ? 12 : 0, shadowColor: '#087f8c55' } });
  }
  const ordered = groupByGold ? state.result.category_order || [] : (state.result.clusters || []).map((item) => `簇 ${item.cluster_id}`);
  const labels = [...new Set([...ordered, ...groups.keys()])];
  const series = labels.map((label, index) => ({ name: label, type: 'scatter', data: groups.get(label) || [], symbolSize: 6, itemStyle: { color: colors[index % colors.length] }, large: false }));
  if (!groupByGold && $('clustering-show-centroids').checked) {
    const centers = state.inspection?.centroids || [];
    series.push({ name: '质心', type: 'scatter', symbol: 'diamond', symbolSize: (value,params) => params.data.assigned ? 32 : 22, z: 5, label: { show: true, formatter: (item) => `C${item.data.cluster}${item.data.assigned ? ' · assigned' : ''}`, position: 'top', color: '#173b52', fontSize: 12, fontWeight: 700, backgroundColor: '#ffffffdd', padding: [4,6], borderRadius: 3 }, data: centers.filter((center) => filtered === 'all' || String(center.cluster_id) === filtered).map((center) => ({ kind: 'centroid', cluster: center.cluster_id, value: center.projection2d, dimension: center.dimension, norm: center.norm, assigned: center.cluster_id === assignments.get(String(state.selectedId))?.cluster_id, itemStyle: { color: colors[center.cluster_id % colors.length], borderColor: '#173b52', borderWidth: center.cluster_id === assignments.get(String(state.selectedId))?.cluster_id ? 4 : 2 } })) });
    const selected = (projection.points || []).find((point) => point.id === state.selectedId);
    const assignment = assignments.get(String(state.selectedId));
    const center = centers.find((item) => item.cluster_id === assignment?.cluster_id);
    if (selected && center && (filtered === 'all' || String(assignment.cluster_id) === filtered)) series.push({ name: '显示投影连线', type: 'line', data: [[selected.x, selected.y], center.projection2d], showSymbol: false, silent: true, lineStyle: { color: '#536176', type: 'solid', width: 3, opacity: .8 }, z: 3 });
  }
  return {
    animation: false, color: colors, grid: { left: 48, right: 18, top: 30, bottom: 52 },
    legend: { type: 'scroll', top: 0, data: labels },
    tooltip: { trigger: 'item', confine: true, appendToBody: false, enterable: true,
      extraCssText: `width:${Math.max(80, Math.min(340, $(groupByGold ? 'clustering-gold-chart' : 'clustering-scatter-chart').clientWidth - 16))}px;white-space:normal;overflow-wrap:anywhere;line-height:1.65;box-sizing:border-box;`,
      position: (point, params, dom, rect, size) => {
        tooltipElements.set(groupByGold ? 'clustering-gold-chart' : 'clustering-scatter-chart', dom);
        dom.style.width = `${Math.max(80, Math.min(340, size.viewSize[0] - 16))}px`;
        dom.style.maxHeight = `${Math.max(80, size.viewSize[1] - 16)}px`;
        dom.style.overflowY = 'auto';
        return boundedTooltipPosition(point, [dom.offsetWidth, dom.offsetHeight], size.viewSize);
      },
      formatter: (item) => pointTooltipHTML(item.data, groupByGold) },
    xAxis: { type: 'value', name: 'SVD 1', nameLocation: 'middle', nameGap: 28, scale: true, splitLine: { lineStyle: { color: '#edf2f5' } } },
    yAxis: { type: 'value', name: 'SVD 2', scale: true, splitLine: { lineStyle: { color: '#edf2f5' } } },
    series,
  };
}

function pointTooltipHTML(d, gold = false) {
  if (d.kind === 'centroid') return `<strong>质心 C${htmlSafe(d.cluster)}</strong><div>${htmlSafe(d.dimension)}D 实际空间成员均值</div><div>范数 ${number(d.norm, 4)} · 点击检视</div>`;
  return `<strong>${htmlSafe(d.id)}</strong><div>${htmlSafe(d.snippet)}</div><div>簇 ${htmlSafe(d.cluster)}${gold ? ` · ${htmlSafe(d.category)}` : ''}</div>`;
}

function updateProjectionCaption() {
  const dimensions = Number($('clustering-projection-dimensions').value);
  const projection = dimensions === 3 ? state.projection3d : state.overview?.projection;
  const ratio = projection?.explained_variance_ratio || [];
  $('clustering-projection-title').textContent = `Document Space Explorer · ${dimensions === 3 ? '三维' : '二维'}投影`;
  $('clustering-projection-caption').textContent = projection ? `${projection.method} · ${dimensions}D 解释方差 ${ratio.map((item) => `${number(item * 100, 1)}%`).join(' / ')}` : '正在准备三维可视化坐标…';
  $('clustering-projection-note').textContent = `颜色代表聚类分配；${dimensions}D 只是低维投影视图，不能直接代表实际聚类空间的距离。文档到质心的连线也仅位于显示投影；实际分配依据下方 actual-space assignment 的高维距离。${dimensions === 3 ? '拖动旋转，滚轮缩放；键盘方向键旋转，+ / − 缩放。' : ''}`;
}

function drawScatter() {
  if ($('clustering-projection-dimensions').value === '2') {
    scatter3d?.dispose(); scatter3d = null;
    showChart('clustering-scatter-chart', scatterOption()); return;
  }
  const element = $('clustering-scatter-chart');
  if (!state.projection3d) { element.textContent = '正在读取三维可视化投影…'; return; }
  const old = charts.get('clustering-scatter-chart'); old?.dispose(); charts.delete('clustering-scatter-chart');
  if (!scatter3d) scatter3d = new Scatter3D(element, pointTooltipHTML, selectPoint);
  const assignments = new Map((state.result?.assignments || []).map((item) => [item.id, item]));
  const groups = (state.result.clusters || []).map((item, index) => ({ label: `簇 ${item.cluster_id}`, color: colors[index % colors.length] }));
  const points = state.projection3d.points.flatMap((point) => {
    const assignment = assignments.get(point.id); if (!assignment) return [];
    const index = (state.result.clusters || []).findIndex((item) => item.cluster_id === assignment.cluster_id);
    return [{ value: [point.x, point.y, point.z], id: point.id, snippet: assignment.snippet, cluster: assignment.cluster_id, color: colors[index % colors.length] }];
  });
  const centers = $('clustering-show-centroids').checked ? (state.inspection?.centroids || []).map((center) => ({ kind: 'centroid', cluster: center.cluster_id, value: center.projection3d, dimension: center.dimension, norm: center.norm, color: colors[center.cluster_id % colors.length] })) : [];
  scatter3d.setBounds(points);
  const filtered = $('clustering-cluster-filter').value;
  scatter3d.setData(points.filter((point) => filtered === 'all' || String(point.cluster) === filtered), groups, { centers: centers.filter((center) => filtered === 'all' || String(center.cluster) === filtered), selectedId: state.selectedId, selectedCluster: assignments.get(state.selectedId)?.cluster_id, selectedCenter: state.selectedCluster });
}

async function changeProjection() {
  const request = ++state.projectionRequest, run = state.run;
  updateProjectionCaption();
  if ($('clustering-projection-dimensions').value === '3' && !state.projection3d && state.overview) {
    // 销毁二维实例后再显示 loading，避免 ECharts 保留已移除的画布。
    charts.get('clustering-scatter-chart')?.dispose(); charts.delete('clustering-scatter-chart');
    drawScatter();
    try {
      const value = await api(`/api/v1/experiments/2/projection?dimensions=3&run_id=${encodeURIComponent(run)}`);
      if (request !== state.projectionRequest || run !== state.run) return;
      state.projection3d = value;
      if ($('page-message').textContent.startsWith('三维投影加载失败')) notice($('page-message'));
    } catch (error) {
      if (request !== state.projectionRequest || run !== state.run) return;
      $('clustering-projection-dimensions').value = '2';
      notice($('page-message'), `三维投影加载失败：${error.message}`, true);
    }
  }
  updateProjectionCaption(); afterRender();
}

function drawVisibleCharts() {
  if (!state.overview || !state.result) return;
  const active = document.querySelector('[data-tab][aria-selected="true"]')?.dataset.tab;
  if (active === 'visualization') {
    const sweep = state.overview.sweep || [];
    showChart('clustering-sweep-chart', {
      color: [colors[0], colors[2]], tooltip: { trigger: 'axis' }, legend: { top: 0, data: ['惯性', '轮廓系数'] },
      grid: { left: 55, right: 54, top: 38, bottom: 38 },
      xAxis: { type: 'category', name: 'K', data: sweep.map((item) => item.k) },
      yAxis: [{ type: 'value', name: '惯性', scale: true }, { type: 'value', name: '轮廓系数', scale: true }],
      series: [{ name: '惯性', type: 'line', data: sweep.map((item) => item.inertia), smooth: true }, { name: '轮廓系数', type: 'line', yAxisIndex: 1, data: sweep.map((item) => item.silhouette), smooth: true }],
    });
    const clusters = state.result.clusters || [];
    showChart('clustering-size-chart', {
      color: [colors[0]], tooltip: { trigger: 'axis' }, grid: { left: 55, right: 18, top: 26, bottom: 40 },
      xAxis: { type: 'category', data: clusters.map((item) => `簇 ${item.cluster_id}`) }, yAxis: { type: 'value', name: '文档' },
      series: [{ type: 'bar', barMaxWidth: 44, data: clusters.map((item) => item.size) }],
    });
    drawScatter();
  }
  if (active === 'evaluation') {
    for (const item of state.comparison?.items || []) {
      const points = item.projection?.points || [];
      showChart(`clustering-mini-${item.representation_id}`, { animation: false, grid: { left: 10, right: 10, top: 10, bottom: 10 }, xAxis: { type: 'value', show: false, scale: true }, yAxis: { type: 'value', show: false, scale: true }, series: item.clusters.map(({cluster_id: cluster}) => ({ type: 'scatter', symbolSize: 4, itemStyle: { color: colors[cluster % colors.length], opacity: .7 }, data: points.filter((point) => point.cluster_id === cluster).map((point) => [point.x, point.y]) })).concat([{ type: 'scatter', symbol: 'diamond', symbolSize: 11, itemStyle: { color: '#173b52' }, data: item.projection.centroids.map((center) => center.projection2d) }]) });
    }
    const categories = state.result.category_order || [];
    const clusters = state.result.clusters || [];
    const data = clusters.flatMap((cluster, row) => categories.map((category, column) => [column, row, Number(cluster.category_counts?.[category] || 0)]));
    showChart('clustering-matrix-chart', {
      tooltip: { formatter: (item) => `${htmlSafe(categories[item.data[0]])} · 簇 ${htmlSafe(clusters[item.data[1]]?.cluster_id)}：${integer(item.data[2])} 篇` },
      grid: { left: 65, right: 25, top: 18, bottom: 70 },
      xAxis: { type: 'category', data: categories, axisLabel: { interval: 0, rotate: categories.length > 5 ? 35 : 0 } },
      yAxis: { type: 'category', data: clusters.map((item) => `簇 ${item.cluster_id}`) },
      visualMap: { min: 0, max: Math.max(1, ...data.map((item) => item[2])), orient: 'horizontal', left: 'center', bottom: 0, calculable: false, inRange: { color: ['#eef8f9', '#087f8c'] } },
      series: [{ type: 'heatmap', data, label: { show: true }, emphasis: { itemStyle: { borderColor: '#173b52', borderWidth: 1 } } }],
    });
    showChart('clustering-gold-chart', scatterOption(true));
  }
}

function afterRender() { requestAnimationFrame(drawVisibleCharts); }

async function loadResult() {
  const request = ++state.request;
  const k = Number($('clustering-k').value);
  if (!state.run || !Number.isInteger(k)) return;
  state.k = k; state.result = null; state.document = null; state.inspection = null; ++state.docRequest; clearCharts();
  for (const key of ['clustering', 'visualization', 'evaluation']) {
    const content = $(`clustering-${{ clustering: 'clusters', visualization: 'visual', evaluation: 'evaluation' }[key]}-content`);
    content.hidden = true;
    let box = $(key).querySelector('.clustering-state');
    if (!box) { box = node('div', '', 'clustering-state'); $(key).prepend(box); }
    box.replaceChildren(node('h2', '正在读取聚类结果…'), node('p', `加载 K = ${k} 的簇与评价指标。`)); box.hidden = false;
  }
  try {
    const [result, summary, inspection] = await Promise.all([
      api(`/api/v1/experiments/2/result?k=${encodeURIComponent(k)}&run_id=${encodeURIComponent(state.run)}`),
      api(`/api/v1/experiments/2/comparison-view?k=${encodeURIComponent(k)}`),
      api(`/api/v1/experiments/2/inspection?k=${k}&run_id=${encodeURIComponent(state.run)}`),
    ]);
    if (request !== state.request) return;
    state.result = result; state.summary = summary; state.comparison = summary; state.inspection = inspection;
    $('clustering-cluster-filter').value = 'all';
    state.selectedId = result.assignments.some((item) => item.id === state.selectedId) ? state.selectedId : state.overview.representation?.example?.id || result.assignments[0]?.id;
    state.selectedCluster = result.assignments.find((item) => item.id === state.selectedId)?.cluster_id;
    workbench.renderOptions();
    renderEvolution(); renderClusters(); renderVisualization(); renderEvaluation(); afterRender();
    notice($('page-message'));
    await selectDocument(state.selectedId);
    if (request !== state.request) return;
    if ($('clustering-projection-dimensions').value === '3' && !state.projection3d) changeProjection();
  } catch (error) {
    if (request !== state.request) return;
    notice($('page-message'), `聚类结果加载失败：${error.message}`, true);
    for (const key of ['clustering', 'visualization', 'evaluation']) {
      const box = $(key).querySelector('.clustering-state');
      box.className = 'clustering-state failure';
      box.replaceChildren(node('h2', '无法读取当前 K 的结果'), node('p', error.message));
    }
  }
}

async function loadOverview() {
  const request = ++state.request;
  state.run = $('clustering-run').value; state.overview = null; state.result = null; state.projection3d = null; state.document = null; ++state.docRequest; ++state.projectionRequest; clearCharts();
  notice($('page-message'));
  $('clustering-k').disabled = true;
  setSectionState('正在读取实验数据…', '加载数据来源、表示方法与参数扫描。');
  try {
    const overview = await api(`/api/v1/experiments/2/overview?run_id=${encodeURIComponent(state.run)}`);
    if (request !== state.request) return;
    state.overview = overview;
    renderData();
    const values = (overview.sweep || []).map((item) => Number(item.k)).filter(Number.isInteger);
    const select = $('clustering-k'); select.replaceChildren();
    for (const value of values) select.append(new Option(`K = ${value}${value === Number(overview.run?.reference_k) ? ' · 5主题参考配置' : ''}`, String(value)));
    if (!values.length) {
      select.append(new Option('暂无可用 K', ''));
      $('clustering-footer-run').textContent = `Run ${plain(overview.run?.run_id)} · ${plain(overview.run?.dataset)}`;
      for (const key of ['clustering', 'visualization', 'evaluation']) {
        const box = $(key).querySelector('.clustering-state');
        box.replaceChildren(node('h2', '尚无参数扫描结果'), node('p', '数据与配置已加载；此 Run 没有可选择的聚类结果。'));
      }
      return;
    }
    select.value = String(values.includes(state.k) ? state.k : values.includes(5) ? 5 : values[0]);
    select.disabled = false;
    $('clustering-footer-run').textContent = `Run ${plain(overview.run?.run_id)} · ${plain(overview.run?.dataset)}`;
    await loadResult();
  } catch (error) {
    if (request !== state.request) return;
    setSectionState('实验数据加载失败', error.message, 'failure');
    notice($('page-message'), `实验数据加载失败：${error.message}`, true);
  }
}

async function loadRuns() {
  notice($('page-message'));
  setSectionState('正在读取实验记录…', '检查可用的文本聚类 Run。');
  try {
    const [value, summary] = await Promise.all([api('/api/v1/experiments/2/runs'), api('/api/v1/experiments/2/summary')]);
    state.summary = summary;
    const runs = (value.items || []).filter((item) => item?.run_id);
    const select = $('clustering-run'); select.replaceChildren();
    if (!runs.length) {
      select.append(new Option('尚无完成 Run', ''));
      $('clustering-k').replaceChildren(new Option('等待 Run', ''));
      setSectionState('尚无完成 Run', '数据准备：请先生成实验二的聚类 Run；完成后这里会展示真实文档、参数与评价结果。');
      $('clustering-footer-run').textContent = '尚无完成 Run';
      notice($('page-message'));
      return;
    }
    for (const run of runs) select.append(new Option(`${run.representation_id ? `${plain(run.representation_id)} · ` : ''}${plain(run.run_id)} · ${plain(run.dataset)}${run.created_at ? ` · ${plain(run.created_at)}` : ''}`, String(run.run_id)));
    const recommended = value.primary ?? value.latest;
    select.value = runs.some((run) => String(run.run_id) === String(recommended)) ? String(recommended) : String(runs[0].run_id);
    select.disabled = false;
    await loadOverview();
  } catch (error) {
    setSectionState('实验记录加载失败', error.message, 'failure');
    notice($('page-message'), `实验记录加载失败：${error.message}`, true);
  }
}

$('clustering-run').addEventListener('change', loadOverview);
$('clustering-k').addEventListener('change', loadResult);
$('clustering-cluster-filter').addEventListener('change', (event) => {
  if (!state.result) return;
  const cluster = event.target.value;
  const selected = state.result.assignments.find((item) => item.id === state.selectedId);
  if (cluster !== 'all' && String(selected?.cluster_id) !== cluster) selectCluster(Number(cluster));
  else drawScatter();
});
$('clustering-projection-dimensions').addEventListener('change', changeProjection);
$('clustering-representation').addEventListener('change', (event) => selectRepresentation(event.target.value));
$('clustering-show-centroids').addEventListener('change', () => { if (state.result) drawScatter(); });
document.addEventListener('toggle', afterRender, true);
document.addEventListener('module-tab', afterRender);
window.addEventListener('resize', () => {
  for (const [id, chart] of charts) {
    chart.dispatchAction({ type: 'hideTip' });
    const tip = tooltipElements.get(id);
    if (tip) { tip.style.transform = 'translate3d(0,0,0)'; tip.style.left = '0px'; tip.style.top = '0px'; tip.style.width = `${Math.max(80, Math.min(340, $(id).clientWidth - 16))}px`; }
    if (!$(id).closest('[role="tabpanel"]')?.hidden) chart.resize();
  }
});
loadRuns();
