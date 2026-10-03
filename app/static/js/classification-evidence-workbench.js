// Experiment 3: inspect saved predictions and representations without fitting models.
import {MODEL_COLORS} from './classification-model-colors.js';
const array = value => Array.isArray(value) ? value : [];
const finite = value => value !== null && value !== undefined && Number.isFinite(Number(value));
const value = (item, fallback = '—') => item === null || item === undefined ? fallback : item;
const textOf = item => String(value(item, ''));

function helpers(ui) {
  const escape = ui?.escape || (item => textOf(item).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char])));
  return {
    escape,
    plainLabel: item => ui?.label ? ui.label(item) : item,
    label: item => escape(ui?.label ? ui.label(item) : item),
    number: item => finite(item) ? escape(ui?.number ? ui.number(item) : Number(item).toFixed(4)) : '—',
    pct: item => finite(item) ? escape(ui?.pct ? ui.pct(item) : `${(Number(item) * 100).toFixed(2)}%`) : '—',
    integer: item => finite(item) ? escape(Number(item).toLocaleString()) : '—',
    table: (headers, rows) => ui?.table ? ui.table(headers, rows) : `<table><thead><tr>${headers.map(header => `<th>${escape(header)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table>`,
  };
}

function detailsTable(h, headers, rows) {
  return `<div class="cew-table-wrap">${h.table(headers, rows)}</div>`;
}

function scoreBars(result, h) {
  const scores = array(result.scores);
  if (!scores.length) return '<p class="cew-empty">此记录没有五类分数。</p>';
  const probability = result.score_type === 'posterior' || result.score_type === 'softmax' || scores.some(score => finite(score.raw_logit));
  const maxAbs = Math.max(1e-12, ...scores.map(score => Math.abs(Number(score.value) || 0)));
  return `<div class="cew-score-list" role="list" aria-label="五类${probability ? '模型输出' : '决策分数'}">${scores.map((score, index) => {
    const numeric = Number(score.value) || 0;
    const width = probability ? Math.min(100, Math.max(0, numeric * 100)) : Math.min(50, Math.abs(numeric) / maxAbs * 50);
    const style = probability ? `width:${width}%` : `left:${numeric < 0 ? 50 - width : 50}%;width:${width}%`;
    return `<button type="button" class="cew-score-row${score.label === result.prediction ? ' is-predicted' : ''}" data-cew-score="${index}" role="listitem"><span class="cew-score-label">${h.label(score.label)}</span><span class="cew-score-track${probability ? '' : ' is-decision'}"><span class="cew-score-fill" style="${style}"></span></span><strong>${probability ? h.pct(score.value) : h.number(score.value)}</strong></button>`;
  }).join('')}</div><p class="cew-note">${probability ? result.score_type === 'posterior' ? 'MultinomialNB 归一化后验，未经校准；并非经验正确率。' : 'Softmax 输出未经校准；并非经验正确率。' : 'LinearSVC 决策分数可为负值，没有原生概率。条形以零为中线，长度按本篇最大绝对分数缩放。'}</p>`;
}

function featureRows(result, kind) {
  return array(kind === 'relative' ? result.top_relative_features : result.top_contributing_features);
}

function spanSource(text, span, document) {
  // A window bridging head and tail has no single contiguous source interval.
  if (span.crosses_truncation_join) return null;
  const offsets = span.character_offsets;
  let start, end;
  if (Array.isArray(offsets)) {
    if (offsets.length === 2 && offsets.every(finite)) [start, end] = offsets.map(Number);
    else {
      const ranges = offsets.map(item => Array.isArray(item) ? {start:item[0], end:item[1]} : item);
      if (ranges.length && ranges.every(item => finite(item?.start) && finite(item?.end))) {
        start = Math.min(...ranges.map(item => Number(item.start)));
        end = Math.max(...ranges.map(item => Number(item.end)));
      }
    }
  } else if (offsets && typeof offsets === 'object') {
    start = offsets.start;
    end = offsets.end;
  }
  if (finite(start) && finite(end) && start >= 0 && end <= text.length && start < end) {
    return {start:Number(start), end:Number(end), method:'offset'};
  }
  const tokens = array(document?.tokens);
  const positions = array(span.original_token_positions);
  if (positions.length && tokens.length) {
    const selected = positions.map(position => tokens[position]);
    const starts = selected.map(token => token?.start ?? token?.start_offset);
    const ends = selected.map(token => token?.end ?? token?.end_offset);
    const contiguous = positions.every((position, index) => index === 0 || position === positions[index - 1] + 1);
    if (contiguous && starts.every(finite) && ends.every(finite)) {
      const start = Math.min(...starts.map(Number)), end = Math.max(...ends.map(Number));
      if (start >= 0 && end <= text.length && start < end) return {start, end, method:'offset'};
    }
  }
  // A sequence span without offsets can only be located by an occurrence in source text.
  const original = array(span.token_span).join('');
  const at = original ? text.indexOf(original) : -1;
  return at >= 0 ? {start:at, end:at + original.length, method:'occurrence'} : null;
}

function sourceWithSpans(source, spans, document, h, selectedIndex=0) {
  const text = textOf(source);
  if (!text) return '<p class="cew-empty">此记录没有保存原文。</p>';
  const candidates = spans.map((span, index) => ({...spanSource(text, span, document), index})).filter(item => finite(item.start) && finite(item.end)).sort((a,b)=>(a.index===selectedIndex?-1:b.index===selectedIndex?1:a.index-b.index));
  const chosen=[];
  for(const item of candidates){if(!chosen.some(other=>item.start<other.end&&other.start<item.end)&&chosen.length<4)chosen.push(item);}
  const located=chosen.sort((a,b)=>a.start-b.start);
  let cursor = 0, content = '';
  for (const item of located) {
    if (item.start < cursor) continue;
    content += h.escape(text.slice(cursor, item.start));
    content += `<button type="button" class="cew-source-span" data-cew-span="${item.index}" title="查看卷积片段 ${item.index + 1}">${h.escape(text.slice(item.start, item.end))}</button>`;
    cursor = item.end;
  }
  content += h.escape(text.slice(cursor));
  const fallback = located.some(item => item.method === 'occurrence');
  return `<div class="cew-source-text">${content}</div><p class="cew-note">${located.length ? fallback ? '高亮按原词片段在原文中的首次连续出现定位；没有精确字符偏移时，重复片段的位置可能不对应。' : '高亮按保存的字符偏移定位。' : '没有可用于原文连续定位的字符偏移；跨 head-tail 拼接窗口不会被伪装成连续原文。下方仍保留原始 token 位置与片段。'}</p>`;
}

function featureInspector(result, index, kind, h) {
  const row = featureRows(result, kind)[index];
  if (!row) return '<p class="cew-empty">没有该特征记录。</p>';
  const isRelative = kind === 'relative';
  const fields = [
    ['特征', row.term], ['TF-IDF', h.number(row.tfidf)],
    [isRelative ? '预测类 − 第二名系数' : '预测类权重', h.number(isRelative ? row.coefficient_difference : row.weight)],
    [isRelative ? '相对贡献' : result.score_type === 'posterior' ? '相对第二名贡献' : '直接贡献', h.number(row.contribution)],
  ];
  if (finite(row.direct_contribution)) fields.push(['对预测类的直接贡献', h.number(row.direct_contribution)]);
  if (finite(row.runner_up_contrast)) fields.push(['对第二名的相对贡献', h.number(row.runner_up_contrast)]);
  return inspectorObject(result.model_id, '当前特征', row.term, [
    [isRelative || result.score_type === 'posterior' ? '相对第二名贡献' : '直接贡献', h.number(row.contribution)],
    ['TF-IDF', h.number(row.tfidf)],
  ], `<p class="wb-inspector-note">${isRelative || result.score_type === 'posterior' ? '正值支持预测类别相对第二名；负值支持第二名。' : 'TF-IDF × 预测类权重，直接改变该类决策分数。'}</p>`, detailsTable(h, ['量', '保存值'], fields), h);
}

function spanInspector(span, h, modelId='C3') {
  if (!span) return '<p class="cew-empty">没有该卷积片段记录。</p>';
  const rows = [
    ['卷积核 / filter', `${value(span.kernel_size)} / ${value(span.filter_id)}`],
    ['通道', value(span.channel)], ['最大响应位置', value(span.argmax_position)],
    ['原始 token 位置', array(span.original_token_positions).join(', ') || '—'],
    ['原词片段', array(span.token_span).join(' ') || '—'],
    ['模型实际 token', array(span.model_token_span).join(' ') || '—'],
    ['activation', h.number(span.max_activation)],
    ['预测类权重', h.number(span.predicted_class_weight)],
    ['直接贡献', h.number(span.contribution)],
    ['跨 head-tail 拼接', span.crosses_truncation_join ? '是' : '否'],
  ];
  return inspectorObject(modelId, '卷积最大响应片段', array(span.token_span).join(' '), [
    ['直接贡献', h.number(span.contribution)], ['最大响应', h.number(span.max_activation)],
  ], `<p class="wb-inspector-caption">kernel ${h.escape(span.kernel_size)} · filter ${h.escape(span.filter_id)} · 位置 ${h.escape(value(span.argmax_position))}</p><div class="wb-token-comparison"><span>原始词片段</span><p>${h.escape(array(span.token_span).join(' '))}</p><span>模型实际 token</span><p>${array(span.model_token_span).map(token=>`<span class="${token==='<UNK>'?'wb-unk':''}">${h.escape(token)}</span>`).join(' ')}</p></div><p class="wb-inspector-note">最大响应 × 预测类权重。${span.crosses_truncation_join?'此窗口跨 head-tail 拼接，不能高亮为连续原文。':'原词与模型 token 分开显示，词表外词为 &lt;UNK&gt;。'}不是 attention 或因果解释。</p>`, detailsTable(h, ['量', '保存值'], rows), h);
}

function inspectorObject(modelId, kind, title, metrics, explanation, complete, h) {
  return `<div class="wb-inspector-heading"><span class="wb-inspector-kicker"><i style="background:${MODEL_COLORS[modelId] || '#148c8a'}"></i>${h.escape(modelId || '')} · ${h.escape(kind)}</span><h3 class="wb-object-title">${h.escape(title)}</h3></div><div class="wb-inspector-metrics">${metrics.map(([label,number])=>`<div><span>${h.escape(label)}</span><strong>${number}</strong></div>`).join('')}</div>${explanation}<details class="wb-inspector-details"><summary>完整保存值与解释</summary>${complete}</details>`;
}

export function renderPrediction({element, inspector, result = {}, text, ui, onEvidenceSelect, onViewErrors, provenance=''}) {
  if (!element || !inspector) return;
  const h = helpers(ui);
  const modelId = result.model_id || (result.sequence_diagnostic || array(result.scores).some(score => finite(score.raw_logit)) ? 'C3' : result.score_type === 'posterior' ? 'C0' : 'C1');
  const neural = modelId === 'C3';
  const spans = array(result.top_convolution_spans);
  const document = result.document || {};
  const source = text ?? result.text ?? result.text_snippet;
  const firstFeatures = featureRows(result, 'direct');
  const relativeFeatures = featureRows(result, 'relative');
  const status = result.correct === undefined ? '' : `<span class="cew-outcome">${result.correct ? '预测正确' : '预测错误'}</span>`;
  const scoreDetail = neural ? detailsTable(h, ['类别', 'Softmax 输出', 'raw logit'], array(result.scores).map(score => [ui?.label?.(score.label) ?? score.label, h.pct(score.value), h.number(score.raw_logit)])) : result.score_type === 'posterior' ? detailsTable(h, ['类别', '后验', 'joint log-score'], array(result.scores).map(score => [ui?.label?.(score.label) ?? score.label, h.pct(score.value), h.number(score.joint_log_score)])) : '';
  const sequence = result.sequence_diagnostic || {};
  const evidence = neural ? `<h3>卷积最大响应片段 <small>Top convolution spans</small></h3><div data-cew-source>${sourceWithSpans(source, spans, document, h)}</div><div class="cew-evidence-list">${spans.map((span, index) => `<button type="button" class="cew-evidence-card" data-cew-span="${index}"><span>${h.escape(array(span.token_span).join(' ') || array(span.model_token_span).join(' '))}</span><strong class="cew-contribution">贡献 ${h.number(span.contribution)}</strong><small>k=${h.escape(value(span.kernel_size))} · filter ${h.escape(value(span.filter_id))}</small></button>`).join('') || '<p class="cew-empty">没有保存 Top 卷积响应片段。</p>'}</div><details><summary>完整 Top 片段数值表</summary>${detailsTable(h, ['原词片段', '模型 token', 'kernel / filter', 'activation', '权重', '贡献'], spans.map(span => [array(span.token_span).join(' '), array(span.model_token_span).join(' '), `${value(span.kernel_size)} / ${value(span.filter_id)}`, h.number(span.max_activation), h.number(span.predicted_class_weight), h.number(span.contribution)]))}</details><p class="cew-note">全部 384 通道贡献和 ${h.number(result.channel_contribution_sum)} + bias ${h.number(result.bias)} = raw logit ${h.number(result.reconstructed_logit)}。这里只展示正贡献 Top 片段，不是 attention 或因果解释。</p>` : `<h3>预测类别的 Top 判别特征</h3><p class="cew-note">${modelId === 'C0' ? 'C0 的贡献相对第二名 joint log-score；正值支持预测类。' : '直接贡献为 TF-IDF × 预测类权重；正值提高该类决策分数。'}</p><div class="cew-evidence-list">${firstFeatures.map((feature, index) => `<button type="button" class="cew-evidence-card" data-cew-feature="${index}"><span>${h.escape(feature.term)}</span><strong class="cew-contribution">贡献 ${h.number(feature.contribution)}</strong></button>`).join('') || '<p class="cew-empty">此记录没有保存 Top 特征。</p>'}</div>${relativeFeatures.length ? `<h3>相对第二名的 Top 特征</h3><div class="cew-evidence-list">${relativeFeatures.map((feature, index) => `<button type="button" class="cew-evidence-card" data-cew-relative="${index}"><span>${h.escape(feature.term)}</span><strong class="cew-contribution">相对贡献 ${h.number(feature.contribution)}</strong></button>`).join('')}</div>` : ''}<details><summary>完整 Top 特征数值表</summary>${detailsTable(h, ['特征', 'TF-IDF', '预测类权重', modelId === 'C0' ? '相对贡献' : '直接贡献'], firstFeatures.map(feature => [feature.term, h.number(feature.tfidf), h.number(feature.weight), h.number(feature.contribution)]))}${relativeFeatures.length ? detailsTable(h, ['特征', 'TF-IDF', '系数差', '相对贡献'], relativeFeatures.map(feature => [feature.term, h.number(feature.tfidf), h.number(feature.coefficient_difference), h.number(feature.contribution)])) : ''}</details><p class="cew-note">${modelId === 'C0' ? `全部直接贡献和 ${h.number(result.feature_sum)} + 截距 ${h.number(result.intercept)} = joint log-score ${h.number(result.reconstructed_score)}；相对第二名截距差 ${h.number(result.contrast_intercept)} + 全部相对贡献 ${h.number(result.contrast_feature_sum)} = gap ${h.number(result.score_gap)}。` : `全部直接贡献和 ${h.number(result.feature_sum)} + 截距 ${h.number(result.intercept)} = 决策分数 ${h.number(result.reconstructed_score)}。`}</p>`;
  element.innerHTML = `<div class="cew-prediction-head"><div><span class="cew-eyebrow">${h.escape(modelId)} · ${neural ? 'Word TextCNN' : modelId === 'C0' ? 'MultinomialNB' : 'LinearSVC'}</span><h2>${h.label(result.prediction)}</h2><p>第二名 ${h.label(result.runner_up)} · ${neural ? 'Softmax 输出' : result.score_type === 'posterior' ? '归一化后验' : '决策分数'} ${finite(result.model_score) ? neural || result.score_type === 'posterior' ? h.pct(result.model_score) : h.number(result.model_score) : '—'}</p></div>${status}</div>${scoreBars(result, h)}${scoreDetail ? `<details><summary>查看五类原始分数</summary>${scoreDetail}</details>` : ''}<div class="cew-prediction-meta">${neural ? `原长 ${h.integer(sequence.original_length)} tokens · 保留 ${h.integer(sequence.retained_length)} · 保留段 UNK ${h.integer(sequence.retained_unk_count)}${sequence.truncated ? ' · head-tail 截断' : ''}` : `${result.feature_kind === 'character_bigram' ? 'char bigram' : '词'}候选 ${h.integer(result.token_count)} · 非零特征 ${h.integer(result.known_feature_count)}${result.zero_vector ? ' · 零向量' : ''}`}</div>${result.notice ? `<p class="cew-note">${h.escape(result.notice)}</p>` : ''}${evidence}${typeof onViewErrors === 'function' ? '<button type="button" class="cew-link-button" data-cew-errors>查看错误样本 →</button>' : ''}`;
  element.style.setProperty('--model-color', MODEL_COLORS[modelId] || '#148c8a');
  const markSelection = button => {
    const key = ['cewSpan','cewFeature','cewRelative','cewScore'].find(key=>button.dataset[key] !== undefined);
    element.querySelectorAll('[data-cew-span],[data-cew-feature],[data-cew-relative],[data-cew-score]').forEach(node=>{
      const selected=key&&node.dataset[key]===button.dataset[key];
      node.classList.toggle('is-selected',Boolean(selected));node.setAttribute('aria-pressed',String(Boolean(selected)));
    });
  };
  const show = (html, payload) => { inspector.innerHTML = html + provenance; if (typeof onEvidenceSelect === 'function') onEvidenceSelect(payload); };
  element.onclick = event => {
    const button = event.target.closest('button[data-cew-score], button[data-cew-feature], button[data-cew-relative], button[data-cew-span], button[data-cew-errors]');
    if (!button || !element.contains(button)) return;
    if (button.hasAttribute('data-cew-errors')) { onViewErrors?.(); return; }
    markSelection(button);
    if (button.hasAttribute('data-cew-score')) {
      const item = array(result.scores)[Number(button.dataset.cewScore)];
      if (item) show(inspectorObject(modelId, '类别输出', h.plainLabel(item.label), [[neural ? 'Softmax 输出' : result.score_type==='posterior' ? 'posterior' : 'decision score', neural || result.score_type==='posterior' ? h.pct(item.value) : h.number(item.value)]], '<p class="wb-inspector-note">'+(neural ? 'Softmax 未经校准。' : result.score_type==='posterior' ? 'NB 归一化后验，未经校准。' : 'SVM 决策分数不是概率。')+'</p>', detailsTable(h, ['量','保存值'], [...(finite(item.raw_logit) ? [['raw logit',h.number(item.raw_logit)]] : []),...(finite(item.joint_log_score)?[['joint log-score',h.number(item.joint_log_score)]]:[])]), h), {type:'score', item});
      return;
    }
    if (button.hasAttribute('data-cew-span')) {
      const index=Number(button.dataset.cewSpan),item=spans[index],sourceView=element.querySelector('[data-cew-source]');
      if(sourceView){sourceView.innerHTML=sourceWithSpans(source,spans,document,h,index);markSelection(button);const active=sourceView.querySelector('.is-selected'),box=sourceView.querySelector('.cew-source-text');if(active&&box)box.scrollTop+=active.getBoundingClientRect().top-box.getBoundingClientRect().top-30;}
      show(spanInspector(item,h),{type:'span',item,index});return;
    }
    const kind = button.hasAttribute('data-cew-relative') ? 'relative' : 'direct';
    const index = Number(button.dataset.cewRelative ?? button.dataset.cewFeature);
    const item = featureRows(result, kind)[index];
    show(featureInspector(result, index, kind, h), {type:'feature', kind, item, index});
  };
  const first=element.querySelector(neural?'button.cew-evidence-card[data-cew-span]':'button[data-cew-feature]') || element.querySelector('[data-cew-score]');
  if(first)first.click();else inspector.innerHTML='<p class="cew-empty">本篇没有保存可展示的证据。</p>';
}

function pooledView(pooled, h) {
  const values = array(pooled).map(item => Number(typeof item === 'object' ? item?.max_activation ?? item?.value : item));
  if (values.length !== 384 || values.some(item => !Number.isFinite(item))) {
    return '<p class="cew-note">当前记录没有提供完整 384 通道 pooled 数值；这里只能说明网络形状。</p>';
  }
  const peak = Math.max(...values, 0);
  const strips = [3, 4, 5].map((kernel, group) => {
    const band = values.slice(group * 128, (group + 1) * 128);
    return `<div class="cew-pool-group"><strong>kernel ${kernel} · 128 filters</strong><div class="cew-pool-strip" role="img" aria-label="kernel ${kernel} 的 128 个真实 pooled activation">${band.map((activation, index) => `<span class="cew-pool-cell" title="channel ${group * 128 + index}: ${h.number(activation)}"><i style="height:${peak ? Math.max(0, Math.round(activation / peak * 100)) : 0}%"></i></span>`).join('')}</div></div>`;
  }).join('');
  return `<p>保存的本篇 384 通道 max pool activation。颜色高度按本篇最大值 ${h.number(peak)} 缩放；灰色表示零响应。</p><div class="cew-pool-groups">${strips}</div><p class="cew-note">这些是实际保存的 pooled 数值。悬停可查看通道与 activation；不是 attention 或因果解释。</p>`;
}

function representationStages(modelId, overview, representation, record, document, h) {
  const neural = modelId === 'C3';
  const character = modelId === 'C2';
  const stage = neural ? overview?.stage3 : character ? overview?.stage2 : overview;
  const global = stage?.representation || {};
  const diagnostic = representation?.sequence_diagnostic || record?.sequence_diagnostic || document?.sequence_diagnostic || {};
  const vocab = neural ? global?.final_train_val?.vocabulary_size ?? global?.vocabulary_size : character ? global?.final?.tfidf?.shape?.[1] : global?.final?.train_val?.shape?.[1];
  const maxLen = global?.max_len ?? stage?.config?.max_len;
  const tokens = array(representation?.tokens ?? document?.tokens);
  const tokenIds = array(representation?.token_ids ?? document?.token_ids);
  const modelTokens = array(representation?.model_tokens ?? document?.model_tokens);
  const membership = array(representation?.feature_membership ?? document?.feature_membership);
  const fullFeatures = array(representation?.top_tfidf_features ?? document?.top_tfidf_features ?? representation?.nonzero_features ?? document?.nonzero_features);
  const matrixBacked = Boolean(representation?.matrix_source ?? document?.matrix_source);
  const savedFeatures = fullFeatures.length ? fullFeatures : array(record?.top_contributing_features);
  const featureRows = savedFeatures.filter(item => finite(item.tfidf ?? item.value)).sort((a, b) => Number(b.tfidf ?? b.value) - Number(a.tfidf ?? a.value)).slice(0, 12);
  const pooled = representation?.pooled ?? document?.pooled;
  const source = textOf(document?.text ?? record?.text ?? document?.text_snippet ?? record?.text_snippet);
  const preprocessing = stage?.config?.preprocessing ?? overview?.config?.preprocessing ?? {};
  const form = preprocessing.unicode_normalization;
  const lower = preprocessing.lowercase;
  const normalized = source ? (form === 'NFC' || form === 'NFKC' ? source.normalize(form) : source) : '';
  const displayed = lower === true ? normalized.toLowerCase() : normalized;
  const steps = [];
  const add = (title, tag, summary, body) => steps.push({title, kind:tag, tag:`${String(steps.length + 1).padStart(2, '0')} · ${tag}`, summary, body});

  add('输入文本', 'SOURCE', '输入为当前样本原文。', source ? `<p class="cew-stage-text">${h.escape(source)}</p>` : '<p class="cew-empty">此记录没有提供全文。</p>');
  add('Unicode 与大小写归一化', 'NORMALIZE', `${form || '配置未提供'}${lower === true ? ' + lowercase' : ''}，这里只显示文本变换。`, `<p>配置：Unicode ${h.escape(form ?? '未提供')} · lowercase ${lower === undefined ? '未提供' : lower ? '启用' : '关闭'}</p>${source ? `<p class="cew-stage-text">${h.escape(displayed)}</p><p class="cew-note">由浏览器按配置展示归一化文本；未调用分词器、向量器或模型 fit。</p>` : '<p class="cew-note">没有原文，无法展示归一化后的文本。</p>'}`);
  const preview = character ? membership.map(item => item.term) : tokens;
  const tokenPreview = preview.length ? `<p class="cew-token-stream">${preview.slice(0, 120).map(token => `<span>${h.escape(typeof token === 'string' ? token : token?.text ?? token?.token ?? '')}</span>`).join('')}</p><p class="cew-note">显示 ${h.integer(Math.min(preview.length, 120))} / ${h.integer(preview.length)} 个已保存${character ? '字符候选' : ' token'}。${character ? '这是保存的字符特征成员关系预览，未将 C3 词序列当作 C2 输入。' : neural ? '这是 C3 head-tail 保留序列。' : '此预览来自 C3 保存的 head-tail 保留序列；C1 的 TF-IDF 使用全文，不能把预览当作完整 C1 输入。'}</p>` : `<p>候选数量：${h.integer(record?.token_count ?? diagnostic.original_length)}</p><p class="cew-note">当前记录没有保存完整${character ? '字符二元组' : '分词'}序列。</p>`;
  add(character ? '字符二元组' : '分词序列', 'ANALYSIS', neural ? '预览是保存的 C3 保留序列。' : 'C1 的 TF-IDF 输入覆盖全文。', tokenPreview);
  if (!neural && !character) add('词级 unigram / bigram', 'WORD FEATURES', '词级一元与二元特征共用当前冻结 TF-IDF 词表。', savedFeatures.length ? `<p>当前新闻已保存的非零特征示例，按词项中的空格区分 unigram 与 bigram；不是完整抽取序列。</p>${detailsTable(h, ['词项', '类型'], savedFeatures.slice(0,20).map(item=>[item.term,item.term.includes(' ')?'bigram':'unigram']))}` : '<p class="cew-note">当前记录没有提供本篇非零词项列表。</p>');

  if (!neural) {
    const membershipBody = membership.length ? `${detailsTable(h, ['候选特征', '本篇 Count', '原 Count 词表', 'χ² 后保留', 'χ² 分数'], membership.map(item => [item.term, h.integer(item.count), item.in_raw_vocabulary === undefined ? '—' : item.in_raw_vocabulary ? '是' : '否', item.retained === undefined ? '—' : item.retained ? '是' : '否', h.number(item.chi2_score)]))}<p class="cew-note">候选与 Count 仅为原文按冻结字符规则的显示枚举；词表成员关系和 χ² 分数读取冻结表示，未拟合或重新预测。</p>` : `<p>最终 χ² 选择维度：${h.integer(vocab)}</p><p class="cew-note">未提供逐特征的原 Count / χ² 保留成员关系。</p>`;
    add(character ? 'Count → χ² 保留' : '冻结词表匹配', 'VOCABULARY', character ? '显示保存的保留与未保留成员关系。' : '输入词项投影到冻结词表。', character ? membershipBody : `<p>最终词表维度：${h.integer(vocab)}</p><p class="cew-note">输入文本投影到已拟合词表。</p>`);
    const featureBody = featureRows.length ? `<p>${matrixBacked ? '当前新闻真实 Top TF-IDF（保存矩阵）' : '已保存特征的 TF-IDF（非整向量 Top）'}</p>${detailsTable(h, ['特征', 'TF-IDF'], featureRows.map(row => [row.term, h.number(row.tfidf ?? row.value)]))}${matrixBacked ? `<p class="cew-note">来源：${h.escape(representation?.matrix_source ?? document?.matrix_source)}。只读取冻结 Test 稀疏矩阵行，未重新拟合表示。</p>` : '<p class="cew-note">此表仅对已保存特征按 TF-IDF 排序，不能称为整行 Top 特征。</p>'}` : `<p>非零特征：${h.integer(representation?.nonzero_count ?? record?.known_feature_count)} · 最终维度：${h.integer(representation?.dimension ?? vocab)}</p><p class="cew-note">当前记录没有保存本篇 TF-IDF 坐标。</p>`;
    add('TF-IDF 稀疏向量', 'VECTOR', matrixBacked ? '读取冻结 Test 矩阵中的本篇真实 Top TF-IDF。' : '只显示已保存特征的真实 TF-IDF。', featureBody);
    const classifier = character ? 'LinearSVC 分数' : modelId === 'C0' ? 'MultinomialNB 后验' : 'LinearSVC 分数';
    add(classifier, 'CLASSIFIER', '五类输出来自冻结分类器。', record?.scores ? detailsTable(h, ['类别', modelId === 'C0' ? '后验' : '决策分数'], record.scores.map(score => [h.plainLabel(score.label), modelId === 'C0' ? h.pct(score.value) : h.number(score.value)])) : '<p class="cew-note">此记录没有五类分数。</p>');
  } else {
    const idsBody = tokenIds.length ? `<p class="cew-token-stream">${tokenIds.slice(0, 120).map((id, index) => `<span>${h.escape(String(id))}${modelTokens[index] ? ` · ${h.escape(modelTokens[index])}` : tokens[index] ? ` · ${h.escape(typeof tokens[index] === 'string' ? tokens[index] : tokens[index]?.text ?? tokens[index]?.token ?? '')}` : ''}</span>`).join('')}</p><p class="cew-note">真实冻结编码 ID；0=&lt;PAD&gt;，1=&lt;UNK&gt;。原长 ${h.integer(diagnostic.original_length)}，保留 ${h.integer(diagnostic.retained_length)}，保留段 UNK ${h.integer(diagnostic.retained_unk_count)}${diagnostic.truncated ? '；head-tail 截断' : ''}。</p>` : `<p>词表（含 PAD/UNK）${h.integer(vocab)} · max_len ${h.integer(maxLen)}</p><p class="cew-note">当前记录没有保存完整 token ID 序列。</p>`;
    add('词表编码与截断', 'TOKEN IDS', '展示真实编码 ID、UNK 和截断诊断。', idsBody);
    add('Embedding', '128D', '每个保留 token 映射为 128 维。', '<p>每个保留 token 进入 128 维 embedding。</p><p class="cew-note">当前 Web 记录没有保存逐 token embedding 数值；这里仅展示冻结网络形状。</p>');
    const spans = array(record?.top_convolution_spans);
    add('三个卷积核的局部窗口', 'K=3/4/5', '各核组 128 filters，展示保存的 Top 响应窗口。', spans.length ? detailsTable(h, ['kernel', '原词窗口', '模型 token', 'activation'], spans.map(span => [span.kernel_size, array(span.token_span).join(' '), array(span.model_token_span).join(' '), h.number(span.max_activation)])) : '<p class="cew-note">核宽 3 / 4 / 5，各 128 filters；当前记录没有保存本篇 Top 窗口。</p>');
    add('Max pool → 384D', 'POOLED', array(pooled).length === 384 ? '384 个通道的实测最大响应。' : '三个核组共 384 通道。', pooledView(pooled, h));
    add('五类 logits → Softmax', 'OUTPUT', '五类 raw logits 与 Softmax 输出。', record?.scores ? detailsTable(h, ['类别', 'raw logit', 'Softmax 输出'], record.scores.map(score => [h.plainLabel(score.label), h.number(score.raw_logit), h.pct(score.value)])) : '<p class="cew-note">此记录没有五类 logits。</p>');
  }
  return {stages:steps, vocab, maxLen, diagnostic, global};
}

export function renderRepresentation({element, inspector, document = {}, record = {}, modelId, overview = {}, representation = {}, ui, provenance=''}) {
  if (!element || !inspector) return;
  const h = helpers(ui), id = modelId || record.model_id || 'C1';
  const {stages, vocab, maxLen, diagnostic, global} = representationStages(id, overview, representation, record, document, h);
  const neural = id === 'C3', savedStage=element.dataset.currentStageKind;
  const tokens=array(representation.tokens), ids=array(representation.token_ids), modelTokens=array(representation.model_tokens);
  const membership=array(representation.feature_membership), features=array(representation.top_tfidf_features).length?array(representation.top_tfidf_features):array(record.top_contributing_features);
  const vectorRows=features.filter(row=>finite(row.tfidf??row.value)).sort((a,b)=>Number(b.tfidf??b.value)-Number(a.tfidf??a.value)).slice(0,12);
  const pools=array(representation.pooled), spans=array(record.top_convolution_spans), scores=array(record.scores);
  const statistics=[['词表维度',h.integer(neural?vocab:representation.dimension??vocab)],['本篇非零特征',h.integer(representation.nonzero_count??record.known_feature_count)],['本篇稀疏率',h.pct(representation.sparsity)],['max_len',h.integer(maxLen)],['参数量',h.integer(global.parameter_count)]];
  let currentStage=0;
  element.style.setProperty('--model-color',MODEL_COLORS[id]);
  element.innerHTML=`<div class="cew-representation-head"><span class="cew-eyebrow">${h.escape(id)} · 冻结表示</span><h2>文本如何进入分类器</h2><p>点击阶段及其中的 token、特征或通道，查看本篇实值。</p></div><div class="cew-stage-list">${stages.map((stage,index)=>`<button type="button" class="cew-stage" data-cew-stage="${index}" aria-pressed="false"><small>${stage.tag}</small><strong>${stage.title}</strong></button>`).join('')}</div><div class="cew-stage-detail" aria-live="polite"></div>`;
  function inspect(index=0) {
    const stage=stages[currentStage], kind=stage.kind;
    let title=stage.title, metrics=[], explanation=`<p class="wb-inspector-note">${h.escape(stage.summary)}</p>`;
    if(kind==='SOURCE'||kind==='NORMALIZE') {
      metrics=[['原文字数',h.integer(Array.from(textOf(document.text)).length)],['保存的 '+(neural?'原长 tokens':'候选词项'),h.integer(neural?diagnostic.original_length:record.token_count)]];
      explanation+=`<p class="wb-inspector-caption">${h.escape(document.document_id || '')}</p>`;
    } else if(kind==='ANALYSIS') {
      const row=membership[index],token=id==='C2'?row?.term:tokens[index];
      title=token??stage.title;
      metrics=id==='C2'?[['本篇 Count',h.integer(row?.count)],['χ² 后保留',row?row.retained?'是':'否':'—']]:[['当前 token 位置',tokens.length?h.integer(index):'—'],['保存序列长度',tokens.length?h.integer(tokens.length):'—']];
      explanation+=`<p class="wb-inspector-caption">${id==='C2'?'候选按原文枚举；χ² 成员关系读取冻结词表。':neural?'当前对象来自保存的 head-tail 保留序列。':'此 token 预览来自 C3 保存的保留序列；C1 TF-IDF 使用全文。'}</p>`;
    } else if(kind==='TOKEN IDS') {
      title=tokens[index]??stage.title;
      metrics=[['当前 Token ID',h.integer(ids[index])],['输入 shape',maxLen?`1 × ${h.integer(maxLen)}`:'—']];
      explanation+=`<div class="wb-token-comparison"><span>模型 token</span><p class="${modelTokens[index]==='<UNK>'?'wb-unk':''}">${h.escape(modelTokens[index]??'未提供')}</p></div><p class="wb-inspector-note">PAD=0 · UNK=1 · 右侧补 PAD；真实编码只读。当前保留 ${h.integer(diagnostic.retained_length)} tokens。</p>`;
    } else if(kind==='WORD FEATURES'||kind==='VOCABULARY'||kind==='VECTOR') {
      const row=kind==='VOCABULARY'&&id==='C2'?membership[index]:kind==='WORD FEATURES'?features[index]:vectorRows[index];
      title=row?.term??stage.title;
      metrics=kind==='VOCABULARY'&&id==='C2'?[['本篇 Count',h.integer(row?.count)],['χ² score',h.number(row?.chi2_score)]]:[['TF-IDF',h.number(row?.tfidf??row?.value)],['向量 shape',`1 × ${h.integer(representation.dimension??vocab)}`]];
      explanation+=kind==='VOCABULARY'&&id==='C2'?`<p class="wb-inspector-caption">原 Count 词表：${row?.in_raw_vocabulary?'存在':'未进入'} · χ² 后：${row?.retained?'保留':'未保留'}</p>`:`<p class="wb-inspector-caption">本篇非零 ${h.integer(representation.nonzero_count??record.known_feature_count)} · 稀疏率 ${h.pct(representation.sparsity)}</p>`;
    } else if(kind==='128D') {
      title=modelTokens[index]??'Embedding';
      metrics=[['Embedding 维度','128'],['输入矩阵 shape',maxLen?`${h.integer(maxLen)} × 128`:'—']];
      explanation+='<p class="wb-inspector-caption">结构 shape 来自冻结网络。未保存逐 token embedding 数值，不展示推测的向量。</p>';
    } else if(kind==='K=3/4/5') {
      const span=spans[index], k=span?.kernel_size??3;
      title=span?array(span.token_span).join(' '):stage.title;
      metrics=[['kernel / filter',`${k} / ${h.escape(value(span?.filter_id))}`],['最大响应',h.number(span?.max_activation)]];
      explanation+=`<p class="wb-inspector-caption">卷积输出 shape：128 × ${maxLen?h.integer(maxLen-k+1):'—'}（结构）。有效窗口仅来自保留 tokens。</p><p class="wb-inspector-note">模型实际 tokens：${h.escape(array(span?.model_token_span).join(' ')||'未提供')}</p>`;
    } else if(kind==='POOLED') {
      const channel=index;
      title=`通道 ${channel} · kernel ${[3,4,5][Math.floor(channel/128)]??'—'}`;
      metrics=[['保存的 activation',h.number(pools[channel])],['向量 shape','1 × 384']];
      explanation+=`<p class="wb-inspector-caption">filter ${channel%128} · 三组 128 个通道。这里只展示保存的 max pool 响应，不是贡献或 attention。</p>`;
    } else if(kind==='OUTPUT'||kind==='CLASSIFIER') {
      const row=scores[index];title=h.plainLabel(row?.label)??stage.title;
      metrics=neural?[['Softmax 输出',h.pct(row?.value)],['raw logit',h.number(row?.raw_logit)]]:[[id==='C0'?'posterior':'decision score',id==='C0'?h.pct(row?.value):h.number(row?.value)]];
      explanation+=`<p class="wb-inspector-note">${neural?'Softmax 未校准。':id==='C0'?'NB 后验未校准。':'SVM 分数不是概率。'}五类结果直接读取冻结预测。</p>`;
    }
    inspector.innerHTML=inspectorObject(id,stage.title,title,metrics,explanation,detailsTable(h,['表示统计','保存值'],statistics)+`<p class="wb-inspector-note">${h.escape(document.document_id||'')} · 文档记录视图 Run ${h.escape(document.source_run||representation.sequence?.source_run||'冻结 Run')}</p><a href="#wb-data-records" data-representation-config>完整实验配置与来源 ↓</a>`,h)+provenance;
    inspector.querySelector('[data-representation-config]').onclick=()=>{const disclosure=window.document.getElementById('wb-data-records');if(disclosure)disclosure.open=true;};
  }
  function select(index) {
    const stage=stages[index];if(!stage)return;
    currentStage=index;element.dataset.currentStageKind=stage.kind;
    element.querySelector('.cew-stage-detail').innerHTML=`<span class="cew-eyebrow">${stage.tag}</span><h3>${stage.title}</h3>${stage.body}`;
    element.querySelectorAll('[data-cew-stage]').forEach(button=>{const active=Number(button.dataset.cewStage)===index;button.classList.toggle('is-active',active);button.setAttribute('aria-pressed',String(active));});
    const content=element.querySelector('.cew-stage-detail');
    let objects=[];
    if(['ANALYSIS','TOKEN IDS'].includes(stage.kind))objects=[...content.querySelectorAll('.cew-token-stream > span')];
    else if(stage.kind==='POOLED')objects=[...content.querySelectorAll('.cew-pool-cell')];
    else if(['WORD FEATURES','VOCABULARY','VECTOR','K=3/4/5','OUTPUT','CLASSIFIER'].includes(stage.kind))objects=[...content.querySelectorAll('.cew-table-wrap tbody tr')];
    objects.forEach((node,i)=>{node.classList.add('cew-inspectable');node.tabIndex=0;node.setAttribute('role','button');node.onclick=()=>{objects.forEach(item=>item.classList.remove('is-selected'));node.classList.add('is-selected');inspect(i);};node.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();node.click();}};});
    if(objects.length)objects[0].click();else inspect();
  }
  element.onclick=event=>{const button=event.target.closest('[data-cew-stage]');if(button&&element.contains(button))select(Number(button.dataset.cewStage));};
  select(Math.max(0,stages.findIndex(stage=>stage.kind===savedStage)));
  return {selectStage:select};
}
