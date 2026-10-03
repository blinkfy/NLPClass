const $ = id => document.getElementById(id);
const labels = {finance:'财经',education:'教育',science:'科技',sports:'体育',entertainment:'娱乐'};
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const number = value => value === null || value === undefined || value === '' ? null : Number.isFinite(Number(value)) ? Number(value) : null;
const pct = value => number(value) === null ? '—' : `${(Number(value) * 100).toFixed(2)}%`;
const pp = value => number(value) === null ? '—' : `${(Number(value) * 100).toFixed(2)} pp`;
const pvalue = value => number(value) === null ? '—' : Number(value) < .0001 ? '<0.0001' : Number(value).toFixed(4);
const label = value => labels[value] || value;

function table(headers, rows) {
  return `<table class="classification-table"><thead><tr>${headers.map(value=>`<th>${escape(value)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(value=>`<td>${escape(value)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

function pairKey(a, b) { return [a, b].sort().join('|'); }

function selectedPairs(data) {
  const all = data.statistics.pairs || [];
  const requested = data.protocol.primary_pairs || data.protocol.primary_comparisons;
  if (!Array.isArray(requested) || !requested.length) return all.slice(0, 3);
  const byKey = new Map(all.map(pair => [pairKey(pair.a, pair.b), pair]));
  const selected = requested.map(item => {
    if (Array.isArray(item) && item.length >= 2) return byKey.get(pairKey(item[0], item[1]));
    if (item && typeof item === 'object') return byKey.get(pairKey(item.a, item.b));
    if (typeof item === 'string') {
      const [a, b] = item.split(/\s*(?:vs\.?|↔|_|:|\/|-)\s*/i);
      return a && b ? byKey.get(pairKey(a, b)) : undefined;
    }
    return undefined;
  }).filter(Boolean);
  return selected.length ? selected.slice(0, 3) : all.slice(0, 3);
}

function renderF1Chart(pairs) {
  const width = Math.max(280, Math.min(900, $('classification-stage6-f1-chart').clientWidth || 700));
  const height = 70 + pairs.length * 54, left = width < 500 ? 100 : 150, right = width - 24;
  const intervals = pairs.map(pair => pair.macro_f1_difference).filter(Boolean)
    .map(row => [number(row.low), number(row.high)]).filter(row => row.every(value => value !== null));
  if (!intervals.length) return '<p class="muted">没有可绘制的 Macro-F1 配对区间。</p>';
  let low = Math.min(0, ...intervals.map(row => row[0]));
  let high = Math.max(0, ...intervals.map(row => row[1]));
  const padding = Math.max((high - low) * .12, .005);
  low -= padding; high += padding;
  const x = value => left + (value - low) / (high - low) * (right - left);
  const zero = x(0);
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="主要配对比较的 Macro-F1 差异与百分之九十五配对 bootstrap 区间；零线表示无差异">`;
  svg += `<line x1="${zero}" y1="15" x2="${zero}" y2="${height-28}" class="stage6-zero"/>`;
  for (let i = 0; i < pairs.length; i++) {
    const pair = pairs[i], interval = pair.macro_f1_difference;
    const lo = number(interval?.low), hi = number(interval?.high), point = number(interval?.point);
    const y = 34 + i * 54, pairName = `${pair.b} − ${pair.a}`;
    svg += `<text x="${left-12}" y="${y+4}" text-anchor="end">${escape(pairName)}</text>`;
    if (lo === null || hi === null || point === null) continue;
    svg += `<line x1="${x(lo)}" y1="${y}" x2="${x(hi)}" y2="${y}" class="stage6-interval"/><line x1="${x(lo)}" y1="${y-5}" x2="${x(lo)}" y2="${y+5}" class="stage6-cap"/><line x1="${x(hi)}" y1="${y-5}" x2="${x(hi)}" y2="${y+5}" class="stage6-cap"/><circle cx="${x(point)}" cy="${y}" r="4" class="stage6-point"><title>${escape(pairName)} · Δ Macro-F1 ${pp(point)} · 95% CI ${pp(lo)} to ${pp(hi)}</title></circle>`;
  }
  svg += `<text x="${left}" y="${height-26}" text-anchor="start">${escape(pp(low))}</text><text x="${right}" y="${height-26}" text-anchor="end">${escape(pp(high))}</text><text x="${(left+right)/2}" y="${height-8}" text-anchor="middle">Δ Macro-F1 (pp)</text></svg>`;
  return svg;
}

function render(data) {
  const models = data.statistics.models || [];
  const ids = ['C0','C1','C2','C3'];
  const byId = new Map(models.map(model => [model.model_id, model]));
  $('classification-stage6-models').innerHTML = table(
    ['模型','Accuracy','Wilson 95% CI','正确 / n','Macro-F1','分层 bootstrap 95% CI'],
    ids.filter(id => byId.has(id)).map(id => {
      const model = byId.get(id), acc = model.accuracy, f1 = model.macro_f1;
      return [id, pct(acc.point), `${pct(acc.low)} – ${pct(acc.high)}`, `${acc.correct} / ${acc.n}`, pct(f1.point), `${pct(f1.low)} – ${pct(f1.high)}`];
    })
  );

  const pairs = selectedPairs(data);
  $('classification-stage6-pairs').innerHTML = table(
    ['比较 A ↔ B（Δ = B − A）','Δ Accuracy（B − A）','两者都正确','仅 A 正确','仅 B 正确','两者都错','McNemar exact raw p','Holm p','Δ Macro-F1（B − A）· 95% paired CI'],
    pairs.map(pair => {
      const f1 = pair.macro_f1_difference;
      return [
        `${pair.a} ↔ ${pair.b}`, pp(pair.accuracy_difference), pair.both_correct, pair.a_only_correct,
        pair.b_only_correct, pair.both_wrong, pvalue(pair.raw_p), pvalue(pair.holm_p),
        f1 ? `${pp(f1.point)} · ${pp(f1.low)} – ${pp(f1.high)}` : '—',
      ];
    })
  );
  $('classification-stage6-f1-chart').innerHTML = renderF1Chart(pairs);

  const recallRows = models.flatMap(model => (model.recall || []).map(row => [
    model.model_id, label(row.label), `${row.correct} / ${row.support}`, pct(row.point), `${pct(row.low)} – ${pct(row.high)}`,
  ]));
  $('classification-stage6-recall').innerHTML = table(['模型','类别','正确 / support','Recall','Wilson 95% CI'], recallRows);

  const interpretation = data.statistics.interpretation || [];
  $('classification-stage6-interpretation').replaceChildren(...interpretation.map(text => {
    const item = document.createElement('li'); item.textContent = text; return item;
  }));
  $('classification-stage6-protocol').textContent = JSON.stringify(data.protocol, null, 2);
  const audit = data.alignment;
  const alignedCount = audit.sample_count ?? audit.aligned_count ?? audit.n ?? audit.document_count;
  $('classification-stage6-alignment').textContent = `对齐审计：${audit.status || '已载入'}${alignedCount == null ? '' : ` · ${alignedCount} 篇对齐记录`}`;
  $('classification-stage6-status').textContent = `Analysis Run ${data.run.run_id} · ${models.length} models · 90 fixed Test documents`;
  for (const link of document.querySelectorAll('[data-stage6-export]')) {
    link.href = `/api/v1/experiments/3/stage6/export/${link.dataset.stage6Export}?run_id=${encodeURIComponent(data.run.run_id)}`;
  }
}

function configureAnalysisOnly() {
  document.body.dataset.stage6Analysis = 'true';
  $('classification-subtitle').textContent = 'Stage6 冻结 Test 不确定性分析；本视图只读取已完成的统计记录。';
  for (const button of document.querySelectorAll('[data-tab]')) if (button.dataset.tab !== 'comparison') button.disabled = true;
  document.querySelector('[data-tab="comparison"]').click();
  $('classification-run').innerHTML = ['20261001_123337_9c1e7341','20261001_133829_68ee50d7','20261001_153153_c99a8d6f'].map(id=>`<option value="${id}">${id}</option>`).join('');
  $('classification-run').value = '20261001_153153_c99a8d6f';
  $('classification-run').disabled = true;
  $('classification-model').value = 'C3';
  $('classification-model').disabled = true;
  $('classification-footer').textContent = 'Stage6 独立 analysis · 历史分类 Run selector 保持只读';
  $('classification-stage-label').textContent = 'Stage6 不确定性分析 · Test 只读统计';
}

export async function initializeStage6(analysisOnly = false) {
  if (analysisOnly) configureAnalysisOnly();
  const panel = $('classification-stage6-panel');
  try {
    const response = await fetch('/api/v1/experiments/3/stage6/overview');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || '尚无已完成的 Stage6 不确定性分析');
    render(data);
    panel.dataset.loaded = 'true';
    if (analysisOnly) $('classification-stage-label').textContent = `Stage6 · ${data.run.run_id}`;
  } catch (error) {
    $('classification-stage6-status').textContent = error.message;
    panel.dataset.loaded = 'false';
  }
}
