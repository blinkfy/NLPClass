// Stage2 仅读取保存资产；保留 Stage1 的原指标、候选与预测显示。
export function renderStage2(data, ui) {
  const {table,pct,number,escape}=ui, s=data.stage2;
  for(const id of ['classification-stage2-data','classification-stage2-paired','classification-stage2-validation']) document.getElementById(id).hidden=!s;
  for(const option of document.querySelectorAll('#classification-sample-filter [data-stage2]')) option.disabled=!s;
  document.getElementById('classification-overall-title').textContent=s?'正式 Test · C0 / C1 / C2':'正式 Test · C0 / C1';
  document.getElementById('classification-overall-note').textContent=s?'同一 90 篇 Test；C1/C2 是同一 LinearSVC 分类器族下两种经 Validation 选择的表示方案。每类仅 18 篇，避免过度解释小幅变化。':'相同最终表示，相同 90 篇 Test；每类 18 篇，一个样本可改变该类 Recall 约 5.56 个百分点。';
  if(!s) return;
  const final=s.representation.final, selection=s.representation.selection, test=s.representation.test;
  document.getElementById('classification-stage2-dimensions').innerHTML=table(['拟合阶段','raw Count 维度','χ² 选择维度 / TF-IDF 维度'],[
    ['Train 420',selection.raw.raw_count_vocabulary_size,selection.tfidf.shape[1]],
    ['Train+Validation 510',final.raw.raw_count_vocabulary_size,final.tfidf.shape[1]]
  ]);
  document.getElementById('classification-stage2-stats').innerHTML=table(['split','Count 选择前稀疏率','选择后稀疏率','TF-IDF 零向量'],[
    ['Final Train+Validation',pct(final.raw.count.sparsity),pct(final.selected_counts.sparsity),final.tfidf.zero_vector_count],
    ['Test，仅 transform',pct(test.before_selection.sparsity),pct(test.after_selection.sparsity),test.tfidf.zero_vector_count]
  ]);
  const coverage=s.coverage.aggregate;
  document.getElementById('classification-stage2-coverage').innerHTML=table(['表示 / fitted vocabulary','候选出现次数','命中次数','出现次数覆盖率','不同候选覆盖率'],Object.entries(coverage).map(([id,c])=>[id,c.candidate_occurrences,c.hit_occurrences,c.occurrence_ratio==null?'无候选':pct(c.occurrence_ratio),c.unique_type_ratio==null?'无候选':pct(c.unique_type_ratio)]));
  document.getElementById('classification-stage2-chi2').innerHTML=table(['char bigram','χ² relevance score','Train+Val DF'],s.top_chi2.map(f=>[f.term,number(f.chi2_score),f.df]));
  document.getElementById('classification-stage2-config').textContent=JSON.stringify(s.config,null,2);
  document.getElementById('classification-stage2-source').textContent=`C2 Run ${s.run.run_id} · k=${s.run.selected.k} · C=${s.run.selected.C} · Stage1 只读引用 ${data.run.run_id}`;
  const names={both_correct:'两者都对',c1_only_correct:'仅 C1 对',c2_only_correct:'仅 C2 对',both_wrong:'两者都错'};
  document.getElementById('classification-stage2-outcomes').innerHTML=Object.entries(s.paired.counts).map(([key,n])=>`<div><strong>${n}</strong><span>${names[key]}</span></div>`).join('');
  document.getElementById('classification-stage2-candidates').innerHTML=table(['k','C','实际维度','Accuracy','Macro P','Macro R','Macro F1','状态'],s.validation.candidates.map(c=>c.status==='evaluated'?[c.k,c.C,c.selected_feature_count,pct(c.metrics.accuracy),pct(c.metrics.macro_precision),pct(c.metrics.macro_recall),pct(c.metrics.macro_f1),c.k===s.run.selected.k&&c.C===s.run.selected.C?'已冻结':'已评价']:[c.k,c.C,'—','—','—','—','—',c.reason]));
  for(const link of document.querySelectorAll('[data-stage2-export]')) link.href=`/api/v1/experiments/3/stage2/export/${escape(link.dataset.stage2Export)}?run_id=${encodeURIComponent(s.run.run_id)}`;
}

export function relativeContributions(result,ui) {
  if(!result.top_relative_features) return '';
  return `<h3>相对第二名的 Top 贡献</h3><div class="table-wrap">${ui.table(['char bigram','TF-IDF','系数差','相对贡献'],result.top_relative_features.map(f=>[f.term,ui.number(f.tfidf),ui.number(f.coefficient_difference),ui.number(f.contribution)]))}</div><p class="muted">截距差 ${ui.number(result.contrast_intercept)} + 全部相对贡献 ${ui.number(result.contrast_feature_sum)} = score gap ${ui.number(result.score_gap)}。</p>`;
}

export function globalCharFeatures(features,ui) {
  return features.map(c=>`<h3>${ui.label(c.label)}</h3><p class="muted">Top positive SVC coefficients · χ² 只表示训练 Count 与类别的关联排序，不是系数、文档贡献或显著性结论。</p><div class="table-wrap">${ui.table(['char bigram','SVC coefficient','χ² relevance score'],c.positive_features.map(f=>[f.term,ui.number(f.svc_coefficient),ui.number(f.chi2_score)]))}</div>`).join('');
}
