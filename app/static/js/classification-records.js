import {renderStage2} from './classification-stage2.js';
import {renderStage3} from './classification-stage3.js';
const $=id=>document.getElementById(id);
export function renderClassificationRecords(overview,ui,selectedModel){
 const {table,pct,number,escape,label}=ui;
 const modelId=()=>selectedModel;
function renderOverview() {
  const {representation:r,config:c,run,dataset} = overview;
  $('classification-representation').innerHTML = `<div class="table-wrap">${table(['拟合阶段 / split','矩阵形状','稀疏率','零向量'],[
    ['选参 · Train',r.selection.train.shape.join(' × '),pct(r.selection.train.sparsity),r.selection.train.zero_vector_count],
    ['选参 · Validation',r.selection.val.shape.join(' × '),pct(r.selection.val.sparsity),r.selection.val.zero_vector_count],
    ['正式 · Train + Validation',r.final.train_val.shape.join(' × '),pct(r.final.train_val.sparsity),r.final.train_val.zero_vector_count],
    ['正式 · Test',r.final.test.shape.join(' × '),pct(r.final.test.sparsity),r.final.test.zero_vector_count]
  ])}</div><p class="muted">C0 / C1 共用同一矩阵；选参与正式模型分别拟合表示，维度可不同。</p>`;
  $('classification-config').textContent = JSON.stringify({preprocessing:c.preprocessing,tfidf:c.representation},null,2);
  $('classification-provenance').innerHTML = `<p>Run：${escape(run.run_id)}</p><p>Dataset SHA256：${escape(dataset.dataset_fingerprint)}</p><p>Split manifest SHA256：${escape(dataset.split_manifest_sha256)}</p><p>环境：Python ${escape(overview.environment.python.split(' ')[0])} · scikit-learn ${escape(overview.environment.dependencies['scikit-learn'])}</p><p>最终参数：C0 alpha=${escape(run.selected.C0)} · C1 C=${escape(run.selected.C1)}</p>`;
  $('classification-overall').innerHTML = table(['模型','Accuracy','Macro P','Macro R','Macro F1','Test 数'],(overview.available_models || ['C0','C1']).map(id=>{
    const m=overview.test[id]; return [id=== 'C0'?'C0 · MultinomialNB':id==='C1'?'C1 · word TF-IDF + LinearSVC':id==='C2'?'C2 · char + χ² + LinearSVC':'C3 · TextCNN',pct(m.accuracy),pct(m.macro_precision),pct(m.macro_recall),pct(m.macro_f1),m.sample_count];
  }));
  $('classification-validation').innerHTML = table(['模型','候选','Accuracy','Macro P','Macro R','Macro F1','选择'],['C0','C1'].flatMap(id=>overview.validation.candidates[id].map(item=>[id,`${id==='C0'?'alpha':'C'} = ${item.parameter}`,pct(item.metrics.accuracy),pct(item.metrics.macro_precision),pct(item.metrics.macro_recall),pct(item.metrics.macro_f1),item.parameter===run.selected[id]?'已冻结':''])));
  for (const link of document.querySelectorAll('[data-export]')) link.href=`/api/v1/experiments/3/export/${link.dataset.export}?run_id=${encodeURIComponent(run.run_id)}`;
  $('classification-footer').textContent = `Run ${run.run_id} · 冻结 C0 / C1 · Test 90` + (overview.stage2 ? ` · C2 Run ${overview.stage2.run.run_id}` : '');
  renderStage2(overview,ui); renderStage3(overview,ui);
}
function renderModel() {
  const m=overview.test[modelId()];
  $('classification-per-class-title').textContent = `${modelId()} · 每类指标`;
  $('classification-per-class').innerHTML = table(['类别','Precision','Recall','F1','Support'],m.per_class.map(c=>[label(c.label),pct(c.precision),pct(c.recall),pct(c.f1),c.support]));
  $('classification-confusion').innerHTML = `<table class="classification-table classification-matrix"><thead><tr><th>真 / 预测</th>${m.label_order.map(c=>`<th>${label(c)}</th>`).join('')}</tr></thead><tbody>${m.confusion_matrix.map((row,i)=>`<tr><th>${label(m.label_order[i])}</th>${row.map(n=>`<td style="background:rgba(21,138,141,${0.04+n/18*0.3})">${n}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

renderOverview();renderModel();
}
