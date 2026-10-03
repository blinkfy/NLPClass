// C3 只展示保存的正式资产与真实卷积响应；历史 Run 隐藏本模块。
const $=id=>document.getElementById(id);

export function renderStage3(data,ui){
  const s=data.stage3, {table,pct,number,escape}=ui;
  for(const id of ['classification-stage3-data','classification-stage3-validation','classification-stage3-paired']) $(id).hidden=!s;
  for(const option of document.querySelectorAll('#classification-sample-filter [data-stage3]')) option.disabled=!s;
  $('classification-subtitle').textContent=s?'小样本 word TextCNN 与冻结稀疏基线：真实结果、训练曲线与卷积最大响应片段。':'同一套词级 TF-IDF，比较朴素贝叶斯与线性支持向量机。';
  $('classification-stage-label').textContent=s?'实验三 · Stage 3 · TextCNN':'实验三 · 第一阶段 C0 / C1';
  if(!s) return;
  $('classification-overall-title').textContent='正式 Test · C0 / C1 / C2 / C3';
  $('classification-overall-note').textContent='同一固定 Test 的后续扩展；C1/C3 共享 HMM 基础，但表示、截断与训练方式不同，不是只改变分类器。每类仅18篇；不根据 Test 调参。';
  const r=s.representation, config=s.config, selected=s.run.selected;
  $('classification-stage3-stats').innerHTML=table(['阶段','词表（含 PAD/UNK）','全文 UNK / tokens','UNK rate','截断篇数 / 总数','截断率','保留 token 比例'],[['Train',r.selection_train],['Validation，transform',r.selection_val],['Final Train+Validation',r.final_train_val],['Test，transform',r.final_test]].map(([name,a])=>[name,a.vocabulary_size,`${a.unk_count} / ${a.original_token_count}`,pct(a.unk_rate),`${a.truncated_count} / ${a.document_count}`,pct(a.truncation_rate),pct(a.retained_token_ratio)]));
  const lengths=r.train_lengths;
  $('classification-stage3-lengths').innerHTML=table(['Train min','mean','median','P90','P95','P99','max','冻结 max_len'],[[lengths.min,number(lengths.mean),lengths.median,lengths.p90,lengths.p95,lengths.p99,lengths.max,r.max_len]]);
  $('classification-stage3-parameters').textContent=`Final trainable parameters：${r.parameter_count.toLocaleString()} · CPU · PyTorch ${s.environment.neural.torch} · Embedding 128D · 384D pooled representation`;
  $('classification-stage3-source').textContent=`C3 Run ${s.run.run_id} · lr=${selected.learning_rate} · seed=42 · final_epochs=${selected.final_epochs}（median ${selected.best_epochs_at_selected_lr.join('/')}）`;
  $('classification-stage3-config').textContent=JSON.stringify(config,null,2);
  const validation=s.validation;
  $('classification-stage3-seeds').innerHTML=table(['lr','seed','best epoch','best Validation Macro-F1','实际 epochs','训练秒'],validation.training_runs.map(a=>[a.learning_rate,a.seed,a.best_epoch,pct(a.best_validation_macro_f1),a.epochs_executed,number(a.runtime_seconds)]));
  $('classification-stage3-stability').innerHTML=table(['lr','三个 seeds mean Macro-F1','population std','选择'],validation.seed_summary.map(a=>[a.learning_rate,pct(a.mean_macro_f1),`${(a.std_macro_f1*100).toFixed(2)} 个百分点`,a.learning_rate===selected.learning_rate?'已冻结':'']));
  const select=$('classification-stage3-curve-run');
  select.innerHTML=validation.training_runs.map(a=>`<option value="${escape(a.candidate_directory)}">lr=${a.learning_rate} · seed=${a.seed}</option>`).join('');
  select.value=validation.training_runs.find(a=>a.learning_rate===selected.learning_rate&&a.seed===42).candidate_directory;
  const render=()=>{
    const record=validation.training_runs.find(a=>a.candidate_directory===select.value);
    $('classification-stage3-loss-curve').innerHTML=curve(record.history,[['train_loss','Train loss','#198b8c'],['validation_loss','Validation loss','#d58e4b']],false);
    $('classification-stage3-f1-curve').innerHTML=curve(record.history,[['validation_macro_f1','Validation Macro-F1','#5853a4']],true);
    $('classification-stage3-curve-caption').textContent=`Stage A 真实 epoch 记录 · best epoch ${record.best_epoch} · early stopping 后停止于 ${record.epochs_executed}。Train accuracy/loss 含训练时 dropout；Validation 在 eval 模式。`;
  }; select.onchange=render;render();
  const pairRows=['c1','c2'].map(id=>{
    const p=s[`paired_${id}`],c=p.counts;
    return [id.toUpperCase()+' ↔ C3',c.both_correct,c.old_only_correct,c.c3_only_correct,c.both_wrong,p.disagreement_ids.length];
  });
  $('classification-stage3-outcomes').innerHTML=table(['配对','两者都对','仅旧模型对','仅 C3 对','两者都错','预测分歧'],pairRows);
  for(const link of document.querySelectorAll('[data-stage3-export]')) link.href=`/api/v1/experiments/3/stage3/export/${link.dataset.stage3Export}?run_id=${encodeURIComponent(s.run.run_id)}`;
  $('classification-footer').textContent+=` · C3 Run ${s.run.run_id}`;
}

function curve(history,series,percent){
  const width=600,height=230,l=58,t=18,bottom=190,right=580,n=history.length;
  const ymax=percent?1:Math.max(0.01,...history.flatMap(r=>series.map(([key])=>r[key])))*1.08;
  const x=epoch=>l+(epoch-1)/Math.max(1,n-1)*(right-l), y=value=>bottom-value/ymax*(bottom-t);
  let content='';
  for(let i=0;i<=4;i++){
    const v=ymax*i/4,cy=y(v);
    content+=`<line x1="${l}" y1="${cy}" x2="${right}" y2="${cy}" stroke="#e1e7e7"/><text x="${l-8}" y="${cy+4}" text-anchor="end">${percent?Math.round(v*100)+'%':v.toFixed(2)}</text>`;
  }
  for(const epoch of [...new Set([1,Math.ceil(n/2),n])]) content+=`<text x="${x(epoch)}" y="211" text-anchor="middle">${epoch}</text>`;
  for(const [key,name,color] of series){
    const points=history.map(r=>`${x(r.epoch)},${y(r[key])}`).join(' ');
    content+=`<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
  }
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${percent?'Validation Macro-F1':'Train 与 Validation loss'} 按 epoch 的真实曲线">${content}<text x="320" y="228" text-anchor="middle">epoch</text></svg><p class="muted">${series.map(([,name,color])=>`<span style="color:${color}">● ${name}</span>`).join(' · ')}</p>`;
}

export function convolutionTable(spans,ui){
  return ui.table(['kernel / filter','卷积最大响应片段','模型实际 tokens','activation','预测类权重','direct contribution'],spans.map(s=>[`${s.kernel_size} / ${s.filter_id}`,s.token_span.join(' ')+(s.crosses_truncation_join?'（跨头尾拼接）':''),s.model_token_span.join(' '),ui.number(s.max_activation),ui.number(s.predicted_class_weight),ui.number(s.contribution)]));
}

export function renderC3Prediction(result,ui){
  const {table,pct,number,label,escape}=ui, d=result.sequence_diagnostic;
  $('classification-prediction-result').innerHTML=`<p class="classification-label">${label(result.prediction)}</p><p class="muted">第二名：${label(result.runner_up)} · Softmax 输出（未校准）</p>${result.scores.map(s=>`<div class="classification-score-row"><span>${label(s.label)}</span><meter min="0" max="1" value="${s.value}"></meter><strong>${pct(s.value)}</strong></div>`).join('')}<details><summary>五类 raw logits</summary>${table(['类别','raw logit'],result.scores.map(s=>[label(s.label),number(s.raw_logit)]))}</details><p class="muted">原长 ${d.original_length} · 保留 ${d.retained_length} · 保留段 UNK ${d.retained_unk_count}${d.truncated?' · head-tail 截断':''}</p><p class="muted">${escape(result.notice)}</p>`;
  $('classification-contributions').innerHTML=`<h3>Top convolution spans · 卷积最大响应片段</h3><div class="table-wrap">${convolutionTable(result.top_convolution_spans,ui)}</div><p class="muted">全部384通道 contribution 和 ${number(result.channel_contribution_sum)} + bias ${number(result.bias)} = raw logit ${number(result.reconstructed_logit)}。上表仅为正贡献 Top spans；不是 attention 或因果解释。</p>`;
}

export function renderC3Samples(data,rows,filter,ui){
  let pairs;
  if(filter==='c1_c3_disagreements') pairs=data.stage3.paired_c1.samples.disagreements;
  if(filter==='c2_c3_disagreements') pairs=data.stage3.paired_c2.samples.disagreements;
  if(filter==='c3_corrected_c2') pairs=data.stage3.paired_c2.samples.corrected_old;
  if(filter==='c3_new_errors') pairs=data.stage3.paired_c2.samples.new_errors;
  const shown=pairs?pairs.map(p=>({...p.c3,old_prediction:p.old.prediction,old_model:filter==='c1_c3_disagreements'?'C1':'C2'})):filter==='all'?rows:rows.filter(r=>!r.correct);
  const {escape,label,table,pct,number}=ui;
  $('classification-error-summary').textContent=`C3 · ${rows.length} 篇 Test · ${rows.filter(r=>!r.correct).length} 篇错误 · 当前显示 ${shown.length} 篇`;
  $('classification-samples').innerHTML=shown.length?shown.map(r=>`<details class="classification-sample"><summary>${escape(r.document_id)} · ${label(r.gold)} → ${label(r.prediction)} · ${r.correct?'正确':'错误'}</summary><p class="classification-news-text">${escape(r.text||r.text_snippet)}</p>${r.old_prediction?`<p>${r.old_model}：${label(r.old_prediction)} · C3：${label(r.prediction)}</p>`:''}<p>第二名：${label(r.runner_up)} · Softmax 输出 ${pct(r.model_score)} · raw logit gap ${number(r.score_gap)}</p><div class="table-wrap">${table(['类别','Softmax 输出','raw logit'],r.scores.map(s=>[label(s.label),pct(s.value),number(s.raw_logit)]))}</div><h3>Top convolution spans</h3><div class="table-wrap">${convolutionTable(r.top_convolution_spans,ui)}</div><p class="muted">bias ${number(r.bias)} + 全部通道贡献 ${number(r.channel_contribution_sum)} = raw logit ${number(r.reconstructed_logit)}。片段不是 attention 或因果解释。</p></details>`).join(''):'<p class="classification-status">该筛选没有样本。</p>';
}

export function c3ModelInfo(meta,ui){
  return `<p>Word TextCNN · ${meta.parameter_count.toLocaleString()} trainable parameters · 384 pooled channels</p><p class="muted">没有全局“关键词”表；卷积最大响应随文本而变。单篇解释展示真实 token spans、最大响应位置与分类器通道贡献。完整384通道记录保存在 predictions artifact。</p>`;
}
