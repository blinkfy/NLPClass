const $=id=>document.getElementById(id);
const conditions=['OO','OS','OT','OA','AA'];
const models=['C0','C1','C2','C3'];
const labels={finance:'财经',education:'教育',science:'科技',sports:'体育',entertainment:'娱乐'};
const conditionNames={OO:'原始输入',OS:'去除来源',OT:'去除模板',OA:'全部清理',AA:'清理后重新训练'};
const directions={correct_to_wrong:'正确 → 错误',wrong_to_correct:'错误 → 正确',wrong_to_wrong:'错误 → 错误'};
const escape=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const label=value=>labels[value]||value||'—';
const pct=value=>Number.isFinite(Number(value))?`${(Number(value)*100).toFixed(2)}%`:'—';
const number=value=>Number.isFinite(Number(value))?Number(value).toFixed(4):'—';
function table(headers,rows){return `<table class="classification-table"><thead><tr>${headers.map(value=>`<th>${escape(value)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(value=>`<td>${escape(value)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;}
async function request(path){
  const response=await fetch(`/api/v1/experiments/3/${path}`);
  const data=await response.json();
  if(!response.ok)throw new Error(data.error?.message||'无法读取 Stage5 分析记录');
  return data;
}

function configureAnalysisOnly(){
  document.body.dataset.stage5Analysis='true';
  document.body.dataset.showStage5Errorpanel='true';
  $('classification-subtitle').textContent='Source / template ablation · 独立开发集分析视图';
  $('classification-run').innerHTML=['20261001_123337_9c1e7341','20261001_133829_68ee50d7','20261001_153153_c99a8d6f'].map(id=>`<option value="${id}">${id}</option>`).join('');
  $('classification-run').value='20261001_153153_c99a8d6f';
  $('classification-run').disabled=true;
  $('classification-model').value='C3';
  $('classification-model').disabled=true;
  for(const button of document.querySelectorAll('[data-tab]'))button.disabled=!['comparison','errors'].includes(button.dataset.tab);
  document.querySelector('[data-tab="comparison"]').click();
  $('classification-footer').textContent='Stage5 独立 analysis · 历史分类 Run 与 classifier selector 保留';
  $('classification-stage-label').textContent='Stage5 开发集分析 · Test 访问禁止';
}

function renderResults(rows){
  const lookup=new Map(rows.map(row=>[`${row.model_id}:${row.condition}`,row]));
  const headers=['模型',...conditions.map(condition=>`${condition} · ${conditionNames[condition]} · Macro-F1 / Δ`)];
  const body=models.map(model=>[
    model,
    ...conditions.map(condition=>{
      const row=lookup.get(`${model}:${condition}`);
      return row?`${pct(row.metrics?.macro_f1)} / ${Number(row.delta_macro_f1)>=0?'+':''}${(Number(row.delta_macro_f1)*100).toFixed(2)} pp`:'—';
    }),
  ]);
  $('classification-stage5-results').innerHTML=table(headers,body);
}

function renderPrevalence(data){
  const rows=['train','validation'].flatMap(split=>(data[split]||[]).map(row=>[
    split==='train'?'Train':'Validation',label(row.label),row.count,row.source_documents,row.template_documents,
    pct(row.source_rate),pct(row.template_rate),pct(row.removed_character_ratio),
  ]));
  $('classification-stage5-prevalence').innerHTML=table(['Split','类别','篇数','Source 篇数','Template 篇数','Source 比例','Template 比例','移除字符比例'],rows);
}

function renderOriginalExplanation(explanation,originalPrediction){
  const item=explanation&&typeof explanation==='object'?explanation:{};
  const scoreType=item.score_type||'score';
  const scoreLabel=scoreType==='posterior'||scoreType==='softmax'?'预测类概率/输出':'预测类分数';
  const summary=[
    ['Prediction',label(item.prediction||originalPrediction)],
    ['Runner-up',label(item.runner_up)],
    ['Score type',scoreType],
    [scoreLabel,item.model_score==null?'—':scoreType==='posterior'||scoreType==='softmax'?pct(item.model_score):number(item.model_score)],
    ['Score gap',number(item.score_gap)],
  ];
  if(item.raw_predicted_logit!=null)summary.push(['Raw predicted logit',number(item.raw_predicted_logit)]);
  else if(item.reconstructed_score!=null)summary.push(['Reconstructed score',number(item.reconstructed_score)]);
  if(item.bias!=null&&item.channel_contribution_sum!=null)summary.push(['Bias + top-channel decomposition',`${number(item.bias)} + ${number(item.channel_contribution_sum)}`]);
  if(item.explanation_note)summary.push(['Note',item.explanation_note]);

  const spans=Array.isArray(item.top_convolution_spans)?item.top_convolution_spans:null;
  const features=Array.isArray(item.top_contributing_features)?item.top_contributing_features:null;
  let detailTable='';
  if(spans){
    detailTable=table(['Top span','Channel','Original positions','Contribution','Join'],spans.map(span=>[
      (span.token_span||span.model_token_span||[]).join(' '),span.channel,
      (span.original_token_positions||[]).join(', '),number(span.contribution),span.crosses_truncation_join?'跨截断拼接':'—',
    ]));
  }else if(features){
    detailTable=table(['Top feature','TF-IDF','Contribution','Source hits','Template hits'],features.map(feature=>[
      feature.term,number(feature.tfidf),number(feature.contribution),
      feature.source_occurrences??'—',feature.template_occurrences??'—',
    ]));
  }
  return `<div class="table-wrap">${table(['原始预测摘要','值'],summary)}${detailTable}</div>`;
}

function renderFlips(rows){
  const direction=$('classification-stage5-flip-direction').value;
  const shown=direction?rows.filter(row=>row.direction===direction):rows;
  $('classification-stage5-flips-status').textContent=`${rows.length} 篇匹配样本 · 当前显示 ${shown.length} 篇`;
  $('classification-stage5-flips').innerHTML=shown.length?shown.map(row=>{
    const spans=(row.removed_spans||[]).map(span=>`<li>${escape(span.kind)} · ${escape(span.rule)} · ${span.start}–${span.end} · ${escape(span.text)}</li>`).join('');
    return `<details class="classification-stage5-flip"><summary>${escape(row.model_id)} · ${escape(row.condition)} · ${escape(row.document_id)} · ${escape(label(row.gold))} · ${escape(label(row.original_prediction))} → ${escape(label(row.prediction))} · ${escape(directions[row.direction]||row.direction)}</summary><p>${escape(row.text_snippet)}</p><h4>Removed spans</h4>${spans?`<ul>${spans}</ul>`:'<p class="muted">无移除片段记录</p>'}<details><summary>Original explanation · compact</summary>${renderOriginalExplanation(row.original_explanation,row.original_prediction)}</details></details>`;
  }).join(''):'<p class="muted">此筛选条件下没有 prediction flips。</p>';
}

export async function initializeStage5(analysisOnly=false){
  if(analysisOnly)configureAnalysisOnly();
  const panel=$('classification-stage5-panel'),errorsPanel=$('classification-stage5-errors');
  panel.hidden=false;
  errorsPanel.hidden=false;
  try{
    const data=await request('stage5/overview');
    const runId=data.run.run_id;
    $('classification-stage5-status').textContent=`Analysis Run ${runId} · ${data.results.length} model-condition rows · Test evaluations=0`;
    $('classification-footer').textContent=`Stage5 Analysis Run ${runId} · Development only`;
    renderResults(data.results);
    renderPrevalence(data.prevalence);
    $('classification-stage5-shortcuts').textContent=JSON.stringify(data.shortcut_summary,null,2);
    for(const link of document.querySelectorAll('[data-stage5-export]'))link.href=`/api/v1/experiments/3/stage5/export/${link.dataset.stage5Export}?run_id=${encodeURIComponent(runId)}`;

    let flipEpoch=0,flipsLoaded=false;
    async function loadFlips(){
      const epoch=++flipEpoch,params=new URLSearchParams();
      const model=$('classification-stage5-flip-model').value,condition=$('classification-stage5-flip-condition').value;
      if(model)params.set('model_id',model);
      if(condition)params.set('condition',condition);
      $('classification-stage5-flips-status').textContent='正在读取开发集 flip 样本…';
      try{
        const rows=await request(`stage5/flips?${params.toString()}`);
        if(epoch===flipEpoch){flipsLoaded=true;renderFlips(rows);}
      }catch(error){if(epoch===flipEpoch)$('classification-stage5-flips-status').textContent=error.message;}
    }
    for(const id of ['classification-stage5-flip-model','classification-stage5-flip-condition'])$(id).addEventListener('change',loadFlips);
    $('classification-stage5-flip-direction').addEventListener('change',loadFlips);
    document.addEventListener('module-tab',event=>{if(event.detail==='errors'&&!flipsLoaded)loadFlips();});
    if(document.querySelector('[data-tab="errors"]').getAttribute('aria-selected')==='true')await loadFlips();
  }catch(error){
    $('classification-stage5-status').textContent=error.message;
    $('classification-stage5-flips-status').textContent='Stage5 flips 无法载入，因为分析 Run 不可用。';
  }
}
