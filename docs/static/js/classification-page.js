import {createErrorDiff} from './classification-error-diff.js';
import {createAlgorithmExplorer} from './classification-algorithm-explorer.js';
import {MODEL_COLORS, MODEL_SCORE_TYPES, modelSource} from './classification-model-colors.js';
import {selectTab} from './layout.js';
import {createClassificationState} from './classification-state.js';
import {initializeComparison} from './classification-comparison-workbench.js';
import {renderPrediction,renderRepresentation} from './classification-evidence-workbench.js';
import {renderClassificationRecords} from './classification-records.js';
import {renderC3Samples,c3ModelInfo} from './classification-stage3.js';
import {globalCharFeatures} from './classification-stage2.js';

const $=id=>document.getElementById(id);
const names={C0:'C0 · MultinomialNB',C1:'C1 · word LinearSVC',C2:'C2 · char χ² LinearSVC',C3:'C3 · TextCNN'};
const labels={finance:'财经',education:'教育',science:'科技',sports:'体育',entertainment:'娱乐'};
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const label=value=>labels[value]||value||'—';
const pct=value=>value==null?'—':`${(Number(value)*100).toFixed(2)}%`;
const number=value=>value==null?'—':Number(value).toFixed(4);
const table=(headers,rows)=>`<table class="classification-table"><thead><tr>${headers.map(h=>`<th>${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(c=>`<td>${escape(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
const ui={escape,label,pct,number,table};
const store=createClassificationState();
const analysisOnly=new URLSearchParams(location.search).get('analysis');
const allowedAnalysis=['stage4','stage5','stage6'].includes(analysisOnly);
const cache=new Map(), predictions={}, featuresCache=new Map();
let overview=null, detail=null, liveResult=null, runEpoch=0, documentEpoch=0, predictEpoch=0;
let flips=null, loadedLegacy=false, settlingRun=false, developmentOverview=null;
function message(text=''){$('page-message').hidden=!text;$('page-message').textContent=text;}
async function request(path,options){
  const response=await fetch(`/api/v1/experiments/3/${path}`,options);
  const data=await response.json();
  if(!response.ok)throw new Error(data.error?.message||`请求失败 (${response.status})`);
  return data;
}
const algorithmExplorer=createAlgorithmExplorer({store,request,ui});

const errorDiff=createErrorDiff({store,request,onOpenPrediction:(selection)=>{
  const evidence=selection&&{...selection};
  const patch={};if(evidence?.model&&evidence.model!==store.get().currentModel)patch.currentModel=evidence.model;
  if(store.get().documentSource==='stage5'&&evidence?.condition&&evidence.condition!==store.get().currentCondition)patch.currentCondition=evidence.condition;
  if(Object.keys(patch).length)store.set(patch);
  if(evidence)store.set({selectedEvidence:evidence});
  if(store.get().documentSource==='stage5')refreshEvidence();
  selectTab('prediction');
}});
const comparison=initializeComparison({element:$('classification-comparison-workbench'),store,request,ui});
const available=()=>overview?.available_models||Object.keys(overview?.test||{});
function modelOptions(){return available().map(id=>`<option value="${id}">${escape(names[id])}</option>`).join('');}
function syncControls(){
  const state=store.get();
  if(overview){
    for(const select of document.querySelectorAll('[data-model-sync]')){select.innerHTML=modelOptions();select.value=state.currentModel;}
    $('classification-model').value=state.currentModel;
    if(!allowedAnalysis)$('classification-run').value=state.currentRun;
    const ids=Object.keys(predictions[state.currentModel]||{});
    const rows=ids.map(id=>predictions[state.currentModel][id]);
    for(const select of document.querySelectorAll('[data-document-sync]')){
      select.innerHTML='<option value="">选择真实 Test 新闻</option>'+rows.map(row=>`<option value="${escape(row.document_id)}">${label(row.gold)} · ${escape((row.text||row.text_snippet||row.document_id).split('\n')[0].slice(0,34))}</option>`).join('');
      select.value=state.documentSource==='test'?state.currentDocument||'':'';
    }
    const other=otherModel();
    $('wb-error-comparison').innerHTML=available().filter(id=>id!==state.currentModel).map(id=>`<option value="${id}">${names[id]}</option>`).join('');
    $('wb-error-comparison').value=other;
    $('wb-error-comparison').disabled=state.documentSource==='stage5';
  }
}
function otherModel(){
  const state=store.get(),pair=String(state.currentComparison||'').split(':');
  return pair.find(id=>available().includes(id)&&id!==state.currentModel)||available().find(id=>id!==state.currentModel);
}
function modelProvenance(id,development=false){
  const s=store.get(),short=development?activeShortcut():null,origin=short?{runId:short.view.model_identity==='clean-trained'?short.bundle.document.source_run:short.bundle.original_model_run,stage:short.view.model_identity==='clean-trained'?'Stage5 · All-clean Train420 AA 模型':'Stage4 · Train420 原始模型'}:development?{runId:developmentOverview?.run?.parent_stage4_run,stage:'Stage4 · Train420 OO 原始模型'}:modelSource(overview,id);
  const viewRun=allowedAnalysis&&development?detail?.document?.source_run:s.currentRun;
  if(!origin.runId || origin.runId===viewRun)return '';
  return `<details class="wb-inspector-details"><summary>Run 来源 · ${escape(id)}</summary>${table(['来源','保存值'],[['当前工作台 Run',viewRun],['模型来源 Run',origin.runId],['模型来源 Stage',origin.stage],...(development?[['验证扰动分析 Run',detail?.document?.source_run]]:[])])}</details>`;
}
function activeShortcut(){
  const s=store.get(),bundle=detail?.shortcut;
  if(s.documentSource!=='stage5'||!bundle||bundle.model_id!==s.currentModel)return null;
  const condition=s.currentCondition||'OO',view=bundle.conditions[condition];if(!view)return null;
  // Saved C3 offsets refer to canonical original text. Project them into the
  // display input segments, without creating any cross-deletion window.
  const segments=view.segments;let prefix=0;const mapped=segments.map(seg=>{const out={...seg,input_start:prefix};prefix+=Array.from(seg.text).length+1;return out;});
  const offset=range=>{const a=range.start??range[0],b=range.end??range[1],seg=mapped.find(t=>a>=t.start&&b<=t.end);return seg?[seg.input_start+a-seg.start,seg.input_start+b-seg.start]:null;};
  const record={...view.record,top_convolution_spans:view.record.top_convolution_spans?.map(span=>({...span,character_offsets:(span.character_offsets||[]).map(offset).filter(Boolean)}))};
  return {record,text:view.input_text,condition,view,bundle};
}
function context(){
  const s=store.get(),shortcut=activeShortcut();
  if(shortcut)return `Validation only · Stage5 ${shortcut.condition} · ${shortcut.view.model_identity==='clean-trained'?'重新训练后的 clean model':'Original Train420 model'} · 保存预测，不重新推理。`;
  return detail?.document?.split?.startsWith('Validation')?`Validation · Stage5 OO 原始模型（Train420）· 分析 Run ${detail.document.source_run}；不是正式 Test 模型。`:`${s.currentModel} · 当前工作台 Run ${s.currentRun} · 冻结 Test 预测，无需重新推理。`;
}
function header(){
  $('classification-subtitle').textContent='冻结实验结果工作台 · 选择模型、新闻与分析问题，点击对象查看证据。';
  $('classification-footer').textContent=allowedAnalysis?`${analysisOnly} · 独立只读分析`:`${store.get().currentRun||'正在加载'} · ${detail?.document?.split||'正式 Run'} · 展示层不回写实验资产`;
  $('classification-stage-label').textContent='Experiment 3 · Frozen C0–C3';
}
function sourceOffsets(text,sequence){
  // Display alignment of saved tokens, not a new tokenizer. NFKC grapheme map
  // preserves original source offsets; head/tail groups are aligned separately.
  const segments=typeof Intl.Segmenter==='function'?[...new Intl.Segmenter('zh',{granularity:'grapheme'}).segment(text)].map(s=>({segment:s.segment,index:s.index})):Array.from(text).map((segment,i,arr)=>({segment,index:arr.slice(0,i).join('').length}));
  let normalized='',map=[];
  for(const s of segments){const norm=s.segment.normalize('NFKC').toLowerCase();for(let i=0;i<norm.length;i++)map.push({start:s.index,end:s.index+s.segment.length});normalized+=norm;}
  const tokens=sequence?.tokens||[],positions=sequence?.original_positions||[],offsets=Array(tokens.length).fill(null);
  let split=positions.findIndex((p,i)=>i&&p!==positions[i-1]+1);if(split<0)split=tokens.length;
  let cursor=0;
  for(let i=0;i<split;i++){const token=tokens[i].normalize('NFKC').toLowerCase(),at=normalized.indexOf(token,cursor);if(at>=0&&token.length){offsets[i]={start:map[at].start,end:map[at+token.length-1].end};cursor=at+token.length;}}
  cursor=normalized.length;
  for(let i=tokens.length-1;i>=split;i--){const token=tokens[i].normalize('NFKC').toLowerCase(),at=normalized.lastIndexOf(token,cursor-token.length);if(at>=0&&token.length){offsets[i]={start:map[at].start,end:map[at+token.length-1].end};cursor=at;}}
  return offsets;
}
function decoratedRecord(row,modelId,representation,text){
  if(!row)return null;
  const result={...row,model_id:modelId};
  if(!representation?.sequence&&modelId==='C3')result.top_convolution_spans=(row.top_convolution_spans||[]).map(span=>({...span,character_offsets:(span.character_offsets||[]).map(r=>sourceRange(text,r))}));
  if(representation?.nonzero_count!=null)result.known_feature_count=representation.nonzero_count;
  const sequence=representation?.sequence;
  if(sequence&&modelId==='C3'){
    const offsets=sourceOffsets(text,sequence),lookup=new Map(sequence.original_positions.map((p,i)=>[p,offsets[i]]));
    result.top_convolution_spans=(row.top_convolution_spans||[]).map(span=>{
      if(span.crosses_truncation_join)return {...span};
      const parts=(span.original_token_positions||[]).map(p=>lookup.get(p));
      return parts.length&&parts.every(Boolean)?{...span,character_offsets:[{start:parts[0].start,end:parts.at(-1).end}]}:{...span};
    });
  }
  return result;
}
function representationData(modelId){
  const raw=detail?.representations?.[modelId]||{},sequence=raw.sequence||{};
  const result={...raw,tokens:sequence.tokens||[],token_ids:sequence.token_ids||[],model_tokens:sequence.model_tokens||[],sequence_diagnostic:sequence.diagnostic,pooled:sequence.pooled};
  if(modelId==='C2')result.tokens=(raw.feature_membership||[]).flatMap(f=>Array(Math.min(f.count,4)).fill(f.term));
  return result;
}
function refreshEvidence(){
  const state=store.get(),shortcut=activeShortcut(),raw=liveResult||shortcut?.record||detail?.records?.[state.currentModel];
  const text=liveResult?$('classification-text').value:shortcut?.text||detail?.document?.text||'';
  if(shortcut&&!liveResult)$('classification-text').value=text;
  const rep=detail?.representations?.[state.currentModel];
  const record=decoratedRecord(raw,state.currentModel,liveResult?null:rep,text);
  if(record){renderPrediction({element:$('classification-prediction-view'),inspector:$('classification-prediction-inspector'),result:record,text,ui,provenance:modelProvenance(state.currentModel,!liveResult&&detail?.document?.split?.startsWith('Validation')),onViewErrors:()=>{selectTab('errors');renderErrorDocument();}});}
  else {$('classification-prediction-view').innerHTML='<p class="muted">选择冻结新闻查看已保存预测，或提交自定义文本。</p>';$('classification-prediction-inspector').innerHTML='<h2>证据检查器</h2><p class="muted">点击当前特征或响应片段。</p>';}
  if(shortcut&&record&&state.selectedEvidence?.model===state.currentModel){
    const selected=state.selectedEvidence,items=selected.kind==='channel'?record.top_convolution_spans||[]:record.top_contributing_features||[];
    const index=items.findIndex(row=>selected.kind==='channel'?row.channel===selected.channel:row.term===selected.term);
    if(index>=0)$('classification-prediction-view').querySelector(`.cew-evidence-card[data-cew-${selected.kind==='channel'?'span':'feature'}="${index}"]`)?.click();
  }
  $('classification-score-note').textContent=state.currentModel==='C0'?'NB posterior / joint log-score，未校准。':state.currentModel==='C3'?'Softmax 输出 + raw logits，未校准；卷积最大响应片段不是 attention 或因果解释。':'LinearSVC decision score，不是概率。';
  $('wb-prediction-context').textContent=liveResult?'自定义新闻的交互预测；不计入正式实验，不保存为 Run。':detail?context():'当前为自定义输入；最多5,000字。';
  $('wb-prediction-errors').disabled=!record;
  $('classification-predict').disabled=!overview;
  if(detail?.document?.split?.startsWith('Validation')&&!liveResult){
    $('classification-representation-view').innerHTML=`<h2>Stage5 Validation · ${shortcut?.condition||'OO'} 输入</h2><p class="muted">${escape(context())} 此处不套用正式 Train+Validation 模型的词表与参数。</p><p class="cew-stage-text">${escape(text)}</p>`;
    $('classification-representation-inspector').innerHTML=`<h2>表示检查器</h2><p>${escape(context())}</p><p>完整 Token IDs / 稀疏矩阵未在此视图提供。已保存的特征与卷积证据可在分类预测和错误分析中查看。</p>${modelProvenance(state.currentModel,true)}`;
    $('wb-representation-source').textContent=`${detail.document.document_id} · ${detail.document.split}`;
  }else if(detail&&!liveResult){
    const representation=representationData(state.currentModel);
    renderRepresentation({element:$('classification-representation-view'),inspector:$('classification-representation-inspector'),document:{...detail.document,sequence_diagnostic:raw?.sequence_diagnostic},record,modelId:state.currentModel,overview,representation,ui,provenance:modelProvenance(state.currentModel)});
    $('wb-representation-source').textContent=`${detail.document.document_id} · ${detail.document.split}`;
  }else{$('classification-representation-view').innerHTML='<p class="muted">表示探索使用冻结真实新闻；请选择左侧示例。</p>';$('classification-representation-inspector').innerHTML='<h2>表示检查器</h2>';}
  header();
  algorithmExplorer.update(!liveResult&&detail?.document?.split==='Test'&&['C1','C2','C3'].includes(state.currentModel)?{runId:state.currentRun,modelId:state.currentModel,document:detail.document,record:raw,representation:detail.representations[state.currentModel],tokenOffsets:detail.representations[state.currentModel]?.sequence?sourceOffsets(detail.document.text,detail.representations[state.currentModel].sequence):[],overview}:null);
}
async function loadDocument(){
  const s=store.get(),epoch=++documentEpoch;
  liveResult=null;
  if(!s.currentDocument){detail=null;refreshEvidence();return;}
  $('classification-prediction-view').innerHTML='<p class="muted">正在读取保存的预测与证据…</p>';
  $('wb-error-document').innerHTML='<h2>文档查看器</h2><p class="muted">正在读取冻结文档…</p>';
  const key=`${s.currentRun}:${s.documentSource}:${s.currentDocument}:${s.currentCondition}`;
  errorDiff.clear('正在读取当前文档…');
  try{
    if(!cache.has(key))cache.set(key,await request(`workbench/document?${new URLSearchParams({run_id:s.currentRun,document_id:s.currentDocument,source:s.documentSource,condition:s.currentCondition==='OO'?'OA':s.currentCondition})}`));
    if(epoch!==documentEpoch)return;
    const next={...cache.get(key)};
    if(next.document.split?.startsWith('Validation')){
      const originKey=`stage5-origin:${next.document.source_run}`;
      if(!cache.has(originKey))cache.set(originKey,await request(`stage5/overview?run_id=${encodeURIComponent(next.document.source_run)}`));
      if(epoch!==documentEpoch)return;
      developmentOverview=cache.get(originKey);
      const shortcutKey=`shortcut:${next.document.source_run}:${s.currentDocument}:${s.currentModel}`;
      if(!cache.has(shortcutKey))cache.set(shortcutKey,await request(`visualization/shortcut-document?${new URLSearchParams({run_id:next.document.source_run,document_id:s.currentDocument,model_id:s.currentModel})}`));
      if(epoch!==documentEpoch)return;
      next.shortcut=cache.get(shortcutKey);
    }
    detail=next;$('classification-text').value=detail.document.text;
    refreshEvidence();renderErrorDocument();renderErrorList();
  }catch(error){if(epoch===documentEpoch){detail=null;refreshEvidence();renderErrorDocument();message(error.message);$('classification-prediction-view').textContent=error.message;$('wb-error-document').textContent=error.message;}}
}
function sourceRange(text,range){
  // Stage5 stores Python Unicode-codepoint offsets; browser slices use UTF-16.
  const chars=Array.from(text),start=Array.isArray(range)?range[0]:range.start,end=Array.isArray(range)?range[1]:range.end;
  return {...range,start:chars.slice(0,start).join('').length,end:chars.slice(0,end).join('').length};
}
const probabilityScore=row=>row?.score_type==='posterior'||row?.score_type==='softmax'||row?.model_id==='C3';
function rangesHTML(text,ranges){
  const sorted=ranges.filter(r=>Number.isInteger(r.start)&&Number.isInteger(r.end)&&r.start>=0&&r.end>r.start&&r.end<=text.length).sort((a,b)=>a.start-b.start||b.end-a.end);
  let html='',cursor=0;
  for(const r of sorted){if(r.start<cursor)continue;html+=escape(text.slice(cursor,r.start))+`<mark class="${r.kind==='removed'?'wb-removed':'wb-highlight'}" data-evidence-index="${r.index??''}" title="${escape(r.title||'保存的证据')}">${escape(text.slice(r.start,r.end))}</mark>`;cursor=r.end;}
  return html+escape(text.slice(cursor));
}
function evidenceRanges(record,text){
  if(record?.top_convolution_spans)return record.top_convolution_spans.slice(0,4).flatMap((s,index)=>(s.character_offsets||[]).map(r=>({...r,index,title:s.token_span.join(' ')})));
  const selected=record?.top_contributing_features||[],ranges=[];
  for(let index=0;index<Math.min(3,selected.length);index++){
    const term=selected[index].term,parts=term.split(' ').filter(Boolean);
    // Exact text occurrence only: separated unigram highlights do not invent a bigram occurrence.
    for(const part of parts){let start=0;while((start=text.indexOf(part,start))>=0){ranges.push({start,end:start+part.length,index,title:term});break;}}
  }
  return ranges;
}
function renderLegacyErrorDocument(){
  const s=store.get(),other=otherModel(),text=liveResult?$('classification-text').value:detail?.document.text||'';
  const record=decoratedRecord(liveResult||detail?.records?.[s.currentModel],s.currentModel,liveResult?null:detail?.representations?.[s.currentModel],text);
  if(!record){$('wb-error-document').innerHTML='<h2>文档查看器</h2><p class="muted">选择左侧新闻。</p>';$('wb-error-inspector').innerHTML='<h2>判断检查器</h2>';return;}
  const b=detail?.records?.[other],removed=detail?.ablation?.removed_spans||[];
  const ranges=[...removed.map(r=>({...sourceRange(text,r),kind:'removed',title:`${r.kind} · ${r.rule}`})),...evidenceRanges(record,text)];
  const source=liveResult?'交互输入 · 非正式样本':`${detail.document.split} · ${detail.document.document_id}`;
  $('wb-error-document').innerHTML=`<div class="wb-object-heading"><h2>${escape(source)}</h2><button class="outline" data-open-prediction>在分类预测中查看 →</button></div><div class="wb-document-tags"><span>真实类别 · ${label(detail?.document.gold)}</span><span>${s.currentModel} · ${label(record.prediction)}</span>${b?`<span>${other} · ${label(b.prediction)}</span>`:''}</div><p class="muted">${escape(liveResult?'当前交互结果':context())}</p><article class="wb-news">${rangesHTML(text,ranges)}</article><p class="muted">${record.top_convolution_spans?'卷积最大响应片段；原词与 <UNK> 模型 token 分开显示。':'高亮只定位原文中实际出现的组成词；分开的两个词不证明它们构成连续 bigram。'}${removed.length?' 粉红标记为 Stage5 已保存的移除区域。':''}</p><div id="wb-document-evidence"></div>`;
  const inspector=$('wb-error-inspector');
  const judgment=(row,id,role)=>{
    const top=(row.top_convolution_spans?.[0]?.token_span||[]).join(' ')||row.top_contributing_features?.[0]?.term||'没有保存 Top 证据';
    const output=MODEL_SCORE_TYPES[id];
    return `<section class="wb-judgment"><span class="wb-inspector-kicker"><i style="background:${MODEL_COLORS[id]}"></i>${role} · ${id}</span><div class="wb-judgment-result"><strong>${label(row.prediction)}</strong><span>${probabilityScore({...row,model_id:id})?pct(row.model_score):number(row.model_score)}</span></div><p class="wb-inspector-caption">${output} · 第二名 ${label(row.runner_up)}</p><p class="wb-judgment-evidence">Top1 · ${escape(top)}</p>${modelProvenance(id,!liveResult&&detail?.document?.split?.startsWith('Validation'))}</section>`;
  };
  inspector.innerHTML=`<div class="wb-inspector-heading"><span class="wb-inspector-kicker">判断检查器</span><h3 class="wb-object-title">当前模型与对比模型</h3></div><div class="wb-judgment-pair">${judgment(record,s.currentModel,'A')}${b?judgment(b,other,'B'):''}</div>${b&&MODEL_SCORE_TYPES[s.currentModel]!==MODEL_SCORE_TYPES[other]?'<p class="wb-inspector-note">分数类型不同，仅用于模型内部判别，不直接比较数值大小。</p>':''}<div id="wb-error-evidence-detail"></div>`;
  renderPrediction({element:$('wb-document-evidence'),inspector:$('wb-error-evidence-detail'),result:record,text,ui,onEvidenceSelect:selection=>{
    if(selection.type==='span'||selection.type==='feature'){
      const selectedRecord=selection.type==='span'?{top_convolution_spans:[selection.item]}:{top_contributing_features:[selection.item]};
      const selected=evidenceRanges(selectedRecord,text).map(range=>({...range,index:selection.index}));
      $('wb-error-document').querySelector('.wb-news').innerHTML=rangesHTML(text,[...removed.map(r=>({...sourceRange(text,r),kind:'removed',title:`${r.kind} · ${r.rule}`})),...selected]);
    }
    $('wb-error-document').querySelectorAll('.wb-news [data-evidence-index]').forEach(mark=>mark.classList.toggle('is-selected',mark.dataset.evidenceIndex===String(selection.index)&&selection.type!=='score'));
    const viewer=$('wb-error-document').querySelector('.wb-news'),active=viewer.querySelector('[data-evidence-index].is-selected');
    if(active){const box=viewer.getBoundingClientRect(),target=active.getBoundingClientRect();viewer.scrollTop+=target.top-box.top-Math.max(28,viewer.clientHeight/3);}
  }});
  if(detail?.ablation){
    const row=detail.ablation.changes[s.currentModel],card=document.createElement('section');card.className='wb-ablation-context';
    card.innerHTML=`<span class="wb-inspector-kicker">Stage5 · Validation only</span><h3>原始 → ${escape(detail.ablation.condition)}</h3><p class="wb-flip-predictions"><strong>${label(record.prediction)}</strong><span>→</span><strong>${label(row.prediction)}</strong></p><p class="wb-inspector-caption">${s.currentModel} · 原模型 Train420 · 冻结清理条件</p>${removed.length?removed.slice(0,3).map(r=>`<div class="wb-removed-evidence"><span>${escape(r.kind)} · ${escape(r.rule)}</span><p>${escape(r.text)}</p></div>`).join(''):'<p class="wb-inspector-note">当前条件没有提供移除片段记录。</p>'}<details class="wb-inspector-details"><summary>完整清理值与区间</summary>${table(['量','保存值'],[['清理后 '+(s.currentModel==='C3'?'Softmax 输出':s.currentModel==='C0'?'posterior':'decision score'),probabilityScore({...row,model_id:s.currentModel})?pct(row.model_score):number(row.model_score)],['清理后第二名',label(row.runner_up)]])}${table(['类型','规则','原文区间','移除文本'],removed.map(r=>[r.kind,r.rule,`${r.start}–${r.end}`,r.text]))}</details>`;
    inspector.insertBefore(card,$('wb-error-evidence-detail'));
  }
  const complete=document.createElement('details');complete.className='wb-inspector-details';complete.innerHTML=`<summary>A / B 完整类别分数</summary>${[record,b].filter(Boolean).map((row,i)=>`<h4>${i?other:s.currentModel}</h4>${table(['类别','保存分数'],row.scores.map(score=>[label(score.label),probabilityScore({...row,model_id:i?other:s.currentModel})?pct(score.value):number(score.value)]))}`).join('')}`;inspector.append(complete);
  $('wb-error-document').querySelector('[data-open-prediction]').hidden=allowedAnalysis;
  $('wb-error-document').querySelector('[data-open-prediction]').onclick=()=>selectTab('prediction');
  $('wb-error-document').querySelector('.wb-news').onclick=event=>{
    const mark=event.target.closest('[data-evidence-index]');if(!mark||mark.dataset.evidenceIndex==='')return;
    $('wb-document-evidence').querySelector(`[${record.top_convolution_spans?'data-cew-span':'data-cew-feature'}="${mark.dataset.evidenceIndex}"]`)?.click();
  };
}
function renderErrorDocument(){
  const s=store.get();
  if(liveResult){errorDiff.clear();renderLegacyErrorDocument();return;}
  if(!detail||detail.document.document_id!==s.currentDocument){errorDiff.clear();return;}
  const records=Object.fromEntries(Object.entries(detail.records).map(([id,row])=>[id,decoratedRecord(row,id,detail.representations?.[id],detail.document.text)]));
  errorDiff.update({detail,records,other:otherModel(),shortcut:detail.shortcut||null,modelSources:Object.fromEntries(available().map(id=>[id,modelSource(overview,id)]))});
}
async function ensureFlips(){if(flips===null)flips=await request('stage5/flips');return flips;}
function renderErrorList(){
  if(!overview){$('wb-error-count').textContent='等待当前 Run';$('wb-error-list').textContent='当前 Run 尚未读取完成。';return;}
  const s=store.get(),other=otherModel();
  if(s.documentSource==='stage5')$('wb-error-set').value='stage5';
  else if($('wb-error-set').value==='stage5')$('wb-error-set').value='errors';
  const mode=$('wb-error-set').value;
  $('wb-flip-controls').hidden=mode!=='stage5';
  let rows;
  if(mode==='stage5'){
    if(flips===null){$('wb-error-list').textContent='正在读取 Validation flips…';ensureFlips().then(renderErrorList).catch(error=>message(error.message));return;}
    rows=flips.filter(r=>r.model_id===s.currentModel&&(!$('wb-error-condition').value||r.condition===$('wb-error-condition').value)&&(!$('wb-error-direction').value||r.direction===$('wb-error-direction').value));
    const grouped=new Map();
    for(const row of rows){const existing=grouped.get(row.document_id);if(existing)existing.flip_conditions.push(row.condition);else grouped.set(row.document_id,{...row,text_snippet:row.original_explanation?.text_snippet||row.text_snippet,flip_conditions:[row.condition]});}
    rows=[...grouped.values()];
  }else{
    rows=Object.values(predictions[s.currentModel]||{}).filter(a=>{const b=predictions[other]?.[a.document_id];if(mode==='all')return true;if(mode==='a_only')return a.correct&&!b?.correct;if(mode==='b_only')return !a.correct&&b?.correct;if(mode==='both_wrong')return !a.correct&&b&&!b.correct;if(mode==='disagreement')return b&&a.prediction!==b.prediction;return !a.correct;});
  }
  if(!s.currentDocument&&rows.length){const row=rows[0];store.set({currentDocument:row.document_id,documentSource:mode==='stage5'?'stage5':'test',currentCondition:row.condition||s.currentCondition});}
  $('wb-error-count').textContent=`${mode==='stage5'?'Validation · Stage5':'Test'} · ${rows.length} 条 · A=${s.currentModel}${mode!=='stage5'&&other?` / B=${other}`:''}`;
  $('wb-error-list').innerHTML=rows.length?rows.map((row,i)=>`<button class="wb-document-item${row.document_id===store.get().currentDocument&&(mode==='stage5'?store.get().documentSource==='stage5':store.get().documentSource==='test')?' is-selected':''}" data-doc-index="${i}" aria-pressed="${row.document_id===store.get().currentDocument&&(mode==='stage5'?store.get().documentSource==='stage5':store.get().documentSource==='test')}"><strong>${escape((row.text||row.text_snippet||row.document_id).split('\n')[0].slice(0,58))}</strong><span>真实 ${label(row.gold)} → ${label(row.prediction)}${row.condition?` · ${(row.flip_conditions||[row.condition]).join(' / ')}`:''}</span><small>${escape(row.document_id)}</small></button>`).join(''):'<p class="muted">这个集合没有样本。</p>';
  $('wb-error-list').onclick=event=>{const button=event.target.closest('[data-doc-index]');if(!button)return;const row=rows[Number(button.dataset.docIndex)];store.set({currentDocument:row.document_id,documentSource:mode==='stage5'?'stage5':'test',currentCondition:row.condition||s.currentCondition});};
  return rows;
}
async function loadGlobalFeatures(){
  const s=store.get(),key=`${s.currentRun}:${s.currentModel}`;
  const path=`${s.currentModel==='C3'?'stage3/':s.currentModel==='C2'?'stage2/':''}features?${new URLSearchParams({run_id:s.currentRun,model_id:s.currentModel})}`;
  try{if(!featuresCache.has(key))featuresCache.set(key,await request(path));if(key!==`${store.get().currentRun}:${store.get().currentModel}`)return;const f=featuresCache.get(key);$('classification-global-features').innerHTML=s.currentModel==='C3'?c3ModelInfo(f,ui):s.currentModel==='C2'?globalCharFeatures(f,ui):f.map(c=>`<h3>${label(c.label)}</h3><p>${s.currentModel==='C0'?`class prior ${pct(c.class_prior)}`:`intercept ${number(c.intercept)}`}</p>${table(['特征','对比权重'],c.discriminative_features.map(r=>[r.term,number(r.contrast_weight)]))}`).join('');}catch(error){message(error.message);}
}
async function loadRun(){
  const s=store.get(),epoch=++runEpoch;message();detail=null;liveResult=null;++predictEpoch;++documentEpoch;
  for(const id of Object.keys(predictions))delete predictions[id];
  $('classification-text').value='';$('wb-representation-source').textContent='正在读取当前 Run…';
  for(const id of ['wb-data-records','wb-comparison-records','wb-error-records'])$(id).hidden=true;
  overview=null;comparison.updateOverview(null);refreshEvidence();renderErrorDocument();$('wb-error-list').textContent='正在读取当前 Run…';
  $('classification-model').disabled=true;
  try{
    const next=await request(`overview?run_id=${encodeURIComponent(s.currentRun)}`);
    const models=next.available_models||Object.keys(next.test);
    const all=await Promise.all(models.map(id=>request(`${id==='C3'?'stage3/':id==='C2'?'stage2/':''}predictions?${new URLSearchParams({run_id:s.currentRun,model_id:id})}`)));
    if(epoch!==runEpoch)return;
    overview=next;for(const id of Object.keys(predictions))delete predictions[id];
    for(const id of ['wb-data-records','wb-comparison-records','wb-error-records'])$(id).hidden=false;
    models.forEach((id,i)=>{predictions[id]=Object.fromEntries(all[i].map(r=>[r.document_id,r]));});
    const current=store.get(),model=models.includes(current.currentModel)?current.currentModel:next.default_model||'C1';
    const keepDevelopment=current.documentSource==='stage5'&&current.currentDocument;
    const doc=keepDevelopment?current.currentDocument:predictions[model][current.currentDocument]?current.currentDocument:(all[models.indexOf(model)].find(row=>!row.correct)||all[models.indexOf(model)][0])?.document_id;
    // The enclosing Run defines model availability; older views never gain later classifiers.
    for(const option of $('classification-model').options)option.disabled=!models.includes(option.value);
    settlingRun=true;store.set({currentModel:model,currentDocument:doc,documentSource:keepDevelopment?'stage5':'test'});settlingRun=false;
    syncControls();renderClassificationRecords(overview,ui,model);comparison.updateOverview(overview);renderErrorList();renderSavedSamples();header();
    await loadDocument();$('classification-model').disabled=false;$('classification-run').disabled=false;
    loadGlobalFeatures();
  }catch(error){if(epoch===runEpoch){message(error.message);$('classification-model').disabled=false;}}
}
store.subscribe((state,previous)=>{
  if(allowedAnalysis){
    if(analysisOnly==='stage5'&&overview&&!settlingRun){
      syncControls();
      if(state.currentDocument!==previous.currentDocument||state.currentCondition!==previous.currentCondition||state.documentSource!==previous.documentSource)loadDocument();
      else if(state.currentModel!==previous.currentModel){loadDocument();renderErrorList();}
      else if(state.currentComparison!==previous.currentComparison){renderErrorList();renderErrorDocument();}
    }
    header();return;
  }
  if(state.currentRun!==previous.currentRun){loadRun();return;}
  if(!overview||settlingRun)return;
  if(state.currentModel!==previous.currentModel)++predictEpoch;
  syncControls();
  if(state.currentDocument!==previous.currentDocument||state.documentSource!==previous.documentSource||state.currentCondition!==previous.currentCondition){loadDocument();}
  else if(state.currentModel!==previous.currentModel){if(state.documentSource==='stage5'){loadDocument();renderErrorList();return;}liveResult=null;refreshEvidence();renderErrorDocument();renderErrorList();renderClassificationRecords(overview,ui,state.currentModel);renderSavedSamples();header();loadGlobalFeatures();}
  else if(state.currentComparison!==previous.currentComparison){renderErrorList();renderErrorDocument();}
});
for(const control of document.querySelectorAll('[data-model-sync]'))control.onchange=()=>store.set({currentModel:control.value});
$('classification-model').onchange=()=>store.set({currentModel:$('classification-model').value});
$('classification-run').onchange=()=>store.set({currentRun:$('classification-run').value});
for(const control of document.querySelectorAll('[data-document-sync]'))control.onchange=()=>{if(control.value)store.set({currentDocument:control.value,documentSource:'test'});};
$('wb-error-comparison').onchange=()=>store.set({currentComparison:`${store.get().currentModel}:${$('wb-error-comparison').value}`});
for(const id of ['wb-error-set','wb-error-condition','wb-error-direction'])$(id).onchange=()=>{
  const development=$('wb-error-set').value==='stage5';
  if(development!==(store.get().documentSource==='stage5')){
    detail=null;store.set({currentDocument:null,documentSource:development?'stage5':'test'});renderErrorDocument();
  }
  const rows=renderErrorList(),s=store.get();
  if(rows&&s.currentDocument&&!rows.some(row=>row.document_id===s.currentDocument&&(!development||row.condition===s.currentCondition))){
    detail=null;store.set({currentDocument:null});renderErrorDocument();renderErrorList();
  }
};
$('wb-prediction-errors').onclick=()=>{selectTab('errors');renderErrorDocument();};
$('classification-text').oninput=()=>{liveResult=null;detail=null;store.set({currentDocument:null,documentSource:'test'});++predictEpoch;refreshEvidence();};
$('classification-predict').onclick=async()=>{
  const s=store.get(),text=$('classification-text').value;
  if(detail&&text===(activeShortcut()?.text||detail.document.text)){liveResult=null;refreshEvidence();return;}
  if(!text.trim()||text.length>5000){message('请输入1–5,000字新闻文本；冻结全文可直接查看保存预测。');return;}
  const epoch=++predictEpoch;$('classification-predict').disabled=true;message();
  try{const result=await request(s.currentModel==='C3'?'stage3/predict':s.currentModel==='C2'?'stage2/predict':'predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,model_id:s.currentModel,run_id:s.currentRun})});if(epoch!==predictEpoch||s.currentModel!==store.get().currentModel||s.currentRun!==store.get().currentRun)return;liveResult=result;refreshEvidence();}catch(error){if(epoch===predictEpoch)message(error.message);}finally{if(epoch===predictEpoch)$('classification-predict').disabled=false;}
};
async function legacyRecords(){
  if(loadedLegacy)return;loadedLegacy=true;
  const [{initializeStage4},{initializeStage5},{initializeStage6}]=await Promise.all([import('./classification-stage4.js'),import('./classification-stage5.js'),import('./classification-stage6.js')]);
  await Promise.all([initializeStage4(false),initializeStage5(false),initializeStage6(false)]);header();
}
$('wb-comparison-records').addEventListener('toggle',()=>{if($('wb-comparison-records').open)legacyRecords().catch(error=>message(error.message));});
function renderSavedSamples(){
  if(!$('wb-error-records').open||!overview)return;
  const filter=$('classification-sample-filter').value,model=filter.startsWith('c3')||filter.includes('_c3_')?'C3':['c2_errors','disagreements','corrected_c1','new_errors'].includes(filter)?'C2':store.get().currentModel;
  const all=Object.values(predictions[model]||{});
  if(model==='C3'){renderC3Samples(overview,all,filter,ui);return;}
  const old=predictions.C1||{};
  const rows=all.filter(r=>filter==='all'?true:filter==='disagreements'?old[r.document_id]?.prediction!==r.prediction:filter==='corrected_c1'?r.correct&&!old[r.document_id]?.correct:filter==='new_errors'?!r.correct&&old[r.document_id]?.correct:!r.correct);
  $('classification-error-summary').textContent=`${model} · 完整保存 ${all.length} 篇 Test · 当前 ${rows.length} 篇`;
  $('classification-samples').innerHTML=rows.map(r=>`<details><summary>${escape(r.document_id)} · ${label(r.gold)} → ${label(r.prediction)}</summary><p>${escape(r.text||r.text_snippet)}</p><p>第二名 ${label(r.runner_up)} · ${model==='C0'?'posterior':'decision score'} ${model==='C0'?pct(r.model_score):number(r.model_score)}</p>${table(['类别',model==='C0'?'posterior':'decision score'],r.scores.map(score=>[label(score.label),model==='C0'?pct(score.value):number(score.value)]))}${table(['特征','TF-IDF','权重','贡献'],r.top_contributing_features.map(f=>[f.term,number(f.tfidf),number(f.weight),number(f.contribution)]))}</details>`).join('')||'<p class="muted">这个集合没有样本。</p>';
}
$('classification-sample-filter').onchange=renderSavedSamples;
$('wb-error-records').addEventListener('toggle',async()=>{if(!$('wb-error-records').open)return;await legacyRecords();renderSavedSamples();});
document.addEventListener('module-tab',event=>{if(event.detail==='errors'){renderErrorList();renderErrorDocument();}});
async function initialize(){
  if(allowedAnalysis){
    selectTab(analysisOnly==='stage5'&&location.hash==='#errors'?'errors':'comparison');for(const button of document.querySelectorAll('[data-tab]'))button.disabled=button.dataset.tab!=='comparison'&&!(analysisOnly==='stage5'&&button.dataset.tab==='errors');
    document.body.dataset.workbenchAnalysisOnly='true';$('classification-run').innerHTML='<option>独立只读分析</option>';$('classification-model').disabled=true;
    if(analysisOnly==='stage5'){
      overview={available_models:['C0','C1','C2','C3']};
      $('classification-model').disabled=false;$('wb-error-set').value='stage5';
      for(const option of $('wb-error-set').options)option.disabled=option.value!=='stage5';
      $('wb-error-records').hidden=true;
      settlingRun=true;store.set({currentModel:store.get().currentModel||'C3',documentSource:'stage5'});settlingRun=false;syncControls();
      if(store.get().currentDocument)await loadDocument();
    }
    $('wb-comparison-records').hidden=true;header();return;
  }
  try{const runs=await request('runs');if(!runs.length)throw new Error('没有可展示的正式分类 Run。');$('classification-run').innerHTML=runs.map(r=>`<option value="${escape(r.run_id)}">${escape(r.run_id)}</option>`).join('');const state=store.get();const chosen=runs.some(r=>r.run_id===state.currentRun)?state.currentRun:runs[0].run_id;if(chosen===state.currentRun)await loadRun();else store.set({currentRun:chosen});}catch(error){message(error.message);}
}
initialize();
