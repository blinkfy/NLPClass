import {MODEL_COLORS, MODEL_SCORE_TYPES} from './classification-model-colors.js';
import {normalizedSource} from './classification-algorithm-explorer.js';

const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={finance:'财经',education:'教育',science:'科技',sports:'体育',entertainment:'娱乐'};
const lab=x=>names[x]||x||'—',num=x=>x==null?'未保存':Number(x).toFixed(4);
const signed=x=>`${x>0?'+':''}${num(x)}`;
const conditions={OO:'原始输入',OS:'去除来源',OT:'去除模板',OA:'全部清理',AA:'清理后重新训练'};
const table=rows=>`<table class="df-table">${rows.map(([k,v])=>`<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join('')}</table>`;

export function codepointRanges(text,ranges){
  const chars=Array.from(text),map=[0];for(const c of chars)map.push(map.at(-1)+c.length);
  return ranges.map(r=>({start:map[r.start??r[0]],end:map[r.end??r[1]]})).filter(r=>Number.isInteger(r.start)&&Number.isInteger(r.end)&&r.end>r.start);
}
export function featureOccurrences(text,term,segments=null){
  const mapped=normalizedSource(text);if(!mapped.offsets)return [];
  const found=[];for(const part of term.split(' ').filter(Boolean)){
    const key=part.normalize('NFKC').toLowerCase();let at=0,count=0;
    while((at=mapped.normalized.indexOf(key,at))>=0&&count++<12){const r={start:mapped.offsets[at].start,end:mapped.offsets[at+key.length-1].end};if(!segments||segments.some(s=>r.start>=s.start&&r.end<=s.end))found.push(r);at+=key.length;}
  }return found;
}
export const regionsOverlap=(a,b)=>a.some(x=>b.some(y=>x.start<y.end&&y.start<x.end));

export function createErrorDiff({store,request,onOpenPrediction}){
  let context=null,lanes=[],epoch=0,key=null;
  const cache=new Map(),main=document.getElementById('wb-error-document'),inspector=document.getElementById('wb-error-inspector');
  const mobileSummary=document.createElement('div');mobileSummary.className='df-mobile-summary';mobileSummary.hidden=true;mobileSummary.setAttribute('role','status');mobileSummary.setAttribute('aria-label','当前文档判断摘要');main.parentElement.prepend(mobileSummary);
  const get=async path=>{if(!cache.has(path)){cache.set(path,request(path).catch(e=>{cache.delete(path);throw e;}));if(cache.size>24)cache.delete(cache.keys().next().value);}return cache.get(path);};
  const state=()=>store.get(),selection=()=>state().selectedEvidence;
  function clear(message='选择左侧冻结文档。'){++epoch;context=null;key=null;lanes=[];mobileSummary.hidden=true;main.innerHTML=`<h2>判别差异</h2><p>${esc(message)}</p>`;inspector.innerHTML='<h2>判断检查器</h2><p>当前没有已加载的证据对象。</p>';}
  function makeLane(model,record,role,linear,condition){
    const text=context.shortcut?.document.text||context.detail.document.text;
    const view=context.shortcut?.conditions[condition];
    const segments=view?codepointRanges(text,view.segments):null;
    let rows;
    if(model==='C3')rows=(record.top_convolution_spans||[]).map(r=>({...r,kind:'channel',term:r.token_span.join(' '),value:r.contribution,metric:'预测类 raw logit 贡献',ranges:context.shortcut?codepointRanges(text,r.character_offsets||[]):r.character_offsets||[]}));
    else rows=(linear?.features||record.top_contributing_features||[]).map(r=>({...r,kind:'feature',value:r.relative_contribution??r.runner_up_contrast??r.contribution,metric:model==='C0'?'相对第二名 joint log-score 贡献':'预测类 − 第二名 margin 贡献',ranges:featureOccurrences(text,r.term,segments)}));
    rows.sort((a,b)=>b.value-a.value||a.term.localeCompare(b.term));
    return {model,role,condition,record,rows:rows.slice(0,8),allRows:rows,linear,view};
  }
  function findSelected(lane,s){return s?.model===lane.model&&(!context.shortcut||s.condition===lane.condition)?lane.allRows.find(r=>r.kind===s.kind&&(r.kind==='channel'?r.channel===s.channel:r.term===s.term)):null;}
  function choose(lane,row){
    const s={kind:row.kind,model:lane.model,documentId:context.detail.document.document_id,origin:'error-diff',condition:lane.condition,targetClass:lane.record.prediction,runnerUpClass:lane.record.runner_up};
    if(row.kind==='channel')Object.assign(s,{channel:row.channel,kernel:row.kernel_size,filter:row.filter_id,position:row.argmax_position,decomposition:'predicted'});
    else Object.assign(s,{term:row.term,column:row.column??null});
    store.set({selectedEvidence:s});
  }
  async function update(next){
    if(!next?.detail?.document){clear();return;}
    const s=state(),identity=[s.currentRun,s.currentDocument,s.currentModel,s.currentComparison,s.currentCondition,s.documentSource].join(':');
    if(identity===key&&context){render();return;}
    key=identity;context=next;lanes=[];mobileSummary.hidden=true;const version=++epoch;
    main.innerHTML='<h2>判别差异</h2><p>正在读取同一文档的冻结证据…</p>';inspector.innerHTML='<h2>判断检查器</h2><p>正在加载当前对象。</p>';
    try{
      if(s.documentSource==='stage5'){
        if(!next.shortcut)throw new Error('Stage5 单文档资产尚未加载');
        const condition=s.currentCondition||'OO',current=next.shortcut.conditions[condition];if(!current)throw new Error('不支持的冻结条件');
        lanes=[makeLane(s.currentModel,next.shortcut.conditions.OO.record,'A',null,'OO'),makeLane(s.currentModel,current.record,'B',null,condition)];
      }else{
        const models=[s.currentModel,next.other].filter(id=>next.detail.records[id]);
        const values=await Promise.all(models.map(id=>['C1','C2'].includes(id)?get(`visualization/decision?${new URLSearchParams({run_id:s.currentRun,document_id:s.currentDocument,model_id:id})}`):null));
        if(version!==epoch||next!==context)return;
        lanes=models.map((id,i)=>makeLane(id,next.records[id],i?'B':'A',values[i],null));
      }
      if(version!==epoch||next!==context)return;
      if(!lanes.some(l=>findSelected(l,selection()))){const lane=context.shortcut?lanes[1]:lanes[0];if(lane.rows.length)choose(lane,lane.rows[0]);else render();}else render();
    }catch(e){if(version===epoch){lanes=[];main.innerHTML=`<h2>无法读取判别差异</h2><p>${esc(e.message)}</p>`;inspector.innerHTML='<h2>诊断未通过</h2><p>不展示替代或推测的证据。</p>';}}
  }
  function rowMarkup(lane,row,index){
    const s=selection(),active=findSelected(lane,s)===row,max=Math.max(...lane.rows.map(r=>Math.abs(r.value)),.001);
    const peers=lanes.filter(l=>l!==lane).flatMap(l=>l.rows).filter(r=>regionsOverlap(row.ranges,r.ranges));
    return `<button class="df-evidence ${active?'is-selected':''}" data-df-role="${lane.role}" data-df-row="${index}" aria-pressed="${active}"><span>${esc(row.term)}${context.shortcut&&state().currentCondition==='OO'?'<small>原始输入中的判别证据</small>':peers.length?'<small>另一侧 Top 证据区域有重合</small>':'<small>当前另一侧 Top 证据无可定位重合</small>'}</span><i><b style="width:${Math.abs(row.value)/max*100}%" class="${row.value<0?'negative':''}"></b></i><strong>${signed(row.value)}</strong></button>`;
  }
  function laneMarkup(lane){
    const selected=findSelected(lane,selection()),rows=lane.rows.slice();if(selected&&!rows.includes(selected))rows.push(selected);
    lane.displayed=rows;
    const r=lane.record,prob=['C0','C3'].includes(lane.model),score=prob?`${(r.model_score*100).toFixed(2)}%`:num(r.model_score);
    const title=context.shortcut?`${lane.condition} · ${conditions[lane.condition]} · ${lane.view.model_identity==='clean-trained'?'清理后训练模型':'原模型'}`:`${lane.role} · ${lane.model}`;
    return `<section class="df-lane" style="--df-model:${MODEL_COLORS[lane.model]}"><button class="df-lane-title" data-df-model="${lane.role}">${esc(title)} · <strong>${lab(r.prediction)}</strong></button><p>${MODEL_SCORE_TYPES[lane.model]} ${score} · 第二名 ${lab(r.runner_up)}</p><div class="df-lane-path">${lane.model==='C3'?'HMM tokens → Embedding → Conv/max pool':lane.model==='C2'?'char bigram → χ² → TF-IDF → margin':'word uni+bi → TF-IDF → '+(lane.model==='C0'?'NB':'margin')}</div><p class="df-measure">${rows[0]?.metric||'没有保存的证据'}${context.shortcut?' · 仅在已保存 Top 证据内排序':''}</p><div class="df-evidence-lane">${rows.map((r,i)=>rowMarkup(lane,r,i)).join('')}</div></section>`;
  }
  function bodyMarkup(text){
    const s=selection(),removed=context.shortcut?.conditions[s?.condition||state().currentCondition]?.removed_spans||[];
    const deleting=codepointRanges(text,context.shortcut?.conditions[state().currentCondition]?.removed_spans||[]);
    const candidates=codepointRanges(text,context.shortcut?.all_removed_spans||[]);
    const selectedLane=lanes.find(l=>findSelected(l,s)),selected=selectedLane&&findSelected(selectedLane,s);
    let visible=[];
    for(const l of (context.shortcut&&state().currentCondition==='OO'?lanes.slice(0,1):lanes)){const row=findSelected(l,s)||l.rows[0];if(row)visible.push(...row.ranges.map(r=>({...r,role:l.role})));}
    if(selected&&!(context.shortcut&&state().currentCondition==='OO'))for(const l of lanes.filter(l=>l!==selectedLane))for(const row of l.rows.filter(r=>regionsOverlap(selected.ranges,r.ranges)).slice(0,2))visible.push(...row.ranges.map(r=>({...r,role:l.role})));
    const ranges=[...visible,...candidates,...deleting],points=[0,text.length,...ranges.flatMap(r=>[r.start,r.end])].filter(p=>Number.isInteger(p)&&p>=0&&p<=text.length).sort((a,b)=>a-b),parts=[];
    for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i];if(a===b)continue;const active=visible.filter(r=>r.start<b&&r.end>a),deleted=deleting.some(r=>r.start<b&&r.end>a),candidate=candidates.some(r=>r.start<b&&r.end>a),roles=[...new Set(active.map(r=>r.role))];
      const cls=deleted?'df-deleted':roles.length>1?'df-overlap':roles.length?'df-highlight-'+roles[0]:candidate?'df-shortcut-candidate':'';
      parts.push(cls?`<mark class="${cls}" data-df-source="${roles[0]||''}" ${deleted||candidate?`data-df-removed="${a}"`:''}>${esc(text.slice(a,b))}</mark>`:esc(text.slice(a,b)));
    }return parts.join('');
  }
  function render(){
    if(!context||!lanes.length)return;
    const doc=context.shortcut?.document||context.detail.document,s=selection(),shortcut=!!context.shortcut,condition=state().currentCondition||'OO',current=shortcut&&context.shortcut.conditions[condition];
    const selectedLane=lanes.find(l=>findSelected(l,s))||lanes[0],selected=findSelected(selectedLane,s)||selectedLane.rows[0];
    const originalOnly=shortcut&&condition==='OO',shownLanes=originalOnly?lanes.slice(0,1):lanes;
    mobileSummary.hidden=false;mobileSummary.innerHTML=`<span><small>${'A · '+lanes[0].model+(shortcut?' · OO':'')}</small><strong style="color:${MODEL_COLORS[lanes[0].model]}">${lab(lanes[0].record.prediction)}</strong></span><span><small>${'B · '+(lanes[1]?.model||'—')+(shortcut?' · '+condition:'')}</small><strong style="color:${MODEL_COLORS[lanes[1]?.model]||'#5b7483'}">${originalOnly?'尚未消融':lab(lanes[1]?.record.prediction)}</strong></span><span><small>Gold · 真实</small><strong>${lab(doc.gold)}</strong></span>`;
    main.style.setProperty('--df-a',MODEL_COLORS[lanes[0].model]);main.style.setProperty('--df-b',MODEL_COLORS[lanes[1]?.model]||MODEL_COLORS[lanes[0].model]);
    const overlap=lanes[0].rows.filter(r=>lanes[1]?.rows.some(q=>regionsOverlap(r.ranges,q.ranges)));
    main.innerHTML=`<div class="wb-object-heading"><div><h2>${shortcut?'单文档来源 / 模板消融':'A vs B · 判别差异'}</h2><p class="df-document-id">${esc(doc.document_id)} · Gold ${lab(doc.gold)}</p></div><button data-df-open class="outline">在分类预测中查看 →</button></div>${shortcut?`<p class="df-context">Development-set diagnostic / Validation only · 冻结 NFKC/lowercase 文本</p><div class="df-conditions">${Object.entries(conditions).map(([c,l])=>`<button data-df-condition="${c}" class="${condition===c?'is-selected':''}">${c} · ${l}</button>`).join('')}</div><div class="df-transition">${originalOnly?`原始预测：<strong>${lab(lanes[0].record.prediction)}</strong> · 基线输入，尚未消融`:`${lab(lanes[0].record.prediction)} → <strong>${lab(lanes[1].record.prediction)}</strong> · ${lanes[0].record.prediction===lanes[1].record.prediction?'预测未改变':'预测已改变'}`}</div><p class="df-context">${condition==='AA'?'AA：清理后重新训练的模型 + 全部清理输入；模型已不同。':originalOnly?'OO：原模型 + 原始输入。选择清理条件后，对照原始与当前证据。':'OO：原模型 + 原始输入；OS/OT/OA：同一原模型 + 清理输入。'}</p>`:`<p class="df-context">${lanes.map(l=>`${l.model}：${lab(l.record.prediction)} / 第二名 ${lab(l.record.runner_up)}`).join('　↔　')}</p>`}<article class="wb-news df-news">${bodyMarkup(doc.text)}</article><div class="df-legend">${originalOnly?'<span>原始证据</span>':'<span>A evidence</span><span>B evidence</span><span>重合区域</span>'}${shortcut?'<span>虚线：冻结 shortcut 区域 · 删除线：当前条件实际删除</span>':''}</div><div class="df-lanes${originalOnly?' is-original-only':''}">${shownLanes.map(laneMarkup).join('')}</div><p class="df-context">${originalOnly?'当前仅显示原始证据；选择清理条件后开启前后对照。':'条长在各侧独立缩放，仅比较各模型内部贡献。'}${!shortcut&&lanes.length===2&&MODEL_SCORE_TYPES[lanes[0].model]!==MODEL_SCORE_TYPES[lanes[1].model]?'分数类型不同，仅用于各模型内部判别，不直接比较数值大小。':''}</p>${originalOnly?'':`<div class="df-overlap-summary">可定位 Top 证据区域重合：${overlap.length} 项；A 独有区域 ${lanes[0].rows.length-overlap.length} 项；B 独有区域 ${lanes[1]?lanes[1].rows.filter(r=>!lanes[0].rows.some(q=>regionsOverlap(r.ranges,q.ranges))).length:0} 项。<small>“独有”只指当前展示的 Top 证据；区域重合不证明语义或权重相同。</small></div>`}<details><summary>${originalOnly?'完整保存的原始证据和类别分数':'完整保存的 A / B 证据和类别分数'}</summary>${shownLanes.map(l=>`<h3>${l.condition||l.role} · ${l.model}</h3>${table(l.record.scores.map(r=>[lab(r.label),`${num(r.value)}${r.raw_logit!=null?' · logit '+num(r.raw_logit):''}`]))}<div class="df-table-scroll">${table(l.allRows.map(r=>[r.term,`${r.metric} ${num(r.value)} · TF-IDF ${num(r.tfidf)}`]))}</div>`).join('')}</details>`;
    inspector.innerHTML=inspectorMarkup(selectedLane,selected,current);
    main.onclick=e=>{
      const evidence=e.target.closest('[data-df-row]'),model=e.target.closest('[data-df-model]'),cond=e.target.closest('[data-df-condition]');
      if(evidence){const l=lanes.find(l=>l.role===evidence.dataset.dfRole);choose(l,l.displayed[Number(evidence.dataset.dfRow)]);}
      else if(model){const l=lanes.find(l=>l.role===model.dataset.dfModel);if(l.rows[0])choose(l,l.rows[0]);}
      else if(cond)store.set({currentCondition:cond.dataset.dfCondition,selectedEvidence:null});
      else if(e.target.closest('[data-df-open]'))onOpenPrediction(s,shortcut);
      else if(e.target.closest('[data-df-removed]')){const start=Number(e.target.closest('[data-df-removed]').dataset.dfRemoved),span=context.shortcut.all_removed_spans.find(r=>{const range=codepointRanges(doc.text,[r])[0];return start>=range.start&&start<range.end;});if(span){store.set({selectedEvidence:{kind:'removed',model:state().currentModel,condition,documentId:doc.document_id,span,origin:'error-diff'}});}}
      else if(e.target.closest('[data-df-source]')){const l=lanes.find(l=>l.role===e.target.closest('[data-df-source]').dataset.dfSource);if(l?.rows[0])choose(l,findSelected(l,s)||l.rows[0]);}
    };
    const article=main.querySelector('.df-news'),mark=article.querySelector(s?.kind==='removed'?'[data-df-removed]':'[data-df-source]');if(mark)article.scrollTop+=mark.getBoundingClientRect().top-article.getBoundingClientRect().top-article.clientHeight/3;
  }
  function inspectorMarkup(lane,row,current){
    const s=selection(),shortcut=!!context.shortcut,other=lanes.find(l=>l!==lane),peers=row?other?.rows.filter(r=>regionsOverlap(row.ranges,r.ranges))||[]:[];
    const matchedChannel=shortcut&&lane.model==='C3'&&current.model_identity==='original'&&state().currentCondition!=='OO'&&row?.kind==='channel'?other.record.all_channels?.find(r=>r.channel===row.channel):null;
    const removed=shortcut?(s?.kind==='removed'?[s.span]:current.removed_spans):[];
    return `<div class="wb-inspector-heading"><span class="wb-inspector-kicker">${shortcut?'Shortcut Diff':'Decision Diff'} · ${lane.condition||lane.role} · ${lane.model}</span><h3 class="wb-object-title">${esc(s?.kind==='removed'?s.span.text:row?.term||'无保存证据')}</h3></div><p class="df-context">${lab(lane.record.prediction)} · 第二名 ${lab(lane.record.runner_up)} · ${MODEL_SCORE_TYPES[lane.model]}</p>${s?.kind==='removed'?'':row?`<div class="df-selected-metric"><span>${row.metric}</span><strong>${signed(row.value)}</strong></div>${table(row.kind==='channel'?[['kernel / filter',`${row.kernel_size} / ${row.filter_id}`],['max activation',num(row.max_activation)],['argmax position',row.argmax_position],['预测类 weight',num(row.predicted_class_weight)],['原词',row.token_span.join(' ')],['模型实际 token',row.model_token_span.join(' ')]]:[['TF-IDF',num(row.tfidf)],['vocab index',row.column??'未保存'],['预测类 weight',num(row.predicted_weight??row.weight)],['第二名 weight',num(row.runner_up_weight)],['direct contribution',num(row.direct_contribution)],['相对贡献',num(row.value)]])}`:'<p>未保存可展示的解释。</p>'}${shortcut&&state().currentCondition==='OO'?'':`<p class="df-context">另一侧相关 Top evidence：${peers.length?peers.map(p=>esc(p.term)).join('、'):'未发现可定位的重合区域'}</p>`}${matchedChannel?`<section class="df-channel-transition"><h3>同一原模型 · channel ${row.channel}</h3>${table([['当前侧最大响应',`${num(row.max_activation)} · p${row.argmax_position} · ${row.token_span.join(' ')}`],['另一条件最大响应',`${num(matchedChannel.max_activation)} · p${matchedChannel.argmax_position} · ${matchedChannel.token_span.join(' ')}`]])}<p>位置属于各自输入序列；原文位置以保存的 offsets 为准。</p></section>`:''}${shortcut?`<div class="df-shortcut-inspector"><h3>${state().currentCondition} · ${conditions[state().currentCondition]} · ${current.model_identity==='clean-trained'?'重新训练后的 clean model':'同一 Original model'}</h3><p>${state().currentCondition==='OO'?`原始预测 ${lab(lanes[0].record.prediction)} · 尚未消融`:`${lab(lanes[0].record.prediction)} → ${lab(lanes[1].record.prediction)} · ${lanes[0].record.prediction===lanes[1].record.prediction?'未改变':'已改变'}`}</p>${removed.length?removed.slice(0,4).map(r=>`<div class="df-removed-detail"><small>${esc(r.kind)} · ${esc(r.rule)} · [${r.start}, ${r.end})</small><p>${esc(r.text)}</p></div>`).join(''):'<p>当前条件没有删除区域。</p>'}${current.model_identity==='clean-trained'?'<p>AA 已更换模型；不把其 filter 编号与原模型视作同一 channel。</p>':''}</div>`:''}<details class="wb-inspector-details"><summary>完整对象与冻结来源</summary>${table([['工作台 Run',state().currentRun||'独立分析视图'],['文档来源 Run',context.detail.document.source_run],['模型 / 条件',`${lane.model} · ${lane.condition||'formal Test'}`],['模型来源',lane.view?.model_reference||lane.linear?.source_run_id||context.modelSources?.[lane.model]?.runId||'未提供'],['Gold',lab(context.detail.document.gold)]])}<pre class="df-json">${esc(JSON.stringify(row||s,null,2))}</pre></details>`;
  }
  store.subscribe((next,old)=>{if(['currentRun','currentDocument','currentModel','currentComparison','currentCondition','documentSource'].some(k=>next[k]!==old[k])){clear('正在切换当前对象…');return;}if(next.selectedEvidence!==old.selectedEvidence&&context&&lanes.length)render();});
  return {update,clear};
}
