import {NEWS_NAMES, CLASS_COLORS} from './comprehensive-workspace.js';

const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>Number.isFinite(v)?v.toFixed(4):'未提供';
const hit=(a,b)=>a&&b&&a.start<b.end&&b.start<a.end;
const valid=(s,n)=>Number.isInteger(s?.start)&&Number.isInteger(s?.end)&&s.start>=0&&s.end>s.start&&s.end<=n;
const kv=rows=>`<dl class="cd-kv">${rows.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`;
const source=v=>typeof v==='string'?v:JSON.stringify(v??'未提供');
const types={PER:'人名',LOC:'地点',ORG:'机构'};

/* This view only joins annotations by original Unicode code-point offsets.
   No model calls, tokenization, classification or statistical evaluation here. */
export function createDocumentView({workspace,host,inspector,onAnalyze,onContextAction,onEntityFilter,onNavigate,onSourceLink}) {
  let closed=false;
  const listeners=[];
  const listen=(el,event,fn)=>{el.addEventListener(event,fn);listeners.push(()=>el.removeEventListener(event,fn));};
  function annotations(profile) {
    const a=profile?.analysis,n=Array.from(profile?.original_text||'').length;
    return ['tokens','pos','entities'].flatMap(kind=>(a?.lexical?.[kind]||[]).filter(s=>valid(s,n)).map(s=>({...s,kind})));
  }
  function currentSpan(profile) {
    const state=workspace.getState();return state.span?.documentKey===workspace.documentKey()?state.span:null;
  }
  function evidenceRows(profile) {
    return (Array.isArray(profile.analysis?.classification?.evidence)?profile.analysis.classification.evidence:[])
      .filter(row=>Number.isFinite(row.relative_contribution)&&row.relative_contribution>0);
  }
  function evidenceOccurrences(row,profile) {
    const chars=Array.from(profile.original_text||'');
    return (Array.isArray(row?.occurrences)?row.occurrences:[]).filter(o=>valid(o,chars.length)&&(!o.text||chars.slice(o.start,o.end).join('')===o.text));
  }
  function selectedRanges(profile) {
    const focus=workspace.getState().focus,span=currentSpan(profile);
    return focus?.kind==='classification-evidence'?evidenceOccurrences(focus,profile):span?[span]:[];
  }
  function marked(s,profile) {
    const state=workspace.getState(),span=currentSpan(profile);
    if(selectedRanges(profile).some(range=>hit(s,range)))return true;
    if(state.focus?.kind==='pos')return s.kind==='pos'&&s.pos===state.focus.tag;
    if(state.focus?.kind==='entity')return s.kind==='entities'&&s.type===state.focus.type;
    return false;
  }
  function textHTML(profile) {
    const chars=Array.from(profile.original_text||''),state=workspace.getState(),all=annotations(profile),span=currentSpan(profile);
    const visible=all.filter(s=>state.layers[s.kind==='entities'?'entities':s.kind]),ranges=selectedRanges(profile);
    const bounds=[...new Set([0,chars.length,...visible.flatMap(s=>[s.start,s.end]),...ranges.flatMap(range=>[range.start,range.end])])].sort((a,b)=>a-b);
    return bounds.slice(0,-1).map((start,i)=>{
      const end=bounds[i+1],value=chars.slice(start,end).join(''),segment={start,end};
      const rows=visible.filter(s=>hit(s,segment)),entity=rows.find(s=>s.kind==='entities'),pos=rows.find(s=>s.kind==='pos');
      const selected=state.focus?.kind==='classification-evidence'?ranges.some(range=>hit(range,segment)):ranges.some(range=>hit(range,segment))||rows.some(s=>marked(s,profile));
      if(!rows.length&&!selected)return esc(value);
      return `<button type="button" data-cd-span="${start}:${end}" class="cd-fragment ${rows.some(s=>s.kind==='tokens')?'is-boundary':''} ${entity?'is-entity entity-'+esc(entity.type):''} ${selected?'is-selected':''}" aria-pressed="${Boolean(selected)}" title="${esc([pos?.pos,entity?.type,`[${start},${end})`].filter(Boolean).join(' · '))}">${esc(value)}</button>`;
    }).join('');
  }
  function tokensHTML(profile) {
    const a=profile.analysis;
    if(a?.modules?.segmentation?.status!=='ok')return '<p class="cd-note">分词尚未完成，不能将未分析文本当作零词。</p>';
    return (a.lexical?.tokens||[]).filter(t=>!t.whitespace).map(t=>{
      const related=(a.lexical?.pos||[]).filter(p=>hit(t,p)),selected=marked({...t,kind:'tokens'},profile)||related.some(p=>marked({...p,kind:'pos'},profile));
      return `<button type="button" data-cd-span="${t.start}:${t.end}" class="cd-token ${selected?'is-selected':''}" aria-pressed="${selected}"><span>${esc(t.form)}</span><small>${esc(related.map(p=>p.pos).join(' / ')||'POS 未知')}</small></button>`;
    }).join('')||'<p class="cd-note">当前分词输出无非空白词项。</p>';
  }
  function entityHTML(profile) {
    if(profile.analysis?.modules?.ner?.status!=='ok')return '<p class="cd-note">NER 尚未完成。</p>';
    return (profile.analysis.lexical?.entities||[]).map((e,index)=>`<div class="cd-entity-row ${marked({...e,kind:'entities'},profile)?'is-selected':''}"><button type="button" data-cd-entity="${index}" aria-pressed="${marked({...e,kind:'entities'},profile)}"><b>${esc(e.text)}</b><small>${esc(e.type)} · [${e.start},${e.end})</small></button><button type="button" class="cd-filter" data-cd-filter-entity="${index}" aria-label="在语料中筛选${esc(e.text)}">筛选语料 ↗</button></div>`).join('')||'<p class="cd-note">已运行 NER，当前没有识别到实体。</p>';
  }
  function countBars(values,kind,profile) {
    const maximum=Math.max(1,...Object.values(values));
    return Object.entries(values).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).map(([label,count])=>`<button type="button" data-cd-count="${esc(kind+':'+label)}" class="cd-count ${workspace.getState().focus?.kind===kind&&(workspace.getState().focus?.tag||workspace.getState().focus?.type)===label?'is-selected':''}"><span>${esc(types[label]||label)}</span><i style="--count:${100*count/maximum}%"></i><b>${count}</b></button>`).join('')||'<p class="cd-note">实际计数为零。</p>';
  }
  function evidenceHTML(profile) {
    const c=profile.analysis?.classification,rows=evidenceRows(profile);
    if(!c)return '';
    if(!rows.length)return '<p class="cd-note">当前记录未提供正向相对贡献特征。</p>';
    const maximum=Math.max(...rows.map(row=>row.relative_contribution));
    return `<div class="cd-evidence"><h4>预测相对第二名的证据 <small>${esc(NEWS_NAMES[c.prediction]||c.prediction)} − ${esc(NEWS_NAMES[c.runner_up]||c.runner_up)}</small></h4>${rows.map((row,index)=>{
      const feature=row.feature||row.term,selected=workspace.getState().focus?.kind==='classification-evidence'&&workspace.getState().focus.feature===feature;
      return `<button type="button" data-cd-evidence="${index}" class="cd-evidence-row ${selected?'is-selected':''}" aria-pressed="${selected}"><span>${esc(feature)}</span><i><em style="width:${100*row.relative_contribution/maximum}%"></em></i><b>+${num(row.relative_contribution)}</b></button>`;
    }).join('')}<p class="cd-note">真实 char bigram 相对贡献；点击按提供的原文位置联动。仅展示正向 Top 特征，不代表完整 margin 分解。</p></div>`;
  }
  function profilesHTML(profile) {
    const a=profile.analysis,c=a?.classification;
    let scores=Array.isArray(c?.scores)?c.scores:Object.entries(c?.scores||{}).map(([label,value])=>({label,value}));
    scores=scores.map(s=>({label:s.label||s.class||s.class_id,value:s.value??s.score??s.decision_score})).filter(s=>Number.isFinite(s.value));
    const max=Math.max(...scores.map(s=>Math.abs(s.value)),1e-9);
    const scoreHTML=c&&scores.length?scores.map(s=>`<button type="button" data-cd-class="${esc(s.label)}" class="cd-score ${workspace.getState().focus?.label===s.label?'is-selected':''}"><span>${esc(NEWS_NAMES[s.label]||s.label)}</span><i class="cd-score-axis"><em style="left:${s.value<0?50-50*Math.abs(s.value)/max:50}%;width:${50*Math.abs(s.value)/max}%;background:${CLASS_COLORS[s.label]||'#168e98'}"></em></i><b>${num(s.value)}</b></button>`).join(''):'<p class="cd-note">当前没有可用分类数值。</p>';
    const pos={},ents={};for(const p of a?.lexical?.pos||[])pos[p.pos]=(pos[p.pos]||0)+1;for(const e of a?.lexical?.entities||[])ents[e.type]=(ents[e.type]||0)+1;
    return `<div class="cd-profile"><section><h3>五类新闻判别 <small>冻结 C2</small></h3>${scoreHTML}<p class="cd-note">中线 0 是超平面参考，不是概率阈值；LinearSVC decision score 不是概率。点击柱形查看整篇分数。</p>${evidenceHTML(profile)}</section><section><h3>词性分布 <small>POS 项数</small></h3>${a?.modules?.pos?.status==='ok'?countBars(pos,'pos',profile):'<p class="cd-note">POS 未知，等待分析。</p>'}</section><section><h3>实体组成 <small>提及次数</small></h3>${a?.modules?.ner?.status==='ok'?countBars(ents,'entity',profile):'<p class="cd-note">NER 未知，等待分析。</p>'}</section></div>`;
  }
  function dependencyHTML(profile) {
    const deep=workspace.getDeep()?.dependency,result=deep?.result,ranges=selectedRanges(profile);
    if(!result||!workspace.getState().layers.dependency)return '';
    const sentences=(result.sentences||[]).map((s,index)=>({...s,_index:index})).filter(s=>(s.items||[]).length);
    if(!sentences.length)return '<p class="cd-note">本次依存结果未包含可用词项；原始输出保留在下方。</p>';
    return `<section class="cd-syntax"><h3>当前文档句法层 <small>真实 head / relation · 与原文跨度对齐</small></h3><p class="cd-note">显示已显式运行的${deep.source_span?`原文 [${deep.source_span.start},${deep.source_span.end}) 选区`:'输入范围'}；不是自动全文句法。横向滚动查看较长句子；选中词项或实体后强化其 incoming / outgoing 关系，其余弧线淡化。</p>${sentences.map((s,si)=>{
      const items=s.items,width=Math.max(450,items.length*82+70),height=175;
      return `<div class="cd-syntax-scroll"><svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="group" aria-label="第${si+1}句真实依存关系"><text x="24" y="153" class="cd-svg-root">ROOT</text>${items.map((t,i)=>{
        const x=85+i*82,head=items.find(h=>h.id===t.head),hx=head?85+items.indexOf(head)*82:24,y=125,bend=34+Math.min(58,Math.abs(x-hx)*.2),selected=ranges.some(range=>hit(t,range)||hit(head,range)),relation=t.relation||t.deprel;
        return `<g tabindex="0" role="button" data-cd-dependency="${s._index}:${i}" class="cd-arc ${selected?'is-selected':ranges.length?'is-muted':''}" aria-pressed="${selected}" aria-label="${esc(t.form)}，head ${t.head}，${esc(relation)}"><path d="M ${hx} ${y} Q ${(hx+x)/2} ${y-bend*2} ${x} ${y}"/><path d="M ${x-4} ${y-7} L ${x} ${y} L ${x+4} ${y-7}" class="cd-arrow"/><text x="${(hx+x)/2}" y="${y-bend-5}">${esc(relation)}</text><text x="${x}" y="153">${esc(t.form)}</text></g>`;
      }).join('')}</svg></div>`;
    }).join('')}</section>`;
  }
  function deepHTML(profile) {
    const deep=workspace.getDeep(),ie=deep.information_extraction?.result,summary=deep.summary?.result;
    let html='';
    if(ie)html+=`<section class="cd-deep-result"><h3>信息抽取 <small>实际规则结果</small></h3><p class="cd-note">NER + 类型 / 触发词规则；仅并列观察，不构成因果链。</p>${ie.status==='unavailable'?`<p class="cd-note">${esc(ie.error||'信息抽取未完成；匹配数未知。')}</p>`:(ie.relations||[]).map((r,i)=>`<button type="button" data-cd-relation="${i}" class="cd-relation">${esc(r.subject?.text)} <small>— ${esc(r.predicate)} →</small> ${esc(r.object?.text)}<span>${esc(r.evidence?.text||'')}</span></button>`).join('')||'<p class="cd-note">当前结果没有匹配到支持的关系规则；不补写推断。</p>'}</section>`;
    if(summary)html+=`<section class="cd-deep-result"><h3>本地摘要 <small>原始生成，不自动修补</small></h3><p class="cd-summary-output">${esc(summary.response??summary.output??summary.generated_text??summary.summary??'结果未包含文本输出')}</p></section>`;
    return html;
  }
  function inspectorHTML(profile) {
    const a=profile.analysis,span=currentSpan(profile),state=workspace.getState(),all=annotations(profile);
    let body='';
    if(state.focus?.kind==='classification-evidence'){
      const row=state.focus.evidence||state.focus,c=a?.classification,occurrences=evidenceOccurrences(state.focus,profile);
      const fields=[['预测 / 第二名',`${NEWS_NAMES[c?.prediction]||c?.prediction||'未知'} / ${NEWS_NAMES[c?.runner_up]||c?.runner_up||'未知'}`],['相对贡献',num(row.relative_contribution)]];
      for(const [key,label] of [['tfidf','TF-IDF'],['coefficient_difference','预测 − 第二名系数'],['weight','预测类权重'],['direct_contribution','预测类直接贡献'],['chi2_score','χ² relevance score']])if(Number.isFinite(row[key]))fields.push([label,num(row[key])]);
      if(typeof row.in_raw_vocabulary==='boolean')fields.push(['原 Count 词表',row.in_raw_vocabulary?'存在':'未进入']);
      if(typeof row.retained==='boolean')fields.push(['χ² 后保留',row.retained?'是':'否']);
      if(Number.isFinite(row.count))fields.push(['本篇 Count',row.count]);
      fields.push(['原文定位',occurrences.length?`${occurrences.length} 处 · ${occurrences.map(o=>`[${o.start}, ${o.end})`).join(' / ')}`:'无稳定原文映射']);
      body=`<span class="cd-eyebrow">当前对象 · C2 分类证据</span><h2>${esc(state.focus.feature)}</h2>${kv(fields)}<p class="cd-note">TF-IDF ×（预测类系数 − 第二名系数）支持两类之间的相对判别。${occurrences.length?'全部已提供的真实 occurrence 同步高亮；位置采用原文 Unicode 字符跨度。':'该特征没有稳定原文 offset / occurrence，保留真实数值，不定位或猜测原文。'}${Number.isFinite(row.chi2_score)?' χ² 是冻结训练 Count 的关联排序量，不是本文贡献或显著性结论。':''}</p>`;
    }else if(state.focus?.kind==='classification'){
      const c=a?.classification,label=state.focus.label;
      body=`<span class="cd-eyebrow">当前对象 · 整篇新闻判别</span><h2>${esc(NEWS_NAMES[label]||label)}</h2>${kv([['预测',NEWS_NAMES[c?.prediction]||c?.prediction||'未知'],['第二名',NEWS_NAMES[c?.runner_up]||c?.runner_up||'未知'],['相对 margin',num(c?.score_gap)],['分数类型','LinearSVC decision score']])}<p class="cd-note">当前柱是整篇分数，不是选中词项的贡献，也不是校准概率。</p>`;
    }else if(span){
      const value=Array.from(profile.original_text).slice(span.start,span.end).join(''),rows=all.filter(s=>hit(s,span));
      body=`<span class="cd-eyebrow">当前对象 · 原文片段</span><h2>${esc(value)}</h2>${kv([['Unicode 位置',`[${span.start}, ${span.end})`],...rows.map(s=>[s.kind==='tokens'?'分词':s.kind==='pos'?'词性':'实体',s.kind==='tokens'?s.form:s.kind==='pos'?`${s.word} / ${s.pos}`:`${s.text} / ${s.type}`])])}`;
      const syntax=workspace.getDeep().dependency?.result?.sentences||[];
      const deps=syntax.flatMap(s=>(s.items||[]).filter(t=>hit(t,span)).map(t=>{const head=s.items.find(h=>h.id===t.head);return ['依存',`${t.form} ← ${head?.form||'ROOT'} / ${t.relation||t.deprel}`];}));
      if(deps.length)body+=kv(deps);
      body+='<p class="cd-note">各模块保留自己的输入协议；当前关联仅由原文位置相交确定。</p>';
    }else if(state.focus?.kind==='pos'||state.focus?.kind==='entity'){
      const f=state.focus,key=f.tag||f.type;
      body=`<span class="cd-eyebrow">当前对象 · ${f.kind==='pos'?'词性':'实体类型'}</span><h2>${esc(types[key]||key)}</h2><p class="cd-note">原文与下方词项同步标出当前类型。点击具体词项进一步钻取。</p>`;
    }else body=`<span class="cd-eyebrow">当前文档</span><h2>${esc(profile.title||profile.id)}</h2>${kv([['原文长度',`${Array.from(profile.original_text||'').length} 字符`],['范围',profile.eligible?'可按需分析':'仅课程历史摘要']])}<p class="cd-note">点击原文、词项或实体；结果尚未完成时保持未知状态。</p>`;
    if(state.focus?.kind==='classification-evidence'&&span){
      const rows=all.filter(s=>selectedRanges(profile).some(range=>hit(s,range)));
      if(rows.length)body+=kv(rows.map(s=>[s.kind==='tokens'?'分词':s.kind==='pos'?'词性':'实体',s.kind==='tokens'?s.form:s.kind==='pos'?`${s.word} / ${s.pos}`:`${s.text} / ${s.type}`]));
      const deps=(workspace.getDeep().dependency?.result?.sentences||[]).flatMap(s=>(s.items||[]).filter(t=>selectedRanges(profile).some(range=>hit(t,range))).map(t=>{const head=s.items.find(h=>h.id===t.head);return ['依存',`${t.form} ← ${head?.form||'ROOT'} / ${t.relation||t.deprel}`];}));
      if(deps.length)body+=kv(deps);
    }
    const selectedEntities=all.filter(s=>s.kind==='entities'&&hit(s,span));
    body+=selectedEntities.map(e=>`<button type="button" data-cd-filter-value="${esc(JSON.stringify({type:e.type,text:e.text}))}" class="cd-context-link">在关联探索中筛选 ${esc(e.text)} ↗</button>`).join('');
    body+=`<div class="cd-context"><h3>当前对象 · 按需分析</h3><p class="cd-note">${span?'默认将选中对象扩展到所在句；运行前可查看和修改实际原文起止位置。':'当前文档，超限时请先选取片段。'} 不随基础分析自动执行。</p><div>${[['dependency','查看依存'],['information_extraction','信息抽取'],['summary','本地摘要']].map(([k,t])=>`<button type="button" data-cd-action="${k}" ${profile.eligible?'':'disabled'}>${t}</button>`).join('')}${span?'<button type="button" data-cd-clear>取消选区</button>':''}</div></div><div class="cd-links"><button data-cd-source="classification">分类方法 → 06</button><button data-cd-source="dependency">句法过程 → 07</button><button data-cd-source="summary">本地模型 → 08</button></div><details data-cd-details="full"><summary>完整数据与来源</summary><pre>${esc(JSON.stringify({provenance:a?.provenance,modules:a?.modules,clustering:profile.clustering,classification:a?.classification,deep:workspace.getDeep()},null,2))}</pre></details>`;
    return body;
  }
  function render() {
    if(closed)return;
    const active=document.activeElement,focused=[...active?.attributes||[]].find(a=>a.name.startsWith('data-cd-'));
    const activeRoot=inspector.contains(active)?inspector:host;
    const open=[...host.querySelectorAll('details[open]'),...inspector.querySelectorAll('details[open]')].map(d=>d.dataset.cdDetails);
    const profile=workspace.getDocument();
    if(!profile){host.innerHTML='<div class="cd-empty"><span class="cd-eyebrow">文档工作台</span><h2>同一份原文，多层真实标注</h2><p>输入文本后运行基础分析，或从语料空间选择一篇可分析文档。</p><p class="cd-note">分词 / POS / NER 保留各自协议，按原文字符位置联动；深层分析由你明确触发。</p></div>';inspector.innerHTML='<h2>当前对象</h2><p class="cd-note">等待文档。选择词项、实体或句法关系后，这里显示真实跨度与模块来源。</p>';return;}
    const a=profile.analysis,all=annotations(profile),state=workspace.getState();
    const current=currentSpan(profile);
    if(a&&!current&&!state.focus){const first=all.find(s=>s.kind==='entities')||all.find(s=>s.kind==='tokens'&&!s.whitespace);if(first){workspace.selectSpan({start:first.start,end:first.end},{kind:first.kind==='entities'?'span':'token'});return;}}
    host.innerHTML=`<div class="cd-document-heading"><div><span class="cd-eyebrow">原文 · ${esc(profile.corpus_id?'语料文档':'输入文档')}</span><h2>${esc(profile.title||'当前文档')}</h2></div>${!a||a.status!=='ok'?`<button type="button" data-cd-analyze ${profile.eligible?'':'disabled'}>${a?'重试未完成模块':'分析当前文档'}</button>`:''}</div><div class="cd-layers">${[['tokens','分词边界'],['pos','词性'],['entities','实体'],['dependency','句法层']].map(([key,label])=>`<label><input type="checkbox" data-cd-layer="${key}" ${state.layers[key]?'checked':''}>${label}</label>`).join('')}<span>Unicode 字符跨度 · 点击联动</span></div><div class="cd-annotation" aria-label="原文多层标注">${textHTML(profile)||'<span class="cd-note">原文为空。</span>'}</div>${dependencyHTML(profile)}<div class="cd-annotation-keys"><span>下划线：词边界</span><span>低饱和背景：实体</span><span>青绿框：当前对象 / 类型</span></div><div class="cd-object-lanes"><section><h3>词项与词性 <small>相交跨度对应，非强制同一 token</small></h3><div class="cd-token-lane">${tokensHTML(profile)}</div></section><section><h3>实体提及 <small>点击联动原文，并带入语料筛选</small></h3><div class="cd-entities">${entityHTML(profile)}</div><p class="cd-note">点击实体会在顶部上下文栏增加筛选条件；多个实体条件取交集，可逐个移除。切换关联探索后仍保留。</p></section></div>${profilesHTML(profile)}${deepHTML(profile)}`;
    inspector.innerHTML=inspectorHTML(profile);
    for(const key of open){for(const root of [host,inspector])root.querySelector(`[data-cd-details="${CSS.escape(key||'')}"]`)?.setAttribute('open','');}
    if(focused&&!active.isConnected)activeRoot.querySelector(`[${focused.name}="${CSS.escape(focused.value)}"]`)?.focus({preventScroll:true});
  }
  function click(event) {
    const node=event.target.closest('[data-cd-evidence],[data-cd-span],[data-cd-entity],[data-cd-filter-entity],[data-cd-count],[data-cd-class],[data-cd-dependency],[data-cd-relation],[data-cd-filter-value],[data-cd-action],[data-cd-source],[data-cd-analyze],[data-cd-clear]');
    if(!node)return;const p=workspace.getDocument();if(!p)return;
    if(node.hasAttribute('data-cd-evidence')){
      const evidence=evidenceRows(p)[Number(node.dataset.cdEvidence)];
      if(evidence){const occurrences=evidenceOccurrences(evidence,p);workspace.selectSpan(occurrences[0]||null,{kind:'classification-evidence',feature:evidence.feature||evidence.term,evidence,occurrences});}
    }
    else if(node.hasAttribute('data-cd-span')){
      const [start,end]=node.dataset.cdSpan.split(':').map(Number);
      const entity=node.matches('.cd-fragment')?(p.analysis?.lexical?.entities||[]).find(e=>valid(e,Array.from(p.original_text||'').length)&&e.start<=start&&e.end>=end):null;
      if(entity){workspace.selectSpan({start:entity.start,end:entity.end},{kind:'span'});onEntityFilter?.({type:entity.type,text:entity.text});}
      else workspace.selectSpan({start,end},{kind:'span'});
    }
    else if(node.hasAttribute('data-cd-entity')){const e=p.analysis?.lexical?.entities?.[Number(node.dataset.cdEntity)];if(e){workspace.selectSpan({start:e.start,end:e.end},{kind:'span'});onEntityFilter?.({type:e.type,text:e.text});}}
    else if(node.hasAttribute('data-cd-filter-entity')){const e=p.analysis?.lexical?.entities?.[Number(node.dataset.cdFilterEntity)];if(e)onEntityFilter?.({type:e.type,text:e.text});}
    else if(node.hasAttribute('data-cd-filter-value'))onEntityFilter?.(JSON.parse(node.dataset.cdFilterValue));
    else if(node.hasAttribute('data-cd-count')){const [kind,label]=node.dataset.cdCount.split(':');workspace.selectSpan(null,kind==='pos'?{kind,tag:label}:{kind,type:label});}
    else if(node.hasAttribute('data-cd-class'))workspace.selectSpan(null,{kind:'classification',label:node.dataset.cdClass});
    else if(node.hasAttribute('data-cd-dependency')){const [si,i]=node.dataset.cdDependency.split(':').map(Number),t=workspace.getDeep().dependency.result.sentences[si].items[i];workspace.selectSpan({start:t.start,end:t.end},{kind:'dependency',head:t.head,relation:t.relation||t.deprel});}
    else if(node.hasAttribute('data-cd-relation')){const r=workspace.getDeep().information_extraction.result.relations[Number(node.dataset.cdRelation)],s=r.evidence||r.subject;if(valid(s,Array.from(p.original_text).length))workspace.selectSpan({start:s.start,end:s.end},{kind:'relation'});}
    else if(node.hasAttribute('data-cd-action'))onContextAction?.(node.dataset.cdAction);
    else if(node.hasAttribute('data-cd-source'))onSourceLink?.(node.dataset.cdSource);
    else if(node.hasAttribute('data-cd-analyze'))onAnalyze?.(p);
    else if(node.hasAttribute('data-cd-clear'))workspace.selectSpan(null,{kind:'document'});
  }
  listen(host,'click',click);listen(inspector,'click',click);
  listen(host,'change',event=>{if(event.target.dataset.cdLayer)workspace.setLayer(event.target.dataset.cdLayer,event.target.checked);});
  listen(host,'keydown',event=>{if(event.target.matches('g[data-cd-dependency]')&&(event.key==='Enter'||event.key===' ')){event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
  const unsubscribe=workspace.subscribe(render);
  render();
  return {render,destroy(){closed=true;unsubscribe();listeners.forEach(fn=>fn());}};
}
