import {matchingDocuments,selectionStats,NEWS_LABELS,NEWS_NAMES,ENTITY_TYPES,CLASS_COLORS,CLUSTER_COLORS} from './comprehensive-workspace.js';

const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clusterColor=(id,state)=>CLUSTER_COLORS[Math.max(0,(state.corpus?.clusters||[]).findIndex(c=>String(c.cluster_id)===String(id)))%CLUSTER_COLORS.length];
const title=d=>d?.title||d?.id||'文档';
const number=v=>Number.isFinite(v)?v.toFixed(3):'未提供';
const kv=rows=>`<dl class="caw-kv">${rows.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`;
const visible=host=>Boolean(host&&host.clientWidth&&host.clientHeight&&host.getClientRects().length);

/* This view never obtains documents or runs models. All counts are selectors of
   the shared snapshot; unknown module outputs remain unknown. */
export function createCorpusView({workspace,scatterHost,listHost,inspectorHost,statsHost,flowHost,entityHost,relationInspectorHost,onOpenDocument,onAnalyzeSelection,onNavigate}) {
  const charts=new Map(),listeners=[];let plotting=false,destroyed=false,lastBrush=null;
  const listen=(node,type,fn)=>{if(node){node.addEventListener(type,fn);listeners.push(()=>node.removeEventListener(type,fn));}};
  const mount=(host,kind)=>{
    if(!host)return null;
    if(!host.querySelector(`[data-caw-chart="${kind}"]`))host.innerHTML=kind==='scatter'?`<div class="caw-canvas-toolbar"><label>着色 <select data-caw-color aria-label="散点着色"><option value="cluster">Cluster</option><option value="prediction">五类新闻预测</option></select></label><label>Cluster <select data-caw-cluster aria-label="筛选 Cluster"></select></label><button type="button" data-caw-brush-reset>清除刷选</button><button type="button" data-caw-reset>重置筛选</button></div><div class="caw-chart caw-scatter" data-caw-chart="scatter"></div><p class="caw-footnote">选择矩形或多边形工具后拖动选区；完成后可直接点击选中点，也可勾选文档。二维仅为显示投影，Cluster 来自高维表示。</p>`:` ${kind==='entity'?'<div class="caw-entity-toolbar"><label>实体统计口径 <select data-caw-entity-metric aria-label="实体统计口径"><option value="mentions_per_ner_document">每篇已分析 NER 文档的提及数</option><option value="mention_count">实体提及总数</option></select></label></div>':''}<div class="caw-chart caw-${kind}" data-caw-chart="${kind}"></div><div data-caw-extra="${kind}"></div>`;
    return host.querySelector(`[data-caw-chart="${kind}"]`);
  };
  const scatterCanvas=mount(scatterHost,'scatter'),flowCanvas=mount(flowHost,'flow'),entityCanvas=mount(entityHost,'entity');
  function chartFor(canvas,key) {
    if(!visible(canvas)||!window.echarts)return null;
    if(!charts.has(key)) {
      canvas.textContent='';const chart=window.echarts.init(canvas);charts.set(key,chart);
      if(key==='scatter') {
        chart.on('click',event=>{if(event.data?.document_id)onOpenDocument?.(event.data.document_id);});
        chart.on('brushSelected',event=>{
          if(plotting||destroyed)return;
          const batch=event.batch?.[0];if(!batch)return;
          // Empty graphical covers do not clear the persistent document selection.
          if(!batch.areas?.length)return;
          const data=chart.getOption().series?.[0]?.data||[];
          const ids=(batch.selected||[]).flatMap(s=>(s.dataIndex||[]).map(i=>data[i]?.document_id)).filter(Boolean);
          workspace.setBrush(ids);
          // End drawing and remove the hit-testing cover so selected points remain clickable.
          chart.dispatchAction({type:'takeGlobalCursor',key:'brush',brushOption:{brushType:false}},{silent:true});
          chart.dispatchAction({type:'brush',areas:[]},{silent:true});
        });
      }
      if(key==='flow')chart.on('click',event=>{
        const data=event.data;if(!data)return;
        if(event.dataType==='edge'){workspace.setFilter('cluster',data.cluster_id);workspace.setFilter('prediction',data.prediction);}
        else if(data.cluster_id!==undefined)workspace.setFilter('cluster',data.cluster_id);
        else if(data.prediction)workspace.setFilter('prediction',data.prediction);
      });
      if(key==='entity')chart.on('click',event=>{
        if(!event.data||event.data.cluster_id===undefined)return;
        workspace.setFilter('cluster',event.data.cluster_id);workspace.addEntity({type:event.data.entity_type});
      });
    }
    return charts.get(key);
  }
  function renderScatter(state,docs) {
    if(!scatterHost)return;
    const current=state.filters.cluster;
    const select=scatterHost.querySelector('[data-caw-cluster]');
    select.innerHTML='<option value="">全部</option>'+(state.corpus?.clusters||[]).map(c=>`<option value="${esc(c.cluster_id)}">Cluster ${esc(c.cluster_id)}</option>`).join('');select.value=current??'';
    scatterHost.querySelector('[data-caw-color]').value=state.colorBy;
    const chart=chartFor(scatterCanvas,'scatter');if(!chart)return;
    const ids=new Set(docs.map(d=>d.id));
    const points=(state.corpus?.documents||[]).filter(d=>Number.isFinite(d.x)&&Number.isFinite(d.y)).map(d=>{
      const summary=state.summaries[d.id],selected=ids.has(d.id),currentDoc=state.documentRef?.corpusId===state.corpus.id&&state.documentRef.id===d.id;
      return {name:title(d),value:[d.x,d.y],document_id:d.id,cluster_id:d.cluster_id,
        prediction:summary?.classification_known?summary.prediction:null,eligible:d.eligible,
        symbolSize:currentDoc?15:selected?9:5,itemStyle:{color:state.colorBy==='prediction'?(CLASS_COLORS[summary?.classification_known?summary.prediction:'unknown']||CLASS_COLORS.unknown):clusterColor(d.cluster_id,state),opacity:selected?.85:.12,borderWidth:currentDoc?2:0,borderColor:'#17394d'}};
    });
    plotting=true;
    try{chart.setOption({animation:false,grid:{left:42,right:22,top:42,bottom:38},
      toolbox:{right:8,feature:{brush:{type:['rect','polygon']}}},
      brush:{toolbox:['rect','polygon'],brushMode:'multiple',xAxisIndex:0,yAxisIndex:0,throttleType:'debounce',throttleDelay:180,brushStyle:{color:'rgba(18,143,151,.08)',borderColor:'#168e98',borderWidth:1}},
      tooltip:{formatter:p=>`${esc(p.data.name)}<br>Cluster ${esc(p.data.cluster_id)}<br>五类预测：${esc(NEWS_NAMES[p.data.prediction]||'未分析 / 未知')}<br>${p.data.eligible?'可按需分析':'仅历史摘要'}`},
      xAxis:{type:'value',name:'投影 1',nameLocation:'middle',nameGap:25,axisLabel:{fontSize:11},splitLine:{lineStyle:{color:'#edf1f4'}}},
      yAxis:{type:'value',axisLabel:{fontSize:11},splitLine:{lineStyle:{color:'#edf1f4'}}},
      series:[{name:'语料文档',type:'scatter',data:points,brushSelect:true}]},{notMerge:false,lazyUpdate:false,silent:true});
      if(state.brushIds===null&&lastBrush!==null)chart.dispatchAction({type:'brush',areas:[]},{silent:true});
      lastBrush=state.brushIds;
    }finally{plotting=false;}
  }
  function renderList(state,docs) {
    if(!listHost)return;
    const checked=new Set(state.brushIds||[]);
    listHost.innerHTML=`<div class="caw-list-heading"><strong>${docs.length} / ${state.corpus?.documents?.length||0} 篇</strong><button type="button" data-caw-select-visible ${docs.length?'':'disabled'}>选择当前集合</button></div><div class="caw-document-scroll">${docs.slice(0,160).map(d=>{
      const s=state.summaries[d.id],active=state.documentRef?.corpusId===state.corpus?.id&&state.documentRef?.id===d.id;
      return `<div class="caw-document ${active?'is-current':''}"><input type="checkbox" aria-label="选择 ${esc(title(d))}" data-caw-select="${esc(d.id)}" ${checked.has(d.id)?'checked':''}><button type="button" data-caw-document="${esc(d.id)}" aria-pressed="${active}"><strong>${esc(title(d))}</strong><span>${esc(d.snippet||'')}</span><small>Cluster ${esc(d.cluster_id)} · ${s?.classification_known?esc(NEWS_NAMES[s.prediction]):'分类未知'} · ${s?.ner_known?`实体 ${(s.entities||[]).length}`:'NER 未分析'}${!d.eligible?' · 仅历史摘要':''}</small></button></div>`;
    }).join('')||'<p class="caw-empty">当前交集没有文档。可逐个移除顶部条件。</p>'}</div>${docs.length>160?'<p class="caw-footnote">仅显示前 160 篇；刷选和统计仍覆盖完整当前集合。</p>':''}<p class="caw-footnote">移动端可先选择当前集合，再取消勾选以缩小范围。</p>`;
  }
  function renderStats(state,stats) {
    if(!statsHost)return;
    const selectionIds=matchingDocuments(state).map(d=>d.id).sort(),saved=state.selectionTerms;
    const termsCurrent=Array.isArray(saved?.document_ids)&&JSON.stringify([...saved.document_ids].sort())===JSON.stringify(selectionIds);
    let termsHTML='<p class="caw-footnote">读取当前 selection 关键词…</p>';
    if(!stats.total)termsHTML='<p class="caw-footnote">当前选择为空，没有词项均值。</p>';
    else if(termsCurrent&&Array.isArray(saved.terms))termsHTML=saved.terms.length?`<div class="caw-selection-terms">${saved.terms.map(t=>`<span><strong>${esc(t.term)}</strong><small>${number(t.weight)}</small></span>`).join('')}</div><p class="caw-footnote">${esc(saved.source||'已有拟合表示')} · 仅汇总当前选择；TF-IDF 均值不是分类贡献，不重新 fit。</p>`:'<p class="caw-footnote">当前选择没有非零词项。</p>';
    else if(termsCurrent&&saved.terms===null)termsHTML='<p class="caw-footnote">当前选择的表示数据未提供，不展示推测关键词。</p>';
    statsHost.innerHTML=`<div class="caw-selection-heading"><div><strong>${stats.total}</strong> 当前集合 <span> / ${state.corpus?.documents?.length||0} 篇</span></div><button type="button" data-caw-analyze ${stats.eligible?'':'disabled'}>分析当前集合 · 每批最多 5 篇</button></div><div class="caw-selection-distributions"><section><h3>五类预测 <small>已知 ${stats.classified} / ${stats.total}</small></h3>${NEWS_LABELS.map(label=>`<button type="button" data-caw-prediction="${label}" class="caw-count-bar ${state.filters.prediction===label?'is-selected':''}"><span>${NEWS_NAMES[label]}</span><i style="--caw-value:${stats.classCounts?stats.classCounts[label]/Math.max(1,stats.classified)*100:0}%;--caw-color:${CLASS_COLORS[label]}"></i><b>${stats.classCounts?stats.classCounts[label]:'未知'}</b></button>`).join('')}<p class="caw-footnote">未分类 ${stats.unknownClassification} 篇不计入已知分布。</p></section><section><h3>实体提及 <small>NER 已知 ${stats.ner} / ${stats.total}</small></h3>${ENTITY_TYPES.map(type=>`<button type="button" data-caw-entity-type="${type}" class="caw-count-bar"><span>${type}</span><i style="--caw-value:${stats.entityCounts?stats.entityCounts[type]/Math.max(1,...Object.values(stats.entityCounts))*100:0}%;--caw-color:#168e98"></i><b>${stats.entityCounts?stats.entityCounts[type]:'未知'}</b></button>`).join('')}<p class="caw-footnote">计数为实体提及次数；未运行 NER 不视为零。</p></section></div><section class="caw-selection-terms-section"><h3>当前选择 · TF-IDF均值（原拟合表示）</h3>${termsHTML}</section>`;
  }
  function renderInspector(state,docs,stats) {
    if(!inspectorHost)return;
    const current=(state.corpus?.documents||[]).find(d=>d.id===state.documentRef?.id&&state.documentRef.corpusId===state.corpus.id),whole=selectionStats(state,state.corpus?.documents||[]);
    const cluster=state.corpus?.clusters?.find(c=>String(c.cluster_id)===String(state.filters.cluster));
    inspectorHost.innerHTML=`<div class="caw-object">${esc(cluster?`Cluster ${cluster.cluster_id}`:state.corpus?.name||state.corpus?.title||'当前语料集合')}</div>${kv([['当前交集',`${stats.total} 篇`],['可分析',`${stats.eligible} 篇`],['实际分类已知',`${stats.classified} 篇`],['NER 已知',`${stats.ner} 篇`]])}<h3>各簇处理覆盖</h3><p class="caw-footnote">全语料可分析文档：分类 ${whole.classifiedEligible} / ${whole.eligible}；NER ${whole.nerEligible} / ${whole.eligible}。各簇分别显示已分析 / 可分析；不是当前交集。</p><div class="caw-cluster-coverage">${whole.clusters.map(c=>`<button type="button" data-caw-cluster-value="${esc(c.id)}" class="${String(state.filters.cluster)===String(c.id)?'is-selected':''}"><span>Cluster ${esc(c.id)}</span><strong>分类 ${c.classifiedEligible} / ${c.eligible}<br>NER ${c.nerEligible} / ${c.eligible}</strong></button>`).join('')}</div>${current?`<div class="caw-rule"></div><h3>当前文档</h3><strong>${esc(title(current))}</strong>${kv([['Cluster',current.cluster_id],['二维位置',`${number(current.x)}, ${number(current.y)}`],['当前筛选内',docs.some(d=>d.id===current.id)?'是':'否 · 文档上下文仍保留']])}<button type="button" data-caw-document="${esc(current.id)}">打开文档工作台 →</button>`:''}${cluster?`<details><summary>已保存的整簇关键词与代表文档</summary><p class="caw-footnote">来自原聚类整簇；不是对当前 selection 重新拟合的关键词。</p><div class="caw-terms">${(cluster.top_terms||[]).slice(0,12).map(t=>`<span>${esc(t.term)}</span>`).join('')}</div>${(cluster.representatives||[]).slice(0,3).map(d=>`<button type="button" data-caw-document="${esc(d.id)}">${esc(d.snippet||d.title||d.id)}</button>`).join('')}</details>`:''}`;
  }
  function renderRelations(state,stats) {
    const docs=matchingDocuments(state),whole=selectionStats(state,state.corpus?.documents||[]);
    const flow=chartFor(flowCanvas,'flow');
    if(flow) {
      const clusterIds=[...new Set(stats.flows.map(x=>x.cluster_id))].sort((a,b)=>Number(a)-Number(b));
      const names=[...new Set(stats.flows.map(x=>x.prediction))];
      flow.setOption({animation:false,tooltip:{trigger:'item',formatter:p=>p.dataType==='edge'?`Cluster ${esc(p.data.cluster_id)} → ${esc(NEWS_NAMES[p.data.prediction])}<br>${p.data.value} 篇实际预测`:`${esc(p.name)}<br>点击筛选当前集合`},series:[{type:'sankey',left:8,right:42,top:18,bottom:18,nodeWidth:15,nodeGap:22,draggable:false,layoutIterations:0,label:{color:'#25435a',fontSize:12},emphasis:{focus:'adjacency'},lineStyle:{color:'source',opacity:.27,curveness:.45},data:[...clusterIds.map(id=>({name:`Cluster ${id}`,cluster_id:id,itemStyle:{color:clusterColor(id,state)}})),...NEWS_LABELS.filter(label=>names.includes(label)).map(prediction=>({name:NEWS_NAMES[prediction],prediction,itemStyle:{color:CLASS_COLORS[prediction]}}))],links:stats.flows.map(f=>({source:`Cluster ${f.cluster_id}`,target:NEWS_NAMES[f.prediction],value:f.count,cluster_id:f.cluster_id,prediction:f.prediction}))}]},true);
    }
    if(flowHost)flowHost.querySelector('[data-caw-extra="flow"]').innerHTML=`<p class="caw-footnote">当前交集 ${stats.classified} / ${stats.total} 篇已有分类结果；可分析文档分类覆盖 ${stats.classifiedEligible} / ${stats.eligible}。流宽为文档数；Cluster 不等于类别。${!stats.classified?'当前集合尚无已知分类，未知结果不会画成流。':''}</p><div class="caw-flow-actions">${stats.flows.map(f=>`<button type="button" data-caw-flow="${esc(JSON.stringify([f.cluster_id,f.prediction]))}">Cluster ${esc(f.cluster_id)} → ${NEWS_NAMES[f.prediction]} · ${f.count}</button>`).join('')}</div><section class="caw-matching-documents"><h3>当前筛选文档 <small>${stats.total} 篇</small></h3><div class="caw-matching-scroll">${docs.slice(0,160).map(d=>{
      const summary=state.summaries[d.id],active=state.documentRef?.corpusId===state.corpus?.id&&state.documentRef?.id===d.id;
      const entitySummary=summary?.ner_known?`${(summary.entities||[]).length} 次实体提及${(summary.entity_types||[]).length?` · ${summary.entity_types.join(' / ')}`:''}`:'NER 未分析 / 未知';
      return `<button type="button" class="caw-matching-document ${active?'is-current':''}" data-caw-document="${esc(d.id)}" aria-pressed="${active}"><strong>${esc(title(d))}</strong><small><span>Cluster ${esc(d.cluster_id)}</span><span style="--caw-prediction-color:${CLASS_COLORS[summary?.classification_known?summary.prediction:'unknown']||CLASS_COLORS.unknown}">${summary?.classification_known?`预测 ${esc(NEWS_NAMES[summary.prediction]||summary.prediction)}`:'分类未分析 / 未知'}</span><span>${esc(entitySummary)}</span>${!d.eligible?'<span>仅历史摘要</span>':''}</small></button>`;
    }).join('')||'<p class="caw-empty">当前交集没有文档。可移除筛选条件。</p>'}</div>${docs.length>160?'<p class="caw-footnote">仅显示前 160 篇；统计覆盖完整当前集合。</p>':''}</section>`;
    const metric=state.entityMetric||'mentions_per_ner_document',normalized=metric==='mentions_per_ner_document';
    const metricSelect=entityHost?.querySelector('[data-caw-entity-metric]');if(metricSelect)metricSelect.value=metric;
    const entity=chartFor(entityCanvas,'entity'),clusterRows=stats.clusters;
    if(entity) {
      const counts=new Map();for(const e of stats.entities){const key=JSON.stringify([e.cluster_id,e.type]);counts.set(key,(counts.get(key)||0)+e.count);}
      const data=clusterRows.flatMap((c,y)=>c.ner?ENTITY_TYPES.map((type,x)=>{const count=counts.get(JSON.stringify([c.id,type]))||0;return {value:[x,y,normalized?count/c.ner:count],mention_count:count,ner_documents:c.ner,cluster_id:c.id,entity_type:type};}):[]);
      entity.setOption({animation:false,grid:{left:84,right:12,top:22,bottom:30},tooltip:{formatter:p=>{const c=clusterRows.find(row=>String(row.id)===String(p.data.cluster_id));return `Cluster ${esc(p.data.cluster_id)} × ${esc(p.data.entity_type)}<br>${normalized?`${p.data.value[2].toFixed(3)} 次 / 已分析 NER 文档<br>`:''}${p.data.mention_count} 次实体提及 / ${p.data.ner_documents} 篇 NER 已知文档<br>当前交集簇内 NER 已分析（可分析文档） ${c.nerEligible} / ${c.eligible} 篇`; }},xAxis:{type:'category',data:ENTITY_TYPES,position:'top',axisLine:{show:false},axisTick:{show:false}},yAxis:{type:'category',data:clusterRows.map(c=>`Cluster ${c.id}`),axisLine:{show:false},axisTick:{show:false}},visualMap:{show:false,min:0,max:Math.max(1,...data.map(x=>x.value[2])),inRange:{color:['#f1f7f7','#168e98']}},series:[{type:'heatmap',data,label:{show:true,color:'#193e50',fontSize:12,formatter:p=>normalized?Number(p.data.value[2].toFixed(3)):p.data.value[2]},itemStyle:{borderColor:'#fff',borderWidth:3},emphasis:{itemStyle:{borderColor:'#153b54',borderWidth:2}}}]},true);
    }
    const named=new Map();for(const e of stats.entities){const key=JSON.stringify([e.type,e.text]),row=named.get(key)||{type:e.type,text:e.text,count:0};row.count+=e.count;named.set(key,row);}
    const topNames=[...named.values()].sort((a,b)=>b.count-a.count||a.text.localeCompare(b.text));
    if(entityHost)entityHost.querySelector('[data-caw-extra="entity"]').innerHTML=`<p class="caw-footnote">${normalized?'热图为实体提及总数 ÷ 当前簇 NER 已分析文档数（含已知历史摘要）。':'热图为当前簇实体提及总数。'}当前交集 NER 已知 ${stats.ner} / ${stats.total} 篇；可分析文档 NER 已分析 ${stats.nerEligible} / ${stats.eligible}。全语料可分析文档 NER 已分析 ${whole.nerEligible} / ${whole.eligible}。${stats.unknownNER} 篇 NER 未知；未观测的簇留空，只有已分析无提及才写 0。点击簇 × 类型可形成交集；下方为当前集合实体提及总数。</p><div class="caw-entity-names">${topNames.slice(0,18).map(e=>`<button type="button" data-caw-entity="${esc(JSON.stringify({type:e.type,text:e.text}))}"><small>${esc(e.type)}</small> ${esc(e.text)} <b>${e.count}</b></button>`).join('')||'<span class="caw-empty">当前交集没有已知实体提及。</span>'}</div>`;
    if(relationInspectorHost)relationInspectorHost.innerHTML=`<div class="caw-object">当前交集 · ${stats.total} 篇</div>${kv([['分类已知 / 当前集合',`${stats.classified} / ${stats.total}`],['NER 已知 / 当前集合',`${stats.ner} / ${stats.total}`],['当前可分析文档 · 分类',`${stats.classifiedEligible} / ${stats.eligible}`],['当前可分析文档 · NER',`${stats.nerEligible} / ${stats.eligible}`],['全语料可分析文档 · 分类',`${whole.classifiedEligible} / ${whole.eligible}`],['全语料可分析文档 · NER',`${whole.nerEligible} / ${whole.eligible}`],['Cluster 条件',state.filters.cluster===null?'无':state.filters.cluster],['五类预测条件',NEWS_NAMES[state.filters.prediction]||'无'],['实体条件',state.filters.entities.map(e=>`${e.type}${e.text?`: ${e.text}`:''}`).join(' ∩ ')||'无'],['刷选',state.brushIds===null?'无':`${state.brushIds.length} 篇（再与条件求交）`]])}<p class="caw-footnote">点击流、实体类型或实体名称会同步筛选语料散点与文档列表。点击当前筛选文档可打开文档工作台，保留筛选条件。所有关系为并列观察，不是因果链。</p><button type="button" data-caw-navigate="collection">返回语料空间 →</button><button type="button" data-caw-analyze ${stats.eligible?'':'disabled'}>分析当前集合 · 每批最多 5 篇</button><details><summary>当前统计的完整数据</summary><pre>${esc(JSON.stringify(stats,null,2))}</pre></details>`;
  }
  function activate(event) {
    const node=event.target.closest('button');if(!node)return;
    if(node.hasAttribute('data-caw-reset'))workspace.reset();
    else if(node.hasAttribute('data-caw-brush-reset')){workspace.setBrush(null);charts.get('scatter')?.dispatchAction({type:'brush',areas:[]});}
    else if(node.hasAttribute('data-caw-select-visible'))workspace.setBrush(matchingDocuments(workspace.getState()).map(d=>d.id));
    else if(node.hasAttribute('data-caw-document'))onOpenDocument?.(node.dataset.cawDocument);
    else if(node.hasAttribute('data-caw-cluster-value'))workspace.setFilter('cluster',node.dataset.cawClusterValue);
    else if(node.hasAttribute('data-caw-prediction'))workspace.setFilter('prediction',node.dataset.cawPrediction);
    else if(node.hasAttribute('data-caw-entity-type'))workspace.addEntity({type:node.dataset.cawEntityType});
    else if(node.hasAttribute('data-caw-entity'))workspace.addEntity(JSON.parse(node.dataset.cawEntity));
    else if(node.hasAttribute('data-caw-flow')){const [cluster,prediction]=JSON.parse(node.dataset.cawFlow);workspace.setFilter('cluster',cluster);workspace.setFilter('prediction',prediction);}
    else if(node.hasAttribute('data-caw-analyze'))onAnalyzeSelection?.(matchingDocuments(workspace.getState()).map(d=>d.id));
    else if(node.hasAttribute('data-caw-navigate'))onNavigate?.(node.dataset.cawNavigate);
  }
  for(const host of [scatterHost,listHost,inspectorHost,statsHost,flowHost,entityHost,relationInspectorHost])listen(host,'click',activate);
  listen(scatterHost,'change',event=>{
    if(event.target.hasAttribute('data-caw-color'))workspace.setColorBy(event.target.value);
    if(event.target.hasAttribute('data-caw-cluster'))workspace.setFilter('cluster',event.target.value||null);
  });
  listen(entityHost,'change',event=>{if(event.target.hasAttribute('data-caw-entity-metric'))workspace.setEntityMetric(event.target.value);});
  listen(listHost,'change',event=>{
    const input=event.target;if(!input.hasAttribute('data-caw-select'))return;
    const state=workspace.getState(),ids=new Set(state.brushIds||[]);
    if(input.checked)ids.add(input.dataset.cawSelect);else ids.delete(input.dataset.cawSelect);
    workspace.setBrush([...ids]);
  });
  function render(){if(destroyed)return;const state=workspace.getState(),docs=matchingDocuments(state),stats=selectionStats(state,docs);
    renderScatter(state,docs);renderList(state,docs);renderStats(state,stats);renderInspector(state,docs,stats);renderRelations(state,stats);
  }
  return {render,resize(){for(const chart of charts.values())chart.resize();render();},destroy(){destroyed=true;listeners.forEach(fn=>fn());charts.forEach(chart=>chart.dispose());charts.clear();}};
}
