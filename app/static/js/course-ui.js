import { api, node, notice } from './api.js';
import { clearAlgorithm, renderTrace, renderCompactTrace, renderTaskModelInfo, resizeAlgorithm, exportProcess } from './algorithm-view.js';
import { resizeCharts } from './charts.js';
export const $=id=>document.getElementById(id);
export const fail=error=>notice($('page-message'),error.message||String(error),true);
export function tableInto(target,headers,rows) {
  target.replaceChildren();const table=node('table'),head=node('tr');headers.forEach(h=>head.append(node('th',h)));table.append(head);
  for(const values of rows){const row=node('tr');values.forEach(v=>row.append(node('td',String(v??'—'))));table.append(row);}target.append(table);
}
export async function resources(kind='cws') {
  const [runs,status]=await Promise.all([api('/api/v1/experiments/1/runs'),api('/api/v1/experiments/1/extensions')]);
  const hmm=runs.items.find(r=>r.run_type==='frozen-final-benchmark'&&r.run_id.endsWith('_smooth'))||runs.items.find(r=>r.run_type!=='course_samples'&&!r.run_id.endsWith('_mle'));
  const ner=runs.items.filter(r=>r.run_type==='course_samples');const extensions=new Map(status.items.map(r=>[r.model,r]));
  const select=$('run-select');
  if(select){select.replaceChildren();for(const run of ner){const o=node('option',`${run.dataset} · ${run.run_id}`);o.value=run.run_id;select.append(o);}if(!ner.length) select.append(node('option','unavailable：没有 NER 课程 Run'));}
  const ctx={kind,hmm,ner,extensions,version:0,model:()=>$('segmentation-model')?.value||'hmm1-smooth',nerId:()=>select?.value||ner[0]?.run_id||null,runId:()=>kind==='ner'?ctx.nerId():hmm?.run_id,modelRun:()=>extensions.get(ctx.model())?.run_id||hmm?.run_id};
  if($('segmentation-model')) for(const option of $('segmentation-model').options){const available=option.value==='hmm1-smooth'?!!hmm:extensions.get(option.value)?.available===true;option.disabled=!available;if(!available)option.textContent+='（unavailable）';}
  ctx.available=()=>ctx.model()==='hmm1-smooth'?!!hmm:extensions.get(ctx.model())?.available===true;
  ctx.footer=()=>{if($('active-run'))$('active-run').textContent=kind==='ner'?`NER · MSRA-200 · Run ${ctx.nerId()||'unavailable'}`:`CWS · SIGHAN MSR · ${ctx.model()} · Run ${ctx.modelRun()||'unavailable'}${kind==='integrated'?` ｜ NER benchmark Run ${ctx.nerId()||'unavailable'}`:''}`;};ctx.footer();
  return ctx;
}
export function clearResults() {
  for(const id of ['original-text','tokens-result','bmes-result','ner-result']) if($(id)){$(id).replaceChildren(node('span','等待分析。'));$(id).className='result-empty';}
  $('result-summary')?.replaceChildren();if($('model-info'))$('model-info').textContent='';
  if($('algorithm-trace')) clearAlgorithm($('algorithm-trace'));$('model-details')?.replaceChildren();if($('process-export'))$('process-export').hidden=true;
  if($('method-run'))$('method-run').textContent='输入文本后显示当前句子的结果。';
  if($('entity-detail'))$('entity-detail').textContent='选择原文中的实体，查看精确跨度。';
  if($('compact-trace'))$('compact-trace').textContent='正在读取当前输入的算法过程…';
  if($('input-codepoints'))$('input-codepoints').textContent='等待当前输入。';
  if($('ner-input-stats'))$('ner-input-stats').textContent='等待识别。';
  if($('ner-current-summary'))$('ner-current-summary').textContent='等待当前句子的实体识别结果。';
  if($('entity-span-track'))$('entity-span-track').textContent='等待识别。';
}
export function entityHighlight(text,entities,target) {
  const chars=Array.from(text);target.replaceChildren();target.className='original';let offset=0;
  for(const e of [...entities].sort((a,b)=>a.start-b.start)){if(e.start<offset)continue;target.append(document.createTextNode(chars.slice(offset,e.start).join('')));const mark=node('mark',chars.slice(e.start,e.end).join(''),`entity ${e.type.toLowerCase()}`);mark.title=`${e.type} [${e.start},${e.end})`;mark.dataset.start=String(e.start);mark.dataset.end=String(e.end);mark.dataset.type=e.type;mark.tabIndex=0;mark.setAttribute('role','button');target.append(mark);offset=e.end;}
  target.append(document.createTextNode(chars.slice(offset).join('')));
}
export function renderAnalysis(value) {
  const entities=value.ner?.status==='ok'?value.ner.entities:[];entityHighlight(value.text,entities,$('original-text'));
  $('model-info').textContent=`${[value.segmentation?.model,value.ner?.model].filter(Boolean).join(' · ')} · ${value.elapsed_ms} ms`;
  if($('input-codepoints')){const chars=Array.from(value.text),target=$('input-codepoints');target.replaceChildren();chars.slice(0,24).forEach((char,index)=>{const cell=node('span',char===' '?'␠':char);cell.title=`offset ${index}`;target.append(cell);});if(chars.length>24)target.append(node('small',`另有 ${chars.length-24} 个字符`));}
  if($('ner-input-stats'))$('ner-input-stats').textContent=`${Array.from(value.text).length} 个 Unicode code point；当前识别到 ${entities.length} 个实体跨度。`;
  if($('ner-current-summary')){const counts=Object.fromEntries(['PER','LOC','ORG'].map(type=>[type,entities.filter(entity=>entity.type===type).length]));$('ner-current-summary').replaceChildren(...['PER','LOC','ORG'].map(type=>{const item=node('span',type);item.append(node('strong',String(counts[type])));return item;}));}
  if($('tokens-result')&&value.segmentation){const s=value.segmentation;$('tokens-result').className='tokens';$('tokens-result').replaceChildren(...s.tokens.map((t,i)=>{const chip=node('span',t,'token');chip.append(node('small',String(i+1)));return chip;}));
    const derived=['longest','bigram'].includes($('segmentation-model')?.value);$('bmes-title').textContent=derived?'Derived BMES（由 tokens 派生）':'BMES（模型解码路径）';$('bmes-result').className='bmes';
    $('bmes-result').replaceChildren(...Array.from(value.text).map((c,i)=>{const cell=node('span','','bmes-cell');cell.dataset.tag=s.tags[i];cell.append(node('span',c===' '?'␠':c),node('small',s.tags[i]));return cell;}));
    $('result-summary').replaceChildren(...[['模型',$('segmentation-model')?.selectedOptions[0]?.textContent||s.model],['字符数',Array.from(value.text).length],['词数',s.tokens.length],['Roundtrip',s.tokens.join('')===value.text?'✓ 原文一致':'不一致'],['耗时',`${value.elapsed_ms} ms`]].map(([label,v])=>{const item=node('span',label);item.append(node('strong',String(v)));return item;}));
    if($('method-run'))$('method-run').textContent=`${Array.from(value.text).length} 字符 → ${s.tokens.length} 词；${value.elapsed_ms} ms。分词拼接${s.tokens.join('')===value.text?'与原文一致':'与原文不一致'}。`;
  }
  if($('ner-result')){const target=$('ner-result');target.replaceChildren();if(value.ner?.status==='ok'){target.className='ner-items';for(const e of entities){const row=node('div','','ner-item');row.append(node('strong',e.text,'ner-entity-text'),node('span',e.type,`entity ${e.type.toLowerCase()}`),node('small',`[${e.start}, ${e.end})`));target.append(row);}if(!entities.length)target.append(node('p','未识别到 PER/LOC/ORG。'));}else{target.className='result-empty';target.textContent=value.ner?`unavailable：${value.ner.reason||value.ner.message||'NER 资源不可用'}`:'未请求 NER。';}}
  if($('entity-span-track')){const target=$('entity-span-track'),length=Array.from(value.text).length;target.replaceChildren();if(entities.length&&length){const track=node('div','','entity-track-line');for(const entity of entities){const bar=node('span',entity.text,`entity-track-bar ${entity.type.toLowerCase()}`);bar.style.left=`${entity.start/length*100}%`;bar.style.width=`${Math.max((entity.end-entity.start)/length*100,5)}%`;bar.title=`${entity.text} · ${entity.type} [${entity.start},${entity.end})`;bar.tabIndex=0;bar.setAttribute('role','button');bar.addEventListener('click',()=>{const mark=[...$('original-text').querySelectorAll('mark.entity')].find(item=>Number(item.dataset.start)===entity.start&&Number(item.dataset.end)===entity.end&&item.dataset.type===entity.type);mark?.click();});bar.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();bar.click();}});track.append(bar);}target.append(track);const axis=node('div','','entity-track-axis');axis.append(node('span','0'),node('span',`${length} code points`));target.append(axis);}else target.append(node('p','当前文本没有可绘制的实体跨度。','muted'));}
  if($('entity-detail')){const detail=$('entity-detail'),marks=[...$('original-text').querySelectorAll('mark.entity')],rows=[...$('ner-result').querySelectorAll('.ner-item')];function selectEntity(entity){detail.textContent=`${entity.text} · ${entity.type} · [${entity.start}, ${entity.end})`;marks.forEach(mark=>mark.classList.toggle('is-active',Number(mark.dataset.start)===entity.start&&Number(mark.dataset.end)===entity.end&&mark.dataset.type===entity.type));rows.forEach((row,i)=>row.classList.toggle('is-active',entities[i]===entity));}marks.forEach(mark=>{const entity=entities.find(e=>e.start===Number(mark.dataset.start)&&e.end===Number(mark.dataset.end)&&e.type===mark.dataset.type);if(!entity)return;mark.addEventListener('click',()=>selectEntity(entity));mark.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectEntity(entity);}});});rows.forEach((row,i)=>{row.tabIndex=0;row.setAttribute('role','button');row.addEventListener('click',()=>selectEntity(entities[i]));row.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectEntity(entities[i]);}});});if(entities.length)selectEntity(entities[0]);else detail.textContent='当前文本未识别到 PER / LOC / ORG 实体。';}
}
export function bindAnalysis(ctx,{tasks=['segmentation'],trace=false}={}) {
  const form=$('analysis-form'),input=$('input-text'),button=$('analyze-button');
  const label=button.textContent;
  let timer,request=0;
  function invalidate() {
    request++;
    clearTimeout(timer);
    clearResults();
    if(!input.value.trim()) {button.disabled=false;button.textContent=label;notice($('page-message'));}
  }
  async function run(current) {
    if(current!==request)return;
    const selected=tasks.filter(task=>$(`task-${task}`).checked);
    if(!selected.length){button.disabled=false;return fail(new Error('请选择至少一个任务。'));}
    const text=input.value,version=ctx.version;
    button.disabled=true;button.textContent='分析中…';notice($('page-message'));
    try {
      if(selected.includes('segmentation')&&!ctx.available())throw new Error('unavailable：当前分词模型资源不可用。');
      const value=await api('/api/v1/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,tasks:selected,run_id:ctx.runId()||'latest',segmentation_model:ctx.model()})});
      if(current!==request||version!==ctx.version)return;
      renderAnalysis(value);
      if(trace&&value.segmentation)await loadProcess(ctx,{text,isCurrent:()=>current===request});
    } catch(error) {
      if(current===request&&version===ctx.version)fail(error);
    } finally {
      if(current===request){button.disabled=false;button.textContent=label;}
    }
  }
  function schedule(delay=0) {
    invalidate();
    if(!input.value.trim())return;
    const current=request;
    timer=setTimeout(()=>run(current),delay);
  }
  form.addEventListener('submit',event=>{event.preventDefault();invalidate();if(input.value.trim())void run(request);});
  input.addEventListener('input',()=>schedule(ctx.kind==='ner'||ctx.kind==='integrated'?850:350));
  for(const task of tasks)$(`task-${task}`).addEventListener('change',()=>schedule());
  ctx.invalidateAnalysis=invalidate;
  ctx.requestAnalysis=()=>schedule();
  return ctx.requestAnalysis;
}
export async function loadProcess(ctx,{text=$('input-text').value,isCurrent=()=>true}={}) {
  const version=ctx.version;if(!text)return;
  if(!ctx.available())return fail(new Error('unavailable：当前模型资源不可用。'));
  if(Array.from(text).length>30){clearAlgorithm($('algorithm-trace'));$('algorithm-trace').append(node('p','算法过程可视化仅对 ≤30 字符输入开放。当前文本仍已正常完成分词。','process-note'));if($('compact-trace'))$('compact-trace').textContent='当前文本已完整分词；算法过程可视化仅对 ≤30 字符输入开放。';}
  else {const value=await api(`/api/v1/experiments/1/trace?${new URLSearchParams({text,model:ctx.model(),run_id:ctx.runId()||'latest'})}`);if(version!==ctx.version||!isCurrent())return;renderTrace($('algorithm-trace'),value);renderCompactTrace($('compact-trace'),value);$('algorithm-title').textContent=`${$('segmentation-model').selectedOptions[0].textContent} · 算法过程`;$('process-export').hidden=ctx.model()!=='bigram';}
  if(version!==ctx.version||!isCurrent())return;
  const value=await api(`/api/v1/models/hmm/parameters?${new URLSearchParams({model:ctx.model(),run_id:ctx.runId()||'latest',topk:'20'})}`);if(version!==ctx.version||!isCurrent())return;renderTaskModelInfo($('model-details'),value,ctx.model());
}
export function bindProcess(ctx) {
  $('trace-button').addEventListener('click',()=>loadProcess(ctx).catch(fail));
  $('process-export').addEventListener('click',()=>exportProcess($('algorithm-trace'),ctx.modelRun()));
}
export function bindSelection(ctx,refresh=()=>{}) {
  for(const id of ['segmentation-model','run-select'])$(id)?.addEventListener('change',()=>{ctx.invalidateAnalysis?.();ctx.version++;ctx.footer();Promise.resolve(refresh()).catch(fail);ctx.requestAnalysis?.();});
}
export function renderNerMetrics(target,ner) {
  if(!ner){target.textContent='当前 Run 没有 NER Gold 评价。';return;}
  const wrap=node('div','','table-wrap');tableInto(wrap,['类别','Precision','Recall','F1','Gold 实体'],['PER','LOC','ORG','Overall'].map(label=>[label,...['precision','recall','f1'].map(k=>`${(ner[label][k]*100).toFixed(2)}%`),ner[label].support]));target.replaceChildren(wrap);
}
export function bindErrors(ctx,{ner=false}={}) {
  let page=1,requestId=0;
  if(ner){$('error-category').closest('.header-controls').hidden=true;}
  function spanText(text,spans) {
    const chars=Array.from(text),target=node('div','','error-text');let offset=0;
    for(const [start,end] of spans.filter(s=>Number.isInteger(s[0])&&Number.isInteger(s[1])&&s[0]>=0&&s[1]<=chars.length&&s[0]<s[1]).sort((a,b)=>a[0]-b[0])){
      if(start<offset)continue;
      target.append(document.createTextNode(chars.slice(offset,start).join('')),node('mark',chars.slice(start,end).join(''),'error-highlight'));
      offset=end;
    }
    target.append(document.createTextNode(chars.slice(offset).join('')));return target;
  }
  function row(label,content) {const target=node('div','','error-row');target.append(node('strong',label),content);return target;}
  function tokenStrip(tokens,spans) {
    const wrap=node('div','','error-token-list');let offset=0;
    for(const word of tokens){const end=offset+Array.from(word).length;const changed=spans.some(([start,stop])=>offset<stop&&end>start);wrap.append(node('span',word,`error-token${changed?' changed':''}`));offset=end;}
    return wrap;
  }
  function entityStrip(text,entities,differences) {
    const wrap=node('div','','error-token-list'),chars=Array.from(text);
    if(!entities?.length){wrap.append(node('span','无实体','muted'));return wrap;}
    for(const entity of entities){const [start,end,type]=Array.isArray(entity)?entity:[entity.start,entity.end,entity.type];
      const changed=differences?.some(e=>{const [s,t,k]=Array.isArray(e)?e:[e.start,e.end,e.type];return s===start&&t===end&&k===type;});
      wrap.append(node('span',`${chars.slice(start,end).join('')} · ${type} [${start},${end})`,`error-token${changed?' changed':''}`));
    }
    return wrap;
  }
  function cwsCase(item) {
    const article=node('article','','error-case'),spans=item.error_spans||[];
    article.append(node('h3',`${item.sample_id} · ${item.category}`));
    const aligned=Array.isArray(item.gold_tokens)&&Array.isArray(item.predicted_tokens)&&item.gold_tokens.join('')===item.text&&item.predicted_tokens.join('')===item.text;
    article.append(spanText(item.text,aligned?spans:[]));
    if(aligned){article.append(row('Gold 分词',tokenStrip(item.gold_tokens,spans)),row('预测分词',tokenStrip(item.predicted_tokens,spans)));}
    else article.append(node('p','词序列与原文未能可靠对齐；保留原始记录。','muted'));
    const detail=node('details','','error-raw'),summary=node('summary','查看原始 BMES 与跨度');detail.append(summary);
    for(const [label,value] of [['Gold 分词',item.gold_tokens?.join(' / ')],['预测分词',item.predicted_tokens?.join(' / ')],['Gold BMES',item.gold_tags?.join(' ')],['预测 BMES',item.predicted_tags?.join(' ')],['错误跨度',JSON.stringify(spans)]])detail.append(node('p',`${label}：${value??'未保存'}`));
    article.append(detail);return article;
  }
  function nerCase(item,index) {
    const article=node('details','','error-case error-case-fold'),chars=Array.from(item.text),summary=node('summary',`${item.sample_id} · ${chars.slice(0,48).join('')}${chars.length>48?'…':''}`);
    article.open=index===0;article.append(summary);
    article.append(spanText(item.text,(item.false_negative||[]).map(e=>Array.isArray(e)?e.slice(0,2):[e.start,e.end])));
    article.append(row('Gold 实体',entityStrip(item.text,item.ner_gold,item.false_negative)));
    article.append(row('预测实体',entityStrip(item.text,item.ner_prediction,item.false_positive)));
    const missed=entityStrip(item.text,item.false_negative),extra=entityStrip(item.text,item.false_positive);
    article.append(row('漏检 / 边界或类型差异',missed),row('多检 / 边界或类型差异',extra));return article;
  }
  async function load(){const call=++requestId;const params=new URLSearchParams({page:String(page),page_size:ner?'4':'5',category:$('error-category').value});if(ner){if(!ctx.nerId())return fail(new Error('unavailable：没有 NER 正式 Run'));params.set('run_id',ctx.nerId());}else{params.set('model',ctx.model());if(ctx.model()==='hmm1-smooth'&&ctx.runId())params.set('run_id',ctx.runId());}
    const value=await api(`${ner?'/api/v1/experiments/1/errors':'/api/v1/views/cws/errors'}?${params}`);if(call!==requestId)return;$('error-list').replaceChildren();
    value.items.forEach((item,index)=>$('error-list').append(ner?nerCase(item,index):cwsCase(item)));
    if(!value.items.length)$('error-list').append(node('p','当前筛选没有错误案例。'));
    $('errors-page').textContent=`第 ${page}/${Math.max(1,Math.ceil(value.total/value.page_size))} 页 · 共 ${value.total} 条`;$('errors-prev').disabled=page<=1;$('errors-next').disabled=page*value.page_size>=value.total;
  }
  $('errors-prev').addEventListener('click',()=>{page--;load().catch(fail);});$('errors-next').addEventListener('click',()=>{page++;load().catch(fail);});$('error-category').addEventListener('change',()=>{page=1;load().catch(fail);});
  return ()=>{page=1;return load();};
}
window.addEventListener('resize',()=>{resizeCharts();resizeAlgorithm();});
