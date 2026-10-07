import {activeTab,selectTab} from './layout.js';
import {workspace,matchingDocuments,selectionStats,NEWS_NAMES} from './comprehensive-workspace.js';
import {sentenceContext,splitDocuments} from './comprehensive-context.js';
import {createDocumentView} from './comprehensive-document.js';
import {createCorpusView} from './comprehensive-corpus.js';

const $=id=>document.getElementById(id),api='/api/v1/comprehensive';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pending=new Map();let disposed=false,termSignature='',termTimer=null,analyzing=false,cases=[];
function message(text,error=false,id='cw-status'){const el=$(id);el.textContent=text;el.classList.toggle('is-error',error);}
function cancel(key){pending.get(key)?.abort();pending.delete(key);}
function cancelDocument(){cancel('document');cancel('analyze');cancel('deep');$('cw-context-action').hidden=true;}
async function request(key,path,body){cancel(key);const controller=new AbortController();pending.set(key,controller);
  try{const response=await fetch(api+path,{method:body===undefined?'GET':'POST',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal});
    const data=await response.json();if(!response.ok)throw new Error(data.error?.message||(Array.isArray(data.detail)?data.detail.map(d=>d.msg).join('；'):data.detail)||`请求失败 ${response.status}`);
    if(pending.get(key)!==controller)throw new DOMException('Stale','AbortError');return data;
  }finally{if(pending.get(key)===controller)pending.delete(key);}
}
function fail(error,id='cw-status'){if(error.name!=='AbortError')message(error.message,true,id);}
function navigate(view,{push=true}={}){view=view==='collection'?'documents':view;workspace.navigate(view,{push});selectTab(view);}
function objectText(){const p=workspace.getDocument();return p?.analysis?.original_text??p?.original_text??'';}
function setButton(id,busy,label){$(id).disabled=busy;if(label)$(id).textContent=label;$(id).setAttribute('aria-busy',String(busy));}

function renderContext(state){
  const c=state.corpus,doc=workspace.getDocument(),docs=matchingDocuments(state);
  const chips=[`<span class="cw-chip">${c?(c.mode==='course'?'THUCNews-600':'自定义语料'):'独立文本'}</span>`];
  if(state.filters.cluster!==null)chips.push(`<button class="cw-chip" data-cw-remove="cluster">Cluster ${esc(state.filters.cluster)} ×</button>`);
  if(state.brushIds!==null)chips.push(`<button class="cw-chip" data-cw-remove="brush">刷选 ${state.brushIds.length} 篇 ×</button>`);
  if(state.filters.prediction!==null)chips.push(`<button class="cw-chip" data-cw-remove="prediction">${esc(NEWS_NAMES[state.filters.prediction])} prediction ×</button>`);
  state.filters.entities.forEach((e,i)=>chips.push(`<button class="cw-chip" data-cw-remove-entity="${i}">${esc(e.type)}${e.text?`: ${esc(e.text)}`:''} ×</button>`));
  if(c)chips.push(`<span class="cw-chip"><strong>${docs.length}</strong> / ${c.documents.length} docs</span>`);
  if(doc)chips.push(`<span class="cw-chip" title="${esc(doc.id)}">文档：${esc((doc.title||doc.id).slice(0,28))}</span>`);
  $('cw-context-path').className='cw-context-path';$('cw-context-path').innerHTML=chips.join('<span>/</span>');$('cw-back').disabled=!state.history.length;
  $('cw-document-source').textContent=doc?(state.documentRef?.corpusId?`${state.corpus?.mode==='course'?'课程文档':'临时文档'} · ${doc.id} · Cluster ${doc.clustering?.cluster_id??'—'}`:'独立输入文本'):'选择一篇文档或输入正文';
  const text=objectText();if($('ca-text').value!==text&&document.activeElement!==$('ca-text'))$('ca-text').value=text;
  $('ca-length').textContent=`${Array.from($('ca-text').value).length} 字`;
  $('ca-analyze').disabled=analyzing||!doc?.eligible;
  $('cw-doc-to-relations').disabled=!state.corpus;
}
let documentView,corpusView;
function render(state){if(disposed)return;renderContext(state);corpusView?.render();queueTerms(state);}
function queueTerms(state){if(!state.corpus)return;const ids=matchingDocuments(state).map(d=>d.id).sort(),signature=JSON.stringify([state.corpus.id,ids]);
  if(signature===termSignature)return;termSignature=signature;clearTimeout(termTimer);cancel('terms');
  if(!ids.length){workspace.setTerms({document_ids:[],terms:[],source:'empty selection'});return;}
  const corpus=state.corpus;
  termTimer=setTimeout(async()=>{try{const data=await request('terms',`/collections/${corpus.id}/terms`,{document_ids:ids});
    if(workspace.getState().corpus===corpus&&termSignature===signature)workspace.setTerms(data);
  }catch(error){if(error.name!=='AbortError'&&workspace.getState().corpus===corpus&&termSignature===signature)workspace.setTerms({document_ids:ids,terms:null,source:error.message});}},250);
}
async function installCorpus(collection){cancelDocument();cancel('batch');cancel('terms');termSignature='';workspace.setCorpus(collection);$('cw-corpus-message').textContent=collection.notice||'';
  try{const snapshot=await request('snapshot',`/collections/${collection.id}/workspace`);if(workspace.getState().corpus?.id===collection.id)workspace.mergeLink(snapshot.link);}catch(error){fail(error);}
}
async function loadCorpus(custom=false){const button=custom?'ca-cluster':'ca-load-course';let body;
  if(custom){const texts=splitDocuments($('ca-documents').value),k=Number($('ca-k').value);
    if(texts.some(t=>!t.trim())||texts.length<3||texts.length>30)return message('提供3～30篇非空正文；空格和空行不会被改写。',true);
    if(!Number.isInteger(k)||k<2||k>=texts.length||k>29)return message('K需为2到文档数-1的整数。',true);
    if(texts.some(t=>Array.from(t).length>5000)||texts.reduce((n,t)=>n+Array.from(t).length,0)>50000)return message('单篇最多5000字，总计最多50000字，不截断。',true);
    body={documents:texts.map((text,i)=>({title:`自定义文档 ${i+1}`,text})),k};
  }setButton(button,true);message(custom?'正在执行隔离临时聚类…':'正在读取冻结课程语料…');
  try{const c=await request('corpus',custom?'/custom':'/course',body);await installCorpus(c);navigate('documents');message(`${c.documents.length}篇已载入；选择Cluster或刷选区域。所有新分析都需明确触发。`);}catch(error){fail(error);}finally{setButton(button,false);}
}
async function buildIndex(){
  setButton('cw-classification-index',true,'分类索引构建中…');message('仅对Train/Validation白名单运行冻结C2；不运行NER，不进行正式评价。');
  try{const result=await request('classification-index','/classification-index/build',{});
    const corpus=workspace.getState().corpus;
    if(corpus?.mode==='course'){const snapshot=await request('snapshot',`/collections/${corpus.id}/workspace`);if(workspace.getState().corpus===corpus){
      workspace.mergeLink(snapshot.link);const ref=workspace.getState().documentRef,key=workspace.documentKey(),version=workspace.getState().documentGeneration;
      if(ref?.corpusId===corpus.id){const profile=await request('index-profile',`/collections/${corpus.id}/documents/${encodeURIComponent(ref.id)}?analyze=false`);
        if(workspace.documentKey()===key&&workspace.getState().documentGeneration===version)workspace.chooseDocument({...profile,corpus_id:corpus.id});}
    }}
    message(`应用分类已知 ${result.classification_known}/${result.eligible}；本次推理 ${result.new_inferences} 篇，NER保持按需。`);
  }catch(error){fail(error);}finally{setButton('cw-classification-index',false,'建立 / 补全应用分类索引');}
}
async function openDocument(id,{analyze=false,drill=true}={}){
  const corpus=workspace.getState().corpus;if(!corpus)return;const row=corpus.documents.find(d=>d.id===id);if(!row)return;
  cancelDocument();workspace.chooseDocument({...row,corpus_id:corpus.id,original_text:null,analysis:null,clustering:{cluster_id:row.cluster_id,x:row.x,y:row.y}}, {push:true});
  const expected=workspace.documentKey();if(drill)navigate('single');message('正在读取当前文档…');
  try{const profile=await request('document',`/collections/${corpus.id}/documents/${encodeURIComponent(id)}?analyze=${analyze}`);
    if(workspace.getState().corpus===corpus&&workspace.documentKey()===expected){workspace.chooseDocument({...profile,corpus_id:corpus.id});message(profile.notice||'文档已打开；选择词项/实体，或显式运行基础分析。');}
  }catch(error){fail(error);}
}
async function analyzeDocument(){const profile=workspace.getDocument(),ref=workspace.getState().documentRef;if(!profile)return;
  const text=objectText();if(!text.trim()||!Array.from(text).some(c=>/[\p{L}\p{N}]/u.test(c)))return message('请输入可分析的正文（不能只有标点）。',true,'ca-single-status');
  if(Array.from(text).length>5000)return message('正文超过5000字；不自动截断。',true,'ca-single-status');
  const key=workspace.documentKey(),version=workspace.getState().documentGeneration;analyzing=true;setButton('ca-analyze',true,'分析中…');message('正在执行分词/POS/NER/C2；不自动触发深层分析。');
  try{let result;if(ref.corpusId){const p=await request('analyze',`/collections/${ref.corpusId}/documents/${encodeURIComponent(ref.id)}?analyze=true`);result={...p,corpus_id:ref.corpusId};}
    else{const a=await request('analyze','/analyze',{text});result={id:a.id,corpus_id:null,title:'当前文本',original_text:a.original_text,analysis:a,eligible:true};}
    if(workspace.documentKey()===key&&workspace.getState().documentGeneration===version){workspace.chooseDocument(result);message(result.analysis.status==='ok'?'分析完成，所有画像与原文选择联动。':'部分模块未完成；有效结果保留，可重试。');}
  }catch(error){fail(error);}finally{analyzing=false;setButton('ca-analyze',false,'分析当前文档');renderContext(workspace.getState());}
}
async function analyzeSelection(){const state=workspace.getState(),corpus=state.corpus;if(!corpus)return message('先载入语料。',true);
  const ids=matchingDocuments(state).map(d=>d.id);setButton('cw-build-sample',true);setButton('cw-analyze-selection',true);message('按当前scope请求确定性簇均衡抽样…');
  try{const plan=await request('plan',`/collections/${corpus.id}/sampling-plan`,{scope_ids:ids,limit:5});
    if(workspace.getState().corpus!==corpus)return;let done=0;
    for(const id of plan.document_ids){if(workspace.getState().corpus!==corpus)return;message(`分析均衡样本 ${done+1}/${plan.document_ids.length}；按已保存Cluster，不使用gold类别。`);
      const profile=await request('batch',`/collections/${corpus.id}/documents/${encodeURIComponent(id)}?analyze=true`);
      if(workspace.getState().corpus!==corpus)return;workspace.putProfile({...profile,corpus_id:corpus.id});done++;
    }const snapshot=await request('snapshot',`/collections/${corpus.id}/workspace`);if(workspace.getState().corpus===corpus){workspace.mergeLink(snapshot.link);message(done?`本批完成${done}篇；所有视图按当前交集即时更新。`:'当前scope无未完成的可分析文档；保留已有统计。');}
  }catch(error){fail(error);}finally{setButton('cw-build-sample',false);setButton('cw-analyze-selection',false);}
}
function contextAction(kind){const profile=workspace.getDocument(),text=objectText();if(!profile?.eligible||!text)return message('当前对象没有允许分析的完整原文。',true);
  const span=workspace.getState().span,limit=kind==='dependency'?300:kind==='summary'?500:5000;
  const range=!span?{start:0,end:Array.from(text).length,text,oversize:Array.from(text).length>limit}:sentenceContext(text,span,limit);
  $('cw-context-action').hidden=false;$('cw-action-kind').value=kind;$('cw-action-start').value=range.start;$('cw-action-end').value=range.end;previewContext();
  message(range.oversize?'当前原文选区超过模块上限，请明确选取句子或调整范围；不会截断。':'确认选区后运行；生成结果仍属于当前完整文档上下文。',Boolean(range.oversize),'cw-action-status');
}
function previewContext(){const text=Array.from(objectText()),a=Number($('cw-action-start').value),b=Number($('cw-action-end').value);$('cw-action-preview').textContent=text.slice(a,b).join('');}
async function runContext(){const key=workspace.documentKey(),version=workspace.getState().documentGeneration,text=objectText(),kind=$('cw-action-kind').value,start=Number($('cw-action-start').value),end=Number($('cw-action-end').value),limit={dependency:300,summary:500,information_extraction:5000}[kind];
  if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end<=start||end>Array.from(text).length||end-start>limit)return message(`选区需连续、有效且不超过${limit}字。`,true,'cw-action-status');
  setButton('cw-run-action',true,'运行中…');message('正在显式运行当前原文选区…',false,'cw-action-status');
  try{const result=await request('deep','/optional',{text,kind,start,end});if(workspace.documentKey()!==key||workspace.getState().documentGeneration!==version||objectText()!==text)return;workspace.setDeep(kind,result);message('真实输出已对齐回全篇原文，点击相关词项/实体观察。',false,'cw-action-status');}
  catch(error){fail(error,'cw-action-status');}finally{setButton('cw-run-action',false,'运行当前选区');}
}
function applyEntity(entity){workspace.addEntity(entity);message(`已带入${entity.type}${entity.text?`: ${entity.text}`:''}条件；可切换关联或语料视图继续交叉筛选。`);}
function sourceLink(module){const paths={classification:'/classification',dependency:'/dependency-parsing',summary:'/deep-learning#llm',information_extraction:'/information-extraction'};window.location.assign(paths[module]||'/tagging');}
function renderCases(items){$('cw-replay-cases').innerHTML=items.map(c=>`<button class="cw-case" data-cw-case="${esc(c.id)}"><strong>${esc(c.title)}</strong><small>${esc(c.expected)}</small></button>`).join('');}
function replayCase(id){const item=cases.find(c=>c.id===id);if(!item)return;if(item.workspace==='single'){$('ca-text').value=item.text;cancelDocument();workspace.setDraft(item.text);navigate('single');message('案例已填入；点击分析真实运行。');}
  else{$('ca-source').value='custom';$('cw-custom-input').open=true;$('ca-documents').value=item.documents.map(d=>d.text).join('\n---\n');$('ca-k').value=item.k;navigate('documents');message('案例已填入；执行临时聚类观察真实状态。');}}
async function tests(){setButton('ca-tests',true,'测试运行中…');message('正在运行应用案例…',false,'ca-test-status');
  try{const data=await request('tests','/system-tests',{});$('ca-test-records').innerHTML=data.records.map(r=>`<div class="cw-test-record ${r.passed?'':'is-failed'}"><strong>${esc(r.name)}<small> · ${esc(r.category)}</small></strong><p>${esc(r.actual)}<br><small>预期：${esc(r.expected)}</small></p><span>${r.passed?'通过':'未通过'} · ${Math.round(r.elapsed_ms||0)}ms</span></div>`).join('');message(`${data.records.filter(r=>r.passed).length}/${data.records.length}通过。${data.notice}`,false,'ca-test-status');}
  catch(error){fail(error,'ca-test-status');}finally{setButton('ca-tests',false,'运行案例测试');}}

documentView=createDocumentView({workspace,host:$('ca-doc-canvas'),inspector:$('ca-doc-inspector'),onAnalyze:analyzeDocument,onDraft:text=>workspace.setDraft(text),onContextAction:contextAction,onEntityFilter:applyEntity,onNavigate:navigate,onSourceLink:sourceLink});
corpusView=createCorpusView({workspace,scatterHost:$('ca-corpus-canvas'),listHost:$('ca-corpus-list'),inspectorHost:$('ca-corpus-inspector'),statsHost:$('ca-selection-stats'),flowHost:$('ca-relations-canvas'),entityHost:$('ca-entity-profile'),relationInspectorHost:$('ca-relations-inspector'),onOpenDocument:id=>openDocument(id),onAnalyzeSelection:analyzeSelection,onNavigate:navigate});
workspace.subscribe(render);
$('ca-text').addEventListener('input',()=>{cancelDocument();workspace.setDraft($('ca-text').value);message('原文已更新；点击分析，不沿用旧模型结果。',false,'ca-single-status');});
$('ca-analyze').addEventListener('click',analyzeDocument);$('ca-clear').addEventListener('click',()=>{cancelDocument();workspace.setDraft('');$('ca-text').value='';$('ca-text').focus();});
$('ca-example').addEventListener('click',()=>{cancelDocument();workspace.setDraft('微软公司在北京举办技术交流活动。张明介绍人工智能软件，并计划与北京大学合作研究。');});
$('cw-classification-index').addEventListener('click',buildIndex);
$('ca-load-course').addEventListener('click',()=>loadCorpus());$('ca-cluster').addEventListener('click',()=>loadCorpus(true));$('ca-source').addEventListener('change',()=>{$('cw-custom-input').open=$('ca-source').value==='custom';});
$('cw-build-sample').addEventListener('click',analyzeSelection);$('cw-analyze-selection').addEventListener('click',analyzeSelection);$('ca-tests').addEventListener('click',tests);
$('cw-reset').addEventListener('click',()=>workspace.reset());$('cw-back').addEventListener('click',()=>{if(workspace.back())selectTab(workspace.getState().view);});$('cw-doc-to-relations').addEventListener('click',()=>navigate('linkage'));
$('cw-close-action').addEventListener('click',()=>{$('cw-context-action').hidden=true;});$('cw-run-action').addEventListener('click',runContext);for(const id of ['cw-action-start','cw-action-end'])$(id).addEventListener('input',previewContext);
$('cw-action-kind').addEventListener('change',()=>contextAction($('cw-action-kind').value));
document.querySelectorAll('[data-cw-layer]').forEach(input=>input.addEventListener('change',()=>workspace.setLayer(input.dataset.cwLayer,input.checked)));
document.querySelector('.cw-page').addEventListener('click',event=>{const remove=event.target.closest('[data-cw-remove]'),entity=event.target.closest('[data-cw-remove-entity]'),replay=event.target.closest('[data-cw-case]');
  if(remove){if(remove.dataset.cwRemove==='brush')workspace.setBrush(null);else workspace.setFilter(remove.dataset.cwRemove,null);}
  if(entity)workspace.removeEntity(Number(entity.dataset.cwRemoveEntity));if(replay)replayCase(replay.dataset.cwCase);
});
document.addEventListener('module-tab',event=>{if(workspace.getState().view!==event.detail)workspace.navigate(event.detail);requestAnimationFrame(()=>{corpusView.resize();documentView.render();});});
window.addEventListener('resize',()=>corpusView.resize());
workspace.navigate(activeTab()||'single',{push:false});workspace.setDraft('');
Promise.allSettled([request('status','/status'),request('cases','/cases')]).then(results=>{const a=results[0],b=results[1];if(a.status==='fulfilled')$('cw-test-resources').textContent=a.value.resources.map(r=>`${r.label}：${r.ready?'资源可用':'未准备'}`).join(' · ');if(b.status==='fulfilled'){cases=b.value.cases;renderCases(cases);}else fail(b.reason);});
