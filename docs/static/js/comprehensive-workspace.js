/* One shared object hierarchy. Pure selectors never invent predictions. */
export const NEWS_LABELS=['finance','education','science','sports','entertainment'];
export const NEWS_NAMES={finance:'财经',education:'教育',science:'科技',sports:'体育',entertainment:'娱乐'};
export const ENTITY_TYPES=['PER','LOC','ORG'];
export const CLASS_COLORS={finance:'#b48d50',education:'#8e7ba4',science:'#168e98',sports:'#5e88b9',entertainment:'#b57c88',unknown:'#b8c2cb'};
export const CLUSTER_COLORS=['#168e98','#6d8db2','#b19b6e','#8c9b72','#9887a9','#ac8388','#668f99'];

export function summaryOf(profile) {
  const a=profile?.analysis;if(!a)return null;
  const ck=a.modules?.classification?.status==='ok'&&NEWS_LABELS.includes(a.classification?.prediction);
  const nk=a.modules?.ner?.status==='ok',pk=a.modules?.pos?.status==='ok';
  const pos_counts={};if(pk)for(const p of a.lexical?.pos||[])pos_counts[p.pos]=(pos_counts[p.pos]||0)+1;
  return {id:profile.id,status:a.status,prediction:ck?a.classification.prediction:null,
    classification_known:ck,ner_known:nk,pos_known:pk,
    entities:nk?a.lexical?.entities||[]:null,entity_count:nk?(a.lexical?.entities||[]).length:null,
    entity_types:nk?[...new Set((a.lexical?.entities||[]).map(e=>e.type))]:[],pos_counts:pk?pos_counts:null};
}

function mergeSummary(previous,row) {
  const merged={...previous,...row};
  // A link or a partial profile may know only one module. Keep observed outputs.
  if(previous?.classification_known&&!row.classification_known){
    for(const [key,value] of Object.entries(previous))if(key.startsWith('classification')||['prediction','runner_up','top_probability','probabilities','confidence'].includes(key))merged[key]=value;
  }
  if(previous?.ner_known&&!row.ner_known)for(const key of ['ner_known','entities','entity_count','entity_types'])merged[key]=previous[key];
  if(previous?.pos_known&&!row.pos_known)for(const key of ['pos_known','pos_counts'])merged[key]=previous[key];
  return merged;
}

export function matchingDocuments(state,{ignoreBrush=false}={}) {
  const f=state.filters;
  return (state.corpus?.documents||[]).filter(d=>{
    const s=state.summaries[d.id];
    if(f.cluster!==null&&String(d.cluster_id)!==String(f.cluster))return false;
    if(!ignoreBrush&&state.brushIds!==null&&!state.brushIds.includes(d.id))return false;
    if(f.prediction!==null&&(!s?.classification_known||s.prediction!==f.prediction))return false;
    return f.entities.every(e=>s?.ner_known&&(s.entities||[]).some(item=>item.type===e.type&&(!e.text||item.text===e.text)));
  });
}

export function selectionStats(state,documents=matchingDocuments(state)) {
  const classCounts=Object.fromEntries(NEWS_LABELS.map(k=>[k,0]));
  const entityCounts=Object.fromEntries(ENTITY_TYPES.map(k=>[k,0]));
  const flows=new Map(),entities=new Map(),clusters=new Map();let classified=0,ner=0,classifiedEligible=0,nerEligible=0;
  for(const d of documents){
    const s=state.summaries[d.id];const c=clusters.get(d.cluster_id)||{id:d.cluster_id,total:0,eligible:0,analyzed:0,classified:0,ner:0,classifiedEligible:0,nerEligible:0};
    c.total++;if(d.eligible)c.eligible++;if(s?.status==='ok')c.analyzed++;
    if(s?.classification_known&&NEWS_LABELS.includes(s.prediction)){
      classified++;c.classified++;if(d.eligible){classifiedEligible++;c.classifiedEligible++;}classCounts[s.prediction]++;
      const key=JSON.stringify([d.cluster_id,s.prediction]);flows.set(key,(flows.get(key)||0)+1);
    }
    if(s?.ner_known){ner++;c.ner++;if(d.eligible){nerEligible++;c.nerEligible++;}for(const e of s.entities||[]){
      if(!ENTITY_TYPES.includes(e.type))continue;entityCounts[e.type]++;
      const key=JSON.stringify([d.cluster_id,e.type,e.text]);entities.set(key,(entities.get(key)||0)+1);
    }}clusters.set(d.cluster_id,c);
  }
  return {total:documents.length,eligible:documents.filter(d=>d.eligible).length,classified,ner,classifiedEligible,nerEligible,
    unknownClassification:documents.length-classified,unknownNER:documents.length-ner,
    classCounts:classified?classCounts:null,entityCounts:ner?entityCounts:null,
    flows:[...flows].map(([key,count])=>{const [cluster_id,prediction]=JSON.parse(key);return {cluster_id,prediction,count};}),
    entities:[...entities].map(([key,count])=>{const [cluster_id,type,text]=JSON.parse(key);return {cluster_id,type,text,count};}).sort((a,b)=>b.count-a.count||a.text.localeCompare(b.text)),
    clusters:[...clusters.values()].sort((a,b)=>a.id-b.id)};
}

export function createWorkspace() {
  let state={corpus:null,summaries:{},profiles:{},documentRef:null,brushIds:null,
    filters:{cluster:null,prediction:null,entities:[]},span:null,focus:null,
    layers:{tokens:true,pos:true,entities:true,dependency:true},deepResults:{},deepTexts:{},
    selectionTerms:null,colorBy:'cluster',entityMetric:'mentions_per_ner_document',view:'single',history:[],generation:0,documentGeneration:0};
  const listeners=new Set();
  const emit=()=>listeners.forEach(fn=>fn(state));
  const docKey=ref=>ref?`${ref.corpusId||'manual'}/${ref.id}`:null;
  function snapshot(){const current=state.profiles[docKey(state.documentRef)];return {draftText:state.documentRef?.id==='draft'&&!state.documentRef.corpusId?current?.original_text:undefined,view:state.view,documentRef:state.documentRef?{...state.documentRef}:null,
    brushIds:state.brushIds===null?null:[...state.brushIds],filters:structuredClone(state.filters),span:state.span?{...state.span}:null,focus:state.focus?{...state.focus}:null};}
  function record(){state.history.push(snapshot());if(state.history.length>24)state.history.shift();}
  function choose(profile,{push=false}={}){
    const ref={corpusId:Object.hasOwn(profile,'corpus_id')?profile.corpus_id:(state.corpus?.id??null),id:profile.id};
    if(push)record();if(docKey(ref)!==docKey(state.documentRef)){state.span=null;state.focus=null;state.documentGeneration++;}
    state.profiles[docKey(ref)]=profile;state.documentRef=ref;
    if(ref.corpusId===state.corpus?.id){const summary=summaryOf(profile);if(summary)state.summaries[profile.id]=mergeSummary(state.summaries[profile.id],summary);}
    emit();
  }
  return {
    getState:()=>state,
    subscribe(fn){listeners.add(fn);return ()=>listeners.delete(fn);},
    getDocument:()=>state.profiles[docKey(state.documentRef)]||null,
    documentKey:()=>docKey(state.documentRef),
    setCorpus(corpus,link=null){state.corpus=corpus;state.summaries={};state.profiles={};state.documentRef=null;
      state.brushIds=null;state.filters={cluster:null,prediction:null,entities:[]};state.span=null;state.focus=null;state.deepResults={};state.deepTexts={};state.selectionTerms=null;state.history=[];
      state.generation++;state.documentGeneration++;if(link)for(const row of link.documents||[])state.summaries[row.id]=mergeSummary(state.summaries[row.id],row);emit();},
    mergeLink(link){for(const row of link.documents||[])state.summaries[row.id]=mergeSummary(state.summaries[row.id],row);emit();},
    chooseDocument:choose,
    putProfile(profile,{activate=false}={}){
      const ref={corpusId:Object.hasOwn(profile,'corpus_id')?profile.corpus_id:(state.corpus?.id??null),id:profile.id};state.profiles[docKey(ref)]=profile;
      if(ref.corpusId===state.corpus?.id){const s=summaryOf(profile);if(s)state.summaries[profile.id]=mergeSummary(state.summaries[profile.id],s);}
      if(activate){choose(profile);return;}emit();
    },
    setDraft(text){const id='draft';if(state.profiles['manual/draft']?.original_text!==text){delete state.deepResults['manual/draft'];delete state.deepTexts['manual/draft'];}state.documentRef={corpusId:null,id};state.profiles[`manual/${id}`]={id,corpus_id:null,original_text:text,analysis:null,title:'输入文本',eligible:true};
      state.span=null;state.focus=null;state.documentGeneration++;emit();},
    setManualDocument(analysis){choose({id:analysis.id,corpus_id:null,title:'当前文本',original_text:analysis.original_text,analysis,eligible:true});},
    selectSpan(span,focus=null){state.span=span?{...span,documentKey:docKey(state.documentRef)}:null;state.focus=focus||{kind:'span'};emit();},
    setFocus(focus){state.focus=focus;emit();},
    setFilter(key,value){record();state.filters[key]=value;emit();},
    addEntity(entity){record();if(!state.filters.entities.some(e=>e.type===entity.type&&e.text===entity.text))state.filters.entities.push({...entity});emit();},
    removeEntity(index){record();state.filters.entities.splice(index,1);emit();},
    setBrush(ids){const value=ids===null?null:[...new Set(ids)].sort();if(JSON.stringify(value)===JSON.stringify(state.brushIds))return;
      record();state.brushIds=value;emit();},
    reset(){record();state.brushIds=null;state.filters={cluster:null,prediction:null,entities:[]};emit();},
    setTerms(value){state.selectionTerms=value;emit();},
    setColorBy(value){state.colorBy=value;emit();},
    setEntityMetric(value){if(['mentions_per_ner_document','mention_count'].includes(value)){state.entityMetric=value;emit();}},
    setLayer(key,value){state.layers[key]=value;emit();},
    setDeep(kind,value){const key=docKey(state.documentRef);const profile=state.profiles[key];state.deepTexts[key]=profile?.analysis?.original_text??profile?.original_text;state.deepResults[key]??={};state.deepResults[key][kind]=value;emit();},
    getDeep:()=>{const key=docKey(state.documentRef),profile=state.profiles[key];return state.deepTexts[key]===(profile?.analysis?.original_text??profile?.original_text)?state.deepResults[key]||{}:{};},
    navigate(view,{push=true}={}){if(view!==state.view&&push)record();state.view=view;emit();},
    back(){const old=state.history.pop();if(!old)return false;const {draftText,...context}=old;Object.assign(state,context);if(draftText!==undefined&&state.profiles['manual/draft'])state.profiles['manual/draft']={...state.profiles['manual/draft'],original_text:draftText};state.documentGeneration++;emit();return true;}
  };
}

export const workspace=createWorkspace();
