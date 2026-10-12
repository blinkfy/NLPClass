import {compute} from './static-compute.js';
import {classify} from './static-classification.js';
import {transformers,whitespace} from './static-transformers.js';

const records=new Map(),labelNames={finance:'财经',entertainment:'娱乐',sports:'体育',science:'科技',education:'教育'};
export function attachEvidence(text,prediction) {
  const chars=Array.from(text),parts=chars.map(c=>c.normalize('NFKC').toLowerCase()),normal=parts.join(''),offsets=[];
  parts.forEach((part,i)=>Array.from(part).forEach(()=>offsets.push(i)));
  const normalized=Array.from(normal),direct=new Map((prediction.top_contributing_features||[]).map(row=>[row.term,row]));
  const evidence=(prediction.top_relative_features||[]).map(row=>{
    const feature=Array.from(row.term),occurrences=[],seen=new Set();
    if(normal===text.normalize('NFKC').toLowerCase()&&feature.length===2)for(let i=0;i+1<normalized.length;i++)if(normalized.slice(i,i+2).join('')===row.term){
      const start=offsets[i],end=offsets[i+1]+1,key=`${start}:${end}`,original=chars.slice(start,end).join('');
      if(!seen.has(key)&&original.normalize('NFKC').toLowerCase()===row.term){seen.add(key);occurrences.push({start,end,text:original});}
    }
    const item={feature:row.term,tfidf:row.tfidf,relative_contribution:row.contribution,coefficient_difference:row.coefficient_difference,occurrences};
    const weight=direct.get(row.term);if(weight)for(const key of ['weight','direct_contribution'])if(key in weight)item[key]=weight[key];
    return item;
  });
  return {...prediction,evidence,evidence_method:'existing C2 top_relative_features; x_j*(w_pred-w_runner)',offset_space:'original_text Unicode code-point',status:'ok',domain:'五类中文新闻'};
}
export async function comprehensive(path,body,manifest) {
  if(!['/api/v1/comprehensive/analyze','/api/v1/comprehensive/optional'].includes(path))return undefined;
  const text=body.text,chars=Array.from(text||'');
  if(typeof text!=='string'||!text.trim()||chars.length>5000||!/[\p{L}\p{N}]/u.test(text))throw new Error('请输入1–5000字正文，不能只有空白或标点');
  if(path.endsWith('/optional')) {
    const started=performance.now();
    const start=body.start??0,end=body.end??chars.length;
    if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end>chars.length||start>=end)throw new Error('原文选区无效');
    const fragment=chars.slice(start,end).join(''),kind=body.kind;
    const route={dependency:'/api/v1/course/dependency/parse',information_extraction:'/api/v1/information-extraction/analyze'}[kind];
    if(!route)throw new Error('摘要生成模型尚未移植，请使用已接入的信息抽取或依存分析');
    const value=await transformers(route,{text:fragment},manifest);if(!value)throw new Error('所选可选模型未导出');
    const shift=span=>{span.start+=start;span.end+=start;};
    if(kind==='dependency')for(const sentence of value.sentences||[]){shift(sentence);for(const item of sentence.items)shift(item);}
    if(kind==='information_extraction'){for(const entity of value.entities||[])shift(entity);for(const relation of value.relations||[])for(const field of ['subject','object','evidence','trigger'])shift(relation[field]);}
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),v=>v.toString(16).padStart(2,'0')).join('');
    return {kind,document_id:'text-'+hash.slice(0,24),result:value,source_span:{start,end,text:fragment},offset_space:'original_text Unicode code-point',scope:'explicit contextual optional analysis',original_text:text,elapsed_ms:performance.now()-started};
  }
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),v=>v.toString(16).padStart(2,'0')).join('');
  const id='text-'+hash.slice(0,24);if(records.has(id))return {...structuredClone(records.get(id)),cached:true};
  const started=performance.now(),value={id,original_text:text,lexical:{tokens:[],pos:[],entities:[]},classification:{},modules:{},provenance:{},cached:false};
  const segmented=await compute('/api/v1/analyze',new URLSearchParams(),{text,tasks:['segmentation'],run_id:'20260926_130612_32a5e168_smooth',segmentation_model:'hmm1-smooth'});
  const tokens=segmented.segmentation.tokens;let at=0;
  value.lexical.tokens=tokens.map((form,i)=>{const end=at+Array.from(form).length,row={id:i+1,form,start:at,end,whitespace:whitespace(form)};at=end;return row;});
  value.modules.segmentation={status:'ok',elapsed_ms:performance.now()-started};value.provenance.segmentation={model:'HMM-1 Smooth',run_id:segmented.run.run_id,input_protocol:'original text',execution:'browser'};
  for(const kind of ['pos','ner','classification']) {
    const tick=performance.now();
    try {
      if(kind==='pos') {
        const chunks=[];let chunk=[],length=0;
        for(const token of tokens){const size=Array.from(token).length;if(chunk.length&&length+size>200){chunks.push(chunk);chunk=[];length=0;}chunk.push(token);length+=size;}if(chunk.length)chunks.push(chunk);
        let offset=0;for(const words of chunks){const part=words.join(''),result=await transformers('/api/v1/course/pos',{text:part,tokens:words},manifest);if(!result)throw new Error('POS权重未导出');value.lexical.pos.push(...result.items.map(row=>({...row,start:row.start+offset,end:row.end+offset})));offset+=Array.from(part).length;}
        value.provenance.pos={model:'hanlp-ctb9-pos-electra-small',input_protocol:'HMM token-boundary batches ≤200 chars',chunk_count:chunks.length,execution:'browser'};
      } else if(kind==='ner') {
        const result=await transformers('/api/v1/analyze',{text,tasks:['ner']},manifest);if(!result)throw new Error('NER权重未导出');value.lexical.entities=result.ner.entities;value.provenance.ner={model:result.ner.model,input_protocol:'original characters; WordPiece-aware chunks',precision:'dynamic-int8',execution:'browser'};
      } else {
        const result=await classify('/api/v1/experiments/3/stage2/predict',{text,run_id:'latest'},manifest);
        value.classification={...attachEvidence(text,result),label_names:labelNames,notice:result.notice||'SVC decision score / margin 不是校准概率；非新闻文本仍会被分到这五类之一。'};
        value.provenance.classification={model:'C2 · char χ² LinearSVC',run_id:result.run_id,input_protocol:'original full text',execution:'browser'};
      }
      value.modules[kind]={status:'ok',elapsed_ms:performance.now()-tick};
    } catch(error){value.modules[kind]={status:'unavailable',error:error.message,elapsed_ms:performance.now()-tick};}
  }
  value.status=Object.values(value.modules).every(m=>m.status==='ok')?'ok':'partial';value.elapsed_ms=performance.now()-started;
  if(value.status==='ok'){records.set(id,structuredClone(value));if(records.size>64)records.delete(records.keys().next().value);}
  return value;
}
