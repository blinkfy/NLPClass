import {compute} from './static-compute.js';
import {sequence} from './static-sequence.js';

const models=new Map(),jobs=new Map();
let worker,nextId=0;
const explanations={NN:'普通名词',NR:'专有名词',NT:'时间名词',VV:'普通动词',VA:'形容词性谓词',VC:'系动词',VE:'存在动词',JJ:'名词前修饰语',PN:'代词',AD:'副词',PU:'标点'};
const category=tag=>['NN','NR','NT'].includes(tag)?'noun':['VV','VC','VE'].includes(tag)?'verb':['VA','JJ'].includes(tag)?'adjective':tag==='PN'?'pronoun':tag==='AD'?'adverb':'other';
export const whitespace=word=>Array.from(word).every(c=>[[9,13],[28,32],[133,133],[160,160],[5760,5760],[8192,8202],[8232,8233],[8239,8239],[8287,8287],[12288,12288]].some(([a,b])=>c.codePointAt(0)>=a&&c.codePointAt(0)<=b));
async function load(descriptor) {
  if (!models.has(descriptor.path)) models.set(descriptor.path,fetch(new URL(descriptor.path,document.baseURI)).then(r=>{
    if(!r.ok)throw new Error('POS 模型词表读取失败');return r.json();
  }));
  return models.get(descriptor.path);
}
function infer(descriptor,ids,spans) {
  if(!worker) {
    worker=new Worker(new URL('./static-transformer-worker.js',import.meta.url),{type:'module'});
    worker.onmessage=({data})=>{const job=jobs.get(data.id);if(!job)return;jobs.delete(data.id);data.error?job.reject(new Error(data.error)):job.resolve(data);};
    worker.onerror=event=>{for(const job of jobs.values())job.reject(new Error(event.message||'Transformer Worker 无法启动'));jobs.clear();worker.terminate();worker=null;};
  }
  return new Promise((resolve,reject)=>{const id=++nextId;jobs.set(id,{resolve,reject});worker.postMessage({id,root:document.baseURI,path:descriptor.path,ids,spans});});
}
export async function transformers(path,body,manifest) {
  if(path==='/api/v1/course/deep/bert'&&manifest.transformers?.bert){
    const started=performance.now(),{bertTrace}=await import('./static-bert.js');
    return {...await bertTrace(body.text,manifest.transformers.bert,{load,infer}),elapsed_ms:performance.now()-started,cached:false};
  }
  if(path==='/api/v1/course/dependency/parse'&&manifest.transformers?.dependency) {
    const started=performance.now(),{parseDependency}=await import('./static-dependency.js');
    return {...await parseDependency(body.text,manifest.transformers.dependency,{load,infer}),elapsed_ms:performance.now()-started,cached:false};
  }
  if(manifest.transformers?.ner && ((path==='/api/v1/analyze'&&(body.tasks||['segmentation','ner']).includes('ner'))||path==='/api/v1/information-extraction/analyze')) {
    const text=body.text,started=performance.now();
    if(typeof text!=='string'||!text||Array.from(text).length>5000)throw new Error('NER 输入限1–5000个字符');
    const entities=await recognize(text,manifest.transformers.ner);
    if(path.includes('/information-extraction/')) {
      const {extractRelations}=await import('./static-relations.js');
      return {text,status:'ok',ner_model:'hanlp-msra-bert',entities,relations:extractRelations(text,entities),method:'NER + explicit typed trigger rules',execution:'browser; dynamic-int8 NER'};
    }
    const request={...body,tasks:['segmentation']};
    let value;
    if((body.tasks||['segmentation','ner']).includes('segmentation')) {
      value=await sequence(path,new URLSearchParams(),request,manifest)||await compute(path,new URLSearchParams(),request);
      if(!value&&body.segmentation_model==='bilstm-crf'){const {neural}=await import('./static-neural.js');value=await neural(path,new URLSearchParams(),request,manifest);}
      if(!value)throw new Error('所选分词模型未导出');
    } else value={text,segmentation:null,classification:null,run:null};
    return {...value,ner:{status:'ok',model:'hanlp-msra-bert',entities,precision:'dynamic-int8'},elapsed_ms:performance.now()-started,execution:'browser'};
  }
  if(path!=='/api/v1/course/pos'||!manifest.transformers?.pos)return undefined;
  const text=body.text;
  if(typeof text!=='string'||!text||Array.from(text).length>500)throw new Error('POS 输入限1–500个字符');
  const start=performance.now(),descriptor=manifest.transformers.pos;
  const model=await load(descriptor);
  const segmentation=body.tokens?{segmentation:{tokens:body.tokens}}:await compute('/api/v1/analyze',new URLSearchParams(),{text,tasks:['segmentation'],run_id:'latest',segmentation_model:'hmm1-smooth'});
  if(segmentation.segmentation.tokens.join('')!==text)throw new Error('POS词项必须无损拼接回原文');
  const tokens=segmentation.segmentation.tokens,words=tokens.filter(w=>!whitespace(w));
  if(!words.length)return {text,tokens,tokenizer:'frozen HMM-1 Smooth',model:model.model,items:[],counts:{},labels:{},elapsed_ms:performance.now()-start,execution:'browser'};
  const {wordpieces}=await import('./static-wordpiece.js');
  const ids=[101],spans=[];
  for(const word of words) {
    const normalized=Array.from(word,c=>model.normalization[c]??c).join('');
    const pieces=wordpieces(normalized,model.vocabulary);
    spans.push(Array.from({length:pieces.length||1},(_,i)=>ids.length+i));
    ids.push(...(pieces.length?pieces:[100]));
  }
  ids.push(102);
  if(ids.length>model.max_subwords)throw new Error(`POS 输入包含${ids.length}个WordPiece，前端上限${model.max_subwords}；请缩短文本，不会自动截断`);
  const width=Math.max(...spans.map(row=>row.length));
  const raw=await infer(descriptor,ids,spans.map(row=>[...row,...Array(width-row.length).fill(0)]));
  const labels=words.map((_,i)=>{let best=0;for(let j=1;j<model.labels.length;j++)if(raw.logits[i*model.labels.length+j]>raw.logits[i*model.labels.length+best])best=j;return model.labels[best];});
  let at=0,index=0;const items=[];
  for(const word of tokens){const end=at+Array.from(word).length;if(!whitespace(word)){const pos=labels[index++];items.push({word,pos,category:category(pos),start:at,end});}at=end;}
  const counts={};for(const item of items)counts[item.category]=(counts[item.category]||0)+1;
  return {text,tokens,tokenizer:'frozen HMM-1 Smooth',model:model.model,items,counts,
    labels:Object.fromEntries([...new Set(labels)].sort().map(tag=>[tag,explanations[tag]||'保留 HanLP 原始标签'])),elapsed_ms:performance.now()-start,execution:'browser'};
}

export function tagsToEntities(text,tags,labelMap,offset=0) {
  const chars=Array.from(text),entities=[];let previous='O',previousType='',begin=0;
  [...tags,'O'].forEach((chunk,i)=>{
    const tag=chunk[0],type=chunk.slice(2);
    const end=['E','S'].includes(previous)||(['B','I'].includes(previous)&&['B','S','O'].includes(tag))||(previous!=='O'&&previous!=='.'&&previousType!==type);
    const start=['B','S'].includes(tag)||(['E','S','O'].includes(previous)&&['E','I'].includes(tag))||(tag!=='O'&&tag!=='.'&&previousType!==type);
    if(end&&labelMap[previousType])entities.push({text:chars.slice(begin,i).join(''),type:labelMap[previousType],start:begin+offset,end:i+offset});
    if(start)begin=i;
    previous=tag;previousType=type;
  });
  return entities;
}
export async function recognize(text,descriptor) {
  const model=await load(descriptor),{wordpieces}=await import('./static-wordpiece.js'),chars=Array.from(text),entities=[];
  let offset=0;
  while(offset<chars.length) {
    const ids=[101],positions=[],chunk=[];
    while(offset+chunk.length<chars.length&&chunk.length<200) {
      const char=chars[offset+chunk.length],pieces=wordpieces(char,model.vocabulary,{lowercase:model.lowercase??false}),row=pieces.length?pieces:[100];
      if(ids.length+row.length+1>model.max_subwords)break;
      positions.push(ids.length);ids.push(...row);chunk.push(char);
      if('。！？!?\n'.includes(char))break;
    }
    ids.push(102);
    const raw=await infer(descriptor,ids,null),tags=positions.map(position=>{
      let best=1;for(let j=2;j<model.labels.length;j++)if(raw.logits[position*model.labels.length+j]>raw.logits[position*model.labels.length+best])best=j;
      return model.labels[best];
    });
    entities.push(...tagsToEntities(chunk.join(''),tags,model.label_map,offset));offset+=chunk.length;
  }
  return entities;
}
