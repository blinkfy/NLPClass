import {compute} from './static-compute.js';
import {wordpieces} from './static-wordpiece.js';
import {whitespace} from './static-transformers.js';

export function eisner(scores) {
  const n=scores.length-1,table=initial=>Array.from({length:n+1},()=>Array(n+1).fill(initial));
  const incomplete=table(-Infinity),complete=table(-Infinity),pi=table(0),pc=table(0),heads=Array(n+1).fill(0);
  for(let i=0;i<=n;i++)complete[i][i]=0;
  for(let width=1;width<=n;width++)for(let i=0;i+width<=n;i++) {
    const j=i+width;let best=-Infinity,split=i;
    for(let r=i;r<j;r++){const value=Math.fround(complete[i][r]+complete[j][r+1]);if(value>best){best=value;split=r;}}
    incomplete[j][i]=Math.fround(best+scores[i][j]);pi[j][i]=split;
    incomplete[i][j]=Math.fround(best+scores[j][i]);pi[i][j]=split;
    best=-Infinity;split=i;
    for(let r=i;r<j;r++){const value=Math.fround(complete[r][i]+incomplete[j][r]);if(value>best){best=value;split=r;}}
    complete[j][i]=best;pc[j][i]=split;
    best=-Infinity;split=i+1;
    for(let r=i+1;r<=j;r++){const value=Math.fround(incomplete[i][r]+complete[r][j]);if(value>best){best=value;split=r;}}
    complete[i][j]=i===0&&j!==n?-Infinity:best;pc[i][j]=split;
  }
  function backtrack(i,j,isComplete) {
    if(i===j)return;
    if(isComplete){const r=pc[i][j];backtrack(i,r,false);backtrack(r,j,true);}
    else {const r=pi[i][j];heads[j]=i;const [a,b]=[i,j].sort((x,y)=>x-y);backtrack(a,r,true);backtrack(b,r+1,true);}
  }
  if(!Number.isFinite(complete[0][n]))throw new Error('依存模型没有有效的单根投射树');
  backtrack(0,n,true);
  return heads.slice(1);
}
export function validateTree(heads) {
  const n=heads.length;if(!n||heads.filter(h=>h===0).length!==1)throw new Error('依存树必须恰有一个ROOT');
  for(let node=1;node<=n;node++){const seen=new Set();let at=node;while(at){if(at<0||at>n||seen.has(at))throw new Error('依存树出现循环或非法head');seen.add(at);at=heads[at-1];}}
  return {single_root:true,acyclic:true,root_reachable:true,node_count:n};
}
export function decodeRelations(arc,relation,labels) {
  const heads=eisner(arc);validateTree(heads);
  const traces=heads.map((head,i)=>{
    const values=relation[i+1][head],order=labels.map((_,j)=>j).sort((a,b)=>values[b]-values[a]||a-b);
    return {dependent:i+1,head,relation:labels[order[0]],verified:true,score_type:'biaffine relation logit, not probability',candidates:order.map(index=>({index,label:labels[index],score:values[index]}))};
  });
  return {heads,relations:traces.map(t=>t.relation),relation_scores:traces};
}
const explanations={root:'句子根节点',nsubj:'名词性主语',nsubjpass:'被动主语',dobj:'直接宾语',iobj:'间接宾语',nn:'名词修饰',amod:'形容词修饰',advmod:'状语修饰',assmod:'关联修饰',assm:'关联标记',det:'限定词',clf:'量词',nummod:'数量修饰',prep:'介词',pobj:'介词宾语',tmod:'时间修饰',loc:'方位关系',lobj:'方位宾语',asp:'体标记',aux:'助动词',neg:'否定修饰',conj:'并列关系',cc:'并列连接词',ccomp:'从句补语',xcomp:'开放从句补语',rcmod:'关系从句修饰',cop:'系词',punct:'标点',dep:'其他依存关系',attr:'属性关系'};
export async function parseDependency(text,descriptor,{load,infer}) {
  if(typeof text!=='string'||whitespace(text)||Array.from(text).length>300)throw new Error('依存输入限1–300个字符');
  const model=await load(descriptor),chars=Array.from(text),spans=[];
  let start=0;
  for(let i=0;i<=chars.length;i++) {
    if(i<chars.length&&!['。','！','？','!','?','\n'].includes(chars[i]))continue;
    if(i<chars.length&&chars[i]!=='\n'){while(i+1<chars.length&&'。！？!?'.includes(chars[i+1]))i++;i++;}
    if(!whitespace(chars.slice(start,i).join('')))spans.push({text:chars.slice(start,i).join(''),start,end:i});
    start=i+((chars[i]==='\n')?1:0);
    if(i===chars.length)break;
    if(chars[i]!=='\n')i--;
  }
  if(!spans.length||spans.length>6)throw new Error('依存分析限1–6个句子');
  const sentences=[];
  for(const span of spans) {
    const value=await compute('/api/v1/analyze',new URLSearchParams(),{text:span.text,tasks:['segmentation'],run_id:'20260926_130612_32a5e168_smooth',segmentation_model:'hmm1-smooth'});
    const words=value.segmentation.tokens.filter(w=>!whitespace(w));
    if(words.length<1||words.length>60)throw new Error('依存分析每句限1–60个非空词');
    const ids=[101],tokenSpans=[[0]];
    for(const word of words){const normalized=Array.from(word,c=>model.normalization[c]??c).join(''),pieces=wordpieces(normalized,model.vocabulary);tokenSpans.push([ids.length]);ids.push(...(pieces.length?pieces:[100]));}
    ids.push(102);if(ids.length>model.max_subwords)throw new Error('依存输入超过512个WordPiece，请缩短文本');
    const raw=await infer(descriptor,ids,tokenSpans),n=words.length+1,l=model.labels.length;
    const arc=Array.from({length:n},(_,i)=>raw.tensors.arc.values.slice(i*n,(i+1)*n));
    const relation=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>raw.tensors.relation.values.slice((i*n+j)*l,(i*n+j+1)*l)));
    const prediction=decodeRelations(arc,relation,model.labels);let cursor=span.start;
    const items=words.map((word,i)=>{while(chars.slice(cursor,cursor+Array.from(word).length).join('')!==word&&cursor<span.end)cursor++;const at=cursor;cursor+=Array.from(word).length;const label=prediction.relations[i];return {id:i+1,form:word,start:at,end:cursor,head:prediction.heads[i],relation:label,explanation:explanations[label]||'保留模型原始标签'};});
    sentences.push({...span,items,checks:validateTree(prediction.heads),arc_scores:arc,relation_scores:prediction.relation_scores,decoder:{tree:true,projective:true,upstream:'Browser port of HanLP Eisner'},score_type:'biaffine arc logit, not probability',conll:items.map(r=>`${r.id}\t${r.form}\t_\t_\t_\t_\t${r.head}\t${r.relation}\t_\t_`).join('\n')});
  }
  return {text,tokenizer:'frozen HMM-1 Smooth',tokenizer_run:'20260926_130612_32a5e168_smooth',sentences,source:{model:model.model,weights_sha256:model.source_sha256,label_scheme:'Chinese Stanford Dependencies / CTB9-SD330',training:false},execution:'browser'};
}
