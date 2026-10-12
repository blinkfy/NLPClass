import {tokenizeWords} from './static-classification.js';
const models=new Map(), jobs=new Map(); let worker=null, counter=0;
const STATES=['B','M','E','S'];
async function load(path) {
  if(!models.has(path)) models.set(path,fetch(new URL(path,document.baseURI)).then(response=>{
    if(!response.ok) throw new Error('神经模型静态资源读取失败'); return response.json();
  }));
  return models.get(path);
}
function run(descriptor,ids) {
  if(!worker) {
    worker=new Worker(new URL('static/js/static-neural-worker.js',document.baseURI),{type:'module'});
    worker.onmessage=({data})=>{
      const job=jobs.get(data.id); if(!job)return; jobs.delete(data.id);
      if(data.error) job.reject(new Error(data.error)); else job.resolve(data.result);
    };
    worker.onerror=event=>{
      for(const job of jobs.values()) job.reject(new Error(event.message||'浏览器神经推理线程失败'));
      jobs.clear(); worker.terminate(); worker=null;
    };
  }
  return new Promise((resolve,reject)=>{
    const id=++counter; jobs.set(id,{resolve,reject});
    worker.postMessage({id,root:document.baseURI,path:descriptor.path,ids});
  });
}
export function encodeSequence(model,tokens) {
  const max=model.vocabulary.max_len, n=tokens.length;
  const positions=n<=max?tokens.map((_,i)=>i):
    [...Array.from({length:Math.ceil(max/2)},(_,i)=>i),...Array.from({length:Math.floor(max/2)},(_,i)=>n-Math.floor(max/2)+i)];
  const vocabulary=new Map(model.vocabulary.terms.map((term,i)=>[term,i]));
  const kept=positions.map(i=>tokens[i]), ids=kept.map(token=>vocabulary.get(token)??1);
  return {tokens:kept,ids,positions,diagnostic:{original_length:n,retained_length:ids.length,
    unk_count:tokens.filter(token=>!vocabulary.has(token)).length,
    retained_unk_count:ids.filter(id=>id===1).length,truncated:n>max}};
}
export function cnnPrediction(model,raw,encoded) {
  const ranking=model.labels.map((_,i)=>i).sort((a,b)=>raw.logits[b]-raw.logits[a]||a-b), [best,second]=ranking;
  const all=raw.pooled.map((value,channel)=>{
    const kernel=Math.floor(channel/128)+3, position=raw.argmax[channel];
    const original=position<0?[]:encoded.positions.slice(position,position+kernel);
    return {channel,kernel_size:kernel,filter_id:channel%128,max_activation:value,argmax_position:position,
      token_span:position<0?[]:encoded.tokens.slice(position,position+kernel),
      model_token_span:position<0?[]:encoded.ids.slice(position,position+kernel).map(id=>model.vocabulary.terms[id]),
      original_token_positions:original,crosses_truncation_join:original.some((v,i)=>i>0&&v!==original[i-1]+1),
      predicted_class_weight:raw.weights[best][channel],contribution:value*raw.weights[best][channel]};
  });
  const sum=all.reduce((total,c)=>total+c.contribution,0);
  return {model_id:'C3',feature_kind:'convolution_spans',score_type:'softmax',prediction:model.labels[best],
    runner_up:model.labels[second],model_score:raw.softmax[best],score_gap:raw.logits[best]-raw.logits[second],
    raw_logits:raw.logits,softmax:raw.softmax,scores:model.labels.map((label,i)=>({label,value:raw.softmax[i],raw_logit:raw.logits[i]})),
    all_channels:all,top_convolution_spans:all.filter(c=>c.contribution>0).sort((a,b)=>b.contribution-a.contribution||a.channel-b.channel).slice(0,12),
    bias:raw.bias[best],channel_contribution_sum:sum,reconstructed_logit:raw.bias[best]+sum,raw_predicted_logit:raw.logits[best],
    explanation_note:'卷积最大响应片段；加法分解不是 attention、因果解释或已校准概率',
    run_id:model.run_id,token_count:encoded.diagnostic.original_length,sequence_diagnostic:encoded.diagnostic,
    notice:'Softmax 输出未校准；卷积片段不是 attention 或因果解释',execution:'browser-worker'};
}
function whitespace(char,ranges) {
  const code=char.codePointAt(0); return ranges.some(([a,b])=>a<=code&&code<=b);
}
function decode(text,tags) {
  const chars=Array.from(text), tokens=[]; let start=0;
  for(let i=0;i<chars.length;i++) if(['E','S'].includes(tags[i])) {tokens.push(chars.slice(start,i+1).join(''));start=i+1;}
  if(start!==chars.length) throw new Error('CRF 路径未构成完整 BMES 分词'); return tokens;
}
async function bilstmTrace(model,descriptor,text) {
  const chars=Array.from(text), vocabulary=new Map(model.vocabulary.map((char,i)=>[char,i]));
  const architecture={embedding_dim:model.config.embedding_dim,hidden_size_per_direction:model.config.hidden_size,
    num_layers:model.config.num_layers,dropout:model.config.dropout,output:'Linear(2*hidden, 4 BMES) -> CRF'};
  const chunks=[], tags=[], tokens=[];
  for(let at=0;at<chars.length;) {
    if(whitespace(chars[at],model.whitespace)) {
      chunks.push({offset:at,text:chars[at],tags:['S'],whitespace:true}); tags.push('S');tokens.push(chars[at]);at++;continue;
    }
    let end=at+1;while(end<chars.length&&!whitespace(chars[end],model.whitespace))end++;
    const chunk=chars.slice(at,end).join(''), raw=await run(descriptor,chars.slice(at,end).map(char=>vocabulary.get(char)??1));
    raw.tokens=decode(chunk,raw.tags); raw.model='bilstm-crf';raw.architecture=architecture;
    raw.trace_type='bilstm_crf_emission';raw.character_count=end-at;raw.bmes_source='model decoded path';
    chunks.push({offset:at,text:chunk,...raw});tags.push(...raw.tags);tokens.push(...raw.tokens);at=end;
  }
  const transitions=chunks.find(c=>!c.whitespace)?.crf_transitions;
  const trace=chunks.length===1&&!chunks[0].whitespace?chunks[0]:
    {model:'bilstm-crf',tokens,tags,chunks,architecture,crf_transitions:transitions};
  return {...trace,trace_type:'bilstm_crf_emission',text,character_count:chars.length,bmes_source:'model decoded path'};
}
export async function neural(path,params,body,manifest) {
  const cnn=manifest.neural?.textcnn, bi=manifest.neural?.bilstm;
  if(path==='/api/v1/models/hmm/parameters'&&params.get('model')==='bilstm-crf'&&bi) {
    const model=await load(bi.path);
    const architecture={embedding_dim:model.config.embedding_dim,hidden_size_per_direction:model.config.hidden_size,
      num_layers:model.config.num_layers,dropout:model.config.dropout,output:'Linear(2*hidden, 4 BMES) -> CRF'};
    const response=await fetch(new URL('crf-transitions.f32',new URL(bi.path,document.baseURI)));
    const matrix=new Float32Array(await response.arrayBuffer());
    return {run:model.run,parameters:{model:'bilstm-crf',architecture,config:model.config,
      vocabulary_size:model.vocabulary.length-2,training:model.training,
      parameter_count:Object.values(model.tensors).reduce((sum,t)=>sum+t.shape.reduce((a,b)=>a*b,1),0),
      crf_transition_matrix:Array.from({length:4},(_,i)=>Array.from(matrix.subarray(i*4,(i+1)*4)))},training_history:model.history};
  }
  if(path==='/api/v1/experiments/3/stage3/predict') {
    if(!cnn) return undefined;
    const requested=body.run_id||'latest';
    if(requested!=='latest'&&requested!==cnn.run_id) throw new Error('C3 需要当前冻结 Stage3 Run');
    if(!body.text?.trim()||Array.from(body.text).length>5000) throw new Error('请输入1–5000字新闻文本');
    const start=performance.now(), model=await load(cnn.path), rules=await load(manifest.classifiers.unicode_path);
    const tokens=await tokenizeWords(body.text,model.preprocessing,model.tokenizer_run_id,rules), encoded=encodeSequence(model,tokens);
    return {...cnnPrediction(model,await run(cnn,encoded.ids),encoded),runtime_seconds:(performance.now()-start)/1000};
  }
  if(path==='/api/v1/experiments/3/visualization/c3-trace') {
    if(!cnn) return undefined;
    const runId=params.get('run_id'), documentId=params.get('document_id');
    if(runId!=='latest'&&runId!==cnn.run_id) throw new Error('C3 trace 需要冻结 Stage3 Run');
    const kernel=Number(params.get('kernel')||3), filter=Number(params.get('filter_id')||0);
    if(![3,4,5].includes(kernel)||!Number.isInteger(filter)||filter<0||filter>127) throw new Error('kernel/filter 越界');
    const response=await fetch(`/api/v1/experiments/3/workbench/document?run_id=${encodeURIComponent(cnn.run_id)}&document_id=${encodeURIComponent(documentId)}`);
    if(!response.ok) throw new Error('冻结文档编码未导出');
    const detail=await response.json(), sequence=detail.representations.C3.sequence;
    const raw=await run(cnn,sequence.token_ids), saved=detail.records.C3, channel=(kernel-3)*128+filter;
    const error=Math.max(...raw.logits.map((value,i)=>Math.abs(value-saved.raw_logits[i])));
    const tolerance=2e-5, logitsMatch=raw.logits.every((v,i)=>Math.abs(v-saved.raw_logits[i])<=tolerance+tolerance*Math.abs(saved.raw_logits[i]));
    const model=await load(cnn.path), prediction=model.labels[raw.logits.indexOf(Math.max(...raw.logits))];
    const pooledMatch=raw.pooled.every((v,i)=>Math.abs(v-sequence.pooled[i])<=tolerance+tolerance*Math.abs(sequence.pooled[i]));
    const argmaxMatch=saved.top_convolution_spans.every(c=>raw.argmax[c.channel]===c.argmax_position);
    if(prediction!==saved.prediction||!pooledMatch||!argmaxMatch)throw new Error('CNN 池化/路径与保存证据不一致');
    if(!logitsMatch) throw new Error(`浏览器 CNN 与保存 logit 偏差 ${error}，不显示未经核验的过程`);
    return {...raw,activations:raw.activations[kernel][filter],frozen_logits:saved.raw_logits,
      label_order:(await load(cnn.path)).labels,prediction:saved.prediction,runner_up:saved.runner_up,
      verification:{logits_match:logitsMatch,prediction_match:prediction===saved.prediction,
        pooled_match:pooledMatch,argmax_match:argmaxMatch,argmax_check_scope:'saved_top_channels',
        max_logit_abs_error:error,atol:tolerance,rtol:tolerance},
      kernel_size:kernel,filter_id:filter,channel,document_id:documentId,retained_length:sequence.token_ids.length,
      max_len:(await load(cnn.path)).vocabulary.max_len,padded_shape:[1,(await load(cnn.path)).vocabulary.max_len],
      embedding_shape:[1,(await load(cnn.path)).vocabulary.max_len,128],view_run_id:runId,source_run_id:cnn.run_id,
      execution:'browser-worker',diagnostic_forward_only:true,formal_evaluation_executed:false,fit_executed:false};
  }
  const isAnalyze=path==='/api/v1/analyze'&&body?.segmentation_model==='bilstm-crf';
  const isTrace=path==='/api/v1/experiments/1/trace'&&params.get('model')==='bilstm-crf';
  const isDeep=path==='/api/v1/course/deep/bilstm';
  if(bi&&(isAnalyze||isTrace||isDeep)) {
    const text=isTrace?params.get('text'):body.text, max=isAnalyze?5000:30;
    if(!text?.trim()||Array.from(text).length>max) throw new Error(`请输入1–${max}字文本`);
    const start=performance.now(), model=await load(bi.path), trace=await bilstmTrace(model,bi,text);
    if(isTrace) return trace;
    if(isAnalyze) return {text,segmentation:{model:'bilstm-crf',tokens:trace.tokens,tags:trace.tags},
      ner:body.tasks?.includes('ner')?{status:'unavailable',reason:'静态版尚未移植 NER'}:null,
      classification:null,run:model.run,elapsed_ms:performance.now()-start,execution:'browser-worker'};
    return {text,trace,parameters:{model:'bilstm-crf',architecture:trace.architecture,training:model.training,
      vocabulary_size:model.vocabulary.length-2,
      parameter_count:Object.values(model.tensors).reduce((sum,t)=>sum+t.shape.reduce((a,b)=>a*b,1),0)},history:model.history,
      source:{model:'浏览器 BiLSTM-CRF',run_id:bi.run_id,training:false},elapsed_ms:performance.now()-start,
      environment:{runtime:'Web Worker / float32 tensors'},execution:'browser-worker'};
  }
  return undefined;
}
