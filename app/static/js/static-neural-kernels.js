/* Frozen neural inference: original float32 weights, eval mode, no training. */
const f32 = Math.fround;
const STATES = ['B','M','E','S'];
const rows = (values, width) => Array.from({length:values.length/width},(_,i) => Array.from(values.subarray(i*width,(i+1)*width)));
const dot = (weights, offset, vector) => { let value=0; for(let i=0;i<vector.length;i++) value+=weights[offset+i]*vector[i]; return value; };

export function textcnn(model, tensors, ids) {
  const embedded = ids.map(id => tensors['embedding.weight'].subarray(id*128,(id+1)*128));
  const pooled = [], argmax = [], activations = {};
  for(let layer=0;layer<3;layer++) {
    const kernel=layer+3, weights=tensors[`convs.${layer}.weight`], bias=tensors[`convs.${layer}.bias`];
    const windows=Math.max(0,ids.length-kernel+1), tracks=[];
    for(let filter=0;filter<128;filter++) {
      let maximum=-Infinity, position=-1; const track=[];
      for(let at=0;at<windows;at++) {
        let value=0;
        for(let dimension=0;dimension<128;dimension++) for(let step=0;step<kernel;step++)
          value+=weights[(filter*128+dimension)*kernel+step]*embedded[at+step][dimension];
        value=Math.max(0,f32(value+bias[filter])); track.push(value);
        if(value>maximum) { maximum=value; position=at; }
      }
      pooled.push(windows?maximum:0); argmax.push(position); tracks.push(track);
    }
    activations[kernel]=tracks;
  }
  const weights=tensors['classifier.weight'], bias=tensors['classifier.bias'];
  const logits=Array.from(bias,(b,i) => f32(dot(weights,i*384,pooled)+b));
  const top=Math.max(...logits), exp=logits.map(v=>Math.exp(v-top)), sum=exp.reduce((a,b)=>a+b,0);
  return {pooled,argmax,activations,logits,softmax:exp.map(v=>v/sum),
    weights:rows(weights,384),bias:Array.from(bias)};
}

function logsumexp(values) {
  const max=Math.max(...values); return max+Math.log(values.reduce((sum,v)=>sum+Math.exp(v-max),0));
}
export function crfTrace(emissions, transitions, start, end) {
  const delta=[start.map((v,j)=>v+emissions[0][j])], alpha=[delta[0].slice()], backpointers=[Array(4).fill(null)];
  for(let t=1;t<emissions.length;t++) {
    const next=[], forward=[], pointers=[];
    for(let target=0;target<4;target++) {
      const candidates=delta[t-1].map((v,source)=>v+transitions[source][target]);
      let best=0; for(let source=1;source<4;source++) if(candidates[source]>candidates[best]) best=source;
      next.push(candidates[best]+emissions[t][target]); pointers.push(best);
      forward.push(logsumexp(alpha[t-1].map((v,source)=>v+transitions[source][target]))+emissions[t][target]);
    }
    delta.push(next); alpha.push(forward); backpointers.push(pointers);
  }
  const last=delta.at(-1).map((v,j)=>v+end[j]); let best=0;
  for(let j=1;j<4;j++) if(last[j]>last[best]) best=j;
  const path=Array(emissions.length); path[path.length-1]=best;
  for(let t=path.length-1;t>0;t--) path[t-1]=backpointers[t][path[t]];
  return {delta,alpha,backpointers,path,path_score:last[best],
    log_partition:logsumexp(alpha.at(-1).map((v,j)=>v+end[j])),start,end};
}
function lstm(tensors, embedded, reverse) {
  const suffix=reverse?'_reverse':'', ih=tensors[`lstm.weight_ih_l0${suffix}`], hh=tensors[`lstm.weight_hh_l0${suffix}`];
  const bi=tensors[`lstm.bias_ih_l0${suffix}`], bh=tensors[`lstm.bias_hh_l0${suffix}`];
  let h=new Float32Array(128), c=new Float32Array(128); const result=Array(embedded.length);
  for(let t=0;t<embedded.length;t++) {
    const position=reverse?embedded.length-1-t:t, x=embedded[position], gates=new Float32Array(512);
    for(let j=0;j<512;j++) gates[j]=f32(f32(dot(ih,j*128,x)+bi[j])+f32(dot(hh,j*128,h)+bh[j]));
    const nextH=new Float32Array(128), nextC=new Float32Array(128);
    for(let j=0;j<128;j++) {
      const input=f32(1/(1+Math.exp(-gates[j]))), forget=f32(1/(1+Math.exp(-gates[128+j])));
      const candidate=f32(Math.tanh(gates[256+j])), output=f32(1/(1+Math.exp(-gates[384+j])));
      nextC[j]=f32(f32(forget*c[j])+f32(input*candidate)); nextH[j]=f32(output*f32(Math.tanh(nextC[j])));
    }
    h=nextH; c=nextC; result[position]=h;
  }
  return result;
}
export function bilstm(model, tensors, ids) {
  const embedded=ids.map(id=>tensors['embedding.weight'].subarray(id*128,(id+1)*128));
  const forward=lstm(tensors,embedded,false), backward=lstm(tensors,embedded,true);
  const emissions=forward.map((h,i)=> {
    const encoded=new Float32Array(256); encoded.set(h); encoded.set(backward[i],128);
    return Array.from(tensors['emission.bias'],(bias,j)=>f32(dot(tensors['emission.weight'],j*256,encoded)+bias));
  });
  const transitions=rows(tensors['crf.transitions'],4), start=Array.from(tensors['crf.start_transitions']);
  const numeric=crfTrace(emissions,transitions,start,Array.from(tensors['crf.end_transitions']));
  const end=Array.from(tensors['crf.end_transitions']);
  let score=start.map((v,j)=>f32(v+emissions[0][j])), alpha=score.slice(); const pointers=[];
  for(let t=1;t<emissions.length;t++) {
    const next=[], forward=[], previous=[];
    for(let target=0;target<4;target++) {
      const candidates=score.map((v,source)=>f32(f32(v+transitions[source][target])+emissions[t][target]));
      let best=0;for(let source=1;source<4;source++)if(candidates[source]>candidates[best])best=source;
      next.push(candidates[best]);previous.push(best);
      forward.push(f32(logsumexp(alpha.map((v,source)=>f32(f32(v+transitions[source][target])+emissions[t][target])))));
    }
    pointers.push(previous);score=next;alpha=forward;
  }
  const last=score.map((v,j)=>f32(v+end[j]));let best=0;
  for(let j=1;j<4;j++)if(last[j]>last[best])best=j;
  const path=Array(emissions.length);path[path.length-1]=best;
  for(let t=path.length-1;t>0;t--)path[t-1]=pointers[t-1][path[t]];
  numeric.verified_path=numeric.path.every((value,i)=>value===path[i]);
  numeric.partition_error=Math.abs(f32(logsumexp(alpha.map((v,j)=>f32(v+end[j]))))-numeric.log_partition);
  if(!numeric.verified_path)throw new Error('CRF float32解码与数值重建路径不一致');
  return {emissions,tags:path.map(i=>STATES[i]),crf_transitions:transitions,numeric_trace:numeric};
}
