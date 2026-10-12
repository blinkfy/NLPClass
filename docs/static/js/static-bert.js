import {wordpieceEncoding} from './static-wordpiece.js';

const nested=(values,shape)=>shape.length===1?values.slice(0,shape[0]):Array.from({length:shape[0]},(_,i)=>{const stride=shape.slice(1).reduce((a,b)=>a*b,1);return nested(values.slice(i*stride,(i+1)*stride),shape.slice(1));});
export async function bertTrace(text,descriptor,{load,infer}) {
  if(typeof text!=='string'||!text.trim()||Array.from(text).length>120)throw new Error('BERT过程探索限1–120字、最多64个WordPiece');
  const model=await load(descriptor),encoding=wordpieceEncoding(text,model.vocabulary,{lowercase:model.lowercase??false}),ids=[101,...encoding.ids,102],offsets=[[0,0],...encoding.offsets,[0,0]];
  if(ids.length>64)throw new Error('BERT输入超过64个WordPiece，请缩短文本，不自动截断');
  const raw=await infer(descriptor,ids,null),tensor=name=>nested(raw.tensors[name].values,raw.tensors[name].shape);
  const logits=tensor('logits'),attentions=tensor('attention'),scaled=tensor('scaled'),norms=tensor('norms'),previews=tensor('previews'),contextNorms=tensor('context_norms');
  const qkv=norms.map((norm,i)=>({q:norm[0],k:norm[1],v:norm[2],q_preview:previews[i][0],k_preview:previews[i][1],v_preview:previews[i][2],context_preview:previews[i][3],context_norm:contextNorms[i]}));
  const checks=attentions.map((layer,l)=>{
    let softmaxError=0,rowError=0,contextError=0;
    layer.forEach((head,h)=>head.forEach((row,t)=>{
      const scores=scaled[l][h][t],max=Math.max(...scores),exp=scores.map(s=>Math.exp(s-max)),sum=exp.reduce((a,b)=>a+b,0);
      softmaxError=Math.max(softmaxError,...row.map((v,i)=>Math.abs(v-exp[i]/sum)));rowError=Math.max(rowError,Math.abs(row.reduce((a,b)=>a+b,0)-1));
      for(let d=0;d<8;d++){const reconstructed=row.reduce((v,a,i)=>v+a*previews[l][2][h][i][d],0);contextError=Math.max(contextError,Math.abs(reconstructed-previews[l][3][h][t][d]));}
    }));
    if(softmaxError>1e-5||contextError>2e-5)throw new Error('浏览器BERT注意力过程未通过数值核对');
    return {layer:l,max_softmax_error:softmaxError,max_context_error:contextError,max_row_sum_error:rowError,context_check_dimensions:8};
  });
  const masks=[];
  ids.forEach((id,position)=>{if(id!==103)return;const scores=logits[position],max=Math.max(...scores),exp=scores.map(s=>Math.exp(s-max)),sum=exp.reduce((a,b)=>a+b,0),order=scores.map((_,i)=>i).sort((a,b)=>scores[b]-scores[a]||a-b).slice(0,8);masks.push({position,candidates:order.map(id=>({id,token:model.vocabulary[id],logit:scores[id],softmax:exp[id]/sum}))});});
  return {text,tokens:ids.map((id,i)=>({position:i,token:model.vocabulary[id],id,start:offsets[i][0],end:offsets[i][1],special:[0,100,101,102,103].includes(id)})),shape:{tokens:ids.length,hidden:768,layers:12,heads:12,head_dim:64},attentions,scaled_scores:scaled,qkv,checks,masks,
    source:{model:model.model,revision:model.revision,weights_sha256:model.source_sha256,format:'ONNX float32',task:'bidirectional masked language modeling',training:false,precision:model.precision,context_verification:'8 displayed dimensions per attention head'},execution:'browser'};
}
