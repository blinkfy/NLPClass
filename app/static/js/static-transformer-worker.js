import * as ort from '../vendor/onnxruntime/ort.wasm.min.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('../vendor/onnxruntime/', import.meta.url).href;
const sessions = new Map();

async function session(root, path) {
  const url = new URL(path, root);
  if (!sessions.has(url.href)) sessions.set(url.href, (async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Transformer 模型清单读取失败');
    const model = await response.json();
    const graphResponse = await fetch(new URL(model.graph, url));
    if (!graphResponse.ok) throw new Error('Transformer 权重下载失败，请重试');
    const graph = await graphResponse.arrayBuffer();
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',graph)), v=>v.toString(16).padStart(2,'0')).join('');
    if (graph.byteLength !== model.bytes || hash !== model.sha256) throw new Error('Transformer 权重校验失败');
    const externalData=[];
    for(const file of model.external||[]) {
      const response=await fetch(new URL(file.file,url));
      if(!response.ok)throw new Error('Transformer 权重分片读取失败');
      const data=new Uint8Array(await response.arrayBuffer());
      const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),v=>v.toString(16).padStart(2,'0')).join('');
      if(data.byteLength!==file.bytes||hash!==file.sha256)throw new Error('Transformer 权重分片校验失败');
      externalData.push({path:file.file,data});
    }
    return {model, runtime:await ort.InferenceSession.create(graph,{executionProviders:['wasm'],externalData})};
  })().catch(error=>{sessions.delete(url.href);throw error;}));
  return sessions.get(url.href);
}

self.onmessage = async ({data}) => {
  try {
    const {runtime} = await session(data.root,data.path);
    const feeds = {
      input_ids:new ort.Tensor('int64',BigInt64Array.from(data.ids,BigInt),[1,data.ids.length])
    };
    if(data.spans)feeds.token_span=new ort.Tensor('int64',BigInt64Array.from(data.spans.flat(),BigInt),[data.spans.length,data.spans[0].length]);
    const result = await runtime.run(feeds);
    const tensors=Object.fromEntries(Object.entries(result).map(([name,tensor])=>[name,{values:Array.from(tensor.data),shape:tensor.dims}]));
    self.postMessage({id:data.id,tensors,logits:tensors.logits?.values,shape:tensors.logits?.shape});
    for (const tensor of Object.values(result)) tensor.dispose();
    for (const tensor of Object.values(feeds)) tensor.dispose();
  } catch (error) { self.postMessage({id:data.id,error:error.message}); }
};
