import {textcnn,bilstm} from './static-neural-kernels.js';
const models=new Map(), results=new Map();
async function load(root,path) {
  if(!models.has(path)) models.set(path,(async()=>{
    const url=new URL(path,root), response=await fetch(url);
    if(!response.ok) throw new Error('神经模型清单读取失败');
    const model=await response.json(), tensors={};
    await Promise.all(Object.entries(model.tensors).map(async([name,info])=>{
      const res=await fetch(new URL(info.file,url)); if(!res.ok) throw new Error(`神经权重读取失败：${name}`);
      const buffer=await res.arrayBuffer(); if(buffer.byteLength!==info.bytes) throw new Error('神经权重文件长度不符');
      if(self.crypto?.subtle) {
        const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),b=>b.toString(16).padStart(2,'0')).join('');
        if(digest!==info.sha256)throw new Error('神经权重 SHA256 校验失败');
      }
      tensors[name]=new Float32Array(buffer);
    }));
    return {model,tensors};
  })());
  return models.get(path);
}
self.onmessage=async({data})=>{
  try {
    const {model,tensors}=await load(data.root,data.path);
    const key=data.path+':'+data.ids.join(',');
    if(!results.has(key)) {
      results.set(key,model.kind==='textcnn'?textcnn(model,tensors,data.ids):bilstm(model,tensors,data.ids));
      while(results.size>3)results.delete(results.keys().next().value);
    }
    const result=results.get(key);
    self.postMessage({id:data.id,result});
  } catch(error) {self.postMessage({id:data.id,error:error.message});}
};
