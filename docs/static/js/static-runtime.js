/* Only loaded by the exported static edition. No network backend or service worker. */
(() => {
  'use strict';
  const nativeFetch = window.fetch.bind(window);
  const root = new URL(document.baseURI);
  const manifestPromise = nativeFetch(new URL('static-manifest.json', root)).then(r => {
    if (!r.ok) throw new Error('静态资源清单读取失败');
    return r.json();
  });
  const computePromise = import(new URL('static/js/static-compute.js', root));
  const cache = new Map();
  const sharedValues = new Map();
  async function expand(value) {
    if (value && typeof value === 'object' && value.__nlplab_static_ref__) {
      const file = value.__nlplab_static_ref__;
      if (!sharedValues.has(file)) sharedValues.set(file, nativeFetch(new URL(file, root)).then(r => {
        if (!r.ok) throw new Error('共享静态数据读取失败');
        return r.json();
      }));
      return structuredClone(await sharedValues.get(file));
    }
    if (Array.isArray(value)) return Promise.all(value.map(expand));
    if (value && typeof value === 'object') {
      const result = {};
      for (const [key, child] of Object.entries(value)) result[key] = await expand(child);
      return result;
    }
    return value;
  }
  const error = message => new Response(JSON.stringify({error:{code:'static_unavailable',message}}),
    {status:503,headers:{'Content-Type':'application/json; charset=utf-8'}});
  const canonical = (path, params) => {
    const entries = [...params].filter(([,v]) => v !== '').sort(([a],[b]) => a.localeCompare(b));
    const query = new URLSearchParams(entries).toString();
    return path + (query ? '?' + query : '');
  };
  function findEntry(manifest, path, params) {
    const entries = manifest.responses;
    const exact = entries[canonical(path, params)];
    if (exact) return exact;
    const defaults = manifest.defaults?.[path] ||
      (path.startsWith('/api/v1/comprehensive/collections/') && path.includes('/documents/')
        ? {analyze:false} : {});
    const reduced = new URLSearchParams(params);
    for (const [key, value] of Object.entries(defaults))
      if (reduced.get(key) === String(value)) reduced.delete(key);
    if (entries[canonical(path, reduced)]) return entries[canonical(path, reduced)];
    // Compare parameter sets with explicit defaults filled in, never ignore a run,
    // document, model or K that determines which frozen result is displayed.
    for (const [key, entry] of Object.entries(entries)) {
      const split = key.indexOf('?');
      if ((split < 0 ? key : key.slice(0,split)) !== path) continue;
      const saved = new URLSearchParams(split < 0 ? '' : key.slice(split+1));
      const names = new Set([...params.keys(), ...saved.keys()]);
      if ([...names].every(name => (params.get(name) ?? String(defaults[name] ?? '')) ===
                                   (saved.get(name) ?? String(defaults[name] ?? '')))) return entry;
    }
    return undefined;
  }
  async function snapshot(manifest, path, params) {
    const entry = findEntry(manifest, path, params);
    if (!entry) return undefined;
    const file = entry.file || entry.path;
    if (!cache.has(file)) cache.set(file, nativeFetch(new URL(file,root)).then(async r => {
      if (!r.ok) throw new Error(`静态文件读取失败：${file}`);
      return new Uint8Array(await r.arrayBuffer());
    }));
    let bytes = await cache.get(file);
    if ((entry.content_type || '').includes('json')) {
      const text = new TextDecoder().decode(bytes);
      if (text.includes('__nlplab_static_ref__')) bytes = JSON.stringify(await expand(JSON.parse(text)));
    }
    // A local runtime being present at export time does not make it available on Pages.
    if (path === '/api/v1/course/modules/resources' || path === '/api/v1/course/resources' ||
        path === '/api/v1/experiments/1/extensions' || path === '/api/v1/comprehensive/status') {
      const value = JSON.parse(new TextDecoder().decode(bytes));
      if (path.endsWith('/modules/resources')) for (const key of Object.keys(value)) {
        if (key.endsWith('_ready')) value[key] = false;
      }
      if (path.endsWith('/modules/resources')) {
        value.bilstm_ready=!!manifest.neural?.bilstm;
        value.bert_ready=!!manifest.transformers?.bert;
        value.dependency_ready=!!manifest.transformers?.dependency;
        value.runtime_ready=!!manifest.neural?.bilstm;
        value.device='Browser CPU';
      }
      if (path === '/api/v1/course/resources') Object.assign(value,{
        perceptron:!!manifest.sequence?.models?.perceptron,
        crf:!!manifest.sequence?.models?.crf,pos:!!manifest.transformers?.pos});
      if (path.endsWith('/extensions')) for (const item of value.items || []) {
        item.available = !!manifest.models?.models?.[item.model] ||
          !!manifest.sequence?.models?.[item.model] ||
          (item.model==='bilstm-crf'&&!!manifest.neural?.bilstm);
        item.unavailable_reason = item.available ? null : '静态版仅保留冻结记录，无此模型实时推理';
      }
      if (path.endsWith('/comprehensive/status')) {
        for (const resource of value.resources || []) resource.ready = resource.key === 'segmentation'||
          (resource.key==='classification'&&!!manifest.classifiers?.runs?.latest?.C2)||
          !!manifest.transformers?.[resource.key];
        value.notice = '静态版：已接入模型在浏览器按需加载；NER使用INT8计算，冻结实验记录保持原结果。';
      }
      bytes = JSON.stringify(value);
    }
    return new Response(bytes, {status:entry.status || 200,
      headers:{'Content-Type':entry.content_type || 'application/json; charset=utf-8'}});
  }
  async function dispatch(path, params, method, body, signal) {
    if (signal?.aborted) throw new DOMException('Aborted','AbortError');
    const manifest = await manifestPromise;
    if(method==='POST'&&path.startsWith('/api/v1/comprehensive/')) {
      const {comprehensive}=await import(new URL('static/js/static-comprehensive.js',root));
      const value=await comprehensive(path,body,manifest);
      if(value!==undefined)return new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json; charset=utf-8'}});
    }
    if (manifest.transformers && method==='POST') {
      const {transformers}=await import(new URL('static/js/static-transformers.js',root));
      const value=await transformers(path,body,manifest);
      if(value!==undefined)return new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json; charset=utf-8'}});
    }
    if (manifest.sequence) {
      const {sequence}=await import(new URL('static/js/static-sequence.js',root));
      const value=await sequence(path,params,body,manifest);
      if(value!==undefined)return new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json; charset=utf-8'}});
    }
    if (manifest.neural && (path==='/api/v1/experiments/3/stage3/predict' ||
        path==='/api/v1/experiments/3/visualization/c3-trace' || path==='/api/v1/course/deep/bilstm' ||
        (path==='/api/v1/models/hmm/parameters'&&params.get('model')==='bilstm-crf') ||
        (path==='/api/v1/analyze'&&body?.segmentation_model==='bilstm-crf') ||
        (path==='/api/v1/experiments/1/trace'&&params.get('model')==='bilstm-crf'))) {
      const {neural}=await import(new URL('static/js/static-neural.js',root));
      const value=await neural(path,params,body,manifest);
      if(value!==undefined)return new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json; charset=utf-8'}});
    }
    if (method === 'POST' || path.endsWith('/trace')) {
      if (method === 'POST' && path.includes('/experiments/3/')) {
        const {classify} = await import(new URL('static/js/static-classification.js',root));
        const value = await classify(path,body,manifest);
        if (value !== undefined) return new Response(JSON.stringify(value),
          {headers:{'Content-Type':'application/json; charset=utf-8'}});
      }
      const {compute} = await computePromise;
      const result = await compute(path, params, body);
      if (result !== undefined) return new Response(JSON.stringify(result),
        {headers:{'Content-Type':'application/json; charset=utf-8'}});
    }
    if (method === 'GET') {
      const query = new URLSearchParams(params);
      const paginated = path.endsWith('/errors');
      const filtered = path.endsWith('/predictions');
      const parameters = path === '/api/v1/models/hmm/parameters';
      if (paginated) ['page','page_size','category'].forEach(key => query.delete(key));
      if (filtered) query.delete('errors_only');
      if (/\/experiments\/3\/stage[23]\/(predictions|features)$/.test(path)) query.delete('model_id');
      if (parameters) {
        query.set('topk','100');
        if (query.get('model') && query.get('model') !== 'hmm1-smooth') query.delete('run_id');
      }
      let response = await snapshot(manifest,path,query);
      if (!response && path === '/api/v1/views/cws/errors' && query.has('run_id')) {
        const expected = query.get('run_id'); query.delete('run_id');
        const candidate = await snapshot(manifest,path,query);
        if (candidate?.ok && (await candidate.clone().json()).run?.run_id === expected) response = candidate;
      }
      if (response && response.ok && (paginated || filtered || parameters)) {
        const value = await response.json();
        if (parameters && value.parameters?.emission_topk) {
          const topk = Math.max(1, Math.min(100, Number(params.get('topk') || 20)));
          for (const [state, entries] of Object.entries(value.parameters.emission_topk))
            value.parameters.emission_topk[state] = entries.slice(0,topk);
        }
        let items = value.items || value.predictions;
        if (Array.isArray(items)) {
          if (params.get('category')) items = items.filter(x =>
            (x.manual_category || x.category) === params.get('category'));
          if (params.get('errors_only') === 'true') items = items.filter(x =>
            x.correct === false || (x.predicted_label != null && x.predicted_label !== x.label));
          if (paginated) {
            const page = Math.max(1, Number(params.get('page') || 1));
            const size = Math.max(1, Math.min(100, Number(params.get('page_size') || 10)));
            value.total = items.length; value.page = page; value.page_size = size;
            items = items.slice((page-1)*size,page*size);
          }
          if (value.items) value.items = items; else value.predictions = items;
        }
        response = new Response(JSON.stringify(value), {headers:{'Content-Type':'application/json'}});
      }
      if (response) return response;
    }
    return error('此功能尚未移植到浏览器，或没有导出当前所选数据组合。已接入的模型可在静态页面实时运行；共享标注和原 Python 训练流程仍需本地服务。');
  }
  window.fetch = async (input, options = {}) => {
    const url = new URL(input instanceof Request ? input.url : input, document.baseURI);
    let path = url.pathname;
    if (path.startsWith(root.pathname)) path = '/' + path.slice(root.pathname.length);
    if (url.origin !== root.origin || !path.startsWith('/api/')) return nativeFetch(input, options);
    const method = (options.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
    let body = options.body;
    if (body === undefined && input instanceof Request && method !== 'GET') body = await input.clone().text();
    if (typeof body === 'string') body = JSON.parse(body);
    try { return await dispatch(path, url.searchParams, method, body, options.signal); }
    catch (e) { if (e.name === 'AbortError') throw e; return error(e.message); }
  };
  function localUrl(path) {
    const url = new URL(path,location.origin);
    const route = ({'/sequence':'/segmentation','/analysis':'/integrated-analysis'})[url.pathname] || url.pathname;
    return new URL(route.replace(/^\//,'').replace(/\/?$/, '/') + url.search + url.hash,root).href;
  }
  window.NLPLAB_STATIC = {url:localUrl, manifest:manifestPromise};
  document.addEventListener('click', async event => {
    const anchor = event.target.closest('a[href]');
    if (!anchor) return;
    const raw = anchor.getAttribute('href');
    if (raw.startsWith('/api/')) {
      event.preventDefault();
      const response = await window.fetch(raw);
      if (!response.ok) { const data = await response.json(); alert(data.error.message); return; }
      const objectUrl = URL.createObjectURL(await response.blob());
      const download = document.createElement('a'); download.href = objectUrl;
      download.download = raw.split('?')[0].split('/').pop() + '.json'; download.click();
      setTimeout(() => URL.revokeObjectURL(objectUrl),1000);
    } else if (raw.startsWith('/') && !raw.startsWith('//')) {
      anchor.href = localUrl(raw);
    }
  }, true);
  document.addEventListener('DOMContentLoaded', () => {
    const unsupported = '此功能尚未移植到浏览器；冻结记录仍可浏览。';
    for (const selector of ['[data-deep-form]:not([data-deep-form="bilstm"]) button[type="submit"]','[data-boundary-compare]',
      '#ca-analyze','#cw-classification-index','#ca-cluster','#pos-analyze','#ie-analyze',
      '#cm-analyze','[data-course-run]']) {
      for (const button of document.querySelectorAll(selector)) {
        button.disabled = true; button.title = unsupported;
      }
    }
    for (const status of document.querySelectorAll('[data-deep-status]')) status.textContent = unsupported;
    for(const selector of ['[data-deep-status="llm"]','[data-deep-status="extensions"]']){
      const status=document.querySelector(selector);
      if(status)status.textContent='Qwen 生成尚未导出：需要量化与额外权重空间；当前静态版已接入 BERT、BiLSTM 与其他模型。';
    }
    manifestPromise.then(manifest=>{
      const pos=document.querySelector('#pos-analyze');
      if(pos&&manifest.transformers?.pos){pos.disabled=false;pos.title='首次运行按需加载约47 MB ELECTRA权重；浏览器CPU推理';}
      if(manifest.transformers?.dependency){const button=document.querySelector('#cm-analyze');if(button){button.disabled=false;button.title='首次运行按需加载约51 MB依存模型权重';}}
      if(manifest.transformers?.bert){
        const button=document.querySelector('[data-deep-form="transformer"] button[type="submit"]');
        if(button){button.disabled=false;button.title='首次运行需下载约390 MB原始BERT权重；建议桌面浏览器';}
        const status=document.querySelector('[data-deep-status="transformer"]');
        if(status)status.textContent='浏览器 BERT 已就绪；首次运行下载约390 MB权重，适合桌面浏览器。';
      }
      if(manifest.transformers?.pos&&manifest.transformers?.ner){const button=document.querySelector('#ca-analyze');if(button){button.disabled=false;button.title='浏览器综合分析：HMM、POS、INT8 NER、C2';}}
      if(manifest.transformers?.ner)for(const selector of ['#ie-analyze','#analyze-button']) {
        const button=document.querySelector(selector);if(button){button.disabled=false;button.title='首次运行按需加载约146 MB NER权重；INT8浏览器推理';}
      }
      const button=document.querySelector('[data-deep-form="bilstm"] button[type="submit"]');
      if(button)button.disabled=!manifest.neural?.bilstm;
      const status=document.querySelector('[data-deep-status="bilstm"]');
      if(status&&manifest.neural?.bilstm)status.textContent='浏览器 BiLSTM-CRF 已就绪；点击运行后加载冻结权重。';
    }).catch(()=>{});
  });
})();
