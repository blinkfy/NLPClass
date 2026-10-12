// Coordinated views over saved numerical artifacts and verified single-document traces.
import {MODEL_COLORS} from './classification-model-colors.js';

const esc = x => String(x ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = x => x == null ? '未保存 / 不适用' : Number(x).toFixed(4);
const labels={finance:'财经',education:'教育',science:'科技',sports:'体育',entertainment:'娱乐'};
const lab=x=>labels[x]||x;
const signed=x=>`${Number(x)>0?'+':''}${num(x)}`;
const table=(rows)=>`<table class="iv-table">${rows.map(([k,v])=>`<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join('')}</table>`;

export function normalizedSource(text){
  let normalized='', offsets=[];
  const segments=typeof Intl.Segmenter==='function'?[...new Intl.Segmenter('zh',{granularity:'grapheme'}).segment(text)]:Array.from(text).map((segment,i,arr)=>({segment,index:arr.slice(0,i).join('').length}));
  for(const s of segments){const n=s.segment.normalize('NFKC').toLowerCase();normalized+=n;for(let i=0;i<n.length;i++)offsets.push({start:s.index,end:s.index+s.segment.length});}
  if(normalized!==text.normalize('NFKC').toLowerCase())return {normalized:text.normalize('NFKC').toLowerCase(),offsets:null};
  return {normalized,offsets};
}
export function charOccurrences(text){
  const mapped=normalizedSource(text),out=[];let previous=null,offset=0;
  for(const c of mapped.normalized){
    const code=c.codePointAt(0),content=/^[a-z\p{Nd}\p{Unified_Ideograph}]$/u.test(c)||(code>=0xf900&&code<=0xfaff)||(code>=0x2f800&&code<=0x2fa1f);
    if(content){if(previous)out.push({term:previous.c+c,start:mapped.offsets?.[previous.offset]?.start,end:mapped.offsets?.[offset+c.length-1]?.end,normalized_position:previous.offset});previous={c,offset};}else previous=null;
    offset+=c.length;
  }
  return out;
}
function featureRanges(text,term){
  const mapped=normalizedSource(text);if(!mapped.offsets)return [];
  const ranges=[];
  for(const part of term.split(' ').filter(Boolean)){let at=0,count=0;const word=part.normalize('NFKC').toLowerCase();while((at=mapped.normalized.indexOf(word,at))>=0&&count++<8){ranges.push({start:mapped.offsets[at].start,end:mapped.offsets[at+word.length-1].end});at+=word.length;}}
  return ranges;
}
function highlighted(text,ranges){
  let at=0,result='';for(const r of ranges.filter(r=>r&&Number.isInteger(r.start)&&Number.isInteger(r.end)).sort((a,b)=>a.start-b.start||a.end-b.end)){
    if(r.start<at||r.end>text.length||r.end<=r.start)continue;result+=esc(text.slice(at,r.start))+`<mark data-iv-source>${esc(text.slice(r.start,r.end))}</mark>`;at=r.end;
  }return result+esc(text.slice(at));
}
const field=(label,value)=>`<div class="iv-key"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;

export function createAlgorithmExplorer({store,request,ui}){
  let context=null,data=null,epoch=0,timer=null,traceEpoch=0;
  const cache=new Map();
  const state=()=>store.get().selectedEvidence;
  const nodes=()=>[['classification-representation-view','classification-representation-inspector','process'],['classification-prediction-view','classification-prediction-inspector','decision']].map(([a,b,mode])=>({main:document.getElementById(a),inspector:document.getElementById(b),mode}));
  const stop=()=>{if(timer){clearInterval(timer);timer=null;}document.querySelectorAll('[data-iv-play]').forEach(n=>n.textContent='播放扫描');};
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  motion.addEventListener('change',()=>{if(motion.matches)stop();document.querySelectorAll('[data-iv-play]').forEach(n=>n.disabled=motion.matches);});
  function localSelection(){
    const selected=state();if(selected&&(!selected.model||selected.model===context.modelId))return selected;
    if(context.modelId==='C3')return {kind:'channel',channel:data.channel,kernel:Math.floor(data.channel/128)+3,filter:data.channel%128,position:data.argmax[data.channel],decomposition:'predicted'};
    const first=data.features.find(f=>f.relative_contribution>0)||data.features[0];return {kind:'feature',term:first?.term||'',column:first?.column??null};
  }
  function choose(patch){const c=context;store.set({selectedEvidence:{...localSelection(),...patch,model:c.modelId,documentId:c.document.document_id,targetClass:data.prediction,runnerUpClass:data.runner_up}});} 
  async function cached(path){if(!cache.has(path)){const value=await request(path);cache.set(path,value);while(cache.size>32)cache.delete(cache.keys().next().value);}return cache.get(path);}
  async function fetchTrace(selection){
    const c=context,version=++traceEpoch;
    const path=`visualization/c3-trace?${new URLSearchParams({run_id:c.runId,document_id:c.document.document_id,kernel:selection.kernel,filter_id:selection.filter})}`;
    try{const result=await cached(path);if(version!==traceEpoch||c!==context)return;data=result;render();}
    catch(error){if(c===context&&version===traceEpoch)failure(error.message);}
  }
  function failure(message){stop();for(const n of nodes()){n.main.innerHTML=`<h2>算法过程浏览器</h2><p class="iv-unavailable">${esc(message)}</p>`;n.inspector.innerHTML='<h2>诊断未通过</h2><p>不展示未经验证的中间数值。正式结果保持不变。</p>';}}
  async function update(next){
    stop();const version=++epoch;context=next;data=null;++traceEpoch;
    if(!next||!['C1','C2','C3'].includes(next.modelId)||next.document?.split!=='Test'){for(const n of nodes())n.main.classList.remove('iv-explorer');return;}
    for(const n of nodes())n.main.innerHTML='<h2>算法过程浏览器</h2><p>正在读取冻结表示与数值分解…</p>';
    try{
      let selection=state();
      if(selection?.model&&selection.model!==next.modelId)selection=null;
      if(next.modelId==='C3'){
        const first=next.record.top_convolution_spans?.[0];
        if(!selection)selection={kind:'channel',kernel:first?.kernel_size||3,filter:first?.filter_id||0,channel:first?.channel||0,position:first?.argmax_position||0,decomposition:'predicted'};
        const path=`visualization/c3-trace?${new URLSearchParams({run_id:next.runId,document_id:next.document.document_id,kernel:selection.kernel,filter_id:selection.filter})}`;
        const result=await cached(path);if(version!==epoch||next!==context)return;data=result;
      }else{
        const result=await cached(`visualization/decision?${new URLSearchParams({run_id:next.runId,document_id:next.document.document_id,model_id:next.modelId})}`);
        if(version!==epoch||next!==context)return;data=result;
        if(!selection){const first=data.features.find(f=>f.relative_contribution>0)||data.features[0];selection={kind:'feature',term:first?.term||'',column:first?.column??null};}
      }
      if(!state()){choose(selection);}else render();
    }catch(error){if(version===epoch&&next===context)failure(error.message);}
  }
  function sourceMarkup(ranges){return `<div class="iv-source" data-iv-source-view>${highlighted(context.document.text,ranges)}</div>`;}
  function provenance(){return `<details class="wb-inspector-details"><summary>冻结来源与完整数值</summary>${table([['工作台 Run',context.runId],['模型来源 Run',data.source_run_id],['文档',context.document.document_id]])}<pre class="iv-json">${esc(JSON.stringify(data.verification||data.provenance,null,2))}</pre></details>`;}
  function outputSummary(mode){
    if(mode!=='decision')return '';
    const neural=context.modelId==='C3',scores=context.record.scores||[];
    return `<div class="iv-output-summary"><p>${neural?'TextCNN Softmax 输出（未校准）':'LinearSVC decision score · 非概率'}</p><div class="iv-score-strip">${scores.map((r,j)=>`<span class="${r.label===data.prediction?'is-predicted':''}"><small>${lab(r.label)}</small><strong>${neural?(Number(r.value)*100).toFixed(2)+'%':num(r.value)}</strong>${neural?`<em>logit ${num(context.record.raw_logits?.[j]??r.raw_logit)}</em>`:''}</span>`).join('')}</div></div>`;
  }
  function renderLinear(n,s){
    const c=context,features=data.features,selected=features.find(f=>f.term===s.term),meta=data.chi2_metadata?.[s.term],membership=c.representation.feature_membership?.find(f=>f.term===s.term);
    const shown=features.slice(0,24);if(selected&&!shown.includes(selected))shown.unshift(selected);
    const maxX=Math.max(...shown.map(f=>f.tfidf),.001),maxC=Math.max(...shown.map(f=>Math.abs(f.relative_contribution)),.001);
    const ranges=c.modelId==='C2'?(charOccurrences(c.document.text).filter(r=>r.term===s.term).slice(0,8)):featureRanges(c.document.text,s.term);
    const status=c.modelId==='C2'?(selected||membership?.retained?'χ² 保留':meta?'raw vocabulary 内 · 未选中':'fitted vocabulary 外'):'词级 '+(s.term.includes(' ')?'bigram':'unigram');
    let lane='';
    if(c.modelId==='C2'){
      const all=charOccurrences(c.document.text);let index=all.findIndex(r=>r.term===s.term);if(s.occurrence!=null&&all[s.occurrence]?.term===s.term)index=s.occurrence;index=Math.max(0,index);
      const start=Math.max(0,index-8),local=all.slice(start,start+22),selectedTerms=new Set(c.representation.selected_features?.map(r=>r.term)||[]);
      lane=`<div class="iv-lane-heading"><strong>局部字符 bigram · 三种选择状态</strong><small>实线：保留 · 虚线：未选中 · 淡色：词表外 · 横向滚动查看更多</small></div><div class="iv-char-strip">${local.map((r,i)=>`<button data-iv-term="${esc(r.term)}" data-iv-occurrence="${start+i}" class="${selectedTerms.has(r.term)?'retained':data.chi2_metadata[r.term]?'discarded':'unseen'} ${r.term===s.term?'is-selected':''}"><small>p${r.normalized_position}</small><strong>${esc(r.term)}</strong></button>`).join('')}</div><p class="iv-status" data-iv-chi2>${esc(s.term)} · ${status}${meta?` · χ² ${num(meta.chi2_score)} · 排名 ${meta.display_rank} · DF ${meta.df}`:''}</p>`;
    }else{
      const sequence=c.representation.sequence,tokens=sequence?.tokens||[],parts=s.term.split(' '),at=tokens.findIndex((t,i)=>parts.every((p,j)=>tokens[i+j]===p&&(!j||sequence.original_positions[i+j]===sequence.original_positions[i+j-1]+1)));
      lane=`<div class="iv-lane-heading"><strong>词项 / bigram 联动</strong><small>${at>=0?'保存的 HMM token 预览来自 C3 保留编码；并非 C1 全文分词序列。':'完整 HMM token 序列未保存；下方仅为真实词表 feature 的词项。'}</small></div><div class="iv-token-lane">${(at>=0?tokens.slice(Math.max(0,at-3),at+parts.length+3):parts).map(t=>`<button data-iv-term="${esc(s.term)}" class="${parts.includes(t)?'is-selected':''}">${esc(t)}</button>`).join('')}</div>`;
    }
    const sum=features.reduce((a,f)=>a+f.relative_contribution,0);
    n.main.innerHTML=`<div class="iv-header"><span style="color:${MODEL_COLORS[c.modelId]}">${c.modelId} · ${c.modelId==='C2'?'χ² 字符表示':'LinearSVC 决策'}浏览器</span><h2>${lab(data.prediction)} 相对 ${lab(data.runner_up)} 的 margin</h2><p>同一篇冻结新闻 · TF-IDF 大不等于判别贡献大。</p></div>${outputSummary(n.mode)}${sourceMarkup(ranges)}${lane}<div class="iv-decision-head"><strong>Feature</strong><strong>TF-IDF magnitude</strong><strong>对 margin 的正 / 负贡献</strong></div><div class="iv-feature-grid">${shown.map(f=>`<button class="iv-feature-row ${s.term===f.term?'is-selected':''}" data-iv-term="${esc(f.term)}" data-iv-column="${f.column}"><span>${esc(f.term)}</span><span class="iv-magnitude"><i style="width:${f.tfidf/maxX*100}%"></i><small>${num(f.tfidf)}</small></span><span class="iv-contribution"><i style="left:${f.relative_contribution>=0?50:50+f.relative_contribution/maxC*50}%;width:${Math.abs(f.relative_contribution)/maxC*50}%" class="${f.relative_contribution<0?'negative':''}"></i><small>${signed(f.relative_contribution)}</small></span></button>`).join('')}</div><div class="iv-margin" data-iv-margin>截距差 ${signed(data.intercept_difference)} + 全部 ${features.length} 个非零特征贡献 ${signed(sum)} = <strong>margin ${num(data.margin)}</strong><small>当前词项：${esc(s.term)} · ${selected?signed(selected.relative_contribution):'未进入最终向量，不参与决策分解'}</small></div><details><summary>完整非零特征与保存分数</summary><div class="iv-table-scroll">${table(features.map(f=>[`${f.column} · ${f.term}`,`x=${num(f.tfidf)} · direct=${num(f.direct_contribution)} · relative=${num(f.relative_contribution)}`]))}</div>${table([['预测类分数',num(data.predicted_score)],['第二名分数',num(data.runner_up_score)],['χ² 显示排名规则',data.rank_definition],['TF', '原始 word TF 未保存，不从归一化 TF-IDF 反推']])}</details>`;
    n.inspector.innerHTML=`<div class="wb-inspector-heading"><span class="wb-inspector-kicker">${c.modelId} · 当前 feature</span><h3 class="wb-object-title">${esc(s.term)}</h3></div><div class="wb-inspector-metrics">${field('TF-IDF',selected?num(selected.tfidf):'未进入最终向量')}${field('相对 margin 贡献',selected?signed(selected.relative_contribution):'不参与')}</div><p class="iv-status">${status}</p>${table([['vocab index',selected?.column??meta?.raw_column??'词表外'],['IDF',num(selected?.idf)],['预测类 weight',num(selected?.predicted_weight)],['第二名 weight',num(selected?.runner_up_weight)],['direct contribution',num(selected?.direct_contribution)],...(meta?[['χ² score',num(meta.chi2_score)],['χ² display rank',meta.display_rank],['Train+Val DF',meta.df],['当前文档 Count',membership?.count??'未保存']]:[])])}<p class="wb-inspector-note">${c.modelId==='C1'&&s.term.includes(' ')?'bigram 的两词分别高亮；不把分离出现伪装成连续短语。':''}TF 不从 L2 归一化数值反推；χ² relevance 与 SVC contribution 是不同量。</p>${provenance()}`;
    n.main.onclick=e=>{const b=e.target.closest('[data-iv-term]');if(!b)return;const f=features.find(f=>f.term===b.dataset.ivTerm);choose({kind:'feature',term:b.dataset.ivTerm,column:f?.column??null,occurrence:b.dataset.ivOccurrence==null?null:Number(b.dataset.ivOccurrence)});};
    const active=n.main.querySelector('.iv-feature-row.is-selected');if(active){const box=n.main.querySelector('.iv-feature-grid');box.scrollTop=active.offsetTop-box.offsetTop-box.clientHeight/3;}
  }
  function decomposition(s){
    const pred=data.label_order.indexOf(data.prediction),runner=data.label_order.indexOf(data.runner_up),target=s.decomposition==='runner'?runner:pred;
    const weights=data.weights[target].map((w,j)=>s.decomposition==='difference'?w-data.weights[runner][j]:w);
    const bias=s.decomposition==='difference'?data.bias[pred]-data.bias[runner]:data.bias[target];
    const expected=s.decomposition==='difference'?data.logits[pred]-data.logits[runner]:data.logits[target];
    const contributions=data.pooled.map((v,j)=>v*weights[j]);
    return {weights,bias,expected,contributions,sum:contributions.reduce((a,v)=>a+v,0)};
  }
  function renderCNN(n,s){
    const c=context,sequence=c.representation.sequence,position=Math.max(0,Math.min(s.position,data.activations.length-1)),kernel=s.kernel,channel=s.channel;
    const tokenStart=Math.max(0,position-3),tokenEnd=Math.min(sequence.tokens.length,position+kernel+3),maxPosition=data.argmax[channel],isMax=data.activations[position]===data.pooled[channel],maxLabel=isMax?(position===maxPosition?'global max · 保存 argmax':'同值 global max · 非保存 argmax'):'非 global max';
    const originals=sequence.original_positions.slice(position,position+kernel),crosses=originals.some((v,i)=>i&&v!==originals[i-1]+1);
    const ranges=(c.tokenOffsets||[]).slice(position,position+kernel).filter(Boolean),calc=decomposition(s),activation=data.activations[position]??0;
    const high=data.pooled.map((v,j)=>({channel:j,value:v,contribution:calc.contributions[j]})).sort((a,b)=>b.contribution-a.contribution||a.channel-b.channel).slice(0,8);
    if(!high.some(x=>x.channel===channel))high.push({channel,value:data.pooled[channel],contribution:calc.contributions[channel]});
    const maxContribution=Math.max(...high.map(x=>Math.abs(x.contribution)),.001),width=Math.max(640,data.activations.length*2),height=92,pad=19,maxActivation=Math.max(...data.activations,.001),x=p=>pad+(width-pad*2)*p/Math.max(1,data.activations.length-1),y=v=>height-pad-(height-pad*2)*v/maxActivation;
    const points=data.activations.map((v,i)=>`${x(i)},${y(v)}`).join(' ');
    n.main.innerHTML=`<div class="iv-header"><span style="color:${MODEL_COLORS.C3}">C3 · 冻结 TextCNN 过程浏览器</span><h2>卷积扫描、最大池化与判别</h2><p>浏览器 trace 核对保存 logits / pooled / prediction 与已保存通道 argmax。</p></div>${outputSummary(n.mode)}<div class="iv-cnn-controls"><label>kernel<select data-iv-kernel>${[3,4,5].map(k=>`<option ${k===kernel?'selected':''}>${k}</option>`).join('')}</select></label><label>filter<input data-iv-filter type="number" min="0" max="127" value="${s.filter}"></label><label>判别目标<select data-iv-mode><option value="predicted" ${s.decomposition==='predicted'?'selected':''}>预测类 · ${lab(data.prediction)}</option><option value="runner" ${s.decomposition==='runner'?'selected':''}>第二名 · ${lab(data.runner_up)}</option><option value="difference" ${s.decomposition==='difference'?'selected':''}>预测类 − 第二名</option></select></label></div>${sourceMarkup(ranges)}<div class="iv-lane-heading"><strong>卷积窗口 [${position}, ${position+kernel})</strong><small>原词与实际模型 token 分列；ID → trainable Embedding 128D。shape ${data.embedding_shape.join(' × ')}。</small></div><div class="iv-token-lane iv-cnn-token-lane">${sequence.tokens.slice(tokenStart,tokenEnd).map((token,i)=>{const j=tokenStart+i;return `<button data-iv-position="${Math.min(j,data.activations.length-1)}" class="${j>=position&&j<position+kernel?'in-window':''}"><small>p${j} · 原序号 ${sequence.original_positions[j]}<br>ID ${sequence.token_ids[j]}</small><strong>${esc(token)}</strong><span class="${sequence.model_tokens[j]==='<UNK>'?'is-unk':''}">${esc(sequence.model_tokens[j])}</span></button>`;}).join('')}</div>${crosses?'<p class="iv-boundary-note">该窗口跨 head-tail 拼接。高亮的是原文中分离的片段，不是连续短语。</p>':''}<div class="iv-lane-heading"><strong>真实 ReLU activation · ${data.activations.length} 个合法位置</strong><small>青色线：当前位置 · 紫色点：global max；PAD 窗口不参与。</small></div><div class="iv-track-scroll"><svg data-iv-track width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="当前filter所有合法位置的真实activation"><line x1="${pad}" x2="${width-pad}" y1="${height-pad}" y2="${height-pad}" stroke="#cad7db"/><polyline points="${points}" fill="none" stroke="${MODEL_COLORS.C3}" stroke-width="1.5"/><line x1="${x(position)}" x2="${x(position)}" y1="10" y2="${height-pad}" stroke="#07878b" stroke-width="2"/><circle cx="${x(maxPosition)}" cy="${y(data.pooled[channel])}" r="4" fill="${MODEL_COLORS.C3}"/><circle cx="${x(position)}" cy="${y(activation)}" r="4" fill="white" stroke="#07878b" stroke-width="2"/><text x="${pad}" y="${height-4}">0</text><text x="${width-pad-20}" y="${height-4}">${data.activations.length-1}</text></svg></div><div class="iv-scan"><button data-iv-step="-1" ${position===0?'disabled':''}>上一个位置</button><input type="range" data-iv-slider min="0" max="${data.activations.length-1}" value="${position}" aria-label="卷积起始位置（释放滑块后定位）"><button data-iv-step="1" ${position===data.activations.length-1?'disabled':''}>下一个位置</button><button data-iv-play ${matchMedia('(prefers-reduced-motion: reduce)').matches?'disabled':''}>${timer?'暂停扫描':'播放扫描'}</button></div><p class="iv-position-summary" data-iv-position-summary>position ${position} · activation ${num(activation)} · ${maxLabel}（argmax ${maxPosition}）${!isMax?' · 此处响应不替换池化值':''}</p><div class="iv-lane-heading"><strong>Global Max Pool · channel → contribution</strong><small>点击通道定位该 filter 的 argmax。始终使用池化最大响应做分类分解。</small></div><div class="iv-pool-bars">${high.map(p=>`<button data-iv-channel="${p.channel}" class="${p.channel===channel?'is-selected':''}"><span>k${Math.floor(p.channel/128)+3} / f${p.channel%128}</span><i><b style="width:${Math.abs(p.contribution)/maxContribution*100}%" class="${p.contribution<0?'negative':''}"></b></i><strong>${signed(p.contribution)}</strong></button>`).join('')}</div><div class="iv-margin" data-iv-decomposition>bias ${signed(calc.bias)} + 全部 384 channel contributions ${signed(calc.sum)} = <strong>${s.decomposition==='difference'?'logit margin':'raw logit'} ${num(calc.expected)}</strong><small>当前 channel ${channel}：pooled ${num(data.pooled[channel])} × weight ${num(calc.weights[channel])} = ${signed(calc.contributions[channel])}</small></div><details><summary>完整 384 通道 / 冻结输出</summary><div class="iv-table-scroll">${table(data.pooled.map((v,j)=>[`${j} · k${Math.floor(j/128)+3}/f${j%128}`,`max=${num(v)} · argmax=${data.argmax[j]} · weight=${num(calc.weights[j])} · contribution=${num(calc.contributions[j])}`]))}</div>${table(data.label_order.map((l,j)=>[lab(l),`frozen logit ${num(data.frozen_logits[j])} · trace logit ${num(data.logits[j])} · Softmax ${(c.record.softmax[j]*100).toFixed(2)}%`]))}</details>`;
    n.inspector.innerHTML=`<div class="wb-inspector-heading"><span class="wb-inspector-kicker">C3 · 卷积窗口 / 池化通道</span><h3 class="wb-object-title">k${kernel} / filter ${s.filter} · position ${position}</h3></div><div class="wb-inspector-metrics">${field('当前位置 activation',num(activation))}${field('pooled contribution',signed(calc.contributions[channel]))}</div><p class="iv-status">${maxLabel} · argmax ${maxPosition}</p><div class="wb-token-comparison"><span>原词片段</span><p>${esc(sequence.tokens.slice(position,position+kernel).join(' '))}</p><span>模型实际 token</span><p>${sequence.model_tokens.slice(position,position+kernel).map(t=>`<span class="${t==='<UNK>'?'wb-unk-token':''}">${esc(t)}</span>`).join(' ')}</p></div>${table([['channel',channel],['conv output shape',`1 × 128 × ${data.max_len-kernel+1} (合法 ${data.activations.length})`],['global max activation',num(data.pooled[channel])],['classifier weight',num(calc.weights[channel])],['当前响应 × weight（未池化）',num(activation*calc.weights[channel])],['池化后 contribution',num(calc.contributions[channel])],['重建 raw logit / margin',num(calc.bias+calc.sum)]])}<p class="wb-inspector-note">${!isMax?'当前位置响应不进入分类器；分类器使用此通道的 global max。':''}卷积最大响应片段不是 attention 或因果解释。Embedding 仅显示实际 shape，不绘制未保存的向量值。</p>${provenance()}`;
    const pool=n.main.querySelector('.iv-pool-bars'),selectedPool=pool.querySelector('.is-selected');if(selectedPool)pool.scrollTop=selectedPool.offsetTop-pool.offsetTop-pool.clientHeight/3;
    const lane=n.main.querySelector('.iv-cnn-token-lane'),firstWindow=lane.querySelector('.in-window');if(firstWindow)lane.scrollLeft=Math.max(0,firstWindow.offsetLeft-lane.offsetLeft-25);
    const track=n.main.querySelector('.iv-track-scroll');track.scrollLeft=Math.max(0,x(position)-track.clientWidth/2);
    n.main.onclick=e=>{
      const b=e.target.closest('[data-iv-channel],[data-iv-position],[data-iv-step],[data-iv-play]');
      if(b?.hasAttribute('data-iv-channel')){stop();selectChannel(Number(b.dataset.ivChannel));}
      else if(b?.hasAttribute('data-iv-position')){stop();choose({position:Number(b.dataset.ivPosition)});}
      else if(b?.hasAttribute('data-iv-step')){stop();choose({position:Math.max(0,Math.min(data.activations.length-1,position+Number(b.dataset.ivStep)))});}
      else if(b?.hasAttribute('data-iv-play')){
        if(timer){stop();return;}if(matchMedia('(prefers-reduced-motion: reduce)').matches)return;
        timer=setInterval(()=>{const current=localSelection();if(current.position>=data.activations.length-1){stop();return;}choose({position:current.position+1});},350);document.querySelectorAll('[data-iv-play]').forEach(b=>b.textContent='暂停扫描');
      }else if(e.target.closest('[data-iv-track]')){stop();const svg=n.main.querySelector('[data-iv-track]'),rect=svg.getBoundingClientRect(),px=(e.clientX-rect.left)*width/rect.width;choose({position:Math.max(0,Math.min(data.activations.length-1,Math.round((px-pad)/(width-pad*2)*(data.activations.length-1))))});}
    };
    n.main.onchange=e=>{stop();if(e.target.matches('[data-iv-kernel]'))selectChannel((Number(e.target.value)-3)*128+s.filter);if(e.target.matches('[data-iv-filter]')){const f=Number(e.target.value);if(Number.isInteger(f)&&f>=0&&f<128)selectChannel((kernel-3)*128+f);}if(e.target.matches('[data-iv-mode]'))choose({decomposition:e.target.value});if(e.target.matches('[data-iv-slider]'))choose({position:Number(e.target.value)});};
    // Commit on change so linked-view rendering cannot detach a dragged slider.
    n.main.querySelector('[data-iv-slider]').oninput=stop;
  }
  function selectChannel(channel){choose({kind:'channel',channel,kernel:Math.floor(channel/128)+3,filter:channel%128,position:data.argmax[channel]});}
  function render(){
    if(!context||!data)return;const s=localSelection();
    if(context.modelId==='C3'&&!data.activations.length){failure('当前编码没有此 kernel 的合法窗口。');return;}
    if(context.modelId==='C3'&&data.channel!==s.channel){for(const n of nodes()){n.main.style.minHeight=`${n.main.getBoundingClientRect().height}px`;n.main.innerHTML='<h2>读取该 filter 的真实 activation…</h2>';}fetchTrace(s);return;}
    for(const n of nodes()){
      n.main.style.minHeight='';n.main.classList.add('iv-explorer');n.main.style.setProperty('--iv-model-color',MODEL_COLORS[context.modelId]);
      if(context.modelId==='C3')renderCNN(n,s);else renderLinear(n,s);
      const source=n.main.querySelector('.iv-source'),mark=source.querySelector('mark');if(mark)source.scrollTop+=mark.getBoundingClientRect().top-source.getBoundingClientRect().top-source.clientHeight/3;
    }
  }
  store.subscribe((next,old)=>{
    if(['currentRun','currentModel','currentDocument','documentSource'].some(k=>next[k]!==old[k])){stop();context=null;data=null;++epoch;++traceEpoch;return;}
    if(next.selectedEvidence!==old.selectedEvidence&&context&&data)render();
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  document.querySelectorAll('[data-tab]').forEach(tab=>tab.addEventListener('click',()=>{if(!['data','prediction'].includes(tab.dataset.tab))stop();else setTimeout(render,0);}));
  return {update,stop};
}
