const $=id=>document.getElementById(id);
const names={C0:'C0 · MultinomialNB',C1:'C1 · word TF-IDF + LinearSVC',C2:'C2 · char χ² + LinearSVC',C3:'C3 · TextCNN'};
import {MODEL_COLORS as colors} from './classification-model-colors.js';
const labels={finance:'财经',education:'教育',science:'科技',sports:'体育',entertainment:'娱乐'};
const escape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=v=>`${(v*100).toFixed(2)}%`;
const pm=v=>`${pct(v.mean)} ± ${(v.std*100).toFixed(2)} pp`;
const count=v=>`${v.mean.toLocaleString('en-US',{maximumFractionDigits:1})} ± ${v.std.toFixed(1)}`;
function table(headers,rows){return `<table class="classification-table"><thead><tr>${headers.map(h=>`<th>${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${escape(v)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;}

function chart(series,width){
  const w=Math.max(310,Math.min(850,width||720)),h=330,left=46,right=w-16,top=20,bottom=282;
  const x=n=>left+(n-80)/340*(right-left),y=v=>bottom-Math.max(0,Math.min(1,v))*(bottom-top);
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="真实开发集学习曲线，误差条为三个子集ladder的population标准差">`;
  for(let i=0;i<=5;i++){const v=i/5;svg+=`<line x1="${left}" y1="${y(v)}" x2="${right}" y2="${y(v)}" stroke="#e3eaf0"/><text x="${left-6}" y="${y(v)+4}" text-anchor="end">${i*20}%</text>`;}
  for(const n of [80,160,240,320,420])svg+=`<text x="${x(n)}" y="${bottom+20}" text-anchor="middle">${n}</text>`;
  svg+=`<text x="${(left+right)/2}" y="${h-9}" text-anchor="middle">训练篇数（五类平衡）</text>`;
  for(const s of series){
    svg+=`<polyline fill="none" stroke="${s.color}" stroke-width="2.4" ${s.dashed?'stroke-dasharray="6 4"':''} points="${s.values.map(p=>`${x(p.train_size)},${y(p.metric.mean)}`).join(' ')}"/>`;
    for(const p of s.values){const px=x(p.train_size),lo=y(p.metric.mean-p.metric.std),hi=y(p.metric.mean+p.metric.std);svg+=`<g stroke="${s.color}" stroke-width="1.2"><line x1="${px}" y1="${hi}" x2="${px}" y2="${lo}"/><line x1="${px-4}" y1="${lo}" x2="${px+4}" y2="${lo}"/><line x1="${px-4}" y1="${hi}" x2="${px+4}" y2="${hi}"/></g><circle cx="${px}" cy="${y(p.metric.mean)}" r="3.2" fill="${s.color}"><title>${escape(s.name)} · n=${p.train_size} · ${pm(p.metric)}</title></circle>`;}
  }
  return svg+'</svg>';
}

function configureAnalysisOnly(){
  document.body.dataset.stage4Analysis='true';
  $('classification-subtitle').textContent='冻结 C0/C1/C2/C3 的开发集样本效率分析；本视图只读取 Train / Validation。';
  for(const button of document.querySelectorAll('[data-tab]'))if(button.dataset.tab!=='comparison')button.disabled=true;
  document.querySelector('[data-tab="comparison"]').click();
  $('classification-run').innerHTML=['20261001_123337_9c1e7341','20261001_133829_68ee50d7','20261001_153153_c99a8d6f'].map(id=>`<option value="${id}">${id}</option>`).join('');
  $('classification-run').value='20261001_153153_c99a8d6f';
  $('classification-model').value='C3';
  $('classification-footer').textContent='Stage4 独立 analysis · 历史分类 Run 与 classifier selector 语义保留';
  $('classification-stage-label').textContent='开发集分析模式 · Test访问禁止';
}

export async function initializeStage4(analysisOnly=false){
  if(analysisOnly)configureAnalysisOnly();
  const panel=$('classification-stage4-panel');
  try{
    const response=await fetch('/api/v1/experiments/3/stage4/overview');
    const data=await response.json();
    if(!response.ok)throw new Error(data.error?.message||'尚无已完成的开发集学习曲线');
    $('classification-stage4-status').textContent=`Analysis Run ${data.run.run_id} · 3 nested ladders × 5 sizes × 4 models · Test evaluation=0 · 420 sanity通过`;
    const seriesFor=model=>data.curves.filter(p=>p.model_id===model);
    let model='all';
    function renderCurve(){
      const models=model==='all'?Object.keys(names):[model];
      const series=models.flatMap(id=>{
        const rows=seriesFor(id);
        const validation={name:`${id} Validation Macro-F1`,color:colors[id],values:rows.map(p=>({train_size:p.train_size,metric:p.validation_macro_f1}))};
        return model==='all'?[validation]:[{name:`${id} Train Macro-F1`,color:'#84949f',dashed:true,values:rows.map(p=>({train_size:p.train_size,metric:p.train_macro_f1}))},validation];
      });
      $('classification-stage4-chart').innerHTML=chart(series,$('classification-stage4-chart').clientWidth);
      $('classification-stage4-legend').innerHTML=series.map(s=>`<span><i style="background:${s.color}"></i>${escape(s.name)}${s.dashed?'（虚线）':''}</span>`).join('');
    }
    function render(){
      model=$('classification-stage4-model').value;
      renderCurve();
      if(model==='all'){
        $('classification-stage4-values').innerHTML=table(['Train size',...Object.keys(names).map(id=>`${id} Validation Macro-F1 mean ± std`)],data.protocol.train_sizes.map(n=>[n,...Object.keys(names).map(id=>pm(seriesFor(id).find(p=>p.train_size===n).validation_macro_f1))]));
      }else{
        $('classification-stage4-values').innerHTML=table(['Train size','Train Macro-F1 mean ± std','Validation Macro-F1 mean ± std','Validation Accuracy mean ± std','Train−Validation gap'],seriesFor(model).map(p=>[p.train_size,pm(p.train_macro_f1),pm(p.validation_macro_f1),pm(p.validation_accuracy),`${(p.gap.mean*100).toFixed(2)} pp`]));
      }
      const selected=model==='all'?data.curves:data.curves.filter(p=>p.model_id===model);
      $('classification-stage4-replicates').innerHTML=table(['模型','Train size','subset seed','Train Macro-F1','Validation Macro-F1','Validation Accuracy'],selected.flatMap(p=>p.replicates.map(r=>[p.model_id,p.train_size,r.subset_seed,pct(r.train_macro_f1),pct(r.validation_macro_f1),pct(r.validation_accuracy)])));
      $('classification-stage4-per-class').innerHTML=table(['模型','Train size',...Object.values(labels).map(c=>`${c} F1 mean ± std`)],selected.map(p=>[p.model_id,p.train_size,...p.per_class_f1.map(pm)]));
    }
    $('classification-stage4-model').addEventListener('change',render);
    const c3=seriesFor('C3');
    $('classification-stage4-c3-diagnostics').innerHTML=table(['Train size','vocab（含 PAD/UNK）mean ± std','Val UNK rate mean ± std','参数数目 mean ± std','Val截断篇数','保留token比例'],c3.map(p=>{const r=p.representation;return[p.train_size,count(r.vocabulary_size),pm(r.validation_unk_rate),count(r.parameter_count),r.validation_truncated_count.mean,pct(r.validation_retained_token_ratio.mean)];}));
    $('classification-stage4-linear-diagnostics').innerHTML=table(['模型','Train size','dimension mean ± std','raw char dimension mean ± std','effective k','Val候选出现次数覆盖 mean ± std','Val零向量 mean'],data.curves.filter(p=>['C1','C2'].includes(p.model_id)).map(p=>{const r=p.representation;return[p.model_id,p.train_size,count(r.dimension),r.raw_dimension?count(r.raw_dimension):'—',r.effective_k?count(r.effective_k):'—',pm(r.validation_coverage),r.validation_zero_vectors.mean];}));
    $('classification-stage4-subsets').innerHTML=table(['subset seed','Train size','每类篇数','ordered IDs SHA256'],data.subsets.map(s=>[s.subset_seed,s.train_size,s.per_class,s.ids_sha256]));
    for(const link of panel.querySelectorAll('[data-stage4-export]'))link.href=`/api/v1/experiments/3/stage4/export/${link.dataset.stage4Export}?run_id=${encodeURIComponent(data.run.run_id)}`;
    let lastWidth=0;
    new ResizeObserver(()=>{const w=$('classification-stage4-chart').clientWidth;if(w>0&&w!==lastWidth){lastWidth=w;renderCurve();}}).observe($('classification-stage4-chart'));
    render();
    panel.dataset.loaded='true';
  }catch(error){$('classification-stage4-status').textContent=error.message;panel.dataset.loaded='false';}
}
