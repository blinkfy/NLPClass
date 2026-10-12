import {MODEL_COLORS as COLORS, MODEL_LIGHT_COLORS, MODEL_NAMES, modelSource} from './classification-model-colors.js';

const MODEL_IDS = ['C0', 'C1', 'C2', 'C3'];
const CONDITIONS = ['OO', 'OS', 'OT', 'OA', 'AA'];
const CONDITION_NAMES = {OO:'原始 → 原始', OS:'原始 → 去来源', OT:'原始 → 去模板', OA:'原始 → 全清理', AA:'全清理 → 全清理'};
const MODES = {formal:'正式结果', learning:'样本效率', ablation:'捷径消融', uncertainty:'统计可靠性'};
const CLASS_NAMES = {finance:'财经', education:'教育', science:'科技', sports:'体育', entertainment:'娱乐'};
const numeric = value => value === null || value === undefined || value === '' ? null : Number.isFinite(Number(value)) ? Number(value) : null;
const percent = value => numeric(value) === null ? '—' : `${(Number(value)*100).toFixed(2)}%`;
const points = value => numeric(value) === null ? '—' : `${(Number(value)*100).toFixed(2)} pp`;
const signedPoints = value => numeric(value) === null ? '—' : `${Number(value)>0?'+':''}${points(value)}`;
const count = value => numeric(value) === null ? '—' : Number(value).toLocaleString('zh-CN',{maximumFractionDigits:1});
const pvalue = value => numeric(value) === null ? '—' : Number(value)<.0001 ? '<0.0001' : Number(value).toFixed(4);
const modelTitle = id => `${id} · ${MODEL_NAMES[id] || id}`;
const classTitle = id => CLASS_NAMES[id] || id || '—';

export function initializeComparison({element, store, request, ui = {}, overview = null}) {
  if (!element) throw new Error('缺少模型比较容器');
  const escape = ui.escape || (value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char])));
  const cache = {}, charts = [];
  let formalOverview = overview;
  let loadEpoch = 0;
  let learningModels = new Set(MODEL_IDS);
  let selectedPoint = null;
  let selectedPair = null;
  let selectedCondition = 'OA';

  const mode = () => MODES[store.get().currentAnalysisMode] ? store.get().currentAnalysisMode : 'formal';
  const model = () => MODEL_IDS.includes(store.get().currentModel) ? store.get().currentModel : 'C0';
  const slot = name => element.querySelector(`[data-comparison-${name}]`);
  const runLabel = data => data?.run?.run_id ? `Run ${escape(data.run.run_id)}` : '冻结实验记录';
  const panel = (title, eyebrow, body, className = '') => `<section class="ccw-card ${className}"><div class="ccw-card-head"><div><span class="ccw-eyebrow">${escape(eyebrow)}</span><h3>${escape(title)}</h3></div></div>${body}</section>`;
  const metric = (label, value, detail = '') => `<div class="ccw-metric"><span>${escape(label)}</span><strong>${escape(value)}</strong>${detail?`<small>${escape(detail)}</small>`:''}</div>`;
  const row = (label, value) => `<div class="ccw-inspector-row"><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`;
  const empty = text => `<div class="ccw-empty">${escape(text)}</div>`;
  const meanStd = value => value && numeric(value.mean) !== null ? `${percent(value.mean)} ± ${points(value.std)}` : '—';
  const numberStd = value => value && numeric(value.mean) !== null ? `${count(value.mean)} ± ${count(value.std)}` : '—';

  function disposeCharts() {
    for (const chart of charts.splice(0)) chart.dispose();
  }
  function draw(selector, options, onClick) {
    const node = element.querySelector(selector);
    if (!node) return;
    const disclosure = node.closest('details');
    if (disclosure && !disclosure.open) {
      disclosure.addEventListener('toggle', () => {
        if (disclosure.open && node.isConnected) draw(selector, options, onClick);
      }, {once:true});
      return;
    }
    if (!window.echarts) { node.innerHTML = empty('图表组件尚未载入。'); return; }
    if (selector === '[data-chart="forest"]') options.series.push({type:'line',data:[],silent:true,markLine:{silent:true,symbol:'none',lineStyle:{color:'#173b49',width:2,type:'dashed'},label:{show:true,formatter:'零差异',position:'insideEndTop',color:'#173b49'},data:[{xAxis:0}]}});
    const chart = window.echarts.init(node);
    chart.setOption(options);
    if (onClick) chart.on('click', onClick);
    charts.push(chart);
  }
  function baseChart(extra = {}) {
    return {animation:false, textStyle:{fontFamily:'inherit',color:'#284553'}, grid:{left:56,right:24,top:44,bottom:56,containLabel:true}, tooltip:{trigger:'axis',backgroundColor:'#fff',borderColor:'#d8e8e8',textStyle:{color:'#183e49'}}, ...extra};
  }
  function shell(active) {
    element.innerHTML = `<div class="ccw-shell"><div class="ccw-heading"><div><span class="ccw-eyebrow">MODEL COMPARISON · FROZEN RECORDS</span><h2>模型比较工作台</h2><p>按同一实验记录查看模型表现、样本效率、清理敏感性与统计区间。</p></div><span class="ccw-readonly">只读分析</span></div><div class="ccw-modes" role="group" aria-label="比较视图">${Object.entries(MODES).map(([key,name])=>`<button type="button" data-mode="${key}" class="${key===active?'is-active':''}" aria-pressed="${key===active}">${name}</button>`).join('')}</div><div data-comparison-body aria-live="polite"></div></div>`;
    element.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
      if (button.dataset.mode !== mode()) store.set({currentAnalysisMode:button.dataset.mode});
    }));
  }
  function setBody(html) { slot('body').innerHTML = html; compactLayout(mode()); }
  function compactLayout(active) {
    const body = slot('body');
    const cards = [...body.querySelectorAll('.ccw-card')];
    if (cards.length < 2) return;
    const grid = document.createElement('div');
    grid.className = 'ccw-workspace';
    const anchor = active === 'learning' ? body.querySelector('.ccw-toggles') : body.querySelector('.ccw-summary');
    anchor.after(grid);
    const detail = (target, title, card) => {
      const disclosure = document.createElement('details');
      disclosure.className = 'ccw-disclosure';
      const summary = document.createElement('summary');
      summary.textContent = title;
      disclosure.append(summary, card);
      target.append(disclosure);
    };
    const inspectorHeading = (card, title, description) => {
      const heading = card.querySelector('.ccw-card-head');
      if (!heading) return;
      heading.className = 'wb-inspector-heading';
      heading.innerHTML = `<span class="ccw-eyebrow">当前对象 · CURRENT OBJECT</span><h3 class="wb-object-title">${escape(title)}</h3><p>${escape(description)}</p>`;
    };
    const explanation = (card, content) => {
      const note = document.createElement('p');
      note.className = 'wb-explanation';
      note.textContent = content;
      card.append(note);
    };
    if (active === 'formal' && cards.length >= 3) {
      body.querySelector('.ccw-kpis')?.remove();
      grid.append(cards[0], cards[2]);
      const id = MODEL_IDS.includes(model()) && formalOverview?.test?.[model()] ? model() : availableFormal()[0];
      const data = formalOverview?.test?.[id];
      inspectorHeading(cards[2], modelTitle(id), `正式 Test · ${count(data?.sample_count)} 篇`);
      const summary = document.createElement('div');
      summary.className = 'wb-key-metrics';
      summary.innerHTML = `${metric('Macro-F1',percent(data?.macro_f1))}${metric('Accuracy',percent(data?.accuracy))}`;
      cards[2].querySelector('.ccw-models')?.after(summary);
      const metrics = cards[2].querySelector('.ccw-inspector');
      if (metrics) detail(cards[2], '完整总体指标', metrics);
      const classes = data?.per_class || [];
      const labels = data?.label_order || classes.map(item=>item.label);
      const matrix = data?.confusion_matrix || [];
      const maximum = Math.max(1,...matrix.flat());
      const snapshot = document.createElement('div');
      snapshot.className = 'wb-formal-snapshot';
      snapshot.innerHTML = `<div class="wb-class-f1"><h4>分类别 F1</h4>${classes.map(item=>`<div class="wb-class-row"><span>${escape(classTitle(item.label))}</span><i><b style="width:${Math.max(0,Math.min(100,Number(item.f1)*100))}%;background:${COLORS[id]}"></b></i><strong>${percent(item.f1)}</strong></div>`).join('')}</div><div class="wb-confusion"><h4>混淆矩阵 <small>行真 · 列预测</small></h4><div class="wb-matrix" role="table" aria-label="${escape(id)} 混淆矩阵"><span></span>${labels.map(label=>`<span title="${escape(classTitle(label))}">${escape(classTitle(label).slice(0,1))}</span>`).join('')}${matrix.map((values,index)=>`<span title="${escape(classTitle(labels[index]))}">${escape(classTitle(labels[index]).slice(0,1))}</span>${values.map(value=>`<strong style="background:rgba(20,140,138,${(0.07+Math.max(0,Number(value))/maximum*.48).toFixed(2)})">${escape(value)}</strong>`).join('')}`).join('')}</div></div>`;
      cards[2].querySelector('.ccw-disclosure')?.before(snapshot);
      explanation(cards[2], '每类 F1 和混淆计数来自所选模型的同一正式 Test 记录。');
      const origin=modelSource(formalOverview,id),viewRun=store.get().currentRun || formalOverview?.view_run_id;
      if(origin.runId && origin.runId!==viewRun){
        const provenance=document.createElement('div');
        provenance.innerHTML=row('当前工作台 Run',viewRun)+row('模型来源 Run',origin.runId)+row('模型来源 Stage',origin.stage);
        detail(cards[2], 'Run 来源 · '+id, provenance);
      }
    } else if (active === 'learning' && cards.length >= 3) {
      grid.append(cards[0], cards[1]);
      const point = selectedPoint;
      inspectorHeading(cards[1],point?`${point.model_id} · ${point.train_size} 篇`: '选择曲线点','Train / Validation · 三个嵌套子集的均值与 population std');
      const diagnostics = document.createElement('div');
      diagnostics.className = 'wb-diagnostics';
      const wanted = ['特征维度','词表大小','Validation 覆盖','Validation UNK rate','参数量'];
      const source = [...cards[2].querySelectorAll('.ccw-inspector-row')];
      diagnostics.innerHTML = '<h4>表示诊断 · 当前点</h4>';
      wanted.forEach(label => { const item=source.find(node=>node.firstElementChild?.textContent===label); if(item) diagnostics.append(item.cloneNode(true)); });
      if (diagnostics.children.length>1) cards[1].append(diagnostics);
      explanation(cards[1], '泛化差为 Train − Validation；误差范围是三个子集结果的 population std。');
      detail(cards[1], '表示诊断 · 维度 / 覆盖 / UNK / 参数', cards[2]);
    } else if (active === 'ablation' && cards.length >= 4) {
      grid.append(cards[0], cards[1]);
      inspectorHeading(cards[1],`${model()} · ${selectedCondition}`,`${CONDITION_NAMES[selectedCondition]} · 开发集条件`);
      const conditionMetrics = cards[2].querySelector('.ccw-kpis-inner');
      cards[1].querySelector('.ccw-models')?.after(conditionMetrics);
      const inspection = cards[1].querySelector('.ccw-inspector');
      const allConditions = document.createElement('div');
      allConditions.className = 'ccw-inspector';
      [...inspection?.children || []].slice(0,5).forEach(node=>allConditions.append(node));
      explanation(cards[1], '条件差值相对 OO；AA 是使用全清理文本重新训练后的结果。');
      detail(cards[1], '五个条件的完整结果', allConditions);
      detail(cards[1], '当前条件说明', cards[2]);
      detail(cards[1], '各类别来源与模板痕迹', cards[3]);
    } else if (active === 'uncertainty' && cards.length >= 3) {
      grid.append(cards[0], cards[1]);
      inspectorHeading(cards[1],selectedPair?.replace(':',' → ') || '配对比较','Δ = B − A · 冻结 Test 配对统计');
      explanation(cards[1], '配对计数、McNemar raw p 与 Holm 调整 p 均取自同一冻结统计记录。');
      detail(cards[1], '单模型 Accuracy / Macro-F1 区间', cards[2]);
    } else return;
    body.querySelectorAll(':scope > .ccw-two').forEach(wrapper => wrapper.remove());
  }
  function modelButtons(ids, selected) {
    return `<div class="ccw-models" role="group" aria-label="选择模型">${ids.map(id=>`<button type="button" data-model="${id}" class="${id===selected?'is-active':''}" aria-pressed="${id===selected}"><i style="background:${COLORS[id]}"></i>${escape(id)}</button>`).join('')}</div>`;
  }
  function bindModels(ids) {
    element.querySelectorAll('[data-model]').forEach(button => button.addEventListener('click', () => {
      if (ids.includes(button.dataset.model)) store.set({currentModel:button.dataset.model});
    }));
  }
  function availableFormal() {
    const available = formalOverview?.available_models || Object.keys(formalOverview?.test || {});
    return MODEL_IDS.filter(id => available.includes(id) && formalOverview?.test?.[id]);
  }

  function renderFormal() {
    const ids = availableFormal();
    if (!ids.length) { setBody(empty('当前历史 Run 尚无正式 Test 模型指标。')); return; }
    const selected = ids.includes(model()) ? model() : ids[0];
    const selectedData = formalOverview.test[selected];
    setBody(`<div class="ccw-summary"><div><span class="ccw-eyebrow">01 · FROZEN TEST</span><h3>正式评价</h3><p>当前历史 Run 的可用模型 · 同一 Test 集；点击模型查看分类别结果。</p></div><span>当前工作台 Run ${escape(store.get().currentRun || formalOverview.view_run_id)}</span></div><div class="ccw-kpis">${metric('可用模型',ids.length,'历史 Run 实际记录')}${metric('Test 样本',count(selectedData.sample_count),'所选模型')}${metric(`${selected} Macro-F1`,percent(selectedData.macro_f1),'正式 Test')}${metric(`${selected} Accuracy`,percent(selectedData.accuracy),'正式 Test')}</div>${panel('整体指标','C0 → C3 · ACTUAL TEST',`<div class="ccw-chart" data-chart="formal-main" role="img" aria-label="各模型正式 Test 的 Macro-F1 与 Accuracy"></div>`)}<div class="ccw-two">${panel('分类别 F1',selected,`<div class="ccw-chart ccw-chart-small" data-chart="formal-class" role="img" aria-label="${selected} 分类别 F1"></div>`)}${panel('模型检视',selected,`${modelButtons(ids,selected)}<div class="ccw-inspector">${row('Macro Precision',percent(selectedData.macro_precision))}${row('Macro Recall',percent(selectedData.macro_recall))}${row('Macro-F1',percent(selectedData.macro_f1))}${row('Accuracy',percent(selectedData.accuracy))}${row('样本数',count(selectedData.sample_count))}</div>`)}</div>`);
    bindModels(ids);
    const main = baseChart({legend:{top:0,formatter:name=>name==='Macro-F1'?'实色 Macro-F1':'浅色 Accuracy'},grid:{left:38,right:15,top:38,bottom:38,containLabel:true},xAxis:{type:'value',min:0,max:100,axisLabel:{formatter:'{value}%'}},yAxis:{type:'category',inverse:true,data:ids,axisTick:{show:false}},series:[{name:'Macro-F1',type:'bar',barMaxWidth:20,itemStyle:{color:'#536777'},label:{show:true,position:'insideRight',color:'#fff',fontSize:11,formatter:p=>`${Number(p.value).toFixed(1)}%`},data:ids.map(id=>({value:formalOverview.test[id].macro_f1*100,itemStyle:{color:COLORS[id]}}))},{name:'Accuracy',type:'bar',barMaxWidth:20,itemStyle:{color:'#cbd5db'},label:{show:true,position:'insideRight',color:'#173b49',fontSize:11,formatter:p=>`${Number(p.value).toFixed(1)}%`},data:ids.map(id=>({value:formalOverview.test[id].accuracy*100,itemStyle:{color:MODEL_LIGHT_COLORS[id]}}))}]});
    draw('[data-chart="formal-main"]',main, event => { if (ids[event.dataIndex]) store.set({currentModel:ids[event.dataIndex]}); });
  }

  function chartErrorBar(api, color) {
    const low = api.coord([api.value(0),api.value(1)-api.value(2)]);
    const high = api.coord([api.value(0),api.value(1)+api.value(2)]);
    return {type:'group',children:[{type:'line',shape:{x1:low[0],y1:low[1],x2:high[0],y2:high[1]},style:{stroke:color,lineWidth:1}},{type:'line',shape:{x1:low[0]-3,y1:low[1],x2:low[0]+3,y2:low[1]},style:{stroke:color}},{type:'line',shape:{x1:high[0]-3,y1:high[1],x2:high[0]+3,y2:high[1]},style:{stroke:color}}]};
  }
  function learningRows(data, id) { return (data.curves || []).filter(item=>item.model_id===id).sort((a,b)=>a.train_size-b.train_size); }
  function selectedLearningPoint(data, ids) {
    const rows = data.curves || [];
    const chosenModel=ids.includes(model())?model():selectedPoint?.model_id;
    let point = rows.find(item=>item.model_id===chosenModel && item.train_size===selectedPoint?.train_size);
    if (!point || !ids.includes(point.model_id)) point = learningRows(data,ids.includes(model())?model():ids[0]).at(-1);
    selectedPoint = point ? {model_id:point.model_id,train_size:point.train_size} : null;
    return point;
  }
  function renderLearning(data) {
    const ids = MODEL_IDS.filter(id=>learningRows(data,id).length);
    if (!ids.length) { setBody(empty('此 Stage4 Run 没有学习曲线记录。')); return; }
    const single = learningModels.size===1 ? [...learningModels][0] : null;
    const visible = ids.filter(id=>learningModels.has(id));
    const point = selectedLearningPoint(data,visible.length?visible:ids);
    const rep = point?.representation || {};
    const representation = [
      ['特征维度',numberStd(rep.dimension)],['词表大小',numberStd(rep.vocabulary_size)],['原始字符维度',numberStd(rep.raw_dimension)],['有效 χ² 维度',numberStd(rep.effective_k)],
      ['Validation 覆盖',meanStd(rep.validation_coverage)],['Validation UNK rate',meanStd(rep.validation_unk_rate)],['Validation 保留 token',meanStd(rep.validation_retained_token_ratio)],['Validation 截断篇数',numberStd(rep.validation_truncated_count)],['参数量',numberStd(rep.parameter_count)]
    ].filter(([,value])=>value!=='—');
    setBody(`<div class="ccw-summary"><div><span class="ccw-eyebrow">02 · DEVELOPMENT ONLY</span><h3>样本效率与泛化</h3><p>每点为三个嵌套子集的均值；误差线为 population std。点击曲线点查看表示诊断。</p></div><span>${runLabel(data)}</span></div><div class="ccw-models ccw-toggles" role="group" aria-label="显示模型">${ids.map(id=>`<button type="button" data-learning-model="${id}" aria-pressed="${learningModels.has(id)}" class="${learningModels.has(id)?'is-active':''}"><i style="background:${COLORS[id]}"></i>${id}</button>`).join('')}</div>${panel('Validation Macro-F1','TRAIN SIZE · MEAN ± POPULATION STD',`<div class="ccw-chart ccw-chart-tall" data-chart="learning" role="img" aria-label="Stage4 开发集学习曲线与标准差"></div>`)}<div class="ccw-two">${panel('选择的子集点',point?`${point.model_id} · ${point.train_size} 篇`:'无数据',point?`<div class="ccw-kpis ccw-kpis-inner">${metric('Train Macro-F1',meanStd(point.train_macro_f1))}${metric('Validation Macro-F1',meanStd(point.validation_macro_f1))}${metric('泛化差',point.gap?`${signedPoints(point.gap.mean)} ± ${points(point.gap.std)}`:'—','Train − Validation')}${metric('Validation Accuracy',meanStd(point.validation_accuracy))}</div>`:empty('选择曲线上的点。'))}${panel('表示诊断',point?modelTitle(point.model_id):'无数据',representation.length?`<div class="ccw-inspector">${representation.map(([label,value])=>row(label,value)).join('')}</div>`:empty('此模型和子集点无表示诊断。'))}</div><p class="ccw-note">分析仅使用 Train / Validation；Stage4 不读取正式 Test。</p>`);
    element.querySelectorAll('[data-learning-model]').forEach(button=>button.addEventListener('click',()=>{
      const id=button.dataset.learningModel;
      if (learningModels.has(id) && learningModels.size>1) learningModels.delete(id);
      else learningModels.add(id);
      render();
    }));
    const series=[];
    for (const id of visible) {
      const rows=learningRows(data,id);
      series.push({name:`${id} Validation`,type:'line',symbolSize:7,showSymbol:true,lineStyle:{width:2.2},itemStyle:{color:COLORS[id]},data:rows.map(p=>[p.train_size,p.validation_macro_f1.mean*100]),modelId:id});
      series.push({name:`${id} std`,type:'custom',silent:true,renderItem:(params,api)=>chartErrorBar(api,`${COLORS[id]}88`),data:rows.map(p=>[p.train_size,p.validation_macro_f1.mean*100,p.validation_macro_f1.std*100]),z:1});
      if (single===id) series.push({name:`${id} Train`,type:'line',symbolSize:7,lineStyle:{type:'dashed',width:2},itemStyle:{color:'#829aa6'},data:rows.map(p=>[p.train_size,p.train_macro_f1.mean*100]),modelId:id});
    }
    draw('[data-chart="learning"]',baseChart({legend:{top:0,data:series.filter(s=>s.type==='line').map(s=>s.name)},grid:{left:36,right:20,top:48,bottom:48,containLabel:true},xAxis:{type:'value',name:'Train 篇数',nameLocation:'middle',nameGap:30,min:0,max:Math.max(...data.curves.map(p=>p.train_size))+25},yAxis:{type:'value',name:'Macro-F1 %',min:0,max:100},tooltip:{trigger:'item',formatter:p=>p.seriesType==='line'?`${escape(p.seriesName)} · n=${p.value[0]}: ${Number(p.value[1]).toFixed(2)}%`:''},series}),event=>{
      const id=series[event.seriesIndex]?.modelId;
      if (id && event.value) {
        selectedPoint={model_id:id,train_size:event.value[0]};
        if(store.get().currentModel===id)render();else store.set({currentModel:id});
      }
    });
  }

  function renderAblation(data) {
    const results=data.results || [];
    const ids=MODEL_IDS.filter(id=>results.some(item=>item.model_id===id));
    if (!ids.length) { setBody(empty('此 Stage5 Run 没有消融结果。')); return; }
    const selected=ids.includes(model())?model():ids[0];
    const byCondition=new Map(results.filter(item=>item.model_id===selected).map(item=>[item.condition,item]));
    const baseline=byCondition.get('OO');
    const cleanup=byCondition.get(selectedCondition);
    setBody(`<div class="ccw-summary"><div><span class="ccw-eyebrow">03 · DEVELOPMENT ONLY</span><h3>来源与模板消融</h3><p>OO、OS、OT、OA、AA 为冻结开发集条件；Δ 相对该模型 OO。点击柱条查看具体条件。</p></div><span>${runLabel(data)}</span></div>${panel('各条件 Macro-F1','MODEL × CONDITION',`<div class="ccw-chart ccw-chart-tall" data-chart="ablation" role="img" aria-label="四个模型在五个消融条件下的开发集 Macro-F1"></div>`)}<div class="ccw-two">${panel('模型敏感性',modelTitle(selected),`${modelButtons(ids,selected)}<div class="ccw-inspector">${CONDITIONS.map(code=>{const item=byCondition.get(code);return row(`${code} · ${CONDITION_NAMES[code]}`,item?`${percent(item.metrics?.macro_f1)} / ${signedPoints(item.delta_macro_f1)}`:'—')}).join('')}${row('OA 清理敏感性',signedPoints(byCondition.get('OA')?.delta_macro_f1))}${row('AA 再训练相对 OO',signedPoints(byCondition.get('AA')?.delta_macro_f1))}${row('AA 相对 OA 恢复',numeric(byCondition.get('AA')?.metrics?.macro_f1)!==null&&numeric(byCondition.get('OA')?.metrics?.macro_f1)!==null?signedPoints(byCondition.get('AA').metrics.macro_f1-byCondition.get('OA').metrics.macro_f1):'—')}</div>`)}${panel('条件检视',cleanup?`${selected} · ${selectedCondition}`:'选择条件',cleanup?`<div class="ccw-kpis ccw-kpis-inner">${metric('Macro-F1',percent(cleanup.metrics?.macro_f1))}${metric('相对 OO',signedPoints(cleanup.delta_macro_f1))}${metric('Accuracy',percent(cleanup.metrics?.accuracy))}${metric('Validation 篇数',count(cleanup.metrics?.sample_count))}</div><p class="ccw-note">OO 基线 Macro-F1：${percent(baseline?.metrics?.macro_f1)}。AA 使用清理后的训练与验证文本。</p>`:empty('此条件没有记录。'))}</div>${panel('类别中的来源与模板痕迹','TRAIN / VALIDATION · PREVALENCE',`<div class="ccw-heat-wrap">${prevalenceHeat(data.prevalence || {})}</div>`)}<p class="ccw-note">Stage5 仅评估开发集清理条件；不读取正式 Test。</p>`);
    bindModels(ids);
    element.querySelectorAll('[data-condition]').forEach(button=>button.addEventListener('click',()=>{selectedCondition=button.dataset.condition;render();}));
    const lookup=new Map(results.map(item=>[`${item.model_id}:${item.condition}`,item]));
    const series=ids.map(id=>({name:id,type:'bar',barMaxWidth:20,itemStyle:{color:COLORS[id]},data:CONDITIONS.map(code=>{const value=lookup.get(`${id}:${code}`)?.metrics?.macro_f1;return numeric(value)===null?null:value*100;})}));
    draw('[data-chart="ablation"]',baseChart({legend:{top:0},grid:{left:42,right:18,top:45,bottom:65,containLabel:true},xAxis:{type:'category',data:CONDITIONS,axisLabel:{interval:0,formatter:value=>`${value}\n${{OO:'原始',OS:'去来源',OT:'去模板',OA:'全清理',AA:'再训练'}[value]}`} },yAxis:{type:'value',min:0,max:100,name:'Macro-F1 %'},series}),event=>{
      if (CONDITIONS[event.dataIndex]) {
        selectedCondition=CONDITIONS[event.dataIndex];
        if (store.get().currentModel===event.seriesName) render();
        else store.set({currentModel:event.seriesName});
      }
    });
  }
  function prevalenceHeat(prevalence) {
    const splits=['train','validation'];
    const cells=splits.flatMap(split=>(prevalence[split]||[]).map(item=>({split,...item})));
    if (!cells.length) return empty('没有类别痕迹占比记录。');
    return `<div class="ccw-prevalence" role="table" aria-label="各类别来源与模板占比"><div class="ccw-prevalence-head" role="row"><span>Split / 类别</span><span>来源痕迹</span><span>模板痕迹</span><span>移除字符</span></div>${cells.map(item=>`<div class="ccw-prevalence-row" role="row"><strong>${item.split==='train'?'Train':'Validation'} · ${escape(classTitle(item.label))}</strong>${[['source_rate','source_documents'],['template_rate','template_documents'],['removed_character_ratio','removed_characters']].map(([rate,num])=>`<span class="ccw-heat-cell" style="--heat:${Math.max(.06,Math.min(.85,Number(item[rate])||0))}" title="${count(item[num])} / ${count(item.count)}">${percent(item[rate])}</span>`).join('')}</div>`).join('')}</div>`;
  }

  function primaryPairs(data) {
    const all=data.statistics?.pairs || [];
    const requested=data.protocol?.primary_pairs || data.protocol?.primary_comparisons || [];
    if (!Array.isArray(requested) || !requested.length) return all.slice(0,3);
    const key=(a,b)=>[a,b].sort().join(':');
    const lookup=new Map(all.map(item=>[key(item.a,item.b),item]));
    const chosen=requested.map(item=>{
      if (Array.isArray(item)) return lookup.get(key(item[0],item[1]));
      if (item && typeof item==='object') return lookup.get(key(item.a,item.b));
      if (typeof item==='string') { const ids=item.match(/C[0-3]/g); return ids?.length>=2 ? lookup.get(key(ids[0],ids[1])) : undefined; }
      return undefined;
    }).filter(Boolean);
    return chosen.length?chosen.slice(0,3):all.slice(0,3);
  }
  function renderUncertainty(data) {
    const pairs=primaryPairs(data);
    const models=data.statistics?.models || [];
    const ids=MODEL_IDS.filter(id=>models.some(item=>item.model_id===id));
    if (!pairs.length && !models.length) { setBody(empty('此 Stage6 Run 没有不确定性统计记录。')); return; }
    const preferred=store.get().currentComparison;
    const pair=pairs.find(item=>`${item.a}:${item.b}`===preferred) || pairs[0];
    if (pair) selectedPair=`${pair.a}:${pair.b}`;
    setBody(`<div class="ccw-summary"><div><span class="ccw-eyebrow">04 · FROZEN TEST STATISTICS</span><h3>配对差异与置信区间</h3><p>Δ = B − A；横线是配对 bootstrap 95% 区间，竖线是零差异。点击区间查看计数与 p 值。</p></div><span>${runLabel(data)}</span></div>${panel('主要配对 · Macro-F1 差异','POINT ESTIMATE · PAIRED 95% CI',`<div class="ccw-chart ccw-chart-forest" data-chart="forest" role="img" aria-label="三个主要模型配对的 Macro-F1 差值与百分之九十五区间"></div>`)}<div class="ccw-two">${panel('配对检视',pair?`${pair.b} − ${pair.a}`:'无配对',pair?`<div class="ccw-inspector">${row('Δ Macro-F1',signedPoints(pair.macro_f1_difference?.point))}${row('配对 95% CI',`${points(pair.macro_f1_difference?.low)} ～ ${points(pair.macro_f1_difference?.high)}`)}${row('Δ Accuracy',signedPoints(pair.accuracy_difference))}${row('两者都正确',count(pair.both_correct))}${row('仅 A 正确',count(pair.a_only_correct))}${row('仅 B 正确',count(pair.b_only_correct))}${row('两者都错',count(pair.both_wrong))}${row('McNemar exact raw p',pvalue(pair.raw_p))}${row('Holm 调整 p',pvalue(pair.holm_p))}</div>`:empty('没有配对记录。'))}${panel('单模型区间','WILSON / STRATIFIED BOOTSTRAP',`<div class="ccw-ci-list">${ids.map(id=>{const item=models.find(row=>row.model_id===id);return `<div class="ccw-ci-row"><strong style="color:${COLORS[id]}">${id}</strong><span>Accuracy ${percent(item.accuracy?.point)}<small>Wilson 95% · ${percent(item.accuracy?.low)} – ${percent(item.accuracy?.high)}</small></span><span>Macro-F1 ${percent(item.macro_f1?.point)}<small>分层 bootstrap 95% · ${percent(item.macro_f1?.low)} – ${percent(item.macro_f1?.high)}</small></span></div>`}).join('')}</div>`)}</div><p class="ccw-note">区间与 p 值描述已冻结 Test 的不确定性；不从图形位置推断额外显著性结论。</p>`);
    const values=pairs.flatMap(item=>[item.macro_f1_difference?.low,item.macro_f1_difference?.high,item.macro_f1_difference?.point].map(numeric).filter(value=>value!==null));
    const extent=Math.max(.01,...values.map(value=>Math.abs(value)))*115;
    const names=pairs.map(item=>`${item.b} − ${item.a}`);
    const forest=pairs.map((item,index)=>[numeric(item.macro_f1_difference?.low)*100,numeric(item.macro_f1_difference?.high)*100,numeric(item.macro_f1_difference?.point)*100,index]);
    draw('[data-chart="forest"]',baseChart({grid:{left:72,right:28,top:20,bottom:48,containLabel:true},xAxis:{type:'value',min:-extent,max:extent,name:'Δ Macro-F1 (pp)',nameLocation:'middle',nameGap:30,axisLine:{show:true},splitLine:{lineStyle:{color:'#e4eeee'}},axisLabel:{formatter:value=>`${value>0?'+':''}${value.toFixed(0)}`}},yAxis:{type:'category',inverse:true,data:names,axisTick:{show:false}},tooltip:{trigger:'item',formatter:p=>{const item=pairs[p.dataIndex];return item?`${escape(item.b)} − ${escape(item.a)}<br>${signedPoints(item.macro_f1_difference?.point)}<br>95% CI ${points(item.macro_f1_difference?.low)} ～ ${points(item.macro_f1_difference?.high)}`:'';}},series:[{type:'custom',data:forest,renderItem:(params,api)=>{const index=api.value(3),low=api.coord([api.value(0),index]),high=api.coord([api.value(1),index]),point=api.coord([api.value(2),index]),pairColor=COLORS[pairs[index]?.b]||COLORS.C2;return {type:'group',children:[{type:'line',shape:{x1:low[0],y1:low[1],x2:high[0],y2:high[1]},style:{stroke:pairColor,lineWidth:2.5}},{type:'line',shape:{x1:low[0],y1:low[1]-7,x2:low[0],y2:low[1]+7},style:{stroke:pairColor,lineWidth:2}},{type:'line',shape:{x1:high[0],y1:high[1]-7,x2:high[0],y2:high[1]+7},style:{stroke:pairColor,lineWidth:2}},{type:'circle',shape:{cx:point[0],cy:point[1],r:6},style:{fill:pairColor,stroke:'#fff',lineWidth:2}}]};}}]}),event=>{
      if (pairs[event.dataIndex]) {
        selectedPair=`${pairs[event.dataIndex].a}:${pairs[event.dataIndex].b}`;
        if (store.get().currentComparison===selectedPair) render();
        else store.set({currentComparison:selectedPair});
      }
    });
  }

  async function render() {
    const active=mode();
    const epoch=++loadEpoch;
    disposeCharts();
    shell(active);
    if (active==='formal') {renderFormal();return;}
    const endpoint={learning:'stage4/overview',ablation:'stage5/overview',uncertainty:'stage6/overview'}[active];
    if (!cache[active]) {
      setBody(empty(`正在读取 ${MODES[active]} 的冻结记录…`));
      try { cache[active]=await request(endpoint); }
      catch (error) { if (epoch===loadEpoch) setBody(`<div class="ccw-error" role="alert">${escape(error?.message || '分析记录读取失败。')}</div>`); return; }
    }
    if (epoch!==loadEpoch || active!==mode()) return;
    if (active==='learning') renderLearning(cache.learning);
    if (active==='ablation') renderAblation(cache.ablation);
    if (active==='uncertainty') renderUncertainty(cache.uncertainty);
  }
  if (store?.subscribe) store.subscribe(() => { render(); });
  if (typeof ResizeObserver!=='undefined') {
    const observer=new ResizeObserver(()=>charts.forEach(chart=>chart.resize()));
    observer.observe(element);
  }
  render();
  return {render,updateOverview(value){formalOverview=value; if (mode()==='formal') render();}};
}
