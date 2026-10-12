import { api, node } from './api.js';
import { activeTab } from './layout.js';
import { loadComparison } from './charts.js';
import { $,resources,bindAnalysis,bindProcess,bindSelection,bindErrors,tableInto,fail } from './course-ui.js';
async function init(){const ctx=await resources();const start=bindAnalysis(ctx,{trace:true});bindProcess(ctx);const errors=bindErrors(ctx);
  async function refresh(){if(activeTab()==='errors')await errors();if(activeTab()==='evaluation'){const version=ctx.version;await loadComparison(()=>version===ctx.version);const value=await api('/api/v1/experiments/1/comparison');tableInto($('reference-table'),['课件方法','P','R','F1','OOV-R','IV-R'],[['longest','Longest Matching'],['bigram','Bigram'],['hmm1-smooth','HMM1'],['hmm2-smooth','HMM2']].map(([id,name])=>[name,...(value.course_reference[id]||[null,null,null,null,null])]));}}
  const panels=[...$('process').querySelectorAll(':scope > .algorithm-panel, :scope > .model-information')],selector=document.querySelector('.page-header .header-controls'),header=document.querySelector('.page-header');
  const copy={
    longest:['Longest Matching','从当前位置选择训练词典中最长的匹配词；未匹配时退化为单字。','规则直接，词典内词语的匹配过程清楚。','多字未登录词难以被召回。'],
    bigram:['Bigram','在词网上使用相邻词概率与固定插值平滑，寻找最小代价路径。','利用相邻词上下文消解词典候选歧义。','依赖训练词典，多字未登录词召回有限。'],
    'hmm1-smooth':['HMM-1 Smooth','用字符发射与一阶 BMES 状态转移建模，通过 Viterbi 解码。','可组合字符形成训练词典外的词。','一阶状态依赖较短，边界仍可能误判。'],
    'hmm2-smooth':['HMM-2 Smooth','以最近两个 BMES 状态为动态规划状态，使用二阶转移。','显式利用更长的标签依赖。','状态组合更稀疏，不保证总体 F1 提升。'],
    'bilstm-crf':['BiLSTM-CRF','BiLSTM 编码双向字符上下文，CRF 联合解码合法 BMES 路径。','对未登录词的上下文泛化较强。','需要更多训练资源，词典内召回不一定领先。']
  };
  const processCopy={
    bigram:[['核心思想','从训练词典生成词网，同一起点可以有多个候选词。'],['二元概率','用前一个词估计当前词，并把概率转为负对数局部代价。'],['动态规划','保存到达状态的最佳累计代价与前驱，再回溯最优路径。'],['OOV / 平滑','未登录字符退化为单字；固定插值：0.9 × P_MLE + 0.1 × P_add-0.1。'],['适用场景','词典内候选有歧义时，利用相邻词统计选择路径。']],
    longest:[['候选扫描','从当前 offset 查找训练词典中的候选词。'],['选择规则','始终选择当前位置最长的词典匹配。'],['未登录字符','没有词典匹配时输出单字并继续扫描。']],
    'hmm1-smooth':[['状态','每个字符对应 B / M / E / S 中一个状态。'],['路径分数','用 π、A、B 的 log-score 累积最优合法路径。'],['前驱回溯','每个时刻保存最佳前驱，终止后回溯整句 BMES。']],
    'hmm2-smooth':[['状态对','DP 状态为 (前一状态, 当前状态)。'],['二阶转移','使用前两个状态估计下一状态的转移分数。'],['前驱回溯','保存更早状态作为 backpointer，终止后回溯 BMES。']],
    'bilstm-crf':[['双向上下文','Embedding 与 BiLSTM 从左右两个方向表示字符。'],['Emission','Linear 层给出 BMES emission score，它不是独立标签概率。'],['联合解码','CRF 根据 emission 与转移分数解码合法 BMES 路径。']]
  };
  const symbols={longest:'L',bigram:'2G','hmm1-smooth':'H¹','hmm2-smooth':'H²','bilstm-crf':'CRF'};
  const formulas={longest:'词典候选中取当前位置最长匹配；无匹配时退化为单字。',bigram:'0.9 × P_MLE(wᵢ | wᵢ₋₁) + 0.1 × P_add-0.1(wᵢ)', 'hmm1-smooth':'log π + Σ log A(stateᵢ₋₁,stateᵢ) + Σ log B(charᵢ | stateᵢ)', 'hmm2-smooth':'log π + 初始状态对 + Σ log A₂(stateᵢ₋₂,stateᵢ₋₁,stateᵢ) + Σ log B', 'bilstm-crf':'Embedding → BiLSTM → Linear emission → CRF 路径解码'};
  const officialMetrics=new Map();let describedModel=null;
  function showBenchmark(){const target=$('method-benchmark'),row=officialMetrics.get(ctx.model());if(!row)return;target.replaceChildren(node('h3','MSR 正式结果摘要'));for(const [label,key] of [['F1','f1'],['OOV-R','oov_recall']]){const line=node('div','','benchmark-mini-row');line.append(node('span',label),node('strong',`${(row[key]*100).toFixed(2)}%`));const rail=node('span','','benchmark-mini-rail');rail.style.setProperty('--value',`${row[key]*100}%`);target.append(line,rail);}target.append(node('small','由冻结的 MSR comparison 只读加载。'));}
  function showMethod(){const model=ctx.model(),[name,idea,strength,limit]=copy[model]||copy['hmm1-smooth'];$('method-name').textContent=name;$('method-idea').textContent=idea;$('method-strength').textContent=strength;$('method-limit').textContent=limit;$('method-emblem').textContent=symbols[model];$('method-formula').textContent=formulas[model];$('process-method-name').textContent=`${name} · 运算说明`;const notes=$('process-method-notes');notes.replaceChildren();for(const [index,[title,body]] of (processCopy[model]||[]).entries()){const card=node('div','','process-explain-step');card.append(node('span',String(index+1).padStart(2,'0'),'process-explain-number'),node('h3',title),node('p',body));notes.append(card);}document.querySelectorAll('.method-strip-item').forEach(item=>item.classList.toggle('is-active',item.dataset.model===model));if(describedModel!==model){$('process-live-step').textContent='选择候选边或播放 Trace，查看当前位置的真实代价与前驱。';describedModel=model;}showBenchmark();}
  function showProcess(){const interactive=activeTab()==='interactive';(interactive?$('workbench-model-slot'):header).append(selector);if(interactive){$('interactive-process-slot').append(panels[0]);$('interactive-deep-slot').append(panels[1]);}else for(const panel of panels)$('process').append(panel);showMethod();}
  async function methodStrip(){const value=await api('/api/v1/experiments/1/comparison'),target=$('method-strip'),methods=[['longest','Longest Matching'],['bigram','Bigram'],['hmm1-smooth','HMM-1 Smooth'],['hmm2-smooth','HMM-2 Smooth'],['bilstm-crf','BiLSTM-CRF']];target.replaceChildren();for(const [model,name] of methods){const row=value.rows.find(item=>item.model===name);if(!row)continue;officialMetrics.set(model,row);const card=node('button','','method-strip-item');card.type='button';card.dataset.model=model;card.disabled=[...$('segmentation-model').options].find(option=>option.value===model)?.disabled??true;const top=node('span','','method-strip-top');top.append(node('span',symbols[model],'method-strip-symbol'),node('strong',name));card.append(top,node('span',`F1 ${(row.f1*100).toFixed(2)}% · OOV-R ${(row.oov_recall*100).toFixed(2)}%`,'method-strip-metrics'));const bar=node('span','','method-strip-bar');bar.style.setProperty('--value',`${row.f1*100}%`);card.append(bar);card.addEventListener('click',()=>{$('segmentation-model').value=model;$('segmentation-model').dispatchEvent(new Event('change',{bubbles:true}));});target.append(card);}showMethod();}
  $('algorithm-trace').addEventListener('trace-playback-step',event=>{
    const {frame,index,total}=event.detail,target=$('process-live-step');target.replaceChildren(node('strong',`第 ${index+1} / ${total} 步`));
    if(frame.edge){const edge=frame.edge;target.append(node('p',`${edge.from[1]} → ${edge.to[1]} · ${edge.to[0]} 位置`),node('small',`P=${(edge.probability??Math.exp(-edge.negative_log_cost)).toExponential(4)} · −log P=${edge.negative_log_cost.toFixed(4)} · 累计 ${edge.cumulative_cost.toFixed(4)}`));}
    else if(frame.step){target.append(node('p',`回溯词语 ${frame.step.word}；前驱 ${frame.step.previous}`));}
    else if(frame.state){const state=['B','M','E','S'].indexOf(frame.state),value=frame.values?.[state];target.append(node('p',`当前字符「${frame.char}」的最优路径标签是 ${frame.state}。`),node('small',`${frame.previous} → ${frame.state} · 累计 log-score ${Number.isFinite(value)?value.toFixed(4):'−∞'}`));}
    else if(frame.char){target.append(node('p',`当前字符「${frame.char}」；查看中栏状态与分数。`));}
  });
  bindSelection(ctx,refresh);$('segmentation-model').addEventListener('change',showMethod);
  document.addEventListener('module-tab',()=>{showProcess();refresh().catch(fail);if(['interactive','process'].includes(activeTab())&&$('original-text').classList.contains('result-empty'))start();});
  showProcess();await refresh();methodStrip().catch(fail);if(['interactive','process'].includes(activeTab()))start();
}init().catch(fail);
