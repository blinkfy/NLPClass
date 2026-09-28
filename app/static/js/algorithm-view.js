import { node } from "./api.js";

const states = ["B", "M", "E", "S"];
const plots = new Map();
const names = {longest: "Longest Matching", bigram: "Bigram", "hmm1-smooth": "HMM-1 Smooth", "hmm2-smooth": "HMM-2 Smooth", "bilstm-crf": "BiLSTM-CRF"};
const score = (v) => v == null || !Number.isFinite(v) ? "−∞" : v.toFixed(4);
const playbackTimers = new WeakMap();

export function clearAlgorithm(target) {
  if(playbackTimers.has(target)){clearInterval(playbackTimers.get(target));playbackTimers.delete(target);}
  for (const [element, chart] of plots) { if (target.contains(element)) { chart.dispose(); plots.delete(element); } }
  delete target.dataset.traceType;
  target.replaceChildren();
}
export function resizeAlgorithm() { plots.forEach((chart) => chart.resize()); }
function heading(target, text, note = "") {
  target.append(node("h3", text));
  if (note) target.append(node("p", note, "muted"));
}
function table(target, headers, rows, className = "") {
  const wrap = node("div", "", "table-wrap"), result = node("table", "", `process-table ${className}`), head = node("tr");
  headers.forEach((text) => head.append(node("th", text)));
  result.append(head);
  rows.forEach((cells) => {
    const tr = node("tr");
    cells.forEach((value) => { const td = node("td"); td.append(value instanceof Node ? value : document.createTextNode(String(value ?? "—"))); tr.append(td); });
    result.append(tr);
  });
  wrap.append(result); target.append(wrap); return result;
}
function path(target, text, tags, label) {
  heading(target, label);
  const strip = node("div", "", "decoded-path");
  Array.from(text).forEach((char, t) => {const cell=node("span", `${char === " " ? "␠" : char} (${tags[t]})`, "path-state");cell.dataset.index=String(t);strip.append(cell);});
  target.append(strip);
}
function flow(target, labels, className = "") {
  const strip = node("div", "", `architecture-flow ${className}`);
  labels.forEach((label, i) => {
    if (i) { const arrow = node("span", "→", "flow-arrow"); arrow.style.setProperty("--step", i); strip.append(arrow); }
    const step = node("span", label, "flow-step"); step.style.setProperty("--step", i); strip.append(step);
  });
  target.append(strip);
}
function playback(target, frames, renderStep, label="计算步骤") {
  if(!frames.length)return;
  const wrap=node("section","","trace-playback"),head=node("div","","trace-playback-head"),title=node("strong",label),count=node("span"),controls=node("div","","trace-playback-controls"),previous=node("button","上一步","outline small"),play=node("button","播放","outline small"),next=node("button","下一步","outline small"),slider=node("input"),stage=node("div","","trace-step-stage");
  slider.type="range";slider.min="0";slider.max=String(frames.length-1);slider.value="0";slider.setAttribute("aria-label",`${label}进度`);
  controls.append(previous,play,next);head.append(title,count,controls);wrap.append(head,slider,stage);target.append(wrap);
  let index=0;
  function stop(){if(playbackTimers.has(target)){clearInterval(playbackTimers.get(target));playbackTimers.delete(target);}play.textContent="播放";}
  function show(value){index=Math.max(0,Math.min(frames.length-1,value));slider.value=String(index);count.textContent=`${index+1} / ${frames.length}`;previous.disabled=index===0;next.disabled=index===frames.length-1;stage.replaceChildren();renderStep(frames[index],index,stage);target.dispatchEvent(new CustomEvent('trace-playback-step',{detail:{frame:frames[index],index,total:frames.length}}));if(index===frames.length-1)stop();}
  previous.addEventListener("click",()=>{stop();show(index-1);});next.addEventListener("click",()=>{stop();show(index+1);});slider.addEventListener("input",()=>{stop();show(Number(slider.value));});
  play.addEventListener("click",()=>{if(playbackTimers.has(target)){stop();return;}if(index===frames.length-1)show(0);play.textContent="暂停";const timer=setInterval(()=>{if(!target.isConnected){stop();return;}show(index+1);},750);playbackTimers.set(target,timer);});show(0);return {show};
}
function stateScores(target, values, chosen, title="状态分数") {
  target.append(node("p",title,"trace-step-caption"));const grid=node("div","","trace-state-grid");
  states.forEach((state,i)=>{const cell=node("span",`${state}  ${score(values?.[i])}`,`trace-state${chosen===state?' is-active':''}`);grid.append(cell);});target.append(grid);
}
function matrixCell(value, predecessor, selected = false, masked = false) {
  const cell = node("span", "", `${selected ? "selected-score " : ""}${masked ? "masked-score" : "score-cell"}`);
  cell.append(node("strong", masked ? "−∞ / impossible" : score(value)));
  if (predecessor != null) cell.append(node("small", predecessor));
  return cell;
}
function scan(target, trace) {
  heading(target, "最长匹配扫描过程", "候选按长度扫描；高亮当前选择，未登录字符退化为单字。offset 使用 Unicode code point。");
  const chars = Array.from(trace.text);
  playback(target,trace.steps,(step,index,stage)=>{
    stage.append(node("p",`Offset ${step.offset} → ${step.next_offset} · 剩余文本：${chars.slice(step.offset).join("")}`,"trace-step-caption"));
    const candidates=node("div","","candidate-list");for(const item of step.candidate_info||[]){candidates.append(node("span",`${item.word} · ${item.kind==="dictionary match"?"词典":item.kind==="single-char fallback"?"单字 fallback":item.kind}`,`candidate${item.selected?" chosen":""}`));}stage.append(candidates);
    stage.append(node("p",`第 ${index+1} 步选中「${step.chosen}」${(step.candidate_info||[]).some(c=>c.selected&&c.kind==="single-char fallback")?"；未登录字符退化为单字":""}。`,"trace-step-result"));
  },"最长匹配逐步扫描");
  const details=node("details","","candidate-details");details.append(node("summary","展开完整扫描表"));
  table(details, ["Step", "Offset", "剩余文本", "候选词", "选择词", "Next Offset"], trace.steps.map((step, i) => {
    const candidates = node("div", "", "candidate-list");
    for (const candidate of step.candidate_info || step.candidates.map(word => ({word, selected: word === step.chosen, kind: "saved candidate"}))) {
      const item = node("span", `${candidate.word} · ${candidate.kind === "dictionary match" ? "词典" : candidate.kind === "single-char fallback" ? "单字 fallback" : candidate.kind}`, `candidate ${candidate.selected ? "chosen" : ""}`);
      candidates.append(item);
    }
    return [i + 1, step.offset, chars.slice(step.offset).join(""), candidates, step.chosen, step.next_offset ?? step.offset + Array.from(step.chosen).length];
  }));target.append(details);
  target.append(node("p", `当前输入 OOV 单字 fallback：${trace.fallback_count ?? "未保存"} 次。`, "process-note"));
}
const svgNS = "http://www.w3.org/2000/svg";
function svgNode(tag, attrs = {}, text = "") {
  const item = document.createElementNS(svgNS, tag);
  Object.entries(attrs).forEach(([key, value]) => item.setAttribute(key, value));
  item.textContent = text; return item;
}
function lattice(target, trace) {
  heading(target, "Word Lattice · 词网", "节点为字符 offset；候选边连接 start → end。青色高亮已保存的最优分词路径。");
  target.append(node("p", `最优分词：${trace.tokens.join(" / ")} · 最优总代价 ${score(trace.cost)}`, "lattice-summary"));
  const chars = Array.from(trace.text), width = Math.max(760, chars.length * 95 + 90);
  const gap = (width - 100) / Math.max(chars.length, 1), base = 154;
  const wrap = node("div", "", "lattice-wrap");
  const svg = svgNode("svg", {viewBox: `0 0 ${width} 235`, width, height: 235, role: "img", "aria-label": "Bigram 候选词网与最优路径"});
  const defs = svgNode("defs"), marker = svgNode("marker", {id: "lattice-arrow", markerWidth: 7, markerHeight: 7, refX: 6, refY: 3, orient: "auto"});
  marker.append(svgNode("path", {d: "M0,0 L6,3 L0,6", fill: "#087f8c"})); defs.append(marker); svg.append(defs);
  const best = new Set((trace.optimal_chain || []).map(e => `${e.start}:${e.end}:${e.word}`));
  let offset = 0;
  for (const word of trace.tokens) { const end = offset + Array.from(word).length; best.add(`${offset}:${end}:${word}`); offset = end; }
  for (const [index, edge] of trace.lattice.entries()) {
    const selected = best.has(`${edge.start}:${edge.end}:${edge.word}`), x1 = 45 + edge.start * gap, x2 = 45 + edge.end * gap;
    const lift = Math.min(124, 25 + (edge.end - edge.start) * 29);
    const curve = svgNode("path", {d: `M${x1},${base} C${x1},${base-lift} ${x2},${base-lift} ${x2},${base}`, fill: "none", stroke: selected ? "#087f8c" : "#b9cbd3", "stroke-width": selected ? 3.5 : 1.3, "marker-end": selected ? "url(#lattice-arrow)" : "", "data-best": String(selected), "data-start": edge.start, "data-end": edge.end, "data-word": edge.word, class: "lattice-edge"});
    curve.style.setProperty("--edge-order", Math.min(index, 16)); svg.append(curve);
    svg.append(svgNode("text", {x: (x1+x2)/2, y: base-lift*0.78-5, "text-anchor": "middle", fill: selected ? "#065c66" : "#6e838e", "font-size": 14, "font-weight": selected ? 700 : 400, "data-start": edge.start, "data-end": edge.end, "data-word": edge.word, class: "lattice-choice", role: "button", tabindex: "0", "aria-label": `查看候选词 ${edge.word} 的代价`}, edge.word));
  }
  for (let i=0; i<=chars.length; i++) {
    const x = 45+i*gap;
    svg.append(svgNode("circle", {cx:x, cy:base, r:5, fill:"#087f8c"}));
    svg.append(svgNode("text", {x, y:base+24, "text-anchor":"middle", "font-size":13}, String(i)));
    if(i<chars.length) svg.append(svgNode("text", {x:x+gap/2, y:base+49, "text-anchor":"middle", "font-size":15}, chars[i]));
  }
  wrap.append(svg); target.append(wrap);
  heading(target, "最优路径", "高亮边组成的词序列；路径由词图上的 Viterbi / 最短路径动态规划得到。");
  flow(target, ["BOS", ...trace.tokens, "EOS"], "best-path");
  target.append(node("p", `最优总代价 ${score(trace.cost)}；EOS 代价 ${score(trace.eos_cost)}`, "process-note"));
  const frames=[...trace.edges.map(edge=>({phase:"候选边代价",edge})),...(trace.optimal_chain||[]).slice().reverse().map(step=>({phase:"最优路径回溯",step}))];
  const controller=playback(target,frames,(frame,index,stage)=>{
    const start=frame.edge?.from[0]??frame.step.start,end=frame.edge?.to[0]??frame.step.end,word=frame.edge?.to[1]??frame.step.word;
    for(const curve of svg.querySelectorAll(".lattice-edge"))curve.classList.toggle("is-active",Number(curve.dataset.start)===start&&Number(curve.dataset.end)===end&&curve.dataset.word===word);
    stage.append(node("strong",`${frame.phase} · ${word} [${start},${end})`));
    if(frame.edge){const edge=frame.edge;stage.append(node("p",`${edge.from[1]} → ${edge.to[1]} · P=${(edge.probability??Math.exp(-edge.negative_log_cost)).toExponential(4)} · −log P=${score(edge.negative_log_cost)} · 累计代价=${score(edge.cumulative_cost)}`,"trace-step-caption"));stage.append(node("small",edge.on_best_path?"最终最优路径上的边":edge.best_to_state?"到达该状态的局部最佳前驱":"候选边，未进入最终路径"));}
    else{const state=trace.states[end]?.[word];stage.append(node("p",`前驱 ${frame.step.previous}；状态累计代价 ${score(state?.cost)}。回溯读取已保存的最优路径。`,"trace-step-caption"));}
  },"候选边计算 → 最优路径回溯");
  const inspectEdge=(element)=>{const index=trace.edges.findIndex(edge=>edge.to[0]===Number(element.dataset.end)&&edge.to[1]===element.dataset.word&&edge.from[0]===Number(element.dataset.start));if(index>=0)controller.show(index);};
  svg.querySelectorAll(".lattice-edge, .lattice-choice").forEach(element=>{element.addEventListener("click",()=>inspectEdge(element));if(element.matches(".lattice-choice"))element.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();inspectEdge(element);}});});
  heading(target, "局部转移代价与前驱摘要", "逐步回放读取已有 Trace，不参与重新求解。");
  const chainRows=(trace.optimal_chain || []).map(step => {
    const state = trace.states[step.end]?.[step.word];
    return [step.start, step.end, step.previous, step.word, state ? score(state.cost) : "—"];
  });
  table(target, ["From", "To", "前一个词", "当前词", "累计代价"], chainRows.slice(0,5));
  if(chainRows.length>5){const fullChain=node("details","","candidate-details");fullChain.append(node("summary",`展开完整最优路径前驱（${chainRows.length} 步）`));table(fullChain,["From","To","前一个词","当前词","累计代价"],chainRows);target.append(fullChain);}
  const edges = node("details", "", "candidate-details");edges.append(node("summary", `展开全部候选边（${trace.edges.length}）`));
  heading(edges, "全部候选边", "Best? 标记最终最优链；局部最佳表示到达该状态的最佳前驱。");
  table(edges, ["From", "To", "Previous", "Current", "Probability", "−log P", "Cumulative Cost", "Best?"], trace.edges.map(edge => [edge.from[0], edge.to[0], edge.from[1], edge.to[1], (edge.probability ?? Math.exp(-edge.negative_log_cost)).toExponential(4), score(edge.negative_log_cost), score(edge.cumulative_cost), edge.on_best_path ? "最优路径 ✓" : edge.best_to_state ? "局部最佳" : "—"]));
  const dp = node("details", "", "candidate-details");dp.append(node("summary", "展开完整 DP 状态与前驱"));
  table(dp, ["End", "词", "累计代价", "前驱"], trace.states.flatMap((frame,t) => Object.entries(frame).map(([word,state]) => [t,word,score(state.cost),JSON.stringify(state.predecessor)])));
  target.append(edges,dp);
}
function hmm1(target, trace) {
  heading(target, "一阶 Viterbi · 最优路径", "当前句子经 BMES 合法状态与 log-score 动态规划后得到的路径。");
  path(target, trace.text, trace.tags, "字符与解码状态");
  const rows = [], bestRows=[], frames=[];
  for (const frame of trace.chunks || [trace]) {
    const chars = Array.from(frame.text || trace.text), offset = frame.offset || 0;
    if (frame.whitespace) { rows.push([offset, "␠", "—", "—", "—", "独立 S", "S"]); bestRows.push([offset,"␠","S","独立空白","—"]);frames.push({offset,char:"␠",state:"S",previous:"独立空白",values:null}); continue; }
    if (!frame.score) continue;
    frame.score.forEach((values,t) => {const state=states.indexOf(frame.tags[t]),previous=frame.backpointer[t][state];rows.push([offset+t,chars[t],...values.map((v,s) => matrixCell(v,frame.backpointer[t][s] >= 0 ? `← ${states[frame.backpointer[t][s]]}` : "起始/无前驱", frame.tags[t] === states[s])),frame.tags[t]]);bestRows.push([offset+t,chars[t],frame.tags[t],previous>=0?states[previous]:"起始",score(values[state])]);frames.push({offset:offset+t,char:chars[t],state:frame.tags[t],previous:previous>=0?states[previous]:"起始",values});});
  }
  const controller=playback(target,frames,(frame,index,stage)=>{
    target.querySelectorAll(".decoded-path .path-state").forEach(cell=>cell.classList.toggle("is-active",Number(cell.dataset.index)===frame.offset));
    target.querySelectorAll(".hmm-predecessor-step").forEach(cell=>cell.classList.toggle("is-active",Number(cell.dataset.index)===index));
    const focus=node("div","","hmm-step-focus"),character=node("span",frame.char,"hmm-focus-char"),detail=node("div","","hmm-focus-detail"),selected=frame.values?.[states.indexOf(frame.state)];
    detail.append(node("small",`当前位置 t=${frame.offset}`),node("strong",`最优路径状态 ${frame.state}`),node("span",`累计 log-score ${score(selected)}`));focus.append(character,detail);stage.append(focus);
    stateScores(stage,frame.values,frame.state,"四个 BMES 状态的累计 log-score");
    stage.append(node("p",`回溯前驱：${frame.previous} → ${frame.state}。高亮状态属于已保存的完整最优路径。`,"trace-step-result"));
  },"一阶 Viterbi 逐字符回放");
  if(bestRows.length){heading(target,"最优路径的前驱摘要","点击字符可跳到对应的回放步骤；完整矩阵在下方折叠区。");const strip=node("div","","hmm-predecessor-strip");bestRows.slice(0,12).forEach(([position,char,state,previous],index)=>{const button=node("button",`${char}  ${previous}→${state}`,`hmm-predecessor-step${index===0?" is-active":""}`);button.type="button";button.dataset.index=String(index);button.addEventListener("click",()=>controller.show(index));strip.append(button);});target.append(strip);if(bestRows.length>12)target.append(node("p",`先显示 12 / ${bestRows.length} 个位置；其余位置可用进度条查看。`,"muted"));}
  if (rows.length) {const details=node("details","","candidate-details");details.append(node("summary","展开完整 B / M / E / S score matrix 与 backpointer"));table(details, ["t", "Char", ...states, "Decoded State"], rows, "viterbi-matrix");target.append(details);}
  else target.append(node("p", trace.trace_note || "当前保存输出没有逐步 score/backpointer。", "process-note"));
}
function hmm2(target, trace) {
  heading(target, "二阶 Viterbi · 最优路径", "DP 状态是 (previous, current)，即 score[t][prev][current]。");
  path(target,trace.text,trace.tags,"最佳 BMES path（最近两个状态为 DP 状态）");
  const frames = [];
  for (const frame of trace.chunks || [trace]) {
    if (!frame.score) continue;
    Array.from(frame.text || trace.text).forEach((char,t) => frames.push({frame,t,char,offset:frame.offset || 0}));
  }
  playback(target,frames,({frame,t,char,offset},index,stage)=>{
    target.querySelectorAll(".decoded-path .path-state").forEach(cell=>cell.classList.toggle("is-active",Number(cell.dataset.index)===offset+t));
    const current=frame.tags[t],previous=t?frame.tags[t-1]:"起始";stage.append(node("strong",`t=${offset+t} · ${char} · ${previous} → ${current}`));
    if(t===0){stateScores(stage,frame.first_score??states.map((_,i)=>frame.score[0][i][i]),current,"首状态初始化 log-score");return;}
    const pairs=node("div","","trace-pair-grid"),detail=node("p",`最优路径状态对 ${previous} → ${current}`,"trace-step-result");
    states.forEach((prev,p)=>states.forEach((next,c)=>{if(!trace.pair_mask?.[p]?.[c])return;const selected=prev===previous&&next===current,cell=node("button",`${prev}→${next}  ${score(frame.score[t][p][c])}`,`trace-state${selected?" is-active":""}`);cell.type="button";cell.addEventListener("click",()=>{const r=frame.backpointer[t][p][c];detail.textContent=t===1?`首状态对初始化：${prev} → ${next}；累计 ${score(frame.score[t][p][c])}`:`前驱 ${r>=0?states[r]:"无有效前驱"} → ${prev} → ${next}；累计 ${score(frame.score[t][p][c])}`;});pairs.append(cell);}));stage.append(pairs,detail);
  },"二阶状态对逐字符回放");
  const bestRows=frames.slice(0,8).map(({frame,t,char,offset})=>{const current=frame.tags[t],previous=t?frame.tags[t-1]:"起始",p=states.indexOf(previous),c=states.indexOf(current);return [offset+t,char,previous,current,t?score(frame.score[t][p][c]):score(frame.first_score?.[c] ?? frame.score[0][c][c])];});
  if(bestRows.length){heading(target,"状态对路径摘要");table(target,["t","字符","前一状态","当前状态","累计 log-score"],bestRows);if(frames.length>8)target.append(node("p",`先显示 8 / ${frames.length} 个位置；完整状态对见下方。`,"muted"));}
  const details=node("details","","candidate-details");details.append(node("summary","展开完整二阶状态对 score matrix 与 backpointer"));
  const label = node("label", "字符位置 "), select = node("select"); select.setAttribute("aria-label", "二阶 DP 字符位置");
  frames.forEach((f,i) => {const option=node("option",`t=${f.offset+f.t} / 字符=${f.char}`); option.value=i; select.append(option);});
  label.append(select); details.append(label);
  const content = node("div"), detail=node("p","请选择一个合法状态对。","process-note"); details.append(content,detail);
  function show() {
    content.replaceChildren(); detail.textContent="请选择一个合法状态对。";
    const selected=frames[Number(select.value)]; if(!selected) return;
    const {frame,t}=selected;
    if(t===0) { table(content,["初始化状态",...states],[["π + 首字符 emission",...states.map((s,i)=>score(frame.first_score?.[i] ?? frame.score[0][i][i]))]]); detail.textContent="t=0 只有首状态；状态对从 t=1 开始初始化。"; return; }
    table(content,["Previous \\ Current",...states],states.map((prev,p)=>[prev,...states.map((current,c)=>{
      const legal=trace.pair_mask?.[p]?.[c] ?? ({B:"ME",M:"ME",E:"BS",S:"BS"}[prev].includes(current));
      const button=node("button",legal?score(frame.score[t][p][c]):"−∞ / impossible",`pair-cell ${legal?"":"masked-score"}`); button.disabled=!legal;
      if(prev===frame.tags[t-1]&&current===frame.tags[t]) button.classList.add("selected-score");
      button.addEventListener("click",()=>{ const r=frame.backpointer[t][p][c]; detail.textContent=t===1?`首对初始化：${prev} → ${current}；score=${score(frame.score[t][p][c])}`:`${r>=0?states[r]:"无有效前驱"} → ${prev} → ${current}；累计 log-score=${score(frame.score[t][p][c])}`; }); return button;
    })]));
  }
  if(frames.length>1) select.value="1"; select.addEventListener("change",show); show();
  target.append(details);
}
function architecture(target) { flow(target,["Characters","Embedding 128","BiLSTM 128 × 2","Linear 4","CRF","BMES"]); }
function crfTransition(target, values) {
    heading(target,"CRF transition · From × To","非法 BMES 转移显式 masked；其余为可学习分数，不是转移概率。");
    const legalValues=values.flat().filter(v=>v>-9999), low=Math.min(...legalValues), high=Math.max(...legalValues);
    table(target,["From \\ To",...states],states.map((from,y)=>[from,...values[y].map((v)=>{
      const cell=matrixCell(v,null,false,v<=-9999);
      if(v>-9999) cell.style.backgroundColor=`rgba(8,127,140,${0.08+0.42*(v-low)/(high-low||1)})`;
      return cell;
    })]));
}
function neural(target, trace) {
  heading(target,"BiLSTM-CRF · 双向上下文与结构化解码"); architecture(target);
  path(target,trace.text,trace.tags,"CRF 最终 BMES decoded path");
  heading(target,"Linear emission 与 CRF 路径","Emission score 不是独立标签概率；最终标签来自 CRF 联合解码。");
  const rows=[],frames=[];
  for(const frame of trace.chunks || [trace]) {
    const chars=Array.from(frame.text || trace.text),offset=frame.offset||0;
    if(frame.whitespace) {rows.push(["␠","—","—","—","—","S（空白保留）"]);frames.push({offset,char:"␠",values:null,decoded:"S"});continue;}
    frame.emissions?.forEach((values,t)=>{rows.push([chars[t],...values.map((v,s)=>matrixCell(v,null,frame.tags[t]===states[s])),frame.tags[t]]);frames.push({offset:offset+t,char:chars[t],values,decoded:frame.tags[t]});});
  }
  playback(target,frames,(frame,index,stage)=>{target.querySelectorAll(".decoded-path .path-state").forEach(cell=>cell.classList.toggle("is-active",Number(cell.dataset.index)===frame.offset));stage.append(node("strong",`字符 ${frame.char} · CRF decoded ${frame.decoded}`));stateScores(stage,frame.values,frame.decoded,"Linear emission score（非概率）");stage.append(node("small","高亮为 CRF 整体解码标签，不代表单点 emission 最大值。"));},"逐字符查看 emission 与解码");
  const emissions=node("details","","candidate-details");emissions.append(node("summary","展开完整 emission score 表"));table(emissions,["Char",...states,"Decoded"],rows,"emission-matrix");target.append(emissions);
  if(trace.crf_transitions){const transitions=node("details","","candidate-details");transitions.append(node("summary","展开 CRF transition matrix"));crfTransition(transitions,trace.crf_transitions);target.append(transitions);}
}
export function renderCompactTrace(target,trace) {
  if(!target)return;
  target.replaceChildren();
  const chars=Array.from(trace.text),limit=Math.min(chars.length,14),type=trace.trace_type;
  if(type==="longest_scan"){
    const strip=node("div","","compact-scan");
    for(const step of trace.steps.slice(0,6)){const card=node("div","","compact-scan-step");card.append(node("small",`OFFSET ${step.offset} → ${step.next_offset}`),node("strong",step.chosen),node("span",`${(step.candidate_info||[]).length} 个候选${(step.candidate_info||[]).some(item=>item.selected&&item.kind==="single-char fallback")?" · 单字 fallback":""}`));strip.append(card);}
    target.append(strip);if(trace.steps.length>6)target.append(node("small",`展示前 6 / ${trace.steps.length} 次扫描；完整过程见下方 Trace。`));return;
  }
  if(type==="bigram_lattice"){
    const width=Math.max(620,chars.length*95+60),gap=(width-64)/Math.max(chars.length,1),base=122,wrap=node("div","","compact-lattice-wrap"),svg=svgNode("svg",{viewBox:`0 0 ${width} 180`,width,height:180,role:"img","aria-label":"由当前 Bigram Trace 绘制的候选词网"});
    const best=new Set((trace.optimal_chain||[]).map(edge=>`${edge.start}:${edge.end}:${edge.word}`));let offset=0;for(const word of trace.tokens){const end=offset+Array.from(word).length;best.add(`${offset}:${end}:${word}`);offset=end;}
    for(const edge of trace.lattice){const x1=32+edge.start*gap,x2=32+edge.end*gap,lift=Math.min(92,23+(edge.end-edge.start)*22),selected=best.has(`${edge.start}:${edge.end}:${edge.word}`);svg.append(svgNode("path",{d:`M${x1},${base} C${x1},${base-lift} ${x2},${base-lift} ${x2},${base}`,fill:"none",stroke:selected?"#087f8c":"#cbdde3","stroke-width":selected?3:1.2}));if(selected)svg.append(svgNode("text",{x:(x1+x2)/2,y:base-lift*.8-5,"text-anchor":"middle",fill:"#075f6d","font-size":13,"font-weight":700},edge.word));}
    for(let i=0;i<=chars.length;i++){const x=32+i*gap;svg.append(svgNode("circle",{cx:x,cy:base,r:4,fill:"#087f8c"}));svg.append(svgNode("text",{x,y:base+19,"text-anchor":"middle","font-size":11},String(i)));}
    wrap.append(svg);target.append(wrap,node("p",`最优路径：${trace.tokens.join(" → ")} · 累计代价 ${score(trace.cost)}`,"compact-path-caption"));return;
  }
  if(type==="hmm2_pair_viterbi"){
    const strip=node("div","","compact-pair-path");for(let t=0;t<limit;t++){const pair=t?`${trace.tags[t-1]}→${trace.tags[t]}`:`起始→${trace.tags[t]}`;const card=node("span","","compact-pair-step");card.append(node("small",chars[t]),node("strong",pair));strip.append(card);}
    target.append(node("p","二阶 DP 以相邻状态对为节点，路径由已保存的二阶 Viterbi 解码得到。","muted"),strip);if(chars.length>limit)target.append(node("small",`展示前 ${limit} / ${chars.length} 个字符。`));return;
  }
  if(type==="bilstm_crf_emission"){
    const strip=node("div","","compact-neural-path");for(let t=0;t<limit;t++){const item=node("span","","compact-neural-step");item.append(node("small",chars[t]),node("strong",trace.tags[t]));strip.append(item);}target.append(node("p","Embedding → BiLSTM → Linear emission → CRF 联合解码","compact-flow-label"),strip,node("small","高亮是 CRF 解码路径，非逐字符 emission 最大值。"));return;
  }
  const values=new Map();for(const chunk of trace.chunks||[trace]){if(!chunk.score)continue;chunk.score.forEach((row,t)=>values.set((chunk.offset||0)+t,row));}
  const matrix=node("div","","mini-trellis");matrix.style.setProperty("--columns",String(limit));
  matrix.append(node("span","状态 / 字符","mini-trellis-corner"));for(let t=0;t<limit;t++)matrix.append(node("span",chars[t],"mini-trellis-char"));
  for(const [s,state] of states.entries()){matrix.append(node("strong",state,"mini-trellis-label"));for(let t=0;t<limit;t++){const cell=node("span",trace.tags[t]===state?state:"·",`mini-trellis-cell${trace.tags[t]===state?" is-path":""}`);cell.title=`${chars[t]} · ${state} · log-score ${score(values.get(t)?.[s])}`;matrix.append(cell);}}
  const scroll=node("div","","mini-trellis-wrap");scroll.append(matrix);target.append(scroll,node("p",`最佳 BMES path：${trace.tags.slice(0,limit).join(" → ")}${chars.length>limit?" …":""}`,"compact-path-caption"));
}
export function renderTrace(target,trace) {
  clearAlgorithm(target); target.dataset.traceType=trace.trace_type || trace.model;
  target.append(node("p",`当前模型：${names[trace.model] || trace.model} · ${Array.from(trace.text).length} code points`,"process-note"));
  const type=trace.trace_type;
  if(type==="longest_scan"||trace.steps) scan(target,trace);
  else if(type==="bigram_lattice"||trace.lattice) lattice(target,trace);
  else if(type==="hmm2_pair_viterbi"||trace.model==="hmm2") hmm2(target,trace);
  else if(type==="bilstm_crf_emission"||trace.architecture) neural(target,trace);
  else hmm1(target,trace);
  target.append(node("p",`最终分词：${trace.tokens.join(" / ")} · ${trace.bmes_source || "saved decoded output"}`,"process-note"));
}
export function renderModelInfo(target,value,model,trace=null,{trainingCurve=true}={}) {
  for (const [element, chart] of plots) { if (target.contains(element)) { chart.dispose(); plots.delete(element); } }
  target.replaceChildren(); const p=value.parameters;
  heading(target,`${names[model]} · 模型信息 / 参数`,"模型参数与本次句子的动态规划过程分开显示。");
  if(model==="longest") {
    table(target,["Dictionary size","Max word length","Fallback strategy","当前 fallback 次数"],[[p.dictionary_size,p.max_word_length,"OOV 字符 → 单字；空白保留",trace?.fallback_count??"分析短文本后显示"]]); return;
  }
  if(model==="bigram") {
    table(target,["Unigram vocabulary","Unigram count","Bigram types","BOS / EOS"],[[p.dictionary_size,p.word_count,p.bigram_types,"<BOS> / <EOS>"]]);
    target.append(node("p","0.9 × P_MLE(wᵢ | wᵢ₋₁) + 0.1 × P_add-0.1(wᵢ)","formula"),node("p","依据课件中的二元语法与低阶平滑思想，本项目采用固定插值实现。未观测上下文完整回退到平滑一元分布。","muted")); return;
  }
  if(model==="bilstm-crf") {
    architecture(target);
    table(target,["Parameters","Vocabulary","Seed","CPU threads","Best dev epoch","Max epoch"],[[p.parameter_count,p.vocabulary_size,p.config.seed,p.config.cpu_threads,p.training.best_epoch,p.config.max_epochs]]);
    target.append(node("p","最优点位于预设训练预算上限，本实验冻结后未继续增加训练轮数。CRF forward / partition / Viterbi 来自 pytorch-crf 0.7.2。","muted"));
    crfTransition(target,p.crf_transition_matrix);
    const history=(value.training_history?.selected_from_development || value.training_history)?.history || [];
    if(trainingCurve && history.length) {
      heading(target,"Frozen development training curve"); const plot=node("div","","chart training-process-chart"); target.append(plot);
      const chart=echarts.init(plot); plots.set(plot,chart);
      chart.setOption({animation:false,tooltip:{trigger:"axis"},legend:{data:["train loss","dev loss","dev F1"]},grid:{left:55,right:50,top:50,bottom:35},xAxis:{type:"category",data:history.map(r=>r.epoch)},yAxis:[{type:"value",name:"NLL"},{type:"value",name:"F1",max:1}],series:[{name:"train loss",type:"line",data:history.map(r=>r.train_loss)},{name:"dev loss",type:"line",data:history.map(r=>r.dev_loss)},{name:"dev F1",type:"line",yAxisIndex:1,data:history.map(r=>r.dev_f1)}]});
    }
    return;
  }
  table(target,["状态",...states],[["π",...p.start.map(v=>v.toFixed(6))]]);
  if(model==="hmm2-smooth") {
    heading(target,"A2 Tensor slice · P(next | prev2, prev1)");
    const controls=node("div","","tensor-controls"), left=node("select"), right=node("select"), display=node("div");
    left.setAttribute("aria-label","Tensor prev2"); right.setAttribute("aria-label","Tensor prev1");
    for(const s of states) {left.append(node("option",s));right.append(node("option",s));}
    right.value="M"; const label1=node("label","prev2 "),label2=node("label","prev1 ");label1.append(left);label2.append(right);controls.append(label1,label2); target.append(controls,display);
    function slice() {display.replaceChildren(); const a=states.indexOf(left.value),b=states.indexOf(right.value); table(display,["next",...states],[[`A2[${left.value},${right.value},:]`,...p.transition_tensor[a][b].map((v,c)=>node("span",p.legal_triples[a][b][c]?v.toFixed(6):"0 / impossible",p.legal_triples[a][b][c]?"":"masked-score"))]]);}
    left.addEventListener("change",slice);right.addEventListener("change",slice);slice();
    heading(target,"Initial state-pair transition"); table(target,["First \\ Second",...states],states.map((s,i)=>[s,...p.initial_transition[i].map(v=>v.toFixed(6))]));
  } else {heading(target,"4 × 4 Transition A");table(target,["From \\ To",...states],states.map((s,i)=>[s,...p.transition[i].map(v=>v.toFixed(6))]));}
  const details=node("details"), body=node("div");details.append(node("summary","B/M/E/S Top-K emission（前5个；完整Top-K见模型参数 Tab）"));
  table(body,["State","Char","P(char | state)"],states.flatMap(s=>p.emission_topk[s].slice(0,5).map(e=>[s,e.character,e.probability.toFixed(6)])));details.append(body);target.append(details);
}
export function renderTaskModelInfo(target, value, model) {
  target.replaceChildren(); const p=value.parameters;
  if(model==='longest'||model==='bigram') { renderModelInfo(target,value,model); return; }
  const neural=model==='bilstm-crf';
  const smoothing=p.config?.smoothing==='add_alpha'?`Add-alpha=${p.config.alpha} · UNK=${p.config.unknown_strategy}`:p.config?.smoothing??'见模型配置';
  table(target,['Model','Vocabulary','Smoothing / structure'],[[names[model],p.vocabulary_size??p.training?.vocabulary_size,neural?'Embedding 128 → BiLSTM 128 × 2 → Linear → CRF':smoothing]]);
  const link=node('a',neural?'查看结构、CRF transition 与冻结训练曲线':'查看 BMES、π / A / B 与二阶 A2 参数');
  link.href=neural?'/deep-learning#bilstm':'/sequence-labeling#hmm';target.append(link);
}
export function exportProcess(target,name) {
  const svg=target.querySelector("svg");
  if(svg) {
    const image=new Image(),url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)],{type:"image/svg+xml;charset=utf-8"}));
    image.onload=()=>{const canvas=document.createElement("canvas");canvas.width=Number(svg.getAttribute("width"));canvas.height=Number(svg.getAttribute("height"));const context=canvas.getContext("2d");context.fillStyle="#ffffff";context.fillRect(0,0,canvas.width,canvas.height);context.drawImage(image,0,0);const link=node("a");link.href=canvas.toDataURL("image/png");link.download=`${name}_word-lattice.png`;link.click();URL.revokeObjectURL(url);};image.src=url;return;
  }
  const entry=[...plots].find(([element])=>target.contains(element));
  if(entry) {const link=node("a");link.href=entry[1].getDataURL({type:"png",pixelRatio:2,backgroundColor:"#fff"});link.download=`${name}_training-curve.png`;link.click();}
}
