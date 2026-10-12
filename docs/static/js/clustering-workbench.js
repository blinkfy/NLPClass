import { node } from './api.js';
import { calculationView, calculationFrames } from './clustering-calculation.js?v=20261003-calc3';
import { coordinateSpectrum, normGlyph, projectionGlyph, semanticCluster, lexicalSpectrum } from './clustering-visual-language.js?v=20261003-calc3';

export class ClusteringWorkbench {
  constructor(context) {
    Object.assign(this, context);
    this.calculationPanel=node('div','','clustering-calculation-panel');
    this.$('clustering-flow').after(this.calculationPanel);
    for (const id of ['clustering-sample-document', 'clustering-document-select']) this.$(id).addEventListener('change', (event) => this.selectDocument(event.target.value));
    this.$('clustering-flow-prev').addEventListener('click', () => this.moveFlowStep(-1));
    this.$('clustering-flow-next').addEventListener('click', () => this.moveFlowStep(1));
    this.$('clustering-flow-play').addEventListener('click', () => this.toggleFlowPlayback());
    for(const tab of document.querySelectorAll('[data-tab]'))tab.addEventListener('click',()=>{if(tab.dataset.tab!=='data')this.stopFlowPlayback();});
    document.addEventListener('visibilitychange',()=>{if(document.hidden)this.stopFlowPlayback();});
    this.motionPreference=window.matchMedia('(prefers-reduced-motion: reduce)');
    this.motionPreference.addEventListener?.('change',event=>{if(event.matches)this.stopFlowPlayback();else this.updateFlowControls();});
    this.updateFlowControls();
  }

  terms(items, limit=8) {
    const box = node('div', '', 'clustering-terms');
    for (const term of (items || []).slice(0, limit)) box.append(node('span', term.term));
    return box;
  }

  bars(items, valueKey='weight', labelKey='term', onClick=null) {
    const box = node('div', '', 'clustering-weight-bars');
    const max = Math.max(...items.map((item) => Number(item[valueKey])), 1e-12);
    for (const item of items) {
      const row = node(onClick ? 'button' : 'div', '', 'clustering-weight-row');
      if (onClick) { row.type = 'button'; row.addEventListener('click', () => onClick(item)); }
      row.append(node('span', String(item[labelKey])));
      const rail = node('i', '', 'clustering-weight-rail'), fill = node('i');
      fill.style.width = `${Number(item[valueKey]) / max * 100}%`; rail.append(fill);
      row.append(rail, node('strong', this.number(item[valueKey], 4))); box.append(row);
    }
    return box;
  }

  coordinates(values) {
    const box = node('div', '', 'clustering-coordinate-chips');
    for (const item of values || []) box.append(node('span', `d${item.dimension} ${this.number(item.value, 4)}`, item.value < 0 ? 'is-negative' : ''));
    return box;
  }

  facts(rows) {
    const box = node('dl', '', 'clustering-inspector-facts');
    for (const [label, value] of rows) box.append(node('dt', label), node('dd', String(value)));
    return box;
  }

  composition(cluster, compact=false) {
    return Object.entries(cluster.category_counts || {}).map(([category, count]) => `${category} ${count}${compact ? '' : ` (${this.number(count / cluster.size * 100, 1)}%)`}`).join(' · ');
  }

  renderOptions() {
    const assignments = this.state.result?.assignments || [];
    for (const id of ['clustering-sample-document', 'clustering-document-select']) {
      const select = this.$(id); select.replaceChildren();
      for (const item of assignments) select.append(new Option(`${item.id} · ${item.snippet.slice(0, 22)}`, item.id));
      select.disabled = !assignments.length;
      select.value = this.state.selectedId || assignments[0]?.id || '';
    }
  }

  renderFlow() {
    const doc = this.state.document, flow = this.$('clustering-flow'), example = this.$('clustering-example');
    flow.replaceChildren(); flow.classList.remove('is-flow-focused'); example.replaceChildren();
    this.calculationPanel.replaceChildren();
    const flowKey=`${this.state.run}:${doc?.id || this.state.selectedId || ''}`;
    if(flowKey!==this.flowDocumentKey){this.stopFlowPlayback();this.flowDocumentKey=flowKey;this.flowStep=null;this.flowFrame=null;}
    if (!doc) { this.stopFlowPlayback();this.flowStep=null;this.$('clustering-sample-id').textContent = this.state.selectedId || ''; flow.append(node('p', this.state.documentError || '正在读取真实样本文档的词与向量…', 'muted')); this.updateFlowControls(); return; }
    const lsa = !!this.state.overview?.config.lsa;
    const stage = (title, label, visual, footer) => {
      const box=node('article','','representation-stage');
      box.dataset.flowStage=String(flow.children.length);
      box.dataset.flowKind=label.split(' · ')[1].toLowerCase().replaceAll(' ','-');
      box.append(node('small',label,'stage-kicker'),node('h4',title),visual,node('small',footer,'stage-footer'));
      flow.append(box);return box;
    };
    const news=node('div','','pipeline-news-sheet');news.append(node('span','新闻原文','sheet-label'),node('p',doc.snippet.slice(0,140)));
    stage('原始新闻','01 · DOCUMENT',news,'真实节选 · id + text');
    stage('中文分词','02 · TOKENIZE',this.terms(doc.tokens.map(term=>({term})),14),`${this.integer(doc.token_count)} tokens · HMM-1 Smooth`);
    const tfidf=node('div','','pipeline-tfidf');tfidf.append(this.bars(doc.top_terms.slice(0,6)),lexicalSpectrum(doc.top_terms));
    stage('TF-IDF','03 · LEXICAL SPACE',tfidf,`${this.integer(doc.tfidf_dimension)} 词维 · 真实词权重`);
    if(lsa) stage(`${this.integer(doc.input_dimension)}D LSA`,'04 · LATENT SPACE',coordinateSpectrum(doc),`${this.number(this.state.overview.representation.lsa.explained_variance_sum*100,2)}% 解释方差 · 坐标已 L2`);
    stage('L2 vector',lsa?'05 · NORMALIZE':'04 · NORMALIZE',normGlyph(doc),`${this.integer(doc.input_dimension)}D · ${doc.vector_values?'完整坐标谱':'坐标谱节选'}，非投影`);
    const mini=projectionGlyph(this.state.overview.projection.points,this.state.result.assignments,this.state.inspection.centroids,this.colors,doc.id);
    const ending=stage('K-means',lsa?'06 · CLUSTER':'05 · CLUSTER',mini,`当前 K=${doc.k} · 2D 显示投影`);ending.classList.add('pipeline-clustering');
    flow.style.setProperty('--flow-count',flow.children.length);
    for(const box of this.flowStages()){
      const pieces=box.querySelectorAll('.clustering-terms span,.clustering-weight-row,svg rect,svg line');
      pieces.forEach((piece,index)=>piece.style.setProperty('--reveal-delay',`${Math.min(index* (pieces.length>20?10:90),1100)}ms`));
    }
    this.flowDescriptions=[
      `读取当前新闻节选 · ${doc.id}`,
      `确定性分词得到 ${this.integer(doc.token_count)} 个 token；展示其中前 14 个。`,
      `词项映射到 ${this.integer(doc.tfidf_dimension)} 维 TF-IDF；条长对应真实权重。`,
      ...(lsa?[`压缩到 ${doc.input_dimension}D 潜在坐标；逐格显示真实数值的正负与幅度。`]:[]),
      `实际文档向量的 L2 范数为 ${this.number(doc.vector_norm,6)}。`,
      `真实 ${doc.input_dimension}D 距离 ${this.number(doc.distance,4)} → 分配簇 C${doc.cluster_id}；图中连线仅为投影。`,
    ];
    this.updateFlowControls();
    this.$('clustering-sample-id').textContent=doc.id;
    example.append(node('p',`当前文档 ${doc.id} · 分配簇 ${doc.cluster_id} · ${doc.input_dimension}D ${doc.space} · 向量范数 ${this.number(doc.vector_norm,6)}`),this.coordinates(doc.vector_preview),this.bars(doc.top_terms));
    this.renderEvolution();
    this.renderCalculation();
  }

  renderCalculation() {
    if(!this.state.document)return;
    const kind=this.flowStages()[this.flowStep??0]?.dataset.flowKind;
    this.calculationPanel.replaceChildren(calculationView({...this.state.document,calculationCenters:this.state.inspection?.centroids},kind,this.flowFrame??null,this.number));
  }

  flowStages() { return [...this.$('clustering-flow').querySelectorAll('.representation-stage')]; }

  updateFlowControls() {
    const stages=this.flowStages(), step=this.flowStep, hasDoc=!!this.state.document&&stages.length>0;
    this.$('clustering-flow-prev').disabled=!hasDoc||step===null||step<=0;
    this.$('clustering-flow-next').disabled=!hasDoc||step===stages.length-1;
    const play=this.$('clustering-flow-play'),reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    play.disabled=!hasDoc||reducedMotion;play.setAttribute('aria-pressed',String(!!this.flowPlaying));
    play.title=reducedMotion?'系统启用了减少动态效果；请使用单步控制查看流程。':'';
    play.textContent=this.flowPlaying?'暂停计算演示':step===stages.length-1?'重播计算过程':'播放计算过程';
    const status=this.$('clustering-flow-status');
    status.textContent=!hasDoc?(this.state.selectedId?`正在读取文档 ${this.state.selectedId} 的真实表示…`:'等待选择文档'):reducedMotion&&step===null?'减少动态效果已开启；使用上一步或下一步逐站查看。':step===null?'播放当前文档的冻结表示路径；不重新训练。':`${step+1}/${stages.length} · ${stages[step].querySelector('h4')?.textContent||'表示阶段'} — ${this.flowDescriptions?.[step]||''}`;
  }

  setFlowStep(index, frame=null) {
    const stages=this.flowStages();if(!stages.length)return;
    this.flowStep=Math.max(0,Math.min(stages.length-1,index));
    this.flowFrame=frame;
    this.$('clustering-flow').classList.add('is-flow-focused');
    stages.forEach((stage,position)=>{
      stage.classList.toggle('is-flow-active',position===this.flowStep);
      stage.classList.toggle('is-flow-complete',position<this.flowStep);
      stage.classList.toggle('is-flow-pending',position>this.flowStep);
      if(position===this.flowStep)stage.setAttribute('aria-current','step');else stage.removeAttribute('aria-current');
    });
    this.updateFlowControls();
    this.renderCalculation();
  }

  stopFlowPlayback() {
    if(this.flowTimer)window.clearTimeout(this.flowTimer);
    this.flowTimer=null;this.flowPlaying=false;
    if(this.$('clustering-flow-play'))this.updateFlowControls();
  }

  moveFlowStep(delta) {
    this.stopFlowPlayback();const stages=this.flowStages();if(!stages.length)return;
    this.setFlowStep(this.flowStep===null?(delta>0?0:stages.length-1):this.flowStep+delta);
  }

  toggleFlowPlayback() {
    if(window.matchMedia('(prefers-reduced-motion: reduce)').matches){this.updateFlowControls();return;}
    if(this.flowPlaying){this.stopFlowPlayback();return;}
    const stages=this.flowStages();if(!stages.length)return;
    if(this.flowStep===null||this.flowStep===stages.length-1)this.setFlowStep(0,0);
    else if(this.flowFrame==null){this.flowFrame=0;this.renderCalculation();}
    this.flowPlaying=true;this.updateFlowControls();
    const delay=1100;
    const advance=()=>{
      if(!this.flowPlaying)return;
      const kind=stages[this.flowStep].dataset.flowKind;
      if(this.flowFrame<calculationFrames[kind]){this.flowFrame++;this.renderCalculation();}
      else if(this.flowStep>=stages.length-1){this.stopFlowPlayback();return;}
      else this.setFlowStep(this.flowStep+1,0);
      this.flowTimer=window.setTimeout(advance,delay);
    };
    this.flowTimer=window.setTimeout(advance,delay);
  }

  renderEvolution() {
    const box=this.$('clustering-representation-evolution');box.replaceChildren();
    for(const item of this.state.summary?.items || []) {
      const doc=this.state.documentViews?.[item.run_id], step=node('button','',`representation-evolution-object ${item.run_id===this.state.run?'is-current':''}`);step.type='button';step.setAttribute('aria-pressed',String(item.run_id===this.state.run));
      step.append(node('small',`${item.representation_id}${item.representation_id==='R2'?' · PRIMARY':''}`),node('strong',this.integer(item.representation.dimension)),node('span',item.representation_id==='R0'?'sparse TF-IDF':item.representation_id==='R1'?'richer lexical features':'compact latent space'));
      if(doc)step.append(coordinateSpectrum(doc,item.representation_id!=='R2'));else step.append(node('p','正在读取同一文档的真实坐标…','muted'));
      step.append(node('small',item.representation_id==='R2'?'100D 实际坐标谱 · 另有2D/3D显示投影':'最高权重稀疏坐标节选'));
      step.addEventListener('click',()=>this.selectRepresentation(item.run_id));box.append(step);
    }
  }

  documentCard(doc, label) {
    const button = node('button', '', 'clustering-representative-link'); button.type = 'button'; button.dataset.documentId = doc.id;
    button.append(node('small', label), node('strong', doc.id), node('p', doc.snippet), node('span', `真实空间距离 ${this.number(doc.distance, 4)} → 检视文档`));
    button.addEventListener('click', () => this.selectDocument(doc.id)); return button;
  }

  renderExplorer() {
    const clusters = this.state.result?.clusters || [];
    const chosen = this.state.selectedCluster ?? clusters[0]?.cluster_id;
    const list = this.$('clustering-cluster-cards'); list.replaceChildren();
    for (const cluster of clusters) {
      const tile = node('button', '', `clustering-cluster-tile${cluster.cluster_id === chosen ? ' is-current' : ''}`); tile.type = 'button';
      tile.dataset.clusterId = cluster.cluster_id; tile.setAttribute('aria-pressed', String(cluster.cluster_id === chosen));
      const head = node('div', '', 'clustering-tile-head');
      const badge = node('span', String(cluster.cluster_id), 'clustering-cluster-dot'); badge.style.background = this.colors[cluster.cluster_id % this.colors.length];
      head.append(badge, node('strong', `簇 ${cluster.cluster_id}`), node('span', `${cluster.size} 篇`));
      tile.append(head, this.terms(cluster.top_terms, 3));
      tile.addEventListener('click', () => this.selectCluster(cluster.cluster_id)); list.append(tile);
    }
    const detail = this.$('clustering-cluster-detail'); detail.replaceChildren();
    const cluster = clusters.find((item) => item.cluster_id === chosen);
    if (!cluster) return;
    const center = this.state.inspection?.centroids.find((item) => item.cluster_id === chosen);
    detail.append(node('span', 'CLUSTER · SEMANTIC PROFILE', 'clustering-workbench-eyebrow'), node('h3', `簇 ${cluster.cluster_id} · ${cluster.top_terms.slice(0, 3).map((term) => term.term).join(' / ')}`), node('p', `${cluster.size} 篇 · 标题来自当前高权重词，簇编号不是类别编号。`, 'muted'));
    const scene=node('div','','semantic-cluster-scene');scene.append(semanticCluster(cluster,center,this.state.document,this.colors,this.selectDocument,this.selectCentroid));detail.append(scene);
    const mobileTerms=this.terms(cluster.top_terms,8);mobileTerms.classList.add('semantic-mobile-terms');detail.append(mobileTerms);
    detail.append(node('p','Semantic cluster summary / schematic view · 半径仅在当前展示的代表/边界文档间相对编码，角度为示意布局；实际判定来自原聚类空间。','semantic-scene-caption'));
    const footer=node('div','','semantic-profile-footer');footer.append(node('span',`质心范数 ${this.number(center?.norm,4)}`),node('span',`${center?.dimension || '—'}D ${this.state.inspection?.space}`),node('span',`post-hoc · ${this.composition(cluster,true)}`));detail.append(footer);
    const more = node('details', '', 'clustering-details'); more.append(node('summary', '完整代表词、更多代表、边界文本与组成'));
    more.append(this.bars(cluster.top_terms), node('h4', '完整代表文本'));
    for (const doc of cluster.representatives) more.append(this.documentCard(doc, '靠近中心'));
    more.append(node('h4', '边界文本 · 距离较远'));
    for (const doc of cluster.boundary_documents) more.append(this.documentCard(doc, '边界'));
    more.append(node('h4', '完整事后组成'), node('p', this.composition(cluster)));
    if(center)more.append(node('h4',`质心 C${center.cluster_id} · 真实坐标节选`),this.coordinates(center.vector_preview));
    detail.append(more);
  }

  renderInspectors() {
    for (const id of ['clustering-document-inspector', 'clustering-analysis-inspector']) {
      const box = this.$(id); box.replaceChildren();
      box.append(node('span', 'DOCUMENT INSPECTOR', 'clustering-workbench-eyebrow'), node('h3', '文档 Inspector'));
      const doc = this.state.document;
      if (!doc) { box.append(node('p', this.state.documentError || '选择文档后，查看真实向量与各质心距离。', 'muted')); continue; }
      box.dataset.documentId = doc.id;
      box.append(node('strong', doc.id, 'clustering-document-id'), this.facts([
        ['assigned cluster', `簇 ${doc.cluster_id}`], ['gold · 仅事后', doc.gold_category],
        ['到所属中心距离', this.number(doc.distance, 6)], ['实际空间', `${doc.input_dimension}D ${doc.space}`],
      ]));
      box.append(node('h4', '该文档 Top terms · TF-IDF'), this.terms(doc.top_terms, 6), node('p', doc.snippet.slice(0,115)+'…', 'clustering-inspector-snippet'));
      const center = this.state.inspection?.centroids.find((item) => item.cluster_id === this.state.selectedCluster);
      const cluster = this.state.result?.clusters.find((item) => item.cluster_id === this.state.selectedCluster);
      if (center && cluster) {
        const profile = node('details', '', 'clustering-details clustering-inspected-centroid');
        profile.dataset.centroidId = center.cluster_id;
        profile.append(node('summary', `当前检视质心 C${center.cluster_id} · ${center.dimension}D`), this.facts([['成员', `${cluster.size} 篇`], ['中心范数', this.number(center.norm, 6)]]), this.terms(cluster.top_terms, 8), this.coordinates(center.vector_preview));
        box.append(profile);
      }
      const full = node('details', '', 'clustering-details'); full.append(node('summary', '完整节选、词权重与真实坐标'), node('p',doc.snippet), this.bars(doc.top_terms), this.coordinates(doc.vector_preview)); box.append(full);
    }
    for (const id of ['clustering-sample-document', 'clustering-document-select']) this.$(id).value = this.state.selectedId || '';
    this.renderAssignment();
  }

  renderAssignment() {
    const box=this.$('clustering-actual-assignment');box.replaceChildren();const doc=this.state.document;
    if(!doc){box.append(node('p',this.state.documentError||'选择文档，观察真实空间中的最近质心判定。','muted'));return;}
    box.dataset.documentId=doc.id;box.dataset.dimension=doc.input_dimension;box.dataset.space=doc.space;
    const distanceKey=`${this.state.run}:${this.state.k}:${doc.id}`, animateDistances=box.dataset.distanceKey!==distanceKey;box.dataset.distanceKey=distanceKey;
    const head=node('div','','assignment-source');head.append(node('small',`${doc.input_dimension}D · ACTUAL-SPACE DISTANCES`),node('strong',doc.id));
    const tracks=node('div','','assignment-distance-tracks');
    const max=Math.max(...doc.centroid_distances.map(item=>Number(item.distance)),1e-12);
    for(const [index,item] of doc.centroid_distances.entries()){
      const nearest=item.cluster_id===doc.nearest_cluster, inspected=item.cluster_id===this.state.selectedCluster;
      const track=node('button','',`assignment-distance-track${nearest?' is-nearest':''}${inspected?' is-inspected':''}`);track.type='button';track.dataset.centroidId=item.cluster_id;track.dataset.distance=item.distance;
      track.setAttribute('aria-label',`质心 C${item.cluster_id}，实际欧氏距离 ${this.number(item.distance,6)}${nearest?'，最近质心并分配到此簇':''}`);
      const rail=node('span','','assignment-distance-rail'),fill=node('i','',`assignment-distance-fill${animateDistances?' is-entering':''}`);
      fill.style.setProperty('--distance-ratio',String(Math.max(0,Math.min(1,Number(item.distance)/max))));fill.style.setProperty('--fill-delay',`${index*65}ms`);rail.append(fill);
      track.append(node('strong',`C${item.cluster_id}`),rail,node('span',this.number(item.distance,4),'assignment-distance-value'),node('small',nearest?'最近 · assigned':'实际欧氏距离'));
      track.addEventListener('click',()=>this.selectCentroid(item.cluster_id));tracks.append(track);
    }
    const verdict=node('div','','assignment-target');verdict.append(node('small','MIN DISTANCE → ASSIGNED'),node('strong',`C${doc.nearest_cluster} → 簇 ${doc.cluster_id}`),node('span','短轨道表示该文档到该质心的实际欧氏距离更小。'));
    const note=node('p','轨道按当前文档到各中心的最大距离归一化，仅用于同一文档内部比较；分配使用完整实际聚类空间，不使用低维投影。','assignment-caption');
    box.append(head,tracks,verdict,note);
  }

  renderKContext() {
    const row=this.state.overview?.sweep.find(item=>item.k===this.state.k), box=this.$('clustering-current-k');box.replaceChildren();if(!row)return;
    const selector=node('div','','k-selector');selector.setAttribute('aria-label','选择冻结 K sweep 配置');
    for(const entry of this.state.overview.sweep){const button=node('button',String(entry.k),entry.k===row.k?'is-current':'');button.type='button';button.dataset.k=entry.k;button.setAttribute('aria-pressed',String(entry.k===row.k));button.addEventListener('click',()=>this.selectK(entry.k));selector.append(button);}
    const facts=node('div','','k-selector-context');facts.append(node('strong',`K=${row.k} · ${row.k===5?'5主题参考配置':'参数扫描'}`),node('span',`silhouette ${this.number(row.silhouette,4)}`),node('span',`stability ${this.number(row.stability.mean_ari,4)}`));
    const sizes=node('div','','k-size-ribbon'), total=row.cluster_sizes.reduce((sum,size)=>sum+size,0);for(const [index,size] of row.cluster_sizes.entries()){const segment=node('span',String(size));segment.style.flex=String(size/total);segment.style.background=this.colors[index%this.colors.length];segment.title=`簇 ${index} · ${size} 篇`;sizes.append(segment);}
    const sweep=this.state.comparison?.items.find(item=>item.representation_id==='R2')?.sweep||[], reference=sweep.find(item=>item.k===5), higher=sweep.filter(item=>reference&&item.silhouette>reference.silhouette).map(item=>item.k);
    box.append(selector,facts,sizes,node('small',`不是自动最优 K。${higher.length?`R2 的 K=${higher.join('/')} silhouette 更高；`:''}K=5 因五主题参考价值与较高稳定性被保留。`));
    const representations=this.$('clustering-representation-switch');representations.replaceChildren();for(const item of this.state.summary?.items||[]){const button=node('button',`${item.representation_id} · ${this.integer(item.representation.dimension)}D`,item.run_id===this.state.run?'is-current':'');button.type='button';button.addEventListener('click',()=>this.selectRepresentation(item.run_id));representations.append(button);}
  }

  renderCenterButtons() {
    const box = this.$('clustering-space-centroids'); box.replaceChildren();
    for (const center of this.state.inspection?.centroids || []) {
      const button = node('button', `◇ C${center.cluster_id}`, center.cluster_id === this.state.selectedCluster ? 'is-current' : ''); button.type = 'button'; button.dataset.centroidId = center.cluster_id;
      button.addEventListener('click', () => this.selectCentroid(center.cluster_id)); box.append(button);
    }
    box.append(node('small', '点击质心查看中心与词空间 profile；点击文档查看分配过程。'));
  }

  renderStructureComparison() {
    const items=this.state.comparison?.items||[];
    this.$('clustering-structure-role').textContent=`K=${this.state.k}${this.state.k===5?' · 5主题参考':''}`;
    const main=this.$('clustering-structure-comparison'), middle=this.$('clustering-structure-r1');main.replaceChildren();middle.replaceChildren();
    for(const item of items){
      const card=node('article','',`experiment-story-state story-${item.representation_id}`);
      card.append(node('small',`${item.representation_id} · ${item.representation_id==='R2'?'PRIMARY':'EXPERIMENT'}`,'clustering-workbench-eyebrow'),node('h3',item.representation_id==='R0'?'高维稀疏词空间':item.representation_id==='R1'?'丰富词汇表示':'紧凑潜在语义空间'),node('strong',this.integer(item.representation.dimension),'story-dimension'),node('span',item.representation_id==='R2'?'LSA + L2 · actual clustering space':'word-level TF-IDF','story-space-label'));
      const chart=node('div','','clustering-comparison-mini');chart.id=`clustering-mini-${item.representation_id}`;chart.setAttribute('aria-label',`${item.representation_id}真实二维显示投影`);card.append(chart);
      const metrics=node('div','','story-internal-metrics');for(const [label,value] of [['silhouette',item.internal.silhouette],['stability',item.stability.mean_ari],['DB ↓',item.internal.davies_bouldin],['CH ↑',item.internal.calinski_harabasz]]){const line=node('div');line.append(node('small',label),node('strong',this.number(value,label==='DB ↓'?3:4)));metrics.append(line);}card.append(metrics);
      const external=node('div','','story-posthoc');external.append(node('small','POST-HOC · 仅类别一致性'),node('span',`ARI ${this.number(item.external.ari,4)} / NMI ${this.number(item.external.nmi,4)}`));card.append(external);
      const button=node('button',`查看 ${item.representation_id}`,'clustering-workbench-action');button.type='button';button.addEventListener('click',()=>this.selectRepresentation(item.run_id));card.append(button);
      (item.representation_id==='R1'?middle:main).append(card);
    }
    const observations=this.$('clustering-observation-cards');observations.replaceChildren();
    const r0=items.find(item=>item.representation_id==='R0'),r1=items.find(item=>item.representation_id==='R1'),r2=items.find(item=>item.representation_id==='R2');
    if(!r0||!r1||!r2)return;
    const notes=[['01 · R0','发现词空间问题',`高维、稀疏；silhouette ${this.number(r0.internal.silhouette,4)}。部分通用新闻词进入 Top terms。`],['02 · R1','词项增强，稳定性下降',`DF 过滤 + bigram；stability ${this.number(r0.stability.mean_ari,4)} → ${this.number(r1.stability.mean_ari,4)}。`],['03 · R2','LSA 改善结构与稳定性',`${r2.representation.dimension}D；silhouette ${this.number(r2.internal.silhouette,4)}，stability ${this.number(r2.stability.mean_ari,4)}。`],['04 · 局限','混合主题仍然存在',`最大簇 ${Math.max(...r2.clusters.map(item=>item.size))} 篇；LSA 解释方差 ${this.number(r2.representation.lsa.explained_variance_sum*100,2)}%。几何指标需结合语义解读。`]];
    for(const [topic,title,text] of notes){
      const step=topic.includes('R0')?'R0':topic.includes('R1')?'R1':topic.includes('R2')?'R2':'limit';
      const card=node('article','','experiment-observation-step');card.dataset.storyStep=step;card.tabIndex=0;card.setAttribute('role','button');
      card.append(node('small',topic),node('h4',title),node('p',text));
      card.addEventListener('click',()=>this.focusEvaluationStory(step));
      card.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();this.focusEvaluationStory(step);}});
      observations.append(card);
    }
    this.focusEvaluationStory('R2');
  }

  focusEvaluationStory(step) {
    const representation=step==='limit'?'R2':step;
    for(const card of this.$('clustering-structure-comparison').querySelectorAll('.experiment-story-state')){
      card.classList.toggle('is-story-focus',card.classList.contains(`story-${representation}`));
      card.classList.toggle('is-story-dimmed',!card.classList.contains(`story-${representation}`));
    }
    for(const card of this.$('clustering-structure-r1').querySelectorAll('.experiment-story-state')){
      card.classList.toggle('is-story-focus',card.classList.contains(`story-${representation}`));
      card.classList.toggle('is-story-dimmed',!card.classList.contains(`story-${representation}`));
    }
    for(const card of this.$('clustering-observation-cards').querySelectorAll('[data-story-step]')){
      const current=card.dataset.storyStep===step;card.classList.toggle('is-current',current);card.setAttribute('aria-pressed',String(current));
    }
  }
}
