import { node } from './api.js';

const NS = 'http://www.w3.org/2000/svg';
export function svgNode(tag, attrs={}, text='') {
  const element = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, String(value));
  if (text) element.textContent = text;
  return element;
}
function graphic(width, height, label) {
  const svg = svgNode('svg', {viewBox:`0 0 ${width} ${height}`, role:'img', 'aria-label':label});
  svg.append(svgNode('title', {}, label)); return svg;
}

export function coordinateSpectrum(document, compact=false) {
  const values = document?.vector_values || document?.vector_preview?.map((item) => item.value) || [];
  const svg = graphic(200, compact ? 72 : 150, document?.vector_values ? '真实100维潜在坐标谱（已L2），颜色表示正负，明暗表示幅度；不是投影' : '真实稀疏向量的最高权重坐标节选');
  if (!values.length) return svg;
  const max = Math.max(...values.map(Math.abs), 1e-12);
  if (document.vector_values && !compact) {
    values.forEach((value, index) => {
      const cell = svgNode('rect', {x:10+(index%10)*18, y:5+Math.floor(index/10)*13, width:15, height:10, rx:2, fill:value<0?'#8b68b5':'#087f8c', opacity:.12+.88*Math.abs(value)/max});
      cell.append(svgNode('title',{},`d${index+1}: ${value.toFixed(6)}`)); svg.append(cell);
    });
    svg.append(svgNode('text',{x:10,y:148,'font-size':10,fill:'#71849a'},'d1 → d100 · 正 / 负坐标'));
  } else {
    svg.append(svgNode('line',{x1:8,y1:36,x2:192,y2:36,stroke:'#dce8ef'}));
    values.forEach((value,index) => {
      const w=180/values.length, h=Math.abs(value)/max*30;
      const bar=svgNode('rect',{x:10+index*w,y:value<0?36:36-h,width:Math.max(1,w-3),height:h,rx:1,fill:value<0?'#8b68b5':'#087f8c'});
      bar.append(svgNode('title',{},`${document.vector_values ? index+1 : document.vector_preview[index].dimension}: ${value.toFixed(6)}`)); svg.append(bar);
    });
  }
  return svg;
}

export function normGlyph(document) {
  const svg=graphic(180,150,'归一化文档向量的真实坐标谱；圆环表示L2范数，角度不是空间位置');
  const values=document.vector_values || document.vector_preview.map((item)=>item.value);
  const max=Math.max(...values.map(Math.abs),1e-12);
  svg.append(svgNode('circle',{cx:90,cy:70,r:53,fill:'#f0f9f8',stroke:'#b7dfe0','stroke-width':1}));
  values.forEach((value,index)=>{
    const angle=index/values.length*Math.PI*2-Math.PI/2, r=33+Math.abs(value)/max*20;
    svg.append(svgNode('line',{x1:90+Math.cos(angle)*33,y1:70+Math.sin(angle)*33,x2:90+Math.cos(angle)*r,y2:70+Math.sin(angle)*r,stroke:value<0?'#8b68b5':'#087f8c','stroke-width':values.length>20?2:4}));
  });
  svg.append(svgNode('text',{x:90,y:68,'text-anchor':'middle',fill:'#173b52','font-size':23,'font-weight':700},document.vector_norm.toFixed(3)),svgNode('text',{x:90,y:87,'text-anchor':'middle',fill:'#71849a','font-size':11},'‖document‖₂'),svgNode('text',{x:90,y:144,'text-anchor':'middle',fill:'#087f8c','font-size':10},'归一化坐标谱 · 非投影'));
  return svg;
}

export function lexicalSpectrum(terms) {
  const svg=graphic(200,45,'原TF-IDF词空间的最高词权重谱，横向顺序为权重排名');
  const max=Math.max(...terms.map(item=>item.weight),1e-12), width=180/Math.max(terms.length,1);
  terms.forEach((item,index)=>{
    const height=item.weight/max*35, bar=svgNode('rect',{x:10+index*width,y:40-height,width:width-3,height,rx:1,fill:'#b88b40'});
    bar.append(svgNode('title',{},`${item.term}: ${item.weight.toFixed(6)}`));svg.append(bar);
  });return svg;
}

export function projectionGlyph(points, assignments, centers, colors, selectedId) {
  const svg=graphic(190,150,'本Run真实二维投影；显示投影不参与聚类');
  if (!points?.length) return svg;
  const minX=Math.min(...points.map(p=>p.x)), maxX=Math.max(...points.map(p=>p.x)), minY=Math.min(...points.map(p=>p.y)), maxY=Math.max(...points.map(p=>p.y));
  const x=v=>12+(v-minX)/(maxX-minX||1)*166, y=v=>132-(v-minY)/(maxY-minY||1)*120;
  const labels=new Map(assignments.map(item=>[item.id,item.cluster_id]));
  const selected=points.find(point=>point.id===selectedId), assigned=centers?.find(center=>center.cluster_id===labels.get(selectedId));
  if(selected&&assigned)svg.append(svgNode('line',{x1:x(selected.x),y1:y(selected.y),x2:x(assigned.projection2d[0]),y2:y(assigned.projection2d[1]),stroke:'#087f8c','stroke-width':2,pathLength:1,class:'pipeline-assignment-link'}));
  for (const point of points) svg.append(svgNode('circle',{cx:x(point.x),cy:y(point.y),r:point.id===selectedId?5:1.5,fill:colors[(labels.get(point.id)??point.cluster_id??0)%colors.length],opacity:point.id===selectedId?1:.35,class:point.id===selectedId?'pipeline-selected-document':''}));
  for(const center of centers||[]) svg.append(svgNode('path',{d:`M ${x(center.projection2d[0])} ${y(center.projection2d[1])-5} l 5 5 l -5 5 l -5 -5 Z`,fill:center===assigned?'#087f8c':'#173b52',stroke:center===assigned?'#fff':'none','stroke-width':1.5,class:center===assigned?'pipeline-assigned-centroid':''}));
  return svg;
}

export function semanticCluster(cluster, center, selected, colors, chooseDocument, chooseCenter) {
  const svg=graphic(580,440,'Semantic cluster summary：示意布局非数学投影；半径编码相对质心距离，标签为真实距离');
  svg.setAttribute('viewBox','30 25 580 440');
  const color=colors[cluster.cluster_id%colors.length], cx=320, cy=225;
  svg.append(svgNode('circle',{cx,cy,r:190,fill:'none',stroke:'#e3ebf0','stroke-dasharray':'3 7'}),svgNode('circle',{cx,cy,r:140,fill:'none',stroke:'#d8e9ec'}));
  const docs=[...cluster.representatives.map((doc,index)=>({...doc,role:`代表 ${index+1}`,boundary:false})),...cluster.boundary_documents.map((doc,index)=>({...doc,role:`边界 ${index+1}`,boundary:true}))];
  if(selected?.cluster_id===cluster.cluster_id && !docs.some(doc=>doc.id===selected.id)) docs.push({...selected,role:'当前文档',boundary:false});
  const distances=docs.map(doc=>doc.distance), min=Math.min(...distances), max=Math.max(...distances);
  docs.forEach((doc,index)=>{
    const angle=(index/docs.length)*Math.PI*2-Math.PI/2, r=155+(doc.distance-min)/(max-min||1)*35;
    const x=cx+Math.cos(angle)*r, y=cy+Math.sin(angle)*r;
    const isSelected=doc.id===selected?.id;
    const linkLength=Math.hypot(x-cx,y-cy);
    const drawSelectedLink=isSelected&&!doc.boundary;
    const link=svgNode('line',{x1:cx,y1:cy,x2:x,y2:y,stroke:isSelected?color:doc.boundary?'#8092a8':color,'stroke-width':isSelected?3:doc.boundary?2:2.5,'stroke-opacity':isSelected?1:.75,'stroke-dasharray':doc.boundary?'7 5':drawSelectedLink?`${linkLength} ${linkLength}`:'none','stroke-dashoffset':drawSelectedLink?linkLength:0,'class':`semantic-distance-link${isSelected?' is-selected':''}${doc.boundary?' is-boundary':''}`,'data-distance':doc.distance,'style':`--schematic-link-length:${linkLength}px`});
    svg.append(link);
    const foreign=svgNode('foreignObject',{x:x-60,y:y-31,width:120,height:65});
    const button=node('button','',`semantic-document ${doc.boundary?'is-boundary':''} ${isSelected?'is-selected':''}`); button.type='button';button.dataset.documentId=doc.id;button.setAttribute('aria-label',`${doc.role} ${doc.id} 距离 ${doc.distance}`);
    button.append(node('strong',doc.role),node('span',doc.snippet.slice(0,15)),node('small',doc.distance.toFixed(4)));
    button.addEventListener('click',()=>chooseDocument(doc.id));foreign.append(button);svg.append(foreign);
  });
  cluster.top_terms.slice(0,8).forEach((item,index)=>{
    const angle=index/8*Math.PI*2-Math.PI/2, r=65;
    svg.append(svgNode('text',{x:cx+Math.cos(angle)*r,y:cy+Math.sin(angle)*r+4,'text-anchor':'middle',fill:color,'font-size':index<3?14:12,'font-weight':index<3?700:500},item.term));
  });
  const hub=svgNode('g',{'class':'semantic-hub',role:'button',tabindex:0,'aria-label':`查看质心 C${cluster.cluster_id}`});
  hub.append(svgNode('circle',{cx,cy,r:47,fill:'#fff',stroke:color,'stroke-width':3}),svgNode('circle',{cx,cy,r:39,fill:'#edf8f8'}),svgNode('text',{x:cx,y:cy-6,'text-anchor':'middle',fill:'#173b52','font-size':26,'font-weight':800},`C${cluster.cluster_id}`),svgNode('text',{x:cx,y:cy+15,'text-anchor':'middle',fill:color,'font-size':14},`${cluster.size} 篇`));
  hub.addEventListener('click',()=>chooseCenter(cluster.cluster_id));hub.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();chooseCenter(cluster.cluster_id);}});svg.append(hub);
  svg.append(svgNode('text',{x:40,y:458,fill:'#71849a','font-size':11},`${center?.dimension || '—'}D 真实空间距离 · 实线代表 / 虚线边界 · 半径仅在当前展示文档中作相对编码 · 角度为示意`));
  return svg;
}
