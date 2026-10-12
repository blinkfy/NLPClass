import { node } from './api.js';

// Playback reveals verified operands and partial sums; it never fits a model.
export function calculationView(doc, kind, frame, number) {
  const box=node('section','','calculation-player'), trace=doc.calculation;
  if(!trace){box.append(node('p','此 Run 尚无已验证的计算中间量；仅显示冻结结果。'));return box;}
  const full=frame===null, shown=limit=>full||frame>=limit;
  const head=node('div','','calculation-heading');
  const titles={document:'原文 → 冻结分词结果',tokenize:'token → unigram / bigram → 词频', 'lexical-space':'词频 × IDF → TF-IDF → L2', 'latent-space':'TF-IDF × 冻结 SVD 基底 → 潜在坐标',normalize:'潜在坐标 → L2 单位向量',cluster:'文档向量 → 欧氏距离 → 最近质心'};
  if(!trace.lsa)titles.normalize='TF-IDF 单位向量 → K-means 输入';
  head.append(node('small','CALCULATION · VERIFIED FROZEN OPERANDS'),node('h4',titles[kind]));box.append(head);
  const row=(label,operations)=>{
    const line=node('div','','calculation-row');line.append(node('strong',label,'calculation-label'));
    operations.forEach(([text,limit],index)=>{
      if(index)line.append(node('span','→','calculation-arrow'));
      const cell=node('span',text,`calculation-operand ${shown(limit)?'is-revealed':''} ${frame===limit?'is-current':''}`);cell.style.setProperty('--operand-delay',`${index*45}ms`);line.append(cell);
    });box.append(line);
  };
  const note=text=>box.append(node('p',`${text} 显示数值经过舍入，校验使用全精度。`,'calculation-note'));
  const result=(text,visible=true)=>box.append(node('div',text,`calculation-result ${visible?'is-revealed':''}`));
  if(kind==='document'){
    row('原文',[[doc.snippet.slice(0,96),0]]);
    row('保留 token',[[doc.tokens.slice(0,14).join(' / '),1]]);
    result(`${doc.token_count} 个保留 token · HMM-1 Smooth → 词长 / 字符模式过滤`,shown(2));
    note('这里读取完整冻结 token 缓存；未保存 HMM 的逐字 Viterbi 路径，故不模拟未记录的解码概率。');
  }else if(kind==='tokenize'){
    const pairs=trace.config.ngram_range[1]>1;
    row('词项构建',[[doc.tokens.slice(0,4).join(' / '),0],[pairs?doc.tokens.slice(0,3).map((token,i)=>`${token} ${doc.tokens[i+1]}`).join(' / '):'仅保留单词 unigram',1]]);
    trace.tfidf.rows.forEach(term=>row(term.term,[[`全文出现 ${term.count} 次`,2],[`${trace.document_count} 篇中 ${term.df} 篇包含`,3]]));
    note(`按冻结词表保留 ${trace.tfidf.nonzero_terms} 个非零词项；DF 来自全部文档的无标签计数。计数使用完整 token，不使用页面节选。`);
  }else if(kind==='lexical-space'){
    trace.tfidf.rows.forEach(term=>row(term.term,[[`count ${term.count}`,0],[trace.config.sublinear_tf?`1 + ln(${term.count}) = ${number(term.tf,4)}`:`TF ${number(term.tf,4)}`,1],[`IDF ${number(term.idf,4)}`,2],[`TF × IDF = ${number(term.raw_weight,4)}`,3],[`÷ ${number(trace.tfidf.norm,4)} = ${number(term.weight,6)}`,4]]));
    result(`‖q‖₂ = √Σ (TF × IDF)² = ${number(trace.tfidf.norm,6)}；x = q / ‖q‖₂`,shown(4));
    const term=trace.tfidf.rows[0], smooth=trace.config.smooth_idf?1:0;
    note(`以「${term.term}」为例：IDF = ln((${trace.document_count} + ${smooth}) / (${term.df} + ${smooth})) + 1 = ${number(term.idf,6)}。L2 分母包含全部 ${trace.tfidf.nonzero_terms} 个非零词项，非仅上面五项。`);
  }else if(kind==='latent-space'){
    const lsa=trace.lsa;
    lsa.rows.forEach((term,i)=>row(term.term,[[`x = ${number(term.weight,6)}`,i+1],[`V₁ = ${number(term.loading,6)}`,i+1],[`贡献 ${number(term.product,6)}`,i+1]]));
    row(`其余 ${lsa.remaining_terms} 项`,[[`贡献之和 ${number(lsa.remaining_sum,6)}`,6]]);
    const sum=lsa.rows.reduce((total,term,i)=>total+(shown(i+1)?term.product:0),0)+(shown(6)?lsa.remaining_sum:0);
    result(`z₁ = Σ xⱼ V₁ⱼ · 当前累加 ${number(sum,6)}${shown(6)?` = 完整坐标 ${number(lsa.raw_value,6)}`:''}`);
    note('演示第一个潜在维度的点积：显示绝对贡献最大的五项，并显式加上其余项之和。100 个维度均使用同一冻结基底计算；正负贡献可能相互抵消。');
  }else if(kind==='normalize'&&trace.lsa){
    const lsa=trace.lsa;
    lsa.raw_preview.forEach((value,i)=>row(`z${i+1}`,[[number(value,6),0],[`平方 ${number(value*value,6)}`,i+1],[`÷ ${number(lsa.norm,6)} = ${number(lsa.normalized_preview[i],6)}`,5]]));
    row('其余 97 维',[[`平方和 ${number(lsa.remaining_squared,6)}`,4]]);
    const squared=lsa.raw_preview.reduce((total,value,i)=>total+(shown(i+1)?value*value:0),0)+(shown(4)?lsa.remaining_squared:0);
    result(`Σ z² · 当前 ${number(squared,6)}${shown(4)?` → √${number(squared,6)} = ${number(lsa.norm,6)}`:''}`);
    note(`完整100维除以同一个范数，得到聚类输入 u；验证 ‖u‖₂ = ${number(doc.vector_norm,6)}。这里只演示运算顺序，不把图形位置当作潜在坐标。`);
  }else if(kind==='normalize'){
    result(`TF-IDF 已完成 L2：‖x‖₂ = ${number(doc.vector_norm,6)}，直接作为 K-means 输入。`);
    note('R0/R1 不经过 LSA；没有额外的潜在空间归一化步骤。');
  }else if(kind==='cluster'){
    const values=doc.vector_values||[];
    for(const center of doc.calculationCenters||[]){
      const distance=doc.centroid_distances.find(item=>item.cluster_id===center.cluster_id).distance;
      const terms=center.vector_preview.slice(0,3).map(item=>{
        const value=values.length?values[item.dimension-1]:doc.vector_preview.find(entry=>entry.dimension===item.dimension)?.value;
        return value===undefined?null:{value,center:item.value,dimension:item.dimension,part:(value-item.value)**2};
      }).filter(Boolean);
      const partial=terms.reduce((total,item)=>total+item.part,0), remainder=Math.max(0,distance*distance-partial);
      row(`C${center.cluster_id}`,[[terms.map(item=>`(d${item.dimension}: ${number(item.value,3)} − ${number(item.center,3)})²`).join(' + ')||'全词空间坐标差',1],[`显示项 ${number(partial,6)} + 其余项 ${number(remainder,6)}`,2],[`√${number(distance*distance,6)} = ${number(distance,6)}`,3]]);
    }
    result(`argmin 距离 = C${doc.nearest_cluster} → assigned cluster = ${doc.cluster_id}`,shown(3));
    note(`距离使用真实 ${doc.input_dimension}D 聚类空间。显示项之外的平方差被保留在「其余项」中；没有使用2D/3D投影距离，也没有重放未保存的K-means训练迭代。`);
  }
  return box;
}

export const calculationFrames={document:2,tokenize:3,'lexical-space':4,'latent-space':6,normalize:5,cluster:3};
