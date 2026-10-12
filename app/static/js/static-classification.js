import {compute as segmentation} from './static-compute.js';

const files = new Map();
const vocabularies = new WeakMap();
async function load(path) {
  if (!files.has(path)) files.set(path, fetch(new URL(path, document.baseURI)).then(response => {
    if (!response.ok) throw new Error('分类模型静态资源读取失败');
    return response.json();
  }));
  return files.get(path);
}
function inRanges(code, ranges) {
  let low = 0, high = ranges.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1, [start,end] = ranges[mid];
    if (code < start) high = mid-1;
    else if (code > end) low = mid+1;
    else return true;
  }
  return false;
}
function normalize(text, rules, form = 'NFKC', lower = true) {
  // Unicode normalization is stable for assigned characters. Leave characters
  // unknown to the frozen Python Unicode version unchanged, including case.
  let block = '', result = '';
  const flush = () => { const value = block.normalize(form); result += lower ? value.toLowerCase() : value; block = ''; };
  for (const char of text) {
    if (inRanges(char.codePointAt(0),rules.assigned)) block += char;
    else { flush(); result += char; }
  }
  flush(); return result;
}
function compare(a,b) {
  const x = Array.from(a, c => c.codePointAt(0)), y = Array.from(b,c => c.codePointAt(0));
  for (let i=0;i<Math.min(x.length,y.length);i++) if (x[i] !== y[i]) return x[i]-y[i];
  return x.length-y.length;
}
async function features(model, text, rules) {
  const normalized = normalize(text,rules,model.preprocessing?.unicode_normalization,
                               model.preprocessing?.lowercase ?? true);
  if (model.model_id === 'C2') {
    const values = []; let previous = null;
    for (const char of normalized) {
      if (inRanges(char.codePointAt(0),rules.content)) {
        if (previous !== null) values.push(previous + char);
        previous = char;
      } else previous = null;
    }
    return {values, tokenCount:values.length};
  }
  const tokens = await tokenizeWords(text,model.preprocessing,model.tokenizer_run_id,rules);
  const values = [], [min,max] = model.tfidf_config.ngram_range;
  for (let n=min;n<=max;n++) for (let i=0;i+n<=tokens.length;i++) values.push(tokens.slice(i,i+n).join(' '));
  return {values, tokenCount:tokens.length};
}
export async function tokenizeWords(text,preprocessing,tokenizerRunId,rules) {
  const normalized=normalize(text,rules,preprocessing.unicode_normalization,preprocessing.lowercase);
  const tokens = [], stopwords = new Set(preprocessing.stopwords);
  const pattern = new RegExp(preprocessing.token_pattern, 'gu');
  for (const [chunk] of normalized.matchAll(pattern)) {
    const words = /^[\x00-\x7f]+$/.test(chunk) ? [chunk] :
      (await segmentation('/api/v1/analyze',{}, {text:chunk,tasks:['segmentation'],
        segmentation_model:'hmm1-smooth',run_id:tokenizerRunId})).segmentation.tokens;
    tokens.push(...words.filter(word => Array.from(word).length >= preprocessing.min_token_length && !stopwords.has(word)));
  }
  return tokens;
}
function explain(model, candidates) {
  if (!vocabularies.has(model)) vocabularies.set(model,new Map(model.terms.map((term,i) => [term,i])));
  const vocabulary = vocabularies.get(model), counts = new Map();
  for (const term of candidates.values) if (vocabulary.has(term)) {
    const column = vocabulary.get(term); counts.set(column,(counts.get(column)||0)+1);
  }
  const vector = [...counts].sort(([a],[b]) => a-b).map(([column,count]) =>
    ({column,term:model.terms[column],value:(1+Math.log(count))*model.idf[column]}));
  const norm = Math.sqrt(vector.reduce((sum,item) => sum+item.value*item.value,0));
  vector.forEach(item => item.value /= norm);
  const scores = model.intercept.map((bias,i) => bias+vector.reduce((sum,f) => sum+f.value*model.weights[i][f.column],0));
  const ranking = model.labels.map((_,i) => i).sort((a,b) => scores[b]-scores[a] || compare(model.labels[a],model.labels[b]));
  const [best,second] = ranking;
  const isNB = model.model_id === 'C0', max = Math.max(...scores);
  const exponentials = scores.map(score => Math.exp(score-max)), total = exponentials.reduce((a,b) => a+b,0);
  const posterior = exponentials.map(value => value/total);
  const contributions = vector.map(item => {
    const weight = model.weights[best][item.column], difference = weight-model.weights[second][item.column];
    return {term:item.term,tfidf:item.value,weight,direct_contribution:item.value*weight,
      runner_up_contrast:item.value*difference,coefficient_difference:difference,
      contribution:item.value*(isNB?difference:weight)};
  });
  const direct = contributions.reduce((sum,f) => sum+f.direct_contribution,0);
  const contrast = contributions.reduce((sum,f) => sum+f.runner_up_contrast,0);
  contributions.sort((a,b) => b.contribution-a.contribution || compare(a.term,b.term));
  const result = {run_id:model.run_id,model_id:model.model_id,prediction:model.labels[best],runner_up:model.labels[second],
    score_type:isNB?'posterior':'decision_score',model_score:isNB?posterior[best]:scores[best],
    scores:model.labels.map((label,i) => ({label,value:isNB?posterior[i]:scores[i],joint_log_score:isNB?scores[i]:null})),
    score_gap:scores[best]-scores[second],zero_vector:vector.length === 0,
    contribution_method:isNB?'x_j * (log_theta_pred - log_theta_runner_up)':'x_j * w_pred,j',
    intercept:model.intercept[best],feature_sum:direct,reconstructed_score:model.intercept[best]+direct,
    contrast_intercept:model.intercept[best]-model.intercept[second],contrast_feature_sum:contrast,
    top_contributing_features:contributions.slice(0,12).map(({coefficient_difference,...feature}) => feature),
    token_count:candidates.tokenCount,known_feature_count:vector.length,
    notice:vector.length ? null : '零向量：未命中词表，预测由先验/截距决定',execution:'browser'};
  if (model.model_id === 'C2') {
    const unique = new Set(candidates.values), hitTypes = [...unique].filter(term => vocabulary.has(term)).length;
    const hits = candidates.values.filter(term => vocabulary.has(term)).length;
    result.feature_kind = 'character_bigram';
    if (!vector.length) result.notice = '零向量：未命中选择后的词表，预测由截距决定';
    result.top_relative_features = [...contributions].sort((a,b) => b.runner_up_contrast-a.runner_up_contrast || compare(a.term,b.term))
      .slice(0,12).map(f => ({term:f.term,tfidf:f.tfidf,contribution:f.runner_up_contrast,coefficient_difference:f.coefficient_difference}));
    result.coverage = {candidate_occurrences:candidates.values.length,hit_occurrences:hits,
      occurrence_ratio:candidates.values.length?hits/candidates.values.length:null,candidate_unique_types:unique.size,
      hit_unique_types:hitTypes,unique_type_ratio:unique.size?hitTypes/unique.size:null};
  }
  return result;
}
export async function classify(path, body, manifest) {
  if (!['/api/v1/experiments/3/predict','/api/v1/experiments/3/stage2/predict'].includes(path)) return undefined;
  const modelId = path.includes('/stage2/') ? 'C2' : body?.model_id || 'C0';
  const descriptor = manifest.classifiers?.runs?.[body?.run_id || 'latest']?.[modelId];
  if (!descriptor) throw new Error('当前冻结 Run 未提供此分类器的浏览器推理参数');
  if (typeof body.text !== 'string' || !body.text.trim() || Array.from(body.text).length > 5000)
    throw new Error('请输入1–5000字新闻文本');
  const [model,rules] = await Promise.all([load(descriptor.path),load(manifest.classifiers.unicode_path)]);
  const start = performance.now(), result = explain(model,await features(model,body.text,rules));
  return {...result,runtime_seconds:(performance.now()-start)/1000};
}
