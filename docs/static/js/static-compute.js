const STATES = ['B', 'M', 'E', 'S'];
const LEGAL_NEXT = { B: ['M', 'E'], M: ['M', 'E'], E: ['B', 'S'], S: ['B', 'S'] };
const LEGAL_START = new Set(['B', 'S']);
const LEGAL_END = new Set(['E', 'S']);
const modelCache = new Map();
const vocabularyCache = new WeakMap();
const dictionaryCache = new WeakMap();
const MODEL_FILES = { 'hmm1-smooth': 'models/hmm1-smooth.json', 'hmm1-mle': 'models/hmm1-mle.json', longest: 'models/longest.json' };

function parameter(params, key, fallback = '') {
  if (params instanceof URLSearchParams) return params.get(key) ?? fallback;
  return params?.[key] ?? fallback;
}

async function loadModel(name) {
  if (!modelCache.has(name)) {
    const url = new URL(MODEL_FILES[name], document.baseURI);
    modelCache.set(name, fetch(url).then(response => {
      if (!response.ok) throw new Error(`静态模型资源不可用：${name} (${response.status})`);
      return response.json();
    }).catch(error => { modelCache.delete(name); throw error; }));
  }
  return modelCache.get(name);
}

function logProbability(value) {
  return value > 0 ? Math.log(value) : -Infinity;
}

function decode(text, tags) {
  const tokens = [];
  let pending = '';
  const characters = Array.from(text);
  for (const [index, tag] of tags.entries()) {
    const character = characters[index];
    if (tag === 'S') {
      if (pending) tokens.push(pending);
      pending = '';
      tokens.push(character);
    } else if (tag === 'B') {
      if (pending) tokens.push(pending);
      pending = character;
    } else if (tag === 'M') {
      pending += character;
    } else {
      tokens.push(pending + character);
      pending = '';
    }
  }
  if (pending) tokens.push(pending);
  return tokens;
}

function viterbi(model, chars) {
  let vocab = vocabularyCache.get(model);
  if (!vocab) {
    vocab = new Map(model.vocabulary.map((character, index) => [character, index]));
    vocabularyCache.set(model, vocab);
  }
  const scores = [], backpointer = [];
  for (let t = 0; t < chars.length; t++) {
    const emissionIndex = vocab.get(chars[t]) ?? 0;
    const row = Array(4).fill(-Infinity), previous = Array(4).fill(-1);
    if (t === 0) {
      for (let s = 0; s < 4; s++) {
        if (LEGAL_START.has(STATES[s])) row[s] = logProbability(model.start[s]) + logProbability(model.emission[s][emissionIndex]);
      }
    } else {
      for (let current = 0; current < 4; current++) {
        for (let prior = 0; prior < 4; prior++) {
          if (!LEGAL_NEXT[STATES[prior]].includes(STATES[current])) continue;
          const candidate = scores[t - 1][prior] + logProbability(model.transition[prior][current]) + logProbability(model.emission[current][emissionIndex]);
          if (candidate > row[current]) {
            row[current] = candidate;
            previous[current] = prior;
          }
        }
      }
    }
    scores.push(row);
    backpointer.push(previous);
  }
  let final = -1;
  for (let s = 0; s < 4; s++) {
    if (LEGAL_END.has(STATES[s]) && (final < 0 || scores.at(-1)[s] > scores.at(-1)[final])) final = s;
  }
  if (final < 0 || !Number.isFinite(scores.at(-1)[final])) {
    throw new Error('HMM 无有效 BMES 路径；检查未知字符覆盖或使用 add_alpha 平滑');
  }
  const path = Array(chars.length);
  path[chars.length - 1] = final;
  for (let t = chars.length - 1; t > 0; t--) path[t - 1] = backpointer[t][path[t]];
  return {
    tags: path.map(index => STATES[index]),
    score: scores.map(row => row.map(value => Number.isFinite(value) ? value : null)),
    backpointer,
    best_log_probability: scores.at(-1)[final],
  };
}

function decodeText(model, text) {
  const chars = Array.from(text), tags = [], chunks = [];
  let offset = 0;
  for (let i = 0; i < chars.length;) {
    const whitespace = /^\s$/u.test(chars[i]);
    let j = i + 1;
    while (j < chars.length && /^\s$/u.test(chars[j]) === whitespace) j++;
    // Python's re.split(r"(\s)") preserves each whitespace code point as its own S chunk.
    if (whitespace) j = i + 1;
    const chunk = chars.slice(i, j).join('');
    if (whitespace) {
      chunks.push({ offset, text: chunk, tags: ['S'], whitespace: true });
      tags.push('S');
    } else {
      const trace = viterbi(model, chars.slice(i, j));
      chunks.push({ offset, text: chunk, ...trace });
      tags.push(...trace.tags);
    }
    offset += j - i;
    i = j;
  }
  return { chars, tags, chunks, tokens: decode(text, tags) };
}

function decodeLongest(model, text) {
  const chars = Array.from(text), tokens = [], tags = [], steps = [];
  let words = dictionaryCache.get(model);
  if (!words) {
    words = new Set(model.words);
    dictionaryCache.set(model, words);
  }
  let offset = 0;
  while (offset < chars.length) {
    const candidates = [chars[offset]];
    if (!/^\s$/u.test(chars[offset])) {
      for (let end = offset + 2; end <= Math.min(chars.length, offset + model.max_word_length); end++) {
        const candidateChars = chars.slice(offset, end);
        if (candidateChars.some(character => /^\s$/u.test(character))) break;
        const candidate = candidateChars.join('');
        if (words.has(candidate)) candidates.push(candidate);
      }
    }
    const chosen = candidates.at(-1);
    steps.push({ offset, candidates, chosen, next_offset: offset + Array.from(chosen).length });
    tokens.push(chosen);
    const tokenChars = Array.from(chosen);
    tags.push(...(tokenChars.length === 1 ? ['S'] : ['B', ...Array(tokenChars.length - 2).fill('M'), 'E']));
    offset += tokenChars.length;
  }
  return { chars, tokens, tags, steps };
}

function runName(model) {
  return model === 'hmm1-mle' ? 'mle' : 'smooth';
}

function tracePacket(modelName, text, result) {
  return {
    model: modelName,
    states: STATES,
    chunks: result.chunks,
    tokens: result.tokens,
    tags: result.tags,
    trace_type: 'hmm1_viterbi',
    text,
    character_count: result.chars.length,
    bmes_source: 'model decoded path',
    pair_mask: STATES.map(source => STATES.map(target => LEGAL_NEXT[source].includes(target))),
  };
}

/** Compute supported frozen HMM-1 requests locally; return undefined for other routes/models. */
export async function compute(path, params, body) {
  const route = String(path).split('?')[0].replace(/\/$/u, '');
  const isAnalyze = route === '/api/v1/analyze' || route === '/api/v1/analyze/';
  const isTrace = route.endsWith('/experiments/1/trace');
  if (!isAnalyze && !isTrace) return undefined;

  const modelName = isAnalyze
    ? (body?.segmentation_model ?? 'hmm1-smooth')
    : parameter(params, 'model', 'hmm1-smooth');
  if (!['hmm1-smooth', 'hmm1-mle', 'longest'].includes(modelName)) return undefined;
  const text = isAnalyze ? body?.text : parameter(params, 'text');
  if (typeof text !== 'string' || !text) return undefined;
  const charLength = Array.from(text).length;
  if (isAnalyze && charLength > 5000) throw new Error('text 长度不得超过 5000 Unicode code point');
  if (isTrace && charLength > 30) throw new Error('算法过程只支持 1–30 Unicode code point');
  const requestedRun = isAnalyze ? (body?.run_id ?? 'latest') : parameter(params, 'run_id', 'latest');
  if (typeof requestedRun !== 'string' || !/^[A-Za-z0-9_-]+$/u.test(requestedRun)) {
    throw new Error('run_id 无效');
  }
  const model = await loadModel(modelName);
  if (modelName !== 'longest' && requestedRun !== 'latest' && requestedRun !== model.run?.run_id) return undefined;
  const started = performance.now();
  const result = modelName === 'longest' ? decodeLongest(model, text) : decodeText(model, text);
  if (isTrace) {
    if (modelName !== 'longest') return tracePacket(modelName, text, result);
    const dictionary = dictionaryCache.get(model);
    const steps = result.steps.map(step => ({
      ...step,
      candidate_info: step.candidates.map(word => ({
        word,
        kind: /^\s+$/u.test(word) ? 'whitespace' : dictionary.has(word) ? 'dictionary match' : 'single-char fallback',
        selected: word === step.chosen,
      })),
    }));
    return {
      model: 'longest', steps, tokens: result.tokens, tags: result.tags,
      trace_type: 'longest_scan', text, character_count: result.chars.length,
      bmes_source: 'derived from tokens',
      pair_mask: STATES.map(source => STATES.map(target => LEGAL_NEXT[source].includes(target))),
      fallback_count: steps.filter(step => step.candidate_info.some(item => item.selected && item.kind === 'single-char fallback')).length,
    };
  }

  const tasks = body?.tasks ?? ['segmentation', 'ner'];
  const includeSegmentation = tasks.includes('segmentation');
  const includeNer = tasks.includes('ner');
  return {
    text,
    segmentation: includeSegmentation ? { model: modelName === 'longest' ? 'longest' : 'hmm-cws', tokens: result.tokens, tags: result.tags } : null,
    ner: includeNer ? {
      status: 'unavailable',
      model: 'hanlp-msra-bert',
      error: '静态浏览器计算不包含 HanLP NER 资源。',
    } : null,
    classification: null,
    run: includeSegmentation ? model.run : null,
    elapsed_ms: Math.round((performance.now() - started) * 100) / 100,
  };
}
