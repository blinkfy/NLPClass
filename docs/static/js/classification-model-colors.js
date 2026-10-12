// Display identity only; never part of a frozen model or experiment config.
export const MODEL_COLORS = Object.freeze({C0:'#bc8549', C1:'#357ca9', C2:'#148c8a', C3:'#7965a6'});
export const MODEL_LIGHT_COLORS = Object.freeze(Object.fromEntries(Object.entries(MODEL_COLORS).map(([id, color]) => [id, '#' + color.slice(1).match(/../g).map(channel => Math.round(parseInt(channel,16)*.3 + 255*.7).toString(16).padStart(2,'0')).join('')])));
export const MODEL_SCORE_TYPES = Object.freeze({C0:'NB posterior', C1:'LinearSVC decision score', C2:'LinearSVC decision score', C3:'TextCNN Softmax'});

// The enclosing workbench Run may include models inherited from earlier stages.
export function modelSource(overview, id) {
  const stage = id === 'C3' ? 'Stage3' : id === 'C2' ? 'Stage2' : 'Stage1';
  const run = id === 'C3' ? overview?.stage3?.run : id === 'C2' ? overview?.stage2?.run : overview?.run;
  return {stage, runId:run?.run_id || null};
}

export const MODEL_NAMES = Object.freeze({C0:'MultinomialNB', C1:'word LinearSVC', C2:'char χ² LinearSVC', C3:'TextCNN'});
