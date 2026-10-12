// Shared objects, URL selection and tab hash; no model or experimental state is written.
export function createClassificationState() {
  const query = new URLSearchParams(location.search);
  const modes = {stage4:'learning',stage5:'ablation',stage6:'uncertainty'};
  let state = {
    currentRun:query.get('run') || null,
    currentModel:query.get('model') || null,
    currentDocument:query.get('doc') || null,
    currentComparison:query.get('pair') || 'C2:C3',
    currentAnalysisMode:query.get('mode') || modes[query.get('analysis')] || 'formal',
    documentSource:query.get('source') === 'stage5' ? 'stage5' : 'test',
    currentCondition:query.get('condition') || 'OA',
    selectedEvidence:null,
  };
  const listeners = new Set();
  return {
    get:()=>({...state}),
    set(patch) {
      const previous = state;
      state = {...state,...patch};
      if(['currentRun','currentModel','currentDocument','documentSource','currentCondition','currentComparison'].some(key=>previous[key]!==state[key]))state.selectedEvidence=null;
      if (Object.keys(patch).every(key => previous[key] === state[key])) return;
      const url = new URL(location.href);
      for (const [key,name] of Object.entries({currentRun:'run',currentModel:'model',currentDocument:'doc',currentComparison:'pair',currentAnalysisMode:'mode',documentSource:'source',currentCondition:'condition'})) {
        if (state[key]) url.searchParams.set(name,state[key]); else url.searchParams.delete(name);
      }
      history.replaceState(null,'',url);
      for (const listener of listeners) listener({...state},{...previous});
    },
    subscribe(listener) {listeners.add(listener);return()=>listeners.delete(listener);},
  };
}
