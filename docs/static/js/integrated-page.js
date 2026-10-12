import { resources,bindAnalysis,bindSelection,fail,$ } from './course-ui.js';
import { activeTab } from './layout.js';
async function init(){const ctx=await resources('integrated');const start=bindAnalysis(ctx,{tasks:['segmentation','ner']});bindSelection(ctx);document.addEventListener('module-tab',()=>{if(activeTab()==='single'&&$('original-text').classList.contains('result-empty'))start();});if(activeTab()==='single')start();}init().catch(fail);
