async (page) => {
  const checks=[],errors=[],warnings=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());if(m.type()==='warning')warnings.push(m.text());});
  page.on('request',r=>{if(r.url().includes('/api/'))requests.push({method:r.method(),url:r.url(),body:r.postData()});});
  const base='http://127.0.0.1:8000',out='F:/projects/NLPClass/outputs/browser/course_ia_20260927_181741/';
  const record=(name,pass,detail={})=>checks.push({name,pass,...detail});
  const overflow=()=>page.evaluate(()=>({width:innerWidth,body:document.documentElement.scrollWidth,overflow:document.documentElement.scrollWidth>innerWidth}));
  try {
    await page.setViewportSize({width:1440,height:1000});await page.route('**/static/**',route=>route.continue());
    const routes=['/','/segmentation','/sequence-labeling','/tagging','/information-extraction','/clustering','/classification','/dependency-parsing','/deep-learning','/integrated-analysis'];
    for(const route of routes){const response=await page.goto(base+route);await page.waitForLoadState('networkidle');record('route '+route,response.status()===200&&await page.locator('nav[aria-label="主导航"] a').count()===10&&await page.locator('nav a[aria-current="page"]').getAttribute('href')===route,{overflow:await overflow()});}
    await page.goto(base+'/segmentation');await page.waitForFunction(()=>document.getElementById('active-run').textContent.includes('Run'));
    record('default HMM Smooth',await page.locator('#segmentation-model').inputValue()==='hmm1-smooth');
    for(const [model,kind] of [['longest','longest_scan'],['bigram','bigram_lattice'],['hmm1-smooth','hmm1_viterbi'],['hmm2-smooth','hmm2_pair_viterbi'],['bilstm-crf','bilstm_crf_emission']]){
      await page.locator('#segmentation-model').selectOption(model);record('switch cleared '+model,await page.locator('#algorithm-trace').textContent()==='');
      await page.locator('#tab-interactive').click();await page.locator('#input-text').fill('商品🙂和服务');await page.locator('#analyze-button').click();
      await page.waitForFunction(kind=>document.getElementById('algorithm-trace').dataset.traceType===kind,kind);await page.waitForFunction(()=>!document.getElementById('analyze-button').disabled);
      record('trace '+model,(await page.locator('#tokens-result').textContent())==='商品🙂和服务',{trace:await page.locator('#algorithm-trace').textContent()});
      await page.locator('#tab-process').click();await page.screenshot({path:out+'segmentation-'+model+'.png',fullPage:true});
    }
    await page.locator('#tab-evaluation').click();await page.waitForSelector('#comparison-table table');record('unique seven models',await page.locator('#comparison-table table tr').count()===8,{table:await page.locator('#comparison-table').textContent()});await page.screenshot({path:out+'segmentation-evaluation.png',fullPage:true});
    await page.locator('#tab-errors').click();await page.waitForSelector('#error-list .error-case');record('CWS errors read only',true);await page.screenshot({path:out+'segmentation-errors.png',fullPage:true});
    await page.goto(base+'/sequence-labeling#hmm');await page.waitForSelector('#model-specific-parameters table');record('HMM parameter charts',await page.locator('#transition-chart canvas').count()>0);await page.screenshot({path:out+'sequence-hmm1.png',fullPage:true});
    await page.locator('#segmentation-model').selectOption('hmm2-smooth');await page.waitForSelector('select[aria-label="Tensor prev2"]');record('HMM2 A2 slice',true);await page.screenshot({path:out+'sequence-hmm2.png',fullPage:true});
    await page.goto(base+'/deep-learning');await page.waitForSelector('#start-chart canvas');record('BiLSTM training and CRF',await page.locator('#model-specific-parameters').textContent().then(t=>t.includes('pytorch-crf 0.7.2')));await page.screenshot({path:out+'deep-learning.png',fullPage:true});
    await page.goto(base+'/tagging#evaluation');await page.waitForSelector('#ner-metrics table');record('NER metrics separate',true,{table:await page.locator('#ner-metrics').textContent()});await page.screenshot({path:out+'ner-evaluation.png',fullPage:true});
    await page.locator('#tab-ner').click();await page.locator('#input-text').fill('中国石油大学（华东）位于青岛。🙂李明在这里学习。');await page.locator('#analyze-button').click();await page.waitForFunction(()=>!document.getElementById('analyze-button').disabled,null,{timeout:60000});record('NER interaction',await page.locator('#original-text').textContent()==='中国石油大学（华东）位于青岛。🙂李明在这里学习。');await page.screenshot({path:out+'ner-interaction.png',fullPage:true});
    await page.locator('#tab-errors').click();await page.waitForSelector('#error-list .error-case');record('NER error view',true);await page.screenshot({path:out+'ner-errors.png',fullPage:true});
    await page.goto(base+'/integrated-analysis');await page.waitForFunction(()=>document.getElementById('active-run').textContent.includes('Run'));await page.locator('#input-text').fill('中国石油大学（华东）位于青岛。');await page.locator('#analyze-button').click();await page.waitForFunction(()=>!document.getElementById('analyze-button').disabled);record('integrated real segmentation NER',(await page.locator('#tokens-result').textContent())==='中国石油大学（华东）位于青岛。');await page.screenshot({path:out+'integrated.png',fullPage:true});
    await page.setViewportSize({width:390,height:844});
    for(const route of ['/','/segmentation#evaluation','/sequence-labeling#hmm','/tagging#evaluation','/deep-learning','/integrated-analysis']){
      await page.goto(base+route);await page.waitForLoadState('networkidle');const size=await overflow();record('mobile '+route,!size.overflow,size);
      await page.locator('#drawer-toggle').click();record('drawer '+route,await page.locator('#drawer-toggle').getAttribute('aria-expanded')==='true'&&await page.locator('nav a').first().isVisible());await page.keyboard.press('Escape');record('drawer closes '+route,await page.locator('#drawer-toggle').getAttribute('aria-expanded')==='false');await page.screenshot({path:out+'mobile-'+route.split('#')[0].slice(1).replaceAll('/','-')+'.png',fullPage:true});
    }
    return {checks,errors,warnings,requests};
  }catch(e){return {checks,errors,warnings,requests,fatal:e.message};}
}
