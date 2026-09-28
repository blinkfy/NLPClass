async (page) => {
  const out = 'F:/projects/NLPClass/outputs/exp1/final_report_assets/20260927_010652';
  const responses = [], errors = [], checks = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => {if(m.type()==='error') errors.push(m.text());});
  page.on('response', async response => {
    if(response.url().includes('/api/v1/') && response.status()===200) {
      try {responses.push({url:response.url(),body:await response.json()});} catch {}
    }
  });
  await page.setViewportSize({width:1440,height:1000});
  await page.route('**/static/**', route => route.continue());
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('#run-select').options.length>1);
  checks.push({default_model:await page.locator('#segmentation-model').inputValue()});
  await page.locator('#task-ner').uncheck();
  const models=[['longest','longest_scan','03-longest-scan.png'],['bigram','bigram_lattice','04-bigram-lattice.png'],['hmm1-smooth','hmm1_viterbi','05-hmm1-viterbi.png'],['hmm2-smooth','hmm2_pair_viterbi','06-hmm2-pair-dp.png'],['bilstm-crf','bilstm_crf_emission','07-bilstm-architecture.png']];
  for(const [model,kind,file] of models) {
    await page.locator('#segmentation-model').selectOption(model);
    checks.push({model,cleared:await page.locator('#algorithm-trace').innerText()===''});
    await page.locator('#input-text').fill('商品🙂和服务');
    await page.locator('#analyze-button').click();
    await page.waitForFunction(kind=>document.querySelector('#algorithm-trace').dataset.traceType===kind && !document.querySelector('#analyze-button').disabled,kind,{timeout:60000});
    await page.waitForFunction(()=>document.querySelector('#model-details').innerText.length>100);
    checks.push({model,trace:await page.locator('#algorithm-trace').innerText(),parameters:await page.locator('#model-details').innerText(),desktop_overflow:await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)});
    await page.locator('.algorithm-panel').screenshot({path:out+'/screenshots/'+file});
    if(model==='bigram') {
      const downloadPromise=page.waitForEvent('download');
      await page.locator('#process-export').click();
      const download=await downloadPromise;
      await download.saveAs(out+'/screenshots/04-bigram-lattice-export.png');
      checks.push({png_export:download.suggestedFilename()});
    }
    if(model==='hmm2-smooth') {
      await page.locator('#algorithm-trace select').first().selectOption('3');
      await page.locator('#algorithm-trace .pair-cell:not(:disabled)').first().click();
      await page.locator('#model-details select').first().selectOption('M');
      await page.locator('#model-details select').nth(1).selectOption('E');
      await page.locator('.model-information').screenshot({path:out+'/screenshots/06-hmm2-tensor-slice.png'});
    }
    if(model==='bilstm-crf') {
      await page.locator('.model-information').screenshot({path:out+'/screenshots/08-bilstm-training-and-crf.png'});
      await page.locator('.training-process-chart').screenshot({path:out+'/screenshots/08-bilstm-training.png'});
      await page.locator('#model-details table').nth(1).screenshot({path:out+'/screenshots/09-crf-transition.png'});
    }
    await page.setViewportSize({width:390,height:844});
    await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth);
    checks.push({model,narrow_overflow:await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)});
    await page.screenshot({path:out+'/screenshots/mobile-'+model+'.png',fullPage:true});
    await page.setViewportSize({width:1440,height:1000});
  }
  await page.locator('#input-text').fill('课程展示文本🙂'.repeat(5));
  await page.locator('#analyze-button').click();
  await page.waitForFunction(()=>!document.querySelector('#analyze-button').disabled && document.querySelector('#algorithm-trace').innerText.includes('仅对'));
  checks.push({long_input:await page.locator('#algorithm-trace').innerText(),summary:await page.locator('#result-summary').innerText()});
  return {checks,errors,responses,demo_inference_executed:true,training_executed:false,official_test_inference_executed:false,scorer_executed:false};
}
