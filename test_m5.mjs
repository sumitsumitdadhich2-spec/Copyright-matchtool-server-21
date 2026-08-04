import puppeteer from 'puppeteer-core';
import fs from 'fs';
import { execSync } from 'child_process';

(async () => {
  const videoFile = process.argv[2] || 'test_2m_real.mp4';
  const isMandelbrot = videoFile.includes('test_2m.mp4') || videoFile === 'test_2m.mp4';
  const serverOutput = isMandelbrot ? 'server_fps_2m.json' : 'server_fps_2m_real.json';
  const browserOutput = isMandelbrot ? 'browser_fps_2m.json' : 'browser_fps_2m_real.json';

  // --- 1. RUN SERVER PIPELINE ---
  console.log(`Running server extraction on ${videoFile}...`);
  try {
    execSync(`npx tsx server/export_fps.ts ${videoFile} ${serverOutput}`, { stdio: 'inherit' });
  } catch (err) {
    console.error('Server extraction failed', err);
    process.exit(1);
  }

  // --- 2. RUN BROWSER PIPELINE ---
  console.log(`Running browser extraction on ${videoFile}...`);
  const browser = await puppeteer.launch({
    executablePath: '/root/.cache/puppeteer/chrome/linux-150.0.7871.24/chrome-linux64/chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--js-flags="--max-old-space-size=4096"', '--enable-precise-memory-info']
  });
  
  const page = await browser.newPage();
  await page.setCacheEnabled(false);
  
  page.on('console', msg => {
    console.log(`[PAGE] ${msg.text()}`);
  });
  
  page.on('pageerror', err => {
    console.log(`[PAGE ERROR] ${err.toString()}`);
  });
  
  page.on('requestfailed', req => {
    console.log(`[REQUEST FAILED] ${req.url()}: ${req.failure()?.errorText}`);
  });
  
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  console.log('Page loaded. Uploading file...');
  
  const fileInput = await page.$('input[type="file"]');
  await fileInput.uploadFile(videoFile);
  
  let processingStarted = false;
  
  // Wait for the button to be enabled and click it
  console.log("Finding and clicking 'Process Reference' button...");
  const startBtn = await page.evaluateHandle(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    return buttons.find(b => b.textContent?.includes('Process Reference'));
  });
  if (!startBtn) {
    throw new Error("Process Reference button not found");
  }
  const buttonElement = startBtn.asElement();
  if (buttonElement) {
    // Wait for the button to be enabled (not disabled)
    let isDisabled = true;
    for (let i = 0; i < 10; i++) {
      isDisabled = await page.evaluate(btn => btn.disabled, buttonElement);
      if (!isDisabled) break;
      await new Promise(r => setTimeout(r, 500));
    }
    console.log("Clicking the enabled Process Reference button...");
    await buttonElement.click();
  } else {
    throw new Error("Could not cast Handle to Element");
  }
  
  let done = false;
  let stagnantCount = 0;
  let lastFrames = "";
  
  while (!done) {
    const textContent = await page.evaluate(() => {
      const divs = Array.from(document.querySelectorAll('div'));
      const statusDiv = divs.find(d => d.textContent?.includes('Status:'));
      const statusVal = statusDiv ? statusDiv.textContent : '';
      
      const progressDiv = divs.find(d => d.textContent?.includes('Progress:'));
      const progressVal = progressDiv ? progressDiv.textContent : '';
      
      return { status: statusVal, progress: progressVal };
    });
    
    if (textContent.status.includes('Processing')) {
      processingStarted = true;
    }
    
    if (processingStarted && textContent.status.toLowerCase().includes('processed')) {
      done = true;
      console.log('BROWSER PROCESSING COMPLETE!');
      break;
    }
    
    if (textContent.status.includes('error')) {
      console.log('PROCESSING ERROR:', textContent);
      break;
    }
    
    if (textContent.progress === lastFrames && processingStarted) {
      stagnantCount++;
    } else {
      stagnantCount = 0;
      lastFrames = textContent.progress;
    }
    
    if (stagnantCount > 60) {
      console.log("Stalled for 60 seconds, breaking...");
      break;
    }
    
    await new Promise(r => setTimeout(r, 1000));
  }
  
  const allFingerprints = await page.evaluate(() => {
    return window.allFingerprints || [];
  });
  
  fs.writeFileSync(browserOutput, JSON.stringify(allFingerprints, null, 2));
  console.log(`Saved ${allFingerprints.length} fingerprints to ${browserOutput}`);
  await browser.close();

  // --- 3. COMPARE 13 VARIANTS ACROSS 6 FRAMES ---
  console.log('\n--- COMPARING TIMESTAMPS ---');
  
  const sFps = JSON.parse(fs.readFileSync(serverOutput, 'utf8'));
  const bFps = JSON.parse(fs.readFileSync(browserOutput, 'utf8'));

  if (!sFps.length || !bFps.length) {
    console.error("Empty fingerprints");
    process.exit(1);
  }

  // Find duration by looking at last timestamp
  const maxTimeS = sFps[sFps.length - 1].timestamp;
  const maxTimeB = bFps[bFps.length - 1].timestamp;
  const maxTime = Math.max(maxTimeS, maxTimeB);

  console.log(`Total duration approx: ${maxTime.toFixed(2)}s`);

  const pctToFind = [0, 10, 25, 50, 75, 90];
  
  // Custom comparison to do character-by-character
  function charCompare(hash1, hash2) {
    if (!hash1 || !hash2) return 0;
    if (hash1.length !== hash2.length) return 0;
    let matches = 0;
    for (let i = 0; i < hash1.length; i++) {
      if (hash1[i] === hash2[i]) matches++;
    }
    return (matches / hash1.length) * 100;
  }

  let totalSim = 0;
  let worstSim = 100;
  let comps = 0;

  const fields = [
    'full',
    'crop_9_16_0', 'crop_9_16_1', 'crop_9_16_2', 'crop_9_16_3', 'crop_9_16_4',
    'zoom_1_25_center', 'zoom_1_25_left', 'zoom_1_25_right',
    'zoom_1_5_center', 'zoom_1_5_left', 'zoom_1_5_right',
    'zoom_2_0_center'
  ];

  for (let pct of pctToFind) {
    const targetTime = (pct / 100) * maxTime;
    
    // Find closest in S
    let sIdx = 0;
    let sDiff = Infinity;
    for (let i = 0; i < sFps.length; i++) {
      let d = Math.abs(sFps[i].timestamp - targetTime);
      if (d < sDiff) { sDiff = d; sIdx = i; }
    }
    
    // Find closest in B
    let bIdx = 0;
    let bDiff = Infinity;
    for (let i = 0; i < bFps.length; i++) {
      let d = Math.abs(bFps[i].timestamp - targetTime);
      if (d < bDiff) { bDiff = d; bIdx = i; }
    }
    
    // Content-aware local alignment: find the best matching server frame in a local search window of +/- 3 frames around sIdx
    let bestSIdx = sIdx;
    let bestAvgSim = 0;
    
    for (let offset = -3; offset <= 3; offset++) {
      const candidateIdx = sIdx + offset;
      if (candidateIdx < 0 || candidateIdx >= sFps.length) continue;
      
      let candidateSum = 0;
      for (let field of fields) {
        const sHash = sFps[candidateIdx].variants[field]?.hash || '';
        const bHash = bFps[bIdx].variants[field]?.hash || '';
        candidateSum += charCompare(sHash, bHash);
      }
      const candidateAvg = candidateSum / fields.length;
      if (candidateAvg > bestAvgSim) {
        bestAvgSim = candidateAvg;
        bestSIdx = candidateIdx;
      }
    }
    
    const sFp = sFps[bestSIdx];
    const bFp = bFps[bIdx];
    
    console.log(`\nFrame at ~${pct}% (Target ${targetTime.toFixed(2)}s)`);
    console.log(`  Server: frame ${sFp.frameIndex} @ ${sFp.timestamp.toFixed(2)}s`);
    console.log(`  Browser: frame ${bFp.frameIndex} @ ${bFp.timestamp.toFixed(2)}s`);
    console.log(`  Time diff (with alignment): ${Math.abs(sFp.timestamp - bFp.timestamp).toFixed(3)}s`);

    let thisFrameTotalSim = 0;

    for (let field of fields) {
      const sHash = sFp.variants[field]?.hash || '';
      const bHash = bFp.variants[field]?.hash || '';
      const sim = charCompare(sHash, bHash);
      
      totalSim += sim;
      comps++;
      if (sim < worstSim) worstSim = sim;
      thisFrameTotalSim += sim;
      
      console.log(`  ${field}: ${sim.toFixed(1)}% match`);
    }
    console.log(`  Frame Average: ${(thisFrameTotalSim / fields.length).toFixed(1)}%`);
  }

  console.log(`\n=== FINAL RESULTS ===`);
  console.log(`Average Similarity: ${(totalSim / comps).toFixed(2)}%`);
  console.log(`Worst-case Field Similarity: ${worstSim.toFixed(2)}%`);

})();
