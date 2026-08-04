import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

(async () => {
  const referenceVideo = 'test_2m_final.mp4';
  const queryVideo = 'cut_clip.mp4';
  const serverUrl = 'http://localhost:3000';

  console.log('==================================================');
  console.log('PART 1: BROWSER CUT-LIST MATCHING FLOW (PUPPETEER)');
  console.log('==================================================');

  console.log('Launching browser...');
  const browser = await puppeteer.launch({
    executablePath: '/root/.cache/puppeteer/chrome/linux-150.0.7871.24/chrome-linux64/chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--js-flags="--max-old-space-size=4096"', '--enable-precise-memory-info']
  });

  const page = await browser.newPage();
  await page.setCacheEnabled(false);

  page.on('console', msg => {
    // Only print interesting log messages to keep output readable
    const txt = msg.text();
    if (txt.includes('processed') || txt.includes('Match') || txt.includes('Status') || txt.includes('extract') || txt.includes('error')) {
      console.log(`[PAGE] ${txt}`);
    }
  });

  page.on('pageerror', err => {
    console.log(`[PAGE ERROR] ${err.toString()}`);
  });

  console.log(`Navigating to ${serverUrl}...`);
  await page.goto(serverUrl, { waitUntil: 'networkidle0' });
  console.log('Page loaded successfully.');

  // --- A. Upload and process Reference Video ---
  console.log(`Selecting reference video: ${referenceVideo}...`);
  const inputs = await page.$$('input[type="file"]');
  if (inputs.length < 2) {
    throw new Error('Could not find file inputs for reference and target videos');
  }
  
  const refInput = inputs[0];
  await refInput.uploadFile(referenceVideo);

  console.log("Clicking 'Process Reference' button...");
  const processRefBtn = await page.evaluateHandle(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    return buttons.find(b => b.textContent?.includes('Process Reference'));
  });

  if (!processRefBtn) {
    throw new Error('Process Reference button not found');
  }
  
  await processRefBtn.asElement().click();
  console.log('Reference extraction started in browser...');

  // Wait for reference processing to complete (the status text will change)
  let refDone = false;
  while (!refDone) {
    const statusText = await page.evaluate(() => {
      const divs = Array.from(document.querySelectorAll('div'));
      const statusDiv = divs.find(d => d.textContent?.includes('Status:'));
      return statusDiv ? statusDiv.textContent : '';
    });

    if (statusText.toLowerCase().includes('reference video processed') || statusText.toLowerCase().includes('processed:')) {
      console.log(`Reference processed successfully. Status: ${statusText}`);
      refDone = true;
      break;
    }
    
    if (statusText.toLowerCase().includes('error')) {
      throw new Error(`Reference processing failed with error: ${statusText}`);
    }
    await new Promise(r => setTimeout(r, 2000));
  }

  // --- B. Upload Target Clip and Run Matching ---
  console.log(`Selecting target query clip: ${queryVideo}...`);
  const targetInput = inputs[1];
  await targetInput.uploadFile(queryVideo);

  console.log('Updating Cut List JSON configuration...');
  // We specify that the cut clip has its start at 0s of this file, and has 125 frames (5 seconds of 25fps)
  const cutListJson = JSON.stringify([{ "startTimeSeconds": 0, "frames": 125 }], null, 2);
  await page.evaluate((json) => {
    const textarea = document.querySelector('textarea');
    if (textarea) {
      textarea.value = json;
      // Trigger input event to update state in React
      textarea.dispatchEvent(new Event('input', { bubbles: true }));
    }
  }, cutListJson);

  console.log("Clicking 'Run Cut-List Matching' button...");
  const runMatchingBtn = await page.evaluateHandle(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    return buttons.find(b => b.textContent?.includes('Run Cut-List Matching'));
  });

  if (!runMatchingBtn) {
    throw new Error('Run Cut-List Matching button not found');
  }

  await runMatchingBtn.asElement().click();
  console.log('Matching execution started in browser...');

  // Wait for matching to complete
  let matchingDone = false;
  while (!matchingDone) {
    const statusText = await page.evaluate(() => {
      const divs = Array.from(document.querySelectorAll('div'));
      const statusDiv = divs.find(d => d.textContent?.includes('Status:'));
      return statusDiv ? statusDiv.textContent : '';
    });

    if (statusText.toLowerCase().includes('matching complete')) {
      console.log(`Matching complete. Status: ${statusText}`);
      matchingDone = true;
      break;
    }

    if (statusText.toLowerCase().includes('error')) {
      throw new Error(`Matching failed with error: ${statusText}`);
    }
    await new Promise(r => setTimeout(r, 1000));
  }

  // Extract match results table data
  console.log('Extracting match results from DOM...');
  const resultsTableData = await page.evaluate(() => {
    const rows = Array.from(document.querySelectorAll('tbody tr'));
    return rows.map(row => {
      const cells = Array.from(row.querySelectorAll('td'));
      return {
        cutIndex: cells[0]?.textContent?.trim(),
        targetStart: cells[1]?.textContent?.trim(),
        frames: cells[2]?.textContent?.trim(),
        refMatchTime: cells[3]?.textContent?.trim(),
        confidence: cells[4]?.textContent?.trim()
      };
    });
  });

  console.log('\n--- BROWSER MATCH RESULTS ---');
  console.log(JSON.stringify(resultsTableData, null, 2));

  await browser.close();

  console.log('\n==================================================');
  console.log('PART 2: SERVER END-TO-END API TEST (CURL/FETCH)');
  console.log('==================================================');

  // Verify the query cut clip is uploaded and analyzed using endpoints:
  // /api/upload -> /api/status/:jobId -> /api/result/:jobId
  console.log(`Initiating file upload to ${serverUrl}/api/upload...`);
  
  const videoFileBuffer = fs.readFileSync(queryVideo);
  const formData = new FormData();
  formData.append('video', new Blob([videoFileBuffer]), queryVideo);

  const uploadRes = await fetch(`${serverUrl}/api/upload`, {
    method: 'POST',
    body: formData
  });

  if (!uploadRes.ok) {
    throw new Error(`Upload API failed: ${uploadRes.status} ${uploadRes.statusText}`);
  }

  const uploadData = await uploadRes.json();
  const { jobId } = uploadData;
  console.log(`Upload successful. Received jobId: ${jobId}`);

  // Poll status endpoint
  let apiDone = false;
  let statusData = null;
  console.log(`Polling status for job ${jobId}...`);

  while (!apiDone) {
    const statusRes = await fetch(`${serverUrl}/api/status/${jobId}`);
    if (!statusRes.ok) {
      throw new Error(`Status API failed: ${statusRes.status} ${statusRes.statusText}`);
    }

    statusData = await statusRes.json();
    console.log(`[Status] ${statusData.status} | processed: ${statusData.processedFrames}/${statusData.totalFrames}`);

    if (statusData.status === 'completed') {
      apiDone = true;
      break;
    }
    if (statusData.status === 'failed') {
      throw new Error(`Server processing failed for job: ${statusData.error}`);
    }

    await new Promise(r => setTimeout(r, 1000));
  }

  // Fetch final results JSON
  console.log(`Fetching results from ${serverUrl}/api/result/${jobId}...`);
  const resultRes = await fetch(`${serverUrl}/api/result/${jobId}`);
  if (!resultRes.ok) {
    throw new Error(`Result API failed: ${resultRes.status} ${resultRes.statusText}`);
  }

  const resultData = await resultRes.json();
  console.log('\n--- SERVER API RESULT (FIRST 3 FRAMES) ---');
  console.log(JSON.stringify(resultData.slice(0, 3), null, 2));
  console.log(`... Total result length: ${resultData.length} frames.`);
  
  console.log('\nAll E2E checks passed successfully!');
})();
