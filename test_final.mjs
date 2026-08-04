import puppeteer from 'puppeteer-core';
import fs from 'fs';
import { execSync } from 'child_process';

(async () => {
  // --- 1. RUN SERVER PIPELINE ---
  console.log('Running server extraction on test_2m_final.mp4...');
  try {
    execSync('npx tsx server/export_fps.ts test_2m_final.mp4 server_fps_final.json', { stdio: 'inherit' });
  } catch (err) {
    console.error('Server extraction failed', err);
    process.exit(1);
  }

  // --- 2. RUN BROWSER PIPELINE ---
  console.log('Running browser extraction on test_2m_final.mp4...');
  const browser = await puppeteer.launch({
    executablePath: '/app/applet/chrome/linux-151.0.7922.34/chrome-linux64/chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--js-flags="--max-old-space-size=4096"', '--enable-precise-memory-info']
  });
  
  const page = await browser.newPage();
  await page.setCacheEnabled(false);
  
  page.on('console', msg => {
    // console.log(`[PAGE] ${msg.text()}`);
  });
  
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle0' });
  console.log('Page loaded. Uploading file...');
  
  const fileInput = await page.$('input[type="file"]');
  await fileInput.uploadFile('test_2m_final.mp4');
  
  let processingStarted = false;
  
  await page.waitForSelector('button');
  const startBtn = await page.$('button');
  await startBtn.click();
  
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
  
  fs.writeFileSync('browser_fps_final.json', JSON.stringify(allFingerprints, null, 2));
  console.log(`Saved ${allFingerprints.length} fingerprints to browser_fps_final.json`);
  await browser.close();

  // Let compare.mjs handle the output
})();
