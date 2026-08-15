import fs from 'fs';

const sFps = JSON.parse(fs.readFileSync('server_fps_2m_real.json', 'utf8'));
const bFps = JSON.parse(fs.readFileSync('browser_fps_2m_real.json', 'utf8'));

const fields = [
  'full',
  'crop_9_16_0', 'crop_9_16_1', 'crop_9_16_2', 'crop_9_16_3', 'crop_9_16_4',
  'zoom_1_25_center', 'zoom_1_25_left', 'zoom_1_25_right',
  'zoom_1_5_center', 'zoom_1_5_left', 'zoom_1_5_right',
  'zoom_2_0_center'
];

function charCompare(hash1, hash2) {
  if (!hash1 || !hash2) return 0;
  if (hash1.length !== hash2.length) return 0;
  let matches = 0;
  for (let i = 0; i < hash1.length; i++) {
    if (hash1[i] === hash2[i]) matches++;
  }
  return (matches / hash1.length) * 100;
}

const pctToFind = [0, 10, 25, 50, 75, 90];
const maxTime = 120;

for (let pct of pctToFind) {
  const targetTime = (pct / 100) * maxTime;
  
  // Find closest browser frame to targetTime
  let bIdx = 0;
  let bDiff = Infinity;
  for (let i = 0; i < bFps.length; i++) {
    let d = Math.abs(bFps[i].timestamp - targetTime);
    if (d < bDiff) { bDiff = d; bIdx = i; }
  }
  
  const bFp = bFps[bIdx];
  console.log(`\n--- Target ${pct}% (${targetTime.toFixed(2)}s), Browser Frame ${bFp.frameIndex} @ ${bFp.timestamp.toFixed(2)}s ---`);
  
  // Now find the best matching server frame across ALL server frames
  let bestSIdx = -1;
  let bestAvg = 0;
  let bestFieldSims = {};
  
  for (let i = 0; i < sFps.length; i++) {
    let sum = 0;
    let fieldSims = {};
    for (let field of fields) {
      const sHash = sFps[i].variants[field]?.hash || '';
      const bHash = bFp.variants[field]?.hash || '';
      const sim = charCompare(sHash, bHash);
      sum += sim;
      fieldSims[field] = sim;
    }
    const avg = sum / fields.length;
    if (avg > bestAvg) {
      bestAvg = avg;
      bestSIdx = i;
      bestFieldSims = fieldSims;
    }
  }
  
  const sFp = sFps[bestSIdx];
  console.log(`Best Server Match: Frame ${sFp.frameIndex} @ ${sFp.timestamp.toFixed(2)}s`);
  console.log(`Average Similarity: ${bestAvg.toFixed(2)}%`);
  for (let field of fields) {
    console.log(`  ${field}: ${bestFieldSims[field].toFixed(1)}%`);
  }
}
