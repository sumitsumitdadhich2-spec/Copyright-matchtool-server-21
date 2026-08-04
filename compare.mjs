import fs from 'fs';

const sFps = JSON.parse(fs.readFileSync('server_fps_final.json', 'utf8'));
const bFps = JSON.parse(fs.readFileSync('browser_fps_final.json', 'utf8'));

if (!sFps.length || !bFps.length) {
  console.error("Empty fingerprints");
  process.exit(1);
}

// Compute timestamps for server frames (assumes 25 fps)
for (let f of sFps) {
  if (f.timestamp === undefined) {
    f.timestamp = f.frameIndex / 25;
  }
}

const maxTimeS = sFps[sFps.length - 1].timestamp;
const maxTimeB = bFps[bFps.length - 1].timestamp;
const maxTime = Math.max(maxTimeS, maxTimeB);

const pctToFind = [0, 10, 25, 50, 75, 90];

function charCompare(hash1, hash2) {
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

for (let pct of pctToFind) {
  const targetTime = (pct / 100) * maxTime;
  let sIdx = 0, sDiff = Infinity;
  for (let i = 0; i < sFps.length; i++) {
    let d = Math.abs(sFps[i].timestamp - targetTime);
    if (d < sDiff) { sDiff = d; sIdx = i; }
  }
  let bIdx = 0, bDiff = Infinity;
  for (let i = 0; i < bFps.length; i++) {
    let d = Math.abs(bFps[i].timestamp - targetTime);
    if (d < bDiff) { bDiff = d; bIdx = i; }
  }
  const sFp = sFps[sIdx], bFp = bFps[bIdx];
  console.log(`\nFrame at ~${pct}% (Target ${targetTime.toFixed(2)}s)`);
  const fields = ['full', 'crop_9_16_0', 'crop_9_16_1', 'crop_9_16_2', 'crop_9_16_3', 'crop_9_16_4'];
  let thisFrameTotalSim = 0;
  for (let field of fields) {
    const sHash = sFp.variants[field]?.hash || '';
    const bHash = bFp.variants[field]?.hash || '';
    const sim = charCompare(sHash, bHash);
    totalSim += sim; comps++;
    if (sim < worstSim) worstSim = sim;
    thisFrameTotalSim += sim;
    console.log(`  ${field}: ${sim.toFixed(1)}% match`);
    if (pct === 50 && (field === 'full' || field === 'crop_9_16_1')) {
      console.log(`    Server:  ${sHash.substring(0, 64)}...`);
      console.log(`    Browser: ${bHash.substring(0, 64)}...`);
    }
  }
}
console.log(`\n=== FINAL RESULTS ===`);
console.log(`Average Similarity: ${(totalSim / comps).toFixed(2)}%`);
console.log(`Worst-case Field Similarity: ${worstSim.toFixed(2)}%`);

