import * as fs from 'fs';

const browserFpRaw = fs.readFileSync('browser_fps.json', 'utf-8');
const serverFpRaw = fs.readFileSync('server_fps.json', 'utf-8');

const browserFps: any[] = JSON.parse(browserFpRaw);
const serverFps: any[] = JSON.parse(serverFpRaw);

console.log(`Loaded ${browserFps.length} browser fps and ${serverFps.length} server fps.`);

// Compare at 0%, 10%, 25%, 50%, 75%, 90%
const totalFrames = Math.min(browserFps.length, serverFps.length);
if (totalFrames === 0) {
  console.log("No frames to compare.");
  process.exit(1);
}

const percentiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9];
const indicesToCompare = percentiles.map(p => Math.floor(p * (totalFrames - 1)));

function hexToBin(hex: string): string {
  let bin = '';
  for (let i = 0; i < hex.length; i++) {
    bin += parseInt(hex[i], 16).toString(2).padStart(4, '0');
  }
  return bin;
}

function compareHashes(h1: string, h2: string): number {
  if (!h1 || !h2) return 0;
  const b1 = hexToBin(h1);
  const b2 = hexToBin(h2);
  let matches = 0;
  const len = Math.min(b1.length, b2.length);
  for (let i = 0; i < len; i++) {
    if (b1[i] === b2[i]) matches++;
  }
  return matches / len;
}

let sumSimilarity = 0;
let totalComparisons = 0;
let worstCase = 1.0;

for (const idx of indicesToCompare) {
  const bf = browserFps[idx];
  const sf = serverFps[idx];
  console.log(`\n--- Comparing Frame Index ${bf.frameIndex} (~${Math.round((idx / totalFrames)*100)}%) ---`);
  
  if (bf.frameIndex !== sf.frameIndex) {
    console.log(`WARNING: frame index mismatch: Browser=${bf.frameIndex}, Server=${sf.frameIndex}`);
  }
  
  // They both have "variants" object with multiple crops
  const variants = Object.keys(bf.variants);
  for (const variant of variants) {
    if (!sf.variants[variant]) {
      console.log(`Server missing variant ${variant}`);
      continue;
    }
    
    const bHash = bf.variants[variant].hash;
    const sHash = sf.variants[variant].hash;
    
    const sim = compareHashes(bHash, sHash);
    sumSimilarity += sim;
    totalComparisons++;
    if (sim < worstCase) worstCase = sim;
    
    console.log(`  Variant '${variant}': ${bHash} vs ${sHash} -> ${(sim * 100).toFixed(2)}% similarity`);
  }
}

const avgSimilarity = sumSimilarity / totalComparisons;
console.log(`\n==============================================`);
console.log(`RESULTS:`);
console.log(`Average Similarity: ${(avgSimilarity * 100).toFixed(2)}%`);
console.log(`Worst-case Similarity: ${(worstCase * 100).toFixed(2)}%`);
console.log(`==============================================\n`);

