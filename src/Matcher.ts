import { FrameFingerprint } from './shared/fingerprint';
import { getFingerprintsBatch } from './utils/db';

export function hammingDistance(s1: string, s2: string): number {
  let dist = 0;
  for (let i = 0; i < s1.length; i++) {
    if (s1[i] !== s2[i]) dist++;
  }
  return dist;
}

export function similarityPercentage(s1: string, s2: string): number {
  const dist = hammingDistance(s1, s2);
  return (1 - dist / s1.length) * 100;
}

export function compareFingerprints(f1: FrameFingerprint, f2: FrameFingerprint): number {
  let maxSim = 0;
  // Compare f1 variants to f2 variants
  // Normally, we check f1.variants against f2.variants, but the prompt says 
  // "Comparison ... primarily on the main 'hash' field, with the crop-variant hashes used to detect cropped/zoomed matches"
  // Let's just compare all against all for robust matching
  for (const v1 of Object.values(f1.variants)) {
    for (const v2 of Object.values(f2.variants)) {
      const sim = similarityPercentage(v1.hash, v2.hash);
      if (sim > maxSim) maxSim = sim;
    }
  }
  return maxSim;
}

export async function loadAllReferenceFingerprints(videoId: string, totalBatches: number): Promise<FrameFingerprint[]> {
  const allFps: FrameFingerprint[] = [];
  for (let i = 0; i < totalBatches; i++) {
    const batch = await getFingerprintsBatch(videoId, i);
    allFps.push(...batch);
  }
  return allFps.sort((a, b) => a.frameIndex - b.frameIndex);
}

export interface Cut {
  startTimeSeconds: number;
  frames: number;
}

export interface MatchResult {
  cutIndex: number;
  cutStartTime: number;
  cutFrames: number;
  refMatchFrameIndex: number;
  refMatchTime: number;
  confidence: number;
  verifiedEndFrameIndex?: number;
}

// Coarse-to-fine search
export function findBestMatch(targetFp: FrameFingerprint, refFps: FrameFingerprint[], step = 10, windowSize = 30): { bestIndex: number, bestSim: number } {
  let bestSim = 0;
  let bestIndex = -1;
  
  // Coarse search
  for (let i = 0; i < refFps.length; i += step) {
    const sim = compareFingerprints(targetFp, refFps[i]);
    if (sim > bestSim) {
      bestSim = sim;
      bestIndex = i;
    }
  }
  
  // Fine search around the best coarse match
  if (bestIndex !== -1) {
    const start = Math.max(0, bestIndex - windowSize);
    const end = Math.min(refFps.length - 1, bestIndex + windowSize);
    for (let i = start; i <= end; i++) {
      const sim = compareFingerprints(targetFp, refFps[i]);
      if (sim > bestSim) {
        bestSim = sim;
        bestIndex = i;
      }
    }
  }
  
  return { bestIndex, bestSim };
}
