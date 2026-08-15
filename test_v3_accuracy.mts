/**
 * E2E accuracy test for matching engine v3.
 * Ground truth (short_edited.mp4 built from ref_movie.mp4):
 *   short 0–6s   → movie 10–16s  (mirror + color grade + 9:16 crop)
 *   short 6–10s  → movie 40–46s  (1.5x speed + 1.3x zoom)
 *   short 10–18s → movie 70–78s  (desaturate + overlay box)
 */
import * as fs from 'fs';
import { extractFingerprints } from './server/pipeline';
import { streamPrecomputeFromFile, groundMatchedSegments, PreSet } from './server/matching-engine';

const MOVIE = '/tmp/ref_movie.mp4';
const SHORT = '/tmp/short_edited.mp4';
const MOVIE_CACHE = '/tmp/ref_movie_fp.json';
const SHORT_CACHE = '/tmp/short_edited_fp.json';

async function getPreSet(video: string, cache: string): Promise<PreSet> {
  if (!fs.existsSync(cache)) {
    console.log(`[Test] Extracting fingerprints: ${video}`);
    await extractFingerprints(video, cache);
  } else {
    console.log(`[Test] Using cached fingerprints: ${cache}`);
  }
  return streamPrecomputeFromFile(cache);
}

const GROUND_TRUTH = [
  { shortStart: 0,  shortEnd: 6,  movieStart: 10, movieEnd: 16, label: 'mirror+grade+crop' },
  { shortStart: 6,  shortEnd: 10, movieStart: 40, movieEnd: 46, label: '1.5x speed+zoom' },
  { shortStart: 10, shortEnd: 18, movieStart: 70, movieEnd: 78, label: 'desat+overlay' },
];

async function main() {
  const moviePreSet = await getPreSet(MOVIE, MOVIE_CACHE);
  const shortPreSet = await getPreSet(SHORT, SHORT_CACHE);
  console.log(`[Test] movie=${moviePreSet.fps.length} frames, short=${shortPreSet.fps.length} frames`);

  const t0 = Date.now();
  // Pass pre-built PreSets so groundMatchedSegments skips the precompute step
  // (empty variants in fps after streaming would otherwise produce wrong hashes).
  const result = await groundMatchedSegments(
    shortPreSet.fps, moviePreSet.fps,
    82, 9, 3,
    shortPreSet, moviePreSet
  );
  console.log(`[Test] Matching took ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  console.log('\n===== SEGMENTS =====');
  for (const s of result.segments) {
    console.log(
      `short ${s.shortStart.toFixed(2)}–${s.shortEnd.toFixed(2)}s → movie ${s.movieStart.toFixed(2)}–${s.movieEnd.toFixed(2)}s ` +
      `conf=${s.confidence.toFixed(1)}% frames=${s.frameCount} gaps=${s.gapCount} approx=${s.isApproximate}`
    );
  }
  console.log('\n===== UNMATCHED =====');
  for (const u of result.unmatchedRanges) {
    console.log(`short ${u.shortStart.toFixed(2)}–${u.shortEnd.toFixed(2)}s`);
  }

  // Score: coverage of ground truth + movie position correctness
  let coveredSec = 0, correctSec = 0;
  const totalSec = 18;
  for (const gt of GROUND_TRUTH) {
    let gtCovered = 0, gtCorrect = 0;
    for (const s of result.segments) {
      const oS = Math.max(gt.shortStart, s.shortStart);
      const oE = Math.min(gt.shortEnd, s.shortEnd);
      if (oE <= oS) continue;
      gtCovered += oE - oS;
      // check movie position: expected movie time at overlap start
      const speed = (gt.movieEnd - gt.movieStart) / (gt.shortEnd - gt.shortStart);
      const expMovieAtOS = gt.movieStart + (oS - gt.shortStart) * speed;
      const segSpeed = (s.movieEnd - s.movieStart) / Math.max(0.04, s.shortEnd - s.shortStart);
      const actMovieAtOS = s.movieStart + (oS - s.shortStart) * segSpeed;
      if (Math.abs(actMovieAtOS - expMovieAtOS) < 2.0) gtCorrect += oE - oS;
    }
    coveredSec += gtCovered;
    correctSec += gtCorrect;
    console.log(`\n[GT ${gt.label}] short ${gt.shortStart}-${gt.shortEnd}s: covered ${gtCovered.toFixed(1)}/${gt.shortEnd - gt.shortStart}s, position-correct ${gtCorrect.toFixed(1)}s`);
  }
  console.log(`\n===== TOTAL: covered ${coveredSec.toFixed(1)}/${totalSec}s (${(coveredSec / totalSec * 100).toFixed(0)}%), position-correct ${correctSec.toFixed(1)}s (${(correctSec / totalSec * 100).toFixed(0)}%) =====`);
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
