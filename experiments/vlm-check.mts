/**
 * EXPERIMENT helper: run the app's REAL verification gate
 * (server/vlm-verify.ts -> verifySegmentByVideo) on one candidate, and print
 * the full verdict INCLUDING the evidence text, so an accept is never taken
 * on a bare "same" verdict alone.
 *
 * Usage:
 *   npx tsx experiments/vlm-check.mts --short 7.00,9.12 --movie 7.00,9.12
 */
import * as dotenv from 'dotenv';
dotenv.config({ path: ['.env.development.local', '.env.local', '.env'], quiet: true });

import { verifySegmentByVideo } from '../server/vlm-verify';
import { degenerateCandidateReason } from '../server/degenerate-guard';

function arg(name: string, dflt: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}

const MOVIE_PATH = arg('movie-path', 'uploads/exp-ref-movie.mp4');
const SHORT_PATH = arg('short-path', 'uploads/exp-short-clip.mp4');
const [shortStart, shortEnd] = arg('short', '7.00,9.12').split(',').map(Number);
const [movieStart, movieEnd] = arg('movie', '7.00,9.12').split(',').map(Number);
const LABEL = arg('label', 'Exp');

async function main() {
  const seg = { shortStart, shortEnd, movieStart, movieEnd };

  // Report what the app's structural guard would say about this candidate.
  const shortSpan = shortEnd - shortStart;
  const movieSpan = movieEnd - movieStart;
  const speedRatio = shortSpan > 0 ? movieSpan / shortSpan : 0;
  const reason = degenerateCandidateReason({
    ...seg,
    speedRatio,
    matchSequence: [
      { shortTime: shortStart, movieTime: movieStart, similarity: 1 },
      { shortTime: (shortStart + shortEnd) / 2, movieTime: (movieStart + movieEnd) / 2, similarity: 1 },
      { shortTime: shortEnd, movieTime: movieEnd, similarity: 1 },
    ],
  });
  console.log(`[${LABEL}] speedRatio=${speedRatio.toFixed(3)} degenerateGuard=${reason ?? 'PASS'}`);

  const result = await verifySegmentByVideo(SHORT_PATH, MOVIE_PATH, seg, LABEL);
  if (!result) {
    console.log(`[${LABEL}] NO VERDICT (Gemini unavailable / cut failed)`);
    process.exit(2);
  }
  console.log(`[${LABEL}] verdict=${result.same ? 'SAME' : 'DIFFERENT'} confidence=${result.confidencePct} likelihood=${result.matchLikelihood}`);
  console.log(`[${LABEL}] evidence:`);
  (result.evidence ?? ['(none supplied)']).forEach((e, i) => console.log(`   ${i + 1}. ${e}`));
}

main().catch((e) => { console.error('[vlm-check] FAILED', e); process.exit(1); });
