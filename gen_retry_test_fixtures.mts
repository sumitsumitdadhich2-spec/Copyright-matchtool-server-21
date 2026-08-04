/**
 * One-off fixture generator for test_candidate_retry.mts. Self-contained
 * (does not depend on any previously-generated /tmp files) — synthesizes a
 * 10s movie made of 5 x 2s blocks (A,B,A,C,A — pattern A repeated 3x at
 * well-separated timestamps ~4s apart, matching the same trick used by
 * gen_vlm_test_fixtures.mts) and a 1.5s short clip cut from the 2nd A
 * occurrence, then fingerprints both.
 */
import * as path from 'path';
import { extractFingerprints } from './server/pipeline';

const ROOT = process.cwd();
const MOVIE_VIDEO = '/tmp/rt_movie.mp4';
const SHORT_VIDEO = '/tmp/rt_short.mp4';
const MOVIE_RESULT = path.join(ROOT, 'uploads/TEST-retry-movie_result.json');
const SHORT_RESULT = path.join(ROOT, 'uploads/TEST-retry-short_result.json');

async function main() {
  console.log('Extracting movie fingerprints...');
  const movieFrames = await extractFingerprints(MOVIE_VIDEO, MOVIE_RESULT);
  console.log(`  movie: ${movieFrames} frames -> ${MOVIE_RESULT}`);

  console.log('Extracting short fingerprints...');
  const shortFrames = await extractFingerprints(SHORT_VIDEO, SHORT_RESULT);
  console.log(`  short: ${shortFrames} frames -> ${SHORT_RESULT}`);

  console.log('Done.');
  process.exit(0);
}

main().catch((err) => { console.error('FATAL:', err); process.exit(1); });
