/**
 * One-off fixture generator for test_vlm_candidate_mock_e2e.mts.
 * Regenerates the two fingerprint result files the test hardcodes paths to.
 * Safe to re-run any time the fixtures go missing.
 *
 * Fingerprint source is a synthetic /tmp/movie_multi.mp4 (testsrc/mandelbrot/
 * rgbtestsrc blocks: A,B,A,C,A — pattern A repeated 3x at well-separated
 * timestamps, ~4s apart) rather than checkpoint_test.mp4, so the resulting
 * candidatePool has real alternates spread far enough apart to exercise
 * deferred recovery (candidates within 0.5s of a rejected timestamp are
 * excluded as duplicates — see getAlternateCandidatesForRange). This is safe
 * because the test's VLM calls are fully mocked (never inspects frame pixel
 * content) and checkpoint_test.mp4/short_clip.mp4 (used only for ffmpeg frame
 * *extraction*, not fingerprinting) are decoupled from the fingerprint data —
 * they only need to be valid, long-enough-to-seek video files, which they are.
 * See /tmp/concat_list.txt generation in shell history for how movie_multi.mp4
 * and short_clip.mp4 (a 1.6s cut from pattern A's 2nd occurrence) were built.
 */
import * as path from 'path';
import { extractFingerprints } from './server/pipeline';

const ROOT = process.cwd();
const MOVIE_VIDEO = '/tmp/movie_multi.mp4';
const SHORT_VIDEO = '/tmp/short_clip.mp4';
const MOVIE_RESULT = path.join(ROOT, 'uploads/1785663028023-es7i25h_result.json');
const SHORT_RESULT = path.join(ROOT, 'uploads/1785663028249-eict9zd_result.json');

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
