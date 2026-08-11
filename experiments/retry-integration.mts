/**
 * Integration check for the dense-scan pool top-up wired into
 * server/candidate-retry.ts.
 *
 * Simulates the exact real-world failure: a segment whose candidate pool is
 * fully exhausted (every candidate checked and rejected, all pointing at wrong
 * movie timestamps, as the hash engine produces for a 1:1 zoomed crop of a
 * 16:9 source). A Retry click on this used to dead-end immediately. It should
 * now dense-scan the movie, append fresh candidates, and Gemini-verify them.
 *
 * Run: npx tsx experiments/retry-integration.mts
 */
import * as fs from 'fs';
import * as path from 'path';
import { retrySegmentCandidates } from '../server/candidate-retry';
import {
  StoredCandidateSet,
  matchCandidatesFilePath,
  readCandidatesFile,
} from '../server/candidate-recovery';

const UPLOAD_DIR = path.resolve('uploads');
const JOB_ID = 'exp-densescan-integration';
const SEG = 0;
const SHORT = path.resolve('uploads/exp-short-clip.mp4');
const MOVIE = path.resolve('uploads/exp-ref-movie.mp4');

// The proven ground truth from the dense-scan experiment.
const SHORT_START = 7.0;
const SHORT_END = 9.12;
const TRUE_MOVIE_START = 7.0;

function wrongCandidate(movieStart: number) {
  const duration = SHORT_END - SHORT_START;
  return {
    segment: {
      shortStart: SHORT_START,
      shortEnd: SHORT_END,
      movieStart,
      movieEnd: movieStart + duration,
      confidence: 62,
      frameCount: 5,
      isApproximate: true,
      gapCount: 0,
      speedRatio: 1,
      matchSequence: [
        { shortTime: SHORT_START, movieTime: movieStart, similarity: 61 },
        { shortTime: (SHORT_START + SHORT_END) / 2, movieTime: movieStart + duration / 2, similarity: 63 },
        { shortTime: SHORT_END, movieTime: movieStart + duration, similarity: 60 },
      ],
    },
    // Fully exhausted pool: already checked and rejected by Gemini.
    checked: true,
    verdict: 'rejected' as const,
    confidencePct: 90,
    matchLikelihood: 10,
    evidence: ['Different scene entirely.'],
  };
}

async function main() {
  // Seed an exhausted candidate set: 3 wrong timestamps, all already rejected.
  const entry: StoredCandidateSet = {
    segmentIndex: SEG,
    shortStart: SHORT_START,
    shortEnd: SHORT_END,
    recordedAt: Date.now(),
    candidates: [wrongCandidate(180.0), wrongCandidate(240.0), wrongCandidate(300.0)],
    dropped: true,
  };
  const file = matchCandidatesFilePath(UPLOAD_DIR, JOB_ID, SEG);
  fs.writeFileSync(file, JSON.stringify(entry, null, 2));

  console.log('=== BEFORE ===');
  console.log(`pool size: ${entry.candidates.length}, unchecked: 0 (fully exhausted)`);
  console.log(`ground truth movieStart: ${TRUE_MOVIE_START}s\n`);

  const result = await retrySegmentCandidates(
    UPLOAD_DIR, JOB_ID, SEG, SHORT, MOVIE, '', '',
  );

  console.log('\n=== RESULT ===');
  console.log(JSON.stringify(result, null, 2));

  const after = readCandidatesFile(UPLOAD_DIR, JOB_ID, SEG)!;
  console.log('\n=== AFTER ===');
  console.log(`pool size: ${after.candidates.length} (was 3)`);
  after.candidates.forEach((c, i) => {
    const used = after.recoveredCandidateIndex === i ? ' <== USED' : '';
    console.log(
      `  #${i} movie=${c.segment.movieStart.toFixed(2)}s-${c.segment.movieEnd.toFixed(2)}s ` +
      `conf=${c.segment.confidence} checked=${c.checked} verdict=${c.verdict ?? '-'} ` +
      `likelihood=${c.matchLikelihood ?? '-'}${used}`,
    );
    if (c.evidence?.length && after.recoveredCandidateIndex === i) {
      c.evidence.forEach(e => console.log(`      · ${e}`));
    }
  });

  const accepted = after.recoveredCandidateIndex !== undefined
    ? after.candidates[after.recoveredCandidateIndex] : undefined;
  console.log('\n=== VERDICT ===');
  if (!accepted) {
    console.log('FAIL: nothing accepted.');
  } else {
    const delta = Math.abs(accepted.segment.movieStart - TRUE_MOVIE_START);
    console.log(`outcome=${result.outcome} bestEffort=${after.bestEffort === true}`);
    console.log(`accepted movieStart=${accepted.segment.movieStart.toFixed(2)}s vs truth ${TRUE_MOVIE_START}s (delta ${delta.toFixed(2)}s)`);
    console.log(delta <= 2.0 ? 'PASS: recovered the correct movie timestamp.' : 'FAIL: wrong timestamp.');
  }

  fs.rmSync(file, { force: true });
}

main().catch(e => { console.error(e); process.exit(1); });
