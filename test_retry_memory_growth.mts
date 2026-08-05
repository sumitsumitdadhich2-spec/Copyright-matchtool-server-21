/**
 * One-off check for the Retry feature's memory-growth acceptance criterion:
 * calling retrySegmentCandidates() repeatedly must not grow server RSS
 * proportionally to the number of retries, since every candidate is
 * persisted to disk and nothing is accumulated in memory across calls.
 *
 * Reuses the same real fingerprint fixtures as test_candidate_retry.mts.
 * Forces `broader_search` mode on every iteration (the more expensive path)
 * by deleting the candidate file and rebuilding a fresh "all-checked, pool
 * exhausted" entry before each call, so every iteration re-triggers
 * matchVideosFromFiles() + a temp fingerprint file write/read/unlink — the
 * most memory-churn-heavy path in the module.
 *
 * Not a pass/fail gate (matching engine + VLM mock work each allocate and
 * free real short-lived memory, so some GC noise is expected) — reports the
 * RSS trend across iterations for manual inspection.
 */
import * as path from 'path';
import * as fs from 'fs';
import { matchVideosFromFiles } from './server/matching-engine';
import { retrySegmentCandidates } from './server/candidate-retry';
import { writeCandidatesFileSync, StoredCandidateSet } from './server/candidate-recovery';

const ROOT = process.cwd();
const UPLOAD_DIR = path.join(ROOT, 'uploads');
const MOVIE_VIDEO = '/tmp/rt_movie.mp4';
const SHORT_VIDEO = '/tmp/rt_short.mp4';
const MOVIE_RESULT = path.join(ROOT, 'uploads/TEST-retry-movie_result.json');
const SHORT_RESULT = path.join(ROOT, 'uploads/TEST-retry-short_result.json');
const FAKE_MATCH_JOB_ID = 'match-TEST-mem-growth';
const SEGMENT_INDEX = 0;
const ITERATIONS = 20;

// Always-reject mock VLM so every iteration exhausts its pool via broader
// search and never short-circuits early on an accept.
const originalFetch = global.fetch;
(global as any).fetch = async (url: any) => {
  const u = String(url);
  if (u.endsWith('/v1/models')) return { ok: true, status: 200, json: async () => ({}) } as any;
  if (/\/slots\/\d+/.test(u)) return { ok: true, status: 200, json: async () => ({}) } as any;
  if (u === process.env.VLM_ENDPOINT_URL) {
    return {
      ok: true, status: 200,
      json: async () => ({ choices: [{ message: { content: JSON.stringify({ same: false, confidence: 5 }) } }] }),
    } as any;
  }
  return originalFetch(url as any);
};

function fmtMB(bytes: number) { return (bytes / 1048576).toFixed(1) + ' MB'; }

async function main() {
  console.log('Building base matched segment from real fingerprints...');
  const matchResult = await matchVideosFromFiles(SHORT_RESULT, MOVIE_RESULT, {});
  const base = matchResult.segments[0];
  if (!base) throw new Error('No base segment matched — fixtures may be stale.');
  console.log(`base segment: short[${base.shortStart}-${base.shortEnd}] -> movie[${base.movieStart}-${base.movieEnd}]`);

  const rss: number[] = [];
  if (global.gc) global.gc();
  rss.push(process.memoryUsage().rss);
  console.log(`iter 0 (baseline): RSS=${fmtMB(rss[0])}`);

  for (let i = 1; i <= ITERATIONS; i++) {
    // Rebuild a fully-checked (pool-exhausted) entry each time so this call
    // is forced into the broader_search branch, not unchecked_pool.
    const entry: StoredCandidateSet = {
      segmentIndex: SEGMENT_INDEX,
      shortStart: base.shortStart,
      shortEnd: base.shortEnd,
      recordedAt: Date.now(),
      dropped: true,
      candidates: [
        { segment: base, checked: true, verdict: 'rejected', confidencePct: 5 },
      ],
    };
    writeCandidatesFileSync(UPLOAD_DIR, FAKE_MATCH_JOB_ID, SEGMENT_INDEX, entry);

    await retrySegmentCandidates(
      UPLOAD_DIR, FAKE_MATCH_JOB_ID, SEGMENT_INDEX,
      SHORT_VIDEO, MOVIE_VIDEO, SHORT_RESULT, MOVIE_RESULT,
    );

    if (global.gc) global.gc();
    rss.push(process.memoryUsage().rss);
    console.log(`iter ${i}: RSS=${fmtMB(rss[i])}  (delta vs baseline: ${fmtMB(rss[i] - rss[0])})`);
  }

  // Cleanup the synthetic candidate file this script wrote.
  try {
    const p = path.join(UPLOAD_DIR, `${FAKE_MATCH_JOB_ID}_seg${SEGMENT_INDEX}_candidates.json`);
    if (fs.existsSync(p)) fs.unlinkSync(p);
  } catch { /* best-effort */ }

  const first5Avg = rss.slice(1, 6).reduce((a, b) => a + b, 0) / 5;
  const last5Avg = rss.slice(-5).reduce((a, b) => a + b, 0) / 5;
  const growthPerIter = (last5Avg - first5Avg) / (ITERATIONS - 5);
  console.log(`\nAvg RSS iters 1-5: ${fmtMB(first5Avg)}`);
  console.log(`Avg RSS last 5 iters: ${fmtMB(last5Avg)}`);
  console.log(`Approx growth per iteration: ${fmtMB(growthPerIter)}`);
  if (growthPerIter < 2 * 1048576) {
    console.log('OK  no meaningful per-iteration RSS growth trend (< 2 MB/iter, consistent with GC noise).');
  } else {
    console.log('WATCH  RSS grew noticeably per iteration — investigate further.');
  }
  process.exit(0);
}

main().catch((err) => { console.error('FATAL:', err); process.exit(1); });
