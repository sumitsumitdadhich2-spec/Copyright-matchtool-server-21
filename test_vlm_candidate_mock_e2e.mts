/**
 * In-process end-to-end test of the candidate/VLM subsystem using a mocked
 * `fetch` instead of a real GPU VLM server (none is reachable in this
 * environment). Runs the REAL resolveSegmentsWithVLM / runDeferredRecoveryPass
 * / buildCandidateHistoryEntry against a REAL matched segment (from actual
 * fingerprint files) and REAL ffmpeg frame extraction — only the VLM verdict
 * itself is scripted. Never touches matching-engine.ts / vlm-verify.ts /
 * vlm-segment-resolver.ts decision logic; this is a black-box test of their
 * existing exported behavior.
 *
 * Uses VLM_MAX_ATTEMPTS=2 for the whole run purely to keep the "exhaust and
 * drop" scenario fast (fewer attempts needed) — every scenario still exits
 * well within that cap except the deliberate always-reject one.
 *
 * IMPORTANT: VLM_ENDPOINT_URL / VLM_MAX_ATTEMPTS / VLM_CONCURRENCY must be
 * set in the environment BEFORE this process starts (e.g. on the shell
 * command line), not assigned to process.env at the top of this file.
 * Static `import` statements are hoisted and run before any other top-level
 * code in the importing module, so by the time a `process.env.X = ...` line
 * here would execute, `./server/vlm-verify.ts` (a transitive import) has
 * already evaluated its module-level `export const VLM_ENDPOINT_URL = ...`
 * with whatever the env var was at process start — an in-file assignment is
 * silently too late and has no effect.
 */
import * as path from 'path';
import { matchVideosFromFiles, MatchedSegment } from './server/matching-engine';
import { resolveSegmentsWithVLM, SegmentResolvedInfo } from './server/vlm-segment-resolver';
import { runDeferredRecoveryPass } from './server/deferred-recovery';
import {
  buildCandidateHistoryEntry,
  writeCandidatesFileSync,
  readCandidatesFile,
  listCandidateFilesForJob,
  deleteCandidateFilesForJob,
} from './server/candidate-recovery';

const ROOT = process.cwd();
const MOVIE_VIDEO = path.join(ROOT, 'checkpoint_test.mp4');
const SHORT_VIDEO = '/tmp/short_clip.mp4';
const MOVIE_RESULT = path.join(ROOT, 'uploads/1785663028023-es7i25h_result.json');
const SHORT_RESULT = path.join(ROOT, 'uploads/1785663028249-eict9zd_result.json');
const UPLOAD_DIR = path.join(ROOT, 'uploads');
const FAKE_MATCH_JOB_ID = 'match-TEST-vlm-mock';

let pass = 0, fail = 0;
function check(label: string, cond: boolean, detail?: any) {
  if (cond) { pass++; console.log(`  OK  ${label}`); }
  else { fail++; console.log(`  FAIL ${label}` + (detail !== undefined ? ` -- ${JSON.stringify(detail)}` : '')); }
}

// ── Mock fetch: scripted VLM verdicts, everything else (models/slots) is a
// plain 200 OK so isVlmAvailable() / resetVlmCache() behave normally. ──────
let script: Array<{ same: boolean; confidence: number }> = [];
let callCount = 0;
const originalFetch = global.fetch;
(global as any).fetch = async (url: any, init: any) => {
  const u = String(url);
  if (u.endsWith('/v1/models')) {
    return { ok: true, status: 200, json: async () => ({}) } as any;
  }
  if (/\/slots\/\d+/.test(u)) {
    return { ok: true, status: 200, json: async () => ({}) } as any;
  }
  if (u === process.env.VLM_ENDPOINT_URL) {
    callCount++;
    const idx = Math.min(callCount, script.length) - 1;
    const verdict = script[idx];
    console.log(`    [mock-fetch] call #${callCount} -> ${JSON.stringify(verdict)}`);
    return {
      ok: true,
      status: 200,
      json: async () => ({ choices: [{ message: { content: JSON.stringify(verdict) } }] }),
    } as any;
  }
  return originalFetch(url, init);
};

async function main() {
  console.log('== Building real matched segment(s) from real fingerprint files ==');
  const matchResult = await matchVideosFromFiles(SHORT_RESULT, MOVIE_RESULT, {});
  check('at least one segment matched', matchResult.segments.length >= 1, matchResult.segments.length);
  const original = matchResult.segments[0];
  console.log(`  base segment: short[${original.shortStart}-${original.shortEnd}] -> movie[${original.movieStart}-${original.movieEnd}]`);
  console.log(`  candidatePool size: ${matchResult.candidatePool?.length ?? 0}`);

  // ── Scenario A: immediate accept ──────────────────────────────────────
  console.log('\n== Scenario A: VLM accepts on first attempt ==');
  callCount = 0;
  script = [{ same: true, confidence: 95 }];
  let resolvedInfoA: SegmentResolvedInfo | null = null;
  const outA = await resolveSegmentsWithVLM(
    [original], SHORT_VIDEO, MOVIE_VIDEO, matchResult.candidatePool,
    undefined, (info) => { resolvedInfoA = info; },
  );
  check('scenario A: 1 segment returned', outA.length === 1, outA.length);
  check('scenario A: accepted matches original movieStart', outA[0]?.movieStart === original.movieStart, outA[0]?.movieStart);
  check('scenario A: exactly 1 VLM call made', callCount === 1, callCount);
  check('scenario A: onSegmentResolved fired with accepted != null', !!resolvedInfoA && (resolvedInfoA as any).accepted !== null);
  check('scenario A: triedCandidates has 1 entry, verdict accepted', (resolvedInfoA as any)?.triedCandidates?.length === 1 && (resolvedInfoA as any)?.triedCandidates?.[0]?.verdict === 'accepted');

  const historyA = buildCandidateHistoryEntry(0, original, (resolvedInfoA as any).triedCandidates, (resolvedInfoA as any).accepted, matchResult.candidatePool);
  check('history A: not null', historyA !== null);
  check('history A: dropped === false', historyA?.dropped === false);
  check('history A: recoveredCandidateIndex === 0 (only 1 tried, accepted)', historyA?.recoveredCandidateIndex === 0, historyA?.recoveredCandidateIndex);
  check('history A: first candidate checked=true verdict=accepted', historyA?.candidates[0]?.checked === true && historyA?.candidates[0]?.verdict === 'accepted');
  if (historyA) {
    const untried = historyA.candidates.slice(1);
    check('history A: extra (untried) candidates all checked=false', untried.every(c => c.checked === false), untried.map(c => c.checked));
  }

  // ── Scenario B: reject at cap (VLM_MAX_ATTEMPTS=2), then deferred recovery ──
  console.log('\n== Scenario B: VLM rejects twice (hits VLM_MAX_ATTEMPTS=2) -> dropped, then deferred recovery accepts ==');
  callCount = 0;
  script = [{ same: false, confidence: 10 }, { same: false, confidence: 15 }];
  let resolvedInfoB: SegmentResolvedInfo | null = null;
  const outB = await resolveSegmentsWithVLM(
    [original], SHORT_VIDEO, MOVIE_VIDEO, matchResult.candidatePool,
    undefined, (info) => { resolvedInfoB = info; },
  );
  check('scenario B: segment dropped (0 returned)', outB.length === 0, outB.length);
  check('scenario B: exactly 2 VLM calls made (respects VLM_MAX_ATTEMPTS)', callCount === 2, callCount);
  check('scenario B: onSegmentResolved fired with accepted === null', !!resolvedInfoB && (resolvedInfoB as any).accepted === null);
  const triedB = (resolvedInfoB as any)?.triedCandidates ?? [];
  check('scenario B: 2 tried candidates, both rejected', triedB.length === 2 && triedB.every((t: any) => t.verdict === 'rejected'), triedB);
  const triedMovieStartsB = triedB.map((t: any) => t.segment.movieStart);
  check('scenario B: the 2 tried candidates point at different movie timestamps (no repeat)', new Set(triedMovieStartsB).size === triedMovieStartsB.length, triedMovieStartsB);

  const historyB = buildCandidateHistoryEntry(0, original, triedB, null, matchResult.candidatePool);
  check('history B: not null', historyB !== null);
  check('history B: dropped === true', historyB?.dropped === true);
  check('history B: recoveredCandidateIndex is undefined (nothing accepted yet)', historyB?.recoveredCandidateIndex === undefined);
  check('history B: has untried candidates left for deferred recovery', (historyB?.candidates.filter(c => !c.checked).length ?? 0) > 0, historyB?.candidates.map(c => c.checked));

  // Persist to disk exactly like server.ts does, so runDeferredRecoveryPass can read it.
  deleteCandidateFilesForJob(UPLOAD_DIR, FAKE_MATCH_JOB_ID); // clean slate from any previous run
  if (historyB) writeCandidatesFileSync(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0, historyB);

  // Now let the deferred pass accept the very next untried candidate.
  script = [{ same: true, confidence: 88 }];
  callCount = 0;
  const recovered = await runDeferredRecoveryPass(UPLOAD_DIR, FAKE_MATCH_JOB_ID, SHORT_VIDEO, MOVIE_VIDEO);
  check('deferred recovery: 1 segment recovered', recovered.length === 1, recovered.length);
  check('deferred recovery: exactly 1 new VLM call (only 1 untried candidate probed before accept)', callCount === 1, callCount);
  const persisted = readCandidatesFile(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0);
  check('deferred recovery: persisted file now has recoveredCandidateIndex set', typeof persisted?.recoveredCandidateIndex === 'number', persisted?.recoveredCandidateIndex);
  if (persisted && typeof persisted.recoveredCandidateIndex === 'number') {
    const recoveredEntry = persisted.candidates[persisted.recoveredCandidateIndex];
    check('deferred recovery: recoveredCandidateIndex points at a checked+accepted candidate', recoveredEntry?.checked === true && recoveredEntry?.verdict === 'accepted', recoveredEntry);
    check('deferred recovery: recovered segment matches the persisted recovered candidate', recovered[0]?.movieStart === recoveredEntry?.segment.movieStart);
  }
  // dropped flag itself is intentionally left as-is by the deferred pass (still true) —
  // server.ts is the one that decides to merge `recovered` back into the final result;
  // just confirm the file wasn't corrupted into some other shape.
  check('deferred recovery: dropped flag untouched (still true) — server.ts merges recovered[] separately', persisted?.dropped === true);

  deleteCandidateFilesForJob(UPLOAD_DIR, FAKE_MATCH_JOB_ID);

  // ── Scenario C: reject once, then accept on retry (within main pass) ───
  console.log('\n== Scenario C: VLM rejects once then accepts (retry/replace within main pass) ==');
  callCount = 0;
  script = [{ same: false, confidence: 30 }, { same: true, confidence: 92 }];
  let resolvedInfoC: SegmentResolvedInfo | null = null;
  const outC = await resolveSegmentsWithVLM(
    [original], SHORT_VIDEO, MOVIE_VIDEO, matchResult.candidatePool,
    undefined, (info) => { resolvedInfoC = info; },
  );
  check('scenario C: 1 segment returned (accepted after retry)', outC.length === 1, outC.length);
  check('scenario C: accepted candidate is NOT the original rejected movieStart', outC[0]?.movieStart !== original.movieStart, { accepted: outC[0]?.movieStart, original: original.movieStart });
  const triedC = (resolvedInfoC as any)?.triedCandidates ?? [];
  check('scenario C: 2 tried, first rejected second accepted', triedC.length === 2 && triedC[0].verdict === 'rejected' && triedC[1].verdict === 'accepted', triedC.map((t:any)=>t.verdict));

  console.log(`\n${pass} passed, ${fail} failed`);
  if (fail > 0) process.exit(1);
}

main().catch((err) => {
  console.error('FATAL:', err);
  process.exit(1);
});
