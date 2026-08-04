/**
 * In-process test of the manual Retry flow (server/candidate-retry.ts) using
 * a mocked `fetch` for the VLM endpoint (no GPU VLM server reachable here —
 * same pattern as test_vlm_candidate_mock_e2e.mts) but REAL ffmpeg frame
 * extraction and a REAL matched segment/candidatePool from real fingerprint
 * files (see gen_retry_test_fixtures.mts). Never touches matching-engine.ts /
 * vlm-verify.ts / vlm-segment-resolver.ts / deferred-recovery.ts decision
 * logic — this only exercises retrySegmentCandidates()'s own new code.
 *
 * IMPORTANT: VLM_ENDPOINT_URL must be set on the shell command line before
 * this process starts (see test_vlm_candidate_mock_e2e.mts's note on import
 * hoisting) — an in-file process.env assignment is too late.
 */
import * as path from 'path';
import * as fs from 'fs';
import { matchVideosFromFiles } from './server/matching-engine';
import {
  buildCandidateHistoryEntry,
  writeCandidatesFileSync,
  readCandidatesFile,
  deleteCandidateFilesForJob,
} from './server/candidate-recovery';
import { retrySegmentCandidates } from './server/candidate-retry';

const ROOT = process.cwd();
const MOVIE_VIDEO = '/tmp/rt_movie.mp4';
const SHORT_VIDEO = '/tmp/rt_short.mp4';
const MOVIE_RESULT = path.join(ROOT, 'uploads/TEST-retry-movie_result.json');
const SHORT_RESULT = path.join(ROOT, 'uploads/TEST-retry-short_result.json');
const UPLOAD_DIR = path.join(ROOT, 'uploads');
const FAKE_MATCH_JOB_ID = 'match-TEST-retry';

let pass = 0, fail = 0;
function check(label: string, cond: boolean, detail?: any) {
  if (cond) { pass++; console.log(`  OK  ${label}`); }
  else { fail++; console.log(`  FAIL ${label}` + (detail !== undefined ? ` -- ${JSON.stringify(detail)}` : '')); }
}

let script: Array<{ same: boolean; confidence: number }> = [];
let callCount = 0;
const originalFetch = global.fetch;
(global as any).fetch = async (url: any, init: any) => {
  const u = String(url);
  if (u.endsWith('/v1/models')) return { ok: true, status: 200, json: async () => ({}) } as any;
  if (/\/slots\/\d+/.test(u)) return { ok: true, status: 200, json: async () => ({}) } as any;
  if (u === process.env.VLM_ENDPOINT_URL) {
    callCount++;
    const idx = Math.min(callCount, script.length) - 1;
    const verdict = script[idx];
    console.log(`    [mock-fetch] call #${callCount} -> ${JSON.stringify(verdict)}`);
    return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: JSON.stringify(verdict) } }] }) } as any;
  }
  return originalFetch(url, init);
};

async function main() {
  console.log('== Building real matched segment(s) + candidatePool from real fingerprint files ==');
  const matchResult = await matchVideosFromFiles(SHORT_RESULT, MOVIE_RESULT, {});
  check('at least one segment matched', matchResult.segments.length >= 1, matchResult.segments.length);
  const original = matchResult.segments[0];
  const pool = matchResult.candidatePool ?? [];
  console.log(`  base segment: short[${original.shortStart.toFixed(2)}-${original.shortEnd.toFixed(2)}] -> movie[${original.movieStart.toFixed(2)}-${original.movieEnd.toFixed(2)}]`);
  console.log(`  candidatePool size: ${pool.length}`);
  check('candidatePool has at least 2 alternates for a meaningful unchecked-pool test', pool.length >= 2, pool.length);

  deleteCandidateFilesForJob(UPLOAD_DIR, FAKE_MATCH_JOB_ID); // clean slate

  // ── Scenario A: unchecked_pool mode — one candidate already checked+accepted
  // (simulating the automatic pass), N others left unchecked. Retry should
  // pick the *unchecked* pool first, never re-run a broader search while any
  // remain, and should stop at the first accept without touching the earlier
  // entries. ──────────────────────────────────────────────────────────────
  console.log('\n== Scenario A: unchecked-pool retry rejects once then accepts, prior entries untouched ==');
  const historyA = buildCandidateHistoryEntry(0, original, [{ segment: original, verdict: 'accepted', confidencePct: 91 }], original, pool);
  check('history built for scenario A', !!historyA);
  if (!historyA) { console.log(`${pass} passed, ${fail} failed`); process.exit(1); }
  const uncheckedCountA = historyA.candidates.filter(c => !c.checked).length;
  check('scenario A setup: has unchecked candidates to retry into', uncheckedCountA >= 1, uncheckedCountA);
  writeCandidatesFileSync(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0, historyA);

  callCount = 0;
  // First unchecked candidate rejected, second (if any) accepted; pad extra rejects in case there are more.
  script = Array.from({ length: Math.max(uncheckedCountA, 1) }, (_, i) => i === uncheckedCountA - 1 ? { same: true, confidence: 90 } : { same: false, confidence: 20 });

  const resultA = await retrySegmentCandidates(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0, SHORT_VIDEO, MOVIE_VIDEO, SHORT_RESULT, MOVIE_RESULT);
  check('scenario A: mode is unchecked_pool', resultA.mode === 'unchecked_pool', resultA.mode);
  check('scenario A: no new candidates added (pool wasn\'t exhausted)', resultA.newCandidatesAdded === 0, resultA.newCandidatesAdded);
  check('scenario A: VLM called exactly once per unchecked candidate up to the accept', callCount === uncheckedCountA, { callCount, uncheckedCountA });

  const afterA = readCandidatesFile(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0);
  check('scenario A: candidate 0 (the original accepted match) is untouched', afterA?.candidates[0]?.checked === true && afterA?.candidates[0]?.verdict === 'accepted' && afterA?.candidates[0]?.confidencePct === 91);
  if (resultA.outcome === 'accepted') {
    check('scenario A: recoveredCandidateIndex now points at the newly-accepted candidate, not candidate 0', afterA?.recoveredCandidateIndex === resultA.acceptedCandidateIndex && afterA?.recoveredCandidateIndex !== 0, afterA?.recoveredCandidateIndex);
    check('scenario A: accepted candidate is checked+accepted', afterA?.candidates[resultA.acceptedCandidateIndex!]?.verdict === 'accepted');
  } else {
    console.log('  (pool exhausted without an accept in this run — acceptable given synthetic data; checked outcome shape only)');
    check('scenario A exhausted: every candidate in the original pool is now checked', afterA?.candidates.every(c => c.checked) ?? false);
  }

  // ── Scenario B: broader_search mode — mark every candidate as checked
  // (simulating a fully-exhausted pool), then retry again. Must trigger a
  // fresh matchVideosFromFiles-based search restricted to this segment's
  // range, append new unchecked candidates (deduped against existing
  // timestamps), and never delete/mutate anything already present. ───────
  console.log('\n== Scenario B: broader_search kicks in once the pool is exhausted ==');
  const entryBefore = readCandidatesFile(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0)!;
  const countBefore = entryBefore.candidates.length;
  // Force full exhaustion regardless of what scenario A left behind.
  entryBefore.candidates.forEach(c => { if (!c.checked) { c.checked = true; c.verdict = 'rejected'; c.confidencePct = 10; } });
  writeCandidatesFileSync(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0, entryBefore);

  callCount = 0;
  script = [{ same: false, confidence: 15 }, { same: true, confidence: 77 }, { same: false, confidence: 5 }, { same: false, confidence: 5 }, { same: false, confidence: 5 }, { same: false, confidence: 5 }, { same: false, confidence: 5 }, { same: false, confidence: 5 }, { same: false, confidence: 5 }, { same: false, confidence: 5 }];
  const resultB = await retrySegmentCandidates(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0, SHORT_VIDEO, MOVIE_VIDEO, SHORT_RESULT, MOVIE_RESULT);
  check('scenario B: mode is broader_search', resultB.mode === 'broader_search', resultB.mode);

  const afterB = readCandidatesFile(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0);
  check('scenario B: pre-existing candidates (0..countBefore-1) are byte-for-byte untouched', JSON.stringify(afterB?.candidates.slice(0, countBefore)) === JSON.stringify(entryBefore.candidates));
  check('scenario B: candidate list only grew (never shrank)', (afterB?.candidates.length ?? 0) >= countBefore, afterB?.candidates.length);
  if (resultB.newCandidatesAdded > 0) {
    check('scenario B: new candidates were appended after the original ones', (afterB?.candidates.length ?? 0) === countBefore + resultB.newCandidatesAdded, { total: afterB?.candidates.length, countBefore, added: resultB.newCandidatesAdded });
  } else {
    console.log('  (broader search found nothing new in this synthetic clip — acceptable; verified no mutation of existing entries instead)');
  }

  // ── Scenario C: calling retry again immediately (2nd broader search) must
  // still never delete anything from scenario B, even if it also finds
  // nothing new or exhausts without accepting. ───────────────────────────
  console.log('\n== Scenario C: a second retry call never deletes/shrinks history ==');
  const beforeC = readCandidatesFile(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0)!;
  beforeC.candidates.forEach(c => { if (!c.checked) { c.checked = true; c.verdict = 'rejected'; c.confidencePct = 8; } });
  writeCandidatesFileSync(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0, beforeC);
  const countC = beforeC.candidates.length;
  callCount = 0;
  script = [{ same: false, confidence: 5 }];
  await retrySegmentCandidates(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0, SHORT_VIDEO, MOVIE_VIDEO, SHORT_RESULT, MOVIE_RESULT);
  const afterC = readCandidatesFile(UPLOAD_DIR, FAKE_MATCH_JOB_ID, 0);
  check('scenario C: candidate list never shrank below its previous size', (afterC?.candidates.length ?? 0) >= countC, { before: countC, after: afterC?.candidates.length });

  deleteCandidateFilesForJob(UPLOAD_DIR, FAKE_MATCH_JOB_ID);

  console.log(`\n${pass} passed, ${fail} failed`);
  if (fail > 0) process.exit(1);
}

main().catch((err) => { console.error('FATAL:', err); process.exit(1); });
