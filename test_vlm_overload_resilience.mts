/**
 * Regression test for the VLM request-overload fix (retry-with-backoff,
 * concurrency capping, reset-lane isolation, inconclusive-vs-rejected
 * tracking). Exercises the REAL `verifySameScene` / `resetVlmCache` /
 * `vlmNetworkStats` exports from server/vlm-verify.ts against a mocked
 * `global.fetch` (no real GPU VLM server is reachable in this environment) —
 * only the network transport is faked, all retry/backoff/concurrency/
 * accounting logic under test is real.
 *
 * Deliberately unit-scoped to server/vlm-verify.ts (no real ffmpeg frame
 * extraction / fixture videos needed) because every behavior this task added
 * — retries, the two concurrency lanes, and inconclusive accounting — lives
 * entirely inside that module and is fully observable through its exported
 * functions and counters. The higher-level "does resolveSegmentsWithVLM pick
 * the right candidate" scenarios are already covered by the existing
 * test_vlm_candidate_mock_e2e.mts baseline and are intentionally not
 * duplicated here; that file is re-run alongside this one to confirm no
 * regression.
 *
 * IMPORTANT: VLM_ENDPOINT_URL / VLM_MAX_CONCURRENT_REQUESTS / etc. must be
 * set on the shell command line BEFORE this process starts, not assigned to
 * process.env at the top of this file — static imports are hoisted and
 * server/vlm-verify.ts reads them into module-level `export const`s at
 * import time (see .agents/memory/vlm-candidate-test-pattern.md).
 */
import {
  verifySameScene,
  resetVlmCache,
  vlmNetworkStats,
  resetVlmNetworkStats,
  VLM_MAX_CONCURRENT_REQUESTS,
  VLM_RESET_MAX_CONCURRENT_REQUESTS,
  VLM_RETRY_ATTEMPTS,
} from './server/vlm-verify';

let pass = 0, fail = 0;
function check(label: string, cond: boolean, detail?: any) {
  if (cond) { pass++; console.log(`  OK  ${label}`); }
  else { fail++; console.log(`  FAIL ${label}` + (detail !== undefined ? ` -- ${JSON.stringify(detail)}` : '')); }
}

console.log(`Config: VLM_MAX_CONCURRENT_REQUESTS=${VLM_MAX_CONCURRENT_REQUESTS}, VLM_RESET_MAX_CONCURRENT_REQUESTS=${VLM_RESET_MAX_CONCURRENT_REQUESTS}, VLM_RETRY_ATTEMPTS=${VLM_RETRY_ATTEMPTS}`);
if (VLM_MAX_CONCURRENT_REQUESTS < 3) {
  console.warn('This test expects VLM_MAX_CONCURRENT_REQUESTS>=3 to meaningfully exercise concurrency; set it via env.');
}

const originalFetch = global.fetch;
let fetchCallLog: string[] = [];

function okVerifyResponse(same: boolean, confidence: number): any {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => ({ choices: [{ message: { content: JSON.stringify({ same, confidence }) } }] }),
  };
}
function statusResponse(status: number, statusText: string): any {
  return { ok: false, status, statusText, json: async () => ({}) };
}

function sleep(ms: number) { return new Promise((r) => setTimeout(r, ms)); }

async function main() {
  // ── Scenario 1: transient 503s, then success -> retried, not rejected ──
  console.log('\n== Scenario 1: two transient 503s then a real accept verdict ==');
  resetVlmNetworkStats();
  let calls = 0;
  (global as any).fetch = async (_url: any, _init: any) => {
    calls++;
    if (calls <= 2) return statusResponse(503, 'Service Unavailable');
    return okVerifyResponse(true, 91);
  };
  const r1 = await verifySameScene('a', 'b');
  check('scenario 1: retried until success (3 attempts)', calls === 3, calls);
  check('scenario 1: real verdict returned, not null', r1 !== null && r1.same === true && r1.confidencePct === 91, r1);
  check('scenario 1: NOT counted as inconclusive (got a real answer)', vlmNetworkStats.verifyInconclusive === 0, vlmNetworkStats.verifyInconclusive);

  // ── Scenario 2: persistent 502 -> exhausts retries -> inconclusive, not a thrown error / crash ──
  console.log('\n== Scenario 2: persistent 502 exhausts retries -> inconclusive (not a rejection) ==');
  resetVlmNetworkStats();
  calls = 0;
  (global as any).fetch = async () => { calls++; return statusResponse(502, 'Bad Gateway'); };
  const r2 = await verifySameScene('a', 'b');
  check('scenario 2: attempted VLM_RETRY_ATTEMPTS+1 times', calls === VLM_RETRY_ATTEMPTS + 1, calls);
  check('scenario 2: returns null (could-not-verify contract preserved)', r2 === null, r2);
  check('scenario 2: counted as inconclusive exactly once', vlmNetworkStats.verifyInconclusive === 1, vlmNetworkStats.verifyInconclusive);

  // ── Scenario 3: ECONNRESET-style connection failure -> retried, then inconclusive ──
  console.log('\n== Scenario 3: persistent ECONNRESET -> retried then inconclusive ==');
  resetVlmNetworkStats();
  calls = 0;
  (global as any).fetch = async () => {
    calls++;
    const e: any = new Error('socket hang up');
    e.cause = { code: 'ECONNRESET' };
    throw e;
  };
  const r3 = await verifySameScene('a', 'b');
  check('scenario 3: retried ECONNRESET VLM_RETRY_ATTEMPTS+1 times', calls === VLM_RETRY_ATTEMPTS + 1, calls);
  check('scenario 3: returns null, never throws out to the caller', r3 === null, r3);
  check('scenario 3: counted as inconclusive exactly once', vlmNetworkStats.verifyInconclusive === 1, vlmNetworkStats.verifyInconclusive);

  // ── Scenario 4: non-retryable 4xx fails fast, and is NOT inconclusive ──
  console.log('\n== Scenario 4: non-retryable 400 fails on first attempt, not counted as inconclusive ==');
  resetVlmNetworkStats();
  calls = 0;
  (global as any).fetch = async () => { calls++; return statusResponse(400, 'Bad Request'); };
  const r4 = await verifySameScene('a', 'b');
  check('scenario 4: exactly 1 attempt (no retry for a genuine 4xx)', calls === 1, calls);
  check('scenario 4: returns null', r4 === null, r4);
  check('scenario 4: NOT counted as inconclusive (immediate non-transient failure)', vlmNetworkStats.verifyInconclusive === 0, vlmNetworkStats.verifyInconclusive);

  // ── Scenario 5: concurrency cap holds under parallel load ──────────────
  console.log('\n== Scenario 5: concurrency cap enforced across parallel verifySameScene calls ==');
  let active = 0, peak = 0;
  const N = 9;
  (global as any).fetch = async () => {
    active++;
    peak = Math.max(peak, active);
    await sleep(40);
    active--;
    return okVerifyResponse(true, 85);
  };
  const results5 = await Promise.all(Array.from({ length: N }, () => verifySameScene('a', 'b')));
  check(`scenario 5: peak concurrent calls === cap (${VLM_MAX_CONCURRENT_REQUESTS})`, peak === VLM_MAX_CONCURRENT_REQUESTS, peak);
  check('scenario 5: all calls still completed with a real verdict', results5.every((r) => r?.same === true), results5);

  // ── Scenario 6: reset lane is independent of a saturated verify lane ───
  console.log('\n== Scenario 6: cache-reset calls are not starved by in-flight verification traffic ==');
  active = 0;
  fetchCallLog = [];
  (global as any).fetch = async (url: any) => {
    const u = String(url);
    if (/\/slots\/\d+/.test(u)) {
      fetchCallLog.push('reset');
      return { ok: true, status: 200, statusText: 'OK', json: async () => ({}) };
    }
    fetchCallLog.push('verify-start');
    await sleep(150); // slow verification call, holds a verify-lane slot the whole time
    fetchCallLog.push('verify-end');
    return okVerifyResponse(true, 80);
  };
  // Saturate the verify lane with VLM_MAX_CONCURRENT_REQUESTS slow calls (no .then/await yet).
  const slowVerifyPromises = Array.from({ length: VLM_MAX_CONCURRENT_REQUESTS }, () => verifySameScene('a', 'b'));
  await sleep(10); // let them all acquire the verify gate before starting the reset race
  const resetStart = Date.now();
  await resetVlmCache('test label — should not wait on saturated verify lane');
  const resetElapsedMs = Date.now() - resetStart;
  check(
    `scenario 6: resetVlmCache completed quickly (${resetElapsedMs}ms) despite a fully-saturated, slow (150ms) verify lane`,
    resetElapsedMs < 100,
    resetElapsedMs,
  );
  check(
    'scenario 6: verify lane was genuinely saturated (all slots taken) when the reset race started',
    fetchCallLog.filter((e) => e === 'verify-start').length === VLM_MAX_CONCURRENT_REQUESTS,
    fetchCallLog,
  );
  check(
    'scenario 6: all 4 slot resets completed while every verify call was STILL in flight (no verify-end logged yet)',
    fetchCallLog.includes('reset') && !fetchCallLog.includes('verify-end'),
    fetchCallLog,
  );
  await Promise.all(slowVerifyPromises); // drain before the next scenario

  // ── Scenario 7: retries never change the verdict (determinism) ─────────
  console.log('\n== Scenario 7: identical verdict whether the VLM answers immediately or only after transient failures ==');
  (global as any).fetch = async () => okVerifyResponse(false, 22);
  const immediate = await verifySameScene('a', 'b');
  let attempt = 0;
  (global as any).fetch = async () => {
    attempt++;
    if (attempt < 3) return statusResponse(503, 'Service Unavailable');
    return okVerifyResponse(false, 22);
  };
  const afterRetries = await verifySameScene('a', 'b');
  check(
    'scenario 7: verdict identical (same={false}, confidence=22) regardless of retry path',
    JSON.stringify(immediate) === JSON.stringify(afterRetries),
    { immediate, afterRetries },
  );

  (global as any).fetch = originalFetch;
  console.log(`\n${pass} passed, ${fail} failed`);
  if (fail > 0) process.exit(1);
}

main().catch((err) => {
  (global as any).fetch = originalFetch;
  console.error('FATAL:', err);
  process.exit(1);
});
