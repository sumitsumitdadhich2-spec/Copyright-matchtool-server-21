/**
 * Sanity test for the dual-model quota manager (scripts/test-quota-rotation.ts).
 * Run: npx tsx scripts/test-quota-rotation.ts
 * RPM=3 so rotation triggers fast. Mock fetch records which model got hit.
 */
process.env.GEMINI_API_KEY = 'test-key';
process.env.GEMINI_RPM = '3';
process.env.GEMINI_RPD = '10';

const calls: string[] = [];
const dailyExhaust: Set<string> = new Set();

(global as any).fetch = async (url: string) => {
  const model = decodeURIComponent(url.match(/models\/([^:]+):/)![1]);
  calls.push(model);
  if (dailyExhaust.has(model)) {
    return {
      status: 429,
      ok: false,
      headers: { get: () => null },
      text: async () => 'GenerateRequestsPerDayPerProjectPerModel-FreeTier',
    };
  }
  return {
    status: 200,
    ok: true,
    headers: { get: () => null },
    json: async () => ({
      candidates: [{ content: { parts: [{ text: '{"same": true, "confidence": 90}' }] } }],
    }),
  };
};

async function main() {
  const { geminiVerifyComposite, getGeminiStatus } = await import('../server/gemini-vlm');

  // Test 1: 6 rapid requests with RPM=3 → first 3 on primary, next 3 on
  // fallback (instant rotation) — no waiting at all.
  console.log('--- Test 1: RPM rotation ---');
  const t0 = Date.now();
  for (let i = 0; i < 6; i++) {
    const r = await geminiVerifyComposite('AAAA', 'test');
    if (!r?.same) throw new Error('expected verdict');
  }
  const elapsed = Date.now() - t0;
  console.log('call sequence:', calls.join(', '));
  console.log(`6 calls took ${elapsed}ms (should be near-instant, < 3000ms)`);
  if (elapsed > 3000) throw new Error('rotation did not avoid waiting!');
  const primaryCalls = calls.filter(c => c.includes('flash-lite-latest')).length;
  const fallbackCalls = calls.filter(c => c.includes('3.1')).length;
  if (primaryCalls !== 3 || fallbackCalls !== 3) {
    throw new Error(`expected 3+3 split, got ${primaryCalls}+${fallbackCalls}`);
  }
  console.log('PASS: 3 on primary, 3 on fallback, zero wait');

  // Test 2: daily 429 on primary → parks it, rotates to fallback.
  console.log('--- Test 2: daily exhaustion rotation (waiting 61s for RPM windows to clear) ---');
  await new Promise(r => setTimeout(r, 61_000));
  calls.length = 0;
  dailyExhaust.add('gemini-flash-lite-latest');
  const r2 = await geminiVerifyComposite('AAAA', 'test');
  console.log('call sequence:', calls.join(', '));
  if (!r2?.same) throw new Error('expected verdict via fallback');
  if (calls[0] !== 'gemini-flash-lite-latest' || calls[1] !== 'gemini-3.1-flash-lite') {
    throw new Error('expected primary(429 daily) then fallback');
  }
  const st = getGeminiStatus();
  const pm = st.models.find((m) => m.model === 'gemini-flash-lite-latest')!;
  if (!pm.dailyLimitReached) throw new Error('primary should be daily-parked');
  if (st.dailyLimitReached) throw new Error('overall flag should be false (fallback alive)');
  console.log('PASS: primary parked, fallback serving, overall flag false');

  // Test 3: next call goes straight to fallback (primary skipped, no request burned).
  calls.length = 0;
  const r3 = await geminiVerifyComposite('AAAA', 'test');
  if (!r3?.same || calls.length !== 1 || calls[0] !== 'gemini-3.1-flash-lite') {
    throw new Error(`expected single fallback call, got: ${calls.join(', ')}`);
  }
  console.log('PASS: parked model skipped without burning a request');

  // Test 4: fallback also daily-dead → overall dailyLimitReached true.
  dailyExhaust.add('gemini-3.1-flash-lite');
  const r4 = await geminiVerifyComposite('AAAA', 'test');
  if (r4 !== null) throw new Error('expected null (both exhausted → unverifiable)');
  const st4 = getGeminiStatus();
  if (!st4.dailyLimitReached) throw new Error('overall dailyLimitReached should be true');
  console.log('PASS: both exhausted → dailyLimitReached flag set for UI');

  console.log('\nALL TESTS PASSED');
  process.exit(0);
}

main().catch(e => { console.error('FAIL:', e.message); process.exit(1); });
