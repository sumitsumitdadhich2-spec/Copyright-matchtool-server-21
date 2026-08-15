/**
 * Regression test for the VLM alt-candidate "1-2 frame" bug.
 *
 * Alternate candidates built for the VLM retry pool (buildAltCandidatesForChunk
 * / buildAltCandidate in server/matching-engine.ts) must always cover the FULL
 * short-clip chunk they stand in for — never collapse to a single frame with a
 * zero-duration movie span, which is what made the VLM fallback pick "correct
 * ~50% of the time, but sometimes 1-2 frame segments" (user report).
 *
 * No ffmpeg / real video needed: fingerprint hashes are plain '0'/'1' text, so
 * we hand-construct exact, controlled Hamming similarities.
 *   - "pattern" frames form a smoothly-evolving sequence (1 bit flipped per
 *     frame, like real motion) so scene-cut detection sees no cut and the
 *     whole short clip is a single scene chunk.
 *   - Short clip = pattern[0..79]. Movie = filler ... pattern[0..79] (at frame
 *     100) ... filler. Ground truth: short[0-79] -> movie[100-179], offset 100.
 *   - Filler frames are independent random 256-bit hashes: ~50% similarity to
 *     anything, far below the 82% Pass-1 threshold (so the true segment is
 *     unambiguous), but enough for the VLM fallback pool's uncapped scan to
 *     have "noise" alternates to build full candidates from.
 */
import { groundMatchedSegments, getAlternateCandidatesForRange, FPData } from './server/matching-engine';

const HASH_BITS = 256;
const PATTERN_LEN = 80;

function seededRandom(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function randomHash(rand: () => number): string {
  let out = '';
  for (let i = 0; i < HASH_BITS; i++) out += rand() < 0.5 ? '0' : '1';
  return out;
}

function flipRandomBit(hash: string, rand: () => number): string {
  const idx = Math.floor(rand() * HASH_BITS);
  const bit = hash[idx] === '1' ? '0' : '1';
  return hash.slice(0, idx) + bit + hash.slice(idx + 1);
}

function makeFP(frameIndex: number, hash: string): FPData {
  return { frameIndex, timestamp: frameIndex / 25, variants: { full: { hash } } };
}

async function main() {
  const rand = seededRandom(42);

  const pattern: string[] = [randomHash(rand)];
  for (let i = 1; i < PATTERN_LEN; i++) pattern.push(flipRandomBit(pattern[i - 1], rand));

  const shortFps: FPData[] = pattern.map((h, i) => makeFP(i, h));

  const movieFps: FPData[] = [];
  let idx = 0;
  for (let i = 0; i < 100; i++) movieFps.push(makeFP(idx++, randomHash(rand)));
  const patternStart = idx;
  for (let i = 0; i < PATTERN_LEN; i++) movieFps.push(makeFP(idx++, pattern[i]));
  for (let i = 0; i < 70; i++) movieFps.push(makeFP(idx++, randomHash(rand)));

  console.log(`[Test] short=${shortFps.length} frames, movie=${movieFps.length} frames, ground-truth movie offset=${patternStart}`);

  const result = await groundMatchedSegments(shortFps, movieFps, 82, 9, 3);

  console.log(`\n[Test] Accepted segments: ${result.segments.length}`);
  for (const s of result.segments) {
    console.log(`  short ${s.shortStart.toFixed(2)}-${s.shortEnd.toFixed(2)}s -> movie ${s.movieStart.toFixed(2)}-${s.movieEnd.toFixed(2)}s frames=${s.frameCount} conf=${s.confidence.toFixed(1)}%`);
  }

  const pool = result.candidatePool ?? [];
  const accepted = new Set(result.segments);
  const altOnly = pool.filter(c => !accepted.has(c));
  console.log(`\n[Test] candidatePool: ${pool.length} total, ${altOnly.length} alt-only (VLM fallback) candidates`);

  let failures = 0;

  // 1. The main pass must still find the true segment, at full length —
  //    proves Pass 1/2/3 ("first-time matching") behavior is untouched.
  const expectedMovieStart = patternStart / 25;
  const truth = result.segments.find(s => Math.abs(s.movieStart - expectedMovieStart) < 0.5);
  if (!truth) {
    console.error(`FAIL: primary match did not find the ground-truth segment at movie ${expectedMovieStart}s`);
    failures++;
  } else if (truth.frameCount !== PATTERN_LEN) {
    console.error(`FAIL: primary segment frameCount=${truth.frameCount}, expected ${PATTERN_LEN}`);
    failures++;
  } else {
    console.log(`PASS: primary segment found with full frameCount=${truth.frameCount}`);
  }

  // 2. No alt-candidate may collapse to a 1-2 frame stub — this was the bug.
  const thin = altOnly.filter(c => c.frameCount <= 2);
  if (thin.length > 0) {
    console.error(`FAIL: ${thin.length}/${altOnly.length} alt-candidate(s) still collapse to <=2 frames`);
    failures++;
  } else {
    console.log(`PASS: 0/${altOnly.length} alt-candidates collapse to <=2 frames`);
  }

  // 3. Every alt-candidate should cover the full chunk width (80 frames) —
  //    the whole short clip is one scene chunk here, so "as many frames as
  //    the short video" means exactly PATTERN_LEN.
  const notFull = altOnly.filter(c => c.frameCount !== PATTERN_LEN);
  if (notFull.length > 0) {
    console.error(`FAIL: ${notFull.length}/${altOnly.length} alt-candidate(s) don't cover the full ${PATTERN_LEN}-frame chunk (counts: ${notFull.map(c => c.frameCount).join(', ')})`);
    failures++;
  } else {
    console.log(`PASS: all ${altOnly.length} alt-candidates cover the full ${PATTERN_LEN}-frame chunk`);
  }

  // 4. movieEnd must differ from movieStart (old bug reported a single
  //    instant regardless of the short-clip range's real duration).
  const zeroDuration = altOnly.filter(c => c.movieEnd === c.movieStart);
  if (zeroDuration.length > 0) {
    console.error(`FAIL: ${zeroDuration.length}/${altOnly.length} alt-candidate(s) report zero movie duration`);
    failures++;
  } else {
    console.log(`PASS: all ${altOnly.length} alt-candidates report a real movie duration`);
  }

  // 5. getAlternateCandidatesForRange must still surface the genuine
  //    alternative (not just noise) as the top pick when one exists.
  const alternatives = getAlternateCandidatesForRange(
    result.candidatePool, shortFps[0].timestamp, shortFps[shortFps.length - 1].timestamp, []
  );
  console.log(`\n[Test] getAlternateCandidatesForRange returned ${alternatives.length} alternative(s)`);
  if (alternatives.length === 0) {
    console.error('FAIL: expected at least one alternative candidate for the short-clip range');
    failures++;
  } else if (Math.abs(alternatives[0].movieStart - expectedMovieStart) > 0.5) {
    console.error(`FAIL: top alternative points at movie ${alternatives[0].movieStart.toFixed(2)}s, expected ~${expectedMovieStart.toFixed(2)}s`);
    failures++;
  } else {
    console.log(`PASS: top alternative correctly points at movie ~${alternatives[0].movieStart.toFixed(2)}s (frames=${alternatives[0].frameCount})`);
  }

  // 6. Simulate a rejection of the true location: the next alternative must
  //    be a DIFFERENT movie location, not a re-offer of the same spot, and
  //    still full-length.
  const afterReject = getAlternateCandidatesForRange(
    result.candidatePool, shortFps[0].timestamp, shortFps[shortFps.length - 1].timestamp, [expectedMovieStart]
  );
  console.log(`[Test] After rejecting movie ${expectedMovieStart.toFixed(2)}s: ${afterReject.length} alternative(s) left`);
  if (afterReject.some(c => Math.abs(c.movieStart - expectedMovieStart) <= 0.5)) {
    console.error('FAIL: a rejected movie timestamp was re-offered');
    failures++;
  } else if (afterReject.length > 0 && afterReject[0].frameCount !== PATTERN_LEN) {
    console.error(`FAIL: fallback alternative after rejection has frameCount=${afterReject[0].frameCount}, expected ${PATTERN_LEN}`);
    failures++;
  } else {
    console.log(`PASS: post-rejection alternatives exclude the rejected spot and stay full-length`);
  }

  console.log(failures === 0 ? '\n✅ ALL CHECKS PASSED' : `\n❌ ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
