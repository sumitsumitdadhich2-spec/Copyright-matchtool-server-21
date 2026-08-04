/**
 * Regression test for candidate-comparison history (buildCandidateHistoryEntry
 * in server/candidate-recovery.ts).
 *
 * Feature being tested: the compare UI must be able to show, for ANY resolved
 * segment (accepted on the first try, accepted after retries, or dropped) —
 * not just previously-dropped ones — every candidate that was actually tried
 * by VLM (with its verdict + confidence) PLUS extra pool alternates that were
 * never checked because a match was already found. This is pure list-merging
 * logic (no ffmpeg, no VLM, no fs) so it's fully testable with synthetic data.
 */
import { buildCandidateHistoryEntry, StoredCandidateSet } from './server/candidate-recovery';
import type { MatchedSegment } from './server/matching-engine';

function mk(movieStart: number, confidence: number): MatchedSegment {
  return {
    shortStart: 10,
    shortEnd: 20,
    movieStart,
    movieEnd: movieStart + 10,
    confidence,
    frameCount: 250,
    isApproximate: false,
    gapCount: 0,
    speedRatio: 1,
    matchSequence: [],
  };
}

const RANGE = { shortStart: 10, shortEnd: 20 };

let failures = 0;
function check(label: string, cond: boolean) {
  if (cond) {
    console.log(`PASS: ${label}`);
  } else {
    console.error(`FAIL: ${label}`);
    failures++;
  }
}

async function main() {
  // ── Case 1: accepted on the very first attempt, with untried pool alternates ──
  {
    const A = mk(100, 91);
    const B = mk(300, 70); // never tried — pool alternate
    const C = mk(500, 60); // never tried — pool alternate
    const pool = [A, B, C];

    const entry = buildCandidateHistoryEntry(
      0, RANGE,
      [{ segment: A, verdict: 'accepted', confidencePct: 91 }],
      A, pool,
    ) as StoredCandidateSet;

    check('case1: entry produced', !!entry);
    check('case1: not dropped', entry.dropped === false);
    check('case1: recoveredCandidateIndex points at the accepted candidate', entry.candidates[entry.recoveredCandidateIndex!]?.segment === A);
    check('case1: exactly 1 tried + 2 untried = 3 total candidates', entry.candidates.length === 3);
    check('case1: tried candidate is checked+accepted+91%', entry.candidates[0].checked === true && entry.candidates[0].verdict === 'accepted' && entry.candidates[0].confidencePct === 91);
    check('case1: untried alternates are present but unchecked', entry.candidates.slice(1).every(c => c.checked === false && c.verdict === undefined));
    check('case1: untried alternates never include the accepted movie timestamp', !entry.candidates.slice(1).some(c => c.segment.movieStart === A.movieStart));
  }

  // ── Case 2: rejected twice, then accepted on the 3rd try ──
  {
    const X = mk(100, 40);
    const Y = mk(200, 55);
    const Z = mk(300, 88);
    const untried = mk(700, 65);
    const pool = [X, Y, Z, untried];

    const entry = buildCandidateHistoryEntry(
      1, RANGE,
      [
        { segment: X, verdict: 'rejected', confidencePct: 40 },
        { segment: Y, verdict: 'rejected', confidencePct: 55 },
        { segment: Z, verdict: 'accepted', confidencePct: 88 },
      ],
      Z, pool,
    ) as StoredCandidateSet;

    check('case2: not dropped', entry.dropped === false);
    check('case2: 3 tried + 1 untried = 4 total', entry.candidates.length === 4);
    check('case2: rejected candidates keep their own confidence', entry.candidates[0].confidencePct === 40 && entry.candidates[1].confidencePct === 55);
    check('case2: recoveredCandidateIndex = 2 (last tried)', entry.recoveredCandidateIndex === 2);
    check('case2: candidate at recoveredCandidateIndex is the accepted one', entry.candidates[2].segment === Z && entry.candidates[2].verdict === 'accepted');
    check('case2: the untried alternate is included unchecked', entry.candidates[3].segment === untried && entry.candidates[3].checked === false);
  }

  // ── Case 3: dropped — every attempt rejected, no acceptance ──
  {
    const P = mk(100, 30);
    const Q = mk(200, 45);
    const extra = mk(900, 50);
    const pool = [P, Q, extra];

    const entry = buildCandidateHistoryEntry(
      2, RANGE,
      [
        { segment: P, verdict: 'rejected', confidencePct: 30 },
        { segment: Q, verdict: 'rejected', confidencePct: 45 },
      ],
      null, pool,
    ) as StoredCandidateSet;

    check('case3: dropped === true', entry.dropped === true);
    check('case3: recoveredCandidateIndex is undefined (nothing accepted yet)', entry.recoveredCandidateIndex === undefined);
    check('case3: 2 tried (both rejected) + 1 untried extra = 3 total', entry.candidates.length === 3);
    check('case3: extra alternate excludes both already-tried timestamps', entry.candidates[2].segment === extra);
  }

  // ── Case 4: a pool "alternate" that happens to sit at an already-tried timestamp must NOT be duplicated ──
  {
    const A = mk(100, 91);
    const dup = mk(100.1, 20); // within default 0.5s drift tolerance of A's movieStart -> must be excluded
    const pool = [A, dup];

    const entry = buildCandidateHistoryEntry(
      3, RANGE,
      [{ segment: A, verdict: 'accepted', confidencePct: 91 }],
      A, pool,
    ) as StoredCandidateSet;

    check('case4: near-duplicate of the tried timestamp is not re-offered as an extra', entry.candidates.length === 1);
  }

  // ── Case 5: nothing tried and no pool alternates -> null (nothing to persist) ──
  {
    const entry = buildCandidateHistoryEntry(4, RANGE, [], null, undefined);
    check('case5: returns null when there is nothing to show', entry === null);
  }

  // ── Case 6: CANDIDATES_MAX cap still applies to the untried-extras portion ──
  {
    const A = mk(100, 91);
    const manyExtras = Array.from({ length: 15 }, (_, i) => mk(1000 + i * 20, 50 + i));
    const pool = [A, ...manyExtras];

    const entry = buildCandidateHistoryEntry(
      5, RANGE,
      [{ segment: A, verdict: 'accepted', confidencePct: 91 }],
      A, pool,
    ) as StoredCandidateSet;

    check('case6: extras capped at 10 (1 tried + 10 extras = 11 total)', entry.candidates.length === 11);
  }

  console.log(failures === 0 ? '\n✅ ALL CHECKS PASSED' : `\n❌ ${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
