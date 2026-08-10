/**
 * Manual, user-triggered "Retry" flow for a single segment in the preview UI.
 * Purely additive on top of the automatic pipeline — never touches
 * groundMatchedSegments()/matchVideosFromFiles() decision logic or the main
 * VLM pass (vlm-segment-resolver.ts). Those stay completely untouched; this
 * module only consumes their outputs (candidate-recovery.ts's disk-backed
 * StoredCandidateSet) and reuses the same verification primitives
 * (vlm-verify.ts) — no second/different similarity-scoring algorithm.
 *
 * ONE mode only: verify the segment's already-discovered candidate pool.
 * Candidates are discovered exactly ONCE — during the main matching pass —
 * and a Retry click simply verifies the next unchecked candidates from that
 * pool (embedding-ranked order) until an accept or the pool runs out. Retry
 * NEVER triggers a new scan of the movie.
 */
import { rankCandidatesCropRobust } from './candidate-embedding-rank';
import { sortIndexesByGreenScore, greenScoreLogTag } from './candidate-green-score';
import { verifySegmentByVideo, VLM_CONFIDENCE_THRESHOLD, VLM_MAX_ATTEMPTS } from './vlm-verify';
import { orderCandidatesByMode } from './clip-description';
import { degenerateCandidateReason } from './degenerate-guard';
import {
  StoredCandidateSet,
  CandidateCheck,
  readCandidatesFile,
  writeCandidatesFileSync,
} from './candidate-recovery';

/**
 * Hard cap on Gemini verifications a SINGLE Retry click may spend. Once spent
 * without a genuine accept, the best-so-far candidate is accepted as a
 * fallback; clicking Retry again starts a fresh budget on the remaining
 * unchecked pool.
 */
const RETRY_MAX_ATTEMPTS = Number(process.env.RETRY_MAX_ATTEMPTS) || VLM_MAX_ATTEMPTS;

export interface RetrySegmentResult {
  /**
   *  accepted    — Gemini genuinely confirmed a candidate.
   *  best_effort — attempts/pool ran out, so the highest-match-likelihood
   *                candidate checked so far was accepted as a fallback.
   *  exhausted   — nothing to accept at all (no candidate ever produced a
   *                usable Gemini score).
   */
  outcome: 'accepted' | 'best_effort' | 'exhausted';
  acceptedCandidateIndex?: number;
  /** Gemini verifications this click actually spent (<= RETRY_MAX_ATTEMPTS). */
  attemptsUsed: number;
}

/** Verify one candidate via the exact same VIDEO-segment Gemini call and
 *  threshold everything else uses (main pass + deferred recovery): both
 *  matched segments are CUT out of their source videos and sent to Gemini
 *  as two real clips in one request — no still frames. Mutates the
 *  candidate in place. */
async function checkCandidate(
  candidate: CandidateCheck,
  shortVideoPath: string,
  movieVideoPath: string,
  logLabel: string,
): Promise<void> {
  // Degenerate-candidate guard: a structurally impossible mapping (frozen
  // matchSequence / near-zero speedRatio) is auto-rejected WITHOUT spending
  // a Gemini call — the VLM cannot be trusted on these (a visually similar
  // duplicate scene makes it answer "same" at high confidence).
  const degenerateReason = degenerateCandidateReason(candidate.segment);
  if (degenerateReason) {
    console.log(`[${logLabel}] auto-rejected degenerate candidate: ${degenerateReason}`);
    candidate.checked = true;
    candidate.verdict = 'rejected';
    candidate.matchLikelihood = 0;
    candidate.evidence = [`Auto-rejected without VLM: ${degenerateReason}`];
    return;
  }
  try {
    const result = await verifySegmentByVideo(
      shortVideoPath,
      movieVideoPath,
      candidate.segment,
      logLabel,
    );
    candidate.checked = true;
    if (result === null) {
      candidate.verdict = 'unverifiable';
      return;
    }
    candidate.confidencePct = result.confidencePct;
    candidate.matchLikelihood = result.matchLikelihood;
    candidate.evidence = result.evidence;
    candidate.verdict =
      result.same && result.confidencePct >= VLM_CONFIDENCE_THRESHOLD
        ? 'accepted'
        : 'rejected';
  } catch {
    candidate.checked = true;
    candidate.verdict = 'unverifiable';
  }
}

/**
 * Run one Retry click for a segment: verify the next unchecked candidates
 * from the already-discovered pool (embedding-ranked first) until a genuine
 * accept, the pool runs out, or the click's attempt budget is spent. Persists
 * every state change to disk incrementally (never held in memory beyond this
 * single call), so history survives a server restart mid-retry.
 */
export async function retrySegmentCandidates(
  uploadDir: string,
  matchJobId: string,
  segmentIndex: number,
  shortVideoPath: string,
  movieVideoPath: string,
  _shortResultPath: string,
  _movieResultPath: string,
): Promise<RetrySegmentResult> {
  const loadedEntry = readCandidatesFile(uploadDir, matchJobId, segmentIndex);
  if (!loadedEntry) throw new Error('No candidate history for this segment');
  const entry: StoredCandidateSet = loadedEntry;

  let attemptsUsed = 0;
  let acceptedIdx: number | undefined;

  const persist = () => writeCandidatesFileSync(uploadDir, matchJobId, segmentIndex, entry);

  /** Already-checked candidates are NEVER re-verified — only fresh ones. */
  const uncheckedIdxs = entry.candidates
    .map((c, i) => (c.checked ? -1 : i))
    .filter(i => i !== -1);

  if (uncheckedIdxs.length > 0) {
    // Mode-aware ranking: when a previously-persisted AI clip profile
    // auto-selected a ranking signal for this clip (hash / embedding /
    // combined), candidates are ordered by THAT signal. Without a profile,
    // the original crop-robust embedding ranking applies unchanged.
    // Reorders only; on failure (model unavailable etc.) the order is kept.
    let order = uncheckedIdxs;
    try {
      if (entry.recommendedMode) {
        order = await orderCandidatesByMode(
          entry.recommendedMode,
          entry.candidates,
          uncheckedIdxs,
          shortVideoPath,
          movieVideoPath,
          'CandidateRetry',
        );
      } else {
        const ranked = await rankCandidatesCropRobust(
          entry.candidates,
          uncheckedIdxs,
          shortVideoPath,
          movieVideoPath,
          'CandidateRetry',
        );
        if (ranked) order = ranked;
      }
    } catch { /* ranking is best-effort only */ }

    // GREEN-QUALITY PRIMARY ordering: candidates with more green timeline
    // frames (similarity >= 80, the UI's own threshold) are verified FIRST.
    // Stable sort over the ranking above, so the embedding/mode order
    // survives as the SECONDARY tie-breaker. Ordering only — never a verdict.
    order = sortIndexesByGreenScore(i => entry.candidates[i]?.segment, order);
    console.log(
      `[CandidateRetry] seg${segmentIndex}: green-score verification order: ` +
      order.map(i => `#${i}(${greenScoreLogTag(entry.candidates[i]?.segment)})`).join(' > ')
    );

    for (const idx of order) {
      if (attemptsUsed >= RETRY_MAX_ATTEMPTS) break;
      attemptsUsed++;
      await checkCandidate(
        entry.candidates[idx],
        shortVideoPath,
        movieVideoPath,
        `CandidateRetry seg${segmentIndex}#${attemptsUsed}`,
      );
      persist();
      if (entry.candidates[idx].verdict === 'accepted') {
        acceptedIdx = idx;
        break;
      }
    }
  } else {
    console.log(
      `[CandidateRetry] seg${segmentIndex}: no unchecked candidates left in the pool — ` +
      `no more candidates to try`
    );
  }

  if (acceptedIdx !== undefined) {
    // Genuine Gemini accept. Supersede — never delete. The previous
    // recoveredCandidateIndex simply stops being marked "★ Used"; its entry
    // stays in `candidates` exactly where it already was.
    entry.recoveredCandidateIndex = acceptedIdx;
    entry.bestEffort = false;
    entry.dropped = false;
    persist();
    return { outcome: 'accepted', acceptedCandidateIndex: acceptedIdx, attemptsUsed };
  }

  // ------------------------------------------------------------------
  // BEST-EFFORT FALLBACK — pool/attempts ran out without a genuine accept:
  // accept the candidate with the HIGHEST Gemini-derived match likelihood
  // across the segment's entire checked history (this click and earlier
  // ones). Marked bestEffort so the UI can distinguish it; a later genuine
  // accept clears the flag.
  // ------------------------------------------------------------------
  let bestIdx = -1;
  let bestScore = -1;
  entry.candidates.forEach((c, i) => {
    // Never fall back to a structurally impossible candidate, even if an
    // earlier (pre-guard) run recorded a high VLM likelihood for it.
    if (degenerateCandidateReason(c.segment)) return;
    const score =
      typeof c.matchLikelihood === 'number'
        ? c.matchLikelihood
        : typeof c.confidencePct === 'number' && c.verdict
          ? (c.verdict === 'accepted' ? c.confidencePct : 100 - c.confidencePct)
          : undefined;
    if (score !== undefined && score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  });

  if (bestIdx !== -1) {
    entry.recoveredCandidateIndex = bestIdx;
    entry.bestEffort = true;
    entry.dropped = false;
    persist();
    console.log(
      `[CandidateRetry] seg${segmentIndex}: no genuine accept after ${attemptsUsed} attempt(s) — ` +
      `accepting best-so-far candidate ${bestIdx} (likelihood ${bestScore}) as fallback`
    );
    return { outcome: 'best_effort', acceptedCandidateIndex: bestIdx, attemptsUsed };
  }

  // Nothing ever produced a usable Gemini score (all unverifiable).
  return { outcome: 'exhausted', attemptsUsed };
}
