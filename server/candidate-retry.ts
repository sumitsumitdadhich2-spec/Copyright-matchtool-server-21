/**
 * Manual, user-triggered "Retry" flow for a single segment in the preview UI.
 * Purely additive on top of the automatic pipeline — never touches
 * groundMatchedSegments()/matchVideosFromFiles() decision logic, the main
 * VLM pass (vlm-segment-resolver.ts), or the deferred recovery pass
 * (deferred-recovery.ts). Those stay completely untouched; this module only
 * consumes their outputs (candidate-recovery.ts's disk-backed
 * StoredCandidateSet) and reuses the same verification primitives
 * (vlm-verify.ts) and the same hash-matching engine (matching-engine.ts) —
 * no second/different similarity-scoring algorithm.
 *
 * Two modes, chosen automatically based on what's left to try for the
 * segment (never based on click count, since the caller doesn't track that):
 *
 *  - `unchecked_pool`: the segment's already-discovered candidate pool (from
 *    background discovery during the main pass, or a previous broader
 *    search) still has entries VLM never got to. Check those first, in
 *    existing order, before doing anything more expensive.
 *  - `broader_search`: every discovered candidate has already been checked
 *    (accepted/rejected/unverifiable). Re-run the real matching engine
 *    (matchVideosFromFiles) restricted to this short-clip range with a wider
 *    frame drift + relaxed similarity floor, purely to surface movie
 *    locations the original single scan didn't surface. New candidates are
 *    appended to the same on-disk history — nothing already there is ever
 *    deleted, even if this retry fails to find an accept.
 */
import { MatchedSegment } from './candidate-matching-engine';
import { rankCandidatesCropRobust } from './candidate-embedding-rank';
import { verifySegmentByVideo, VLM_CONFIDENCE_THRESHOLD } from './vlm-verify';
import { VLM_TOTAL_MAX_ATTEMPTS } from './vlm-segment-resolver';
import { describeTargetClip, orderCandidatesByMode, RankMode } from './clip-description';
import {
  broaderSearchForRange,
  BROADER_SEARCH_MAX_NEW,
  SAME_LOCATION_TOLERANCE,
} from './broader-search';
import { degenerateCandidateReason } from './degenerate-guard';
import {
  StoredCandidateSet,
  CandidateCheck,
  readCandidatesFile,
  writeCandidatesFileSync,
} from './candidate-recovery';

/**
 * Hard cap on Gemini verifications a SINGLE Retry click may spend, across as
 * many broader-search rounds as it takes. Once spent without a genuine accept,
 * the best-so-far candidate is accepted as a fallback; clicking Retry again
 * starts a fresh budget and keeps hunting for NEW candidates.
 *
 * DEEP-SEARCH upgrade: a Retry click is the user explicitly saying "search
 * harder", so it gets a fresh 30-verification budget (same hard total the
 * main pass's auto-extend uses) instead of the old 10.
 */
const RETRY_MAX_ATTEMPTS = Number(process.env.RETRY_MAX_ATTEMPTS) || VLM_TOTAL_MAX_ATTEMPTS;

export interface RetrySegmentResult {
  /**
   *  accepted    — Gemini genuinely confirmed a candidate.
   *  best_effort — attempts ran out, so the highest-match-likelihood candidate
   *                checked so far was accepted as a fallback.
   *  exhausted   — nothing to accept at all (no candidate ever produced a
   *                usable Gemini score).
   */
  outcome: 'accepted' | 'best_effort' | 'exhausted';
  mode: 'unchecked_pool' | 'broader_search';
  acceptedCandidateIndex?: number;
  newCandidatesAdded: number;
  /** Gemini verifications this click actually spent (<= RETRY_MAX_ATTEMPTS). */
  attemptsUsed: number;
  /** Broader-search rounds this click ran. */
  searchRounds: number;
}

/**
 * Re-run the real matching engine restricted to this segment's short-clip
 * range, with a wider drift + relaxed similarity floor, to surface movie
 * locations the original full-clip scan didn't. Delegates to the shared
 * broader-search primitive (broader-search.ts) — same logic, now also used
 * by the main pass for weak initial candidate pools.
 */
function runBroaderSearch(
  entry: StoredCandidateSet,
  shortResultPath: string,
  movieResultPath: string,
  round: number,
): Promise<MatchedSegment[]> {
  return broaderSearchForRange(entry.shortStart, entry.shortEnd, shortResultPath, movieResultPath, round);
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
 * Run one Retry click for a segment. Determines internally whether this is
 * "search the unchecked pool" or "broader search" based on whether any
 * unchecked candidates remain — never based on a click counter the caller
 * would have to track. Persists every state change to disk incrementally
 * (never held in memory beyond this single call), so history survives a
 * server restart mid-retry and RAM never grows with retry count.
 */
export async function retrySegmentCandidates(
  uploadDir: string,
  matchJobId: string,
  segmentIndex: number,
  shortVideoPath: string,
  movieVideoPath: string,
  shortResultPath: string,
  movieResultPath: string,
): Promise<RetrySegmentResult> {
  const loadedEntry = readCandidatesFile(uploadDir, matchJobId, segmentIndex);
  if (!loadedEntry) throw new Error('No candidate history for this segment');
  const entry: StoredCandidateSet = loadedEntry;

  let attemptsUsed = 0;
  let searchRounds = 0;
  let newCandidatesAdded = 0;
  let acceptedIdx: number | undefined;

  const persist = () => writeCandidatesFileSync(uploadDir, matchJobId, segmentIndex, entry);

  /** Already-checked candidates are NEVER re-verified — only fresh ones. */
  const uncheckedIdxsNow = () =>
    entry.candidates.map((c, i) => (c.checked ? -1 : i)).filter(i => i !== -1);

  /**
   * Verify a batch of unchecked candidates (embedding-ranked first) until a
   * genuine accept or the click's attempt budget runs out. Every verification
   * is the same VIDEO-segment Gemini call the main pass uses; every state
   * change is persisted to disk immediately.
   */
  async function verifyBatch(idxs: number[]): Promise<void> {
    if (idxs.length === 0) return;
    // Mode-aware ranking (deep-search upgrade): when the AI clip profile has
    // auto-selected a ranking signal for this clip (hash / embedding /
    // combined), candidates are ordered by THAT signal. Without a profile,
    // the original crop-robust embedding ranking applies unchanged.
    // Reorders only; on failure (model unavailable etc.) the order is kept.
    let order = idxs;
    try {
      if (entry.recommendedMode) {
        order = await orderCandidatesByMode(
          entry.recommendedMode,
          entry.candidates,
          idxs,
          shortVideoPath,
          movieVideoPath,
          'CandidateRetry',
        );
      } else {
        const ranked = await rankCandidatesCropRobust(
          entry.candidates,
          idxs,
          shortVideoPath,
          movieVideoPath,
          'CandidateRetry',
        );
        if (ranked) order = ranked;
      }
    } catch { /* ranking is best-effort only */ }

    for (const idx of order) {
      if (attemptsUsed >= RETRY_MAX_ATTEMPTS) return;
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
        return;
      }
    }
  }

  // ------------------------------------------------------------------
  // PHASE 1 — unchecked pool: candidates already discovered but never
  // verified are checked FIRST. If a genuine match is found here, no
  // broader search runs at all (no wasted work).
  // ------------------------------------------------------------------
  const initialUnchecked = uncheckedIdxsNow();
  const mode: 'unchecked_pool' | 'broader_search' =
    initialUnchecked.length > 0 ? 'unchecked_pool' : 'broader_search';
  await verifyBatch(initialUnchecked);

  // ------------------------------------------------------------------
  // PHASE 2 — broader-search rounds: as long as no genuine accept and
  // attempt budget remains, keep hunting for NEW candidates (each round
  // widens the net via the persisted broaderSearchRounds counter, so
  // repeat Retry clicks surface genuinely new movie locations) and run
  // them through Gemini. Stops on accept, budget exhaustion, or when two
  // consecutive rounds find nothing new (search saturated).
  // ------------------------------------------------------------------
  let emptyRounds = 0;
  while (acceptedIdx === undefined && attemptsUsed < RETRY_MAX_ATTEMPTS && emptyRounds < 2) {
    const round = entry.broaderSearchRounds ?? 0;
    const existingTimestamps = entry.candidates.map(c => c.segment.movieStart);
    const fresh = (await runBroaderSearch(entry, shortResultPath, movieResultPath, round))
      .filter(seg => !existingTimestamps.some(t => Math.abs(t - seg.movieStart) <= SAME_LOCATION_TOLERANCE))
      .slice(0, BROADER_SEARCH_MAX_NEW);

    entry.broaderSearchRounds = round + 1;
    searchRounds++;

    if (fresh.length === 0) {
      emptyRounds++;
      persist(); // keep the round counter even when nothing was found
      continue;
    }
    emptyRounds = 0;
    newCandidatesAdded += fresh.length;

    const freshIdxs: number[] = [];
    for (const seg of fresh) {
      entry.candidates.push({ segment: seg, checked: false });
      freshIdxs.push(entry.candidates.length - 1);
    }
    // Persist the newly-discovered (not-yet-checked) candidates immediately,
    // even before verification runs, so the visible history grows right
    // away and nothing is lost if verification below throws.
    persist();

    await verifyBatch(freshIdxs);
  }

  if (acceptedIdx !== undefined) {
    // Genuine Gemini accept. Supersede — never delete. The previous
    // recoveredCandidateIndex simply stops being marked "★ Used"; its entry
    // stays in `candidates` exactly where it already was.
    entry.recoveredCandidateIndex = acceptedIdx;
    entry.bestEffort = false;
    entry.dropped = false;
    persist();
    return {
      outcome: 'accepted', mode, acceptedCandidateIndex: acceptedIdx,
      newCandidatesAdded, attemptsUsed, searchRounds,
    };
  }

  // ------------------------------------------------------------------
  // BEST-EFFORT FALLBACK — attempts ran out without a genuine accept:
  // accept the candidate with the HIGHEST Gemini-derived match likelihood
  // across the segment's entire checked history (this click and earlier
  // ones). Marked bestEffort so the UI can distinguish it; a later Retry
  // click starts a fresh budget, hunts new candidates again, and a genuine
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
    return {
      outcome: 'best_effort', mode, acceptedCandidateIndex: bestIdx,
      newCandidatesAdded, attemptsUsed, searchRounds,
    };
  }

  // Nothing ever produced a usable Gemini score (all unverifiable).
  return { outcome: 'exhausted', mode, newCandidatesAdded, attemptsUsed, searchRounds };
}
