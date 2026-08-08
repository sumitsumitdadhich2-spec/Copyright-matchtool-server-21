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
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { MatchedSegment, matchVideosFromFiles } from './candidate-matching-engine';
import { rankCandidatesCropRobust } from './candidate-embedding-rank';
import { verifySegmentByVideo, VLM_CONFIDENCE_THRESHOLD, VLM_MAX_ATTEMPTS } from './vlm-verify';
import {
  StoredCandidateSet,
  CandidateCheck,
  readCandidatesFile,
  writeCandidatesFileSync,
} from './candidate-recovery';

/** Cap on how many freshly-discovered candidates a single broader search appends. */
const BROADER_SEARCH_MAX_NEW = 10;
/** Deliberately looser than the app default (82) — this is a fallback pass
 *  whose only job is to surface locations the original scan missed; VLM
 *  verification below is still the real accept/reject gate. */
const BROADER_SEARCH_MIN_SIMILARITY = 40;
const BROADER_SEARCH_MIN_CONSECUTIVE_FRAMES = 3;
/** Wider than the app default (3) so a slightly different edit speed/frame
 *  offset than the original scan assumed can still line up. */
const BROADER_SEARCH_FRAME_DRIFT = 8;
/** Small amount of extra context on each side of the short-clip range so the
 *  scene-chunk splitter has more than a couple of frames to work with. */
const BROADER_SEARCH_PAD_SECONDS = 1.5;
/** Same drift tolerance used elsewhere (getAlternateCandidatesForRange) to
 *  treat two movie timestamps as "the same location". */
const SAME_LOCATION_TOLERANCE = 0.5;

/**
 * Hard cap on Gemini verifications a SINGLE Retry click may spend, across as
 * many broader-search rounds as it takes. Once spent without a genuine accept,
 * the best-so-far candidate is accepted as a fallback; clicking Retry again
 * starts a fresh budget and keeps hunting for NEW candidates.
 */
const RETRY_MAX_ATTEMPTS = Number(process.env.RETRY_MAX_ATTEMPTS) || VLM_MAX_ATTEMPTS;

/**
 * Each broader-search round widens the net a little more, so repeat Retry
 * clicks surface genuinely new movie locations instead of rediscovering the
 * same ones. Round 0 uses the base values above.
 */
function broaderSearchParamsForRound(round: number): {
  minSimilarity: number;
  minConsecutiveFrames: number;
  frameDrift: number;
  padSeconds: number;
} {
  const step = Math.max(0, round);
  return {
    // Floor at 20 — below that the engine returns mostly noise.
    minSimilarity: Math.max(20, BROADER_SEARCH_MIN_SIMILARITY - step * 5),
    minConsecutiveFrames: Math.max(2, BROADER_SEARCH_MIN_CONSECUTIVE_FRAMES - (step > 0 ? 1 : 0)),
    frameDrift: Math.min(24, BROADER_SEARCH_FRAME_DRIFT + step * 4),
    padSeconds: Math.min(6, BROADER_SEARCH_PAD_SECONDS + step * 0.75),
  };
}

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
 * Write a temporary fingerprint file containing only the short-clip frames
 * within [start-pad, end+pad], preserving their original timestamps, so a
 * fresh matchVideosFromFiles() call can be scoped to just this range without
 * touching the real per-job fingerprint files on disk.
 */
async function writeFilteredShortFingerprint(
  shortResultPath: string,
  rangeStart: number,
  rangeEnd: number,
): Promise<string | null> {
  const tempPath = path.join(
    os.tmpdir(),
    `retry-broader-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`,
  );
  const rl = readline.createInterface({
    input: fs.createReadStream(shortResultPath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });
  const lines: string[] = [];
  for await (const line of rl) {
    if (!line.trim()) continue;
    try {
      const frame = JSON.parse(line);
      if (typeof frame.timestamp === 'number' && frame.timestamp >= rangeStart && frame.timestamp <= rangeEnd) {
        lines.push(line);
      }
    } catch { /* skip corrupt line */ }
  }
  if (lines.length === 0) return null;
  await fs.promises.writeFile(tempPath, lines.join('\n') + '\n');
  return tempPath;
}

/**
 * Re-run the real matching engine restricted to this segment's short-clip
 * range, with a wider drift + relaxed similarity floor, to surface movie
 * locations the original full-clip scan didn't. Reuses matchVideosFromFiles
 * verbatim (same chunked/full-load RAM-safety logic, same scene-chunk
 * scan) — this is not a second scoring algorithm.
 */
async function runBroaderSearch(
  entry: StoredCandidateSet,
  shortResultPath: string,
  movieResultPath: string,
  round: number,
): Promise<MatchedSegment[]> {
  const params = broaderSearchParamsForRound(round);
  const rangeStart = Math.max(0, entry.shortStart - params.padSeconds);
  const rangeEnd = entry.shortEnd + params.padSeconds;

  const tempShortPath = await writeFilteredShortFingerprint(shortResultPath, rangeStart, rangeEnd);
  if (!tempShortPath) return [];

  try {
    const result = await matchVideosFromFiles(tempShortPath, movieResultPath, {
      minSimilarity: params.minSimilarity,
      minConsecutiveFrames: params.minConsecutiveFrames,
      frameDrift: params.frameDrift,
    });

    const pool = [...(result.segments || []), ...(result.candidatePool || [])];
    const overlapping = pool.filter(seg =>
      Math.min(seg.shortEnd, entry.shortEnd) - Math.max(seg.shortStart, entry.shortStart) > 0.15,
    );

    const deduped: MatchedSegment[] = [];
    for (const seg of overlapping) {
      if (!deduped.some(d => Math.abs(d.movieStart - seg.movieStart) <= SAME_LOCATION_TOLERANCE)) {
        deduped.push(seg);
      }
    }
    deduped.sort((a, b) => b.confidence - a.confidence);
    return deduped;
  } finally {
    fs.promises.unlink(tempShortPath).catch(() => { /* best-effort cleanup */ });
  }
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
    // Crop-robust embedding ranking (candidate system only): try the
    // candidates whose movie frame — full OR any left/center/right 9:16
    // window of it — looks most like the short frame FIRST, so a vertically
    // cropped short doesn't burn Gemini attempts on wrong locations.
    // Reorders only; on failure (model unavailable etc.) the order is kept.
    let order = idxs;
    try {
      const ranked = await rankCandidatesCropRobust(
        entry.candidates,
        idxs,
        shortVideoPath,
        movieVideoPath,
        'CandidateRetry',
      );
      if (ranked) order = ranked;
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
