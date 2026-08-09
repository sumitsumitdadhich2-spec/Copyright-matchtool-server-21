/**
 * Retry/replace loop that gates matched segments through VLM scene
 * verification. Runs strictly AFTER groundMatchedSegments() has already
 * produced its candidates — it never touches hash-matching speed or accuracy.
 */
import type { MatchedSegment } from './matching-engine';
import { getAlternateCandidatesForRange } from './matching-engine';
import {
  verifySegmentByVideo,
  VLM_CONFIDENCE_THRESHOLD,
  VLM_MAX_ATTEMPTS,
  VLM_CONCURRENCY,
} from './vlm-verify';
import { rankCandidatesCropRobust } from './candidate-embedding-rank';
import { geminiConfigured } from './gemini-vlm';
import { degenerateCandidateReason } from './degenerate-guard';
import { broaderSearchForRange, BROADER_SEARCH_MAX_NEW, SAME_LOCATION_TOLERANCE } from './broader-search';
import { describeTargetClip, orderCandidatesByMode, RankMode } from './clip-description';

/**
 * How many candidates the fast pre-filters (SSCD/CLIP embeddings) build and
 * rank per segment BEFORE Gemini verification starts. The embeddings never
 * accept/reject anything — they only pick WHICH frames Gemini looks at first.
 */
const CANDIDATE_POOL_TARGET = Number(process.env.CANDIDATE_POOL_TARGET) || 10;

/**
 * FIRST-PASS BROADER SEARCH (bug fix): when the strongest candidate in a
 * segment's initial pool has a hash confidence below this threshold, the
 * true scene very likely isn't in the pool at all (observed bug: best
 * initial candidate 75.6% while the real scene — found only by a manual
 * Retry's broader search — scored 87.3%). In that case one broader-search
 * round runs immediately, BEFORE any Gemini attempt, so the main pass can
 * find the real location on the first try. Requires the caller to supply
 * the fingerprint file paths; without them behavior is unchanged.
 */
const LOW_CONFIDENCE_BROADEN_THRESHOLD =
  Number(process.env.LOW_CONFIDENCE_BROADEN_THRESHOLD) || 80;

/**
 * AUTO-EXTEND (candidate-system upgrade): when ALL of the first
 * VLM_MAX_ATTEMPTS (10) candidates are genuinely rejected, the segment does
 * NOT stop there anymore. Gemini writes a detailed description of the target
 * clip, auto-selects the most reliable ranking signal for it (hash /
 * embedding / combined — see clip-description.ts), and broader-search rounds
 * keep discovering NEW movie locations which are verified in that ranked
 * order — up to this HARD TOTAL CAP of verifications per segment (first 10
 * included). The first-10 flow itself is byte-for-byte unchanged; this phase
 * is purely additive and only runs after it.
 */
export const VLM_TOTAL_MAX_ATTEMPTS =
  Number(process.env.VLM_TOTAL_MAX_ATTEMPTS) || 30;

// Fixed batch size for VLM server cache-reset points. Purely a cache-hygiene
// boundary — does not change segment order, candidate selection, or verdicts.
const VLM_RESET_BATCH_SIZE = 48;

/**
 * Infrastructure-failure retry policy (BUG FIX): when Gemini yields no
 * verdict (quota exhausted / network failure) or the segment cut/upload
 * throws, the SAME candidate is retried with exponential backoff before
 * giving up on it — and the loop then CONTINUES to the next ranked candidate
 * instead of stopping. Infrastructure retries never consume the
 * VLM_MAX_ATTEMPTS candidate budget (only content rejections do), and an
 * unverified candidate is NEVER assigned as the segment's accepted answer.
 */
const VLM_INFRA_RETRIES = Number(process.env.VLM_INFRA_RETRIES) || 3;
const VLM_INFRA_BACKOFF_MS: number[] = (process.env.VLM_INFRA_BACKOFF_MS || '2000,8000,30000')
  .split(',')
  .map((s) => Math.max(0, Number(s.trim()) || 0));

function infraBackoffDelay(retry: number): number {
  return VLM_INFRA_BACKOFF_MS[Math.min(retry, VLM_INFRA_BACKOFF_MS.length - 1)] ?? 2000;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export interface VlmProgressInfo {
segmentIndex: number;
totalSegments: number;
attempt: number;
verdict: 'accepted' | 'rejected' | 'unverifiable' | 'dropped';
/**
 * Which verification phase emitted this event (additive, optional):
 *  - 'initial'     — the untouched first-10 loop (budget VLM_MAX_ATTEMPTS).
 *  - 'deep-search' — the auto-extend loop after every initial candidate was
 *                    rejected (budget VLM_TOTAL_MAX_ATTEMPTS).
 * Display-only; never consulted by any accept/reject decision.
 */
phase?: 'initial' | 'deep-search';
/** Total per-segment verification budget for the current phase. */
totalBudget?: number;
}

/**
 * Pick the representative (shortTime, movieTime) pair to verify for a segment.
 * `bestFrameDetail` carries per-channel similarity scores but not timestamps,
 * so the actual frame pair always comes from the matchSequence midpoint
 * (falling back to the segment bounds if the sequence is empty).
 */
export function pickRepresentativeFrames(seg: MatchedSegment): { shortTime: number; movieTime: number } {
  const mid = seg.matchSequence[Math.floor(seg.matchSequence.length / 2)];
  if (mid) return { shortTime: mid.shortTime, movieTime: mid.movieTime };
  return { shortTime: seg.shortStart, movieTime: seg.movieStart };
}

/**
 * Minimum spacing (seconds, on the short-clip timeline) between the frame
 * pairs sent to the VLM for one segment — prevents sending three
 * near-identical frames from the same instant, which would add tokens
 * without adding any real cross-check value.
 */
const PAIR_MIN_SPACING_S = 0.75;

/**
 * Pick up to 3 (shortTime, movieTime) pairs for VLM verification of one
 * segment, all drawn from timestamps the hash matcher already computed —
 * no video re-scan:
 *  1. the segment midpoint (identical to pickRepresentativeFrames — the
 *     exact pair the old single-pair flow verified),
 *  2. the HIGHEST-similarity frame pair in the segment's matchSequence,
 *  3. the SECOND-highest-similarity pair,
 * deduplicated so pairs are at least PAIR_MIN_SPACING_S apart on the short
 * timeline. Falls back gracefully to fewer pairs (down to 1) for very short
 * or sparse segments.
 */
export function pickVerificationFramePairs(
  seg: MatchedSegment,
): Array<{ shortTime: number; movieTime: number }> {
  const pairs: Array<{ shortTime: number; movieTime: number }> = [];
  const addIfSpaced = (p: { shortTime: number; movieTime: number }) => {
    if (pairs.length >= 3) return;
    if (pairs.some(q => Math.abs(q.shortTime - p.shortTime) < PAIR_MIN_SPACING_S)) return;
    pairs.push(p);
  };

  // 1. Midpoint — keeps the old behavior's pair always included.
  const mid = seg.matchSequence[Math.floor(seg.matchSequence.length / 2)];
  if (mid) {
    pairs.push({ shortTime: mid.shortTime, movieTime: mid.movieTime });
  } else {
    pairs.push({ shortTime: seg.shortStart, movieTime: seg.movieStart });
  }

  // 2 & 3. Highest- and second-highest-similarity pairs from the sequence.
  const bySimilarity = [...seg.matchSequence].sort((a, b) => b.similarity - a.similarity);
  for (const f of bySimilarity) {
    if (pairs.length >= 3) break;
    addIfSpaced({ shortTime: f.shortTime, movieTime: f.movieTime });
  }

  return pairs;
}

/**
 * Resolve a set of matched segments through VLM verification. Rejected
 * candidates are replaced with the next-best alternative for the same
 * short-clip range (drawn from `candidatePool`, never a small time-shift of
 * the same rejected spot). After VLM_MAX_ATTEMPTS rejections for a range (or
 * no more candidates), the range is dropped from the result entirely.
 *
 * Gracefully returns `segments` unchanged if the VLM endpoint is not
 * configured or unreachable — the tool must keep working with the AWS GPU
 * server off.
 */
export interface SegmentCandidateAttempt {
  /** The candidate segment that was actually run through VLM at this attempt. */
  segment: MatchedSegment;
  verdict: 'accepted' | 'rejected' | 'unverifiable';
  /** Present whenever the VLM call itself returned a parsed result (accepted or rejected). */
  confidencePct?: number;
  /** Gemini-derived 0-100 "is this a real match" score (see vlm-verify.ts). */
  matchLikelihood?: number;
  /** Concrete shared/contradictory details Gemini cited. */
  evidence?: string[];
}

export interface SegmentResolvedInfo {
  segmentIndex: number;
  original: MatchedSegment;
  /** Every candidate this pass actually ran through VLM for this range, in attempt order. */
  triedCandidates: SegmentCandidateAttempt[];
  /** The candidate ultimately kept for this range, or null if the range was dropped entirely. */
  accepted: MatchedSegment | null;
  /**
   * When `accepted` is null and the auto-extend phase spent the segment's
   * FULL total budget (VLM_TOTAL_MAX_ATTEMPTS) without a genuine accept:
   * index into `triedCandidates` of the highest-match-likelihood candidate —
   * the single "best effort" the UI shows for this range (clearly NOT
   * VLM-confirmed). All rejected candidates stay in the history untouched.
   */
  bestEffortIndex?: number;
  /** Gemini's target-clip description, when the auto-extend phase generated one. */
  clipDescription?: string;
  /** Ranking signal Gemini auto-selected for this clip (hash/embedding/combined). */
  recommendedMode?: RankMode;
}

export async function resolveSegmentsWithVLM(
  segments: MatchedSegment[],
  shortVideoPath: string,
  movieVideoPath: string,
  candidatePool: MatchedSegment[] | undefined,
  onProgress?: (info: VlmProgressInfo) => void,
  /**
   * Optional side-effect fired once per segment, the instant this pass has a
   * final verdict for it (accepted immediately, accepted after retries, or
   * dropped after VLM_MAX_ATTEMPTS/no more candidates). Purely additive —
   * does not change which segments are accepted/dropped, their order, or
   * timing. Used by server.ts to persist full candidate-comparison history
   * (for every segment, not only dropped ones) and to kick off background
   * candidate discovery for a later deferred recovery pass; never awaited here.
   */
  onSegmentResolved?: (info: SegmentResolvedInfo) => void,
  /**
   * Optional fingerprint file paths. When supplied, a segment whose initial
   * candidate pool looks weak (top hash confidence below
   * LOW_CONFIDENCE_BROADEN_THRESHOLD) gets ONE broader-search round up front
   * to pull in movie locations the original scan missed — before any Gemini
   * attempt is spent. Purely additive: strong pools are untouched.
   */
  fingerprintPaths?: { shortResultPath: string; movieResultPath: string },
): Promise<MatchedSegment[]> {
  if (segments.length === 0) return segments;

  // ------------------------------------------------------------------------
  // Provider availability: Gemini is the ONLY verdict-maker now. The fast
  // pre-filters (SSCD/CLIP embeddings) only build/rank candidates and can
  // never verify anything on their own — so without Gemini, skip the pass.
  // ------------------------------------------------------------------------
  if (!geminiConfigured()) {
    console.warn(
      `[VLM] Skipping verification pass — Gemini not available ` +
      `(GEMINI_API_KEY ${process.env.GEMINI_API_KEY ? 'set but unusable' : 'unset'})`
    );
    return segments;
  }

  // Slot for each segment's final outcome, filled in whatever order segments
  // within a batch happen to finish — collected and sorted at the end, so
  // completion order never affects the returned result.
  const outcomes: (MatchedSegment | null)[] = new Array(segments.length).fill(null);

  /**
   * Runs the exact same retry/replace loop as before for a single segment.
   * Segments are fully independent of each other: each only reads its own
   * `original` bounds and the shared, read-only `candidatePool` — so running
   * several of these concurrently changes nothing about which candidates are
   * tried, in what preference order, or what verdict each one gets. Only
   * wall-clock time changes.
   */
  async function resolveOneSegment(i: number): Promise<void> {
    const original = segments[i];
    const triedCandidates: SegmentCandidateAttempt[] = [];
    let attempt = 0;
    let accepted: MatchedSegment | null = null;

    // ------------------------------------------------------------------
    // CANDIDATE GENERATION — the fast pre-filters' ONLY job now.
    // Build up to CANDIDATE_POOL_TARGET candidates for this short-clip
    // range upfront (the original + alternates from the pool, deduped by
    // movie location), then rank them with the crop-robust SSCD/CLIP
    // embedding ranker so Gemini — the one and only verdict-maker — checks
    // the most promising frames first. Embeddings never accept/reject
    // anything here; ranking failure just keeps the confidence order.
    // ------------------------------------------------------------------
    const candidates: MatchedSegment[] = [original];
    const alternates = getAlternateCandidatesForRange(
      candidatePool,
      original.shortStart,
      original.shortEnd,
      [original.movieStart],
    );
    for (const alt of alternates) {
      if (candidates.length >= CANDIDATE_POOL_TARGET) break;
      if (candidates.some(c => Math.abs(c.movieStart - alt.movieStart) <= 0.5)) continue;
      candidates.push(alt);
    }

    // ------------------------------------------------------------------
    // FIRST-PASS BROADER SEARCH (bug fix): if even the STRONGEST candidate
    // in the pool has a weak hash confidence, the real scene probably isn't
    // in the pool at all. Run one broader-search round NOW (same primitive
    // the manual Retry uses) and merge any genuinely new movie locations,
    // so Gemini gets a chance to see the true scene on the first attempt
    // instead of only after a manual Retry.
    // ------------------------------------------------------------------
    if (fingerprintPaths) {
      const topConfidence = Math.max(...candidates.map(c => c.confidence));
      if (topConfidence < LOW_CONFIDENCE_BROADEN_THRESHOLD) {
        try {
          console.log(
            `[VLM] seg${i}: weak initial pool (top hash confidence ` +
            `${topConfidence.toFixed(1)} < ${LOW_CONFIDENCE_BROADEN_THRESHOLD}) — ` +
            `running first-pass broader search`
          );
          const fresh = (await broaderSearchForRange(
            original.shortStart,
            original.shortEnd,
            fingerprintPaths.shortResultPath,
            fingerprintPaths.movieResultPath,
            0,
          ))
            .filter(seg => !candidates.some(c => Math.abs(c.movieStart - seg.movieStart) <= SAME_LOCATION_TOLERANCE))
            .slice(0, BROADER_SEARCH_MAX_NEW);
          if (fresh.length > 0) {
            console.log(
              `[VLM] seg${i}: first-pass broader search added ${fresh.length} new candidate(s) ` +
              `(best new confidence ${fresh[0].confidence.toFixed(1)})`
            );
            candidates.push(...fresh);
          }
        } catch (err: any) {
          // Broadening is best-effort only — never fail the segment over it.
          console.warn(`[VLM] seg${i}: first-pass broader search failed: ${err?.message || err}`);
        }
      }
    }

    let order = candidates.map((_, k) => k);
    if (candidates.length > 1) {
      try {
        const ranked = await rankCandidatesCropRobust(
          candidates.map((segment) => ({ segment })),
          order,
          shortVideoPath,
          movieVideoPath,
          'MainPassRank',
        );
        if (ranked) order = ranked;
      } catch {
        // Ranking is best-effort only — keep hash-confidence order.
      }
    }

    // ------------------------------------------------------------------
    // VERIFICATION — Gemini only, in ranked order, up to VLM_MAX_ATTEMPTS.
    // ------------------------------------------------------------------
    let sawUnverifiable = false;

    for (const candIdx of order) {
      if (attempt >= VLM_MAX_ATTEMPTS) break;
      const candidate = candidates[candIdx];

      // ----------------------------------------------------------------
      // DEGENERATE-CANDIDATE GUARD (bug fix): a structurally impossible
      // mapping — near-zero speedRatio / frozen matchSequence (every short
      // frame stuck on one movie instant) — can NEVER be a real copy, yet
      // Gemini may still accept it at high confidence when the movie has a
      // visually similar duplicate scene. Auto-reject WITHOUT spending a
      // Gemini call or a candidate-budget slot, and never let one become
      // the accepted answer.
      // ----------------------------------------------------------------
      const degenerateReason = degenerateCandidateReason(candidate);
      if (degenerateReason) {
        console.log(`[VLM] seg${i}: auto-rejected degenerate candidate (${degenerateReason})`);
        triedCandidates.push({
          segment: candidate,
          verdict: 'rejected',
          matchLikelihood: 0,
          evidence: [`Auto-rejected without VLM: ${degenerateReason}`],
        });
        onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'rejected', phase: 'initial', totalBudget: VLM_MAX_ATTEMPTS });
        continue;
      }

      attempt++;

      // ----------------------------------------------------------------
      // BUG FIX: infrastructure failures (Gemini no-verdict / cut-upload
      // errors) retry the SAME candidate with backoff instead of stopping
      // the whole loop. Retries do NOT consume the candidate budget
      // (`attempt` is incremented once per candidate above). If all
      // retries fail, the candidate is recorded as unverifiable and the
      // loop CONTINUES to the next ranked candidate. An unverified
      // candidate is never assigned to `accepted`.
      // ----------------------------------------------------------------
      let candidateDone = false;
      for (let retry = 0; retry <= VLM_INFRA_RETRIES && !candidateDone; retry++) {
        if (retry > 0) {
          const delayMs = infraBackoffDelay(retry - 1);
          console.log(
            `[VLM] seg${i}#${attempt} retry ${retry}/${VLM_INFRA_RETRIES} after ${delayMs}ms backoff`
          );
          await sleep(delayMs);
        }

        let infraFailReason: string | null = null;
        try {
          // Cut BOTH matched segments out of their source videos and send
          // the two real clips to Gemini in one request — no still frames.
          const result = await verifySegmentByVideo(
            shortVideoPath,
            movieVideoPath,
            candidate,
            `VLM seg${i}#${attempt}${retry > 0 ? `r${retry}` : ''}`,
          );

          if (result === null) {
            // Gemini could not produce a verdict (quota exhausted /
            // network failure / malformed response) — infrastructure
            // failure, not a content rejection.
            infraFailReason = 'no-verdict (quota/network/malformed)';
          } else if (result.same && result.confidencePct >= VLM_CONFIDENCE_THRESHOLD) {
            accepted = candidate;
            triedCandidates.push({
              segment: candidate,
              verdict: 'accepted',
              confidencePct: result.confidencePct,
              matchLikelihood: result.matchLikelihood,
              evidence: result.evidence,
            });
            onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'accepted', phase: 'initial', totalBudget: VLM_MAX_ATTEMPTS });
            candidateDone = true;
          } else {
            triedCandidates.push({
              segment: candidate,
              verdict: 'rejected',
              confidencePct: result.confidencePct,
              matchLikelihood: result.matchLikelihood,
              evidence: result.evidence,
            });
            onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'rejected', phase: 'initial', totalBudget: VLM_MAX_ATTEMPTS });
            candidateDone = true; // content rejection — move to next candidate
          }
        } catch (err: any) {
          // Segment cut / upload failure — infrastructure failure for this
          // attempt, not a content rejection.
          infraFailReason = `cut/upload failure: ${err?.message || err}`;
        }

        if (infraFailReason !== null && !candidateDone) {
          if (retry < VLM_INFRA_RETRIES) {
            console.warn(
              `[VLM] seg${i}#${attempt} infrastructure failure (${infraFailReason}) — will retry same candidate`
            );
          } else {
            // Retries exhausted for THIS candidate only: record it as
            // unverifiable and continue to the next ranked candidate.
            console.warn(
              `[VLM] seg${i}#${attempt} unverifiable after ${VLM_INFRA_RETRIES} retries (${infraFailReason}) — ` +
              `moving on to next candidate`
            );
            sawUnverifiable = true;
            triedCandidates.push({ segment: candidate, verdict: 'unverifiable' });
            onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'unverifiable', phase: 'initial', totalBudget: VLM_MAX_ATTEMPTS });
            candidateDone = true;
          }
        }
      }

      if (accepted) break;
    }

    // ------------------------------------------------------------------
    // AUTO-EXTEND PHASE (candidate-system upgrade — purely additive; the
    // first-10 flow above is untouched). Runs ONLY when every candidate in
    // the initial pool was checked/rejected without a genuine accept:
    //   1. Gemini writes a DETAILED DESCRIPTION of the target clip and
    //      auto-selects the most reliable ranking signal for it.
    //   2. Any leftover discovered-but-unverified candidates are verified
    //      first (ordered by that signal) — cheapest wins first.
    //   3. Broader-search rounds keep surfacing NEW movie locations (each
    //      round widens the net), ranked by the same signal and verified
    //      one by one — never re-checking any already-tried location.
    //   4. Hard stop at VLM_TOTAL_MAX_ATTEMPTS (30) total verifications or
    //      when the search saturates (2 consecutive empty rounds).
    // The segment is NOT abandoned until this full budget is spent.
    // ------------------------------------------------------------------
    let clipDescription: string | undefined;
    let recommendedMode: RankMode | undefined;

    if (!accepted && fingerprintPaths && geminiConfigured() && attempt < VLM_TOTAL_MAX_ATTEMPTS) {
      console.log(
        `[VLM] seg${i}: all ${attempt} initial candidate(s) rejected — AUTO-EXTEND starts ` +
        `(budget ${attempt}/${VLM_TOTAL_MAX_ATTEMPTS} used)`
      );

      // 1. AI clip profile — description + auto-selected ranking mode.
      //    Best-effort: on failure the search still runs in 'combined' mode.
      const profile = await describeTargetClip(
        shortVideoPath, original.shortStart, original.shortEnd, 0, `ClipProfile seg${i}`,
      );
      const mode: RankMode = profile?.recommendedMode ?? 'combined';
      clipDescription = profile?.description;
      recommendedMode = mode;

      /** Verify ONE extended candidate with the exact same degenerate guard,
       *  Gemini video call, threshold, and infra-retry policy the first-10
       *  loop uses. Returns true when the segment is done (accepted). */
      const verifyExtended = async (candidate: MatchedSegment): Promise<boolean> => {
        const degenerateReason = degenerateCandidateReason(candidate);
        if (degenerateReason) {
          console.log(`[VLM] seg${i} (extend): auto-rejected degenerate candidate (${degenerateReason})`);
          triedCandidates.push({
            segment: candidate,
            verdict: 'rejected',
            matchLikelihood: 0,
            evidence: [`Auto-rejected without VLM: ${degenerateReason}`],
          });
          onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'rejected', phase: 'deep-search', totalBudget: VLM_TOTAL_MAX_ATTEMPTS });
          return false;
        }
        attempt++;
        for (let retry = 0; retry <= VLM_INFRA_RETRIES; retry++) {
          if (retry > 0) await sleep(infraBackoffDelay(retry - 1));
          try {
            const result = await verifySegmentByVideo(
              shortVideoPath, movieVideoPath, candidate,
              `VLM seg${i}#${attempt}ext${retry > 0 ? `r${retry}` : ''}`,
            );
            if (result === null) {
              if (retry < VLM_INFRA_RETRIES) continue;
              sawUnverifiable = true;
              triedCandidates.push({ segment: candidate, verdict: 'unverifiable' });
              onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'unverifiable', phase: 'deep-search', totalBudget: VLM_TOTAL_MAX_ATTEMPTS });
              return false;
            }
            if (result.same && result.confidencePct >= VLM_CONFIDENCE_THRESHOLD) {
              accepted = candidate;
              triedCandidates.push({
                segment: candidate, verdict: 'accepted',
                confidencePct: result.confidencePct,
                matchLikelihood: result.matchLikelihood,
                evidence: result.evidence,
              });
              onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'accepted', phase: 'deep-search', totalBudget: VLM_TOTAL_MAX_ATTEMPTS });
              return true;
            }
            triedCandidates.push({
              segment: candidate, verdict: 'rejected',
              confidencePct: result.confidencePct,
              matchLikelihood: result.matchLikelihood,
              evidence: result.evidence,
            });
            onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'rejected', phase: 'deep-search', totalBudget: VLM_TOTAL_MAX_ATTEMPTS });
            return false;
          } catch (err: any) {
            if (retry < VLM_INFRA_RETRIES) continue;
            sawUnverifiable = true;
            triedCandidates.push({ segment: candidate, verdict: 'unverifiable' });
            onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'unverifiable', phase: 'deep-search', totalBudget: VLM_TOTAL_MAX_ATTEMPTS });
            return false;
          }
        }
        return false;
      };

      // Every movie location this segment has already considered — never
      // re-check any of them.
      const triedLocations = () => [
        ...candidates.map(c => c.movieStart),
        ...triedCandidates.map(t => t.segment.movieStart),
      ];

      // 2. Leftover discovered-but-unverified candidates first (the initial
      //    pool can be larger than the first-10 budget), in mode order.
      //    IMPORTANT: the capped `candidates` array (CANDIDATE_POOL_TARGET)
      //    may have LEFT OUT alternates the matching engine already
      //    discovered — pull those from the full candidatePool too, so the
      //    deep-search phase spends its budget on every known location
      //    before paying for broader-search re-scans.
      const poolLeftovers = getAlternateCandidatesForRange(
        candidatePool,
        original.shortStart,
        original.shortEnd,
        candidates.map(c => c.movieStart),
      );
      for (const alt of poolLeftovers) {
        if (candidates.some(c => Math.abs(c.movieStart - alt.movieStart) <= SAME_LOCATION_TOLERANCE)) continue;
        candidates.push(alt);
      }
      const verifiedStarts = new Set(triedCandidates.map(t => t.segment.movieStart));
      const leftoverIdxs = candidates
        .map((c, k) => (verifiedStarts.has(c.movieStart) ? -1 : k))
        .filter(k => k !== -1);
      if (leftoverIdxs.length > 0) {
        const orderedLeftovers = await orderCandidatesByMode(
          mode, candidates.map(segment => ({ segment })), leftoverIdxs,
          shortVideoPath, movieVideoPath, `ExtendRank seg${i}`,
        );
        for (const k of orderedLeftovers) {
          if (accepted || attempt >= VLM_TOTAL_MAX_ATTEMPTS) break;
          await verifyExtended(candidates[k]);
        }
      }

      // 3. Broader-search rounds for genuinely NEW movie locations.
      let extRound = 1; // round 0 may already have run as the first-pass broaden
      let emptyRounds = 0;
      while (!accepted && attempt < VLM_TOTAL_MAX_ATTEMPTS && emptyRounds < 2) {
        let fresh: MatchedSegment[] = [];
        try {
          const known = triedLocations();
          fresh = (await broaderSearchForRange(
            original.shortStart, original.shortEnd,
            fingerprintPaths.shortResultPath, fingerprintPaths.movieResultPath,
            extRound,
          ))
            .filter(seg => !known.some(t => Math.abs(t - seg.movieStart) <= SAME_LOCATION_TOLERANCE))
            .slice(0, BROADER_SEARCH_MAX_NEW);
        } catch (err: any) {
          console.warn(`[VLM] seg${i} (extend): broader-search round ${extRound} failed: ${err?.message || err}`);
        }
        extRound++;

        if (fresh.length === 0) { emptyRounds++; continue; }
        emptyRounds = 0;
        console.log(`[VLM] seg${i} (extend): round ${extRound - 1} surfaced ${fresh.length} new candidate(s)`);

        // Track them so later rounds/dedupe see them, then verify in mode order.
        const baseIdx = candidates.length;
        candidates.push(...fresh);
        const freshIdxs = fresh.map((_, k) => baseIdx + k);
        const orderedFresh = await orderCandidatesByMode(
          mode, candidates.map(segment => ({ segment })), freshIdxs,
          shortVideoPath, movieVideoPath, `ExtendRank seg${i}`,
        );
        for (const k of orderedFresh) {
          if (accepted || attempt >= VLM_TOTAL_MAX_ATTEMPTS) break;
          await verifyExtended(candidates[k]);
        }
      }

      if (!accepted) {
        console.log(
          `[VLM] seg${i}: auto-extend exhausted (${attempt}/${VLM_TOTAL_MAX_ATTEMPTS} verifications) ` +
          `without a genuine accept — best-effort fallback will be surfaced`
        );
      }
    }

    // ------------------------------------------------------------------
    // BEST-EFFORT PICK: when the FULL budget ran out without a genuine
    // accept, the single highest-match-likelihood (non-degenerate) checked
    // candidate is surfaced as the range's visible "best effort" — clearly
    // NOT VLM-confirmed. Every rejected candidate stays in history exactly
    // as before; nothing is hidden or deleted.
    // ------------------------------------------------------------------
    let bestEffortIndex: number | undefined;
    if (!accepted && triedCandidates.length > 0) {
      let bestScore = -1;
      triedCandidates.forEach((t, k) => {
        if (degenerateCandidateReason(t.segment)) return;
        const score = typeof t.matchLikelihood === 'number' ? t.matchLikelihood : -1;
        if (score > bestScore) { bestScore = score; bestEffortIndex = k; }
      });
    }

    if (accepted) {
      outcomes[i] = accepted;
    } else {
      console.log(
        `[VLM] No genuine match found after ${attempt} attempt(s) for short-clip range ` +
        `[${original.shortStart.toFixed(2)}s–${original.shortEnd.toFixed(2)}s]` +
        (sawUnverifiable
          ? ' (some candidates unverifiable — eligible for deferred recovery) — dropping from main pass.'
          : ' — dropping.')
      );
      onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'dropped' });
    }

    // Fires for EVERY segment (accepted first try, accepted after retries, or
    // dropped) so the caller can persist full candidate-comparison history —
    // not just what happened to previously-dropped segments.
    onSegmentResolved?.({
      segmentIndex: i, original, triedCandidates, accepted,
      bestEffortIndex, clipDescription, recommendedMode,
    });
  }

  /**
   * Runs `indices` with up to VLM_CONCURRENCY segments in flight at once.
   * Each worker pulls the next unclaimed index off the shared cursor, so
   * segments that finish quickly (fewer retries) immediately pick up the
   * next one instead of waiting for the slowest one in a fixed pairing.
   */
  async function runPool(indices: number[]): Promise<void> {
    let cursor = 0;
    const workerCount = Math.min(VLM_CONCURRENCY, indices.length);
    const workers = Array.from({ length: workerCount }, async () => {
      while (cursor < indices.length) {
        const idx = indices[cursor++];
        await resolveOneSegment(idx);
      }
    });
    await Promise.all(workers);
  }

  // Process in fixed-size batches with up to VLM_CONCURRENCY segments within
  // each batch running at once. (The batch boundary is a historical artifact
  // of the removed Qwen server's cache-reset points — kept as a harmless
  // pacing boundary, no reset calls are made anymore.)
  for (let batchStart = 0; batchStart < segments.length; batchStart += VLM_RESET_BATCH_SIZE) {
    const batchEnd = Math.min(batchStart + VLM_RESET_BATCH_SIZE, segments.length);
    const indices = Array.from({ length: batchEnd - batchStart }, (_, k) => batchStart + k);
    await runPool(indices);
  }

  const resolved = outcomes.filter((s): s is MatchedSegment => s !== null);
  resolved.sort((a, b) => a.shortStart - b.shortStart);
  return resolved;
}
