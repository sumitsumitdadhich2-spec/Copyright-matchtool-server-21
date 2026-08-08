/**
 * Retry/replace loop that gates matched segments through GEMINI scene
 * verification. Runs strictly AFTER groundMatchedSegments() has already
 * produced its candidates — it never touches hash-matching speed or accuracy.
 *
 * Gemini is the ONLY decision-maker here. The SSCD/embedding systems are
 * used exclusively by the candidate system (candidate discovery + crop-robust
 * ranking of the 10-candidate pool) — they never accept or reject a segment.
 */
import type { MatchedSegment } from './matching-engine';
import { getAlternateCandidatesForRange } from './matching-engine';
import {
  extractFrameAsBase64,
  verifySameSceneChecked,
  VLM_CONFIDENCE_THRESHOLD,
  VLM_MAX_ATTEMPTS,
  VLM_CONCURRENCY,
} from './vlm-verify';
import { geminiConfigured } from './gemini-vlm';

export interface VlmProgressInfo {
  segmentIndex: number;
  totalSegments: number;
  attempt: number;
  verdict: 'accepted' | 'rejected' | 'unverifiable' | 'dropped';
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
}

export interface SegmentResolvedInfo {
  segmentIndex: number;
  original: MatchedSegment;
  /** Every candidate this pass actually ran through VLM for this range, in attempt order. */
  triedCandidates: SegmentCandidateAttempt[];
  /** The candidate ultimately kept for this range, or null if the range was dropped entirely. */
  accepted: MatchedSegment | null;
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
): Promise<MatchedSegment[]> {
  if (segments.length === 0) return segments;

  // ------------------------------------------------------------------------
  // Provider availability: Gemini is the only verification provider. Skip
  // the entire pass (keeping segments unchanged) only when it is not
  // configured at all.
  // ------------------------------------------------------------------------
  if (!geminiConfigured()) {
    console.warn(
      '[Verify] Skipping verification pass — GEMINI_API_KEY is not set. ' +
      'Gemini is the only final checker; add a key to enable verification.'
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
    let candidate: MatchedSegment | undefined = original;
    const rejectedMovieTimestamps: number[] = [];
    const triedCandidates: SegmentCandidateAttempt[] = [];
    let attempt = 0;
    let accepted: MatchedSegment | null = null;

    while (candidate && attempt < VLM_MAX_ATTEMPTS) {
      attempt++;
      const triedCandidate = candidate;
      const framePairs = pickVerificationFramePairs(candidate);

      let verdict: VlmProgressInfo['verdict'] = 'unverifiable';
      try {
        // Extract every needed frame in parallel (still one ffmpeg spawn per
        // frame, all concurrent), then make ONE VLM call with all pairs —
        // same request count per attempt as the old single-pair flow.
        const extracted = await Promise.all(
          framePairs.map(async (p) => {
            const [shortFrameB64, movieFrameB64] = await Promise.all([
              extractFrameAsBase64(shortVideoPath, p.shortTime),
              extractFrameAsBase64(movieVideoPath, p.movieTime),
            ]);
            return { shortFrameB64, movieFrameB64 };
          }),
        );
        // ------------------------------------------------------------------
        // GEMINI is the only decision-maker. No frame-based gate decides
        // accept/reject anymore — the SSCD/embedding systems only build and
        // rank the candidate pool elsewhere.
        // ------------------------------------------------------------------
        const result = await verifySameSceneChecked(extracted);

        if (result === null) {
          // Gemini could not produce a verdict (unconfigured/quota/network).
          // Conservative policy: keep the candidate as 'unverifiable' rather
          // than dropping it on an infrastructure failure.
          verdict = 'unverifiable';
          accepted = candidate;
          triedCandidates.push({ segment: triedCandidate, verdict: 'unverifiable' });
          onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
          break;
        } else if (result.same && result.confidencePct >= VLM_CONFIDENCE_THRESHOLD) {
          verdict = 'accepted';
          accepted = candidate;
          triedCandidates.push({ segment: triedCandidate, verdict: 'accepted', confidencePct: result.confidencePct });
          onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
          break;
        } else {
          verdict = 'rejected';
          triedCandidates.push({ segment: triedCandidate, verdict: 'rejected', confidencePct: result.confidencePct });
          onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
        }
      } catch (err: any) {
        // Frame extraction failure (e.g. bad timestamp) — treat as
        // unverifiable for this attempt rather than crashing the match.
        console.warn(`[VLM] Frame extraction failed for segment ${i}, attempt ${attempt}: ${err?.message || err}`);
        verdict = 'unverifiable';
        accepted = candidate;
        triedCandidates.push({ segment: triedCandidate, verdict: 'unverifiable' });
        onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
        break;
      }

      // Rejected — look for the next-best alternative elsewhere in the movie,
      // never re-showing an already-rejected movie timestamp for this range.
      rejectedMovieTimestamps.push(candidate.movieStart);
      const alternatives = getAlternateCandidatesForRange(
        candidatePool,
        original.shortStart,
        original.shortEnd,
        rejectedMovieTimestamps,
      );
      candidate = alternatives[0];
    }

    if (accepted) {
      outcomes[i] = accepted;
    } else {
      console.log(
        `[VLM] No genuine match found after ${attempt} attempt(s) for short-clip range ` +
        `[${original.shortStart.toFixed(2)}s–${original.shortEnd.toFixed(2)}s] — dropping.`
      );
      onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'dropped' });
    }

    // Fires for EVERY segment (accepted first try, accepted after retries, or
    // dropped) so the caller can persist full candidate-comparison history —
    // not just what happened to previously-dropped segments.
    onSegmentResolved?.({ segmentIndex: i, original, triedCandidates, accepted });
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

  // Run every segment through the pool — Gemini has no server-side cache,
  // so no batch/reset boundaries are needed anymore.
  await runPool(Array.from({ length: segments.length }, (_, k) => k));

  const resolved = outcomes.filter((s): s is MatchedSegment => s !== null);
  resolved.sort((a, b) => a.shortStart - b.shortStart);
  return resolved;
}
