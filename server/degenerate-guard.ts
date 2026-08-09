/**
 * Degenerate-candidate guard (VLM false-accept fix).
 *
 * Root cause it protects against: a movie can contain two visually
 * near-identical scenes (same actors, same wardrobe). The hash matcher can
 * then produce a candidate whose matchSequence is "frozen" — every short
 * frame mapped onto the SAME movie instant (speedRatio ≈ 0) — which can
 * never be a real copied segment, yet Gemini, looking only at the frames,
 * may still answer "same" with high confidence.
 *
 * This guard structurally rejects such candidates BEFORE they can be
 * accepted, regardless of what the VLM says. It never rejects a plausible
 * candidate: a real match always advances through the movie timeline at a
 * roughly comparable rate to the short clip.
 *
 * Used by every verification path (main pass, deferred recovery, manual
 * retry) so a degenerate candidate can never be accepted anywhere.
 */

/** Structural subset of MatchedSegment needed for the check — identical in
 *  matching-engine.ts and candidate-matching-engine.ts, so both engines'
 *  segments satisfy it. */
export interface DegenerateCheckableSegment {
  shortStart: number;
  shortEnd: number;
  movieStart: number;
  movieEnd: number;
  speedRatio: number;
  matchSequence: Array<{ shortTime: number; movieTime: number; similarity: number }>;
}

/**
 * A real copied segment plays the movie forward at a speed comparable to the
 * short clip. Below this ratio the mapping is effectively frozen (e.g. the
 * observed bug: speedRatio 0.1 — 53 short frames all stuck on one movie
 * frame). 0.5× is already the extreme legitimate slow-mo the engine models.
 */
export const DEGENERATE_MIN_SPEED_RATIO =
  Number(process.env.DEGENERATE_MIN_SPEED_RATIO) || 0.35;

/** matchSequence spans below which the movie side counts as "frozen" when
 *  the short side spans at least DEGENERATE_MIN_SHORT_SPAN_S. */
const DEGENERATE_FROZEN_MOVIE_SPAN_S = 0.25;
const DEGENERATE_MIN_SHORT_SPAN_S = 1.0;

/**
 * Returns a human-readable reason when the candidate is structurally
 * degenerate (can never be a real match), or null when it is plausible.
 */
export function degenerateCandidateReason(
  seg: DegenerateCheckableSegment,
): string | null {
  // 1. Near-zero (or backwards) speed ratio — the regression over the whole
  //    matchSequence says the movie timeline barely advances while the short
  //    plays. A real copy cannot look like this.
  if (Number.isFinite(seg.speedRatio) && seg.speedRatio < DEGENERATE_MIN_SPEED_RATIO) {
    return (
      `speedRatio ${seg.speedRatio.toFixed(2)} < ${DEGENERATE_MIN_SPEED_RATIO} ` +
      `(movie timeline frozen relative to short clip)`
    );
  }

  // 2. Frozen matchSequence — every matched short frame points at (almost)
  //    the same movie instant even though the short side spans real time.
  //    Catches degenerate mappings even when the regression slope is noisy.
  if (seg.matchSequence.length >= 3) {
    let minMovie = Infinity, maxMovie = -Infinity;
    let minShort = Infinity, maxShort = -Infinity;
    for (const f of seg.matchSequence) {
      if (f.movieTime < minMovie) minMovie = f.movieTime;
      if (f.movieTime > maxMovie) maxMovie = f.movieTime;
      if (f.shortTime < minShort) minShort = f.shortTime;
      if (f.shortTime > maxShort) maxShort = f.shortTime;
    }
    const movieSpan = maxMovie - minMovie;
    const shortSpan = maxShort - minShort;
    if (shortSpan >= DEGENERATE_MIN_SHORT_SPAN_S && movieSpan < DEGENERATE_FROZEN_MOVIE_SPAN_S) {
      return (
        `frozen matchSequence (${seg.matchSequence.length} short frames spanning ` +
        `${shortSpan.toFixed(2)}s all map to a ${movieSpan.toFixed(2)}s movie window)`
      );
    }
  }

  return null;
}
