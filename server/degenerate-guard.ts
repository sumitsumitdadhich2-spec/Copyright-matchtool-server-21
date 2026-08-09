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

/**
 * Upper bound of the same guard. A real copied segment cannot map a tiny
 * short-clip span onto a much longer movie window: 2.0× is already the
 * extreme fast-forward the engine models, so anything past 2.5× means the
 * mapping is structurally wrong (observed bug: 0.64s clip → 2.40s movie
 * window, speedRatio 7.35 — Gemini still said "same" because the movie
 * repeats the same actress/wardrobe/location throughout).
 */
export const DEGENERATE_MAX_SPEED_RATIO =
  Number(process.env.DEGENERATE_MAX_SPEED_RATIO) || 2.5;

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

  // 2. Excessive speed ratio — the movie timeline races ahead far faster
  //    than any legitimate fast-forward edit. A tiny short span mapped onto
  //    a long movie window is structurally impossible for a real copy.
  if (Number.isFinite(seg.speedRatio) && seg.speedRatio > DEGENERATE_MAX_SPEED_RATIO) {
    return (
      `speedRatio ${seg.speedRatio.toFixed(2)} > ${DEGENERATE_MAX_SPEED_RATIO} ` +
      `(movie window far longer than any legitimate fast-forward of the short clip)`
    );
  }

  // 3. Frozen matchSequence — every matched short frame points at (almost)
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

/** Minimal shape needed for the timeline-monotonicity check. */
export interface TimelineCheckableSegment {
  shortStart: number;
  movieStart: number;
  /** Set to true (display-only) when the segment breaks the monotonic movie timeline. */
  timelineOutlier?: boolean;
}

/** Backward jitter (seconds) tolerated before a movie-time step counts as a
 *  backwards jump. Covers regression noise on segment boundaries. */
const TIMELINE_BACKWARD_TOLERANCE_S = 0.5;

/**
 * Timeline-monotonicity check (display-only flagging).
 *
 * A short clip cut from a movie almost always walks the movie timeline
 * forward: consecutive segments' movieStart values increase along with
 * shortStart. A segment whose movie position jumps BACKWARDS relative to
 * its neighbours (observed bug: seg 8 → movie 90.88s followed by seg 9 →
 * movie 84.28s) is very likely a false accept on repeated visuals.
 *
 * Legitimate edits CAN reorder scenes, so this never rejects — it only
 * sets `timelineOutlier: true` on the minority of segments that fall
 * outside the longest non-decreasing movie-time subsequence, so the UI can
 * warn the user and suggest a manual Retry.
 *
 * Mutates the passed segments in place and returns the number flagged.
 */
export function flagTimelineOutliers(segments: TimelineCheckableSegment[]): number {
  if (segments.length < 3) return 0; // too little context to call anything an outlier

  const ordered = [...segments].sort((a, b) => a.shortStart - b.shortStart);

  // Longest non-decreasing subsequence over movieStart (O(n²) — segment
  // counts are small). Segments in the LIS follow the dominant forward
  // timeline; the rest are outliers.
  const n = ordered.length;
  const lisLen: number[] = new Array(n).fill(1);
  const prev: number[] = new Array(n).fill(-1);
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < i; j++) {
      if (
        ordered[i].movieStart >= ordered[j].movieStart - TIMELINE_BACKWARD_TOLERANCE_S &&
        lisLen[j] + 1 > lisLen[i]
      ) {
        lisLen[i] = lisLen[j] + 1;
        prev[i] = j;
      }
    }
  }
  let bestEnd = 0;
  for (let i = 1; i < n; i++) if (lisLen[i] > lisLen[bestEnd]) bestEnd = i;

  const inLis = new Set<TimelineCheckableSegment>();
  for (let i = bestEnd; i !== -1; i = prev[i]) inLis.add(ordered[i]);

  // If the "dominant" timeline covers less than 60% of segments, the clip
  // is probably a genuinely reordered edit — flag nothing.
  if (inLis.size < Math.ceil(n * 0.6)) {
    for (const s of segments) s.timelineOutlier = false;
    return 0;
  }

  let flagged = 0;
  for (const s of segments) {
    const isOutlier = !inLis.has(s);
    s.timelineOutlier = isOutlier;
    if (isOutlier) flagged++;
  }
  return flagged;
}
