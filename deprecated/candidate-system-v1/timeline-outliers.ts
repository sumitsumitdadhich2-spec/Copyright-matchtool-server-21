/**
 * Timeline monotonicity check (display-only flag).
 *
 * A short clip cut from a movie almost always uses movie material in
 * chronological order: as the clip plays forward, the matched movie
 * timestamps should also move forward (or stay put across a cut). When one
 * segment suddenly jumps BACKWARDS in the movie relative to its neighbours
 * (observed bug: Seg 8 → 90.88s followed by Seg 9 → 84.28s), that segment is
 * very likely a false match onto a visually similar duplicate scene.
 *
 * This module finds the dominant forward movie timeline via the longest
 * non-decreasing subsequence (LNDS) of movieStart values (segments ordered by
 * shortStart) and flags every segment OUTSIDE that subsequence with
 * `timelineOutlier: true`.
 *
 * Strictly display-only:
 *  - segments are never removed or re-ordered — legit re-ordered edits stay;
 *  - if the "outliers" would be a large fraction of all segments
 *    (>= TIMELINE_OUTLIER_MAX_FRACTION), the short is treated as a genuinely
 *    re-ordered edit and NOTHING is flagged.
 */

interface TimelineCheckableSegment {
  shortStart: number;
  movieStart: number;
  timelineOutlier?: boolean;
}

/** Minimum number of segments before the check is meaningful at all. */
const TIMELINE_OUTLIER_MIN_SEGMENTS = 4;

/** If flagging would mark this fraction (or more) of segments as outliers,
 *  the short is a genuinely re-ordered edit — flag nothing. */
const TIMELINE_OUTLIER_MAX_FRACTION =
  Number(process.env.TIMELINE_OUTLIER_MAX_FRACTION) || 0.4;

/**
 * Returns a NEW array (same order) where segments off the dominant forward
 * movie timeline carry `timelineOutlier: true`. Input objects are not
 * mutated; non-outlier segments get any stale `timelineOutlier` flag cleared.
 */
export function flagTimelineOutliers<T extends TimelineCheckableSegment>(
  segments: T[],
): T[] {
  if (segments.length < TIMELINE_OUTLIER_MIN_SEGMENTS) {
    return segments.map(s =>
      s.timelineOutlier ? { ...s, timelineOutlier: false } : s);
  }

  // Order by position in the short clip (the play order the viewer sees).
  const order = segments
    .map((s, i) => ({ i, shortStart: s.shortStart, movieStart: s.movieStart }))
    .sort((a, b) => a.shortStart - b.shortStart);

  // Longest non-decreasing subsequence over movieStart — O(n^2) is fine for
  // the segment counts involved (tens, not thousands).
  const n = order.length;
  const lisLen = new Array<number>(n).fill(1);
  const prev = new Array<number>(n).fill(-1);
  let bestEnd = 0;
  for (let j = 1; j < n; j++) {
    for (let k = 0; k < j; k++) {
      if (order[k].movieStart <= order[j].movieStart && lisLen[k] + 1 > lisLen[j]) {
        lisLen[j] = lisLen[k] + 1;
        prev[j] = k;
      }
    }
    if (lisLen[j] > lisLen[bestEnd]) bestEnd = j;
  }

  // Members of the dominant forward timeline.
  const onTimeline = new Set<number>();
  for (let j = bestEnd; j !== -1; j = prev[j]) onTimeline.add(order[j].i);

  const outlierCount = n - onTimeline.size;
  if (outlierCount === 0 || outlierCount / n >= TIMELINE_OUTLIER_MAX_FRACTION) {
    // Nothing suspicious, or the short is genuinely re-ordered — flag nothing.
    return segments.map(s =>
      s.timelineOutlier ? { ...s, timelineOutlier: false } : s);
  }

  return segments.map((s, i) => {
    const isOutlier = !onTimeline.has(i);
    if (Boolean(s.timelineOutlier) === isOutlier) return s;
    return { ...s, timelineOutlier: isOutlier };
  });
}
