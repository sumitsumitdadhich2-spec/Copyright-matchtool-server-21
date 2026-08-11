/**
 * Display-only timeline sanity flag.
 * ---------------------------------------------------------------------------
 * A short clip normally walks forward through the reference film: as
 * shortStart increases, movieStart increases too. A segment that jumps
 * backwards against that trend is usually a mislocalised match, so the UI
 * marks it for human attention.
 *
 * This NEVER removes, re-orders, or rewrites a segment — it only sets a
 * boolean. It is also deliberately silent when the clip is genuinely
 * re-ordered (a supercut / non-linear edit), where "backwards" carries no
 * information: if fewer than 60% of consecutive pairs move forward, no flag is
 * set at all.
 */

import type { MatchedSegment } from '../matching-engine';

export function flagTimelineOutliers<T extends MatchedSegment>(segments: T[]): T[] {
  if (segments.length < 3) return segments.map(s => ({ ...s, timelineOutlier: false }));

  const ordered = [...segments].sort((a, b) => a.shortStart - b.shortStart);

  let forward = 0;
  for (let i = 1; i < ordered.length; i++) {
    if (ordered[i].movieStart >= ordered[i - 1].movieStart) forward++;
  }
  const forwardRatio = forward / (ordered.length - 1);

  // Non-linear edit — "backwards" is the norm here, so flag nothing.
  if (forwardRatio < 0.6) return segments.map(s => ({ ...s, timelineOutlier: false }));

  const outliers = new Set<T>();
  for (let i = 1; i < ordered.length; i++) {
    const prev = ordered[i - 1];
    const cur = ordered[i];
    // Only a real backwards jump counts: allow a small tolerance so a
    // slightly-overlapping neighbour is not flagged.
    if (cur.movieStart < prev.movieStart - 0.5) outliers.add(cur);
  }

  return segments.map(s => ({ ...s, timelineOutlier: outliers.has(s) }));
}
