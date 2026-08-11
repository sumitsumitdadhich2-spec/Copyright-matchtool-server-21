/**
 * Timeline-locality ranking for VLM candidates.
 *
 * Why this exists: a copied clip is assembled from a handful of movie scenes,
 * so segments that are ADJACENT in the short clip land in the same
 * neighbourhood of the movie. The hash-based candidate pool ignores this — it
 * ranks purely on per-frame similarity, so for a short segment the pool can
 * scatter candidates across the entire movie. Observed in a 42-segment run:
 * one 0.84 s segment (index 14) had candidates at 24 s, 53 s, 56 s, 118 s,
 * 174 s, 241 s, 250 s, 268 s, 295 s and 424 s, while its verified neighbours
 * (segments 11–17) all sat between 108 s and 205 s. With a limited Gemini
 * attempt budget, the one plausible candidate can fall outside the prefix that
 * actually gets checked.
 *
 * The signal that works is NOT distance to a single predicted point — real
 * segment-to-segment gaps in the movie run to tens of seconds, so a point
 * prediction is far too tight. What discriminates cleanly is whether a
 * candidate falls inside the movie-time WINDOW spanned by the segment's
 * nearby neighbours. On the run above this reduces segment 14 from 10
 * candidates to 2 and segment 18 from 11 to 2, keeping the correct one in
 * both cases.
 *
 * This module only REORDERS candidates. It never accepts, rejects, filters or
 * rewrites one — Gemini remains the sole verdict-maker, and a promoted
 * candidate still has to pass verification. Out-of-window candidates are
 * demoted, never removed, so a genuinely displaced segment is still reachable.
 */

/** Minimal shape needed to place a segment on the movie timeline. */
export interface ContinuityCheckableSegment {
  movieStart: number;
  movieEnd: number;
}

/**
 * How many neighbouring segments on each side define the expected movie
 * neighbourhood. Wide enough that a couple of unmatched or mislocated
 * neighbours cannot collapse the window, narrow enough to stay local to the
 * current scene group.
 */
export const CONTINUITY_NEIGHBOURS =
  Number(process.env.CONTINUITY_NEIGHBOURS) || 3;

/**
 * Slack added to each side of the neighbour window, absorbing the normal gap
 * between one copied scene and the next.
 */
export const CONTINUITY_SLACK_S =
  Number(process.env.CONTINUITY_SLACK_S) || 8;

/** Minimum neighbours required before a window is considered meaningful. */
const CONTINUITY_MIN_NEIGHBOURS = 2;

export interface ContinuityWindow {
  lower: number;
  upper: number;
  neighbourCount: number;
}

/**
 * Movie-time window where this segment is expected to live, from the segments
 * immediately around it in short-clip order.
 *
 * Returns null when there are too few neighbours to say anything — in that
 * case ordering is left completely untouched.
 */
export function computeContinuityWindow(
  neighbours: readonly ContinuityCheckableSegment[],
  index: number,
): ContinuityWindow | null {
  const from = Math.max(0, index - CONTINUITY_NEIGHBOURS);
  const to = Math.min(neighbours.length - 1, index + CONTINUITY_NEIGHBOURS);

  let lower = Infinity;
  let upper = -Infinity;
  let neighbourCount = 0;

  for (let n = from; n <= to; n++) {
    if (n === index) continue;
    const seg = neighbours[n];
    if (!seg || !Number.isFinite(seg.movieStart) || !Number.isFinite(seg.movieEnd)) continue;
    if (seg.movieStart < lower) lower = seg.movieStart;
    if (seg.movieEnd > upper) upper = seg.movieEnd;
    neighbourCount++;
  }

  if (neighbourCount < CONTINUITY_MIN_NEIGHBOURS) return null;
  if (!Number.isFinite(lower) || !Number.isFinite(upper)) return null;

  return {
    lower: Math.max(0, lower - CONTINUITY_SLACK_S),
    upper: upper + CONTINUITY_SLACK_S,
    neighbourCount,
  };
}

/** Whether a candidate starts inside the expected movie neighbourhood. */
export function isWithinContinuityWindow(
  candidate: ContinuityCheckableSegment,
  window: ContinuityWindow,
): boolean {
  return candidate.movieStart >= window.lower && candidate.movieStart <= window.upper;
}

/**
 * Stable reorder of `order` promoting candidates inside the expected movie
 * neighbourhood, WITHOUT displacing the incoming top-ranked candidate.
 *
 * The leading candidate stays pinned at position 1 because the window is a
 * locality prior, not evidence: a copied clip legitimately jumps to a far part
 * of the movie, and on the reference run four already-correct segments (7, 26,
 * 35, 38) had their verified answer sitting OUTSIDE the neighbour window —
 * demoting it would have spent Gemini calls on worse candidates and risked a
 * false accept on a closer-looking one. Pinning costs nothing, since position
 * 1 is already the best candidate by green-score/embedding rank.
 *
 * Everything after position 1 is partitioned: in-window candidates first, then
 * the rest, each group keeping its incoming relative order. On the reference
 * run this is exactly what lifts the correct candidate for segment 18 from
 * deep in the pool to position 2, while leaving every previously-accepted
 * segment's winner first.
 */
export function sortIndexesByContinuity(
  getSegment: (index: number) => ContinuityCheckableSegment,
  order: readonly number[],
  window: ContinuityWindow | null,
): number[] {
  if (!window || order.length <= 2) return [...order];

  const [pinned, ...rest] = order;
  const inside: number[] = [];
  const outside: number[] = [];
  for (const idx of rest) {
    (isWithinContinuityWindow(getSegment(idx), window) ? inside : outside).push(idx);
  }
  // Nothing (or everything) qualifies — the window adds no information.
  if (inside.length === 0 || outside.length === 0) return [...order];

  return [pinned, ...inside, ...outside];
}

/** Short log tag describing a candidate's fit to the expected window. */
export function continuityLogTag(
  candidate: ContinuityCheckableSegment,
  window: ContinuityWindow | null,
): string {
  if (!window) return 'win:n/a';
  return isWithinContinuityWindow(candidate, window) ? 'in-window' : 'out';
}
