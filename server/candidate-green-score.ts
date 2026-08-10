/**
 * Green-quality score for a candidate segment's "Match quality timeline".
 *
 * The preview UI paints each matchSequence frame green when its similarity is
 * >= 80 (yellow >= 60, red below — see src/App.tsx). Observation: candidates
 * whose timeline is mostly GREEN are almost always the genuinely correct
 * match. This module turns that free, already-computed per-frame data into a
 * PRIMARY ordering signal:
 *
 *   score = greenFraction  (fraction of frames with similarity >= 80) FIRST,
 *           avgSimilarity  (plain matchSequence average) as the tie-break.
 *
 * STRICTLY ordering-only. It never accepts, rejects, or removes a candidate —
 * Gemini VLM verification remains the one and only verdict-maker. The
 * crop-robust embedding rank (rankCandidatesCropRobust) stays intact as the
 * SECONDARY refinement: callers sort by green-score with a STABLE sort over
 * the embedding-ranked order, so equal-green candidates keep their embedding
 * order exactly.
 *
 * Fail-safe: a candidate with an empty/missing matchSequence scores 0 (never
 * crashes) and simply ranks last — old persisted jobs keep loading fine.
 */

/** Same threshold the UI's timeline uses for a GREEN bar — do not change. */
export const GREEN_SIMILARITY_THRESHOLD = 80;

/** Minimal structural shape — works for both MatchedSegment declarations
 *  (matching-engine.ts and candidate-matching-engine.ts). */
export interface GreenScorableSegment {
  matchSequence?: Array<{ similarity: number }> | null;
}

export interface GreenScore {
  /** Fraction (0..1) of matchSequence frames with similarity >= 80. PRIMARY. */
  greenFraction: number;
  /** Plain average similarity (0..100) across the matchSequence. Tie-break. */
  avgSimilarity: number;
}

/** Score one candidate segment. Empty/missing matchSequence -> all zeros. */
export function greenScoreOfSegment(segment: GreenScorableSegment | null | undefined): GreenScore {
  const seq = segment?.matchSequence;
  if (!Array.isArray(seq) || seq.length === 0) {
    return { greenFraction: 0, avgSimilarity: 0 };
  }
  let greenCount = 0;
  let simSum = 0;
  for (const frame of seq) {
    const sim = typeof frame?.similarity === 'number' && Number.isFinite(frame.similarity)
      ? frame.similarity
      : 0;
    simSum += sim;
    if (sim >= GREEN_SIMILARITY_THRESHOLD) greenCount++;
  }
  return { greenFraction: greenCount / seq.length, avgSimilarity: simSum / seq.length };
}

/** Descending comparator: more green first, then higher average similarity. */
export function compareGreenScoreDesc(a: GreenScore, b: GreenScore): number {
  if (b.greenFraction !== a.greenFraction) return b.greenFraction - a.greenFraction;
  return b.avgSimilarity - a.avgSimilarity;
}

/**
 * Reorder candidate INDEXES so green-score is the PRIMARY key (descending)
 * while the incoming index order (e.g. crop-robust embedding rank) survives
 * as the SECONDARY tie-breaker — Array.prototype.sort is stable, so equal
 * scores keep their relative order. Returns a reordered COPY of exactly the
 * given indexes; never adds, drops, or mutates anything.
 */
export function sortIndexesByGreenScore(
  segmentAt: (index: number) => GreenScorableSegment | null | undefined,
  indexes: number[],
): number[] {
  if (indexes.length <= 1) return [...indexes];
  const scores = new Map<number, GreenScore>();
  for (const i of indexes) {
    let score: GreenScore;
    try {
      score = greenScoreOfSegment(segmentAt(i));
    } catch {
      score = { greenFraction: 0, avgSimilarity: 0 };
    }
    scores.set(i, score);
  }
  return [...indexes].sort((a, b) => compareGreenScoreDesc(scores.get(a)!, scores.get(b)!));
}

/**
 * Stable-sort a SEGMENT ARRAY by green-score descending (discovery-pool use:
 * applied BEFORE a top-N cut so the N kept candidates are the greenest ones,
 * not merely the first-found ones). Returns a new array; the input and its
 * members are never mutated, and nothing is added or removed.
 */
export function sortSegmentsByGreenScore<T extends GreenScorableSegment>(segments: T[]): T[] {
  if (!Array.isArray(segments) || segments.length <= 1) return [...(segments ?? [])];
  const scored = segments.map((segment, i) => ({ segment, i, score: greenScoreOfSegment(segment) }));
  scored.sort((a, b) => {
    const byScore = compareGreenScoreDesc(a.score, b.score);
    return byScore !== 0 ? byScore : a.i - b.i;
  });
  return scored.map(s => s.segment);
}

/** Compact "g=0.87 avg=87.1" tag for verification-order log lines. */
export function greenScoreLogTag(segment: GreenScorableSegment | null | undefined): string {
  const s = greenScoreOfSegment(segment);
  return `g=${s.greenFraction.toFixed(2)} avg=${s.avgSimilarity.toFixed(1)}`;
}
