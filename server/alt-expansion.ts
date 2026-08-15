/**
 * Anchor-based alt-candidate expansion — Tasks 2, 3, 4 of the FPS-aware,
 * speed-tolerant alt-candidate pipeline.
 * ---------------------------------------------------------------------------
 * STRICTLY ADDITIVE POST-PASS. This module never touches Pass 1/2/3, the seed
 * scan, the walks, or the dedup that selects accepted segments. It runs AFTER
 * the engine has finished, receives its outputs read-only, and returns extra
 * candidates that the engine appends to `candidatePool` — the pool consumed
 * only by the Gemini verification retry loop. Precision is Gemini's job;
 * recall is this module's job ("garbage aa bhi jaye to Gemini filter kar
 * dega").
 *
 * Pipeline per short-clip scene chunk that lacks a confident match:
 *   1. ANCHORS (Task 2): high-confidence segments elsewhere in the clip act
 *      as anchors. Two anchors give the edit's speed in ONE division:
 *      speed = Δmovie / Δshort. One anchor falls back to its own speedRatio.
 *   2. TIMESTAMP-CORRECTED RELAXED SEARCH (Task 2): the missing range is
 *      searched only in a small movie window predicted from the anchors +
 *      inferred speed, with a loose similarity threshold.
 *   3. RANSAC LINE FIT (Task 3): scattered (shortTime, movieTime) hash
 *      matches are fit to movieTime = speed·shortTime + offset. Outliers
 *      drop away; each surviving line (slope clamped to 0.1x–8x) seeds one
 *      full-coverage candidate built by the engine's own alt builder.
 *   4. SPEED-AWARE GROUPING (Task 4): built candidates whose line-predicted
 *      positions agree within ALT_GROUP_TOLERANCE_SECONDS merge into one
 *      candidate with unioned frame support, so drifted fragments of the
 *      same real match pool their confidence instead of splitting it.
 *
 * All math here is SECONDS-based (Task 1): match positions, offsets, window
 * bounds and grouping tolerances are computed from timestamps, never raw
 * frame counts, so a 10fps short vs a 60fps movie compares on one time scale.
 *
 * Engine access is via injected closures (scanWindow / buildCandidate), so
 * this file imports only TYPES from the protected matching engine.
 */

import type { FPData, MatchedSegment } from './matching-engine';
import {
  buildDuplicateFrameMask,
  medianFrameDuration,
  type DuplicateFrameMask,
} from './video-metadata';

// ---------------------------------------------------------------------------
// Env tunables — all safe-defaulted, all overridable
// ---------------------------------------------------------------------------

function envNum(name: string, fallback: number, min: number, max: number): number {
  const n = Number(process.env[name]);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

/** Kill switch — set ALT_EXPANSION_ENABLED=0 to disable the whole post-pass. */
const ALT_EXPANSION_ENABLED = process.env.ALT_EXPANSION_ENABLED !== '0';
/** Minimum engine confidence (%) for a segment to serve as an anchor. */
const ALT_ANCHOR_MIN_CONF = envNum('ALT_ANCHOR_MIN_CONF', 85, 50, 100);
/** Half-width (seconds of MOVIE time) of the relaxed-search window around the
 *  anchor-predicted position. */
const ALT_EXPAND_WINDOW_SECONDS = envNum('ALT_EXPAND_WINDOW_SECONDS', 45, 5, 600);
/** Loose hash-similarity floor (%) for relaxed-search points. */
const ALT_RELAXED_MIN_SIM = envNum('ALT_RELAXED_MIN_SIM', 55, 30, 95);
/** Short frames sampled per chunk for the relaxed search (duplicates skipped). */
const ALT_EXPANSION_SAMPLES = Math.round(envNum('ALT_EXPANSION_SAMPLES', 12, 4, 48));
/** RANSAC iterations (random pair draws; small point sets use all pairs). */
const ALT_RANSAC_ITERS = Math.round(envNum('ALT_RANSAC_ITERS', 200, 20, 5000));
/** Inlier tolerance (seconds) around the fitted line. */
const ALT_RANSAC_TOLERANCE_SECONDS = envNum('ALT_RANSAC_TOLERANCE_SECONDS', 0.5, 0.05, 5);
/** Minimum inliers for a line to produce a candidate. */
const ALT_RANSAC_MIN_INLIERS = Math.round(envNum('ALT_RANSAC_MIN_INLIERS', 4, 2, 50));
/** Max expansion candidates built per chunk (distinct RANSAC lines). */
const ALT_EXPANSION_MAX_PER_CHUNK = Math.round(envNum('ALT_EXPANSION_MAX_PER_CHUNK', 2, 1, 6));
/** Task 4: candidates whose line-predicted positions agree within this many
 *  seconds merge into one candidate with unioned frame support. */
const ALT_GROUP_TOLERANCE_SECONDS = envNum('ALT_GROUP_TOLERANCE_SECONDS', 2.0, 0.2, 10);
/** Speed clamps — recap edits use 0.1x super-slow-mo up to 8x fast-forward.
 *  Mirrors the engine's SLOPE_MIN/SLOPE_MAX without importing them. */
const ALT_SPEED_MIN = envNum('ALT_SPEED_MIN', 0.1, 0.01, 1);
const ALT_SPEED_MAX = envNum('ALT_SPEED_MAX', 8.0, 1, 32);

// ---------------------------------------------------------------------------
// Public contract
// ---------------------------------------------------------------------------

/** A relaxed-search hit handed back by the engine's injected scanner. */
export interface ScanPoint {
  si: number;
  mi: number;
  sim: number;
}

export interface AltExpansionInput {
  shortFps: FPData[];
  /** Global movie frame metadata — timestamps are all that is read. */
  movieFps: Array<{ timestamp: number }>;
  /** The engine's scene chunks (short-frame index ranges). */
  chunks: Array<{ start: number; end: number }>;
  /** Final accepted segments from the engine (read-only; anchor source). */
  acceptedSegments: MatchedSegment[];
  /**
   * Relaxed window scan, provided by the engine so this module needs no
   * access to PreSets or similarity internals. Must return every (si, mi)
   * pair within [loMi, hiMi] whose similarity ≥ minSim, using the SAME
   * scoring the engine's own seed scan uses.
   */
  scanWindow: (
    sis: number[],
    loMi: number,
    hiMi: number,
    minSim: number,
  ) => Promise<ScanPoint[]>;
  /**
   * Build one full-coverage candidate for a chunk at a given alignment
   * offset (movie frame index minus short frame index) — the engine wraps
   * its own buildAltCandidate / buildAltCandidateChunked here, so expansion
   * candidates carry the exact same full-coverage guarantees as the rest of
   * the alt pool.
   */
  buildCandidate: (
    chunkStart: number,
    chunkEnd: number,
    offsetFrames: number,
  ) => Promise<MatchedSegment | null>;
  /** Log prefix, e.g. '[Matcher]' / '[MatchChunked]'. */
  logTag?: string;
}

/** Debug metadata attached to every expansion-built candidate. */
export interface AltExpansionMeta {
  source: 'anchor-expansion';
  /** Speed inferred from anchors for the window prediction. */
  anchorSpeed: number;
  /** Slope of the RANSAC line that produced this candidate. */
  ransacSpeed: number;
  ransacInliers: number;
  ransacPoints: number;
  anchorsUsed: number;
  /** How many raw candidates were merged into this one by Task 4 grouping. */
  groupedFrom: number;
}

// ---------------------------------------------------------------------------
// Anchors + speed inference (Task 2)
// ---------------------------------------------------------------------------

interface Anchor {
  shortMid: number;
  movieMid: number;
  speedRatio: number;
}

function toAnchor(seg: MatchedSegment): Anchor {
  return {
    shortMid: (seg.shortStart + seg.shortEnd) / 2,
    movieMid: (seg.movieStart + seg.movieEnd) / 2,
    speedRatio: clampSpeed(seg.speedRatio || 1),
  };
}

function clampSpeed(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 1;
  return Math.max(ALT_SPEED_MIN, Math.min(ALT_SPEED_MAX, v));
}

/**
 * Infer speed + predicted movie time for a short-clip timestamp from the
 * nearest anchors. Two straddling anchors → ONE division gives the speed
 * (0.1x..8x all fall out naturally, no brute-force buckets). A single anchor
 * falls back to that segment's own regression speedRatio.
 */
function predictFromAnchors(
  anchors: Anchor[],
  shortTime: number,
): { speed: number; predictedMovieTime: number; anchorsUsed: number } | null {
  if (anchors.length === 0) return null;

  let prev: Anchor | null = null;
  let next: Anchor | null = null;
  for (const a of anchors) {
    if (a.shortMid <= shortTime) prev = a;
    if (a.shortMid > shortTime && next === null) next = a;
  }

  if (prev && next && next.shortMid - prev.shortMid > 0.25) {
    // Speed in one division: movie gap over short gap between the anchors.
    const speed = clampSpeed((next.movieMid - prev.movieMid) / (next.shortMid - prev.shortMid));
    return {
      speed,
      predictedMovieTime: prev.movieMid + speed * (shortTime - prev.shortMid),
      anchorsUsed: 2,
    };
  }

  const nearest = prev ?? next;
  if (!nearest) return null;
  const speed = nearest.speedRatio;
  return {
    speed,
    predictedMovieTime: nearest.movieMid + speed * (shortTime - nearest.shortMid),
    anchorsUsed: 1,
  };
}

// ---------------------------------------------------------------------------
// RANSAC line fit (Task 3): movieTime = speed·shortTime + offset
// ---------------------------------------------------------------------------

interface TimePoint {
  shortTime: number;
  movieTime: number;
  sim: number;
}

interface RansacModel {
  speed: number;
  offset: number; // seconds: movieTime at shortTime=0
  inliers: TimePoint[];
  score: number; // inlier count, tie-broken by summed similarity
}

/** Deterministic LCG so identical inputs always produce identical output. */
function seededRandom(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function evaluateLine(points: TimePoint[], speed: number, offset: number): RansacModel {
  const inliers: TimePoint[] = [];
  let simSum = 0;
  for (const p of points) {
    const predicted = speed * p.shortTime + offset;
    if (Math.abs(p.movieTime - predicted) <= ALT_RANSAC_TOLERANCE_SECONDS) {
      inliers.push(p);
      simSum += p.sim;
    }
  }
  return { speed, offset, inliers, score: inliers.length + simSum / 1e6 };
}

/** Least-squares refit over a model's inliers (clamped slope). */
function refit(model: RansacModel, points: TimePoint[]): RansacModel {
  const pts = model.inliers;
  if (pts.length < 2) return model;
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const p of pts) {
    sx += p.shortTime; sy += p.movieTime;
    sxx += p.shortTime * p.shortTime; sxy += p.shortTime * p.movieTime;
    n++;
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-9) return model;
  const slope = clampSpeed((n * sxy - sx * sy) / denom);
  const intercept = (sy - slope * sx) / n;
  return evaluateLine(points, slope, intercept);
}

/**
 * Fit up to `maxModels` distinct lines through the scattered points. Distinct
 * means the lines predict movie positions further apart than the grouping
 * tolerance at the chunk midpoint — otherwise they are the same real match.
 */
function ransacLines(
  points: TimePoint[],
  chunkMidShortTime: number,
  maxModels: number,
): RansacModel[] {
  if (points.length < ALT_RANSAC_MIN_INLIERS) return [];

  const candidates: RansacModel[] = [];
  const tryPair = (a: TimePoint, b: TimePoint): void => {
    const ds = b.shortTime - a.shortTime;
    if (Math.abs(ds) < 1e-6) return;
    const speed = (b.movieTime - a.movieTime) / ds;
    if (!Number.isFinite(speed) || speed < ALT_SPEED_MIN || speed > ALT_SPEED_MAX) return;
    const offset = a.movieTime - speed * a.shortTime;
    candidates.push(evaluateLine(points, speed, offset));
  };

  const totalPairs = (points.length * (points.length - 1)) / 2;
  if (totalPairs <= ALT_RANSAC_ITERS) {
    // Small point set — exhaustive pairs, fully deterministic.
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) tryPair(points[i], points[j]);
    }
  } else {
    const rand = seededRandom(points.length * 7919 + 17);
    for (let it = 0; it < ALT_RANSAC_ITERS; it++) {
      const i = Math.floor(rand() * points.length);
      let j = Math.floor(rand() * points.length);
      if (j === i) j = (j + 1) % points.length;
      tryPair(points[i], points[j]);
    }
  }

  candidates.sort((a, b) => b.score - a.score);

  const picked: RansacModel[] = [];
  for (const cand of candidates) {
    if (cand.inliers.length < ALT_RANSAC_MIN_INLIERS) continue;
    const refined = refit(cand, points);
    if (refined.inliers.length < ALT_RANSAC_MIN_INLIERS) continue;
    const predictedMid = refined.speed * chunkMidShortTime + refined.offset;
    const dup = picked.some(m =>
      Math.abs((m.speed * chunkMidShortTime + m.offset) - predictedMid) <= ALT_GROUP_TOLERANCE_SECONDS,
    );
    if (dup) continue;
    picked.push(refined);
    if (picked.length >= maxModels) break;
  }
  return picked;
}

// ---------------------------------------------------------------------------
// Speed-aware grouping (Task 4) — runs on BUILT candidates only
// ---------------------------------------------------------------------------

/** Predicted movie time of a candidate at a given short-clip time, using the
 *  candidate's own regression speed. */
function predictedMovieTime(c: MatchedSegment, shortTime: number): number {
  return c.movieStart + clampSpeed(c.speedRatio || 1) * (shortTime - c.shortStart);
}

function shortOverlap(a: MatchedSegment, b: MatchedSegment): number {
  return Math.min(a.shortEnd, b.shortEnd) - Math.max(a.shortStart, b.shortStart);
}

/** Merge two candidates that Task 4 decided are the same real match:
 *  union the frame support so combined confidence rises naturally. */
function mergeCandidates(base: MatchedSegment, other: MatchedSegment): MatchedSegment {
  const byShortTime = new Map<string, { shortTime: number; movieTime: number; similarity: number }>();
  for (const f of base.matchSequence) byShortTime.set(f.shortTime.toFixed(4), f);
  for (const f of other.matchSequence) {
    const key = f.shortTime.toFixed(4);
    const existing = byShortTime.get(key);
    if (!existing || f.similarity > existing.similarity) byShortTime.set(key, f);
  }
  const union = [...byShortTime.values()].sort((a, b) => a.shortTime - b.shortTime);
  const confidence = union.length > 0
    ? union.reduce((s, f) => s + f.similarity, 0) / union.length
    : base.confidence;
  const totalFrames = base.frameCount + other.frameCount;

  const baseMeta = base.altExpansion;
  const otherMeta = other.altExpansion;

  return {
    ...base,
    shortStart: Math.min(base.shortStart, other.shortStart),
    shortEnd: Math.max(base.shortEnd, other.shortEnd),
    movieStart: Math.min(base.movieStart, other.movieStart),
    movieEnd: Math.max(base.movieEnd, other.movieEnd),
    frameCount: union.length,
    confidence,
    gapCount: Math.min(base.gapCount, other.gapCount),
    speedRatio:
      (base.speedRatio * base.frameCount + other.speedRatio * other.frameCount) / totalFrames,
    matchSequence: union,
    bestFrameDetail: base.confidence >= other.confidence
      ? base.bestFrameDetail ?? other.bestFrameDetail
      : other.bestFrameDetail ?? base.bestFrameDetail,
    altExpansion: baseMeta
      ? { ...baseMeta, groupedFrom: (baseMeta.groupedFrom ?? 1) + (otherMeta?.groupedFrom ?? 1) }
      : otherMeta,
  };
}

/**
 * Task 4: speed-aware dedup grouping over built candidates. Two candidates
 * merge when their short ranges overlap and their line-predicted movie
 * positions at the overlap midpoint agree within ALT_GROUP_TOLERANCE_SECONDS
 * — i.e. they are drift-shifted fragments of one real match, regardless of
 * a constant-offset comparison at the range edges.
 */
export function groupCandidatesSpeedAware(cands: MatchedSegment[]): MatchedSegment[] {
  if (cands.length <= 1) return cands;
  const sorted = [...cands].sort((a, b) => b.confidence * b.frameCount - a.confidence * a.frameCount);
  const out: MatchedSegment[] = [];

  for (const cand of sorted) {
    let merged = false;
    for (let i = 0; i < out.length; i++) {
      const kept = out[i];
      const overlap = shortOverlap(kept, cand);
      if (overlap <= 0.15) continue;
      const midT = Math.max(kept.shortStart, cand.shortStart) + overlap / 2;
      const drift = Math.abs(predictedMovieTime(kept, midT) - predictedMovieTime(cand, midT));
      if (drift <= ALT_GROUP_TOLERANCE_SECONDS) {
        out[i] = mergeCandidates(kept, cand);
        merged = true;
        break;
      }
    }
    if (!merged) out.push(cand);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Main entry — never throws, returns [] on any failure
// ---------------------------------------------------------------------------

export async function expandAltCandidates(input: AltExpansionInput): Promise<MatchedSegment[]> {
  const tag = input.logTag ?? '[AltExpand]';
  if (!ALT_EXPANSION_ENABLED) {
    console.log(`${tag} Alt-candidate expansion disabled via ALT_EXPANSION_ENABLED=0.`);
    return [];
  }

  try {
    const { shortFps, movieFps, chunks, acceptedSegments } = input;
    if (shortFps.length === 0 || movieFps.length === 0 || chunks.length === 0) return [];

    const anchors = acceptedSegments
      .filter(s => s.confidence >= ALT_ANCHOR_MIN_CONF)
      .map(toAnchor)
      .sort((a, b) => a.shortMid - b.shortMid);
    if (anchors.length === 0) {
      console.log(`${tag} Expansion: no anchors (no segment ≥ ${ALT_ANCHOR_MIN_CONF}% conf) — skipping.`);
      return [];
    }

    // Seconds-based scales (Task 1): everything below works in timestamps.
    const shortFrameDur = medianFrameDuration(shortFps) || 0.04;
    const movieFrameDur = medianFrameDuration(movieFps) || 0.04;
    const movieDuration = movieFps[movieFps.length - 1].timestamp;

    // Duplicate-frame mask (Task 1): a low-fps short resampled onto the 25fps
    // grid repeats frames; skip repeats when sampling seeds so each sample is
    // a real, distinct source frame.
    let dupMask: DuplicateFrameMask | null = null;
    try {
      dupMask = buildDuplicateFrameMask(shortFps);
      if (dupMask.duplicateCount > 0) {
        console.log(
          `${tag} Expansion: short-clip duplicate-frame mask — ` +
          `${dupMask.duplicateCount}/${shortFps.length} duplicates, ` +
          `effective fps ≈ ${dupMask.effectiveFps?.toFixed(1) ?? 'n/a'}.`,
        );
      }
    } catch { /* mask is best-effort */ }

    const built: MatchedSegment[] = [];

    for (const chunk of chunks) {
      const chunkStartT = shortFps[chunk.start].timestamp;
      const chunkEndT = shortFps[chunk.end].timestamp;
      const chunkMidT = (chunkStartT + chunkEndT) / 2;

      // Only expand chunks WITHOUT a confident engine match — anchors don't
      // need help, and skipping them keeps the pass cheap and focused.
      const covered = acceptedSegments.some(s =>
        Math.min(s.shortEnd, chunkEndT) - Math.max(s.shortStart, chunkStartT) > 0.15 &&
        s.confidence >= ALT_ANCHOR_MIN_CONF,
      );
      if (covered) continue;

      const prediction = predictFromAnchors(anchors, chunkMidT);
      if (!prediction) continue;

      // Timestamp-corrected relaxed search window around the prediction.
      const loT = Math.max(0, prediction.predictedMovieTime - ALT_EXPAND_WINDOW_SECONDS);
      const hiT = Math.min(movieDuration, prediction.predictedMovieTime + ALT_EXPAND_WINDOW_SECONDS);
      if (hiT <= loT) continue;
      const loMi = Math.max(0, Math.floor(loT / movieFrameDur));
      const hiMi = Math.min(movieFps.length - 1, Math.ceil(hiT / movieFrameDur));

      // Sample distinct (non-duplicate) short frames across the chunk.
      const sis: number[] = [];
      const step = Math.max(1, Math.floor((chunk.end - chunk.start) / ALT_EXPANSION_SAMPLES));
      for (let si = chunk.start; si <= chunk.end && sis.length < ALT_EXPANSION_SAMPLES; si += step) {
        let pick = si;
        // Walk forward past masked duplicates (bounded to the chunk).
        while (pick <= chunk.end && dupMask?.mask[pick] === 1) pick++;
        if (pick > chunk.end) break;
        if (sis[sis.length - 1] !== pick) sis.push(pick);
      }
      if (sis.length === 0) continue;

      const scanned = await input.scanWindow(sis, loMi, hiMi, ALT_RELAXED_MIN_SIM);
      if (scanned.length === 0) continue;

      const points: TimePoint[] = scanned.map(p => ({
        shortTime: shortFps[p.si].timestamp,
        movieTime: movieFps[p.mi].timestamp,
        sim: p.sim,
      }));

      const models = ransacLines(points, chunkMidT, ALT_EXPANSION_MAX_PER_CHUNK);
      if (models.length === 0) continue;

      const midSi = Math.round((chunk.start + chunk.end) / 2);
      const midShortT = shortFps[midSi].timestamp;

      for (const model of models) {
        const predictedMidMovieT = model.speed * midShortT + model.offset;
        if (predictedMidMovieT < 0 || predictedMidMovieT > movieDuration) continue;
        const seedMi = Math.max(0, Math.min(movieFps.length - 1, Math.round(predictedMidMovieT / movieFrameDur)));
        const offsetFrames = seedMi - midSi;

        const cand = await input.buildCandidate(chunk.start, chunk.end, offsetFrames);
        if (!cand) continue;
        built.push({
          ...cand,
          altExpansion: {
            source: 'anchor-expansion',
            anchorSpeed: prediction.speed,
            ransacSpeed: model.speed,
            ransacInliers: model.inliers.length,
            ransacPoints: points.length,
            anchorsUsed: prediction.anchorsUsed,
            groupedFrom: 1,
          },
        });
      }

      console.log(
        `${tag} Expansion: chunk [${chunk.start}–${chunk.end}] ` +
        `anchorSpeed=${prediction.speed.toFixed(3)}x window=[${loT.toFixed(1)}s–${hiT.toFixed(1)}s] ` +
        `points=${points.length} lines=${models.length}.`,
      );
    }

    // Task 4: speed-aware grouping over the built outputs only.
    const grouped = groupCandidatesSpeedAware(built);
    if (grouped.length !== built.length) {
      console.log(`${tag} Expansion: speed-aware grouping merged ${built.length} → ${grouped.length} candidate(s).`);
    }
    if (grouped.length > 0) {
      console.log(`${tag} Expansion: appending ${grouped.length} anchor-expansion candidate(s) to the alt pool.`);
    }
    return grouped;
  } catch (e: any) {
    console.warn(`${tag} Alt-candidate expansion failed (non-fatal): ${e?.message || e}`);
    return [];
  }
}

/** Shared seconds-based alt-pool dedup separation: converts ALT_DEDUP_SECONDS
 *  into an alignment-offset separation in frames for the given frame duration.
 *  Default (2.0s @ 25fps grid) reproduces the historical 50-frame constant
 *  exactly, so existing behavior is unchanged until the env var is set. */
export const ALT_DEDUP_SECONDS = envNum('ALT_DEDUP_SECONDS', 2.0, 0.2, 30);

export function altDedupSeparationFrames(frameDurationSeconds: number, fallbackFrames: number): number {
  if (!(frameDurationSeconds > 0)) return fallbackFrames;
  return Math.max(1, Math.round(ALT_DEDUP_SECONDS / frameDurationSeconds));
}
