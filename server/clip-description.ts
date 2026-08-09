/**
 * AI target-clip profiling for the candidate auto-extend / deep-search flow.
 *
 * When every candidate in a segment's initial pool has been rejected, the
 * system doesn't stop — it asks Gemini to WRITE A DETAILED DESCRIPTION of the
 * target clip segment (scene, people, objects, colors, camera work, action,
 * overlays) and to pick which analysis signal is most reliable for THIS clip:
 *
 *   - "hash":      clean, uncropped clip — the perceptual-hash confidence
 *                  ordering is trustworthy, verify highest-confidence first.
 *   - "embedding": cropped / zoomed / filtered / mirrored clip — the
 *                  crop-robust SSCD/CLIP embedding ranking is the better
 *                  signal, verify its order first.
 *   - "combined":  ambiguous — blend both orderings.
 *
 * The description and mode NEVER accept or reject anything. Gemini video
 * verification (vlm-verify.ts) remains the one and only verdict-maker; this
 * module only decides WHICH freshly-discovered candidates get verified first,
 * so the (hard-capped) attempt budget is spent on the most promising ones.
 *
 * Deep-search depth: each Retry click passes a higher `depth`, which asks for
 * a progressively finer description (fine background details, motion
 * patterns, edge content) so repeat searches rank with sharper information
 * instead of re-reading the same coarse summary.
 */
import { MatchedSegment } from './candidate-matching-engine';
import { rankCandidatesCropRobust } from './candidate-embedding-rank';
import { geminiDescribeVideo } from './gemini-vlm';
import { cutSegmentToTempFile, deleteTempSegment } from './segment-cutter';
import { fpsForDuration } from './vlm-verify';

export type RankMode = 'hash' | 'embedding' | 'combined';

export interface TargetClipProfile {
  /** Gemini's detailed prose description of the target clip segment. */
  description: string;
  /** Which candidate-ordering signal Gemini judged most reliable for this clip. */
  recommendedMode: RankMode;
  /** Deep-search depth this profile was generated at (0 = first auto-extend). */
  depth: number;
}

/** Depth-0 profiling prompt: describe + pick the best ranking signal. */
const PROFILE_PROMPT_BASE = `You are profiling a short TARGET CLIP that we are trying to locate inside a
full-length reference movie. Watch the clip carefully, then answer in TWO
parts.

PART 1 — DESCRIPTION (plain text, 5-10 sentences):
Describe everything identifiable: the scene/location, every visible person
(appearance, clothing, accessories), key objects and background details
(furniture, walls, posters, vehicles, signage), dominant colors and lighting,
the exact physical action that unfolds and its order, camera angle and camera
movement (pan/zoom/handheld/static), any cuts/transitions, and any text,
captions, watermarks or UI overlays ON TOP of the footage.

PART 2 — SIGNAL SELECTION (JSON, last line only):
We have two ways to order which movie locations get verified first:
- "hash": perceptual-hash confidence. Reliable when the clip looks like a
  clean, full-frame, unedited excerpt of the movie.
- "embedding": crop-robust visual-embedding similarity. Better when the clip
  looks vertically CROPPED (9:16 slice of a widescreen frame), zoomed,
  mirrored, heavily color-graded/filtered, letterboxed, or buried under large
  overlays.
- "combined": use both, when you cannot tell.

Finish your answer with EXACTLY one JSON object on its own line:
{"recommendedMode": "hash" | "embedding" | "combined"}`;

/** Extra instruction appended for deep-search rounds (depth >= 1). */
function deepAddendum(depth: number): string {
  return `

DEEP-SEARCH ROUND ${depth}: previous coarser searches failed to locate this
clip. Go FINER this time: describe small background elements near the frame
EDGES, secondary/background people, precise motion trajectories and timing,
light direction and shadows, texture/pattern details on clothing and
surfaces, and anything unusual that would fingerprint this exact recording.`;
}

/** Pull the {"recommendedMode": ...} JSON off the tail of Gemini's answer. */
export function parseClipProfile(raw: string, depth: number): TargetClipProfile | null {
  if (!raw || !raw.trim()) return null;
  let mode: RankMode = 'combined';
  let description = raw.trim();
  const jsonMatch = raw.match(/\{[^{}]*"recommendedMode"[^{}]*\}/g);
  if (jsonMatch && jsonMatch.length > 0) {
    try {
      const parsed = JSON.parse(jsonMatch[jsonMatch.length - 1]);
      if (parsed.recommendedMode === 'hash' || parsed.recommendedMode === 'embedding' || parsed.recommendedMode === 'combined') {
        mode = parsed.recommendedMode;
      }
      description = raw.slice(0, raw.lastIndexOf(jsonMatch[jsonMatch.length - 1])).trim() || description;
    } catch { /* keep defaults */ }
  }
  return { description, recommendedMode: mode, depth };
}

/**
 * Cut the target-clip segment out of the short video and ask Gemini to
 * profile it. Returns null on ANY failure (Gemini off, quota parked, cut
 * failed) — callers fall back to mode 'combined' with no description; the
 * search itself still runs.
 */
export async function describeTargetClip(
  shortVideoPath: string,
  shortStart: number,
  shortEnd: number,
  depth: number,
  logLabel = 'ClipProfile',
): Promise<TargetClipProfile | null> {
  let clipPart: string | null = null;
  try {
    clipPart = await cutSegmentToTempFile(shortVideoPath, shortStart, shortEnd, 'target');
    if (!clipPart) return null;
    const prompt = depth > 0 ? PROFILE_PROMPT_BASE + deepAddendum(depth) : PROFILE_PROMPT_BASE;
    const fps = fpsForDuration(Math.max(0, shortEnd - shortStart));
    const raw = await geminiDescribeVideo(clipPart, prompt, fps);
    if (!raw) return null;
    const profile = parseClipProfile(raw, depth);
    if (profile) {
      console.log(
        `[${logLabel}] depth=${depth} recommendedMode=${profile.recommendedMode} ` +
        `description=${profile.description.slice(0, 160).replace(/\s+/g, ' ')}…`
      );
    }
    return profile;
  } catch (err: any) {
    console.warn(`[${logLabel}] describeTargetClip failed: ${err?.message || err}`);
    return null;
  } finally {
    deleteTempSegment(clipPart);
  }
}

/**
 * Order candidate indexes according to the profile's recommended mode.
 * Reorders ONLY — never accepts/rejects/removes a candidate; on any failure
 * the hash-confidence order is kept, exactly like the existing best-effort
 * ranking everywhere else in the pipeline.
 *
 *  hash      -> sort by the matching engine's hash confidence, descending.
 *  embedding -> crop-robust SSCD/CLIP embedding rank (existing ranker).
 *  combined  -> average of each candidate's position in BOTH orderings.
 */
export async function orderCandidatesByMode(
  mode: RankMode,
  candidates: Array<{ segment: MatchedSegment }>,
  indexes: number[],
  shortVideoPath: string,
  movieVideoPath: string,
  logLabel = 'ModeRank',
): Promise<number[]> {
  if (indexes.length <= 1) return indexes;

  const hashOrder = [...indexes].sort(
    (a, b) => (candidates[b].segment.confidence ?? 0) - (candidates[a].segment.confidence ?? 0),
  );
  if (mode === 'hash') return hashOrder;

  let embeddingOrder: number[] | null = null;
  try {
    embeddingOrder = await rankCandidatesCropRobust(
      candidates, indexes, shortVideoPath, movieVideoPath, logLabel,
    );
  } catch { /* ranking is best-effort only */ }

  if (mode === 'embedding') return embeddingOrder ?? hashOrder;

  // combined: average rank position across both signals.
  if (!embeddingOrder) return hashOrder;
  const pos = (order: number[], idx: number) => {
    const p = order.indexOf(idx);
    return p === -1 ? order.length : p;
  };
  return [...indexes].sort(
    (a, b) =>
      (pos(hashOrder, a) + pos(embeddingOrder!, a)) -
      (pos(hashOrder, b) + pos(embeddingOrder!, b)),
  );
}
