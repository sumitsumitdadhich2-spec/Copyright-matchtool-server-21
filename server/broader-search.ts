/**
 * Shared broader-search primitive: re-run the real matching engine
 * (candidate-matching-engine's matchVideosFromFiles) restricted to one
 * short-clip range, with a wider frame drift + relaxed similarity floor,
 * purely to surface movie locations the original full-clip scan missed.
 *
 * Extracted from candidate-retry.ts so it can ALSO run during the MAIN
 * pass (vlm-segment-resolver.ts) when the initial candidate pool looks weak
 * (low top hash confidence) — the fix for the observed bug where the true
 * scene (movie 6.88s, confidence 87.3%) was never in the first 11
 * candidates and only surfaced after a manual Retry's broader search.
 *
 * This is NOT a second scoring algorithm — it reuses matchVideosFromFiles
 * verbatim. VLM verification remains the only accept/reject gate.
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { MatchedSegment, matchVideosFromFiles } from './candidate-matching-engine';
import { degenerateCandidateReason } from './degenerate-guard';

/** Cap on how many freshly-discovered candidates a single broader search returns. */
export const BROADER_SEARCH_MAX_NEW = 10;
/** Deliberately looser than the app default (82) — this is a fallback pass
 *  whose only job is to surface locations the original scan missed; VLM
 *  verification is still the real accept/reject gate. */
export const BROADER_SEARCH_MIN_SIMILARITY = 40;
export const BROADER_SEARCH_MIN_CONSECUTIVE_FRAMES = 3;
/** Wider than the app default (3) so a slightly different edit speed/frame
 *  offset than the original scan assumed can still line up. */
export const BROADER_SEARCH_FRAME_DRIFT = 8;
/** Small amount of extra context on each side of the short-clip range so the
 *  scene-chunk splitter has more than a couple of frames to work with. */
export const BROADER_SEARCH_PAD_SECONDS = 1.5;
/** Same drift tolerance used elsewhere (getAlternateCandidatesForRange) to
 *  treat two movie timestamps as "the same location". */
export const SAME_LOCATION_TOLERANCE = 0.5;

/**
 * Each broader-search round widens the net a little more, so repeat rounds
 * surface genuinely new movie locations instead of rediscovering the same
 * ones. Round 0 uses the base values above.
 */
export function broaderSearchParamsForRound(round: number): {
  minSimilarity: number;
  minConsecutiveFrames: number;
  frameDrift: number;
  padSeconds: number;
} {
  const step = Math.max(0, round);
  return {
    // Floor at 20 — below that the engine returns mostly noise.
    minSimilarity: Math.max(20, BROADER_SEARCH_MIN_SIMILARITY - step * 5),
    minConsecutiveFrames: Math.max(2, BROADER_SEARCH_MIN_CONSECUTIVE_FRAMES - (step > 0 ? 1 : 0)),
    frameDrift: Math.min(24, BROADER_SEARCH_FRAME_DRIFT + step * 4),
    padSeconds: Math.min(6, BROADER_SEARCH_PAD_SECONDS + step * 0.75),
  };
}

/**
 * Write a temporary fingerprint file containing only the short-clip frames
 * within [start-pad, end+pad], preserving their original timestamps, so a
 * fresh matchVideosFromFiles() call can be scoped to just this range without
 * touching the real per-job fingerprint files on disk.
 */
async function writeFilteredShortFingerprint(
  shortResultPath: string,
  rangeStart: number,
  rangeEnd: number,
): Promise<string | null> {
  const tempPath = path.join(
    os.tmpdir(),
    `retry-broader-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`,
  );
  const rl = readline.createInterface({
    input: fs.createReadStream(shortResultPath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });
  const lines: string[] = [];
  for await (const line of rl) {
    if (!line.trim()) continue;
    try {
      const frame = JSON.parse(line);
      if (typeof frame.timestamp === 'number' && frame.timestamp >= rangeStart && frame.timestamp <= rangeEnd) {
        lines.push(line);
      }
    } catch { /* skip corrupt line */ }
  }
  if (lines.length === 0) return null;
  await fs.promises.writeFile(tempPath, lines.join('\n') + '\n');
  return tempPath;
}

/**
 * Run one broader-search round for a short-clip range. Returns candidates
 * overlapping the range, deduped by movie location and sorted by hash
 * confidence descending. Never throws on cleanup.
 */
export async function broaderSearchForRange(
  shortStart: number,
  shortEnd: number,
  shortResultPath: string,
  movieResultPath: string,
  round: number,
): Promise<MatchedSegment[]> {
  const params = broaderSearchParamsForRound(round);
  const rangeStart = Math.max(0, shortStart - params.padSeconds);
  const rangeEnd = shortEnd + params.padSeconds;

  const tempShortPath = await writeFilteredShortFingerprint(shortResultPath, rangeStart, rangeEnd);
  if (!tempShortPath) return [];

  try {
    const result = await matchVideosFromFiles(tempShortPath, movieResultPath, {
      minSimilarity: params.minSimilarity,
      minConsecutiveFrames: params.minConsecutiveFrames,
      frameDrift: params.frameDrift,
    });

    const pool = [...(result.segments || []), ...(result.candidatePool || [])];
    const overlapping = pool.filter(seg =>
      Math.min(seg.shortEnd, shortEnd) - Math.max(seg.shortStart, shortStart) > 0.15,
    );

    const deduped: MatchedSegment[] = [];
    for (const seg of overlapping) {
      if (!deduped.some(d => Math.abs(d.movieStart - seg.movieStart) <= SAME_LOCATION_TOLERANCE)) {
        deduped.push(seg);
      }
    }
    deduped.sort((a, b) => b.confidence - a.confidence);
    return deduped;
  } finally {
    fs.promises.unlink(tempShortPath).catch(() => { /* best-effort cleanup */ });
  }
}
