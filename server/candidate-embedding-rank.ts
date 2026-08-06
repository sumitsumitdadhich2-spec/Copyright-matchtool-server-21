/**
 * Crop-robust embedding ranking for the CANDIDATE system only.
 *
 * Problem this solves: short-form videos are usually a 9:16 vertical crop
 * cut from ANY horizontal position (left / center / right) of the 16:9
 * movie frame. A plain full-frame comparison (hash or embedding) sees only
 * ~40% shared pixels in that case, so the correct movie location often
 * ranks BELOW wrong-but-globally-similar locations, and the VLM then wastes
 * its attempts on the wrong candidates (or worse, accepts one).
 *
 * Fix: for every candidate, the movie frame is compared against the short
 * frame in FOUR variants — the full frame plus left / center / right 9:16
 * crops (extracted directly by ffmpeg, no extra image library). Each
 * variant is embedded with the same lazily-downloaded local CLIP model the
 * embedding gate already uses, and the candidate's score is the MAX cosine
 * similarity across variants. Wherever the editor cropped from, one of the
 * variants lines up and the true location rises to the top of the pool.
 *
 * Scope guarantees (user requirement — never touch the main matching pass):
 *  - Consumed ONLY by candidate-side modules (candidate-retry.ts,
 *    deferred-recovery.ts). Never imported by matching-engine.ts,
 *    vlm-segment-resolver.ts's main pass, or server.ts's first pass.
 *  - Ranking only REORDERS which untried candidates get VLM-verified first.
 *    It never adds/removes candidates, never accepts/rejects anything by
 *    itself — the VLM (+ embedding gate) verdicts stay the only gate.
 *  - Fully fail-safe: if the CLIP model can't load (first-run download
 *    blocked, unsupported platform) or any ffmpeg/embedding step fails, the
 *    ranker returns null and callers keep their previous candidate order.
 *
 * Runtime: everything is local CPU inside this app — the ~150MB CLIP
 * weights auto-download to the HuggingFace cache on first use. Nothing to
 * install on the Colab VLM side.
 */
import { spawn } from 'child_process';
import { makeCleanEnv } from './pipeline';
import { MatchedSegment } from './candidate-matching-engine';
import { pickVerificationFramePairs } from './vlm-segment-resolver';
import { embedFrameB64, cosineSimilarity } from './embedding-gate';

/** Horizontal 9:16 windows tested on the movie frame, plus the full frame. */
export type MovieCropVariant = 'full' | 'left' | 'center' | 'right';

const CROP_VARIANTS: MovieCropVariant[] = ['full', 'left', 'center', 'right'];

/**
 * ffmpeg crop-filter expression for a 9:16 vertical window at the given
 * horizontal position. `min(iw\,ih*9/16)` keeps it valid even for movies
 * narrower than 9:16 (crop then degenerates to the full width). The comma
 * inside min() must be escaped so the filtergraph parser doesn't treat it
 * as a filter separator.
 */
function cropFilterFor(variant: Exclude<MovieCropVariant, 'full'>): string {
  const w = 'min(iw\\,ih*9/16)';
  const x =
    variant === 'left' ? '0' :
    variant === 'center' ? '(iw-ow)/2' :
    'iw-ow';
  return `crop=w=${w}:h=ih:x=${x}:y=0`;
}

/**
 * Grab one JPEG frame at `timestampSeconds`, optionally cropped to a 9:16
 * window, base64-encoded. Same ffmpeg invocation style as
 * vlm-verify.ts's extractFrameAsBase64 (which stays untouched).
 */
function extractFrameVariantAsBase64(
  videoPath: string,
  timestampSeconds: number,
  variant: MovieCropVariant,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const ts = Math.max(0, timestampSeconds).toFixed(3);
    const args = ['-ss', ts, '-i', videoPath, '-frames:v', '1'];
    if (variant !== 'full') args.push('-vf', cropFilterFor(variant));
    args.push('-f', 'image2pipe', '-vcodec', 'mjpeg', '-q:v', '3', 'pipe:1');

    const proc = spawn('ffmpeg', args, { env: makeCleanEnv() });
    const chunks: Buffer[] = [];
    let stderr = '';
    proc.stdout.on('data', (c: Buffer) => chunks.push(c));
    proc.stderr.on('data', (c: Buffer) => { stderr += c.toString(); });
    proc.on('error', reject);
    proc.on('close', (code) => {
      const buf = Buffer.concat(chunks);
      if (code !== 0 || buf.length === 0) {
        reject(new Error(`ffmpeg ${variant}-crop frame extraction failed (code ${code}) at t=${ts}s: ${stderr.slice(-300)}`));
        return;
      }
      resolve(buf.toString('base64'));
    });
  });
}

export interface CandidateEmbedScore {
  /** Index into the caller's candidates array. */
  index: number;
  /** Max cosine similarity across movie-frame variants (higher = better). */
  score: number;
  /** Which movie-frame variant produced the max — great for log forensics. */
  bestVariant: MovieCropVariant;
}

/**
 * Score one candidate: short mid-frame embedding vs. the movie frame's four
 * variants, MAX cosine wins. Returns null when scoring was impossible
 * (model unavailable / extraction failed) — caller decides the fallback.
 */
async function scoreCandidate(
  segment: MatchedSegment,
  shortVideoPath: string,
  movieVideoPath: string,
  shortEmbedCache: Map<string, Float32Array | null>,
): Promise<{ score: number; bestVariant: MovieCropVariant } | null> {
  const pair = pickVerificationFramePairs(segment)[0];
  if (!pair) return null;

  // Short frame embedding — cached by (rounded) timestamp, since many
  // candidates for the same short-clip range share the same short frame.
  const cacheKey = pair.shortTime.toFixed(2);
  let shortEmb = shortEmbedCache.get(cacheKey);
  if (shortEmb === undefined) {
    try {
      const shortB64 = await extractFrameVariantAsBase64(shortVideoPath, pair.shortTime, 'full');
      shortEmb = await embedFrameB64(shortB64);
    } catch {
      shortEmb = null;
    }
    shortEmbedCache.set(cacheKey, shortEmb ?? null);
  }
  if (!shortEmb) return null;

  let best = -1;
  let bestVariant: MovieCropVariant = 'full';
  for (const variant of CROP_VARIANTS) {
    try {
      const movieB64 = await extractFrameVariantAsBase64(movieVideoPath, pair.movieTime, variant);
      const movieEmb = await embedFrameB64(movieB64);
      if (!movieEmb) continue;
      const sim = cosineSimilarity(shortEmb, movieEmb);
      if (sim > best) {
        best = sim;
        bestVariant = variant;
      }
    } catch {
      // One failed variant never sinks the candidate — try the rest.
    }
  }
  return best >= 0 ? { score: best, bestVariant } : null;
}

/**
 * Rank a set of candidate indexes by crop-robust embedding similarity,
 * best first. `indexes` point into `candidates`; the returned array is a
 * REORDERED COPY of exactly those indexes (same members, new order).
 *
 * Returns null when ranking is unavailable as a whole (e.g. embedding model
 * failed to load) so callers keep their existing order — identical
 * fail-safe contract to embedding-gate.ts. Candidates that individually
 * fail to score are kept at the END in their original relative order (an
 * unscorable candidate should not beat a scored one, but must never be
 * dropped either — the VLM still gets to try it).
 */
export async function rankCandidatesCropRobust(
  candidates: Array<{ segment: MatchedSegment }>,
  indexes: number[],
  shortVideoPath: string,
  movieVideoPath: string,
  label = 'CandidateRank',
): Promise<number[] | null> {
  if (indexes.length <= 1) return null; // nothing to reorder

  const shortEmbedCache = new Map<string, Float32Array | null>();
  const scored: CandidateEmbedScore[] = [];
  const unscored: number[] = [];

  for (const idx of indexes) {
    const cand = candidates[idx];
    if (!cand) { unscored.push(idx); continue; }
    const res = await scoreCandidate(cand.segment, shortVideoPath, movieVideoPath, shortEmbedCache);
    if (res) scored.push({ index: idx, score: res.score, bestVariant: res.bestVariant });
    else unscored.push(idx);
  }

  // If NOTHING could be scored the model is effectively unavailable —
  // signal "no ranking" so callers keep their previous order untouched.
  if (scored.length === 0) return null;

  scored.sort((a, b) => b.score - a.score);
  console.log(
    `[${label}] Crop-robust embedding ranking: ` +
    scored.map(s => `#${s.index} sim=${s.score.toFixed(3)} (${s.bestVariant})`).join(', ') +
    (unscored.length ? ` | unscored kept last: ${unscored.join(', ')}` : ''),
  );

  return [...scored.map(s => s.index), ...unscored];
}
