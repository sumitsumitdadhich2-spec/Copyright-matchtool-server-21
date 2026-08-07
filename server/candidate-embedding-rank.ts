/**
 * Crop-robust embedding ranking for the CANDIDATE system only.
 *
 * Problem this solves: short-form videos are usually a 9:16 vertical crop
 * cut from ANY horizontal position (left / center / right) of the 16:9
 * movie frame — and editors also ZOOM (1.25x / 1.5x / 2x) before cropping.
 * A plain full-frame comparison (hash or embedding) sees only ~40% shared
 * pixels in that case, so the correct movie location often ranks BELOW
 * wrong-but-globally-similar locations, and the VLM then wastes its
 * attempts on the wrong candidates (or worse, accepts one).
 *
 * Fix: for every candidate, the movie frame is compared against the short
 * frame across THIRTEEN variants — the exact same crop geometry the hash
 * fingerprint system uses (src/shared/fingerprint.ts getCropRects(), logic
 * copied here, file untouched):
 *   1.     full
 *   2-6.   crop_9_16_0 .. crop_9_16_4  (5 horizontal 9:16 windows,
 *          step = (W - cropW) / 4)
 *   7-9.   zoom_1_25_center / _left / _right
 *   10-12. zoom_1_5_center / _left / _right
 *   13.    zoom_2_0_center
 * Each variant is extracted directly by ffmpeg (no extra image library) and
 * the candidate's score is the MAX cosine similarity across variants.
 * Wherever the editor cropped/zoomed from, one of the variants lines up and
 * the true location rises to the top of the pool.
 *
 * Scoring engines (in preference order, all fail-safe):
 *   1. GPU embedding service (gpu_embedding_service.py on a Colab T4,
 *      reachable via the GPU_EMBED_SERVICE_URL env var — same ngrok pattern
 *      as the VLM service). Primary model: SSCD (copy-detection-specific).
 *      When the SSCD top-2 margin is tight (< 0.05), DINOv2 embeddings
 *      tie-break the top two. For large pools (> 10 candidates) the
 *      service's FAISS index endpoints are used instead of per-pair cosine.
 *   2. Local CLIP ViT-B/32 on CPU (the same lazily-downloaded model the
 *      embedding gate uses) — the exact pre-GPU behavior, used whenever the
 *      GPU service is not configured, unhealthy, times out, or errors.
 *   3. If both engines are unavailable: return null so callers keep their
 *      previous candidate order untouched.
 *
 * Scope guarantees (user requirement — never touch the main matching pass):
 *  - Consumed ONLY by candidate-side modules (candidate-retry.ts,
 *    deferred-recovery.ts). Never imported by matching-engine.ts,
 *    vlm-segment-resolver.ts's main pass, or server.ts's first pass.
 *  - Ranking only REORDERS which untried candidates get VLM-verified first.
 *    It never adds/removes candidates, never accepts/rejects anything by
 *    itself — the VLM (+ embedding gate) verdicts stay the only gate.
 *  - Fully fail-safe: any GPU/network/model/ffmpeg failure silently steps
 *    down to the next engine; candidates that individually fail to score
 *    are kept (at the end), never dropped.
 */
import { spawn } from 'child_process';
import { makeCleanEnv } from './pipeline';
import { MatchedSegment } from './candidate-matching-engine';
import { pickVerificationFramePairs } from './vlm-segment-resolver';
import { embedFrameB64, cosineSimilarity } from './embedding-gate';

// ---------------------------------------------------------------------------
// Config (all overridable via env, all with safe defaults)
// ---------------------------------------------------------------------------

/** Base URL of the GPU embedding service (ngrok link). Empty = disabled. */
function gpuServiceUrl(): string {
  // Users often paste the full health-check link (".../health") instead of the
  // base URL — strip a trailing "/health" and trailing slashes defensively.
  return (process.env.GPU_EMBED_SERVICE_URL || '')
    .replace(/\/+$/, '')
    .replace(/\/health$/i, '');
}

/** Hard timeout for GPU embed/search calls. On expiry -> CPU fallback. */
const GPU_EMBED_TIMEOUT_MS = Number(process.env.GPU_EMBED_TIMEOUT_MS) || 30_000;

/** Health-check timeout (kept short so a dead ngrok link stalls nothing). */
const GPU_HEALTH_TIMEOUT_MS = Number(process.env.GPU_HEALTH_TIMEOUT_MS) || 5_000;

/** Health result is cached this long, so it runs once per run — never per candidate. */
const GPU_HEALTH_CACHE_MS = Number(process.env.GPU_HEALTH_CACHE_MS) || 120_000;

/** SSCD top-2 margin under which DINOv2 tie-breaks the top two candidates. */
const DINO_TIEBREAK_MARGIN = Number(process.env.DINO_TIEBREAK_MARGIN) || 0.05;

/** Above this many candidates, use the service's FAISS index endpoints. */
const FAISS_CANDIDATE_THRESHOLD = Number(process.env.FAISS_CANDIDATE_THRESHOLD) || 10;

/** Max images per /embed request (mirrors the service's T4-safe batch cap). */
const GPU_MAX_IMAGES_PER_REQUEST = 64;

// ---------------------------------------------------------------------------
// Movie-frame crop variants — 13 rects, geometry copied verbatim from
// src/shared/fingerprint.ts getCropRects() (that file is NEVER imported or
// modified; the candidate system keeps its own copy by design).
// ---------------------------------------------------------------------------

export type MovieCropVariant = string;

interface VariantSpec {
  name: MovieCropVariant;
  /** ffmpeg -vf crop filter, or null for the full frame. */
  filter: string | null;
}

interface CropRect { name: string; sx: number; sy: number; sw: number; sh: number }

/**
 * EXACT copy of the geometry in src/shared/fingerprint.ts getCropRects().
 * Keep in lockstep with the fingerprint system so the ranker tests the same
 * 13 windows the hash matcher indexes.
 */
function getCropRectsCopy(width: number, height: number): CropRect[] {
  const rects: CropRect[] = [];

  // 1. Full variant
  rects.push({ name: 'full', sx: 0, sy: 0, sw: width, sh: height });

  // 2. 9:16 variants (5 crops)
  let cropWidth = Math.round(height * (9 / 16));
  if (cropWidth % 2 !== 0) cropWidth--;

  if (cropWidth <= width) {
    const step = (width - cropWidth) / 4;
    for (let i = 0; i < 5; i++) {
      let sx = Math.round(i * step);
      if (sx % 2 !== 0) sx--;
      rects.push({ name: `crop_9_16_${i}`, sx, sy: 0, sw: cropWidth, sh: height });
    }
  } else {
    for (let i = 0; i < 5; i++) {
      rects.push({ name: `crop_9_16_${i}`, sx: 0, sy: 0, sw: width, sh: height });
    }
  }

  // 3. Zoom crops helper
  const addZoomCrops = (zoom: number, namePrefix: string) => {
    let sw = Math.min(width, Math.max(1, Math.round(width / zoom)));
    let sh = Math.min(height, Math.max(1, Math.round(height / zoom)));
    if (sw % 2 !== 0) sw--;
    if (sh % 2 !== 0) sh--;

    let sy = Math.min(height - sh, Math.max(0, Math.round((height - sh) / 2)));
    if (sy % 2 !== 0) sy--;

    let sxCenter = Math.min(width - sw, Math.max(0, Math.round((width - sw) / 2)));
    if (sxCenter % 2 !== 0) sxCenter--;

    rects.push({ name: `${namePrefix}_center`, sx: sxCenter, sy, sw, sh });
    rects.push({ name: `${namePrefix}_left`, sx: 0, sy, sw, sh });

    let sxRight = Math.min(width - sw, Math.max(0, width - sw));
    if (sxRight % 2 !== 0) sxRight--;
    rects.push({ name: `${namePrefix}_right`, sx: sxRight, sy, sw, sh });
  };

  addZoomCrops(1.25, 'zoom_1_25');
  addZoomCrops(1.5, 'zoom_1_5');

  // 2.0x zoom (Center only)
  let sw2 = Math.min(width, Math.max(1, Math.round(width / 2.0)));
  let sh2 = Math.min(height, Math.max(1, Math.round(height / 2.0)));
  if (sw2 % 2 !== 0) sw2--;
  if (sh2 % 2 !== 0) sh2--;

  let sx2 = Math.min(width - sw2, Math.max(0, Math.round((width - sw2) / 2)));
  if (sx2 % 2 !== 0) sx2--;

  let sy2 = Math.min(height - sh2, Math.max(0, Math.round((height - sh2) / 2)));
  if (sy2 % 2 !== 0) sy2--;

  rects.push({ name: 'zoom_2_0_center', sx: sx2, sy: sy2, sw: sw2, sh: sh2 });

  return rects;
}

/**
 * Legacy expression-based 9:16 crop filters (the pre-upgrade 4-variant set).
 * Used ONLY as a fallback when ffprobe can't report the movie's dimensions,
 * so behavior in that edge case is identical to before this upgrade.
 */
function legacyCropFilterFor(variant: 'left' | 'center' | 'right'): string {
  const w = 'min(iw\\,ih*9/16)';
  const x =
    variant === 'left' ? '0' :
    variant === 'center' ? '(iw-ow)/2' :
    'iw-ow';
  return `crop=w=${w}:h=ih:x=${x}:y=0`;
}

/** Probe the video's coded width/height once per run via ffprobe. */
function probeDimensions(videoPath: string): Promise<{ width: number; height: number } | null> {
  return new Promise((resolve) => {
    const proc = spawn(
      'ffprobe',
      ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', videoPath],
      { env: makeCleanEnv() },
    );
    let out = '';
    proc.stdout.on('data', (c: Buffer) => { out += c.toString(); });
    proc.on('error', () => resolve(null));
    proc.on('close', (code) => {
      if (code !== 0) { resolve(null); return; }
      const [w, h] = out.trim().split('x').map(Number);
      if (!w || !h || !isFinite(w) || !isFinite(h)) { resolve(null); return; }
      resolve({ width: w, height: h });
    });
  });
}

/**
 * Build the movie-frame variant list for this run: the 13 fingerprint-style
 * rects when the movie's dimensions are known, otherwise the legacy 4-variant
 * expression crops (identical to pre-upgrade behavior).
 */
async function buildMovieVariants(movieVideoPath: string): Promise<VariantSpec[]> {
  const dims = await probeDimensions(movieVideoPath);
  if (dims) {
    return getCropRectsCopy(dims.width, dims.height).map((r) => ({
      name: r.name,
      filter: r.name === 'full' ? null : `crop=w=${r.sw}:h=${r.sh}:x=${r.sx}:y=${r.sy}`,
    }));
  }
  return [
    { name: 'full', filter: null },
    { name: 'left', filter: legacyCropFilterFor('left') },
    { name: 'center', filter: legacyCropFilterFor('center') },
    { name: 'right', filter: legacyCropFilterFor('right') },
  ];
}

/**
 * Grab one JPEG frame at `timestampSeconds`, optionally cropped by the given
 * ffmpeg filter, base64-encoded. Same ffmpeg invocation style as
 * vlm-verify.ts's extractFrameAsBase64 (which stays untouched).
 */
function extractFrameVariantAsBase64(
  videoPath: string,
  timestampSeconds: number,
  cropFilter: string | null,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const ts = Math.max(0, timestampSeconds).toFixed(3);
    const args = ['-ss', ts, '-i', videoPath, '-frames:v', '1'];
    if (cropFilter) args.push('-vf', cropFilter);
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
        reject(new Error(`ffmpeg crop frame extraction failed (code ${code}) at t=${ts}s: ${stderr.slice(-300)}`));
        return;
      }
      resolve(buf.toString('base64'));
    });
  });
}

// ---------------------------------------------------------------------------
// GPU embedding service client (fail-safe: every helper returns null on any
// error, and callers step down to the local CLIP CPU path)
// ---------------------------------------------------------------------------

let gpuHealthCache: { ok: boolean; at: number } | null = null;

async function fetchJsonWithTimeout(
  url: string,
  init: RequestInit | undefined,
  timeoutMs: number,
): Promise<any | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * One health check per run (cached GPU_HEALTH_CACHE_MS) — never hit per
 * candidate. A dead ngrok link costs at most GPU_HEALTH_TIMEOUT_MS once.
 */
async function gpuServiceAvailable(label: string): Promise<boolean> {
  const base = gpuServiceUrl();
  if (!base) return false;
  const now = Date.now();
  if (gpuHealthCache && now - gpuHealthCache.at < GPU_HEALTH_CACHE_MS) {
    return gpuHealthCache.ok;
  }
  const health = await fetchJsonWithTimeout(`${base}/health`, undefined, GPU_HEALTH_TIMEOUT_MS);
  const ok = !!health && health.status === 'ok' && !!health.models_loaded?.sscd;
  gpuHealthCache = { ok, at: now };
  if (!ok) {
    console.log(`[${label}] GPU embed service unhealthy/unreachable — falling back to local CLIP CPU.`);
  }
  return ok;
}

type GpuModel = 'sscd' | 'dino' | 'clip';

/** POST /embed in <=64-image chunks. Returns L2-normalized embeddings or null. */
async function gpuEmbed(imagesB64: string[], model: GpuModel): Promise<number[][] | null> {
  const base = gpuServiceUrl();
  if (!base) return null;
  const out: number[][] = [];
  for (let i = 0; i < imagesB64.length; i += GPU_MAX_IMAGES_PER_REQUEST) {
    const chunk = imagesB64.slice(i, i + GPU_MAX_IMAGES_PER_REQUEST);
    const json = await fetchJsonWithTimeout(
      `${base}/embed`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: chunk, model }),
      },
      GPU_EMBED_TIMEOUT_MS,
    );
    if (!json || !Array.isArray(json.embeddings) || json.embeddings.length !== chunk.length) {
      return null;
    }
    out.push(...json.embeddings);
  }
  return out;
}

interface GpuBatchItem { short: string; variants: string[] }
interface GpuBatchResult { sims: number[]; max_sim: number; best_index: number }

/**
 * POST /embed_batch — one GPU roundtrip scores every (short, 13-variant)
 * group at once; cosine happens server-side so no embeddings travel back.
 */
async function gpuEmbedBatchScore(
  batches: GpuBatchItem[],
  model: GpuModel,
): Promise<GpuBatchResult[] | null> {
  const base = gpuServiceUrl();
  if (!base) return null;
  const json = await fetchJsonWithTimeout(
    `${base}/embed_batch`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batches, model }),
    },
    GPU_EMBED_TIMEOUT_MS,
  );
  if (!json || !Array.isArray(json.results) || json.results.length !== batches.length) {
    return null;
  }
  return json.results;
}

/**
 * FAISS path for large pools: index all movie-variant embeddings once, then
 * search with each short-frame embedding. Any failure returns null and the
 * caller recomputes with plain cosine (embeddings are cheap to redo via
 * /embed_batch) — never a hard failure.
 */
async function gpuFaissScores(
  movieEmbeddings: number[][],
  shortEmbeddings: number[][],
): Promise<number[][] | null> {
  const base = gpuServiceUrl();
  if (!base) return null;
  const built = await fetchJsonWithTimeout(
    `${base}/index_build`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ embeddings: movieEmbeddings }),
    },
    GPU_EMBED_TIMEOUT_MS,
  );
  if (!built || !built.index_id) return null;

  const searched = await fetchJsonWithTimeout(
    `${base}/index_search`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index_id: built.index_id, queries: shortEmbeddings, k: movieEmbeddings.length }),
    },
    GPU_EMBED_TIMEOUT_MS,
  );
  if (!searched || !Array.isArray(searched.ids) || !Array.isArray(searched.scores)) return null;

  // Re-expand (ids, scores) into dense per-query score rows.
  const rows: number[][] = shortEmbeddings.map(() => new Array(movieEmbeddings.length).fill(-1));
  for (let q = 0; q < shortEmbeddings.length; q++) {
    const ids: number[] = searched.ids[q] || [];
    const scores: number[] = searched.scores[q] || [];
    for (let j = 0; j < ids.length; j++) {
      if (ids[j] >= 0 && ids[j] < movieEmbeddings.length) rows[q][ids[j]] = scores[j];
    }
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

export interface CandidateEmbedScore {
  /** Index into the caller's candidates array. */
  index: number;
  /** Max cosine similarity across movie-frame variants (higher = better). */
  score: number;
  /** Which movie-frame variant produced the max — great for log forensics. */
  bestVariant: MovieCropVariant;
}

interface PreparedCandidate {
  index: number;
  shortB64: string;
  movieVariants: Array<{ name: MovieCropVariant; b64: string }>;
}

/**
 * Extract all frames needed for one candidate: the short mid-frame (cached
 * across candidates sharing the same short timestamp) plus every movie
 * variant that extracts successfully. Returns null when the candidate can't
 * be prepared at all (kept unscored at the end, never dropped).
 */
async function prepareCandidateFrames(
  index: number,
  segment: MatchedSegment,
  shortVideoPath: string,
  movieVideoPath: string,
  variants: VariantSpec[],
  shortFrameCache: Map<string, string | null>,
): Promise<PreparedCandidate | null> {
  const pair = pickVerificationFramePairs(segment)[0];
  if (!pair) return null;

  const cacheKey = pair.shortTime.toFixed(2);
  let shortB64 = shortFrameCache.get(cacheKey);
  if (shortB64 === undefined) {
    try {
      shortB64 = await extractFrameVariantAsBase64(shortVideoPath, pair.shortTime, null);
    } catch {
      shortB64 = null;
    }
    shortFrameCache.set(cacheKey, shortB64 ?? null);
  }
  if (!shortB64) return null;

  const movieVariants: Array<{ name: MovieCropVariant; b64: string }> = [];
  for (const variant of variants) {
    try {
      const b64 = await extractFrameVariantAsBase64(movieVideoPath, pair.movieTime, variant.filter);
      movieVariants.push({ name: variant.name, b64 });
    } catch {
      // One failed variant never sinks the candidate — try the rest.
    }
  }
  if (movieVariants.length === 0) return null;

  return { index, shortB64, movieVariants };
}

/**
 * GPU (SSCD-primary) scoring for all prepared candidates.
 *  - <= FAISS_CANDIDATE_THRESHOLD candidates: one /embed_batch call, cosine
 *    on the server, MAX across variants.
 *  - larger pools: /embed + FAISS /index_build + /index_search (falls back
 *    to /embed_batch, then to null, on any hiccup).
 *  - tight SSCD top-2 margin (< DINO_TIEBREAK_MARGIN): DINOv2 re-scores the
 *    top two and may swap them.
 * Returns null when the GPU path failed as a whole (caller -> CPU path).
 */
async function scoreWithGpu(
  prepared: PreparedCandidate[],
  label: string,
): Promise<CandidateEmbedScore[] | null> {
  if (prepared.length === 0) return null;

  const batches: GpuBatchItem[] = prepared.map((p) => ({
    short: p.shortB64,
    variants: p.movieVariants.map((v) => v.b64),
  }));

  let scored: CandidateEmbedScore[] | null = null;

  // ── FAISS path for big pools ─────────────────────────────────────────────
  if (prepared.length > FAISS_CANDIDATE_THRESHOLD) {
    scored = await scoreWithGpuFaiss(prepared, label);
    if (!scored) {
      console.log(`[${label}] FAISS path failed — retrying via /embed_batch cosine.`);
    }
  }

  // ── Standard single-roundtrip batch path ────────────────────────────────
  if (!scored) {
    const results = await gpuEmbedBatchScore(batches, 'sscd');
    if (!results) return null;
    scored = prepared.map((p, i) => {
      const r = results[i];
      const bestIdx = Math.min(Math.max(0, r.best_index), p.movieVariants.length - 1);
      return { index: p.index, score: r.max_sim, bestVariant: p.movieVariants[bestIdx].name };
    });
  }

  scored.sort((a, b) => b.score - a.score);

  // ── DINOv2 tie-break when SSCD can't separate the top two ───────────────
  if (scored.length >= 2 && scored[0].score - scored[1].score < DINO_TIEBREAK_MARGIN) {
    const topTwo = [scored[0], scored[1]].map(
      (s) => prepared.find((p) => p.index === s.index)!,
    );
    if (topTwo[0] && topTwo[1]) {
      const dinoResults = await gpuEmbedBatchScore(
        topTwo.map((p) => ({ short: p.shortB64, variants: p.movieVariants.map((v) => v.b64) })),
        'dino',
      );
      if (dinoResults) {
        console.log(
          `[${label}] SSCD margin ${(scored[0].score - scored[1].score).toFixed(3)} < ${DINO_TIEBREAK_MARGIN} — ` +
          `DINOv2 tie-break: #${scored[0].index}=${dinoResults[0].max_sim.toFixed(3)} vs ` +
          `#${scored[1].index}=${dinoResults[1].max_sim.toFixed(3)}`,
        );
        if (dinoResults[1].max_sim > dinoResults[0].max_sim) {
          const tmp = scored[0];
          scored[0] = scored[1];
          scored[1] = tmp;
        }
      }
      // dino failure = keep SSCD order, no error (tie-break is best-effort)
    }
  }

  return scored;
}

/** FAISS variant of the GPU path (embeddings via /embed, search on-service). */
async function scoreWithGpuFaiss(
  prepared: PreparedCandidate[],
  label: string,
): Promise<CandidateEmbedScore[] | null> {
  // Deduplicate short frames (many candidates share the same short frame).
  const shortKeys: string[] = [];
  const shortKeyOf = new Map<string, number>(); // b64 -> index into shortKeys
  for (const p of prepared) {
    if (!shortKeyOf.has(p.shortB64)) {
      shortKeyOf.set(p.shortB64, shortKeys.length);
      shortKeys.push(p.shortB64);
    }
  }

  // Flat movie-variant image list, remembering which (candidate, variant)
  // each flat position belongs to.
  const movieImages: string[] = [];
  const flatOwner: Array<{ prepIdx: number; variantIdx: number }> = [];
  prepared.forEach((p, prepIdx) => {
    p.movieVariants.forEach((v, variantIdx) => {
      movieImages.push(v.b64);
      flatOwner.push({ prepIdx, variantIdx });
    });
  });

  const [movieEmbs, shortEmbs] = await Promise.all([
    gpuEmbed(movieImages, 'sscd'),
    gpuEmbed(shortKeys, 'sscd'),
  ]);
  if (!movieEmbs || !shortEmbs) return null;

  const rows = await gpuFaissScores(movieEmbs, shortEmbs);
  if (!rows) return null;

  return prepared.map((p, prepIdx) => {
    const row = rows[shortKeyOf.get(p.shortB64)!] || [];
    let best = -1;
    let bestVariant: MovieCropVariant = p.movieVariants[0].name;
    for (let flat = 0; flat < flatOwner.length; flat++) {
      if (flatOwner[flat].prepIdx !== prepIdx) continue;
      const sim = row[flat];
      if (typeof sim === 'number' && sim > best) {
        best = sim;
        bestVariant = p.movieVariants[flatOwner[flat].variantIdx].name;
      }
    }
    return { index: p.index, score: best, bestVariant };
  });
}

/**
 * Local CLIP CPU scoring — the exact pre-GPU behavior (embedFrameB64 from
 * the embedding gate, MAX cosine across variants), now over 13 variants.
 * Returns null per candidate when the model is unavailable.
 */
async function scoreCandidateCpu(
  p: PreparedCandidate,
  shortEmbedCache: Map<string, Float32Array | null>,
): Promise<{ score: number; bestVariant: MovieCropVariant } | null> {
  let shortEmb = shortEmbedCache.get(p.shortB64);
  if (shortEmb === undefined) {
    shortEmb = await embedFrameB64(p.shortB64);
    shortEmbedCache.set(p.shortB64, shortEmb ?? null);
  }
  if (!shortEmb) return null;

  let best = -1;
  let bestVariant: MovieCropVariant = p.movieVariants[0].name;
  for (const variant of p.movieVariants) {
    const movieEmb = await embedFrameB64(variant.b64);
    if (!movieEmb) continue;
    const sim = cosineSimilarity(shortEmb, movieEmb);
    if (sim > best) {
      best = sim;
      bestVariant = variant.name;
    }
  }
  return best >= 0 ? { score: best, bestVariant } : null;
}

// ---------------------------------------------------------------------------
// Public API — signature and fail-safe contract unchanged
// ---------------------------------------------------------------------------

/**
 * Rank a set of candidate indexes by crop-robust embedding similarity,
 * best first. `indexes` point into `candidates`; the returned array is a
 * REORDERED COPY of exactly those indexes (same members, new order).
 *
 * Returns null when ranking is unavailable as a whole (e.g. every embedding
 * engine failed) so callers keep their existing order — identical fail-safe
 * contract to embedding-gate.ts. Candidates that individually fail to score
 * are kept at the END in their original relative order (an unscorable
 * candidate should not beat a scored one, but must never be dropped either —
 * the VLM still gets to try it).
 */
export async function rankCandidatesCropRobust(
  candidates: Array<{ segment: MatchedSegment }>,
  indexes: number[],
  shortVideoPath: string,
  movieVideoPath: string,
  label = 'CandidateRank',
): Promise<number[] | null> {
  if (indexes.length <= 1) return null; // nothing to reorder

  const variants = await buildMovieVariants(movieVideoPath);

  // ── Frame extraction (shared by both engines) ────────────────────────────
  const shortFrameCache = new Map<string, string | null>();
  const prepared: PreparedCandidate[] = [];
  const unscored: number[] = [];

  for (const idx of indexes) {
    const cand = candidates[idx];
    if (!cand) { unscored.push(idx); continue; }
    const prep = await prepareCandidateFrames(
      idx, cand.segment, shortVideoPath, movieVideoPath, variants, shortFrameCache,
    );
    if (prep) prepared.push(prep);
    else unscored.push(idx);
  }

  if (prepared.length === 0) return null; // extraction impossible — keep old order

  // ── Engine 1: GPU service (SSCD primary, DINOv2 tie-break, FAISS >10) ───
  let scored: CandidateEmbedScore[] | null = null;
  let engine = 'clip-cpu';

  if (await gpuServiceAvailable(label)) {
    scored = await scoreWithGpu(prepared, label);
    if (scored) {
      engine = 'sscd-gpu';
    } else {
      console.log(`[${label}] GPU scoring failed mid-run — falling back to local CLIP CPU.`);
    }
  }

  // ── Engine 2: local CLIP CPU (identical to pre-upgrade behavior) ────────
  if (!scored) {
    const shortEmbedCache = new Map<string, Float32Array | null>();
    const cpuScored: CandidateEmbedScore[] = [];
    for (const p of prepared) {
      const res = await scoreCandidateCpu(p, shortEmbedCache);
      if (res) cpuScored.push({ index: p.index, score: res.score, bestVariant: res.bestVariant });
      else unscored.push(p.index);
    }
    scored = cpuScored;
  } else {
    // GPU path scores every prepared candidate; guard against holes anyway.
    const scoredIdx = new Set(scored.map((s) => s.index));
    for (const p of prepared) {
      if (!scoredIdx.has(p.index)) unscored.push(p.index);
    }
  }

  // If NOTHING could be scored, every engine is effectively unavailable —
  // signal "no ranking" so callers keep their previous order untouched.
  if (scored.length === 0) return null;

  // GPU results arrive already ordered (SSCD sort + possible DINOv2 swap of
  // the top two — re-sorting by raw SSCD score would undo that swap). Only
  // the CPU path needs sorting here.
  if (engine !== 'sscd-gpu') scored.sort((a, b) => b.score - a.score);
  console.log(
    `[${label}] engine=${engine} Crop-robust embedding ranking (${variants.length} variants): ` +
    scored.map(s => `#${s.index} sim=${s.score.toFixed(3)} (${s.bestVariant})`).join(', ') +
    (unscored.length ? ` | unscored kept last: ${unscored.join(', ')}` : ''),
  );

  return [...scored.map(s => s.index), ...unscored];
}
