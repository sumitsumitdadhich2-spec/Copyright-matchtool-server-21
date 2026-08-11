/**
 * Dense full-movie embedding scan — a CANDIDATE SOURCE of last resort.
 *
 * ---------------------------------------------------------------------------
 * Why this module exists (proven on real data, not assumed)
 * ---------------------------------------------------------------------------
 * Segments kept ending up in "retry needed" with a top SSCD similarity of only
 * 0.15–0.40, versus 0.6–0.9 for segments that matched. The experiment
 * (experiments/dense-scan.mts, short 7.00s–9.12s of the reference clip pair)
 * showed the cause is NOT the VLM and NOT the ranker's ordering:
 *
 *   The true movie location was never in the candidate pool, because every
 *   crop window the system tests has the WRONG ASPECT RATIO.
 *
 * `src/shared/fingerprint.ts` getCropRects() — mirrored in
 * candidate-embedding-rank.ts — hardcodes 9:16 vertical crops plus 16:9
 * zooms, on the assumption that short-form video is always a 9:16 vertical cut.
 * The failing clip is 1:1 SQUARE (720x720) cut from a 16:9 movie (426x240),
 * so NONE of the 13 windows ever lines up with it and the correct location
 * scores like noise.
 *
 * Measured result for short 7.00s–9.12s, scanning the FULL 492s movie at 1fps
 * with crop windows derived from the SHORT clip's real aspect ratio:
 *
 *   movie t=7.00s  sim=0.7583  variant=s1.00_x0.25_y0.50   <- true match
 *   movie t=8.00s  sim=0.7237  variant=s1.00_x0.25_y0.50
 *   movie t=13.00s sim=0.4834  (next-best, unrelated scene)
 *   pool mean=0.1839
 *
 * A 0.28 margin over the runner-up, and Gemini confirmed it at confidence 100
 * citing the identical burned-in subtitle, wardrobe and setting. The winning
 * window is a full-height 1:1 square crop — a geometry the old 13-variant set
 * cannot express.
 *
 * ---------------------------------------------------------------------------
 * What this module does
 * ---------------------------------------------------------------------------
 * 1. Derives crop windows from the SHORT clip's measured aspect ratio
 *    (`buildAspectAwareVariants`) instead of assuming 9:16.
 * 2. Embeds every crop window of every movie frame at a coarse fps ONCE per
 *    (movie, aspect) and caches that index in memory — so scanning a second
 *    failing segment costs one short-frame embedding, not a second full pass.
 * 3. For a given short-clip range: embeds the short frame, cosine-scores it
 *    against the whole index, applies non-max suppression to get distinct
 *    movie regions, then refines the best regions at a finer fps.
 * 4. Returns MatchedSegment candidates with speedRatio 1.0 (so the
 *    degenerate-guard passes them) for the EXISTING VLM gate to judge.
 *
 * Scope guarantees:
 *  - Never accepts or rejects anything itself. It only ADDS candidates; the
 *    Gemini video-pair verdict in vlm-verify.ts stays the only gate.
 *  - Never touches the hash matching pass or its scoring.
 *  - Fully fail-safe: any ffmpeg/GPU/network failure returns [] and callers
 *    behave exactly as they did before this module existed.
 */
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { createCanvas, loadImage } from 'canvas';
import { FFMPEG_BIN, FFPROBE_BIN } from './ffmpeg-path';
import { makeCleanEnv } from './pipeline';
import { MatchedSegment } from './candidate-matching-engine';

// ---------------------------------------------------------------------------
// Config — all env-overridable, all with the values proven in the experiment
// ---------------------------------------------------------------------------

/** Master switch. Set to "0" to disable the dense scan entirely. */
export const DENSE_SCAN_ENABLED = process.env.DENSE_SCAN_ENABLED !== '0';

/** Coarse full-movie sampling rate. 1fps localized the true match cleanly. */
const COARSE_FPS = Number(process.env.DENSE_SCAN_COARSE_FPS) || 1;

/** Fine sampling rate used only inside the best coarse regions. */
const FINE_FPS = Number(process.env.DENSE_SCAN_FINE_FPS) || 5;

/** Half-width (seconds) of the fine refinement window around a coarse peak. */
const FINE_RADIUS_S = Number(process.env.DENSE_SCAN_FINE_RADIUS_S) || 1.5;

/** How many distinct movie regions to return as candidates. */
const TOP_REGIONS = Number(process.env.DENSE_SCAN_TOP_REGIONS) || 5;

/** Minimum separation (seconds) between two returned regions (NMS radius). */
const NMS_RADIUS_S = Number(process.env.DENSE_SCAN_NMS_RADIUS_S) || 4;

/**
 * Floor on the coarse similarity of a returned region. The true match scored
 * 0.758 while the unrelated pool averaged 0.184, so 0.30 keeps real matches
 * comfortably while dropping pure noise. Never an accept — only a "worth a
 * Gemini call" filter.
 */
const MIN_SIMILARITY = Number(process.env.DENSE_SCAN_MIN_SIMILARITY) || 0.3;

/** Square edge every crop is resized to before embedding. */
const EMBED_SIZE = 224;

/** JPEG quality for crops sent to the GPU service. */
const EMBED_JPEG_QUALITY = 0.72;

/** Images per /embed request (mirrors the service's T4-safe batch cap). */
const GPU_MAX_IMAGES_PER_REQUEST = 64;

const GPU_TIMEOUT_MS = Number(process.env.GPU_EMBED_TIMEOUT_MS) || 30_000;

/** Cap on cached movie indexes (each is a few MB). */
const INDEX_CACHE_MAX = Number(process.env.DENSE_SCAN_INDEX_CACHE_MAX) || 2;

function gpuServiceUrl(): string {
  return (process.env.GPU_EMBED_SERVICE_URL || '')
    .replace(/\/+$/, '')
    .replace(/\/health$/i, '');
}

// ---------------------------------------------------------------------------
// Aspect-aware crop windows — the actual fix
// ---------------------------------------------------------------------------

export interface CropWindow {
  name: string;
  sx: number;
  sy: number;
  sw: number;
  sh: number;
}

/**
 * Build crop windows of the SHORT clip's aspect ratio across a scale/position
 * grid, plus the whole frame as a baseline.
 *
 * `scales` are fractions of the largest window of that aspect ratio which
 * fits inside the movie frame; `xs`/`ys` are 0..1 positions of that window
 * inside the frame. For a 1:1 short over a 426x240 movie, scale 1.0 gives the
 * 240x240 square that the failing clip was actually cut from.
 */
export function buildAspectAwareVariants(
  movieWidth: number,
  movieHeight: number,
  shortAspect: number,
  scales: number[],
  xs: number[],
  ys: number[],
): CropWindow[] {
  const out: CropWindow[] = [{ name: 'full', sx: 0, sy: 0, sw: movieWidth, sh: movieHeight }];
  if (!(shortAspect > 0) || !(movieWidth > 0) || !(movieHeight > 0)) return out;

  // Largest window of the short's aspect ratio that fits in the movie frame.
  let baseW = movieWidth;
  let baseH = movieWidth / shortAspect;
  if (baseH > movieHeight) {
    baseH = movieHeight;
    baseW = movieHeight * shortAspect;
  }

  const seen = new Set<string>(['full']);
  for (const scale of scales) {
    const sw = Math.max(16, Math.round(baseW * scale));
    const sh = Math.max(16, Math.round(baseH * scale));
    if (sw > movieWidth || sh > movieHeight) continue;
    for (const fx of xs) {
      for (const fy of ys) {
        const name = `s${scale.toFixed(2)}_x${fx.toFixed(2)}_y${fy.toFixed(2)}`;
        if (seen.has(name)) continue;
        seen.add(name);
        out.push({
          name,
          sx: Math.round((movieWidth - sw) * Math.min(1, Math.max(0, fx))),
          sy: Math.round((movieHeight - sh) * Math.min(1, Math.max(0, fy))),
          sw,
          sh,
        });
      }
    }
  }
  return out;
}

/** The coarse grid used for the full-movie pass (16 windows for a 1:1 short). */
export function coarseVariants(movieWidth: number, movieHeight: number, shortAspect: number): CropWindow[] {
  return buildAspectAwareVariants(
    movieWidth, movieHeight, shortAspect,
    [1.0, 0.75, 0.55],
    [0, 0.25, 0.5, 0.75, 1],
    [0.5],
  );
}

/** The denser grid used only inside a refinement window. */
export function fineVariants(movieWidth: number, movieHeight: number, shortAspect: number): CropWindow[] {
  return buildAspectAwareVariants(
    movieWidth, movieHeight, shortAspect,
    [1.0, 0.85, 0.7, 0.55, 0.4],
    [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    [0.25, 0.5, 0.75],
  );
}

// ---------------------------------------------------------------------------
// ffmpeg / canvas helpers
// ---------------------------------------------------------------------------

function runProc(bin: string, args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const proc = spawn(bin, args, { env: makeCleanEnv() });
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (c: Buffer) => { stdout += c.toString(); });
    proc.stderr.on('data', (c: Buffer) => { stderr += c.toString(); });
    proc.on('error', (e) => resolve({ code: -1, stdout, stderr: String(e) }));
    proc.on('close', (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

interface Dimensions { width: number; height: number; duration: number }

async function probeVideo(videoPath: string): Promise<Dimensions | null> {
  const { code, stdout } = await runProc(FFPROBE_BIN, [
    '-v', 'error',
    '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height:format=duration',
    '-of', 'default=nw=1:nk=1',
    videoPath,
  ]);
  if (code !== 0) return null;
  const [width, height, duration] = stdout.trim().split('\n').map(Number);
  if (!width || !height) return null;
  return { width, height, duration: Number.isFinite(duration) ? duration : 0 };
}

/** Extract a frame series in ONE ffmpeg pass (hundreds of -ss spawns is the
 *  slow way; this is what makes a full-movie scan affordable). */
async function extractFrameSeries(
  videoPath: string,
  fps: number,
  outDir: string,
  startSeconds?: number,
  durationSeconds?: number,
): Promise<string[]> {
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });
  const args: string[] = ['-y', '-v', 'error'];
  if (startSeconds !== undefined) args.push('-ss', Math.max(0, startSeconds).toFixed(3));
  args.push('-i', videoPath);
  if (durationSeconds !== undefined) args.push('-t', Math.max(0.04, durationSeconds).toFixed(3));
  args.push('-vf', `fps=${fps}`, '-q:v', '3', path.join(outDir, '%06d.jpg'));
  const { code, stderr } = await runProc(FFMPEG_BIN, args);
  if (code !== 0) throw new Error(`ffmpeg frame series failed: ${stderr.slice(-300)}`);
  return fs.readdirSync(outDir).filter((f) => f.endsWith('.jpg')).sort();
}

async function extractSingleFrame(videoPath: string, timestamp: number, outPath: string): Promise<void> {
  const { code, stderr } = await runProc(FFMPEG_BIN, [
    '-y', '-v', 'error',
    '-ss', Math.max(0, timestamp).toFixed(3),
    '-i', videoPath,
    '-frames:v', '1',
    outPath,
  ]);
  if (code !== 0) throw new Error(`ffmpeg single frame failed at ${timestamp}s: ${stderr.slice(-300)}`);
}

// One reusable canvas — crop + resize + JPEG encode, no extra image library.
const cropCanvas = createCanvas(EMBED_SIZE, EMBED_SIZE);
const cropCtx = cropCanvas.getContext('2d');

function cropToBase64(image: any, window: CropWindow): string {
  cropCtx.clearRect(0, 0, EMBED_SIZE, EMBED_SIZE);
  cropCtx.drawImage(image, window.sx, window.sy, window.sw, window.sh, 0, 0, EMBED_SIZE, EMBED_SIZE);
  return cropCanvas.toBuffer('image/jpeg', { quality: EMBED_JPEG_QUALITY }).toString('base64');
}

function wholeFrameToBase64(image: any): string {
  return cropToBase64(image, { name: 'full', sx: 0, sy: 0, sw: image.width, sh: image.height });
}

// ---------------------------------------------------------------------------
// GPU embedding client (fail-safe: null on any problem)
// ---------------------------------------------------------------------------

async function postJson(endpoint: string, body: unknown): Promise<any | null> {
  const base = gpuServiceUrl();
  if (!base) return null;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GPU_TIMEOUT_MS);
  try {
    const res = await fetch(`${base}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** POST /embed in <=64-image chunks. Returns L2-normalized embeddings. */
async function embedImages(imagesB64: string[], model = 'sscd'): Promise<number[][] | null> {
  const out: number[][] = [];
  for (let i = 0; i < imagesB64.length; i += GPU_MAX_IMAGES_PER_REQUEST) {
    const chunk = imagesB64.slice(i, i + GPU_MAX_IMAGES_PER_REQUEST);
    const json = await postJson('/embed', { images: chunk, model });
    if (!json || !Array.isArray(json.embeddings) || json.embeddings.length !== chunk.length) return null;
    out.push(...json.embeddings);
  }
  return out;
}

interface BatchScore { sims: number[]; max_sim: number; best_index: number }

async function embedBatchScore(
  batches: Array<{ short: string; variants: string[] }>,
  model = 'sscd',
): Promise<BatchScore[] | null> {
  const json = await postJson('/embed_batch', { batches, model });
  if (!json || !Array.isArray(json.results) || json.results.length !== batches.length) return null;
  return json.results;
}

// ---------------------------------------------------------------------------
// Movie embedding index — built once per (movie, aspect, fps), cached
// ---------------------------------------------------------------------------

interface MovieIndex {
  /** Movie timestamp of each sampled frame. */
  times: number[];
  /** Crop windows applied to every frame. */
  windows: CropWindow[];
  /** Embedding dimensionality. */
  dim: number;
  /** Flat [frame][window] embeddings: row = frameIdx * windows.length + windowIdx. */
  data: Float32Array;
  dimensions: Dimensions;
}

const indexCache = new Map<string, MovieIndex>();

function indexCacheKey(movieVideoPath: string, shortAspect: number): string {
  return `${movieVideoPath}|${shortAspect.toFixed(4)}|${COARSE_FPS}`;
}

function rememberIndex(key: string, index: MovieIndex): void {
  indexCache.set(key, index);
  while (indexCache.size > INDEX_CACHE_MAX) {
    const oldest = indexCache.keys().next().value;
    if (oldest === undefined) break;
    indexCache.delete(oldest);
  }
}

async function buildMovieIndex(
  movieVideoPath: string,
  shortAspect: number,
  label: string,
): Promise<MovieIndex | null> {
  const key = indexCacheKey(movieVideoPath, shortAspect);
  const cached = indexCache.get(key);
  if (cached) {
    console.log(`[${label}] dense-scan: reusing cached movie index (${cached.times.length} frames x ${cached.windows.length} windows)`);
    return cached;
  }

  const dimensions = await probeVideo(movieVideoPath);
  if (!dimensions) {
    console.log(`[${label}] dense-scan: could not probe movie dimensions — skipping.`);
    return null;
  }

  const windows = coarseVariants(dimensions.width, dimensions.height, shortAspect);
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'dense-scan-'));
  const started = Date.now();

  try {
    const files = await extractFrameSeries(movieVideoPath, COARSE_FPS, workDir);
    if (files.length === 0) {
      console.log(`[${label}] dense-scan: movie frame series empty — skipping.`);
      return null;
    }
    console.log(
      `[${label}] dense-scan: building movie index — ${files.length} frames @ ${COARSE_FPS}fps x ` +
      `${windows.length} aspect-aware windows (short aspect ${shortAspect.toFixed(3)}, ` +
      `movie ${dimensions.width}x${dimensions.height})`,
    );

    const times: number[] = [];
    let data: Float32Array | null = null;
    let dim = 0;
    let written = 0;

    // Embed frame-by-frame in GPU-sized chunks so peak memory stays flat.
    const framesPerRequest = Math.max(1, Math.floor(GPU_MAX_IMAGES_PER_REQUEST / windows.length)) || 1;
    for (let i = 0; i < files.length; i += framesPerRequest) {
      const slice = files.slice(i, i + framesPerRequest);
      const images: string[] = [];
      for (const file of slice) {
        const image = await loadImage(path.join(workDir, file));
        for (const window of windows) images.push(cropToBase64(image, window));
      }
      const embeddings = await embedImages(images);
      if (!embeddings || embeddings.length !== images.length) {
        console.log(`[${label}] dense-scan: GPU embedding failed mid-index — skipping dense scan.`);
        return null;
      }
      if (!data) {
        dim = embeddings[0].length;
        data = new Float32Array(files.length * windows.length * dim);
      }
      for (const embedding of embeddings) {
        data.set(embedding, written);
        written += dim;
      }
      for (const file of slice) {
        const frameNumber = Number(file.replace(/\D/g, ''));
        times.push((frameNumber - 1) / COARSE_FPS);
      }
    }

    if (!data || dim === 0) return null;
    const index: MovieIndex = { times, windows, dim, data, dimensions };
    rememberIndex(key, index);
    console.log(
      `[${label}] dense-scan: movie index ready in ${((Date.now() - started) / 1000).toFixed(1)}s ` +
      `(${times.length} frames, dim=${dim})`,
    );
    return index;
  } catch (err: any) {
    console.log(`[${label}] dense-scan: index build errored (${err?.message || err}) — skipping.`);
    return null;
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

/** Cosine similarity against an L2-normalized row of the flat index. */
function dotWithRow(query: Float32Array, index: MovieIndex, row: number): number {
  const offset = row * index.dim;
  let sum = 0;
  for (let d = 0; d < index.dim; d++) sum += query[d] * index.data[offset + d];
  return sum;
}

export interface DenseScanRegion {
  movieStart: number;
  movieEnd: number;
  /** Best embedding similarity found for this region (0..1). */
  similarity: number;
  /** Crop window that produced it — the forensic record of WHY it matched. */
  variant: string;
  /** Per-frame (shortTime, movieTime, similarity 0-100) trace for the UI. */
  matchSequence: Array<{ shortTime: number; movieTime: number; similarity: number }>;
}

export interface DenseScanOptions {
  shortVideoPath: string;
  movieVideoPath: string;
  shortStart: number;
  shortEnd: number;
  /** Movie timestamps already tried; regions near these are suppressed. */
  excludeMovieTimestamps?: number[];
  maxRegions?: number;
  label?: string;
}

/**
 * Locate the most likely movie regions for one short-clip range by densely
 * scanning the WHOLE movie with aspect-aware crop windows. Returns [] on any
 * failure — callers must treat an empty result as "no extra candidates".
 */
export async function denseScanRegions(options: DenseScanOptions): Promise<DenseScanRegion[]> {
  const {
    shortVideoPath, movieVideoPath, shortStart, shortEnd,
    excludeMovieTimestamps = [], maxRegions = TOP_REGIONS, label = 'DenseScan',
  } = options;

  if (!DENSE_SCAN_ENABLED) return [];
  if (!gpuServiceUrl()) {
    console.log(`[${label}] dense-scan: GPU_EMBED_SERVICE_URL not set — skipping.`);
    return [];
  }
  const shortDuration = shortEnd - shortStart;
  if (!(shortDuration > 0)) return [];

  const shortDimensions = await probeVideo(shortVideoPath);
  if (!shortDimensions) return [];
  const shortAspect = shortDimensions.width / shortDimensions.height;

  const index = await buildMovieIndex(movieVideoPath, shortAspect, label);
  if (!index) return [];

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'dense-scan-q-'));
  try {
    // ── Query embedding: the middle frame of the short-clip range ──────────
    const shortMid = (shortStart + shortEnd) / 2;
    const queryPath = path.join(workDir, 'query.jpg');
    await extractSingleFrame(shortVideoPath, shortMid, queryPath);
    const queryImage = await loadImage(queryPath);
    const queryEmbeddings = await embedImages([wholeFrameToBase64(queryImage)]);
    if (!queryEmbeddings || queryEmbeddings.length !== 1) {
      console.log(`[${label}] dense-scan: could not embed the short query frame — skipping.`);
      return [];
    }
    const query = Float32Array.from(queryEmbeddings[0]);

    // ── Coarse pass: best window per movie frame ──────────────────────────
    const windowCount = index.windows.length;
    const perFrame: Array<{ time: number; similarity: number; variant: string }> = [];
    for (let frame = 0; frame < index.times.length; frame++) {
      let best = -1;
      let bestWindow = 0;
      for (let w = 0; w < windowCount; w++) {
        const similarity = dotWithRow(query, index, frame * windowCount + w);
        if (similarity > best) { best = similarity; bestWindow = w; }
      }
      perFrame.push({ time: index.times[frame], similarity: best, variant: index.windows[bestWindow].name });
    }

    const ranked = [...perFrame].sort((a, b) => b.similarity - a.similarity);
    const mean = perFrame.reduce((a, f) => a + f.similarity, 0) / (perFrame.length || 1);
    console.log(
      `[${label}] dense-scan: short [${shortStart.toFixed(2)}s-${shortEnd.toFixed(2)}s] coarse top: ` +
      ranked.slice(0, 5).map((r) => `t=${r.time.toFixed(2)}s sim=${r.similarity.toFixed(4)} (${r.variant})`).join(', ') +
      ` | pool mean=${mean.toFixed(4)}`,
    );

    // ── Non-max suppression into distinct regions ─────────────────────────
    const picked: Array<{ time: number; similarity: number; variant: string }> = [];
    for (const candidate of ranked) {
      if (picked.length >= maxRegions) break;
      if (candidate.similarity < MIN_SIMILARITY) break;
      if (picked.some((p) => Math.abs(p.time - candidate.time) < NMS_RADIUS_S)) continue;
      if (excludeMovieTimestamps.some((t) => Math.abs(t - candidate.time) < NMS_RADIUS_S)) continue;
      picked.push(candidate);
    }

    if (picked.length === 0) {
      console.log(`[${label}] dense-scan: no region cleared the ${MIN_SIMILARITY} similarity floor.`);
      return [];
    }

    // ── Fine pass: pin down the exact start inside each picked region ─────
    const regions: DenseScanRegion[] = [];
    for (const peak of picked) {
      const refined = await refineRegion(
        peak, index, query, shortVideoPath, movieVideoPath, shortStart, shortEnd, shortAspect, label,
      );
      regions.push(refined);
    }

    regions.sort((a, b) => b.similarity - a.similarity);
    console.log(
      `[${label}] dense-scan: ${regions.length} candidate region(s): ` +
      regions.map((r) => `[${r.movieStart.toFixed(2)}s-${r.movieEnd.toFixed(2)}s] sim=${r.similarity.toFixed(4)} (${r.variant})`).join(', '),
    );
    return regions;
  } catch (err: any) {
    console.log(`[${label}] dense-scan: scan errored (${err?.message || err}) — returning no candidates.`);
    return [];
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
  }
}

/**
 * Refine one coarse peak: re-scan ±FINE_RADIUS_S at FINE_FPS with the denser
 * window grid, and build the per-frame trace the UI timeline needs. On any
 * failure the coarse peak is returned unchanged (never a hard failure).
 */
async function refineRegion(
  peak: { time: number; similarity: number; variant: string },
  index: MovieIndex,
  query: Float32Array,
  shortVideoPath: string,
  movieVideoPath: string,
  shortStart: number,
  shortEnd: number,
  shortAspect: number,
  label: string,
): Promise<DenseScanRegion> {
  const shortDuration = shortEnd - shortStart;
  const fallback: DenseScanRegion = {
    movieStart: peak.time,
    movieEnd: peak.time + shortDuration,
    similarity: peak.similarity,
    variant: peak.variant,
    matchSequence: [
      { shortTime: shortStart, movieTime: peak.time, similarity: Math.round(peak.similarity * 100) },
      { shortTime: (shortStart + shortEnd) / 2, movieTime: peak.time + shortDuration / 2, similarity: Math.round(peak.similarity * 100) },
      { shortTime: shortEnd, movieTime: peak.time + shortDuration, similarity: Math.round(peak.similarity * 100) },
    ],
  };

  const windows = fineVariants(index.dimensions.width, index.dimensions.height, shortAspect);
  const windowStart = Math.max(0, peak.time - FINE_RADIUS_S);
  const windowDuration = FINE_RADIUS_S * 2 + shortDuration;
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'dense-fine-'));

  try {
    const [movieFiles, shortFiles] = await Promise.all([
      extractFrameSeries(movieVideoPath, FINE_FPS, path.join(workDir, 'movie'), windowStart, windowDuration),
      extractFrameSeries(shortVideoPath, FINE_FPS, path.join(workDir, 'short'), shortStart, shortDuration),
    ]);
    if (movieFiles.length === 0 || shortFiles.length === 0) return fallback;

    // Score the short's MIDDLE frame against every fine movie frame/window to
    // find the exact alignment offset.
    const midShort = shortFiles[Math.floor(shortFiles.length / 2)];
    const midImage = await loadImage(path.join(workDir, 'short', midShort));
    const midB64 = wholeFrameToBase64(midImage);

    const batches: Array<{ short: string; variants: string[] }> = [];
    for (const file of movieFiles) {
      const image = await loadImage(path.join(workDir, 'movie', file));
      batches.push({ short: midB64, variants: windows.map((w) => cropToBase64(image, w)) });
    }
    const scores = await embedBatchScore(batches);
    if (!scores) return fallback;

    let bestIdx = 0;
    for (let i = 1; i < scores.length; i++) {
      if (scores[i].max_sim > scores[bestIdx].max_sim) bestIdx = i;
    }
    const bestScore = scores[bestIdx];
    if (bestScore.max_sim < peak.similarity - 0.05) return fallback;

    const midOffsetInShort = Math.floor(shortFiles.length / 2) / FINE_FPS;
    const bestMovieTime = windowStart + bestIdx / FINE_FPS;
    const movieStart = Math.max(0, bestMovieTime - midOffsetInShort);
    const bestWindow = windows[Math.min(Math.max(0, bestScore.best_index), windows.length - 1)];

    // Per-frame trace: each short frame paired with the movie frame at the
    // same offset from the aligned start (speedRatio 1.0 mapping).
    const matchSequence = movieFiles.map((_, i) => {
      const movieTime = windowStart + i / FINE_FPS;
      return {
        shortTime: Math.min(shortEnd, shortStart + (movieTime - movieStart)),
        movieTime,
        similarity: Math.round(Math.max(0, Math.min(1, scores[i].max_sim)) * 100),
      };
    }).filter((f) => f.movieTime >= movieStart - 1e-6 && f.movieTime <= movieStart + shortDuration + 1e-6);

    console.log(
      `[${label}] dense-scan: refined t=${peak.time.toFixed(2)}s -> ${movieStart.toFixed(2)}s ` +
      `sim ${peak.similarity.toFixed(4)} -> ${bestScore.max_sim.toFixed(4)} (${bestWindow.name})`,
    );

    return {
      movieStart,
      movieEnd: movieStart + shortDuration,
      similarity: bestScore.max_sim,
      variant: bestWindow.name,
      matchSequence: matchSequence.length >= 3 ? matchSequence : fallback.matchSequence,
    };
  } catch {
    return fallback;
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// Public API: dense-scan regions as MatchedSegment candidates
// ---------------------------------------------------------------------------

/**
 * Dense-scan the movie for a short-clip range and return the results as
 * MatchedSegment candidates ready to be appended to a candidate pool.
 *
 * speedRatio is 1.0 by construction (the movie span equals the short span),
 * so these candidates always pass degenerate-guard.ts — the structural guard
 * exists to kill frozen/racing hash mappings, which a 1:1 aligned dense-scan
 * region can never be.
 *
 * Returns [] on any failure.
 */
export async function denseScanCandidatesForRange(
  options: DenseScanOptions,
): Promise<MatchedSegment[]> {
  const regions = await denseScanRegions(options);
  const { shortStart, shortEnd } = options;

  return regions.map((region) => ({
    shortStart,
    shortEnd,
    movieStart: region.movieStart,
    movieEnd: region.movieEnd,
    // Embedding similarity expressed on the engine's 0-100 confidence scale.
    // Ordering/display only — never an accept.
    confidence: Math.round(Math.max(0, Math.min(1, region.similarity)) * 100),
    frameCount: region.matchSequence.length,
    isApproximate: true,
    gapCount: 0,
    speedRatio: 1,
    matchSequence: region.matchSequence,
  }));
}
