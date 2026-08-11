/**
 * EXPERIMENT (not app code): dense full-movie embedding scan for ONE segment.
 *
 * Purpose: find out whether the true movie location for a "retry needed"
 * segment is even present in the candidate pool the app builds, and which
 * crop geometry surfaces it.
 *
 * Why this exists: the app's candidate ranker (server/candidate-embedding-rank.ts)
 * tests 13 movie-frame variants whose geometry is hardcoded to 9:16 vertical
 * crops + 16:9 zooms. This short clip is 1:1 SQUARE and heavily zoomed, so
 * none of those 13 windows ever lines up with it. This script instead derives
 * the crop aspect from the SHORT clip and sweeps a 2D scale/position grid.
 *
 * Usage:
 *   npx tsx experiments/dense-scan.mts --short-start 7.00 --short-end 9.12
 */
import { spawn } from 'child_process';
import { createCanvas, loadImage } from 'canvas';
import * as fs from 'fs';
import * as path from 'path';

import ffmpegStatic from 'ffmpeg-static';
import ffprobeStatic from 'ffprobe-static';

const FFMPEG = ffmpegStatic as unknown as string;
const FFPROBE = ffprobeStatic.path;

const GPU = (process.env.GPU_EMBED_SERVICE_URL || 'https://uncover-given-pebbly.ngrok-free.dev')
  .replace(/\/+$/, '')
  .replace(/\/health$/i, '');

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------
function arg(name: string, dflt: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}

const MOVIE = arg('movie', 'uploads/exp-ref-movie.mp4');
const SHORT = arg('short', 'uploads/exp-short-clip.mp4');
const SHORT_START = Number(arg('short-start', '7.00'));
const SHORT_END = Number(arg('short-end', '9.12'));
const SCAN_FPS = Number(arg('scan-fps', '1'));
const TOP_N = Number(arg('top', '30'));
const EMBED_SIZE = 224;
const MODEL = arg('model', 'sscd');
const OUT = arg('out', `experiments/out/scan_${SHORT_START}_${SHORT_END}.json`);

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
function run(bin: string, args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const p = spawn(bin, args);
    let stdout = '';
    let stderr = '';
    p.stdout.on('data', (c) => (stdout += c.toString()));
    p.stderr.on('data', (c) => (stderr += c.toString()));
    p.on('error', reject);
    p.on('close', (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

async function probe(file: string): Promise<{ width: number; height: number; duration: number }> {
  const { stdout } = await run(FFPROBE, [
    '-v', 'error',
    '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height:format=duration',
    '-of', 'default=nw=1:nk=1',
    file,
  ]);
  const [w, h, d] = stdout.trim().split('\n').map(Number);
  return { width: w, height: h, duration: d };
}

/** Extract every frame at `fps` into dir in ONE ffmpeg pass (fast). */
async function extractSeries(file: string, fps: number, dir: string): Promise<void> {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const r = await run(FFMPEG, [
    '-y', '-v', 'error',
    '-i', file,
    '-vf', `fps=${fps}`,
    '-q:v', '3',
    path.join(dir, '%06d.jpg'),
  ]);
  if (r.code !== 0) throw new Error(`ffmpeg series failed: ${r.stderr.slice(-400)}`);
}

async function extractOne(file: string, t: number, out: string): Promise<void> {
  const r = await run(FFMPEG, ['-y', '-v', 'error', '-ss', t.toFixed(3), '-i', file, '-frames:v', '1', out]);
  if (r.code !== 0) throw new Error(`ffmpeg frame failed at ${t}: ${r.stderr.slice(-300)}`);
}

// ---------------------------------------------------------------------------
// Crop-variant grid, derived from the SHORT clip's real aspect ratio.
// This is the part the app currently gets wrong (hardcoded 9:16).
// ---------------------------------------------------------------------------
interface Variant { name: string; sx: number; sy: number; sw: number; sh: number }

function buildVariants(
  movieW: number,
  movieH: number,
  shortAspect: number,
  scales: number[],
  xs: number[],
  ys: number[],
): Variant[] {
  const out: Variant[] = [];
  // 'full' = whole movie frame, squashed to the embed square (baseline).
  out.push({ name: 'full', sx: 0, sy: 0, sw: movieW, sh: movieH });

  for (const scale of scales) {
    // Largest window of the short's aspect ratio that fits, then scaled down.
    let sh = movieH * scale;
    let sw = sh * shortAspect;
    if (sw > movieW * scale) {
      sw = movieW * scale;
      sh = sw / shortAspect;
    }
    sw = Math.max(16, Math.round(sw));
    sh = Math.max(16, Math.round(sh));
    if (sw > movieW || sh > movieH) continue;

    for (const fx of xs) {
      for (const fy of ys) {
        const sx = Math.round((movieW - sw) * fx);
        const sy = Math.round((movieH - sh) * fy);
        out.push({
          name: `s${scale.toFixed(2)}_x${fx.toFixed(2)}_y${fy.toFixed(2)}`,
          sx, sy, sw, sh,
        });
      }
    }
  }
  return out;
}

/** Resize an image region into EMBED_SIZE² JPEG base64 using node-canvas. */
const canvas = createCanvas(EMBED_SIZE, EMBED_SIZE);
const ctx = canvas.getContext('2d');
function cropToB64(img: any, v: Variant): string {
  ctx.clearRect(0, 0, EMBED_SIZE, EMBED_SIZE);
  ctx.drawImage(img, v.sx, v.sy, v.sw, v.sh, 0, 0, EMBED_SIZE, EMBED_SIZE);
  return canvas.toBuffer('image/jpeg', { quality: 0.72 }).toString('base64');
}

// ---------------------------------------------------------------------------
// GPU service
// ---------------------------------------------------------------------------
interface BatchResult { sims: number[]; max_sim: number; best_index: number }

async function embedBatch(
  batches: Array<{ short: string; variants: string[] }>,
  model: string,
): Promise<BatchResult[]> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180_000);
  try {
    const res = await fetch(`${GPU}/embed_batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
      body: JSON.stringify({ batches, model }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`/embed_batch HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const json: any = await res.json();
    if (!Array.isArray(json.results) || json.results.length !== batches.length) {
      throw new Error(`/embed_batch bad shape`);
    }
    return json.results;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  const movieInfo = await probe(MOVIE);
  const shortInfo = await probe(SHORT);
  const shortAspect = shortInfo.width / shortInfo.height;

  console.log(`[scan] movie ${movieInfo.width}x${movieInfo.height} ${movieInfo.duration.toFixed(2)}s`);
  console.log(`[scan] short ${shortInfo.width}x${shortInfo.height} ${shortInfo.duration.toFixed(2)}s aspect=${shortAspect.toFixed(3)}`);
  console.log(`[scan] target short range ${SHORT_START}s-${SHORT_END}s, scanning FULL movie at ${SCAN_FPS}fps, model=${MODEL}`);

  // ── short query frame (mid of the segment) ────────────────────────────────
  const shortMid = (SHORT_START + SHORT_END) / 2;
  const shortDir = '/tmp/exp/short';
  fs.mkdirSync(shortDir, { recursive: true });
  const shortPath = path.join(shortDir, `q_${shortMid.toFixed(2)}.jpg`);
  await extractOne(SHORT, shortMid, shortPath);
  const shortImg = await loadImage(shortPath);
  const shortB64 = cropToB64(shortImg, { name: 'full', sx: 0, sy: 0, sw: shortImg.width, sh: shortImg.height });

  // ── dense movie frame series (one ffmpeg pass) ────────────────────────────
  const movieDir = '/tmp/exp/movie_series';
  console.log('[scan] extracting movie frame series...');
  await extractSeries(MOVIE, SCAN_FPS, movieDir);
  const files = fs.readdirSync(movieDir).filter((f) => f.endsWith('.jpg')).sort();
  console.log(`[scan] ${files.length} movie frames`);

  // ── crop variant grid derived from the short's aspect ratio ───────────────
  const scales = [1.0, 0.75, 0.55];
  const xs = [0, 0.25, 0.5, 0.75, 1];
  const ys = [0.5];
  const variants = buildVariants(movieInfo.width, movieInfo.height, shortAspect, scales, xs, ys);
  console.log(`[scan] ${variants.length} crop variants/frame (aspect-derived): ${variants.map((v) => v.name).join(', ')}`);

  // ── batched scoring ──────────────────────────────────────────────────────
  const perRequest = Math.max(1, Math.floor(480 / (variants.length + 1)));
  const scores: Array<{ t: number; sim: number; variant: string }> = [];
  const t0 = Date.now();

  for (let i = 0; i < files.length; i += perRequest) {
    const slice = files.slice(i, i + perRequest);
    const batches: Array<{ short: string; variants: string[] }> = [];
    for (const f of slice) {
      const img = await loadImage(path.join(movieDir, f));
      batches.push({ short: shortB64, variants: variants.map((v) => cropToB64(img, v)) });
    }
    const results = await embedBatch(batches, MODEL);
    results.forEach((r, k) => {
      const frameNo = Number(slice[k].replace(/\D/g, ''));
      scores.push({
        t: (frameNo - 1) / SCAN_FPS,
        sim: r.max_sim,
        variant: variants[Math.min(Math.max(0, r.best_index), variants.length - 1)].name,
      });
    });
    const done = Math.min(i + perRequest, files.length);
    process.stdout.write(`\r[scan] ${done}/${files.length} frames (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }
  console.log('');

  scores.sort((a, b) => b.sim - a.sim);
  console.log(`\n[scan] TOP ${TOP_N} movie timestamps for short ${SHORT_START}-${SHORT_END}s:`);
  scores.slice(0, TOP_N).forEach((s, i) => {
    console.log(`  ${String(i + 1).padStart(2)}. t=${s.t.toFixed(2)}s  sim=${s.sim.toFixed(4)}  variant=${s.variant}`);
  });

  const sims = scores.map((s) => s.sim);
  const mean = sims.reduce((a, b) => a + b, 0) / sims.length;
  console.log(`\n[scan] pool stats: max=${sims[0].toFixed(4)} mean=${mean.toFixed(4)} min=${sims[sims.length - 1].toFixed(4)}`);

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify({
    shortRange: [SHORT_START, SHORT_END], model: MODEL, scanFps: SCAN_FPS,
    shortAspect, variants: variants.map((v) => v.name), scores,
  }, null, 2));
  console.log(`[scan] wrote ${OUT}`);
}

main().catch((e) => { console.error('[scan] FAILED', e); process.exit(1); });
