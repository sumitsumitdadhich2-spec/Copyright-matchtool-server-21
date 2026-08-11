/**
 * EXPERIMENT helper: render a side-by-side [short frame | cropped movie frame]
 * JPEG so a human (or the agent) can eyeball whether a candidate is the real
 * match, and re-score that exact pair on the GPU service.
 *
 * Usage:
 *   npx tsx experiments/preview-pair.mts --short-t 8.06 --movie-t 7.0 \
 *     --variant s1.00_x0.25_y0.50 --out /tmp/exp/pair.jpg
 */
import { spawn } from 'child_process';
import { createCanvas, loadImage } from 'canvas';
import * as fs from 'fs';
import * as path from 'path';
import ffmpegStatic from 'ffmpeg-static';
import ffprobeStatic from 'ffprobe-static';

const FFMPEG = ffmpegStatic as unknown as string;
const FFPROBE = ffprobeStatic.path;
const GPU = (process.env.GPU_EMBED_SERVICE_URL || '').replace(/\/+$/, '').replace(/\/health$/i, '');

function arg(name: string, dflt: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}

const MOVIE = arg('movie', 'uploads/exp-ref-movie.mp4');
const SHORT = arg('short', 'uploads/exp-short-clip.mp4');
const SHORT_T = Number(arg('short-t', '8.06'));
const MOVIE_T = Number(arg('movie-t', '7.0'));
const VARIANT = arg('variant', 's1.00_x0.25_y0.50');
const OUT = arg('out', '/tmp/exp/pair.jpg');
const SIZE = 320;

function run(bin: string, args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const p = spawn(bin, args);
    let stdout = '', stderr = '';
    p.stdout.on('data', (c) => (stdout += c.toString()));
    p.stderr.on('data', (c) => (stderr += c.toString()));
    p.on('error', reject);
    p.on('close', (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

async function probe(file: string) {
  const { stdout } = await run(FFPROBE, [
    '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height', '-of', 'default=nw=1:nk=1', file,
  ]);
  const [width, height] = stdout.trim().split('\n').map(Number);
  return { width, height };
}

async function grab(file: string, t: number, out: string) {
  const r = await run(FFMPEG, ['-y', '-v', 'error', '-ss', t.toFixed(3), '-i', file, '-frames:v', '1', out]);
  if (r.code !== 0) throw new Error(r.stderr.slice(-300));
}

/** Parse "s1.00_x0.25_y0.50" (or "full") into a pixel rect for the movie frame. */
export function variantRect(name: string, movieW: number, movieH: number, shortAspect: number) {
  if (name === 'full') return { sx: 0, sy: 0, sw: movieW, sh: movieH };
  const m = /^s([\d.]+)_x([\d.]+)_y([\d.]+)$/.exec(name);
  if (!m) throw new Error(`bad variant name ${name}`);
  const [scale, fx, fy] = [Number(m[1]), Number(m[2]), Number(m[3])];
  let sh = movieH * scale;
  let sw = sh * shortAspect;
  if (sw > movieW * scale) { sw = movieW * scale; sh = sw / shortAspect; }
  sw = Math.max(16, Math.round(sw));
  sh = Math.max(16, Math.round(sh));
  return { sx: Math.round((movieW - sw) * fx), sy: Math.round((movieH - sh) * fy), sw, sh };
}

async function main() {
  fs.mkdirSync('/tmp/exp', { recursive: true });
  const movieInfo = await probe(MOVIE);
  const shortInfo = await probe(SHORT);
  const shortAspect = shortInfo.width / shortInfo.height;

  const sPath = '/tmp/exp/_pv_short.jpg';
  const mPath = '/tmp/exp/_pv_movie.jpg';
  await grab(SHORT, SHORT_T, sPath);
  await grab(MOVIE, MOVIE_T, mPath);

  const sImg = await loadImage(sPath);
  const mImg = await loadImage(mPath);
  const rect = variantRect(VARIANT, movieInfo.width, movieInfo.height, shortAspect);

  const c = createCanvas(SIZE * 2 + 12, SIZE);
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(sImg, 0, 0, sImg.width, sImg.height, 0, 0, SIZE, SIZE);
  ctx.drawImage(mImg, rect.sx, rect.sy, rect.sw, rect.sh, SIZE + 12, 0, SIZE, SIZE);
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, c.toBuffer('image/jpeg', { quality: 0.9 }));

  console.log(`[pair] short t=${SHORT_T}s | movie t=${MOVIE_T}s variant=${VARIANT} rect=${JSON.stringify(rect)}`);
  console.log(`[pair] wrote ${OUT}`);

  if (GPU) {
    const sq = createCanvas(224, 224);
    const sctx = sq.getContext('2d');
    const toB64 = (img: any, r: { sx: number; sy: number; sw: number; sh: number }) => {
      sctx.clearRect(0, 0, 224, 224);
      sctx.drawImage(img, r.sx, r.sy, r.sw, r.sh, 0, 0, 224, 224);
      return sq.toBuffer('image/jpeg', { quality: 0.72 }).toString('base64');
    };
    for (const model of ['sscd', 'dino', 'clip']) {
      const res = await fetch(`${GPU}/embed_batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          batches: [{
            short: toB64(sImg, { sx: 0, sy: 0, sw: sImg.width, sh: sImg.height }),
            variants: [toB64(mImg, rect)],
          }],
          model,
        }),
      });
      const j: any = await res.json().catch(() => null);
      console.log(`[pair] ${model} sim=${j?.results?.[0]?.max_sim?.toFixed?.(4) ?? 'n/a'}`);
    }
  }
}

main().catch((e) => { console.error('[pair] FAILED', e); process.exit(1); });
