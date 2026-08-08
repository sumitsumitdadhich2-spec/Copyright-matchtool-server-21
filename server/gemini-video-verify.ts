/**
 * FULL-SEGMENT VIDEO verification via the Gemini Files API.
 *
 * Instead of comparing a handful of still frames, the ENTIRE matched segment
 * (~10s) is cut from both videos with ffmpeg, uploaded through the Files API,
 * and judged by Gemini in ONE request containing BOTH clips (2 videos per
 * request is confirmed to work; costs exactly 1 RPD per comparison).
 *
 * Model chain (free tier, confirmed from the AI Studio rate-limit page):
 *   1. gemini-3.5-flash-lite  — 15 RPM / 250K TPM / 500 RPD
 *   2. gemini-3.1-flash-lite  — 15 RPM / 250K TPM / 500 RPD (separate pool)
 * Each model has its OWN daily pool, so the chain gives ~1000 comparisons/day
 * per API key (per Google Cloud project). When BOTH pools are exhausted the
 * status below flags dailyLimitReached so the app shows
 * "Gemini key limit over for today — add a new key".
 *
 * Contracts (same as gemini-vlm.ts):
 *  - Every failure path returns null, NEVER throws. Callers fall back to the
 *    legacy frame-composite flow.
 *  - Per-minute 429 -> wait (honoring retryDelay) and retry, work never dies.
 *  - Per-day 429 on a model -> that model is benched until Pacific midnight,
 *    the next model in the chain takes over immediately.
 *
 * Uploaded segment clips are cached by (path, start, end) for 47h — the Files
 * API keeps files 48h, so re-checking the same movie segment against several
 * shorts never re-uploads.
 */
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { makeCleanEnv } from './pipeline';
import { FFMPEG_BIN } from './ffmpeg-path';
import { parseVerdictJson } from './gemini-vlm';

const API_BASE = 'https://generativelanguage.googleapis.com';

export function geminiVideoConfigured(): boolean {
  return !!process.env.GEMINI_API_KEY && process.env.GEMINI_VIDEO_VERIFY !== '0';
}

/** Model chain, highest priority first. Override: GEMINI_VIDEO_MODELS="a,b,c" */
function modelChain(): string[] {
  const env = process.env.GEMINI_VIDEO_MODELS;
  if (env) return env.split(',').map((s) => s.trim()).filter(Boolean);
  return ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite'];
}

const RPD_LIMIT = Number(process.env.GEMINI_VIDEO_RPD) || 500; // per model per day
const RPM_LIMIT = Number(process.env.GEMINI_VIDEO_RPM) || 15;
const REQUEST_TIMEOUT_MS = Number(process.env.GEMINI_VIDEO_TIMEOUT_MS) || 120_000;
const UPLOAD_TIMEOUT_MS = 90_000;
const FILE_ACTIVE_POLL_MS = 2_000;
const FILE_ACTIVE_MAX_WAIT_MS = 120_000;
const FILE_CACHE_TTL_MS = 47 * 60 * 60 * 1000; // Files API keeps 48h; stay under
const MAX_CLIP_SECONDS = Number(process.env.GEMINI_VIDEO_MAX_CLIP_S) || 120;
const PER_MINUTE_MAX_RETRIES = Number(process.env.GEMINI_VIDEO_429_MAX_RETRIES) || 30;

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ---------------------------------------------------------------------------
// Per-day budget, keyed by PACIFIC-TIME date (Google resets free-tier daily
// quotas at midnight Pacific, not UTC).
// ---------------------------------------------------------------------------
function pacificDay(): string {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/Los_Angeles' });
}

interface ModelBudget {
  usedToday: number;
  dailyExhausted: boolean;
  exhaustedSince: number;
}

let budgetDay = '';
const budgets = new Map<string, ModelBudget>();
let rateLimitWaiting = false;

function rollDay(): void {
  const today = pacificDay();
  if (today !== budgetDay) {
    budgetDay = today;
    budgets.clear();
  }
}

function budgetFor(model: string): ModelBudget {
  rollDay();
  let b = budgets.get(model);
  if (!b) {
    b = { usedToday: 0, dailyExhausted: false, exhaustedSince: 0 };
    budgets.set(model, b);
  }
  return b;
}

/** First model in the chain that still has daily budget, or null. */
function pickModel(): string | null {
  for (const m of modelChain()) {
    const b = budgetFor(m);
    if (!b.dailyExhausted && b.usedToday < RPD_LIMIT) return m;
  }
  return null;
}

export function geminiVideoAvailable(): boolean {
  return geminiVideoConfigured() && pickModel() !== null;
}

export interface GeminiVideoStatus {
  configured: boolean;
  models: Array<{
    model: string;
    usedToday: number;
    rpdLimit: number;
    dailyExhausted: boolean;
  }>;
  activeModel: string | null;
  usedToday: number;
  rpmLimit: number;
  /** True only when EVERY model in the chain is out of daily quota. */
  dailyLimitReached: boolean;
  dailyLimitSince: number | null;
  rateLimitWaiting: boolean;
}

export function getGeminiVideoStatus(): GeminiVideoStatus {
  rollDay();
  const models = modelChain().map((m) => {
    const b = budgetFor(m);
    return {
      model: m,
      usedToday: b.usedToday,
      rpdLimit: RPD_LIMIT,
      dailyExhausted: b.dailyExhausted || b.usedToday >= RPD_LIMIT,
    };
  });
  const allOut = models.every((m) => m.dailyExhausted);
  const since = models
    .map((m) => budgetFor(m.model).exhaustedSince)
    .filter((t) => t > 0)
    .sort((a, b) => b - a)[0];
  return {
    configured: geminiVideoConfigured(),
    models,
    activeModel: pickModel(),
    usedToday: models.reduce((s, m) => s + m.usedToday, 0),
    rpmLimit: RPM_LIMIT,
    dailyLimitReached: geminiVideoConfigured() && allOut,
    dailyLimitSince: allOut && since ? since : null,
    rateLimitWaiting,
  };
}

// ---------------------------------------------------------------------------
// Sliding-window RPM pacer (shared across models — conservative and simple).
// ---------------------------------------------------------------------------
const requestTimestamps: number[] = [];
let pacerChain: Promise<void> = Promise.resolve();

function takeRpmSlot(): Promise<void> {
  const waitTurn = pacerChain.then(async () => {
    for (;;) {
      const now = Date.now();
      while (requestTimestamps.length && now - requestTimestamps[0] >= 60_000) {
        requestTimestamps.shift();
      }
      if (requestTimestamps.length < RPM_LIMIT) {
        requestTimestamps.push(Date.now());
        return;
      }
      const waitMs = 60_000 - (now - requestTimestamps[0]) + 250;
      await sleep(Math.max(250, waitMs));
    }
  });
  pacerChain = waitTurn.catch(() => {});
  return waitTurn;
}

// ---------------------------------------------------------------------------
// ffmpeg: cut [start, end] of a video into a small temp mp4 (video-only —
// shorts ka audio aksar replace hota hai, isliye judgement sirf visuals par).
// 480p / CRF 30 keeps a 10s clip ~1MB, upload fast, tokens low.
// ---------------------------------------------------------------------------
function extractClip(
  videoPath: string,
  startS: number,
  endS: number,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const start = Math.max(0, startS);
    const dur = Math.min(Math.max(0.5, endS - start), MAX_CLIP_SECONDS);
    const out = path.join(
      os.tmpdir(),
      `gvv-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.mp4`,
    );
    const proc = spawn(
      FFMPEG_BIN,
      [
        '-ss', start.toFixed(3),
        '-t', dur.toFixed(3),
        '-i', videoPath,
        '-an',
        '-vf', "scale='min(854,iw)':-2",
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '30',
        '-movflags', '+faststart',
        '-y', out,
      ],
      { env: makeCleanEnv() },
    );
    let stderr = '';
    proc.stderr.on('data', (c: Buffer) => { stderr += c.toString(); });
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0 || !fs.existsSync(out) || fs.statSync(out).size === 0) {
        try { fs.unlinkSync(out); } catch { /* ignore */ }
        reject(new Error(`ffmpeg clip extraction failed (code ${code}): ${stderr.slice(-300)}`));
        return;
      }
      resolve(out);
    });
  });
}

// ---------------------------------------------------------------------------
// Files API: resumable upload -> wait ACTIVE -> return file URI.
// Uploaded-clip cache: same (path,start,end) is never uploaded twice in 47h.
// ---------------------------------------------------------------------------
interface CachedFile { uri: string; expiresAt: number; }
const fileCache = new Map<string, CachedFile>();

function cacheKey(videoPath: string, startS: number, endS: number): string {
  return `${videoPath}::${startS.toFixed(2)}-${endS.toFixed(2)}`;
}

async function uploadFile(localPath: string, displayName: string): Promise<string | null> {
  const apiKey = process.env.GEMINI_API_KEY as string;
  const size = fs.statSync(localPath).size;

  // 1. Start resumable session.
  const startController = new AbortController();
  const startTimer = setTimeout(() => startController.abort(), UPLOAD_TIMEOUT_MS);
  let uploadUrl: string | null = null;
  try {
    const res = await fetch(`${API_BASE}/upload/v1beta/files`, {
      method: 'POST',
      headers: {
        'x-goog-api-key': apiKey,
        'X-Goog-Upload-Protocol': 'resumable',
        'X-Goog-Upload-Command': 'start',
        'X-Goog-Upload-Header-Content-Length': String(size),
        'X-Goog-Upload-Header-Content-Type': 'video/mp4',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file: { display_name: displayName } }),
      signal: startController.signal,
    });
    if (!res.ok) {
      console.warn(`[GeminiVideo] Upload start failed: HTTP ${res.status}`);
      return null;
    }
    uploadUrl = res.headers.get('x-goog-upload-url');
  } catch (err: any) {
    console.warn(`[GeminiVideo] Upload start error: ${err?.message || err}`);
    return null;
  } finally {
    clearTimeout(startTimer);
  }
  if (!uploadUrl) {
    console.warn('[GeminiVideo] Upload start returned no upload URL');
    return null;
  }

  // 2. Upload bytes + finalize.
  const upController = new AbortController();
  const upTimer = setTimeout(() => upController.abort(), UPLOAD_TIMEOUT_MS);
  let fileName: string | null = null;
  let fileUri: string | null = null;
  let state: string | null = null;
  try {
    const data = fs.readFileSync(localPath);
    const res = await fetch(uploadUrl, {
      method: 'POST',
      headers: {
        'Content-Length': String(size),
        'X-Goog-Upload-Offset': '0',
        'X-Goog-Upload-Command': 'upload, finalize',
      },
      body: data,
      signal: upController.signal,
    });
    if (!res.ok) {
      console.warn(`[GeminiVideo] Upload finalize failed: HTTP ${res.status}`);
      return null;
    }
    const json: any = await res.json();
    fileName = json?.file?.name || null;
    fileUri = json?.file?.uri || null;
    state = json?.file?.state || null;
  } catch (err: any) {
    console.warn(`[GeminiVideo] Upload error: ${err?.message || err}`);
    return null;
  } finally {
    clearTimeout(upTimer);
  }
  if (!fileUri || !fileName) return null;

  // 3. Poll until ACTIVE (processing is ~8s for a 7.5MB clip; ours are ~1MB).
  const deadline = Date.now() + FILE_ACTIVE_MAX_WAIT_MS;
  while (state !== 'ACTIVE') {
    if (state === 'FAILED') {
      console.warn('[GeminiVideo] File processing FAILED');
      return null;
    }
    if (Date.now() > deadline) {
      console.warn('[GeminiVideo] File never became ACTIVE within the wait budget');
      return null;
    }
    await sleep(FILE_ACTIVE_POLL_MS);
    try {
      const res = await fetch(`${API_BASE}/v1beta/${fileName}`, {
        headers: { 'x-goog-api-key': apiKey },
      });
      if (!res.ok) continue;
      const json: any = await res.json();
      state = json?.state || null;
    } catch { /* transient poll failure — keep polling until deadline */ }
  }
  return fileUri;
}

/** Extract + upload one segment clip, with 47h URI caching. */
async function getSegmentFileUri(
  videoPath: string,
  startS: number,
  endS: number,
  label: string,
): Promise<string | null> {
  const key = cacheKey(videoPath, startS, endS);
  const cached = fileCache.get(key);
  if (cached && cached.expiresAt > Date.now()) return cached.uri;
  fileCache.delete(key);

  let clipPath: string | null = null;
  try {
    clipPath = await extractClip(videoPath, startS, endS);
    const uri = await uploadFile(clipPath, `${label}-${startS.toFixed(1)}s`);
    if (uri) {
      fileCache.set(key, { uri, expiresAt: Date.now() + FILE_CACHE_TTL_MS });
    }
    return uri;
  } catch (err: any) {
    console.warn(`[GeminiVideo] Clip prep failed (${label}): ${err?.message || err}`);
    return null;
  } finally {
    if (clipPath) { try { fs.unlinkSync(clipPath); } catch { /* ignore */ } }
  }
}

// ---------------------------------------------------------------------------
// THE STRICT PROMPT — the old frame-based VLM made too many mistakes, so this
// one forces a frame-level, moment-by-moment matching procedure and defaults
// to NO on any doubt.
// ---------------------------------------------------------------------------
const VIDEO_VERIFY_PROMPT = `You are a forensic video copyright analyst. You are given TWO video clips:

- VIDEO 1: a segment from a short-form vertical video (a "short").
- VIDEO 2: a candidate segment from a full movie.

QUESTION: Was VIDEO 1 copied from the EXACT same original footage as VIDEO 2 — i.e. the same shot(s), the same take, the same moments of the same recording?

MANDATORY PROCEDURE — do ALL of these steps before answering:
1. Pick at least 3 distinct moments spread across VIDEO 1 (beginning, middle, end).
2. For EACH moment, search VIDEO 2 for the SAME moment: the same subject in the same pose performing the same instant of the same action, with the same camera position and the same background details.
3. Verify TEMPORAL continuity: the ORDER and PROGRESSION of actions/camera movement in VIDEO 1 must appear in VIDEO 2 in the same order (allowing for speed-up/slow-down).
4. Verify at least 2 BACKGROUND MICRO-DETAILS per matched moment (specific objects, signage, furniture, terrain, clothing details, light sources) that appear in BOTH videos.

TRANSFORMATIONS YOU MUST IGNORE (these do NOT make videos different):
- VIDEO 1 is usually a 9:16 VERTICAL CROP cut from ANY horizontal position of VIDEO 2's widescreen frame (left, center, or right — scan VIDEO 2's FULL width including edges). VIDEO 2 containing extra people/objects/scenery on the sides is NOT a difference.
- Resolution, compression artifacts, blur, sharpening.
- Color grading, filters, brightness, contrast, saturation.
- Watermarks, logos, subtitles, captions, stickers, UI overlays, borders.
- Horizontal mirroring (flipped video).
- Speed changes (slow-motion or fast-forward).
- Letterboxing / pillarboxing / zoom.

REASONS YOU MUST ANSWER same=false (be ruthless about these):
- A DIFFERENT shot, angle, take, or moment of the SAME movie (same actors, same costumes, same location, but not the exact same recording instants) -> same=false.
- Merely similar-looking footage (similar rooms, landscapes, crowds, similar action scenes) -> same=false.
- You matched the subjects but the background micro-details do not line up -> same=false.
- The action order/progression differs between the videos -> same=false.
- Any moment from step 1 has NO true counterpart in VIDEO 2 -> same=false.
- You are uncertain, the clips are too dark/blurry to verify micro-details, or you cannot point to CONCRETE shared evidence -> same=false. NEVER guess in favor of a match.

same=true requires ALL of: every sampled moment found in VIDEO 2, temporal order consistent, and background micro-details confirmed. Same footage = same recording, not same movie.

First write your moment-by-moment analysis (2-6 short lines). Then your FINAL line must be ONLY this JSON, nothing after it:
{"same": true|false, "confidence": 0-100}`;

// ---------------------------------------------------------------------------
// 429 handling helpers.
// ---------------------------------------------------------------------------
function is429Daily(bodyText: string): boolean {
  return /PerDay|per day|daily/i.test(bodyText);
}

function retryDelayMsFrom(bodyText: string, res: Response): number {
  const m = bodyText.match(/"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"/);
  if (m) return Math.min(Number(m[1]) * 1000 + 500, 65_000);
  const header = res.headers.get('retry-after');
  const secs = header ? Number(header) : NaN;
  if (isFinite(secs) && secs > 0) return Math.min(secs * 1000, 65_000);
  return 5_000;
}

function benchModel(model: string): void {
  const b = budgetFor(model);
  b.dailyExhausted = true;
  b.exhaustedSince = b.exhaustedSince || Date.now();
  const next = pickModel();
  console.warn(
    `[GeminiVideo] DAILY quota over for ${model}` +
    (next ? ` — switching to ${next}` : ' — NO models left, key limit over for today (UI flagged)'),
  );
}

// ---------------------------------------------------------------------------
// The main entry point: verify one (short segment, movie segment) pair by
// sending BOTH full clips in one generateContent request.
// ---------------------------------------------------------------------------
export async function geminiVerifySegmentVideos(args: {
  shortVideoPath: string;
  shortStart: number;
  shortEnd: number;
  movieVideoPath: string;
  movieStart: number;
  movieEnd: number;
  label?: string;
}): Promise<{ same: boolean; confidencePct: number } | null> {
  if (!geminiVideoConfigured()) return null;
  if (!pickModel()) return null; // all daily pools out — status flags the UI

  const label = args.label || 'segment';

  // Upload both clips (movie side is the cache winner across multiple shorts).
  const [shortUri, movieUri] = await Promise.all([
    getSegmentFileUri(args.shortVideoPath, args.shortStart, args.shortEnd, `short-${label}`),
    getSegmentFileUri(args.movieVideoPath, args.movieStart, args.movieEnd, `movie-${label}`),
  ]);
  if (!shortUri || !movieUri) {
    console.warn(`[GeminiVideo] ${label}: clip upload failed — falling back to frame-based verification`);
    return null;
  }

  const body = {
    contents: [
      {
        parts: [
          { text: 'VIDEO 1 (short clip segment):' },
          { file_data: { mime_type: 'video/mp4', file_uri: shortUri } },
          { text: 'VIDEO 2 (movie candidate segment):' },
          { file_data: { mime_type: 'video/mp4', file_uri: movieUri } },
          { text: VIDEO_VERIFY_PROMPT },
        ],
      },
    ],
    generationConfig: { temperature: 0, maxOutputTokens: 2048 },
  };

  for (let attempt = 1; attempt <= PER_MINUTE_MAX_RETRIES + 1; attempt++) {
    const model = pickModel();
    if (!model) return null; // every pool died mid-loop

    rateLimitWaiting = true;
    await takeRpmSlot();
    rateLimitWaiting = false;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      budgetFor(model).usedToday++;
      const res = await fetch(
        `${API_BASE}/v1beta/models/${encodeURIComponent(model)}:generateContent`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-goog-api-key': process.env.GEMINI_API_KEY as string,
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        },
      );

      if (res.status === 429) {
        clearTimeout(timer);
        const bodyText = await res.text().catch(() => '');
        if (is429Daily(bodyText)) {
          benchModel(model);
          continue; // immediately retry on the next model in the chain
        }
        const delay = retryDelayMsFrom(bodyText, res);
        console.warn(
          `[GeminiVideo] 429 per-minute on ${model} — waiting ${Math.round(delay / 1000)}s ` +
          `(attempt ${attempt}, work will not stop)`,
        );
        rateLimitWaiting = true;
        await sleep(delay);
        rateLimitWaiting = false;
        continue;
      }

      if (res.status === 403 || res.status === 404) {
        // Model not available on this key/project — bench it and move on.
        clearTimeout(timer);
        console.warn(`[GeminiVideo] HTTP ${res.status} on ${model} — model unavailable, trying next in chain`);
        benchModel(model);
        continue;
      }

      if (!res.ok) {
        console.warn(`[GeminiVideo] HTTP ${res.status} ${res.statusText} — falling back to frame-based verification`);
        return null;
      }

      const json: any = await res.json();
      const raw: string = (json?.candidates?.[0]?.content?.parts || [])
        .map((p: any) => p?.text || '')
        .join('');
      const verdict = parseVerdictJson(raw);
      if (!verdict) {
        console.warn(`[GeminiVideo] Unparseable response — fallback: ${String(raw).slice(0, 200)}`);
        return null;
      }
      console.log(
        `[GeminiVideo] ${label}: model=${model} verdict=${verdict.same ? 'SAME' : 'DIFFERENT'} ` +
        `conf=${verdict.confidence} (used today: ${getGeminiVideoStatus().usedToday})`,
      );
      return { same: verdict.same, confidencePct: verdict.confidence };
    } catch (err: any) {
      const reason = err?.name === 'AbortError' ? 'timed out' : (err?.message || String(err));
      console.warn(`[GeminiVideo] Request failed (${reason}) — falling back to frame-based verification`);
      return null;
    } finally {
      clearTimeout(timer);
      rateLimitWaiting = false;
    }
  }
  console.warn('[GeminiVideo] Gave up after prolonged rate-limit waiting — fallback');
  return null;
}
