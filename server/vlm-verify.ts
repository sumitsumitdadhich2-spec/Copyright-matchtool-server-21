/**
 * VLM (Qwen2.5-VL) scene verification helpers.
 *
 * These are on-demand, per-frame calls — they never re-decode a whole video.
 * All network/parse failures are treated as "could not verify" (returns null),
 * never as a silent pass or fail, so a segment falls back to being kept as-is
 * rather than the whole match crashing.
 */
import { spawn } from 'child_process';
import { makeCleanEnv } from './pipeline';

export const VLM_ENDPOINT_URL =
  process.env.VLM_ENDPOINT_URL || 'http://localhost:8000/v1/chat/completions';
export const VLM_CONFIDENCE_THRESHOLD =
  Number(process.env.VLM_CONFIDENCE_THRESHOLD) || 80;
export const VLM_MAX_ATTEMPTS =
  Number(process.env.VLM_MAX_ATTEMPTS) || 10;

const VLM_MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct';

/**
 * Node's fetch() collapses DNS failures, connection refusals, timeouts, and
 * TLS errors into the same generic "fetch failed" message — the real reason
 * lives in `err.cause`. Surface it so a dead/misconfigured VLM_ENDPOINT_URL
 * can be diagnosed from logs alone (wrong host vs. server down vs. slow
 * response) instead of guessing. Display-only: never changes control flow,
 * retries, or verdicts — only the text written to the log.
 */
function describeFetchError(err: any): string {
  if (err?.name === 'AbortError') return 'timed out';
  const cause = err?.cause;
  if (cause?.code) return `${cause.code}${cause.message ? ` — ${cause.message}` : ''}`;
  if (cause?.message) return cause.message;
  return err?.message || String(err);
}

// ---------------------------------------------------------------------------
// Concurrency — the llama.cpp VLM server exposes VLM_NUM_SLOTS independent
// KV-cache slots (see resetVlmCache below), so it can genuinely serve this
// many requests in parallel without one call's context bleeding into
// another's. Defaults to that same slot count so segment verification never
// asks the server for more concurrent work than it was provisioned for.
// Override with VLM_CONCURRENCY if the endpoint is reconfigured with a
// different slot count.
//
// NOTE: this governs how many *segments* a caller (vlm-segment-resolver.ts /
// deferred-recovery.ts) processes at once, which indirectly bounds how many
// verifySameScene() calls they issue concurrently. VLM_MAX_CONCURRENT_REQUESTS
// below is a second, independent cap enforced here at the actual network-call
// layer, covering every caller (main pass + deferred pass + any future one)
// combined — belt-and-suspenders against the server/tunnel getting more
// concurrent HTTP requests than it can handle.
// ---------------------------------------------------------------------------
export const VLM_CONCURRENCY =
  Number(process.env.VLM_CONCURRENCY) || 4;

// ---------------------------------------------------------------------------
// Request-layer resilience — added to fix VLM request overload on large
// movies (500+ segments). Purely about pacing/retrying network calls to the
// VLM server; never changes which candidate is tried, in what order, or what
// counts as an accept/reject verdict. See replit.md for the full writeup.
//
// - VLM_MAX_CONCURRENT_REQUESTS caps how many verifySameScene() HTTP calls
//   are in flight at once, across the whole pipeline (main pass + deferred
//   pass combined), independent of how many segments are being *processed*
//   concurrently (VLM_CONCURRENCY above). Defaults to the server's slot count.
// - VLM_RESET_MAX_CONCURRENT_REQUESTS is a separate, small allowance for
//   cache-reset/slot-erase calls, so a burst of real verification requests
//   never queues up behind reset traffic and vice versa (point 4: resets are
//   hygiene, not correctness, and must not compete with verification for the
//   same budget).
// - VLM_RETRY_ATTEMPTS / VLM_RETRY_BASE_DELAY_MS control exponential backoff
//   retry of *transient* failures (5xx, connection resets, timeouts) — this
//   is distinct from VLM_MAX_ATTEMPTS (how many different candidates get
//   tried for a segment). A retry here is about getting a real answer for
//   one specific request that failed to complete; it never advances to a
//   different candidate.
// ---------------------------------------------------------------------------
export const VLM_MAX_CONCURRENT_REQUESTS =
  Number(process.env.VLM_MAX_CONCURRENT_REQUESTS) || 4;
export const VLM_RESET_MAX_CONCURRENT_REQUESTS =
  Number(process.env.VLM_RESET_MAX_CONCURRENT_REQUESTS) || Math.min(2, VLM_MAX_CONCURRENT_REQUESTS);
export const VLM_RETRY_ATTEMPTS =
  Number(process.env.VLM_RETRY_ATTEMPTS) || 3;
export const VLM_RETRY_BASE_DELAY_MS =
  Number(process.env.VLM_RETRY_BASE_DELAY_MS) || 1000;

/** Simple counting semaphore — no external deps, just a queue of resolvers. */
class Semaphore {
  private active = 0;
  private queue: Array<() => void> = [];
  constructor(private readonly max: number) {}

  acquire(): Promise<() => void> {
    if (this.active < this.max) {
      this.active++;
      return Promise.resolve(() => this.release());
    }
    return new Promise<() => void>((resolve) => {
      this.queue.push(() => {
        this.active++;
        resolve(() => this.release());
      });
    });
  }

  private release(): void {
    this.active--;
    const next = this.queue.shift();
    if (next) next();
  }
}

// Two independent lanes (point 4): verification calls never wait behind
// reset calls, and a slow/hung reset never eats into verification's budget.
const verifyGate = new Semaphore(Math.max(1, VLM_MAX_CONCURRENT_REQUESTS));
const resetGate = new Semaphore(Math.max(1, VLM_RESET_MAX_CONCURRENT_REQUESTS));

/**
 * Aggregate, process-wide counters for the "inconclusive" outcome (a request
 * that never got a real answer even after retries — server overload/network
 * failure, NOT the VLM saying "no match"). server.ts snapshots/resets these
 * around each match job so it can log and report how many "unmatched"
 * outcomes were genuine content rejections vs. inconclusive-due-to-overload
 * (point 3) — purely additive observability, never consulted by any
 * accept/reject decision.
 */
export const vlmNetworkStats = {
  verifyInconclusive: 0,
  resetInconclusive: 0,
};

export function resetVlmNetworkStats(): void {
  vlmNetworkStats.verifyInconclusive = 0;
  vlmNetworkStats.resetInconclusive = 0;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function backoffDelayMs(attemptJustFailed: number): number {
  return VLM_RETRY_BASE_DELAY_MS * Math.pow(2, attemptJustFailed - 1);
}

/** 5xx = server-side overload/unavailability — exactly the symptom this fixes. */
function isRetryableStatus(status: number): boolean {
  return status >= 500 && status <= 599;
}

/**
 * Connection-level failures that mean "the request never completed", not a
 * real answer — ECONNRESET et al, plus our own client-side timeout
 * (AbortError), which under heavy load is indistinguishable from a slow/
 * overloaded server not responding in time.
 */
function isRetryableError(err: any): boolean {
  if (err?.name === 'AbortError') return true;
  const code = err?.cause?.code || err?.code;
  if (typeof code === 'string' && [
    'ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'EPIPE', 'ENOTFOUND', 'EAI_AGAIN', 'ECONNABORTED',
  ].includes(code)) {
    return true;
  }
  // Node's fetch collapses many socket-level failures into a generic
  // "fetch failed" TypeError with the specific reason only in `cause`. If we
  // can't identify a specific cause at all, still treat it as transient —
  // safer to retry an unrecognized network hiccup than to give up instantly.
  if (err instanceof TypeError && /fetch failed/i.test(err.message || '')) return true;
  return false;
}

interface RetryableFetchResult {
  response: Response;
  /** Total attempts made (1 = succeeded/failed on the first try, no retry needed). */
  attempts: number;
}

/**
 * fetch() with exponential-backoff retry for transient failures only
 * (5xx status, or connection-level errors/timeouts). Non-retryable outcomes
 * (2xx/3xx/4xx responses, or errors that don't look transient) are returned/
 * thrown immediately on the first attempt, so callers' existing handling for
 * those cases is completely unchanged. Capped at VLM_RETRY_ATTEMPTS retries
 * (default 3: waits ~1s, 2s, 4s between attempts).
 */
async function fetchWithRetry(
  url: string,
  init: RequestInit,
  timeoutMs: number,
  label: string,
): Promise<RetryableFetchResult> {
  const maxAttempts = VLM_RETRY_ATTEMPTS + 1;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...init, signal: controller.signal });
      clearTimeout(timeout);
      if (res.ok || !isRetryableStatus(res.status) || attempt === maxAttempts) {
        return { response: res, attempts: attempt };
      }
      console.warn(
        `[VLM] ${label}: transient ${res.status} ${res.statusText} — retrying ` +
        `(attempt ${attempt + 1}/${maxAttempts} in ${backoffDelayMs(attempt)}ms)`
      );
      await sleep(backoffDelayMs(attempt));
    } catch (err: any) {
      clearTimeout(timeout);
      if (!isRetryableError(err) || attempt === maxAttempts) {
        err.vlmAttempts = attempt;
        throw err;
      }
      console.warn(
        `[VLM] ${label}: transient error (${describeFetchError(err)}) — retrying ` +
        `(attempt ${attempt + 1}/${maxAttempts} in ${backoffDelayMs(attempt)}ms)`
      );
      await sleep(backoffDelayMs(attempt));
    }
  }
  // Unreachable — the loop above always returns or throws by the last attempt.
  throw new Error(`${label}: fetchWithRetry exhausted attempts unexpectedly`);
}

/** Grab a single JPEG frame from a video at `timestampSeconds`, base64-encoded. */
export function extractFrameAsBase64(
  videoPath: string,
  timestampSeconds: number,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const ts = Math.max(0, timestampSeconds).toFixed(3);
    const proc = spawn(
      'ffmpeg',
      [
        '-ss', ts,
        '-i', videoPath,
        '-frames:v', '1',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-q:v', '3',
        'pipe:1',
      ],
      { env: makeCleanEnv() },
    );

    const chunks: Buffer[] = [];
    let stderr = '';
    proc.stdout.on('data', (c: Buffer) => chunks.push(c));
    proc.stderr.on('data', (c: Buffer) => { stderr += c.toString(); });
    proc.on('error', reject);
    proc.on('close', (code) => {
      const buf = Buffer.concat(chunks);
      if (code !== 0 || buf.length === 0) {
        reject(new Error(`ffmpeg frame extraction failed (code ${code}) at t=${ts}s: ${stderr.slice(-400)}`));
        return;
      }
      resolve(buf.toString('base64'));
    });
  });
}

/**
 * Ask the VLM whether two frames show the same scene/subject.
 * Returns null (not false) when the call could not be completed or parsed —
 * callers must treat null as "skip VLM gating for this attempt", not a verdict.
 *
 * Transient network failures (5xx, ECONNRESET, timeouts) are retried with
 * backoff before giving up (see fetchWithRetry) and concurrency to the VLM
 * server is capped at VLM_MAX_CONCURRENT_REQUESTS — neither changes this
 * function's contract with callers: same inputs still produce the same
 * {same, confidencePct} | null outcome, just more resilient to overload.
 */
export async function verifySameScene(
  shortFrameB64: string,
  movieFrameB64: string,
): Promise<{ same: boolean; confidencePct: number } | null> {
  return verifySameSceneMulti([{ shortFrameB64, movieFrameB64 }]);
}

/** One (short-frame, movie-frame) pair from the same candidate segment. */
export interface VlmFramePair {
  shortFrameB64: string;
  movieFrameB64: string;
}

/**
 * Multi-pair variant of verifySameScene: sends up to 3 (short, movie) frame
 * pairs from the SAME candidate segment in ONE request (max 6 images), so
 * the VLM can cross-check its verdict across several moments of the segment
 * instead of judging from a single frame pair. Still exactly one HTTP call
 * per attempt — same request count, same timeout, same retry behavior as
 * before, so wall-clock time per attempt does not grow.
 *
 * Decision rule baked into the prompt: if ANY pair clearly shows the same
 * scene, the answer is "same". This is deliberately recall-biased — the
 * whole point of sending extra pairs is to stop the VLM from rejecting a
 * genuinely-matching segment because one unlucky frame (motion blur, cut
 * boundary, overlay) looked different.
 */
export async function verifySameSceneMulti(
  pairs: VlmFramePair[],
): Promise<{ same: boolean; confidencePct: number } | null> {
  const usable = pairs.slice(0, 3);
  if (usable.length === 0) return null;

  const n = usable.length;
  const pairList = usable
    .map((_, i) => `Pair ${i + 1} = images ${i * 2 + 1} & ${i * 2 + 2}`)
    .join('; ');

  const prompt =
    `You are verifying whether a short video clip was copied from a movie. ` +
    `You are given ${n} pair(s) of video frames, ${n * 2} images total, in order: ${pairList}. ` +
    `In each pair the FIRST image is a frame from the short clip and the SECOND image is a frame ` +
    `from the movie. All pairs come from the SAME candidate segment, sampled at different moments.\n\n` +
    `Task: decide if the pairs show the same underlying scene/footage.\n\n` +
    `IMPORTANT — treat frames as the SAME scene even if they differ in:\n` +
    `- compression artifacts, resolution, blur, or sharpness\n` +
    `- cropping, zooming, letterboxing/pillarboxing, or aspect ratio\n` +
    `- color grading, filters, brightness, contrast, or saturation\n` +
    `- watermarks, logos, subtitles, captions, or UI overlays added on top\n` +
    `- horizontal mirroring (flipped image)\n` +
    `- a slightly different instant of the same continuous shot (people/objects moved a little)\n\n` +
    `Judge only the underlying content: same location, same people/characters, same objects, ` +
    `same camera setup or same continuous action.\n\n` +
    `Decision rule: if AT LEAST ONE pair clearly shows the same scene, answer same=true — ` +
    `do NOT reject just because one pair is ambiguous or blurry. ` +
    `Answer same=false ONLY if the pairs clearly show different content ` +
    `(different location AND different people/objects/action).\n\n` +
    `Reply with ONLY JSON, no other text: {"same": true|false, "confidence": 0-100} ` +
    `where confidence is how certain you are of your verdict.`;

  const content: any[] = [{ type: 'text', text: prompt }];
  for (const p of usable) {
    content.push({ type: 'image_url', image_url: { url: `data:image/jpeg;base64,${p.shortFrameB64}` } });
    content.push({ type: 'image_url', image_url: { url: `data:image/jpeg;base64,${p.movieFrameB64}` } });
  }

  const body = {
    model: VLM_MODEL,
    messages: [
      {
        role: 'user',
        content,
      },
    ],
    max_tokens: 100,
    temperature: 0,
  };

  const release = await verifyGate.acquire();
  try {
    const { response: res, attempts } = await fetchWithRetry(
      VLM_ENDPOINT_URL,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      },
      20_000,
      'verifySameScene',
    );

    if (!res.ok) {
      if (attempts > 1) vlmNetworkStats.verifyInconclusive++;
      console.warn(
        `[VLM] Endpoint returned ${res.status} ${res.statusText}` +
        (attempts > 1
          ? ` after ${attempts} attempts — no verdict obtained, treating as inconclusive (not a rejection)`
          : '')
      );
      return null;
    }

    const json: any = await res.json();
    let raw: string = json?.choices?.[0]?.message?.content ?? '';
    if (!raw) {
      console.warn('[VLM] Empty response content');
      return null;
    }

    // Strip markdown code fences if the model wrapped its JSON in them.
    raw = raw.trim().replace(/^```(?:json)?/i, '').replace(/```$/, '').trim();

    const parsed = JSON.parse(raw);
    if (typeof parsed.same !== 'boolean' || typeof parsed.confidence !== 'number') {
      console.warn(`[VLM] Unexpected response shape: ${raw.slice(0, 200)}`);
      return null;
    }

    return { same: parsed.same, confidencePct: parsed.confidence };
  } catch (err: any) {
    const attempts = err?.vlmAttempts || 1;
    if (attempts > 1) vlmNetworkStats.verifyInconclusive++;
    console.warn(
      `[VLM] verifySameScene failed calling ${VLM_ENDPOINT_URL}` +
      (attempts > 1
        ? ` after ${attempts} attempts — no verdict obtained, treating as inconclusive (not a rejection)`
        : '') +
      `: ${describeFetchError(err)}`
    );
    return null;
  } finally {
    release();
  }
}

// ---------------------------------------------------------------------------
// Cache reset — llama.cpp's KV-cache/prompt-cache per slot is not released
// between individual chat/completions calls, only on slot-erase or a full
// server restart. This is best-effort cache hygiene: a failed reset must
// never abort or fail the match job, only log a warning.
// ---------------------------------------------------------------------------
const VLM_NUM_SLOTS = 4;
const VLM_RESET_TIMEOUT_MS = 5_000;

function getVlmBaseUrl(): string {
  return VLM_ENDPOINT_URL.replace(/\/v1\/chat\/completions\/?$/, '');
}

async function eraseSlot(baseUrl: string, slotId: number): Promise<void> {
  const release = await resetGate.acquire();
  try {
    const { response: res, attempts } = await fetchWithRetry(
      `${baseUrl}/slots/${slotId}?action=erase`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' },
      VLM_RESET_TIMEOUT_MS,
      `Slot ${slotId} erase`,
    );
    if (!res.ok) {
      if (attempts > 1) vlmNetworkStats.resetInconclusive++;
      console.warn(
        `[VLM] Slot ${slotId} erase returned ${res.status} ${res.statusText}` +
        (attempts > 1 ? ` after ${attempts} attempts (best-effort, does not affect match result)` : '')
      );
    }
  } catch (err: any) {
    const attempts = err?.vlmAttempts || 1;
    if (attempts > 1) vlmNetworkStats.resetInconclusive++;
    console.warn(
      `[VLM] Slot ${slotId} erase failed calling ${baseUrl}` +
      (attempts > 1 ? ` after ${attempts} attempts` : '') +
      ` (best-effort, does not affect match result): ${describeFetchError(err)}`
    );
  } finally {
    release();
  }
}

/**
 * Erase all VLM server slots' KV-cache. Best-effort: failures are logged as
 * warnings and never thrown — a failed reset must not fail the match job.
 * `label` is only used for the log line (e.g. "after batch (segments 1-48)").
 */
export async function resetVlmCache(label: string): Promise<void> {
  const baseUrl = getVlmBaseUrl();
  await Promise.all(
    Array.from({ length: VLM_NUM_SLOTS }, (_, slotId) => eraseSlot(baseUrl, slotId)),
  );
  console.log(`[VLM] Cache reset ${label}`);
}

// ---------------------------------------------------------------------------
// Availability check — avoids paying a 20s timeout per segment/attempt when
// the AWS GPU server is simply off. Cached briefly so a single /api/match
// request only checks once, and repeated requests don't hammer a down host.
// ---------------------------------------------------------------------------
let cachedAvailable: boolean | null = null;
let cachedAt = 0;
let warnedUnavailable = false;
const AVAILABILITY_TTL_MS = 30_000;

export async function isVlmAvailable(): Promise<boolean> {
  if (!process.env.VLM_ENDPOINT_URL) {
    if (!warnedUnavailable) {
      console.warn('[VLM] VLM_ENDPOINT_URL is not set — skipping VLM verification pass.');
      warnedUnavailable = true;
    }
    return false;
  }

  const now = Date.now();
  if (cachedAvailable !== null && now - cachedAt < AVAILABILITY_TTL_MS) {
    return cachedAvailable;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3_000);
  let lastError: any = null;
  try {
    const healthUrl = VLM_ENDPOINT_URL.replace(/\/chat\/completions\/?$/, '/models');
    const res = await fetch(healthUrl, { signal: controller.signal });
    cachedAvailable = res.ok;
    if (!res.ok) lastError = new Error(`HTTP ${res.status} ${res.statusText}`);
  } catch (err) {
    cachedAvailable = false;
    lastError = err;
  } finally {
    clearTimeout(timeout);
  }

  cachedAt = now;
  if (!cachedAvailable && !warnedUnavailable) {
    const reason = lastError ? describeFetchError(lastError) : 'unknown error';
    console.warn(`[VLM] Endpoint ${VLM_ENDPOINT_URL} is unreachable (${reason}) — skipping VLM verification pass until it comes back.`);
    warnedUnavailable = true;
  }
  if (cachedAvailable) warnedUnavailable = false;

  return cachedAvailable;
}
