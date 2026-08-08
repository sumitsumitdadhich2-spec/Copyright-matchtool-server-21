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
import { FFMPEG_BIN } from './ffmpeg-path';
import { sscdGateCheck } from './sscd-verify-gate';
import { geminiConfigured, geminiVerifyComposite, parseVerdictJson } from './gemini-vlm';
import { buildSideBySideComposite } from './frame-composite';

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
// Default 2, matching the server's actual slot count (llama.cpp logs
// "n_slots = 2"). The previous default of 4 sent twice as many concurrent
// requests as the server had slots — the excess queued server-side until the
// client's timeout fired, producing the flood of "cancel task" log lines and
// unverifiable (timed-out) verdicts.
export const VLM_CONCURRENCY =
  Number(process.env.VLM_CONCURRENCY) || 2;

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
  Number(process.env.VLM_MAX_CONCURRENT_REQUESTS) || 2;

// How long to wait for one VLM chat/completions call. The server needs
// 15–25s of PROMPT PROCESSING alone for a 6-image request (~6000 tokens at
// ~300 tok/s), so the old 20s timeout was routinely cancelling requests that
// were still working — visible as "cancel task ... progress = 0.80" in the
// server logs. 90s gives slow-but-successful requests room to finish.
export const VLM_REQUEST_TIMEOUT_MS =
  Number(process.env.VLM_REQUEST_TIMEOUT_MS) || 90_000;
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
      FFMPEG_BIN,
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
 * Decision rule: each pair is judged independently and the segment counts as
 * "same" only when a strict MAJORITY of pairs agree (2 of 3, 2 of 2, 1 of 1).
 * The majority is recomputed client-side from the per-pair verdicts (see the
 * parse step below) rather than trusting the model's own aggregate — the old
 * "any one pair => same" rule let a single hallucinated pair accept a wrong
 * segment, which was a major source of false accepts.
 *
 * LEGACY QWEN PROTOCOL — this is the exact pre-upgrade implementation,
 * unchanged. It remains the default whenever GEMINI_API_KEY is not set, and
 * the last-resort fallback when the new composite protocol cannot run.
 * The exported verifySameSceneMulti below routes into it.
 */
async function verifySameSceneMultiLegacy(
  pairs: VlmFramePair[],
  options?: {
    /**
     * When true, each pair is sent with the MOVIE frame first and the short
     * frame second (and the prompt describes that order). Used by the
     * self-consistency double-check: the same content in a genuinely
     * different presentation, so agreement between the two calls is a real
     * signal rather than the model repeating its own cached answer.
     */
    swapped?: boolean;
  },
): Promise<{ same: boolean; confidencePct: number } | null> {
  const usable = pairs.slice(0, 3);
  if (usable.length === 0) return null;

  const swapped = options?.swapped === true;
  const n = usable.length;
  const pairList = usable
    .map((_, i) => `Pair ${i + 1} = images ${i * 2 + 1} & ${i * 2 + 2}`)
    .join('; ');

  const firstDesc = swapped ? 'movie' : 'short clip';
  const secondDesc = swapped ? 'short clip' : 'movie';
  const prompt =
    `You are verifying whether a short video clip was copied from a movie. ` +
    `You are given ${n} pair(s) of video frames, ${n * 2} images total, in order: ${pairList}. ` +
    `In each pair the FIRST image is a frame from the ${firstDesc} and the SECOND image is a frame ` +
    `from the ${secondDesc}. All pairs come from the SAME candidate segment, sampled at different moments.\n\n` +
    `Task: decide if the pairs show the same underlying scene/footage.\n\n` +
    `CRITICAL — VERTICAL CROP: the short clip is usually a 9:16 vertical crop cut from ` +
    `SOME horizontal position (left, center, OR right — the editor chooses freely) of the ` +
    `widescreen movie frame. This means the ${swapped ? 'second' : 'first'} image may show only a NARROW SLICE ` +
    `of what the ${swapped ? 'first' : 'second'} image shows. Before judging a pair, mentally scan the ENTIRE ` +
    `movie frame from left edge to right edge and check whether the short frame's content appears ` +
    `in ANY region of it — including near the edges. A person/object visible in the short frame ` +
    `may sit at the far left or far right of the movie frame. NEVER reject a pair merely because ` +
    `the movie frame contains extra people, objects, or scenery on either side that the short ` +
    `frame does not show, or because the subject is positioned differently within the frame.\n\n` +
    `Also treat frames as the SAME scene even if they differ in:\n` +
    `- compression artifacts, resolution, blur, or sharpness\n` +
    `- zooming, letterboxing/pillarboxing, or aspect ratio\n` +
    `- color grading, filters, brightness, contrast, or saturation\n` +
    `- watermarks, logos, subtitles, captions, or UI overlays added on top\n` +
    `- horizontal mirroring (flipped image)\n` +
    `- a slightly different instant of the same continuous shot (people/objects moved a little)\n\n` +
    `Judge only the underlying content: same location, same people/characters, same objects, ` +
    `same camera setup or same continuous action.\n\n` +
    `EQUALLY IMPORTANT — do NOT confuse merely similar footage with the same footage. ` +
    `A DIFFERENT moment of the same movie (same characters/costumes/location but a different ` +
    `shot, camera angle, or action) is NOT the same scene. Generic look-alikes ` +
    `(similar rooms, similar landscapes, similar crowds) are NOT the same scene. ` +
    `To count a pair as same, the cropped region of the movie frame must show the SAME shot: ` +
    `matching subject pose/action, matching background details, matching lighting — not just ` +
    `the same characters somewhere similar.\n\n` +
    `Decision rule: judge EACH pair independently and give a per-pair verdict. ` +
    `The overall answer is same=true ONLY if a MAJORITY of the pairs show the same scene ` +
    `(${n === 3 ? '2 of 3' : n === 2 ? '2 of 2' : '1 of 1'} pairs). ` +
    `A pair counts as "same" only when you can point to concrete shared content ` +
    `(same location AND same people/characters/objects or the same continuous action). ` +
    `If a pair is too ambiguous or blurry to tell, count that pair as NOT same — ` +
    `do not guess in favor of a match.\n\n` +
    `Reply with ONLY JSON, no other text: ` +
    `{"pairs": [true|false${n > 1 ? ', ...' : ''}], "same": true|false, "confidence": 0-100} ` +
    `where "pairs" has exactly ${n} boolean(s) (one per pair, in order) and ` +
    `confidence is how certain you are of your overall verdict.`;

  const content: any[] = [{ type: 'text', text: prompt }];
  for (const p of usable) {
    const first = swapped ? p.movieFrameB64 : p.shortFrameB64;
    const second = swapped ? p.shortFrameB64 : p.movieFrameB64;
    content.push({ type: 'image_url', image_url: { url: `data:image/jpeg;base64,${first}` } });
    content.push({ type: 'image_url', image_url: { url: `data:image/jpeg;base64,${second}` } });
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
      VLM_REQUEST_TIMEOUT_MS,
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

    // Enforce the majority rule OURSELVES from the per-pair verdicts instead
    // of trusting the model's own aggregate — 7B models frequently give
    // per-pair answers that contradict their overall "same" field, and the
    // per-pair answers are the more reliable of the two. Strict majority
    // (floor(n/2)+1): 3 pairs need 2, 2 pairs need 2, 1 pair needs 1.
    let same = parsed.same;
    if (
      Array.isArray(parsed.pairs) &&
      parsed.pairs.length === n &&
      parsed.pairs.every((v: any) => typeof v === 'boolean')
    ) {
      const trueCount = parsed.pairs.filter(Boolean).length;
      const needed = Math.floor(n / 2) + 1;
      same = trueCount >= needed;
      if (same !== parsed.same) {
        console.log(
          `[VLM] Overriding model aggregate same=${parsed.same} with per-pair majority ` +
          `(${trueCount}/${n} pairs same, need ${needed}) -> same=${same}`
        );
      }
    }

    return { same, confidencePct: parsed.confidence };
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
// Self-consistency double-check — a 7B model's single "same=true" verdict is
// not trustworthy enough on its own. When the first call ACCEPTS (same=true
// at/above the confidence threshold), a second call is made with the image
// order inside every pair swapped (movie frame first). Only if BOTH calls
// independently say "same" does the accept stand. Rejections and
// inconclusive results are never double-checked (they can only cost recall,
// not precision), so the extra VLM traffic applies only to would-be accepts
// — which the embedding gate has already thinned out.
//
// Disable with VLM_SELF_CONSISTENCY=0 if VLM capacity is too tight.
// ---------------------------------------------------------------------------
export const VLM_SELF_CONSISTENCY = process.env.VLM_SELF_CONSISTENCY !== '0';

/**
 * verifySameSceneMulti + swap self-consistency. Same return contract:
 *  - {same:true, ...}  => both calls agreed the footage is the same
 *                         (confidence = the LOWER of the two calls)
 *  - {same:false, ...} => first call rejected, OR the swapped re-check
 *                         disagreed with an initial accept
 *  - null              => no reliable verdict (first call inconclusive, or
 *                         the confirmation call could not be completed) —
 *                         callers apply their existing unverifiable policy
 */
async function verifySameSceneCheckedLegacy(
  pairs: VlmFramePair[],
): Promise<{ same: boolean; confidencePct: number } | null> {
  const first = await verifySameSceneMultiLegacy(pairs);
  if (first === null) return null;

  // Only a passing accept triggers the confirmation call — a same=true below
  // the confidence threshold is already treated as a rejection by callers.
  const passing = first.same && first.confidencePct >= VLM_CONFIDENCE_THRESHOLD;
  if (!VLM_SELF_CONSISTENCY || !passing) return first;

  const second = await verifySameSceneMultiLegacy(pairs, { swapped: true });
  if (second === null) {
    // Accept claimed but could not be confirmed (timeout/overload on the
    // re-check). Surface as "no reliable verdict" so callers fall back to
    // their unverifiable policy (embedding-similarity backed) instead of
    // accepting on a single unconfirmed opinion.
    console.log('[VLM] Self-consistency: initial accept could not be confirmed (re-check inconclusive) — treating as unverifiable');
    return null;
  }
  if (!second.same) {
    console.log(
      `[VLM] Self-consistency: swapped re-check DISAGREED with initial accept ` +
      `(conf ${first.confidencePct} vs ${second.confidencePct}) — rejecting`
    );
    return { same: false, confidencePct: Math.min(first.confidencePct, second.confidencePct) };
  }
  return { same: true, confidencePct: Math.min(first.confidencePct, second.confidencePct) };
}

// ===========================================================================
// VERIFICATION ROUTER — SSCD gate + Gemini-primary composite protocol
// ===========================================================================
// The exported entry points (verifySameSceneMulti / verifySameSceneChecked)
// keep their EXACT signatures and return contracts, so the callers
// (vlm-segment-resolver.ts, candidate-retry.ts, deferred-recovery.ts) are
// completely untouched. Routing happens transparently inside:
//
//   1. SSCD DECISION GATE (sscd-verify-gate.ts): clear-cut accept/reject
//      decided from copy-detection similarity, VLM skipped entirely.
//      Gate disabled/unhealthy -> transparent pass-through.
//   2. If GEMINI_API_KEY is set: NEW COMPOSITE PROTOCOL — each frame pair is
//      joined into ONE side-by-side labeled image, judged independently
//      (Gemini primary, Qwen automatic fallback per call), final verdict by
//      majority vote across pairs.
//   3. Otherwise, or if the composite protocol yields no verdict at all:
//      the LEGACY Qwen path, byte-for-byte the pre-upgrade behavior.
//
// With no GEMINI_API_KEY and no GPU_EMBED_SERVICE_URL, steps 1–2 are inert
// and every call lands directly in step 3 — zero behavioral change.
// ===========================================================================

/** Strict single-composite prompt (Task 3) — used for BOTH providers. */
const COMPOSITE_PROMPT =
  `The image shows two video frames side by side, separated by a gray divider: ` +
  `frame A on the LEFT (from a short vertical clip) and frame B on the RIGHT (from a movie).\n\n` +
  `Question: Are these two frames from the SAME scene of the SAME video ` +
  `(one may be cropped/zoomed/flipped/color-graded)?\n\n` +
  `Important: frame A is often a narrow 9:16 vertical crop taken from ANY horizontal ` +
  `position of frame B — scan frame B's full width, including the edges, before deciding. ` +
  `Ignore differences in resolution, compression, letterboxing, color grading, watermarks, ` +
  `subtitles, overlays, horizontal mirroring, or a slightly different instant of the same ` +
  `continuous shot. BUT a different shot/angle/moment of the same movie, or merely ` +
  `similar-looking footage, is NOT the same scene.\n\n` +
  `You may reason briefly, but your FINAL line must be ONLY this JSON and nothing else: ` +
  `{"same": true|false, "confidence": 0-100}`;

/** One per-pair provider verdict in the composite protocol. */
interface CompositeVote {
  same: boolean;
  confidence: number;
  provider: 'gemini' | 'qwen';
}

/**
 * Qwen fallback for the composite protocol: ONE composite image per request
 * to the existing VLM_ENDPOINT_URL, reusing the same semaphore/retry/timeout
 * infrastructure as the legacy path. Returns null on any failure.
 */
async function qwenVerifyComposite(
  compositeB64: string,
): Promise<{ same: boolean; confidence: number } | null> {
  const body = {
    model: VLM_MODEL,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: COMPOSITE_PROMPT },
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${compositeB64}` } },
        ],
      },
    ],
    max_tokens: 300,
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
      VLM_REQUEST_TIMEOUT_MS,
      'qwenVerifyComposite',
    );
    if (!res.ok) {
      if (attempts > 1) vlmNetworkStats.verifyInconclusive++;
      console.warn(`[VLM] Composite endpoint returned ${res.status} ${res.statusText}`);
      return null;
    }
    const json: any = await res.json();
    const raw: string = json?.choices?.[0]?.message?.content ?? '';
    const verdict = parseVerdictJson(raw);
    if (!verdict) {
      console.warn(`[VLM] Composite response unparseable: ${String(raw).slice(0, 200)}`);
      return null;
    }
    return verdict;
  } catch (err: any) {
    const attempts = err?.vlmAttempts || 1;
    if (attempts > 1) vlmNetworkStats.verifyInconclusive++;
    console.warn(`[VLM] qwenVerifyComposite failed: ${describeFetchError(err)}`);
    return null;
  } finally {
    release();
  }
}

/** Gemini primary, Qwen automatic fallback — for a single composite image. */
async function providerVerifyComposite(
  compositeB64: string,
): Promise<CompositeVote | null> {
  if (geminiConfigured()) {
    const g = await geminiVerifyComposite(compositeB64, COMPOSITE_PROMPT);
    if (g) return { ...g, provider: 'gemini' };
    console.log('[Verify] gemini unavailable for this pair — falling back to qwen');
  }
  const q = await qwenVerifyComposite(compositeB64);
  if (q) return { ...q, provider: 'qwen' };
  return null;
}

function medianNum(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : Math.round((s[mid - 1] + s[mid]) / 2);
}

interface CompositeMajorityResult {
  same: boolean;
  confidencePct: number;
  provider: string;
  votesFor: number;
  votesTotal: number;
}

/**
 * Task 3 — multi-frame majority vote over up to 3 composite images (same
 * frame pairs the SSCD gate sampled). Each pair gets an independent verdict;
 * a strict majority (2/3, 2/2, 1/1) decides. Returns null only when NO pair
 * could be judged at all (composites failed or both providers failed on
 * every pair) — the caller then falls back to the legacy path.
 */
async function compositeMajorityVerify(
  pairs: VlmFramePair[],
): Promise<CompositeMajorityResult | null> {
  const usable = pairs.slice(0, 3);
  if (usable.length === 0) return null;

  const composites = await Promise.all(
    usable.map((p) => buildSideBySideComposite(p.shortFrameB64, p.movieFrameB64)),
  );
  const validComposites = composites.filter((c): c is string => !!c);
  if (validComposites.length === 0) return null;

  // Sequential per-pair calls: gentle on Gemini free-tier rate limits and on
  // the Qwen server's slot budget.
  const votes: CompositeVote[] = [];
  for (const composite of validComposites) {
    const vote = await providerVerifyComposite(composite);
    if (vote) votes.push(vote);
  }
  if (votes.length === 0) return null;

  const trueCount = votes.filter((v) => v.same).length;
  const needed = Math.floor(votes.length / 2) + 1;
  const same = trueCount >= needed;
  const agreeing = votes.filter((v) => v.same === same);
  const confidencePct = medianNum(agreeing.map((v) => v.confidence));

  const providers = new Set(votes.map((v) => v.provider));
  const provider =
    providers.size === 1 ? votes[0].provider : 'gemini+qwen';

  return {
    same,
    confidencePct,
    provider,
    votesFor: same ? trueCount : votes.length - trueCount,
    votesTotal: votes.length,
  };
}

/**
 * The router itself. `legacyFn` is the exact pre-upgrade behavior for the
 * specific entry point being wrapped, invoked verbatim when the new layers
 * are disabled or cannot produce a verdict.
 */
async function routedVerify(
  pairs: VlmFramePair[],
  legacyFn: () => Promise<{ same: boolean; confidencePct: number } | null>,
): Promise<{ same: boolean; confidencePct: number } | null> {
  // ---- Layer 1: SSCD decision gate (fail-safe: null = pass-through) ----
  let gateLabel = 'skipped';
  let simLabel = '-';
  try {
    const gate = await sscdGateCheck(pairs);
    if (gate) {
      gateLabel = 'sscd';
      simLabel = gate.medianSim.toFixed(2);
      if (gate.verdict === 'accept') {
        console.log(`[Verify] gate=sscd sim=${simLabel} provider=none votes=0/0 verdict=accept conf=100`);
        return { same: true, confidencePct: 100 };
      }
      if (gate.verdict === 'reject') {
        console.log(`[Verify] gate=sscd sim=${simLabel} provider=none votes=0/0 verdict=reject conf=100`);
        return { same: false, confidencePct: 100 };
      }
      // 'ambiguous' -> fall through to the VLM layers below.
    }
  } catch {
    // Gate must never break verification — proceed as if it were disabled.
  }

  // ---- Layer 2: composite protocol (only when Gemini is configured) ----
  // When GEMINI_API_KEY is absent, this whole layer is skipped so the
  // no-new-env-vars behavior is EXACTLY the pre-upgrade Qwen path.
  if (geminiConfigured()) {
    const result = await compositeMajorityVerify(pairs);
    if (result) {
      console.log(
        `[Verify] gate=${gateLabel} sim=${simLabel} provider=${result.provider} ` +
        `votes=${result.votesFor}/${result.votesTotal} ` +
        `verdict=${result.same ? 'accept' : 'reject'} conf=${result.confidencePct}`
      );
      return { same: result.same, confidencePct: result.confidencePct };
    }
    console.log('[Verify] composite protocol yielded no verdict — falling back to legacy Qwen path');
  }

  // ---- Layer 3: legacy path (byte-for-byte pre-upgrade behavior) ----
  const legacy = await legacyFn();
  if (legacy) {
    const verdict = legacy.same && legacy.confidencePct >= VLM_CONFIDENCE_THRESHOLD
      ? 'accept'
      : 'reject';
    console.log(
      `[Verify] gate=${gateLabel} sim=${simLabel} provider=qwen votes=- ` +
      `verdict=${verdict} conf=${legacy.confidencePct}`
    );
  }
  return legacy;
}

/**
 * Exported entry point — same signature/contract as always. Callers
 * (candidate-retry.ts) are untouched; routing happens inside.
 */
export async function verifySameSceneMulti(
  pairs: VlmFramePair[],
  options?: { swapped?: boolean },
): Promise<{ same: boolean; confidencePct: number } | null> {
  // Swapped calls only exist as the legacy self-consistency re-check —
  // they must not re-run the gate or the composite protocol.
  if (options?.swapped) return verifySameSceneMultiLegacy(pairs, options);
  return routedVerify(pairs, () => verifySameSceneMultiLegacy(pairs, options));
}

/**
 * GEMINI-FIRST entry point for the manual Retry button. Unlike routedVerify,
 * this SKIPS the SSCD shortcut so a user-triggered Retry is always judged by
 * Gemini itself (with the exact same rate-limit system: 10/min sliding-window
 * pacer, wait-and-retry on per-minute 429s so the process never dies, and
 * daily-limit detection that flags the UI warning). Only when Gemini can
 * yield no verdict at all (not configured, or daily quota exhausted between
 * probes) does it fall back to the normal routed path so Retry still works.
 */
export async function verifySameSceneGeminiFirst(
  pairs: VlmFramePair[],
): Promise<{ same: boolean; confidencePct: number } | null> {
  if (geminiConfigured()) {
    const result = await compositeMajorityVerify(pairs);
    if (result) {
      console.log(
        `[Verify] (retry) gate=skipped provider=${result.provider} ` +
        `votes=${result.votesFor}/${result.votesTotal} ` +
        `verdict=${result.same ? 'accept' : 'reject'} conf=${result.confidencePct}`
      );
      return { same: result.same, confidencePct: result.confidencePct };
    }
    console.log('[Verify] (retry) Gemini yielded no verdict — falling back to routed path');
  }
  return routedVerify(pairs, () => verifySameSceneMultiLegacy(pairs));
}

/**
 * Exported entry point — same signature/contract as always. Callers
 * (vlm-segment-resolver.ts, deferred-recovery.ts) are untouched.
 * In the composite protocol the per-pair majority vote replaces the legacy
 * swap self-consistency check; the legacy path keeps it as before.
 */
export async function verifySameSceneChecked(
  pairs: VlmFramePair[],
): Promise<{ same: boolean; confidencePct: number } | null> {
  return routedVerify(pairs, () => verifySameSceneCheckedLegacy(pairs));
}

// ---------------------------------------------------------------------------
// Cache reset — llama.cpp's KV-cache/prompt-cache per slot is not released
// between individual chat/completions calls, only on slot-erase or a full
// server restart. This is best-effort cache hygiene: a failed reset must
// never abort or fail the match job, only log a warning.
// ---------------------------------------------------------------------------
// Must match the llama.cpp server's actual slot count ("n_slots = 2" in its
// startup logs). Override with VLM_NUM_SLOTS if the server is reconfigured.
const VLM_NUM_SLOTS = Number(process.env.VLM_NUM_SLOTS) || 2;
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
