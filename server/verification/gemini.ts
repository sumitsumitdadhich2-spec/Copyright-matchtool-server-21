/**
 * Gemini transport for the verification stage — the ONE verdict-maker.
 * ---------------------------------------------------------------------------
 * This is a fresh, deliberately small client written for the rebuilt
 * verification system. It knows how to do exactly one thing well: upload two
 * short MP4 clips and ask Gemini whether they are the same moment of the same
 * source footage, returning a structured verdict.
 *
 * Design rules (learned from the system this replaces):
 *  - ONE decision maker. There is no embedding gate, no heuristic fallback and
 *    no "best effort" auto-accept path in here. Gemini answers, or the caller
 *    gets `null` and treats the candidate as UNVERIFIABLE.
 *  - Every unavailability reason is DISTINGUISHABLE. `availability()` returns a
 *    specific, loggable reason string, and every failed request increments a
 *    specific counter. The old bug class — verification silently no-op'ing
 *    because an unrelated provider was unconfigured — is impossible here
 *    because the caller must consult `availability()` and log its reason.
 *  - Quota is respected, never guessed around: a sliding-window pacer for
 *    requests/minute, and per-model daily counters that park a model for the
 *    rest of the UTC day once Google reports its daily quota gone.
 *
 * Configuration (all optional except the key):
 *   GEMINI_API_KEY              required — without it verification is skipped
 *   GEMINI_MODELS               comma-separated model ids, tried in order
 *   GEMINI_RPM / GEMINI_RPD     free-tier pacing limits (defaults 15 / 200)
 *   GEMINI_TIMEOUT_MS           per-request timeout (default 120000)
 */

import * as fs from 'fs';
import * as path from 'path';

const API_ROOT = 'https://generativelanguage.googleapis.com/v1beta';

/** Gemini samples video at 1 fps by default, and accepts at most 24. */
export const GEMINI_MAX_FPS = 24;

const RPM = Number(process.env.GEMINI_RPM) || 15;
const RPD = Number(process.env.GEMINI_RPD) || 200;
const TIMEOUT_MS = Number(process.env.GEMINI_TIMEOUT_MS) || 120_000;

function modelNames(): string[] {
  const raw = (process.env.GEMINI_MODELS || '').trim();
  if (raw) {
    const list = raw.split(',').map(s => s.trim()).filter(Boolean);
    if (list.length > 0) return list;
  }
  return ['gemini-flash-lite-latest', 'gemini-2.5-flash'];
}

interface ModelState {
  name: string;
  usedToday: number;
  /** UTC day (YYYY-MM-DD) the counters above belong to. */
  day: string;
  /** Set when Google reports the daily quota gone for this model. */
  dailyExhausted: boolean;
  dailyExhaustedSince: number | null;
}

function utcDay(): string {
  return new Date().toISOString().slice(0, 10);
}

const models: ModelState[] = modelNames().map(name => ({
  name,
  usedToday: 0,
  day: utcDay(),
  dailyExhausted: false,
  dailyExhaustedSince: null,
}));

function rollDay(m: ModelState) {
  const today = utcDay();
  if (m.day !== today) {
    m.day = today;
    m.usedToday = 0;
    m.dailyExhausted = false;
    m.dailyExhaustedSince = null;
  }
}

function isParked(m: ModelState): boolean {
  rollDay(m);
  return m.dailyExhausted || m.usedToday >= RPD;
}

let lastUsedModel = '';
let rateLimitWaiting = false;

export function geminiConfigured(): boolean {
  return !!(process.env.GEMINI_API_KEY || '').trim();
}

/** ── Availability ─────────────────────────────────────────────────────────
 * The single question the orchestrator asks before starting a verification
 * pass, and again before each candidate. Every negative answer carries a
 * distinct, human-readable reason that goes straight into the logs and the
 * job result — so "why didn't verification run?" is always answerable.
 */
export type Availability =
  | { ok: true }
  | { ok: false; reason: string };

export function availability(): Availability {
  if (!geminiConfigured()) {
    return { ok: false, reason: 'GEMINI_API_KEY is not set — no verification provider configured.' };
  }
  models.forEach(rollDay);
  if (models.every(isParked)) {
    return {
      ok: false,
      reason:
        `Gemini daily quota exhausted for every configured model ` +
        `(${models.map(m => `${m.name}:${m.usedToday}/${RPD}`).join(', ')}).`,
    };
  }
  return { ok: true };
}

// ── Status (polled by the UI) ───────────────────────────────────────────────

export interface GeminiModelStatus {
  model: string;
  usedToday: number;
  rpdLimit: number;
  remaining: number;
  rpmLimit: number;
  dailyLimitReached: boolean;
}

export interface GeminiStatus {
  configured: boolean;
  model: string;
  rpmLimit: number;
  usedToday: number;
  dailyLimitReached: boolean;
  dailyLimitSince: number | null;
  rateLimitWaiting: boolean;
  models: GeminiModelStatus[];
}

export function getGeminiStatus(): GeminiStatus {
  models.forEach(rollDay);
  const allParked = models.every(isParked);
  const since = allParked
    ? models.reduce<number>((acc, m) => Math.max(acc, m.dailyExhaustedSince ?? 0), 0) || Date.now()
    : null;
  return {
    configured: geminiConfigured(),
    model: lastUsedModel || models[0].name,
    rpmLimit: RPM,
    usedToday: models.reduce((n, m) => n + m.usedToday, 0),
    dailyLimitReached: allParked,
    dailyLimitSince: since,
    rateLimitWaiting,
    models: models.map(m => ({
      model: m.name,
      usedToday: m.usedToday,
      rpdLimit: RPD,
      remaining: Math.max(0, RPD - m.usedToday),
      rpmLimit: RPM,
      dailyLimitReached: isParked(m),
    })),
  };
}

// ── Request counters (per match job; reset by the orchestrator) ─────────────

export interface GeminiCounters {
  /** Requests that produced a parsed verdict. */
  answered: number;
  /** Requests that failed transport-side (timeout, 5xx, network). */
  transportFailed: number;
  /** Requests skipped because every model was daily-parked. */
  quotaBlocked: number;
  /** Responses that came back but could not be parsed into a verdict. */
  unparseable: number;
  /** Clips that could not be uploaded to the Files API. */
  uploadFailed: number;
}

export const geminiCounters: GeminiCounters = {
  answered: 0,
  transportFailed: 0,
  quotaBlocked: 0,
  unparseable: 0,
  uploadFailed: 0,
};

export function resetGeminiCounters() {
  geminiCounters.answered = 0;
  geminiCounters.transportFailed = 0;
  geminiCounters.quotaBlocked = 0;
  geminiCounters.unparseable = 0;
  geminiCounters.uploadFailed = 0;
}

// ── Sliding-window RPM pacer ────────────────────────────────────────────────

const recentRequests: number[] = [];

function sleep(ms: number) {
  return new Promise<void>(r => setTimeout(r, ms));
}

async function acquireSlot(): Promise<void> {
  for (;;) {
    const now = Date.now();
    while (recentRequests.length > 0 && now - recentRequests[0] >= 60_000) recentRequests.shift();
    if (recentRequests.length < RPM) {
      recentRequests.push(now);
      rateLimitWaiting = false;
      return;
    }
    const waitMs = 60_000 - (now - recentRequests[0]) + 250;
    rateLimitWaiting = true;
    console.log(`[Gemini] ${RPM} req/min window full — waiting ${Math.ceil(waitMs / 1000)}s for a slot.`);
    await sleep(waitMs);
  }
}

// ── Files API ───────────────────────────────────────────────────────────────

interface UploadedFile {
  /** "files/abc123" — used for the delete call. */
  name: string;
  uri: string;
  mimeType: string;
}

async function apiFetch(url: string, init: RequestInit, timeoutMs = TIMEOUT_MS): Promise<Response> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...init,
      signal: ac.signal,
      headers: {
        ...(init.headers as Record<string, string> | undefined),
        'x-goog-api-key': (process.env.GEMINI_API_KEY || '').trim(),
      },
    });
  } finally {
    clearTimeout(timer);
  }
}

/** Upload one clip and wait until the Files API reports it ACTIVE. */
async function uploadClip(filePath: string, label: string): Promise<UploadedFile | null> {
  let size = 0;
  try {
    size = fs.statSync(filePath).size;
  } catch {
    console.warn(`[Gemini] Cannot stat ${label} clip at ${filePath} — skipping upload.`);
    geminiCounters.uploadFailed++;
    return null;
  }
  if (size === 0) {
    console.warn(`[Gemini] ${label} clip is empty (${filePath}) — skipping upload.`);
    geminiCounters.uploadFailed++;
    return null;
  }

  try {
    const start = await apiFetch(`${API_ROOT}/files`, {
      method: 'POST',
      headers: {
        'X-Goog-Upload-Protocol': 'resumable',
        'X-Goog-Upload-Command': 'start',
        'X-Goog-Upload-Header-Content-Length': String(size),
        'X-Goog-Upload-Header-Content-Type': 'video/mp4',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file: { display_name: `${label}-${path.basename(filePath)}` } }),
    });
    const uploadUrl = start.headers.get('x-goog-upload-url');
    if (!start.ok || !uploadUrl) {
      console.warn(`[Gemini] Upload handshake failed for ${label} clip (HTTP ${start.status}).`);
      geminiCounters.uploadFailed++;
      return null;
    }

    const bytes = await fs.promises.readFile(filePath);
    const put = await apiFetch(uploadUrl, {
      method: 'POST',
      headers: {
        'Content-Length': String(size),
        'X-Goog-Upload-Offset': '0',
        'X-Goog-Upload-Command': 'upload, finalize',
      },
      body: new Uint8Array(bytes),
    });
    if (!put.ok) {
      console.warn(`[Gemini] Upload of ${label} clip failed (HTTP ${put.status}).`);
      geminiCounters.uploadFailed++;
      return null;
    }
    const json: any = await put.json().catch(() => null);
    const file = json?.file;
    if (!file?.uri || !file?.name) {
      console.warn(`[Gemini] Upload of ${label} clip returned no file handle.`);
      geminiCounters.uploadFailed++;
      return null;
    }

    // Video files need processing before they can be referenced.
    let state: string = file.state || 'PROCESSING';
    const deadline = Date.now() + 90_000;
    while (state === 'PROCESSING' && Date.now() < deadline) {
      await sleep(1000);
      const poll = await apiFetch(`${API_ROOT}/${file.name}`, { method: 'GET' }, 20_000);
      const pj: any = await poll.json().catch(() => null);
      state = pj?.state || state;
      if (state === 'FAILED') break;
    }
    if (state !== 'ACTIVE') {
      console.warn(`[Gemini] ${label} clip never became ACTIVE (state=${state}) — cannot verify with it.`);
      geminiCounters.uploadFailed++;
      await deleteClipFile(file.name);
      return null;
    }

    return { name: file.name, uri: file.uri, mimeType: file.mimeType || 'video/mp4' };
  } catch (e: any) {
    console.warn(`[Gemini] Upload of ${label} clip threw: ${e?.message || e}`);
    geminiCounters.uploadFailed++;
    return null;
  }
}

async function deleteClipFile(name: string | undefined) {
  if (!name) return;
  try {
    await apiFetch(`${API_ROOT}/${name}`, { method: 'DELETE' }, 20_000);
  } catch {
    /* best effort — Gemini expires files after 48h anyway */
  }
}

// ── generateContent with model rotation ─────────────────────────────────────

type CallOutcome =
  | { kind: 'text'; text: string }
  | { kind: 'quotaBlocked' }
  | { kind: 'failed'; detail: string };

/** True when a 429 body describes a per-DAY quota rather than a per-minute one. */
function isDailyQuota(body: string): boolean {
  return /PerDay|per day|GenerateRequestsPerDayPerProject/i.test(body);
}

async function generateContent(body: unknown): Promise<CallOutcome> {
  let lastDetail = 'no model produced a response';

  for (let pass = 0; pass < 2; pass++) {
    for (const m of models) {
      if (isParked(m)) continue;

      await acquireSlot();
      lastUsedModel = m.name;

      let res: Response;
      try {
        res = await apiFetch(`${API_ROOT}/models/${m.name}:generateContent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
      } catch (e: any) {
        lastDetail = `${m.name}: ${e?.name === 'AbortError' ? `timed out after ${TIMEOUT_MS}ms` : e?.message || e}`;
        console.warn(`[Gemini] ${lastDetail}`);
        continue;
      }

      if (res.ok) {
        m.usedToday++;
        const json: any = await res.json().catch(() => null);
        const text: string = (json?.candidates?.[0]?.content?.parts ?? [])
          .map((p: any) => p?.text ?? '')
          .join('')
          .trim();
        if (!text) {
          lastDetail = `${m.name}: empty response body`;
          console.warn(`[Gemini] ${lastDetail}`);
          continue;
        }
        return { kind: 'text', text };
      }

      const errBody = await res.text().catch(() => '');

      if (res.status === 429) {
        if (isDailyQuota(errBody)) {
          m.dailyExhausted = true;
          m.dailyExhaustedSince = Date.now();
          console.warn(`[Gemini] ${m.name}: daily quota exhausted — parking it for the rest of the UTC day.`);
        } else {
          // Per-minute throttle: the pacer under-estimated. Back off once and
          // let the outer loop try the next model / second pass.
          console.warn(`[Gemini] ${m.name}: per-minute rate limit hit — backing off 20s.`);
          rateLimitWaiting = true;
          await sleep(20_000);
          rateLimitWaiting = false;
        }
        lastDetail = `${m.name}: HTTP 429`;
        continue;
      }

      lastDetail = `${m.name}: HTTP ${res.status} ${errBody.slice(0, 200)}`;
      console.warn(`[Gemini] ${lastDetail}`);
      // 4xx other than 429 is a request-shape problem — retrying other models
      // will fail identically, so give up immediately.
      if (res.status >= 400 && res.status < 500) return { kind: 'failed', detail: lastDetail };
    }

    if (models.every(isParked)) return { kind: 'quotaBlocked' };
  }

  return { kind: 'failed', detail: lastDetail };
}

// ── The verdict ─────────────────────────────────────────────────────────────

export interface GeminiVerdict {
  /** Gemini's decision: is the target clip the same footage as the reference? */
  same: boolean;
  /** 0-100 self-reported certainty. */
  confidence: number;
  /** Short human-readable justification, surfaced in logs and the UI. */
  reason: string;
}

const VERDICT_PROMPT = `You are auditing an automated video-copyright match.

VIDEO 1 is a segment cut from the REFERENCE film.
VIDEO 2 is a segment cut from the TARGET clip that an automated hash matcher
believes was copied from that exact moment of VIDEO 1.

Decide whether VIDEO 2 really is the SAME footage as VIDEO 1 — the same shot of
the same scene, same actors, same action, same camera move — allowing for
re-encoding, cropping, zooming, letterboxing, colour grading, overlaid text or
logos, mirroring, and speed changes.

Answer "same": false when the two clips merely LOOK similar in a generic way:
same show but a different moment, a different shot of the same location, similar
lighting or colour palette, similar framing of different people, or both being
dark/blurry/static frames. A generic visual resemblance is NOT a match.

First state, in at most two sentences, the concrete visual evidence you are
relying on (a specific action, object, face, camera move, or on-screen text that
appears in both — or the specific difference that rules the match out).

Then output ONLY a JSON object on its own final line:
{"same": true|false, "confidence": 0-100, "reason": "<one short sentence>"}`;

function parseVerdict(raw: string): GeminiVerdict | null {
  // Take the LAST JSON object in the response — the prompt asks for evidence
  // first, verdict last.
  const matches = raw.match(/\{[^{}]*\}/g);
  if (!matches) return null;
  for (let i = matches.length - 1; i >= 0; i--) {
    try {
      const obj = JSON.parse(matches[i]);
      if (typeof obj?.same !== 'boolean') continue;
      const conf = Number(obj.confidence);
      return {
        same: obj.same,
        confidence: Number.isFinite(conf) ? Math.max(0, Math.min(100, conf)) : 0,
        reason: typeof obj.reason === 'string' ? obj.reason.slice(0, 300) : '',
      };
    } catch {
      /* try the next candidate object */
    }
  }
  return null;
}

/**
 * THE verification call. Uploads both clips, asks for one verdict, cleans up.
 *
 * Returns `null` for every "no real answer" case (unconfigured, quota gone,
 * upload failure, transport failure, unparseable response) — each of which has
 * already been counted in `geminiCounters` and logged with its specific cause.
 * Never throws.
 */
export async function verifyClipPair(
  referenceClipPath: string,
  targetClipPath: string,
  fps = 2,
): Promise<GeminiVerdict | null> {
  const avail = availability();
  if (!avail.ok) {
    geminiCounters.quotaBlocked++;
    console.warn(`[Gemini] Skipping verification request — ${avail.reason}`);
    return null;
  }

  const [ref, target] = await Promise.all([
    uploadClip(referenceClipPath, 'reference'),
    uploadClip(targetClipPath, 'target'),
  ]);

  try {
    if (!ref || !target) return null;

    let currentFps = Math.max(1, Math.min(Math.round(fps), GEMINI_MAX_FPS));

    for (;;) {
      const videoPart = (f: UploadedFile) => ({
        file_data: { mime_type: f.mimeType, file_uri: f.uri },
        ...(currentFps > 1 ? { video_metadata: { fps: currentFps } } : {}),
      });

      const outcome = await generateContent({
        contents: [
          {
            role: 'user',
            parts: [
              { text: 'VIDEO 1 — reference film segment:' },
              videoPart(ref),
              { text: 'VIDEO 2 — target clip segment:' },
              videoPart(target),
              { text: VERDICT_PROMPT },
            ],
          },
        ],
        generationConfig: { temperature: 0, maxOutputTokens: 1024 },
      });

      if (outcome.kind === 'quotaBlocked') {
        geminiCounters.quotaBlocked++;
        return null;
      }
      if (outcome.kind === 'failed') {
        // Very short clips sample to zero frames at low fps and come back as
        // HTTP 400. Escalating the sampling rate is the documented fix.
        if (/HTTP 400/.test(outcome.detail) && currentFps < GEMINI_MAX_FPS) {
          const next = Math.min(currentFps * 2, GEMINI_MAX_FPS);
          console.warn(`[Gemini] HTTP 400 at fps=${currentFps} — retrying both clips at fps=${next}.`);
          currentFps = next;
          continue;
        }
        geminiCounters.transportFailed++;
        console.warn(`[Gemini] No verdict — ${outcome.detail}`);
        return null;
      }

      const verdict = parseVerdict(outcome.text);
      if (!verdict) {
        geminiCounters.unparseable++;
        console.warn(`[Gemini] Unparseable verdict: ${outcome.text.slice(0, 300)}`);
        return null;
      }
      geminiCounters.answered++;
      return verdict;
    }
  } finally {
    await Promise.all([deleteClipFile(ref?.name), deleteClipFile(target?.name)]);
  }
}
