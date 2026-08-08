/**
 * Gemini Flash provider for last-pass VLM verification — DUAL-MODEL
 * QUOTA MANAGER edition.
 *
 * Two free-tier models, each with its OWN independent quota pool:
 *   PRIMARY : gemini-flash-lite-latest (3.5 flash-lite) — 500 RPD, 15 RPM
 *   FALLBACK: gemini-3.1-flash-lite                     — 500 RPD, 15 RPM
 *
 * Rotation rules (zero-wait design):
 *   1. Always prefer PRIMARY when its per-minute window has room.
 *   2. PRIMARY RPM full  -> INSTANTLY switch to FALLBACK (no waiting).
 *   3. FALLBACK RPM full -> switch back to PRIMARY the moment its window
 *      frees up. Only when BOTH windows are full do we wait — and only for
 *      the shortest time until either window opens (~30 req/min effective).
 *   4. RPD hit (daily 429) on a model -> that model is parked until
 *      midnight PACIFIC TIME; the other model keeps working alone.
 *   5. BOTH models daily-exhausted -> dailyLimitReached flag for the UI
 *      ("Gemini key limit over — new API key needed") + periodic re-probe
 *      to detect the midnight-PT reset automatically.
 *
 * Contracts (unchanged from the single-model version):
 *  - Every failure path returns null — NEVER throws. Caller treats null
 *    as unverifiable.
 *  - Module is completely inert without GEMINI_API_KEY.
 *
 * Env vars:
 *  - GEMINI_API_KEY          (required to activate)
 *  - GEMINI_MODEL_PRIMARY    (default "gemini-flash-lite-latest" = 3.5 lite)
 *  - GEMINI_MODEL_FALLBACK   (default "gemini-3.1-flash-lite")
 *  - GEMINI_RPM              (default 15  — per model)
 *  - GEMINI_RPD              (default 500 — per model)
 *  - GEMINI_TIMEOUT_MS       (default 60000)
 *  - GEMINI_DAILY_PROBE_INTERVAL_MS (default 10 min)
 */

import * as fs from 'fs';
import { Readable } from 'stream';

export function geminiConfigured(): boolean {
  return !!process.env.GEMINI_API_KEY;
}

const GEMINI_TIMEOUT_MS = Number(process.env.GEMINI_TIMEOUT_MS) || 60_000;
const GEMINI_429_RETRY_DELAY_MS = 4_000;
const GEMINI_429_RETRY_DELAY_MAX_MS = 65_000;
// Per model. Free tier lite models: 15 RPM / 500 RPD each.
const GEMINI_RPM = Number(process.env.GEMINI_RPM) || 15;
const GEMINI_RPD = Number(process.env.GEMINI_RPD) || 500;
// How often to re-probe after BOTH daily quotas looked exhausted.
const GEMINI_DAILY_PROBE_INTERVAL_MS =
  Number(process.env.GEMINI_DAILY_PROBE_INTERVAL_MS) || 10 * 60_000;
// Retry cap guards against a truly stuck endpoint, not normal pacing.
const GEMINI_429_MAX_RETRIES = Number(process.env.GEMINI_429_MAX_RETRIES) || 30;

function primaryModelName(): string {
  return process.env.GEMINI_MODEL_PRIMARY || process.env.GEMINI_MODEL || 'gemini-flash-lite-latest';
}
function fallbackModelName(): string {
  return process.env.GEMINI_MODEL_FALLBACK || 'gemini-3.1-flash-lite';
}

// ---------------------------------------------------------------------------
// Pacific-time day key — free-tier daily quotas reset at midnight PT.
// ---------------------------------------------------------------------------
function pacificDay(): string {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/Los_Angeles',
    year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(new Date());
}

// ---------------------------------------------------------------------------
// Per-model quota state
// ---------------------------------------------------------------------------
interface ModelState {
  name: () => string;
  /** Sliding-window timestamps of requests sent in the last 60s. */
  window: number[];
  /** Requests sent today (Pacific day). */
  usedToday: number;
  /** Pacific day the counters belong to. */
  day: string;
  /** Daily quota confirmed exhausted (429 PerDay) — parked till midnight PT. */
  dailyExhausted: boolean;
  dailyExhaustedSince: number;
  /** Per-minute 429 server-imposed cooldown (honors retryDelay). */
  cooldownUntil: number;
}

function freshState(name: () => string): ModelState {
  return {
    name,
    window: [],
    usedToday: 0,
    day: pacificDay(),
    dailyExhausted: false,
    dailyExhaustedSince: 0,
    cooldownUntil: 0,
  };
}

const primary = freshState(primaryModelName);
const fallback = freshState(fallbackModelName);
const MODELS: ModelState[] = [primary, fallback];

function rollDayIfNeeded(m: ModelState): void {
  const today = pacificDay();
  if (today !== m.day) {
    m.day = today;
    m.usedToday = 0;
    m.dailyExhausted = false;
    m.dailyExhaustedSince = 0;
    m.cooldownUntil = 0;
    m.window.length = 0;
  }
}

function pruneWindow(m: ModelState, now: number): void {
  while (m.window.length && now - m.window[0] >= 60_000) m.window.shift();
}

/** Model can take a request RIGHT NOW (no waiting)? */
function slotFree(m: ModelState, now: number): boolean {
  rollDayIfNeeded(m);
  if (m.dailyExhausted) return false;
  if (m.usedToday >= GEMINI_RPD) {
    // Local counter says the daily pool is gone — park it proactively so we
    // don't burn a request just to receive the 429.
    m.dailyExhausted = true;
    m.dailyExhaustedSince = m.dailyExhaustedSince || now;
    return false;
  }
  if (now < m.cooldownUntil) return false;
  pruneWindow(m, now);
  return m.window.length < GEMINI_RPM;
}

/** ms until this model could take a request (Infinity if daily-parked). */
function msUntilFree(m: ModelState, now: number): number {
  rollDayIfNeeded(m);
  if (m.dailyExhausted || m.usedToday >= GEMINI_RPD) return Infinity;
  pruneWindow(m, now);
  const cooldownWait = Math.max(0, m.cooldownUntil - now);
  const rpmWait = m.window.length < GEMINI_RPM
    ? 0
    : 60_000 - (now - m.window[0]) + 250;
  return Math.max(cooldownWait, rpmWait);
}

function bothDailyExhausted(): boolean {
  MODELS.forEach(rollDayIfNeeded);
  return MODELS.every(m => m.dailyExhausted || m.usedToday >= GEMINI_RPD);
}

// ---------------------------------------------------------------------------
// FIFO slot acquisition — serialized so queued callers proceed in order.
// Returns the ModelState whose slot was taken, or null when BOTH models are
// daily-exhausted (probe handling happens in the caller).
// ---------------------------------------------------------------------------
let pacerChain: Promise<unknown> = Promise.resolve();
let rateLimitWaiting = false;
let lastUsedModel = '';

function acquireSlot(): Promise<ModelState | null> {
  const turn = pacerChain.then(async (): Promise<ModelState | null> => {
    for (;;) {
      const now = Date.now();
      if (bothDailyExhausted()) return null;

      // Rule 1-2: primary first, instant switch to fallback on RPM hit.
      if (slotFree(primary, now)) {
        primary.window.push(now);
        return primary;
      }
      if (slotFree(fallback, now)) {
        if (lastUsedModel !== fallback.name()) {
          console.log(
            `[Gemini] ${primary.name()} RPM window full — instantly rotating to ` +
            `${fallback.name()} (zero wait)`
          );
        }
        fallback.window.push(now);
        return fallback;
      }

      // Rule 3: both windows full — wait only the SHORTEST time.
      const wait = Math.min(msUntilFree(primary, now), msUntilFree(fallback, now));
      if (!isFinite(wait)) return null; // both got daily-parked meanwhile
      rateLimitWaiting = true;
      await sleep(Math.max(250, Math.min(wait, 61_000)));
      rateLimitWaiting = false;
    }
  });
  pacerChain = turn.catch(() => {});
  return turn;
}

// ---------------------------------------------------------------------------
// Both-exhausted probing — detect the midnight-PT reset (or a fresh key).
// ---------------------------------------------------------------------------
let lastDailyProbeAt = 0;

/** When both models are parked, occasionally un-park one to probe. */
function takeProbeModel(): ModelState | null {
  const now = Date.now();
  if (now - lastDailyProbeAt < GEMINI_DAILY_PROBE_INTERVAL_MS) return null;
  lastDailyProbeAt = now;
  console.warn('[Gemini] Both models daily-parked — sending a probe request to check if quota is back');
  return primary;
}

// ---------------------------------------------------------------------------
// Status for the UI
// ---------------------------------------------------------------------------
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
  /** Model currently being used (last request sent). */
  model: string;
  rpmLimit: number;
  /** Combined requests today across both models. */
  usedToday: number;
  /** True only when BOTH models' daily pools are gone. */
  dailyLimitReached: boolean;
  dailyLimitSince: number | null;
  /** True while a request is parked waiting (both RPM windows full). */
  rateLimitWaiting: boolean;
  /** Per-model breakdown for the quota display. */
  models: GeminiModelStatus[];
}

export function getGeminiStatus(): GeminiStatus {
  MODELS.forEach(rollDayIfNeeded);
  const allGone = bothDailyExhausted();
  const since = allGone
    ? Math.max(primary.dailyExhaustedSince, fallback.dailyExhaustedSince) || Date.now()
    : null;
  return {
    configured: geminiConfigured(),
    model: lastUsedModel || primary.name(),
    rpmLimit: GEMINI_RPM,
    usedToday: primary.usedToday + fallback.usedToday,
    dailyLimitReached: allGone,
    dailyLimitSince: since,
    rateLimitWaiting,
    models: MODELS.map(m => ({
      model: m.name(),
      usedToday: m.usedToday,
      rpdLimit: GEMINI_RPD,
      remaining: Math.max(0, GEMINI_RPD - m.usedToday),
      rpmLimit: GEMINI_RPM,
      dailyLimitReached: m.dailyExhausted || m.usedToday >= GEMINI_RPD,
    })),
  };
}

/**
 * Classify a 429 body: daily quota exhausted vs per-minute throttle.
 * Google returns QuotaFailure details with quotaId strings like
 * "GenerateRequestsPerDayPerProjectPerModel-FreeTier" (daily) or
 * "...PerMinute..." (minute window).
 */
function is429Daily(bodyText: string): boolean {
  return /PerDay|per day|daily/i.test(bodyText);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface GeminiVerdict {
  same: boolean;
  confidence: number;
  /** Concrete shared/contradictory details the model cited, when it supplied them. */
  evidence?: string[];
  /** The aligned time windows the model reported, when it supplied them. */
  matchedTimeranges?: { short?: string; movie?: string } | null;
}

/**
 * Collect every BALANCED top-level `{...}` substring in a blob of text.
 * A brace-counting scan (rather than a regex) is required because the verdict
 * JSON now contains NESTED objects/arrays (`matchedTimeranges`, `evidence`).
 * Strings are tracked so braces inside quoted evidence text can't unbalance
 * the scan.
 */
function balancedJsonObjects(text: string): string[] {
  const found: string[] = [];
  let depth = 0;
  let startIdx = -1;
  let inString = false;
  let escaped = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') { inString = true; continue; }
    if (ch === '{') {
      if (depth === 0) startIdx = i;
      depth++;
    } else if (ch === '}') {
      if (depth > 0) {
        depth--;
        if (depth === 0 && startIdx !== -1) {
          found.push(text.slice(startIdx, i + 1));
          startIdx = -1;
        }
      }
    }
  }
  return found;
}

/**
 * Extract the FINAL JSON object containing a "same" field from model text.
 * The prompt allows reasoning/evidence before the JSON, so we scan for the
 * last balanced object that actually carries a verdict.
 */
export function parseVerdictJson(raw: string): GeminiVerdict | null {
  if (!raw) return null;
  const cleaned = raw
    .trim()
    .replace(/```(?:json)?/gi, '')
    .trim();

  const matches = balancedJsonObjects(cleaned);
  if (matches.length === 0) return null;

  for (let i = matches.length - 1; i >= 0; i--) {
    try {
      const parsed = JSON.parse(matches[i]);
      if (typeof parsed?.same === 'boolean' && typeof parsed?.confidence === 'number') {
        const evidence = Array.isArray(parsed.evidence)
          ? parsed.evidence.filter((e: unknown) => typeof e === 'string' && e.trim()).slice(0, 8)
          : undefined;
        const ranges =
          parsed.matchedTimeranges && typeof parsed.matchedTimeranges === 'object'
            ? {
                short: typeof parsed.matchedTimeranges.short === 'string' ? parsed.matchedTimeranges.short : undefined,
                movie: typeof parsed.matchedTimeranges.movie === 'string' ? parsed.matchedTimeranges.movie : undefined,
              }
            : null;
        return {
          same: parsed.same,
          confidence: Math.max(0, Math.min(100, parsed.confidence)),
          evidence: evidence && evidence.length ? evidence : undefined,
          matchedTimeranges: ranges,
        };
      }
    } catch {
      // Try the previous candidate object.
    }
  }
  return null;
}

function retryAfterMs(res: Response): number {
  const header = res.headers.get('retry-after');
  const secs = header ? Number(header) : NaN;
  if (isFinite(secs) && secs > 0) {
    return Math.min(secs * 1000, GEMINI_429_RETRY_DELAY_MAX_MS);
  }
  return GEMINI_429_RETRY_DELAY_MS;
}

/**
 * Send ONE generateContent request through the dual-model quota rotation and
 * return the concatenated response text (or null when no answer could be
 * obtained). This is the single shared transport for every Gemini call —
 * video verification, plain-text probes, everything — so all of them get the
 * same RPM pacing, per-minute 429 rotation, daily-quota parking and probing.
 */
async function generateContentWithRotation(
  body: unknown,
  timeoutMs: number,
): Promise<string | null> {
  if (!geminiConfigured()) return null;

  for (let attempt = 1; attempt <= GEMINI_429_MAX_RETRIES + 1; attempt++) {
    // -- Pick a model: normal rotation, or a probe when both daily-parked --
    let m: ModelState | null = null;
    let isProbe = false;
    if (bothDailyExhausted()) {
      m = takeProbeModel();
      isProbe = true;
      if (!m) return null; // probe not due yet — no verdict possible right now
    } else {
      rateLimitWaiting = true;
      m = await acquireSlot();
      rateLimitWaiting = false;
      if (!m) {
        // Both pools drained while we were queued — try a probe or bail.
        m = takeProbeModel();
        isProbe = true;
        if (!m) return null;
      }
    }

    const modelName = m.name();
    lastUsedModel = modelName;
    const url =
      `https://generativelanguage.googleapis.com/v1beta/models/` +
      `${encodeURIComponent(modelName)}:generateContent`;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      rollDayIfNeeded(m);
      m.usedToday++;
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': process.env.GEMINI_API_KEY as string,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (res.status === 429) {
        clearTimeout(timer);
        const bodyText = await res.text().catch(() => '');
        if (is429Daily(bodyText)) {
          // RPD hit — park THIS model till midnight PT; other model carries on.
          m.dailyExhausted = true;
          m.dailyExhaustedSince = m.dailyExhaustedSince || Date.now();
          if (bothDailyExhausted()) {
            lastDailyProbeAt = Date.now();
            console.warn(
              '[Gemini] BOTH models\' daily quotas exhausted — flagging the UI ' +
              '("Gemini key limit over — new API key needed"), treating as unverifiable. ' +
              `Re-probe every ${Math.round(GEMINI_DAILY_PROBE_INTERVAL_MS / 60000)} min.`
            );
            return null;
          }
          console.warn(
            `[Gemini] ${modelName} DAILY quota exhausted — parked till midnight PT, ` +
            'rotating to the other model with its own fresh pool'
          );
          continue; // instantly retry on the other model
        }
        // Per-minute throttle: cooldown THIS model, instantly rotate to the
        // other one — zero waiting unless both are throttled.
        const delay = retryAfterMs(res);
        m.cooldownUntil = Date.now() + delay;
        console.warn(
          `[Gemini] ${modelName} 429 per-minute — cooling it ${Math.round(delay / 1000)}s ` +
          `and rotating models instantly (attempt ${attempt}/${GEMINI_429_MAX_RETRIES + 1})`
        );
        continue;
      }

      // A successful probe proves the daily quota reset — un-park everything.
      if (isProbe && res.ok) {
        MODELS.forEach(s => {
          s.dailyExhausted = false;
          s.dailyExhaustedSince = 0;
          s.usedToday = 0;
          s.day = pacificDay();
        });
        console.warn('[Gemini] Probe succeeded — daily quotas are back, resuming Gemini verification');
      }

      if (!res.ok) {
        console.warn(`[Gemini] ${modelName} HTTP ${res.status} ${res.statusText} — treating as unverifiable`);
        return null;
      }

      const json: any = await res.json();
      const raw: string = (json?.candidates?.[0]?.content?.parts || [])
        .map((p: any) => p?.text || '')
        .join('');
      return raw;
    } catch (err: any) {
      const reason = err?.name === 'AbortError' ? 'timed out' : (err?.message || String(err));
      console.warn(`[Gemini] ${modelName} request failed (${reason}) — treating as unverifiable`);
      return null;
    } finally {
      clearTimeout(timer);
      rateLimitWaiting = false;
    }
  }
  console.warn('[Gemini] Gave up after prolonged rate-limit waiting — treating as unverifiable');
  return null;
}

// ===========================================================================
// FILES API — required for VIDEO input.
//
// Video segments are far too large for inline base64 (the inline request cap
// is 20 MB total and we deliberately upload at the ORIGINAL resolution), so
// each segment goes through the resumable Files API and is referenced by URI.
// Uploads do NOT consume generateContent RPM/RPD quota, so they happen
// outside the model-rotation loop; only the verification call itself is paced.
// ===========================================================================

const GEMINI_UPLOAD_TIMEOUT_MS =
  Number(process.env.GEMINI_UPLOAD_TIMEOUT_MS) || 300_000;
const GEMINI_VIDEO_TIMEOUT_MS =
  Number(process.env.GEMINI_VIDEO_TIMEOUT_MS) || 300_000;
/** How long to wait for an uploaded file to finish PROCESSING. */
const GEMINI_FILE_ACTIVE_TIMEOUT_MS =
  Number(process.env.GEMINI_FILE_ACTIVE_TIMEOUT_MS) || 240_000;
const FILE_POLL_INTERVAL_MS = 2_000;

interface GeminiFileRef {
  /** Resource name, e.g. "files/abc123". */
  name: string;
  uri: string;
  mimeType: string;
}

function apiKeyHeader(): Record<string, string> {
  return { 'x-goog-api-key': process.env.GEMINI_API_KEY as string };
}

/** Poll a freshly-uploaded file until it reports ACTIVE (or fails/times out). */
async function waitForFileActive(name: string): Promise<boolean> {
  const url = `https://generativelanguage.googleapis.com/v1beta/${name}`;
  const deadline = Date.now() + GEMINI_FILE_ACTIVE_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url, { headers: apiKeyHeader() });
      if (!res.ok) return false;
      const json: any = await res.json();
      const state = json?.state || json?.file?.state;
      if (state === 'ACTIVE') return true;
      if (state === 'FAILED') {
        console.warn(`[Gemini] Uploaded file ${name} failed processing`);
        return false;
      }
    } catch {
      // Transient — keep polling until the deadline.
    }
    await sleep(FILE_POLL_INTERVAL_MS);
  }
  console.warn(`[Gemini] Uploaded file ${name} never became ACTIVE within the timeout`);
  return false;
}

/**
 * Upload one local file via the resumable Files API. The body is STREAMED
 * from disk so a large original-resolution segment never has to be held in
 * RAM as a single buffer. Returns null on any failure.
 */
async function uploadFileToGemini(
  filePath: string,
  mimeType: string,
  displayName: string,
): Promise<GeminiFileRef | null> {
  let size = 0;
  try {
    size = fs.statSync(filePath).size;
  } catch {
    return null;
  }
  if (size === 0) return null;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GEMINI_UPLOAD_TIMEOUT_MS);
  try {
    const startRes = await fetch(
      'https://generativelanguage.googleapis.com/upload/v1beta/files',
      {
        method: 'POST',
        headers: {
          ...apiKeyHeader(),
          'Content-Type': 'application/json',
          'X-Goog-Upload-Protocol': 'resumable',
          'X-Goog-Upload-Command': 'start',
          'X-Goog-Upload-Header-Content-Length': String(size),
          'X-Goog-Upload-Header-Content-Type': mimeType,
        },
        body: JSON.stringify({ file: { display_name: displayName } }),
        signal: controller.signal,
      },
    );
    if (!startRes.ok) {
      console.warn(`[Gemini] Upload start failed for ${displayName}: HTTP ${startRes.status}`);
      return null;
    }
    const uploadUrl = startRes.headers.get('x-goog-upload-url');
    if (!uploadUrl) {
      console.warn(`[Gemini] Upload start for ${displayName} returned no upload URL`);
      return null;
    }

    const stream = Readable.toWeb(fs.createReadStream(filePath)) as unknown as ReadableStream;
    const upRes = await fetch(uploadUrl, {
      method: 'POST',
      headers: {
        'Content-Length': String(size),
        'X-Goog-Upload-Offset': '0',
        'X-Goog-Upload-Command': 'upload, finalize',
      },
      body: stream,
      // Required by undici when the request body is a stream.
      duplex: 'half',
      signal: controller.signal,
    } as RequestInit & { duplex: 'half' });

    if (!upRes.ok) {
      console.warn(`[Gemini] Upload of ${displayName} failed: HTTP ${upRes.status}`);
      return null;
    }
    const json: any = await upRes.json();
    const file = json?.file;
    if (!file?.uri || !file?.name) {
      console.warn(`[Gemini] Upload of ${displayName} returned no file URI`);
      return null;
    }
    if (file.state !== 'ACTIVE' && !(await waitForFileActive(file.name))) {
      await deleteGeminiFile(file.name);
      return null;
    }
    console.log(
      `[Gemini] Uploaded ${displayName} (${(size / 1_048_576).toFixed(1)} MB) -> ${file.name}`
    );
    return { name: file.name, uri: file.uri, mimeType: file.mimeType || mimeType };
  } catch (err: any) {
    const reason = err?.name === 'AbortError' ? 'timed out' : (err?.message || String(err));
    console.warn(`[Gemini] Upload of ${displayName} errored (${reason})`);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** Best-effort remote cleanup so uploads don't pile up against the storage cap. */
async function deleteGeminiFile(name: string): Promise<void> {
  try {
    await fetch(`https://generativelanguage.googleapis.com/v1beta/${name}`, {
      method: 'DELETE',
      headers: apiKeyHeader(),
    });
  } catch {
    // Files expire on their own after 48h — a failed delete is harmless.
  }
}

/**
 * THE verification call: upload BOTH cut segments and ask for a single verdict
 * in ONE request — reference movie segment first (labeled VIDEO 1), target
 * clip segment second (labeled VIDEO 2), then the task prompt.
 *
 * Returns null (never throws) when no verdict could be obtained, so the
 * caller's "unverifiable" policy applies.
 */
export async function geminiVerifyVideoPair(
  referenceSegmentPath: string,
  targetSegmentPath: string,
  video1Label: string,
  video2Label: string,
  prompt: string,
): Promise<GeminiVerdict | null> {
  if (!geminiConfigured()) return null;
  // Don't burn an upload when there is no chance of a verdict anyway.
  if (bothDailyExhausted()) {
    console.warn('[Gemini] Both models daily-parked — skipping video upload, treating as unverifiable');
    return null;
  }

  const [refFile, targetFile] = await Promise.all([
    uploadFileToGemini(referenceSegmentPath, 'video/mp4', 'reference-segment'),
    uploadFileToGemini(targetSegmentPath, 'video/mp4', 'target-segment'),
  ]);

  try {
    if (!refFile || !targetFile) return null;

    const body = {
      contents: [
        {
          role: 'user',
          parts: [
            { text: video1Label },
            { file_data: { mime_type: refFile.mimeType, file_uri: refFile.uri } },
            { text: video2Label },
            { file_data: { mime_type: targetFile.mimeType, file_uri: targetFile.uri } },
            { text: prompt },
          ],
        },
      ],
      generationConfig: {
        temperature: 0,
        // Larger than the old image path: the prompt requires cited evidence
        // before the JSON verdict.
        maxOutputTokens: Number(process.env.GEMINI_VIDEO_MAX_TOKENS) || 2048,
      },
    };

    const raw = await generateContentWithRotation(body, GEMINI_VIDEO_TIMEOUT_MS);
    if (raw === null) return null;

    const verdict = parseVerdictJson(raw);
    if (!verdict) {
      console.warn(
        `[Gemini] Malformed/unparseable video verdict — treating as unverifiable: ${String(raw).slice(0, 300)}`
      );
      return null;
    }
    return verdict;
  } finally {
    await Promise.all([
      refFile ? deleteGeminiFile(refFile.name) : Promise.resolve(),
      targetFile ? deleteGeminiFile(targetFile.name) : Promise.resolve(),
    ]);
  }
}

/**
 * Text-only verdict request. Used by the quota-rotation test script so it can
 * exercise the exact same pacing/rotation transport without needing media.
 */
export async function geminiVerifyText(prompt: string): Promise<GeminiVerdict | null> {
  const raw = await generateContentWithRotation(
    {
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0, maxOutputTokens: 256 },
    },
    GEMINI_TIMEOUT_MS,
  );
  return raw === null ? null : parseVerdictJson(raw);
}
