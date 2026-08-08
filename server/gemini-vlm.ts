/**
 * Gemini Flash provider for last-pass VLM verification (Task 2).
 *
 * Primary VLM when GEMINI_API_KEY is set; the Qwen endpoint
 * (VLM_ENDPOINT_URL) remains as automatic fallback. This module is
 * COMPLETELY inert when GEMINI_API_KEY is not set — geminiConfigured()
 * returns false and nothing here is ever called, so the app behaves exactly
 * as it did before this file existed.
 *
 * Contracts:
 *  - Every failure path (HTTP error, timeout, 429 after retry, malformed
 *    JSON) returns null — NEVER throws. The caller then falls back to Qwen.
 *  - 429 handling: exactly ONE retry with backoff (honoring Retry-After when
 *    present, capped), then null so the Qwen fallback kicks in immediately.
 *  - Free-tier awareness: a simple in-memory per-day request counter logs a
 *    warning when usage passes GEMINI_DAILY_WARN (default 200) — it never
 *    hard-blocks requests (per spec).
 *
 * Env vars:
 *  - GEMINI_API_KEY   (required for this provider to activate)
 *  - GEMINI_MODEL     (default "gemini-3.6-flash" — newest free-tier Flash
 *                      as of Aug 2026; gemini-2.5-flash retires Oct 2026.
 *                      "gemini-flash-latest" alias also works.)
 *  - GEMINI_TIMEOUT_MS   (default 60000)
 *  - GEMINI_DAILY_WARN   (default 200 — warn-only soft limit)
 */

export function geminiConfigured(): boolean {
  return !!process.env.GEMINI_API_KEY;
}

function geminiModel(): string {
  return process.env.GEMINI_MODEL || 'gemini-3.6-flash';
}

const GEMINI_TIMEOUT_MS = Number(process.env.GEMINI_TIMEOUT_MS) || 60_000;
const GEMINI_DAILY_WARN = Number(process.env.GEMINI_DAILY_WARN) || 200;
const GEMINI_429_RETRY_DELAY_MS = 4_000;
const GEMINI_429_RETRY_DELAY_MAX_MS = 65_000;
// Free tier: ~10 requests/minute. The pacer below keeps us under this so we
// rarely hit a 429 at all; when we do, we WAIT and retry (never abandon the
// segment) until either the per-minute window frees up or the DAILY quota is
// confirmed exhausted.
const GEMINI_RPM = Number(process.env.GEMINI_RPM) || 10;
// How often to re-probe Gemini after the daily quota looked exhausted
// ("keep checking whether the limit came back").
const GEMINI_DAILY_PROBE_INTERVAL_MS =
  Number(process.env.GEMINI_DAILY_PROBE_INTERVAL_MS) || 10 * 60_000;
// Per-minute 429s are retried indefinitely in spirit; this cap only guards
// against a truly stuck endpoint (~30 min of waiting per request).
const GEMINI_429_MAX_RETRIES = Number(process.env.GEMINI_429_MAX_RETRIES) || 30;

// ---------------------------------------------------------------------------
// Sliding-window RPM pacer. Every real HTTP request must first take a slot;
// if GEMINI_RPM requests already went out in the last 60s, the caller waits
// (queued, FIFO) until the oldest one ages out. This keeps a big batch of
// segments flowing at a steady 10/min instead of blasting the API and dying.
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
      if (requestTimestamps.length < GEMINI_RPM) {
        requestTimestamps.push(Date.now());
        return;
      }
      const waitMs = 60_000 - (now - requestTimestamps[0]) + 250;
      await sleep(Math.max(250, waitMs));
    }
  });
  // Serialize slot acquisition so queued callers proceed FIFO.
  pacerChain = waitTurn.catch(() => {});
  return waitTurn;
}

// ---------------------------------------------------------------------------
// Daily-quota state — exposed to the app via getGeminiStatus() so the UI can
// show "daily limit over — bring a new API key". After the daily quota looks
// exhausted we stop hammering the API, but re-probe every
// GEMINI_DAILY_PROBE_INTERVAL_MS to detect when the quota resets.
// ---------------------------------------------------------------------------
let dailyLimitReached = false;
let dailyLimitSince = 0;
let lastDailyProbeAt = 0;
let rateLimitWaiting = false;

export interface GeminiStatus {
  configured: boolean;
  model: string;
  rpmLimit: number;
  usedToday: number;
  dailyLimitReached: boolean;
  dailyLimitSince: number | null;
  /** True while a request is parked waiting for the per-minute window. */
  rateLimitWaiting: boolean;
}

export function getGeminiStatus(): GeminiStatus {
  // Roll the day counter forward if needed so usedToday is fresh.
  const today = new Date().toISOString().slice(0, 10);
  if (today !== usageDay) {
    usageDay = today;
    usageCount = 0;
    warnedThisDay = false;
    dailyLimitReached = false;
    dailyLimitSince = 0;
  }
  return {
    configured: geminiConfigured(),
    model: geminiModel(),
    rpmLimit: GEMINI_RPM,
    usedToday: usageCount,
    dailyLimitReached,
    dailyLimitSince: dailyLimitReached ? dailyLimitSince : null,
    rateLimitWaiting,
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

// ---------------------------------------------------------------------------
// In-memory daily usage counter (warn-only, resets on UTC day change or
// process restart — good enough for free-tier awareness, per spec).
// ---------------------------------------------------------------------------
let usageDay = '';
let usageCount = 0;
let warnedThisDay = false;

function trackUsage(): void {
  const today = new Date().toISOString().slice(0, 10);
  if (today !== usageDay) {
    usageDay = today;
    usageCount = 0;
    warnedThisDay = false;
  }
  usageCount++;
  if (usageCount > GEMINI_DAILY_WARN && !warnedThisDay) {
    warnedThisDay = true;
    console.warn(
      `[Gemini] ${usageCount} requests today — past the soft daily warning ` +
      `threshold (${GEMINI_DAILY_WARN}). Free-tier daily limits may kick in soon; ` +
      `429s will automatically fall back to Qwen.`
    );
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Extract the FINAL JSON object containing a "same" field from model text.
 * The prompt allows reasoning before the JSON, so we scan for the last
 * matching object rather than requiring the whole response to be JSON.
 */
export function parseVerdictJson(
  raw: string,
): { same: boolean; confidence: number } | null {
  if (!raw) return null;
  const cleaned = raw
    .trim()
    .replace(/```(?:json)?/gi, '')
    .trim();

  const matches = cleaned.match(/\{[^{}]*"same"[^{}]*\}/g);
  if (!matches || matches.length === 0) return null;

  for (let i = matches.length - 1; i >= 0; i--) {
    try {
      const parsed = JSON.parse(matches[i]);
      if (typeof parsed.same === 'boolean' && typeof parsed.confidence === 'number') {
        return {
          same: parsed.same,
          confidence: Math.max(0, Math.min(100, parsed.confidence)),
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
 * Ask Gemini whether the composite image shows the same scene.
 * Returns {same, confidence} or null (caller falls back to Qwen).
 */
export async function geminiVerifyComposite(
  compositeB64: string,
  prompt: string,
): Promise<{ same: boolean; confidence: number } | null> {
  if (!geminiConfigured()) return null;

  const model = geminiModel();
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/` +
    `${encodeURIComponent(model)}:generateContent`;

  const body = {
    contents: [
      {
        parts: [
          { inline_data: { mime_type: 'image/jpeg', data: compositeB64 } },
          { text: prompt },
        ],
      },
    ],
    generationConfig: {
      temperature: 0,
      maxOutputTokens: 1024,
    },
  };

  // Daily quota already exhausted? Don't burn time on every segment — only
  // send a probe request every GEMINI_DAILY_PROBE_INTERVAL_MS to detect the
  // quota reset; everything in between falls back to Qwen instantly.
  if (dailyLimitReached) {
    const now = Date.now();
    if (now - lastDailyProbeAt < GEMINI_DAILY_PROBE_INTERVAL_MS) {
      return null;
    }
    lastDailyProbeAt = now;
    console.warn('[Gemini] Daily limit flag set — sending a probe request to check if quota is back');
  }

  // Wait-and-retry loop: per-minute 429s NEVER abandon the segment — we wait
  // for the window to free up and try again, so the pipeline keeps running
  // in 10-request bursts until every segment is checked. Only a confirmed
  // DAILY-quota 429 (or a non-429 failure) gives up on Gemini for this call.
  for (let attempt = 1; attempt <= GEMINI_429_MAX_RETRIES + 1; attempt++) {
    // Pace ourselves under the free-tier RPM before every real request.
    rateLimitWaiting = true;
    await takeRpmSlot();
    rateLimitWaiting = false;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), GEMINI_TIMEOUT_MS);
    try {
      trackUsage();
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
          dailyLimitReached = true;
          dailyLimitSince = dailyLimitSince || Date.now();
          lastDailyProbeAt = Date.now();
          console.warn(
            '[Gemini] DAILY quota exhausted — flagging for the app UI ' +
            '("daily limit over — new API key needed") and falling back to Qwen. ' +
            `Will re-probe every ${Math.round(GEMINI_DAILY_PROBE_INTERVAL_MS / 60000)} min.`
          );
          return null;
        }
        // Per-minute throttle: wait for the window and KEEP GOING.
        const delay = retryAfterMs(res);
        console.warn(
          `[Gemini] 429 per-minute limit — waiting ${Math.round(delay / 1000)}s ` +
          `then continuing (attempt ${attempt}/${GEMINI_429_MAX_RETRIES + 1}, work will not stop)`
        );
        rateLimitWaiting = true;
        await sleep(delay);
        rateLimitWaiting = false;
        continue;
      }

      // Any successful (or at least non-429) response proves the daily quota
      // is back — clear the flag so the UI warning goes away.
      if (dailyLimitReached && res.ok) {
        dailyLimitReached = false;
        dailyLimitSince = 0;
        console.warn('[Gemini] Probe succeeded — daily quota is back, resuming Gemini verification');
      }

      if (!res.ok) {
        console.warn(`[Gemini] HTTP ${res.status} ${res.statusText} — falling back to Qwen`);
        return null;
      }

      const json: any = await res.json();
      const raw: string = (json?.candidates?.[0]?.content?.parts || [])
        .map((p: any) => p?.text || '')
        .join('');

      const verdict = parseVerdictJson(raw);
      if (!verdict) {
        console.warn(`[Gemini] Malformed/unparseable response — falling back to Qwen: ${String(raw).slice(0, 200)}`);
        return null;
      }
      return verdict;
    } catch (err: any) {
      const reason = err?.name === 'AbortError' ? 'timed out' : (err?.message || String(err));
      console.warn(`[Gemini] Request failed (${reason}) — falling back to Qwen`);
      return null;
    } finally {
      clearTimeout(timer);
      rateLimitWaiting = false;
    }
  }
  console.warn('[Gemini] Gave up after prolonged rate-limit waiting — falling back to Qwen');
  return null;
}
