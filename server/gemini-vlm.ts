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
const GEMINI_429_RETRY_DELAY_MAX_MS = 30_000;

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

  // Exactly one retry, only for 429 (per spec). All other failures -> null.
  for (let attempt = 1; attempt <= 2; attempt++) {
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
        if (attempt === 1) {
          const delay = retryAfterMs(res);
          console.warn(`[Gemini] 429 rate-limited — retrying once in ${delay}ms`);
          clearTimeout(timer);
          await sleep(delay);
          continue;
        }
        console.warn('[Gemini] 429 rate-limited after retry — falling back to Qwen');
        return null;
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
    }
  }
  return null;
}
