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
 *  - Every failure path returns null — NEVER throws. Caller falls back to
 *    Qwen.
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
 * Rotates between the two lite models per the quota-manager rules.
 * Returns {same, confidence} or null (caller falls back to Qwen).
 */
export async function geminiVerifyComposite(
  compositeB64: string,
  prompt: string,
): Promise<{ same: boolean; confidence: number } | null> {
  return geminiGenerateVerdict(
    [
      { inline_data: { mime_type: 'image/jpeg', data: compositeB64 } },
      { text: prompt },
    ],
    { maxOutputTokens: 1024, timeoutMs: GEMINI_TIMEOUT_MS },
  );
}

// Video requests carry two full clips — prompt processing takes far longer
// than a single composite image, so they get their own (longer) timeout.
const GEMINI_VIDEO_TIMEOUT_MS = Number(process.env.GEMINI_VIDEO_TIMEOUT_MS) || 240_000;

/**
 * VIDEO-SEGMENT verification: sends TWO full video clips (VIDEO 1 = segment
 * cut from the reference movie, VIDEO 2 = segment cut from the target clip)
 * plus the verification prompt in ONE request. Same dual-model quota
 * manager, same rotation/429 handling, same null-on-failure contract.
 */
export async function geminiVerifySegmentClips(
  movieClipB64: string,
  shortClipB64: string,
  prompt: string,
): Promise<{ same: boolean; confidence: number } | null> {
  return geminiGenerateVerdict(
    [
      { text: prompt },
      { text: 'VIDEO 1 (segment cut from the reference movie):' },
      { inline_data: { mime_type: 'video/mp4', data: movieClipB64 } },
      { text: 'VIDEO 2 (segment cut from the target clip):' },
      { inline_data: { mime_type: 'video/mp4', data: shortClipB64 } },
    ],
    { maxOutputTokens: 2048, timeoutMs: GEMINI_VIDEO_TIMEOUT_MS },
  );
}

/**
 * Generic quota-managed generateContent call. `parts` is the raw parts array
 * (text / inline_data in any order). Returns the parsed verdict or null.
 */
async function geminiGenerateVerdict(
  parts: any[],
  opts: { maxOutputTokens: number; timeoutMs: number },
): Promise<{ same: boolean; confidence: number } | null> {
  if (!geminiConfigured()) return null;

  const body = {
    contents: [
      {
        parts,
      },
    ],
    generationConfig: {
      temperature: 0,
      maxOutputTokens: opts.maxOutputTokens,
    },
  };

  for (let attempt = 1; attempt <= GEMINI_429_MAX_RETRIES + 1; attempt++) {
    // -- Pick a model: normal rotation, or a probe when both daily-parked --
    let m: ModelState | null = null;
    let isProbe = false;
    if (bothDailyExhausted()) {
      m = takeProbeModel();
      isProbe = true;
      if (!m) return null; // probe not due yet — instant Qwen fallback
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
    const timer = setTimeout(() => controller.abort(), opts.timeoutMs);
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
              '("Gemini key limit over — new API key needed"), falling back to Qwen. ' +
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
        console.warn(`[Gemini] ${modelName} HTTP ${res.status} ${res.statusText} — falling back to Qwen`);
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
      console.warn(`[Gemini] ${modelName} request failed (${reason}) — falling back to Qwen`);
      return null;
    } finally {
      clearTimeout(timer);
      rateLimitWaiting = false;
    }
  }
  console.warn('[Gemini] Gave up after prolonged rate-limit waiting — falling back to Qwen');
  return null;
}
