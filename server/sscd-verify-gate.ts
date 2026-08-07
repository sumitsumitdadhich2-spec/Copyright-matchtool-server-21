/**
 * SSCD DECISION GATE — runs BEFORE any VLM verification call.
 *
 * Uses the SAME GPU embedding service the candidate ranker already talks to
 * (gpu_embedding_service.py via GPU_EMBED_SERVICE_URL, POST /embed with
 * model:"sscd") to compute copy-detection similarity for the frame pairs a
 * segment is about to be VLM-verified with. Clear-cut cases are decided here
 * and the VLM call is skipped entirely:
 *
 *   median sim >= SSCD_ACCEPT_THRESHOLD (default 0.75)  -> AUTO-ACCEPT
 *   median sim <= SSCD_REJECT_THRESHOLD (default 0.35)  -> AUTO-REJECT
 *   in between                                          -> 'ambiguous'
 *                                                          (VLM decides)
 *
 * Scope / safety guarantees:
 *  - Consumed ONLY by the verification layer (vlm-verify.ts routing hook).
 *    Never imported by matching-engine.ts, server.ts's first pass, or the
 *    candidate ranking logic — the main hash matching system and
 *    candidate-embedding-rank.ts stay 100% untouched.
 *  - Fully fail-safe: GPU_EMBED_SERVICE_URL unset, SSCD_GATE_ENABLED=0,
 *    health check failing, /embed erroring/timing out, or a malformed
 *    response ALL return null — the caller then proceeds to the VLM exactly
 *    as before (transparent pass-through). This module never throws.
 *  - The frame pairs are the ones the callers already sampled for VLM
 *    verification (segment midpoint + highest-similarity moments, up to 3
 *    pairs spread across the segment) — the gate reuses those same frames,
 *    so no extra video decoding happens here.
 */

// Local structural copy of vlm-verify.ts's VlmFramePair — kept here (not
// imported) so this module has zero imports from vlm-verify.ts and the
// dependency graph stays acyclic (vlm-verify.ts imports THIS module).
export interface SscdGatePair {
  shortFrameB64: string;
  movieFrameB64: string;
}

export interface SscdGateOutcome {
  verdict: 'accept' | 'reject' | 'ambiguous';
  /** Median SSCD cosine similarity across the sampled pairs. */
  medianSim: number;
}

// ---------------------------------------------------------------------------
// Config (all env-tunable, all with safe defaults)
// ---------------------------------------------------------------------------

function gpuServiceUrl(): string {
  // Users often paste the full health-check link (".../health") instead of the
  // base URL — strip a trailing "/health" and trailing slashes defensively.
  return (process.env.GPU_EMBED_SERVICE_URL || '')
    .replace(/\/+$/, '')
    .replace(/\/health$/i, '');
}

/**
 * Gate enabled by default whenever GPU_EMBED_SERVICE_URL is set.
 * SSCD_GATE_ENABLED=0 (or "false") force-disables it even with the URL set.
 */
export function sscdGateEnabled(): boolean {
  if (!gpuServiceUrl()) return false;
  const flag = (process.env.SSCD_GATE_ENABLED || '').toLowerCase();
  return flag !== '0' && flag !== 'false';
}

function acceptThreshold(): number {
  const v = Number(process.env.SSCD_ACCEPT_THRESHOLD);
  return isFinite(v) && v > 0 ? v : 0.75;
}

function rejectThreshold(): number {
  const v = Number(process.env.SSCD_REJECT_THRESHOLD);
  return isFinite(v) && v > 0 ? v : 0.35;
}

/** Hard timeout for the /embed call. On expiry -> null (pass-through). */
const SSCD_GATE_EMBED_TIMEOUT_MS =
  Number(process.env.SSCD_GATE_EMBED_TIMEOUT_MS) || 30_000;

/** Health-check timeout (short, so a dead ngrok link stalls nothing). */
const SSCD_GATE_HEALTH_TIMEOUT_MS =
  Number(process.env.SSCD_GATE_HEALTH_TIMEOUT_MS) || 5_000;

/**
 * Health result cached this long — same "once per run, never per segment"
 * pattern candidate-embedding-rank.ts uses for the same service.
 */
const SSCD_GATE_HEALTH_CACHE_MS =
  Number(process.env.SSCD_GATE_HEALTH_CACHE_MS) || 120_000;

// ---------------------------------------------------------------------------
// GPU service client (every failure path returns null — never throws)
// ---------------------------------------------------------------------------

let healthCache: { ok: boolean; at: number } | null = null;

async function fetchJsonWithTimeout(
  url: string,
  init: RequestInit | undefined,
  timeoutMs: number,
): Promise<any | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function gateServiceHealthy(): Promise<boolean> {
  const base = gpuServiceUrl();
  if (!base) return false;
  const now = Date.now();
  if (healthCache && now - healthCache.at < SSCD_GATE_HEALTH_CACHE_MS) {
    return healthCache.ok;
  }
  const health = await fetchJsonWithTimeout(
    `${base}/health`, undefined, SSCD_GATE_HEALTH_TIMEOUT_MS,
  );
  const ok = !!health && health.status === 'ok' && !!health.models_loaded?.sscd;
  healthCache = { ok, at: now };
  if (!ok) {
    console.log('[SSCDGate] GPU embed service unhealthy/unreachable — gate pass-through, VLM decides as before.');
  }
  return ok;
}

/** POST /embed with model:"sscd". Returns L2-normalized embeddings or null. */
async function sscdEmbed(imagesB64: string[]): Promise<number[][] | null> {
  const base = gpuServiceUrl();
  if (!base) return null;
  const json = await fetchJsonWithTimeout(
    `${base}/embed`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images: imagesB64, model: 'sscd' }),
    },
    SSCD_GATE_EMBED_TIMEOUT_MS,
  );
  if (!json || !Array.isArray(json.embeddings) || json.embeddings.length !== imagesB64.length) {
    return null;
  }
  return json.embeddings;
}

function cosine(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run the SSCD decision gate on the frame pairs that are about to be sent to
 * the VLM (up to 3 pairs — the same ones callers sampled from the segment's
 * start/high-similarity/middle moments).
 *
 * Returns:
 *  - { verdict: 'accept' | 'reject', medianSim }  -> decided, skip the VLM
 *  - { verdict: 'ambiguous', medianSim }          -> middle band, VLM decides
 *  - null -> gate disabled/unhealthy/failed       -> transparent pass-through
 *            (caller behaves EXACTLY as it did before this gate existed)
 */
export async function sscdGateCheck(
  pairs: SscdGatePair[],
): Promise<SscdGateOutcome | null> {
  try {
    if (!sscdGateEnabled()) return null;
    if (!pairs || pairs.length === 0) return null;
    if (!(await gateServiceHealthy())) return null;

    const usable = pairs.slice(0, 3);
    // Flatten [short0, movie0, short1, movie1, ...] into one batched call.
    const images: string[] = [];
    for (const p of usable) {
      images.push(p.shortFrameB64, p.movieFrameB64);
    }
    const embeddings = await sscdEmbed(images);
    if (!embeddings) return null;

    const sims: number[] = [];
    for (let i = 0; i < usable.length; i++) {
      const shortEmb = embeddings[i * 2];
      const movieEmb = embeddings[i * 2 + 1];
      if (!Array.isArray(shortEmb) || !Array.isArray(movieEmb)) return null;
      sims.push(cosine(shortEmb, movieEmb));
    }
    if (sims.length === 0) return null;

    const med = median(sims);
    if (med >= acceptThreshold()) {
      console.log(`[SSCDGate] verdict=accept sim=${med.toFixed(2)} vlm=skipped`);
      return { verdict: 'accept', medianSim: med };
    }
    if (med <= rejectThreshold()) {
      console.log(`[SSCDGate] verdict=reject sim=${med.toFixed(2)} vlm=skipped`);
      return { verdict: 'reject', medianSim: med };
    }
    console.log(`[SSCDGate] verdict=ambiguous sim=${med.toFixed(2)} vlm=required`);
    return { verdict: 'ambiguous', medianSim: med };
  } catch {
    // Absolute last line of defense — the gate must never break verification.
    return null;
  }
}
