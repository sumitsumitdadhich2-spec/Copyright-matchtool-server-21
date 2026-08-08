/**
 * VLM scene-verification helpers — GEMINI-ONLY edition.
 *
 * The old Qwen2.5-VL server path has been fully removed. Verification is now:
 *   1. SSCD decision gate (sscd-verify-gate.ts) — clear-cut accept/reject
 *      decided from copy-detection similarity, no VLM call at all.
 *   2. Gemini composite protocol (gemini-vlm.ts) — each frame pair is joined
 *      into ONE side-by-side labeled image, judged independently, final
 *      verdict by majority vote across pairs. Gemini is the FINAL check.
 *   3. If Gemini yields no verdict (not configured, daily quota exhausted,
 *      network failure), the result is null — "could not verify". Callers
 *      treat null as unverifiable (their existing embedding-similarity-backed
 *      policy), NEVER as a silent pass or fail.
 *
 * All network/parse failures return null so a segment falls back to being
 * handled by the caller's unverifiable policy rather than crashing the match.
 */
import { spawn } from 'child_process';
import { makeCleanEnv } from './pipeline';
import { FFMPEG_BIN } from './ffmpeg-path';
import { sscdGateCheck } from './sscd-verify-gate';
import { geminiConfigured, geminiVerifyComposite } from './gemini-vlm';
import { buildSideBySideComposite } from './frame-composite';

export const VLM_CONFIDENCE_THRESHOLD =
  Number(process.env.VLM_CONFIDENCE_THRESHOLD) || 80;
export const VLM_MAX_ATTEMPTS =
  Number(process.env.VLM_MAX_ATTEMPTS) || 10;

// How many *segments* a caller (vlm-segment-resolver.ts / deferred-recovery.ts)
// processes at once. Gemini's own per-minute pacing lives inside gemini-vlm.ts,
// so this only bounds concurrent frame extraction + in-flight verifications.
export const VLM_CONCURRENCY =
  Number(process.env.VLM_CONCURRENCY) || 2;

/**
 * Aggregate, process-wide counters for the "inconclusive" outcome (a request
 * that never got a real answer — quota exhausted/network failure, NOT the
 * model saying "no match"). server.ts snapshots/resets these around each
 * match job so it can report how many "unmatched" outcomes were genuine
 * content rejections vs. inconclusive — purely additive observability,
 * never consulted by any accept/reject decision.
 */
export const vlmNetworkStats = {
  verifyInconclusive: 0,
  resetInconclusive: 0,
};

export function resetVlmNetworkStats(): void {
  vlmNetworkStats.verifyInconclusive = 0;
  vlmNetworkStats.resetInconclusive = 0;
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

/** One (short-frame, movie-frame) pair from the same candidate segment. */
export interface VlmFramePair {
  shortFrameB64: string;
  movieFrameB64: string;
}

/**
 * Ask the verifier whether two frames show the same scene/subject.
 * Returns null (not false) when the call could not be completed or parsed —
 * callers must treat null as "unverifiable for this attempt", not a verdict.
 */
export async function verifySameScene(
  shortFrameB64: string,
  movieFrameB64: string,
): Promise<{ same: boolean; confidencePct: number } | null> {
  return verifySameSceneMulti([{ shortFrameB64, movieFrameB64 }]);
}

// ===========================================================================
// VERIFICATION ROUTER — SSCD gate + Gemini composite protocol (final check)
// ===========================================================================

/** Strict single-composite prompt — used for every Gemini verification call. */
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

function medianNum(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : Math.round((s[mid - 1] + s[mid]) / 2);
}

interface CompositeMajorityResult {
  same: boolean;
  confidencePct: number;
  votesFor: number;
  votesTotal: number;
}

/**
 * Multi-frame majority vote over up to 3 composite images. Each pair gets an
 * independent Gemini verdict; a strict majority (2/3, 2/2, 1/1) decides.
 * Returns null only when NO pair could be judged at all (composites failed
 * or Gemini failed on every pair — e.g. daily quota exhausted).
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

  // Sequential per-pair calls: gentle on Gemini free-tier rate limits.
  const votes: Array<{ same: boolean; confidence: number }> = [];
  for (const composite of validComposites) {
    const vote = await geminiVerifyComposite(composite, COMPOSITE_PROMPT);
    if (vote) votes.push(vote);
  }
  if (votes.length === 0) return null;

  const trueCount = votes.filter((v) => v.same).length;
  const needed = Math.floor(votes.length / 2) + 1;
  const same = trueCount >= needed;
  const agreeing = votes.filter((v) => v.same === same);
  const confidencePct = medianNum(agreeing.map((v) => v.confidence));

  return {
    same,
    confidencePct,
    votesFor: same ? trueCount : votes.length - trueCount,
    votesTotal: votes.length,
  };
}

/**
 * The router itself:
 *   Layer 1 — SSCD gate: clear-cut accept/reject without any VLM call.
 *   Layer 2 — Gemini composite majority vote: the FINAL check.
 *   No verdict at all -> null (caller's unverifiable policy applies).
 */
async function routedVerify(
  pairs: VlmFramePair[],
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
      // 'ambiguous' -> fall through to Gemini below.
    }
  } catch {
    // Gate must never break verification — proceed as if it were disabled.
  }

  // ---- Layer 2: Gemini composite protocol — the final check ----
  if (geminiConfigured()) {
    const result = await compositeMajorityVerify(pairs);
    if (result) {
      console.log(
        `[Verify] gate=${gateLabel} sim=${simLabel} provider=gemini ` +
        `votes=${result.votesFor}/${result.votesTotal} ` +
        `verdict=${result.same ? 'accept' : 'reject'} conf=${result.confidencePct}`
      );
      return { same: result.same, confidencePct: result.confidencePct };
    }
    vlmNetworkStats.verifyInconclusive++;
    console.log(
      '[Verify] Gemini yielded no verdict (quota exhausted or network failure) — ' +
      'treating this attempt as unverifiable'
    );
    return null;
  }

  // No Gemini key at all — no verdict possible.
  return null;
}

/**
 * Exported entry point — same signature/contract as always. Callers
 * (candidate-retry.ts) are untouched; routing happens inside.
 * The legacy `swapped` self-consistency option no longer has a separate
 * provider path — the per-pair majority vote replaces it.
 */
export async function verifySameSceneMulti(
  pairs: VlmFramePair[],
  _options?: { swapped?: boolean },
): Promise<{ same: boolean; confidencePct: number } | null> {
  return routedVerify(pairs);
}

/**
 * GEMINI-FIRST entry point for the manual Retry button. SKIPS the SSCD
 * shortcut so a user-triggered Retry is always judged by Gemini itself
 * (with the exact same rate-limit system: sliding-window pacer,
 * wait-and-retry on per-minute 429s, and daily-limit detection that flags
 * the UI warning). If Gemini can yield no verdict at all, falls back to the
 * normal routed path (SSCD gate may still decide clear-cut cases).
 */
export async function verifySameSceneGeminiFirst(
  pairs: VlmFramePair[],
): Promise<{ same: boolean; confidencePct: number } | null> {
  if (geminiConfigured()) {
    const result = await compositeMajorityVerify(pairs);
    if (result) {
      console.log(
        `[Verify] (retry) gate=skipped provider=gemini ` +
        `votes=${result.votesFor}/${result.votesTotal} ` +
        `verdict=${result.same ? 'accept' : 'reject'} conf=${result.confidencePct}`
      );
      return { same: result.same, confidencePct: result.confidencePct };
    }
    console.log('[Verify] (retry) Gemini yielded no verdict — falling back to SSCD-gated path');
  }
  return routedVerify(pairs);
}

/**
 * Exported entry point — same signature/contract as always. Callers
 * (vlm-segment-resolver.ts, deferred-recovery.ts) are untouched.
 * The per-pair majority vote replaces the legacy swap self-consistency check.
 */
export async function verifySameSceneChecked(
  pairs: VlmFramePair[],
): Promise<{ same: boolean; confidencePct: number } | null> {
  return routedVerify(pairs);
}
