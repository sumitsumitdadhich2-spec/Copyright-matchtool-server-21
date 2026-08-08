/**
 * Scene verification — GEMINI ONLY.
 *
 * The legacy Qwen VLM server (VLM_ENDPOINT_URL / llama.cpp slots / KV-cache
 * resets) has been FULLY REMOVED from the app. Gemini is now the one and
 * only final checker for every matched segment.
 *
 * The SSCD / embedding systems are NOT decision-makers anymore — they are
 * used exclusively by the CANDIDATE system (candidate-embedding-rank.ts) to
 * build and rank the 10-candidate pool so the right movie location is tried
 * first. Accept/reject verdicts come only from Gemini.
 *
 * All network/parse failures are treated as "could not verify" (returns
 * null), never as a silent pass or fail — callers apply their own
 * unverifiable policy.
 */
import { spawn } from 'child_process';
import { makeCleanEnv } from './pipeline';
import { FFMPEG_BIN } from './ffmpeg-path';
import { geminiConfigured, geminiVerifyComposite } from './gemini-vlm';
import { buildSideBySideComposite } from './frame-composite';

export const VLM_CONFIDENCE_THRESHOLD =
  Number(process.env.VLM_CONFIDENCE_THRESHOLD) || 80;
export const VLM_MAX_ATTEMPTS =
  Number(process.env.VLM_MAX_ATTEMPTS) || 10;

/**
 * How many segments the callers (vlm-segment-resolver.ts /
 * deferred-recovery.ts) process at once. Gemini's own sliding-window pacer
 * (gemini-vlm.ts) is the real rate limiter — this just bounds how many
 * segments are mid-flight at any moment.
 */
export const VLM_CONCURRENCY =
  Number(process.env.VLM_CONCURRENCY) || 2;

/**
 * Aggregate, process-wide counters for the "inconclusive" outcome (a request
 * that never got a real answer — Gemini unavailable/exhausted/failed, NOT
 * Gemini saying "no match"). server.ts snapshots/resets these around each
 * match job — purely additive observability, never consulted by any
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

/** Strict single-composite prompt — Gemini judges one side-by-side image. */
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
 * Returns null only when NO pair could be judged at all (composites failed,
 * Gemini unconfigured, or Gemini failed on every pair).
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
 * The one verification entry point — GEMINI ONLY.
 *
 * Return contract (unchanged for all callers):
 *  - {same:true, ...}  => Gemini majority says same footage
 *  - {same:false, ...} => Gemini majority says different footage
 *  - null              => no reliable verdict (Gemini unconfigured, daily
 *                         quota gone, or every call failed) — callers apply
 *                         their existing unverifiable policy.
 */
export async function verifySameSceneChecked(
  pairs: VlmFramePair[],
): Promise<{ same: boolean; confidencePct: number } | null> {
  if (!geminiConfigured()) {
    return null;
  }
  const result = await compositeMajorityVerify(pairs);
  if (!result) {
    vlmNetworkStats.verifyInconclusive++;
    console.log('[Verify] Gemini yielded no verdict — treating as unverifiable');
    return null;
  }
  console.log(
    `[Verify] provider=gemini votes=${result.votesFor}/${result.votesTotal} ` +
    `verdict=${result.same ? 'accept' : 'reject'} conf=${result.confidencePct}`
  );
  return { same: result.same, confidencePct: result.confidencePct };
}

/**
 * Entry point for the manual Retry button — identical to
 * verifySameSceneChecked now that Gemini is the only provider. Kept as a
 * separate export so candidate-retry.ts's import stays unchanged.
 */
export async function verifySameSceneGeminiFirst(
  pairs: VlmFramePair[],
): Promise<{ same: boolean; confidencePct: number } | null> {
  return verifySameSceneChecked(pairs);
}
