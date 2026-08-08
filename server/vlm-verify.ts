/**
 * Segment verification — GEMINI VIDEO edition.
 *
 * Gemini is the ONE AND ONLY verdict-maker. What changed: it no longer
 * receives a single still frame per side. Both matched segments are CUT out of
 * their source videos and the two real video clips are uploaded in ONE request
 * (reference movie segment = VIDEO 1, target clip segment = VIDEO 2). A still
 * frame discards action continuity, camera movement, cut timing and audio —
 * exactly the evidence that makes a copy-detection verdict accurate.
 *
 * The fast frame pre-filters (SSCD / CLIP embeddings in
 * candidate-embedding-rank) are unchanged and still accept/reject NOTHING —
 * they only rank which candidate Gemini looks at first.
 *
 * All failure paths return null ("unverifiable") so a segment falls back to
 * the caller's unverifiable policy rather than crashing the match.
 */
import { spawn } from 'child_process';
import { makeCleanEnv } from './pipeline';
import { FFMPEG_BIN } from './ffmpeg-path';
import { geminiConfigured, geminiVerifyVideoPair } from './gemini-vlm';
import { cutSegmentToTempFile, deleteTempSegment, SEGMENT_MAX_SECONDS } from './segment-cutter';

export const VLM_CONFIDENCE_THRESHOLD =
  Number(process.env.VLM_CONFIDENCE_THRESHOLD) || 80;
export const VLM_MAX_ATTEMPTS =
  Number(process.env.VLM_MAX_ATTEMPTS) || 10;

// How many *segments* a caller (vlm-segment-resolver.ts / deferred-recovery.ts)
// processes at once. Gemini's own per-minute pacing lives inside gemini-vlm.ts,
// so this only bounds concurrent segment cutting + in-flight verifications.
// Kept low by default: each in-flight verification now re-encodes and uploads
// two real video segments.
export const VLM_CONCURRENCY =
  Number(process.env.VLM_CONCURRENCY) || 2;

/** Re-exported so callers can surface the cap in logs/UI. */
export { SEGMENT_MAX_SECONDS };

/**
 * Aggregate, process-wide counters for the "inconclusive" outcome (a request
 * that never got a real answer — quota exhausted/network/cut failure, NOT the
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

/**
 * Grab a single JPEG frame from a video at `timestampSeconds`, base64-encoded.
 * Gemini no longer sees these — this stays only for the LOCAL candidate
 * ranking pre-filter (candidate-embedding-rank.ts).
 */
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

// ===========================================================================
// PROMPT — user-authored, verbatim. Do not edit without asking.
// ===========================================================================

/** Label part sent immediately before the reference-movie segment. */
export const VIDEO_1_LABEL = 'VIDEO 1: a segment cut from a rafrence movie';
/** Label part sent immediately before the target-clip segment. */
export const VIDEO_2_LABEL = 'VIDEO 2: a segment cut from a target clip.';

export const VIDEO_VERIFY_PROMPT = `TASK: Determine whether VIDEO 2 was COPIED from VIDEO 1 — i.e. both contain
the SAME underlying footage (same shots, same recording), not merely similar
footage.
=== CRITICAL: ASPECT RATIO / CROP RULES ===
The short is almost always a VERTICAL CROP cut from SOME horizontal position
of the widescreen movie frame — LEFT, CENTER, RIGHT, or ANY arbitrary slice
the editor chose. The editor may even have cropped a small corner or an
off-center region of the screen.

Before judging, mentally scan the ENTIRE movie frame edge-to-edge and check
whether the short's content appears in ANY region of it.
NEVER say "different" just because the movie frame shows EXTRA people,
objects, or scenery that the short does not show. The short seeing LESS is
EXPECTED — it is a narrow slice.
NEVER say "different" because the subject sits at a different position
within the frame, or because framing/composition looks different.
The short may also be zoomed, mirrored (horizontally flipped), letterboxed,
color-graded, filtered, sped up/slowed slightly, or have text/watermarks/
captions/UI overlays added. NONE of these make it a different video.
=== WHAT TO COMPARE (check ALL of these) ===

ACTION CONTINUITY: does the same physical action unfold in the same order
with the same motion? (e.g. a punch, a turn, a vehicle passing — the exact
same movement, not a similar one)
BACKGROUND DETAILS: walls, furniture, posters, trees, vehicles, buildings,
crowd members, lighting fixtures — the small details BEHIND the subjects.
These are the strongest fingerprint. Same scene = same background details
in the overlapping region.
PEOPLE: same individuals — face structure, hair, clothing, accessories,
body build. Also every background/secondary person visible in both.
WARDROBE & PROPS: exact same clothing items, colors, patterns, objects
being held or used.
CAMERA WORK: same camera angle, same camera movement (pan/zoom/handheld
shake), same shot transitions AT THE SAME MOMENTS within the overlap.
LIGHTING & TIME: same light direction, shadows, time-of-day, weather.
SHOT SEQUENCE: if multiple shots/cuts occur in the clips, the same cuts
must occur showing the same content (order may offset if the segments
don't align perfectly — judge the overlapping portion).
=== STRICT REJECTION RULES (do NOT be lenient) ===

Same movie but a DIFFERENT scene/moment = NOT a match. Same actors in the
same costumes at a different location or doing a different action = NO.
Same location but different take/angle/action = NOT a match.
Generic look-alikes (similar rooms, similar streets, similar fights,
similar dances) = NOT a match. You must identify the EXACT same recording.
If the clips are too blurry/dark/ambiguous to verify concrete shared
details, answer same=false. NEVER guess in favor of a match.
Do NOT hallucinate differences either: "person looks slightly different"
due to crop/compression/color-grade is NOT a valid difference. A valid
difference is a CONCRETE contradiction — different background object,
different clothing item, different action, different location geometry.
=== EVIDENCE REQUIREMENT ===
Your verdict MUST cite at least 3 concrete, specific visual details
(background objects, clothing items, exact actions, camera moves) that are
either SHARED (for match) or CONTRADICTORY (for no-match). "Looks similar"
or "looks different" is NOT evidence.

=== OUTPUT (JSON only, after your evidence) ===
{
"same": true|false,
"confidence": 0-100,
"matchedTimeranges": {"short": "start-end sec", "movie": "start-end sec"} | null,
"evidence": ["detail 1", "detail 2", "detail 3"]
}

same=true ONLY when confidence >= 90 AND you cited concrete shared details
including at least one BACKGROUND detail match.
If audio is available in both clips, use dialogue/music/sound-effects
alignment as additional evidence.`;

// ===========================================================================
// SEGMENT VERIFICATION
// ===========================================================================

/** The timeline bounds of one candidate — the only fields verification needs. */
export interface VerifiableSegmentRange {
  shortStart: number;
  shortEnd: number;
  movieStart: number;
  movieEnd: number;
}

export interface SegmentVerifyResult {
  same: boolean;
  confidencePct: number;
  /**
   * How likely THIS candidate is a real match, on a single 0-100 scale
   * derived from Gemini's own confidence:
   *   same=true  -> confidence
   *   same=false -> 100 - confidence  (a weakly-rejected candidate scores
   *                 higher than a confidently-rejected one)
   * Used only to pick the "best so far" fallback after retries run out; it
   * never overrides an accept/reject verdict.
   */
  matchLikelihood: number;
  /** Concrete details the model cited, when it supplied them. */
  evidence?: string[];
}

/**
 * Duration-based Gemini frame-sampling tiers (confirmed by controlled API
 * testing): Gemini samples at 1 fps by default, which yields ZERO frames —
 * and an HTTP 400 — for clips shorter than ~0.5s. Instead of padding the
 * cuts, we raise video_metadata.fps for short segments only:
 *   < 0.5s      -> 7 fps
 *   0.5s to ~2s -> 3 fps
 *   bigger      -> 1 fps (Gemini default — request shape unchanged)
 */
export function fpsForDuration(durationSeconds: number): number {
  if (durationSeconds < 0.5) return 7;
  if (durationSeconds <= 2.0) return 3;
  return 1;
}

/**
 * fps for a verification PAIR: both clips MUST be sampled at the SAME fps
 * for a fair comparison, so the tier is picked from the SHORTER of the two
 * durations — the short clip must never be under-sampled.
 */
export function fpsForSegmentPair(seg: VerifiableSegmentRange): number {
  const shortDur = Math.max(0, seg.shortEnd - seg.shortStart);
  const movieDur = Math.max(0, seg.movieEnd - seg.movieStart);
  return fpsForDuration(Math.min(shortDur, movieDur));
}

/**
 * Cut both sides of a candidate out of their source videos and ask Gemini —
 * in ONE request — whether the target segment was copied from the reference
 * segment. Returns null when no verdict could be obtained (Gemini off/quota
 * exhausted/network failure, or either cut failed).
 */
export async function verifySegmentByVideo(
  shortVideoPath: string,
  movieVideoPath: string,
  seg: VerifiableSegmentRange,
  logLabel = 'Verify',
): Promise<SegmentVerifyResult | null> {
  if (!geminiConfigured()) return null;

  let moviePart: string | null = null;
  let shortPart: string | null = null;
  try {
    // Cut both sides in parallel — two independent ffmpeg processes.
    [moviePart, shortPart] = await Promise.all([
      cutSegmentToTempFile(movieVideoPath, seg.movieStart, seg.movieEnd, 'reference'),
      cutSegmentToTempFile(shortVideoPath, seg.shortStart, seg.shortEnd, 'target'),
    ]);

    if (!moviePart || !shortPart) {
      vlmNetworkStats.verifyInconclusive++;
      console.warn(`[${logLabel}] Could not cut both segments — treating as unverifiable`);
      return null;
    }

    // Same fps on BOTH clips, tiered by the SHORTER duration in the pair —
    // prevents the HTTP 400 zero-frame failure on sub-0.5s segments without
    // touching the cut bounds (no padding).
    const pairFps = fpsForSegmentPair(seg);
    if (pairFps > 1) {
      console.log(`[${logLabel}] Short-segment sampling: fps=${pairFps} applied to both clips`);
    }

    const verdict = await geminiVerifyVideoPair(
      moviePart,
      shortPart,
      VIDEO_1_LABEL,
      VIDEO_2_LABEL,
      VIDEO_VERIFY_PROMPT,
      pairFps,
    );

    if (!verdict) {
      vlmNetworkStats.verifyInconclusive++;
      console.log(
        `[${logLabel}] Gemini yielded no verdict (quota exhausted or network failure) — ` +
        'treating this attempt as unverifiable'
      );
      return null;
    }

    const matchLikelihood = verdict.same
      ? verdict.confidence
      : 100 - verdict.confidence;

    console.log(
      `[${logLabel}] provider=gemini-video ` +
      `short=[${seg.shortStart.toFixed(2)}s–${seg.shortEnd.toFixed(2)}s] ` +
      `movie=[${seg.movieStart.toFixed(2)}s–${seg.movieEnd.toFixed(2)}s] ` +
      `verdict=${verdict.same ? 'same' : 'different'} conf=${verdict.confidence} ` +
      `likelihood=${matchLikelihood}` +
      (verdict.evidence?.length ? ` evidence=${verdict.evidence.length}` : '')
    );

    return {
      same: verdict.same,
      confidencePct: verdict.confidence,
      matchLikelihood,
      evidence: verdict.evidence,
    };
  } catch (err: any) {
    vlmNetworkStats.verifyInconclusive++;
    console.warn(`[${logLabel}] Segment verification errored: ${err?.message || err}`);
    return null;
  } finally {
    deleteTempSegment(moviePart);
    deleteTempSegment(shortPart);
  }
}
