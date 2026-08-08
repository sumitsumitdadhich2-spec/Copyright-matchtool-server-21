/**
 * VLM scene-verification helpers — GEMINI VIDEO-SEGMENT edition.
 *
 * The old frame-composite protocol (side-by-side JPEG stills) is GONE.
 * Verification now works on the actual matched SEGMENTS: the overlapping
 * timeline ranges are cut out of BOTH videos with a high-quality re-encode
 * (frame-accurate, CRF 18 — visually lossless) and sent to Gemini in ONE
 * request as two separate videos, each with its own label, plus the task
 * prompt. Gemini judges the full motion/action/audio, not a single frame.
 *
 *   1. Segments longer than VLM_VIDEO_SEGMENT_CAP_S (default 60s) are capped
 *      to the CENTER 60s of the match, with the movie side mapped through
 *      the same proportional offset so both cuts stay aligned.
 *   2. If Gemini yields no verdict (not configured, daily quota exhausted,
 *      network failure), the result is null — "could not verify". Callers
 *      treat null as unverifiable, NEVER as a silent pass or fail.
 *
 * All network/parse failures return null so a segment falls back to being
 * handled by the caller's unverifiable policy rather than crashing the match.
 */
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { makeCleanEnv } from './pipeline';
import { FFMPEG_BIN } from './ffmpeg-path';
import { geminiConfigured, geminiVerifyVideoPair } from './gemini-vlm';

export const VLM_CONFIDENCE_THRESHOLD =
  Number(process.env.VLM_CONFIDENCE_THRESHOLD) || 80;
export const VLM_MAX_ATTEMPTS =
  Number(process.env.VLM_MAX_ATTEMPTS) || 10;

// How many *segments* a caller (vlm-segment-resolver.ts / deferred-recovery.ts)
// processes at once. Gemini's own per-minute pacing lives inside gemini-vlm.ts,
// so this only bounds concurrent segment cutting + in-flight verifications.
export const VLM_CONCURRENCY =
  Number(process.env.VLM_CONCURRENCY) || 2;

/** Hard cap on the duration (seconds) of each cut segment sent to Gemini. */
export const VLM_VIDEO_SEGMENT_CAP_S =
  Number(process.env.VLM_VIDEO_SEGMENT_CAP_S) || 60;

/** Minimum duration (seconds) of a cut — very short matches get a little
 *  context so Gemini has actual motion to judge, not a near-still. */
const VLM_VIDEO_SEGMENT_MIN_S = 1;

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

/** Grab a single JPEG frame from a video at `timestampSeconds`, base64-encoded.
 *  (No longer used for Gemini verification — kept for tooling/compat.) */
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
// SEGMENT CUTTING — frame-accurate, high-quality re-encode (CRF 18)
// ===========================================================================

/**
 * Cut [startSeconds, startSeconds+durationSeconds] out of a video into a
 * temp MP4 with a high-quality re-encode:
 *  - `-ss` BEFORE `-i` + re-encode = frame-accurate cut (not keyframe-snapped)
 *  - libx264 CRF 18 = visually lossless, no meaningful quality loss
 *  - audio kept (AAC) when the source has an audio stream, so Gemini can use
 *    dialogue/music alignment as extra evidence
 * Returns the cut file's bytes and deletes the temp file.
 */
export async function extractSegmentAsMp4(
  videoPath: string,
  startSeconds: number,
  durationSeconds: number,
): Promise<Buffer> {
  const start = Math.max(0, startSeconds);
  const dur = Math.max(VLM_VIDEO_SEGMENT_MIN_S, durationSeconds);
  const tempPath = path.join(
    os.tmpdir(),
    `vlm-seg-${Date.now()}-${Math.random().toString(36).slice(2)}.mp4`,
  );

  await new Promise<void>((resolve, reject) => {
    const proc = spawn(
      FFMPEG_BIN,
      [
        '-ss', start.toFixed(3),
        '-i', videoPath,
        '-t', dur.toFixed(3),
        '-map', '0:v:0',
        '-map', '0:a:0?',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-y', tempPath,
      ],
      { env: makeCleanEnv() },
    );
    let stderr = '';
    proc.stderr.on('data', (c: Buffer) => { stderr += c.toString(); });
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`ffmpeg segment cut failed (code ${code}) at t=${start.toFixed(2)}s: ${stderr.slice(-400)}`));
        return;
      }
      resolve();
    });
  });

  try {
    const bytes = await fs.promises.readFile(tempPath);
    if (bytes.length === 0) throw new Error('ffmpeg produced an empty segment file');
    return bytes;
  } finally {
    fs.promises.unlink(tempPath).catch(() => { /* best-effort cleanup */ });
  }
}

// ===========================================================================
// VIDEO-PAIR VERIFICATION — the ONE AND ONLY Gemini check
// ===========================================================================

const VIDEO1_LABEL = 'VIDEO 1: a segment cut from a reference movie';
const VIDEO2_LABEL = 'VIDEO 2: a segment cut from a target clip';

const VIDEO_PAIR_PROMPT =
  `TASK: Determine whether VIDEO 2 was COPIED from VIDEO 1 — i.e. both contain
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
alignment as additional evidence`;

/** The minimal shape needed to locate a matched segment on both timelines. */
export interface SegmentTimeRange {
  shortStart: number;
  shortEnd: number;
  movieStart: number;
  movieEnd: number;
}

/**
 * Compute the aligned (short, movie) cut windows for a matched segment,
 * capped to VLM_VIDEO_SEGMENT_CAP_S. When capping, the CENTER of the match
 * is kept and the movie window is mapped through the same proportional
 * offset so both cuts show the same underlying moment even when the short
 * was sped up/slowed down.
 */
export function computeAlignedCutWindows(seg: SegmentTimeRange): {
  shortStart: number; shortDur: number;
  movieStart: number; movieDur: number;
} {
  const shortDurFull = Math.max(0.04, seg.shortEnd - seg.shortStart);
  const movieDurFull = Math.max(0.04, seg.movieEnd - seg.movieStart);
  const scale = movieDurFull / shortDurFull;

  let winStart = seg.shortStart;
  let winDur = shortDurFull;
  if (winDur > VLM_VIDEO_SEGMENT_CAP_S) {
    const mid = seg.shortStart + shortDurFull / 2;
    winStart = mid - VLM_VIDEO_SEGMENT_CAP_S / 2;
    winDur = VLM_VIDEO_SEGMENT_CAP_S;
  }

  const movieWinStart = seg.movieStart + (winStart - seg.shortStart) * scale;
  const movieWinDur = Math.min(winDur * scale, VLM_VIDEO_SEGMENT_CAP_S);

  return {
    shortStart: Math.max(0, winStart),
    shortDur: Math.max(VLM_VIDEO_SEGMENT_MIN_S, winDur),
    movieStart: Math.max(0, movieWinStart),
    movieDur: Math.max(VLM_VIDEO_SEGMENT_MIN_S, movieWinDur),
  };
}

/**
 * THE verification entry point. Cuts the matched segment out of BOTH videos
 * (high-quality re-encode, ≤60s each) and sends them to Gemini in ONE
 * request — VIDEO 1 = reference-movie segment, VIDEO 2 = target-clip
 * segment — with the strict copied-footage prompt.
 *
 * Returns null (not false) when the call could not be completed or parsed —
 * callers must treat null as "unverifiable for this attempt", not a verdict.
 */
export async function verifySameSceneVideo(
  shortVideoPath: string,
  movieVideoPath: string,
  seg: SegmentTimeRange,
): Promise<{ same: boolean; confidencePct: number } | null> {
  if (!geminiConfigured()) return null;

  const win = computeAlignedCutWindows(seg);
  let movieBytes: Buffer;
  let shortBytes: Buffer;
  try {
    [movieBytes, shortBytes] = await Promise.all([
      extractSegmentAsMp4(movieVideoPath, win.movieStart, win.movieDur),
      extractSegmentAsMp4(shortVideoPath, win.shortStart, win.shortDur),
    ]);
  } catch (err: any) {
    console.warn(`[Verify] Segment cutting failed: ${err?.message || err} — treating as unverifiable`);
    vlmNetworkStats.verifyInconclusive++;
    return null;
  }

  const result = await geminiVerifyVideoPair(
    movieBytes,
    VIDEO1_LABEL,
    shortBytes,
    VIDEO2_LABEL,
    VIDEO_PAIR_PROMPT,
  );

  if (result === null) {
    vlmNetworkStats.verifyInconclusive++;
    console.log(
      '[Verify] Gemini yielded no verdict (quota exhausted or network failure) — ' +
      'treating this attempt as unverifiable'
    );
    return null;
  }

  console.log(
    `[Verify] provider=gemini-video ` +
    `short=[${seg.shortStart.toFixed(1)}s–${seg.shortEnd.toFixed(1)}s] ` +
    `movie=[${seg.movieStart.toFixed(1)}s–${seg.movieEnd.toFixed(1)}s] ` +
    `verdict=${result.same ? 'accept' : 'reject'} conf=${result.confidence}`
  );
  return { same: result.same, confidencePct: result.confidence };
}
