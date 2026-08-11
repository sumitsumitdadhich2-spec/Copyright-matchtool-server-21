/**
 * Cut a matched segment out of a video into a small, self-contained temp MP4
 * so it can be uploaded to Gemini for VIDEO-level verification.
 *
 * This replaces the old "send one JPEG frame per side" protocol: a single
 * frame throws away every temporal cue (action continuity, camera movement,
 * cut points, audio), which is exactly what makes a copy-detection verdict
 * reliable. Now the ACTUAL overlapping footage from both timelines is cut and
 * handed to the model.
 *
 * Decisions locked in with the user:
 *  - HIGH-QUALITY RE-ENCODE (not stream copy), so the cut lands on the exact
 *    requested timestamps instead of snapping to the nearest keyframe.
 *  - ORIGINAL RESOLUTION, no downscaling whatsoever.
 *  - EXACT segment bounds — no padding, no minimum-duration expansion.
 *  - Hard cap of SEGMENT_MAX_SECONDS (60s); anything longer is trimmed to the
 *    MIDDLE window of that length, which is the part most likely to contain
 *    the strongest shared footage.
 *
 * FAIL-SAFE: every failure path returns null and logs — a cut failure must
 * degrade to "unverifiable", never crash a match job.
 */
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { makeCleanEnv } from './pipeline';
import { FFMPEG_BIN } from './ffmpeg-path';

/** Hard cap on how long a single uploaded segment may be. */
export const SEGMENT_MAX_SECONDS =
  Number(process.env.GEMINI_SEGMENT_MAX_SECONDS) || 60;

/** x264 quality for the re-encode. 18 is visually transparent for this use. */
const SEGMENT_CRF = Number(process.env.GEMINI_SEGMENT_CRF) || 18;
const SEGMENT_PRESET = process.env.GEMINI_SEGMENT_PRESET || 'veryfast';
/** Give up on a single cut after this long. */
const SEGMENT_CUT_TIMEOUT_MS =
  Number(process.env.GEMINI_SEGMENT_CUT_TIMEOUT_MS) || 240_000;

export interface CutWindow {
  start: number;
  duration: number;
}

/**
 * Clamp a [start, end] range to a valid, non-empty cut window of at most
 * SEGMENT_MAX_SECONDS. Over-long ranges keep their MIDDLE window; short
 * ranges are used exactly as-is (no expansion — user's explicit choice).
 */
export function clampCutWindow(start: number, end: number): CutWindow {
  const safeStart = Math.max(0, Number.isFinite(start) ? start : 0);
  const safeEnd = Number.isFinite(end) ? end : safeStart;
  // Guard against inverted/degenerate ranges from upstream rounding.
  const rawDuration = Math.max(0.04, safeEnd - safeStart);

  if (rawDuration <= SEGMENT_MAX_SECONDS) {
    return { start: safeStart, duration: rawDuration };
  }
  const center = safeStart + rawDuration / 2;
  return {
    start: Math.max(0, center - SEGMENT_MAX_SECONDS / 2),
    duration: SEGMENT_MAX_SECONDS,
  };
}

function tempSegmentPath(label: string): string {
  return path.join(
    os.tmpdir(),
    `gemini-seg-${label}-${Date.now()}-${Math.random().toString(36).slice(2)}.mp4`,
  );
}

/**
 * Cut [start, end] out of `videoPath` into a temp MP4 and return its path.
 * Caller owns the file and must call `deleteTempSegment` when done.
 */
export async function cutSegmentToTempFile(
  videoPath: string,
  start: number,
  end: number,
  label: string,
): Promise<string | null> {
  const { start: cutStart, duration } = clampCutWindow(start, end);
  const outPath = tempSegmentPath(label);

  const args = [
    '-hide_banner',
    '-loglevel', 'error',
    '-y',
    // -ss BEFORE -i keeps the seek fast; because we re-encode, ffmpeg still
    // decodes from the preceding keyframe and discards, so the output starts
    // at the exact requested timestamp.
    '-ss', cutStart.toFixed(3),
    '-i', videoPath,
    '-t', duration.toFixed(3),
    // Video stream is mandatory; audio is taken only if the source has one.
    '-map', '0:v:0',
    '-map', '0:a:0?',
    // High-quality re-encode at the SOURCE resolution — no scale filter here
    // on purpose, so nothing is downsampled before the model sees it.
    '-c:v', 'libx264',
    '-crf', String(SEGMENT_CRF),
    '-preset', SEGMENT_PRESET,
    '-pix_fmt', 'yuv420p',
    '-c:a', 'aac',
    '-b:a', '128k',
    '-movflags', '+faststart',
    '-avoid_negative_ts', 'make_zero',
    outPath,
  ];

  return new Promise((resolve) => {
    const proc = spawn(FFMPEG_BIN, args, { env: makeCleanEnv() });
    let stderr = '';
    let settled = false;

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { proc.kill('SIGKILL'); } catch { /* already gone */ }
      console.warn(`[SegmentCut] ${label} cut timed out after ${SEGMENT_CUT_TIMEOUT_MS}ms`);
      deleteTempSegment(outPath);
      resolve(null);
    }, SEGMENT_CUT_TIMEOUT_MS);

    proc.stderr.on('data', (c: Buffer) => { stderr += c.toString(); });
    proc.on('error', (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      console.warn(`[SegmentCut] ${label} ffmpeg spawn failed: ${err?.message || err}`);
      resolve(null);
    });
    proc.on('close', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      let size = 0;
      try { size = fs.statSync(outPath).size; } catch { size = 0; }
      if (code !== 0 || size === 0) {
        console.warn(
          `[SegmentCut] ${label} cut failed (code ${code}, ${size} bytes) ` +
          `at ${cutStart.toFixed(2)}s +${duration.toFixed(2)}s: ${stderr.slice(-300)}`
        );
        deleteTempSegment(outPath);
        resolve(null);
        return;
      }
      console.log(
        `[SegmentCut] ${label}: ${cutStart.toFixed(2)}s +${duration.toFixed(2)}s ` +
        `-> ${(size / 1_048_576).toFixed(1)} MB`
      );
      resolve(outPath);
    });
  });
}

/** Best-effort temp-file cleanup — never throws. */
export function deleteTempSegment(filePath: string | null | undefined): void {
  if (!filePath) return;
  fs.promises.unlink(filePath).catch(() => { /* already gone */ });
}
