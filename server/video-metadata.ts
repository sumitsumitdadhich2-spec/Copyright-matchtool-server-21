/**
 * Video metadata probing + duplicate-frame masking — Task 1 foundation of the
 * FPS-aware, speed-tolerant alt-candidate pipeline.
 * ---------------------------------------------------------------------------
 * Everything here is ADDITIVE and consumed ONLY by:
 *   - the alt-candidate expansion pass (server/alt-expansion.ts), and
 *   - the verification records (server/verification/) as debug metadata.
 * Nothing in this file is read by the protected Pass 1/2/3 matching engine.
 *
 * Why no CFR re-encode: the fingerprint pipeline (server/pipeline.ts) already
 * decodes every video through `ffmpeg -r 25`, which resamples ANY source
 * (10fps..120fps, CFR or VFR) onto a constant 25fps timestamp grid before
 * hashing. Timestamps in the fingerprint NDJSON are therefore already
 * normalized. What that resampling CANNOT hide is a low-fps source: a 10fps
 * short exported to the 25fps grid repeats each real frame ~2.5×, producing
 * runs of (near-)identical consecutive hashes that distort line fits and
 * waste relaxed-search samples. The duplicate-frame mask below detects those
 * runs directly from the hashes so downstream consumers can skip them.
 */

import { execFile } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { FFPROBE_BIN } from './ffmpeg-path';
import { makeCleanEnv } from './pipeline';

// ---------------------------------------------------------------------------
// Env tunables (all safe-defaulted; see also alt-expansion.ts / verify.ts)
// ---------------------------------------------------------------------------

function envNum(name: string, fallback: number, min: number, max: number): number {
  const n = Number(process.env[name]);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

/** Max Hamming distance (bits) between consecutive full-variant aHashes for a
 *  frame to count as a duplicate of its predecessor. */
const ALT_DUP_HAMMING = Math.round(envNum('ALT_DUP_HAMMING', 2, 0, 32));

// ---------------------------------------------------------------------------
// ffprobe metadata
// ---------------------------------------------------------------------------

export interface VideoStreamMetadata {
  /** Path that was probed (for debugging; may no longer exist later). */
  videoPath: string;
  /** Declared frame rate from r_frame_rate (e.g. 30000/1001 → 29.97). */
  declaredFps: number | null;
  /** Real average frame rate from avg_frame_rate. */
  averageFps: number | null;
  /** True when r_frame_rate and avg_frame_rate disagree beyond tolerance —
   *  the source is (or was) variable frame rate. Informational only: the
   *  fingerprint pipeline's `-r 25` resample already normalizes timestamps. */
  isVFR: boolean;
  durationSeconds: number | null;
  width: number | null;
  height: number | null;
  probedAt: number;
  /** Present when ffprobe failed — metadata is best-effort, never fatal. */
  error?: string;
}

/** Both sides of a match job, saved durably like verify records. */
export interface MatchVideoMetadata {
  short: VideoStreamMetadata | null;
  movie: VideoStreamMetadata | null;
}

function parseRational(raw: unknown): number | null {
  if (typeof raw !== 'string' || raw.length === 0) return null;
  const slash = raw.indexOf('/');
  if (slash === -1) {
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : null;
  }
  const num = Number(raw.slice(0, slash));
  const den = Number(raw.slice(slash + 1));
  if (!Number.isFinite(num) || !Number.isFinite(den) || den === 0) return null;
  const v = num / den;
  return Number.isFinite(v) && v > 0 ? v : null;
}

/**
 * Probe a video file with ffprobe. NEVER throws — on any failure it returns a
 * metadata object with `error` set and null fields, so callers can save it
 * durably and move on without risking the match job.
 */
export function probeVideoMetadata(videoPath: string): Promise<VideoStreamMetadata> {
  const base: VideoStreamMetadata = {
    videoPath,
    declaredFps: null,
    averageFps: null,
    isVFR: false,
    durationSeconds: null,
    width: null,
    height: null,
    probedAt: Date.now(),
  };

  return new Promise(resolve => {
    try {
      execFile(
        FFPROBE_BIN,
        [
          '-v', 'error',
          '-select_streams', 'v:0',
          '-show_entries', 'stream=r_frame_rate,avg_frame_rate,width,height,duration',
          '-show_entries', 'format=duration',
          '-of', 'json',
          videoPath,
        ],
        { env: makeCleanEnv(), timeout: 30_000 },
        (err, stdout) => {
          if (err) {
            resolve({ ...base, error: err.message || String(err) });
            return;
          }
          try {
            const parsed = JSON.parse(stdout);
            const stream = parsed?.streams?.[0] ?? {};
            const declaredFps = parseRational(stream.r_frame_rate);
            const averageFps = parseRational(stream.avg_frame_rate);
            const streamDur = Number(stream.duration);
            const formatDur = Number(parsed?.format?.duration);
            const durationSeconds = Number.isFinite(streamDur) && streamDur > 0
              ? streamDur
              : (Number.isFinite(formatDur) && formatDur > 0 ? formatDur : null);
            // VFR: declared vs average disagree by > 0.5% (covers 29.97-vs-30
            // style rounding without false positives).
            const isVFR = declaredFps !== null && averageFps !== null &&
              Math.abs(declaredFps - averageFps) / Math.max(declaredFps, averageFps) > 0.005;
            resolve({
              ...base,
              declaredFps,
              averageFps,
              isVFR,
              durationSeconds,
              width: Number.isFinite(Number(stream.width)) ? Number(stream.width) : null,
              height: Number.isFinite(Number(stream.height)) ? Number(stream.height) : null,
            });
          } catch (e: any) {
            resolve({ ...base, error: `ffprobe output parse failed: ${e?.message || e}` });
          }
        },
      );
    } catch (e: any) {
      resolve({ ...base, error: e?.message || String(e) });
    }
  });
}

// ---------------------------------------------------------------------------
// Durable per-match-job storage — same pattern as verify records
// ---------------------------------------------------------------------------

function metadataFilePath(uploadDir: string, matchJobId: string): string {
  return path.join(uploadDir, `${matchJobId}_videometa.json`);
}

/** Write the probed metadata for a match job. Never throws. */
export async function saveMatchVideoMetadata(
  uploadDir: string,
  matchJobId: string,
  meta: MatchVideoMetadata,
): Promise<void> {
  try {
    await fs.promises.writeFile(metadataFilePath(uploadDir, matchJobId), JSON.stringify(meta));
  } catch (e: any) {
    console.error(`[VideoMeta] Failed to save metadata for ${matchJobId}: ${e?.message || e}`);
  }
}

/** Read back a job's saved metadata. Returns null when absent/corrupt. */
export function readMatchVideoMetadata(uploadDir: string, matchJobId: string): MatchVideoMetadata | null {
  const p = metadataFilePath(uploadDir, matchJobId);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, 'utf-8')) as MatchVideoMetadata;
  } catch (e: any) {
    console.warn(`[VideoMeta] Corrupt metadata file for ${matchJobId}: ${e?.message || e}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Duplicate consecutive-frame mask (low-fps sources on the 25fps grid)
// ---------------------------------------------------------------------------

/** Minimal frame shape needed by the mask — matches FPData structurally
 *  without importing the matching engine (keeps this module dependency-free
 *  from the protected engine). */
export interface HashedFrame {
  timestamp: number;
  variants: Record<string, { hash: string }>;
}

export interface DuplicateFrameMask {
  /** mask[i] === 1 → frame i is a (near-)exact repeat of frame i-1. */
  mask: Uint8Array;
  duplicateCount: number;
  /** unique frames / total frames — 1.0 for a true 25fps source,
   *  ~0.4 for a 10fps source resampled onto the 25fps grid. */
  uniqueRatio: number;
  /** Effective real fps of the source, inferred from the sampled grid rate ×
   *  uniqueRatio. Null when the grid rate cannot be estimated. */
  effectiveFps: number | null;
}

function hammingStrings(a: string, b: string): number {
  const len = Math.min(a.length, b.length);
  let d = Math.abs(a.length - b.length);
  for (let i = 0; i < len; i++) {
    if (a.charCodeAt(i) !== b.charCodeAt(i)) d++;
  }
  return d;
}

/** Median seconds-per-frame of the sampled timestamp grid. 0 when unknown. */
export function medianFrameDuration(frames: Array<{ timestamp: number }>): number {
  if (frames.length < 2) return 0;
  const sampleCount = Math.min(frames.length - 1, 200);
  const deltas: number[] = [];
  for (let i = 1; i <= sampleCount; i++) {
    const d = frames[i].timestamp - frames[i - 1].timestamp;
    if (d > 0) deltas.push(d);
  }
  if (deltas.length === 0) return 0;
  deltas.sort((a, b) => a - b);
  return deltas[Math.floor(deltas.length / 2)];
}

/**
 * Build a mask of duplicate consecutive frames from the full-variant hashes.
 * A frame counts as a duplicate when its hash is within ALT_DUP_HAMMING bits
 * of the previous frame's hash. Consumers (alt-expansion) skip masked frames
 * when sampling relaxed-search seeds and RANSAC points so a low-fps source's
 * repeated frames don't dominate the fit.
 */
export function buildDuplicateFrameMask(frames: HashedFrame[]): DuplicateFrameMask {
  const mask = new Uint8Array(frames.length);
  let duplicateCount = 0;

  const hashOf = (f: HashedFrame): string => {
    const v = f.variants['full'] ?? f.variants[Object.keys(f.variants)[0]];
    return v?.hash ?? '';
  };

  let prevHash = frames.length > 0 ? hashOf(frames[0]) : '';
  for (let i = 1; i < frames.length; i++) {
    const h = hashOf(frames[i]);
    if (prevHash.length > 0 && h.length > 0 && hammingStrings(prevHash, h) <= ALT_DUP_HAMMING) {
      mask[i] = 1;
      duplicateCount++;
    }
    prevHash = h;
  }

  const uniqueRatio = frames.length > 0 ? (frames.length - duplicateCount) / frames.length : 1;
  const frameDur = medianFrameDuration(frames);
  const gridFps = frameDur > 0 ? 1 / frameDur : null;
  const effectiveFps = gridFps !== null ? gridFps * uniqueRatio : null;

  return { mask, duplicateCount, uniqueRatio, effectiveFps };
}
