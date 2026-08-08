/**
 * FULL-VIDEO segment verification (replaces the frame-pair final check).
 *
 * For a matched segment, the exact matched time ranges are CUT out of both
 * videos with ffmpeg:
 *   VIDEO 1 = [movieStart, movieEnd]  cut from the reference movie
 *   VIDEO 2 = [shortStart, shortEnd]  cut from the target clip
 * Both clips (with audio, compressed to fit the request budget) are sent to
 * Gemini in ONE request so the model can judge the ENTIRE motion, shot
 * sequence, and audio — not isolated frames.
 *
 * Contract: returns {same, confidencePct} or null (could not verify —
 * callers fall back to the legacy frame-based path so the tool keeps
 * working when Gemini is unavailable/out of quota).
 */
import { spawn } from 'child_process';
import { makeCleanEnv } from './pipeline';
import { FFMPEG_BIN } from './ffmpeg-path';
import { geminiConfigured, geminiVerifySegmentClips } from './gemini-vlm';

// ---------------------------------------------------------------------------
// Size budget — Gemini inline requests are capped at ~20 MB total. Base64
// inflates by 4/3, so the two RAW clips together must stay under ~14 MB.
// Each clip gets ~6.5 MB; the video bitrate is computed from the segment
// duration so ANY length segment fits (longer segment -> lower bitrate,
// never trimmed — the whole matched range is always sent).
// ---------------------------------------------------------------------------
const CLIP_BYTE_BUDGET = Number(process.env.SEGMENT_CLIP_BYTE_BUDGET) || 6_500_000;
const CLIP_HEIGHT = Number(process.env.SEGMENT_CLIP_HEIGHT) || 360;
const CLIP_FPS = Number(process.env.SEGMENT_CLIP_FPS) || 10;
const AUDIO_KBPS = 32;
const MIN_VIDEO_KBPS = 80;
const MAX_VIDEO_KBPS = 1000;

/** The full-video comparison prompt (VIDEO 1 = movie segment, VIDEO 2 = target segment). */
export const SEGMENT_VIDEO_PROMPT = `- VIDEO 1: a segment cut from a reference movie
- VIDEO 2: a segment cut from a target clip.
TASK: Determine whether VIDEO 2 was COPIED from VIDEO 1 — i.e. both contain
the SAME underlying footage (same shots, same recording), not merely similar
footage.

=== CRITICAL: ASPECT RATIO / CROP RULES ===
The short is almost always a VERTICAL CROP cut from SOME horizontal position
of the widescreen movie frame — LEFT, CENTER, RIGHT, or ANY arbitrary slice
the editor chose. The editor may even have cropped a small corner or an
off-center region of the screen.
- Before judging, mentally scan the ENTIRE movie frame edge-to-edge and check
  whether the short's content appears in ANY region of it.
- NEVER say "different" just because the movie frame shows EXTRA people,
  objects, or scenery that the short does not show. The short seeing LESS is
  EXPECTED — it is a narrow slice.
- NEVER say "different" because the subject sits at a different position
  within the frame, or because framing/composition looks different.
- The short may also be zoomed, mirrored (horizontally flipped), letterboxed,
  color-graded, filtered, sped up/slowed slightly, or have text/watermarks/
  captions/UI overlays added. NONE of these make it a different video.

=== WHAT TO COMPARE (check ALL of these) ===
1. ACTION CONTINUITY: does the same physical action unfold in the same order
   with the same motion? (e.g. a punch, a turn, a vehicle passing — the exact
   same movement, not a similar one)
2. BACKGROUND DETAILS: walls, furniture, posters, trees, vehicles, buildings,
   crowd members, lighting fixtures — the small details BEHIND the subjects.
   These are the strongest fingerprint. Same scene = same background details
   in the overlapping region.
3. PEOPLE: same individuals — face structure, hair, clothing, accessories,
   body build. Also every background/secondary person visible in both.
4. WARDROBE & PROPS: exact same clothing items, colors, patterns, objects
   being held or used.
5. CAMERA WORK: same camera angle, same camera movement (pan/zoom/handheld
   shake), same shot transitions AT THE SAME MOMENTS within the overlap.
6. LIGHTING & TIME: same light direction, shadows, time-of-day, weather.
7. SHOT SEQUENCE: if multiple shots/cuts occur in the clips, the same cuts
   must occur showing the same content (order may offset if the segments
   don't align perfectly — judge the overlapping portion).

=== STRICT REJECTION RULES (do NOT be lenient) ===
- Same movie but a DIFFERENT scene/moment = NOT a match. Same actors in the
  same costumes at a different location or doing a different action = NO.
- Same location but different take/angle/action = NOT a match.
- Generic look-alikes (similar rooms, similar streets, similar fights,
  similar dances) = NOT a match. You must identify the EXACT same recording.
- If the clips are too blurry/dark/ambiguous to verify concrete shared
  details, answer same=false. NEVER guess in favor of a match.
- Do NOT hallucinate differences either: "person looks slightly different"
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
- same=true ONLY when confidence >= 80 AND you cited concrete shared details
  including at least one BACKGROUND detail match.
- If audio is available in both clips, use dialogue/music/sound-effects
  alignment as additional evidence.`;

/**
 * Cut [start, end] out of `videoPath` as a compact H.264/AAC fragmented MP4,
 * bitrate scaled so the whole clip fits the byte budget regardless of
 * duration. Returned base64-encoded. Rejects on ffmpeg failure.
 */
export function extractSegmentClipAsBase64(
  videoPath: string,
  startSeconds: number,
  endSeconds: number,
): Promise<string> {
  const start = Math.max(0, startSeconds);
  const duration = Math.max(0.5, endSeconds - start);

  // Bits available for video after audio takes its share.
  const totalKbps = (CLIP_BYTE_BUDGET * 8) / duration / 1000;
  const videoKbps = Math.round(
    Math.min(MAX_VIDEO_KBPS, Math.max(MIN_VIDEO_KBPS, totalKbps - AUDIO_KBPS)),
  );

  return new Promise((resolve, reject) => {
    const args = [
      '-ss', start.toFixed(3),
      '-t', duration.toFixed(3),
      '-i', videoPath,
      '-vf', `scale=-2:${CLIP_HEIGHT},fps=${CLIP_FPS}`,
      '-c:v', 'libx264',
      '-preset', 'veryfast',
      '-b:v', `${videoKbps}k`,
      '-maxrate', `${videoKbps}k`,
      '-bufsize', `${videoKbps * 2}k`,
      '-pix_fmt', 'yuv420p',
      '-c:a', 'aac',
      '-b:a', `${AUDIO_KBPS}k`,
      '-ac', '1',
      '-sn', '-dn',
      '-movflags', 'frag_keyframe+empty_moov',
      '-f', 'mp4',
      'pipe:1',
    ];
    const proc = spawn(FFMPEG_BIN, args, { env: makeCleanEnv() });

    const chunks: Buffer[] = [];
    let stderr = '';
    proc.stdout.on('data', (c: Buffer) => chunks.push(c));
    proc.stderr.on('data', (c: Buffer) => { stderr += c.toString(); });
    proc.on('error', reject);
    proc.on('close', (code) => {
      const buf = Buffer.concat(chunks);
      if (code !== 0 || buf.length === 0) {
        reject(new Error(
          `ffmpeg segment cut failed (code ${code}) at [${start.toFixed(2)}s +${duration.toFixed(2)}s]: ${stderr.slice(-400)}`,
        ));
        return;
      }
      resolve(buf.toString('base64'));
    });
  });
}

/** The time ranges a segment verification needs — matches MatchedSegment's fields. */
export interface SegmentRanges {
  shortStart: number;
  shortEnd: number;
  movieStart: number;
  movieEnd: number;
}

/**
 * FULL-VIDEO verification of one matched segment: cuts the matched range out
 * of BOTH videos and asks Gemini whether VIDEO 2 (target segment) was copied
 * from VIDEO 1 (movie segment). Returns null when no verdict could be
 * obtained (Gemini unconfigured / out of quota / cut failed) — callers then
 * fall back to the legacy frame path.
 */
export async function verifySegmentByVideo(
  shortVideoPath: string,
  movieVideoPath: string,
  seg: SegmentRanges,
): Promise<{ same: boolean; confidencePct: number } | null> {
  if (!geminiConfigured()) return null;

  let movieClipB64: string;
  let shortClipB64: string;
  try {
    [movieClipB64, shortClipB64] = await Promise.all([
      extractSegmentClipAsBase64(movieVideoPath, seg.movieStart, seg.movieEnd),
      extractSegmentClipAsBase64(shortVideoPath, seg.shortStart, seg.shortEnd),
    ]);
  } catch (err: any) {
    console.warn(`[VideoVerify] Segment cut failed: ${err?.message || err}`);
    return null;
  }

  const movieMB = (movieClipB64.length * 0.75 / 1_000_000).toFixed(1);
  const shortMB = (shortClipB64.length * 0.75 / 1_000_000).toFixed(1);
  console.log(
    `[VideoVerify] Sending full segments to Gemini — ` +
    `movie [${seg.movieStart.toFixed(1)}s–${seg.movieEnd.toFixed(1)}s] (${movieMB} MB), ` +
    `short [${seg.shortStart.toFixed(1)}s–${seg.shortEnd.toFixed(1)}s] (${shortMB} MB)`,
  );

  const verdict = await geminiVerifySegmentClips(movieClipB64, shortClipB64, SEGMENT_VIDEO_PROMPT);
  if (!verdict) return null;

  console.log(
    `[VideoVerify] verdict=${verdict.same ? 'MATCH' : 'no-match'} conf=${verdict.confidence} ` +
    `for short [${seg.shortStart.toFixed(1)}s–${seg.shortEnd.toFixed(1)}s]`,
  );
  return { same: verdict.same, confidencePct: verdict.confidence };
}
