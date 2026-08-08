/**
 * FINAL-DECISION video-clip verification via Gemini.
 *
 * Instead of comparing sampled frames, this module cuts the ACTUAL two video
 * segments — the short-clip segment and the matched movie segment — encodes
 * them as small MP4 clips with ffmpeg, and sends BOTH clips inline to Gemini
 * with a forensic copyright-analysis prompt. Gemini's verdict on the clips
 * is the FINAL accept/reject decision for a candidate segment.
 *
 * Contracts:
 *  - Inert without GEMINI_API_KEY (geminiConfigured() === false -> null).
 *  - Every failure path (ffmpeg error, oversized clips, quota exhaustion,
 *    network failure, unparseable response) returns null — NEVER throws.
 *    Callers treat null as "no verdict" (unverifiable), not accept/reject.
 *  - Reuses the exact dual-model quota manager in gemini-vlm.ts, so video
 *    requests share the same RPM/RPD budget and rotation rules as before.
 *
 * Env vars:
 *  - GEMINI_VIDEO_MAX_CLIP_S   (default 30)  — max seconds sent per clip
 *  - GEMINI_VIDEO_TIMEOUT_MS   (default 180000) — per-request timeout
 *  - GEMINI_VIDEO_MAX_TOKENS   (default 2048)
 */
import { spawn } from 'child_process';
import { makeCleanEnv } from './pipeline';
import { FFMPEG_BIN } from './ffmpeg-path';
import { geminiConfigured, geminiGenerateText } from './gemini-vlm';

export const VIDEO_VERIFY_MAX_CLIP_S =
  Number(process.env.GEMINI_VIDEO_MAX_CLIP_S) || 30;
const VIDEO_VERIFY_TIMEOUT_MS =
  Number(process.env.GEMINI_VIDEO_TIMEOUT_MS) || 180_000;
const VIDEO_VERIFY_MAX_TOKENS =
  Number(process.env.GEMINI_VIDEO_MAX_TOKENS) || 2048;

/**
 * Per-clip byte budget. Gemini inline requests cap around 20MB total, and
 * base64 inflates bytes by ~33% — 6MB raw per clip keeps the whole request
 * comfortably under the limit even with two clips + prompt.
 */
const MAX_CLIP_BYTES = 6 * 1024 * 1024;

/** Extra movie-side context so a slightly-off alignment still overlaps. */
const MOVIE_PAD_S = 1.0;

/** The forensic analyst prompt — verdict rules live in the prompt itself. */
const FORENSIC_VIDEO_PROMPT = `You are a forensic video copyright analyst. You are given TWO video clips:
- VIDEO 1: a segment cut from a SHORT-FORM video (vertical 9:16 or similar)
- VIDEO 2: a segment cut from a FULL MOVIE (widescreen 16:9 or wider)

TASK: Determine whether VIDEO 1 was COPIED from VIDEO 2 — i.e. both contain
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

// ---------------------------------------------------------------------------
// Clip extraction
// ---------------------------------------------------------------------------

interface EncodeProfile {
  maxHeight: number;
  crf: number;
  fps: number;
}

/** First attempt: decent quality. Retry: aggressive shrink if oversized. */
const ENCODE_PROFILES: EncodeProfile[] = [
  { maxHeight: 480, crf: 30, fps: 12 },
  { maxHeight: 360, crf: 36, fps: 8 },
];

/**
 * Cut [start, start+duration] from a video and encode it as a small
 * fragmented MP4 (h264 + mono AAC audio so dialogue/music alignment is
 * usable as evidence), returned as a Buffer. Rejects on ffmpeg failure.
 */
function extractClipMp4(
  videoPath: string,
  startSeconds: number,
  durationSeconds: number,
  profile: EncodeProfile,
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const ss = Math.max(0, startSeconds).toFixed(3);
    const t = Math.max(0.5, durationSeconds).toFixed(3);
    const proc = spawn(
      FFMPEG_BIN,
      [
        '-ss', ss,
        '-t', t,
        '-i', videoPath,
        // \, escapes the comma for the filtergraph parser (no shell here).
        '-vf', `scale=-2:min(${profile.maxHeight}\\,ih)`,
        '-r', String(profile.fps),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', String(profile.crf),
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '48k',
        '-ac', '1',
        '-movflags', 'frag_keyframe+empty_moov',
        '-f', 'mp4',
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
        reject(new Error(
          `ffmpeg clip extraction failed (code ${code}) at t=${ss}s+${t}s: ${stderr.slice(-400)}`,
        ));
        return;
      }
      resolve(buf);
    });
  });
}

/** Extract a clip, retrying with the aggressive profile if it's too big. */
async function extractClipWithinBudget(
  videoPath: string,
  startSeconds: number,
  durationSeconds: number,
): Promise<Buffer | null> {
  for (const profile of ENCODE_PROFILES) {
    const buf = await extractClipMp4(videoPath, startSeconds, durationSeconds, profile);
    if (buf.length <= MAX_CLIP_BYTES) return buf;
    console.warn(
      `[VideoVerify] Clip too large (${(buf.length / 1024 / 1024).toFixed(1)}MB > ` +
      `${(MAX_CLIP_BYTES / 1024 / 1024).toFixed(0)}MB) at ${profile.maxHeight}p crf${profile.crf} — ` +
      `retrying with a smaller encode`,
    );
  }
  return null;
}

// ---------------------------------------------------------------------------
// Verdict parsing — the prompt's JSON includes a NESTED matchedTimeranges
// object, so the flat regex in gemini-vlm.ts's parseVerdictJson can't grab
// it. Balanced-brace scan for the LAST object containing a "same" field.
// ---------------------------------------------------------------------------

export interface VideoVerifyResult {
  same: boolean;
  confidence: number;
  evidence: string[];
  matchedTimeranges: { short?: string; movie?: string } | null;
}

export function parseForensicVerdict(raw: string): VideoVerifyResult | null {
  if (!raw) return null;
  const cleaned = raw.trim().replace(/```(?:json)?/gi, '').trim();

  let best: VideoVerifyResult | null = null;
  for (let i = 0; i < cleaned.length; i++) {
    if (cleaned[i] !== '{') continue;
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let j = i; j < cleaned.length; j++) {
      const ch = cleaned[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === '\\') escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') { inString = true; continue; }
      if (ch === '{') depth++;
      else if (ch === '}') {
        depth--;
        if (depth === 0) {
          const candidate = cleaned.slice(i, j + 1);
          if (candidate.includes('"same"')) {
            try {
              const parsed = JSON.parse(candidate);
              if (typeof parsed.same === 'boolean' && typeof parsed.confidence === 'number') {
                best = {
                  same: parsed.same,
                  confidence: Math.max(0, Math.min(100, parsed.confidence)),
                  evidence: Array.isArray(parsed.evidence)
                    ? parsed.evidence.filter((e: any) => typeof e === 'string').slice(0, 10)
                    : [],
                  matchedTimeranges:
                    parsed.matchedTimeranges && typeof parsed.matchedTimeranges === 'object'
                      ? parsed.matchedTimeranges
                      : null,
                };
              }
            } catch { /* keep scanning */ }
          }
          i = j; // resume outer scan past this object
          break;
        }
      }
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export interface SegmentBounds {
  shortStart: number;
  shortEnd: number;
  movieStart: number;
  movieEnd: number;
}

/**
 * Cut the two matched segments from the actual videos and ask Gemini for the
 * forensic verdict. Returns the parsed verdict, or null when no reliable
 * verdict could be obtained (never throws).
 */
export async function geminiVerifyVideoClips(
  shortVideoPath: string,
  movieVideoPath: string,
  seg: SegmentBounds,
): Promise<VideoVerifyResult | null> {
  if (!geminiConfigured()) return null;

  try {
    const shortDur = Math.max(0.5, seg.shortEnd - seg.shortStart);
    const dur = Math.min(shortDur, VIDEO_VERIFY_MAX_CLIP_S);
    // Center the verification window within the segment when it's longer
    // than the cap, and apply the same relative offset on the movie side.
    const shortOff = (shortDur - dur) / 2;
    const shortClipStart = seg.shortStart + shortOff;
    const movieClipStart = Math.max(0, seg.movieStart + shortOff - MOVIE_PAD_S);
    const movieClipDur = dur + 2 * MOVIE_PAD_S;

    const [shortClip, movieClip] = await Promise.all([
      extractClipWithinBudget(shortVideoPath, shortClipStart, dur),
      extractClipWithinBudget(movieVideoPath, movieClipStart, movieClipDur),
    ]);
    if (!shortClip || !movieClip) {
      console.warn('[VideoVerify] Could not fit clips within the size budget — no verdict');
      return null;
    }

    console.log(
      `[VideoVerify] Sending clips: short ${shortClipStart.toFixed(2)}s+${dur.toFixed(2)}s ` +
      `(${(shortClip.length / 1024).toFixed(0)}KB), movie ${movieClipStart.toFixed(2)}s+` +
      `${movieClipDur.toFixed(2)}s (${(movieClip.length / 1024).toFixed(0)}KB)`,
    );

    const raw = await geminiGenerateText(
      [
        { text: 'VIDEO 1 (segment cut from the short-form video):' },
        { inline_data: { mime_type: 'video/mp4', data: shortClip.toString('base64') } },
        { text: 'VIDEO 2 (segment cut from the full movie):' },
        { inline_data: { mime_type: 'video/mp4', data: movieClip.toString('base64') } },
        { text: FORENSIC_VIDEO_PROMPT },
      ],
      { maxOutputTokens: VIDEO_VERIFY_MAX_TOKENS, timeoutMs: VIDEO_VERIFY_TIMEOUT_MS },
    );
    if (raw === null) return null;

    const verdict = parseForensicVerdict(raw);
    if (!verdict) {
      console.warn(`[VideoVerify] Unparseable response — no verdict: ${String(raw).slice(0, 200)}`);
      return null;
    }
    console.log(
      `[VideoVerify] Verdict same=${verdict.same} conf=${verdict.confidence}` +
      (verdict.evidence.length ? ` evidence: ${verdict.evidence.join(' | ').slice(0, 300)}` : ''),
    );
    return verdict;
  } catch (err: any) {
    console.warn(`[VideoVerify] Failed: ${err?.message || err}`);
    return null;
  }
}
