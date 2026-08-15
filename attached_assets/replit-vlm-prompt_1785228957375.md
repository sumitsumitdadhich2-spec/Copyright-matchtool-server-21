# Task: Add Qwen2.5-VL verification layer on top of existing matching engine

## CRITICAL CONSTRAINT — read first
Do NOT modify, refactor, or change the behavior of the existing fingerprinting
pipeline (`server/pipeline.ts`, `server/worker.ts`) or the core matching
algorithm inside `groundMatchedSegments()` in `server/matching-engine.ts`.
Matching speed and accuracy must stay EXACTLY as they are today. This is a new
verification pass that runs AFTER the existing engine already produced its
matched segments — it must not change how those segments are computed.

## What already exists (do not touch the logic, only add to it)
- `groundMatchedSegments()` in `server/matching-engine.ts` returns
  `{ segments: MatchedSegment[], unmatchedRanges }`.
- Each `MatchedSegment` has `shortStart`, `shortEnd`, `movieStart`, `movieEnd`,
  `confidence`, `matchSequence` (array of `{shortTime, movieTime, similarity}`).
- Original uploaded video files are now KEPT on disk after fingerprinting
  (not deleted — storage is not a constraint). Their path is stored in each
  job's meta file (`uploads/<jobId>_meta.json`) under the field `videoPath`.
  Use the existing `getVideoPathForJob(jobId)` helper in `server.ts` to read it.

## What to build

### 1. Auto-download Qwen2.5-VL-7B — zero manual steps
Set up the AWS GPU server (assume Ubuntu + NVIDIA GPU, e.g. g6.xlarge) to run
Qwen2.5-VL-7B-Instruct via vLLM's OpenAI-compatible Docker image, with the
model auto-pulled from Hugging Face on first container start (vLLM does this
automatically — it downloads and caches the model the first time
`--model Qwen/Qwen2.5-VL-7B-Instruct` is used). Persist the Hugging Face cache
to a mounted volume so re-starts of the container do NOT re-download the model.
Set `--restart unless-stopped` so the container (and therefore the model
server) comes back up automatically after any crash or reboot — the user
should never have to manually start it again. Provide the docker run command
and a systemd or docker-compose setup so it survives an EC2 reboot too.

### 2. New file: `server/vlm-verify.ts`
Responsibilities:
- `extractFrameAsBase64(videoPath: string, timestampSeconds: number): Promise<string>`
  — use ffmpeg (same spawn pattern/env as `server/pipeline.ts`'s `makeCleanEnv()`)
  to grab a single JPEG frame at a timestamp, return as base64. Do this
  on-demand, only for the specific frames being verified — never re-decode
  the whole video.
- `verifySameScene(shortFrameB64: string, movieFrameB64: string): Promise<{ same: boolean; confidencePct: number }>`
  — call the AWS-hosted vLLM OpenAI-compatible endpoint
  (`POST http://<AWS_VLM_HOST>:8000/v1/chat/completions`) with model
  `Qwen/Qwen2.5-VL-7B-Instruct`, sending both images and a prompt that forces
  a strict JSON reply, e.g.:
  ```
  Compare these two video frames. Are they showing the same scene/subject
  (allowing for compression, crop, or color grading differences)? Reply with
  ONLY JSON: {"same": true|false, "confidence": 0-100}
  ```
  Parse the JSON response (strip markdown fences if present). Read the host
  from an env var, e.g. `VLM_ENDPOINT_URL`, with a sane default.
- Treat any network/parse failure as "could not verify" — do NOT silently
  treat it as a pass or fail; log it and skip VLM gating for that attempt
  (fall back to keeping the segment as-is) rather than crashing the whole match.

### 3. New file: `server/vlm-segment-resolver.ts`
This holds the retry/replace loop described below. Export one function:

```ts
async function resolveSegmentsWithVLM(
  segments: MatchedSegment[],
  shortVideoPath: string,
  movieVideoPath: string,
  allMovieCandidates: (shortSeg: MatchedSegment) => MatchedSegment[] // see below
): Promise<MatchedSegment[]>
```

Logic per segment:
1. Take the segment's best representative frame pair — use the short-clip
   timestamp and matched movie timestamp from `bestFrameDetail` if present,
   otherwise the midpoint of `matchSequence`.
2. Extract both frames, call `verifySameScene`.
3. If `same === true` AND `confidencePct >= 80` → ACCEPT. Keep this segment
   in the output, move to the next segment.
4. Otherwise → REJECT this candidate. Look up the next-best alternative
   match for this same short-clip time range from the matching engine's
   full candidate pool (see point 4 below) — an alternative location
   ANYWHERE else in the movie, not a small time-shift near the rejected one.
   Never re-show a candidate (movie timestamp) already rejected for this
   short-clip range in this resolution loop — keep a `Set` of rejected
   movie-timestamp keys per short-clip range to enforce this.
5. Repeat steps 2–4 with the new candidate. Cap at 10 attempts total per
   short-clip range. If all 10 are rejected (or no more candidates exist),
   drop this short-clip range entirely from the final result — log it as
   "no genuine match found after 10 attempts" and move to the next segment.
   Do not loop forever.

### 4. Expose alternative candidates from the matching engine
`groundMatchedSegments()` currently keeps only the single best segment per
short-clip time range (see the dedup block that sorts by confidence and
drops overlaps). Without changing its output contract or performance, add an
internal-only way to retrieve the N-best alternative candidates for a given
short-clip range — e.g. an optional export like
`getAlternateCandidatesForRange(shortStart, shortEnd, excludeMovieTimestamps: number[])`
that reuses the already-computed similarity data (do NOT re-run the full
O(n×m) scan — reuse whatever intermediate candidate list already exists
before the final dedup step, just expose it). If no such intermediate list
currently exists, cache the pre-dedup candidate list per short-clip range in
memory for the duration of that single `/api/match` request only (not
persisted, not global state across requests).

### 5. Wire it into `server.ts`'s `/api/match` route
After `matchVideosFromFiles(...)` returns its `result`, and only if both
`getVideoPathForJob(movieJobId)` and `getVideoPathForJob(shortJobId)` resolve
to existing files, run `resolveSegmentsWithVLM(...)` on `result.segments`
before sending the `done` SSE event. If either video file is missing (e.g.
old jobs from before this feature), skip VLM verification entirely and
return the original segments unchanged — do not error out.

Stream progress for this pass too, e.g. `sendEvent({ type: 'vlm-progress', segmentIndex, attempt, verdict })`
so the frontend can show "Verifying segment 2/5 with AI (attempt 3)..." live.

### 6. Config
- `VLM_ENDPOINT_URL` env var (default `http://localhost:8000/v1/chat/completions`
  or the AWS server's address).
- `VLM_CONFIDENCE_THRESHOLD` env var, default `80`.
- `VLM_MAX_ATTEMPTS` env var, default `10`.
- If `VLM_ENDPOINT_URL` is unset or unreachable at request time, skip VLM
  verification gracefully (log a warning once) and return the original
  matching-engine segments untouched — the tool must keep working even if
  the AWS GPU server is off.

## Non-negotiables (repeat for emphasis)
- Existing hash-matching speed and accuracy: UNCHANGED.
- No re-processing of already-fingerprinted videos.
- No re-decoding of full videos for VLM — only single on-demand frame grabs.
- Never show the same rejected movie-timestamp candidate twice for the same
  short-clip range.
- Hard cap of 10 attempts per short-clip range, then give up and drop it.
- Qwen2.5-VL-7B must auto-start and auto-download on first run — no manual
  steps for the user going forward.
