# Nexus Video Match

A high-performance video copyright matching tool that locates clips inside a reference movie using perceptual fingerprinting and sequence-alignment.

## How it works

1. **Extract fingerprints** — server-side pipeline (ffmpeg → Node.js worker_threads → node-canvas) decodes video at 25fps and computes a 256-bit perceptual hash for 13 crop/zoom variants of each frame, plus a spatial color/skin/detail signature.
2. **Fingerprint storage** — results stored as `uploads/<jobId>_result.json` on disk.
3. **Matching** — `POST /api/match` runs `groundMatchedSegments()` (two-pass sequence-alignment engine) comparing the short clip against the reference movie.
4. **Preview** — results shown in the browser with side-by-side video playback and a per-frame similarity timeline.

## Stack

- **Frontend**: React 19 + TypeScript + Tailwind CSS v4 (served via Vite middleware)
- **Backend**: Express 5 + tsx (dev) / esbuild (prod)
- **Fingerprinting**: ffmpeg + worker_threads + node-canvas (server-side)
- **Matching**: Pure TypeScript — Uint32Array XOR+popcount Hamming, O(n×m) brute-force scan

## Key files

| File | Purpose |
|------|---------|
| `server.ts` | Express server with all API routes including `/api/match` |
| `server/pipeline.ts` | ffmpeg → worker_threads fingerprint extraction pipeline |
| `server/worker.ts` | Per-frame hash + signature computation (node-canvas) |
| `server/matching-engine.ts` | `groundMatchedSegments()` — the core matching algorithm |
| `src/shared/fingerprint.ts` | Shared types + `computeSignature()` + `computeHashAndFeatures()` |
| `src/App.tsx` | Main React UI — upload, progress, results, side-by-side preview |
| `src/VideoProcessor.ts` | Browser + server video processing, returns `jobId` |

## Running locally

```bash
npm install
PORT=5000 npm run dev
```

## API

- `POST /api/upload-chunk` — chunked video upload (5 MB chunks)
- `GET /api/status/:jobId` — fingerprint extraction progress
- `GET /api/result/:jobId` — download fingerprint JSON
- `POST /api/match` — `{ movieJobId, shortJobId }` → SSE stream of `progress` /
  `vlm-progress` events, ending in `done` → `{ segments, movieFrames, shortFrames }`

## Optional VLM (Qwen2.5-VL) scene verification pass

After `groundMatchedSegments()` produces its matched segments (hash-matching
speed/accuracy unchanged), `/api/match` can optionally run each segment
through a vision-language model to confirm the short clip and matched movie
frame actually show the same scene. Rejected candidates are retried against
the next-best alternative location in the movie (never a small time-shift of
the same spot), up to `VLM_MAX_ATTEMPTS` per short-clip range; if all
attempts are rejected, that range is dropped from the result.

- `server/vlm-verify.ts` — on-demand ffmpeg frame grab + vLLM
  OpenAI-compatible `chat/completions` call.
- `server/vlm-segment-resolver.ts` — the retry/replace loop
  (`resolveSegmentsWithVLM`).
- `getAlternateCandidatesForRange()` in `server/matching-engine.ts` exposes
  the pre-dedup candidate pool so alternatives don't require re-scanning.
- Original uploaded videos are now kept on disk after fingerprinting
  (`uploads/<jobId>_meta.json` → `videoPath`) so frames can be extracted
  on demand; `getVideoPathForJob()` in `server.ts` resolves the path.
- Config (env vars): `VLM_ENDPOINT_URL` (unset = feature off),
  `VLM_CONFIDENCE_THRESHOLD` (default 80), `VLM_MAX_ATTEMPTS` (default 10),
  `VLM_CONCURRENCY` (default 4, matching the vLLM server's KV-cache slot
  count — segments are verified with this many in flight at once instead of
  strictly one at a time; both `resolveSegmentsWithVLM` and
  `runDeferredRecoveryPass` use it, since segments are independent of each
  other and produce identical verdicts either way).
- If the endpoint is unset or unreachable, the pass is skipped gracefully
  (logged once) and the original hash-matched segments are returned
  unchanged — matching keeps working with the GPU server off.
- AWS-side vLLM + Qwen2.5-VL-7B-Instruct server setup (Docker Compose +
  systemd, auto-downloads and caches the model, survives reboots):
  `deploy/vlm-server/README.md`.

### Request-layer resilience (large-movie VLM overload fix)

On large movies (500+ segments, large candidate pools) enough concurrent VLM
HTTP calls could overwhelm the remote GPU server/tunnel, turning into a burst
of 500/502/503/`ECONNRESET`. This is a pacing/resilience fix only — it never
changes which candidate is tried, what counts as accept/reject, or matching
accuracy; the final verdict for any segment is identical whether a call
succeeds immediately or only after retries.

- `VLM_MAX_CONCURRENT_REQUESTS` (default 4) — caps how many `verifySameScene`
  HTTP calls are in flight at once, across the whole pipeline (main pass +
  deferred pass combined). Independent of `VLM_CONCURRENCY` above, which
  bounds how many *segments* are processed at once — this is a second cap at
  the actual network-call layer.
- `VLM_RESET_MAX_CONCURRENT_REQUESTS` (default `min(2, VLM_MAX_CONCURRENT_REQUESTS)`)
  — separate, smaller concurrency lane for cache-reset/slot-erase calls, so a
  burst of real verification requests never queues up behind reset traffic
  (or vice versa) — resets are hygiene, not correctness, and must not compete
  with verification for the same budget.
- `VLM_RETRY_ATTEMPTS` (default 3) / `VLM_RETRY_BASE_DELAY_MS` (default 1000)
  — exponential-backoff retry for *transient* failures only (HTTP 5xx,
  `ECONNRESET`/`ECONNREFUSED`/`ETIMEDOUT`/timeouts). Distinct from
  `VLM_MAX_ATTEMPTS` (how many different candidates get tried for a segment):
  this retry is about getting a real answer for one request that failed to
  complete, never about advancing to a different candidate.
- A call that exhausts its retries without a real response returns `null`
  (same contract as before) and is tracked as **inconclusive** — distinct
  from a genuine "VLM said no match" rejection. Per-job counters
  (`vlmNetworkStats` in `server/vlm-verify.ts`, reset per match job) roll up
  into `MatchJob.vlmStats` (`accepted` / `rejected` / `inconclusiveNetwork` /
  `otherUnverifiable` / `recoveredByDeferred` / `stillUnmatchedAfterDeferred`)
  returned in the `/api/match` `done` event and logged as a breakdown line —
  so "why is this range unmatched" is answerable (content mismatch vs. server
  overload) from results or logs, not silently merged together.
- Regression test: `test_vlm_overload_resilience.mts` (retry-then-succeed,
  retries-exhausted-is-inconclusive-not-rejection, non-retryable 4xx fails
  fast, concurrency cap holds under parallel load, reset lane isn't starved
  by verification traffic, retried vs. immediate success produce an identical
  verdict).

## User preferences

- Server mode is the default (faster, uses ffmpeg pipeline)
- Do not restructure the existing ffmpeg + worker_threads pipeline unless explicitly asked
- Keep Docker/deployment config untouched
