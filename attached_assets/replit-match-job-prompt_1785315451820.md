# Task: Make `/api/match` a persistent background job — same pattern as fingerprinting

## Problem

Fingerprinting jobs (`/api/upload-chunk` → background pipeline → `/api/status/:jobId`)
already survive tab backgrounding, app switching, and reconnects, because they are
tracked as **server-side jobs** that the frontend polls with `GET /api/status/:jobId`.

`/api/match`, however, is a single long-lived `POST` request that streams Server-Sent
Events (SSE) over one HTTP connection. When the tab is backgrounded on mobile (Chrome/
Android throttles or kills background network connections after ~10–15s), that
connection drops, `fetch`'s stream reader fails client-side, and the UI shows
"failed" / "connection lost" — even though nothing crashed. The problem is **only**
that the frontend has no way to reconnect and re-fetch the result, unlike
fingerprinting jobs.

## Goal

Convert `/api/match` to the **exact same background-job model** already used for
fingerprinting, so the match/VLM process:

- Keeps running on the server regardless of what the client does.
- Can be reconnected to from Job History (Reconnect / Stop / Delete), exactly like
  fingerprint jobs.
- Shows live progress (including VLM verification progress) via polling, not a
  persistent SSE/fetch-stream connection.
- Survives the user backgrounding the tab, switching apps, or losing signal for any
  length of time — when they come back, it should sync instantly (reuse the existing
  `visibilitychange` re-sync pattern already implemented for fingerprint jobs).

## Non-negotiable constraints — READ CAREFULLY

**Do NOT touch matching accuracy or speed in any way.** This is purely a
plumbing/transport change (how progress and results reach the client), not a change
to *what* is computed or *how fast* it's computed.

Specifically forbidden:
- Do NOT modify `groundMatchedSegments()`, `matchVideosFromFiles()`, or any function
  inside `server/matching-engine.ts`.
- Do NOT modify the VLM verification logic in `server/vlm-verify.ts` or
  `server/vlm-segment-resolver.ts` (timeouts, retry counts, confidence thresholds,
  model name, prompt content — none of it).
- Do NOT change `minSimilarity`, `minConsecutiveFrames`, `frameDrift`, or any other
  matching parameter defaults or bounds.
- Do NOT change the fingerprint extraction pipeline (`server/pipeline.ts`,
  `server/worker.ts`) at all — it already works correctly and is out of scope.
- Do NOT add polling delay, batching, or throttling that slows down when the match
  result becomes available to a client that stays connected. A user who keeps the tab
  open must see the result just as fast as today.
- Do NOT change how many CPU workers / threads matching or VLM verification use.

If any change you're about to make would alter a numeric result (segments found,
confidence %, timing of the algorithm itself), STOP — that is out of scope. This task
is 100% about **transport/reconnection**, not the algorithm.

## What to build

### 1. Server: convert `/api/match` into a job

Reuse the **same in-memory/disk job-registry pattern** already used for fingerprint
jobs (look at how `uploads/<jobId>_meta.json` and the video registry / job status map
work in `server.ts` and `server/pipeline.ts` — mirror that structure, don't invent a
new one).

- `POST /api/match` — accepts `{ movieJobId, shortJobId, minSimilarity,
  minConsecutiveFrames, frameDrift }` exactly as today. Instead of opening an SSE
  stream, it:
  1. Generates a new `matchJobId`.
  2. Registers a job entry with status `processing`, 0 progress.
  3. Immediately responds `{ matchJobId }` (HTTP 200, fast).
  4. Kicks off the **exact same** `matchVideosFromFiles(...)` call and (if
     applicable) the **exact same** `resolveSegmentsWithVLM(...)` call
     asynchronously in the background — unchanged internals, unchanged call
     signatures, unchanged options.
  5. The existing `onProgress` callbacks (`type: 'progress'` and
     `type: 'vlm-progress'`) should update the job's stored progress state instead
     of `sendEvent(...)` over SSE.
  6. On success, store `{ status: 'completed', segments, unmatchedRanges,
     movieFrames, shortFrames }` on the job.
  7. On failure, store `{ status: 'failed', error }` on the job — same error
     surfacing as today (e.g. VLM failure still falls back to hash-matched segments
     exactly as the current try/catch around `resolveSegmentsWithVLM` does; do not
     change that fallback behavior).
  8. Support a `stopped` status if the user explicitly stops the job (see below).

- `GET /api/match-status/:matchJobId` — returns current job state: `status`
  (`processing` | `completed` | `failed` | `stopped`), latest progress fields
  (phase, pct, chunkIdx, totalChunks, shortStart, shortEnd, segmentsFound — same
  shape as today's `progress` events; plus vlm-progress fields when in that phase),
  and on `completed`, the full result payload (`segments`, `unmatchedRanges`,
  `movieFrames`, `shortFrames`).

- `POST /api/match-stop/:matchJobId` — mirror however fingerprint jobs currently
  support being stopped (check `server.ts` for the existing stop endpoint for
  fingerprint jobs and copy that mechanism — likely an abort flag / AbortController
  checked between chunks/segments). Must cleanly stop between segment-scan steps
  without corrupting partial results; matching itself doesn't need mid-algorithm
  cancellation precision beyond what fingerprinting already does.

- `DELETE /api/match/:matchJobId` — mirror the existing fingerprint job delete
  endpoint: removes the job's tracked state (and its stored result if persisted to
  disk under `uploads/`, matching whatever convention fingerprint results use).

- Match jobs should appear in whatever **job history / registry** structure already
  powers the fingerprinting history list, so they show up in the same "Job History"
  UI panel alongside fingerprint jobs — with a way to distinguish a match job from a
  fingerprint job (e.g. a `type: 'match'` vs `type: 'fingerprint'` field).

- If the process/server restarts while a match job is `processing`, follow
  whatever the existing fingerprint-job startup reconciliation already does (see
  `[Startup] Rebuilt video registry: N cached job(s)...` logic) — mirror it for
  match jobs rather than inventing new recovery semantics.

### 2. Frontend: replace the SSE fetch-stream with polling

In `src/App.tsx`, replace the current `/api/match` `fetch` + `reader.read()` SSE loop
(the block that builds `matchStartTime`, reads `payload.type === 'progress' | 'done'
| 'error'`) with:

- `POST /api/match` → get `matchJobId`, then call a new `pollMatchUntilDone(matchJobId)`
  that mirrors `pollUntilDone(...)` **exactly** in structure:
  - Same 1500ms interval with the same exponential backoff on network errors
    (`Math.min(1500 * Math.pow(2, consecutiveErrors - 1), 15_000)`, same
    `MAX_CONSECUTIVE_ERRORS = 8` cutoff).
  - Same three outcomes: `completed` → populate `segments` / `unmatched` /
    `matchStats` exactly as today; `failed` → `setErrorMsg` exactly as today;
    `stopped` → clear state exactly as the fingerprint `stopped` case does.
  - On persistent connection loss (hits `MAX_CONSECUTIVE_ERRORS`), do **not** show a
    hard failure — show the same "Connection lost. Use Job History → Reconnect to
    resume tracking progress." message the fingerprint flow already shows, and keep
    the job's session so it can be reattached.

- Extend the existing `visibilitychange` handler (around the block that re-syncs
  `refJobId` / `targetJobId` on tab-focus) to **also** immediately re-fetch
  `/api/match-status/:matchJobId` when a match job is in-flight, exactly the way it
  already re-fetches `/api/status/:jobId` for ref/target jobs. This is the fix for
  the reported bug: a 10–15s background dip must resync instantly instead of
  showing "failed".

- Extend `handleReattach(jobId)` so it also recognizes match jobs (not just
  `reference`/`target` fingerprint jobs) and calls `pollMatchUntilDone(jobId)` to
  resume tracking — same Job History → "Reconnect" button UX already in place for
  fingerprint jobs, just wired to match jobs too.

- Add "Stop" and "Delete" actions for match jobs in the Job History panel, calling
  the new `/api/match-stop/:matchJobId` and `/api/match/:matchJobId` (DELETE)
  endpoints — same buttons/pattern already present for fingerprint jobs in that
  panel, just pointed at the new endpoints for jobs of type `match`.

- The live progress UI (currently driven by `matchProgress` state on each SSE
  `progress`/`vlm-progress` event) should update identically on each poll tick —
  same fields, same rendering — just sourced from `GET /api/match-status/:matchJobId`
  responses instead of SSE messages.

## Acceptance criteria

1. Start a match on a small clip (e.g. the ~1000-frame movie / ~450-frame short
   clip test case). Background the tab for 30+ seconds mid-match, then return —
   progress must resync immediately (via `visibilitychange`) and the match must
   complete and show results, with **no error shown**, even though the tab was
   backgrounded well past the old 10–15s failure window.
2. Start a match, close the tab entirely, reopen the app, go to Job History, and
   click "Reconnect" on the in-progress match job — it must resume live progress
   tracking and land on the same final result as if you'd never left.
3. Stop a running match job from Job History — server-side processing must
   actually stop (not just hide in the UI).
4. Delete a completed or stopped match job from Job History — it disappears from
   the list and its stored data is cleaned up.
5. For a fixed pair of test videos, **segment counts, confidence percentages, and
   total wall-clock matching+VLM time must be identical (within normal run-to-run
   noise) to what the current SSE implementation produces** — prove this with a
   before/after comparison on the same input. Any measurable regression in accuracy
   or speed is a failed implementation, even if the reconnect feature works.
6. VLM fallback-to-hash-only-segments behavior (when VLM endpoint is unset/
   unreachable/fails) must be unchanged and still verified working.

## Summary of the one rule that matters most

**Change only how progress and results travel from server to browser. Do not touch
what is computed, in what order, with what parameters, or how many workers/threads
do it.** If you're unsure whether a change affects accuracy or speed, don't make it —
ask first instead of guessing.
