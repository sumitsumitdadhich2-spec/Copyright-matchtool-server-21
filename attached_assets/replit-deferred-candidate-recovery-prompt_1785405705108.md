# Task: Background candidate discovery for rejected segments + deferred verification pass + preview-only candidate display

## Problem

The current matching + VLM verification pipeline works like this: the hash-
matching engine (`groundMatchedSegments` in `server/matching-engine.ts`) finds
one best candidate per short-clip range and puts it in `candidatePool`. When VLM
rejects that candidate, `getAlternateCandidatesForRange()` looks for another
entry in `candidatePool` for the same range — but because the matching engine
only ever records one best candidate per range (not multiple), there is usually
nothing else to try, so the range gets dropped as unmatched even when the movie
genuinely contains a matching (but differently-scored) segment elsewhere.

The matching engine's accuracy (roughly 80% correct on the first candidate) is
good and must not be touched. This task is entirely about giving the other ~20%
(the rejected segments) a real second chance, without slowing down or changing
the 80% that already works.

## Non-negotiable constraints — READ CAREFULLY

**Do NOT modify the matching engine's core logic, scoring, or the first-pass
candidate selection in any way.** `groundMatchedSegments()`, `mergeAdjacentSegments()`,
`contextValidateSegments()`, scene-chunk detection, and all existing pass 1/2/3
logic in `server/matching-engine.ts` stay exactly as they are. The 80% that
already gets accepted on the first VLM check must be produced through the exact
same code path as today, with identical timing.

Specifically forbidden:
- Do NOT change VLM_MAX_ATTEMPTS, confidence thresholds, timeout values, or the
  VLM prompt content.
- Do NOT slow down the main verification pass for segments that get accepted on
  the first try — the new background candidate search must run fully in
  parallel with ongoing VLM verification, never blocking it or competing for the
  same VLM request slots until the deferred pass explicitly begins (see below).
- Do NOT include any of these newly-discovered alternate candidates in the
  final result JSON — the JSON that the API returns and that persists as the
  job's result must only ever contain final accepted segments, exactly as today's
  schema. Candidate data is for on-screen preview only (see step 4) and must
  live in a separate structure, never merged into the segments array that ships
  in the JSON response.
- Do NOT hold the growing candidate data in server memory/RAM as it accumulates
  across a run with many rejections — persist it to disk (under the existing
  `uploads/` job-storage convention already used elsewhere in `server.ts`), and
  read it back in when needed, so RAM usage stays flat regardless of how many
  segments get rejected in one job.

## What to build

### 1. Background candidate discovery, triggered immediately on each rejection

In `server/vlm-segment-resolver.ts` (wherever a segment's VLM verification
ultimately fails after exhausting attempts and the segment is about to be
dropped), add a side-effect: as soon as a segment is confirmed rejected, kick off
an async, non-blocking search for **10 alternate candidate locations** for that
short-clip range elsewhere in the movie's fingerprint data. This search:

- Runs in the background — it must not delay or block the ongoing VLM
  verification loop for other segments. Use the existing worker-thread pool
  pattern already present in the codebase (`server/worker.ts` / the pipeline's
  worker pool) rather than inventing a new concurrency primitive, if there is
  spare capacity; otherwise queue it as fire-and-forget work that doesn't
  compete with active VLM request slots.
- Uses the same fingerprint/hash-matching approach the engine already has
  (reuse existing similarity-scoring functions from `matching-engine.ts` — do
  not write a second, different similarity algorithm) to find the next-best 10
  distinct movie-timestamp candidates for that short-clip range, beyond the one
  that was already tried and rejected.
- As each batch of 10 candidates is found for a rejected segment, write them to
  disk immediately (under the job's `uploads/` directory, e.g.
  `<jobId>_candidates.json` or appended per-segment file — follow whatever
  naming convention the existing job-result files use) rather than keeping them
  in memory. Do not accumulate this in a JS array/Map that grows for the
  lifetime of the job.
- This happens for every segment that gets rejected during the main pass, for
  the entire duration of that pass — multiple segments' candidate searches may
  be in flight or queued at once.

### 2. Deferred verification pass, after the main pass finishes

Once the main VLM verification pass for a match job has processed every segment
(all accepted or exhausted-and-rejected), start a second pass:

- Process each rejected segment **one at a time, in the order they were
  originally rejected** (not parallel across segments — sequential, one
  rejected segment fully resolved before moving to the next).
- For the current rejected segment, load its 10 background-discovered
  candidates from disk and verify them with VLM **one at a time** (reusing the
  exact same VLM verification call/logic as the main pass — same prompt, same
  confidence handling, same accept/reject criteria) until one is accepted or all
  10 are exhausted.
- If one of the 10 is accepted, this becomes the segment's final result. If
  none are accepted, the segment remains unmatched, exactly as it would have
  before this change (no regression — this is strictly additive, a segment
  that was going to be dropped can now potentially be recovered, never
  worse-off than today's outcome).
- Move to the next originally-rejected segment and repeat.

### 3. Chronological reinsertion into final results

When a deferred-pass candidate is accepted for a previously-rejected segment,
insert it into the job's final `segments` array at the position corresponding
to its **original short-clip timestamp range** (i.e., where it would have
appeared if it had been accepted on the first attempt) — not appended at the
end. The final segments array must remain sorted by `shortStart` exactly as
`groundMatchedSegments()` already sorts it today (see the existing
`final.sort((a, b) => a.shortStart - b.shortStart)` pattern) — apply the same
sort (or a stable insert) after merging in any deferred-pass results.

### 4. Preview-only candidate display (not in the result JSON)

The result JSON returned by the match API and persisted as the job's stored
result must contain only final segments — unchanged shape from today, no new
fields with candidate data mixed in.

Separately, expose the per-segment candidate data (from the disk files written
in step 1) through a distinct read path — e.g. a new endpoint like
`GET /api/match/:matchJobId/candidates/:segmentId` or an aggregate
`GET /api/match/:matchJobId/candidates` — that the frontend preview UI can call
to show, for a given point in the timeline that was originally rejected, the
list of candidate frames that were checked (whether or not one was ultimately
accepted). This is purely for the human to visually compare in the preview UI;
it must never be embedded in or returned as part of the primary match result
JSON.

## Acceptance criteria

1. Run a match job with a mix of clearly-accepted and some rejected segments
   (reuse an existing test video pair known to produce a few rejections).
   Confirm via logs that background candidate searches start immediately upon
   each rejection during the main pass, and do not delay the main pass's
   completion time compared to before this change (measure and compare).
2. Confirm the deferred pass only starts after the main pass fully finishes,
   processes rejected segments strictly in their original rejection order, and
   tries their 10 candidates sequentially.
3. Confirm any segment recovered in the deferred pass appears in the final
   segments array at its correct chronological (shortStart-sorted) position,
   not appended out of order at the end.
4. Confirm the final result JSON contains zero candidate-related fields —
   inspect the raw JSON response/stored file to verify its shape is identical
   to a pre-change job's JSON, just with (potentially) more segments recovered.
5. Confirm candidate data written to disk during the run does not remain
   resident in server memory for the life of the job — verify via a memory
   check during a run with many (10+) rejections that RAM stays flat rather
   than growing proportionally to rejection count.
6. Confirm the preview UI's new candidate-fetching endpoint returns the checked
   candidates for a given previously-rejected segment, independent of the main
   result JSON.
7. For the segments that were already being accepted on the first try (the
   ~80% case), confirm identical output and identical timing to before this
   change — this task must not alter or slow down that path at all.

## Summary of the one rule that matters most

**The first-pass matching engine and its ~80% first-try accuracy path must be
completely untouched — same code, same speed, same output.** Everything new in
this task is additive: background-discover candidates the moment a segment is
rejected (without slowing the main pass), verify them only after the main pass
ends, insert any recovered segment back in chronological order, keep all
candidate data on disk (not in RAM), and keep candidates out of the final JSON
entirely (preview-only, via a separate endpoint). If any part of this would
require changing the existing matching or first-pass VLM logic to work, stop
and ask rather than modifying it.
