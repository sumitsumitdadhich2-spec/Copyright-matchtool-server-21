# Task: Batch VLM verification in groups of 48 + reset VLM server cache between batches AND between videos

## Problem

The VLM server (llama.cpp, running on a remote GPU machine, reached via
`VLM_ENDPOINT_URL`) accumulates GPU RAM usage across requests within a single
match job. This happens because llama.cpp's KV-cache / prompt-cache for each of
its 4 parallel slots is not released between individual VLM verification calls —
only a full server restart fully clears it. Confirmed symptom: GPU RAM climbs
continuously through a long run (e.g. 500+ segments in one movie) and is never
observed to reset mid-run — it only appears to "restart clean" when a completely
new video begins, because in practice the server has been getting bounced between
runs. If a single match job has many segments to verify (500+ is common for a
full movie), GPU RAM can climb high enough to risk an out-of-memory failure
partway through, since nothing currently forces a mid-job cache clear.

## Solution overview

Two related fixes, both using the same mechanism (llama.cpp's slot-erase
endpoint), triggered at two different points:

1. **Between batches within one match job**: instead of sending all VLM
   verification requests for a match job in one continuous stream, process them
   in **fixed batches of 48 segments**. After each batch of 48 completes (all 48
   results received — not just started), send a reset call to the VLM server to
   erase its slots' KV-cache, then continue with the next batch of 48. Repeat
   until all segments in the job are verified. 48 was chosen deliberately because
   the server runs 4 parallel slots (`n_slots = 4` from server startup logs) — 48
   is a clean multiple of 4 (12 full rounds), so every round is a complete
   4-segment round with no partial/orphaned round at the batch boundary. This
   guarantees all 4 slots are idle at the moment of reset, so erasing them cannot
   cancel or corrupt an in-flight verification.

2. **Between videos**: this part already exists conceptually (confirmed by the
   observed behavior that GPU RAM only drops when a genuinely new video/model
   session starts) but must be made an explicit, guaranteed step rather than an
   incidental side effect: when a match job for one video pair finishes (or a
   new movie/short-clip pair is loaded for a new match job), send the same
   slot-erase reset call before starting VLM verification on the new pair.

## Non-negotiable constraints — READ CAREFULLY

**Do NOT reduce verification speed or throughput in any way.** This is a memory-
management change, not an accuracy or performance change.

Specifically forbidden:
- Do NOT reduce the number of parallel slots used per batch (keep sending
  requests to all 4 slots in parallel within a batch of 48, exactly as today).
- Do NOT add any artificial delay, sleep, or throttling beyond the reset call
  itself between batches.
- Do NOT change VLM_MAX_ATTEMPTS, confidence thresholds, timeout values, the
  VLM prompt content, or any part of `groundMatchedSegments()` /
  `matchVideosFromFiles()` in `server/matching-engine.ts`.
- Do NOT change how `resolveSegmentsWithVLM` / `vlm-segment-resolver.ts` decides
  which candidate to try next on rejection — that logic (described in an earlier
  task) is unchanged; batching only affects *when* a reset is issued, not *how*
  candidates are chosen or verified.
- The reset call itself must be fast (erasing an idle slot is a cheap
  operation per llama.cpp docs) — if in testing it turns out to measurably slow
  down the pipeline, stop and report that instead of leaving it in silently.
- Do NOT change the fallback-to-hash-only-segments behavior when VLM is
  unset/unreachable/fails — unchanged as before.

## What to build

### 1. Reset helper function

Add a small helper (e.g. in `server/vlm-verify.ts` or a new
`server/vlm-reset.ts`) that calls the VLM server's slot-erase endpoint for each
of its slots:

```
POST {VLM_BASE_URL}/slots/{slot_id}?action=erase
Content-Type: application/json
Body: {}
```

- `VLM_BASE_URL` is the same host/port as `VLM_ENDPOINT_URL` but without the
  `/v1/chat/completions` path suffix — derive it by stripping that suffix, don't
  hardcode a separate env var.
- Loop over slot ids `0` through `3` (4 slots, matching `n_slots = 4` seen in
  server startup logs) and erase each one. Do this with a short timeout (a few
  seconds) and log a warning if any erase call fails, but do NOT abort the match
  job if a reset call fails — treat it as best-effort cache hygiene, not a
  correctness requirement. A failed reset should never surface as a job failure
  to the user.
- Log a single line per reset event, e.g.
  `[VLM] Cache reset after batch (segments 1-48)` or
  `[VLM] Cache reset for new video pair` — reuse whatever logging convention
  `vlm-verify.ts` already uses (see the existing `[VLM] ...` log lines in that
  file).

### 2. Batch the verification loop by 48

In `server/vlm-segment-resolver.ts` (or wherever the loop currently issues VLM
verification calls for all segments of a match job back-to-back), restructure it
to process segments in **groups of 48**:

- Keep the existing per-segment retry/candidate-selection logic and existing
  parallelism across the 4 slots exactly as it is today — batching only adds a
  boundary every 48 segments, it does not change how work is distributed to
  slots within a batch.
- After all 48 segments in a batch have a final result (accepted, or exhausted
  all VLM_MAX_ATTEMPTS and dropped) — i.e. after confirming no in-flight
  requests remain for that batch — call the reset helper from step 1.
- Then continue to the next batch of up to 48 remaining segments, and repeat
  until the job's segment list is exhausted. The final batch may have fewer
  than 48 segments (whatever remains) — still reset after it completes, same as
  any other batch.
- This batching is purely about grouping requests and inserting a reset point —
  it must not change the order segments are processed in, which candidate is
  picked for each, or the final accept/reject outcome for any segment compared
  to today's unbatched behavior.

### 3. Reset between videos / match jobs

Wherever a new match job begins loading a new movie+short-clip pair for VLM
verification (the start of `resolveSegmentsWithVLM` or equivalent), call the
same reset helper once, before the first batch of that job starts. This
guarantees each new video pair starts with genuinely clean VLM server state,
matching the currently-observed-but-incidental "clean start on new video"
behavior — making it an explicit guarantee instead of a side effect.

## Acceptance criteria

1. Run a match job with a large number of segments (e.g. simulate 100+
   candidate segments, or use a long movie known to produce 100+ from earlier
   testing). Confirm via server logs that a `[VLM] Cache reset after batch`
   message appears roughly every 48 segments, and confirm (via the Colab GPU RAM
   graph or `nvidia-smi` on the VLM host) that GPU RAM after each reset drops
   back down rather than climbing indefinitely across the whole job.
2. Run two match jobs back-to-back (different video pairs) in the same server
   session without restarting anything manually. Confirm a
   `[VLM] Cache reset for new video pair` log appears at the start of the second
   job, and GPU RAM at the start of job 2 is back near baseline rather than
   continuing to climb from job 1.
3. For a fixed pair of test videos used in earlier testing, segment counts,
   confidence percentages, and total wall-clock VLM verification time must be
   the same (within normal run-to-run noise) as before this change — prove this
   with a before/after timing comparison. Any measurable slowdown is a failed
   implementation.
4. If a reset call to the VLM server fails (network hiccup, VLM temporarily
   unreachable), the match job must still complete normally using whatever
   results it already has — a failed reset is logged as a warning, never
   surfaced to the user as a job error.
5. Existing VLM fallback-to-hash-only-segments behavior (VLM unset/unreachable/
   fails entirely) remains unchanged and still verified working.

## Summary of the one rule that matters most

**Change only when a cache-reset call is sent (every 48 segments, and at the
start of each new video pair's VLM verification) — never change what is
verified, in what order, with what parameters, or how many slots/workers do the
verification work in parallel.** If unsure whether a change affects speed or
correctness, don't make it — ask first instead of guessing.
