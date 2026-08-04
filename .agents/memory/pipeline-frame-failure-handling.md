---
name: Extraction pipeline frame-failure handling
description: Why a rejected per-frame promise in server/pipeline.ts must be tracked separately from success, not silently swallowed.
---

# Frame-failure handling in the extraction pipeline

`server/pipeline.ts` processes video frames concurrently (worker pool) and
flushes completed fingerprints to disk in contiguous order. Before this was
fixed, a rejected per-frame promise (worker crash, decode error, missing
native addon) was not distinguished from a successful one.

**Two compounding failure modes this caused:**
1. If **every** frame failed (e.g. a broken native dependency killing every
   worker on load), the job still resolved as `"completed"` with a 0-byte /
   empty result file — a silent total failure with no error surfaced
   anywhere.
2. If **one** frame failed mid-stream, the flush logic's contiguous-walk
   (`fingerprints.has(nextIndex)`) got permanently stuck at that index —
   every later successfully-processed frame after it would never flush to
   disk, silently truncating output and leaking memory for the rest of the
   job.

**Fix pattern:** track a `successCount` and a `failedFrameIndices` set
alongside the existing fingerprints map. The flush walk steps past known
failed indices instead of blocking on them; the completion handler rejects
the job explicitly when 100% of attempted frames failed. Partial failure
still behaves exactly as before (tolerated, no behavior change for the
normal/working path).

**Why this matters beyond this one bug:** any pipeline stage that flushes
"in order" based on a contiguous index/counter needs an explicit skip-list
for permanently-missing indices, or a single failure anywhere upstream
silently caps everything downstream of it. Watch for the same shape
(contiguous-walk + a map keyed by index) elsewhere in this codebase.

**Diagnostic signature to recognize this class of bug fast:** a job reports
`status: "completed"` but the output file is missing, 0 bytes, or
suspiciously truncated, with no error anywhere — check whether the
per-item failure path is silently counted as success before assuming the
matching/business logic is wrong.
