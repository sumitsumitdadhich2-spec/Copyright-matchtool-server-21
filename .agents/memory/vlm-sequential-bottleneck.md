---
name: VLM verification concurrency
description: Why the VLM scene-verification passes were sequential and how they were made concurrent without changing verdicts.
---

`resolveSegmentsWithVLM` (main pass) and `runDeferredRecoveryPass` (recovery
pass) originally verified one segment at a time, each with up to
`VLM_MAX_ATTEMPTS` retries × a 20s call timeout. For jobs with 100+ segments
this could take hours, during which the UI looked "stuck" and concurrent
`/api/match-status` polls could read as "Failed to fetch".

**Why it was safe to parallelize:** each segment's retry loop only reads its
own bounds and the shared, read-only candidate pool — no segment's outcome
depends on another's. The self-hosted vLLM/llama.cpp endpoint exposes a fixed
number of independent KV-cache slots (`VLM_NUM_SLOTS`, currently 4), so that
slot count is the real concurrency ceiling the server was provisioned for.

**How it was applied:** added `VLM_CONCURRENCY` (env-overridable, defaults to
the slot count) and ran segments through a worker-pool (cursor-based, N
workers pulling the next index) instead of a `for` loop — same per-segment
logic, same verdicts, only wall-clock time changes. The main pass's
batch-based VLM cache-reset boundary was preserved by pooling within each
fixed-size batch and resetting once the whole batch settles, same as before.
