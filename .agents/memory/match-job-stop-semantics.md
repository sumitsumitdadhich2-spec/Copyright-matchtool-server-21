---
name: Match job "Stop" cancellation semantics
description: Why stopping a background /api/match job doesn't preempt the in-flight hash scan or VLM verification loop.
---

Stopping a background match job (`/api/match-stop/:id`) flips the job's status
to `stopped` immediately and discards any result that arrives afterward — but
it does not preempt the CPU-bound work already in flight inside
`matchVideosFromFiles()` / `resolveSegmentsWithVLM()`.

**Why:** `server/matching-engine.ts` and the VLM verification files
(`server/vlm-verify.ts`, `server/vlm-segment-resolver.ts`) are treated as
locked/protected — their internals (algorithm, thresholds, retries, prompts)
must never be modified, since any change there risks altering match accuracy
or timing. Adding a real abort-checked early-exit inside those loops would
require touching those files, so it was intentionally left out.

**How to apply:** If a future task asks for "instant"/mid-algorithm
cancellation of matching or VLM verification, that requires deliberately
relaxing the "don't touch matching-engine/VLM files" constraint — flag this
tradeoff to the user rather than assuming it's fine to add abort checks there.

**Update:** "Protected" means don't touch the accept/reject algorithm, scoring,
thresholds, retries, or prompts — it does not forbid small additive hooks. A
background-candidate-recovery feature added one optional callback param to
`resolveSegmentsWithVLM` (fired only at final-drop time, never awaited) and
exported an already-private helper (`pickRepresentativeFrames`), with zero
change to accept/reject behavior or timing. That kind of purely-additive,
optional-callback extension is safe; changing existing call signatures'
required args, defaults, or internal control flow is not.
