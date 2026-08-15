---
name: VLM concurrency deadlock audit (unreproduced)
description: A "VLM verification never starts" bug report blamed the Semaphore/retry-backoff work in server/vlm-verify.ts; how it was investigated and why it stayed unreproduced.
---

A bug report claimed VLM verification produced zero `[VLM]` log lines (not even
cache-reset) after `[Match ...] Done`, blaming the `Semaphore` /
`fetchWithRetry` / concurrency-capping work in `server/vlm-verify.ts` as a
permit-leak or deadlock, even on a tiny 6-segment baseline, against a
confirmed-reachable real VLM server.

**Testing technique that worked well:** don't stop at the existing mocked-
`global.fetch` unit tests (they can pass while still hiding a real-socket bug).
Also stand up a genuine `http.createServer` in the *same* process (no ad hoc
background process — those die between tool calls in this sandbox) and hit it
with real `fetch()`/loopback sockets, including: realistic latency + jitter,
a transient-failure rate, batch-boundary `resetVlmCache` calls interleaved
with in-flight `verifySameScene` calls, and a deliberate "black hole" endpoint
(accepts the connection, never responds) to confirm `AbortController` timeouts
actually bound worst-case wait time.

**Result:** every scenario resolved correctly — concurrency cap held exactly,
retries/backoff worked, no permit leaks, no permanent hangs (even the black-
hole case resolved in ~27s via timeout+retry, never forever). The Semaphore's
acquire/release pairing was also read line-by-line with no leak found on any
path (`try/finally` releases in both `verifySameScene` and `eraseSlot`).

**Conclusion:** could not reproduce the reported hang from code alone. Likely
candidates if it resurfaces: a real proxy/tunnel-specific connection cap (e.g.
ngrok) rather than an app bug, or something only visible in real production
logs. Follow-up task filed asking for real AWS-host logs / a way to hit the
actual failing endpoint before assuming a specific code fix.

**Why this matters:** don't re-derive this whole audit from scratch if the
same report resurfaces — check real production logs/env first instead of
re-reading the Semaphore for the Nth time; the logic has already been
stress-tested from multiple angles and held up.
