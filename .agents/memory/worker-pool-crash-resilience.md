---
name: Worker pool crash resilience
description: A worker_threads pool must track which in-flight task each worker owns, or a worker crash orphans that task forever and hangs the whole job.
---

## The Rule
When a worker in a `worker_threads` pool dies (native-addon fault, OOM, etc.) mid-task, explicitly reject only the task(s) that were in flight on that specific worker, then spawn a replacement to keep pool size steady. Track assignment with a `Map<Worker, Set<taskId>>` populated at dispatch time and cleared on completion/death.

**Why:** The original extraction pipeline's `worker.on('error', ...)` handler removed the dead worker from the pool but never touched the pending task map — the specific task(s) that worker was holding never resolved or rejected. Since job completion was gated on a "processed count reached total" check, one dead worker meant that count could never be reached, and the job hung in "processing" forever with no error surfaced. This risk scales with job length/frame count (a 2-hour movie has ~180k frames and many hours of sustained native-canvas work — far more chances to hit a rare worker crash than a short clip), making it a plausible cause of "large video never finishes" reports.

**How to apply:** Any worker-pool-based pipeline (frame hashing, image processing, etc.) needs this pattern. Also listen for `worker.on('exit', code)` with `code !== 0` as a second signal (crashes don't always emit `'error'`), guarded so 'error' and 'exit' firing for the same death don't double-process.

## Related: process-level crash isolation
For a server that runs multiple long (multi-hour) background jobs concurrently, add `process.on('uncaughtException', ...)` and `process.on('unhandledRejection', ...)` handlers that log loudly but do NOT exit. Without this, Node's default behavior is to crash the entire process on any single unhandled error anywhere — silently killing every OTHER in-flight job too, not just the one that triggered it. This is a last-resort net only; each job should still have its own try/catch that marks itself failed on its own errors.

## Confirmed already-solid: job persistence on restart
If a job's on-disk meta file still says "processing" when the server rebuilds job history at startup, the rebuild path should convert it to "failed" with an explanatory error (e.g. "Interrupted by server restart") rather than silently dropping it or leaving a phantom in-progress entry. This project's match-job rebuild already does this correctly — worth preserving as the pattern for any future job type.
