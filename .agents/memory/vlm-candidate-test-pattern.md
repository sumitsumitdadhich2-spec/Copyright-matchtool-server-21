---
name: In-process VLM/candidate testing pattern
description: How to exercise resolveSegmentsWithVLM / deferred recovery / candidate history end-to-end without a real GPU VLM server, plus sandbox gotchas hit while building this.
---

# Testing the VLM/candidate subsystem without a real VLM server

No real Qwen2.5-VL/vLLM server is reachable in this environment. Standing up
a second ad hoc HTTP server (mock VLM) on another port, plus a second app
instance pointed at it, is the obvious approach but is **unreliable in this
sandbox**: a background listener started in one ShellExec call can become
unreachable (`curl` connection-refused) from a *separate* subsequent
ShellExec call even though `ps aux` still shows it alive. The configured
workflow's port (5000) is special-cased and stays reachable across calls;
ad hoc ports are not reliable across calls. Prefer a single script over a
second server.

**Working pattern:** a standalone `tsx` script (repo convention:
`test_*.mts` in the project root) that:
1. Calls the real `matchVideosFromFiles(...)` against real fingerprint
   result files to get a real `MatchedSegment` + real `candidatePool` (no
   fabricated data).
2. Monkey-patches `global.fetch` to intercept calls to the VLM endpoint and
   return a scripted `{same, confidence}` verdict synchronously, while
   passing through `/v1/models` and `/slots/:id` as plain 200 OKs so
   `isVlmAvailable()` / `resetVlmCache()` behave normally.
3. Calls the real `resolveSegmentsWithVLM` / `runDeferredRecoveryPass` /
   `buildCandidateHistoryEntry` with real local ffmpeg frame extraction
   (real short/movie video files) — only the verdict is fake, the rest of
   the pipeline is exercised for real.

This validated the full accept / reject-then-drop / deferred-recovery-accept
/ reject-then-retry-then-accept matrix in one process, no second port
needed.

**Two sharp edges hit while building this:**

- **ES module import hoisting breaks env-var setup.** Static `import`
  statements execute before any other top-level code in the *same* file,
  regardless of source order. A `process.env.VLM_ENDPOINT_URL = '...'` line
  written above the imports still runs *after* `server/vlm-verify.ts` (a
  transitive import) has already evaluated its module-level
  `export const VLM_ENDPOINT_URL = process.env.VLM_ENDPOINT_URL || '...'`
  — the assignment is silently too late. Fix: set `VLM_ENDPOINT_URL` /
  `VLM_MAX_ATTEMPTS` / `VLM_CONCURRENCY` etc. on the shell command line
  that launches the script (`VLM_ENDPOINT_URL=... npx tsx test_x.mts`), or
  use a dynamic `await import(...)` after the assignment instead of a
  static `import`.
- **`/tmp` is not reliably persistent across a long session** — a file
  written to `/tmp` earlier in a session was later found gone (`No such
  file or directory`) with no deliberate deletion. Regenerate `/tmp` test
  fixtures immediately before the command that consumes them rather than
  relying on one created many turns earlier.

**Also:** never `pkill -f "tsx server.ts"` to kill a stray secondary test
instance — that pattern also matches the main dev workflow's process (same
command line) and will kill the real app. Capture the PID at spawn time
(`$!` or `ps` right after starting it) and kill that specific PID instead.

A reusable regression script for this exact matrix lives at
`test_vlm_candidate_mock_e2e.mts` in the project root.
