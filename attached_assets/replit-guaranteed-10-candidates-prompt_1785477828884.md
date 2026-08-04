# Task: Guarantee at least 10 real alternate candidates per short-clip range, sourced from the whole movie — not just the top-8 pool

## Confirmed root cause (from code, not a guess)

Every short-clip scene chunk is already scanned against the **entire movie**
during the initial matching pass, in `groundMatchedSegmentsChunked()`
(`server/matching-engine.ts`, the chunked scan loop around line 1913). For
each seed position `si`, `hashSimFastCross(shortSet, si, chunkSet, localMi)`
computes a similarity score against **every frame in every chunk of the
movie** — the full-movie comparison already happens. The problem is what
happens to those scores afterward:

```ts
// Prune to top MAX_SEED_CANDIDATES periodically to cap list growth
if (list.length > MAX_SEED_CANDIDATES * 4) {
  list.sort((a, b) => b.sim - a.sim);
  list.splice(MAX_SEED_CANDIDATES * 2);
}
...
// Final sort + trim
for (const [, list] of allCands) {
  list.sort((a, b) => b.sim - a.sim);
  if (list.length > MAX_SEED_CANDIDATES) list.splice(MAX_SEED_CANDIDATES); // MAX_SEED_CANDIDATES = 8
}
```

Only the **top 8** scored candidates per seed survive into `candidatePool`.
Everything else computed during the full-movie scan is thrown away. Then in
`server/vlm-segment-resolver.ts`, when the VLM rejects a candidate, the
retry loop asks `getAlternateCandidatesForRange(candidatePool, ...)` for
something else to try — but if `candidatePool` only had 1–2 entries for that
range to begin with (or they all get used up across a few rejections), there
is nothing left, even though the movie was already fully scanned and had
more (lower-scoring, but real and previously-discarded) matches available the
whole time. That's why segments are dropping after exactly 1 VLM attempt
instead of using the 10 attempts `VLM_MAX_ATTEMPTS` allows for.

## What you're asking for, restated precisely

Every scene chunk's every seed range must end up with **at least 10 distinct
candidate movie timestamps** available for the VLM retry loop to try, sourced
from the real full-movie similarity scan that already runs today — not
invented, not padded with duplicates, not a different/weaker algorithm.
If the movie genuinely doesn't contain 10 distinct plausible matches for a
given short-clip range (rare, but possible for a very short or generic
clip), give as many as were actually found — never fabricate candidates that
don't correspond to real scan results.

Verification must stay fully sequential and non-blocking: as soon as a
candidate is available, send it to the VLM immediately and keep going through
the up-to-10 candidates one at a time until one is accepted or all are
exhausted — never pause the pipeline waiting to "collect all 10 first."

## What must NOT change — read carefully

- Do NOT change the scoring formula, the hash comparison
  (`hashSimFastCross`, `hammingN`, `popcount32`), or how `minSimilarity` /
  `fastFloor` are computed. The candidates you're recovering already exist in
  the scan output — you're changing how many of them survive the trim, not
  how they're scored.
- Do NOT change which candidate becomes the *first* one tried, or how
  segments get walked/merged/context-validated into the final `segments`
  list in Pass 1/Pass 2/Pass 3. The ~80%-correct-on-first-pass behavior must
  be bit-for-bit identical on a rerun of a known-good test video pair —
  this task only affects what's available as a fallback *after* a rejection.
- Do NOT change `VLM_MAX_ATTEMPTS`, `VLM_CONFIDENCE_THRESHOLD`, the VLM
  prompt, or `verifySameScene`'s accept/reject logic.
- Do NOT let a low-scoring candidate silently become part of the *accepted*
  final result without VLM approval — every candidate below the original
  `minSimilarity` cutoff, even though now retained for retry purposes, must
  still go through the same VLM verification as any other candidate before
  being accepted. Nothing bypasses VLM.
- Do NOT re-scan the movie a second time to find these extra candidates —
  the data is already produced by the existing scan; this is about keeping
  more of what's already computed, not scanning again.
- Do NOT change the "never re-show an already-rejected movie timestamp"
  behavior — `rejectedMovieTimestamps` / `driftTolerance` exclusion in
  `getAlternateCandidatesForRange` must still apply to this larger pool
  exactly as it does today.
- Do NOT block or pause segment processing to "wait and collect 10
  candidates before starting VLM" — the first (highest-scoring) candidate
  must still be sent to VLM immediately, same as today; only the *fallback*
  behavior on rejection changes, by having more real options already sitting
  there to fall back to instead of coming up empty.

## What to build

### 1. Raise the retained-candidate cap for VLM fallback purposes, without changing what's used for the first pass

In the chunked scan (`groundMatchedSegmentsChunked`), keep `MAX_SEED_CANDIDATES`
(8) as the number used for the actual Pass 1/2/3 walk logic — that stays
exactly as-is so first-pass matching is unaffected. Separately, retain a
**second, larger trimmed list per seed** — e.g. top 10 (or slightly more, to
allow for some being excluded later as duplicates/too-close movie
timestamps) — sourced from the exact same `list` already being scored and
sorted in that loop, just trimmed to a bigger number before being discarded.
This larger list becomes (or feeds into) `candidatePool`, which is what
`getAlternateCandidatesForRange` searches during VLM retries — so the retry
path sees up to 10 real distinct candidates instead of whatever was left
over from the top-8 pass list.

- Do this by widening the existing trim/splice limits for the pool that
  becomes `candidatePool`, not by adding a separate scan.
- `SEED_SEPARATION` (distinct-timestamp deduplication, currently 50
  frames / 2s) must still apply, so the 10 candidates are genuinely 10
  different movie moments, not 10 near-duplicates of the same moment.
- Same treatment applies to the equivalent seed-collection logic around line
  1294 (`topCands = cands.slice(0, MAX_SEED_CANDIDATES)`) and line 1996 if
  those paths also feed into what eventually becomes `candidatePool` for a
  segment — check both `groundMatchedSegments` and
  `groundMatchedSegmentsChunked` since matching-engine.ts has both a
  non-chunked and chunked implementation; whichever one is actually used in
  the current server flow (confirm via `matchVideosFromFiles`) is the one
  that must be fixed; if both are reachable, fix both consistently.

### 2. Confirm the retry loop already does the right thing once supply is fixed

`resolveSegmentsWithVLM`'s existing loop (`server/vlm-segment-resolver.ts`)
already does exactly what's wanted here — try the best candidate, send to
VLM, if rejected pull the next-best remaining candidate from
`getAlternateCandidatesForRange(candidatePool, ...)`, repeat up to
`VLM_MAX_ATTEMPTS`, never blocking on anything. No changes should be needed
here once `candidatePool` genuinely contains up to 10 real options — verify
this is true rather than assuming it, and only touch this file if the loop
itself turns out to have a separate bug beyond candidate supply.

## Acceptance criteria

1. Re-run the same match job that produced the "1 attempt(s) ... dropping"
   log lines. For segments that previously dropped after 1 attempt, confirm
   the log now shows more than 1 attempt (up to 10) before either accepting
   or dropping, and confirm via `candidatePool` inspection/logging that up to
   10 distinct movie timestamps were genuinely available for that range.
2. Confirm the previously-correct ~80% of segments that matched on the first
   pass are byte-for-byte unchanged (same segments, same timestamps, same
   confidence) on a rerun of a known-good test video pair.
3. Confirm processing never pauses to "wait for all 10 candidates to be
   ready" — the VLM call for the first (best) candidate must start exactly
   when it does today; only fallback behavior on rejection changes.
4. Confirm candidates below the original `minSimilarity` threshold that get
   accepted via this fallback path still went through full VLM verification —
   spot check a few accepted segments that came from a lower-ranked
   candidate and confirm a `[VLM]` accept log line exists for them.
5. Report before/after: "VLM verification: 52 → 41 segments" should improve
   (more of the original segments survive verification) without changing any
   of the segments that were already correct.

## Summary of the one rule that matters most

**The movie is already fully scanned for every short-clip range — the fix is
to stop throwing away 90%+ of that scan's results down to just the top 8, and
instead keep up to 10 real, distinct, already-computed candidates available
for the VLM retry loop to fall back on.** No new scanning, no new scoring
algorithm, no change to which candidate is tried first or how the first-pass
walk builds `segments` — only how many already-discovered candidates survive
into `candidatePool` for retries after a rejection. VLM verification must
stay strictly sequential and must never pause waiting to collect candidates
before proceeding.
