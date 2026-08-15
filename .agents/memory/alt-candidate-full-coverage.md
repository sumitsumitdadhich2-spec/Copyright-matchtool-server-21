---
name: VLM alt-candidate full-coverage walk
description: Why matching-engine.ts has two separate walk functions (walkOneDir vs walkFullCoverage) and what guarantee the second one exists to preserve.
---

The VLM-retry alternate-candidate pool (`buildAltCandidate`, `buildAltCandidatesForChunk`,
and their chunked-path siblings `buildAltCandidateChunked` /
`buildAltCandidatesForChunkWindowed`, all in `server/matching-engine.ts`) must build
each candidate via a **forced, full-coverage walk** that never stops early and
never reads/writes the shared `usedShort` bitmap — kept fully isolated from
`walkOneDir`/`buildSegment`, which back the protected first-pass (Pass 1/2/3)
matching and intentionally stop early on weak similarity.

**Why:** the original alt-candidate builder produced a single-frame stub
(`movieStart === movieEnd`, `frameCount: 1`) for every alternate. That stub's
confidence came from an ungated per-frame scan, so it could out-rank a real
multi-frame candidate in the confidence sort — surfacing as a "correct
alignment found, but the reported segment is only 1-2 frames" bug after a
segment was rejected by VLM verification. Reusing `walkOneDir` directly isn't
an option either: it exits as soon as similarity drops or misses run out, and
it treats `usedShort` as authoritative (breaks immediately if the next frame
is already claimed by the accepted segment) — both would still collapse
alt-candidates to a handful of frames.

**How to apply:** any alternate/candidate a VLM-rejection path offers for a
short-clip range of N frames must itself span N frames (weak/forced frames
still report their real similarity so confidence isn't inflated — only
genuinely good matches steer the running slope estimate). If you touch
candidate-pool construction again, keep it calling the full-coverage walk
(or an equivalent that shares its no-early-exit, no-`usedShort` guarantees)
rather than the first-pass walk, so Pass 1/2/3 behavior (~80% baseline
accuracy, must stay bit-for-bit identical per explicit user constraint) is
never put at risk by fixes scoped to the ~50%-accurate fallback path.
