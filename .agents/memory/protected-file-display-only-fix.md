---
name: Fixing display bugs inside protected/locked algorithm files
description: Pattern for correcting a cosmetic/display bug inside a file whose decision logic must stay bit-for-bit identical.
---

When a file's accept/reject/merge/scoring decisions are locked (must stay
bit-for-bit identical per explicit user constraint), but it also has a
genuine display-layer bug (e.g. an off-by-one-frame end timestamp), do NOT
fix it inline where the buggy value is first computed if that value also
feeds internal decisions (gap checks, merge thresholds, dedup overlap, etc.).

**Why:** Correcting the value at its source changes the internal decision
logic's inputs too, even though the fix's intent is purely cosmetic — this
risks shifting which segments get merged/dropped/accepted, breaking the
"must stay identical" constraint.

**How to apply:** Let every internal decision keep consuming the original
(uncorrected) values exactly as before. Apply the correction as one final
pass over the *already-decided* output, at the last point before it's
returned to the caller/API/UI — and apply it uniformly to every parallel
output structure that shares the same convention (e.g. both the primary
result list and any candidate pool other passes will later draw from), so
nothing downstream ends up half-corrected.
