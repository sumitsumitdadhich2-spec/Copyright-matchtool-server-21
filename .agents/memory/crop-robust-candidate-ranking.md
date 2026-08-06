# Crop-robust embedding ranking (candidate system only)

## Problem
Shorts are usually a 9:16 vertical crop cut from ANY horizontal position
(left/center/right) of the 16:9 movie frame. Full-frame comparison sees only
~40% shared pixels, so the correct movie location often ranked below wrong
locations and the VLM burned attempts on (or accepted) wrong candidates —
user reported ~50% VLM error on cropped shorts.

## Fix
`server/candidate-embedding-rank.ts` — for each untried candidate, embed the
short mid-frame vs FOUR movie-frame variants (full + left/center/right 9:16
ffmpeg crops: `crop=w=min(iw\,ih*9/16):h=ih:x=...:y=0`) with the same lazy
CLIP model embedding-gate uses (`embedFrameB64`/`cosineSimilarity` exported
additively from embedding-gate.ts). Candidate score = MAX cosine across
variants; untried candidates are VLM-verified best-score-first.

## Wiring (candidate side ONLY — main pass untouched)
- `candidate-retry.ts`: ranks `uncheckedIdxs` before the verification loop.
- `deferred-recovery.ts`: ranks unchecked candidate order in
  `recoverOneSegment` (loop converted from index scan to ranked list).
- NEVER import this from matching-engine.ts / server.ts main pass /
  vlm-segment-resolver.ts main pass.

## Guarantees
- Reorder-only: never adds/removes/accepts/rejects candidates itself.
- Fail-safe: returns null (keep old order) if CLIP can't load or nothing
  could be scored; per-candidate failures go to the END, never dropped.
- All CPU, in-app; CLIP weights (~150MB) auto-download on first use — nothing
  to install on the Colab/ngrok VLM side.

## VLM prompt (vlm-verify.ts, text-only change)
Prompt now leads with a CRITICAL vertical-crop instruction (scan the whole
movie frame left-to-right, never reject for extra side content) and a
counterweight anti-false-positive rule (different moment/shot of the same
movie is NOT the same scene). Control flow/verdict parsing unchanged.

## Verified
`test_crop_rank.mts`: synthetic mandelbrot long video, RIGHT-crop 9:16 short
from t=10s; ranker put correct candidate first (sim 0.997, variant=right)
over wrong t=25 candidate (0.949). Run with ffmpeg on PATH
(`FFMPEG_BIN` override supported by the test itself).
