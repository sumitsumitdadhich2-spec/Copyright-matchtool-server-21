# Candidate matching engine split (protected main engine)

## Decision
`server/matching-engine.ts` (main first-pass engine, ~80% accuracy, user-validated) is PROTECTED. The candidate-find system runs on a dedicated 1:1 copy: `server/candidate-matching-engine.ts`.

## Why
User explicitly required that tuning the candidate-find system must NEVER risk changing first-pass match results. A shared engine meant any candidate-side tweak (thresholds, drift, walk params) would silently alter the main pass too.

## Wiring (as of the split)
Files importing the CANDIDATE copy (`./candidate-matching-engine`):
- `server/candidate-retry.ts` — manual Retry broader search (`matchVideosFromFiles`)
- `server/candidate-recovery.ts` — alt-candidate pool (`getAlternateCandidatesForRange`)
- `server/deferred-recovery.ts` — type import only

Files that STAY on the main engine (`./matching-engine`) — do not move them:
- `server.ts` — main match pass
- `server/vlm-segment-resolver.ts` — main VLM pass + its in-pass alternate lookup

## Bulletproof passes (candidate engine only, in-memory path)
- Pass 2.5 consensus: PROBE_STEP 2, VOTES_PER_PROBE 10, TOP_OFFSETS 20, accept floor 32 (ADAPTIVE_FLOOR parity).
- Pass 2.7 speed-ratio sweep (`speedSweepRescueSeeds`): tests slopes 0.5–2.0x for chunks consensus can't crack (constant-offset voting is blind to re-timed edits). Fast-cross sweep, frameSim refine, accept floor 32; all anchors always feed the VLM pool.
- Pass 4 (pool-only): every chunk with leftover unmatched frames gets a fresh 3-probe full-movie best-cross scan whose top offsets ALWAYS enter the VLM pool — the retry loop is never empty-handed. Never touches `segments`/`usedShort`.
- Pool-loss fix: Pass 2 banks `chunkAltRaw` into the pool even when the walk fails (previously failed chunks silently lost all scan data).
- Verified: 2x sped-up synthetic clip found at speed 2.01 / conf 99.7; self-match still 100%.

## Rules
- All candidate-system tuning goes in `candidate-matching-engine.ts` ONLY.
- Never edit `matching-engine.ts` for candidate behavior.
- The two files' exported types (`MatchedSegment`, `FPData`, etc.) are structurally identical; TS structural typing keeps cross-module usage compatible. If a type shape ever diverges between the two files, cross-imports will break — keep shared type SHAPES in sync even when tuning constants diverge.
