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

## Rules
- All candidate-system tuning goes in `candidate-matching-engine.ts` ONLY.
- Never edit `matching-engine.ts` for candidate behavior.
- The two files' exported types (`MatchedSegment`, `FPData`, etc.) are structurally identical; TS structural typing keeps cross-module usage compatible. If a type shape ever diverges between the two files, cross-imports will break — keep shared type SHAPES in sync even when tuning constants diverge.
