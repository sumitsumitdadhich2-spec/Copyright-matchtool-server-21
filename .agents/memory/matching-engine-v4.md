---
name: Matching Engine v4
description: Key changes made to the matching engine and UI in v4, and what the parameters do.
---

# Matching Engine v4 — Key Facts

## Algorithm tuning (server/matching-engine.ts)
- `WALK_MIN_SIM` lowered 56 → 50: catches lightly-encoded / low-contrast matches
- `MAX_SEED_CANDIDATES` raised 6 → 8: tries more candidate positions per short frame
- Fast-floor margin: -20 (was -18): admits more candidates in seed scan
- `frameDrift` parameter (default 3): added to per-step search window (`WALK_LOOK_AHEAD + frameDrift`)
  - Expands the window for clips with minor frame drops/duplicates/speed drift
  - User-controllable 0–10 frames
- `FrameDetail` exported type: per-segment best-frame breakdown (structureSim, colorSim, skinSim, detailSim, cropRegion, movieHash, shortHash)

## Server API (server.ts)
- `/api/match` now accepts `frameDrift` body param (int 0–15, default 3)
- Confidence threshold validation changed: `>= 20` (was `>= 40`) to allow 20 % floor
- `/api/sanity-test` POST endpoint: tests 10 deterministic 16×16 frames for hash determinism
  - Main thread: uses fake ImageData (no canvas) with `computeHashAndFeatures` directly
  - Worker: spawns server/worker.ts via worker_threads; gracefully handles canvas unavailability
  - Returns: `{pass, totalFrames, workerAvailable, results[]}`

## UI additions (src/App.tsx)
- Confidence slider min: 20 % (was 60 %)
- New "Sequence Frame Drift" slider: 0–10, default 3, shown as "±N frames"
- Settings summary now shows: "Confidence ≥82% · Min duration 0.5s · Drift ±3f"
- "Worker Accuracy Calibration" collapsible section with Run Sanity Test button
- In preview panel (after match quality timeline):
  - Live Perceptual Fingerprint: 16×16 bit grids for movie hash + short hash
  - Match Integrity Similarity: gauge bar with threshold marker
  - Matched Crop Region: badge showing best crop variant
  - Double-Check Verification Checklist: 4 bar rows (Structure 84%, Colors, Human/Character, Edges/Details)
  - Fingerprint Bitstream: 48-bit prefix of each hash

**Why:** WALK_MIN_SIM at 50 is safer than lower values — the adaptive floor catches really weak matches; raising it would miss clips with encoding artifacts. The frameDrift expansion helps with re-encoded clips where frame indices don't align perfectly.
