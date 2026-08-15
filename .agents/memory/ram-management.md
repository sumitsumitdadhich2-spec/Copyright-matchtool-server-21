---
name: RAM management — pipeline + matching
description: How extraction and matching control RAM; what was changed and why.
---

# RAM Management in Nexus Video Match

## Extraction pipeline (server/pipeline.ts)

**Problem:** Task queue used a hardcoded limit of 100 frames regardless of resolution.
- 1080p = 8.3 MB/frame × 100 = 830 MB (OK)
- 4K    = 33  MB/frame × 100 = 3.3 GB (crashes 8 GB machine)

**Fix:** After ffprobe returns `width × height`, compute dynamic queue limit:
```typescript
const QUEUE_RAM_CAP = 1.5 * 1024 * 1024 * 1024; // 1.5 GB
dynamicQueueLimit = Math.max(4, Math.min(500, Math.floor(QUEUE_RAM_CAP / frameBytes)));
```
Pause ffmpeg when `taskQueue.length >= dynamicQueueLimit || RSS >= RAM_FLUSH_THRESHOLD`.
Resume when queue < half the limit AND RSS < 85% of threshold.

**Already present (pre-existing):**
- NDJSON disk spill of completed fingerprints when RSS > 5.5 GB or 1500 frames accumulated
- ffmpeg stdout backpressure

## Matching engine (server/matching-engine.ts)

**Full-load path (existing):** Streams NDJSON into flat TypedArrays.
- 2-hour movie: ~350 MB total (aFlat ~71 MB + JS objects ~243 MB + tDelta ~33 MB)
- This is already very efficient; fits easily in 8 GB RAM.

**Chunked path (added):** Activated when estimated movie PreSet > 4 GB OR free RAM < estimate + 500 MB headroom.
- Phase 1: `buildMovieLineIndex` — single pass, records byte offset of each NDJSON line (8 bytes/frame).
- Phase 2: Scan movie in 10,000-frame chunks via `loadMovieWindowPreset`; collect seed candidates per short-scene seed position.
- Phase 3: For each seed, load ±3,000-frame window, run bidirectional walk; convert local mi → global mi using `winStart` offset.
- Phase 4: Post-process (merge, dedup, context-validate) — identical to full path.

**Why:** Prevents OOM on very long movies (>12 hours on 8 GB). For 2-hour movies the full path is used automatically.
