/**
 * Manual, user-triggered "Retry" flow for a single segment in the preview UI.
 * Purely additive on top of the automatic pipeline — never touches
 * groundMatchedSegments()/matchVideosFromFiles() decision logic, the main
 * VLM pass (vlm-segment-resolver.ts), or the deferred recovery pass
 * (deferred-recovery.ts). Those stay completely untouched; this module only
 * consumes their outputs (candidate-recovery.ts's disk-backed
 * StoredCandidateSet) and reuses the same verification primitives
 * (vlm-verify.ts) and the same hash-matching engine (matching-engine.ts) —
 * no second/different similarity-scoring algorithm.
 *
 * Two modes, chosen automatically based on what's left to try for the
 * segment (never based on click count, since the caller doesn't track that):
 *
 *  - `unchecked_pool`: the segment's already-discovered candidate pool (from
 *    background discovery during the main pass, or a previous broader
 *    search) still has entries VLM never got to. Check those first, in
 *    existing order, before doing anything more expensive.
 *  - `broader_search`: every discovered candidate has already been checked
 *    (accepted/rejected/unverifiable). Re-run the real matching engine
 *    (matchVideosFromFiles) restricted to this short-clip range with a wider
 *    frame drift + relaxed similarity floor, purely to surface movie
 *    locations the original single scan didn't surface. New candidates are
 *    appended to the same on-disk history — nothing already there is ever
 *    deleted, even if this retry fails to find an accept.
 */
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';
import { MatchedSegment, matchVideosFromFiles } from './matching-engine';
import { pickRepresentativeFrames } from './vlm-segment-resolver';
import { extractFrameAsBase64, verifySameScene, VLM_CONFIDENCE_THRESHOLD } from './vlm-verify';
import {
  StoredCandidateSet,
  CandidateCheck,
  readCandidatesFile,
  writeCandidatesFileSync,
} from './candidate-recovery';

/** Cap on how many freshly-discovered candidates a single broader search appends. */
const BROADER_SEARCH_MAX_NEW = 10;
/** Deliberately looser than the app default (82) — this is a fallback pass
 *  whose only job is to surface locations the original scan missed; VLM
 *  verification below is still the real accept/reject gate. */
const BROADER_SEARCH_MIN_SIMILARITY = 40;
const BROADER_SEARCH_MIN_CONSECUTIVE_FRAMES = 3;
/** Wider than the app default (3) so a slightly different edit speed/frame
 *  offset than the original scan assumed can still line up. */
const BROADER_SEARCH_FRAME_DRIFT = 8;
/** Small amount of extra context on each side of the short-clip range so the
 *  scene-chunk splitter has more than a couple of frames to work with. */
const BROADER_SEARCH_PAD_SECONDS = 1.5;
/** Same drift tolerance used elsewhere (getAlternateCandidatesForRange) to
 *  treat two movie timestamps as "the same location". */
const SAME_LOCATION_TOLERANCE = 0.5;

export interface RetrySegmentResult {
  outcome: 'accepted' | 'exhausted';
  mode: 'unchecked_pool' | 'broader_search';
  acceptedCandidateIndex?: number;
  newCandidatesAdded: number;
}

/**
 * Write a temporary fingerprint file containing only the short-clip frames
 * within [start-pad, end+pad], preserving their original timestamps, so a
 * fresh matchVideosFromFiles() call can be scoped to just this range without
 * touching the real per-job fingerprint files on disk.
 */
async function writeFilteredShortFingerprint(
  shortResultPath: string,
  rangeStart: number,
  rangeEnd: number,
): Promise<string | null> {
  const tempPath = path.join(
    os.tmpdir(),
    `retry-broader-${Date.now()}-${Math.random().toString(36).slice(2)}.jsonl`,
  );
  const rl = readline.createInterface({
    input: fs.createReadStream(shortResultPath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });
  const lines: string[] = [];
  for await (const line of rl) {
    if (!line.trim()) continue;
    try {
      const frame = JSON.parse(line);
      if (typeof frame.timestamp === 'number' && frame.timestamp >= rangeStart && frame.timestamp <= rangeEnd) {
        lines.push(line);
      }
    } catch { /* skip corrupt line */ }
  }
  if (lines.length === 0) return null;
  await fs.promises.writeFile(tempPath, lines.join('\n') + '\n');
  return tempPath;
}

/**
 * Re-run the real matching engine restricted to this segment's short-clip
 * range, with a wider drift + relaxed similarity floor, to surface movie
 * locations the original full-clip scan didn't. Reuses matchVideosFromFiles
 * verbatim (same chunked/full-load RAM-safety logic, same scene-chunk
 * scan) — this is not a second scoring algorithm.
 */
async function runBroaderSearch(
  entry: StoredCandidateSet,
  shortResultPath: string,
  movieResultPath: string,
): Promise<MatchedSegment[]> {
  const rangeStart = Math.max(0, entry.shortStart - BROADER_SEARCH_PAD_SECONDS);
  const rangeEnd = entry.shortEnd + BROADER_SEARCH_PAD_SECONDS;

  const tempShortPath = await writeFilteredShortFingerprint(shortResultPath, rangeStart, rangeEnd);
  if (!tempShortPath) return [];

  try {
    const result = await matchVideosFromFiles(tempShortPath, movieResultPath, {
      minSimilarity: BROADER_SEARCH_MIN_SIMILARITY,
      minConsecutiveFrames: BROADER_SEARCH_MIN_CONSECUTIVE_FRAMES,
      frameDrift: BROADER_SEARCH_FRAME_DRIFT,
    });

    const pool = [...(result.segments || []), ...(result.candidatePool || [])];
    const overlapping = pool.filter(seg =>
      Math.min(seg.shortEnd, entry.shortEnd) - Math.max(seg.shortStart, entry.shortStart) > 0.15,
    );

    const deduped: MatchedSegment[] = [];
    for (const seg of overlapping) {
      if (!deduped.some(d => Math.abs(d.movieStart - seg.movieStart) <= SAME_LOCATION_TOLERANCE)) {
        deduped.push(seg);
      }
    }
    deduped.sort((a, b) => b.confidence - a.confidence);
    return deduped;
  } finally {
    fs.promises.unlink(tempShortPath).catch(() => { /* best-effort cleanup */ });
  }
}

/** Verify one candidate via the exact same VLM call/threshold everything
 *  else uses, mutating it in place and returning its verdict. */
async function checkCandidate(
  candidate: CandidateCheck,
  shortVideoPath: string,
  movieVideoPath: string,
): Promise<void> {
  const { shortTime, movieTime } = pickRepresentativeFrames(candidate.segment);
  try {
    const [shortFrame, movieFrame] = await Promise.all([
      extractFrameAsBase64(shortVideoPath, shortTime),
      extractFrameAsBase64(movieVideoPath, movieTime),
    ]);
    const result = await verifySameScene(shortFrame, movieFrame);
    if (result === null) {
      candidate.checked = true;
      candidate.verdict = 'unverifiable';
    } else if (result.same && result.confidencePct >= VLM_CONFIDENCE_THRESHOLD) {
      candidate.checked = true;
      candidate.verdict = 'accepted';
      candidate.confidencePct = result.confidencePct;
    } else {
      candidate.checked = true;
      candidate.verdict = 'rejected';
      candidate.confidencePct = result.confidencePct;
    }
  } catch {
    candidate.checked = true;
    candidate.verdict = 'unverifiable';
  }
}

/**
 * Run one Retry click for a segment. Determines internally whether this is
 * "search the unchecked pool" or "broader search" based on whether any
 * unchecked candidates remain — never based on a click counter the caller
 * would have to track. Persists every state change to disk incrementally
 * (never held in memory beyond this single call), so history survives a
 * server restart mid-retry and RAM never grows with retry count.
 */
export async function retrySegmentCandidates(
  uploadDir: string,
  matchJobId: string,
  segmentIndex: number,
  shortVideoPath: string,
  movieVideoPath: string,
  shortResultPath: string,
  movieResultPath: string,
): Promise<RetrySegmentResult> {
  const entry = readCandidatesFile(uploadDir, matchJobId, segmentIndex);
  if (!entry) throw new Error('No candidate history for this segment');

  let uncheckedIdxs = entry.candidates
    .map((c, i) => (c.checked ? -1 : i))
    .filter(i => i !== -1);

  let mode: 'unchecked_pool' | 'broader_search';
  let newCandidatesAdded = 0;

  if (uncheckedIdxs.length > 0) {
    mode = 'unchecked_pool';
  } else {
    mode = 'broader_search';
    const existingTimestamps = entry.candidates.map(c => c.segment.movieStart);
    const fresh = (await runBroaderSearch(entry, shortResultPath, movieResultPath))
      .filter(seg => !existingTimestamps.some(t => Math.abs(t - seg.movieStart) <= SAME_LOCATION_TOLERANCE))
      .slice(0, BROADER_SEARCH_MAX_NEW);

    newCandidatesAdded = fresh.length;
    for (const seg of fresh) {
      entry.candidates.push({ segment: seg, checked: false });
      uncheckedIdxs.push(entry.candidates.length - 1);
    }
    // Persist the newly-discovered (not-yet-checked) candidates immediately,
    // even before verification runs, so the visible history grows right
    // away and nothing is lost if verification below throws.
    writeCandidatesFileSync(uploadDir, matchJobId, segmentIndex, entry);
  }

  let acceptedIdx: number | undefined;
  for (const idx of uncheckedIdxs) {
    await checkCandidate(entry.candidates[idx], shortVideoPath, movieVideoPath);
    writeCandidatesFileSync(uploadDir, matchJobId, segmentIndex, entry);
    if (entry.candidates[idx].verdict === 'accepted') {
      acceptedIdx = idx;
      break;
    }
  }

  if (acceptedIdx !== undefined) {
    // Supersede — never delete. The previous recoveredCandidateIndex simply
    // stops being marked "★ Used"; its entry stays in `candidates` exactly
    // where it already was.
    entry.recoveredCandidateIndex = acceptedIdx;
    entry.dropped = false;
    writeCandidatesFileSync(uploadDir, matchJobId, segmentIndex, entry);
    return { outcome: 'accepted', mode, acceptedCandidateIndex: acceptedIdx, newCandidatesAdded };
  }

  return { outcome: 'exhausted', mode, newCandidatesAdded };
}
