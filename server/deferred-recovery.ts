/**
 * Deferred verification pass: runs strictly AFTER the main VLM verification
 * pass (resolveSegmentsWithVLM) has finished with every segment. Gives each
 * segment that was ultimately dropped one more chance — its 10
 * background-discovered candidates (written to disk by
 * server/candidate-recovery.ts the instant it was dropped) are checked with
 * VLM one at a time, in original-rejection order, sequentially.
 *
 * Reuses the exact same VLM call/verdict logic as the main pass
 * (verifySameScene + VLM_CONFIDENCE_THRESHOLD from server/vlm-verify.ts —
 * untouched) and the same representative-frame picking helper
 * (pickRepresentativeFrames from server/vlm-segment-resolver.ts — untouched).
 * Never modifies matching-engine.ts / vlm-verify.ts / vlm-segment-resolver.ts
 * decision logic.
 */
import { MatchedSegment } from './matching-engine';
import { pickVerificationFramePairs } from './vlm-segment-resolver';
import { extractFrameAsBase64, verifySameSceneMulti, VLM_CONFIDENCE_THRESHOLD, VLM_CONCURRENCY } from './vlm-verify';
import {
  listCandidateFilesForJob,
  readCandidatesFile,
  writeCandidatesFileSync,
  StoredCandidateSet,
} from './candidate-recovery';

export interface DeferredRecoveryProgress {
  segmentIndex: number;
  totalRejected: number;
  candidateAttempt: number;
  totalCandidates: number;
  verdict: 'accepted' | 'rejected' | 'unverifiable' | 'exhausted';
}

/**
 * Runs the deferred pass for one match job and returns any segments that
 * were recovered (accepted on one of their background candidates). Callers
 * are responsible for merging these back into the final segments array and
 * re-sorting by shortStart — this function never touches the primary result.
 */
export async function runDeferredRecoveryPass(
  uploadDir: string,
  matchJobId: string,
  shortVideoPath: string,
  movieVideoPath: string,
  onProgress?: (info: DeferredRecoveryProgress) => void,
): Promise<MatchedSegment[]> {
  const segmentIndexes = listCandidateFilesForJob(uploadDir, matchJobId);
  if (segmentIndexes.length === 0) return [];

  console.log(`[DeferredRecovery] Match ${matchJobId}: ${segmentIndexes.length} previously-rejected segment(s) to re-examine.`);

  const recovered: (MatchedSegment | null)[] = new Array(segmentIndexes.length).fill(null);

  /**
   * Re-examines one dropped segment's background candidates, in order,
   * stopping at the first accepted one — identical logic to before. Segments
   * are independent of each other (each reads/writes only its own candidate
   * file), so running several of these concurrently changes nothing about
   * which candidate gets accepted for any given segment.
   */
  async function recoverOneSegment(orderIdx: number): Promise<void> {
    const segmentIndex = segmentIndexes[orderIdx];
    const entry: StoredCandidateSet | null = readCandidatesFile(uploadDir, matchJobId, segmentIndex);
    if (!entry || entry.candidates.length === 0) return;
    // Only genuinely-dropped ranges are eligible for recovery. Segments the
    // main pass already accepted also get a candidate file now (for the
    // compare UI's history), but re-verifying them here would be wasteful
    // and could wrongly duplicate an already-matched segment.
    if (entry.dropped === false) return;

    let acceptedIdx: number | null = null;

    for (let c = 0; c < entry.candidates.length; c++) {
      const candidateEntry = entry.candidates[c];
      // Skip candidates the main pass already ran through VLM (now included
      // for comparison history) — only fresh, never-tried candidates need
      // deferred verification.
      if (candidateEntry.checked) continue;
      const framePairs = pickVerificationFramePairs(candidateEntry.segment);

      let verdict: DeferredRecoveryProgress['verdict'] = 'unverifiable';
      try {
        const extracted = await Promise.all(
          framePairs.map(async (p) => {
            const [shortFrameB64, movieFrameB64] = await Promise.all([
              extractFrameAsBase64(shortVideoPath, p.shortTime),
              extractFrameAsBase64(movieVideoPath, p.movieTime),
            ]);
            return { shortFrameB64, movieFrameB64 };
          }),
        );
        const result = await verifySameSceneMulti(extracted);

        if (result === null) {
          candidateEntry.checked = true;
          candidateEntry.verdict = 'unverifiable';
          verdict = 'unverifiable';
        } else if (result.same && result.confidencePct >= VLM_CONFIDENCE_THRESHOLD) {
          candidateEntry.checked = true;
          candidateEntry.verdict = 'accepted';
          candidateEntry.confidencePct = result.confidencePct;
          verdict = 'accepted';
          acceptedIdx = c;
        } else {
          candidateEntry.checked = true;
          candidateEntry.verdict = 'rejected';
          candidateEntry.confidencePct = result.confidencePct;
          verdict = 'rejected';
        }
      } catch (err: any) {
        console.warn(`[DeferredRecovery] Frame extraction failed for match ${matchJobId} segment ${segmentIndex} candidate ${c}: ${err?.message || err}`);
        candidateEntry.checked = true;
        candidateEntry.verdict = 'unverifiable';
        verdict = 'unverifiable';
      }

      onProgress?.({
        segmentIndex,
        totalRejected: segmentIndexes.length,
        candidateAttempt: c + 1,
        totalCandidates: entry.candidates.length,
        verdict,
      });

      if (acceptedIdx !== null) break;
    }

    if (acceptedIdx !== null) {
      entry.recoveredCandidateIndex = acceptedIdx;
      recovered[orderIdx] = entry.candidates[acceptedIdx].segment;
      console.log(`[DeferredRecovery] Match ${matchJobId}: recovered short-clip range [${entry.shortStart.toFixed(2)}s–${entry.shortEnd.toFixed(2)}s] via candidate ${acceptedIdx + 1}/${entry.candidates.length}.`);
    } else {
      onProgress?.({
        segmentIndex,
        totalRejected: segmentIndexes.length,
        candidateAttempt: entry.candidates.length,
        totalCandidates: entry.candidates.length,
        verdict: 'exhausted',
      });
      console.log(`[DeferredRecovery] Match ${matchJobId}: short-clip range [${entry.shortStart.toFixed(2)}s–${entry.shortEnd.toFixed(2)}s] remains unmatched after ${entry.candidates.length} candidate(s).`);
    }

    // Persist per-candidate verdicts for the preview UI (does not affect the
    // primary match result JSON, which lives in a separate file).
    writeCandidatesFileSync(uploadDir, matchJobId, segmentIndex, entry);
  }

  // Bounded concurrency, matching the main VLM pass — up to VLM_CONCURRENCY
  // segments re-examined at once instead of strictly one at a time.
  let cursor = 0;
  const workerCount = Math.min(VLM_CONCURRENCY, segmentIndexes.length);
  const workers = Array.from({ length: workerCount }, async () => {
    while (cursor < segmentIndexes.length) {
      const orderIdx = cursor++;
      await recoverOneSegment(orderIdx);
    }
  });
  await Promise.all(workers);

  return recovered.filter((s): s is MatchedSegment => s !== null);
}
