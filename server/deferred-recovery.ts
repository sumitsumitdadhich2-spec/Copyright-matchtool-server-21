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
import { MatchedSegment } from './candidate-matching-engine';
import { rankCandidatesCropRobust } from './candidate-embedding-rank';
import { pickVerificationFramePairs } from './vlm-segment-resolver';
import { extractFrameAsBase64, verifySameSceneChecked, VLM_CONFIDENCE_THRESHOLD, VLM_CONCURRENCY } from './vlm-verify';
import { geminiVideoAvailable, geminiVerifySegmentVideos } from './gemini-video-verify';
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

    // Only fresh, never-tried candidates need deferred verification —
    // candidates the main pass already ran through VLM are history-only.
    let uncheckedOrder = entry.candidates
      .map((cand, i) => (cand.checked ? -1 : i))
      .filter(i => i !== -1);

    // Crop-robust embedding ranking (candidate system only): verify the
    // candidates whose movie frame — full OR any left/center/right 9:16
    // window — best matches the short frame FIRST, so a vertically cropped
    // short is recovered on an early attempt instead of exhausting the
    // pool on wrong locations. Reorder-only and fail-safe: if ranking is
    // unavailable, the original candidate order is used exactly as before.
    const ranked = await rankCandidatesCropRobust(
      entry.candidates,
      uncheckedOrder,
      shortVideoPath,
      movieVideoPath,
      'DeferredRecovery',
    );
    if (ranked) uncheckedOrder = ranked;

    for (const c of uncheckedOrder) {
      const candidateEntry = entry.candidates[c];
      const framePairs = pickVerificationFramePairs(candidateEntry.segment);

      let verdict: DeferredRecoveryProgress['verdict'] = 'unverifiable';

      // FULL-SEGMENT VIDEO check first (Gemini priority) — whole candidate
      // segment from both videos in one request; frame-based flow is only
      // the fallback when no video verdict is possible.
      let videoDecided = false;
      if (geminiVideoAvailable()) {
        const seg = candidateEntry.segment;
        const vr = await geminiVerifySegmentVideos({
          shortVideoPath,
          shortStart: seg.shortStart,
          shortEnd: seg.shortEnd,
          movieVideoPath,
          movieStart: seg.movieStart,
          movieEnd: seg.movieEnd,
          label: `recover-seg${segmentIndex}-c${c}`,
        });
        if (vr !== null) {
          videoDecided = true;
          candidateEntry.checked = true;
          candidateEntry.confidencePct = vr.confidencePct;
          if (vr.same && vr.confidencePct >= VLM_CONFIDENCE_THRESHOLD) {
            candidateEntry.verdict = 'accepted';
            verdict = 'accepted';
            acceptedIdx = c;
          } else {
            candidateEntry.verdict = 'rejected';
            verdict = 'rejected';
          }
        }
      }

      if (!videoDecided) try {
        const extracted = await Promise.all(
          framePairs.map(async (p) => {
            const [shortFrameB64, movieFrameB64] = await Promise.all([
              extractFrameAsBase64(shortVideoPath, p.shortTime),
              extractFrameAsBase64(movieVideoPath, p.movieTime),
            ]);
            return { shortFrameB64, movieFrameB64 };
          }),
        );
        // Accept-side self-consistency re-check included: a recovered
        // candidate is accepted only if the swapped-order re-check agrees.
        const result = await verifySameSceneChecked(extracted);

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
