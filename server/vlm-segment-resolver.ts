/**
 * Retry/replace loop that gates matched segments through VLM scene
 * verification. Runs strictly AFTER groundMatchedSegments() has already
 * produced its candidates — it never touches hash-matching speed or accuracy.
 */
import type { MatchedSegment } from './matching-engine';
import { getAlternateCandidatesForRange } from './matching-engine';
import {
  extractFrameAsBase64,
  verifySameSceneChecked,
  isVlmAvailable,
  resetVlmCache,
  VLM_CONFIDENCE_THRESHOLD,
  VLM_MAX_ATTEMPTS,
  VLM_CONCURRENCY,
} from './vlm-verify';
import { embeddingGateCheck, EMBED_UNVERIFIABLE_KEEP_SIM } from './embedding-gate';
import { sscdGateEnabled } from './sscd-verify-gate';
import { geminiConfigured } from './gemini-vlm';
import { geminiVerifyVideoClips } from './gemini-video-verify';

// Fixed batch size for VLM server cache-reset points. Purely a cache-hygiene
// boundary — does not change segment order, candidate selection, or verdicts.
const VLM_RESET_BATCH_SIZE = 48;

export interface VlmProgressInfo {
  segmentIndex: number;
  totalSegments: number;
  attempt: number;
  verdict: 'accepted' | 'rejected' | 'unverifiable' | 'dropped';
}

/**
 * Pick the representative (shortTime, movieTime) pair to verify for a segment.
 * `bestFrameDetail` carries per-channel similarity scores but not timestamps,
 * so the actual frame pair always comes from the matchSequence midpoint
 * (falling back to the segment bounds if the sequence is empty).
 */
export function pickRepresentativeFrames(seg: MatchedSegment): { shortTime: number; movieTime: number } {
  const mid = seg.matchSequence[Math.floor(seg.matchSequence.length / 2)];
  if (mid) return { shortTime: mid.shortTime, movieTime: mid.movieTime };
  return { shortTime: seg.shortStart, movieTime: seg.movieStart };
}

/**
 * Minimum spacing (seconds, on the short-clip timeline) between the frame
 * pairs sent to the VLM for one segment — prevents sending three
 * near-identical frames from the same instant, which would add tokens
 * without adding any real cross-check value.
 */
const PAIR_MIN_SPACING_S = 0.75;

/**
 * Pick up to 3 (shortTime, movieTime) pairs for VLM verification of one
 * segment, all drawn from timestamps the hash matcher already computed —
 * no video re-scan:
 *  1. the segment midpoint (identical to pickRepresentativeFrames — the
 *     exact pair the old single-pair flow verified),
 *  2. the HIGHEST-similarity frame pair in the segment's matchSequence,
 *  3. the SECOND-highest-similarity pair,
 * deduplicated so pairs are at least PAIR_MIN_SPACING_S apart on the short
 * timeline. Falls back gracefully to fewer pairs (down to 1) for very short
 * or sparse segments.
 */
export function pickVerificationFramePairs(
  seg: MatchedSegment,
): Array<{ shortTime: number; movieTime: number }> {
  const pairs: Array<{ shortTime: number; movieTime: number }> = [];
  const addIfSpaced = (p: { shortTime: number; movieTime: number }) => {
    if (pairs.length >= 3) return;
    if (pairs.some(q => Math.abs(q.shortTime - p.shortTime) < PAIR_MIN_SPACING_S)) return;
    pairs.push(p);
  };

  // 1. Midpoint — keeps the old behavior's pair always included.
  const mid = seg.matchSequence[Math.floor(seg.matchSequence.length / 2)];
  if (mid) {
    pairs.push({ shortTime: mid.shortTime, movieTime: mid.movieTime });
  } else {
    pairs.push({ shortTime: seg.shortStart, movieTime: seg.movieStart });
  }

  // 2 & 3. Highest- and second-highest-similarity pairs from the sequence.
  const bySimilarity = [...seg.matchSequence].sort((a, b) => b.similarity - a.similarity);
  for (const f of bySimilarity) {
    if (pairs.length >= 3) break;
    addIfSpaced({ shortTime: f.shortTime, movieTime: f.movieTime });
  }

  return pairs;
}

/**
 * Resolve a set of matched segments through VLM verification. Rejected
 * candidates are replaced with the next-best alternative for the same
 * short-clip range (drawn from `candidatePool`, never a small time-shift of
 * the same rejected spot). After VLM_MAX_ATTEMPTS rejections for a range (or
 * no more candidates), the range is dropped from the result entirely.
 *
 * Gracefully returns `segments` unchanged if the VLM endpoint is not
 * configured or unreachable — the tool must keep working with the AWS GPU
 * server off.
 */
export interface SegmentCandidateAttempt {
  /** The candidate segment that was actually run through VLM at this attempt. */
  segment: MatchedSegment;
  verdict: 'accepted' | 'rejected' | 'unverifiable';
  /** Present whenever the VLM call itself returned a parsed result (accepted or rejected). */
  confidencePct?: number;
}

export interface SegmentResolvedInfo {
  segmentIndex: number;
  original: MatchedSegment;
  /** Every candidate this pass actually ran through VLM for this range, in attempt order. */
  triedCandidates: SegmentCandidateAttempt[];
  /** The candidate ultimately kept for this range, or null if the range was dropped entirely. */
  accepted: MatchedSegment | null;
}

export async function resolveSegmentsWithVLM(
  segments: MatchedSegment[],
  shortVideoPath: string,
  movieVideoPath: string,
  candidatePool: MatchedSegment[] | undefined,
  onProgress?: (info: VlmProgressInfo) => void,
  /**
   * Optional side-effect fired once per segment, the instant this pass has a
   * final verdict for it (accepted immediately, accepted after retries, or
   * dropped after VLM_MAX_ATTEMPTS/no more candidates). Purely additive —
   * does not change which segments are accepted/dropped, their order, or
   * timing. Used by server.ts to persist full candidate-comparison history
   * (for every segment, not only dropped ones) and to kick off background
   * candidate discovery for a later deferred recovery pass; never awaited here.
   */
  onSegmentResolved?: (info: SegmentResolvedInfo) => void,
): Promise<MatchedSegment[]> {
  if (segments.length === 0) return segments;

  // ------------------------------------------------------------------------
  // Provider availability: verification must proceed if ANY configured
  // provider can handle it — not just the legacy Qwen endpoint. The routing
  // inside verifySameSceneChecked (SSCD gate -> Gemini composite -> legacy
  // Qwen) already fails safe per provider, so all we must NOT do here is
  // short-circuit before that routing ever runs. Only skip the entire pass
  // when zero providers are configured/available at all.
  // ------------------------------------------------------------------------
  const vlmAvailable = await isVlmAvailable();
  const sscdConfigured = sscdGateEnabled();
  const geminiAvailable = geminiConfigured();
  if (!vlmAvailable && !sscdConfigured && !geminiAvailable) {
    console.warn(
      `[VLM] Skipping verification pass — no verification provider available (` +
      `VLM_ENDPOINT_URL ${process.env.VLM_ENDPOINT_URL ? 'set but unreachable' : 'unset'}, ` +
      `GPU_EMBED_SERVICE_URL ${process.env.GPU_EMBED_SERVICE_URL ? 'set but SSCD gate disabled' : 'unset'}, ` +
      `GEMINI_API_KEY ${process.env.GEMINI_API_KEY ? 'set but unusable' : 'unset'})`
    );
    return segments;
  }

  // Explicit guarantee (not an incidental side effect) that each new
  // video pair's VLM verification starts from clean server-side cache state.
  // Only meaningful (and only attempted) when the Qwen VLM server itself is
  // reachable — the SSCD gate and Gemini have no server-side KV cache.
  if (vlmAvailable) await resetVlmCache('for new video pair');

  // Slot for each segment's final outcome, filled in whatever order segments
  // within a batch happen to finish — collected and sorted at the end, so
  // completion order never affects the returned result.
  const outcomes: (MatchedSegment | null)[] = new Array(segments.length).fill(null);

  /**
   * Runs the exact same retry/replace loop as before for a single segment.
   * Segments are fully independent of each other: each only reads its own
   * `original` bounds and the shared, read-only `candidatePool` — so running
   * several of these concurrently changes nothing about which candidates are
   * tried, in what preference order, or what verdict each one gets. Only
   * wall-clock time changes.
   */
  async function resolveOneSegment(i: number): Promise<void> {
    const original = segments[i];
    let candidate: MatchedSegment | undefined = original;
    const rejectedMovieTimestamps: number[] = [];
    const triedCandidates: SegmentCandidateAttempt[] = [];
    let attempt = 0;
    let accepted: MatchedSegment | null = null;

    while (candidate && attempt < VLM_MAX_ATTEMPTS) {
      attempt++;
      const triedCandidate = candidate;
      const framePairs = pickVerificationFramePairs(candidate);

      let verdict: VlmProgressInfo['verdict'] = 'unverifiable';

      // ----------------------------------------------------------------
      // FINAL-DECISION VIDEO-CLIP CHECK (Gemini configured): the actual
      // short segment and movie segment are cut with ffmpeg and sent as
      // two video clips to Gemini with the forensic copyright prompt.
      // Its verdict is FINAL for this candidate — no frame pairs, no
      // embedding gate, no Qwen. Only when Gemini can produce NO verdict
      // at all (quota exhausted / network / ffmpeg failure) does the
      // candidate fall back to the conservative "keep as unverifiable"
      // policy. The legacy frame path below runs ONLY when GEMINI_API_KEY
      // is not set at all.
      // ----------------------------------------------------------------
      if (geminiConfigured()) {
        const vv = await geminiVerifyVideoClips(shortVideoPath, movieVideoPath, candidate);
        if (vv === null) {
          verdict = 'unverifiable';
          accepted = candidate;
          triedCandidates.push({ segment: triedCandidate, verdict: 'unverifiable' });
          onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
          console.log(`[VideoVerify] Segment ${i} attempt ${attempt}: no verdict obtainable — keeping as unverifiable`);
          break;
        }
        if (vv.same && vv.confidence >= VLM_CONFIDENCE_THRESHOLD) {
          verdict = 'accepted';
          accepted = candidate;
          triedCandidates.push({ segment: triedCandidate, verdict: 'accepted', confidencePct: vv.confidence });
          onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
          break;
        }
        verdict = 'rejected';
        triedCandidates.push({ segment: triedCandidate, verdict: 'rejected', confidencePct: vv.confidence });
        onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
      } else try {
        // LEGACY FRAME PATH — only reachable without GEMINI_API_KEY.
        // Extract every needed frame in parallel (still one ffmpeg spawn per
        // frame, all concurrent), then make ONE VLM call with all pairs —
        // same request count per attempt as the old single-pair flow.
        const extracted = await Promise.all(
          framePairs.map(async (p) => {
            const [shortFrameB64, movieFrameB64] = await Promise.all([
              extractFrameAsBase64(shortVideoPath, p.shortTime),
              extractFrameAsBase64(movieVideoPath, p.movieTime),
            ]);
            return { shortFrameB64, movieFrameB64 };
          }),
        );
        // ------------------------------------------------------------------
        // Embedding gate: a deterministic CLIP-similarity pre-check on the
        // exact frames that would go to the VLM. Clear-cut cases are decided
        // here (no VLM call at all); only the ambiguous middle band falls
        // through to the VLM. If the gate is unavailable (model failed to
        // load), gate === null and behavior is identical to before.
        // ------------------------------------------------------------------
        const gate = await embeddingGateCheck(extracted);
        if (gate?.decision === 'accept') {
          verdict = 'accepted';
          accepted = candidate;
          triedCandidates.push({ segment: triedCandidate, verdict: 'accepted', confidencePct: Math.round(gate.medianSim * 100) });
          onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
          console.log(`[EmbedGate] Segment ${i} attempt ${attempt}: auto-accept (median sim ${gate.medianSim.toFixed(3)})`);
          break;
        }
        if (gate?.decision === 'reject') {
          verdict = 'rejected';
          triedCandidates.push({ segment: triedCandidate, verdict: 'rejected', confidencePct: Math.round(gate.medianSim * 100) });
          onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
          console.log(`[EmbedGate] Segment ${i} attempt ${attempt}: auto-reject (max sim ${gate.maxSim.toFixed(3)})`);
        } else {
          // Ambiguous (or gate unavailable) — the VLM is the tie-breaker,
          // with an accept-side self-consistency re-check (swapped image
          // order): an accept stands only if both calls agree.
          const result = await verifySameSceneChecked(extracted);

          if (result === null) {
            // VLM could not produce a verdict (timeout/overload). The old
            // policy silently ACCEPTED here, which meant every VLM outage
            // passed unverified segments straight through — a major source
            // of false accepts. New policy: keep the candidate only if the
            // embedding similarity independently supports it; otherwise
            // treat it as rejected and try the next candidate.
            const keep = gate !== null && gate.medianSim >= EMBED_UNVERIFIABLE_KEEP_SIM;
            if (keep || gate === null) {
              // gate === null: no independent signal at all — keep the old
              // conservative "don't drop on infrastructure failure" behavior.
              verdict = 'unverifiable';
              accepted = candidate;
              triedCandidates.push({ segment: triedCandidate, verdict: 'unverifiable' });
              onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
              break;
            }
            verdict = 'rejected';
            triedCandidates.push({ segment: triedCandidate, verdict: 'rejected', confidencePct: Math.round(gate.medianSim * 100) });
            onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
            console.log(
              `[VLM] Segment ${i} attempt ${attempt}: VLM inconclusive and embedding sim ` +
              `${gate.medianSim.toFixed(3)} < ${EMBED_UNVERIFIABLE_KEEP_SIM} — rejecting instead of blind-accepting`
            );
          } else if (result.same && result.confidencePct >= VLM_CONFIDENCE_THRESHOLD) {
            verdict = 'accepted';
            accepted = candidate;
            triedCandidates.push({ segment: triedCandidate, verdict: 'accepted', confidencePct: result.confidencePct });
            onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
            break;
          } else {
            verdict = 'rejected';
            triedCandidates.push({ segment: triedCandidate, verdict: 'rejected', confidencePct: result.confidencePct });
            onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
          }
        }
      } catch (err: any) {
        // Frame extraction failure (e.g. bad timestamp) — treat as
        // unverifiable for this attempt rather than crashing the match.
        console.warn(`[VLM] Frame extraction failed for segment ${i}, attempt ${attempt}: ${err?.message || err}`);
        verdict = 'unverifiable';
        accepted = candidate;
        triedCandidates.push({ segment: triedCandidate, verdict: 'unverifiable' });
        onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict });
        break;
      }

      // Rejected — look for the next-best alternative elsewhere in the movie,
      // never re-showing an already-rejected movie timestamp for this range.
      rejectedMovieTimestamps.push(candidate.movieStart);
      const alternatives = getAlternateCandidatesForRange(
        candidatePool,
        original.shortStart,
        original.shortEnd,
        rejectedMovieTimestamps,
      );
      candidate = alternatives[0];
    }

    if (accepted) {
      outcomes[i] = accepted;
    } else {
      console.log(
        `[VLM] No genuine match found after ${attempt} attempt(s) for short-clip range ` +
        `[${original.shortStart.toFixed(2)}s–${original.shortEnd.toFixed(2)}s] — dropping.`
      );
      onProgress?.({ segmentIndex: i, totalSegments: segments.length, attempt, verdict: 'dropped' });
    }

    // Fires for EVERY segment (accepted first try, accepted after retries, or
    // dropped) so the caller can persist full candidate-comparison history —
    // not just what happened to previously-dropped segments.
    onSegmentResolved?.({ segmentIndex: i, original, triedCandidates, accepted });
  }

  /**
   * Runs `indices` with up to VLM_CONCURRENCY segments in flight at once.
   * Each worker pulls the next unclaimed index off the shared cursor, so
   * segments that finish quickly (fewer retries) immediately pick up the
   * next one instead of waiting for the slowest one in a fixed pairing.
   */
  async function runPool(indices: number[]): Promise<void> {
    let cursor = 0;
    const workerCount = Math.min(VLM_CONCURRENCY, indices.length);
    const workers = Array.from({ length: workerCount }, async () => {
      while (cursor < indices.length) {
        const idx = indices[cursor++];
        await resolveOneSegment(idx);
      }
    });
    await Promise.all(workers);
  }

  // Process in fixed-size batches — identical cache-reset boundaries to the
  // previous sequential version (every VLM_RESET_BATCH_SIZE segments, plus
  // the final partial batch), just with up to VLM_CONCURRENCY segments
  // within each batch running at once instead of one at a time.
  for (let batchStart = 0; batchStart < segments.length; batchStart += VLM_RESET_BATCH_SIZE) {
    const batchEnd = Math.min(batchStart + VLM_RESET_BATCH_SIZE, segments.length);
    const indices = Array.from({ length: batchEnd - batchStart }, (_, k) => batchStart + k);
    await runPool(indices);
    if (vlmAvailable) await resetVlmCache(`after batch (segments ${batchStart + 1}-${batchEnd})`);
  }

  const resolved = outcomes.filter((s): s is MatchedSegment => s !== null);
  resolved.sort((a, b) => a.shortStart - b.shortStart);
  return resolved;
}
