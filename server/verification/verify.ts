/**
 * The new candidate/verification system.
 * ===========================================================================
 * Input:  MatchedSegment[] from the untouched matching engine + the two
 *         original uploaded video files.
 * Output: the finalised segment list (accepted matches, plus rejected ones
 *         kept visible and flagged) with one durable record per range.
 *
 * The whole design is one loop and one decision:
 *
 *   for each short-clip range the engine matched:
 *     take its candidates (the engine's own pick + the engine's alternates,
 *     ordered by hash confidence)
 *     cut both clips, ask Gemini "same footage?" once per candidate
 *     the FIRST accepted candidate wins; stop asking
 *     if none are accepted, the range is dropped (kept visible, flagged)
 *
 * Deliberately NOT here (all of it was in the old system and none of it
 * improved accuracy):
 *   - no duplicate of the matching engine; alternates come from the engine's
 *     own `getAlternateCandidatesForRange` export
 *   - no embedding gate, SSCD gate, dense re-scan, or degenerate guard
 *   - no cascade of fallback verifiers that each get a second opinion
 *   - no re-fingerprinting; the only source data is MatchedSegment + the files
 *
 * Gemini (server/gemini-vlm.ts) is used exactly as-is and is the sole judge.
 */

import {
  geminiConfigured,
  geminiVerifyVideoPair,
  getGeminiStatus,
  type GeminiVerdict,
} from '../gemini-vlm';
import {
  getAlternateCandidatesForRange,
  type MatchedSegment,
} from '../matching-engine';
import { cutClip, deleteClip } from './clip';
import { buildVerificationPrompt, VIDEO1_LABEL, VIDEO2_LABEL } from './prompt';
import {
  writeRecordAsync,
  type CandidateRecord,
  type VerificationRecord,
} from './store';
import { flagTimelineOutliers } from './timeline';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/** Extra candidates fetched from the engine's pool, on top of its own pick. */
const MAX_ALTERNATES = clampInt(process.env.VERIFY_MAX_ALTERNATES, 2, 0, 6);
/** Ranges verified in parallel. Gemini's own RPM pacing lives in gemini-vlm.ts. */
const CONCURRENCY = clampInt(process.env.VERIFY_CONCURRENCY, 2, 1, 8);
/** A verdict below this confidence is not trusted either way. */
const MIN_CONFIDENCE = clampInt(process.env.VERIFY_MIN_CONFIDENCE, 55, 0, 100);
/** Frame-sampling rate handed to Gemini for both clips of a pair. */
const FPS = clampInt(process.env.VERIFY_FPS, 2, 1, 24);

function clampInt(raw: string | undefined, fallback: number, min: number, max: number): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.round(n)));
}

// ---------------------------------------------------------------------------
// Public contract
// ---------------------------------------------------------------------------

export interface VerifyRequest {
  /** Raw segments from matchVideosFromFiles(). Never mutated. */
  segments: MatchedSegment[];
  /** The engine's pre-dedup pool, used only to offer alternates. */
  candidatePool?: MatchedSegment[];
  /** Absolute path of the uploaded short clip (undefined when not retained). */
  shortVideoPath: string | undefined;
  /** Absolute path of the uploaded reference movie (undefined when not retained). */
  movieVideoPath: string | undefined;
  /** Where per-range records are written. */
  uploadDir: string;
  matchJobId: string;
  onProgress?: (done: number, total: number, message: string) => void;
}

export interface VerifySummary {
  /** False when verification did not run at all. `reason` says why. */
  ran: boolean;
  reason: string;
  rangesTotal: number;
  rangesVerified: number;
  accepted: number;
  rejected: number;
  unverifiable: number;
  /** Ranges whose match came from an alternate rather than the engine's pick. */
  switched: number;
  geminiCalls: number;
}

export interface VerifyResult {
  segments: MatchedSegment[];
  summary: VerifySummary;
}

/**
 * Verify every matched segment. Never throws and never rejects: any failure
 * degrades to "unverifiable" for that range and the engine's original segment
 * is kept, so a match job always completes.
 */
export async function verifyMatchedSegments(req: VerifyRequest): Promise<VerifyResult> {
  const total = req.segments.length;

  if (total === 0) {
    console.log('[Verify] No matched segments — nothing to verify.');
    return { segments: [], summary: emptySummary('no matched segments', 0) };
  }

  // --- Graceful degradation, loudly ------------------------------------------
  // The old system's worst bug was silently no-opping when its provider was
  // unconfigured. Every skip path below names itself explicitly.
  if (!req.shortVideoPath || !req.movieVideoPath) {
    const reason =
      'original uploaded video file(s) are no longer on disk — verification SKIPPED, all segments pass through unverified';
    console.warn(`[Verify] ${reason}.`);
    await writeSkippedRecords(req, reason);
    return {
      segments: flagTimelineOutliers(req.segments),
      summary: emptySummary(reason, total),
    };
  }

  if (!geminiConfigured()) {
    const reason =
      'GEMINI_API_KEY is not set — verification SKIPPED, all segments pass through unverified';
    console.warn(`[Verify] ${reason}.`);
    await writeSkippedRecords(req, reason);
    return {
      segments: flagTimelineOutliers(req.segments),
      summary: emptySummary(reason, total),
    };
  }

  const status = getGeminiStatus();
  if (status.dailyLimitReached) {
    const reason =
      'Gemini daily quota exhausted on every model — verification SKIPPED, all segments pass through unverified';
    console.warn(`[Verify] ${reason}.`);
    await writeSkippedRecords(req, reason);
    return {
      segments: flagTimelineOutliers(req.segments),
      summary: emptySummary(reason, total),
    };
  }

  console.log(
    `[Verify] Verifying ${total} matched range(s) with Gemini (model=${status.model}, ` +
    `concurrency=${CONCURRENCY}, alternates<=${MAX_ALTERNATES}, minConfidence=${MIN_CONFIDENCE}).`,
  );

  const summary: VerifySummary = {
    ran: true,
    reason: 'verified with Gemini',
    rangesTotal: total,
    rangesVerified: 0,
    accepted: 0,
    rejected: 0,
    unverifiable: 0,
    switched: 0,
    geminiCalls: 0,
  };

  // Stable range index by short-clip order, so record files line up with the
  // order the UI renders and survive a re-check later.
  const ordered = [...req.segments].sort((a, b) => a.shortStart - b.shortStart);
  const finalised: Array<MatchedSegment | null> = new Array(ordered.length).fill(null);

  let done = 0;
  let cursor = 0;

  const runWorker = async (): Promise<void> => {
    for (;;) {
      const index = cursor++;
      if (index >= ordered.length) return;

      const outcome = await verifyOneRange(req, ordered[index], index);

      summary.geminiCalls += outcome.calls;
      summary.rangesVerified++;
      if (outcome.record.dropped) {
        summary.rejected++;
      } else if (outcome.record.candidates.some(c => c.verdict === 'accepted')) {
        summary.accepted++;
        if ((outcome.record.usedCandidateIndex ?? 0) > 0) summary.switched++;
      } else {
        summary.unverifiable++;
      }

      finalised[index] = outcome.segment;
      await writeRecordAsync(req.uploadDir, req.matchJobId, outcome.record);

      done++;
      req.onProgress?.(done, ordered.length, `Verified ${done}/${ordered.length} matched ranges`);
    }
  };

  await Promise.all(
    Array.from({ length: Math.min(CONCURRENCY, ordered.length) }, () => runWorker()),
  );

  const segments = flagTimelineOutliers(
    finalised.filter((s): s is MatchedSegment => s !== null),
  );

  console.log(
    `[Verify] Done: ${summary.accepted} accepted, ${summary.rejected} rejected, ` +
    `${summary.unverifiable} unverifiable, ${summary.switched} switched to an alternate ` +
    `(${summary.geminiCalls} Gemini call(s)).`,
  );

  return { segments, summary };
}

// ---------------------------------------------------------------------------
// One range
// ---------------------------------------------------------------------------

interface RangeOutcome {
  segment: MatchedSegment;
  record: VerificationRecord;
  calls: number;
}

async function verifyOneRange(
  req: VerifyRequest,
  primary: MatchedSegment,
  segmentIndex: number,
): Promise<RangeOutcome> {
  const candidates = collectCandidates(primary, req.candidatePool);

  const record: VerificationRecord = {
    segmentIndex,
    shortStart: primary.shortStart,
    shortEnd: primary.shortEnd,
    recordedAt: Date.now(),
    candidates: candidates.map(segment => ({
      segment,
      checked: false,
      hashConfidence: segment.confidence,
    })),
    dropped: false,
  };

  // The short-clip side is identical for every candidate of this range, so it
  // is cut once and reused across all Gemini calls.
  const shortClip = await cutClip(
    req.shortVideoPath!,
    primary.shortStart,
    primary.shortEnd,
    `short-${segmentIndex}`,
  );

  if (!shortClip) {
    const reason = 'could not cut the short-clip side with ffmpeg — range left unverified';
    console.warn(`[Verify] Range ${segmentIndex} (${fmt(primary.shortStart)}-${fmt(primary.shortEnd)}): ${reason}.`);
    record.skippedReason = reason;
    record.usedCandidateIndex = 0;
    return { segment: primary, record, calls: 0 };
  }

  let calls = 0;
  try {
    for (let i = 0; i < candidates.length; i++) {
      const candidate = candidates[i];
      const entry = record.candidates[i];

      const movieClip = await cutClip(
        req.movieVideoPath!,
        candidate.movieStart,
        candidate.movieEnd,
        `movie-${segmentIndex}-${i}`,
      );
      if (!movieClip) {
        entry.verdict = 'unverifiable';
        entry.reason = 'reference clip could not be cut';
        continue;
      }

      let verdict: GeminiVerdict | null = null;
      try {
        verdict = await geminiVerifyVideoPair(
          movieClip,
          shortClip,
          VIDEO1_LABEL,
          VIDEO2_LABEL,
          buildVerificationPrompt({
            movieStart: candidate.movieStart,
            movieEnd: candidate.movieEnd,
            shortStart: candidate.shortStart,
            shortEnd: candidate.shortEnd,
            speedRatio: candidate.speedRatio,
          }),
          FPS,
        );
        calls++;
      } catch (e: any) {
        console.warn(
          `[Verify] Range ${segmentIndex} candidate ${i}: Gemini call threw (${e?.message || e}) — unverifiable.`,
        );
      } finally {
        deleteClip(movieClip);
      }

      entry.checked = verdict !== null;

      if (!verdict) {
        entry.verdict = 'unverifiable';
        entry.reason = 'Gemini returned no usable verdict';
        continue;
      }

      entry.confidencePct = Math.round(verdict.confidence);
      entry.reason = (verdict.evidence || []).join('; ').slice(0, 400) || undefined;

      const trusted = verdict.confidence >= MIN_CONFIDENCE;
      if (verdict.same && trusted) {
        entry.verdict = 'accepted';
        record.usedCandidateIndex = i;
        console.log(
          `[Verify] Range ${segmentIndex} (${fmt(primary.shortStart)}-${fmt(primary.shortEnd)}) ` +
          `ACCEPTED candidate ${i} @ movie ${fmt(candidate.movieStart)}s ` +
          `(Gemini ${entry.confidencePct}%${i > 0 ? ', switched from the engine pick' : ''}).`,
        );
        // A winner ends the range — no point spending quota on the rest.
        return { segment: candidates[i], record, calls };
      }

      if (!verdict.same && trusted) {
        entry.verdict = 'rejected';
      } else {
        entry.verdict = 'unverifiable';
        entry.reason =
          `low certainty (${entry.confidencePct}% < ${MIN_CONFIDENCE}%)` +
          (entry.reason ? ` — ${entry.reason}` : '');
      }
    }
  } finally {
    deleteClip(shortClip);
  }

  // Nobody won. Keep the engine's own pick visible so the user can review it
  // and hit re-check, but mark it clearly as not a verified match.
  const anyRejected = record.candidates.some(c => c.verdict === 'rejected');
  record.dropped = anyRejected;
  record.usedCandidateIndex = 0;
  if (!anyRejected) {
    record.skippedReason = 'no candidate could be judged — Gemini gave no usable verdict';
  }

  console.log(
    `[Verify] Range ${segmentIndex} (${fmt(primary.shortStart)}-${fmt(primary.shortEnd)}): ` +
    `${anyRejected ? 'REJECTED' : 'UNVERIFIABLE'} after ${record.candidates.length} candidate(s) — ` +
    `keeping the engine pick, flagged for review.`,
  );

  return {
    segment: { ...primary, vlmRejectedKept: anyRejected },
    record,
    calls,
  };
}

/**
 * The engine's own pick first, then its alternates for the same short range by
 * descending hash confidence. Alternates pointing at essentially the same movie
 * timestamp as an earlier candidate are dropped — re-asking Gemini about the
 * same footage is pure quota waste.
 */
function collectCandidates(
  primary: MatchedSegment,
  pool: MatchedSegment[] | undefined,
): MatchedSegment[] {
  const out = [primary];
  if (MAX_ALTERNATES === 0) return out;

  const alternates = getAlternateCandidatesForRange(
    pool,
    primary.shortStart,
    primary.shortEnd,
    [],
    0.5,
  );

  for (const alt of alternates) {
    if (out.length >= MAX_ALTERNATES + 1) break;
    if (out.some(existing => Math.abs(existing.movieStart - alt.movieStart) < 1)) continue;
    out.push(alt);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Manual re-check of a single range (used by the preview UI's retry action)
// ---------------------------------------------------------------------------

export interface RecheckRequest extends Omit<VerifyRequest, 'segments' | 'onProgress'> {
  /** The range's current segment, as shown in the UI. */
  segment: MatchedSegment;
  /** Its stable range index — the record file to overwrite. */
  segmentIndex: number;
}

export interface RecheckResult {
  segment: MatchedSegment;
  record: VerificationRecord;
  /** True when a candidate was accepted this time. */
  accepted: boolean;
  /** Human-readable outcome for the UI toast. */
  message: string;
}

/**
 * Re-run verification for exactly one range and overwrite its record.
 * Same code path as the bulk pass, so a retry can never disagree with it.
 */
export async function recheckSegment(req: RecheckRequest): Promise<RecheckResult> {
  if (!req.shortVideoPath || !req.movieVideoPath || !geminiConfigured()) {
    const message = !geminiConfigured()
      ? 'GEMINI_API_KEY is not set — cannot re-check this segment'
      : 'original uploaded video file(s) are no longer on disk — cannot re-check this segment';
    console.warn(`[Verify] Re-check of range ${req.segmentIndex} refused: ${message}.`);
    const record: VerificationRecord = {
      segmentIndex: req.segmentIndex,
      shortStart: req.segment.shortStart,
      shortEnd: req.segment.shortEnd,
      recordedAt: Date.now(),
      candidates: [{ segment: req.segment, checked: false, hashConfidence: req.segment.confidence }],
      usedCandidateIndex: 0,
      dropped: false,
      skippedReason: message,
    };
    await writeRecordAsync(req.uploadDir, req.matchJobId, record);
    return { segment: req.segment, record, accepted: false, message };
  }

  console.log(`[Verify] Manual re-check requested for range ${req.segmentIndex}.`);

  const outcome = await verifyOneRange(
    {
      segments: [req.segment],
      candidatePool: req.candidatePool,
      shortVideoPath: req.shortVideoPath,
      movieVideoPath: req.movieVideoPath,
      uploadDir: req.uploadDir,
      matchJobId: req.matchJobId,
    },
    req.segment,
    req.segmentIndex,
  );

  await writeRecordAsync(req.uploadDir, req.matchJobId, outcome.record);

  const winner = outcome.record.candidates[outcome.record.usedCandidateIndex ?? 0];
  const accepted = winner?.verdict === 'accepted';
  const message = accepted
    ? `Confirmed as a match (Gemini ${winner.confidencePct ?? 0}% confident)`
    : outcome.record.dropped
      ? 'Gemini rejected every candidate for this segment'
      : outcome.record.skippedReason || 'Gemini could not judge this segment';

  return { segment: outcome.segment, record: outcome.record, accepted, message };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function emptySummary(reason: string, total: number): VerifySummary {
  return {
    ran: false,
    reason,
    rangesTotal: total,
    rangesVerified: 0,
    accepted: 0,
    rejected: 0,
    unverifiable: 0,
    switched: 0,
    geminiCalls: 0,
  };
}

/** One record per range explaining why nothing was checked, so the UI can say so. */
async function writeSkippedRecords(req: VerifyRequest, reason: string): Promise<void> {
  const ordered = [...req.segments].sort((a, b) => a.shortStart - b.shortStart);
  await Promise.all(
    ordered.map((segment, segmentIndex) =>
      writeRecordAsync(req.uploadDir, req.matchJobId, {
        segmentIndex,
        shortStart: segment.shortStart,
        shortEnd: segment.shortEnd,
        recordedAt: Date.now(),
        candidates: [{ segment, checked: false, hashConfidence: segment.confidence }],
        usedCandidateIndex: 0,
        dropped: false,
        skippedReason: reason,
      }),
    ),
  );
}

function fmt(seconds: number): string {
  return seconds.toFixed(2);
}
