/**
 * Preview-only, disk-backed storage for background-discovered alternate
 * candidates. Populated the instant a segment is finally dropped by the main
 * VLM pass (server/vlm-segment-resolver.ts), consumed by the deferred
 * recovery pass (server/deferred-recovery.ts) and by the preview-candidate
 * API routes in server.ts.
 *
 * This module intentionally does NOT run any hash-matching or scoring of its
 * own — it only reuses `getAlternateCandidatesForRange`, the pre-dedup
 * candidate pool already produced by the matching engine, so there is no
 * second similarity algorithm anywhere in this feature.
 *
 * Nothing here is ever merged into a match job's primary result JSON
 * (`<matchJobId>_matchresult.json`). Candidate data lives only in its own
 * `<matchJobId>_candidates_<segmentIndex>.json` files under uploads/, so RAM
 * usage does not grow with the number of rejected segments in a run.
 */
import * as fs from 'fs';
import * as path from 'path';
import { MatchedSegment, getAlternateCandidatesForRange } from './matching-engine';

export interface CandidateCheck {
  segment: MatchedSegment;
  checked: boolean;
  verdict?: 'accepted' | 'rejected' | 'unverifiable';
  confidencePct?: number;
}

export interface StoredCandidateSet {
  segmentIndex: number;
  shortStart: number;
  shortEnd: number;
  recordedAt: number;
  candidates: CandidateCheck[];
  /**
   * Index into `candidates` that was ultimately used for this short-clip
   * range — set as soon as the main VLM pass accepts a candidate (whether on
   * the first attempt or after retries), or later by the deferred pass if a
   * dropped segment gets recovered from its background candidates.
   */
  recoveredCandidateIndex?: number;
  /**
   * True when the main VLM pass could not find any acceptable candidate for
   * this range (only these are eligible for the deferred recovery pass).
   * False means the range was already accepted in the main pass — its
   * `candidates` list is comparison history only, not a to-do list.
   */
  dropped: boolean;
}

const CANDIDATES_MAX = 10;

function candidatesFileName(matchJobId: string, segmentIndex: number): string {
  return `${matchJobId}_candidates_${segmentIndex}.json`;
}

export function matchCandidatesFilePath(uploadDir: string, matchJobId: string, segmentIndex: number): string {
  return path.join(uploadDir, candidatesFileName(matchJobId, segmentIndex));
}

/**
 * Pick up to `max` next-best alternate movie-timestamp candidates for a
 * short-clip range, beyond everything already tried. Pure reuse of the
 * matching engine's own `getAlternateCandidatesForRange` — no new scoring.
 */
export function discoverBackgroundCandidates(
  candidatePool: MatchedSegment[] | undefined,
  shortStart: number,
  shortEnd: number,
  excludeMovieTimestamps: number[],
  max = CANDIDATES_MAX,
): MatchedSegment[] {
  return getAlternateCandidatesForRange(candidatePool, shortStart, shortEnd, excludeMovieTimestamps).slice(0, max);
}

/**
 * Merge a segment's real VLM attempt history with untried pool alternates
 * into one comparison-ready `StoredCandidateSet`, for the compare UI. Pure
 * and deterministic apart from the `recordedAt` timestamp — no fs, no VLM
 * calls — so it's unit-testable on its own.
 *
 * `triedCandidates` must be in attempt order; when `accepted` is non-null it
 * is always the last entry (resolveSegmentsWithVLM breaks the instant a
 * candidate is accepted), so its index in the merged list is simply
 * `triedCandidates.length - 1`. Extra alternates are pulled from
 * `candidatePool` excluding every already-tried movie timestamp, so the two
 * halves of the list never overlap.
 *
 * Returns null when there is nothing at all to show (no attempts and no
 * pool alternates) — callers should skip writing a file in that case.
 */
export function buildCandidateHistoryEntry(
  segmentIndex: number,
  original: { shortStart: number; shortEnd: number },
  triedCandidates: Array<{
    segment: MatchedSegment;
    verdict: 'accepted' | 'rejected' | 'unverifiable';
    confidencePct?: number;
  }>,
  accepted: MatchedSegment | null,
  candidatePool: MatchedSegment[] | undefined,
  extraMax = CANDIDATES_MAX,
): StoredCandidateSet | null {
  const triedTimestamps = triedCandidates.map(t => t.segment.movieStart);
  const extraCandidates = discoverBackgroundCandidates(
    candidatePool,
    original.shortStart,
    original.shortEnd,
    triedTimestamps,
    extraMax,
  );

  const candidates: CandidateCheck[] = [
    ...triedCandidates.map(t => ({
      segment: t.segment,
      checked: true,
      verdict: t.verdict,
      confidencePct: t.confidencePct,
    })),
    ...extraCandidates.map(segment => ({ segment, checked: false })),
  ];
  if (candidates.length === 0) return null;

  return {
    segmentIndex,
    shortStart: original.shortStart,
    shortEnd: original.shortEnd,
    recordedAt: Date.now(),
    dropped: accepted === null,
    candidates,
    recoveredCandidateIndex: accepted ? triedCandidates.length - 1 : undefined,
  };
}

/**
 * Build a comparison-ready `StoredCandidateSet` for a segment accepted
 * purely by hash-matching, with no VLM verification ever run for it (VLM
 * off/unreachable, or original videos not retained for frame extraction).
 *
 * The accepted segment is always `candidates[0]` (so `recoveredCandidateIndex`
 * is always 0), followed by up to `extraMax` untried pool alternates — same
 * `StoredCandidateSet` shape `buildCandidateHistoryEntry` produces, so the
 * existing compare UI needs no special-casing for the no-VLM case. All
 * entries are `checked: false` since no verification actually ran; the
 * accepted one is distinguished purely by `recoveredCandidateIndex` (the UI
 * already renders that as a "Used" marker independent of checked/verdict).
 *
 * Callers write this immediately after hash-matching, before any VLM pass
 * runs. If VLM does run afterward, its own richer per-segment entry
 * (`buildCandidateHistoryEntry`) simply overwrites this one for that index —
 * no coordination needed here.
 */
export function buildHashOnlyCandidateHistoryEntry(
  segmentIndex: number,
  accepted: MatchedSegment,
  candidatePool: MatchedSegment[] | undefined,
  extraMax = CANDIDATES_MAX,
): StoredCandidateSet {
  const extraCandidates = discoverBackgroundCandidates(
    candidatePool,
    accepted.shortStart,
    accepted.shortEnd,
    [accepted.movieStart],
    extraMax,
  );

  return {
    segmentIndex,
    shortStart: accepted.shortStart,
    shortEnd: accepted.shortEnd,
    recordedAt: Date.now(),
    dropped: false,
    candidates: [
      { segment: accepted, checked: false },
      ...extraCandidates.map(segment => ({ segment, checked: false })),
    ],
    recoveredCandidateIndex: 0,
  };
}

/**
 * Fire-and-forget write of a freshly-discovered candidate set to disk.
 * Never awaited by the caller — a failure here must not affect (or slow)
 * the ongoing main VLM verification pass, only be logged.
 */
export function writeCandidatesFileAsync(
  uploadDir: string,
  matchJobId: string,
  segmentIndex: number,
  entry: StoredCandidateSet,
): void {
  const filePath = matchCandidatesFilePath(uploadDir, matchJobId, segmentIndex);
  fs.promises.writeFile(filePath, JSON.stringify(entry)).catch(err => {
    console.warn(`[CandidateRecovery] Failed to write candidates for match ${matchJobId} segment ${segmentIndex}: ${err?.message || err}`);
  });
}

export function readCandidatesFile(uploadDir: string, matchJobId: string, segmentIndex: number): StoredCandidateSet | null {
  const filePath = matchCandidatesFilePath(uploadDir, matchJobId, segmentIndex);
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return null;
  }
}

export function writeCandidatesFileSync(uploadDir: string, matchJobId: string, segmentIndex: number, entry: StoredCandidateSet): void {
  try {
    fs.writeFileSync(matchCandidatesFilePath(uploadDir, matchJobId, segmentIndex), JSON.stringify(entry));
  } catch (err: any) {
    console.warn(`[CandidateRecovery] Failed to persist updated candidates for match ${matchJobId} segment ${segmentIndex}: ${err?.message || err}`);
  }
}

/** List every candidate-set file for a match job, sorted by segmentIndex ascending
 *  (== original rejection order, since the main pass iterates segments in that order). */
export function listCandidateFilesForJob(uploadDir: string, matchJobId: string): number[] {
  if (!fs.existsSync(uploadDir)) return [];
  const prefix = `${matchJobId}_candidates_`;
  const indexes: number[] = [];
  for (const file of fs.readdirSync(uploadDir)) {
    if (!file.startsWith(prefix) || !file.endsWith('.json')) continue;
    const idxStr = file.slice(prefix.length, -'.json'.length);
    const idx = Number(idxStr);
    if (Number.isFinite(idx)) indexes.push(idx);
  }
  indexes.sort((a, b) => a - b);
  return indexes;
}

export function deleteCandidateFilesForJob(uploadDir: string, matchJobId: string): void {
  for (const idx of listCandidateFilesForJob(uploadDir, matchJobId)) {
    try { fs.unlinkSync(matchCandidatesFilePath(uploadDir, matchJobId, idx)); } catch { /* ignore */ }
  }
}
