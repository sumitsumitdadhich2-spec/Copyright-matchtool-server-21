import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  CloudUpload, Video, Server, Monitor, Play, Pause, Download, Search,
  Film, ScanLine, Activity, X, AlertCircle, CheckCircle2, Layers,
  Sliders, RotateCcw, RefreshCw, ChevronDown, ChevronUp, Repeat,
  ShieldCheck, Cpu, Zap, Trash2, Database, History, ChevronLeft, ChevronRight, ListChecks,
  Plus, Minus, XCircle
} from 'lucide-react';
import { processVideoFile, processVideoOnServer } from './VideoProcessor';
import ApiSettings from './components/ApiSettings';
import { clearVideoFingerprints } from './utils/db';
import { saveJobSession, getJobSession, clearJobSession, saveMatchJobId, getMatchJobId, clearMatchJobId } from './utils/session';
import type { CachedJob } from './utils/session';
import { JobHistory } from './components/JobHistory';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FrameDetail {
  cropRegion: string;
  structureSim: number;
  colorSim: number;
  skinSim: number;
  detailSim: number;
  movieHash: string;
  shortHash: string;
}

interface MatchedSegment {
  shortStart: number;
  shortEnd: number;
  movieStart: number;
  movieEnd: number;
  confidence: number;
  frameCount: number;
  isApproximate: boolean;
  gapCount?: number;
  /**
   * Effective speed ratio of the short clip vs the reference movie.
   * 1.0 = normal speed · 0.5 = slowed (clip longer than movie section) · 2.0 = sped up
   * Computed via linear regression over matchSequence in the server engine.
   */
  speedRatio?: number;
  matchSequence: Array<{ shortTime: number; movieTime: number; similarity: number }>;
  bestFrameDetail?: FrameDetail;
  /** Server-set: every candidate for this range was VLM-rejected; this is the
   *  highest-confidence rejected candidate kept visible for manual Retry. */
  vlmRejectedKept?: boolean;
  /** Server-set (display-only): segment jumps off the dominant forward movie timeline. */
  timelineOutlier?: boolean;
}

interface UnmatchedRange {
  shortStart: number;
  shortEnd: number;
}

// Preview-only candidate data (deferred-candidate-recovery feature). Mirrors
// server/candidate-recovery.ts's StoredCandidateSet — never part of the
// primary match result JSON, fetched separately for the compare UI only.
interface CandidateCheck {
  segment: MatchedSegment;
  checked: boolean;
  verdict?: 'accepted' | 'rejected' | 'unverifiable';
  confidencePct?: number;
}

interface StoredCandidateSet {
  segmentIndex: number;
  shortStart: number;
  shortEnd: number;
  recordedAt: number;
  candidates: CandidateCheck[];
  /** Index into `candidates` that was actually used for this short-clip range. */
  recoveredCandidateIndex?: number;
  /** False when the main VLM pass accepted this range outright (comparison history only). */
  dropped: boolean;
  /** Server-reported: a manual Retry is currently running for this segment. */
  retrying?: boolean;
  /** True when the used candidate is only the best-scoring fallback after the
   *  full verification budget ran out — NOT a genuine AI-confirmed match. */
  bestEffort?: boolean;
  /** AI-written detailed description of the target clip (deep-search flow). */
  clipDescription?: string;
  /** Ranking signal the AI auto-selected for this clip. */
  recommendedMode?: 'hash' | 'embedding' | 'combined';
  /** How many deep-search description rounds have run for this segment. */
  deepSearchDepth?: number;
}

interface SanityResult {
  pass: boolean;
  totalFrames: number;
  workerAvailable?: boolean;
  workerError?: string;
  results: Array<{
    frameIndex: number;
    pass: boolean;
    hashBits: number;
    mainHashPrefix: string;
    workerHashPrefix: string;
  }>;
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

function fmt(secs: number) {
  const m = Math.floor(secs / 60);
  const s = (secs % 60).toFixed(2).padStart(5, '0');
  return `${m}:${s}`;
}

function fmtDur(secs: number) {
  if (secs < 60) return `${secs.toFixed(2)}s`;
  const m = Math.floor(secs / 60);
  const s = (secs % 60).toFixed(1);
  return `${m}m ${s}s`;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Red badge for segments where every candidate was VLM-rejected and only the
 *  best rejected candidate is kept visible — NOT a verified match. */
function RejectedKeptBadge() {
  return (
    <span
      title="Every candidate for this range was rejected by scene verification — this is only the highest-confidence rejected candidate, kept visible so you can review it and hit Retry."
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-xs font-semibold bg-red-500/10 text-red-400 border border-red-500/30"
    >
      <XCircle className="w-3 h-3" /> Rejected — Retry needed
    </span>
  );
}

/** Amber badge for segments that jump off the dominant forward movie timeline. */
function TimelineJumpBadge() {
  return (
    <span
      title="This segment jumps backwards against the forward movie timeline established by the surrounding segments — it may be a false match onto a similar-looking scene. Display-only flag; the segment is kept."
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded font-mono text-[10px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30"
    >
      <AlertCircle className="w-3 h-3" /> Timeline jump
    </span>
  );
}

function ConfidenceBadge({ confidence, isApproximate }: { confidence: number; isApproximate: boolean }) {
  if (!isApproximate && confidence >= 80) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-xs font-semibold bg-green-500/10 text-green-400 border border-green-500/25">
        <CheckCircle2 className="w-3 h-3" /> {confidence.toFixed(1)}%
      </span>
    );
  }
  if (!isApproximate && confidence >= 60) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-xs font-semibold bg-yellow-500/10 text-yellow-400 border border-yellow-500/25">
        <AlertCircle className="w-3 h-3" /> {confidence.toFixed(1)}%
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-xs font-semibold bg-orange-500/10 text-orange-400 border border-orange-500/25">
      <AlertCircle className="w-3 h-3" /> {confidence.toFixed(1)}%{isApproximate ? ' ~' : ''}
    </span>
  );
}

/**
 * Verdict + confidence for a single candidate in the compare panel's
 * candidate stepper — lets the user judge, at a glance, whether the
 * currently-shown candidate was accepted, rejected, unverifiable, or never
 * even checked (because a match was already found earlier in the pool).
 */
function CandidateVerdictBadge({ candidate, isUsed, isBestEffort }: { candidate: CandidateCheck; isUsed: boolean; isBestEffort?: boolean }) {
  const pct = candidate.confidencePct !== undefined ? ` ${candidate.confidencePct.toFixed(0)}%` : '';
  let badge: React.ReactNode;
  if (!candidate.checked) {
    badge = (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-[11px] font-semibold bg-slate-800 text-slate-500 border border-dashed border-slate-700">
        Not checked
      </span>
    );
  } else if (candidate.verdict === 'accepted') {
    badge = (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-[11px] font-semibold bg-green-500/10 text-green-400 border border-green-500/25">
        <CheckCircle2 className="w-3 h-3" /> Accepted{pct}
      </span>
    );
  } else if (candidate.verdict === 'rejected') {
    badge = (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-[11px] font-semibold bg-red-500/10 text-red-400 border border-red-500/25">
        <X className="w-3 h-3" /> Rejected{pct}
      </span>
    );
  } else {
    badge = (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-[11px] font-semibold bg-yellow-500/10 text-yellow-400 border border-yellow-500/25">
        <AlertCircle className="w-3 h-3" /> Unverifiable
      </span>
    );
  }
  return (
    <div className="flex items-center gap-1.5">
      {badge}
      {isUsed && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-[11px] font-semibold bg-indigo-500/10 text-indigo-300 border border-indigo-500/25">
          ★ Used
        </span>
      )}
      {isUsed && isBestEffort && (
        <span
          title="Highest-scoring candidate after the full verification budget ran out — NOT confirmed by AI video verification. Click Retry to search deeper."
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono text-[11px] font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/25">
          <AlertCircle className="w-3 h-3" /> Best effort — not AI-confirmed
        </span>
      )}
    </div>
  );
}

/** 16×16 perceptual hash visualized as a colored bit-grid */
function HashGrid({ hash, label, accent }: { hash: string; label: string; accent: string }) {
  const bits = hash.length >= 256 ? hash.slice(0, 256).split('') : null;
  return (
    <div className="space-y-2">
      <p className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">{label}</p>
      {bits ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(16, 1fr)', gap: '1.5px' }}>
          {bits.map((bit, i) => (
            <div key={i}
              className={`rounded-[1px] ${bit === '1' ? accent : 'bg-slate-800/80'}`}
              style={{ aspectRatio: '1' }}
            />
          ))}
        </div>
      ) : (
        <div className="text-xs text-slate-600 font-mono">No hash data</div>
      )}
      <p className="text-[9px] font-mono text-slate-700 break-all leading-tight">
        {hash ? hash.slice(0, 48) + '…' : '—'}
      </p>
    </div>
  );
}

/** Single row of the double-check checklist */
function ChecklistRow({ label, value, weight }: { label: string; value: number; weight?: string }) {
  const pct = Math.max(0, Math.min(100, value));
  const bar = pct >= 75 ? 'bg-green-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-red-500/70';
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] font-mono">
        <span className="text-slate-400">{label}{weight ? <span className="text-slate-600 ml-1">({weight})</span> : ''}</span>
        <span className={`font-bold ${pct >= 75 ? 'text-green-400' : pct >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>{pct.toFixed(1)}%</span>
      </div>
      <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-500 ${bar}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function SliderParam({
  label, hint, value, min, max, step, display, onChange, disabled
}: {
  label: string; hint: string; value: number; min: number; max: number;
  step: number; display: string; onChange: (v: number) => void; disabled?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs font-mono">
        <span className="text-slate-400">{label}</span>
        <span className="text-white font-bold">{display}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="w-full h-1.5 rounded appearance-none cursor-pointer accent-blue-500 bg-slate-700 disabled:opacity-40"
      />
      <p className="text-[10px] text-slate-600 leading-tight">{hint}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Matching progress panel (shown while /api/match SSE stream is active)
// ---------------------------------------------------------------------------

function MatchProgressPanel({ progress }: {
  progress: {
    phase: string; pct: number;
    chunkIdx?: number; totalChunks?: number;
    shortStart?: number; shortEnd?: number;
    segmentsFound?: number;
    vlmSegmentIndex?: number; vlmTotalSegments?: number; vlmVerdict?: string;
    vlmAttempt?: number; vlmTotalBudget?: number;
    startTime: number;
  };
}) {
  const { phase, pct, chunkIdx, totalChunks, shortStart, shortEnd, segmentsFound, vlmSegmentIndex, vlmTotalSegments, vlmVerdict, vlmAttempt, vlmTotalBudget, startTime } = progress;
  const elapsed = (Date.now() - startTime) / 1000;
  const eta = pct > 2 ? Math.max(0, Math.round(elapsed * (100 - pct) / pct)) : null;

  const phaseLabel: Record<string, string> = {
    loading_short: 'Analysing short clip…',
    indexing:      'Building movie frame index…',
    loading_movie: 'Loading movie fingerprints…',
    scanning:      'Scanning movie for matches…',
    matching:      'Matching scene chunks…',
    finalizing:    'Finalising results…',
    vlm_verify:    'Verifying scenes with AI…',
    vlm_deep_search:   'Deep search — AI hunting new candidates…',
    deferred_recovery: 'Recovering dropped segments…',
  };

  const showChunk = phase === 'matching' && totalChunks != null && chunkIdx != null;
  const showVlm = (phase === 'vlm_verify' || phase === 'vlm_deep_search') && vlmTotalSegments != null && vlmSegmentIndex != null;
  const isDeepSearch = phase === 'vlm_deep_search';

  return (
    <div className="space-y-2 pt-0.5">
      {/* Progress bar */}
      <div className="relative h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 bg-indigo-500 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Top row: phase label + pct + ETA */}
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span className="shrink-0">{phaseLabel[phase] ?? phase}</span>
        <div className="flex items-center gap-2 shrink-0 ml-3">
          <span className="text-indigo-400 font-medium">{pct}%</span>
          {eta !== null && (
            <span className="text-slate-500">
              ~{eta < 60 ? `${eta}s` : `${Math.round(eta / 60)}m`} bacha hai
            </span>
          )}
        </div>
      </div>

      {/* Detail row: current clip section + segments found */}
      {showChunk && (
        <div className="flex items-center justify-between text-xs rounded-md bg-slate-800/60 px-2.5 py-1.5">
          {/* Which part of the short clip is being searched */}
          <div className="flex items-center gap-1.5 min-w-0 text-slate-300">
            <span className="text-slate-500 shrink-0">Dhundh raha hai:</span>
            <span className="font-medium truncate">
              scene {chunkIdx! + 1}/{totalChunks}
              {shortStart != null && shortEnd != null && (
                <span className="text-slate-400 font-normal">
                  {' '}({fmt(shortStart)}–{fmt(shortEnd)})
                </span>
              )}
            </span>
          </div>
          {/* Live segment count */}
          <div className="shrink-0 ml-3 flex items-center gap-1">
            <span className="text-slate-500">Segments mile:</span>
            <span className={`font-bold tabular-nums ${(segmentsFound ?? 0) > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
              {segmentsFound ?? 0}
            </span>
          </div>
        </div>
      )}

      {/* Detail row: AI scene-verification pass (only if VLM is configured) */}
      {showVlm && (
        <div className={`flex items-center justify-between text-xs rounded-md px-2.5 py-1.5 ${isDeepSearch ? 'bg-purple-950/40' : 'bg-indigo-950/40'}`}>
          <div className="flex items-center gap-1.5 min-w-0 text-slate-300">
            <span className="text-slate-500 shrink-0">{isDeepSearch ? 'Deep search — segment:' : 'Verifying segment:'}</span>
            <span className="font-medium truncate">
              {vlmSegmentIndex! + 1}/{vlmTotalSegments}
            </span>
            {vlmAttempt != null && vlmTotalBudget != null && (
              <span className={`shrink-0 font-mono ${isDeepSearch ? 'text-purple-300' : 'text-slate-400'}`}>
                · candidate {vlmAttempt}/{vlmTotalBudget}
              </span>
            )}
          </div>
          {vlmVerdict && (
            <span className={`shrink-0 ml-3 font-medium capitalize ${
              vlmVerdict === 'accepted' ? 'text-emerald-400'
                : vlmVerdict === 'rejected' || vlmVerdict === 'dropped' ? 'text-orange-400'
                : 'text-slate-400'
            }`}>
              {vlmVerdict}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [processMode, setProcessMode] = useState<'browser' | 'server'>('server');
  const [showHistory, setShowHistory] = useState(false);

  // Reference video state
  const [refFile, setRefFile]       = useState<File | null>(null);
  const [refFileUrl, setRefFileUrl] = useState<string>('');
  const [refJobId, setRefJobId]     = useState<string>('');
  const [refProgress, setRefProgress] = useState({ processed: 0, total: 0, startTime: 0 });
  const [refDone, setRefDone]       = useState(false);
  const [refBatches, setRefBatches]       = useState(0);
  const [targetBatches, setTargetBatches] = useState(0);

  // Target clip state
  const [targetFile, setTargetFile]       = useState<File | null>(null);
  const [targetFileUrl, setTargetFileUrl] = useState<string>('');
  const [targetJobId, setTargetJobId]     = useState<string>('');
  const [targetProgress, setTargetProgress] = useState({ processed: 0, total: 0, startTime: 0 });
  const [targetDone, setTargetDone]       = useState(false);

  // Match params (sent to server)
  const [similarityThreshold, setSimilarityThreshold] = useState(82);
  const [minSegmentDuration, setMinSegmentDuration]   = useState(0.5);   // seconds → converted to frames
  const [frameDrift, setFrameDrift]                   = useState(3);      // extra search window frames
  const [showSettings, setShowSettings]               = useState(false);

  // Sanity test
  const [sanityRunning, setSanityRunning] = useState(false);
  const [sanityResult, setSanityResult]   = useState<SanityResult | null>(null);
  const [showSanity, setShowSanity]       = useState(false);

  // Match results
  const [segments, setSegments]           = useState<MatchedSegment[]>([]);
  const [unmatchedRanges, setUnmatched]   = useState<UnmatchedRange[]>([]);
  const [isMatching, setIsMatching]       = useState(false);
  const [matchStats, setMatchStats]       = useState<{ movieFrames: number; shortFrames: number } | null>(null);

  // Live matching progress — polled from /api/match-status/:matchJobId
  const [matchProgress, setMatchProgress] = useState<{
    phase: string; pct: number;
    chunkIdx?: number; totalChunks?: number;
    shortStart?: number; shortEnd?: number;
    segmentsFound?: number;
    vlmSegmentIndex?: number; vlmTotalSegments?: number; vlmVerdict?: string;
    vlmAttempt?: number; vlmTotalBudget?: number;
    startTime: number;
  } | null>(null);
  const [matchJobId, setMatchJobId] = useState<string>('');

  // Gemini free-tier quota status (polled alongside match-status). When the
  // daily quota is exhausted the server keeps auto-probing for the reset;
  // here we just surface a persistent warning telling the user to bring a
  // fresh API key.
  const [geminiQuota, setGeminiQuota] = useState<{
    configured: boolean;
    model?: string;
    usedToday: number;
    rpmLimit: number;
    dailyLimitReached: boolean;
    rateLimitWaiting: boolean;
    models?: {
      model: string;
      usedToday: number;
      rpdLimit: number;
      remaining: number;
      rpmLimit: number;
      dailyLimitReached: boolean;
    }[];
  } | null>(null);

  // Status / error
  const [status, setStatus]     = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string>('');

  // Processing in-flight
  const [isProcessingRef, setIsProcessingRef]       = useState(false);
  const [isProcessingTarget, setIsProcessingTarget] = useState(false);

  // Cache / session state
  const [refCached, setRefCached]           = useState(false);
  const [targetCached, setTargetCached]     = useState(false);
  const [refCachedMeta, setRefCachedMeta]   = useState<{ fileName: string; totalFrames: number } | null>(null);
  const [targetCachedMeta, setTargetCachedMeta] = useState<{ fileName: string; totalFrames: number } | null>(null);

  // Preview panel
  const [previewSegment, setPreviewSegment] = useState<MatchedSegment | null>(null);
  const [isPlaying, setIsPlaying]           = useState(false);
  const [loopSegment, setLoopSegment]       = useState(true);
  const [playbackSpeed, setPlaybackSpeed]   = useState(1.0);

  // Deferred-candidate-recovery preview data — fetched once per match job,
  // consumed purely by the Next/Previous candidate navigation below. Never
  // merged into `segments` / the primary result.
  const [candidateSets, setCandidateSets]   = useState<StoredCandidateSet[]>([]);
  const [candidateIndex, setCandidateIndex] = useState(0);

  // "View all candidates" panel (Fix 2) — keyed by segmentIndex rather than
  // holding the StoredCandidateSet object directly, so the panel always
  // reflects the latest fetch (e.g. right after a Retry completes and
  // candidateSets refreshes) instead of a stale snapshot.
  const [viewAllCandidatesForKey, setViewAllCandidatesForKey] = useState<number | null>(null);

  // Set by handleJumpToCandidate right before switching `previewSegment`, so
  // the "reset candidate index" effect below lands on the candidate the user
  // actually clicked instead of overwriting it with the recovered default.
  const pendingCandidateIndexRef = useRef<number | null>(null);

  // Manual per-segment Retry (preview UI). Keyed by segmentIndex (from the
  // segment's StoredCandidateSet) rather than the segments array position,
  // since deferred recovery can reorder/insert segments. Purely additive UI
  // state — never touches `segments`/`candidateSets` directly except via the
  // same fetch paths already used elsewhere.
  const [retryingSegments, setRetryingSegments] = useState<Set<number>>(new Set());
  const [retryError, setRetryError] = useState<string>('');

  // Inline per-segment candidate expansion (additive UI) — keyed by the
  // candidate set's segmentIndex, same keying discipline as
  // viewAllCandidatesForKey above so a Retry-triggered refetch never leaves a
  // stale panel open. Multiple segments can be expanded at once.
  const [expandedCandidateKeys, setExpandedCandidateKeys] = useState<Set<number>>(new Set());

  // In-flight "Make main segment" selection — `${segmentIndex}:${candidateIndex}`
  // while the POST is running, '' otherwise. Disables just that one button.
  const [selectingCandidateKey, setSelectingCandidateKey] = useState<string>('');

  // Manual ±1 s boundary trim (adjust-candidate feature). One adjustment in
  // flight at a time — key is `${segmentIndex}:${candidateIndex}:${edge}`.
  const [adjustingKey, setAdjustingKey] = useState<string>('');
  const [adjustError, setAdjustError] = useState<string>('');
  const [selectError, setSelectError] = useState<string>('');

  const refVideoRef  = useRef<HTMLVideoElement>(null);
  const clipVideoRef = useRef<HTMLVideoElement>(null);
  const loopRef      = useRef({ loop: true, seg: null as MatchedSegment | null });

  // Look up the StoredCandidateSet (if any) recorded for an arbitrary
  // segment's short-clip range — matched by range, not array index, since a
  // recovered segment's final position in `segments` differs from the
  // `segmentIndex` its candidate file was written under. Shared by the
  // preview panel (below) and every row of the results table (View all
  // candidates button) so there is only one place that defines "this segment
  // has candidate history".
  const findCandidateSetForSegment = (seg: MatchedSegment): StoredCandidateSet | undefined => {
    // Exact-range match first (0.05s tolerance) — the common case.
    const exact = candidateSets.find(cs =>
      Math.abs(cs.shortStart - seg.shortStart) < 0.05 &&
      Math.abs(cs.shortEnd - seg.shortEnd) < 0.05);
    if (exact) return exact;
    // Fallback: overlap match. A VLM-accepted alternate or a manual-Retry
    // swap can leave the active segment with a slightly different short-clip
    // range than the original range its candidate file was keyed under, which
    // made the exact match above miss and hid the candidate/Retry buttons.
    // Pick the candidate set with the LARGEST overlap so adjacent segments
    // can never grab each other's history.
    let best: StoredCandidateSet | undefined;
    let bestOverlap = 0;
    for (const cs of candidateSets) {
      const overlap = Math.min(cs.shortEnd, seg.shortEnd) - Math.max(cs.shortStart, seg.shortStart);
      if (overlap > bestOverlap) { bestOverlap = overlap; best = cs; }
    }
    return best;
  };

  const activeCandidateSet = previewSegment ? findCandidateSetForSegment(previewSegment) : undefined;

  // "View all candidates" panel data — resolved fresh from candidateSets on
  // every render (see viewAllCandidatesForKey declaration above) so a Retry
  // completing while the panel is open updates it live.
  const viewAllCandidatesFor = viewAllCandidatesForKey !== null
    ? candidateSets.find(cs => cs.segmentIndex === viewAllCandidatesForKey) ?? null
    : null;

  // What the Reference Movie pane actually shows: the currently-stepped
  // candidate's movie-side location when candidate navigation is in use,
  // otherwise the selected segment's own matched location. The short-clip
  // side always stays anchored to `previewSegment` — candidate stepping never
  // changes which segment is selected.
  const movieViewSegment = activeCandidateSet?.candidates[candidateIndex]?.segment ?? previewSegment;

  // Which segmentIndex a manual Retry for the currently-previewed segment
  // should target. Prefers the candidate file's own segmentIndex; when no
  // candidate set was fetched (yet), falls back to the segment's position in
  // the results array — hash-only candidate files are written under exactly
  // that index, so the Retry button stays permanently available instead of
  // disappearing whenever candidate data hasn't loaded.
  const retrySegmentIndex = activeCandidateSet
    ? activeCandidateSet.segmentIndex
    : previewSegment
      ? segments.findIndex(s =>
          Math.abs(s.shortStart - previewSegment.shortStart) < 0.05 &&
          Math.abs(s.shortEnd - previewSegment.shortEnd) < 0.05)
      : -1;

  // True while a manual Retry is running for the segment currently shown in
  // the preview — drives the spinner overlay. Independent of any other
  // segment's retry state, so navigating elsewhere is never blocked by it.
  const isCurrentSegmentRetrying = retrySegmentIndex >= 0
    ? retryingSegments.has(retrySegmentIndex) || !!activeCandidateSet?.retrying
    : false;

  // Fetch candidate data once a match job is available — and again when
  // matching finishes. `matchJobId` is set the moment a match STARTS, but
  // candidate files are only written when it completes, so a fetch keyed on
  // matchJobId alone always ran too early, got an empty list, and never
  // refetched — which is why the "View all candidates" button never showed.
  // Depending on `isMatching` too makes the completion transition
  // (isMatching → false) trigger a fresh fetch once the files exist.
  useEffect(() => {
    if (!matchJobId) { setCandidateSets([]); return; }
    let cancelled = false;
    fetch(`/api/match/${matchJobId}/candidates`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (!cancelled) setCandidateSets(data?.segments ?? []); })
      .catch(() => { if (!cancelled) setCandidateSets([]); });
    return () => { cancelled = true; };
  }, [matchJobId, isMatching]);

  // Close the "View all candidates" modal with the Escape key — click-outside
  // and the X button already work; this adds the standard keyboard path.
  useEffect(() => {
    if (viewAllCandidatesForKey === null) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setViewAllCandidatesForKey(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [viewAllCandidatesForKey]);

  // Whenever the selected segment changes, reset candidate stepping to the
  // candidate that was actually accepted (if this segment was recovered),
  // so the pane starts in sync with pair 1's selection.
  useEffect(() => {
    if (!previewSegment) { setCandidateIndex(0); return; }
    // handleJumpToCandidate (View all candidates panel) sets this just before
    // switching previewSegment when the user jumps to a specific candidate on
    // a different segment — honor that instead of the recovered default.
    if (pendingCandidateIndexRef.current !== null) {
      setCandidateIndex(pendingCandidateIndexRef.current);
      pendingCandidateIndexRef.current = null;
      return;
    }
    const cs = findCandidateSetForSegment(previewSegment);
    setCandidateIndex(cs?.recoveredCandidateIndex ?? 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewSegment, candidateSets]);

  // Keep loopRef in sync so the timeupdate handler always sees current values
  useEffect(() => {
    loopRef.current = { loop: loopSegment, seg: movieViewSegment };
  }, [loopSegment, movieViewSegment]);

  // Apply playback speed whenever it changes
  useEffect(() => {
    if (refVideoRef.current)  refVideoRef.current.playbackRate  = playbackSpeed;
    if (clipVideoRef.current) clipVideoRef.current.playbackRate = playbackSpeed;
  }, [playbackSpeed]);

  // Clean up blob URLs on unmount
  useEffect(() => {
    return () => {
      if (refFileUrl)    URL.revokeObjectURL(refFileUrl);
      if (targetFileUrl) URL.revokeObjectURL(targetFileUrl);
    };
  }, []);

  // Restore previous sessions on page load so the user doesn't have to re-upload
  useEffect(() => {
    restoreSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function restoreSessions() {
    // Restore an in-progress match job first, same as fingerprint sessions below.
    const savedMatchJobId = getMatchJobId();
    if (savedMatchJobId) {
      try {
        const res = await fetch(`/api/match-status/${savedMatchJobId}`);
        if (res.ok) {
          const job = await res.json();
          if (job.status === 'processing') {
            setMatchJobId(savedMatchJobId);
            setIsMatching(true);
            setStatus('Reconnecting to match processing…');
            pollMatchUntilDone(savedMatchJobId);
          } else if (job.status === 'completed') {
            setMatchJobId(savedMatchJobId);
            setSegments(job.segments || []);
            setUnmatched(job.unmatchedRanges || []);
            setMatchStats({ movieFrames: job.movieFrames, shortFrames: job.shortFrames });
            clearMatchJobId();
          } else {
            clearMatchJobId();
          }
        } else {
          clearMatchJobId();
        }
      } catch {
        clearMatchJobId();
      }
    }

    for (const role of ['reference', 'target'] as const) {
      const session = getJobSession(role);
      if (!session) continue;
      try {
        const res = await fetch(`/api/status/${session.jobId}`);
        if (!res.ok) { clearJobSession(role); continue; }
        const job = await res.json();
        if (job.status === 'completed') {
          applyRestoredSession(role, session, job.totalFrames || session.totalFrames);
        } else if (job.status === 'processing' || job.status === 'pending') {
          // Job still running on server — reconnect and poll
          if (role === 'reference') {
            setRefJobId(session.jobId);
            setIsProcessingRef(true);
            setStatus('Reconnecting to reference processing…');
          } else {
            setTargetJobId(session.jobId);
            setIsProcessingTarget(true);
            setStatus('Reconnecting to target processing…');
          }
          pollUntilDone(session.jobId, role, session);
        } else {
          clearJobSession(role);
        }
      } catch {
        clearJobSession(role);
      }
    }
  }

  function applyRestoredSession(
    role: 'reference' | 'target',
    session: CachedJob,
    totalFrames: number
  ) {
    if (role === 'reference') {
      setRefJobId(session.jobId);
      setRefDone(true);
      setRefCached(true);
      setRefCachedMeta({ fileName: session.fileName, totalFrames });
    } else {
      setTargetJobId(session.jobId);
      setTargetDone(true);
      setTargetCached(true);
      setTargetCachedMeta({ fileName: session.fileName, totalFrames });
    }
  }

  async function pollUntilDone(
    jobId: string,
    role: 'reference' | 'target',
    session: CachedJob
  ) {
    // Track a local start time so ETA is meaningful from the point of (re)connection.
    const pollStartTime = performance.now();
    // Consecutive network-error counter for exponential backoff.
    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 8;

    while (true) {
      // Normal interval 1 500 ms; exponential back-off on network errors (caps at ~15 s).
      const delay = consecutiveErrors === 0
        ? 1500
        : Math.min(1500 * Math.pow(2, consecutiveErrors - 1), 15_000);
      await new Promise(r => setTimeout(r, delay));

      try {
        const res = await fetch(`/api/status/${jobId}`);
        consecutiveErrors = 0; // reset back-off on any successful HTTP response

        if (!res.ok) {
          clearJobSession(role);
          if (role === 'reference') { setIsProcessingRef(false); setRefJobId(''); }
          else { setIsProcessingTarget(false); setTargetJobId(''); }
          break;
        }

        const job = await res.json();
        if (role === 'reference') setRefProgress({ processed: job.processedFrames, total: job.totalFrames, startTime: pollStartTime });
        else                      setTargetProgress({ processed: job.processedFrames, total: job.totalFrames, startTime: pollStartTime });

        if (job.status === 'completed') {
          const updated: CachedJob = { ...session, totalFrames: job.totalFrames };
          saveJobSession(role, updated);
          if (role === 'reference') {
            setIsProcessingRef(false);
            setStatus(`Reference ready: ${job.totalFrames} frames.`);
          } else {
            setIsProcessingTarget(false);
            setStatus(`Target ready: ${job.totalFrames} frames.`);
          }
          applyRestoredSession(role, updated, job.totalFrames);
          break;
        } else if (job.status === 'failed') {
          clearJobSession(role);
          if (role === 'reference') { setIsProcessingRef(false); setErrorMsg(`Reference failed: ${job.error}`); }
          else { setIsProcessingTarget(false); setErrorMsg(`Target failed: ${job.error}`); }
          break;
        } else if (job.status === 'stopped') {
          clearJobSession(role);
          if (role === 'reference') setIsProcessingRef(false);
          else setIsProcessingTarget(false);
          break;
        }
      } catch {
        // Network error (tab was backgrounded, mobile throttling, brief drop).
        consecutiveErrors++;
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          // Too many consecutive failures — stop polling but keep the session so
          // the user can reconnect via Job History → Reconnect button.
          if (role === 'reference') setIsProcessingRef(false);
          else setIsProcessingTarget(false);
          setErrorMsg('Connection lost. Use Job History → Reconnect to resume tracking progress.');
          break;
        }
        // Otherwise silently retry with backoff — do NOT show an error yet.
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Poll a background /api/match job until it finishes — mirrors pollUntilDone
  // above exactly (same interval, same backoff, same error threshold) so a
  // background tab or brief network drop behaves identically to fingerprinting.
  // ---------------------------------------------------------------------------
  async function pollMatchUntilDone(jobId: string) {
    const pollStartTime = performance.now();
    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 8;

    while (true) {
      const delay = consecutiveErrors === 0
        ? 1500
        : Math.min(1500 * Math.pow(2, consecutiveErrors - 1), 15_000);
      await new Promise(r => setTimeout(r, delay));

      try {
        const res = await fetch(`/api/match-status/${jobId}`);
        consecutiveErrors = 0;

        if (!res.ok) {
          clearMatchJobId();
          setIsMatching(false);
          setMatchProgress(null);
          setErrorMsg('Match job not found — it may have been deleted.');
          break;
        }

        const job = await res.json();
        if (job.gemini) setGeminiQuota(job.gemini);
        if (job.progress) {
          setMatchProgress({
            phase: job.progress.phase,
            pct: job.progress.pct,
            chunkIdx: job.progress.chunkIdx,
            totalChunks: job.progress.totalChunks,
            shortStart: job.progress.shortStart,
            shortEnd: job.progress.shortEnd,
            segmentsFound: job.progress.segmentsFound,
            vlmSegmentIndex: job.progress.vlmSegmentIndex,
            vlmTotalSegments: job.progress.vlmTotalSegments,
            vlmVerdict: job.progress.vlmVerdict,
            vlmAttempt: job.progress.vlmAttempt,
            vlmTotalBudget: job.progress.vlmTotalBudget,
            startTime: pollStartTime,
          });
        }

        if (job.status === 'completed') {
          clearMatchJobId();
          setMatchProgress(null);
          setSegments(job.segments || []);
          setUnmatched(job.unmatchedRanges || []);
          setMatchStats({ movieFrames: job.movieFrames, shortFrames: job.shortFrames });
          setIsMatching(false);
          const segs = job.segments || [];
          const unmatched = job.unmatchedRanges || [];
          setStatus(`Matching complete. ${segs.length} segment(s) found${unmatched.length > 0 ? `, ${unmatched.length} unmatched range(s)` : ' — full clip covered'}.`);
          break;
        } else if (job.status === 'failed') {
          clearMatchJobId();
          setMatchProgress(null);
          setIsMatching(false);
          setErrorMsg(`Match failed: ${job.error}`);
          setStatus('');
          break;
        } else if (job.status === 'stopped') {
          clearMatchJobId();
          setMatchProgress(null);
          setIsMatching(false);
          setStatus('Matching stopped.');
          break;
        }
      } catch {
        consecutiveErrors++;
        if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          setIsMatching(false);
          setErrorMsg('Connection lost. Use Job History → Reconnect to resume tracking progress.');
          break;
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Reattach to a running job from the Job History panel
  // ---------------------------------------------------------------------------
  async function handleReattach(jobId: string, type: 'fingerprint' | 'match' = 'fingerprint') {
    if (type === 'match') {
      setShowHistory(false);
      setMatchJobId(jobId);
      saveMatchJobId(jobId);
      setIsMatching(true);
      setSegments([]);
      setUnmatched([]);
      setMatchStats(null);
      setErrorMsg('');
      setStatus('Reconnecting to match processing…');
      pollMatchUntilDone(jobId);
      return;
    }
    // Determine which role this job belongs to by checking stored sessions.
    const refSession    = getJobSession('reference');
    const targetSession = getJobSession('target');

    let role: 'reference' | 'target' | null = null;
    let session: CachedJob | null = null;

    if (refSession?.jobId === jobId) {
      role = 'reference'; session = refSession;
    } else if (targetSession?.jobId === jobId) {
      role = 'target'; session = targetSession;
    }

    setShowHistory(false);

    if (!role || !session) {
      // Job is running on server but not in this browser's session — nothing to restore.
      setStatus(`Job ${jobId.slice(0, 12)}… is running on the server but is not linked to this session.`);
      return;
    }

    if (role === 'reference') {
      setRefJobId(jobId);
      setIsProcessingRef(true);
      setStatus('Reconnecting to reference processing…');
    } else {
      setTargetJobId(jobId);
      setIsProcessingTarget(true);
      setStatus('Reconnecting to target processing…');
    }

    pollUntilDone(jobId, role, session);
  }

  // ---------------------------------------------------------------------------
  // Page Visibility API — re-sync immediately when the tab comes back into focus
  // (prevents "connection lost" on mobile Chrome after backgrounding)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState !== 'visible') return;

      // Immediately re-fetch status for whichever video(s) are currently polling
      // so the counters snap back to the real server value without waiting for
      // the next 1.5 s tick.  This is read-only and does not affect processing.
      const now = performance.now();

      if (isProcessingRef && refJobId) {
        fetch(`/api/status/${refJobId}`)
          .then(r => r.json())
          .then(job => setRefProgress({ processed: job.processedFrames, total: job.totalFrames, startTime: now - (job.processedFrames / (job.totalFrames || 1)) * 5000 }))
          .catch(() => {});
      }
      if (isProcessingTarget && targetJobId) {
        fetch(`/api/status/${targetJobId}`)
          .then(r => r.json())
          .then(job => setTargetProgress({ processed: job.processedFrames, total: job.totalFrames, startTime: now - (job.processedFrames / (job.totalFrames || 1)) * 5000 }))
          .catch(() => {});
      }
      if (isMatching && matchJobId) {
        fetch(`/api/match-status/${matchJobId}`)
          .then(r => r.json())
          .then(job => {
            if (job.gemini) setGeminiQuota(job.gemini);
            if (job.progress) {
              setMatchProgress(prev => ({
                phase: job.progress.phase,
                pct: job.progress.pct,
                chunkIdx: job.progress.chunkIdx,
                totalChunks: job.progress.totalChunks,
                shortStart: job.progress.shortStart,
                shortEnd: job.progress.shortEnd,
                segmentsFound: job.progress.segmentsFound,
                vlmSegmentIndex: job.progress.vlmSegmentIndex,
                vlmTotalSegments: job.progress.vlmTotalSegments,
                vlmVerdict: job.progress.vlmVerdict,
                vlmAttempt: job.progress.vlmAttempt,
                vlmTotalBudget: job.progress.vlmTotalBudget,
                startTime: prev?.startTime ?? now,
              }));
            }
          })
          .catch(() => {});
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [isProcessingRef, refJobId, isProcessingTarget, targetJobId, isMatching, matchJobId]);

  // ---------------------------------------------------------------------------
  // Loop / segment-end detection
  // ---------------------------------------------------------------------------
  const handleRefTimeUpdate = useCallback(() => {
    const { loop, seg } = loopRef.current;
    if (!seg || !refVideoRef.current) return;
    if (refVideoRef.current.currentTime >= seg.movieEnd - 0.08) {
      // Segment ended — loop or pause
      if (loop) {
        refVideoRef.current.currentTime  = seg.movieStart;
        if (clipVideoRef.current) clipVideoRef.current.currentTime = seg.shortStart;
      } else {
        refVideoRef.current.pause();
        clipVideoRef.current?.pause();
        setIsPlaying(false);
      }
    }
  }, []);

  // ---------------------------------------------------------------------------
  // File handlers — also check for cached fingerprints on selection
  // ---------------------------------------------------------------------------
  const handleRefFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setRefFile(file);
    setRefDone(false);
    setRefJobId('');
    setRefCached(false);
    setRefCachedMeta(null);
    setSegments([]);
    setMatchStats(null);
    setPreviewSegment(null);
    setStatus('');
    if (refFileUrl) URL.revokeObjectURL(refFileUrl);
    setRefFileUrl(file ? URL.createObjectURL(file) : '');

    if (file && processMode === 'server') {
      try {
        const res = await fetch(
          `/api/lookup-video?name=${encodeURIComponent(file.name)}&size=${file.size}`
        );
        if (res.ok) {
          const { jobId, totalFrames } = await res.json();
          setRefJobId(jobId);
          setRefDone(true);
          setRefCached(true);
          setRefCachedMeta({ fileName: file.name, totalFrames });
          saveJobSession('reference', { jobId, fileName: file.name, fileSize: file.size, totalFrames, savedAt: Date.now() });
          setStatus(`Saved fingerprints found for reference (${totalFrames} frames) — extraction skipped.`);
        }
      } catch { /* lookup failure is non-fatal */ }
    }
  };

  const handleTargetFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setTargetFile(file);
    setTargetDone(false);
    setTargetJobId('');
    setTargetCached(false);
    setTargetCachedMeta(null);
    setSegments([]);
    setMatchStats(null);
    setPreviewSegment(null);
    if (targetFileUrl) URL.revokeObjectURL(targetFileUrl);
    setTargetFileUrl(file ? URL.createObjectURL(file) : '');

    if (file && processMode === 'server') {
      try {
        const res = await fetch(
          `/api/lookup-video?name=${encodeURIComponent(file.name)}&size=${file.size}`
        );
        if (res.ok) {
          const { jobId, totalFrames } = await res.json();
          setTargetJobId(jobId);
          setTargetDone(true);
          setTargetCached(true);
          setTargetCachedMeta({ fileName: file.name, totalFrames });
          saveJobSession('target', { jobId, fileName: file.name, fileSize: file.size, totalFrames, savedAt: Date.now() });
        }
      } catch { /* non-fatal */ }
    }
  };

  // ---------------------------------------------------------------------------
  // Process reference
  // ---------------------------------------------------------------------------
  const handleProcessReference = async () => {
    if (!refFile) return;

    // If fingerprints are already cached for this exact file, skip re-extraction
    if (refCached && refJobId) {
      setStatus(`Using saved fingerprints for reference (${refCachedMeta?.totalFrames ?? '?'} frames). Ready to match.`);
      return;
    }

    setIsProcessingRef(true);
    setRefDone(false);
    setRefJobId('');
    setRefCached(false);
    setRefCachedMeta(null);
    setErrorMsg('');
    setStatus(`Processing reference video${processMode === 'server' ? ' on server' : ' in browser'}…`);

    try {
      await clearVideoFingerprints('reference');
      const startTime = performance.now();
      const run = processMode === 'server' ? processVideoOnServer : processVideoFile;

      const { totalFrames, batches, jobId } = await run(refFile, 'reference', (p, t) => {
        setRefProgress({ processed: p, total: t, startTime });
      });

      setRefBatches(batches);
      setRefJobId(jobId || '');
      setRefDone(true);
      setStatus(`Reference processed: ${totalFrames} frames.`);

      if (jobId && processMode === 'server') {
        const session = { jobId, fileName: refFile.name, fileSize: refFile.size, totalFrames, savedAt: Date.now() };
        saveJobSession('reference', session);
        setRefCached(true);
        setRefCachedMeta({ fileName: refFile.name, totalFrames });
      }
    } catch (e: any) {
      setErrorMsg(`Reference error: ${e.message}`);
      setStatus('');
    } finally {
      setIsProcessingRef(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Process target + match
  // ---------------------------------------------------------------------------
  const handleRunAnalysis = async () => {
    if (!targetFile && !targetCached) return;
    setIsMatching(false);
    setSegments([]);
    setUnmatched([]);
    setMatchStats(null);
    setPreviewSegment(null);
    setErrorMsg('');

    let finalTargetJobId = targetJobId;
    let finalTotalFrames = targetCachedMeta?.totalFrames ?? 0;

    // If target fingerprints already cached, skip extraction
    if (targetCached && targetJobId) {
      setStatus('Using saved target fingerprints. Running segment matching…');
    } else if (targetFile) {
      setIsProcessingTarget(true);
      setStatus(`Processing target clip${processMode === 'server' ? ' on server' : ' in browser'}…`);

      try {
        await clearVideoFingerprints('target');
        const startTime = performance.now();
        const run = processMode === 'server' ? processVideoOnServer : processVideoFile;

        const { totalFrames, batches, jobId } = await run(targetFile, 'target', (p, t) => {
          setTargetProgress({ processed: p, total: t, startTime });
        });

        finalTargetJobId = jobId || '';
        finalTotalFrames = totalFrames;
        setTargetBatches(batches);
        setTargetJobId(finalTargetJobId);
        setTargetDone(true);
        setIsProcessingTarget(false);

        if (jobId && processMode === 'server') {
          const session = { jobId, fileName: targetFile.name, fileSize: targetFile.size, totalFrames, savedAt: Date.now() };
          saveJobSession('target', session);
          setTargetCached(true);
          setTargetCachedMeta({ fileName: targetFile.name, totalFrames });
        }
      } catch (e: any) {
        setErrorMsg(`Error: ${e.message}`);
        setStatus('');
        setIsProcessingTarget(false);
        return;
      }
    } else {
      return;
    }

    // --- matching (server mode) ---
    if (processMode === 'server') {
      if (!refJobId) {
        setErrorMsg('Reference job ID not found — please process the reference video first.');
        return;
      }
      if (!finalTargetJobId) {
        setErrorMsg('Target job ID missing — re-process the target clip.');
        return;
      }
      setIsMatching(true);
      setMatchProgress(null);
      setStatus(`Fingerprints ready (${finalTotalFrames} frames). Running segment matching…`);

      // Convert minSegmentDuration (seconds) to min consecutive frames @ 25fps
      const minConsecutiveFrames = Math.max(5, Math.round(minSegmentDuration * 25));

      try {
        // /api/match starts a background job and returns immediately; progress
        // and the final result are retrieved via polling (pollMatchUntilDone),
        // which survives tab backgrounding/reconnects the same way fingerprint
        // job polling already does.
        const matchRes = await fetch('/api/match', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            movieJobId: refJobId,
            shortJobId: finalTargetJobId,
            minSimilarity: similarityThreshold,
            minConsecutiveFrames,
            frameDrift
          })
        });

        if (!matchRes.ok) {
          const errData = await matchRes.json().catch(() => ({}));
          throw new Error(errData.error || `Match API returned ${matchRes.status}`);
        }

        const { matchJobId: newMatchJobId } = await matchRes.json();
        if (!newMatchJobId) throw new Error('Match API did not return a matchJobId');

        setMatchJobId(newMatchJobId);
        saveMatchJobId(newMatchJobId);
        await pollMatchUntilDone(newMatchJobId);
      } catch (e: any) {
        setMatchProgress(null);
        setErrorMsg(`Error: ${e.message}`);
        setStatus('');
        setIsMatching(false);
      }
      return;
    }

    // --- matching (browser mode) ---
    setIsMatching(true);
    setStatus('Running browser-side matching…');
    try {
      const { loadAllReferenceFingerprints, compareFingerprints } = await import('./Matcher');
      const refFps    = await loadAllReferenceFingerprints('reference', refBatches);
      const targetFps = await loadAllReferenceFingerprints('target', targetBatches);

      let bestSim = 0, bestMi = 0;
      for (let mi = 0; mi < refFps.length; mi += 5) {
        const sim = compareFingerprints(targetFps[0], refFps[mi]);
        if (sim > bestSim) { bestSim = sim; bestMi = mi; }
      }

      // Browser mode: rough best-frame match only — speed ratio unknown.
      setSegments([{
        shortStart: targetFps[0]?.timestamp ?? 0,
        shortEnd:   targetFps[targetFps.length - 1]?.timestamp ?? 0,
        movieStart: refFps[bestMi]?.timestamp ?? 0,
        movieEnd:   refFps[Math.min(refFps.length - 1, bestMi + targetFps.length)]?.timestamp ?? 0,
        confidence: bestSim, frameCount: targetFps.length,
        isApproximate: true, speedRatio: 1, matchSequence: []
      }]);
      setIsMatching(false);
      setStatus('Browser matching complete.');
    } catch (e: any) {
      setErrorMsg(`Error: ${e.message}`);
      setStatus('');
      setIsMatching(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Delete cached fingerprints
  // ---------------------------------------------------------------------------
  const handleDeleteRef = async () => {
    if (refJobId) {
      try { await fetch(`/api/job/${refJobId}`, { method: 'DELETE' }); } catch { /* ignore */ }
    }
    clearJobSession('reference');
    setRefJobId(''); setRefDone(false); setRefCached(false); setRefCachedMeta(null);
    setSegments([]); setMatchStats(null); setPreviewSegment(null);
    setStatus('Reference fingerprints deleted.');
  };

  const handleDeleteTarget = async () => {
    if (targetJobId) {
      try { await fetch(`/api/job/${targetJobId}`, { method: 'DELETE' }); } catch { /* ignore */ }
    }
    clearJobSession('target');
    setTargetJobId(''); setTargetDone(false); setTargetCached(false); setTargetCachedMeta(null);
    setSegments([]); setMatchStats(null); setPreviewSegment(null);
    setStatus('Target fingerprints deleted.');
  };

  // ---------------------------------------------------------------------------
  // Preview: seek + auto-play
  // ---------------------------------------------------------------------------
  // play() returns a promise that rejects with AbortError if a pause() (or a
  // new load/seek) interrupts it before playback starts — e.g. when the user
  // quickly steps to another segment/candidate. Swallow that rejection so it
  // never surfaces as an unhandled runtime error.
  const safePlay = (video: HTMLVideoElement | null) => {
    video?.play().catch(() => { /* interrupted by pause/seek — safe to ignore */ });
  };

  // `movieSeekOverride` lets callers (e.g. jumping to a specific candidate
  // from the "View all candidates" panel) land the reference video on a
  // different movie timestamp than this segment's own accepted match,
  // without touching the short-clip side (which always stays anchored to
  // the segment, per `movieViewSegment`'s contract elsewhere in this file).
  const handlePreviewSegment = (seg: MatchedSegment, movieSeekOverride?: number) => {
    setPreviewSegment(seg);
    setIsPlaying(false);
    setTimeout(() => {
      if (refVideoRef.current) {
        refVideoRef.current.currentTime  = movieSeekOverride ?? seg.movieStart;
        refVideoRef.current.playbackRate = playbackSpeed;
      }
      if (clipVideoRef.current) {
        clipVideoRef.current.currentTime  = seg.shortStart;
        clipVideoRef.current.playbackRate = playbackSpeed;
      }
      // Auto-play after seeking
      setTimeout(() => {
        safePlay(refVideoRef.current);
        safePlay(clipVideoRef.current);
        setIsPlaying(true);
      }, 200);
    }, 100);
  };

  const handleSyncPlay = () => {
    if (!refVideoRef.current || !clipVideoRef.current) return;
    if (isPlaying) {
      refVideoRef.current.pause();
      clipVideoRef.current.pause();
      setIsPlaying(false);
    } else {
      refVideoRef.current.playbackRate  = playbackSpeed;
      clipVideoRef.current.playbackRate = playbackSpeed;
      safePlay(refVideoRef.current);
      safePlay(clipVideoRef.current);
      setIsPlaying(true);
    }
  };

  const handleRestartPreview = () => {
    if (!previewSegment) return;
    const movieSeg = activeCandidateSet?.candidates[candidateIndex]?.segment ?? previewSegment;
    if (refVideoRef.current)  refVideoRef.current.currentTime  = movieSeg.movieStart;
    if (clipVideoRef.current) clipVideoRef.current.currentTime = previewSegment.shortStart;
    safePlay(refVideoRef.current);
    safePlay(clipVideoRef.current);
    setIsPlaying(true);
  };

  // ---------------------------------------------------------------------------
  // Preview navigation — segment Next/Previous (pair 1) and candidate
  // Next/Previous (pair 2). UI-only: consumes existing `segments` /
  // candidate data, never re-fetches or recomputes matches/candidates.
  // ---------------------------------------------------------------------------
  const previewIndex = previewSegment ? segments.indexOf(previewSegment) : -1;

  const handleSegmentStep = (dir: 1 | -1) => {
    if (previewIndex === -1) return;
    const nextIdx = previewIndex + dir;
    if (nextIdx < 0 || nextIdx >= segments.length) return;
    handlePreviewSegment(segments[nextIdx]);
  };

  const handleCandidateStep = (dir: 1 | -1) => {
    if (!activeCandidateSet) return;
    const nextIdx = candidateIndex + dir;
    if (nextIdx < 0 || nextIdx >= activeCandidateSet.candidates.length) return;
    const seg = activeCandidateSet.candidates[nextIdx].segment;
    setCandidateIndex(nextIdx);
    setIsPlaying(false);
    refVideoRef.current?.pause();
    clipVideoRef.current?.pause();
    if (refVideoRef.current)  refVideoRef.current.currentTime  = seg.movieStart;
    if (clipVideoRef.current) clipVideoRef.current.currentTime = previewSegment?.shortStart ?? seg.shortStart;
  };

  // ---------------------------------------------------------------------------
  // Jump to a specific candidate from the "View all candidates" panel.
  // Sets candidateIndex on the correct segment and closes the panel.
  // ---------------------------------------------------------------------------
  const handleJumpToCandidate = useCallback((cs: StoredCandidateSet, idx: number) => {
    const candidate = cs.candidates[idx];
    if (!candidate) return;
    // Find the live segment that matches this candidate set's short-clip range
    const seg = segments.find(s =>
      Math.abs(s.shortStart - cs.shortStart) < 0.05 &&
      Math.abs(s.shortEnd - cs.shortEnd) < 0.05);

    if (seg && seg !== previewSegment) {
      // Switching to a different segment. pendingCandidateIndexRef makes the
      // "reset candidate index" effect land on this candidate instead of the
      // recovered default, and movieSeekOverride seeks the reference video
      // straight to THIS candidate's location — not the segment's own
      // accepted match, which is what handlePreviewSegment would otherwise
      // seek to on its own.
      pendingCandidateIndexRef.current = idx;
      handlePreviewSegment(seg, candidate.segment.movieStart);
    } else if (seg) {
      // Same segment already previewed — the "previewSegment changed" effect
      // won't fire again (same object reference), so update candidateIndex
      // and reposition the video directly here, mirroring handleCandidateStep.
      setCandidateIndex(idx);
      setIsPlaying(false);
      refVideoRef.current?.pause();
      clipVideoRef.current?.pause();
      if (refVideoRef.current)  refVideoRef.current.currentTime  = candidate.segment.movieStart;
      if (clipVideoRef.current) clipVideoRef.current.currentTime = seg.shortStart;
    } else {
      // Segment was dropped from results (all candidates rejected) — no
      // previewSegment to switch to, just reposition the reference video.
      setCandidateIndex(idx);
      setIsPlaying(false);
      refVideoRef.current?.pause();
      if (refVideoRef.current) refVideoRef.current.currentTime = candidate.segment.movieStart;
    }
    setViewAllCandidatesForKey(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segments, previewSegment]);

  // ---------------------------------------------------------------------------
  // "Make main segment" — user promotes a candidate they visually verified to
  // be the active match for its short-clip range. Purely additive: calls the
  // new select-candidate endpoint (which reuses the exact segment-swap logic
  // Retry already uses on acceptance), then refreshes state through the same
  // fetch paths the rest of the app already uses. Never touches matching.
  // ---------------------------------------------------------------------------
  const handleMakeMainCandidate = async (cs: StoredCandidateSet, idx: number) => {
    if (!matchJobId) { setSelectError('No active match job — re-run matching to enable selection.'); return; }
    const key = `${cs.segmentIndex}:${idx}`;
    if (selectingCandidateKey) return; // one selection at a time keeps state simple & safe
    setSelectError('');
    setSelectingCandidateKey(key);
    try {
      const res = await fetch(`/api/match/${matchJobId}/segment/${cs.segmentIndex}/select-candidate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidateIndex: idx }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setSelectError(err?.error || 'Could not make this candidate the main segment.');
        return;
      }
      const data = await res.json();
      const newSegments: MatchedSegment[] = data.segments || [];
      setSegments(newSegments);
      await refreshCandidateSets();

      // Keep the preview in sync if the previewed segment covered this range —
      // same range-overlap sync the Retry flow performs after a swap.
      if (previewSegment) {
        const overlapsPreview =
          Math.min(previewSegment.shortEnd, cs.shortEnd) - Math.max(previewSegment.shortStart, cs.shortStart) > 0;
        if (overlapsPreview) {
          const promoted = newSegments.find(s =>
            Math.min(s.shortEnd, cs.shortEnd) - Math.max(s.shortStart, cs.shortStart) > 0);
          if (promoted && promoted !== previewSegment) {
            pendingCandidateIndexRef.current = idx;
            setPreviewSegment(promoted);
          }
        }
      }
      setStatus(`Candidate ${idx + 1} is now the main match for segment range ${fmt(cs.shortStart)}–${fmt(cs.shortEnd)}.`);
    } catch {
      setSelectError('Network error while making this candidate the main segment.');
    } finally {
      setSelectingCandidateKey('');
    }
  };

  // ---------------------------------------------------------------------------
  // Manual ±1 s candidate boundary trim — user nudges the movie-side start or
  // end of a candidate they like. Purely additive: calls the adjust-candidate
  // endpoint, then refreshes through the exact same fetch paths the rest of
  // the app already uses. Never touches matching. The timeline is
  // one-directional, so the server clamps at 0:00 and enforces a minimum
  // segment length — the UI just surfaces those errors.
  // ---------------------------------------------------------------------------
  const handleAdjustCandidate = async (cs: StoredCandidateSet, idx: number, edge: 'start' | 'end', deltaSec: 1 | -1) => {
    if (!matchJobId) { setAdjustError('No active match job — re-run matching to enable adjustments.'); return; }
    if (adjustingKey || selectingCandidateKey) return; // one mutation at a time keeps state simple & safe
    const key = `${cs.segmentIndex}:${idx}:${edge}`;
    setAdjustError('');
    setAdjustingKey(key);
    try {
      const res = await fetch(`/api/match/${matchJobId}/segment/${cs.segmentIndex}/adjust-candidate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidateIndex: idx, edge, deltaSec }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setAdjustError(err?.error || 'Could not adjust this candidate.');
        return;
      }
      const data = await res.json();
      const updatedSeg: MatchedSegment | undefined = data?.entry?.candidates?.[idx]?.segment;

      // Refresh candidate history so every list/stepper shows the new bounds.
      await refreshCandidateSets();

      // If the adjusted candidate was the main match, the server also updated
      // the active segments — re-link previewSegment to its new object so
      // Next/Previous and the stepper stay consistent (same pattern as
      // handleMakeMainCandidate).
      if (data.segmentsUpdated && Array.isArray(data.segments)) {
        const newSegments: MatchedSegment[] = data.segments;
        setSegments(newSegments);
        if (previewSegment) {
          const relinked = newSegments.find(s =>
            Math.abs(s.shortStart - previewSegment.shortStart) < 0.05 &&
            Math.abs(s.shortEnd - previewSegment.shortEnd) < 0.05);
          if (relinked && relinked !== previewSegment) {
            pendingCandidateIndexRef.current = candidateIndex;
            setPreviewSegment(relinked);
          }
        }
      }

      // Make the preview react instantly to the adjustment: if the adjusted
      // candidate is the one currently shown in the compare panel, seek ONLY
      // the reference (movie) video to the edge that just moved — the new
      // START, or a couple of seconds before the new END. The short/clip
      // video is deliberately left untouched: the user is only changing the
      // movie side, the short stays exactly as it is.
      if (updatedSeg &&
          activeCandidateSet?.segmentIndex === cs.segmentIndex && candidateIndex === idx &&
          refVideoRef.current) {
        if (edge === 'start') {
          refVideoRef.current.currentTime = updatedSeg.movieStart;
        } else {
          refVideoRef.current.currentTime = Math.max(updatedSeg.movieStart, updatedSeg.movieEnd - 2);
        }
      }

      if (updatedSeg) {
        setStatus(`Candidate ${idx + 1} ${edge === 'start' ? 'start' : 'end'} moved ${deltaSec > 0 ? '+1s' : '-1s'} → movie ${fmt(updatedSeg.movieStart)}–${fmt(updatedSeg.movieEnd)}.`);
      }
    } catch {
      setAdjustError('Network error while adjusting this candidate.');
    } finally {
      setAdjustingKey('');
    }
  };

  // Compact [−|+ Start … End −|+] button group rendered next to a candidate.
  // Left "+" extends the segment 1 s earlier (start − 1), left "−" shrinks it
  // (start + 1); right "+" extends 1 s later (end + 1), right "−" shrinks it
  // (end − 1). Reused by the preview stepper and the inline candidate list.
  const renderTrimControls = (cs: StoredCandidateSet, idx: number, disabled: boolean) => {
    const seg = cs.candidates[idx]?.segment;
    if (!seg) return null;
    const busy = !!adjustingKey || disabled;
    const btnCls = 'inline-flex items-center justify-center w-6 h-6 rounded-md border border-slate-700 bg-slate-800/80 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-slate-800/80 text-slate-300 transition cursor-pointer';
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-cyan-500/20 bg-cyan-500/5 px-2 py-1">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-cyan-300/70">Start</span>
        <button onClick={() => handleAdjustCandidate(cs, idx, 'start', -1)} disabled={busy || seg.movieStart < 1}
          title="Extend start 1 s earlier (e.g. 4s → 3s)" className={btnCls}>
          <Plus className="w-3 h-3" />
        </button>
        <button onClick={() => handleAdjustCandidate(cs, idx, 'start', 1)} disabled={busy || seg.movieEnd - seg.movieStart < 1.5}
          title="Shrink from the start by 1 s" className={btnCls}>
          <Minus className="w-3 h-3" />
        </button>
        <span className="font-mono text-[11px] text-cyan-200/90 px-1 whitespace-nowrap">
          {fmt(seg.movieStart)}<span className="text-slate-600 mx-0.5">→</span>{fmt(seg.movieEnd)}
        </span>
        <button onClick={() => handleAdjustCandidate(cs, idx, 'end', -1)} disabled={busy || seg.movieEnd - seg.movieStart < 1.5}
          title="Shrink from the end by 1 s" className={btnCls}>
          <Minus className="w-3 h-3" />
        </button>
        <button onClick={() => handleAdjustCandidate(cs, idx, 'end', 1)} disabled={busy}
          title="Extend end 1 s later (e.g. 6s → 7s)" className={btnCls}>
          <Plus className="w-3 h-3" />
        </button>
        <span className="text-[10px] font-semibold uppercase tracking-wider text-cyan-300/70">End</span>
        {adjustingKey.startsWith(`${cs.segmentIndex}:${idx}:`) && (
          <RefreshCw className="w-3 h-3 text-cyan-300 animate-spin" />
        )}
      </div>
    );
  };

  const toggleCandidateExpansion = (segmentKey: number) => {
    setExpandedCandidateKeys(prev => {
      const next = new Set(prev);
      if (next.has(segmentKey)) next.delete(segmentKey);
      else next.add(segmentKey);
      return next;
    });
  };

  // ---------------------------------------------------------------------------
  // Manual per-segment Retry — user-triggered, independent of whatever verdict
  // the automatic pipeline already reached. Kicks off the server-side retry,
  // then polls the same per-segment candidates endpoint (now carrying a
  // `retrying` flag) until it reports done, refreshing both the full
  // candidate history and (if the segment's active match changed) the main
  // match-status result so the preview reflects the newly-accepted match.
  // ---------------------------------------------------------------------------
  const refreshCandidateSets = useCallback(async () => {
    if (!matchJobId) return;
    try {
      const r = await fetch(`/api/match/${matchJobId}/candidates`);
      if (r.ok) {
        const data = await r.json();
        setCandidateSets(data?.segments ?? []);
      }
    } catch { /* transient — next poll/refresh will retry */ }
  }, [matchJobId]);

  const handleRetrySegment = async () => {
    if (!matchJobId || retrySegmentIndex < 0) return;
    const segmentIndex = retrySegmentIndex;
    setRetryError('');
    setRetryingSegments(prev => new Set(prev).add(segmentIndex));

    try {
      const startRes = await fetch(`/api/match/${matchJobId}/segment/${segmentIndex}/retry`, { method: 'POST' });
      if (!startRes.ok) {
        const err = await startRes.json().catch(() => ({}));
        setRetryError(err?.error || 'Could not start Retry for this segment.');
        setRetryingSegments(prev => { const next = new Set(prev); next.delete(segmentIndex); return next; });
        return;
      }

      // Poll the per-segment endpoint until the server reports it is no
      // longer running — same fetch-and-poll shape as pollMatchUntilDone,
      // just scoped to one segment instead of the whole job.
      let consecutiveErrors = 0;
      while (true) {
        await new Promise(r => setTimeout(r, 1500));
        try {
          const res = await fetch(`/api/match/${matchJobId}/candidates/${segmentIndex}`);
          if (!res.ok) { consecutiveErrors++; if (consecutiveErrors >= 8) break; continue; }
          consecutiveErrors = 0;
          const entry: StoredCandidateSet = await res.json();
          if (!entry.retrying) break;
        } catch {
          consecutiveErrors++;
          if (consecutiveErrors >= 8) break;
        }
      }

      // Refresh candidate history (always) and the main match result (in
      // case this segment's active match changed) so the rest of the UI —
      // segment list, stepper, preview — reflects the outcome.
      await refreshCandidateSets();
      const statusRes = await fetch(`/api/match-status/${matchJobId}`);
      if (statusRes.ok) {
        const job = await statusRes.json();
        const newSegments = job.segments || [];
        setSegments(newSegments);
        // If the currently-previewed segment's range now points at a
        // different (or newly-added) active match, keep the preview in sync.
        if (previewSegment) {
          const stillThere = newSegments.find((s: MatchedSegment) =>
            Math.abs(s.shortStart - previewSegment.shortStart) < 0.05 &&
            Math.abs(s.shortEnd - previewSegment.shortEnd) < 0.05);
          if (stillThere && stillThere !== previewSegment) setPreviewSegment(stillThere);
        }
      }
    } finally {
      setRetryingSegments(prev => { const next = new Set(prev); next.delete(segmentIndex); return next; });
    }
  };

  // ---------------------------------------------------------------------------
  // Sanity test
  // ---------------------------------------------------------------------------
  const handleSanityTest = async () => {
    setSanityRunning(true);
    setSanityResult(null);
    try {
      const res = await fetch('/api/sanity-test', { method: 'POST' });
      const data = await res.json();
      setSanityResult(data);
    } catch (e: any) {
      setSanityResult({ pass: false, totalFrames: 10, results: [] });
    } finally {
      setSanityRunning(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Download JSON
  // ---------------------------------------------------------------------------
  const handleDownloadJson = () => {
    const payload = { segments, matchStats, params: { similarityThreshold, minSegmentDuration } };
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2));
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = 'match_results.json';
    a.click();
  };

  // ---------------------------------------------------------------------------
  // Progress renderer
  // ---------------------------------------------------------------------------
  const renderProgress = (prog: typeof refProgress, accent: string) => {
    if (prog.total === 0 && prog.processed === 0) return null;
    const elapsed   = (performance.now() - prog.startTime) / 1000;
    const fps       = prog.processed / (elapsed || 1);
    const remaining = prog.total - prog.processed;
    const eta       = fps > 0 ? remaining / fps : 0;
    const etaStr    = isFinite(eta) ? `${Math.floor(eta / 60)}m ${Math.round(eta % 60)}s` : '…';
    const pct       = prog.total > 0 ? Math.min(100, Math.round((prog.processed / prog.total) * 100)) : 0;
    return (
      <div className="mt-3 space-y-1.5">
        <div className="flex justify-between text-xs font-mono text-slate-400">
          <span>{prog.processed.toLocaleString()} / {prog.total ? prog.total.toLocaleString() : '?'} frames</span>
          <span>{fps.toFixed(1)} fps · ETA {etaStr}</span>
        </div>
        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <div className={`h-full transition-all duration-300 ${accent}`} style={{ width: `${pct}%` }} />
        </div>
      </div>
    );
  };

  // ---------------------------------------------------------------------------
  // Derived
  // ---------------------------------------------------------------------------
  const canRunAnalysis = (!!targetFile || (targetCached && !!targetJobId)) && refDone && !isProcessingRef && !isProcessingTarget && !isMatching;
  const busy = isProcessingRef || isProcessingTarget || isMatching;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="min-h-screen bg-[#0d1117] text-slate-300 font-sans">
      <div className="max-w-5xl mx-auto p-6 md:p-10 space-y-5">

        {/* ── Header ── */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-slate-800 pb-6">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
              <ScanLine className="w-7 h-7 text-blue-500" />
              Shiva Video Match
            </h1>
            <p className="text-slate-500 text-sm mt-1">Sequence-alignment video fingerprint matching</p>
          </div>
          <div className="flex items-center gap-2">
            {/* Job History toggle */}
            <button
              onClick={() => setShowHistory(h => !h)}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-lg border transition-all cursor-pointer ${showHistory ? 'bg-violet-600 border-violet-500 text-white shadow' : 'bg-slate-900 border-slate-700 text-slate-400 hover:text-white hover:border-slate-600'}`}
            >
              <History className="w-3.5 h-3.5" />
              History
            </button>

            {/* Browser / Server mode toggle */}
            <div className="flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800">
              <button onClick={() => setProcessMode('browser')}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${processMode === 'browser' ? 'bg-slate-700 text-white shadow ring-1 ring-slate-600' : 'text-slate-400 hover:text-white'}`}>
                <Monitor className="w-3.5 h-3.5" /> Browser
              </button>
              <button onClick={() => setProcessMode('server')}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${processMode === 'server' ? 'bg-blue-600 text-white shadow ring-1 ring-blue-500' : 'text-slate-400 hover:text-white'}`}>
                <Server className="w-3.5 h-3.5" /> Server
              </button>
            </div>
          </div>
        </div>

        {/* ── Error banner ── */}
        {errorMsg && (
          <div className="flex items-start gap-3 bg-red-950/40 border border-red-800/50 rounded-xl p-4 text-red-300 text-sm">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0 text-red-400" />
            <span>{errorMsg}</span>
            <button onClick={() => setErrorMsg('')} className="ml-auto shrink-0 text-red-500 hover:text-red-300 cursor-pointer">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* ── Job History panel ── */}
        {showHistory && (
          <JobHistory onClose={() => setShowHistory(false)} onReattach={handleReattach} />
        )}

        {/* ── Saved Sessions panel ── */}
        {(refCached || targetCached) && (
          <section className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-teal-400" />
              <h3 className="text-sm font-semibold text-slate-300">Saved Fingerprints</h3>
              <span className="text-xs text-slate-600 font-mono ml-1">— persists until deleted</span>
            </div>
            <div className="space-y-2">
              {refCached && refCachedMeta && (
                <div className="flex items-center justify-between gap-3 bg-slate-800/50 rounded-xl px-3 py-2.5">
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-slate-300 truncate">{refCachedMeta.fileName}</p>
                    <p className="text-[10px] text-slate-500 font-mono">{refCachedMeta.totalFrames.toLocaleString()} frames · Reference</p>
                  </div>
                  <button onClick={handleDeleteRef} disabled={busy}
                    className="shrink-0 flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-medium text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg border border-red-500/20 transition disabled:opacity-40 cursor-pointer">
                    <Trash2 className="w-3 h-3" /> Delete
                  </button>
                </div>
              )}
              {targetCached && targetCachedMeta && (
                <div className="flex items-center justify-between gap-3 bg-slate-800/50 rounded-xl px-3 py-2.5">
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-slate-300 truncate">{targetCachedMeta.fileName}</p>
                    <p className="text-[10px] text-slate-500 font-mono">{targetCachedMeta.totalFrames.toLocaleString()} frames · Target</p>
                  </div>
                  <button onClick={handleDeleteTarget} disabled={busy}
                    className="shrink-0 flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-medium text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg border border-red-500/20 transition disabled:opacity-40 cursor-pointer">
                    <Trash2 className="w-3 h-3" /> Delete
                  </button>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ── Step 1: Reference ── */}
        <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="bg-blue-500/10 p-2 rounded-lg border border-blue-500/20">
              <Film className="w-4 h-4 text-blue-400" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Step 1 — Reference Movie</h2>
              <p className="text-xs text-slate-500">The full-length video to search within</p>
            </div>
            <div className="ml-auto flex items-center gap-2">
              {refDone && refCached && (
                <span className="flex items-center gap-1 text-xs text-teal-400 font-medium">
                  <Database className="w-3 h-3" /> Cached
                </span>
              )}
              {refDone && !refCached && (
                <span className="flex items-center gap-1 text-xs text-green-400 font-medium">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                </span>
              )}
            </div>
          </div>

          {/* Show cached-file banner when session was restored without a file */}
          {refCached && !refFile && refCachedMeta && (
            <div className="flex items-center gap-3 bg-teal-950/30 border border-teal-700/30 rounded-xl px-4 py-3">
              <Database className="w-4 h-4 text-teal-400 shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium text-teal-300 truncate">{refCachedMeta.fileName}</p>
                <p className="text-[10px] text-teal-600 font-mono">{refCachedMeta.totalFrames.toLocaleString()} frames saved · select the file below only if you want to preview video</p>
              </div>
            </div>
          )}

          <div className="relative group">
            <input type="file" accept="video/mp4" onChange={handleRefFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
            <div className={`flex items-center gap-3 p-4 border-2 border-dashed rounded-xl transition-colors ${refFile ? 'border-blue-500/40 bg-blue-500/5' : 'border-slate-700 bg-slate-800/40 group-hover:border-slate-600'}`}>
              <CloudUpload className={`w-5 h-5 shrink-0 ${refFile ? 'text-blue-400' : 'text-slate-500'}`} />
              <div className="min-w-0">
                <p className="text-sm font-medium text-slate-300 truncate">
                  {refFile ? refFile.name : (refCached ? 'Select file to re-extract (optional — cached fingerprints are ready)' : 'Drop reference video here or click to browse')}
                </p>
                {refFile && <p className="text-xs text-slate-500">{(refFile.size / 1024 / 1024).toFixed(1)} MB</p>}
              </div>
            </div>
          </div>

          <button onClick={handleProcessReference} disabled={(!refFile && !refCached) || isProcessingRef}
            className="w-full flex justify-center items-center gap-2 py-2.5 px-4 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl text-sm font-medium transition-colors cursor-pointer">
            {isProcessingRef
              ? <><RefreshCw className="w-4 h-4 animate-spin" /> Processing…</>
              : refCached && refJobId
              ? <><CheckCircle2 className="w-4 h-4" /> Using Saved Fingerprints</>
              : <><Activity className="w-4 h-4" /> Extract Fingerprints</>}
          </button>
          {renderProgress(refProgress, 'bg-blue-500')}
        </section>

        {/* ── Settings Panel ── */}
        <section className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
          <button
            onClick={() => setShowSettings(s => !s)}
            className="w-full flex items-center justify-between px-6 py-4 hover:bg-slate-800/50 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <div className="bg-purple-500/10 p-2 rounded-lg border border-purple-500/20">
                <Sliders className="w-4 h-4 text-purple-400" />
              </div>
              <div className="text-left">
                <h2 className="text-base font-semibold text-white">Match Parameters</h2>
                <p className="text-xs text-slate-500">
                  Confidence ≥{similarityThreshold}% · Min duration {minSegmentDuration.toFixed(1)}s · Drift ±{frameDrift}f
                </p>
              </div>
            </div>
            {showSettings ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
          </button>

          {showSettings && (
            <div className="px-6 pb-6 pt-2 border-t border-slate-800 grid grid-cols-1 sm:grid-cols-2 gap-6">
              <SliderParam
                label="Confidence Threshold"
                hint="Minimum similarity % for a frame to count as a match. Lower = catch more (but more false positives). Default: 82%."
                value={similarityThreshold}
                min={20} max={95} step={1}
                display={`≥ ${similarityThreshold}%`}
                onChange={setSimilarityThreshold}
                disabled={busy}
              />
              <SliderParam
                label="Min Segment Duration"
                hint="Shortest accepted match sequence. Raise to filter out brief blips. Default: 0.5s."
                value={minSegmentDuration}
                min={0.2} max={5.0} step={0.1}
                display={`${minSegmentDuration.toFixed(1)}s`}
                onChange={setMinSegmentDuration}
                disabled={busy}
              />
              <SliderParam
                label="Sequence Frame Drift"
                hint="Extra frames added to the per-step search window. Allows matching clips with minor frame drops or slight speed drift. Default: ±3 frames."
                value={frameDrift}
                min={0} max={10} step={1}
                display={`± ${frameDrift} frame${frameDrift !== 1 ? 's' : ''}`}
                onChange={setFrameDrift}
                disabled={busy}
              />
            </div>
          )}
        </section>

        {/* ── API & GPU Connection ── */}
        <ApiSettings />

        {/* ── Worker Accuracy Calibration ── */}
        <section className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
          <button
            onClick={() => setShowSanity(s => !s)}
            className="w-full flex items-center justify-between px-6 py-4 hover:bg-slate-800/50 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <div className="bg-cyan-500/10 p-2 rounded-lg border border-cyan-500/20">
                <Cpu className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="text-left">
                <h2 className="text-base font-semibold text-white">Worker Accuracy Calibration</h2>
                <p className="text-xs text-slate-500">Verify 100% hash integrity between Main Thread and Web Worker</p>
              </div>
              {sanityResult && (
                <span className={`ml-2 flex items-center gap-1 text-xs font-semibold ${sanityResult.pass ? 'text-green-400' : 'text-red-400'}`}>
                  {sanityResult.pass ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                  {sanityResult.pass ? 'PASS' : 'FAIL'}
                </span>
              )}
            </div>
            {showSanity ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
          </button>

          {showSanity && (
            <div className="px-6 pb-6 pt-2 border-t border-slate-800 space-y-4">
              <p className="text-xs text-slate-500 leading-relaxed">
                Executes side-by-side processing of 10 complex test frames on the <span className="text-cyan-400 font-medium">Main Thread</span> vs.{' '}
                <span className="text-purple-400 font-medium">Web Worker</span> to guarantee 100% hash value integrity.
              </p>

              <button
                onClick={handleSanityTest}
                disabled={sanityRunning}
                className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition cursor-pointer"
              >
                {sanityRunning
                  ? <><RefreshCw className="w-4 h-4 animate-spin" /> Running…</>
                  : <><ShieldCheck className="w-4 h-4" /> Run Sanity Test</>}
              </button>

              {sanityResult && (
                <div className="space-y-3">
                  <div className={`flex items-center gap-3 p-3 rounded-xl border text-sm font-semibold ${sanityResult.pass ? 'bg-green-900/20 border-green-700/40 text-green-300' : 'bg-red-900/20 border-red-700/40 text-red-300'}`}>
                    {sanityResult.pass
                      ? <><CheckCircle2 className="w-4 h-4" /> All {sanityResult.totalFrames} frames passed — 256-bit hash determinism verified{sanityResult.workerAvailable ? ' (Main Thread ≡ Worker)' : ' (Main Thread — worker path requires ffmpeg install)'}</>
                      : <><AlertCircle className="w-4 h-4" /> Hash mismatch detected — worker output differs from main thread</>}
                  </div>

                  <div className="space-y-1.5 max-h-52 overflow-y-auto pr-1">
                    {sanityResult.results.map(r => (
                      <div key={r.frameIndex} className={`grid grid-cols-[auto_1fr_1fr] gap-3 p-2 rounded-lg text-[10px] font-mono border ${r.pass ? 'bg-slate-800/40 border-slate-800' : 'bg-red-900/20 border-red-700/30'}`}>
                        <span className={`font-bold self-center ${r.pass ? 'text-green-400' : 'text-red-400'}`}>
                          F{r.frameIndex + 1} {r.pass ? '✓' : '✗'}
                        </span>
                        <div>
                          <p className="text-cyan-500/70 mb-0.5">Main Thread ({r.hashBits}-bit)</p>
                          <p className="text-slate-400 break-all">{r.mainHashPrefix}…</p>
                        </div>
                        <div>
                          <p className="text-purple-500/70 mb-0.5">Worker</p>
                          <p className={`break-all ${r.pass ? 'text-slate-400' : 'text-red-400'}`}>{r.workerHashPrefix}…</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {/* ── Step 2: Target + Match ── */}
        <section className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-500/10 p-2 rounded-lg border border-indigo-500/20">
              <Search className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Step 2 — Target Clip &amp; Find Matches</h2>
              <p className="text-xs text-slate-500">Upload the clip to locate inside the reference</p>
            </div>
            <div className="ml-auto flex items-center gap-2">
              {targetDone && targetCached && (
                <span className="flex items-center gap-1 text-xs text-teal-400 font-medium">
                  <Database className="w-3 h-3" /> Cached
                </span>
              )}
              {targetDone && !targetCached && (
                <span className="flex items-center gap-1 text-xs text-green-400 font-medium">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Processed
                </span>
              )}
            </div>
          </div>

          {/* Cached-file banner when session restored without a file */}
          {targetCached && !targetFile && targetCachedMeta && (
            <div className="flex items-center gap-3 bg-teal-950/30 border border-teal-700/30 rounded-xl px-4 py-3">
              <Database className="w-4 h-4 text-teal-400 shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium text-teal-300 truncate">{targetCachedMeta.fileName}</p>
                <p className="text-[10px] text-teal-600 font-mono">{targetCachedMeta.totalFrames.toLocaleString()} frames saved · ready to match</p>
              </div>
            </div>
          )}

          <div className="relative group">
            <input type="file" accept="video/mp4" onChange={handleTargetFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
            <div className={`flex items-center gap-3 p-4 border-2 border-dashed rounded-xl transition-colors ${targetFile ? 'border-indigo-500/40 bg-indigo-500/5' : 'border-slate-700 bg-slate-800/40 group-hover:border-slate-600'}`}>
              <CloudUpload className={`w-5 h-5 shrink-0 ${targetFile ? 'text-indigo-400' : 'text-slate-500'}`} />
              <div className="min-w-0">
                <p className="text-sm font-medium text-slate-300 truncate">
                  {targetFile ? targetFile.name : (targetCached ? 'Select file to re-extract (optional — cached fingerprints are ready)' : 'Drop target clip here or click to browse')}
                </p>
                {targetFile && <p className="text-xs text-slate-500">{(targetFile.size / 1024 / 1024).toFixed(1)} MB</p>}
              </div>
            </div>
          </div>

          {!refDone && (targetFile || targetCached) && (
            <p className="text-xs text-amber-400/80 flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5" /> Process the reference video first (Step 1).
            </p>
          )}

          <button onClick={handleRunAnalysis} disabled={!canRunAnalysis}
            className="w-full flex justify-center items-center gap-2 py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl text-sm font-medium transition-colors cursor-pointer">
            <ScanLine className="w-4 h-4" />
            {isProcessingTarget ? 'Extracting fingerprints…' : isMatching ? 'Running matching algorithm…' : 'Process & Find Matches'}
          </button>
          {renderProgress(targetProgress, 'bg-indigo-500')}
        </section>

        {/* ── Gemini daily-limit warning (BOTH models exhausted) ── */}
        {geminiQuota?.dailyLimitReached && (
          <div className="bg-red-950/60 border border-red-500/40 rounded-xl p-3.5 text-sm text-red-200 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-semibold text-red-300">
                Gemini key limit over (day) — nayi API key lekar aao
              </p>
              <p className="text-xs text-red-200/80 leading-relaxed">
                Dono models ki daily quota khatam ho gayi hai ({geminiQuota.usedToday} requests aaj).
                Processing rukegi nahi — server har 10 minute mein check karta rahega ki limit
                wapas aayi ya nahi (midnight Pacific pe reset). Tab tak jin segments ko Gemini
                verify nahi kar paya, wo unverifiable mark honge (embedding similarity se decide
                hoga ki rakhna hai ya nahi). Nayi key (alag Cloud project wali) Settings mein daal
                do to Gemini turant wapas chalu ho jayega.
              </p>
            </div>
          </div>
        )}

        {/* ── Gemini per-model quota display ── */}
        {!geminiQuota?.dailyLimitReached && geminiQuota?.configured && geminiQuota.models && isMatching && (
          <div className="bg-slate-900/70 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-300 flex flex-wrap items-center gap-x-4 gap-y-1.5">
            <span className="text-slate-500 font-medium">Gemini quota</span>
            {geminiQuota.models.map(mq => (
              <span key={mq.model} className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                  mq.dailyLimitReached
                    ? 'bg-red-500'
                    : geminiQuota.model === mq.model
                      ? 'bg-green-400 animate-pulse'
                      : 'bg-slate-600'
                }`} />
                <span className={mq.dailyLimitReached ? 'text-red-300 line-through' : ''}>
                  {mq.model.replace('gemini-', '').replace('-latest', '')}: {mq.remaining}/{mq.rpdLimit} left
                </span>
              </span>
            ))}
          </div>
        )}

        {/* ── Gemini per-minute pacing indicator (both RPM windows full) ── */}
        {!geminiQuota?.dailyLimitReached && geminiQuota?.rateLimitWaiting && isMatching && (
          <div className="bg-amber-950/40 border border-amber-500/30 rounded-xl px-3.5 py-2.5 text-xs text-amber-200 flex items-center gap-2.5">
            <div className="w-2 h-2 rounded-full bg-amber-400 animate-pulse shrink-0" />
            Dono models ka {geminiQuota.rpmLimit} req/min window full hai — shortest wait chal raha
            hai, kaam nahi rukega ({geminiQuota.usedToday} requests aaj tak). RPM hit pe model
            auto-rotate hota hai, isliye ye rarely dikhega.
          </div>
        )}

        {/* ── Status bar ── */}
        {(status || isMatching) && (
          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5 text-sm text-slate-300 space-y-2.5">
            <div className="flex items-center gap-3">
              <div className={`w-2 h-2 rounded-full shrink-0 ${busy ? 'bg-blue-500 animate-pulse' : 'bg-green-500'}`} />
              {status}
            </div>

            {/* Live matching progress — shown only while matching via server SSE */}
            {isMatching && matchProgress && (
              <MatchProgressPanel progress={matchProgress} />
            )}
          </div>
        )}

        {/* ── Results table ── */}
        {segments.length > 0 && (
          <section className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
            <div className="p-5 border-b border-slate-800 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
              <div className="flex items-center gap-3">
                <div className="bg-green-500/10 p-2 rounded-lg border border-green-500/20">
                  <Layers className="w-4 h-4 text-green-400" />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-white">
                    {segments.length} Matched Segment{segments.length !== 1 ? 's' : ''}
                  </h2>
                  {matchStats && (
                    <p className="text-xs text-slate-500">
                      {matchStats.shortFrames} clip frames · {matchStats.movieFrames} reference frames ·
                      threshold {similarityThreshold}% · min {minSegmentDuration.toFixed(1)}s
                    </p>
                  )}
                </div>
              </div>
              <button onClick={handleDownloadJson}
                className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 rounded-lg text-xs font-medium transition cursor-pointer">
                <Download className="w-3.5 h-3.5" /> Export JSON
              </button>
            </div>

            {/* ── Clip coverage timeline ── */}
            {matchStats && matchStats.shortFrames > 0 && (() => {
              const clipDur = segments.length > 0
                ? Math.max(...segments.map(s => s.shortEnd), ...unmatchedRanges.map(u => u.shortEnd))
                : 0;
              if (clipDur <= 0) return null;
              return (
                <div className="px-5 py-4 border-b border-slate-800 space-y-2">
                  <div className="flex justify-between text-xs text-slate-500 font-mono">
                    <span>Clip coverage</span>
                    <span>
                      {unmatchedRanges.length === 0
                        ? '✓ Full clip matched'
                        : `${unmatchedRanges.length} unmatched range${unmatchedRanges.length !== 1 ? 's' : ''}`}
                    </span>
                  </div>
                  {/* Stacked bar: green=matched, orange=unmatched */}
                  <div className="relative h-6 rounded overflow-hidden bg-slate-800 flex">
                    {(() => {
                      // Build a sorted list of intervals with type
                      type Bar = { start: number; end: number; kind: 'match' | 'gap'; conf: number; idx: number };
                      const bars: Bar[] = [];
                      segments.forEach((s, i) => bars.push({ start: s.shortStart, end: s.shortEnd, kind: 'match', conf: s.confidence, idx: i }));
                      unmatchedRanges.forEach((u, i) => bars.push({ start: u.shortStart, end: u.shortEnd, kind: 'gap', conf: 0, idx: i }));
                      bars.sort((a, b) => a.start - b.start);
                      return bars.map((bar, i) => {
                        const left  = (bar.start / clipDur) * 100;
                        const width = Math.max(0.3, ((bar.end - bar.start) / clipDur) * 100);
                        const conf  = bar.conf;
                        const bg    = bar.kind === 'gap'
                          ? 'bg-orange-700/60'
                          : conf >= 80 ? 'bg-green-500' : conf >= 60 ? 'bg-yellow-500' : 'bg-blue-400';
                        const label = bar.kind === 'match'
                          ? `Seg ${bar.idx + 1}: ${fmt(bar.start)}–${fmt(bar.end)} (${conf.toFixed(0)}%)`
                          : `Unmatched: ${fmt(bar.start)}–${fmt(bar.end)}`;
                        return (
                          <div key={i} title={label}
                            className={`absolute top-0 h-full ${bg} transition-all`}
                            style={{ left: `${left}%`, width: `${width}%` }} />
                        );
                      });
                    })()}
                  </div>
                  <div className="flex items-center gap-4 text-[10px] text-slate-600">
                    <span className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-sm bg-green-500" /> High confidence match</span>
                    <span className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-sm bg-yellow-500" /> Medium confidence</span>
                    <span className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-sm bg-orange-700/60" /> No match found</span>
                  </div>
                </div>
              );
            })()}

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-950 text-slate-500 text-xs font-medium uppercase tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3 text-left">#</th>
                    <th className="px-4 py-3 text-left">Clip Time</th>
                    <th className="px-4 py-3 text-left">Movie Time</th>
                    <th className="px-4 py-3 text-left">Duration</th>
                    <th className="px-4 py-3 text-left">Frames</th>
                    <th className="px-4 py-3 text-left">Confidence</th>
                    <th className="px-4 py-3 text-right">Compare</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {segments.map((seg, i) => {
                    const clipDur  = seg.shortEnd - seg.shortStart;
                    const movieDur = seg.movieEnd - seg.movieStart;
                    const isActive = previewSegment === seg;
                    const rowCs = findCandidateSetForSegment(seg);
                    const isExpanded = rowCs ? expandedCandidateKeys.has(rowCs.segmentIndex) : false;
                    return (
                      <React.Fragment key={i}>
                      <tr
                        className={`transition-colors hover:bg-slate-800/40 ${
                          isActive
                            ? 'bg-indigo-900/20 ring-1 ring-inset ring-indigo-500/30'
                            : seg.vlmRejectedKept
                              ? 'bg-red-950/30 ring-1 ring-inset ring-red-500/20'
                              : ''
                        }`}>
                        <td className="px-4 py-3 font-mono text-slate-500 text-xs">{i + 1}</td>

                        {/* Clip time */}
                        <td className="px-4 py-3">
                          <div className="font-mono text-xs">
                            <span className="text-white">{fmt(seg.shortStart)}</span>
                            <span className="text-slate-600 mx-1">→</span>
                            <span className="text-slate-400">{fmt(seg.shortEnd)}</span>
                          </div>
                        </td>

                        {/* Movie time */}
                        <td className="px-4 py-3">
                          <div className="font-mono text-xs">
                            <span className="text-white">{fmt(seg.movieStart)}</span>
                            <span className="text-slate-600 mx-1">→</span>
                            <span className="text-slate-400">{fmt(seg.movieEnd)}</span>
                          </div>
                        </td>

                        {/* Duration + Speed ratio */}
                        <td className="px-4 py-3">
                          <div className="font-mono text-xs text-slate-400">
                            <div>{fmtDur(clipDur)} clip</div>
                            <div className="text-slate-600">{fmtDur(movieDur)} movie</div>
                            {seg.speedRatio !== undefined && Math.abs(seg.speedRatio - 1) > 0.08 && (
                              <div
                                title={
                                  seg.speedRatio < 1
                                    ? `Clip was slowed ~${(1 / seg.speedRatio).toFixed(2)}× — movie section is shorter than clip`
                                    : `Clip was sped up ~${seg.speedRatio.toFixed(2)}�� — movie section is longer than clip`
                                }
                                className={`mt-0.5 font-semibold ${
                                  seg.speedRatio < 0.92
                                    ? 'text-blue-400'   // slow-mo
                                    : 'text-amber-400'  // fast-forward
                                }`}
                              >
                                {seg.speedRatio.toFixed(2)}× speed
                              </div>
                            )}
                          </div>
                        </td>

                        {/* Frame count */}
                        <td className="px-4 py-3 font-mono text-xs text-slate-400">
                          {seg.frameCount}
                          {seg.gapCount != null && seg.gapCount > 0 && (
                            <span className="ml-1 text-amber-500/70" title={`${seg.gapCount} low-confidence frame(s) skipped within segment`}>
                              +{seg.gapCount}↗
                            </span>
                          )}
                        </td>

                        {/* Confidence */}
                        <td className="px-4 py-3">
                          <div className="flex flex-col items-start gap-1">
                            {seg.vlmRejectedKept
                              ? <RejectedKeptBadge />
                              : <ConfidenceBadge confidence={seg.confidence} isApproximate={seg.isApproximate} />}
                            {seg.timelineOutlier && <TimelineJumpBadge />}
                          </div>
                        </td>

                        {/* Compare + View all candidates buttons */}
                        <td className="px-4 py-3 text-right">
                          <div className="inline-flex items-center gap-1.5">
                            {/* Inline candidate expansion toggle — shows every
                                candidate (accepted / rejected / not checked)
                                for this segment right below the row */}
                            {rowCs && (
                              <button
                                onClick={() => toggleCandidateExpansion(rowCs.segmentIndex)}
                                title={isExpanded ? 'Hide candidates' : `Show all ${rowCs.candidates.length} candidate(s) below this row`}
                                className={`inline-flex items-center gap-1 px-2 py-1.5 border rounded-lg text-xs font-medium transition cursor-pointer ${isExpanded ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300' : 'bg-emerald-500/10 hover:bg-emerald-500/20 border-emerald-500/20 text-emerald-400'}`}>
                                {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                <span className="hidden sm:inline">{rowCs.candidates.length}</span>
                              </button>
                            )}
                            {/* View all candidates — only shown when candidate history exists */}
                            {(() => {
                              const cs = findCandidateSetForSegment(seg);
                              if (!cs) return null;
                              const isOpen = viewAllCandidatesForKey === cs.segmentIndex;
                              return (
                                <button
                                  onClick={() => setViewAllCandidatesForKey(isOpen ? null : cs.segmentIndex)}
                                  title={`View all ${cs.candidates.length} candidate(s) for this segment`}
                                  className={`inline-flex items-center gap-1 px-2 py-1.5 border rounded-lg text-xs font-medium transition cursor-pointer ${isOpen ? 'bg-purple-500/20 border-purple-500/40 text-purple-300' : 'bg-purple-500/10 hover:bg-purple-500/20 border-purple-500/20 text-purple-400'}`}>
                                  <ListChecks className="w-3 h-3" />
                                </button>
                              );
                            })()}
                            <button onClick={() => handlePreviewSegment(seg)}
                              className={`inline-flex items-center gap-1 px-3 py-1.5 border rounded-lg text-xs font-medium transition cursor-pointer ${isActive ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' : 'bg-blue-500/10 hover:bg-blue-500/20 border-blue-500/20 text-blue-400'}`}>
                              <Play className="w-3 h-3" /> Compare
                            </button>
                          </div>
                        </td>
                      </tr>

                      {/* ── Inline candidate list (expanded) — additive UI ── */}
                      {rowCs && isExpanded && (
                        <tr className="bg-slate-950/60">
                          <td colSpan={7} className="px-4 py-3">
                            <div className="rounded-xl border border-slate-800 bg-slate-900/70 overflow-hidden">
                              <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-800 bg-slate-950/50">
                                <p className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                                  <ListChecks className="w-3.5 h-3.5 text-emerald-400" />
                                  Candidates for Segment {i + 1}
                                  <span className="font-normal font-mono text-slate-500 normal-case">
                                    {fmt(rowCs.shortStart)}–{fmt(rowCs.shortEnd)}
                                  </span>
                                </p>
                                <span className="text-[10px] font-mono text-slate-600">
                                  {rowCs.candidates.length} candidate(s)
                                </span>
                              </div>
                              {rowCs.candidates.length === 0 ? (
                                <p className="px-4 py-4 text-xs text-slate-500 text-center">No candidates recorded for this segment.</p>
                              ) : (
                                <div className="divide-y divide-slate-800/60 max-h-64 overflow-y-auto">
                                  {rowCs.candidates.map((c, cIdx) => {
                                    const isUsed = cIdx === (rowCs.recoveredCandidateIndex ?? -1);
                                    const selKey = `${rowCs.segmentIndex}:${cIdx}`;
                                    const isSelecting = selectingCandidateKey === selKey;
                                    const segmentBusy = retryingSegments.has(rowCs.segmentIndex) || !!rowCs.retrying;
                                    return (
                                      <div key={cIdx}
                                        className={`px-4 py-2.5 flex flex-wrap items-center gap-3 transition-colors ${isUsed ? 'bg-indigo-500/5' : 'hover:bg-slate-800/40'}`}>
                                        <span className="text-[11px] font-mono text-slate-600 w-5 shrink-0">{cIdx + 1}</span>
                                        <div className="flex-1 min-w-[140px]">
                                          <div className="font-mono text-xs text-slate-300">
                                            {fmt(c.segment.movieStart)}
                                            <span className="text-slate-600 mx-1">→</span>
                                            {fmt(c.segment.movieEnd)}
                                          </div>
                                          {c.confidencePct != null && (
                                            <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                                              {c.confidencePct.toFixed(1)}% confidence
                                            </div>
                                          )}
                                        </div>
                                        <CandidateVerdictBadge candidate={c} isUsed={isUsed} isBestEffort={rowCs.bestEffort === true} />
                                        {renderTrimControls(rowCs, cIdx, isSelecting || !!selectingCandidateKey || segmentBusy || isMatching)}
                                        <div className="flex items-center gap-1.5 ml-auto">
                                          <button
                                            onClick={() => handleJumpToCandidate(rowCs, cIdx)}
                                            title="Preview this candidate in the compare panel below"
                                            className="inline-flex items-center gap-1 px-2.5 py-1.5 border rounded-lg text-[11px] font-medium bg-blue-500/10 hover:bg-blue-500/20 border-blue-500/20 text-blue-400 transition cursor-pointer">
                                            <Play className="w-3 h-3" /> Preview
                                          </button>
                                          {isUsed ? (
                                            <span className="inline-flex items-center gap-1 px-2.5 py-1.5 border rounded-lg text-[11px] font-semibold bg-indigo-500/10 border-indigo-500/25 text-indigo-300">
                                              ★ Main
                                            </span>
                                          ) : (
                                            <button
                                              onClick={() => handleMakeMainCandidate(rowCs, cIdx)}
                                              disabled={isSelecting || !!selectingCandidateKey || segmentBusy || isMatching}
                                              title="Make this candidate the main match for this segment (you verified it yourself)"
                                              className="inline-flex items-center gap-1 px-2.5 py-1.5 border rounded-lg text-[11px] font-semibold bg-emerald-500/10 hover:bg-emerald-500/20 disabled:opacity-40 disabled:cursor-not-allowed border-emerald-500/25 text-emerald-300 transition cursor-pointer">
                                              {isSelecting
                                                ? <><RefreshCw className="w-3 h-3 animate-spin" /> Saving…</>
                                                : <><CheckCircle2 className="w-3 h-3" /> Make Main</>}
                                            </button>
                                          )}
                                        </div>
                                      </div>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {selectError && (
              <div className="px-4 py-2 border-t border-red-900/40 bg-red-500/5 flex items-center gap-2 text-xs text-red-300">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                {selectError}
                <button onClick={() => setSelectError('')} className="ml-auto text-red-400/70 hover:text-red-300 cursor-pointer">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {adjustError && (
              <div className="px-4 py-2 border-t border-red-900/40 bg-red-500/5 flex items-center gap-2 text-xs text-red-300">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                {adjustError}
                <button onClick={() => setAdjustError('')} className="ml-auto text-red-400/70 hover:text-red-300 cursor-pointer">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </section>
        )}

        {/* No results message */}
        {segments.length === 0 && !isMatching && matchStats && (
          <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 rounded-xl p-5 text-slate-400 text-sm">
            <AlertCircle className="w-5 h-5 text-amber-500 shrink-0" />
            No matching segments found. Try lowering the confidence threshold or min duration in Match Parameters, or ensure the clip actually appears in the reference.
          </div>
        )}

        {/* ── Side-by-side Preview Panel ── */}
        {previewSegment && (
          <section className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">

            {/* Panel header */}
            <div className="p-4 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="bg-blue-500/10 p-1.5 rounded-lg border border-blue-500/20">
                  <Video className="w-4 h-4 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    Side-by-Side Comparison
                    {previewSegment.vlmRejectedKept
                      ? <RejectedKeptBadge />
                      : <ConfidenceBadge confidence={previewSegment.confidence} isApproximate={previewSegment.isApproximate} />}
                    {previewSegment.timelineOutlier && <TimelineJumpBadge />}
                  </h3>
                  <p className="text-xs text-slate-500 font-mono">
                    Clip {fmt(previewSegment.shortStart)}–{fmt(previewSegment.shortEnd)} ({fmtDur(previewSegment.shortEnd - previewSegment.shortStart)}) ·
                    Movie {fmt(previewSegment.movieStart)}–{fmt(previewSegment.movieEnd)} ·
                    {previewSegment.frameCount} frames
                  </p>
                </div>
              </div>
              <button onClick={() => { setPreviewSegment(null); setIsPlaying(false); refVideoRef.current?.pause(); clipVideoRef.current?.pause(); }}
                className="p-1.5 text-slate-500 hover:text-slate-300 rounded-lg hover:bg-slate-800 transition cursor-pointer">
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Playback controls bar */}
            <div className="px-4 py-3 border-b border-slate-800 bg-slate-950/50 flex flex-wrap items-center gap-3">

              {/* Play / Pause */}
              <button onClick={handleSyncPlay}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition cursor-pointer ${isPlaying ? 'bg-amber-500/20 border border-amber-500/40 text-amber-300 hover:bg-amber-500/30' : 'bg-green-600 border border-green-500 text-white hover:bg-green-500'}`}>
                {isPlaying ? <><Pause className="w-3.5 h-3.5" /> Pause Both</> : <><Play className="w-3.5 h-3.5" /> Play Both</>}
              </button>

              {/* Restart */}
              <button onClick={handleRestartPreview}
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition cursor-pointer">
                <RotateCcw className="w-3.5 h-3.5" /> Restart
              </button>

              {/* Separator */}
              <div className="w-px h-6 bg-slate-700" />

              {/* Loop toggle */}
              <button onClick={() => setLoopSegment(l => !l)}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium border transition cursor-pointer ${loopSegment ? 'bg-indigo-500/15 border-indigo-500/40 text-indigo-300' : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-300'}`}>
                <Repeat className="w-3.5 h-3.5" />
                {loopSegment ? 'Loop: ON' : 'Loop: OFF'}
              </button>

              {/* Separator */}
              <div className="w-px h-6 bg-slate-700" />

              {/* Speed controls */}
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-slate-500 font-mono">Speed:</span>
                {[0.25, 0.5, 1.0].map(sp => (
                  <button key={sp} onClick={() => setPlaybackSpeed(sp)}
                    className={`px-2.5 py-1.5 rounded-md text-xs font-mono font-semibold border transition cursor-pointer ${playbackSpeed === sp ? 'bg-blue-600 border-blue-500 text-white' : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white hover:bg-slate-700'}`}>
                    {sp === 1 ? '1×' : `${sp}×`}
                  </button>
                ))}
              </div>
            </div>

            {/* Segment info row */}
            <div className="px-4 py-2.5 bg-slate-950/30 border-b border-slate-800 grid grid-cols-3 gap-4 text-xs font-mono">
              <div>
                <span className="text-slate-600 block">Clip duration</span>
                <span className="text-slate-300">{fmtDur(previewSegment.shortEnd - previewSegment.shortStart)}</span>
              </div>
              <div>
                <span className="text-slate-600 block">Movie duration</span>
                <span className="text-slate-300">{fmtDur(previewSegment.movieEnd - previewSegment.movieStart)}</span>
              </div>
              <div>
                <span className="text-slate-600 block">Speed ratio</span>
                <span className="text-slate-300">
                  {((previewSegment.movieEnd - previewSegment.movieStart) /
                    Math.max(0.001, previewSegment.shortEnd - previewSegment.shortStart)).toFixed(3)}×
                </span>
              </div>
            </div>

            {/* Dual video panes */}
            <div className="relative grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-800">
              {isCurrentSegmentRetrying && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-slate-950/85 backdrop-blur-sm">
                  <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
                  <p className="text-xs font-medium text-slate-300">Retrying this segment…</p>
                </div>
              )}

              {/* Reference movie */}
              <div className="p-4 space-y-2">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Film className="w-3.5 h-3.5 text-blue-400" /> Reference Movie
                  <span className="font-normal font-mono text-slate-600 normal-case ml-1">@ {fmt(movieViewSegment?.movieStart ?? previewSegment.movieStart)}</span>
                </p>
                <div className="bg-black rounded-xl overflow-hidden border border-slate-800 relative">
                  {refFileUrl ? (
                    <video ref={refVideoRef} src={refFileUrl} controls
                      className="w-full max-h-72 object-contain"
                      onTimeUpdate={handleRefTimeUpdate}
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-40 text-slate-600 text-sm">
                      Reference video not available for preview
                    </div>
                  )}
                </div>
              </div>

              {/* Target clip */}
              <div className="p-4 space-y-2">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Search className="w-3.5 h-3.5 text-indigo-400" /> Target Clip
                  <span className="font-normal font-mono text-slate-600 normal-case ml-1">@ {fmt(previewSegment.shortStart)}</span>
                </p>
                <div className="bg-black rounded-xl overflow-hidden border border-slate-800">
                  {targetFileUrl ? (
                    <video ref={clipVideoRef} src={targetFileUrl} controls
                      className="w-full max-h-72 object-contain"
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-40 text-slate-600 text-sm">
                      Target clip not available for preview
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* ── Quick adjust bar — sits right under the videos so the user
                can trim the boundaries and confirm without scrolling up.
                Same handlers as everywhere else (renderTrimControls /
                handleMakeMainCandidate) — purely a second placement. ── */}
            {activeCandidateSet && activeCandidateSet.candidates.length > 0 && (() => {
              const currentCandidate = activeCandidateSet.candidates[candidateIndex];
              if (!currentCandidate) return null;
              const isUsed = candidateIndex === (activeCandidateSet.recoveredCandidateIndex ?? -1);
              const busy = !!selectingCandidateKey || isCurrentSegmentRetrying || isMatching;
              return (
                <div className="px-4 py-3 border-t border-cyan-500/15 bg-cyan-500/[0.03] flex flex-wrap items-center gap-3">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-cyan-300/60 flex items-center gap-1.5">
                    <Zap className="w-3 h-3" /> Quick Adjust
                  </span>
                  {/* ±1 s trim — preview seeks to the moved edge instantly */}
                  {renderTrimControls(activeCandidateSet, candidateIndex, busy)}
                  {/* Confirm — make this candidate the main segment */}
                  {isUsed ? (
                    <span className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Main Segment Set
                    </span>
                  ) : (
                    <button
                      onClick={() => handleMakeMainCandidate(activeCandidateSet, candidateIndex)}
                      disabled={busy}
                      title="Confirm — make this candidate the main segment"
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed border border-emerald-500 text-white transition cursor-pointer">
                      {selectingCandidateKey === `${activeCandidateSet.segmentIndex}:${candidateIndex}`
                        ? <><RefreshCw className="w-3.5 h-3.5 animate-spin" /> Saving…</>
                        : <><CheckCircle2 className="w-3.5 h-3.5" /> Confirm — Make Main</>}
                    </button>
                  )}
                </div>
              );
            })()}

            {/* ── Preview navigation ── */}
            <div className="px-4 py-3 border-t border-slate-800 bg-slate-950/40 flex flex-wrap items-center gap-4">

              {/* Pair 1 — Segment Next/Previous (always present, primary) */}
              <div className="flex items-center gap-2">
                <button onClick={() => handleSegmentStep(-1)} disabled={previewIndex <= 0}
                  className="flex items-center gap-1 px-3 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-slate-800 border border-slate-700 text-slate-200 transition cursor-pointer">
                  <ChevronLeft className="w-3.5 h-3.5" /> Previous Segment
                </button>
                <span className="text-xs font-mono text-slate-500 px-1">
                  Segment {previewIndex + 1} of {segments.length}
                </span>
                <button onClick={() => handleSegmentStep(1)} disabled={previewIndex === -1 || previewIndex >= segments.length - 1}
                  className="flex items-center gap-1 px-3 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-slate-800 border border-slate-700 text-slate-200 transition cursor-pointer">
                  Next Segment <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Pair 2 — Candidate Next/Previous (only when this segment has candidate data) */}
              {activeCandidateSet && activeCandidateSet.candidates.length > 0 && (() => {
                const currentCandidate = activeCandidateSet.candidates[candidateIndex];
                const isUsed = candidateIndex === (activeCandidateSet.recoveredCandidateIndex ?? -1);
                return (
                  <>
                    <div className="w-px h-6 bg-slate-700 hidden sm:block" />
                    <div className="flex items-center gap-1.5 bg-purple-500/5 border border-purple-500/20 rounded-lg px-2 py-1">
                      <button onClick={() => handleCandidateStep(-1)} disabled={candidateIndex <= 0}
                        className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium bg-slate-800/80 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-slate-800/80 border border-slate-700 text-purple-200 transition cursor-pointer">
                        <ChevronLeft className="w-3 h-3" /> Prev Candidate
                      </button>
                      <span className="text-[11px] font-mono text-purple-300/80 px-1">
                        Candidate {candidateIndex + 1} of {activeCandidateSet.candidates.length}
                      </span>
                      <button onClick={() => handleCandidateStep(1)} disabled={candidateIndex >= activeCandidateSet.candidates.length - 1}
                        className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium bg-slate-800/80 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-slate-800/80 border border-slate-700 text-purple-200 transition cursor-pointer">
                        Next Candidate <ChevronRight className="w-3 h-3" />
                      </button>
                    </div>
                    {currentCandidate && <CandidateVerdictBadge candidate={currentCandidate} isUsed={isUsed} isBestEffort={activeCandidateSet.bestEffort === true} />}
                    {/* ±1 s boundary trim for the currently-stepped candidate */}
                    {currentCandidate && renderTrimControls(
                      activeCandidateSet, candidateIndex,
                      !!selectingCandidateKey || isCurrentSegmentRetrying || isMatching)}
                    {/* Make Main — promote the currently-stepped candidate the
                        user just visually verified (hidden when it already IS
                        the main match). Same handler as the inline table. */}
                    {currentCandidate && !isUsed && (
                      <button
                        onClick={() => handleMakeMainCandidate(activeCandidateSet, candidateIndex)}
                        disabled={!!selectingCandidateKey || isCurrentSegmentRetrying || isMatching}
                        title="Make this candidate the main match for this segment"
                        className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-emerald-500/10 hover:bg-emerald-500/20 disabled:opacity-40 disabled:cursor-not-allowed border border-emerald-500/30 text-emerald-300 transition cursor-pointer">
                        {selectingCandidateKey === `${activeCandidateSet.segmentIndex}:${candidateIndex}`
                          ? <><RefreshCw className="w-3.5 h-3.5 animate-spin" /> Saving…</>
                          : <><CheckCircle2 className="w-3.5 h-3.5" /> Make Main</>}
                      </button>
                    )}
                  </>
                );
              })()}

              {/* Manual Retry — independent of the automatic verdict; the
                  user clicks this when they visually judge the current match
                  to be wrong. PERMANENT: always visible for the previewed
                  segment (candidates exist even without a VLM check, so the
                  button must not depend on VLM/candidate data having loaded).
                  Disabled only while this same segment's own retry is running
                  (other segments stay fully interactive). */}
              <div className="w-px h-6 bg-slate-700 hidden sm:block" />
              <button onClick={handleRetrySegment} disabled={isCurrentSegmentRetrying || retrySegmentIndex < 0}
                title="Manually re-search for a better match for this segment"
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-amber-500/10 hover:bg-amber-500/20 disabled:opacity-50 disabled:cursor-not-allowed border border-amber-500/30 text-amber-300 transition cursor-pointer">
                <RefreshCw className={`w-3.5 h-3.5 ${isCurrentSegmentRetrying ? 'animate-spin' : ''}`} />
                {isCurrentSegmentRetrying ? 'Retrying…' : 'Retry Segment'}
              </button>
            </div>

            {retryError && (
              <div className="px-4 py-2 border-t border-red-900/40 bg-red-500/5 flex items-center gap-2 text-xs text-red-300">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                {retryError}
              </div>
            )}

            {adjustError && (
              <div className="px-4 py-2 border-t border-red-900/40 bg-red-500/5 flex items-center gap-2 text-xs text-red-300">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                {adjustError}
                <button onClick={() => setAdjustError('')} className="ml-auto text-red-400/70 hover:text-red-300 cursor-pointer">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Match quality timeline */}
            {previewSegment.matchSequence.length > 0 && (
              <div className="p-4 border-t border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-slate-500">Match quality timeline — {previewSegment.matchSequence.length} frames</p>
                  <p className="text-xs font-mono text-slate-500">
                    avg {(previewSegment.matchSequence.reduce((a, f) => a + f.similarity, 0) / previewSegment.matchSequence.length).toFixed(1)}%
                  </p>
                </div>
                <div className="flex gap-px h-8 rounded overflow-hidden">
                  {previewSegment.matchSequence.map((item, i) => {
                    const pct = item.similarity;
                    const bg  = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500';
                    return (
                      <div key={i} className={`flex-1 ${bg}`} style={{ opacity: 0.4 + (pct / 100) * 0.6 }}
                        title={`Frame ${i + 1}: ${item.similarity.toFixed(1)}% @ clip ${fmt(item.shortTime)} → movie ${fmt(item.movieTime)}`} />
                    );
                  })}
                </div>
                <div className="flex justify-between text-xs text-slate-700 mt-1 font-mono">
                  <span>{fmt(previewSegment.shortStart)}</span>
                  <span className="text-slate-600">clip timeline</span>
                  <span>{fmt(previewSegment.shortEnd)}</span>
                </div>
              </div>
            )}

            {/* ── Perceptual Fingerprint & Integrity Panels ── */}
            {previewSegment.bestFrameDetail && (() => {
              const d = previewSegment.bestFrameDetail!;
              const intSim = d.structureSim * 0.84 + d.colorSim * 0.053 + d.skinSim * 0.053 + d.detailSim * 0.054;
              return (
                <>
                  {/* Live Perceptual Fingerprint */}
                  <div className="p-4 border-t border-slate-800 space-y-4">
                    <div className="flex items-center gap-2">
                      <Zap className="w-3.5 h-3.5 text-yellow-400" />
                      <p className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                        Live Perceptual Fingerprint <span className="text-slate-600 normal-case font-normal">(16×16 / 256-bit Hex Map)</span>
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                      <HashGrid hash={d.movieHash} label="Movie Hash" accent="bg-blue-400" />
                      <HashGrid hash={d.shortHash} label="Short Hash" accent="bg-indigo-400" />
                    </div>
                  </div>

                  {/* Match Integrity + Crop Region */}
                  <div className="p-4 border-t border-slate-800 space-y-4">
                    <div className="flex flex-wrap items-start gap-6">
                      {/* Integrity gauge */}
                      <div className="flex-1 min-w-[180px] space-y-2">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="text-slate-400">Match Integrity Similarity</span>
                          <span className={`font-bold ${intSim >= similarityThreshold ? 'text-green-400' : intSim >= 60 ? 'text-yellow-400' : 'text-orange-400'}`}>
                            {intSim.toFixed(1)}%
                          </span>
                        </div>
                        <div className="relative w-full h-3 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ${intSim >= similarityThreshold ? 'bg-green-500' : intSim >= 60 ? 'bg-yellow-500' : 'bg-orange-500'}`}
                            style={{ width: `${intSim}%` }}
                          />
                          {/* Threshold marker */}
                          <div
                            className="absolute top-0 h-full w-0.5 bg-white/40"
                            style={{ left: `${similarityThreshold}%` }}
                            title={`Threshold: ${similarityThreshold}%`}
                          />
                        </div>
                        <div className="flex justify-between text-[9px] font-mono text-slate-700">
                          <span>Low Match</span>
                          <span>Threshold ({similarityThreshold}%)</span>
                          <span>Exact Match</span>
                        </div>
                      </div>

                      {/* Crop region badge */}
                      <div className="space-y-1">
                        <p className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Matched Crop Region</p>
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-purple-500/10 border border-purple-500/25 rounded-lg text-xs font-mono text-purple-300">
                          <Layers className="w-3 h-3" />
                          {d.cropRegion}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Double-Check Verification Checklist */}
                  <div className="p-4 border-t border-slate-800 space-y-3">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                      <p className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Double-Check Verification Checklist</p>
                      <span className="text-[10px] text-slate-600 font-mono">Active Frame-by-Frame</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <ChecklistRow label="Structure" value={d.structureSim} weight="84%" />
                      <ChecklistRow label="Colors & Background" value={d.colorSim} />
                      <ChecklistRow label="Human / Character" value={d.skinSim} />
                      <ChecklistRow label="Edges / Details" value={d.detailSim} />
                    </div>
                    <p className="text-[9px] text-slate-700 leading-relaxed">
                      * The system prioritizes structural fingerprints <span className="text-slate-600">(84% weight)</span> to remain robust under style changes,
                      while cross-checking ambient colors, characters (hands/heads), and background detail ratios <span className="text-slate-600">(16% weight)</span>{' '}
                      for extreme frame alignment precision.
                    </p>

                    {/* Fingerprint Bitstream */}
                    <div className="mt-2 space-y-2 pt-3 border-t border-slate-800/60">
                      <p className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Fingerprint Bitstream (Prefix 48-bit)</p>
                      <div className="space-y-1">
                        <div className="flex items-start gap-2">
                          <span className="text-[9px] text-blue-500/70 font-mono w-24 shrink-0 pt-0.5">Movie Hash</span>
                          <code className="text-[9px] font-mono text-slate-400 break-all leading-tight">
                            {d.movieHash ? d.movieHash.slice(0, 48) + '…' : '—'}
                          </code>
                        </div>
                        <div className="flex items-start gap-2">
                          <span className="text-[9px] text-indigo-500/70 font-mono w-24 shrink-0 pt-0.5">Short Hash</span>
                          <code className="text-[9px] font-mono text-slate-400 break-all leading-tight">
                            {d.shortHash ? d.shortHash.slice(0, 48) + '…' : '—'}
                          </code>
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              );
            })()}
          </section>
        )}

      </div>

      {/* ── View All Candidates modal overlay ── */}
      {viewAllCandidatesFor && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm"
          onClick={() => setViewAllCandidatesForKey(null)}>
          <div
            className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-lg max-h-[80vh] flex flex-col overflow-hidden"
            onClick={e => e.stopPropagation()}>

            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
              <div className="flex items-center gap-2 min-w-0">
                <ListChecks className="w-4 h-4 text-purple-400 shrink-0" />
                <h3 className="text-sm font-semibold text-slate-200 truncate">
                  All Candidates — Segment {viewAllCandidatesFor.segmentIndex + 1}
                </h3>
                <span className="text-xs text-slate-500 font-mono shrink-0">
                  {fmt(viewAllCandidatesFor.shortStart)}–{fmt(viewAllCandidatesFor.shortEnd)}
                </span>
              </div>
              <button
                onClick={() => setViewAllCandidatesForKey(null)}
                className="ml-3 text-slate-500 hover:text-slate-300 transition cursor-pointer shrink-0">
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Candidate list */}
            <div className="overflow-y-auto flex-1 divide-y divide-slate-800/60">
              {viewAllCandidatesFor.candidates.length === 0 ? (
                <p className="px-5 py-6 text-xs text-slate-500 text-center">No candidates recorded.</p>
              ) : (
                viewAllCandidatesFor.candidates.map((c, idx) => {
                  const isUsed = idx === (viewAllCandidatesFor.recoveredCandidateIndex ?? -1);
                  return (
                    <button
                      key={idx}
                      onClick={() => handleJumpToCandidate(viewAllCandidatesFor, idx)}
                      className="w-full px-5 py-3 flex items-center gap-3 hover:bg-slate-800/60 transition text-left cursor-pointer group">
                      <span className="text-[11px] font-mono text-slate-600 w-5 shrink-0 group-hover:text-slate-500">
                        {idx + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="font-mono text-xs text-slate-300">
                          {fmt(c.segment.movieStart)}
                          <span className="text-slate-600 mx-1">→</span>
                          {fmt(c.segment.movieEnd)}
                        </div>
                        {c.confidencePct != null && (
                          <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                            {c.confidencePct.toFixed(1)}% confidence
                          </div>
                        )}
                      </div>
                      <CandidateVerdictBadge candidate={c} isUsed={isUsed} isBestEffort={viewAllCandidatesFor.bestEffort === true} />
                    </button>
                  );
                })
              )}
            </div>

            <div className="px-5 py-3 border-t border-slate-800 text-[11px] text-slate-600">
              Click any row to jump the preview to that candidate&apos;s movie location.
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
