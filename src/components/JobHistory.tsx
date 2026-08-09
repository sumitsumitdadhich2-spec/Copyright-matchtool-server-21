import React, { useState, useEffect, useCallback, useRef } from 'react';
import { X, Trash2, Square, CheckCircle2, AlertCircle, Clock, RefreshCw, Film, FolderOpen, HardDrive, PlayCircle } from 'lucide-react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface MatchJobProgress {
  phase: string;
  pct: number;
  vlmSegmentIndex?: number;
  vlmTotalSegments?: number;
  vlmVerdict?: string;
  vlmAttempt?: number;
  vlmTotalBudget?: number;
}

interface JobEntry {
  id: string;
  type?: 'fingerprint' | 'match';
  status: 'uploading' | 'pending' | 'processing' | 'completed' | 'failed' | 'stopped';
  totalFrames: number;
  processedFrames: number;
  error?: string;
  originalName?: string;
  startedAt?: number;
  completedAt?: number;
  // Match-job-only extras
  progress?: MatchJobProgress;
  segmentCount?: number;
  movieJobId?: string;
  shortJobId?: string;
  // Original uploaded video still saved on the server. For match jobs this is
  // true only when BOTH source videos survive.
  hasVideo?: boolean;
  /** Bytes the saved video(s) for this job occupy on the server. */
  videoSize?: number;
  /** Match-job-only: per-source availability of the two saved videos. */
  movieHasVideo?: boolean;
  shortHasVideo?: boolean;
}

function fmtBytes(bytes?: number) {
  if (!bytes || bytes <= 0) return null;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

const MATCH_PHASE_LABEL: Record<string, string> = {
  loading_short: 'Analysing short clip',
  indexing:      'Building movie frame index',
  loading_movie: 'Loading movie fingerprints',
  scanning:      'Scanning movie for matches',
  matching:      'Matching scene chunks',
  finalizing:    'Finalising results',
  vlm_verify:    'Verifying scenes with AI',
  vlm_deep_search:   'Deep search — AI hunting new candidates',
  deferred_recovery: 'Recovering dropped segments',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtDate(ts?: number) {
  if (!ts) return '—';
  return new Date(ts).toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function fmtDuration(startedAt?: number, completedAt?: number) {
  if (!startedAt) return null;
  const end = completedAt ?? Date.now();
  const secs = Math.round((end - startedAt) / 1000);
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}m ${s}s`;
}

function StatusBadge({ status }: { status: JobEntry['status'] }) {
  const cfg: Record<JobEntry['status'], { label: string; cls: string; dot?: string }> = {
    uploading:  { label: 'Uploading',  cls: 'bg-blue-500/15 text-blue-400 border-blue-500/30',   dot: 'animate-pulse bg-blue-400' },
    processing: { label: 'Processing', cls: 'bg-green-500/15 text-green-400 border-green-500/30', dot: 'animate-pulse bg-green-400' },
    pending:    { label: 'Pending',    cls: 'bg-blue-500/15 text-blue-400 border-blue-500/30',   dot: 'bg-blue-400' },
    completed:  { label: 'Completed', cls: 'bg-teal-500/15 text-teal-400 border-teal-500/30' },
    failed:     { label: 'Failed',    cls: 'bg-red-500/15 text-red-400 border-red-500/30' },
    stopped:    { label: 'Stopped',   cls: 'bg-orange-500/15 text-orange-400 border-orange-500/30' },
  };
  // Fallback for any unknown status value coming from the server so a
  // missing config entry can never crash the render.
  const c = cfg[status] ?? {
    label: status ?? 'Unknown',
    cls: 'bg-slate-500/15 text-slate-400 border-slate-500/30',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold border ${c.cls}`}>
      {c.dot && <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${c.dot}`} />}
      {c.label}
    </span>
  );
}

function StatusIcon({ status }: { status: JobEntry['status'] }) {
  if (status === 'completed') return <CheckCircle2 className="w-4 h-4 text-teal-400 shrink-0" />;
  if (status === 'failed')    return <AlertCircle  className="w-4 h-4 text-red-400 shrink-0" />;
  if (status === 'stopped')   return <Square       className="w-4 h-4 text-orange-400 shrink-0" />;
  if (status === 'processing' || status === 'pending')
    return <RefreshCw className="w-4 h-4 text-green-400 shrink-0 animate-spin" />;
  return <Clock className="w-4 h-4 text-slate-400 shrink-0" />;
}

// ---------------------------------------------------------------------------
// Single job card
// ---------------------------------------------------------------------------

const JobCard = React.memo(function JobCard({
  job,
  onStop,
  onDelete,
  onReattach,
  onOpenMatch,
  onOpenFingerprint,
  onDeleteVideo,
}: {
  job: JobEntry;
  onStop: (id: string, type: 'fingerprint' | 'match') => void;
  onDelete: (id: string, type: 'fingerprint' | 'match') => void;
  onReattach?: (id: string, type: 'fingerprint' | 'match') => void;
  onOpenMatch?: (id: string) => void;
  onOpenFingerprint?: (id: string, role: 'reference' | 'target') => void;
  onDeleteVideo?: (id: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [pickOpenRole, setPickOpenRole] = useState(false);
  const [stopping, setStopping] = useState(false);
  // Id of the video job currently previewed inline (this job for a fingerprint
  // job, or one of the two source jobs for a match job). null = player closed.
  const [watchId, setWatchId] = useState<string | null>(null);
  // Which saved video the per-source "remove" confirmation is armed for.
  const [confirmRemoveId, setConfirmRemoveId] = useState<string | null>(null);

  const isMatch = job.type === 'match';
  const isRunning = job.status === 'processing' || job.status === 'pending' || job.status === 'uploading';
  const pct = job.totalFrames > 0
    ? Math.min(100, Math.round((job.processedFrames / job.totalFrames) * 100))
    : 0;

  // Estimate time remaining for running fingerprint jobs (frame-rate based).
  // Match jobs don't have a stable "frames/sec" rate across phases, so no ETA.
  let etaStr: string | null = null;
  if (!isMatch && isRunning && job.startedAt && job.processedFrames > 0 && job.totalFrames > 0) {
    const elapsed = (Date.now() - job.startedAt) / 1000;
    const fps = job.processedFrames / (elapsed || 1);
    const remaining = job.totalFrames - job.processedFrames;
    const eta = fps > 0 ? remaining / fps : 0;
    if (isFinite(eta) && eta > 0) {
      const m = Math.floor(eta / 60);
      const s = Math.round(eta % 60);
      etaStr = m > 0 ? `~${m}m ${s}s left` : `~${s}s left`;
    }
  }

  // Every saved video this card can play / remove. A fingerprint job owns one
  // video; a match job points at the two fingerprint jobs it was built from,
  // so both stay watchable (and individually removable) from history.
  const videoSources: { id: string; label: string }[] = isMatch
    ? [
        ...(job.movieHasVideo && job.movieJobId ? [{ id: job.movieJobId, label: 'movie' }] : []),
        ...(job.shortHasVideo && job.shortJobId ? [{ id: job.shortJobId, label: 'clip' }] : []),
      ]
    : job.hasVideo ? [{ id: job.id, label: '' }] : [];

  const handleStop = async () => {
    setStopping(true);
    onStop(job.id, job.type ?? 'fingerprint');
  };

  const handleDeleteConfirmed = () => {
    onDelete(job.id, job.type ?? 'fingerprint');
  };

  return (
    <div className={`rounded-xl border p-3 space-y-2.5 transition-colors ${
      isRunning
        ? 'bg-green-950/20 border-green-700/30'
        : 'bg-slate-800/40 border-slate-700/50'
    }`}>
      {/* Row 1: Icon + filename + status */}
      <div className="flex items-start gap-2.5">
        <StatusIcon status={job.status} />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-slate-200 truncate leading-tight">
            {job.originalName ?? job.id}
          </p>
          <p className="text-[11px] text-slate-500 font-mono mt-0.5">
            {fmtDate(job.startedAt)}
            {job.completedAt && job.startedAt && (
              <> · {fmtDuration(job.startedAt, job.completedAt)}</>
            )}
            {isRunning && job.startedAt && (
              <> · running {fmtDuration(job.startedAt)}</>
            )}
          </p>
        </div>
        <StatusBadge status={job.status} />
      </div>

      {/* Progress bar (running jobs) */}
      {isRunning && isMatch && (
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] font-mono text-slate-400">
            <span>
              {MATCH_PHASE_LABEL[job.progress?.phase ?? ''] ?? 'Matching…'}
              {(job.progress?.phase === 'vlm_verify' || job.progress?.phase === 'vlm_deep_search') && job.progress.vlmTotalSegments != null && (
                <>
                  {' '}({(job.progress.vlmSegmentIndex ?? 0) + 1}/{job.progress.vlmTotalSegments})
                  {job.progress.vlmAttempt != null && job.progress.vlmTotalBudget != null && (
                    <> · candidate {job.progress.vlmAttempt}/{job.progress.vlmTotalBudget}</>
                  )}
                </>
              )}
            </span>
            <span className="text-indigo-400 font-semibold">{job.progress?.pct ?? 0}%</span>
          </div>
          <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all duration-500"
              style={{ width: `${job.progress?.pct ?? 0}%` }}
            />
          </div>
        </div>
      )}
      {isRunning && !isMatch && (
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] font-mono text-slate-400">
            <span>
              {job.processedFrames.toLocaleString()}
              {job.totalFrames > 0 && <> / {job.totalFrames.toLocaleString()} frames</>}
            </span>
            <span className="text-right">
              {job.totalFrames > 0 && <span className="text-green-400 font-semibold">{pct}%</span>}
              {etaStr && <span className="text-slate-500 ml-2">{etaStr}</span>}
            </span>
          </div>
          <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {/* Frame count / segment count (completed) */}
      {job.status === 'completed' && !isMatch && job.totalFrames > 0 && (
        <p className="text-[11px] font-mono text-slate-500 flex items-center gap-2 flex-wrap">
          <span>{job.totalFrames.toLocaleString()} frames fingerprinted</span>
          {job.hasVideo && (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-sky-500/25 bg-sky-500/10 text-sky-400 text-[10px] font-semibold">
              <HardDrive className="w-3 h-3" /> Video saved{fmtBytes(job.videoSize) ? ` · ${fmtBytes(job.videoSize)}` : ''}
            </span>
          )}
        </p>
      )}
      {job.status === 'completed' && isMatch && (
        <p className="text-[11px] font-mono text-slate-500 flex items-center gap-2 flex-wrap">
          <span>{job.segmentCount?.toLocaleString() ?? 0} segment{job.segmentCount === 1 ? '' : 's'} found</span>
          {job.hasVideo ? (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-sky-500/25 bg-sky-500/10 text-sky-400 text-[10px] font-semibold">
              <HardDrive className="w-3 h-3" /> Videos saved — preview ready{fmtBytes(job.videoSize) ? ` · ${fmtBytes(job.videoSize)}` : ''}
            </span>
          ) : job.movieHasVideo || job.shortHasVideo ? (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-amber-500/25 bg-amber-500/10 text-amber-400 text-[10px] font-semibold">
              <AlertCircle className="w-3 h-3" />
              Only the {job.movieHasVideo ? 'movie' : 'clip'} video is still saved
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-amber-500/25 bg-amber-500/10 text-amber-400 text-[10px] font-semibold">
              <AlertCircle className="w-3 h-3" /> Videos removed — results only
            </span>
          )}
        </p>
      )}

      {/* Error message */}
      {job.error && (
        <p className="text-[11px] text-red-400 font-mono leading-snug break-all">
          {job.error}
        </p>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-2 pt-0.5">
        {/* Running job actions */}
        {isRunning && (
          <>
            {/* Reconnect — reopens live progress view in the main panel */}
            {onReattach && (
              <button
                onClick={() => onReattach(job.id, job.type ?? 'fingerprint')}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-green-500/30 text-green-400 bg-green-500/10 hover:bg-green-500/20 transition cursor-pointer"
              >
                <RefreshCw className="w-3 h-3" />
                Reconnect
              </button>
            )}
            <button
              onClick={handleStop}
              disabled={stopping}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-orange-500/30 text-orange-400 bg-orange-500/10 hover:bg-orange-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition cursor-pointer"
            >
              <Square className="w-3 h-3" />
              {stopping ? 'Stopping…' : 'Stop'}
            </button>
          </>
        )}

        {/* Open — reload a completed match job's results into the main panel */}
        {!isRunning && isMatch && job.status === 'completed' && onOpenMatch && (
          <button
            onClick={() => onOpenMatch(job.id)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-indigo-500/30 text-indigo-400 bg-indigo-500/10 hover:bg-indigo-500/20 transition cursor-pointer"
          >
            <FolderOpen className="w-3 h-3" />
            Open
          </button>
        )}

        {/* Open — restore a completed fingerprint job (and its saved video) into the main panel */}
        {!isRunning && !isMatch && job.status === 'completed' && onOpenFingerprint && (
          !pickOpenRole ? (
            <button
              onClick={() => setPickOpenRole(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-indigo-500/30 text-indigo-400 bg-indigo-500/10 hover:bg-indigo-500/20 transition cursor-pointer"
            >
              <FolderOpen className="w-3 h-3" />
              Open
            </button>
          ) : (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[11px] text-slate-400">Open as:</span>
              <button
                onClick={() => { setPickOpenRole(false); onOpenFingerprint(job.id, 'reference'); }}
                className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition cursor-pointer"
              >
                Movie
              </button>
              <button
                onClick={() => { setPickOpenRole(false); onOpenFingerprint(job.id, 'target'); }}
                className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-violet-600 hover:bg-violet-500 text-white transition cursor-pointer"
              >
                Clip
              </button>
              <button
                onClick={() => setPickOpenRole(false)}
                className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 transition cursor-pointer"
              >
                Cancel
              </button>
            </div>
          )
        )}

        {/* Watch the server-saved copy right here, without loading it into a
            slot. Match jobs expose BOTH of their saved source videos. */}
        {!isRunning && videoSources.map(src => (
          <button
            key={`watch-${src.id}`}
            onClick={() => setWatchId(prev => (prev === src.id ? null : src.id))}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-teal-500/30 text-teal-400 bg-teal-500/10 hover:bg-teal-500/20 transition cursor-pointer"
          >
            <PlayCircle className="w-3 h-3" />
            {watchId === src.id ? 'Hide' : 'Watch'}{src.label ? ` ${src.label}` : ''}
          </button>
        ))}

        {/* Remove a saved video — frees server disk space, keeps fingerprints
            and results so matching and "Open" still work. */}
        {!isRunning && onDeleteVideo && videoSources.map(src => (
          confirmRemoveId !== src.id ? (
            <button
              key={`rm-${src.id}`}
              onClick={() => setConfirmRemoveId(src.id)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-sky-500/25 text-sky-400 bg-sky-500/10 hover:bg-sky-500/20 transition cursor-pointer"
            >
              <HardDrive className="w-3 h-3" />
              Remove{src.label ? ` ${src.label}` : ' video'}
            </button>
          ) : (
            <div key={`rm-${src.id}`} className="flex items-center gap-2">
              <span className="text-[11px] text-slate-400">
                Remove saved {src.label || 'video'}? Previews will stop working.
              </span>
              <button
                onClick={() => {
                  setConfirmRemoveId(null);
                  if (watchId === src.id) setWatchId(null);
                  onDeleteVideo(src.id);
                }}
                className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-sky-600 hover:bg-sky-500 text-white transition cursor-pointer"
              >
                Yes, remove
              </button>
              <button
                onClick={() => setConfirmRemoveId(null)}
                className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 transition cursor-pointer"
              >
                Cancel
              </button>
            </div>
          )
        ))}

        {/* Delete button */}
        {!confirmDelete ? (
          <button
            onClick={() => setConfirmDelete(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-red-500/25 text-red-400 bg-red-500/8 hover:bg-red-500/20 transition cursor-pointer ml-auto"
          >
            <Trash2 className="w-3 h-3" />
            Delete
          </button>
        ) : (
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-[11px] text-slate-400">Delete this job?</span>
            <button
              onClick={handleDeleteConfirmed}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-red-600 hover:bg-red-500 text-white transition cursor-pointer"
            >
              Yes, delete
            </button>
            <button
              onClick={() => setConfirmDelete(false)}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 transition cursor-pointer"
            >
              Cancel
            </button>
          </div>
        )}
      </div>

      {/* Inline preview of the copy kept on the server */}
      {watchId && videoSources.some(s => s.id === watchId) && (
        <div className="bg-black rounded-lg overflow-hidden border border-slate-700/60">
          <video
            key={watchId}
            src={`/api/video/${watchId}`}
            controls
            preload="metadata"
            className="w-full max-h-56 object-contain"
            onError={() => setWatchId(null)}
          />
        </div>
      )}
    </div>
  );
});

// ---------------------------------------------------------------------------
// JobHistory panel
// ---------------------------------------------------------------------------

interface JobHistoryProps {
  onClose: () => void;
  onReattach?: (jobId: string, type: 'fingerprint' | 'match') => void;
  onOpenMatch?: (jobId: string) => void;
  /** Open a completed fingerprint job's saved video back into the main panel
   *  as the reference movie or the target clip. */
  onOpenFingerprint?: (jobId: string, role: 'reference' | 'target') => void;
  /** A job was deleted here — lets the main panel drop it if it was loaded. */
  onJobDeleted?: (jobId: string, type: 'fingerprint' | 'match') => void;
  /** A saved video was removed — lets the main panel drop its dead preview. */
  onVideoRemoved?: (jobId: string) => void;
}

export function JobHistory({
  onClose, onReattach, onOpenMatch, onOpenFingerprint, onJobDeleted, onVideoRemoved,
}: JobHistoryProps) {
  const [jobs, setJobs] = useState<JobEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<'all' | 'match' | 'fingerprint' | 'saved'>('all');
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Serialized snapshot of the last applied jobs payload — lets us skip
  // setState entirely when nothing changed, so the panel doesn't visibly
  // "refresh" (re-render/flicker) on every poll tick.
  const lastPayloadRef = useRef<string>('');
  // In-flight guard — with a 2s interval and a 15s request timeout, a slow
  // server would otherwise pile up overlapping /api/jobs requests, which
  // makes the panel feel like it is constantly refreshing (or hangs it).
  const inFlightRef = useRef(false);

  // The parent (App) re-renders every 1-2s while anything is processing, and
  // it passes plain (non-memoized) functions as these props. Routing them
  // through refs gives every JobCard STABLE callback props, so React.memo
  // actually prevents the whole list from re-rendering on each App render —
  // this is what stops the visible "refresh" flicker of the history panel.
  const handlersRef = useRef({ onReattach, onOpenMatch, onOpenFingerprint });
  useEffect(() => {
    handlersRef.current = { onReattach, onOpenMatch, onOpenFingerprint };
  }, [onReattach, onOpenMatch, onOpenFingerprint]);

  const stableReattach = useCallback((id: string, type: 'fingerprint' | 'match') => {
    handlersRef.current.onReattach?.(id, type);
  }, []);
  const stableOpenMatch = useCallback((id: string) => {
    handlersRef.current.onOpenMatch?.(id);
  }, []);
  const stableOpenFingerprint = useCallback((id: string, role: 'reference' | 'target') => {
    handlersRef.current.onOpenFingerprint?.(id, role);
  }, []);

  const fetchJobs = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const res = await fetch('/api/jobs', { signal: AbortSignal.timeout(15_000) });
      if (!res.ok) return;
      const data: JobEntry[] = await res.json();
      const payload = JSON.stringify(data);
      if (payload !== lastPayloadRef.current) {
        lastPayloadRef.current = payload;
        // Keep the previous object reference for any job whose data didn't
        // change — combined with React.memo(JobCard) this means only the
        // cards that actually changed re-render, so the list never flickers
        // or "refreshes" while a job is running.
        setJobs(prev => {
          const prevById = new Map(prev.map(j => [j.id, j]));
          return data.map(j => {
            const old = prevById.get(j.id);
            return old && JSON.stringify(old) === JSON.stringify(j) ? old : j;
          });
        });
      }
    } catch {
      /* network error — keep old state */
    } finally {
      inFlightRef.current = false;
      setLoading(false);
    }
  }, []);

  // Adaptive polling: 2s while something is running, 6s when everything is
  // finished (history rarely changes on its own — no need to hammer the API).
  const hasRunning = jobs.some(j => j.status === 'processing' || j.status === 'pending' || j.status === 'uploading');

  useEffect(() => {
    fetchJobs();
    intervalRef.current = setInterval(fetchJobs, hasRunning ? 2000 : 6000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchJobs, hasRunning]);

  const handleStop = useCallback(async (jobId: string, type: 'fingerprint' | 'match') => {
    try {
      const url = type === 'match' ? `/api/match-stop/${jobId}` : `/api/jobs/${jobId}/stop`;
      await fetch(url, { method: 'POST' });
      // Optimistically update status while waiting for next poll
      setJobs(prev => prev.map(j =>
        j.id === jobId ? { ...j, status: 'stopped' as const } : j
      ));
    } catch { /* ignore */ }
  }, []);

  const handleDelete = useCallback(async (jobId: string, type: 'fingerprint' | 'match') => {
    try {
      const url = type === 'match' ? `/api/match/${jobId}` : `/api/job/${jobId}`;
      const res = await fetch(url, { method: 'DELETE' });
      if (!res.ok) {
        setActionError('Could not delete that job — the server refused the request.');
        return;
      }
      setActionError('');
      lastPayloadRef.current = ''; // force next poll to re-apply fresh data
      setJobs(prev => prev.filter(j => j.id !== jobId));
      handlersRef.current.onJobDeleted?.(jobId, type);
    } catch {
      setActionError('Could not delete that job — the server is unreachable.');
    }
  }, []);

  // Delete ONLY the saved video file from the server — fingerprints and job
  // history remain, so matching still works; only the video preview is gone.
  const handleDeleteVideo = useCallback(async (jobId: string) => {
    try {
      const res = await fetch(`/api/video/${jobId}`, { method: 'DELETE' });
      if (!res.ok) {
        setActionError('Could not remove that saved video.');
        return;
      }
      setActionError('');
      lastPayloadRef.current = '';
      // Reflect the removal on the video's own job AND on every match job that
      // was built from it, so their Watch buttons disappear immediately.
      setJobs(prev => prev.map(j => {
        if (j.id === jobId) return { ...j, hasVideo: false, videoSize: 0 };
        if (j.type !== 'match') return j;
        if (j.movieJobId !== jobId && j.shortJobId !== jobId) return j;
        return {
          ...j,
          hasVideo: false,
          movieHasVideo: j.movieJobId === jobId ? false : j.movieHasVideo,
          shortHasVideo: j.shortJobId === jobId ? false : j.shortHasVideo,
        };
      }));
      handlersRef.current.onVideoRemoved?.(jobId);
    } catch {
      setActionError('Could not remove that saved video — the server is unreachable.');
    }
  }, []);

  const runningJobs  = jobs.filter(j => j.status === 'processing' || j.status === 'pending' || j.status === 'uploading');
  const finishedJobs = jobs.filter(j => j.status !== 'processing' && j.status !== 'pending' && j.status !== 'uploading');

  return (
    <section className="bg-slate-900 border border-slate-700 rounded-2xl overflow-hidden shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800 bg-slate-900/80">
        <div className="flex items-center gap-2.5">
          <div className="bg-violet-500/10 p-1.5 rounded-lg border border-violet-500/20">
            <Film className="w-4 h-4 text-violet-400" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-white">Job History</h2>
            <p className="text-[11px] text-slate-500">
              {loading ? 'Loading…' : `${jobs.length} job${jobs.length !== 1 ? 's' : ''}`}
              {runningJobs.length > 0 && (
                <span className="text-green-400 ml-1">
                  · {runningJobs.length} running
                </span>
              )}
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 text-slate-500 hover:text-slate-300 rounded-lg hover:bg-slate-800 transition cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div className="p-4 space-y-4 max-h-[70vh] overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center py-8 text-slate-500 text-sm gap-2">
            <RefreshCw className="w-4 h-4 animate-spin" />
            Loading jobs…
          </div>
        )}

        {!loading && jobs.length === 0 && (
          <div className="flex flex-col items-center justify-center py-10 text-center gap-2">
            <Film className="w-8 h-8 text-slate-700" />
            <p className="text-slate-500 text-sm">No jobs yet</p>
            <p className="text-slate-600 text-xs">Upload a video to start processing</p>
          </div>
        )}

        {/* Running jobs section */}
        {runningJobs.length > 0 && (
          <div className="space-y-2">
            <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider px-0.5">
              Active
            </p>
            {runningJobs.map(job => (
              <React.Fragment key={job.id}>
                <JobCard
                  job={job}
                  onStop={handleStop}
                  onDelete={handleDelete}
                  onReattach={stableReattach}
                />
              </React.Fragment>
            ))}
          </div>
        )}

        {/* Finished jobs section */}
        {finishedJobs.length > 0 && (
          <div className="space-y-2">
            {runningJobs.length > 0 && (
              <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider px-0.5">
                History
              </p>
            )}
            {finishedJobs.map(job => (
              <React.Fragment key={job.id}>
                <JobCard
                  job={job}
                  onStop={handleStop}
                  onDelete={handleDelete}
                  onOpenMatch={stableOpenMatch}
                  onOpenFingerprint={stableOpenFingerprint}
                  onDeleteVideo={handleDeleteVideo}
                />
              </React.Fragment>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
