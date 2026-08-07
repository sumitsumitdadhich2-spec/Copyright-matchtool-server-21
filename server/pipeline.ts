import { spawn, execSync } from 'child_process';
import { Worker } from 'worker_threads';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import { FrameSignature } from '../src/shared/fingerprint';

const getDirname = () => {
  if (typeof __dirname !== 'undefined') {
    return __dirname;
  }
  try {
    const metaUrl = typeof import.meta !== 'undefined' && import.meta.url ? import.meta.url : '';
    if (metaUrl) {
      return path.dirname(fileURLToPath(metaUrl));
    }
  } catch (e) {
    // ignore
  }
  return process.cwd();
};

const currentDirname = getDirname();

export const NUM_WORKERS = Math.max(1, Math.min(os.cpus().length, 128));

/**
 * Build a clean env for Nix-provided binaries (ffprobe, ffmpeg).
 * When server.ts prepends /lib/x86_64-linux-gnu to LD_LIBRARY_PATH so that
 * canvas.node can find libuuid.so.1, the system libmount.so.1 ends up before
 * the Nix-provided one in the search order.  The system copy is older and
 * lacks the MOUNT_2_40 versioned symbol that Nix glib requires, causing:
 *   ffprobe: libmount.so.1: version `MOUNT_2_40' not found
 * Fix: strip the system lib paths from LD_LIBRARY_PATH before spawning any
 * Nix binary.  canvas workers inherit the unmodified process env and still
 * find libuuid via the $ORIGIN symlink in node_modules/canvas/build/Release/.
 */
export function makeCleanEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  if (env.LD_LIBRARY_PATH) {
    const cleaned = env.LD_LIBRARY_PATH
      .split(':')
      .filter(p => p !== '/lib/x86_64-linux-gnu' && p !== '/usr/lib/x86_64-linux-gnu')
      .join(':');
    if (cleaned) {
      env.LD_LIBRARY_PATH = cleaned;
    } else {
      delete env.LD_LIBRARY_PATH;
    }
  }
  return env;
}

// ---------------------------------------------------------------------------
// How often to attempt a flush (every N processed frames)
// ---------------------------------------------------------------------------
const FLUSH_EVERY = 100;

// Flush when contiguous completed frames in Map >= this many
const FLUSH_BATCH = 1500;

// Flush when process RSS exceeds this (bytes).  Scaled off actual machine RAM
// (os.totalmem()) instead of a hardcoded figure, so the pipeline automatically
// uses more headroom on bigger machines (e.g. a 16 GB box gets ~11 GB instead
// of being stuck at the 8 GB-era 5.5 GB figure) while leaving room for the OS
// + workers. The multiplier reproduces the original 5.5 GB value exactly on an
// 8 GB machine (5.5 / 8 = 0.6875).
const RAM_FLUSH_THRESHOLD_BYTES = os.totalmem() * 0.6875;

// ---------------------------------------------------------------------------
// Checkpoint: save progress to disk every N frames flushed so processing can
// resume after a server restart without re-processing from scratch.
// ---------------------------------------------------------------------------
const CHECKPOINT_EVERY = 5000;

/** Options for resumable extraction */
export interface ExtractionOptions {
  /** Number of frames already written to outputPath from a previous run (0 = fresh start) */
  resumeFrom?: number;
  /** Full path of the checkpoint JSON file to create/update during processing */
  checkpointPath?: string;
  /** Job ID — stored in the checkpoint so server.ts can match it on resume */
  jobId?: string;
  /** "filename:filesize" key — stored in checkpoint for fast lookup */
  checkpointKey?: string;
  /**
   * AbortSignal to cancel the pipeline mid-run.
   * When aborted, ffmpeg and workers are killed, the write stream is destroyed,
   * and the returned Promise rejects with Error('STOPPED').
   */
  abortSignal?: AbortSignal;
}

export interface FingerprintResult {
  frameIndex: number;
  timestamp: number;
  variants: Record<string, { hash: string }>;
  signature?: FrameSignature;
}

/**
 * Extract per-frame fingerprints from a video file and write them as NDJSON
 * (one JSON object per line) directly to `outputPath`.
 *
 * RAM usage is kept low by flushing completed frames to disk as they arrive,
 * rather than accumulating the entire fingerprint set in memory until the end.
 *
 * @returns Promise that resolves with the total frame count once done.
 */
export function extractFingerprints(
  videoPath: string,
  outputPath: string,
  onProgress?: (decoded: number, processed: number) => void,
  options: ExtractionOptions = {}
): Promise<number> {
  const { resumeFrom = 0, checkpointPath, jobId, checkpointKey, abortSignal } = options;

  return new Promise((resolve, reject) => {
    const workers: Worker[] = [];
    let idleWorkers: Worker[] = [];
    const activeTasks = new Map<number, { resolve: Function; reject: Function }>();
    // Tracks which task ids are currently in flight on each worker, so a worker
    // crash (native canvas fault, transient OOM, etc.) can reject exactly its
    // own orphaned frames instead of leaving them stuck in activeTasks forever.
    // Without this, one dead worker means `processed` never reaches `decoded`
    // and the whole job hangs in "processing" indefinitely — far more likely
    // across the tens/hundreds of thousands of frames in a long movie than in
    // a short clip.
    const workerTaskIds = new Map<Worker, Set<number>>();
    let taskIdCounter = 0;
    let decoded = 0;
    let processed = 0;
    // Frames skipped at the start of a resumed run (already written in a previous run).
    let skipped = 0;
    // Total frames that finished successfully (distinct from `processed`, which
    // counts both successes and failures for the completion-check below).
    let successCount = 0;
    // Frame indices that failed permanently (worker crash, decode error, etc.)
    // and will never appear in `fingerprints`. Consulted by flushToStream's
    // contiguous walk so a single failed frame doesn't block every later frame
    // from ever being flushed — see note there for why that matters.
    const failedFrameIndices = new Set<number>();
    // Holds frames that have been computed but not yet written to disk.
    const fingerprints = new Map<number, { variants: any; signature?: FrameSignature }>();
    const taskQueue: { id: number; frameBuffer: Buffer; width: number; height: number; frameIndex: number }[] = [];
    let ffmpegProcess: any = null;
    let isFinished = false;

    // Track the highest frame index already written to disk.
    // Initialised to resumeFrom so flushToStream starts writing from the right offset.
    let lastFlushedFrame = resumeFrom;

    // Checkpoint state — prevents concurrent writes and tracks last saved position.
    let checkpointPending = false;
    let lastCheckpointAt = resumeFrom; // lastFlushedFrame value at last checkpoint save

    // Set after ffprobe; controls how many raw-pixel frames we allow in the
    // task queue at once.  At 1 080p a frame is ~8 MB; at 4K it is ~33 MB.
    // Hardcoding 100 was safe for 1 080p (800 MB queue) but blew past 8 GB for
    // 4K content (100 × 33 MB = 3.3 GB before any fingerprints are computed).
    let frameBytes       = 0;
    let dynamicQueueLimit = 100; // default until ffprobe fills this in

    // ── Write stream ─────────────────────────────────────────────────────────
    // Use append mode when resuming so already-written frames are preserved.
    let writeStreamErr: Error | null = null;
    const writeStream = fs.createWriteStream(outputPath, {
      encoding: 'utf8',
      flags: resumeFrom > 0 ? 'a' : 'w',
    });
    writeStream.on('error', (err) => { writeStreamErr = err; });

    // ── Abort / cancel support ────────────────────────────────────────────────
    // When the caller signals cancellation (e.g. from the Stop button), we kill
    // ffmpeg, terminate all workers, destroy the write stream, and reject the
    // Promise with the sentinel string 'STOPPED' so server.ts can distinguish
    // a user-initiated stop from a processing error.
    if (abortSignal) {
      if (abortSignal.aborted) {
        // Already cancelled before the pipeline even started — bail immediately.
        writeStream.destroy();
        reject(new Error('STOPPED'));
        return;
      }
      abortSignal.addEventListener('abort', () => {
        if (isFinished) return;
        isFinished = true;
        stopDiagnostics();
        try { ffmpegProcess?.kill('SIGTERM'); } catch { /* ignore */ }
        cleanupWorkers();
        try { writeStream.destroy(); } catch { /* ignore */ }
        reject(new Error('STOPPED'));
      }, { once: true });
    }

    // ── Flush helper ─────────────────────────────────────────────────────────
    /**
     * Write the longest contiguous run of completed frames (starting at
     * lastFlushedFrame+1) to disk, then delete them from the Map.
     *
     * Called periodically from the worker message handler and once more
     * (force=true) after all frames are done.
     */
    function flushToStream(force = false): void {
      const rss = process.memoryUsage().rss;
      const shouldFlush =
        force ||
        rss >= RAM_FLUSH_THRESHOLD_BYTES ||
        fingerprints.size >= FLUSH_BATCH;
      if (!shouldFlush) return;

      // Walk forward from the last flushed position while frames are present.
      // A permanently-failed frame (in failedFrameIndices, never in
      // `fingerprints`) must still let the walk step past it — otherwise one
      // failed frame in the middle of a long job would wedge `hi` there
      // forever, and every later frame (even though successfully computed)
      // would sit unflushed in the Map for the rest of the job: unbounded
      // memory growth plus a silently truncated output file with everything
      // after the failure point missing, despite the job reporting success.
      let hi = lastFlushedFrame;
      while (fingerprints.has(hi + 1) || failedFrameIndices.has(hi + 1)) hi++;
      if (hi === lastFlushedFrame) return; // nothing contiguous to flush

      const FRAME_RATE = 25;
      for (let i = lastFlushedFrame + 1; i <= hi; i++) {
        const fp = fingerprints.get(i);
        if (fp) {
          const line =
            JSON.stringify({
              frameIndex: i,
              timestamp: (i - 1) / FRAME_RATE,
              variants: fp.variants,
              signature: fp.signature
            }) + '\n';
          writeStream.write(line);
          fingerprints.delete(i); // free memory
        }
      }
      lastFlushedFrame = hi;

      // ── Async checkpoint write (non-blocking, every CHECKPOINT_EVERY frames) ─
      // We write after lastFlushedFrame advances so the checkpoint always reflects
      // data that is actually on disk. checkpointPending prevents concurrent writes.
      if (
        checkpointPath && jobId && checkpointKey &&
        lastFlushedFrame - lastCheckpointAt >= CHECKPOINT_EVERY &&
        !checkpointPending
      ) {
        lastCheckpointAt = lastFlushedFrame;
        checkpointPending = true;
        const cpData = JSON.stringify({ jobId, checkpointKey, updatedAt: Date.now() });
        fs.promises.writeFile(checkpointPath, cpData)
          .catch(e => console.error(`[Checkpoint] Write failed for ${jobId}:`, e))
          .finally(() => { checkpointPending = false; });
      }
    }

    // ── Worker lifecycle ─────────────────────────────────────────────────────
    const cleanupWorkers = () => {
      for (const w of workers) {
        w.terminate().catch(() => {});
      }
    };

    function assignTasks() {
      while (idleWorkers.length > 0 && taskQueue.length > 0) {
        const worker = idleWorkers.pop()!;
        const task = taskQueue.shift()!;
        workerTaskIds.get(worker)?.add(task.id);
        dispatchedAt.set(task.id, Date.now());
        // Transfer (not clone) the frame's backing ArrayBuffer. Without a
        // transferList, worker_threads structured-clones the payload — for an
        // ~8 MB 1080p frame that clone is a synchronous main-thread copy
        // (measured ~6-7ms) that runs once per frame regardless of NUM_WORKERS,
        // capping total achievable throughput no matter how many workers are
        // idle. Transferring ownership instead is near-zero-cost (~0.04ms) —
        // safe here because `frameBuffer` is always a dedicated standalone
        // copy (see the `Buffer.from(...)` above), never a view sharing memory
        // with anything the main thread still needs.
        worker.postMessage({
          id: task.id,
          frameBuffer: task.frameBuffer,
          width: task.width,
          height: task.height
        }, [task.frameBuffer.buffer]);
      }
      // Resume when the queue has drained to 75% of the limit (instead of
      // 50%) AND RSS is comfortably below the flush threshold.
      //
      // Why 75%? With NUM_WORKERS hashing frames in parallel, the 50% threshold
      // created windows where workers drained the remaining half-queue faster
      // than a single-threaded ffmpeg could refill it, leaving workers idle
      // (visible in [PipelineDiag] as idleWorkers > 0 with taskQueue == 0).
      // At 75% ffmpeg restarts while the queue still has 3× NUM_WORKERS of
      // headroom, almost eliminating those starvation gaps.
      const resumeAt = Math.max(NUM_WORKERS * 2, Math.floor(dynamicQueueLimit * 3 / 4));
      if (
        taskQueue.length < resumeAt &&
        process.memoryUsage().rss < RAM_FLUSH_THRESHOLD_BYTES * 0.85 &&
        ffmpegProcess?.stdout.isPaused()
      ) {
        ffmpegProcess.stdout.resume();
      }
    }

    // ── Diagnostic sampler ───────────────────────────────────────────────────
    // Periodically logs pool-utilization signals so a slow run can be diagnosed
    // (worker-starved vs decode-starved vs RAM-throttled) without attaching a
    // profiler. Cheap (one line every 3s) — left in permanently since it is the
    // only visibility into why a deploy's CPU usage doesn't reach NUM_WORKERS
    // cores. Does not affect pause/resume/RAM logic in any way; read-only.
    // Wall-clock time each task spent dispatched-but-not-yet-complete, purely
    // for the sampler below — does not read or influence any hashing logic.
    const dispatchedAt = new Map<number, number>();
    let busyMsSinceLastSample = 0;
    let framesSinceLastSample = 0;

    let diagnosticTimer: NodeJS.Timeout | null = null;
    if (process.env.PIPELINE_DIAGNOSTICS !== '0') {
      diagnosticTimer = setInterval(() => {
        if (isFinished) return;
        const rssMb = Math.round(process.memoryUsage().rss / 1_048_576);
        const paused = ffmpegProcess?.stdout?.isPaused?.() ?? false;
        const avgMsPerFrame = framesSinceLastSample > 0
          ? (busyMsSinceLastSample / framesSinceLastSample).toFixed(0)
          : 'n/a';
        console.log(
          `[PipelineDiag] decoded=${decoded} processed=${processed + skipped} ` +
          `idleWorkers=${idleWorkers.length}/${NUM_WORKERS} taskQueue=${taskQueue.length}/${dynamicQueueLimit} ` +
          `rss=${rssMb}MB ffmpegPaused=${paused} avgWorkerMs/frame=${avgMsPerFrame}`
        );
        busyMsSinceLastSample = 0;
        framesSinceLastSample = 0;
      }, 3000);
      diagnosticTimer.unref?.();
    }
    const stopDiagnostics = () => { if (diagnosticTimer) clearInterval(diagnosticTimer); };

    try {
      const isProd = process.env.NODE_ENV === 'production';
      // In dev, import.meta.url is undefined under tsx ESM mode, so getDirname()
      // falls back to process.cwd() (project root) — not server/.  Resolve
      // explicitly from CWD so the path is always correct regardless of how
      // tsx initialises import.meta.
      const workerPath = isProd
        ? path.join(currentDirname, 'worker.cjs')
        : path.resolve(process.cwd(), 'server', 'worker.ts');

      // Spawn one worker, wiring up message/error/exit handling. Used both for
      // the initial pool and to replace a worker that crashes mid-run so pool
      // throughput doesn't silently shrink over a long job.
      function spawnWorker(): Worker {
        // In dev the worker entry is a .ts file, so tsx must be active inside
        // the worker thread. `-r tsx/cjs` only installs the CommonJS require
        // hook — it does nothing when the worker is loaded through the ESM
        // loader chain (which is what happens whenever the parent process has
        // its own `--import` loader in NODE_OPTIONS, e.g. under the hosted
        // dev runtime). In that case worker.ts is resolved as ESM and its
        // extensionless `../src/shared/fingerprint` import throws
        // ERR_MODULE_NOT_FOUND, killing every worker and failing 100% of
        // frames. `--import tsx` registers BOTH the ESM resolve/load hooks
        // and the CJS hook, so it works either way.
        const worker = new Worker(workerPath, isProd ? {} : {
          execArgv: ['--import', 'tsx']
        });
        workerTaskIds.set(worker, new Set());

        worker.on('message', (msg) => {
          workerTaskIds.get(worker)?.delete(msg.id);
          idleWorkers.push(worker);
          const dispatchTs = dispatchedAt.get(msg.id);
          if (dispatchTs !== undefined) {
            dispatchedAt.delete(msg.id);
            busyMsSinceLastSample += Date.now() - dispatchTs;
            framesSinceLastSample++;
          }
          const task = activeTasks.get(msg.id);
          if (task) {
            activeTasks.delete(msg.id);
            if (msg.error) {
              task.reject(new Error(msg.error));
            } else {
              processed++;
              task.resolve(msg.result);
              if (onProgress) {
                onProgress(decoded, processed + skipped);
              }
            }
          }
          // Periodically attempt to flush completed frames to disk.
          if (processed % FLUSH_EVERY === 0) {
            flushToStream();
          }
          assignTasks();
        });

        worker.on('error', (err) => {
          console.error(`Worker error:`, err);
          handleWorkerDeath(worker);
        });

        worker.on('exit', (code) => {
          // worker.terminate() (normal end-of-job / abort cleanup) itself
          // reports a non-zero code, so check isFinished BEFORE logging —
          // otherwise every successful job would print a false "unexpectedly"
          // alarm for its own routine shutdown.
          if (code !== 0 && !isFinished) {
            console.error(`Worker exited unexpectedly with code ${code}`);
            handleWorkerDeath(worker);
          }
        });

        workers.push(worker);
        idleWorkers.push(worker);
        return worker;
      }

      // A worker can die mid-frame (native canvas fault, transient OOM, etc.).
      // Reject only the task(s) that were actually in flight on THIS worker —
      // identified via workerTaskIds — so the overall processed+skipped count
      // can still reach `decoded` and the job reaches a terminal state instead
      // of hanging forever, then spawn a replacement to keep pool size steady.
      function handleWorkerDeath(worker: Worker) {
        if (isFinished) return; // pipeline already ending/aborted — nothing to do
        const orphaned = workerTaskIds.get(worker);
        if (!orphaned) return; // 'error' and 'exit' can both fire — handle once
        workerTaskIds.delete(worker);

        for (const id of orphaned) {
          const task = activeTasks.get(id);
          if (task) {
            activeTasks.delete(id);
            task.reject(new Error('Worker crashed before finishing this frame'));
          }
        }

        const index = workers.indexOf(worker);
        if (index !== -1) workers.splice(index, 1);
        const idleIndex = idleWorkers.indexOf(worker);
        if (idleIndex !== -1) idleWorkers.splice(idleIndex, 1);

        try {
          spawnWorker();
        } catch (e) {
          console.error('Failed to respawn worker after crash:', e);
        }
        assignTasks();
      }

      for (let i = 0; i < NUM_WORKERS; i++) {
        spawnWorker();
      }

      // ── Query video dimensions via ffprobe ───────────────────────────────
      let width = 0;
      let height = 0;
      try {
        const ffprobeOutput = execSync(
          `ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "${videoPath}"`,
          { env: makeCleanEnv() }
        ).toString().trim();
        const [w, h] = ffprobeOutput.split('x').map(Number);
        if (!w || !h || isNaN(w) || isNaN(h)) {
          throw new Error(`Failed to parse resolution: ${ffprobeOutput}`);
        }
        width = w;
        height = h;
      } catch (err: any) {
        console.error(`ffprobe error on ${videoPath}:`, err);
        throw new Error(`Could not determine video dimensions: ${err.message}`);
      }

      // ── Dynamic queue limit ──────────────────────────────────────────────
      // Cap raw-frame RAM in the task queue, scaled off actual machine RAM
      // (os.totalmem()) instead of a hardcoded 1.5 GB figure, so a bigger
      // machine can buffer more frames per worker and go faster.  Multiplier
      // reproduces the original 1.5 GB cap exactly on an 8 GB machine
      // (1.5 / 8 = 0.1875), e.g. ~3 GB on a 16 GB machine.
      //   1 080p  (8.3 MB/frame) → ~180 frames in queue  (~1.5 GB @ 8 GB RAM)
      //   4K      (33  MB/frame) →  ~45 frames in queue  (~1.5 GB @ 8 GB RAM)
      frameBytes        = width * height * 4;
      const QUEUE_RAM_CAP = os.totalmem() * 0.1875;
      dynamicQueueLimit = Math.max(4, Math.min(1000, Math.floor(QUEUE_RAM_CAP / frameBytes)));
      console.log(
        `Pipeline starting for ${videoPath} (${width}x${height})` +
        ` — frame ${(frameBytes / 1_048_576).toFixed(1)} MB` +
        ` — queue limit ${dynamicQueueLimit} frames (~${(dynamicQueueLimit * frameBytes / 1_073_741_824).toFixed(2)} GB)`
      );

      ffmpegProcess = spawn('ffmpeg', [
        '-i', videoPath,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgba',
        '-r', '25',
        '-'
      ], { env: makeCleanEnv() });

      const frameSize = width * height * 4;

      // ── Frame accumulator ────────────────────────────────────────────────
      // ffmpeg's stdout arrives in small OS-pipe-sized chunks (measured ~64 KB
      // on this host) — roughly 130-170 chunks per raw frame at 1080p/4K.
      // The previous approach (`buffer = Buffer.concat([buffer, chunk])` every
      // chunk, then `.slice()`) reallocated and copied the ENTIRE accumulated
      // leftover on every single chunk arrival: while accumulating one frame,
      // that leftover grows from ~0 up to just under `frameSize`, so the total
      // copying work to assemble one frame was O(frameSize × chunks/frame) —
      // e.g. ~170 copies averaging several MB each for a single 8 MB frame,
      // squarely on the main thread, blocking event-loop turns (ffmpeg stdout
      // reads, worker dispatch/completion) the whole time.
      // Fix: append each incoming chunk into a fixed backing buffer with
      // `.copy()` (cost = chunk size only, not accumulated size), and once a
      // full frame is available, shift the small remainder down. This is the
      // standard growable-buffer pattern and makes accumulation cost O(bytes
      // received) instead of O(bytes received × frame size).
      let accum = Buffer.alloc(frameSize + 1_048_576); // frame + 1MB slack for one chunk
      let accumLen = 0;

      ffmpegProcess.stdout.on('data', (chunk: Buffer) => {
        if (accumLen + chunk.length > accum.length) {
          // Extremely large single chunk (bigger than our slack) — fall back
          // to growing the backing buffer rather than dropping data.
          const grown = Buffer.alloc(accumLen + chunk.length);
          accum.copy(grown, 0, 0, accumLen);
          accum = grown;
        }
        chunk.copy(accum, accumLen);
        accumLen += chunk.length;

        while (accumLen >= frameSize) {
          // Deliberate standalone copy (not a view into `accum`, which is
          // reused/mutated for subsequent chunks). A dedicated copy is also
          // required so this frame's ArrayBuffer can be handed to the worker
          // via the zero-copy `transferList` below without detaching memory
          // `accum` still needs. See postMessage call for why.
          const frameBuffer = Buffer.from(accum.subarray(0, frameSize));
          // Shift any leftover bytes (start of the next frame) down to the
          // front of the accumulator.
          accum.copy(accum, 0, frameSize, accumLen);
          accumLen -= frameSize;

          decoded++;

          // ── Resume skip: discard frames already written in a previous run ──
          // ffmpeg decodes from the start; we simply throw away raw pixel data
          // for frames we already have.  No workers are involved, so fingerprint
          // quality for the resumed portion is identical to a fresh run.
          if (decoded <= resumeFrom) {
            skipped++;
            if (onProgress && skipped % 1000 === 0) {
              onProgress(decoded, skipped); // show fast-forward progress
            }
            // Don't pause ffmpeg during skip — queue is empty, no backpressure.
            continue;
          }

          const id = ++taskIdCounter;
          const currentFrame = decoded;

          const p = new Promise<{ variants: any; signature?: FrameSignature }>((res, rej) => {
            activeTasks.set(id, { resolve: res, reject: rej });
          });

          p.then((result) => {
            fingerprints.set(currentFrame, result);
            successCount++;
          }).catch((err) => {
            console.error(`Error processing frame ${currentFrame}:`, err);
            failedFrameIndices.add(currentFrame);
            processed++;
            if (onProgress) {
              onProgress(decoded, processed + skipped);
            }
          });

          taskQueue.push({
            id,
            frameBuffer,
            width,
            height,
            frameIndex: currentFrame
          });

          // Pause ffmpeg when the queue is full OR when total RSS is high.
          // Both conditions are checked so a very large video resolution
          // triggers pause even before the frame count limit is reached.
          const rssNow = process.memoryUsage().rss;
          if (
            (taskQueue.length >= dynamicQueueLimit || rssNow >= RAM_FLUSH_THRESHOLD_BYTES) &&
            !ffmpegProcess.stdout.isPaused()
          ) {
            ffmpegProcess.stdout.pause();
          }

          assignTasks();
        }
      });

      ffmpegProcess.stderr.on('data', (_data: Buffer) => {
        // suppress ffmpeg stderr
      });

      // stdout/stderr are separate EventEmitters from the process itself — an
      // unhandled 'error' on either (broken pipe, resource exhaustion, etc.)
      // would otherwise crash the whole Node process instead of just failing
      // this one job.
      const failPipeline = (err: any) => {
        if (isFinished) return;
        isFinished = true;
        stopDiagnostics();
        console.error('ffmpeg pipeline error:', err);
        try { ffmpegProcess.kill('SIGTERM'); } catch { /* ignore */ }
        cleanupWorkers();
        reject(err);
      };
      ffmpegProcess.stdout.on('error', failPipeline);
      ffmpegProcess.stderr.on('error', failPipeline);

      ffmpegProcess.on('error', failPipeline);

      ffmpegProcess.on('close', (_code: any) => {
        const checkInterval = setInterval(() => {
          if ((processed + skipped) >= decoded) {
            clearInterval(checkInterval);
            if (!isFinished) {
              isFinished = true;
              stopDiagnostics();
              cleanupWorkers();

              // Final flush — write any frames still in the Map.
              flushToStream(true);

              writeStream.end(() => {
                const attempted = successCount + failedFrameIndices.size;
                if (writeStreamErr) {
                  reject(writeStreamErr);
                } else if (attempted > 0 && successCount === 0) {
                  // Every single frame that was actually attempted (i.e. not
                  // skipped by a resume) failed — almost certainly a systemic
                  // problem (missing native dependency, corrupt input, etc.),
                  // not ordinary per-frame flakiness. Previously this still
                  // resolved "successfully" with an empty result file, which
                  // silently produced a "0 segments matched" result downstream
                  // with no indication anything was wrong. Partial failures
                  // (some frames ok, some not) are still tolerated as before.
                  reject(new Error(
                    `All ${attempted} processed frame(s) failed — fingerprinting produced no usable data. Check server logs for the underlying error (e.g. a missing native dependency).`
                  ));
                } else {
                  resolve(decoded);
                }
              });
            }
          }
        }, 100);
      });

    } catch (err) {
      stopDiagnostics();
      cleanupWorkers();
      reject(err);
    }
  });
}
