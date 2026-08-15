import * as MP4Box from 'mp4box';
import { FrameFingerprint } from './shared/fingerprint';
import { saveFingerprintsBatch } from './utils/db';

let videoFramesCreated = 0;
let videoFramesClosed = 0;
let imageBitmapsCreated = 0;
let imageBitmapsClosed = 0;

export class WorkerPool {
  workers: Worker[] = [];
  tasks: { bitmap: ImageBitmap; resolve: (val: FrameFingerprint) => void; reject: (err: any) => void; frameIndex: number; timestamp: number }[] = [];
  activeWorkers: number = 0;
  inflightBitmaps: number = 0;

  constructor(size: number) {
    for (let i = 0; i < size; i++) {
      const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (e) => this.handleMessage(worker, e.data);
      worker.onerror = (e) => this.handleError(worker, e);
      this.workers.push(worker);
    }
  }

  handleMessage(worker: Worker, data: { result: FrameFingerprint; error?: string; closed?: boolean }) {
    this.activeWorkers--;
    this.inflightBitmaps--;
    if (data.closed) {
      imageBitmapsClosed++;
    }
    const currentTask = (worker as any).currentTask;
    if (currentTask) {
      if (data.error) {
        currentTask.reject(new Error(data.error));
      } else {
        currentTask.resolve(data.result);
      }
      (worker as any).currentTask = null;
    }
    this.pump();
  }

  handleError(worker: Worker, e: ErrorEvent) {
    this.activeWorkers--;
    this.inflightBitmaps--;
    const currentTask = (worker as any).currentTask;
    if (currentTask) {
      currentTask.reject(new Error(e.message));
      (worker as any).currentTask = null;
    }
    this.pump();
  }

  async enqueue(bitmap: ImageBitmap, frameIndex: number, timestamp: number): Promise<FrameFingerprint> {
    this.inflightBitmaps++;
    return new Promise((resolve, reject) => {
      this.tasks.push({ bitmap, resolve, reject, frameIndex, timestamp });
      this.pump();
    });
  }

  pump() {
    if (this.tasks.length === 0) return;
    const availableWorker = this.workers.find(w => !(w as any).currentTask);
    if (!availableWorker) return;

    const task = this.tasks.shift()!;
    (availableWorker as any).currentTask = task;
    this.activeWorkers++;
    availableWorker.postMessage({
      bitmap: task.bitmap,
      frameIndex: task.frameIndex,
      timestamp: task.timestamp
    }, [task.bitmap]);
  }

  terminate() {
    for (const w of this.workers) {
      w.terminate();
    }
  }
}

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export async function processVideoFile(
  file: File,
  videoId: string,
  onProgress: (framesProcessed: number, totalFrames: number, inflight: number) => void
): Promise<{ totalFrames: number, batches: number, jobId: string }> {
  const mp4boxfile = MP4Box.createFile();
  let chunkSize = 4096;
  let offset = 0;
  let trackInfo: any = null;
  const maxInflight = 300;

  const numWorkers = Math.min(navigator.hardwareConcurrency || 4, 8);
  const workerPool = new WorkerPool(numWorkers);

  const batchSize = 300;
  let currentBatch: FrameFingerprint[] = [];
  let batchIndex = 0;
  let framesDispatched = 0;
  let framesProcessed = 0;
  let totalFramesEst = 0;
  const recentDurations: number[] = [];

  return new Promise((resolve, reject) => {
    let decoder: VideoDecoder;
    let demuxCompleted = false;
    let pendingSamples: any[] = [];
    let isDecoding = false;
    let isProcessingSamples = false;

    console.log("[DEBUG] processVideoFile promise initialized. File size:", file.size);

    let resolveDecoderConfigured: () => void;
    let rejectDecoderConfigured: (err: any) => void;
    const decoderConfiguredPromise = new Promise<void>((res, rej) => {
      resolveDecoderConfigured = res;
      rejectDecoderConfigured = rej;
    });

    let isExtractionStarted = false;
    let resumeReading: (() => void) | null = null;

    mp4boxfile.onError = (e: any) => {
      console.error("[DEBUG] MP4Box error event fired:", e);
      reject(new Error("MP4Box parsing error: " + String(e)));
    };

    mp4boxfile.onReady = (info: any) => {
      if (trackInfo) {
        console.log("[DEBUG] MP4Box onReady event fired again, ignoring.");
        return;
      }
      console.log("[DEBUG] MP4Box onReady event fired. Tracks count:", info.tracks.length);
      info.tracks.forEach((t: any) => {
        console.log(`[DEBUG] Track ID: ${t.id}, Type: ${t.track_width ? "Video" : "Audio/Other"}, Codec: ${t.codec}, Samples: ${t.nb_samples}`);
      });
      const videoTrack = info.videoTracks[0];
      if (!videoTrack) return reject(new Error('No video track found'));
      trackInfo = videoTrack;
      totalFramesEst = videoTrack.nb_samples;

      const stsdEntry = mp4boxfile.getTrackById(videoTrack.id).mdia.minf.stbl.stsd.entries[0] as any;
      const box = stsdEntry.avcC || stsdEntry.hvcC || null;
      let description: Uint8Array | undefined = undefined;
      if (box) {
        const stream = new (MP4Box as any).DataStream(undefined, 0, (MP4Box as any).DataStream.BIG_ENDIAN);
        box.write(stream);
        description = new Uint8Array(stream.buffer, 8);
        console.log(`[DEBUG] Serialized description box of type ${box.type} into Uint8Array of length: ${description.length}`);
      }
                          
      let codec = videoTrack.codec;
      if (codec.startsWith('avc1')) {
        codec = 'avc1.640028'; 
      }
      
      const config: VideoDecoderConfig = {
        codec: codec,
        codedHeight: videoTrack.video.height,
        codedWidth: videoTrack.video.width,
        description: description,
      };

      decoder = new VideoDecoder({
        output: async (frame) => {
          videoFramesCreated++;
          while (workerPool.inflightBitmaps >= maxInflight) {
            await delay(10);
          }
          
          try {
            const bitmap = await createImageBitmap(frame);
            imageBitmapsCreated++;
            const bitmapStartTime = performance.now();
            const ts = frame.timestamp / 1_000_000;
            const fIndex = framesDispatched++;
            
            frame.close();
            videoFramesClosed++;

            workerPool.enqueue(bitmap, fIndex, ts).then(fp => {
              const duration = performance.now() - bitmapStartTime;
              framesProcessed++;
              
              recentDurations.push(duration);
              if (recentDurations.length > 50) {
                recentDurations.shift();
              }
              const rollingAvg = recentDurations.reduce((a, b) => a + b, 0) / recentDurations.length;
              
              if (framesProcessed % 50 === 0) {
                console.log(`[Diag] Frame ${framesProcessed} - Processing duration: ${duration.toFixed(2)}ms | Rolling Avg (last 50): ${rollingAvg.toFixed(2)}ms`);
              }
              if (framesProcessed % 200 === 0) {
                console.log(`[Diag] Frame ${framesProcessed} - VideoFrames: ${videoFramesCreated} created / ${videoFramesClosed} closed, ImageBitmaps: ${imageBitmapsCreated} created / ${imageBitmapsClosed} closed, inflightBitmaps: ${workerPool.inflightBitmaps}`);
              }

              currentBatch.push(fp);
              if (currentBatch.length >= batchSize) {
                saveFingerprintsBatch(videoId, batchIndex++, [...currentBatch]);
                currentBatch = [];
              }
              onProgress(framesProcessed, totalFramesEst, workerPool.inflightBitmaps);
              
              if (demuxCompleted && pendingSamples.length === 0 && decoder.decodeQueueSize === 0 && framesProcessed === framesDispatched) {
                 if (currentBatch.length > 0) {
                    saveFingerprintsBatch(videoId, batchIndex++, currentBatch);
                 }
                 workerPool.terminate();
                 resolve({ totalFrames: framesProcessed, batches: batchIndex, jobId: '' });
              }
            }).catch(e => {
              console.error("Worker error", e);
            });
          } catch (err) {
            console.error("Failed to create bitmap", err);
            frame.close();
            videoFramesClosed++;
          }
        },
        error: (e) => {
          console.error('Decoder error', e);
          workerPool.terminate();
          reject(e);
        }
      });

      console.log("[DEBUG] Checking codec config support for:", config);
      VideoDecoder.isConfigSupported(config).then(support => {
        console.log("[DEBUG] Codec support check returned:", support);
        if (support.supported) {
          decoder.configure(config);
          console.log("[DEBUG] Decoder configured successfully. Setting extraction options.");
          mp4boxfile.setExtractionOptions(videoTrack.id, videoTrack, { nbSamples: 1 });
          mp4boxfile.start();
          console.log("[DEBUG] MP4Box extraction started. Seeking to 0.");
          const seekResult = mp4boxfile.seek(0, true);
          console.log("[DEBUG] Seek result:", seekResult);
          offset = (seekResult && typeof seekResult === 'object' && 'offset' in seekResult) 
            ? (seekResult as any).offset 
            : 0;
          console.log(`[DEBUG] Updated file reading offset to seek result offset: ${offset}`);
          isExtractionStarted = true;
          chunkSize = 1024 * 1024 * 5;
          resolveDecoderConfigured();
          if (resumeReading) {
            console.log("[DEBUG] Resuming file reading after decoder configuration.");
            resumeReading();
          }
        } else {
          console.error("[DEBUG] Codec NOT supported!");
          const err = new Error(`Codec ${config.codec} not supported`);
          rejectDecoderConfigured(err);
          reject(err);
        }
      }).catch(err => {
        console.error("[DEBUG] Error checking config support:", err);
        rejectDecoderConfigured(err);
        reject(err);
      });
    };

    mp4boxfile.onSamples = async (id: any, user: any, samples: any[]) => {
      console.log(`[DEBUG] MP4Box onSamples event fired: received ${samples.length} samples.`);
      pendingSamples.push(...samples);
      processSamples();
    };

    async function processSamples() {
      if (isProcessingSamples) return;
      isProcessingSamples = true;
      
      while (pendingSamples.length > 0) {
        if (decoder.decodeQueueSize > maxInflight || workerPool.inflightBitmaps > maxInflight) {
          await delay(10);
          continue;
        }
        
        const sample = pendingSamples.shift();
        const type = sample.is_sync ? 'key' : 'delta';
        const chunk = new EncodedVideoChunk({
          type,
          timestamp: sample.cts * 1_000_000 / sample.timescale,
          duration: sample.duration * 1_000_000 / sample.timescale,
          data: sample.data
        });
        decoder.decode(chunk);
      }
      
      isProcessingSamples = false;
      if (demuxCompleted && decoder && decoder.state === 'configured' && decoder.decodeQueueSize === 0) {
        await decoder.flush();
      }
    }

    const readNextChunk = () => {
      if (offset >= file.size) {
        if (!isExtractionStarted) {
          console.log("[DEBUG] File fully read, but waiting for extraction to start before flushing...");
          resumeReading = () => {
            if (offset < file.size) {
              console.log(`[DEBUG] Resuming file reading from offset ${offset} instead of flushing.`);
              readNextChunk();
            } else {
              console.log("[DEBUG] Flushing after file fully read and extraction started.");
              mp4boxfile.flush();
              demuxCompleted = true;
              processSamples();
            }
          };
          return;
        }
        console.log("[DEBUG] File fully read. Flushing mp4box.");
        mp4boxfile.flush();
        demuxCompleted = true;
        processSamples();
        return;
      }

      if (trackInfo && !isExtractionStarted) {
        console.log("[DEBUG] Pausing file reading until extraction starts...");
        resumeReading = () => readNextChunk();
        return;
      }

      const reader = new FileReader();
      const end = Math.min(offset + chunkSize, file.size);
      const slice = file.slice(offset, end);
      console.log(`[DEBUG] readNextChunk: reading chunk range [${offset} - ${end}] of file`);

      reader.onload = (e) => {
        const buffer = e.target?.result as ArrayBuffer;
        (buffer as any).fileStart = offset;
        offset += buffer.byteLength;
        mp4boxfile.appendBuffer(buffer as any);

        readNextChunk();
      };
      reader.readAsArrayBuffer(slice);
    };

    readNextChunk();
  });
}

export async function processVideoOnServer(
  file: File,
  videoId: string,
  onProgress: (framesProcessed: number, totalFrames: number, inflight: number) => void,
  onJobIdKnown?: (jobId: string) => void
): Promise<{ totalFrames: number, batches: number, jobId: string }> {
  const CHUNK_SIZE = 5 * 1024 * 1024;
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  const uploadId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  let jobId = '';

  // The server reuses this upload's own id as the eventual fingerprint job id
  // (see /api/upload-chunk), so the caller can persist a resumable session
  // immediately — before a large file even finishes transferring — instead of
  // waiting for the whole upload+extraction round trip to resolve. If a
  // checkpoint from a previous interrupted upload of the same file is found
  // server-side, the last chunk's response corrects it below.
  onJobIdKnown?.(uploadId);

  for (let i = 0; i < totalChunks; i++) {
    const chunk = file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
    const formData = new FormData();
    formData.append('chunk', chunk, file.name);
    formData.append('uploadId', uploadId);
    formData.append('chunkIndex', i.toString());
    formData.append('totalChunks', totalChunks.toString());
    formData.append('filename', file.name);
    formData.append('totalSize', file.size.toString());

    const uploadRes = await fetch('/api/upload-chunk', {
      method: 'POST',
      body: formData,
    });

    if (!uploadRes.ok) {
      const errText = await uploadRes.text();
      throw new Error(`Server upload failed at chunk ${i + 1}/${totalChunks}: ${errText}`);
    }

    if (i === totalChunks - 1) {
      const data = await uploadRes.json();
      jobId = data.jobId;
      if (jobId && jobId !== uploadId) onJobIdKnown?.(jobId);
    }
  }
  console.log(`[Server Process] File uploaded, Job ID: ${jobId}`);

  // Poll status endpoint. Retries transient network errors with exponential
  // backoff (same pattern used by the main App.tsx polling loops) instead of
  // throwing on the first hiccup — a single dropped request used to abort the
  // whole extraction with "Failed to fetch job status", which was especially
  // likely to be hit on long-running jobs (large videos / many frames).
  let done = false;
  let totalFrames = 0;
  let processedFrames = 0;
  let consecutiveErrors = 0;
  const MAX_CONSECUTIVE_ERRORS = 8;

  while (!done) {
    const delay = consecutiveErrors === 0
      ? 1000
      : Math.min(1000 * Math.pow(2, consecutiveErrors - 1), 15_000);
    await new Promise(resolve => setTimeout(resolve, delay));

    let statusRes: Response;
    try {
      statusRes = await fetch(`/api/status/${jobId}`);
    } catch {
      consecutiveErrors++;
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        throw new Error(`Failed to fetch job status for ${jobId} after ${MAX_CONSECUTIVE_ERRORS} attempts`);
      }
      continue;
    }
    if (!statusRes.ok) {
      consecutiveErrors++;
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        throw new Error(`Failed to fetch job status for ${jobId}`);
      }
      continue;
    }
    consecutiveErrors = 0;

    const job = await statusRes.json();
    totalFrames = job.totalFrames || 0;
    processedFrames = job.processedFrames || 0;

    onProgress(processedFrames, totalFrames, 0);

    if (job.status === 'completed') {
      done = true;
    } else if (job.status === 'failed') {
      throw new Error(`Server processing failed: ${job.error || 'Unknown error'}`);
    }
  }

  // Fetch processed results — retry transient failures the same way.
  let resultRes: Response;
  {
    let attempt = 0;
    while (true) {
      try {
        resultRes = await fetch(`/api/result/${jobId}`);
        if (resultRes.ok) break;
      } catch {
        // fall through to retry
      }
      attempt++;
      if (attempt >= MAX_CONSECUTIVE_ERRORS) {
        throw new Error(`Failed to fetch job results for ${jobId}`);
      }
      await new Promise(resolve => setTimeout(resolve, Math.min(1000 * Math.pow(2, attempt - 1), 15_000)));
    }
  }

  const fingerprints: FrameFingerprint[] = await resultRes.json();
  console.log(`[Server Process] Fetched ${fingerprints.length} fingerprints for job ${jobId}.`);

  const batchSize = 300;
  let batchIndex = 0;
  for (let i = 0; i < fingerprints.length; i += batchSize) {
    const chunk = fingerprints.slice(i, i + batchSize);
    await saveFingerprintsBatch(videoId, batchIndex++, chunk);
  }

  return { totalFrames: fingerprints.length, batches: batchIndex, jobId };
}
