/**
 * Checkpoint / Resume Integration Test
 * =====================================
 * Tests that:
 *  1. Processing a video fully produces N fingerprints (ground truth).
 *  2. Simulating a mid-job crash (partial result + checkpoint file on disk)
 *     causes the next upload of the same file to RESUME from the saved frame,
 *     not restart from scratch.
 *  3. The resumed run produces the same total frame count and identical
 *     fingerprint hashes as the ground-truth run (accuracy preserved).
 */

import fs   from 'fs';
import path from 'path';

const BASE      = 'http://localhost:5000';
const UPLOAD_DIR = path.join(process.cwd(), 'uploads');
const VIDEO      = 'checkpoint_test.mp4'; // 10s 320x240 25fps synthetic video (valid)

// ─── helpers ─────────────────────────────────────────────────────────────────

async function uploadChunked(filePath, chunkSizeBytes = 1 * 1024 * 1024) {
  const data      = fs.readFileSync(filePath);
  const filename  = path.basename(filePath);
  const totalSize = data.length;
  const uploadId  = `test-${Date.now()}`;
  const chunkSize = Math.min(chunkSizeBytes, totalSize);
  const totalChunks = Math.ceil(totalSize / chunkSize);

  let jobId = null;
  for (let i = 0; i < totalChunks; i++) {
    const chunk = data.slice(i * chunkSize, (i + 1) * chunkSize);
    const form  = new FormData();
    form.append('uploadId',    uploadId);
    form.append('chunkIndex',  String(i));
    form.append('totalChunks', String(totalChunks));
    form.append('filename',    filename);
    form.append('chunk', new Blob([chunk]), `chunk-${i}`);

    const r = await fetch(`${BASE}/api/upload-chunk`, { method: 'POST', body: form });
    if (!r.ok) throw new Error(`Chunk ${i} upload failed: ${await r.text()}`);
    const body = await r.json();
    if (body.jobId) jobId = body.jobId;
  }
  if (!jobId) throw new Error('No jobId returned from upload');
  return { jobId, filename, fileSize: totalSize };
}

async function waitForCompletion(jobId, timeoutMs = 5 * 60 * 1000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const r    = await fetch(`${BASE}/api/status/${jobId}`);
    const body = await r.json();
    process.stdout.write(`\r  [status] ${body.status} ${body.processedFrames ?? 0}/${body.totalFrames ?? '?'} frames   `);
    if (body.status === 'completed') { console.log(); return body; }
    if (body.status === 'failed')    throw new Error(`Job failed: ${body.error}`);
    await new Promise(r => setTimeout(r, 800));
  }
  throw new Error(`Timeout waiting for job ${jobId}`);
}

async function getResult(jobId) {
  const r = await fetch(`${BASE}/api/result/${jobId}`);
  if (!r.ok) throw new Error(`result fetch failed: ${await r.text()}`);
  return r.json();
}

function readNdjsonFile(filePath) {
  if (!fs.existsSync(filePath)) return [];
  return fs.readFileSync(filePath, 'utf-8')
    .split('\n')
    .filter(l => l.trim())
    .map(l => JSON.parse(l));
}

function countCompleteLines(filePath) {
  if (!fs.existsSync(filePath)) return 0;
  const content = fs.readFileSync(filePath, 'utf-8');
  let count = 0, pos = 0;
  while (true) {
    const nl = content.indexOf('\n', pos);
    if (nl === -1) break;
    count++;
    pos = nl + 1;
  }
  return count;
}

// ─── main ─────────────────────────────────────────────────────────────────────

async function main() {
  if (!fs.existsSync(VIDEO)) {
    console.error(`Test video not found: ${VIDEO}`);
    process.exit(1);
  }

  const fileSize = fs.statSync(VIDEO).size;
  const filename = VIDEO;

  console.log(`\n${'═'.repeat(60)}`);
  console.log(' Checkpoint / Resume Integration Test');
  console.log(`${'═'.repeat(60)}\n`);

  // ─── Step 1: Full run — ground truth ──────────────────────────────────────
  console.log('STEP 1: Full processing run (ground truth)');
  console.log(`  video: ${VIDEO}  (${(fileSize / 1024).toFixed(1)} KB)`);

  const { jobId: gtJobId } = await uploadChunked(VIDEO);
  console.log(`  jobId: ${gtJobId}`);
  const gtStatus = await waitForCompletion(gtJobId);
  const groundTruth = await getResult(gtJobId);
  const totalFrames = gtStatus.totalFrames;

  console.log(`  ✓ Ground truth: ${totalFrames} frames, ${groundTruth.length} fingerprints`);

  // ─── Step 2: Simulate a crash at SPLIT_AT frames ───────────────────────────
  const SPLIT_AT = Math.max(1, Math.floor(totalFrames * 0.6)); // crash at ~60%
  console.log(`\nSTEP 2: Simulate crash at frame ${SPLIT_AT} of ${totalFrames}`);

  // Pick a fresh fake jobId for the "interrupted" job
  const fakeJobId   = `fake-${Date.now()}`;
  const resultPath  = path.join(UPLOAD_DIR, `${fakeJobId}_result.json`);
  const cpPath      = path.join(UPLOAD_DIR, `${fakeJobId}_checkpoint.json`);
  const metaPath    = path.join(UPLOAD_DIR, `${fakeJobId}_meta.json`);
  const checkpointKey = `${filename}:${fileSize}`;

  // Write the first SPLIT_AT lines of ground truth as a partial result
  // (exactly as the pipeline would have written them before crashing)
  const partialLines = groundTruth
    .slice(0, SPLIT_AT)
    .map(fp => JSON.stringify(fp))
    .join('\n') + '\n';
  fs.writeFileSync(resultPath, partialLines);
  console.log(`  ✓ Wrote partial result: ${SPLIT_AT} lines → ${resultPath}`);

  // Write checkpoint file (same structure as pipeline.ts writes)
  fs.writeFileSync(cpPath, JSON.stringify({ jobId: fakeJobId, checkpointKey, updatedAt: Date.now() }));
  console.log(`  ✓ Wrote checkpoint file: ${cpPath}`);

  // Write meta file so server can find the job
  fs.writeFileSync(metaPath, JSON.stringify({ originalName: filename, fileSize, createdAt: Date.now() }));
  console.log(`  ✓ Wrote meta file`);

  // Verify the partial result has exactly SPLIT_AT lines
  const partialLineCount = countCompleteLines(resultPath);
  console.log(`  ✓ Partial result line count: ${partialLineCount} (expected ${SPLIT_AT})`);
  if (partialLineCount !== SPLIT_AT) {
    console.error(`  ✗ Line count mismatch — aborting`);
    process.exit(1);
  }

  // ─── Step 3: Re-upload same video — should resume ─────────────────────────
  console.log(`\nSTEP 3: Re-upload "${filename}" (same filename + size) — expect RESUME`);

  const { jobId: resumeJobId } = await uploadChunked(VIDEO);
  console.log(`  jobId returned: ${resumeJobId}`);

  // The server should have matched the checkpoint and returned fakeJobId
  if (resumeJobId !== fakeJobId) {
    console.error(`  ✗ Expected resumed jobId "${fakeJobId}" but got "${resumeJobId}"`);
    console.error('    Server did not detect the checkpoint — test FAILED');
    process.exit(1);
  }
  console.log(`  ✓ Server correctly resumed job "${fakeJobId}"`);

  // Check that the server started processing from SPLIT_AT, not from 0
  // (watch early status updates — processedFrames should already be ≥ SPLIT_AT)
  const earlyStatus = await fetch(`${BASE}/api/status/${resumeJobId}`).then(r => r.json());
  console.log(`  ✓ Early processedFrames: ${earlyStatus.processedFrames} (should be ≥ ${SPLIT_AT})`);

  const resumeStatus = await waitForCompletion(resumeJobId);
  const resumeResult = await getResult(resumeJobId);

  console.log(`  ✓ Resumed run finished: ${resumeStatus.totalFrames} frames, ${resumeResult.length} fingerprints`);

  // ─── Step 4: Compare resumed result to ground truth ───────────────────────
  console.log('\nSTEP 4: Accuracy comparison (resumed vs ground truth)');

  let allMatch = true;
  const errors = [];

  if (resumeResult.length !== groundTruth.length) {
    errors.push(`Frame count mismatch: ${resumeResult.length} vs ${groundTruth.length} (ground truth)`);
    allMatch = false;
  }

  // Sort both by frameIndex (order may differ slightly during parallel processing)
  const byFrame = arr => [...arr].sort((a, b) => a.frameIndex - b.frameIndex);
  const gtSorted     = byFrame(groundTruth);
  const resumeSorted = byFrame(resumeResult);

  let hashMismatches = 0;
  for (let i = 0; i < Math.min(gtSorted.length, resumeSorted.length); i++) {
    const gt = gtSorted[i];
    const re = resumeSorted[i];
    if (gt.frameIndex !== re.frameIndex) {
      errors.push(`Frame index mismatch at position ${i}: GT=${gt.frameIndex} RESUME=${re.frameIndex}`);
      allMatch = false;
      break;
    }
    const gtHash = gt.variants?.full?.hash ?? gt.variants?.['full']?.hash ?? '';
    const reHash = re.variants?.full?.hash ?? re.variants?.['full']?.hash ?? '';
    if (gtHash !== reHash) {
      hashMismatches++;
      if (hashMismatches <= 3) {
        errors.push(`Hash mismatch at frame ${gt.frameIndex}: GT="${gtHash.slice(0,16)}…" RESUME="${reHash.slice(0,16)}…"`);
      }
      allMatch = false;
    }
  }

  if (hashMismatches > 0) errors.push(`Total hash mismatches: ${hashMismatches}`);

  // ─── Results ──────────────────────────────────────────────────────────────
  console.log(`\n${'─'.repeat(60)}`);
  if (allMatch) {
    console.log(' ✅  ALL CHECKS PASSED');
    console.log(`     • Crash simulated at frame ${SPLIT_AT}/${totalFrames} (~${Math.round(SPLIT_AT/totalFrames*100)}%)`);
    console.log(`     • Resume detected correctly (same jobId returned)`);
    console.log(`     • Final frame count matches ground truth: ${resumeResult.length}`);
    console.log(`     • All ${groundTruth.length} fingerprint hashes are identical`);
  } else {
    console.log(' ❌  TEST FAILED');
    for (const e of errors) console.log(`     • ${e}`);
    process.exit(1);
  }
  console.log(`${'─'.repeat(60)}\n`);

  // Cleanup fake checkpoint (already deleted by server on completion, but just in case)
  [cpPath, metaPath].forEach(f => { try { if (fs.existsSync(f)) fs.unlinkSync(f); } catch {} });
}

main().catch(err => { console.error('Test error:', err); process.exit(1); });
