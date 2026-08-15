import { parentPort } from 'worker_threads';
import { createCanvas } from 'canvas';
import { getCropRects, processSubtitles, detectContentBounds, maskWatermarkRegions, computeHashAndFeatures, computeSignature, FrameSignature, VariantHashes } from '../src/shared/fingerprint';

parentPort?.on('message', async (message) => {
  const { id, frameBuffer, width, height } = message;
  
  try {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);
    imgData.data.set(new Uint8ClampedArray(frameBuffer));
    ctx.putImageData(imgData, 0, 0);
    
    const variants: Record<string, VariantHashes> = {};
    
    // Downscale full frame to a standard intermediate size
    const H_down = 120;
    const W_down = Math.round(width * (H_down / height));
    
    const fullDownCanvas = createCanvas(W_down, H_down);
    const fullDownCtx = fullDownCanvas.getContext('2d');
    fullDownCtx.patternQuality = 'best';
    fullDownCtx.quality = 'best';
    fullDownCtx.imageSmoothingEnabled = true;
    
    fullDownCtx.fillStyle = '#000000';
    fullDownCtx.fillRect(0, 0, W_down, H_down);
    fullDownCtx.drawImage(canvas, 0, 0, width, height, 0, 0, W_down, H_down);
    
    // ── 1. Auto black-bar / letterbox / border trim ─────────────────────────
    // Applied at fingerprint time on the downscaled frame; when a clip has
    // letterbox bars or a frame-around-frame border, hashing proceeds on the
    // content region only, so query hashes align with bar-free movie hashes.
    const imgDataDown = fullDownCtx.getImageData(0, 0, W_down, H_down);
    const bounds = detectContentBounds(imgDataDown as any);

    let workCanvas = fullDownCanvas;
    let workCtx = fullDownCtx;
    let workW = W_down;
    let workH = H_down;
    if (bounds) {
      workCanvas = createCanvas(bounds.sw, bounds.sh);
      workCtx = workCanvas.getContext('2d');
      workCtx.patternQuality = 'best';
      workCtx.quality = 'best';
      workCtx.imageSmoothingEnabled = true;
      workCtx.drawImage(fullDownCanvas, bounds.sx, bounds.sy, bounds.sw, bounds.sh, 0, 0, bounds.sw, bounds.sh);
      workW = bounds.sw;
      workH = bounds.sh;
    }

    // ── 2. Full-frame text masking + watermark masking ──────────────────────
    // forceFullPass=true: borderline subtitle cases are no longer skipped
    // (the relaxed gate lives inside processSubtitles v2).
    const workData = workCtx.getImageData(0, 0, workW, workH);
    const subChanged = processSubtitles(workData as any, true);
    const wmChanged  = maskWatermarkRegions(workData as any);
    if (subChanged || wmChanged) {
      workCtx.putImageData(workData, 0, 0);
    }
    const changed = bounds !== null || subChanged || wmChanged;

    // Crop rects follow the effective (trimmed) frame when preprocessing ran
    const rects = changed ? getCropRects(workW, workH) : getCropRects(width, height);
    
    const finalCanvas = createCanvas(16, 16);
    const finalCtx = finalCanvas.getContext('2d');
    finalCtx.patternQuality = 'best';
    finalCtx.quality = 'best';
    finalCtx.imageSmoothingEnabled = true;

    let signature: FrameSignature | undefined;
    
    for (const rect of rects) {
      finalCtx.fillStyle = '#000000';
      finalCtx.fillRect(0, 0, 16, 16);
      if (!changed) {
        finalCtx.drawImage(canvas, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, 16, 16);
      } else {
        // Rects are already in workCanvas coordinates
        finalCtx.drawImage(workCanvas, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, 16, 16);
      }
      const finalImgData = finalCtx.getImageData(0, 0, 16, 16);

      // Compute signature only for the 'full' variant (one per frame)
      const isFullVariant = rect.name === 'full';
      const features = computeHashAndFeatures(finalImgData as any, isFullVariant);
      variants[rect.name] = {
        hash: features.hash,
        dhash: features.dhash,
        fhash: features.fhash,
        fdhash: features.fdhash,
        phash: features.phash,
      };
      if (isFullVariant && features.signature) {
        signature = features.signature;
      }
    }
    
    parentPort?.postMessage({ id, result: { variants, signature } });
  } catch (error: any) {
    parentPort?.postMessage({ id, error: error.message || String(error) });
  }
});
