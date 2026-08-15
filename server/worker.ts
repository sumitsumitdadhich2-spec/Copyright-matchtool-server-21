import { parentPort } from 'worker_threads';
import { createCanvas } from 'canvas';
import { getCropRects, processSubtitles, computeHashAndFeatures, computeSignature, FrameSignature, VariantHashes } from '../src/shared/fingerprint';

parentPort?.on('message', async (message) => {
  const { id, frameBuffer, width, height } = message;
  
  try {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);
    imgData.data.set(new Uint8ClampedArray(frameBuffer));
    ctx.putImageData(imgData, 0, 0);
    
    const rects = getCropRects(width, height);
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
    
    const imgDataDown = fullDownCtx.getImageData(0, 0, W_down, H_down);
    const changed = processSubtitles(imgDataDown as any, false);
    if (changed) {
      fullDownCtx.putImageData(imgDataDown, 0, 0);
    }
    
    const finalCanvas = createCanvas(16, 16);
    const finalCtx = finalCanvas.getContext('2d');
    finalCtx.patternQuality = 'best';
    finalCtx.quality = 'best';
    finalCtx.imageSmoothingEnabled = true;
    
    const scaleX = W_down / width;
    const scaleY = H_down / height;

    let signature: FrameSignature | undefined;
    
    for (const rect of rects) {
      finalCtx.fillStyle = '#000000';
      finalCtx.fillRect(0, 0, 16, 16);
      if (!changed) {
        finalCtx.drawImage(canvas, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, 16, 16);
      } else {
        const sx = rect.sx * scaleX;
        const sy = rect.sy * scaleY;
        const sw = rect.sw * scaleX;
        const sh = rect.sh * scaleY;
        finalCtx.drawImage(fullDownCanvas, sx, sy, sw, sh, 0, 0, 16, 16);
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
