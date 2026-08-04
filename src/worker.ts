import { computeFingerprint } from './shared/fingerprint';

self.onmessage = async (e) => {
  const { bitmap, frameIndex, timestamp } = e.data;
  
  // Create an offscreen canvas
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as any;
  ctx.drawImage(bitmap, 0, 0);

  try {
    const fingerprint = computeFingerprint(ctx, bitmap.width, bitmap.height, frameIndex, timestamp);
    bitmap.close();
    self.postMessage({ result: fingerprint, closed: true });
  } catch (err: any) {
    bitmap.close();
    self.postMessage({ error: err.message, closed: true });
  }
};
