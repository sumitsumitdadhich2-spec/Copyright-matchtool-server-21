/**
 * Side-by-side frame composite for VLM verification (Task 3 protocol fix).
 *
 * Joins the two frames of a verification pair into ONE image — short-clip
 * frame on the LEFT (labeled "A"), movie frame on the RIGHT (labeled "B"),
 * separated by an 8px divider. Sending a single composite instead of two
 * separate images removes a major source of VLM confusion (models mixing up
 * which image is which, or attending to only one of them).
 *
 * Uses the `canvas` package (already a project dependency) — no ffmpeg
 * subprocess, no temp files, no fontconfig dependency issues.
 *
 * FAIL-SAFE: any decode/render failure returns null. Callers must treat null
 * as "composite unavailable" and fall back to the legacy two-image protocol
 * — never throw, never drop the verification.
 */
import { createCanvas, loadImage, Image } from 'canvas';

/** Both halves are scaled to this common height before joining. */
const COMPOSITE_HEIGHT = Number(process.env.VLM_COMPOSITE_HEIGHT) || 448;
const DIVIDER_PX = 8;
/** JPEG quality of the output composite. */
const COMPOSITE_JPEG_QUALITY = 0.85;

function scaledWidth(img: Image): number {
  if (!img.height) return 0;
  return Math.max(1, Math.round((img.width * COMPOSITE_HEIGHT) / img.height));
}

function drawLabel(ctx: any, text: string, x: number): void {
  const boxW = 34;
  const boxH = 34;
  ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
  ctx.fillRect(x, 0, boxW, boxH);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 24px sans-serif';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.fillText(text, x + boxW / 2, boxH / 2 + 1);
}

/**
 * Build the labeled side-by-side composite.
 *
 * @param leftB64  base64 JPEG of the LEFT frame (labeled "A" — short clip)
 * @param rightB64 base64 JPEG of the RIGHT frame (labeled "B" — movie)
 * @returns base64 JPEG of the composite, or null on ANY failure.
 */
export async function buildSideBySideComposite(
  leftB64: string,
  rightB64: string,
): Promise<string | null> {
  try {
    const [left, right] = await Promise.all([
      loadImage(Buffer.from(leftB64, 'base64')),
      loadImage(Buffer.from(rightB64, 'base64')),
    ]);

    const leftW = scaledWidth(left);
    const rightW = scaledWidth(right);
    if (leftW === 0 || rightW === 0) return null;

    const totalW = leftW + DIVIDER_PX + rightW;
    const canvas = createCanvas(totalW, COMPOSITE_HEIGHT);
    const ctx = canvas.getContext('2d');

    // Divider — solid mid-gray so it reads as a separator, not content.
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, totalW, COMPOSITE_HEIGHT);

    ctx.drawImage(left, 0, 0, leftW, COMPOSITE_HEIGHT);
    ctx.drawImage(right, leftW + DIVIDER_PX, 0, rightW, COMPOSITE_HEIGHT);

    drawLabel(ctx, 'A', 0);
    drawLabel(ctx, 'B', leftW + DIVIDER_PX);

    return canvas
      .toBuffer('image/jpeg', { quality: COMPOSITE_JPEG_QUALITY })
      .toString('base64');
  } catch (err: any) {
    console.warn(`[Verify] composite build failed (falling back to two-image protocol): ${err?.message || err}`);
    return null;
  }
}
