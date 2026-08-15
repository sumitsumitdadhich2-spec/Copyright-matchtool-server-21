export interface FrameSignature {
  /** Flat array: 4x4 grid of cells, each cell has avg R, G, B = 48 values total (0-255) */
  colorGrid: number[];
  /** 4x4 = 16 values, fraction of pixels per cell matching skin-tone heuristic (0-1) */
  skinScoreGrid: number[];
  /** 4x4 = 16 values, mean-absolute-deviation of grayscale per cell (texture measure, 0-255) */
  detailGrid: number[];
}

export interface VariantHashes {
  /** Average hash (256-bit for 16x16), brightness-threshold based */
  hash: string;
  /** Gradient/difference hash (480-bit: horizontal + vertical), robust to color grading & brightness/contrast edits */
  dhash?: string;
  /** Average hash of the horizontally flipped frame (mirror-edit detection) */
  fhash?: string;
  /** Gradient hash of the horizontally flipped frame */
  fdhash?: string;
  /** DCT-based perceptual hash (64-bit): robust to compression and colour-grading edits */
  phash?: string;
}

export interface FrameFingerprint {
  frameIndex: number;
  timestamp: number;
  variants: Record<string, VariantHashes>;
  /** Optional: computed from the 'full' variant 16x16 image for weighted similarity */
  signature?: FrameSignature;
}

export interface CropRect {
  name: string;
  sx: number;
  sy: number;
  sw: number;
  sh: number;
}

export function getCropRects(width: number, height: number): CropRect[] {
  const rects: CropRect[] = [];
  
  // 1. Full variant
  rects.push({ name: 'full', sx: 0, sy: 0, sw: width, sh: height });
  
  // 2. 9:16 variants (5 crops)
  let cropWidth = Math.round(height * (9 / 16));
  if (cropWidth % 2 !== 0) cropWidth--;
  
  if (cropWidth <= width) {
    const step = (width - cropWidth) / 4;
    for (let i = 0; i < 5; i++) {
      let sx = Math.round(i * step);
      if (sx % 2 !== 0) sx--;
      rects.push({
        name: `crop_9_16_${i}`,
        sx,
        sy: 0,
        sw: cropWidth,
        sh: height
      });
    }
  } else {
    for (let i = 0; i < 5; i++) {
      rects.push({
        name: `crop_9_16_${i}`,
        sx: 0,
        sy: 0,
        sw: width,
        sh: height
      });
    }
  }

  // 3. Zoom crops helper
  const addZoomCrops = (zoom: number, namePrefix: string) => {
    let sw = Math.min(width, Math.max(1, Math.round(width / zoom)));
    let sh = Math.min(height, Math.max(1, Math.round(height / zoom)));
    if (sw % 2 !== 0) sw--;
    if (sh % 2 !== 0) sh--;
    
    let sy = Math.min(height - sh, Math.max(0, Math.round((height - sh) / 2)));
    if (sy % 2 !== 0) sy--;
    
    let sxCenter = Math.min(width - sw, Math.max(0, Math.round((width - sw) / 2)));
    if (sxCenter % 2 !== 0) sxCenter--;
    
    rects.push({ name: `${namePrefix}_center`, sx: sxCenter, sy, sw, sh });
    rects.push({ name: `${namePrefix}_left`, sx: 0, sy, sw, sh });
    
    let sxRight = Math.min(width - sw, Math.max(0, width - sw));
    if (sxRight % 2 !== 0) sxRight--;
    rects.push({ name: `${namePrefix}_right`, sx: sxRight, sy, sw, sh });
  };

  addZoomCrops(1.25, 'zoom_1_25');
  addZoomCrops(1.5, 'zoom_1_5');

  // 2.0x zoom (Center only)
  let sw2 = Math.min(width, Math.max(1, Math.round(width / 2.0)));
  let sh2 = Math.min(height, Math.max(1, Math.round(height / 2.0)));
  if (sw2 % 2 !== 0) sw2--;
  if (sh2 % 2 !== 0) sh2--;
  
  let sx2 = Math.min(width - sw2, Math.max(0, Math.round((width - sw2) / 2)));
  if (sx2 % 2 !== 0) sx2--;
  
  let sy2 = Math.min(height - sh2, Math.max(0, Math.round((height - sh2) / 2)));
  if (sy2 % 2 !== 0) sy2--;
  
  rects.push({ name: 'zoom_2_0_center', sx: sx2, sy: sy2, sw: sw2, sh: sh2 });
  
  return rects;
}

// ---------------------------------------------------------------------------
// Shared inpaint helper — fills masked pixels from the nearest unmasked pixel
// above; falls back to the nearest unmasked pixel below (needed for text at
// the very top of the frame, which has no source rows above it).
// ---------------------------------------------------------------------------
function inpaintMask(data: Uint8ClampedArray, width: number, height: number, mask: Uint8Array): void {
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y * width + x] !== 1) continue;
      let sourceY = -1;
      for (let sy = y - 1; sy >= 0; sy--) {
        if (mask[sy * width + x] === 0) { sourceY = sy; break; }
      }
      if (sourceY === -1) {
        for (let sy = y + 1; sy < height; sy++) {
          if (mask[sy * width + x] === 0) { sourceY = sy; break; }
        }
      }
      if (sourceY !== -1) {
        const targetIdx = (y * width + x) * 4;
        const sourceIdx = (sourceY * width + x) * 4;
        data[targetIdx]     = data[sourceIdx];
        data[targetIdx + 1] = data[sourceIdx + 1];
        data[targetIdx + 2] = data[sourceIdx + 2];
        data[targetIdx + 3] = data[sourceIdx + 3];
      }
    }
  }
}

/** 5×5 dilation of a binary mask (2px radius) */
function dilateMask(mask: Uint8Array, width: number, height: number, radius = 2): Uint8Array {
  const dilated = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y * width + x] !== 1) continue;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            dilated[ny * width + nx] = 1;
          }
        }
      }
    }
  }
  return dilated;
}

/**
 * Detect black bars / letterbox / pillarbox / uniform frame-around-frame
 * borders and return the content rectangle (in this image's coordinates),
 * or null when no trimming is needed.
 *
 * A row/column counts as a "bar" when nearly all of its pixels are close to
 * the border reference color (sampled from the outermost row/column) AND
 * that reference color is itself very flat (uniform). Works for classic
 * black letterbox, white borders, and colored frame-around-frame edits.
 *
 * Designed to run on the downscaled (~120px-high) frame — cheap, and small
 * 1-2px trims at this scale correspond to real bars at full resolution.
 */
export function detectContentBounds(
  imageData: ImageData
): { sx: number; sy: number; sw: number; sh: number } | null {
  const { width, height, data } = imageData;
  if (width < 8 || height < 8) return null;

  const COLOR_TOL   = 26;   // per-channel distance to reference color to count as "bar pixel"
  const BAR_FRAC    = 0.94; // ≥94% of row/col pixels must match reference
  const MAX_TRIM    = 0.42; // never trim more than 42% from one side

  const rowMatches = (y: number, ref: [number, number, number]): boolean => {
    let matches = 0;
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      if (
        Math.abs(data[idx]     - ref[0]) <= COLOR_TOL &&
        Math.abs(data[idx + 1] - ref[1]) <= COLOR_TOL &&
        Math.abs(data[idx + 2] - ref[2]) <= COLOR_TOL
      ) matches++;
    }
    return matches / width >= BAR_FRAC;
  };
  const colMatches = (x: number, ref: [number, number, number]): boolean => {
    let matches = 0;
    for (let y = 0; y < height; y++) {
      const idx = (y * width + x) * 4;
      if (
        Math.abs(data[idx]     - ref[0]) <= COLOR_TOL &&
        Math.abs(data[idx + 1] - ref[1]) <= COLOR_TOL &&
        Math.abs(data[idx + 2] - ref[2]) <= COLOR_TOL
      ) matches++;
    }
    return matches / height >= BAR_FRAC;
  };

  /** Median-ish reference color of a row/col: average of its pixels */
  const rowRef = (y: number): [number, number, number] => {
    let r = 0, g = 0, b = 0;
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      r += data[idx]; g += data[idx + 1]; b += data[idx + 2];
    }
    return [r / width, g / width, b / width];
  };
  const colRef = (x: number): [number, number, number] => {
    let r = 0, g = 0, b = 0;
    for (let y = 0; y < height; y++) {
      const idx = (y * width + x) * 4;
      r += data[idx]; g += data[idx + 1]; b += data[idx + 2];
    }
    return [r / height, g / height, b / height];
  };

  const maxTrimY = Math.floor(height * MAX_TRIM);
  const maxTrimX = Math.floor(width * MAX_TRIM);

  // Top
  let top = 0;
  {
    const ref = rowRef(0);
    if (rowMatches(0, ref)) {
      while (top < maxTrimY && rowMatches(top, ref)) top++;
    }
  }
  // Bottom
  let bottom = 0;
  {
    const ref = rowRef(height - 1);
    if (rowMatches(height - 1, ref)) {
      while (bottom < maxTrimY && rowMatches(height - 1 - bottom, ref)) bottom++;
    }
  }
  // Left
  let left = 0;
  {
    const ref = colRef(0);
    if (colMatches(0, ref)) {
      while (left < maxTrimX && colMatches(left, ref)) left++;
    }
  }
  // Right
  let right = 0;
  {
    const ref = colRef(width - 1);
    if (colMatches(width - 1, ref)) {
      while (right < maxTrimX && colMatches(width - 1 - right, ref)) right++;
    }
  }

  // Ignore sub-pixel noise: require at least 2px trimmed somewhere (at
  // ~120px-high downscale, 2px ≈ a real bar at source resolution).
  if (top + bottom < 2 && left + right < 2) return null;
  if (top + bottom < 2) { top = 0; bottom = 0; }
  if (left + right < 2) { left = 0; right = 0; }
  if (top === 0 && bottom === 0 && left === 0 && right === 0) return null;

  const sw = width  - left - right;
  const sh = height - top  - bottom;
  // Content must remain a meaningful portion of the frame
  if (sw < width * 0.3 || sh < height * 0.3) return null;

  return { sx: left, sy: top, sw, sh };
}

/**
 * Detect and inpaint watermark-like overlays (channel logos, app watermarks)
 * in the four corners and the center of the frame.
 *
 * Heuristic per region: bright, low-saturation pixels (typical of white /
 * semi-transparent logos) that occupy a small-but-real fraction of the
 * region box. Only masks when the overlay footprint is plausible (2–40% of
 * the corner box; tighter 2–18% for the center to avoid nuking faces/skin),
 * so ordinary bright content is left untouched.
 */
export function maskWatermarkRegions(imageData: ImageData): boolean {
  const { width, height, data } = imageData;
  if (width < 16 || height < 16) return false;

  const boxW = Math.max(4, Math.floor(width  * 0.18));
  const boxH = Math.max(4, Math.floor(height * 0.18));
  const regions: Array<{ x0: number; y0: number; maxFrac: number }> = [
    { x0: 0,             y0: 0,              maxFrac: 0.40 }, // top-left
    { x0: width - boxW,  y0: 0,              maxFrac: 0.40 }, // top-right
    { x0: 0,             y0: height - boxH,  maxFrac: 0.40 }, // bottom-left
    { x0: width - boxW,  y0: height - boxH,  maxFrac: 0.40 }, // bottom-right
    {
      x0: Math.floor((width - boxW) / 2),
      y0: Math.floor((height - boxH) / 2),
      maxFrac: 0.18,                                          // center (conservative)
    },
  ];

  const mask = new Uint8Array(width * height);
  let masked = false;

  for (const region of regions) {
    const boxMask: Array<number> = [];
    let count = 0;
    for (let y = region.y0; y < region.y0 + boxH; y++) {
      for (let x = region.x0; x < region.x0 + boxW; x++) {
        const idx = (y * width + x) * 4;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        const mx = Math.max(r, g, b);
        const mn = Math.min(r, g, b);
        // Bright + low saturation = typical white/gray logo overlay
        if (mx > 195 && (mx - mn) < 45) {
          boxMask.push(y * width + x);
          count++;
        }
      }
    }
    const frac = count / (boxW * boxH);
    if (frac >= 0.02 && frac <= region.maxFrac) {
      for (const i of boxMask) mask[i] = 1;
      masked = true;
    }
  }

  if (!masked) return false;
  const dilated = dilateMask(mask, width, height, 1);
  inpaintMask(data, width, height, dilated);
  return true;
}

/**
 * Detect and inpaint on-screen text ANYWHERE in the frame (v2).
 *
 * v1 limitations fixed here:
 *  - Only scanned the bottom 25% (startY = 75% height) → Shorts/Reels-style
 *    center or top captions were never caught. Now scans the FULL frame.
 *  - Only caught bright white-ish pixels (r>180 && g>180) → black text,
 *    colored captions, and outlined text were missed. Detection now covers:
 *      1. Bright white-ish text  (classic subtitles)
 *      2. Bright saturated text  (yellow / green / cyan / magenta captions)
 *      3. Dark / outlined text   (near-black pixels with a high-contrast
 *                                 bright neighbour — text outlines)
 *
 * `forceFullPass` now acts as a smarter borderline heuristic: when true, the
 * minimum-pixel gate is relaxed (4 instead of 8) so borderline subtitle
 * cases are no longer skipped. When nothing text-like is found the function
 * returns false without touching the image (no wasted redraw).
 */
export function processSubtitles(imageData: ImageData, forceFullPass: boolean): boolean {
  const { width, height, data } = imageData;
  const n = width * height;

  // Precompute luminance for contrast checks (dark-outlined text detection)
  const luma = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    luma[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }

  const mask = new Uint8Array(n);
  let textPixelCount = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const idx = i * 4;
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const mx = Math.max(r, g, b);
      const mn = Math.min(r, g, b);

      let isText = false;
      // 1. Bright white-ish (classic subtitles)
      if (r > 170 && g > 170 && b > 140) {
        isText = true;
      }
      // 2. Bright saturated caption colors (yellow, green, cyan, magenta…)
      else if (mx > 170 && (mx - mn) > 90) {
        isText = true;
      }
      // 3. Dark / outlined text: near-black pixel with a very bright
      //    immediate neighbour (typical text outline contrast)
      else if (mx < 45) {
        const l  = luma[i];
        const up = y > 0          ? luma[i - width] : l;
        const dn = y < height - 1 ? luma[i + width] : l;
        const lf = x > 0          ? luma[i - 1]     : l;
        const rt = x < width - 1  ? luma[i + 1]     : l;
        if (Math.max(up, dn, lf, rt) - l > 95) isText = true;
      }

      if (isText) {
        mask[i] = 1;
        textPixelCount++;
      }
    }
  }

  // Plausibility gates: too few pixels = noise, too many = actual bright
  // scene content (sky, snow, spotlight) — do NOT mask in that case.
  const minCount = forceFullPass ? 4 : 8;
  if (textPixelCount < minCount || textPixelCount > n * 0.30) {
    return false;
  }

  const dilated = dilateMask(mask, width, height, 2);
  inpaintMask(data, width, height, dilated);
  return true;
}

/**
 * Compute a 4×4 spatial signature from a 16×16 ImageData.
 * Each cell covers 4×4 pixels, giving:
 *   colorGrid:     16 cells × 3 (avg R,G,B) = 48 values  [0-255]
 *   skinScoreGrid: 16 cells × 1 (skin fraction)            [0-1]
 *   detailGrid:    16 cells × 1 (MAD of gray, texture)    [0-255]
 */
export function computeSignature(imageData: ImageData): FrameSignature {
  const { width, height, data } = imageData;
  const cellW = Math.max(1, Math.floor(width / 4));
  const cellH = Math.max(1, Math.floor(height / 4));

  const colorGrid: number[] = [];
  const skinScoreGrid: number[] = [];
  const detailGrid: number[] = [];

  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      const startY = row * cellH;
      const startX = col * cellW;
      const endY = Math.min(startY + cellH, height);
      const endX = Math.min(startX + cellW, width);

      let sumR = 0, sumG = 0, sumB = 0;
      let skinCount = 0;
      const grays: number[] = [];

      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const idx = (y * width + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];

          sumR += r; sumG += g; sumB += b;

          // Simple skin-tone heuristic (works on both light and medium skin)
          if (r > 80 && g > 40 && b > 20 && r > g && r > b && (r - b) > 20) {
            skinCount++;
          }

          grays.push(0.299 * r + 0.587 * g + 0.114 * b);
        }
      }

      const n = grays.length || 1;
      colorGrid.push(sumR / n, sumG / n, sumB / n);
      skinScoreGrid.push(skinCount / n);

      // Mean-absolute-deviation as texture measure
      const meanGray = grays.reduce((a, v) => a + v, 0) / n;
      const mad = grays.reduce((a, v) => a + Math.abs(v - meanGray), 0) / n;
      detailGrid.push(mad);
    }
  }

  return { colorGrid, skinScoreGrid, detailGrid };
}

/** Build average hash from gray values against a threshold */
function buildAHash(g: Float32Array, len: number, threshold: number): string {
  const bits = new Array<string>(len);
  for (let i = 0; i < len; i++) bits[i] = g[i] >= threshold ? '1' : '0';
  return bits.join('');
}

/**
 * Build gradient (difference) hash: horizontal comparisons ((w-1)*h bits)
 * followed by vertical comparisons (w*(h-1) bits).
 * Gradient sign is invariant to brightness/contrast/gamma edits and most
 * color grading, making it far more robust than average hash for edited videos.
 */
function buildDHash(g: Float32Array, width: number, height: number): string {
  const bits: string[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width - 1; x++) {
      bits.push(g[y * width + x + 1] > g[y * width + x] ? '1' : '0');
    }
  }
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width; x++) {
      bits.push(g[(y + 1) * width + x] > g[y * width + x] ? '1' : '0');
    }
  }
  return bits.join('');
}

// ---------------------------------------------------------------------------
// pHash (DCT-based perceptual hash) — module-level cosine cache
// ---------------------------------------------------------------------------

/** Cosine table for the 2D DCT-II on a 16×16 image, computed once on first use. */
let _pHashCos: Float64Array | null = null;

function getPHashCos(): Float64Array {
  if (_pHashCos) return _pHashCos;
  // cos((2x+1)*u*π/32) for x in [0,16), u in [0,8)
  // Shared for rows and columns (image is square 16×16, K=8 kept frequencies)
  _pHashCos = new Float64Array(16 * 8);
  for (let x = 0; x < 16; x++) {
    for (let u = 0; u < 8; u++) {
      _pHashCos[x * 8 + u] = Math.cos(((2 * x + 1) * u * Math.PI) / 32);
    }
  }
  return _pHashCos;
}

/**
 * Build a 64-bit DCT-based perceptual hash (pHash) from a 16×16 grayscale grid.
 *
 * Algorithm:
 *  1. Separable 2D DCT-II on the 16×16 input → extract top-left 8×8 (64 values).
 *  2. Compute median of the 63 AC coefficients (exclude DC at [0,0]).
 *  3. Each bit is '1' if coefficient > median, '0' otherwise → 64-bit binary string.
 *
 * Robust to JPEG/video compression artefacts and colour-grading edits that
 * confuse aHash (brightness-based) and dHash (edge-gradient-based).
 */
function buildPHash(g: Float32Array, width: number, height: number): string {
  const K = 8;
  const cos = getPHashCos();

  // ── Row-wise DCT: compute only first K frequency outputs per row ──────────
  const rowDCT = new Float64Array(height * K);
  for (let y = 0; y < height; y++) {
    for (let u = 0; u < K; u++) {
      let sum = 0;
      for (let x = 0; x < width; x++) {
        sum += g[y * width + x] * cos[x * K + u];
      }
      rowDCT[y * K + u] = sum;
    }
  }

  // ── Column-wise DCT: first K frequency outputs for each u-column ──────────
  const dct = new Float64Array(K * K);
  for (let u = 0; u < K; u++) {
    for (let v = 0; v < K; v++) {
      let sum = 0;
      for (let y = 0; y < height; y++) {
        sum += rowDCT[y * K + u] * cos[y * K + v];
      }
      dct[v * K + u] = sum;
    }
  }

  // ── Median of the 63 AC coefficients (all except DC at index 0) ──────────
  const ac = new Float64Array(K * K - 1);
  for (let i = 1; i < K * K; i++) ac[i - 1] = dct[i];
  ac.sort();
  const median = ac[31]; // floor(63/2) = 31

  // ── 64-bit hash string ────────────────────────────────────────────────────
  const bits = new Array<string>(K * K);
  for (let i = 0; i < K * K; i++) {
    bits[i] = dct[i] > median ? '1' : '0';
  }
  return bits.join('');
}

/** Horizontally flip a gray grid */
function flipGraysH(g: Float32Array, width: number, height: number): Float32Array {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[y * width + x] = g[y * width + (width - 1 - x)];
    }
  }
  return out;
}

export function computeHashAndFeatures(
  imageData: ImageData,
  includeSignature = false
): { hash: string; dhash: string; fhash: string; fdhash: string; phash: string; signature?: FrameSignature } {
  const { width, height, data } = imageData;
  let totalGray = 0;
  const grays = new Float32Array(width * height);
  
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];
    
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    grays[i] = gray;
    totalGray += gray;
  }
  
  const avgGray = totalGray / (width * height);
  
  let sumSqDiff = 0;
  for (let i = 0; i < width * height; i++) {
    const diff = grays[i] - avgGray;
    sumSqDiff += diff * diff;
  }
  const variance = sumSqDiff / (width * height);
  
  let hash: string;
  let dhash: string;
  let fhash: string;
  let fdhash: string;
  let phash: string;

  if (variance < 1.0) {
    // Flat frame — deterministic all-zero hashes
    hash = '0'.repeat(width * height);
    dhash = '0'.repeat((width - 1) * height + width * (height - 1));
    fhash = hash;
    fdhash = dhash;
    phash = '0'.repeat(64);
  } else {
    let currentGrays = grays;
    for (let pass = 0; pass < 3; pass++) {
      const smoothedGrays = new Float32Array(width * height);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let sum = 0;
          let count = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const ny = y + dy;
              const nx = x + dx;
              if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                sum += currentGrays[ny * width + nx];
                count++;
              }
            }
          }
          smoothedGrays[y * width + x] = sum / count;
        }
      }
      currentGrays = smoothedGrays;
    }
    
    const flipped = flipGraysH(currentGrays, width, height);
    hash   = buildAHash(currentGrays, width * height, avgGray);
    dhash  = buildDHash(currentGrays, width, height);
    fhash  = buildAHash(flipped, width * height, avgGray);
    fdhash = buildDHash(flipped, width, height);
    phash  = buildPHash(currentGrays, width, height);
  }

  const signature = includeSignature ? computeSignature(imageData) : undefined;
  return { hash, dhash, fhash, fdhash, phash, signature };
}

export function computeFingerprint(
  ctx: any,
  width: number,
  height: number,
  frameIndex: number,
  timestamp: number
): FrameFingerprint {
  const variants: Record<string, VariantHashes> = {};
  
  const H_down = 120;
  const W_down = Math.round(width * (H_down / height));
  
  const fullDownCanvas = new OffscreenCanvas(W_down, H_down);
  const fullDownCtx = fullDownCanvas.getContext('2d', { willReadFrequently: true });
  if (!fullDownCtx) throw new Error('Failed to get 2d context for fullDownCanvas');
  fullDownCtx.imageSmoothingEnabled = true;
  fullDownCtx.imageSmoothingQuality = 'high';
  
  fullDownCtx.fillStyle = '#000000';
  fullDownCtx.fillRect(0, 0, W_down, H_down);
  fullDownCtx.drawImage(ctx.canvas, 0, 0, width, height, 0, 0, W_down, H_down);
  
  // ── 1. Auto black-bar / letterbox / border trim (query-side) ──────────────
  // Detected on the downscaled frame; when bars are found the content region
  // becomes the effective frame for all crop variants, so query hashes align
  // with the original (bar-free) movie hashes without regenerating movie
  // fingerprints.
  const downData = fullDownCtx.getImageData(0, 0, W_down, H_down);
  const bounds = detectContentBounds(downData);

  let workCanvas: OffscreenCanvas = fullDownCanvas;
  let workCtx = fullDownCtx;
  let workW = W_down;
  let workH = H_down;
  if (bounds) {
    workCanvas = new OffscreenCanvas(bounds.sw, bounds.sh);
    const c = workCanvas.getContext('2d', { willReadFrequently: true });
    if (!c) throw new Error('Failed to get 2d context for content canvas');
    c.imageSmoothingEnabled = true;
    c.imageSmoothingQuality = 'high';
    c.drawImage(fullDownCanvas, bounds.sx, bounds.sy, bounds.sw, bounds.sh, 0, 0, bounds.sw, bounds.sh);
    workCtx = c;
    workW = bounds.sw;
    workH = bounds.sh;
  }

  // ── 2. Text (full-frame) + watermark masking ──────────────────────────────
  const workData = workCtx.getImageData(0, 0, workW, workH);
  const subChanged = processSubtitles(workData, true);
  const wmChanged  = maskWatermarkRegions(workData);
  if (subChanged || wmChanged) {
    workCtx.putImageData(workData, 0, 0);
  }
  const changed = bounds !== null || subChanged || wmChanged;

  // Crop rects are computed against the effective (trimmed) frame when any
  // preprocessing ran; otherwise against the original full-res frame.
  const rects = changed ? getCropRects(workW, workH) : getCropRects(width, height);
  
  const finalCanvas = new OffscreenCanvas(16, 16);
  const finalCtx = finalCanvas.getContext('2d', { willReadFrequently: true });
  if (!finalCtx) throw new Error('Failed to get 2d context for final canvas');
  finalCtx.imageSmoothingEnabled = true;
  finalCtx.imageSmoothingQuality = 'high';

  let fullVariantSignature: FrameSignature | undefined;
  
  for (const rect of rects) {
    finalCtx.fillStyle = '#000000';
    finalCtx.fillRect(0, 0, 16, 16);
    if (!changed) {
      finalCtx.drawImage(ctx.canvas, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, 16, 16);
    } else {
      // Rects are already in workCanvas coordinates
      finalCtx.drawImage(workCanvas, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, 16, 16);
    }
    
    const finalImgData = finalCtx.getImageData(0, 0, 16, 16);
    // Only compute signature for the 'full' variant
    const isFullVariant = rect.name === 'full';
    const features = computeHashAndFeatures(finalImgData, isFullVariant);
    variants[rect.name] = {
      hash: features.hash,
      dhash: features.dhash,
      fhash: features.fhash,
      fdhash: features.fdhash,
      phash: features.phash,
    };
    if (isFullVariant && features.signature) {
      fullVariantSignature = features.signature;
    }
  }
  
  return {
    frameIndex,
    timestamp,
    variants,
    signature: fullVariantSignature
  };
}
