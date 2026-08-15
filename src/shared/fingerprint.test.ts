import { processSubtitles, computeHashAndFeatures, getCropRects } from './fingerprint';

// Mock ImageData for the test
class MockImageData {
  width: number;
  height: number;
  data: Uint8ClampedArray;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.data = new Uint8ClampedArray(width * height * 4);
    // Initialize with a non-subtitle background (e.g., dark gray)
    for (let i = 0; i < width * height; i++) {
      this.data[i * 4] = 40;     // R
      this.data[i * 4 + 1] = 40; // G
      this.data[i * 4 + 2] = 40; // B
      this.data[i * 4 + 3] = 255; // A
    }
  }
}

function verifyCropRects(width: number, height: number) {
  const rects = getCropRects(width, height);
  if (rects.length !== 13) {
    throw new Error(`Expected exactly 13 crop rects, got ${rects.length} for ${width}x${height}`);
  }

  const expectedNames = [
    'full',
    'crop_9_16_0', 'crop_9_16_1', 'crop_9_16_2', 'crop_9_16_3', 'crop_9_16_4',
    'zoom_1_25_center', 'zoom_1_25_left', 'zoom_1_25_right',
    'zoom_1_5_center', 'zoom_1_5_left', 'zoom_1_5_right',
    'zoom_2_0_center'
  ];

  for (const name of expectedNames) {
    const rect = rects.find(r => r.name === name);
    if (!rect) {
      throw new Error(`Missing crop rect variant '${name}' for ${width}x${height}`);
    }

    // Check integer properties
    if (!Number.isInteger(rect.sx) || !Number.isInteger(rect.sy) || !Number.isInteger(rect.sw) || !Number.isInteger(rect.sh)) {
      throw new Error(`Non-integer coordinates found in rect '${name}': sx=${rect.sx}, sy=${rect.sy}, sw=${rect.sw}, sh=${rect.sh}`);
    }

    // Check in-bounds properties
    if (rect.sx < 0 || rect.sy < 0 || rect.sx + rect.sw > width || rect.sy + rect.sh > height) {
      throw new Error(`Rect '${name}' out of bounds for ${width}x${height}: sx=${rect.sx}, sy=${rect.sy}, sw=${rect.sw}, sh=${rect.sh}`);
    }

    if (rect.sw <= 0 || rect.sh <= 0) {
      throw new Error(`Rect '${name}' has invalid dimensions: sw=${rect.sw}, sh=${rect.sh}`);
    }
  }
  console.log(`  [PASS] Successfully verified all 13 crops for resolution ${width}x${height}`);
}

async function runMilestone1Tests() {
  console.log("=== MILESTONE 1 VERIFICATION ===");
  
  // TEST A: Subtitle-removal early-exit correctness
  console.log("\nTesting (a) Subtitle-removal early-exit correctness on zero-masked frames...");
  
  let earlyExitByteIdentical = true;
  for (let i = 0; i < 50; i++) {
    // Generate a random 160x90 frame without subtitle pixels
    const frame = new MockImageData(160, 90);
    for (let j = 0; j < frame.data.length; j += 4) {
      // Keep R,G,B below 180 to ensure no white/yellow subtitle pixels
      frame.data[j] = Math.floor(Math.random() * 180);
      frame.data[j + 1] = Math.floor(Math.random() * 180);
      frame.data[j + 2] = Math.floor(Math.random() * 180);
      frame.data[j + 3] = 255;
    }

    const frameClone1 = new MockImageData(160, 90);
    const frameClone2 = new MockImageData(160, 90);
    frameClone1.data.set(frame.data);
    frameClone2.data.set(frame.data);

    // Run normal path (should early exit)
    const changed1 = processSubtitles(frameClone1 as any, false);
    
    // Run forced full pass (dilate + inpaint on empty mask)
    const changed2 = processSubtitles(frameClone2 as any, true);

    if (changed1 !== false || changed2 !== true) {
       console.error("  Mismatch in return values. changed1:", changed1, "changed2:", changed2);
    }

    // Byte-by-byte comparison
    let isIdentical = true;
    for (let j = 0; j < frameClone1.data.length; j++) {
      if (frameClone1.data[j] !== frameClone2.data[j]) {
        isIdentical = false;
        break;
      }
    }
    
    if (!isIdentical) {
      earlyExitByteIdentical = false;
      console.log(`  Failed at randomized frame iteration ${i}`);
      break;
    }
  }
  
  if (earlyExitByteIdentical) {
    console.log("  [PASS] Early-exit produces byte-identical output to running the full path on zero-masked frames across 50 iterations.");
  } else {
    console.log("  [FAIL] Early-exit output differed from full path output.");
  }

  // TEST B: Determinism
  console.log("\nTesting (b) Determinism of identical synthetic frames...");
  const hashFrame1 = new MockImageData(16, 16);
  const hashFrame2 = new MockImageData(16, 16);
  
  // Seed them with identical random pixel data
  for (let i = 0; i < hashFrame1.data.length; i++) {
    const val = Math.floor(Math.random() * 255);
    hashFrame1.data[i] = val;
    hashFrame2.data[i] = val;
  }

  const result1 = computeHashAndFeatures(hashFrame1 as any);
  const result2 = computeHashAndFeatures(hashFrame2 as any);

  let hashesIdentical = result1.hash === result2.hash;
  if (hashesIdentical) {
    console.log("  [PASS] Identical synthetic frames produced identical hash output.");
    console.log(`         Hash: ${result1.hash.substring(0, 32)}...`);
  } else {
    console.log("  [FAIL] Identical synthetic frames produced differing hashes!");
    console.log(`         Hash 1: ${result1.hash}`);
    console.log(`         Hash 2: ${result2.hash}`);
  }

  // TEST C: getCropRects resolution tests (landscape, portrait, square)
  console.log("\nTesting (c) Crop rects validity on multiple aspect ratios...");
  try {
    verifyCropRects(1920, 1080); // Landscape
    verifyCropRects(1080, 1920); // Portrait
    verifyCropRects(1000, 1000); // Square
    console.log("  [PASS] All crop rects assertions passed successfully!");
  } catch (error) {
    console.error("  [FAIL] Crop rects assertions failed:", error);
    process.exit(1);
  }

  if (earlyExitByteIdentical && hashesIdentical) {
    console.log("\n=== MILESTONE 1 PASSED ===");
  } else {
    console.log("\n=== MILESTONE 1 FAILED ===");
    process.exit(1);
  }
}

runMilestone1Tests().catch(console.error);
