/**
 * Resolve the ffmpeg / ffprobe executables once, at import time.
 *
 * The whole fingerprinting + candidate-crop pipeline shells out to ffmpeg and
 * ffprobe. Historically both were expected to be on PATH (provided by Nix on
 * Replit / by the base image on EC2). On hosts where they are NOT installed —
 * e.g. the plain Node sandbox this repo now also runs in — every ffmpeg spawn
 * fails with ENOENT and the job dies with a confusing "no usable data" error.
 *
 * Resolution order (first hit wins):
 *   1. FFMPEG_PATH / FFPROBE_PATH env vars — explicit operator override.
 *   2. A real `ffmpeg` / `ffprobe` on PATH (Nix, apt, Docker image, …).
 *   3. The static binaries shipped by the `ffmpeg-static` / `ffprobe-static`
 *      npm packages, which are plain self-contained ELF binaries.
 *
 * Step 2 is kept ahead of step 3 so a host-provided build (usually newer and
 * hardware-accelerated) still wins when one exists.
 */
import { spawnSync } from 'child_process';
import { createRequire } from 'module';
import * as path from 'path';

// Resolve from the project root rather than `import.meta.url` so this module
// behaves identically whether tsx loads it as ESM (server / worker threads) or
// the compiled build loads it as CommonJS.
const require_ = createRequire(path.join(process.cwd(), 'noop.js'));

function onPath(bin: string): boolean {
  try {
    return spawnSync(bin, ['-version'], { stdio: 'ignore' }).status === 0;
  } catch {
    return false;
  }
}

function fromStaticPackage(pkg: 'ffmpeg-static' | 'ffprobe-static'): string | null {
  try {
    const mod = require_(pkg);
    const p = typeof mod === 'string' ? mod : mod?.path ?? mod?.default;
    return typeof p === 'string' && p.length > 0 ? p : null;
  } catch {
    return null;
  }
}

function resolve(
  bin: 'ffmpeg' | 'ffprobe',
  envVar: string,
  pkg: 'ffmpeg-static' | 'ffprobe-static',
): string {
  const override = (process.env[envVar] || '').trim();
  if (override) return override;
  if (onPath(bin)) return bin;
  const staticPath = fromStaticPackage(pkg);
  if (staticPath) {
    console.log(`[ffmpeg-path] ${bin} not on PATH — using bundled binary ${staticPath}`);
    return staticPath;
  }
  // Nothing found: return the bare name so the eventual ENOENT names the
  // missing binary instead of silently pointing at a bogus path.
  console.warn(`[ffmpeg-path] ${bin} could not be resolved (not on PATH, no ${pkg})`);
  return bin;
}

export const FFMPEG_BIN = resolve('ffmpeg', 'FFMPEG_PATH', 'ffmpeg-static');
export const FFPROBE_BIN = resolve('ffprobe', 'FFPROBE_PATH', 'ffprobe-static');

/** Shell-quoted ffprobe path, for the one execSync call site. */
export const FFPROBE_BIN_QUOTED =
  FFPROBE_BIN === 'ffprobe' ? 'ffprobe' : JSON.stringify(FFPROBE_BIN);
