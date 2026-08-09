/**
 * Process-wide cache for expensive, immutable per-movie precomputations
 * (fingerprint PreSets, NDJSON line indexes, timestamp metadata).
 *
 * PERFORMANCE FIX (bottleneck 1): every matchVideosFromFiles() call used to
 * re-stream the ENTIRE movie fingerprint file from disk — a 50–90 s cost per
 * call for a 12 K-frame movie — and the VLM auto-extend fallback calls the
 * matcher repeatedly for the same movie. The movie file never changes within
 * a job, so these assets are computed exactly once per (path, mtime, size)
 * and reused by every subsequent round.
 *
 * Design notes:
 *  - Keyed by resolved file path; validated by mtimeMs + size, so replacing
 *    the file (new upload at the same path) automatically invalidates.
 *  - Values are stored as PROMISES: concurrent callers (VLM_CONCURRENCY
 *    segments auto-extending at once) share ONE in-flight load instead of
 *    racing duplicate precomputes. A rejected load is evicted so the next
 *    caller retries.
 *  - Small LRU (default 2 movies) so long-running servers processing many
 *    movies don't accumulate PreSets forever.
 *  - Cached assets MUST be treated as read-only by consumers — both matching
 *    engines only read the PreSet flats/fps during a scan, never mutate them.
 */
import * as fs from 'fs';
import * as path from 'path';

const MAX_CACHED_MOVIES = Math.max(1, Number(process.env.MOVIE_ASSET_CACHE_MAX) || 2);

interface MovieCacheEntry {
  /** `${mtimeMs}:${size}` of the file when the entry was created. */
  statKey: string;
  /** kind → in-flight or settled loader promise. */
  assets: Map<string, Promise<unknown>>;
  lastUsed: number;
}

const cache = new Map<string, MovieCacheEntry>();

function statKeyFor(filePath: string): string | null {
  try {
    const st = fs.statSync(filePath);
    return `${st.mtimeMs}:${st.size}`;
  } catch {
    return null;
  }
}

/** True when a settled/in-flight cached asset exists for this exact file version. */
export function hasMovieAsset(filePath: string, kind: string): boolean {
  const key = path.resolve(filePath);
  const entry = cache.get(key);
  if (!entry) return false;
  const statKey = statKeyFor(key);
  if (statKey === null || statKey !== entry.statKey) return false;
  return entry.assets.has(kind);
}

/**
 * Return the cached asset for (filePath, kind), or load it via `loader`
 * exactly once per file version. `cacheHit` is true when no load ran for
 * this call (a previous call's result — possibly still in flight — was reused).
 */
export async function getOrLoadMovieAsset<T>(
  filePath: string,
  kind: string,
  loader: () => Promise<T>,
): Promise<{ value: T; cacheHit: boolean }> {
  const key = path.resolve(filePath);
  const statKey = statKeyFor(key);
  // Unstatable file (deleted mid-job?) — degrade gracefully, no caching.
  if (statKey === null) return { value: await loader(), cacheHit: false };

  let entry = cache.get(key);
  if (entry && entry.statKey !== statKey) {
    // File replaced since caching — drop every stale asset for it.
    cache.delete(key);
    entry = undefined;
  }
  if (!entry) {
    entry = { statKey, assets: new Map(), lastUsed: Date.now() };
    cache.set(key, entry);
    // LRU eviction: never evict the entry we just created.
    while (cache.size > MAX_CACHED_MOVIES) {
      let oldestKey: string | null = null;
      let oldestUsed = Infinity;
      for (const [k, e] of cache) {
        if (k === key) continue;
        if (e.lastUsed < oldestUsed) { oldestUsed = e.lastUsed; oldestKey = k; }
      }
      if (oldestKey === null) break;
      cache.delete(oldestKey);
      console.log(`[MovieAssetCache] Evicted LRU movie assets for ${oldestKey}`);
    }
  }
  entry.lastUsed = Date.now();

  const existing = entry.assets.get(kind);
  if (existing) {
    try {
      return { value: (await existing) as T, cacheHit: true };
    } catch {
      // Previous load failed — evict and retry below.
      entry.assets.delete(kind);
    }
  }

  const inFlight = loader();
  entry.assets.set(kind, inFlight as Promise<unknown>);
  try {
    return { value: await inFlight, cacheHit: false };
  } catch (err) {
    entry.assets.delete(kind);
    throw err;
  }
}

/** Test/maintenance helper: drop everything (or one movie's assets). */
export function clearMovieAssetCache(filePath?: string): void {
  if (filePath) cache.delete(path.resolve(filePath));
  else cache.clear();
}
