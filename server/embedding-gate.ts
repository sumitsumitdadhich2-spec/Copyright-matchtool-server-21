/**
 * CLIP embedding gate — a fast, deterministic pre-check that runs BEFORE the
 * VLM for every candidate segment. It embeds each (short-frame, movie-frame)
 * pair with a local CLIP vision model and compares cosine similarity:
 *
 *   - clearly the same footage  -> auto-ACCEPT (no VLM call needed)
 *   - clearly different footage -> auto-REJECT (no VLM call needed)
 *   - ambiguous middle band     -> defer to the VLM as tie-breaker
 *
 * Why this exists: the 7B VLM is the weakest link in the pipeline — it is
 * slow (which caused timeouts under load) and inconsistent on borderline
 * frames. Embeddings are consistent, run locally on CPU, and are very good
 * at exactly this "is it the same image content?" question. Routing only the
 * genuinely ambiguous cases to the VLM cuts VLM traffic dramatically AND
 * removes most of its false accepts/rejects.
 *
 * Fully optional and fail-safe: if the model cannot be loaded (no network to
 * download weights on first run, unsupported platform, etc.) every function
 * returns null and callers fall back to the previous VLM-only behavior.
 */

// Cosine similarity above which a pair is considered "same footage" without
// asking the VLM. CLIP image-image similarity for re-encoded/cropped/filtered
// copies of the same frame is typically 0.90+; unrelated scenes rarely
// exceed 0.85. 0.93 is deliberately conservative.
export const EMBED_ACCEPT_SIM = Number(process.env.EMBED_ACCEPT_SIM) || 0.93;

// Cosine similarity below which a pair is considered clearly different
// footage. Unrelated frames usually sit in the 0.4–0.7 band; genuinely same
// footage almost never drops under 0.6 even with heavy edits.
export const EMBED_REJECT_SIM = Number(process.env.EMBED_REJECT_SIM) || 0.55;

// When the VLM could not produce a verdict (timeout/overload), the candidate
// is kept ONLY if its embedding similarity clears this bar — this replaces
// the old "unverifiable => silently accept" policy that let bad segments
// through whenever the VLM server was overloaded.
export const EMBED_UNVERIFIABLE_KEEP_SIM =
  Number(process.env.EMBED_UNVERIFIABLE_KEEP_SIM) || 0.78;

const EMBED_MODEL = process.env.EMBED_MODEL || 'Xenova/clip-vit-base-patch32';

export type EmbeddingGateDecision = 'accept' | 'reject' | 'ambiguous';

export interface EmbeddingGateResult {
  decision: EmbeddingGateDecision;
  /** Per-pair cosine similarities, same order as the input pairs. */
  similarities: number[];
  /** Median similarity across pairs — the primary aggregate signal. */
  medianSim: number;
  /** Best (max) similarity across pairs. */
  maxSim: number;
}

// ---------------------------------------------------------------------------
// Lazy model loading — the ~150MB CLIP weights are only downloaded/loaded the
// first time the gate is actually used, and a load failure permanently
// disables the gate for this process (logged once) instead of retrying and
// stalling every segment.
// ---------------------------------------------------------------------------
let extractorPromise: Promise<any> | null = null;
let gateDisabled = false;

async function getExtractor(): Promise<any | null> {
  if (gateDisabled) return null;
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      // Keep everything local/CPU; never try WebGPU in Node.
      (env as any).allowLocalModels = true;
      const extractor = await pipeline('image-feature-extraction', EMBED_MODEL);
      console.log(`[EmbedGate] Loaded ${EMBED_MODEL} for embedding pre-checks`);
      return extractor;
    })().catch((err: any) => {
      gateDisabled = true;
      extractorPromise = null;
      console.warn(
        `[EmbedGate] Could not load embedding model (${err?.message || err}) — ` +
        `embedding gate disabled, falling back to VLM-only verification.`
      );
      return null;
    });
  }
  return extractorPromise;
}

function cosineSim(a: Float32Array | number[], b: Float32Array | number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += (a[i] as number) * (b[i] as number);
    na += (a[i] as number) * (a[i] as number);
    nb += (b[i] as number) * (b[i] as number);
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
}

/** Embed one base64 JPEG frame; returns null if the gate is unavailable. */
async function embedFrame(extractor: any, b64: string): Promise<Float32Array | null> {
  const { RawImage } = await import('@huggingface/transformers');
  const blob = new Blob([Buffer.from(b64, 'base64')], { type: 'image/jpeg' });
  const image = await RawImage.fromBlob(blob);
  const output = await extractor(image, { pooling: 'mean', normalize: true });
  return output?.data ? new Float32Array(output.data) : null;
}

/**
 * Run the embedding gate over the frame pairs of one candidate segment.
 * Returns null when the gate is unavailable (model failed to load, or an
 * unexpected embedding error) — callers must then behave exactly as before
 * the gate existed.
 *
 * Decision rules (conservative on purpose):
 *  - accept: median similarity across pairs >= EMBED_ACCEPT_SIM
 *            (most sampled moments look like the same footage)
 *  - reject: even the BEST pair is below EMBED_REJECT_SIM
 *            (no sampled moment looks remotely like the same footage)
 *  - ambiguous: everything else -> let the VLM decide
 */
export async function embeddingGateCheck(
  pairs: Array<{ shortFrameB64: string; movieFrameB64: string }>,
): Promise<EmbeddingGateResult | null> {
  if (pairs.length === 0) return null;
  const extractor = await getExtractor();
  if (!extractor) return null;

  try {
    const similarities: number[] = [];
    for (const p of pairs) {
      const [embShort, embMovie] = await Promise.all([
        embedFrame(extractor, p.shortFrameB64),
        embedFrame(extractor, p.movieFrameB64),
      ]);
      if (!embShort || !embMovie) return null;
      similarities.push(cosineSim(embShort, embMovie));
    }

    const med = median(similarities);
    const max = Math.max(...similarities);

    let decision: EmbeddingGateDecision = 'ambiguous';
    if (med >= EMBED_ACCEPT_SIM) decision = 'accept';
    else if (max < EMBED_REJECT_SIM) decision = 'reject';

    return { decision, similarities, medianSim: med, maxSim: max };
  } catch (err: any) {
    console.warn(`[EmbedGate] Embedding check failed (${err?.message || err}) — deferring to VLM.`);
    return null;
  }
}
