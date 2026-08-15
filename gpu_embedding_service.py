# =============================================================================
# GPU EMBEDDING SERVICE — SSCD + DINOv2 + OpenCLIP + FAISS
# =============================================================================
# Standalone FastAPI service meant to run on a Google Colab T4 (16GB) GPU.
# The AWS app talks to it through an ngrok tunnel via GPU_EMBED_SERVICE_URL —
# the exact same pattern as the VLM service.
#
# Serves the CANDIDATE ranking system only (candidate-embedding-rank.ts).
# The app is fully fail-safe: if this service is down/unreachable it silently
# falls back to local CLIP on CPU, so this service can be started/stopped at
# any time without breaking anything.
#
# Models (all three fit comfortably in T4 16GB together, fp16):
#   1. SSCD   (facebookresearch/sscd-copy-detection, sscd_disc_mixup
#              torchscript) — PRIMARY. Purpose-built for copy detection:
#              crops, zooms, re-encodes, filters.
#   2. DINOv2 ViT-S/14 (torch.hub facebookresearch/dinov2) — SECONDARY,
#              used by the app as a tie-breaker when SSCD's top-2 margin
#              is tight.
#   3. OpenCLIP ViT-B/32 — TERTIARY / compatibility with the app's local
#              CPU CLIP path.
#
# Endpoints:
#   GET  /health        -> { status, models_loaded: {sscd,dino,clip}, gpu }
#   POST /embed         -> { images: [b64,...], model: "sscd"|"dino"|"clip" }
#                          -> { embeddings: [[...],...] }   (L2-normalized)
#   POST /embed_batch   -> { batches: [{short: b64, variants: [b64,...]}],
#                            model }
#                          -> { results: [{sims, max_sim, best_index}] }
#                          One GPU roundtrip scores every (short, 13-variant)
#                          group; cosine happens here so no embeddings travel
#                          back over ngrok.
#   POST /index_build   -> { embeddings: [[...],...] } -> { index_id, size }
#   POST /index_search  -> { index_id, queries: [[...],...], k }
#                          -> { ids: [[...]], scores: [[...]] }
#                          FAISS inner-product search (embeddings are
#                          L2-normalized, so IP == cosine). Used by the app
#                          when a candidate pool is large; also ready for a
#                          future full-movie pre-index pipeline.
#
# Safety: requests are processed in micro-batches of MAX_BATCH (64) images so
# a T4 never OOMs, with fp16 autocast inference.
#
# -----------------------------------------------------------------------------
# COLAB SETUP (run these cells on a T4 runtime):
# -----------------------------------------------------------------------------
# # Cell 1 — dependencies
# !pip install -q fastapi uvicorn pillow open_clip_torch faiss-cpu pyngrok nest_asyncio
# # torch + torchvision are preinstalled on Colab. faiss-gpu wheels are
# # flaky on Colab; faiss-cpu is plenty fast for these index sizes.
#
# # Cell 2 — get this file onto the runtime (upload it, or paste its contents
# # into a %%writefile cell):
# # %%writefile gpu_embedding_service.py
# # <paste this file>
#
# # Cell 3 — start the server + ngrok tunnel (same pattern as the VLM service)
# import nest_asyncio, threading, uvicorn
# from pyngrok import ngrok
# nest_asyncio.apply()
# ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")   # https://dashboard.ngrok.com
# public_url = ngrok.connect(8000, "http").public_url
# print("GPU_EMBED_SERVICE_URL =", public_url)   # <- set this env var on AWS
# def run():
#     import gpu_embedding_service  # loads models on import of app startup
#     uvicorn.run(gpu_embedding_service.app, host="0.0.0.0", port=8000)
# threading.Thread(target=run, daemon=True).start()
#
# # Then on the AWS app set:  GPU_EMBED_SERVICE_URL=<printed ngrok url>
# # Verify with:              curl <url>/health
# =============================================================================

import base64
import io
import threading
import uuid
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

try:
    import faiss  # faiss-cpu or faiss-gpu
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
MAX_BATCH = 64  # T4-safe micro-batch cap; larger requests are chunked

SSCD_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
)

app = FastAPI(title="gpu-embedding-service")

# -----------------------------------------------------------------------------
# Model loading (each fail-soft: a model that fails to load is reported as
# missing in /health and its endpoints return an error, but the service and
# the other models keep working)
# -----------------------------------------------------------------------------
_models: Dict[str, Optional[torch.nn.Module]] = {"sscd": None, "dino": None, "clip": None}
_clip_preprocess = None
_load_lock = threading.Lock()

from torchvision import transforms  # noqa: E402  (after torch import)

# SSCD official preprocessing: resize to 288, ImageNet normalization.
_sscd_tf = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# DINOv2 ViT-S/14: side must be a multiple of 14 — 224 works well.
_dino_tf = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_models() -> None:
    """Load all three models onto the GPU. Called once at startup."""
    global _clip_preprocess

    # 1) SSCD (torchscript — downloads ~120MB on first run)
    try:
        sscd = torch.hub.load_state_dict_from_url  # noqa: F841 (cache helper warm)
    except Exception:
        pass
    try:
        import urllib.request, os, tempfile
        cache = os.path.join(tempfile.gettempdir(), "sscd_disc_mixup.torchscript.pt")
        if not os.path.exists(cache):
            print("[gpu-embed] downloading SSCD weights ...")
            urllib.request.urlretrieve(SSCD_WEIGHTS_URL, cache)
        m = torch.jit.load(cache, map_location=DEVICE)
        m.eval()
        # BUG FIX (SSCD saturation): the SSCD TorchScript model must stay in
        # fp32. Running it in fp16 (.half()) degrades/saturates its
        # copy-detection features so badly that after L2 normalization EVERY
        # image pair scores cosine ~1.000 — which made all candidate rankings
        # degenerate (all sim=1.000, margin 0.000). fp32 on a T4 is still
        # fast enough for this workload.
        _models["sscd"] = m
        print("[gpu-embed] SSCD loaded (fp32 — fp16 saturates SSCD features)")
    except Exception as e:  # noqa: BLE001
        print(f"[gpu-embed] SSCD load FAILED: {e}")

    # 2) DINOv2 ViT-S/14
    try:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        m.eval().to(DEVICE)
        if USE_FP16:
            m = m.half()
        _models["dino"] = m
        print("[gpu-embed] DINOv2 ViT-S/14 loaded")
    except Exception as e:  # noqa: BLE001
        print(f"[gpu-embed] DINOv2 load FAILED: {e}")

    # 3) OpenCLIP ViT-B/32
    try:
        import open_clip
        m, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        m.eval().to(DEVICE)
        if USE_FP16:
            m = m.half()
        _models["clip"] = m
        _clip_preprocess = preprocess
        print("[gpu-embed] OpenCLIP ViT-B/32 loaded")
    except Exception as e:  # noqa: BLE001
        print(f"[gpu-embed] OpenCLIP load FAILED: {e}")


with _load_lock:
    _load_models()

# -----------------------------------------------------------------------------
# Embedding core
# -----------------------------------------------------------------------------

def _decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


@torch.inference_mode()
def _embed_images(images_b64: List[str], model_name: str) -> np.ndarray:
    """Embed a list of base64 JPEGs; returns (N, D) float32, L2-normalized.

    Processes in MAX_BATCH micro-batches with fp16 autocast so a T4 never
    OOMs regardless of request size.
    """
    model = _models.get(model_name)
    if model is None:
        raise RuntimeError(f"model '{model_name}' not loaded")

    if model_name == "sscd":
        tf = _sscd_tf
    elif model_name == "dino":
        tf = _dino_tf
    else:
        if _clip_preprocess is None:
            raise RuntimeError("clip preprocess unavailable")
        tf = _clip_preprocess

    # BUG FIX (SSCD saturation): SSCD runs in strict fp32 — no .half() inputs
    # and no autocast — because fp16 saturates its features and every cosine
    # similarity collapses to ~1.000 (degenerate ranking). DINO/CLIP keep fp16.
    use_half = USE_FP16 and model_name != "sscd"

    outputs: List[np.ndarray] = []
    for start in range(0, len(images_b64), MAX_BATCH):
        chunk = images_b64[start : start + MAX_BATCH]
        tensors = torch.stack([tf(_decode_image(b)) for b in chunk]).to(DEVICE)
        if use_half:
            tensors = tensors.half()
        with torch.autocast(device_type="cuda", enabled=use_half):
            if model_name == "clip":
                feats = model.encode_image(tensors)
            else:
                feats = model(tensors)
        outputs.append(feats.float().cpu().numpy())

    embs = np.concatenate(outputs, axis=0).astype(np.float32)
    return _l2_normalize(embs)


# -----------------------------------------------------------------------------
# Request / response models
# -----------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    images: List[str]
    model: str = "sscd"


class BatchItem(BaseModel):
    short: str
    variants: List[str]


class EmbedBatchRequest(BaseModel):
    batches: List[BatchItem]
    model: str = "sscd"


class IndexBuildRequest(BaseModel):
    embeddings: List[List[float]]


class IndexSearchRequest(BaseModel):
    index_id: str
    queries: List[List[float]]
    k: int = 10


# -----------------------------------------------------------------------------
# FAISS index registry (in-memory; indexes die with the Colab runtime, which
# is fine — the app rebuilds per run)
# -----------------------------------------------------------------------------
_indexes: Dict[str, "faiss.Index"] = {}
_index_lock = threading.Lock()
MAX_INDEXES = 16  # simple LRU-ish cap so a long session can't leak RAM
_index_order: List[str] = []


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {k: v is not None for k, v in _models.items()},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "faiss": FAISS_AVAILABLE,
        "max_batch": MAX_BATCH,
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        if not req.images:
            return {"embeddings": []}
        embs = _embed_images(req.images, req.model)
        return {"embeddings": embs.tolist()}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/embed_batch")
def embed_batch(req: EmbedBatchRequest):
    """Score every (short, variants[]) group in ONE GPU pass.

    All images across all batches are embedded together (micro-batched at
    MAX_BATCH internally), then cosine similarity — embeddings are
    L2-normalized so it's a plain dot product — is computed here, avoiding
    13x network roundtrips AND avoiding shipping embeddings back over ngrok.
    """
    try:
        if not req.batches:
            return {"results": []}

        # Flatten: shorts first (deduplicated), then all variants.
        short_map: Dict[str, int] = {}
        flat: List[str] = []
        for b in req.batches:
            if b.short not in short_map:
                short_map[b.short] = len(flat)
                flat.append(b.short)
        variant_pos: List[List[int]] = []
        for b in req.batches:
            positions = []
            for v in b.variants:
                positions.append(len(flat))
                flat.append(v)
            variant_pos.append(positions)

        embs = _embed_images(flat, req.model)

        results = []
        for i, b in enumerate(req.batches):
            s = embs[short_map[b.short]]
            sims = [float(np.dot(s, embs[p])) for p in variant_pos[i]]
            if sims:
                best_index = int(np.argmax(sims))
                results.append(
                    {"sims": sims, "max_sim": sims[best_index], "best_index": best_index}
                )
            else:
                results.append({"sims": [], "max_sim": -1.0, "best_index": 0})
        return {"results": results}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/index_build")
def index_build(req: IndexBuildRequest):
    """Build a FAISS inner-product index over L2-normalized embeddings
    (IP == cosine). Returns an index_id for /index_search.

    Also the building block for a FUTURE full-movie pre-index pipeline:
    embed every movie frame once, index here, then search shorts directly.
    (App-side pre-indexing is intentionally NOT built yet.)
    """
    if not FAISS_AVAILABLE:
        return JSONResponse(status_code=500, content={"error": "faiss not installed"})
    try:
        arr = np.asarray(req.embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return JSONResponse(status_code=400, content={"error": "embeddings must be a non-empty 2D array"})
        arr = _l2_normalize(arr)
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        index_id = uuid.uuid4().hex
        with _index_lock:
            _indexes[index_id] = index
            _index_order.append(index_id)
            while len(_index_order) > MAX_INDEXES:
                old = _index_order.pop(0)
                _indexes.pop(old, None)
        return {"index_id": index_id, "size": int(arr.shape[0])}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/index_search")
def index_search(req: IndexSearchRequest):
    if not FAISS_AVAILABLE:
        return JSONResponse(status_code=500, content={"error": "faiss not installed"})
    try:
        with _index_lock:
            index = _indexes.get(req.index_id)
        if index is None:
            return JSONResponse(status_code=404, content={"error": "unknown index_id"})
        q = np.asarray(req.queries, dtype=np.float32)
        if q.ndim != 2 or q.shape[0] == 0:
            return JSONResponse(status_code=400, content={"error": "queries must be a non-empty 2D array"})
        q = _l2_normalize(q)
        k = max(1, min(int(req.k), index.ntotal))
        scores, ids = index.search(q, k)
        return {"ids": ids.tolist(), "scores": scores.tolist()}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


# -----------------------------------------------------------------------------
# Direct launch (outside Colab): python gpu_embedding_service.py
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
