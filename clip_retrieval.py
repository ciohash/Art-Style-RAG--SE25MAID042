"""
clip_retrieval.py
=================
Art Style RAG — Milestone 1 Preliminary Model
CLIP-ViT-B/32 text-to-image retrieval on a 1k WikiArt subset.

Usage:
    # Step 1: Build CLIP index from dataset
    python clip_retrieval.py --mode index --data data/wikiart_1k.json

    # Step 2: Query with text
    python clip_retrieval.py --mode query --query "Impressionist landscape at sunset"

    # Step 3: Ablation — compare text-only vs CLIP retrieval
    python clip_retrieval.py --mode ablation

Requirements:
    pip install open-clip-torch datasets Pillow faiss-cpu
"""

import os
import json
import argparse
import logging
import pickle
import time
from pathlib import Path

import torch
import open_clip
import numpy as np
from PIL import Image
from datasets import load_dataset

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME   = "ViT-B-32"          # CLIP backbone
PRETRAINED   = "openai"            # OpenAI's original CLIP weights
INDEX_PATH   = "data/clip_index.pkl"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_clip_model():
    """Load CLIP ViT-B/32 model, transforms, and tokenizer."""
    log.info(f"Loading {MODEL_NAME} ({PRETRAINED}) on {DEVICE}…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE).eval()
    log.info("Model loaded.")
    return model, preprocess, tokenizer


# ─── Embedding helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def embed_image(model, preprocess, pil_image: Image.Image) -> np.ndarray:
    """Return L2-normalised CLIP image embedding as numpy (512,)."""
    tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    feat   = model.encode_image(tensor)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze(0)


@torch.no_grad()
def embed_text(model, tokenizer, text: str) -> np.ndarray:
    """Return L2-normalised CLIP text embedding as numpy (512,)."""
    tokens = tokenizer([text]).to(DEVICE)
    feat   = model.encode_text(tokens)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze(0)


# ─── FAISS-free cosine similarity search ─────────────────────────────────────

def cosine_search(query_vec: np.ndarray, index_vecs: np.ndarray, top_k: int = 5):
    """Brute-force cosine similarity (vectors are already L2-normalised → dot product)."""
    scores = index_vecs @ query_vec          # shape (N,)
    top_ids = np.argsort(-scores)[:top_k]   # descending
    return [(int(idx), float(scores[idx])) for idx in top_ids]


# ─── Index builder ────────────────────────────────────────────────────────────

def build_index(data_path: str = "data/wikiart_1k.json",
                index_path: str = INDEX_PATH,
                subset: int = 1000) -> None:
    """
    Stream WikiArt, embed each image with CLIP, and save:
      - embeddings matrix  (N × 512) float32
      - metadata list      list of dicts {id, artist, style, genre, text_description}
    """
    model, preprocess, _ = load_clip_model()

    log.info("Streaming WikiArt from HuggingFace…")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True, trust_remote_code=True)

    # Load metadata from the pipeline output (labels decoded there)
    with open(data_path, "r") as f:
        meta_records = json.load(f)

    embeddings = []
    metadata   = []
    failed     = 0

    log.info(f"Embedding {subset} images with CLIP {MODEL_NAME}…")
    t0 = time.time()

    for i, sample in enumerate(ds):
        if len(embeddings) >= subset:
            break
        if i >= len(meta_records):
            break

        try:
            pil_img = sample["image"].convert("RGB")
            vec     = embed_image(model, preprocess, pil_img)
            embeddings.append(vec)
            metadata.append(meta_records[i])

            if len(embeddings) % 100 == 0:
                elapsed = time.time() - t0
                log.info(f"  Embedded {len(embeddings)}/{subset}  ({elapsed:.1f}s elapsed)")

        except Exception as e:
            log.warning(f"  Skipped sample {i}: {e}")
            failed += 1

    emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)  # (N, 512)
    index_data = {"embeddings": emb_matrix, "metadata": metadata}

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)

    log.info(f"\nIndex saved → {index_path}")
    log.info(f"  Embedded : {len(embeddings)}")
    log.info(f"  Failed   : {failed}")
    log.info(f"  Shape    : {emb_matrix.shape}")
    log.info(f"  Time     : {time.time() - t0:.1f}s")


# ─── Query runner ─────────────────────────────────────────────────────────────

def query(text: str, top_k: int = 5, index_path: str = INDEX_PATH) -> None:
    """
    Embed a text query with CLIP and retrieve top-k most similar paintings.
    Prints a ranked result table with cosine similarity scores.
    """
    if not Path(index_path).exists():
        log.error(f"Index not found at {index_path}. Run with --mode index first.")
        return

    model, _, tokenizer = load_clip_model()

    with open(index_path, "rb") as f:
        index_data = pickle.load(f)

    emb_matrix = index_data["embeddings"]   # (N, 512)
    metadata   = index_data["metadata"]

    log.info(f"\nQuery: \"{text}\"")
    query_vec = embed_text(model, tokenizer, text)
    results   = cosine_search(query_vec, emb_matrix, top_k=top_k)

    print(f"\n{'Rank':<6} {'Score':<8} {'Artist':<28} {'Style':<28} {'Genre':<22}")
    print("─" * 100)
    for rank, (idx, score) in enumerate(results, 1):
        m = metadata[idx]
        print(f"{rank:<6} {score:<8.4f} {m['artist']:<28} {m['style']:<28} {m['genre']:<22}")
        print(f"       └─ {m['text_description']}")

    print()


# ─── Ablation: text-only (BM25-style keyword) vs CLIP ────────────────────────

def run_ablation(index_path: str = INDEX_PATH) -> None:
    """
    Ablation study comparing:
      A) Keyword matching on text_description (baseline)
      B) CLIP semantic embedding retrieval (proposed)

    Metric: Precision@5 — fraction of top-5 results whose style matches the query style.
    """
    if not Path(index_path).exists():
        log.error(f"Index not found at {index_path}. Run with --mode index first.")
        return

    model, _, tokenizer = load_clip_model()

    with open(index_path, "rb") as f:
        index_data = pickle.load(f)

    emb_matrix = index_data["embeddings"]
    metadata   = index_data["metadata"]

    # Test queries: (query_text, target_style_keyword)
    test_queries = [
        ("dreamy water lilies soft brush strokes nature", "Impressionism"),
        ("dark dramatic portrait candlelight shadows",    "Baroque"),
        ("bold geometric shapes primary colors abstract", "Cubism"),
        ("emotional raw brush strokes vivid colour",     "Expressionism"),
        ("serene Japanese woodblock ocean wave",          "Ukiyo-e"),
    ]

    print("\n" + "=" * 70)
    print("  ABLATION: Keyword Matching vs CLIP Semantic Retrieval (P@5)")
    print("=" * 70)

    keyword_scores = []
    clip_scores    = []

    for query_text, target_style in test_queries:
        k = 5

        # ── Keyword baseline ──────────────────────────────────────────────
        keyword_hits = 0
        for m in metadata:
            if target_style.lower() in m["text_description"].lower():
                keyword_hits += 1
            if keyword_hits >= k:
                break
        kw_precision = keyword_hits / k

        # ── CLIP retrieval ────────────────────────────────────────────────
        query_vec = embed_text(model, tokenizer, query_text)
        results   = cosine_search(query_vec, emb_matrix, top_k=k)
        clip_hits = sum(
            1 for idx, _ in results
            if target_style.lower() in metadata[idx]["style"].lower()
        )
        clip_precision = clip_hits / k

        keyword_scores.append(kw_precision)
        clip_scores.append(clip_precision)

        print(f"\n  Query : \"{query_text}\"")
        print(f"  Target style       : {target_style}")
        print(f"  Keyword P@{k}      : {kw_precision:.2f}")
        print(f"  CLIP P@{k}         : {clip_precision:.2f}   {'✓ better' if clip_precision > kw_precision else ('= tie' if clip_precision == kw_precision else '✗ worse')}")

    avg_kw   = np.mean(keyword_scores)
    avg_clip = np.mean(clip_scores)

    print("\n" + "─" * 70)
    print(f"  Average Keyword P@5 : {avg_kw:.3f}")
    print(f"  Average CLIP P@5    : {avg_clip:.3f}")
    print(f"  Delta               : {avg_clip - avg_kw:+.3f}")
    print("=" * 70 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP text-to-image retrieval for WikiArt")
    parser.add_argument("--mode",   choices=["index", "query", "ablation"], default="query")
    parser.add_argument("--data",   default="data/wikiart_1k.json",  help="Metadata JSON from data_pipeline.py")
    parser.add_argument("--index",  default=INDEX_PATH,              help="Where to save/load CLIP index")
    parser.add_argument("--query",  default="Impressionist water scene at dusk")
    parser.add_argument("--topk",   type=int, default=5)
    parser.add_argument("--subset", type=int, default=1000)
    args = parser.parse_args()

    if args.mode == "index":
        build_index(data_path=args.data, index_path=args.index, subset=args.subset)
    elif args.mode == "query":
        query(text=args.query, top_k=args.topk, index_path=args.index)
    elif args.mode == "ablation":
        run_ablation(index_path=args.index)
