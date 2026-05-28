"""
data_pipeline.py
================
Art Style RAG — Milestone 1 Data Pipeline
Loads WikiArt from HuggingFace, preprocesses images and metadata,
and saves a ready-to-use 1k-sample subset for the CLIP model.

Usage:
    python data_pipeline.py --subset 1000 --output data/wikiart_1k.json

Requirements:
    pip install datasets Pillow open-clip-torch
"""

import os
import json
import argparse
import logging
from pathlib import Path
from io import BytesIO

from datasets import load_dataset
from PIL import Image

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── WikiArt label maps (from dataset card) ──────────────────────────────────
ARTIST_MAP = {
    0: "Albrecht Dürer", 1: "Boris Kustodiev", 2: "Camille Pissarro",
    3: "Childe Hassam", 4: "Claude Monet", 5: "Edgar Degas",
    6: "Eugene Boudin", 7: "Gustave Courbet", 8: "Henri Matisse",
    9: "Henri de Toulouse-Lautrec", 10: "Ivan Aivazovsky", 11: "Ivan Shishkin",
    12: "John Singer Sargent", 13: "Marc Chagall", 14: "Martiros Saryan",
    15: "Nicholas Roerich", 16: "Pablo Picasso", 17: "Paul Cézanne",
    18: "Paul Gauguin", 19: "Paul Klee", 20: "Pierre-Auguste Renoir",
    21: "Pyotr Konchalovsky", 22: "Raphael Kirchner", 23: "Rembrandt",
    24: "Salvador Dali", 25: "Valentin Serov", 26: "Vincent van Gogh",
    27: "Vrubel", 28: "Wassily Kandinsky", 29: "William Turner",
}

STYLE_MAP = {
    0: "Abstract Expressionism", 1: "Action painting", 2: "Analytical Cubism",
    3: "Art Nouveau (Modern)", 4: "Baroque", 5: "Color Field Painting",
    6: "Contemporary Realism", 7: "Cubism", 8: "Early Renaissance",
    9: "Expressionism", 10: "Fauvism", 11: "High Renaissance",
    12: "Impressionism", 13: "Mannerism (Late Renaissance)", 14: "Minimalism",
    15: "Naive Art (Primitivism)", 16: "New Realism", 17: "Northern Renaissance",
    18: "Pointillism", 19: "Pop Art", 20: "Post-Impressionism",
    21: "Realism", 22: "Rococo", 23: "Romanticism", 24: "Symbolism",
    25: "Synthetic Cubism", 26: "Ukiyo-e",
}

GENRE_MAP = {
    0: "abstract painting", 1: "cityscape", 2: "genre painting",
    3: "illustration", 4: "landscape", 5: "nude painting (nu)",
    6: "portrait", 7: "religious painting", 8: "sketch and study",
    9: "still life", 10: "Unknown Genre",
}


# ─── Preprocessing helpers ────────────────────────────────────────────────────

def decode_label(value, label_map: dict, fallback: str = "Unknown") -> str:
    """Convert a numeric label (int or ClassLabel) to a human-readable string."""
    if value is None:
        return fallback
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return str(value)
    return label_map.get(idx, fallback)


def preprocess_image(image_obj, target_size: int = 224) -> dict:
    """
    Resize and centre-crop a PIL image to target_size × target_size.
    Returns a dict with width, height, mode, and base64-free metadata.
    """
    if image_obj is None:
        return {"valid": False, "width": None, "height": None, "mode": None}

    img: Image.Image = image_obj.convert("RGB")
    w, h = img.size

    # Centre crop to square, then resize
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((target_size, target_size), Image.LANCZOS)

    return {
        "valid": True,
        "original_width": w,
        "original_height": h,
        "processed_size": target_size,
        "mode": "RGB",
    }


def build_text_description(artist: str, style: str, genre: str) -> str:
    """
    Construct a natural-language CLIP query string from metadata fields.
    Example: "A Impressionism landscape painting by Claude Monet"
    """
    parts = []
    if style and style != "Unknown":
        parts.append(style)
    if genre and genre not in ("Unknown Genre", "Unknown"):
        parts.append(genre)
    desc = " ".join(parts) if parts else "painting"
    if artist and artist != "Unknown":
        desc += f" by {artist}"
    return f"A {desc}"


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(subset_size: int = 1000, output_path: str = "data/wikiart_1k.json",
                 image_size: int = 224, seed: int = 42) -> None:
    """
    Full data pipeline:
      1. Load WikiArt from HuggingFace (streaming to avoid full download)
      2. Decode numeric labels → human-readable strings
      3. Preprocess images (resize + centre-crop)
      4. Build text descriptions for CLIP
      5. Save metadata JSON for downstream indexing
    """
    log.info("Loading WikiArt dataset from HuggingFace (streaming)…")
    ds = load_dataset(
        "huggan/wikiart",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0

    log.info(f"Processing {subset_size} samples (image_size={image_size}px)…")
    for i, sample in enumerate(ds):
        if len(records) >= subset_size:
            break

        try:
            # ── Decode labels ──────────────────────────────────────────────
            artist = decode_label(sample.get("artist"), ARTIST_MAP)
            style  = decode_label(sample.get("style"),  STYLE_MAP)
            genre  = decode_label(sample.get("genre"),  GENRE_MAP)

            # ── Preprocess image ───────────────────────────────────────────
            img_meta = preprocess_image(sample.get("image"), target_size=image_size)
            if not img_meta["valid"]:
                skipped += 1
                continue

            # ── Build text description ─────────────────────────────────────
            text_desc = build_text_description(artist, style, genre)

            record = {
                "id": len(records),
                "artist": artist,
                "style": style,
                "genre": genre,
                "text_description": text_desc,
                "image_meta": img_meta,
            }
            records.append(record)

            if len(records) % 100 == 0:
                log.info(f"  Processed {len(records)}/{subset_size} samples…")

        except Exception as e:
            log.warning(f"  Skipped sample {i}: {e}")
            skipped += 1

    log.info(f"Done. {len(records)} records saved, {skipped} skipped.")

    # ── Save metadata JSON ──────────────────────────────────────────────────
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    log.info(f"Saved metadata → {output_file}")

    # ── Print summary stats ─────────────────────────────────────────────────
    styles  = {}
    genres  = {}
    artists = {}
    for r in records:
        styles[r["style"]]   = styles.get(r["style"], 0) + 1
        genres[r["genre"]]   = genres.get(r["genre"], 0) + 1
        artists[r["artist"]] = artists.get(r["artist"], 0) + 1

    log.info("\n── Dataset Summary ──────────────────────────────")
    log.info(f"  Total records : {len(records)}")
    log.info(f"  Unique styles : {len(styles)}")
    log.info(f"  Unique genres : {len(genres)}")
    log.info(f"  Unique artists: {len(artists)}")
    log.info(f"  Top styles    : {sorted(styles.items(), key=lambda x: -x[1])[:5]}")
    log.info(f"  Top genres    : {sorted(genres.items(), key=lambda x: -x[1])[:5]}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WikiArt data pipeline for Art Style RAG")
    parser.add_argument("--subset",  type=int,   default=1000,                help="Number of samples to extract")
    parser.add_argument("--output",  type=str,   default="data/wikiart_1k.json", help="Output JSON path")
    parser.add_argument("--imgsize", type=int,   default=224,                 help="Image crop size (px)")
    parser.add_argument("--seed",    type=int,   default=42,                  help="Random seed")
    args = parser.parse_args()

    run_pipeline(
        subset_size=args.subset,
        output_path=args.output,
        image_size=args.imgsize,
        seed=args.seed,
    )
