"""
app.py  —  Art Style RAG  FastAPI Backend (Milestone 2)
Images are proxied through /image?url=... so the browser only talks to localhost.
This bypasses all CORS, Wikimedia blocking, and encoding issues.
"""

import os, re, pickle, logging, httpx
from pathlib import Path

import numpy as np
import torch
import open_clip
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
INDEX_PATH   = os.environ.get("INDEX_PATH", "data/clip_index.pkl")
BACKBONE     = os.environ.get("BACKBONE", "ViT-B-32")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Art Style RAG", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

clip_model = clip_tokenizer = index_embeddings = index_metadata = None

@app.on_event("startup")
async def load_models():
    global clip_model, clip_tokenizer, index_embeddings, index_metadata
    log.info(f"Loading CLIP {BACKBONE} on {DEVICE}...")
    model, _, _ = open_clip.create_model_and_transforms(BACKBONE, pretrained="openai")
    clip_model     = model.to(DEVICE).eval()
    clip_tokenizer = open_clip.get_tokenizer(BACKBONE)
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    index_embeddings = data["embeddings"].astype(np.float32)
    index_metadata   = data["metadata"]
    log.info(f"Ready — {len(index_metadata)} paintings indexed.")


# ── Image proxy endpoint ──────────────────────────────────────────────────────
# Browser calls /image?url=<encoded_url>
# Backend fetches from Wikimedia and streams back — no CORS issues ever.

# No proxy endpoint needed — images are served as local static files


# ── Artist → Wikimedia image URL map (all 24 artists in 1k dataset) ───────────
# Key: lowercase artist name as it appears in metadata (no accents normalised)
ARTIST_IMAGES = {
    # Impressionism / Post-Impressionism
    "claude monet":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/320px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg",
    "pierre-auguste renoir":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Pierre-Auguste_Renoir_-_Le_Moulin_de_la_Galette.jpg/320px-Pierre-Auguste_Renoir_-_Le_Moulin_de_la_Galette.jpg",
    "edgar degas":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Edgar_Germain_Hilaire_Degas_074.jpg/320px-Edgar_Germain_Hilaire_Degas_074.jpg",
    "camille pissarro":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Camille_Pissarro_-_Boulevard_Montmartre%2C_Afternoon_Sun.jpg/320px-Camille_Pissarro_-_Boulevard_Montmartre%2C_Afternoon_Sun.jpg",
    "childe hassam":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Childe_Hassam_-_Allies_Day%2C_May_1917.jpg/320px-Childe_Hassam_-_Allies_Day%2C_May_1917.jpg",
    "eugene boudin":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Eug%C3%A8ne_Boudin_-_The_Beach_at_Trouville_-_WGA02980.jpg/320px-Eug%C3%A8ne_Boudin_-_The_Beach_at_Trouville_-_WGA02980.jpg",
    "vincent van gogh":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/320px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    # Cézanne comes through as "Paul C\u00e9zanne" or "Paul Cézanne" — match both
    "paul c\u00e9zanne":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Paul_C%C3%A9zanne%2C_Les_Grandes_Baigneuses_%281894%E2%80%931906%29.jpg/320px-Paul_C%C3%A9zanne%2C_Les_Grandes_Baigneuses_%281894%E2%80%931906%29.jpg",
    "paul cezanne":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Paul_C%C3%A9zanne%2C_Les_Grandes_Baigneuses_%281894%E2%80%931906%29.jpg/320px-Paul_C%C3%A9zanne%2C_Les_Grandes_Baigneuses_%281894%E2%80%931906%29.jpg",
    "paul gauguin":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Paul_Gauguin_-_D%27ou_venons-nous.jpg/320px-Paul_Gauguin_-_D%27ou_venons-nous.jpg",
    # Modernism
    "paul klee":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Paul_Klee_-_Twittering_Machine_%281922%29.jpg/320px-Paul_Klee_-_Twittering_Machine_%281922%29.jpg",
    "henri matisse":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Matissedance.jpg/320px-Matissedance.jpg",
    "pablo picasso":
        "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/PicassoGuernica.jpg/320px-PicassoGuernica.jpg",
    "wassily kandinsky":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/320px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    "marc chagall":
        "https://upload.wikimedia.org/wikipedia/en/thumb/3/35/Marc_Chagall_I_and_the_Village.jpg/320px-Marc_Chagall_I_and_the_Village.jpg",
    "salvador dali":
        "https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/The_Persistence_of_Memory.jpg/320px-The_Persistence_of_Memory.jpg",
    # Old Masters
    "rembrandt":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/The_Night_Watch_-_HD.jpg/320px-The_Night_Watch_-_HD.jpg",
    "albrecht d\u00fcrer":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Albrecht_D%C3%BCrer_-_Self-Portrait_at_28_%28Alte_Pinakothek%29.jpg/320px-Albrecht_D%C3%BCrer_-_Self-Portrait_at_28_%28Alte_Pinakothek%29.jpg",
    "albrecht durer":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Albrecht_D%C3%BCrer_-_Self-Portrait_at_28_%28Alte_Pinakothek%29.jpg/320px-Albrecht_D%C3%BCrer_-_Self-Portrait_at_28_%28Alte_Pinakothek%29.jpg",
    # Russian / Eastern European — the artists causing blank images
    "ivan aivazovsky":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Ivan_Konstantinovich_Aivazovsky_-_The_Ninth_Wave.jpg/320px-Ivan_Konstantinovich_Aivazovsky_-_The_Ninth_Wave.jpg",
    "ivan shishkin":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Ivan_Shishkin_-_Morning_in_a_Pine_Forest.jpg/320px-Ivan_Shishkin_-_Morning_in_a_Pine_Forest.jpg",
    "boris kustodiev":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Boris_Kustodiev_-_Bolshevik.jpg/320px-Boris_Kustodiev_-_Bolshevik.jpg",
    "nicholas roerich":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Nicholas_Roerich_-_Himalayas_%281933%29.jpg/320px-Nicholas_Roerich_-_Himalayas_%281933%29.jpg",
    "martiros saryan":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Martiros_Saryan._Self-portrait_with_mask._1933.jpg/320px-Martiros_Saryan._Self-portrait_with_mask._1933.jpg",
    "pyotr konchalovsky":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Konchalovsky_Lilac.jpg/320px-Konchalovsky_Lilac.jpg",
    "valentin serov":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Serov_Valentin_Girl_with_Peaches_1887.jpg/320px-Serov_Valentin_Girl_with_Peaches_1887.jpg",
    "vrubel":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Vrubel_Demon.jpg/320px-Vrubel_Demon.jpg",
    # Others
    "john singer sargent":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/John_Singer_Sargent_-_Carnation%2C_Lily%2C_Lily%2C_Rose_-_Google_Art_Project.jpg/320px-John_Singer_Sargent_-_Carnation%2C_Lily%2C_Lily%2C_Rose_-_Google_Art_Project.jpg",
    "henri de toulouse-lautrec":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Toulouse-Lautrec_-_At_the_Moulin_Rouge.jpg/320px-Toulouse-Lautrec_-_At_the_Moulin_Rouge.jpg",
    "raphael kirchner":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Raphael_Kirchner_-_Geisha_Liane_02.jpg/320px-Raphael_Kirchner_-_Geisha_Liane_02.jpg",
    "gustave courbet":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Gustave_Courbet_-_Bonjour_Monsieur_Courbet_-_Google_Art_Project.jpg/320px-Gustave_Courbet_-_Bonjour_Monsieur_Courbet_-_Google_Art_Project.jpg",
    "william turner":
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Turner_-_Rain%2C_Steam_and_Speed_-_National_Gallery_file.jpg/320px-Turner_-_Rain%2C_Steam_and_Speed_-_National_Gallery_file.jpg",
}

STYLE_FALLBACKS = {
    "Impressionism":                "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/320px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg",
    "Post-Impressionism":           "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/320px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "Realism":                      "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Gustave_Courbet_-_Bonjour_Monsieur_Courbet_-_Google_Art_Project.jpg/320px-Gustave_Courbet_-_Bonjour_Monsieur_Courbet_-_Google_Art_Project.jpg",
    "Romanticism":                  "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Caspar_David_Friedrich_-_Wanderer_above_the_sea_of_fog.jpg/320px-Caspar_David_Friedrich_-_Wanderer_above_the_sea_of_fog.jpg",
    "Baroque":                      "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/The_Night_Watch_-_HD.jpg/320px-The_Night_Watch_-_HD.jpg",
    "Symbolism":                    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Fernand_Khnopff_-_I_Lock_My_Door_upon_Myself_-_Google_Art_Project.jpg/320px-Fernand_Khnopff_-_I_Lock_My_Door_upon_Myself_-_Google_Art_Project.jpg",
    "Expressionism":                "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/320px-The_Scream.jpg",
    "Abstract Expressionism":       "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/320px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    "Cubism":                       "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/PicassoGuernica.jpg/320px-PicassoGuernica.jpg",
    "Analytical Cubism":            "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/PicassoGuernica.jpg/320px-PicassoGuernica.jpg",
    "Synthetic Cubism":             "https://upload.wikimedia.org/wikipedia/en/thumb/7/74/PicassoGuernica.jpg/320px-PicassoGuernica.jpg",
    "Art Nouveau (Modern)":         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Mucha-Sarah_Bernhardt-La_Dame_aux_Camelias.jpg/320px-Mucha-Sarah_Bernhardt-La_Dame_aux_Camelias.jpg",
    "Ukiyo-e":                      "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/320px-Tsunami_by_hokusai_19th_century.jpg",
    "Pointillism":                  "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Georges_Seurat_-_A_Sunday_on_La_Grande_Jatte_--_1884.jpg/320px-Georges_Seurat_-_A_Sunday_on_La_Grande_Jatte_--_1884.jpg",
    "Fauvism":                      "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Matissedance.jpg/320px-Matissedance.jpg",
    "Surrealism":                   "https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/The_Persistence_of_Memory.jpg/320px-The_Persistence_of_Memory.jpg",
    "Naive Art (Primitivism)":      "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Henri_Rousseau_-_Le_R%C3%AAve.jpg/320px-Henri_Rousseau_-_Le_R%C3%AAve.jpg",
    "Rococo":                       "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Fragonard%2C_The_Swing.jpg/320px-Fragonard%2C_The_Swing.jpg",
    "Northern Renaissance":         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Albrecht_D%C3%BCrer_-_Self-Portrait_at_28_%28Alte_Pinakothek%29.jpg/320px-Albrecht_D%C3%BCrer_-_Self-Portrait_at_28_%28Alte_Pinakothek%29.jpg",
    "Early Renaissance":            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/320px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
    "High Renaissance":             "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/320px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
    "Mannerism (Late Renaissance)": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/320px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
    "Minimalism":                   "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/320px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    "Color Field Painting":         "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/320px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    "Action painting":              "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/320px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
}

DEFAULT_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/320px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"


def get_raw_image_url(artist: str, style: str) -> str:
    """Return Wikimedia URL for artist — match on normalised lowercase name."""
    key = artist.lower().strip()
    if key in ARTIST_IMAGES:
        return ARTIST_IMAGES[key]
    # Partial match: handles "Henri de Toulouse-Lautrec" variant spellings
    for name, url in ARTIST_IMAGES.items():
        if name in key or key in name:
            return url
    return STYLE_FALLBACKS.get(style, DEFAULT_IMAGE)


def proxy_url(artist: str, style: str) -> str:
    """Return a local static URL for the artist image.
    Images are pre-downloaded to static/artist_images/ by download_images.py.
    Falls back to a placeholder SVG if the file doesn't exist.
    """
def proxy_url(artist: str, style: str) -> str:

    from pathlib import Path
    import unicodedata

    def slugify(name: str):

        name = name.lower().strip()

        # remove accents
        name = unicodedata.normalize("NFD", name)

        name = "".join(
            c for c in name
            if unicodedata.category(c) != "Mn"
        )

        name = name.replace(" ", "_")

        return name

    artist_slug = slugify(artist)

    image_path = Path("static/artist_images") / f"{artist_slug}.jpg"

    if image_path.exists():

        return f"/static/artist_images/{artist_slug}.jpg"

    return "/static/artist_images/default.jpg"


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class PaintingResult(BaseModel):
    rank: int; artist: str; style: str; genre: str
    description: str; similarity: float; image_url: str

class QueryResponse(BaseModel):
    query: str; paintings: list[PaintingResult]
    explanation: str; metrics: dict


# ── Retrieval ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def retrieve(query: str, top_k: int) -> list[dict]:
    tokens = clip_tokenizer([query]).to(DEVICE)
    feat   = clip_model.encode_text(tokens)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    qvec   = feat.cpu().numpy().squeeze(0)
    scores = index_embeddings @ qvec
    top_ids = np.argsort(-scores)[:top_k]
    results = []
    for r, i in enumerate(top_ids):
        m = index_metadata[i]
        img = proxy_url(m["artist"], m["style"])
        log.info(f"  P{r+1}. {m['artist']} ({m['style']}) → {img}")
        results.append({
            "rank": r+1, "artist": m["artist"], "style": m["style"],
            "genre": m["genre"], "description": m["text_description"],
            "similarity": float(scores[i]), "image_url": img,
        })
    return results


# ── LLM ───────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert art historian and curator.
Given a user query and retrieved paintings P1-P5 from WikiArt, write a grounded 3-paragraph recommendation.
- Always cite paintings as P1, P2, P3, P4, P5
- Connect each to specific aesthetic qualities in the query
- Never invent titles or dates not in the metadata
- End with one sentence on how fine-tuning CLIP helps
Prose only, 200-300 words."""

def explain(query: str, paintings: list[dict]) -> str:
    ctx = f'Query: "{query}"\n\nRetrieved:\n' + "\n".join(
        f'  P{p["rank"]}. {p["artist"]} · {p["style"]} · {p["genre"]} (score {p["similarity"]:.4f})'
        for p in paintings)
    if GROQ_API_KEY:
        try:
            from groq import Groq
            r = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model="llama-3.3-70b-versatile", max_tokens=1024,
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ctx}])
            return r.choices[0].message.content
        except Exception as e:
            log.error(f"Groq error: {e}")
    styles  = list(dict.fromkeys(p["style"] for p in paintings))
    artists = [p["artist"] for p in paintings[:3]]
    return (
        f"The retrieved paintings reflect your query through {' and '.join(styles[:2])} traditions. "
        f"P1 ({paintings[0]['artist']}) leads with similarity {paintings[0]['similarity']:.3f}. "
        f"Works by {', '.join(artists)} collectively capture the mood of your search.\n\n"
        f"Scores ({paintings[0]['similarity']:.3f}–{paintings[-1]['similarity']:.3f}) reflect zero-shot CLIP confidence. "
        f"Fine-tuning on WikiArt labels substantially improves retrieval for minority styles.\n\n"
        f"Set GROQ_API_KEY for a full art-historical explanation."
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status":"ok","paintings_indexed":len(index_metadata) if index_metadata else 0,
            "backbone":BACKBONE,"llm_available":bool(GROQ_API_KEY)}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not clip_model: raise HTTPException(503, "Model not loaded")
    if not req.query.strip(): raise HTTPException(400, "Empty query")
    paintings = retrieve(req.query, req.top_k)
    style_counts = {}
    for p in paintings: style_counts[p["style"]] = style_counts.get(p["style"],0)+1
    return QueryResponse(
        query=req.query,
        paintings=[PaintingResult(**p) for p in paintings],
        explanation=explain(req.query, paintings),
        metrics={"top_score":round(paintings[0]["similarity"],4),
                 "avg_score":round(float(np.mean([p["similarity"] for p in paintings])),4),
                 "style_dist":style_counts,"backbone":BACKBONE})

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html") if Path("static/index.html").exists() \
        else {"message":"Art Style RAG API"}
