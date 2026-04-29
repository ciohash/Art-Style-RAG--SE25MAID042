# Art-Style-RAG--SE25MAID018
A user uploads or describes a painting style (e.g., "impressionist, warm, outdoors") and the system retrieves visually and thematically similar artworks with context about the artist and movement.


> **Milestone 1 · Preliminary Model**  
> Retrieve paintings from WikiArt by natural-language aesthetic descriptions using CLIP-ViT-B/32.

---

## Overview

Finding artwork by visual style is hard with keyword search alone. A query like *"melancholic late-19th century night scenes with loose, expressive brushwork"* cannot be meaningfully answered by a standard image database.

This project builds an **Art Style RAG** system:

1. **Retrieve** — embed a free-text query with CLIP and find visually/thematically matching paintings from a 1k WikiArt subset.
2. **Generate** *(final milestone)* — produce a grounded explanation for each result using an LLM, with every recommended artwork traceable to real WikiArt metadata.

---

## Project Structure

```
.
├── data_pipeline.py          # Download, preprocess & save WikiArt metadata
├── clip_retrieval.py         # Build CLIP index, query, and run ablation
├── requirements.txt          # Python dependencies
├── data/
│   ├── wikiart_1k.json       # Output of data_pipeline.py (1k metadata records)
│   └── clip_index.pkl        # Output of clip_retrieval.py --mode index
└── outputs.txt               # Sample run logs and ablation results
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
datasets>=2.14.0
open-clip-torch>=2.24.0
Pillow>=10.0.0
numpy>=1.24.0
torch>=2.0.0
```

> No GPU required — all steps run on CPU (~10 min for 1k images).

---

### 2. Build the dataset

Stream 1,000 paintings from [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart) on HuggingFace, decode labels, preprocess images, and save metadata:

```bash
python data_pipeline.py --subset 1000 --output data/wikiart_1k.json
```

| Argument | Default | Description |
|---|---|---|
| `--subset` | `1000` | Number of paintings to extract |
| `--output` | `data/wikiart_1k.json` | Output metadata JSON path |
| `--imgsize` | `224` | Image crop size in pixels |
| `--seed` | `42` | Random seed |

**Sample output:**
```
Total records : 1000
Unique styles : 16
Unique genres : 10
Unique artists: 24
Top styles    : [('Impressionism', 276), ('Realism', 210), ('Post-Impressionism', 93), ...]
Top genres    : [('landscape', 225), ('portrait', 145), ('genre painting', 131), ...]
```

---

### 3. Build the CLIP index

Embed all 1,000 images with CLIP ViT-B/32 into 512-d L2-normalised vectors and save to disk:

```bash
python clip_retrieval.py --mode index --data data/wikiart_1k.json
```

This takes ~10 minutes on CPU. Output: `data/clip_index.pkl` — a 1000×512 float32 embedding matrix plus metadata.

---

### 4. Query by text

Retrieve the top-5 most similar paintings for a natural-language query:

```bash
python clip_retrieval.py --mode query --query "Impressionist water scene at dusk"
```

| Argument | Default | Description |
|---|---|---|
| `--query` | `"Impressionist water scene at dusk"` | Free-text aesthetic query |
| `--topk` | `5` | Number of results to return |
| `--index` | `data/clip_index.pkl` | Path to saved index |

**Sample output:**
```
Rank   Score    Artist                       Style                        Genre
────────────────────────────────────────────────────────────────────────────────
1      0.2777   Henri Matisse                Realism                      religious painting
       └─ A Realism religious painting by Henri Matisse
...
```

---

### 5. Run the ablation study

Compare keyword matching vs. CLIP semantic retrieval across 5 style-targeted queries:

```bash
python clip_retrieval.py --mode ablation
```

---

## Ablation Results

Metric: **Precision@5** — fraction of top-5 results whose style matches the query target.

| Query | Target Style | Keyword P@5 | CLIP P@5 |
|---|---|---|---|
| "dreamy water lilies soft brush strokes nature" | Impressionism | 1.00 | **1.00** |
| "dark dramatic portrait candlelight shadows" | Baroque | 1.00 | 0.40 |
| "bold geometric shapes primary colors abstract" | Cubism | 1.00 | 0.60 |
| "emotional raw brush strokes vivid colour" | Expressionism | 1.00 | 0.00 |
| "serene Japanese woodblock ocean wave" | Ukiyo-e | 0.00 | 0.00 |
| **Average** | | **0.800** | **0.400** |

### Why the −0.40 delta is misleading

The keyword baseline exploits **label leakage** — each painting's text description contains the style name verbatim (e.g., *"A Impressionism landscape by Claude Monet"*). A query with the word "Impressionism" trivially matches all Impressionism records. This is a label lookup, not real semantic retrieval.

Real users don't type style names; they describe *what they're looking for*. The Ukiyo-e query — *"serene Japanese woodblock ocean wave"* — is the honest test: both methods fail equally at P@5 = 0.00.

**Two key findings:**

- **Keyword retrieval fails on unseen vocabulary.** Any query without the exact style name scores P@5 = 0.00, regardless of semantic relevance. CLIP's embedding space is specifically designed to bridge this vocabulary gap.
- **CLIP's zero-shot style recognition is uneven.** Iconic, well-documented styles (Impressionism) perform well. Visually ambiguous or underrepresented styles (Expressionism, Ukiyo-e) require WikiArt-specific fine-tuning — the primary goal of the final milestone.

---

## Architecture

```
User query (natural language)
        │
        ▼
  CLIP Text Encoder (ViT-B/32)
        │  512-d L2-normalised embedding
        ▼
  Cosine Similarity Search
        │  against 1000×512 image embedding matrix
        ▼
  Top-K Retrieved Paintings
  (artist, style, genre, description)
        │
        ▼
  [Final Milestone] LLM generates
  grounded explanation
```

**Why CLIP over CNN classifiers?** Traditional supervised art classifiers (ResNet, EfficientNet) can classify into known styles but cannot handle open-ended natural language queries. CLIP's joint embedding space allows arbitrary text queries to retrieve semantically relevant paintings without any labelled training data at query time.

**Why RAG over direct LLM generation?** LLMs hallucinate artwork details. RAG grounds the response in actual retrieved paintings from WikiArt, so every recommended artwork exists and its metadata is verifiable.

---

## Dataset

**WikiArt** ([huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart))

- 81,444 paintings across 27 art styles, 10 genres, 195 artists
- Spans Renaissance to Abstract Expressionism
- Milestone 1 uses a 1k streaming subset — no full 15 GB download required
- Labels are numeric integers mapped to strings via lookup tables in `data_pipeline.py`

The 1k subset is **heavily skewed**: Impressionism (276 samples) and Realism (210 samples) dominate, creating a class imbalance challenge. Resolving this via weighted sampling is a key ablation target for the final milestone.

---

## Expected Performance

| Metric | Milestone 1 (zero-shot) | Final Milestone (fine-tuned) |
|---|---|---|
| Average CLIP P@5 | 0.40 | 0.80+ (target) |
| Average Keyword P@5 | 0.80* | — |
| ROUGE-L (LLM explanations) | — | >0.35 (target) |

*\*Keyword P@5 inflated by label leakage — not a fair comparison.*

---

## Roadmap

- [x] **Milestone 1** — Data pipeline, CLIP ViT-B/32 index, text-to-image retrieval, ablation study
- [ ] **Final Milestone** — Fine-tune CLIP on WikiArt labels, weighted sampling for class balance, LLM-generated grounded explanations, ViT-B/32 vs ViT-L/14 comparison
<img width="1440" height="1736" alt="image" src="https://github.com/user-attachments/assets/a6ae29dd-967c-4499-ade3-6d2164453b8b" />

---

## References

- Radford, A. et al. (2021). *Learning transferable visual models from natural language supervision.* ICML 2021. [CLIP]
- Saleh, B. & Elgammal, A. (2016). *Large-scale classification of fine-art paintings.* arXiv:1505.00855. [WikiArt benchmark]
- Lewis, P. et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS 2020. [RAG]
- Shen, S. et al. (2022). *How much can CLIP benefit vision-and-language tasks?* ICLR 2022.
- Tourani, A. et al. (2026). *RAG-VisualRec: Vision- and text-enhanced RAG in recommendation.* arXiv:2506.20817.
- HuggingFace dataset card: [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart)
