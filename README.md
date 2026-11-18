# ÉLAN — AI-Powered Fashion Discovery

**A full-stack contemporary fashion storefront pairing multimodal visual discovery and collaborative filtering with reproducible offline evaluation.**

[**Live Demo**](https://elan-fashion-store.vercel.app/) · [**Backend Health**](https://elan-fashion-api.onrender.com/health) · [**Source Repository**](https://github.com/vukhiem2005/ai-fashion-recommender)

ÉLAN is a full-stack AI-powered fashion e-commerce simulation featuring 328 active products and 343 SKU variants across six apparel categories.
Built with a minimalist editorial aesthetic, the platform integrates only recommendation models that have been validated through offline benchmarks.

## Key Features

- **Collection Browsing:**
  Keyword search, category filtering (`Outerwear`, `Tops`, `Bottoms`, `Dresses`, `Knitwear`, `Accessories`), price sorting, and 24-item pagination.
- **Product Detail & Shopping Bag:**
  Variant selection by color and size, persistent browser shopping bag with quantity controls, and local guest profile preferences.
- **Similar-Product Discovery (`You May Also Like`):**
  Top-5 visually and textually related product recommendations surfaced on every product detail page.
- **Synthetic Shopper Personalization (`Your Style Edit`):**
  Interactive style-profile switcher on `/shop` demonstrating personalized Top-5 recommendations across distinct shopper personas and cold-start visitors.

## AI Recommendation Systems

- **CLIP Late Fusion (Primary Similar-Item Engine):**
  Combines pretrained visual image embeddings with textual attribute similarity to rank complementary styles across the catalog.
- **TF-IDF + Cosine Similarity (Content Fallback):**
  Ranks products from weighted category, title, color, and description metadata whenever an item lacks an image embedding or multimodal serving is disabled.
- **ItemKNN Collaborative Filtering (Synthetic Personalization):**
  Recommends Top-5 unseen pieces for selectable simulated shopper profiles based on item co-interaction neighborhoods, with popularity fallback for cold-start profiles.

## Evaluation Highlights

Models are evaluated separately by recommendation task and deployed only when supported by offline evidence:

- **Similar-Item Retrieval:**
  On the 18-reference Holdout benchmark (`140` judged candidate pairs), **CLIP Late Fusion** achieved `nDCG@5 = 0.7625` compared to `0.6967` for the **TF-IDF** baseline (`+9.44%`), with `Precision@5 = 71.11%` and `Strict Precision@5 = 26.67%`.
- **Personalized Next-Item Ranking:**
  On the chronological synthetic interaction split (Test Protocol A1, `545` users), **ItemKNN** achieved `Graded nDCG@5 = 0.0514` (`Precision@5 = 4.11%`, `Catalog Coverage@5 = 36.59%`), outperforming `TruncatedSVD` (`0.0464`) and popularity/content baselines.
- **Deferred Research Models:**
  `TruncatedSVD`, `Apriori` / `FP-Growth` association rule mining (pure test `HitRate@5 = 1.29%`, `0/256` configs meeting validation coverage floors), and linear `Hybrid` fusion (`Graded nDCG@5 = 0.0501`) are preserved offline rather than deployed to the storefront.

> **Evaluation Scope:**
> The 18-reference holdout is a historically reused exploratory benchmark with AI-assisted relevance labels rather than an independent human-annotated blind test.
> Personalization metrics are computed on computer-generated synthetic interactions (`SYNTHETIC_SIMULATION_V5`) and do not measure real customer behavior.
> Item-similarity and user-personalization metrics use distinct protocols and are not directly comparable.

## Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4 |
| **Backend** | Python, FastAPI, SQLModel, SQLAlchemy, Uvicorn |
| **AI / ML** | CLIP (`clip-vit-B-32` precomputed `.npz` embeddings), scikit-learn (`TF-IDF`), NumPy (`ItemKNN`, `TruncatedSVD`, `Apriori`, `FP-Growth`) |
| **Database** | SQLite (versioned read-only release snapshot `backend/data/elan_release.db`) |
| **Deployment** | Vercel (frontend), Render (backend) |

## Run Locally (Windows PowerShell)

### 1. Backend (Terminal 1)

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:ELAN_CATALOG_MODE = "release"
$env:ELAN_CLIP_RECOMMENDATIONS_ENABLED = "true"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

### 2. Frontend (Terminal 2)

```powershell
cd frontend
npm ci
$env:ELAN_BACKEND_URL = "http://127.0.0.1:8001"
npm run dev
```

Open `http://localhost:3000` (health check: `http://127.0.0.1:8001/health`).

## Research & Documentation

Detailed experimental protocols, hyperparameter grids, ablation tables, and SHA-256 artifact digests are committed in the repository:

- [Milestone 1 — Reproducible Synthetic Interaction Dataset (`SYNTHETIC_SIMULATION_V5`)](backend/evaluation/v5/synthetic_dataset_report.md)
- [Milestone 2 — Collaborative Filtering (`ItemKNN` vs. `TruncatedSVD`) Report](backend/evaluation/v5/cf_milestone2_report.md)
- [Milestone 3 — Association Rule Mining (`Apriori` & `FP-Growth`) Report](backend/evaluation/v5/arm_milestone3_report.md)
- [Milestone 4A — Lightweight Multimodal Retrieval Report](backend/evaluation/v5/multimodal_milestone4_report.md)
- [Milestone 4B — Neural CLIP Multimodal (`Dev-18` & `Holdout-18`) Final Report](backend/evaluation/v5/neural_multimodal_milestone4b_final_report.md)
- [Milestone 5 — Calibrated Hybrid Recommendation & Ablation Report](backend/evaluation/v5/hybrid_milestone5_report.md)
- [Historical Content Baseline Evaluation (V1–V4 Holdout Report)](backend/evaluation/holdout_v4_evaluation_report.md)

**Academic Simulation & Data Provenance:**
ÉLAN is an academic simulation and engineering portfolio project.
The release catalog (`328` products, `335` tracked image files) combines `320` sampled articles from the *H&M Personalized Fashion Recommendations* dataset with `8` original ÉLAN demo items, and personalized profiles use deterministic synthetic interactions (`seed=20260501`).
All product prices are illustrative, shopping bag state is stored locally in the browser, and real checkout, orders, or payment processing are unavailable.
