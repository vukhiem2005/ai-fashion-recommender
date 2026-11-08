# ÉLAN — AI Fashion Recommender

**A full-stack fashion storefront with content-based product recommendations and reproducible offline model evaluation.**

ÉLAN combines a Next.js shopping interface, a FastAPI product service, SQLite inventory, and a TF-IDF recommendation engine. The project explores the engineering trade-offs between text similarity, heuristic reranking, and structured fashion metadata, rather than assuming that a newer model must be deployed.

> **Project status:** Local portfolio demo. **V1 is the active recommendation engine**; V2, V3, and V4 are saved offline experiments. A public Live Demo is not yet available.

## Features

- **Fashion storefront:** product search, category filters, price sorting, and pagination (24 items per page).
- **Product details:** images, descriptions, color/size variants, and SKU selection.
- **AI-powered discovery:** a **“You May Also Like”** section with the Top-5 most similar active products, similarity scores, and short matching reasons.
- **Catalog administration:** create and edit products, hide/restore products, and manage product images through an admin-key-protected interface.
- **Model experimentation:** compare four content-based recommenders using frozen candidate rankings and relevance labels.
- **Reproducible evaluation:** scripts and CSV/JSON/Markdown reports for Precision@5, Strict Precision@5, and pooled-candidate nDCG@5.

The local demonstration catalog contains **328 products**: **320 sampled H&M articles** and **8 original ÉLAN demo products**. This is a portfolio storefront, not a transactional checkout system.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| Backend | Python, FastAPI, SQLModel, SQLAlchemy, Uvicorn |
| Database | SQLite |
| Recommendation engine | scikit-learn TF-IDF, cosine similarity, heuristic reranking |
| Evaluation | Python, CSV/JSON, Precision@5, Strict Precision@5, nDCG@5 |
| Development | VS Code, PowerShell, Git, GitHub |

## Architecture

```text
Browser
  |-- /shop and /shop/[slug]
  |       |
  |       `-- Next.js API routes (/api/products, /api/recommendations/...)
  |                    |
  |                    v
  |                FastAPI (port 8001)
  |                    |-- Product and variant endpoints
  |                    |-- Recommendation endpoint -> V1 TF-IDF + cosine
  |                    `-- SQLModel / SQLite catalog
  |
  `-- /admin/products -> admin API (requires ELAN_ADMIN_KEY)
```

**Production recommendation path:** `frontend/src/components/YouMayAlsoLike.tsx` → Next.js proxy → `GET /recommendations/{product_id}` → `backend/app/recommender.py` (**V1**).

The recommendations are **content-based similar-item suggestions**, not behavior-based personalization. No user profiles, clickstream training, or collaborative filtering are implemented.

## Recommendation Engines

| Version | Approach | Status |
|---|---|---|
| **V1** | Weighted product text (category, name, description, colors) → TF-IDF (unigrams/bigrams) → cosine similarity | **Active application baseline** |
| **V2** | V1 similarity with category and name-derived subtype reranking | Offline experiment |
| **V3** | Refined subtype recognition and type-relation heuristics | Offline experiment |
| **V4** | Reranking with H&M `product_type_name` metadata | Offline experiment |

V1 ranks active catalog products and excludes the reference product itself. V2–V4 preserve the baseline and are **not** wired into the public recommendation endpoint.

## Benchmark: V1 vs. V2 vs. V3 vs. V4

The frozen blind-review holdout contains **18 reference products** (3 per category), **140 unique reference–candidate pairs**, and **90 Top-5 recommendation positions per model**.

| Model | Precision@5 | Strict Precision@5 | nDCG@5* |
|---|---:|---:|---:|
| **V1 — Active baseline** | 66.7% | **24.4%** | **0.784** |
| V2 | 70.0% | 22.2% | 0.741 |
| V3 | 70.0% | 22.2% | 0.741 |
| V4 | **71.1%** | **24.4%** | 0.759 |

**Label definitions:** `0` = irrelevant, `1` = related, `2` = close substitute. Precision@5 counts grades ≥ 1; Strict Precision@5 counts grade 2 only.

* **nDCG@5** uses gain `2^relevance - 1` and an ideal ranking from the **pooled retrieved candidates**, not the entire catalog. It is averaged over **16** references with defined ideal DCG; two references without a relevant pooled candidate are excluded from the main nDCG mean.
* Images and names were assessed **without access to model ranking identities**. Labels were assigned by an **AI evaluator**, not independently validated by human annotators; these results are exploratory, not a verified consumer-quality benchmark.
* V4 improves Precision@5 over V1 by **4.4 percentage points (4 of 90 positions)**, but ranks behind V1 on nDCG@5 and ties V1 on strict precision. V4 also reduces Tops-category Precision@5 from **73.3% to 53.3%**.
* A paired reference-bootstrap 95% percentile interval for V4 − V1 Precision@5 is approximately **−7.8 to +15.6 percentage points**, so this small holdout does not establish a robust improvement.

### Deployment Decision

**V1 remains active.** It has the highest measured nDCG@5 in this holdout, ties the highest Strict Precision@5, and has an established, stable API integration. V4 has slightly higher Precision@5, but its category-level regressions, uncertainty, and dependence on a private local metadata CSV do not justify changing the deployed recommendation path.

The goal is **evidence-based model selection**, not maximizing a single metric.

Full evidence and reproduction assets:

- [Evaluation report](backend/evaluation/holdout_v4_evaluation_report.md)
- [Structured benchmark metrics](backend/evaluation/holdout_v4_metrics.json)
- [Blind relevance labels](backend/evaluation/holdout_v4_blind_labeled.csv)
- [Frozen model rankings](backend/evaluation/holdout_v4_candidate_union.csv)
- [Evaluation script](backend/evaluation/evaluate_holdout_v4_results.py)

### Reproduce the Saved Benchmark (Windows PowerShell)

From the repository root, run:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\evaluation\evaluate_holdout_v4_results.py --out-dir .\backend\evaluation\recomputed
```

The script recalculates metrics from committed CSV inputs, without invoking recommendation models or accessing SQLite. It writes outputs to a separate `recomputed` directory to avoid overwriting the committed reports. On repeated runs, choose a different empty output directory (or explicitly use `--overwrite`).

## Run Locally

### Prerequisites

- Python and Node.js/npm installed
- Windows PowerShell (commands below), or equivalent commands on your platform
- Access to this repository (currently private) and a local product catalog for full storefront functionality (see **Data & Assets**)

### 1. Backend (Terminal 1)

```powershell
git clone https://github.com/khimneae/ai-fashion-recommender.git
cd ai-fashion-recommender\backend

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Required only for protected admin endpoints:
$env:ELAN_ADMIN_KEY = "replace-with-your-own-local-key"

python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Check [API health](http://127.0.0.1:8001/health) or open the [FastAPI documentation](http://127.0.0.1:8001/docs).

### 2. Frontend (Terminal 2)

From the repository root:

```powershell
cd frontend
npm ci
npm run dev
```

Open these local pages:

- **Storefront:** http://localhost:3000/shop
- **Product management:** http://localhost:3000/admin/products

Next.js proxies API requests to `http://127.0.0.1:8001` by default. For another backend URL, set `ELAN_BACKEND_URL` in the frontend environment before starting the dev server.

**Important:** On a clean clone, the backend may create an **empty** `backend/data/elan.db`. A running API does not mean the full 328-product demo catalog has been restored. Product and recommendation features require populated inventory; the admin UI/API can add demo records with a configured admin key. Original H&M catalog reconstruction requires appropriately obtained source data and the repository's import tools. Do not run destructive or `--apply` data scripts against an existing database without reviewing their safeguards and backups.

## Data & Assets

- Source material: H&M article metadata from the *H&M Personalized Fashion Recommendations* dataset; a **320-item local sample** was used for the demo.
- `data/private/hm/articles.csv`, downloaded H&M imagery, and `backend/data/elan.db` are **not redistributed** in the repository.
- Product prices for the imported H&M sample are **illustrative demo values**, not current retail prices.
- Some storefront products may display an image placeholder where an image is unavailable.
- The frozen evaluation CSV/JSON files **are committed** so benchmark metrics can be recalculated without shipping the private catalog or image assets.
- Use source data and imagery only under their applicable usage terms.

## Project Structure

```text
ai-fashion-recommender/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application
│   │   ├── models.py               # SQLModel product / variant schemas
│   │   ├── recommendations.py      # API route using V1
│   │   ├── recommender.py          # Active TF-IDF baseline (V1)
│   │   ├── recommender_v2.py       # Experimental subtype reranking
│   │   ├── recommender_v3.py       # Experimental refined heuristics
│   │   └── recommender_v4.py       # Experimental H&M metadata reranking
│   ├── evaluation/                 # Frozen rankings, labels, scripts, reports
│   ├── scripts/                    # Local H&M data / image utilities
│   └── requirements.txt
├── frontend/
│   ├── src/app/shop/               # Catalog and product pages
│   ├── src/app/admin/products/     # Inventory admin interface
│   ├── src/app/api/                # Next.js backend proxies
│   └── src/components/YouMayAlsoLike.tsx
└── README.md
```

## Limitations & Next Steps

- Evaluate recommendations with independent human raters and a larger, more diverse holdout.
- Improve catalog taxonomy consistency and candidate coverage for uncommon product types.
- Package metadata safely if a future experiment is selected for deployment.
- Add production-scale regression tests, a public demo deployment, and UI screenshots when available.

---

**Portfolio focus:** full-stack integration, content-based retrieval, offline experimentation, reproducible evaluation, and conservative production model selection.
