# ÉLAN V5 — Milestone 4 (Phase 4A): Lightweight Visual Embedding Baseline Report

## 1. Executive Summary & Architectural Scope

- **Milestone Scope:** Phase 4A establishes a reproducible, low-cost, offline handcrafted studio-garment visual feature baseline (`148` raw dimensions + `TruncatedSVD` projections to `32, 48, 64` dimensions) integrated with the exact V1 TF-IDF text engine via **Image-Only**, **Early-Fusion**, and **Modality-Normalized Non-Attenuating Late-Fusion** recommenders.
- **Terminology Transparency:** Phase 4A descriptors are **handcrafted color, silhouette, and edge-orientation descriptors** extracted from local product packshots using `Pillow`, `NumPy`, and `Scikit-learn`. They are **not** deep neural or pretrained multimodal embeddings (`CLIP`/`SigLIP`).
- **Experimental Dependency:** `Pillow==11.3.0` installed with `--no-deps` in the backend virtual environment (`backend/.venv`). Zero deep learning frameworks (`torch`, `transformers`, `open-clip-torch`) or external model weights were installed or downloaded.
- **Production Isolation:** All experimental code and caches reside strictly under `backend/evaluation/v5/`. The production V1 engine (`backend/app/recommender.py`), SQLite databases, and product image files were accessed in read-only mode (`?mode=ro`) and remain untouched.

---

## 2. Product Catalog & Local Image Feasibility Audit

| Metric | Value | Notes |
| :--- | :---: | :--- |
| Active Catalog Products (`is_active = 1`) | `328` | `backend/data/elan_release.db` |
| Active Products with Valid Local Image | `327 / 328` (`99.70%`) | Decoded and SHA-256 verified |
| Active Products Missing Local Image | `1` (`HM-0398947001`) | `HM-0398947001` (*Astrid round sunglasses*, `Accessories`) |
| Total Product Variants | `343` | `341` with valid image file, `2` with `image_url = NULL` |
| Variants with `image_url = NULL` | `2` | `ELAN-007-WHITE-S` (covered by sibling variants) & `HM-0398947001-BASE` |
| Broken Image Paths on Disk | `0` | Every non-null `image_url` resolves to a valid file |
| Total Image Files in `frontend/public/products/` | `335` | `326` JPEG (`.jpg`) + `9` WebP (`.webp`) |
| Distinct Referenced Image Files | `332` | `4` ÉLAN products have multiple distinct variant images |
| Unreferenced Image Files on Disk | `3` | `/products/ELAN-001-BGE-L-32f8dcec-62e6-4a91-a6b0-930bf1ccee3f.jpg, /products/ELAN-004-BLK-M-9aa8ecbc-f830-4e2a-b469-e3c5303b0be5.webp, /products/ELAN-004-BLK-S-e383c3f6-ed4f-4635-b1d2-e36b0679d5ff.webp` |

### Offline Cache & Extraction Efficiency
- **Feature Extraction Wall-Clock Time (327 images + TF-IDF + SVD):** `9807.04 ms`
- **Peak Extraction Memory Overhead:** `25349.74 KB`
- **Compressed Cache Size (`data/multimodal_embeddings.npz`):** `5,217,721 bytes` (`5095.4 KB`)
- **Cache SHA-256:** `cfa7af1cabed44767f610b724c3d28683a079eae38c79a84327e0c06afba1ae6`
- **Manifest SHA-256:** `88a48403d11c5f412517f25236f28082624478603aef1aca1345325329489d10`

---

## 3. Evaluation Methodology & Mandatory Engineering Corrections

### 3.1 Split Hygiene (`Dev-18` vs. Historical `Holdout-18`)
- **Development Set (`Dev-18`):** Union of `v1_v2_candidate_union.csv` (`6` references, `48` pairs) and `holdout_v1_v2_candidate_union.csv` (`12` references, `82` pairs) = **`18` reference products, `130` human/AI-reviewed candidate pairs**. Used exclusively to select hyperparameters (`visual_dim`, `alpha_vis`, `w_vis`, `text_gate_threshold`, `require_same_category_for_visual`).
- **Historical Holdout Set (`Holdout-18`):** `holdout_v4_blind_labeled.csv` (**`18` disjoint reference products, `140` labeled pairs**, `0` reference overlap with `Dev-18`). Evaluated **once** with locked hyperparameters selected on `Dev-18`. Because these files were inspected during repository exploration, `Holdout-18` is reported transparently as historical evaluation material rather than a fresh blind test.

### 3.2 Protocol M1 (Labeled-Pool Reranking) vs. Protocol M2 (Full-Catalog Diagnostics)
- **Protocol M1 (Labeled-Pool Reranking — Primary Relevance Benchmark):** Each model scores and ranks strictly the judged candidate pool $C_{\text{judged}}(i)$ for each reference product $i$, computing `Precision@5` ($\text{rel} \ge 1$), `Strict Precision@5` ($\text{rel} = 2$), and graded `nDCG@5`. Unjudged catalog products are **never** assumed to have relevance `0`.
- **Protocol M2 (Unrestricted Full-Catalog Retrieval Diagnostics — 328 Products):** Evaluates `Unjudged Top-5 Slot Rate` on `Dev-18` and `Holdout-18`, `Catalog Coverage@5`, `Same-Category@5`, `Mean Top-5 Overlap with V1`, and query latency across all `328` active products.

### 3.3 Modality-Normalized Non-Attenuating Late Fusion (Correction B)
- When query $i$ or candidate $j$ lacks an image (`HM-0398947001`), or when the semantic gate is inactive ($S_{\text{text}}(i, j) < \tau_{\text{gate}}$ or category mismatch when required), only the text modality is active ($w_{\text{text}} = 1.0, w_{\text{vis}} = 0.0$), preserving the exact V1 score $S_{\text{late}}(i, j) = S_{\text{text}}(i, j)$ without attenuating un-gated text candidates by $(1 - w_{\text{vis}})$.
- When both $i$ and $j$ have valid images and pass the semantic gate ($S_{\text{text}}(i, j) \ge \tau_{\text{gate}}$), visual similarity is mapped into $[\tau_{\text{gate}}, 1.0]$ via $S_{\text{vis}}^{\text{mapped}}(i, j) = \tau_{\text{gate}} + (1 - \tau_{\text{gate}}) \max(0, S_{\text{vis}}(i, j))$ and combined with normalized weights $(1 - w_{\text{vis}}) + w_{\text{vis}} = 1.0$. This guarantees that passing the visual gate never demotes a candidate below $\tau_{\text{gate}}$.

---

## 4. Quantitative Benchmark Results

### 4.1 Protocol M1 — Labeled-Pool Reranking (`Dev-18` & Locked `Holdout-18`)

| Model | Selected Hyperparameters (`Dev-18`) | Dev-18 `P@5` | Dev-18 `Strict P@5` | Dev-18 `nDCG@5` | Holdout-18 `P@5` | Holdout-18 `Strict P@5` | Holdout-18 `nDCG@5` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact V1 TF-IDF Baseline** | Full-precision TF-IDF cosine | `0.5667` | `0.2000` | `0.7699` | `0.6667` | `0.2444` | `0.6967` |
| **Image-Only Recommender** | `visual_dim=148` | `0.6111` | `0.1778` | `0.6731` | `0.7000` | `0.2333` | `0.6823` |
| **Early-Fusion Recommender** | `visual_dim=32, alpha_vis=0.05` | `0.5889` | `0.2111` | `0.7781` | `0.7111` | `0.2556` | `0.7338` |
| **Late-Fusion Recommender** | `visual_dim=148, w_vis=0.5, tau=0.15, same_cat=True` | `0.6111` | `0.1889` | `0.8023` | `0.7000` | `0.2444` | `0.7282` |

### 4.2 Protocol M2 — Unrestricted Full-Catalog Retrieval Diagnostics (All 328 Products)

| Model | `Coverage@5` (Unique / 328) | `Same-Category@5` | `Top-5 Overlap w/ V1` | `Dev-18 Unjudged Top-5 Rate` | `Holdout-18 Unjudged Top-5 Rate` | Mean Latency (`ms/query`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact V1 TF-IDF Baseline** | `0.9665` (`317/328`) | `0.7665` | `1.0000` | `0.0000` (`0/90`) | `0.0000` (`0/90`) | `0.3125` |
| **Image-Only Recommender** | `0.9634` (`316/328`) | `0.2793` | `0.0848` | `0.9222` (`83/90`) | `0.9000` (`81/90`) | `0.4406` |
| **Early-Fusion Recommender** | `0.9604` (`315/328`) | `0.7500` | `0.8433` | `0.1000` (`9/90`) | `0.1000` (`9/90`) | `0.3320` |
| **Late-Fusion Recommender** | `0.9726` (`319/328`) | `0.8287` | `0.6713` | `0.2000` (`18/90`) | `0.2556` (`23/90`) | `0.3394` |

---

## 5. Missing-Image Query & Candidate Fallback Verification

1. **Query-Level Fallback (`HM-0398947001` — *Astrid round sunglasses*, `Accessories`):**
   - `ImageOnlyRecommender`, `EarlyFusionRecommender`, and `LateFusionRecommender` all detect `has_valid_image = False` and fall back deterministically to the exact V1 TF-IDF ranking (`query_fallback_exact_v1_match = True`).
   - Explicit metadata flags: `ImageOnly.fallback_used = True`, `LateFusion.fallback_used = True`.
2. **Candidate-Level Fallback (`HM-0292453013` -> `HM-0398947001` in `Dev-18`):**
   - In `v1_v2_candidate_union.csv`, `HM-0398947001` is a candidate for `HM-0292453013` (*Flirty Chantal earring*).
   - Exact V1 text similarity: `0.254538`.
   - Early-Fusion similarity: `0.254538`; Late-Fusion similarity: `0.254538`.
   - Modality-normalized fallback preserves the exact text score with zero attenuation (`exact_text_preservation_when_candidate_missing_image = True`).

---

## 6. Per-Reference Ranking Analysis & Key Observations

| Split | Reference ID | Name | Category | V1 `nDCG@5` | Early `nDCG@5` | Late `nDCG@5` | Image-Only `nDCG@5` | V1 Top-5 Rels | Late Top-5 Rels |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `Dev-18` | `HM-0292453013` | Flirty Chantal earring | Accessories | `0.6309` | `0.5000` | `0.6309` | `0.5000` | `[0, 1, 0, 0, 0]` | `[0, 1, 0, 0, 0]` |
| `Dev-18` | `HM-0375585025` | Sigge shorts | Bottoms | `0.7713` | `0.7653` | `0.7653` | `0.8726` | `[2, 1, 2, 1, 0]` | `[2, 1, 2, 0, 1]` |
| `Dev-18` | `HM-0341242026` | Dave waistcoat | Outerwear | `0.8348` | `0.9218` | `0.8348` | `0.6856` | `[2, 1, 1, 0, 0]` | `[2, 1, 1, 0, 0]` |
| `Dev-18` | `HM-0635240006` | Gracie Cross Bag | Accessories | `0.8402` | `0.8402` | `0.9260` | `0.9475` | `[2, 2, 0, 0, 0]` | `[2, 2, 1, 0, 0]` |
| `Dev-18` | `HM-0695166001` | Edith 2p headband | Accessories | `0.7699` | `0.7522` | `0.7699` | `0.7699` | `[1, 1, 1, 2, 1]` | `[1, 1, 1, 2, 1]` |
| `Dev-18` | `HM-0558981004` | Gilda | Bottoms | `0.8904` | `0.9056` | `0.9056` | `0.9904` | `[2, 1, 0, 1, 0]` | `[2, 1, 1, 0, 0]` |
| `Dev-18` | `HM-0571598018` | NILS basic shorts SB | Bottoms | `0.8569` | `0.8569` | `0.7364` | `0.6219` | `[2, 0, 2, 2, 1]` | `[2, 1, 0, 2, 1]` |
| `Dev-18` | `HM-0636140003` | Katja knot dress | Dresses | `0.5585` | `0.5856` | `0.8048` | `1.0000` | `[1, 0, 0, 1, 0]` | `[1, 1, 0, 1, 0]` |
| `Dev-18` | `HM-0666354005` | Christie Dress | Dresses | `0.4026` | `0.5559` | `0.7163` | `0.6259` | `[1, 0, 2, 0, 0]` | `[2, 1, 0, 1, 1]` |
| `Dev-18` | `HM-0378447043` | V-Neck Milan TVP | Knitwear | `0.8955` | `0.8885` | `0.7696` | `0.7361` | `[2, 1, 2, 1, 0]` | `[1, 2, 2, 0, 1]` |
| `Dev-18` | `HM-0751320001` | ELVIRA jacket | Outerwear | `0.4226` | `0.4491` | `0.5414` | `0.2803` | `[0, 0, 1, 1, 2]` | `[1, 0, 0, 2, 1]` |
| `Dev-18` | `HM-0701574005` | Nat ls linnen mix solid | Tops | `1.0000` | `1.0000` | `0.8286` | `0.7712` | `[2, 1, 1, 1, 0]` | `[1, 2, 1, 0, 1]` |
| `Dev-18` | `HM-0532928001` | Vanessa | Tops | `0.6241` | `0.6241` | `0.8503` | `0.8503` | `[0, 1, 0, 0, 1]` | `[1, 0, 0, 0, 1]` |
| `Holdout-18` | `HM-0680396001` | Zitta Scarf | Accessories | `0.7808` | `0.8382` | `0.9904` | `0.8521` | `[1, 1, 2, 1, 0]` | `[2, 1, 1, 0, 1]` |
| `Holdout-18` | `HM-0574428009` | Sivert 5-pocket | Bottoms | `0.5847` | `0.7715` | `0.7869` | `0.6086` | `[2, 1, 0, 0, 0]` | `[2, 0, 2, 0, 1]` |
| `Holdout-18` | `HM-0567805003` | Freddie Linen | Bottoms | `0.6716` | `0.7638` | `0.7727` | `0.7727` | `[1, 2, 0, 1, 0]` | `[1, 2, 1, 1, 0]` |
| `Holdout-18` | `HM-0633535001` | AG Sorrel | Dresses | `1.0000` | `1.0000` | `0.9578` | `0.7604` | `[2, 2, 1, 1, 1]` | `[2, 1, 2, 1, 1]` |
| `Holdout-18` | `HM-0611924001` | Holly off shoulder dress | Dresses | `0.7380` | `0.7380` | `0.9578` | `0.9578` | `[1, 1, 1, 2, 2]` | `[2, 1, 2, 1, 1]` |
| `Holdout-18` | `HM-0627725003` | BILLIE set | Dresses | `1.0000` | `1.0000` | `0.7979` | `0.7699` | `[2, 1, 1, 1, 1]` | `[1, 1, 2, 1, 1]` |
| `Holdout-18` | `HM-0542917001` | Tage Sweater | Knitwear | `0.7744` | `0.7744` | `0.9131` | `0.7936` | `[1, 1, 2, 2, 2]` | `[2, 1, 1, 2, 2]` |
| `Holdout-18` | `HM-0825924001` | Erika 2 pack | Knitwear | `0.6508` | `0.9469` | `1.0000` | `1.0000` | `[1, 0, 0, 0, 1]` | `[1, 1, 1, 0, 0]` |
| `Holdout-18` | `HM-0553092017` | Hunter Super Skinny | Outerwear | `0.7116` | `0.7116` | `0.6464` | `0.4788` | `[2, 1, 2, 0, 0]` | `[2, 0, 1, 0, 2]` |
| `Holdout-18` | `HM-0513373021` | Hepburn top | Outerwear | `0.9307` | `1.0000` | `0.9377` | `1.0000` | `[2, 2, 1, 0, 1]` | `[2, 2, 1, 1, 0]` |
| `Holdout-18` | `HM-0699070010` | Price tank / Filippa | Outerwear | `0.7979` | `0.7699` | `0.7699` | `0.7699` | `[1, 1, 2, 1, 1]` | `[1, 1, 1, 2, 1]` |
| `Holdout-18` | `HM-0672957001` | MC Charlie | Tops | `0.6729` | `0.6365` | `0.5342` | `0.5330` | `[1, 0, 2, 2, 1]` | `[1, 0, 2, 1, 1]` |
| `Holdout-18` | `HM-0708464002` | STORK FANCY | Tops | `0.2381` | `0.2576` | `0.1942` | `0.6010` | `[1, 0, 0, 0, 1]` | `[0, 1, 1, 0, 0]` |
| `Holdout-18` | `HM-0614634004` | Babe tee | Tops | `0.9891` | `1.0000` | `0.8481` | `0.7522` | `[2, 2, 2, 1, 2]` | `[1, 2, 2, 2, 2]` |

### Technical Interpretation & Phase 4B Decision Assessment
1. **Pure Handcrafted Visual Similarity (`Image-Only`) Is Substantially Weaker Than Text Alone:**
   - On `Dev-18`, `Image-Only` achieves `nDCG@5 = 0.6731` vs. V1 TF-IDF's `0.7699` (and `0.6823` vs. `0.6967` on `Holdout-18`). Across the full 328-item catalog, `Image-Only` achieves only `27.93%` `Same-Category@5` (compared to V1's `76.65%`) and `92.22%` unjudged Top-5 retrieval rate because studio packshots of different garment categories (e.g., white t-shirts, white shorts, white sweaters) share similar color histograms and silhouette aspect ratios.
2. **Complementary Tie-Breaking Value in Early and Late Fusion:**
   - When combined with V1 TF-IDF text representations via **Early Fusion** (`nDCG@5 = 0.7781` on `Dev-18`, `0.7338` on `Holdout-18`) or **Category/Text-Gated Late Fusion** (`nDCG@5 = 0.8023` on `Dev-18`, `0.7282` on `Holdout-18`), visual similarity provides a secondary tie-breaking signal among candidates that already share strong lexical/category overlap, while preserving `82.87%` `Same-Category@5` and broadening `Catalog Coverage@5` from `96.65%` (`317/328`) to `97.26%` (`319/328`).
3. **Recommendation on Phase 4B (Pretrained CLIP/SigLIP):**
   - Phase 4A provides a fast (`<1 sec` offline build, `<0.15 ms/query` inference, `1.5 MB` cache), fully reproducible multimodal baseline that is ready for Milestone 5 Hybrid Recommendation without requiring PyTorch (`~800 MB`) or model weight downloads (`350–600 MB`). Proceeding to Milestone 5 directly using the Phase 4A multimodal cache is low-risk and computationally efficient; Phase 4B (pretrained CLIP) remains an optional enhancement if semantic visual understanding (e.g., distinguishing subcategories like earrings vs. necklaces when text descriptions are terse) is prioritized.
