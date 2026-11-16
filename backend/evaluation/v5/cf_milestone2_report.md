# ELAN V5 — Milestone 2: Collaborative Filtering Benchmark Report

> **CRITICAL SCIENTIFIC DISCLAIMER (`SYNTHETIC_SIMULATION_ONLY`):**
> Track A personalized user-to-item metrics in this report are evaluated on the
> Milestone 1 synthetic interaction dataset (`SYNTHETIC_SIMULATION_V5`). They measure
> how effectively each algorithm recovers simulated persona preferences and outfit
> co-occurrences from sparse training interactions. **They do NOT measure real human
> preference or justify replacing the production V1 recommender.**

## 1. Experimental Setup & Chronological Hygiene

- **Random Seed:** `20260501`
- **Catalog Size:** `328` active products (`backend/data/elan_release.db`, read-only `mode=ro`)
- **Training Set (`train`):** `16091` events across `600` users (`2026-01-01` .. `2026-03-04`)
- **Validation Set (`val`):** `3609` events (`557` users with novel targets unseen in `train`)
- **Test Set (`test`):** `3635` events (`545` users with novel targets vs. `train`; `542` users with novel targets vs. `train ∪ val`)
- **Model Fitting Policy:** All model parameters (`PopularityBaseline`, `V1_TFIDF_Content`, `ItemKNN_CF`, `TruncatedSVD_MF`) are fitted **exclusively on `train`**. No model parameters are refitted on `val` or `test`.

## 2. Validation Hyperparameter Selection (`MIN_VAL_COVERAGE = 25.0%`)

- **Selection Protocol:** Maximize validation `Graded nDCG@5` among configurations satisfying `Catalog Coverage@5 >= 25.0%` (`82 / 328` items). If `0` configurations satisfy the coverage floor, deterministically fall back to the highest validation `Graded nDCG@5` across the grid and flag `coverage_threshold_satisfied = False`.
- **Matrix Configurations Evaluated (`6` pairs):** `dedup_mode ∈ {'max_weight', 'log_sum'} × iuf_exponent ∈ {0.0, 0.5, 1.0}`

### 2.1. `ItemKNN_CF` Grid Selection (`192` configurations evaluated)
- **Coverage Floor Status:** `192 / 192` configurations met `Coverage@5 >= 25.0%` (`coverage_threshold_satisfied = True`, `fallback_used = False`).
- **Validation Coverage Range Across Grid:** `35.98%` .. `100.00%`
- **Selected Configuration:** `dedup_mode = 'log_sum'`, `iuf_exponent = 0.5`, `shrinkage = 20.0`, `k_neighbors = 100`, `normalize_sim_columns = False` (Validation `Coverage@5 = 35.98%`, `Graded nDCG@5 = 0.0506`).

### 2.2. `TruncatedSVD_MF` Grid Selection (`60` configurations evaluated; `30` unique Track A subspace projections)
- **Coverage Floor Status:** `48 / 60` configurations met `Coverage@5 >= 25.0%` (`coverage_threshold_satisfied = True`, `fallback_used = False`).
- **Validation Coverage Range Across Grid:** `21.04%` .. `70.43%`
- **Selection Policy Applied:** `highest_val_ndcg_graded_among_configs_meeting_25pct_coverage_floor`
- **Selected Configuration:** `dedup_mode = 'log_sum'`, `iuf_exponent = 0.0`, `n_components = 16`, `singular_value_power = 0.5` (Explained Variance Ratio Sum: `18.93%`, Validation `Coverage@5 = 33.23%`, `Graded nDCG@5 = 0.0449`).
- **Parameter Semantics Note (`singular_value_power`):** Track A personalized user scoring projects user profiles onto the rank-$k$ orthonormal subspace `(r_u @ V_k) @ V_k.T`, which is mathematically invariant to `singular_value_power`. The parameter `singular_value_power` only scales item embeddings `V_k @ diag(sigma_k ** p)` for Track B item-to-item cosine similarity.

## 3. Track A — Personalized Next-Item Recommendation (`[SIMULATION VALIDATION ONLY]`)

### 3.1. Validation Split Results (`conditioning = train`, `targets = val novel`)

| Model | Precision@5 | Recall@5 | Binary nDCG@5 | Graded nDCG@5 | Catalog Coverage@5 | Meets >=25% Floor |
|---|---:|---:|---:|---:|---:|---|
| **PopularityBaseline** | 2.94% | 3.85% | 0.0367 | **0.0352** | 3.35% (11/328) | No |
| **V1_TFIDF_Content** | 2.37% | 3.18% | 0.0298 | **0.0288** | 82.32% (270/328) | Yes |
| **ItemKNN_CF** | 4.60% | 5.67% | 0.0532 | **0.0506** | 35.98% (118/328) | Yes |
| **TruncatedSVD_MF** | 3.81% | 5.28% | 0.0464 | **0.0449** | 33.23% (109/328) | Yes |

### 3.2. Test Split — Protocol A1: Frozen Train History (`conditioning = train`, `targets = test novel vs. train`)

| Model | Precision@5 | Recall@5 | Binary nDCG@5 | Graded nDCG@5 | Catalog Coverage@5 | Unique Items |
|---|---:|---:|---:|---:|---:|---:|
| **PopularityBaseline** | 3.27% | 4.59% | 0.0434 | **0.0383** | 3.35% | 11/328 |
| **V1_TFIDF_Content** | 2.35% | 2.82% | 0.0248 | **0.0220** | 80.79% | 265/328 |
| **ItemKNN_CF** | 4.11% | 5.61% | 0.0538 | **0.0514** | 36.59% | 120/328 |
| **TruncatedSVD_MF** | 3.71% | 4.86% | 0.0470 | **0.0464** | 33.54% | 110/328 |

### 3.3. Test Split — Protocol A2: Folded-In `train + val` History (`conditioning = train ∪ val`, `targets = test novel vs. train ∪ val`, frozen `train` model parameters)

| Model | Precision@5 | Recall@5 | Binary nDCG@5 | Graded nDCG@5 | Catalog Coverage@5 | Unique Items |
|---|---:|---:|---:|---:|---:|---:|
| **PopularityBaseline** | 3.10% | 4.40% | 0.0410 | **0.0363** | 4.27% | 14/328 |
| **V1_TFIDF_Content** | 2.18% | 2.67% | 0.0242 | **0.0240** | 78.66% | 258/328 |
| **ItemKNN_CF** | 4.54% | 6.21% | 0.0567 | **0.0544** | 30.18% | 99/328 |
| **TruncatedSVD_MF** | 4.06% | 5.60% | 0.0508 | **0.0485** | 33.54% | 110/328 |

## 4. Track B — Item-to-Item Similarity vs. V1 on the 18-Reference Holdout

> **Methodological Note (No Unjudged-as-Zero Bias):**
> The frozen holdout (`holdout_v4_blind_labeled.csv`) contains relevance judgments (`0=irrelevant`,
> `1=related`, `2=close substitute`) ONLY for the **140 candidate pairs** pooled from content-based
> models V1–V4 across 18 reference products. In full-catalog retrieval, collaborative models retrieve
> cross-category complementary items that are **unjudged** in this pool. Therefore, Table 4.1 evaluates
> **Labeled-Pool Reranking** (ranking only the judged candidates for each reference), while Table 4.2
> reports **Full-Catalog Retrieval Diagnostics** (unjudged slot rate and Top-5 overlap with V1).

### 4.1. Labeled-Pool Reranking (18 References, 140 Judged Candidate Pairs)

| Model | Labeled-Pool P@5 | Labeled-Pool Strict P@5 | Labeled-Pool nDCG@5 (n=16) |
|---|---:|---:|---:|
| **PopularityBaseline** | 65.56% | 21.11% | **0.6376** |
| **V1_TFIDF_Content** | 66.67% | 24.44% | **0.7838** |
| **ItemKNN_CF** | 66.67% | 17.78% | **0.6087** |
| **TruncatedSVD_MF** | 65.56% | 18.89% | **0.6277** |

### 4.2. Unconstrained Full-Catalog Retrieval Diagnostics (328 Products)

| Model | Unjudged Top-5 Slot Rate (18 Holdout Refs) | Mean Top-5 Overlap with V1 (All 328 Products) |
|---|---:|---:|
| **PopularityBaseline** | 97.78% (88/90) | 1.65% |
| **V1_TFIDF_Content** | 0.00% (0/90) | 100.00% |
| **ItemKNN_CF** | 94.44% (85/90) | 4.76% |
| **TruncatedSVD_MF** | 90.00% (81/90) | 4.21% |

## 5. Runtime & Memory Telemetry

| Model | Fit Time (ms) | Peak Fit Memory (KB) | Full-Split Inference Time (ms) |
|---|---:|---:|---:|
| **PopularityBaseline** | 0.34 ms | 0.00 KB | 237.58 ms |
| **V1_TFIDF_Content** | 49.70 ms | 0.00 KB | 375.57 ms |
| **ItemKNN_CF** | 287.59 ms | 5757.82 KB | 211.40 ms |
| **TruncatedSVD_MF** | 22.89 ms | 1872.74 KB | 189.05 ms |

## 6. Reproducibility & Verification

- **`data/cf_milestone2_metrics.json` SHA-256:** `a03bd2d0818193f7532637325d147f5c5e0a9b3f89422ff3afbed7642a026a84`
