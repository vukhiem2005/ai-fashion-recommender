# ELAN V5 — Milestone 3: Association Rule Mining (Frequently Bought Together) Benchmark Report

> **CRITICAL SCIENTIFIC DISCLAIMER (`SYNTHETIC_SIMULATION_ONLY`):**
> All transaction baskets and evaluation metrics in this report derive from the Milestone 1
> synthetic dataset (`SYNTHETIC_SIMULATION_V5`). Retrospective basket-completion accuracy
> measures conditional co-occurrence recovery under simulation rules; **it does NOT establish
> real-world recommendation effectiveness or causal sales uplift.**

## 1. Transaction Dataset & Chronological Cart-Order Verification

- **Ordering Verification:** Verified across 100% of orders (1,470/1,470) that order_item_id sequence in synthetic_order_items.csv is identical to the session's chronological add_to_cart sequence_number and event_ts in synthetic_interactions.csv.
- **Within-Order Deduplication:** Repeated product IDs within a single order are deduplicated into distinct itemsets (`duplicate_product_occurrences_removed = 0`).

| Split | Total Orders | Single-Item Orders | Multi-Item Orders (`|B| >= 2`) | Multi-Item Ratio | Mean Distinct Items/Order | Verified Cart Order |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 1016 | 324 | **692** | 68.11% | 1.9715 | 1016/1016 |
| `val` | 229 | 83 | **146** | 63.76% | 1.8908 | 229/229 |
| `test` | 225 | 70 | **155** | 68.89% | 1.9956 | 225/225 |

## 2. Apriori vs. FP-Growth Equivalence & Complexity Benchmark (`train` Orders)

| `min_support_count` | Relative `min_support` | `|L1|` | `|L2|` | `|L3|` | Total Itemsets | Exact Equivalence | Apriori Fit (ms) | FP-Growth Fit (ms) | Apriori Mem (KB) | FP-Growth Mem (KB) |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2 | 0.001969 | 299 | 52 | 0 | **351** | `True` | 3435.46 ms | 79.76 ms | 489.95 KB | 396.35 KB |
| 3 | 0.002953 | 253 | 6 | 0 | **259** | `True` | 2562.95 ms | 50.89 ms | 404.46 KB | 334.78 KB |
| 4 | 0.003937 | 222 | 0 | 0 | **222** | `True` | 1622.69 ms | 56.44 ms | 385.32 KB | 327.38 KB |
| 5 | 0.004921 | 181 | 0 | 0 | **181** | `True` | 1122.16 ms | 56.05 ms | 376.36 KB | 296.59 KB |

## 3. Validation Hyperparameter Selection (`256` Configurations Evaluated)

- **Threshold Semantics:** `lift > 1.0` indicates positive statistical association; `lift = 1.0` indicates independence; `lift < 1.0` indicates negative association. All grid configurations enforce `lift >= min_lift > 1.0`.
- **Selection Constraints on `val`:** `Rule Coverage Rate >= 25.0%` and `Catalog Coverage@5 >= 20.0%`.
- **Eligibility & Status:** `0 / 256` configurations satisfied both constraints (`constraints_satisfied = False`, `fallback_used = True`).
- **Selected Configuration:** `min_support_count = 2` (`support >= 0.001969`), `min_confidence = 0.05`, `min_lift = 1.05`, `alpha_smooth = 0.0`, `beta_lift = 1.0`.
- **Discovered Itemsets & Rules (`train`):** `L1=299`, `L2=52`, `L3=0`; **Total Rules = `104`** (`104` single-antecedent `1->1` rules and `0` multi-antecedent `2->1` rules).

## 4. Offline Basket-Completion Evaluation (`[SIMULATION VALIDATION ONLY]`)

### 4.1. Validation Split — Protocol B1: Leave-Last-Item-Out (`146` Multi-Item Orders)

| Model | HitRate@5 | Recall@5 | Precision@5 | nDCG@5 | Catalog Cov@5 | Rule Cov Rate | Basket Fallback Rate | Cond. HitRate@5 (Matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **OrderPopularityBaseline** | 6.16% | 6.16% | 1.23% | **0.0340** | 1.83% (6/328) | 0.00% | 100.00% | 0.00% (0) |
| **V1_TFIDF_Content** | 5.48% | 5.48% | 1.10% | **0.0368** | 78.35% (257/328) | 100.00% | 0.00% | 5.48% (146) |
| **PairCooccurrence_Pure** | 4.79% | 4.79% | 0.96% | **0.0261** | 48.17% (158/328) | 99.32% | 0.00% | 4.83% (145) |
| **PairCooccurrence_With_Fallback** | 5.48% | 5.48% | 1.10% | **0.0287** | 48.17% (158/328) | 99.32% | 10.27% | 5.52% (145) |
| **Apriori_ARM_Pure** | 2.05% | 2.05% | 0.41% | **0.0129** | 15.24% (50/328) | 41.78% | 0.00% | 4.92% (61) |
| **Apriori_ARM_With_Fallback** | 7.53% | 7.53% | 1.51% | **0.0384** | 15.55% (51/328) | 41.78% | 94.52% | 9.84% (61) |

### 4.2. Test Split — Protocol B1 (Primary): Leave-Last-Item-Out (`155` Multi-Item Orders)

| Model | HitRate@5 | Recall@5 | Precision@5 | nDCG@5 | Catalog Cov@5 | Rule Cov Rate | Rule Hits / Fallback Hits | Cond. HitRate@5 (Matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **OrderPopularityBaseline** | 7.74% | 7.74% | 1.55% | **0.0446** | 1.83% (6/328) | 0.00% | 0 / 12 | 0.00% (0) |
| **V1_TFIDF_Content** | 5.81% | 5.81% | 1.16% | **0.0300** | 77.13% (253/328) | 100.00% | 9 / 0 | 5.81% (155) |
| **PairCooccurrence_Pure** | 5.16% | 5.16% | 1.03% | **0.0384** | 49.70% (163/328) | 100.00% | 8 / 0 | 5.16% (155) |
| **PairCooccurrence_With_Fallback** | 5.16% | 5.16% | 1.03% | **0.0384** | 49.70% (163/328) | 100.00% | 8 / 0 | 5.16% (155) |
| **Apriori_ARM_Pure** | 1.29% | 1.29% | 0.26% | **0.0097** | 14.02% (46/328) | 38.71% | 2 / 0 | 3.33% (60) |
| **Apriori_ARM_With_Fallback** | 7.74% | 7.74% | 1.55% | **0.0448** | 14.63% (48/328) | 38.71% | 2 / 10 | 6.67% (60) |

### 4.3. Test Split — Protocol B2 (Secondary): First-Item-Only Seed (`155` Multi-Item Orders)

| Model | HitRate@5 | Recall@5 | Precision@5 | nDCG@5 | Catalog Cov@5 | Rule Cov Rate | Cond. HitRate@5 (Matched) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **OrderPopularityBaseline** | 9.03% | 6.56% | 1.81% | **0.0399** | 1.83% (6/328) | 0.00% | 0.00% (0) |
| **V1_TFIDF_Content** | 8.39% | 6.51% | 1.68% | **0.0362** | 75.91% (249/328) | 100.00% | 8.39% (155) |
| **PairCooccurrence_Pure** | 7.10% | 4.41% | 1.42% | **0.0356** | 46.34% (152/328) | 100.00% | 7.10% (155) |
| **PairCooccurrence_With_Fallback** | 8.39% | 4.95% | 1.68% | **0.0388** | 46.34% (152/328) | 100.00% | 8.39% (155) |
| **Apriori_ARM_Pure** | 0.65% | 0.32% | 0.13% | **0.0025** | 12.20% (40/328) | 32.26% | 2.00% (50) |
| **Apriori_ARM_With_Fallback** | 8.39% | 5.91% | 1.68% | **0.0358** | 13.11% (43/328) | 32.26% | 4.00% (50) |

## 5. Isolated Optional Experiment: Session-Level `add_to_cart` Baskets vs. Purchase Orders

- **Training Cart Sessions:** `2091` sessions with cart additions (`1128` multi-item cart sessions).
- **Discovered Rules (`min_support_count = 2`):** `254` total rules (`245` `1->1` rules, `9` `2->1` rules).
- **Test Protocol B1 (`Session-Cart ARM Pure`):** `HitRate@5 = 1.94%`, `nDCG@5 = 0.0109`, `Rule Coverage Rate = 62.58%`, `Catalog Coverage@5 = 23.78%`.
- **Test Protocol B1 (`Session-Cart ARM With Fallback`):** `HitRate@5 = 7.10%`, `nDCG@5 = 0.0376`, `Catalog Coverage@5 = 23.78%`.

## 6. Runtime & Memory Telemetry

| Model | Fit Time (ms) | Peak Fit Memory (KB) | Test Protocol B1 Inference Time (ms) |
|---|---:|---:|---:|
| **OrderPopularityBaseline** | 1.85 ms | 0.00 KB | 120.38 ms |
| **V1_TFIDF_Content** | 100.26 ms | 0.00 KB | 114.67 ms |
| **PairCooccurrenceBaseline** | 56.71 ms | 241.93 KB | 76.46 ms |
| **Apriori_ARM** | 3092.76 ms | 463.64 KB | 250.38 ms |

## 7. Reproducibility & Verification

- **`data/arm_milestone3_metrics.json` SHA-256:** `a6b5eaaabf3c0e4df49b5ddcaac7f5dbc5a23cd769810a1bb4c9256a58341ee7`
