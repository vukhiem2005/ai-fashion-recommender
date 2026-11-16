# ELAN V5 — Milestone 1: Reproducible Synthetic Interaction Dataset Report

> **CRITICAL SCIENTIFIC DISCLAIMER (`SYNTHETIC_SIMULATION_ONLY`):**
> All records in this dataset are computer-generated synthetic interactions derived from
> mathematical persona and outfit rules over the 328-product ELAN release catalog.
> Offline metrics computed on this dataset validate algorithmic implementation, data pipeline
> integrity, and absence of leakage under simulation assumptions. **They do NOT measure real
> human preference or justify replacing the production V1 recommender on the live website.**

## 1. Provenance & Reproducibility Summary

- **Dataset Label:** `SYNTHETIC_SIMULATION_V5` (`is_synthetic = True` across all rows)
- **Random Seed:** `20260501`
- **Source Catalog:** `backend/data/elan_release.db` (opened strictly in SQLite `mode=ro`)
- **Measured Generation Runtime:** `2.6529` seconds

## 2. Design Assumptions vs. Measured Outcomes

| Metric | Design Assumption / Target | Measured Actual Outcome | Status |
|---|---:|---:|---|
| Active Catalog Products | 328 | **328** | Verified |
| Synthetic Users | 600 | **600** | Verified |
| Shopping Sessions | ~3,600 (assumption) | **3827** | Measured |
| Total Interaction Events | ~21,500 (assumption) | **23335** | Measured |
| Synthetic Orders (Baskets) | ~1,150 (assumption) | **1470** (2885 line items) | Measured |
| Multi-Item Order Ratio | 50.0%–75.0% | **67.55%** (993/1470) | PASS |
| Overall User–Item Sparsity | 91.0%–96.0% | **92.94%** (13885 unique pairs) | PASS |
| Train User–Item Sparsity | >= 91.0% | **95.01%** (9821 unique pairs) | Verified |
| Train Product Coverage | >= 98.0% (>= 322/328) | **100.00%** (328/328) | PASS |
| Overall Product Coverage | Measured | **100.00%** (328/328) | Measured |

## 3. Funnel & Chronological Split Breakdown

- **Event Funnel Counts:**
  - `view`: **14953**
  - `add_to_cart`: **5497** (View-to-Cart Rate: **36.76%**)
  - `purchase`: **2885** (Cart-Session-to-Order Rate: **48.60%**)
  - Mean Distinct Items per Order: **1.96**
  - Top-10% Popular Products Share of Views: **22.49%**

| Split | Window (UTC) | Sessions | Interactions | Orders | Unique $(u, i)$ Pairs | Novel $(u, i)$ Pairs (Eval-Derived) |
|---|---|---:|---:|---:|---:|---:|
| `train` | `2026-01-01` .. `2026-03-04` | 2624 | 16091 | 1016 | 9821 | 9821 (baseline) |
| `val` | `2026-03-05` .. `2026-03-17` | 609 | 3609 | 229 | 2333 | 2094 (239 repeat) |
| `test` | `2026-03-18` .. `2026-03-31` | 594 | 3635 | 225 | 2311 | 1970 (341 repeat) |

## 4. Anti-Leakage Architecture

1. **Strict Quarantine of Generator-Internal Variables:** Latent simulation parameters (`persona_id`,
   `primary_categories`, `preferred_subtypes`, `preferred_color_family`, `preferred_style`,
   `price_tier_preference`, `price_sensitivity`, `exploration_noise_rate`, `session_intent`,
   `anchor_product_id`, `origin_signal`, `latent_utility`) are **completely excluded** from
   `backend/evaluation/v5/data/*.csv` and stored separately in `backend/evaluation/v5/audit/`.
2. **Evaluation-Time Derived Repeat Flag:** `is_repeat_interaction` is not stored in raw interaction CSVs;
   it is computed dynamically at evaluation time by checking whether `(user_id, product_id)` appeared in
   prior chronological splits.
3. **Atomic Session & Order Partitioning:** Every session, along with all its events, orders, and order
   items, is contained strictly within a single chronological split window.

## 5. File SHA-256 Digests (`seed=20260501`)

- `audit/synthetic_interactions_latent_audit_only.csv`: `15cbb8a33de4fa62ff06ddefa8ca9e1d7929ee67b3321c9bbda9a109407ecc4d`
- `audit/synthetic_sessions_latent_audit_only.csv`: `2daa613d63c5e697c1fc58b516608fcd7b0529ae6a0debc777e4b2329b256f5b`
- `audit/synthetic_users_latent_audit_only.csv`: `64ac8a0a73e3d0dc5e166bd3c819721a5f2a80c9848ab15c5489e2ed46dc43f9`
- `data/synthetic_interactions.csv`: `4e622ac11125c581f74d8ace392ef55fc0e3ba6e02cba16f9c18291181f6d656`
- `data/synthetic_order_items.csv`: `d0ed873b68fba883fb68c7e93a09d7eb27d4817065754d4295259f94d2f4c75f`
- `data/synthetic_orders.csv`: `f25988ebba2ff7fcba9fb570a290b5931b61c75fed99be4ace5c139d245d5edc`
- `data/synthetic_sessions.csv`: `a42748c60fb24f95a4d5142561107fabc824f57f46dbe59a9374a13b26db9167`
- `data/synthetic_users.csv`: `a2bc316a124c1cdf5fc4ca995d01b8990f312f0ab76948be5693eea919a8b02a`
