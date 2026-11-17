# ÉLAN V5 — Milestone 5: Hybrid Recommendation Report

> **Data Provenance:** Synthetic user-item and basket-completion metrics validate hybrid pipeline integration, modality normalization, and cold-start routing under simulation rules. They do not establish real-world consumer uplift.

## 1. Hybrid Architecture & Calibrated Fusion
- **Content / Multimodal Signal (`w_content = 0.50`):** Phase 4B `CLIPLateFusionRecommender` (`text_source='v1_tfidf'`, `w_vis=0.5`, `tau=0.0`, `same_cat=False`) with exact V1 TF-IDF fallback for `HM-0398947001`.
- **Collaborative Filtering (`w_cf = 0.35`):** Milestone 2 `ItemKNNRecommender` (`log_sum`, `iuf_exponent=0.5`, `shrinkage=20.0`, `k_neighbors=100`).
- **Association Rule Mining (`w_arm = 0.15`):** Milestone 3 `AprioriAssociationRuleMiner` (`104` rules from `train` orders), activated strictly when basket context is provided and matching rules exist.
- **Normalization:** Per-query min-max normalization into `[0.0, 1.0]` over eligible candidates; uninformative/inactive modalities receive zero weight so active weights always sum to `1.0`.

## 2. Track 1 — Synthetic User-Item Next-Item Ranking (`test` split, Protocol A1)

| Model / Ablation | Users | `P@5` | `Recall@5` | `nDCG@5 (Bin)` | `nDCG@5 (Graded)` | `Low-Hist (<=5) nDCG@5` | `Coverage@5` | `ARM Rate` | Latency (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_v1_content_tfidf` | `545` | `0.0239` | `0.0294` | `0.0259` | `0.0230` | `0.0000` | `0.8049` | `0.0000` | `1.1382` |
| `2_clip_late_fusion_content` | `545` | `0.0183` | `0.0238` | `0.0199` | `0.0161` | `0.0000` | `0.5976` | `0.0000` | `1.0575` |
| `3_itemknn_cf` | `545` | `0.0411` | `0.0561` | `0.0538` | `0.0514` | `0.0000` | `0.3659` | `0.0000` | `0.9986` |
| `4_hybrid_content_cf` | `545` | `0.0396` | `0.0568` | `0.0502` | `0.0501` | `0.0000` | `0.4573` | `0.0000` | `1.1894` |
| `5_hybrid_content_cf_arm` | `545` | `0.0389` | `0.0559` | `0.0500` | `0.0493` | `0.0000` | `0.4695` | `0.8239` | `1.2873` |

## 3. Track 2 — Synthetic Basket Completion (`test` multi-item orders, `155` baskets)

| Model / Ablation | Baskets | `HitRate@5` | `nDCG@5` | `Coverage@5` | `ARM Activation` | Latency (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_v1_content_tfidf` | `155` | `0.0645` | `0.0325` | `0.7652` | `0.0000` | `1.4792` |
| `2_clip_late_fusion_content` | `155` | `0.0645` | `0.0466` | `0.7561` | `0.0000` | `1.4200` |
| `3_itemknn_cf` | `155` | `0.0516` | `0.0354` | `0.5793` | `0.0000` | `1.1228` |
| `4_apriori_arm_pop_fallback` | `155` | `0.0839` | `0.0536` | `0.1463` | `0.3871` | `0.9150` |
| `5_hybrid_content_cf` | `155` | `0.0581` | `0.0379` | `0.6159` | `0.0000` | `0.9198` |
| `6_hybrid_content_cf_arm` | `155` | `0.0645` | `0.0379` | `0.6220` | `0.3871` | `0.9729` |

## 4. Cold-Start & Missing-Asset Routing
- **Zero-History User Fallback:** `cold_start_fallback=True`, `active_modalities=['popularity_fallback']`.
- **Missing-Image Anchor (`HM-0398947001`):** `missing_image_fallback=True`, exact V1 TF-IDF Top-5 preserved.

## 5. Reproducibility Command
```powershell
C:\Users\vuvan\Projects\_venvs\elan-v5-clip\Scripts\python.exe -m backend.evaluation.v5.evaluate_hybrid_milestone5
C:\Users\vuvan\Projects\_venvs\elan-v5-clip\Scripts\python.exe -m backend.evaluation.v5.validate_hybrid_milestone5
```
