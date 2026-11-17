# ÉLAN V5 — Milestone 4B.3C: Final Holdout Benchmark Report

> **Historical Holdout Provenance:** Historically reused holdout set (holdout_v4_blind_labeled.csv: 18 disjoint references, 140 judged pairs). Previously consulted during V4/Phase 4A; reported transparently as historical evaluation material rather than a newly collected blind test.

- **Locked Config File SHA-256:** `f6ed98abe2dd630e10acebb4f7af1265ac666d3a09b67c4794cedca857a0ad17`
- **CLIP NPZ SHA-256:** `4265a3a69611ef57c3f3350cdcdd971e31e10d56db4005b07a328ec410af1e7f`
- **Phase 4A NPZ SHA-256:** `cfa7af1cabed44767f610b724c3d28683a079eae38c79a84327e0c06afba1ae6`
- **Holdout-18 Dataset:** `18` disjoint references, `140` judged pairs (`0` overlap with `Dev-18`).

## 1. Seven-Model Holdout-18 & Dev-18 Comparison (Protocol M1: Pooled Reranking)

| Model | Dev `P@5` | Dev `nDCG@5` | Holdout `P@5` | Holdout `Strict P@5` | Holdout `nDCG@5` | `Δ nDCG@5` vs V1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact V1 TF-IDF** | `0.5667` | `0.7699` | `0.6667` | `0.2444` | `0.6967` | `+0.0000` (`+0.00%`) |
| **Phase 4A Image-Only** | `0.6111` | `0.6731` | `0.7000` | `0.2333` | `0.6823` | `-0.0144` (`-2.07%`) |
| **Phase 4A Early Fusion** | `0.5889` | `0.7781` | `0.7111` | `0.2556` | `0.7338` | `+0.0371` (`+5.33%`) |
| **Phase 4A Late Fusion** | `0.6111` | `0.8023` | `0.7000` | `0.2444` | `0.7282` | `+0.0315` (`+4.52%`) |
| **Phase 4B CLIP Image-Only** | `0.7000` | `0.8425` | `0.7222` | `0.2333` | `0.7091` | `+0.0124` (`+1.78%`) |
| **Phase 4B CLIP Text-Image** | `0.7556` | `0.8819` | `0.7667` | `0.2333` | `0.7066` | `+0.0099` (`+1.42%`) |
| **Phase 4B CLIP Late Fusion** | `0.6889` | `0.8933` | `0.7111` | `0.2667` | `0.7625` | `+0.0658` (`+9.44%`) |

## 2. Protocol M2 — Unrestricted Full-Catalog Diagnostics (All 328 Products)

> Note: Unjudged Top-5 slot rates reflect retrieval outside historical V1/V2 pools and must **not** be interpreted as irrelevance.

| Model | `Coverage@5` | `Same-Category@5` | `Overlap w/ V1` | `Dev-18 Unjudged` | `Holdout-18 Unjudged` | Latency (`ms/query`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact V1 TF-IDF** | `0.9665` (`317/328`) | `0.7665` | `1.0000` | `0.0000` | `0.0000` | `0.7448` |
| **Phase 4A Image-Only** | `0.9634` (`316/328`) | `0.2793` | `0.0848` | `0.9222` | `0.9000` | `0.7201` |
| **Phase 4A Early Fusion** | `0.9604` (`315/328`) | `0.7500` | `0.8433` | `0.1000` | `0.1000` | `0.7439` |
| **Phase 4A Late Fusion** | `0.9726` (`319/328`) | `0.8287` | `0.6713` | `0.2000` | `0.2556` | `0.6886` |
| **Phase 4B CLIP Image-Only** | `0.9116` (`299/328`) | `0.6232` | `0.1896` | `0.6889` | `0.8444` | `0.8547` |
| **Phase 4B CLIP Text-Image** | `0.8598` (`282/328`) | `0.7299` | `0.2841` | `0.6222` | `0.7333` | `0.6568` |
| **Phase 4B CLIP Late Fusion** | `0.9543` (`313/328`) | `0.8030` | `0.6683` | `0.2444` | `0.2111` | `0.4850` |

## 3. Per-Query Holdout-18 `nDCG@5` Breakdown

| Reference ID | Name | Category | V1 | P4A Late | CLIP Img | CLIP Text-Img | CLIP Late | `Δ CLIP Late vs V1` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `HM-0680396001` | Zitta Scarf | Accessories | `0.7808` | `0.9904` | `0.6094` | `0.8286` | `1.0000` | `+0.2192` |
| `HM-0752869003` | MAGIC GLOVES | Accessories | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `+0.0000` |
| `HM-0715599001` | Alma Onesie | Accessories | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `+0.0000` |
| `HM-0574428009` | Sivert 5-pocket | Bottoms | `0.5847` | `0.7869` | `0.8684` | `0.9355` | `0.8262` | `+0.2415` |
| `HM-0786627001` | Mini Sunrise Jumpsuit | Bottoms | `1.0000` | `1.0000` | `1.0000` | `0.5000` | `1.0000` | `+0.0000` |
| `HM-0567805003` | Freddie Linen | Bottoms | `0.6716` | `0.7727` | `0.7638` | `0.7979` | `0.9130` | `+0.2414` |
| `HM-0633535001` | AG Sorrel | Dresses | `1.0000` | `0.9578` | `1.0000` | `0.9355` | `1.0000` | `+0.0000` |
| `HM-0611924001` | Holly off shoulder dress | Dresses | `0.7380` | `0.9578` | `0.8390` | `0.7968` | `0.8167` | `+0.0787` |
| `HM-0627725003` | BILLIE set | Dresses | `1.0000` | `0.7979` | `0.7979` | `0.8508` | `0.8508` | `-0.1492` |
| `HM-0542917001` | Tage Sweater | Knitwear | `0.7744` | `0.9131` | `0.7226` | `0.7034` | `0.8107` | `+0.0363` |
| `HM-0825924001` | Erika 2 pack | Knitwear | `0.6508` | `1.0000` | `0.7654` | `0.7328` | `0.7654` | `+0.1146` |
| `HM-0555364011` | Spencer Shirt Sweater | Knitwear | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `+0.0000` |
| `HM-0553092017` | Hunter Super Skinny | Outerwear | `0.7116` | `0.6464` | `1.0000` | `0.8437` | `0.9271` | `+0.2155` |
| `HM-0513373021` | Hepburn top | Outerwear | `0.9307` | `0.9377` | `0.6135` | `0.8732` | `0.9195` | `-0.0112` |
| `HM-0699070010` | Price tank / Filippa | Outerwear | `0.7979` | `0.7699` | `1.0000` | `0.7979` | `0.8508` | `+0.0529` |
| `HM-0672957001` | MC Charlie | Tops | `0.6729` | `0.5342` | `0.5994` | `0.5994` | `0.6729` | `+0.0000` |
| `HM-0708464002` | STORK FANCY | Tops | `0.2381` | `0.1942` | `0.4161` | `0.6899` | `0.4794` | `+0.2413` |
| `HM-0614634004` | Babe tee | Tops | `0.9891` | `0.8481` | `0.7694` | `0.8328` | `0.8933` | `-0.0958` |
