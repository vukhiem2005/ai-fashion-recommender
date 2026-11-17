"""Offline Evaluation & Ablation Benchmark for ELAN V5 Milestone 5 (Hybrid Recommender).

Evaluates strictly on existing Milestone 1 synthetic splits (`SYNTHETIC_SIMULATION_V5`)
with zero test-label leakage:
- Track 1 (Synthetic User-Item Next-Item Ranking, `test` split, Protocol A1 `frozen_train_history`):
  Compares `V1 Content-Based`, `CLIP Late Fusion Content`, `ItemKNN CF`,
  `Hybrid Content + CF`, and `Hybrid Content + CF + ARM` (with user train cart/purchase
  basket context), including a low-history/cold-start slice.
- Track 2 (Synthetic Basket Completion, `test` multi-item orders):
  Compares `V1 Content-Based`, `CLIP Late Fusion`, `ItemKNN CF`, `Apriori ARM`,
  `Hybrid Content + CF`, and `Hybrid Content + CF + ARM`.

Zero runtime PyTorch, Transformers, or image decoding.
Does NOT retune on historical Holdout-18 or present synthetic metrics as real-world proof.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from backend.evaluation.v5.arm_models import (
    AprioriAssociationRuleMiner,
    build_order_transactions,
)
from backend.evaluation.v5.cf_models import (
    ItemKNNRecommender,
    build_train_matrix,
    load_active_catalog_readonly,
)
from backend.evaluation.v5.clip_recommenders import (
    CLIPLateFusionRecommender,
    load_offline_clip_data,
)
from backend.evaluation.v5.evaluate_cf_milestone2 import (
    build_target_ground_truth,
    dcg_at_k,
    load_csv_records,
)
from backend.evaluation.v5.hybrid_models import HybridFusionConfig, HybridRecommender
from backend.evaluation.v5.multimodal_models import DEFAULT_DATA_DIR, DEFAULT_RELEASE_DB

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_M5_METRICS_JSON = DEFAULT_DATA_DIR / "hybrid_milestone5_metrics.json"
DEFAULT_M5_REPORT_MD = THIS_DIR / "hybrid_milestone5_report.md"


def build_fitted_hybrid_system() -> tuple[
    HybridRecommender,
    Any,
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    tuple[str, ...],
]:
    """Fit CF and ARM strictly on `train`, load frozen CLIP Late Fusion, and assemble Hybrid."""
    users_rows = load_csv_records(DEFAULT_DATA_DIR / "synthetic_users.csv")
    interactions_rows = load_csv_records(DEFAULT_DATA_DIR / "synthetic_interactions.csv")
    orders_rows = load_csv_records(DEFAULT_DATA_DIR / "synthetic_orders.csv")
    order_items_rows = load_csv_records(DEFAULT_DATA_DIR / "synthetic_order_items.csv")

    catalog = load_active_catalog_readonly(DEFAULT_RELEASE_DB)
    user_ids = tuple(sorted(r["user_id"].strip() for r in users_rows))
    product_ids = tuple(p["id"] for p in catalog)

    train_interactions = [r for r in interactions_rows if r["split"].strip() == "train"]
    train_bundle = build_train_matrix(
        train_interactions=train_interactions,
        user_ids=user_ids,
        product_ids=product_ids,
        dedup_mode="log_sum",
        iuf_exponent=0.5,
        l2_normalize_users=True,
    )

    cf_model = ItemKNNRecommender(
        k_neighbors=100, shrinkage=20.0, normalize_sim_columns=False
    ).fit(train_bundle)

    train_orders, _ = build_order_transactions(
        orders_rows=orders_rows,
        order_items_rows=order_items_rows,
        interactions_rows=interactions_rows,
        split_filter="train",
    )
    arm_model = AprioriAssociationRuleMiner(
        min_support_count=2,
        min_confidence=0.05,
        min_lift=1.05,
        alpha_smooth=0.0,
        beta_lift=1.0,
        max_itemset_len=3,
    ).fit(train_orders, catalog_product_ids=product_ids)

    clip_data = load_offline_clip_data()
    clip_late = CLIPLateFusionRecommender(
        w_vis=0.5,
        text_gate_threshold=0.0,
        require_same_category_for_visual=False,
        text_source="v1_tfidf",
    ).fit(clip_data)

    hybrid = HybridRecommender(
        clip_late_fusion=clip_late,
        cf_model=cf_model,
        arm_model=arm_model,
        clip_data=clip_data,
        config=HybridFusionConfig(w_content=0.50, w_cf=0.35, w_arm=0.15),
    )
    return hybrid, train_bundle, interactions_rows, orders_rows, order_items_rows, user_ids


def _evaluate_track1_user_ranking(
    hybrid: HybridRecommender,
    train_bundle: Any,
    user_ids: tuple[str, ...],
    test_gt: dict[str, dict[str, int]],
    user_train_baskets: dict[str, tuple[str, ...]],
    user_train_counts: dict[str, int],
    mode_kwargs: dict[str, Any],
    use_basket_context: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs_bin: list[float] = []
    ndcgs_grad: list[float] = []
    low_hist_ndcg: list[float] = []
    high_hist_ndcg: list[float] = []
    unique_recs: set[str] = set()
    arm_activated_count = 0
    cold_fallback_count = 0

    t0 = time.perf_counter()
    for u_idx, uid in enumerate(user_ids):
        gt = test_gt.get(uid)
        if not gt:
            continue
        basket = user_train_baskets.get(uid, ()) if use_basket_context else None
        out = hybrid.recommend(
            user_profile=train_bundle.normalized_matrix[u_idx],
            seen_mask_row=train_bundle.seen_mask[u_idx],
            basket_product_ids=basket,
            top_k=top_k,
            exclude_seen=True,
            **mode_kwargs,
        )
        if out["arm_activated"]:
            arm_activated_count += 1
        if out["cold_start_fallback"]:
            cold_fallback_count += 1

        top_pids = [cid for cid, _ in out["recommendations"]]
        unique_recs.update(top_pids)

        hits = [1.0 if pid in gt else 0.0 for pid in top_pids]
        n_hits = sum(hits)
        precisions.append(n_hits / float(top_k))
        recalls.append(n_hits / float(len(gt)))

        idcg_b = dcg_at_k([1.0] * min(top_k, len(gt)), k=top_k)
        ndcgs_bin.append(dcg_at_k(hits, k=top_k) / idcg_b if idcg_b > 0 else 0.0)

        gains = [float((2 ** gt[pid]) - 1) if pid in gt else 0.0 for pid in top_pids]
        ideal_g = [float((2 ** g) - 1) for g in sorted(gt.values(), reverse=True)[:top_k]]
        idcg_g = dcg_at_k(ideal_g, k=top_k)
        u_ndcg_g = dcg_at_k(gains, k=top_k) / idcg_g if idcg_g > 0 else 0.0
        ndcgs_grad.append(u_ndcg_g)

        if user_train_counts.get(uid, 0) <= 5:
            low_hist_ndcg.append(u_ndcg_g)
        else:
            high_hist_ndcg.append(u_ndcg_g)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    n_eval = len(precisions)
    return {
        "evaluated_users": n_eval,
        "precision_at_5": round(float(np.mean(precisions)), 4),
        "recall_at_5": round(float(np.mean(recalls)), 4),
        "ndcg_binary_at_5": round(float(np.mean(ndcgs_bin)), 4),
        "ndcg_graded_at_5": round(float(np.mean(ndcgs_grad)), 4),
        "low_history_users_le5_count": len(low_hist_ndcg),
        "low_history_ndcg_graded_at_5": round(float(np.mean(low_hist_ndcg)), 4) if low_hist_ndcg else 0.0,
        "active_users_gt5_count": len(high_hist_ndcg),
        "active_users_ndcg_graded_at_5": round(float(np.mean(high_hist_ndcg)), 4) if high_hist_ndcg else 0.0,
        "catalog_coverage_at_5": round(len(unique_recs) / float(hybrid.n_items), 4),
        "unique_recommended_items_at_5": len(unique_recs),
        "arm_activation_rate": round(arm_activated_count / float(max(1, n_eval)), 4),
        "cold_start_fallback_count": cold_fallback_count,
        "mean_query_latency_ms": round(elapsed_ms / float(max(1, n_eval)), 4),
    }


def _evaluate_track2_basket_completion(
    hybrid: HybridRecommender,
    test_multi_orders: list[Any],
    mode_kwargs: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    hits: list[float] = []
    ndcgs: list[float] = []
    unique_recs: set[str] = set()
    arm_activated_count = 0

    t0 = time.perf_counter()
    for tx in test_multi_orders:
        prefix = tx.ordered_products[:-1]
        target = tx.ordered_products[-1]

        out = hybrid.recommend(
            anchor_product_id=prefix[-1],
            basket_product_ids=prefix,
            top_k=top_k,
            **mode_kwargs,
        )
        if out["arm_activated"]:
            arm_activated_count += 1
        top_pids = [cid for cid, _ in out["recommendations"]]
        unique_recs.update(top_pids)

        if target in top_pids:
            rank = top_pids.index(target)
            hits.append(1.0)
            ndcgs.append(1.0 / math.log2(rank + 2.0))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    n_baskets = len(hits)
    return {
        "evaluated_baskets": n_baskets,
        "hit_rate_at_5": round(float(np.mean(hits)), 4),
        "ndcg_at_5": round(float(np.mean(ndcgs)), 4),
        "catalog_coverage_at_5": round(len(unique_recs) / float(hybrid.n_items), 4),
        "unique_recommended_items_at_5": len(unique_recs),
        "arm_activation_rate": round(arm_activated_count / float(max(1, n_baskets)), 4),
        "mean_query_latency_ms": round(elapsed_ms / float(max(1, n_baskets)), 4),
    }


def run_milestone5_evaluation() -> dict[str, Any]:
    (
        hybrid,
        train_bundle,
        interactions_rows,
        orders_rows,
        order_items_rows,
        user_ids,
    ) = build_fitted_hybrid_system()

    train_seen_pairs = {
        (r["user_id"].strip(), r["product_id"].strip())
        for r in interactions_rows
        if r["split"].strip() == "train"
    }
    test_interactions = [r for r in interactions_rows if r["split"].strip() == "test"]
    test_gt = build_target_ground_truth(
        test_interactions, conditioning_seen_pairs=train_seen_pairs, require_novel=True
    )

    # Derive each user's train cart/purchase basket context strictly from `train`
    user_train_baskets_map: dict[str, list[str]] = defaultdict(list)
    user_train_items: dict[str, set[str]] = defaultdict(set)
    for r in interactions_rows:
        if r["split"].strip() != "train":
            continue
        uid = r["user_id"].strip()
        pid = r["product_id"].strip()
        user_train_items[uid].add(pid)
        if r["event_type"].strip() in ("add_to_cart", "purchase") and pid not in user_train_baskets_map[uid]:
            user_train_baskets_map[uid].append(pid)

    user_train_baskets = {uid: tuple(pids) for uid, pids in user_train_baskets_map.items()}
    user_train_counts = {uid: len(pids) for uid, pids in user_train_items.items()}

    # Track 1 ablations
    track1_configs = {
        "1_v1_content_tfidf": (
            {"enable_content": True, "enable_cf": False, "enable_arm": False, "use_v1_content_only": True},
            False,
        ),
        "2_clip_late_fusion_content": (
            {"enable_content": True, "enable_cf": False, "enable_arm": False, "use_v1_content_only": False},
            False,
        ),
        "3_itemknn_cf": (
            {"enable_content": False, "enable_cf": True, "enable_arm": False},
            False,
        ),
        "4_hybrid_content_cf": (
            {"enable_content": True, "enable_cf": True, "enable_arm": False},
            False,
        ),
        "5_hybrid_content_cf_arm": (
            {"enable_content": True, "enable_cf": True, "enable_arm": True},
            True,
        ),
    }
    track1_results = {
        name: _evaluate_track1_user_ranking(
            hybrid=hybrid,
            train_bundle=train_bundle,
            user_ids=user_ids,
            test_gt=test_gt,
            user_train_baskets=user_train_baskets,
            user_train_counts=user_train_counts,
            mode_kwargs=kwargs,
            use_basket_context=use_basket,
        )
        for name, (kwargs, use_basket) in track1_configs.items()
    }

    # Track 2: Synthetic Basket Completion on test multi-item orders
    test_orders, _ = build_order_transactions(
        orders_rows=orders_rows,
        order_items_rows=order_items_rows,
        interactions_rows=interactions_rows,
        split_filter="test",
    )
    test_multi_orders = [tx for tx in test_orders if len(tx.ordered_products) >= 2]

    track2_configs = {
        "1_v1_content_tfidf": {"enable_content": True, "enable_cf": False, "enable_arm": False, "use_v1_content_only": True},
        "2_clip_late_fusion_content": {"enable_content": True, "enable_cf": False, "enable_arm": False, "use_v1_content_only": False},
        "3_itemknn_cf": {"enable_content": False, "enable_cf": True, "enable_arm": False},
        "4_apriori_arm_pop_fallback": {"enable_content": False, "enable_cf": False, "enable_arm": True},
        "5_hybrid_content_cf": {"enable_content": True, "enable_cf": True, "enable_arm": False},
        "6_hybrid_content_cf_arm": {"enable_content": True, "enable_cf": True, "enable_arm": True},
    }
    track2_results = {
        name: _evaluate_track2_basket_completion(hybrid, test_multi_orders, kwargs)
        for name, kwargs in track2_configs.items()
    }

    # Cold-start & missing-image behavioral probes
    cold_user_out = hybrid.recommend(
        user_profile=np.zeros(hybrid.n_items, dtype=np.float64),
        top_k=5,
    )
    missing_img_out = hybrid.recommend(
        anchor_product_id="HM-0398947001",
        enable_content=True,
        enable_cf=False,
        enable_arm=False,
        top_k=5,
    )

    payload: dict[str, Any] = {
        "milestone": "Milestone 5 — Hybrid Recommendation MVP & Offline Evaluation",
        "data_provenance": {
            "user_item_and_basket_splits": "SYNTHETIC_SIMULATION_V5 (train=16,091 events / 1,016 orders; test=3,635 events / 225 orders, 155 multi-item baskets)",
            "item_similarity_reference": "Phase 4B locked Holdout-18 historical benchmark (CLIP Late Fusion nDCG@5 = 0.7625 vs V1 0.6967, +9.44%); NOT retuned in Milestone 5.",
            "disclaimer": "Synthetic user-item and basket-completion metrics validate hybrid pipeline integration, modality normalization, and cold-start routing under simulation rules. They do not establish real-world consumer uplift.",
        },
        "default_fusion_config": {
            "w_content": hybrid.config.w_content,
            "w_cf": hybrid.config.w_cf,
            "w_arm": hybrid.config.w_arm,
            "normalization": "Per-query min-max normalization into [0, 1] over eligible candidates + modality-normalized weight sum",
        },
        "track1_synthetic_user_ranking_test": track1_results,
        "track2_synthetic_basket_completion_test": track2_results,
        "behavioral_probes": {
            "cold_start_zero_history_user": {
                "cold_start_fallback": cold_user_out["cold_start_fallback"],
                "active_modalities": cold_user_out["active_modalities"],
                "top5_ids": [cid for cid, _ in cold_user_out["recommendations"]],
            },
            "missing_image_anchor_HM_0398947001": {
                "missing_image_fallback": missing_img_out["missing_image_fallback"],
                "active_modalities": missing_img_out["active_modalities"],
                "top5_ids": [cid for cid, _ in missing_img_out["recommendations"]],
            },
        },
    }

    DEFAULT_M5_METRICS_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    DEFAULT_M5_REPORT_MD.write_text(
        _build_m5_markdown(payload), encoding="utf-8", newline="\n"
    )
    return payload


def _build_m5_markdown(p: dict[str, Any]) -> str:
    t1 = p["track1_synthetic_user_ranking_test"]
    t2 = p["track2_synthetic_basket_completion_test"]
    lines = [
        "# ÉLAN V5 — Milestone 5: Hybrid Recommendation Report",
        "",
        f"> **Data Provenance:** {p['data_provenance']['disclaimer']}",
        "",
        "## 1. Hybrid Architecture & Calibrated Fusion",
        "- **Content / Multimodal Signal (`w_content = 0.50`):** Phase 4B `CLIPLateFusionRecommender` (`text_source='v1_tfidf'`, `w_vis=0.5`, `tau=0.0`, `same_cat=False`) with exact V1 TF-IDF fallback for `HM-0398947001`.",
        "- **Collaborative Filtering (`w_cf = 0.35`):** Milestone 2 `ItemKNNRecommender` (`log_sum`, `iuf_exponent=0.5`, `shrinkage=20.0`, `k_neighbors=100`).",
        "- **Association Rule Mining (`w_arm = 0.15`):** Milestone 3 `AprioriAssociationRuleMiner` (`104` rules from `train` orders), activated strictly when basket context is provided and matching rules exist.",
        "- **Normalization:** Per-query min-max normalization into `[0.0, 1.0]` over eligible candidates; uninformative/inactive modalities receive zero weight so active weights always sum to `1.0`.",
        "",
        "## 2. Track 1 — Synthetic User-Item Next-Item Ranking (`test` split, Protocol A1)",
        "",
        "| Model / Ablation | Users | `P@5` | `Recall@5` | `nDCG@5 (Bin)` | `nDCG@5 (Graded)` | `Low-Hist (<=5) nDCG@5` | `Coverage@5` | `ARM Rate` | Latency (`ms`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, r in t1.items():
        lines.append(
            f"| `{name}` | `{r['evaluated_users']}` | `{r['precision_at_5']:.4f}` | `{r['recall_at_5']:.4f}` | `{r['ndcg_binary_at_5']:.4f}` | `{r['ndcg_graded_at_5']:.4f}` | `{r['low_history_ndcg_graded_at_5']:.4f}` | `{r['catalog_coverage_at_5']:.4f}` | `{r['arm_activation_rate']:.4f}` | `{r['mean_query_latency_ms']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## 3. Track 2 — Synthetic Basket Completion (`test` multi-item orders, `155` baskets)",
            "",
            "| Model / Ablation | Baskets | `HitRate@5` | `nDCG@5` | `Coverage@5` | `ARM Activation` | Latency (`ms`) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for name, r in t2.items():
        lines.append(
            f"| `{name}` | `{r['evaluated_baskets']}` | `{r['hit_rate_at_5']:.4f}` | `{r['ndcg_at_5']:.4f}` | `{r['catalog_coverage_at_5']:.4f}` | `{r['arm_activation_rate']:.4f}` | `{r['mean_query_latency_ms']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Cold-Start & Missing-Asset Routing",
            f"- **Zero-History User Fallback:** `cold_start_fallback={p['behavioral_probes']['cold_start_zero_history_user']['cold_start_fallback']}`, `active_modalities={p['behavioral_probes']['cold_start_zero_history_user']['active_modalities']}`.",
            f"- **Missing-Image Anchor (`HM-0398947001`):** `missing_image_fallback={p['behavioral_probes']['missing_image_anchor_HM_0398947001']['missing_image_fallback']}`, exact V1 TF-IDF Top-5 preserved.",
            "",
            "## 5. Reproducibility Command",
            "```powershell",
            r"C:\Users\vuvan\Projects\_venvs\elan-v5-clip\Scripts\python.exe -m backend.evaluation.v5.evaluate_hybrid_milestone5",
            r"C:\Users\vuvan\Projects\_venvs\elan-v5-clip\Scripts\python.exe -m backend.evaluation.v5.validate_hybrid_milestone5",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    out = run_milestone5_evaluation()
    print(
        json.dumps(
            {
                "status": "OK",
                "track1": out["track1_synthetic_user_ranking_test"],
                "track2": out["track2_synthetic_basket_completion_test"],
            },
            indent=2,
        )
    )
